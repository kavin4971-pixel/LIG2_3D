from __future__ import annotations

import argparse
import csv
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from transformer_ppo_remus import (
    CausalTransformerActorCritic,
    DEFAULT_CURRICULUM,
    PPOConfig,
    PPOTrainer,
    RemusRewardWrapper,
    RewardConfig,
    RunningMeanStd,
    SequenceContext,
    load_env_class,
    set_seed,
)


def rotation_matrix_body_to_inertial(phi: float, theta: float, psi: float) -> np.ndarray:
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)

    rz = np.array(
        [
            [cps, -sps, 0.0],
            [sps, cps, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    ry = np.array(
        [
            [cth, 0.0, sth],
            [0.0, 1.0, 0.0],
            [-sth, 0.0, cth],
        ],
        dtype=np.float64,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cphi, -sphi],
            [0.0, sphi, cphi],
        ],
        dtype=np.float64,
    )
    return rz @ ry @ rx


def _load_checkpoint(path: Path, device: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _config_from_checkpoint(args: argparse.Namespace, checkpoint: Dict[str, Any]) -> PPOConfig:
    cfg = PPOConfig()
    ckpt_cfg = checkpoint.get("ppo_config", {})
    if isinstance(ckpt_cfg, dict):
        valid_fields = {field.name for field in fields(PPOConfig)}
        for key, value in ckpt_cfg.items():
            if key in valid_fields:
                setattr(cfg, key, value)

    cfg.seed = args.seed
    cfg.device = args.device
    cfg.save_dir = str(args.output_dir)
    cfg.run_dir = str(args.output_dir / "_loader_run")
    cfg.log_csv = False
    cfg.resume_from = str(args.checkpoint)
    cfg.start_stage = args.stage
    cfg.use_guided_action_prior = not args.no_guided_action_prior
    cfg.plot_stage4 = False
    cfg.show_stage4_plots = False
    return cfg


def _infer_dims(cfg: PPOConfig, reward_cfg: RewardConfig) -> Tuple[int, int]:
    REMUSAUVEnv = load_env_class()
    tmp_stage = DEFAULT_CURRICULUM[0]
    tmp_env = REMUSAUVEnv(
        seed=cfg.seed,
        current_enabled=tmp_stage.current_enabled,
        include_current_in_obs=False,
        n_obstacles=tmp_stage.n_obstacles,
    )
    tmp_env.target_z_min = tmp_stage.target_z_min
    tmp_env.target_z_max = tmp_stage.target_z_max
    if tmp_stage.target_distance_min is not None and hasattr(tmp_env, "min_start_target_distance"):
        tmp_env.min_start_target_distance = tmp_stage.target_distance_min
    if hasattr(tmp_env, "max_start_target_distance"):
        tmp_env.max_start_target_distance = tmp_stage.target_distance_max
    if tmp_stage.target_boundary_margin is not None and hasattr(tmp_env, "target_boundary_margin"):
        tmp_env.target_boundary_margin = tmp_stage.target_boundary_margin

    tmp_env = RemusRewardWrapper(tmp_env, reward_cfg)
    obs_dim = int(np.prod(tmp_env.observation_space.shape))
    act_dim = int(np.prod(tmp_env.action_space.shape))
    tmp_env.close()
    return obs_dim, act_dim


def build_trainer(args: argparse.Namespace) -> PPOTrainer:
    checkpoint = _load_checkpoint(args.checkpoint, args.device)
    reward_cfg = RewardConfig()
    cfg = _config_from_checkpoint(args, checkpoint)
    obs_dim, act_dim = _infer_dims(cfg, reward_cfg)

    policy = CausalTransformerActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        seq_len=cfg.seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        init_log_std=cfg.init_log_std,
        init_propeller_bias=cfg.init_propeller_bias,
    )
    obs_rms = RunningMeanStd(shape=(obs_dim,))
    trainer = PPOTrainer(
        policy=policy,
        cfg=cfg,
        reward_cfg=reward_cfg,
        obs_rms=obs_rms,
        curriculum=DEFAULT_CURRICULUM,
    )
    trainer.policy.eval()
    return trainer


def deterministic_action_from_context(trainer: PPOTrainer, context: SequenceContext) -> np.ndarray:
    obs_hist = list(context.obs_hist)
    prev_act_hist = list(context.prev_act_hist)
    if not obs_hist:
        return np.zeros(trainer.act_dim, dtype=np.float32)

    # Avoid fully-masked padding queries in torch TransformerEncoder by running
    # only the real history tokens. Use the rightmost positional embeddings so
    # the policy sees the same positions as the right-aligned training context.
    obs_seq = torch.as_tensor(np.stack(obs_hist, axis=0), dtype=torch.float32, device=trainer.device).unsqueeze(0)
    prev_act_seq = torch.as_tensor(
        np.stack(prev_act_hist, axis=0),
        dtype=torch.float32,
        device=trainer.device,
    ).unsqueeze(0)
    n_tokens = obs_seq.shape[1]

    policy = trainer.policy
    with torch.no_grad():
        x = torch.cat([obs_seq, prev_act_seq], dim=-1)
        x = policy.in_proj(x) + policy.pos_embed[:, -n_tokens:, :]
        x = policy.encoder(x, mask=policy._causal_mask(n_tokens, trainer.device))
        x = policy.final_norm(x)
        mean = policy.actor(x[:, -1])
        action = torch.tanh(mean)

    if not torch.isfinite(action).all():
        raise FloatingPointError("Policy produced a non-finite action during GIF rollout.")
    return action.squeeze(0).cpu().numpy().astype(np.float32)


def rollout_episode(trainer: PPOTrainer, seed: int, sample_stride: int) -> Dict[str, Any]:
    env = trainer.eval_env
    obs, info = env.reset(seed=seed)
    reset_start = np.asarray(info.get("start", np.zeros(3)), dtype=np.float64).copy()
    reset_target = np.asarray(info.get("target", np.zeros(3)), dtype=np.float64).copy()
    reset_yaw = float(np.arctan2(reset_target[1] - reset_start[1], reset_target[0] - reset_start[0]))
    context = SequenceContext(trainer.cfg.seq_len, trainer.obs_dim, trainer.act_dim)
    prev_action = np.zeros(trainer.act_dim, dtype=np.float32)
    context.append(obs, prev_action)

    positions: List[np.ndarray] = [reset_start.copy()]
    orientations: List[np.ndarray] = [np.array([0.0, 0.0, reset_yaw], dtype=np.float64)]
    currents: List[np.ndarray] = [np.asarray(info.get("current_inertial", np.zeros(3)), dtype=np.float64).copy()]
    obstacles: List[np.ndarray] = [
        np.asarray(info.get("obstacle_centers", np.zeros((0, 3))), dtype=np.float64).copy()
    ]
    obstacle_radii = np.asarray(info.get("obstacle_radii", np.zeros((0,))), dtype=np.float64).copy()

    done = False
    steps = 0
    event = ""
    final_distance = float(info.get("mission_distance", 0.0))
    min_clearance = float("inf")
    sample_stride = max(int(sample_stride), 1)

    try:
        while not done:
            action = deterministic_action_from_context(trainer, context)
            next_obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            event = str(info.get("event") or event)
            final_distance = float(info.get("distance_to_goal", final_distance))

            clearance = info.get("min_obstacle_clearance")
            if clearance is not None and np.isfinite(float(clearance)):
                min_clearance = min(min_clearance, float(clearance))

            if steps % sample_stride == 0 or done:
                positions.append(np.asarray(info.get("position", np.zeros(3)), dtype=np.float64).copy())
                orientations.append(
                    np.array(
                        [
                            float(info.get("roll", 0.0)),
                            float(info.get("pitch", 0.0)),
                            float(info.get("yaw", reset_yaw)),
                        ],
                        dtype=np.float64,
                    )
                )
                currents.append(np.asarray(info.get("current_inertial", np.zeros(3)), dtype=np.float64).copy())
                obstacles.append(
                    np.asarray(info.get("obstacle_centers", np.zeros((0, 3))), dtype=np.float64).copy()
                )

            context.append(next_obs, action)
            obs = next_obs
    except FloatingPointError as exc:
        event = str(exc)
        done = True

    return {
        "seed": seed,
        "success": bool(info.get("goal_reached", False)),
        "event": event,
        "steps": steps,
        "final_distance": final_distance,
        "min_clearance": min_clearance if np.isfinite(min_clearance) else np.nan,
        "start": reset_start,
        "target": reset_target,
        "positions": np.asarray(positions, dtype=np.float64),
        "orientations": np.asarray(orientations, dtype=np.float64),
        "currents": np.asarray(currents, dtype=np.float64),
        "obstacles": obstacles,
        "obstacle_radii": obstacle_radii,
    }


def _frame_indices(n_frames: int, max_frames: int) -> np.ndarray:
    if n_frames <= max_frames:
        return np.arange(n_frames, dtype=int)
    return np.unique(np.linspace(0, n_frames - 1, max_frames, dtype=int))


def save_trajectory_gif(trajectory: Dict[str, Any], path: Path, fps: int, max_frames: int, dpi: int) -> None:
    positions = np.asarray(trajectory["positions"], dtype=np.float64)
    currents = np.asarray(trajectory["currents"], dtype=np.float64)
    obstacles = trajectory["obstacles"]
    obstacle_radii = np.asarray(trajectory["obstacle_radii"], dtype=np.float64)
    target = np.asarray(trajectory["target"], dtype=np.float64)
    start = positions[0]
    world_size = 60.0

    frame_ids = _frame_indices(len(positions), max_frames)
    fig = plt.figure(figsize=(8.4, 7.2), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    ax.scatter([start[0]], [start[1]], [start[2]], c="#2ca02c", s=58, label="Start", depthshade=True)
    ax.scatter([target[0]], [target[1]], [target[2]], c="#d62728", s=90, marker="*", label="Target", depthshade=True)
    trail_line, = ax.plot([], [], [], color="#1f77b4", linewidth=2.2, label="AUV trail")
    auv_point, = ax.plot([], [], [], marker="o", color="#1f77b4", markersize=7, linestyle="")
    projection_line, = ax.plot([], [], [], color="#1f77b4", linewidth=1.0, alpha=0.25, linestyle="--")

    obstacle_scat = None
    if obstacles and len(obstacles[0]) > 0:
        sizes = np.maximum(obstacle_radii, 0.5) * 62.0
        obstacle_scat = ax.scatter(
            obstacles[0][:, 0],
            obstacles[0][:, 1],
            obstacles[0][:, 2],
            s=sizes,
            c="#ff7f0e",
            alpha=0.38,
            edgecolors="#b85c00",
            linewidths=0.5,
            label="Moving obstacles",
            depthshade=True,
        )

    current_quiver = ax.quiver(
        positions[0, 0],
        positions[0, 1],
        positions[0, 2],
        0.0,
        0.0,
        0.0,
        color="#9467bd",
        linewidth=2.0,
        arrow_length_ratio=0.28,
        label="Current",
    )

    # Workspace box outline. Depth z is positive downward, so the z-axis is inverted below.
    corners = np.array(
        [
            [-world_size, -world_size, 0.0],
            [world_size, -world_size, 0.0],
            [world_size, world_size, 0.0],
            [-world_size, world_size, 0.0],
            [-world_size, -world_size, 0.0],
            [-world_size, -world_size, world_size],
            [world_size, -world_size, world_size],
            [world_size, world_size, world_size],
            [-world_size, world_size, world_size],
            [-world_size, -world_size, world_size],
        ],
        dtype=np.float64,
    )
    ax.plot(corners[:5, 0], corners[:5, 1], corners[:5, 2], color="#777777", alpha=0.35, linewidth=0.8)
    ax.plot(corners[5:, 0], corners[5:, 1], corners[5:, 2], color="#777777", alpha=0.35, linewidth=0.8)
    for corner_idx in range(4):
        ax.plot(
            [corners[corner_idx, 0], corners[corner_idx + 5, 0]],
            [corners[corner_idx, 1], corners[corner_idx + 5, 1]],
            [corners[corner_idx, 2], corners[corner_idx + 5, 2]],
            color="#777777",
            alpha=0.25,
            linewidth=0.8,
        )

    ax.set_xlim(-world_size - 5.0, world_size + 5.0)
    ax.set_ylim(-world_size - 5.0, world_size + 5.0)
    ax.set_zlim(world_size, 0.0)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z depth [m]")
    ax.view_init(elev=24.0, azim=-52.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        f"3D Stage 4 Success | seed={trajectory['seed']} | steps={trajectory['steps']} | "
        f"final_dist={trajectory['final_distance']:.2f}m | min_clear={trajectory['min_clearance']:.2f}m",
        fontsize=10,
    )

    def update(frame_number: int):
        nonlocal current_quiver
        idx = int(frame_ids[frame_number])
        pos_slice = positions[: idx + 1]
        trail_line.set_data(pos_slice[:, 0], pos_slice[:, 1])
        trail_line.set_3d_properties(pos_slice[:, 2])
        auv_point.set_data([positions[idx, 0]], [positions[idx, 1]])
        auv_point.set_3d_properties([positions[idx, 2]])
        projection_line.set_data(pos_slice[:, 0], pos_slice[:, 1])
        projection_line.set_3d_properties(np.full(len(pos_slice), world_size))

        if obstacles and obstacle_scat is not None:
            obs_idx = min(idx, len(obstacles) - 1)
            obstacle_scat._offsets3d = (
                obstacles[obs_idx][:, 0],
                obstacles[obs_idx][:, 1],
                obstacles[obs_idx][:, 2],
            )

        current_quiver.remove()
        if currents.ndim == 2 and idx < len(currents):
            current = currents[idx]
            anchor = positions[idx]
            current_quiver = ax.quiver(
                anchor[0],
                anchor[1],
                anchor[2],
                current[0] * 8.0,
                current[1] * 8.0,
                current[2] * 8.0,
                color="#9467bd",
                linewidth=2.0,
                arrow_length_ratio=0.28,
            )
        ax.set_title(f"3D trajectory | frame {frame_number + 1}/{len(frame_ids)}")
        artists = [trail_line, auv_point, projection_line, current_quiver]
        if obstacle_scat is not None:
            artists.append(obstacle_scat)
        return artists

    animation = FuncAnimation(fig, update, frames=len(frame_ids), blit=False, repeat=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def _body_frame_angles(
    point: np.ndarray,
    position: np.ndarray,
    orientation: np.ndarray,
) -> Tuple[float, float, float, np.ndarray]:
    rot = rotation_matrix_body_to_inertial(float(orientation[0]), float(orientation[1]), float(orientation[2]))
    rel_body = rot.T @ (point - position)
    forward = max(float(rel_body[0]), 1e-6)
    lateral = float(rel_body[1])
    vertical_down = float(rel_body[2])
    distance = float(np.linalg.norm(rel_body))
    bearing_deg = float(np.degrees(np.arctan2(lateral, forward)))
    elevation_deg = float(-np.degrees(np.arctan2(vertical_down, max(np.linalg.norm(rel_body[:2]), 1e-6))))
    return bearing_deg, elevation_deg, distance, rel_body


def save_first_person_gif(trajectory: Dict[str, Any], path: Path, fps: int, max_frames: int, dpi: int) -> None:
    positions = np.asarray(trajectory["positions"], dtype=np.float64)
    orientations = np.asarray(trajectory["orientations"], dtype=np.float64)
    currents = np.asarray(trajectory["currents"], dtype=np.float64)
    obstacles = trajectory["obstacles"]
    obstacle_radii = np.asarray(trajectory["obstacle_radii"], dtype=np.float64)
    target = np.asarray(trajectory["target"], dtype=np.float64)

    frame_ids = _frame_indices(len(positions), max_frames)
    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=dpi)
    ax.set_facecolor("#f8fbff")
    ax.axhline(0.0, color="#444444", linewidth=0.9, alpha=0.55)
    ax.axvline(0.0, color="#444444", linewidth=0.9, alpha=0.55)
    ax.plot([-6.0, 6.0], [0.0, 0.0], color="#1f77b4", linewidth=2.0, alpha=0.7)
    ax.plot([0.0, 0.0], [-4.0, 4.0], color="#1f77b4", linewidth=2.0, alpha=0.7)
    ax.set_xlim(-75.0, 75.0)
    ax.set_ylim(-45.0, 45.0)
    ax.grid(True, alpha=0.22)
    ax.set_xlabel("bearing in AUV frame [deg]")
    ax.set_ylabel("elevation [deg, up positive]")

    target_scatter = ax.scatter([], [], c="#d62728", s=160, marker="*", label="Target", zorder=5)
    obstacle_scatter = ax.scatter([], [], c="#ff7f0e", alpha=0.45, edgecolors="#b85c00", label="Obstacles", zorder=4)
    current_quiver = ax.quiver(
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        color="#9467bd",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.006,
        label="Current",
        zorder=6,
    )
    info_text = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.82},
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(
        f"AUV First-Person View | seed={trajectory['seed']} | "
        f"steps={trajectory['steps']} | final_dist={trajectory['final_distance']:.2f}m",
        fontsize=10,
    )

    def update(frame_number: int):
        idx = int(frame_ids[frame_number])
        position = positions[idx]
        orientation = orientations[min(idx, len(orientations) - 1)]

        target_bearing, target_elevation, target_distance, target_body = _body_frame_angles(
            target,
            position,
            orientation,
        )
        if target_body[0] > 0.1:
            target_scatter.set_offsets([[target_bearing, target_elevation]])
        else:
            target_scatter.set_offsets([[np.nan, np.nan]])

        obstacle_offsets: List[List[float]] = []
        obstacle_sizes: List[float] = []
        if obstacles:
            obs_idx = min(idx, len(obstacles) - 1)
            for center, radius in zip(obstacles[obs_idx], obstacle_radii):
                bearing, elevation, distance, rel_body = _body_frame_angles(center, position, orientation)
                if rel_body[0] <= 0.1:
                    continue
                if abs(bearing) > 90.0 or abs(elevation) > 60.0:
                    continue
                angular_radius = np.degrees(np.arctan2(float(radius) + 0.25, max(distance, 1e-6)))
                obstacle_offsets.append([bearing, elevation])
                obstacle_sizes.append(float(np.clip((angular_radius * 9.0) ** 2, 18.0, 800.0)))

        if obstacle_offsets:
            obstacle_scatter.set_offsets(np.asarray(obstacle_offsets, dtype=np.float64))
            obstacle_scatter.set_sizes(np.asarray(obstacle_sizes, dtype=np.float64))
        else:
            obstacle_scatter.set_offsets(np.empty((0, 2), dtype=np.float64))
            obstacle_scatter.set_sizes(np.empty((0,), dtype=np.float64))

        if currents.ndim == 2 and idx < len(currents):
            rot = rotation_matrix_body_to_inertial(float(orientation[0]), float(orientation[1]), float(orientation[2]))
            current_body = rot.T @ currents[idx]
            current_bearing = np.degrees(np.arctan2(current_body[1], max(abs(current_body[0]), 1e-6)))
            current_elevation = -np.degrees(
                np.arctan2(current_body[2], max(np.linalg.norm(current_body[:2]), 1e-6))
            )
            norm = max(float(np.hypot(current_bearing, current_elevation)), 1e-6)
            arrow_scale = min(float(np.linalg.norm(current_body)) * 18.0, 18.0)
            current_quiver.set_UVC([current_bearing / norm * arrow_scale], [current_elevation / norm * arrow_scale])

        info_text.set_text(
            f"frame {frame_number + 1}/{len(frame_ids)}\n"
            f"target range {target_distance:.1f} m\n"
            f"bearing {target_bearing:.1f} deg | elev {target_elevation:.1f} deg\n"
            f"yaw {np.degrees(orientation[2]):.1f} deg"
        )
        ax.set_title("Body-frame camera projection")
        return [target_scatter, obstacle_scatter, current_quiver, info_text]

    animation = FuncAnimation(fig, update, frames=len(frame_ids), blit=False, repeat=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def write_summary(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = ["gif", "first_person_gif", "seed", "target", "steps", "final_distance", "min_clearance", "event"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render GIFs for successful REMUS stage episodes.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints_transformer_remus_v4/transformer_ppo_best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("success_gifs_stage4"))
    parser.add_argument("--stage", type=int, default=4)
    parser.add_argument("--num-gifs", type=int, default=10)
    parser.add_argument("--max-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed-start", type=int, default=50_000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=140)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--no-guided-action-prior", action="store_true")
    parser.add_argument("--no-first-person", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    set_seed(args.seed)
    trainer = build_trainer(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    successes = 0
    for trial_idx in range(args.max_trials):
        eval_seed = args.seed + args.eval_seed_start + trial_idx
        trajectory = rollout_episode(trainer, eval_seed, args.sample_stride)
        status = "success" if trajectory["success"] else trajectory["event"]
        print(
            f"[{trial_idx + 1:04d}/{args.max_trials:04d}] seed={eval_seed} "
            f"event={status} steps={trajectory['steps']} final_dist={trajectory['final_distance']:.2f}"
        )
        if not trajectory["success"]:
            continue

        successes += 1
        gif_path = args.output_dir / f"success_{successes:02d}_seed_{eval_seed}_3d.gif"
        fpv_path = args.output_dir / f"success_{successes:02d}_seed_{eval_seed}_fpv.gif"
        save_trajectory_gif(trajectory, gif_path, fps=args.fps, max_frames=args.max_frames, dpi=args.dpi)
        print(f"  saved {gif_path}")
        if not args.no_first_person:
            save_first_person_gif(trajectory, fpv_path, fps=args.fps, max_frames=args.max_frames, dpi=args.dpi)
            print(f"  saved {fpv_path}")
        summaries.append(
            {
                "gif": gif_path.name,
                "first_person_gif": "" if args.no_first_person else fpv_path.name,
                "seed": eval_seed,
                "target": np.asarray(trajectory["target"], dtype=np.float64).round(3).tolist(),
                "steps": trajectory["steps"],
                "final_distance": trajectory["final_distance"],
                "min_clearance": trajectory["min_clearance"],
                "event": trajectory["event"],
            }
        )
        if successes >= args.num_gifs:
            break

    write_summary(args.output_dir / "summary.csv", summaries)
    if successes < args.num_gifs:
        raise RuntimeError(f"Only found {successes}/{args.num_gifs} successful episodes in {args.max_trials} trials.")
    print(f"Done. Saved {successes} GIFs to {args.output_dir}")


if __name__ == "__main__":
    main()
