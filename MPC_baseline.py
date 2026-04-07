import argparse
import csv
import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from REMUSAUVEnv import REMUSAUVEnv, wrap_angle


@dataclass
class EpisodeResult:
    episode: int
    seed: int
    event: str
    success: int
    steps: int
    episode_return: float
    final_distance_to_goal: float
    min_distance_to_goal: float
    min_obstacle_clearance: float


class GuidanceField:
    """
    Goal-attractive / obstacle-repulsive guidance used by the MPC.

    The environment itself is not modified. This only computes a short-horizon
    desired heading, desired pitch, and desired cruise speed from the current
    state, target, obstacles, boundaries, and (when available) ocean current.
    """

    def __init__(self, env: REMUSAUVEnv):
        self.env = env

        # Guidance tuning
        self.influence_radius = 2.5
        self.attractive_gain = 1.5
        self.repulsive_gain = 1.5
        self.vertical_repulsive_scale = 0.7
        self.boundary_margin = 1.2
        self.boundary_gain_xy = 2.0
        self.boundary_gain_z = 1.5
        self.current_comp_gain = 1.0
        self.max_pitch_ref = np.deg2rad(18.0)

    def _get_current_world(self) -> np.ndarray:
        current = getattr(self.env, "current_inertial", None)
        if current is None:
            return np.zeros(3, dtype=np.float64)
        return np.asarray(current, dtype=np.float64)

    def _minimum_clearance(self, pos: np.ndarray) -> float:
        min_clearance = np.inf
        for obs in self.env.obstacles:
            clearance = np.linalg.norm(pos - obs.center) - (obs.radius + self.env.auv_radius)
            min_clearance = min(min_clearance, clearance)
        return float(min_clearance)

    def compute(self) -> Dict[str, Any]:
        eta = self.env.state[:6]
        nu = self.env.state[6:]
        pos = eta[:3]
        psi = eta[5]
        theta = eta[4]
        u, v, w, _, q, r = nu

        target_vec = self.env.target - pos
        target_dist = float(np.linalg.norm(target_vec) + 1e-9)
        attractive = target_vec / target_dist

        repulsive = np.zeros(3, dtype=np.float64)
        for obs in self.env.obstacles:
            rel = pos - obs.center
            dist_center = np.linalg.norm(rel) + 1e-9
            clearance = dist_center - (obs.radius + self.env.auv_radius)
            if clearance < self.influence_radius:
                unit = rel / dist_center
                effective_clearance = max(clearance, 0.05)
                strength = self.repulsive_gain * (
                    (1.0 / effective_clearance - 1.0 / self.influence_radius)
                    / (effective_clearance ** 2)
                )
                repulsive += strength * np.array(
                    [unit[0], unit[1], self.vertical_repulsive_scale * unit[2]],
                    dtype=np.float64,
                )

        # Boundary repulsion so that the controller does not hug the walls.
        for axis in (0, 1):
            if pos[axis] > self.env.world_size - self.boundary_margin:
                repulsive[axis] -= self.boundary_gain_xy / (self.env.world_size - pos[axis] + 0.08)
            if pos[axis] < -self.env.world_size + self.boundary_margin:
                repulsive[axis] += self.boundary_gain_xy / (pos[axis] + self.env.world_size + 0.08)

        if pos[2] < self.boundary_margin:
            repulsive[2] += self.boundary_gain_z / (pos[2] + 0.08)
        if pos[2] > self.env.world_size - self.boundary_margin:
            repulsive[2] -= self.boundary_gain_z / (self.env.world_size - pos[2] + 0.08)

        desired_world = self.attractive_gain * attractive + repulsive

        # If the environment exposes a current estimate, bias the desired
        # direction slightly against it.
        current_world = self._get_current_world()
        desired_world = desired_world - self.current_comp_gain * current_world

        horiz = float(np.linalg.norm(desired_world[:2]))
        psi_ref = float(np.arctan2(desired_world[1], desired_world[0]) if horiz > 1e-6 else psi)
        theta_ref = float(-np.arctan2(desired_world[2], max(horiz, 1e-6)))
        theta_ref = float(np.clip(theta_ref, -self.max_pitch_ref, self.max_pitch_ref))

        u_ref = float(np.clip(0.6 + 0.08 * target_dist, 0.30, 1.25))

        heading_abs_err = abs(wrap_angle(psi_ref - psi))
        if heading_abs_err > np.deg2rad(70.0):
            u_ref *= 0.35
        elif heading_abs_err > np.deg2rad(40.0):
            u_ref *= 0.60

        min_clearance = self._minimum_clearance(pos)
        if min_clearance < 1.2:
            u_ref *= float(np.clip((min_clearance - 0.02) / 1.18, 0.18, 1.0))

        heading_err = wrap_angle(psi_ref - psi)
        pitch_err = theta_ref - theta

        nominal = np.array(
            [
                np.clip(1.05 * (u_ref - u) + 0.15 * np.cos(heading_err), -0.3, 1.0),
                np.clip(1.25 * heading_err - 0.35 * r + 0.08 * v, -1.0, 1.0),
                np.clip(1.80 * (-pitch_err) + 0.30 * w + 0.35 * q, -1.0, 1.0),
            ],
            dtype=np.float64,
        )

        return {
            "psi_ref": psi_ref,
            "theta_ref": theta_ref,
            "u_ref": u_ref,
            "nominal_action": nominal,
            "target_distance": target_dist,
            "min_clearance": min_clearance,
            "current_world": current_world,
        }


class ShootingMPCController:
    """
    A lightweight shooting-based MPC (CEM-style) for the REMUS environment.

    - The real environment is left unchanged.
    - The controller uses a short reduced-order predictive model:
      [surge speed, yaw, pitch, yaw rate, pitch rate]
    - Obstacles and current are handled in the outer guidance layer.
    - The optimizer samples candidate action sequences around a warm start,
      evaluates a finite-horizon quadratic objective, then applies the first
      action of the best sequence in receding-horizon fashion.
    """

    def __init__(
        self,
        env: REMUSAUVEnv,
        horizon: int = 6,
        plan_dt: float = 0.15,
        candidates: int = 24,
        elites: int = 6,
        replan_interval: int = 4,
        seed: Optional[int] = None,
    ):
        self.env = env
        self.horizon = horizon
        self.plan_dt = plan_dt
        self.candidates = candidates
        self.elites = elites
        self.replan_interval = replan_interval
        self.guidance = GuidanceField(env)
        self.rng = np.random.default_rng(seed)

        self.mean = np.zeros((self.horizon, 3), dtype=np.float64)
        self.std = np.tile(np.array([0.18, 0.22, 0.22], dtype=np.float64), (self.horizon, 1))
        self.last_action = np.zeros(3, dtype=np.float64)
        self.cached_action = np.zeros(3, dtype=np.float64)
        self.replan_counter = 0

        # Reduced-order model coefficients.
        self.u_lin_drag = 0.34
        self.u_quad_drag = 0.10
        self.yaw_gain = 1.55
        self.yaw_damping = 0.98
        self.yaw_quad = 0.12
        self.pitch_gain = 1.90
        self.pitch_damping = 1.05
        self.pitch_quad = 0.08
        self.min_effective_speed = 0.35
        self.max_pitch = np.deg2rad(22.0)

        # Cost weights.
        self.w_speed = 3.0
        self.w_heading = 18.0
        self.w_pitch = 14.0
        self.w_yaw_rate = 0.7
        self.w_pitch_rate = 0.7
        self.w_act = np.array([0.08, 0.05, 0.05], dtype=np.float64)
        self.w_delta = np.array([0.15, 0.10, 0.10], dtype=np.float64)
        self.w_terminal_speed = 4.0
        self.w_terminal_heading = 22.0
        self.w_terminal_pitch = 18.0

    def reset(self) -> None:
        self.mean.fill(0.0)
        self.std[:] = np.array([0.18, 0.22, 0.22], dtype=np.float64)
        self.last_action.fill(0.0)
        self.cached_action.fill(0.0)
        self.replan_counter = 0

    def _simulate_reduced_model(self, x: np.ndarray, action: np.ndarray) -> np.ndarray:
        u, psi, theta, r, q = x
        prop, rudder, stern = action

        dt = self.plan_dt
        effective_u = max(abs(u), self.min_effective_speed)

        u = u + dt * (1.15 * prop - self.u_lin_drag * u - self.u_quad_drag * u * abs(u))
        r = r + dt * (
            self.yaw_gain * effective_u * rudder
            - self.yaw_damping * r
            - self.yaw_quad * r * abs(r)
        )
        psi = wrap_angle(psi + dt * r)

        q = q + dt * (
            -self.pitch_gain * effective_u * stern
            - self.pitch_damping * q
            - self.pitch_quad * q * abs(q)
        )
        theta = np.clip(theta + dt * q, -self.max_pitch, self.max_pitch)

        return np.array([u, psi, theta, r, q], dtype=np.float64)

    def _sequence_cost(self, sequence: np.ndarray, ref: Dict[str, Any]) -> float:
        eta = self.env.state[:6]
        nu = self.env.state[6:]
        x = np.array([nu[0], eta[5], eta[4], nu[5], nu[4]], dtype=np.float64)

        cost = 0.0
        prev_action = self.last_action
        for action in sequence:
            x = self._simulate_reduced_model(x, action)
            heading_err = wrap_angle(ref["psi_ref"] - x[1])
            pitch_err = ref["theta_ref"] - x[2]
            delta = action - prev_action

            cost += self.w_speed * (x[0] - ref["u_ref"]) ** 2
            cost += self.w_heading * heading_err ** 2
            cost += self.w_pitch * pitch_err ** 2
            cost += self.w_yaw_rate * x[3] ** 2
            cost += self.w_pitch_rate * x[4] ** 2
            cost += float(np.sum(self.w_act * (action ** 2)))
            cost += float(np.sum(self.w_delta * (delta ** 2)))

            prev_action = action

        heading_err = wrap_angle(ref["psi_ref"] - x[1])
        pitch_err = ref["theta_ref"] - x[2]
        cost += self.w_terminal_speed * (x[0] - ref["u_ref"]) ** 2
        cost += self.w_terminal_heading * heading_err ** 2
        cost += self.w_terminal_pitch * pitch_err ** 2
        return float(cost)

    def _plan(self) -> np.ndarray:
        ref = self.guidance.compute()
        warm_start = np.vstack([self.mean[1:], ref["nominal_action"][None, :]])

        samples = np.empty((self.candidates + 1, self.horizon, 3), dtype=np.float64)
        samples[0] = warm_start
        samples[1:] = warm_start[None, :, :] + self.std[None, :, :] * self.rng.standard_normal(
            size=(self.candidates, self.horizon, 3)
        )

        samples[:, :, 0] = np.clip(samples[:, :, 0], -0.3, 1.0)
        samples[:, :, 1:] = np.clip(samples[:, :, 1:], -1.0, 1.0)

        costs = np.array([self._sequence_cost(seq, ref) for seq in samples], dtype=np.float64)
        elite_idx = np.argsort(costs)[: self.elites]
        elites = samples[elite_idx]

        self.mean = elites[0].copy()
        self.std = np.maximum(
            elites.std(axis=0) + np.array([0.02, 0.03, 0.03], dtype=np.float64),
            np.array([0.04, 0.05, 0.05], dtype=np.float64),
        )

        action = self.mean[0].copy()
        self.cached_action = action
        self.last_action = action.copy()
        self.replan_counter = self.replan_interval
        return action

    def act(self) -> np.ndarray:
        if self.replan_counter <= 0:
            return self._plan()
        self.replan_counter -= 1
        return self.cached_action.copy()


class MPCDemoRecorder:
    def __init__(self) -> None:
        self.episodes: List[Dict[str, Any]] = []

    def add_episode(self, episode_dict: Dict[str, Any]) -> None:
        self.episodes.append(episode_dict)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.episodes, f)


def minimum_clearance(env: REMUSAUVEnv) -> float:
    pos = env.state[:3]
    min_clearance = np.inf
    for obs in env.obstacles:
        clearance = np.linalg.norm(pos - obs.center) - (obs.radius + env.auv_radius)
        min_clearance = min(min_clearance, clearance)
    return float(min_clearance)


def evaluate_mpc(
    episodes: int,
    seed: int,
    out_dir: Path,
    save_success_demos: bool = True,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    episode_csv = out_dir / "mpc_episode_summary.csv"
    aggregate_json = out_dir / "mpc_aggregate_summary.json"
    demo_path = out_dir / "mpc_success_demos.pkl"

    recorder = MPCDemoRecorder() if save_success_demos else None

    results: List[EpisodeResult] = []
    event_counts: Dict[str, int] = {"goal": 0, "collision": 0, "out_of_bounds": 0, "timeout": 0, "other": 0}

    for ep in range(episodes):
        env_seed = seed + ep
        env = REMUSAUVEnv(seed=env_seed)
        obs, info = env.reset(seed=env_seed)
        controller = ShootingMPCController(env=env, seed=env_seed)
        controller.reset()

        episode_return = 0.0
        min_goal_dist = float(np.linalg.norm(env.state[:3] - env.target))
        min_clear = minimum_clearance(env)

        traj_obs: List[np.ndarray] = [np.asarray(obs, dtype=np.float32)]
        traj_actions: List[np.ndarray] = []
        traj_rewards: List[float] = []
        traj_infos: List[Dict[str, Any]] = [info]

        final_event = "timeout"
        final_distance = min_goal_dist
        steps = 0

        for t in range(env.max_steps):
            action = controller.act()
            obs, reward, terminated, truncated, step_info = env.step(action)
            episode_return += float(reward)
            steps = t + 1

            goal_dist = float(step_info.get("distance_to_goal", np.linalg.norm(env.state[:3] - env.target)))
            min_goal_dist = min(min_goal_dist, goal_dist)
            min_clear = min(min_clear, minimum_clearance(env))

            traj_actions.append(np.asarray(action, dtype=np.float32))
            traj_rewards.append(float(reward))
            traj_obs.append(np.asarray(obs, dtype=np.float32))
            traj_infos.append(step_info)

            if terminated or truncated:
                final_event = str(step_info.get("event", "timeout" if truncated else "other"))
                final_distance = goal_dist
                break

        success = int(final_event == "goal")
        if final_event not in event_counts:
            final_event = "other"
        event_counts[final_event] += 1

        result = EpisodeResult(
            episode=ep,
            seed=env_seed,
            event=final_event,
            success=success,
            steps=steps,
            episode_return=episode_return,
            final_distance_to_goal=final_distance,
            min_distance_to_goal=min_goal_dist,
            min_obstacle_clearance=min_clear,
        )
        results.append(result)

        if recorder is not None and success:
            recorder.add_episode(
                {
                    "seed": env_seed,
                    "start": np.asarray(info.get("start"), dtype=np.float32),
                    "target": np.asarray(info.get("target"), dtype=np.float32),
                    "observations": np.asarray(traj_obs, dtype=np.float32),
                    "actions": np.asarray(traj_actions, dtype=np.float32),
                    "rewards": np.asarray(traj_rewards, dtype=np.float32),
                    "infos": traj_infos,
                    "final_event": final_event,
                }
            )

        success_rate = sum(r.success for r in results) / len(results)
        print(
            f"ep={ep:03d} | seed={env_seed:04d} | event={final_event:<13} | "
            f"steps={steps:4d} | return={episode_return:8.2f} | "
            f"final_dist={final_distance:6.3f} | min_dist={min_goal_dist:6.3f} | "
            f"succ_rate={success_rate:5.3f}",
            flush=True,
        )

    with episode_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    aggregate = {
        "episodes": episodes,
        "seed_start": seed,
        "success_rate": float(sum(r.success for r in results) / max(len(results), 1)),
        "goal_count": event_counts["goal"],
        "collision_count": event_counts["collision"],
        "out_of_bounds_count": event_counts["out_of_bounds"],
        "timeout_count": event_counts["timeout"],
        "other_count": event_counts["other"],
        "mean_return": float(np.mean([r.episode_return for r in results])),
        "mean_steps": float(np.mean([r.steps for r in results])),
        "mean_final_distance": float(np.mean([r.final_distance_to_goal for r in results])),
        "mean_min_distance": float(np.mean([r.min_distance_to_goal for r in results])),
        "success_demo_count": int(sum(r.success for r in results)) if recorder is not None else 0,
        "episode_csv": str(episode_csv),
        "success_demo_path": str(demo_path) if recorder is not None else None,
    }

    with aggregate_json.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    if recorder is not None:
        recorder.dump(demo_path)

    print(
        "final | "
        f"episodes={aggregate['episodes']} | success_rate={aggregate['success_rate']:.3f} | "
        f"goal/coll/oob/to={aggregate['goal_count']}/{aggregate['collision_count']}/"
        f"{aggregate['out_of_bounds_count']}/{aggregate['timeout_count']} | "
        f"mean_return={aggregate['mean_return']:.2f} | mean_steps={aggregate['mean_steps']:.1f}",
        flush=True,
    )

    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a shooting-MPC baseline on REMUSAUVEnv.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./runs/mpc_baseline"),
        help="Directory to save CSV / JSON / demo files.",
    )
    parser.add_argument(
        "--no-save-demos",
        action="store_true",
        help="Do not save successful demonstration trajectories.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_mpc(
        episodes=args.episodes,
        seed=args.seed,
        out_dir=args.out_dir,
        save_success_demos=not args.no_save_demos,
    )
