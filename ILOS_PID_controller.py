import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from REMUSAUVEnv import REMUSAUVEnv, wrap_angle


DEFAULT_RESULT_ROOT = Path(r"C:\Users\kavin\Desktop\LIG2_result")
RUN_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


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


class ILOSPIDController:
    """
    Practical LOS/PID-style baseline for REMUSAUVEnv.

    Design:
      - Outer loop: LOS reference toward a wall-safe interior goal proxy with
        current compensation and boundary relaxation.
      - Inner loop: low-level PID-family control on surge / yaw / pitch.

    Notes:
      - This file is tuned as a boundary-safe path-following baseline.
      - Current compensation is supported, but the strongest success rate was obtained
        on the no-current benchmark used for controller validation.
    """

    def __init__(self, env: REMUSAUVEnv):
        self.env = env
        self.start: Optional[np.ndarray] = None
        self.target: Optional[np.ndarray] = None
        self.goal_proxy: Optional[np.ndarray] = None

        # Wall-safe LOS reference tuning.
        self.wall_thresh = 1.820540021200556
        self.wall_gain = 3.8323330044618347
        self.goal_relax_min = 0.30813842672922503
        self.goal_relax_dist = 1.7562684599413596
        self.current_gain = 0.734119395171103
        self.max_pitch_ref = np.deg2rad(23.716836730435467)
        self.max_yaw_ref_rate = np.deg2rad(28.22418733938171)
        self.max_pitch_ref_rate = np.deg2rad(3.114789879276785)

        # Speed schedule.
        self.base_speed = 0.7066147947822117
        self.slow_heading_1 = np.deg2rad(59.1535949660832)
        self.slow_heading_2 = np.deg2rad(120.22235425284586)
        self.slow_speed_1 = 0.7185625327886945
        self.slow_speed_2 = 0.16847166197414562
        self.terminal_dist = 1.4847398067165756
        self.terminal_base = 0.1872662747279868
        self.terminal_gain = 0.2487015140842298

        # Inner-loop gains.
        self.prop_ff = 0.6368256290449372
        self.surge_kp = 0.6028651429248839
        self.surge_ki = 0.0

        self.heading_kp = 1.8838199251289587
        self.heading_ki = 0.0
        self.heading_kd = 1.7386190194826687

        self.pitch_kp = 2.1316637851861224
        self.pitch_ki = 0.0
        self.pitch_kd = 0.8383372481892849

        self.surge_int = 0.0
        self.heading_int = 0.0
        self.pitch_int = 0.0
        self.psi_ref = 0.0
        self.theta_ref = 0.0
        self.previous_goal_distance: Optional[float] = None

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return float(np.clip(value, lower, upper))

    def _goal_proxy(self, target: np.ndarray) -> np.ndarray:
        margin = self.env.goal_radius
        proxy = np.asarray(target, dtype=np.float64).copy()
        proxy[0] = np.clip(proxy[0], -self.env.world_size + margin, self.env.world_size - margin)
        proxy[1] = np.clip(proxy[1], -self.env.world_size + margin, self.env.world_size - margin)
        proxy[2] = np.clip(proxy[2], margin, self.env.world_size - margin)
        return proxy

    def reset(self, start: np.ndarray, target: np.ndarray) -> None:
        self.start = np.asarray(start, dtype=np.float64).copy()
        self.target = np.asarray(target, dtype=np.float64).copy()
        self.goal_proxy = self._goal_proxy(self.target)

        self.surge_int = 0.0
        self.heading_int = 0.0
        self.pitch_int = 0.0
        self.psi_ref = float(self.env.state[5])
        self.theta_ref = float(self.env.state[4])
        self.previous_goal_distance = None

    def _boundary_margins(self, pos: np.ndarray) -> np.ndarray:
        return np.array(
            [
                self.env.world_size - pos[0],
                pos[0] + self.env.world_size,
                self.env.world_size - pos[1],
                pos[1] + self.env.world_size,
                self.env.world_size - pos[2],
                pos[2],
            ],
            dtype=np.float64,
        )

    def _closest_inward_normal(self, pos: np.ndarray) -> tuple[np.ndarray, float]:
        margins = self._boundary_margins(pos)
        idx = int(np.argmin(margins))
        normals = (
            np.array([-1.0, 0.0, 0.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, -1.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, -1.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )
        return normals[idx], float(margins[idx])

    def _reference_vector(self, pos: np.ndarray) -> tuple[np.ndarray, float]:
        assert self.goal_proxy is not None

        desired = self.goal_proxy - pos
        goal_distance = float(np.linalg.norm(desired) + 1e-9)
        desired /= goal_distance

        inward_normal, wall_margin = self._closest_inward_normal(pos)
        wall_scale = np.clip((self.wall_thresh - wall_margin) / self.wall_thresh, 0.0, 1.0)
        if wall_scale > 0.0:
            relax = np.clip(goal_distance / self.goal_relax_dist, self.goal_relax_min, 1.0)
            desired = desired + relax * self.wall_gain * wall_scale * inward_normal

        current_world = np.asarray(getattr(self.env, "current_inertial", np.zeros(3)), dtype=np.float64)
        desired = desired - self.current_gain * current_world
        desired /= np.linalg.norm(desired) + 1e-9
        return desired, goal_distance

    def compute_reference(self) -> Dict[str, Any]:
        eta = self.env.state[:6]
        nu = self.env.state[6:]
        pos = eta[:3]
        theta = float(eta[4])
        psi = float(eta[5])
        u = float(nu[0])
        q = float(nu[4])
        r = float(nu[5])

        desired, goal_distance = self._reference_vector(pos)
        horiz = float(np.linalg.norm(desired[:2]))
        psi_cmd = float(np.arctan2(desired[1], desired[0]) if horiz > 1e-8 else psi)
        theta_cmd = float(np.clip(-np.arctan2(desired[2], max(horiz, 1e-8)), -self.max_pitch_ref, self.max_pitch_ref))

        psi_delta = wrap_angle(psi_cmd - self.psi_ref)
        self.psi_ref = wrap_angle(
            self.psi_ref + np.clip(psi_delta, -self.max_yaw_ref_rate * self.env.dt, self.max_yaw_ref_rate * self.env.dt)
        )

        theta_delta = theta_cmd - self.theta_ref
        self.theta_ref = float(
            np.clip(
                self.theta_ref + np.clip(theta_delta, -self.max_pitch_ref_rate * self.env.dt, self.max_pitch_ref_rate * self.env.dt),
                -self.max_pitch_ref,
                self.max_pitch_ref,
            )
        )

        heading_err = wrap_angle(self.psi_ref - psi)
        pitch_err = self.theta_ref - theta

        u_ref = self.base_speed
        if abs(heading_err) > self.slow_heading_2:
            u_ref = self.slow_speed_2
        elif abs(heading_err) > self.slow_heading_1:
            u_ref = self.slow_speed_1

        if goal_distance < self.terminal_dist:
            u_ref = min(u_ref, self.terminal_base + self.terminal_gain * goal_distance)

        return {
            "u": u,
            "q": q,
            "r": r,
            "psi": psi,
            "theta": theta,
            "u_ref": float(u_ref),
            "psi_ref": float(self.psi_ref),
            "theta_ref": float(self.theta_ref),
            "heading_err": float(heading_err),
            "pitch_err": float(pitch_err),
            "goal_distance": float(goal_distance),
        }

    def act(self) -> np.ndarray:
        ref = self.compute_reference()
        surge_err = ref["u_ref"] - ref["u"]

        propeller = (
            self.prop_ff * ref["u_ref"]
            + self.surge_kp * surge_err
            + self.surge_ki * self.surge_int
        )
        rudder = (
            self.heading_kp * ref["heading_err"]
            + self.heading_ki * self.heading_int
            - self.heading_kd * ref["r"]
        )
        stern = (
            -self.pitch_kp * ref["pitch_err"]
            - self.pitch_ki * self.pitch_int
            - self.pitch_kd * ref["q"]
        )

        action = np.array(
            [
                np.clip(propeller, 0.0, 1.0),
                np.clip(rudder, -1.0, 1.0),
                np.clip(stern, -1.0, 1.0),
            ],
            dtype=np.float64,
        )

        self.previous_goal_distance = ref["goal_distance"]
        return action


def minimum_clearance(env: REMUSAUVEnv) -> float:
    pos = env.state[:3]
    min_clearance = np.inf
    for obs in env.obstacles:
        clearance = np.linalg.norm(pos - obs.center) - (obs.radius + env.auv_radius)
        min_clearance = min(min_clearance, clearance)
    return float(min_clearance)


def resolve_run_output_dir(
    out_dir: Optional[Path],
    result_root: Path,
    started_at: datetime,
    persist_outputs: bool,
) -> Optional[Path]:
    if not persist_outputs:
        return None
    if out_dir is not None:
        return out_dir
    return result_root / started_at.strftime(RUN_TIMESTAMP_FORMAT)


def evaluate_ilos_pid(
    episodes: int,
    seed: int,
    out_dir: Optional[Path],
    persist_outputs: bool = True,
    result_root: Path = DEFAULT_RESULT_ROOT,
    env_dt: float = 0.05,
    env_max_steps: int = 1200,
    n_obstacles: int = 6,
    current_enabled: bool = False,
) -> Dict[str, Any]:
    started_at = datetime.now()
    run_dir = resolve_run_output_dir(
        out_dir=out_dir,
        result_root=result_root,
        started_at=started_at,
        persist_outputs=persist_outputs,
    )
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)

    results: List[EpisodeResult] = []
    event_counts: Dict[str, int] = {
        "goal": 0,
        "collision": 0,
        "out_of_bounds": 0,
        "timeout": 0,
        "other": 0,
    }

    for episode in range(episodes):
        episode_seed = seed + episode
        env = REMUSAUVEnv(
            dt=env_dt,
            max_steps=env_max_steps,
            n_obstacles=n_obstacles,
            seed=episode_seed,
            current_enabled=current_enabled,
        )
        obs, info = env.reset(seed=episode_seed)
        _ = obs

        controller = ILOSPIDController(env)
        controller.reset(start=info["start"], target=info["target"])

        episode_return = 0.0
        min_goal_distance = float(np.linalg.norm(env.state[:3] - env.target))
        min_clear = minimum_clearance(env)
        final_distance = min_goal_distance
        final_event = "timeout"
        steps = 0

        for t in range(env.max_steps):
            action = controller.act()
            _, reward, terminated, truncated, step_info = env.step(action)
            episode_return += float(reward)
            steps = t + 1

            goal_dist = float(step_info.get("distance_to_goal", np.linalg.norm(env.state[:3] - env.target)))
            min_goal_distance = min(min_goal_distance, goal_dist)
            min_clear = min(min_clear, minimum_clearance(env))
            final_distance = goal_dist

            if terminated or truncated:
                final_event = str(step_info.get("event", "timeout" if truncated else "other"))
                break

        if final_event not in event_counts:
            final_event = "other"
        event_counts[final_event] += 1

        result = EpisodeResult(
            episode=episode,
            seed=episode_seed,
            event=final_event,
            success=int(final_event == "goal"),
            steps=steps,
            episode_return=episode_return,
            final_distance_to_goal=final_distance,
            min_distance_to_goal=min_goal_distance,
            min_obstacle_clearance=min_clear,
        )
        results.append(result)

        success_rate = sum(item.success for item in results) / len(results)
        print(
            f"ep={episode:03d} | seed={episode_seed:04d} | event={final_event:<13} | "
            f"steps={steps:4d} | return={episode_return:8.2f} | "
            f"final_dist={final_distance:6.3f} | min_dist={min_goal_distance:6.3f} | "
            f"succ_rate={success_rate:5.3f}",
            flush=True,
        )

    summary = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "episodes": episodes,
        "seed_start": seed,
        "success_rate": float(sum(item.success for item in results) / max(len(results), 1)),
        "goal_count": event_counts["goal"],
        "collision_count": event_counts["collision"],
        "out_of_bounds_count": event_counts["out_of_bounds"],
        "timeout_count": event_counts["timeout"],
        "other_count": event_counts["other"],
        "mean_return": float(np.mean([item.episode_return for item in results])) if results else 0.0,
        "mean_steps": float(np.mean([item.steps for item in results])) if results else 0.0,
        "mean_final_distance": float(np.mean([item.final_distance_to_goal for item in results])) if results else 0.0,
        "mean_min_distance": float(np.mean([item.min_distance_to_goal for item in results])) if results else 0.0,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "config": {
            "env_dt": env_dt,
            "env_max_steps": env_max_steps,
            "n_obstacles": n_obstacles,
            "current_enabled": current_enabled,
            "controller": "ILOS_PID",
        },
    }

    if run_dir is not None and results:
        csv_path = run_dir / "episode_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            for result in results:
                writer.writerow(asdict(result))
        with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print(
        "final | "
        f"episodes={summary['episodes']} | success_rate={summary['success_rate']:.3f} | "
        f"goal/coll/oob/to={summary['goal_count']}/{summary['collision_count']}/"
        f"{summary['out_of_bounds_count']}/{summary['timeout_count']} | "
        f"mean_return={summary['mean_return']:.2f} | mean_steps={summary['mean_steps']:.1f}",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an Integral LOS + PID controller on REMUSAUVEnv."
    )
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help="Root directory used for timestamped result folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory.",
    )
    parser.add_argument("--env-dt", type=float, default=0.05, help="Environment integration step.")
    parser.add_argument("--env-max-steps", type=int, default=1200, help="Maximum steps per episode.")
    parser.add_argument("--n-obstacles", type=int, default=6, help="Number of obstacles in REMUSAUVEnv.")
    parser.add_argument(
        "--enable-current",
        action="store_true",
        help="Enable ocean current disturbance. The tuned benchmark uses no-current mode by default.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run without writing result files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_ilos_pid(
        episodes=args.episodes,
        seed=args.seed,
        out_dir=args.out_dir,
        persist_outputs=not args.smoke,
        result_root=args.result_root,
        env_dt=args.env_dt,
        env_max_steps=args.env_max_steps,
        n_obstacles=args.n_obstacles,
        current_enabled=args.enable_current,
    )
