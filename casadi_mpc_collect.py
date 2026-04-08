import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import casadi as ca
import h5py
import numpy as np

from REMUSAUVEnv import REMUSAUVEnv, wrap_angle


DEFAULT_RESULT_ROOT = Path(r"C:\Users\kavin\Desktop\LIG2_result")
RUN_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


@dataclass
class EpisodeSummary:
    episode: int
    seed: int
    event: str
    success: int
    steps: int
    episode_return: float
    final_distance_to_goal: float
    min_distance_to_goal: float
    min_obstacle_clearance: float


class EasyCollectionREMUSAUVEnv(REMUSAUVEnv):
    """
    Easier collection environment used by the CasADi MPC data generator.

    Differences from REMUSAUVEnv:
      - ocean current disabled
      - no obstacles
      - Coriolis / centripetal coupling removed from integration
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["current_enabled"] = False
        kwargs["n_obstacles"] = 0
        super().__init__(*args, **kwargs)

    def _integrate(self, eta: np.ndarray, nu: np.ndarray, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Easy-mode dynamics: keep drag and restoring terms, but remove
        # Coriolis / centripetal coupling and current-induced relative flow.
        drag = self._drag(nu)
        restoring = self._restoring_force(eta)

        rhs = tau - drag - restoring
        nu_dot = self.M_inv @ rhs
        nu_next = nu + self.dt * nu_dot
        nu_next = np.clip(nu_next, -self.velocity_clip, self.velocity_clip)

        eta_dot = self._kinematics(eta, nu_next)
        eta_next = eta + self.dt * eta_dot
        eta_next[3] = wrap_angle(eta_next[3])
        eta_next[4] = np.clip(wrap_angle(eta_next[4]), -self.theta_limit, self.theta_limit)
        eta_next[5] = wrap_angle(eta_next[5])
        return eta_next, nu_next


class CasadiREMUSMPC:
    """
    Simplified nonlinear MPC for REMUSAUVEnv.

    The predictive model keeps the dominant states for path tracking:
      [x, y, z, psi, theta, u, r, q]
    where
      - position evolves from surge speed and pitch/yaw orientation
      - surge, yaw rate, and pitch rate respond to propeller/rudder/stern inputs
      - ocean current is treated as a constant disturbance over the MPC horizon
    """

    def __init__(
        self,
        env: REMUSAUVEnv,
        horizon: int = 12,
        replan_every: int = 3,
        plan_dt: Optional[float] = None,
        solver_name: str = "ipopt",
    ) -> None:
        self.env = env
        self.horizon = horizon
        self.replan_every = replan_every
        self.plan_dt = float(plan_dt if plan_dt is not None else env.dt * replan_every)
        self.solver_name = solver_name

        self.nx = 8
        self.nu = 3
        self.n_obs = env.n_obstacles
        self.world_margin_xy = 0.30
        self.world_margin_z = 0.35
        self.obstacle_buffer = 0.25

        self.cached_action = np.zeros(self.nu, dtype=np.float64)
        self.last_action = np.zeros(self.nu, dtype=np.float64)
        self.replan_counter = 0
        self.u_guess = np.zeros((self.horizon, self.nu), dtype=np.float64)
        self.x_guess = np.zeros((self.horizon + 1, self.nx), dtype=np.float64)

        self._build_problem()

    def reset(self) -> None:
        self.cached_action.fill(0.0)
        self.last_action.fill(0.0)
        self.replan_counter = 0
        self.u_guess.fill(0.0)
        self.x_guess.fill(0.0)

    def _smooth_positive(self, value: ca.MX) -> ca.MX:
        return 0.5 * (value + ca.sqrt(value * value + 1e-4))

    def _dynamics_symbolic(self, x: ca.MX, u_cmd: ca.MX, current_world: ca.MX) -> ca.MX:
        px, py, pz, psi, theta, surge, yaw_rate, pitch_rate = ca.vertsplit(x)
        prop, rudder, stern = ca.vertsplit(u_cmd)

        dt = self.plan_dt
        u_eff = ca.sqrt(surge * surge + 0.25 ** 2)

        surge_next = surge + dt * (1.35 * prop - 0.42 * surge - 0.10 * surge * ca.fabs(surge))
        yaw_rate_next = yaw_rate + dt * (1.80 * u_eff * rudder - 1.12 * yaw_rate)
        pitch_rate_next = pitch_rate + dt * (-1.95 * u_eff * stern - 1.08 * pitch_rate)

        psi_next = psi + dt * yaw_rate_next
        theta_next = theta + dt * pitch_rate_next
        theta_next = ca.fmin(ca.fmax(theta_next, -float(self.env.theta_limit)), float(self.env.theta_limit))

        surge_forward = self._smooth_positive(surge_next)
        forward_world = ca.vertcat(
            ca.cos(theta_next) * ca.cos(psi_next),
            ca.cos(theta_next) * ca.sin(psi_next),
            -ca.sin(theta_next),
        )
        pos_next = ca.vertcat(px, py, pz) + dt * (surge_forward * forward_world + current_world)

        return ca.vertcat(
            pos_next[0],
            pos_next[1],
            pos_next[2],
            psi_next,
            theta_next,
            surge_next,
            yaw_rate_next,
            pitch_rate_next,
        )

    def _rollout_guess(self, state0: np.ndarray, current_world: np.ndarray) -> np.ndarray:
        rollout = np.zeros((self.horizon + 1, self.nx), dtype=np.float64)
        rollout[0] = state0
        for k in range(self.horizon):
            rollout[k + 1] = self._dynamics_numpy(rollout[k], self.u_guess[k], current_world)
        return rollout

    def _dynamics_numpy(self, x: np.ndarray, u_cmd: np.ndarray, current_world: np.ndarray) -> np.ndarray:
        px, py, pz, psi, theta, surge, yaw_rate, pitch_rate = x
        prop, rudder, stern = u_cmd

        dt = self.plan_dt
        u_eff = np.sqrt(surge * surge + 0.25 ** 2)
        surge_next = surge + dt * (1.35 * prop - 0.42 * surge - 0.10 * surge * abs(surge))
        yaw_rate_next = yaw_rate + dt * (1.80 * u_eff * rudder - 1.12 * yaw_rate)
        pitch_rate_next = pitch_rate + dt * (-1.95 * u_eff * stern - 1.08 * pitch_rate)

        psi_next = psi + dt * yaw_rate_next
        theta_next = np.clip(theta + dt * pitch_rate_next, -self.env.theta_limit, self.env.theta_limit)

        surge_forward = max(surge_next, 0.0)
        forward_world = np.array(
            [
                np.cos(theta_next) * np.cos(psi_next),
                np.cos(theta_next) * np.sin(psi_next),
                -np.sin(theta_next),
            ],
            dtype=np.float64,
        )
        pos_next = np.array([px, py, pz], dtype=np.float64) + dt * (
            surge_forward * forward_world + current_world
        )

        return np.array(
            [
                pos_next[0],
                pos_next[1],
                pos_next[2],
                psi_next,
                theta_next,
                surge_next,
                yaw_rate_next,
                pitch_rate_next,
            ],
            dtype=np.float64,
        )

    def _build_problem(self) -> None:
        self.opti = ca.Opti()
        self.X = self.opti.variable(self.nx, self.horizon + 1)
        self.U = self.opti.variable(self.nu, self.horizon)

        self.p_x0 = self.opti.parameter(self.nx)
        self.p_target = self.opti.parameter(3)
        self.p_current = self.opti.parameter(3)
        self.p_prev_action = self.opti.parameter(self.nu)
        self.p_obs_centers = self.opti.parameter(3, self.n_obs)
        self.p_obs_radii = self.opti.parameter(self.n_obs)

        self.opti.subject_to(self.X[:, 0] == self.p_x0)
        self.opti.subject_to(self.opti.bounded(-0.25, self.U[0, :], 1.0))
        self.opti.subject_to(self.opti.bounded(-1.0, self.U[1, :], 1.0))
        self.opti.subject_to(self.opti.bounded(-1.0, self.U[2, :], 1.0))

        self.opti.subject_to(
            self.opti.bounded(
                -self.env.world_size + self.world_margin_xy,
                self.X[0, :],
                self.env.world_size - self.world_margin_xy,
            )
        )
        self.opti.subject_to(
            self.opti.bounded(
                -self.env.world_size + self.world_margin_xy,
                self.X[1, :],
                self.env.world_size - self.world_margin_xy,
            )
        )
        self.opti.subject_to(
            self.opti.bounded(
                self.world_margin_z,
                self.X[2, :],
                self.env.world_size - self.world_margin_z,
            )
        )
        self.opti.subject_to(
            self.opti.bounded(-float(self.env.theta_limit), self.X[4, :], float(self.env.theta_limit))
        )
        self.opti.subject_to(self.opti.bounded(-0.20, self.X[5, :], 2.50))
        self.opti.subject_to(self.opti.bounded(-1.50, self.X[6, :], 1.50))
        self.opti.subject_to(self.opti.bounded(-1.20, self.X[7, :], 1.20))

        cost = 0
        prev_u = self.p_prev_action
        wall_alpha = 2.6
        obs_alpha = 3.2

        for k in range(self.horizon):
            xk = self.X[:, k]
            uk = self.U[:, k]
            x_next = self._dynamics_symbolic(xk, uk, self.p_current)
            self.opti.subject_to(self.X[:, k + 1] == x_next)

            pos = xk[0:3]
            psi = xk[3]
            theta = xk[4]
            surge = xk[5]
            yaw_rate = xk[6]
            pitch_rate = xk[7]

            target_vec = self.p_target - pos
            dist_sq = ca.sumsqr(target_vec)
            dist = ca.sqrt(dist_sq + 1e-6)
            horiz = ca.sqrt(target_vec[0] ** 2 + target_vec[1] ** 2 + 1e-6)
            psi_ref = ca.atan2(target_vec[1], target_vec[0])
            theta_ref = -ca.atan2(target_vec[2], horiz)
            heading_err = ca.atan2(ca.sin(psi_ref - psi), ca.cos(psi_ref - psi))
            pitch_err = theta_ref - theta
            surge_ref = 0.45 + 0.35 * ca.tanh(0.18 * dist)

            cost += 5.0 * dist_sq
            cost += 10.0 * (1.0 - ca.cos(heading_err))
            cost += 8.0 * pitch_err ** 2
            cost += 1.8 * (surge - surge_ref) ** 2
            cost += 0.5 * yaw_rate ** 2 + 0.5 * pitch_rate ** 2
            cost += 0.07 * uk[0] ** 2 + 0.04 * uk[1] ** 2 + 0.04 * uk[2] ** 2
            cost += 0.10 * ca.sumsqr(uk - prev_u)

            margins = [
                self.env.world_size - pos[0],
                pos[0] + self.env.world_size,
                self.env.world_size - pos[1],
                pos[1] + self.env.world_size,
                pos[2],
                self.env.world_size - pos[2],
            ]
            for margin in margins:
                cost += 3.0 * ca.exp(-wall_alpha * margin)

            for obs_idx in range(self.n_obs):
                obs_center = self.p_obs_centers[:, obs_idx]
                obs_radius = self.p_obs_radii[obs_idx]
                dist_to_obs = ca.sqrt(ca.sumsqr(pos - obs_center) + 1e-6)
                clearance = dist_to_obs - (obs_radius + self.env.auv_radius + self.obstacle_buffer)
                cost += 2.2 * ca.exp(-obs_alpha * clearance)

            prev_u = uk

        pos_terminal = self.X[0:3, self.horizon]
        target_terminal = self.p_target - pos_terminal
        terminal_dist_sq = ca.sumsqr(target_terminal)
        cost += 22.0 * terminal_dist_sq

        self.opti.minimize(cost)
        self.opti.solver(
            self.solver_name,
            {
                "print_time": False,
                "expand": True,
            },
            {
                "print_level": 0,
                "sb": "yes",
                "max_iter": 80,
                "tol": 1e-3,
                "acceptable_tol": 1e-2,
            },
        )

    def _extract_state(self) -> np.ndarray:
        eta = self.env.state[:6]
        nu = self.env.state[6:]
        return np.array(
            [
                eta[0],
                eta[1],
                eta[2],
                eta[5],
                eta[4],
                nu[0],
                nu[5],
                nu[4],
            ],
            dtype=np.float64,
        )

    def _obstacle_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        centers = np.zeros((3, self.n_obs), dtype=np.float64)
        radii = np.full(self.n_obs, 0.30, dtype=np.float64)
        for idx, obs in enumerate(self.env.obstacles[: self.n_obs]):
            centers[:, idx] = obs.center
            radii[idx] = obs.radius
        return centers, radii

    def _fallback_action(self) -> np.ndarray:
        eta = self.env.state[:6]
        nu = self.env.state[6:]
        pos = eta[:3]
        psi = eta[5]
        theta = eta[4]
        q = nu[4]
        r = nu[5]

        target_vec = self.env.target - pos
        goal_dist = float(np.linalg.norm(target_vec) + 1e-9)
        psi_ref = float(np.arctan2(target_vec[1], target_vec[0]))
        theta_ref = float(
            -np.arctan2(target_vec[2], max(np.linalg.norm(target_vec[:2]), 1e-6))
        )
        theta_ref = float(np.clip(theta_ref, -np.deg2rad(18.0), np.deg2rad(18.0)))

        heading_err = wrap_angle(psi_ref - psi)
        pitch_err = theta_ref - theta

        action = np.array(
            [
                np.clip(0.45 + 0.30 * np.tanh(0.18 * goal_dist), -0.10, 1.0),
                np.clip(1.6 * heading_err - 0.45 * r, -1.0, 1.0),
                np.clip(-1.8 * pitch_err - 0.40 * q, -1.0, 1.0),
            ],
            dtype=np.float64,
        )
        return action

    def act(self) -> np.ndarray:
        if self.replan_counter > 0:
            self.replan_counter -= 1
            return self.cached_action.copy()

        state0 = self._extract_state()
        centers, radii = self._obstacle_arrays()
        current_world = np.asarray(self.env.current_inertial, dtype=np.float64)

        self.opti.set_value(self.p_x0, state0)
        self.opti.set_value(self.p_target, np.asarray(self.env.target, dtype=np.float64))
        self.opti.set_value(self.p_current, current_world)
        self.opti.set_value(self.p_prev_action, self.last_action)
        self.opti.set_value(self.p_obs_centers, centers)
        self.opti.set_value(self.p_obs_radii, radii)

        self.x_guess = self._rollout_guess(state0, current_world)
        self.opti.set_initial(self.X, self.x_guess.T)
        self.opti.set_initial(self.U, self.u_guess.T)

        try:
            sol = self.opti.solve()
            u_seq = np.asarray(sol.value(self.U)).T
            self.u_guess = np.vstack([u_seq[1:], u_seq[-1:]])
            action = u_seq[0]
        except RuntimeError:
            action = self._fallback_action()
            self.u_guess = np.vstack([self.u_guess[1:], self.u_guess[-1:]])

        action = np.asarray(action, dtype=np.float64)
        action[0] = np.clip(action[0], -0.25, 1.0)
        action[1:] = np.clip(action[1:], -1.0, 1.0)
        self.cached_action = action
        self.last_action = action.copy()
        self.replan_counter = self.replan_every - 1
        return action.copy()


class HDF5TransitionWriter:
    def __init__(self, path: Path, obs_dim: int, action_dim: int, flush_size: int = 2048) -> None:
        self.path = path
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.flush_size = flush_size
        self.file = h5py.File(path, "w")
        self.string_dtype = h5py.string_dtype(encoding="utf-8")
        self.count = 0

        self.datasets = {
            "observations": self.file.create_dataset(
                "observations",
                shape=(0, obs_dim),
                maxshape=(None, obs_dim),
                chunks=(flush_size, obs_dim),
                dtype=np.float32,
            ),
            "actions": self.file.create_dataset(
                "actions",
                shape=(0, action_dim),
                maxshape=(None, action_dim),
                chunks=(flush_size, action_dim),
                dtype=np.float32,
            ),
            "next_observations": self.file.create_dataset(
                "next_observations",
                shape=(0, obs_dim),
                maxshape=(None, obs_dim),
                chunks=(flush_size, obs_dim),
                dtype=np.float32,
            ),
            "rewards": self.file.create_dataset(
                "rewards",
                shape=(0,),
                maxshape=(None,),
                chunks=(flush_size,),
                dtype=np.float32,
            ),
            "done": self.file.create_dataset(
                "done",
                shape=(0,),
                maxshape=(None,),
                chunks=(flush_size,),
                dtype=np.bool_,
            ),
            "truncated": self.file.create_dataset(
                "truncated",
                shape=(0,),
                maxshape=(None,),
                chunks=(flush_size,),
                dtype=np.bool_,
            ),
            "episode": self.file.create_dataset(
                "episode",
                shape=(0,),
                maxshape=(None,),
                chunks=(flush_size,),
                dtype=np.int32,
            ),
            "step": self.file.create_dataset(
                "step",
                shape=(0,),
                maxshape=(None,),
                chunks=(flush_size,),
                dtype=np.int32,
            ),
            "seed": self.file.create_dataset(
                "seed",
                shape=(0,),
                maxshape=(None,),
                chunks=(flush_size,),
                dtype=np.int32,
            ),
            "event": self.file.create_dataset(
                "event",
                shape=(0,),
                maxshape=(None,),
                chunks=(flush_size,),
                dtype=self.string_dtype,
            ),
        }
        self.buffer: Dict[str, List[Any]] = {key: [] for key in self.datasets}

    def append(self, **kwargs: Any) -> None:
        key_map = {
            "observations": "observations",
            "actions": "actions",
            "next_observations": "next_observations",
            "rewards": "reward",
            "done": "done",
            "truncated": "truncated",
            "episode": "episode",
            "step": "step",
            "seed": "seed",
            "event": "event",
        }
        for buffer_key, input_key in key_map.items():
            self.buffer[buffer_key].append(kwargs[input_key])
        if len(self.buffer["rewards"]) >= self.flush_size:
            self.flush()

    def flush(self) -> None:
        buffer_len = len(self.buffer["rewards"])
        if buffer_len == 0:
            return

        start = self.count
        end = self.count + buffer_len
        for key, dataset in self.datasets.items():
            dataset.resize((end,) + dataset.shape[1:])
            values = self.buffer[key]
            if key == "event":
                dataset[start:end] = np.asarray(values, dtype=object)
            else:
                dataset[start:end] = np.asarray(values, dtype=dataset.dtype)
            self.buffer[key].clear()
        self.count = end

    def close(self, attrs: Optional[Dict[str, Any]] = None) -> None:
        self.flush()
        if attrs is not None:
            for key, value in attrs.items():
                self.file.attrs[key] = value
        self.file.close()


class CSVTransitionWriter:
    def __init__(self, path: Path, obs_dim: int, action_dim: int) -> None:
        self.path = path
        self.file = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        header = ["episode", "step", "seed", "reward", "done", "truncated", "event"]
        header.extend([f"obs_{idx}" for idx in range(obs_dim)])
        header.extend([f"action_{idx}" for idx in range(action_dim)])
        header.extend([f"next_obs_{idx}" for idx in range(obs_dim)])
        self.writer.writerow(header)

    def append(self, **kwargs: Any) -> None:
        row: List[Any] = [
            kwargs["episode"],
            kwargs["step"],
            kwargs["seed"],
            kwargs["reward"],
            int(kwargs["done"]),
            int(kwargs["truncated"]),
            kwargs["event"],
        ]
        row.extend(np.asarray(kwargs["observations"], dtype=np.float32).tolist())
        row.extend(np.asarray(kwargs["actions"], dtype=np.float32).tolist())
        row.extend(np.asarray(kwargs["next_observations"], dtype=np.float32).tolist())
        self.writer.writerow(row)

    def close(self, attrs: Optional[Dict[str, Any]] = None) -> None:
        _ = attrs
        self.file.close()


def minimum_clearance(env: REMUSAUVEnv) -> float:
    pos = env.state[:3]
    min_clearance = np.inf
    for obs in env.obstacles:
        clearance = np.linalg.norm(pos - obs.center) - (obs.radius + env.auv_radius)
        min_clearance = min(min_clearance, clearance)
    return float(min_clearance)


def make_run_dir(result_root: Path, started_at: datetime) -> Path:
    run_dir = result_root / f"casadi_mpc_dataset_{started_at.strftime(RUN_TIMESTAMP_FORMAT)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_writer(run_dir: Path, file_format: str, obs_dim: int, action_dim: int):
    if file_format == "hdf5":
        return HDF5TransitionWriter(run_dir / "transitions.h5", obs_dim=obs_dim, action_dim=action_dim)
    if file_format == "csv":
        return CSVTransitionWriter(run_dir / "transitions.csv", obs_dim=obs_dim, action_dim=action_dim)
    raise ValueError(f"Unsupported format: {file_format}")


def save_episode_summary(path: Path, summaries: List[EpisodeSummary]) -> None:
    if not summaries:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))


def collect_dataset(
    episodes: Optional[int],
    seed: int,
    run_dir: Path,
    file_format: str,
    env_dt: float,
    env_max_steps: int,
    n_obstacles: int,
    current_enabled: bool,
    horizon: int,
    replan_every: int,
    plan_dt: Optional[float],
    success_target: Optional[int] = None,
    easy_env: bool = True,
) -> Dict[str, Any]:
    started_at = datetime.now().isoformat(timespec="seconds")
    env_cls = EasyCollectionREMUSAUVEnv if easy_env else REMUSAUVEnv
    env_n_obstacles = 0 if easy_env else n_obstacles
    env_current_enabled = False if easy_env else current_enabled

    env_probe = env_cls(
        dt=env_dt,
        max_steps=env_max_steps,
        n_obstacles=env_n_obstacles,
        seed=seed,
        current_enabled=env_current_enabled,
    )
    obs_dim = int(env_probe.observation_space.shape[0])
    action_dim = int(env_probe.action_space.shape[0])
    writer = build_writer(run_dir, file_format=file_format, obs_dim=obs_dim, action_dim=action_dim)

    summaries: List[EpisodeSummary] = []
    event_counts: Dict[str, int] = {
        "goal": 0,
        "collision": 0,
        "out_of_bounds": 0,
        "timeout": 0,
        "other": 0,
    }

    total_transitions = 0
    collected_successes = 0
    stop_reason = "episodes_exhausted"
    episode = 0

    while True:
        if success_target is not None and collected_successes >= success_target:
            stop_reason = "success_target_reached"
            break
        if episodes is not None and episode >= episodes:
            stop_reason = "episodes_exhausted"
            break

        episode_seed = seed + episode
        env = env_cls(
            dt=env_dt,
            max_steps=env_max_steps,
            n_obstacles=env_n_obstacles,
            seed=episode_seed,
            current_enabled=env_current_enabled,
        )
        observation, info = env.reset(seed=episode_seed)
        controller = CasadiREMUSMPC(
            env=env,
            horizon=horizon,
            replan_every=replan_every,
            plan_dt=plan_dt,
        )
        controller.reset()

        episode_return = 0.0
        min_goal_distance = float(np.linalg.norm(env.state[:3] - env.target))
        min_clearance = minimum_clearance(env)
        final_event = "timeout"
        final_distance = min_goal_distance
        steps = 0

        for step in range(env.max_steps):
            action = controller.act()
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            event = str(step_info.get("event", "")) if (terminated or truncated) else ""

            writer.append(
                observations=np.asarray(observation, dtype=np.float32),
                actions=np.asarray(action, dtype=np.float32),
                next_observations=np.asarray(next_obs, dtype=np.float32),
                reward=np.float32(reward),
                done=bool(terminated or truncated),
                truncated=bool(truncated),
                episode=np.int32(episode),
                step=np.int32(step),
                seed=np.int32(episode_seed),
                event=event,
            )
            total_transitions += 1

            observation = next_obs
            episode_return += float(reward)
            steps = step + 1

            goal_distance = float(
                step_info.get("distance_to_goal", np.linalg.norm(env.state[:3] - env.target))
            )
            final_distance = goal_distance
            min_goal_distance = min(min_goal_distance, goal_distance)
            min_clearance = min(min_clearance, minimum_clearance(env))

            if terminated or truncated:
                final_event = event if event else ("timeout" if truncated else "other")
                break

        if final_event not in event_counts:
            final_event = "other"
        event_counts[final_event] += 1
        collected_successes += int(final_event == "goal")

        summaries.append(
            EpisodeSummary(
                episode=episode,
                seed=episode_seed,
                event=final_event,
                success=int(final_event == "goal"),
                steps=steps,
                episode_return=episode_return,
                final_distance_to_goal=final_distance,
                min_distance_to_goal=min_goal_distance,
                min_obstacle_clearance=min_clearance,
            )
        )
        success_rate = sum(item.success for item in summaries) / len(summaries)
        print(
            f"ep={episode:03d} | seed={episode_seed:04d} | event={final_event:<13} | "
            f"steps={steps:4d} | return={episode_return:8.2f} | "
            f"final_dist={final_distance:6.3f} | succ_rate={success_rate:5.3f} | "
            f"succ_count={collected_successes}"
            + (
                f"/{success_target}"
                if success_target is not None
                else ""
            ),
            flush=True,
        )
        episode += 1

    summary = {
        "started_at": started_at,
        "episodes": len(summaries),
        "requested_episode_limit": episodes,
        "seed_start": seed,
        "transitions": total_transitions,
        "success_target": success_target,
        "successful_trajectories": collected_successes,
        "stop_reason": stop_reason,
        "success_rate": float(sum(item.success for item in summaries) / max(len(summaries), 1)),
        "goal_count": event_counts["goal"],
        "collision_count": event_counts["collision"],
        "out_of_bounds_count": event_counts["out_of_bounds"],
        "timeout_count": event_counts["timeout"],
        "other_count": event_counts["other"],
        "mean_return": float(np.mean([item.episode_return for item in summaries])) if summaries else 0.0,
        "mean_steps": float(np.mean([item.steps for item in summaries])) if summaries else 0.0,
        "dataset_format": file_format,
        "dataset_path": str((run_dir / "transitions.h5") if file_format == "hdf5" else (run_dir / "transitions.csv")),
        "episode_summary_csv": str(run_dir / "episode_summary.csv"),
        "config": {
            "easy_env": easy_env,
            "env_dt": env_dt,
            "env_max_steps": env_max_steps,
            "n_obstacles": env_n_obstacles,
            "current_enabled": env_current_enabled,
            "horizon": horizon,
            "replan_every": replan_every,
            "plan_dt": float(plan_dt if plan_dt is not None else env_dt * replan_every),
        },
    }

    writer.close(
        attrs={
            "episodes": len(summaries),
            "requested_episode_limit": -1 if episodes is None else episodes,
            "seed_start": seed,
            "dataset_format": file_format,
            "created_at": started_at,
            "success_target": -1 if success_target is None else success_target,
            "successful_trajectories": collected_successes,
            "stop_reason": stop_reason,
        }
    )
    save_episode_summary(run_dir / "episode_summary.csv", summaries)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(summary["config"], f, indent=2)

    print(
        "final | "
        f"episodes={summary['episodes']} | transitions={summary['transitions']} | "
        f"successes={summary['successful_trajectories']}"
        + (
            f"/{summary['success_target']}"
            if summary["success_target"] is not None
            else ""
        )
        + " | "
        f"success_rate={summary['success_rate']:.3f} | "
        f"goal/coll/oob/to={summary['goal_count']}/{summary['collision_count']}/"
        f"{summary['out_of_bounds_count']}/{summary['timeout_count']} | "
        f"stop={summary['stop_reason']}",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect (observation, action) data from REMUSAUVEnv using a CasADi-based simplified MPC."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Episode limit. When omitted, --success-target can run without an episode cap.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument(
        "--success-target",
        type=int,
        default=None,
        help="Keep collecting until this many successful goal trajectories are obtained.",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help="Root directory for timestamped output folders.",
    )
    parser.add_argument(
        "--format",
        choices=("hdf5", "csv"),
        default="hdf5",
        help="Dataset storage format.",
    )
    parser.add_argument("--env-dt", type=float, default=0.05, help="Environment integration step.")
    parser.add_argument("--env-max-steps", type=int, default=1200, help="Maximum steps per episode.")
    parser.add_argument(
        "--n-obstacles",
        type=int,
        default=6,
        help="Number of obstacles in REMUSAUVEnv when using --full-env.",
    )
    parser.add_argument(
        "--disable-current",
        action="store_true",
        help="Disable ocean current disturbance when using --full-env.",
    )
    parser.add_argument(
        "--full-env",
        action="store_true",
        help="Use the original REMUSAUVEnv instead of the default easy collection environment.",
    )
    parser.add_argument("--horizon", type=int, default=12, help="MPC prediction horizon.")
    parser.add_argument(
        "--replan-every",
        type=int,
        default=3,
        help="Repeat the first MPC action for this many env steps before replanning.",
    )
    parser.add_argument(
        "--plan-dt",
        type=float,
        default=None,
        help="Optional MPC model time step. Default: env_dt * replan_every.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.success_target is not None and args.success_target <= 0:
        raise ValueError("--success-target must be a positive integer.")
    if args.episodes is not None and args.episodes <= 0:
        raise ValueError("--episodes must be a positive integer when provided.")

    episode_limit = args.episodes
    if episode_limit is None and args.success_target is None:
        episode_limit = 20

    run_dir = make_run_dir(args.result_root, datetime.now())
    collect_dataset(
        episodes=episode_limit,
        seed=args.seed,
        run_dir=run_dir,
        file_format=args.format,
        env_dt=args.env_dt,
        env_max_steps=args.env_max_steps,
        n_obstacles=args.n_obstacles,
        current_enabled=not args.disable_current,
        horizon=args.horizon,
        replan_every=args.replan_every,
        plan_dt=args.plan_dt,
        success_target=args.success_target,
        easy_env=not args.full_env,
    )
