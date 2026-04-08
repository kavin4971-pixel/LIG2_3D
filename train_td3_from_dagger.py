from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - compatibility fallback
    import gym

import numpy as np

from REMUSAUVEnv import REMUSAUVEnv, wrap_angle


DEFAULT_DAGGER_POLICY_PATH = Path(r"C:\Users\kavin\Desktop\LIG2_result\20260408_143015\dagger_policy.pt")
DEFAULT_RESULT_ROOT = Path(r"C:\Users\kavin\Desktop\LIG2_result")
RUN_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


@dataclass
class EvalSummary:
    name: str
    episodes: int
    seed_start: int
    success_rate: float
    goal_count: int
    collision_count: int
    out_of_bounds_count: int
    timeout_count: int
    other_count: int
    mean_return: float
    mean_steps: float


@dataclass
class CurriculumStage:
    name: str
    end_timestep: int
    current_enabled: bool
    current_scale: float
    n_obstacles: int
    obstacle_velocity_scale: float
    goal_radius: float
    min_start_target_distance: float
    max_start_target_distance: Optional[float]
    yaw_align_std_deg: Optional[float]
    pitch_align_std_deg: Optional[float]
    initial_forward_speed: float


class TimeoutDistancePenaltyWrapper(gym.Wrapper):
    """
    Adds an explicit timeout penalty proportional to the final distance to goal.

    This keeps the underlying REMUSAUVEnv unchanged while making "safe but
    aimless" timeouts less attractive than actually reaching the goal.
    """

    def __init__(self, env: gym.Env, penalty_scale: float) -> None:
        super().__init__(env)
        self.penalty_scale = max(float(penalty_scale), 0.0)

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if truncated and str(info.get("event", "")) == "timeout" and self.penalty_scale > 0.0:
            final_distance = float(info.get("distance_to_goal", 0.0))
            timeout_penalty = self.penalty_scale * final_distance
            reward -= timeout_penalty
            info = dict(info)
            info["timeout_distance_penalty"] = timeout_penalty
            info["reward_after_timeout_penalty"] = float(reward)
        return obs, reward, terminated, truncated, info


class CurriculumREMUSAUVEnv(REMUSAUVEnv):
    """
    Training-only curriculum wrapper implemented as an REMUSAUVEnv subclass.

    The underlying physics remain unchanged. Difficulty is modulated only by
    reset-time scenario generation:
      - current on/off + current magnitude scaling
      - obstacle count and obstacle speed scaling
      - goal radius
      - start-target distance band
      - initial yaw/pitch alignment
      - initial forward speed boost for early control-surface authority
    """

    def __init__(
        self,
        curriculum_stages: Sequence[CurriculumStage],
        dt: float,
        max_steps: int,
        world_size: float,
        n_obstacles: int,
        seed: Optional[int],
        include_current_in_obs: bool,
    ) -> None:
        if not curriculum_stages:
            raise ValueError("curriculum_stages must not be empty.")

        self.curriculum_stages = list(curriculum_stages)
        self.stage_index = 0
        self.active_stage = self.curriculum_stages[0]
        self._curriculum_min_start_target_distance = self.active_stage.min_start_target_distance
        self._curriculum_max_start_target_distance = self.active_stage.max_start_target_distance
        self._curriculum_yaw_align_std = (
            None
            if self.active_stage.yaw_align_std_deg is None
            else np.deg2rad(self.active_stage.yaw_align_std_deg)
        )
        self._curriculum_pitch_align_std = (
            None
            if self.active_stage.pitch_align_std_deg is None
            else np.deg2rad(self.active_stage.pitch_align_std_deg)
        )

        super().__init__(
            dt=dt,
            max_steps=max_steps,
            world_size=world_size,
            n_obstacles=self.active_stage.n_obstacles,
            seed=seed,
            current_enabled=self.active_stage.current_enabled,
            include_current_in_obs=include_current_in_obs,
        )
        self.set_stage(0)

    def set_stage(self, stage_index: int) -> None:
        stage_index = int(np.clip(stage_index, 0, len(self.curriculum_stages) - 1))
        self.stage_index = stage_index
        self.active_stage = self.curriculum_stages[stage_index]
        self.current_enabled = bool(self.active_stage.current_enabled)
        self.n_obstacles = int(self.active_stage.n_obstacles)
        self.goal_radius = float(self.active_stage.goal_radius)
        self._curriculum_min_start_target_distance = float(self.active_stage.min_start_target_distance)
        self._curriculum_max_start_target_distance = (
            None
            if self.active_stage.max_start_target_distance is None
            else float(self.active_stage.max_start_target_distance)
        )
        self._curriculum_yaw_align_std = (
            None
            if self.active_stage.yaw_align_std_deg is None
            else np.deg2rad(self.active_stage.yaw_align_std_deg)
        )
        self._curriculum_pitch_align_std = (
            None
            if self.active_stage.pitch_align_std_deg is None
            else np.deg2rad(self.active_stage.pitch_align_std_deg)
        )

    def _generate_start_target(self) -> Tuple[np.ndarray, np.ndarray]:
        min_dist = max(float(self._curriculum_min_start_target_distance), 0.0)
        max_dist = (
            np.inf
            if self._curriculum_max_start_target_distance is None
            else max(float(self._curriculum_max_start_target_distance), min_dist + 1e-6)
        )

        for _ in range(2000):
            start = self._sample_point()
            target = self._sample_point()
            dist = float(np.linalg.norm(start - target))
            if dist >= min_dist and dist <= max_dist:
                return start, target
        raise RuntimeError(
            f"Failed to sample start/target for curriculum stage {self.active_stage.name}."
        )

    def _apply_stage_post_reset(self) -> None:
        stage = self.active_stage

        if not stage.current_enabled:
            self.current_base_inertial[:] = 0.0
            self.current_osc_amp[:] = 0.0
            self.current_inertial[:] = 0.0
        else:
            self.current_base_inertial *= float(stage.current_scale)
            self.current_osc_amp *= float(stage.current_scale)
            self.current_inertial *= float(stage.current_scale)

        for obs in self.obstacles:
            obs.velocity *= float(stage.obstacle_velocity_scale)

        pos = self.state[:3]
        target_vec = self.target - pos
        horiz = float(np.linalg.norm(target_vec[:2]))
        psi_ref = float(np.arctan2(target_vec[1], target_vec[0]))
        theta_ref = float(-np.arctan2(target_vec[2], max(horiz, 1e-6)))
        theta_ref = float(np.clip(theta_ref, -self.theta_limit, self.theta_limit))

        if self._curriculum_yaw_align_std is not None:
            self.state[5] = wrap_angle(
                psi_ref + self.rng.normal(loc=0.0, scale=self._curriculum_yaw_align_std)
            )
        if self._curriculum_pitch_align_std is not None:
            self.state[4] = float(
                np.clip(
                    theta_ref + self.rng.normal(loc=0.0, scale=self._curriculum_pitch_align_std),
                    -self.theta_limit,
                    self.theta_limit,
                )
            )
            self.state[3] = float(np.clip(self.rng.normal(loc=0.0, scale=np.deg2rad(1.5)), -0.05, 0.05))

        if stage.initial_forward_speed > 0.0:
            self.state[6] = max(float(self.state[6]), float(stage.initial_forward_speed))

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, info = super().reset(seed=seed, options=options)
        _ = obs
        self._apply_stage_post_reset()
        info = dict(info)
        info["curriculum_stage"] = self.active_stage.name
        info["curriculum_stage_index"] = int(self.stage_index)
        return self._get_obs(), info


def require_td3_dependencies() -> Dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
        from stable_baselines3 import TD3
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.noise import NormalActionNoise
    except ImportError as exc:
        raise SystemExit(
            "Missing TD3 dependencies. Run this script from the environment that has "
            "`torch` and `stable-baselines3` installed.\n"
            "Example:\n"
            "  C:\\anaconda3\\envs\\lig2\\python train_td3_from_dagger.py\n"
            f"Original import error: {exc}"
        ) from exc

    return {
        "torch": torch,
        "nn": nn,
        "TD3": TD3,
        "BaseCallback": BaseCallback,
        "CallbackList": CallbackList,
        "CheckpointCallback": CheckpointCallback,
        "EvalCallback": EvalCallback,
        "Monitor": Monitor,
        "NormalActionNoise": NormalActionNoise,
    }


def build_actor_freeze_callback_class(base_callback_cls: Any):
    class ActorFreezeCallback(base_callback_cls):
        def __init__(self, freeze_steps: int, unfrozen_policy_delay: int, verbose: int = 0) -> None:
            super().__init__(verbose=verbose)
            self.freeze_steps = max(int(freeze_steps), 0)
            self.unfrozen_policy_delay = max(int(unfrozen_policy_delay), 1)
            self._released = self.freeze_steps <= 0

        def _on_step(self) -> bool:
            if not self._released and self.num_timesteps >= self.freeze_steps:
                self.model.policy_delay = self.unfrozen_policy_delay
                self._released = True
                if self.verbose > 0:
                    print(
                        f"Actor freeze released at timestep {self.num_timesteps}. "
                        f"policy_delay -> {self.unfrozen_policy_delay}",
                        flush=True,
                    )
            return True

    return ActorFreezeCallback


def build_curriculum_callback_class(base_callback_cls: Any):
    class CurriculumCallback(base_callback_cls):
        def __init__(self, curriculum_env: CurriculumREMUSAUVEnv, verbose: int = 0) -> None:
            super().__init__(verbose=verbose)
            self.curriculum_env = curriculum_env
            self.stage_history: List[Dict[str, Any]] = []
            self._last_stage_index = -1

        def _stage_index_from_timestep(self, timestep: int) -> int:
            for idx, stage in enumerate(self.curriculum_env.curriculum_stages):
                if timestep <= stage.end_timestep:
                    return idx
            return len(self.curriculum_env.curriculum_stages) - 1

        def _record_stage(self, timestep: int, stage: CurriculumStage) -> None:
            record = {
                "timestep": int(timestep),
                "stage_index": int(self.curriculum_env.stage_index),
                "stage": asdict(stage),
            }
            self.stage_history.append(record)
            if self.verbose > 0:
                print(
                    "curriculum | "
                    f"t={timestep} | stage={stage.name} | current_scale={stage.current_scale:.2f} | "
                    f"obstacles={stage.n_obstacles} | obs_speed_scale={stage.obstacle_velocity_scale:.2f} | "
                    f"goal_radius={stage.goal_radius:.2f}",
                    flush=True,
                )

        def _update_stage(self, timestep: int) -> None:
            stage_index = self._stage_index_from_timestep(timestep)
            if stage_index == self._last_stage_index:
                return
            self.curriculum_env.set_stage(stage_index)
            self._last_stage_index = stage_index
            self._record_stage(timestep, self.curriculum_env.active_stage)

        def _on_training_start(self) -> None:
            self._update_stage(0)

        def _on_step(self) -> bool:
            self._update_stage(self.num_timesteps)
            return True

    return CurriculumCallback


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def resolve_torch_device(torch_mod: Any, requested_device: str) -> Any:
    if requested_device == "auto":
        return torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
    return torch_mod.device(requested_device)


def resolve_run_output_dir(out_dir: Optional[Path], result_root: Path, started_at: datetime) -> Path:
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    run_dir = result_root / started_at.strftime(RUN_TIMESTAMP_FORMAT)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_default_curriculum(total_timesteps: int, full_n_obstacles: int) -> List[CurriculumStage]:
    total_timesteps = max(int(total_timesteps), 1)
    fractions = [0.16, 0.36, 0.60, 0.80, 1.00]
    end_steps: List[int] = []
    prev = 0
    for frac in fractions:
        step = max(prev + 1, int(round(frac * total_timesteps)))
        end_steps.append(min(step, total_timesteps))
        prev = end_steps[-1]
    end_steps[-1] = total_timesteps

    return [
        CurriculumStage(
            name="bridge_nominal",
            end_timestep=end_steps[0],
            current_enabled=False,
            current_scale=0.0,
            n_obstacles=0,
            obstacle_velocity_scale=0.0,
            goal_radius=0.60,
            min_start_target_distance=4.0,
            max_start_target_distance=7.0,
            yaw_align_std_deg=8.0,
            pitch_align_std_deg=4.0,
            initial_forward_speed=0.20,
        ),
        CurriculumStage(
            name="mild_current",
            end_timestep=end_steps[1],
            current_enabled=True,
            current_scale=0.20,
            n_obstacles=0,
            obstacle_velocity_scale=0.0,
            goal_radius=0.55,
            min_start_target_distance=4.5,
            max_start_target_distance=7.5,
            yaw_align_std_deg=12.0,
            pitch_align_std_deg=6.0,
            initial_forward_speed=0.18,
        ),
        CurriculumStage(
            name="static_obstacles",
            end_timestep=end_steps[2],
            current_enabled=True,
            current_scale=0.45,
            n_obstacles=min(2, full_n_obstacles),
            obstacle_velocity_scale=0.0,
            goal_radius=0.50,
            min_start_target_distance=5.0,
            max_start_target_distance=8.5,
            yaw_align_std_deg=18.0,
            pitch_align_std_deg=8.0,
            initial_forward_speed=0.15,
        ),
        CurriculumStage(
            name="slow_moving_obstacles",
            end_timestep=end_steps[3],
            current_enabled=True,
            current_scale=0.70,
            n_obstacles=min(4, full_n_obstacles),
            obstacle_velocity_scale=0.35,
            goal_radius=0.42,
            min_start_target_distance=5.5,
            max_start_target_distance=9.0,
            yaw_align_std_deg=25.0,
            pitch_align_std_deg=10.0,
            initial_forward_speed=0.10,
        ),
        CurriculumStage(
            name="full_env",
            end_timestep=end_steps[4],
            current_enabled=True,
            current_scale=1.00,
            n_obstacles=full_n_obstacles,
            obstacle_velocity_scale=1.00,
            goal_radius=0.35,
            min_start_target_distance=6.0,
            max_start_target_distance=None,
            yaw_align_std_deg=None,
            pitch_align_std_deg=None,
            initial_forward_speed=0.0,
        ),
    ]


def make_training_env(
    seed: Optional[int],
    monitor_cls: Any,
    env_dt: float,
    env_max_steps: int,
    world_size: float,
    n_obstacles: int,
    timeout_distance_penalty_scale: float,
    use_curriculum: bool,
    curriculum_stages: Optional[Sequence[CurriculumStage]],
) -> Tuple[Any, Optional[CurriculumREMUSAUVEnv]]:
    if use_curriculum:
        if not curriculum_stages:
            raise ValueError("Curriculum requested but no curriculum stages were provided.")
        base_env: REMUSAUVEnv = CurriculumREMUSAUVEnv(
            curriculum_stages=curriculum_stages,
            dt=env_dt,
            max_steps=env_max_steps,
            world_size=world_size,
            n_obstacles=n_obstacles,
            seed=seed,
            include_current_in_obs=True,
        )
        curriculum_ref: Optional[CurriculumREMUSAUVEnv] = base_env
    else:
        base_env = REMUSAUVEnv(
            dt=env_dt,
            max_steps=env_max_steps,
            world_size=world_size,
            n_obstacles=n_obstacles,
            current_enabled=True,
            include_current_in_obs=True,
            seed=seed,
        )
        curriculum_ref = None

    env: gym.Env = TimeoutDistancePenaltyWrapper(base_env, penalty_scale=timeout_distance_penalty_scale)
    return monitor_cls(env), curriculum_ref


def make_full_eval_env(
    seed: Optional[int],
    monitor_cls: Any,
    env_dt: float,
    env_max_steps: int,
    world_size: float,
    n_obstacles: int,
    timeout_distance_penalty_scale: float,
) -> Any:
    env: gym.Env = REMUSAUVEnv(
        dt=env_dt,
        max_steps=env_max_steps,
        world_size=world_size,
        n_obstacles=n_obstacles,
        current_enabled=True,
        include_current_in_obs=True,
        seed=seed,
    )
    env = TimeoutDistancePenaltyWrapper(env, penalty_scale=timeout_distance_penalty_scale)
    return monitor_cls(env)


def evaluate_policy_events(
    policy_like: Any,
    monitor_cls: Any,
    episodes: int,
    seed: int,
    env_dt: float,
    env_max_steps: int,
    world_size: float,
    n_obstacles: int,
    timeout_distance_penalty_scale: float,
    name: str,
) -> EvalSummary:
    event_counts: Dict[str, int] = {
        "goal": 0,
        "collision": 0,
        "out_of_bounds": 0,
        "timeout": 0,
        "other": 0,
    }
    returns: List[float] = []
    steps_taken: List[int] = []

    for episode in range(episodes):
        env_seed = seed + episode
        env = make_full_eval_env(
            seed=env_seed,
            monitor_cls=monitor_cls,
            env_dt=env_dt,
            env_max_steps=env_max_steps,
            world_size=world_size,
            n_obstacles=n_obstacles,
            timeout_distance_penalty_scale=timeout_distance_penalty_scale,
        )
        obs, _ = env.reset(seed=env_seed)
        episode_return = 0.0
        final_event = "timeout"
        steps = 0

        for t in range(env.unwrapped.max_steps):
            action, _ = policy_like.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            steps = t + 1
            if terminated or truncated:
                final_event = str(info.get("event", "timeout" if truncated else "other"))
                break

        if final_event not in event_counts:
            final_event = "other"
        event_counts[final_event] += 1
        returns.append(episode_return)
        steps_taken.append(steps)
        env.close()

    return EvalSummary(
        name=name,
        episodes=episodes,
        seed_start=seed,
        success_rate=event_counts["goal"] / max(episodes, 1),
        goal_count=event_counts["goal"],
        collision_count=event_counts["collision"],
        out_of_bounds_count=event_counts["out_of_bounds"],
        timeout_count=event_counts["timeout"],
        other_count=event_counts["other"],
        mean_return=float(np.mean(returns)) if returns else 0.0,
        mean_steps=float(np.mean(steps_taken)) if steps_taken else 0.0,
    )


def load_dagger_policy(policy_path: Path, torch_mod: Any, device: str) -> Any:
    torch_device = resolve_torch_device(torch_mod, device)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=r"You are using `torch\.load` with `weights_only=False`",
        )
        policy = torch_mod.load(policy_path, map_location=torch_device)
    if not hasattr(policy, "state_dict"):
        raise TypeError(f"Loaded object from {policy_path} is not a torch policy.")
    if hasattr(policy, "set_training_mode"):
        policy.set_training_mode(False)
    return policy


def transfer_dagger_actor_to_td3(model: Any, dagger_policy: Any, torch_mod: Any) -> Dict[str, Any]:
    src_state = dagger_policy.state_dict()
    actor = model.policy.actor
    actor_target = model.policy.actor_target

    mapping = {
        "mlp_extractor.policy_net.0.weight": "mu.0.weight",
        "mlp_extractor.policy_net.0.bias": "mu.0.bias",
        "mlp_extractor.policy_net.2.weight": "mu.2.weight",
        "mlp_extractor.policy_net.2.bias": "mu.2.bias",
        "action_net.weight": "mu.4.weight",
        "action_net.bias": "mu.4.bias",
    }

    copied: List[Dict[str, Any]] = []
    with torch_mod.no_grad():
        actor_state = actor.state_dict()
        actor_target_state = actor_target.state_dict()
        for src_key, dst_key in mapping.items():
            if src_key not in src_state:
                raise KeyError(f"Missing source key in dagger policy: {src_key}")
            if dst_key not in actor_state:
                raise KeyError(f"Missing destination key in TD3 actor: {dst_key}")
            src_tensor = src_state[src_key]
            if tuple(src_tensor.shape) != tuple(actor_state[dst_key].shape):
                raise ValueError(
                    f"Shape mismatch for {src_key} -> {dst_key}: "
                    f"{tuple(src_tensor.shape)} != {tuple(actor_state[dst_key].shape)}"
                )
            actor_state[dst_key].copy_(src_tensor)
            actor_target_state[dst_key].copy_(src_tensor)
            copied.append(
                {
                    "src": src_key,
                    "dst": dst_key,
                    "shape": list(src_tensor.shape),
                }
            )

    return {
        "copied_parameters": copied,
        "actor_net_arch": [32, 32],
        "activation": "Tanh",
    }


def train_td3_from_dagger(args: argparse.Namespace) -> Dict[str, Any]:
    deps = require_td3_dependencies()
    torch_mod = deps["torch"]
    nn = deps["nn"]
    td3_cls = deps["TD3"]
    base_callback_cls = deps["BaseCallback"]
    callback_list_cls = deps["CallbackList"]
    checkpoint_callback_cls = deps["CheckpointCallback"]
    eval_callback_cls = deps["EvalCallback"]
    monitor_cls = deps["Monitor"]
    normal_action_noise_cls = deps["NormalActionNoise"]

    dagger_policy_path = args.dagger_policy_path.resolve()
    if not dagger_policy_path.exists():
        raise FileNotFoundError(f"Dagger policy not found: {dagger_policy_path}")

    started_at = datetime.now()
    run_dir = resolve_run_output_dir(args.out_dir, args.result_root, started_at)
    checkpoints_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    actor_freeze_callback_cls = build_actor_freeze_callback_class(base_callback_cls)
    curriculum_callback_cls = build_curriculum_callback_class(base_callback_cls)

    curriculum_stages = None if args.disable_curriculum else build_default_curriculum(
        total_timesteps=args.total_timesteps,
        full_n_obstacles=args.n_obstacles,
    )

    train_env, curriculum_env = make_training_env(
        seed=args.seed,
        monitor_cls=monitor_cls,
        env_dt=args.env_dt,
        env_max_steps=args.env_max_steps,
        world_size=args.world_size,
        n_obstacles=args.n_obstacles,
        timeout_distance_penalty_scale=args.timeout_distance_penalty_scale,
        use_curriculum=not args.disable_curriculum,
        curriculum_stages=curriculum_stages,
    )
    eval_env = make_full_eval_env(
        seed=args.eval_seed,
        monitor_cls=monitor_cls,
        env_dt=args.env_dt,
        env_max_steps=args.env_max_steps,
        world_size=args.world_size,
        n_obstacles=args.n_obstacles,
        timeout_distance_penalty_scale=args.timeout_distance_penalty_scale,
    )

    dagger_policy = load_dagger_policy(dagger_policy_path, torch_mod=torch_mod, device=args.device)

    action_dim = int(train_env.action_space.shape[0])
    action_noise = normal_action_noise_cls(
        mean=np.zeros(action_dim, dtype=np.float32),
        sigma=np.full(action_dim, args.action_noise_std, dtype=np.float32),
    )
    freeze_policy_delay = int(10**12) if args.actor_freeze_steps > 0 else int(args.policy_delay)

    model = td3_cls(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        policy_delay=freeze_policy_delay,
        action_noise=action_noise,
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=str(run_dir / "tb"),
        policy_kwargs={
            "net_arch": [32, 32],
            "activation_fn": nn.Tanh,
        },
    )

    transfer_report = transfer_dagger_actor_to_td3(model, dagger_policy, torch_mod=torch_mod)
    save_json(run_dir / "transfer_report.json", transfer_report)

    dagger_eval = evaluate_policy_events(
        policy_like=dagger_policy,
        monitor_cls=monitor_cls,
        episodes=args.eval_episodes,
        seed=args.eval_seed,
        env_dt=args.env_dt,
        env_max_steps=args.env_max_steps,
        world_size=args.world_size,
        n_obstacles=args.n_obstacles,
        timeout_distance_penalty_scale=args.timeout_distance_penalty_scale,
        name="dagger_full_env",
    )
    td3_init_eval = evaluate_policy_events(
        policy_like=model,
        monitor_cls=monitor_cls,
        episodes=args.eval_episodes,
        seed=args.eval_seed,
        env_dt=args.env_dt,
        env_max_steps=args.env_max_steps,
        world_size=args.world_size,
        n_obstacles=args.n_obstacles,
        timeout_distance_penalty_scale=args.timeout_distance_penalty_scale,
        name="td3_initialized_from_dagger",
    )

    checkpoint_callback = checkpoint_callback_cls(
        save_freq=max(args.checkpoint_freq, 1),
        save_path=str(checkpoints_dir),
        name_prefix="td3_dagger_init",
        save_replay_buffer=args.save_replay_buffer,
        save_vecnormalize=False,
    )
    eval_callback = eval_callback_cls(
        eval_env,
        best_model_save_path=str(eval_dir),
        log_path=str(eval_dir),
        eval_freq=max(args.eval_freq, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    callbacks_list: List[Any] = [checkpoint_callback, eval_callback]
    curriculum_callback = None
    if curriculum_env is not None:
        curriculum_callback = curriculum_callback_cls(curriculum_env=curriculum_env, verbose=1)
        callbacks_list.append(curriculum_callback)
    if args.actor_freeze_steps > 0:
        callbacks_list.append(
            actor_freeze_callback_cls(
                freeze_steps=args.actor_freeze_steps,
                unfrozen_policy_delay=args.policy_delay,
                verbose=1,
            )
        )
    callbacks = callback_list_cls(callbacks_list)

    model.learn(
        total_timesteps=args.total_timesteps,
        log_interval=args.log_interval,
        progress_bar=True,
        callback=callbacks,
    )

    final_model_path = run_dir / "td3_dagger_init_final"
    model.save(str(final_model_path))
    if args.save_replay_buffer:
        model.save_replay_buffer(str(run_dir / "td3_dagger_init_replay_buffer"))

    td3_final_eval = evaluate_policy_events(
        policy_like=model,
        monitor_cls=monitor_cls,
        episodes=args.eval_episodes,
        seed=args.eval_seed,
        env_dt=args.env_dt,
        env_max_steps=args.env_max_steps,
        world_size=args.world_size,
        n_obstacles=args.n_obstacles,
        timeout_distance_penalty_scale=args.timeout_distance_penalty_scale,
        name="td3_final",
    )

    summary = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "dagger_policy_path": str(dagger_policy_path),
        "final_model_path": str(final_model_path) + ".zip",
        "best_model_dir": str(eval_dir),
        "environment": {
            "current_enabled": True,
            "n_obstacles": args.n_obstacles,
            "env_dt": args.env_dt,
            "env_max_steps": args.env_max_steps,
            "world_size": args.world_size,
            "timeout_distance_penalty_scale": args.timeout_distance_penalty_scale,
            "curriculum_enabled": not args.disable_curriculum,
        },
        "td3_config": {
            "seed": args.seed,
            "total_timesteps": args.total_timesteps,
            "learning_rate": args.learning_rate,
            "buffer_size": args.buffer_size,
            "learning_starts": args.learning_starts,
            "batch_size": args.batch_size,
            "tau": args.tau,
            "gamma": args.gamma,
            "train_freq": args.train_freq,
            "gradient_steps": args.gradient_steps,
            "policy_delay": args.policy_delay,
            "initial_policy_delay": freeze_policy_delay,
            "actor_freeze_steps": args.actor_freeze_steps,
            "action_noise_std": args.action_noise_std,
            "policy_net_arch": [32, 32],
            "policy_activation": "Tanh",
        },
        "transfer": transfer_report,
        "curriculum": {
            "enabled": not args.disable_curriculum,
            "stages": [asdict(stage) for stage in curriculum_stages] if curriculum_stages is not None else None,
            "history": curriculum_callback.stage_history if curriculum_callback is not None else [],
        },
        "evaluation": {
            "dagger_full_env": asdict(dagger_eval),
            "td3_initialized": asdict(td3_init_eval),
            "td3_final": asdict(td3_final_eval),
        },
    }
    save_json(run_dir / "train_summary.json", summary)
    save_json(run_dir / "dagger_full_env_eval.json", asdict(dagger_eval))
    save_json(run_dir / "td3_initialized_eval.json", asdict(td3_init_eval))
    save_json(run_dir / "td3_final_eval.json", asdict(td3_final_eval))
    if curriculum_callback is not None:
        save_json(
            run_dir / "curriculum_stage_history.json",
            {"history": curriculum_callback.stage_history, "stages": [asdict(stage) for stage in curriculum_stages]},
        )

    print(
        f"dagger full-env success_rate={dagger_eval.success_rate:.3f} | "
        f"td3 init success_rate={td3_init_eval.success_rate:.3f} | "
        f"td3 final success_rate={td3_final_eval.success_rate:.3f}",
        flush=True,
    )
    print(f"saved outputs to {run_dir}", flush=True)

    eval_env.close()
    train_env.close()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train TD3 on REMUSAUVEnv, initializing the actor from a saved DAgger policy. "
            "Training can optionally use a curriculum that gradually enables current and obstacles, "
            "while evaluation remains on the full environment."
        )
    )
    parser.add_argument(
        "--dagger-policy-path",
        type=Path,
        default=DEFAULT_DAGGER_POLICY_PATH,
        help="Path to the saved dagger_policy.pt used for actor initialization.",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help="Root directory used for timestamped output folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Training seed.")
    parser.add_argument("--eval-seed", type=int, default=10_000, help="Evaluation base seed.")
    parser.add_argument("--total-timesteps", type=int, default=500_000, help="TD3 training timesteps.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Low TD3 learning rate for gentle fine-tuning from the DAgger initialization.",
    )
    parser.add_argument("--buffer-size", type=int, default=500_000, help="Replay buffer size.")
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=0,
        help="Keep 0 to act with the DAgger-initialized policy immediately instead of random warmup.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="TD3 batch size.")
    parser.add_argument("--tau", type=float, default=0.005, help="TD3 target smoothing coefficient.")
    parser.add_argument("--gamma", type=float, default=0.99, help="TD3 discount factor.")
    parser.add_argument("--train-freq", type=int, default=1, help="TD3 train frequency.")
    parser.add_argument("--gradient-steps", type=int, default=1, help="TD3 gradient steps per update.")
    parser.add_argument("--policy-delay", type=int, default=2, help="TD3 actor update period after the freeze stage.")
    parser.add_argument(
        "--actor-freeze-steps",
        type=int,
        default=20_000,
        help="During the first N environment steps, keep actor updates frozen so only the critic adapts.",
    )
    parser.add_argument(
        "--action-noise-std",
        type=float,
        default=0.03,
        help="Small Gaussian exploration noise. The DAgger actor is already near a good nominal solution.",
    )
    parser.add_argument("--env-dt", type=float, default=0.05, help="Environment integration step.")
    parser.add_argument("--env-max-steps", type=int, default=1200, help="Maximum steps per episode.")
    parser.add_argument("--world-size", type=float, default=10.0, help="Environment world size.")
    parser.add_argument(
        "--n-obstacles",
        type=int,
        default=6,
        help="Number of dynamic obstacles in the final full-difficulty environment.",
    )
    parser.add_argument(
        "--timeout-distance-penalty-scale",
        type=float,
        default=12.0,
        help="Timeout penalty: reward -= scale * final_distance_to_goal.",
    )
    parser.add_argument("--eval-episodes", type=int, default=20, help="Evaluation episodes per checkpoint/final report.")
    parser.add_argument("--eval-freq", type=int, default=20_000, help="SB3 eval callback frequency in timesteps.")
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50_000,
        help="Checkpoint frequency in timesteps.",
    )
    parser.add_argument("--log-interval", type=int, default=10, help="SB3 log interval.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device for TD3 and dagger policy loading.",
    )
    parser.add_argument(
        "--disable-curriculum",
        action="store_true",
        help="Disable curriculum learning and train directly on the full environment.",
    )
    parser.add_argument(
        "--save-replay-buffer",
        action="store_true",
        help="Also save the TD3 replay buffer at the end and on checkpoints.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_td3_from_dagger(parse_args())
