from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from REMUSAUVEnv import REMUSAUVEnv


DEFAULT_DAGGER_POLICY_PATH = Path(r"C:\Users\kavin\Desktop\LIG2_result\20260408_110153\dagger_policy.pt")
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


def require_td3_dependencies() -> Dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
        from stable_baselines3 import TD3
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
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
        "CallbackList": CallbackList,
        "CheckpointCallback": CheckpointCallback,
        "EvalCallback": EvalCallback,
        "Monitor": Monitor,
        "NormalActionNoise": NormalActionNoise,
    }


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


def make_full_env(
    seed: Optional[int],
    monitor_cls: Any,
    env_dt: float,
    env_max_steps: int,
    n_obstacles: int,
) -> Any:
    env = REMUSAUVEnv(
        dt=env_dt,
        max_steps=env_max_steps,
        n_obstacles=n_obstacles,
        current_enabled=True,
        include_current_in_obs=True,
        seed=seed,
    )
    return monitor_cls(env)


def evaluate_policy_events(
    policy_like: Any,
    monitor_cls: Any,
    episodes: int,
    seed: int,
    env_dt: float,
    env_max_steps: int,
    n_obstacles: int,
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
        env = make_full_env(
            seed=env_seed,
            monitor_cls=monitor_cls,
            env_dt=env_dt,
            env_max_steps=env_max_steps,
            n_obstacles=n_obstacles,
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

    train_env = make_full_env(
        seed=args.seed,
        monitor_cls=monitor_cls,
        env_dt=args.env_dt,
        env_max_steps=args.env_max_steps,
        n_obstacles=args.n_obstacles,
    )
    eval_env = make_full_env(
        seed=args.eval_seed,
        monitor_cls=monitor_cls,
        env_dt=args.env_dt,
        env_max_steps=args.env_max_steps,
        n_obstacles=args.n_obstacles,
    )

    dagger_policy = load_dagger_policy(dagger_policy_path, torch_mod=torch_mod, device=args.device)

    action_dim = int(train_env.action_space.shape[0])
    action_noise = normal_action_noise_cls(
        mean=np.zeros(action_dim, dtype=np.float32),
        sigma=np.full(action_dim, args.action_noise_std, dtype=np.float32),
    )

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
        n_obstacles=args.n_obstacles,
        name="dagger_full_env",
    )
    td3_init_eval = evaluate_policy_events(
        policy_like=model,
        monitor_cls=monitor_cls,
        episodes=args.eval_episodes,
        seed=args.eval_seed,
        env_dt=args.env_dt,
        env_max_steps=args.env_max_steps,
        n_obstacles=args.n_obstacles,
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
    callbacks = callback_list_cls([checkpoint_callback, eval_callback])

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
        n_obstacles=args.n_obstacles,
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
            "action_noise_std": args.action_noise_std,
            "policy_net_arch": [32, 32],
            "policy_activation": "Tanh",
        },
        "transfer": transfer_report,
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
        description="Train TD3 on full REMUSAUVEnv, initializing the actor from a saved DAgger policy."
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
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="TD3 learning rate.")
    parser.add_argument("--buffer-size", type=int, default=500_000, help="Replay buffer size.")
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=0,
        help="Random warmup steps before TD3 uses its policy. Defaults to 0 to exploit the dagger initialization immediately.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="TD3 batch size.")
    parser.add_argument("--tau", type=float, default=0.005, help="TD3 target smoothing coefficient.")
    parser.add_argument("--gamma", type=float, default=0.99, help="TD3 discount factor.")
    parser.add_argument("--train-freq", type=int, default=1, help="TD3 train frequency.")
    parser.add_argument("--gradient-steps", type=int, default=1, help="TD3 gradient steps per update.")
    parser.add_argument("--action-noise-std", type=float, default=0.10, help="Stddev of Gaussian action noise.")
    parser.add_argument("--env-dt", type=float, default=0.05, help="Environment integration step.")
    parser.add_argument("--env-max-steps", type=int, default=1200, help="Maximum steps per episode.")
    parser.add_argument(
        "--n-obstacles",
        type=int,
        default=6,
        help="Number of dynamic obstacles. Keep > 0 for the full environment.",
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
        "--save-replay-buffer",
        action="store_true",
        help="Also save the TD3 replay buffer at the end and on checkpoints.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_td3_from_dagger(parse_args())
