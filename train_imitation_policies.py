from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import h5py
import numpy as np

from REMUSAUVEnv import REMUSAUVEnv, wrap_angle


DEFAULT_DATASET_DIR = Path(r"C:\Users\kavin\Desktop\LIG2_result\casadi_mpc_dataset_20260407_202725")
DEFAULT_RESULT_ROOT = Path(r"C:\Users\kavin\Desktop\LIG2_result")
RUN_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


@dataclass
class EpisodeSpan:
    episode: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


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


class EasyCollectionREMUSAUVEnv(REMUSAUVEnv):
    """
    Matches the easy-mode environment used by casadi_mpc_collect.py.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["current_enabled"] = False
        kwargs["n_obstacles"] = 0
        super().__init__(*args, **kwargs)

    def _integrate(self, eta: np.ndarray, nu: np.ndarray, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


class HDF5ObsActBatchIterable:
    """
    Re-iterable batch loader that streams success-only transitions from HDF5.

    `imitation.algorithms.bc.BC` accepts any iterator yielding mappings with
    `obs` and `acts`, so we avoid loading the full dataset into RAM.
    """

    def __init__(
        self,
        h5_path: Path,
        spans: Sequence[EpisodeSpan],
        batch_size: int,
        seed: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> None:
        self.h5_path = h5_path
        self.spans = list(spans)
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.total_transitions = int(sum(span.length for span in self.spans))
        self._epoch_index = 0

    def __len__(self) -> int:
        if self.drop_last:
            return self.total_transitions // self.batch_size
        return int(np.ceil(self.total_transitions / self.batch_size))

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        rng = np.random.default_rng(self.seed + self._epoch_index)
        self._epoch_index += 1

        span_order = np.arange(len(self.spans))
        if self.shuffle:
            rng.shuffle(span_order)

        obs_parts: List[np.ndarray] = []
        act_parts: List[np.ndarray] = []
        buffered = 0

        with h5py.File(self.h5_path, "r") as f:
            obs_ds = f["observations"]
            act_ds = f["actions"]

            for order_idx in span_order:
                span = self.spans[int(order_idx)]
                obs = obs_ds[span.start:span.end]
                acts = act_ds[span.start:span.end]

                local_indices = np.arange(len(obs))
                if self.shuffle:
                    rng.shuffle(local_indices)
                    obs = obs[local_indices]
                    acts = acts[local_indices]

                cursor = 0
                while cursor < len(obs):
                    needed = self.batch_size - buffered
                    take = min(needed, len(obs) - cursor)
                    obs_parts.append(obs[cursor:cursor + take])
                    act_parts.append(acts[cursor:cursor + take])
                    buffered += take
                    cursor += take

                    if buffered == self.batch_size:
                        yield {
                            "obs": np.concatenate(obs_parts, axis=0).astype(np.float32, copy=False),
                            "acts": np.concatenate(act_parts, axis=0).astype(np.float32, copy=False),
                        }
                        obs_parts.clear()
                        act_parts.clear()
                        buffered = 0

        if buffered and not self.drop_last:
            yield {
                "obs": np.concatenate(obs_parts, axis=0).astype(np.float32, copy=False),
                "acts": np.concatenate(act_parts, axis=0).astype(np.float32, copy=False),
            }


def require_training_dependencies() -> Dict[str, Any]:
    try:
        import torch
        from imitation.algorithms import bc, dagger
        from imitation.data import types
        from imitation.data.wrappers import RolloutInfoWrapper
        from stable_baselines3.common.policies import BasePolicy
        from stable_baselines3.common.torch_layers import FlattenExtractor
        from stable_baselines3.common.vec_env import DummyVecEnv
        from casadi_mpc_collect import CasadiREMUSMPC
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Install them with:\n"
            "  python -m pip install torch stable-baselines3 imitation\n"
            f"Original import error: {exc}"
        ) from exc

    return {
        "torch": torch,
        "bc": bc,
        "dagger": dagger,
        "types": types,
        "RolloutInfoWrapper": RolloutInfoWrapper,
        "BasePolicy": BasePolicy,
        "FlattenExtractor": FlattenExtractor,
        "DummyVecEnv": DummyVecEnv,
        "CasadiREMUSMPC": CasadiREMUSMPC,
    }


def resolve_run_output_dir(out_dir: Optional[Path], result_root: Path, started_at: datetime) -> Path:
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    run_dir = result_root / started_at.strftime(RUN_TIMESTAMP_FORMAT)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_selected_episode_ids(summary_csv: Path, success_only: bool, max_episodes: Optional[int]) -> List[int]:
    selected: List[int] = []
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            is_success = row.get("success", "0") == "1"
            if success_only and not is_success:
                continue
            selected.append(int(row["episode"]))
            if max_episodes is not None and len(selected) >= max_episodes:
                break
    return selected


def collect_episode_spans(h5_path: Path, episode_ids: Sequence[int]) -> List[EpisodeSpan]:
    wanted = set(int(ep) for ep in episode_ids)
    if not wanted:
        return []

    with h5py.File(h5_path, "r") as f:
        episodes = f["episode"][:]

    change_points = np.nonzero(np.diff(episodes) != 0)[0] + 1
    starts = np.concatenate(([0], change_points))
    ends = np.concatenate((change_points, [len(episodes)]))

    spans: List[EpisodeSpan] = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        episode_id = int(episodes[start])
        if episode_id in wanted:
            spans.append(EpisodeSpan(episode=episode_id, start=int(start), end=int(end)))

    spans.sort(key=lambda span: span.episode)
    return spans


def load_trajectories_for_spans(h5_path: Path, spans: Sequence[EpisodeSpan], types_mod: Any) -> List[Any]:
    trajectories: List[Any] = []
    with h5py.File(h5_path, "r") as f:
        obs_ds = f["observations"]
        next_obs_ds = f["next_observations"]
        act_ds = f["actions"]

        for span in spans:
            obs = obs_ds[span.start:span.end]
            next_obs_last = next_obs_ds[span.end - 1]
            acts = act_ds[span.start:span.end]

            traj_obs = np.empty((span.length + 1, obs.shape[1]), dtype=np.float32)
            traj_obs[:-1] = obs
            traj_obs[-1] = next_obs_last

            trajectories.append(
                types_mod.Trajectory(
                    obs=traj_obs,
                    acts=np.asarray(acts, dtype=np.float32),
                    infos=None,
                    terminal=True,
                )
            )
    return trajectories


def make_env_from_config(config: Dict[str, Any], seed: Optional[int] = None) -> REMUSAUVEnv:
    easy_env = bool(config.get("easy_env", False))
    env_cls = EasyCollectionREMUSAUVEnv if easy_env else REMUSAUVEnv
    env_n_obstacles = 0 if easy_env else int(config.get("n_obstacles", 6))
    env_current_enabled = False if easy_env else bool(config.get("current_enabled", True))

    env = env_cls(
        dt=float(config.get("env_dt", 0.05)),
        max_steps=int(config.get("env_max_steps", 1200)),
        n_obstacles=env_n_obstacles,
        current_enabled=env_current_enabled,
        seed=seed,
    )
    return env


def make_vec_env(env_config: Dict[str, Any], seed: int, dummy_vec_env_cls: Any, rollout_info_wrapper_cls: Any):
    def make_single_env(offset: int):
        def _thunk():
            env = make_env_from_config(env_config, seed=seed + offset)
            return rollout_info_wrapper_cls(env)

        return _thunk

    return dummy_vec_env_cls([make_single_env(0)])


def build_mpc_oracle_policy_class(base_policy_cls: Any, flatten_extractor_cls: Any, torch_mod: Any, mpc_cls: Any):
    class MPCOraclePolicy(base_policy_cls):
        def __init__(
            self,
            venv: Any,
            observation_space: Any,
            action_space: Any,
            horizon: int,
            replan_every: int,
            plan_dt: Optional[float],
        ) -> None:
            super().__init__(
                observation_space=observation_space,
                action_space=action_space,
                features_extractor_class=flatten_extractor_cls,
                squash_output=False,
            )
            self.venv = venv
            self._torch = torch_mod
            self._controllers = [
                mpc_cls(
                    env=env.unwrapped,
                    horizon=horizon,
                    replan_every=replan_every,
                    plan_dt=plan_dt,
                )
                for env in self.venv.envs
            ]
            self._last_step_counts = [None for _ in self._controllers]

        def _predict(self, observation: Any, deterministic: bool = False) -> Any:
            del deterministic
            actions: List[np.ndarray] = []
            for idx, wrapped_env in enumerate(self.venv.envs):
                env = wrapped_env.unwrapped
                if self._last_step_counts[idx] is None or int(env.step_count) == 0:
                    self._controllers[idx].reset()
                action = np.asarray(self._controllers[idx].act(), dtype=np.float32)
                actions.append(action)
                self._last_step_counts[idx] = int(env.step_count)

            actions_np = np.stack(actions, axis=0)
            return self._torch.as_tensor(actions_np, device=observation.device)

    return MPCOraclePolicy


def evaluate_policy(policy: Any, env_config: Dict[str, Any], episodes: int, seed: int, name: str) -> EvalSummary:
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
        env = make_env_from_config(env_config, seed=env_seed)
        obs, _ = env.reset(seed=env_seed)
        ep_return = 0.0
        event = "timeout"
        steps = 0

        for t in range(env.max_steps):
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            steps = t + 1
            if terminated or truncated:
                event = str(info.get("event", "timeout" if truncated else "other"))
                break

        if event not in event_counts:
            event = "other"
        event_counts[event] += 1
        returns.append(ep_return)
        steps_taken.append(steps)

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


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_policy(owner: Any, path: Path, torch_mod: Any) -> None:
    if hasattr(owner, "save_policy"):
        owner.save_policy(path)
        return
    torch_mod.save(owner.policy, path)


def build_bc_trainer(
    bc_mod: Any,
    observation_space: Any,
    action_space: Any,
    rng: np.random.Generator,
    batch_size: int,
    minibatch_size: int,
    device: str,
    demonstrations: Optional[Iterable[Dict[str, np.ndarray]]] = None,
) -> Any:
    base_kwargs: Dict[str, Any] = {
        "observation_space": observation_space,
        "action_space": action_space,
        "rng": rng,
        "batch_size": batch_size,
        "minibatch_size": minibatch_size,
        "device": device,
    }
    if demonstrations is not None:
        base_kwargs["demonstrations"] = demonstrations

    constructor_attempts = [
        dict(base_kwargs),
        {k: v for k, v in base_kwargs.items() if k != "device"},
        {k: v for k, v in base_kwargs.items() if k not in {"device", "minibatch_size"}},
    ]
    last_error: Optional[TypeError] = None
    for kwargs in constructor_attempts:
        try:
            return bc_mod.BC(**kwargs)
        except TypeError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def train_bc_compat(bc_trainer: Any, n_epochs: int) -> None:
    train_attempts = [
        {"n_epochs": n_epochs, "progress_bar": True, "log_rollouts_n_episodes": 0},
        {"n_epochs": n_epochs, "progress_bar": True},
        {"n_epochs": n_epochs},
    ]
    last_error: Optional[TypeError] = None
    for kwargs in train_attempts:
        try:
            bc_trainer.train(**kwargs)
            return
        except TypeError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def train_dagger_compat(
    dagger_trainer: Any,
    total_timesteps: int,
    round_min_episodes: int,
    round_min_timesteps: int,
    dagger_bc_epochs: int,
) -> None:
    bc_train_kwargs = {"n_epochs": dagger_bc_epochs, "progress_bar": True, "log_rollouts_n_episodes": 0}
    attempts = [
        {
            "total_timesteps": total_timesteps,
            "rollout_round_min_episodes": round_min_episodes,
            "rollout_round_min_timesteps": round_min_timesteps,
            "bc_train_kwargs": bc_train_kwargs,
        },
        {
            "total_timesteps": total_timesteps,
            "rollout_round_min_episodes": round_min_episodes,
            "rollout_round_min_timesteps": round_min_timesteps,
            "bc_train_kwargs": {"n_epochs": dagger_bc_epochs},
        },
        {
            "total_timesteps": total_timesteps,
        },
    ]
    last_error: Optional[TypeError] = None
    for kwargs in attempts:
        try:
            dagger_trainer.train(**kwargs)
            return
        except TypeError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def train_policies(args: argparse.Namespace) -> Dict[str, Any]:
    deps = require_training_dependencies()
    torch_mod = deps["torch"]
    bc = deps["bc"]
    dagger = deps["dagger"]
    types_mod = deps["types"]
    rollout_info_wrapper_cls = deps["RolloutInfoWrapper"]
    base_policy_cls = deps["BasePolicy"]
    flatten_extractor_cls = deps["FlattenExtractor"]
    dummy_vec_env_cls = deps["DummyVecEnv"]
    casadi_mpc_cls = deps["CasadiREMUSMPC"]

    dataset_dir = args.dataset_dir.resolve()
    if args.batch_size % args.minibatch_size != 0:
        raise ValueError("--batch-size must be divisible by --minibatch-size.")

    h5_path = dataset_dir / "transitions.h5"
    config_path = dataset_dir / "config.json"
    summary_csv = dataset_dir / "episode_summary.csv"
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 dataset not found: {h5_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")
    if not summary_csv.exists():
        raise FileNotFoundError(f"Episode summary CSV not found: {summary_csv}")

    started_at = datetime.now()
    run_dir = resolve_run_output_dir(args.out_dir, args.result_root, started_at)
    bc_policy_path = run_dir / "bc_policy.pt"
    dagger_policy_path = run_dir / "dagger_policy.pt"
    dagger_scratch_dir = run_dir / "dagger_scratch"
    dagger_scratch_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = load_json(config_path)
    selected_episode_ids = load_selected_episode_ids(
        summary_csv=summary_csv,
        success_only=not args.include_failures,
        max_episodes=args.max_success_episodes,
    )
    spans = collect_episode_spans(h5_path, selected_episode_ids)
    if not spans:
        raise RuntimeError("No episodes selected from the expert dataset.")

    probe_env = make_env_from_config(dataset_config, seed=args.seed)
    observation_space = probe_env.observation_space
    action_space = probe_env.action_space

    bc_demos = HDF5ObsActBatchIterable(
        h5_path=h5_path,
        spans=spans,
        batch_size=args.minibatch_size,
        seed=args.seed,
        shuffle=True,
        drop_last=True,
    )

    rng_bc = np.random.default_rng(args.seed)
    bc_trainer = build_bc_trainer(
        bc_mod=bc,
        observation_space=observation_space,
        action_space=action_space,
        rng=rng_bc,
        demonstrations=bc_demos,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size,
        device=args.device,
    )
    train_bc_compat(bc_trainer, n_epochs=args.bc_epochs)
    save_policy(bc_trainer, bc_policy_path, torch_mod)

    dagger_seed_count = args.dagger_initial_trajectories
    if dagger_seed_count is None or dagger_seed_count <= 0:
        dagger_seed_spans = spans
    else:
        dagger_seed_spans = spans[:dagger_seed_count]
    dagger_seed_trajs = load_trajectories_for_spans(h5_path, dagger_seed_spans, types_mod)

    # Installed imitation versions build DAgger demo batches with
    # `bc_trainer.batch_size`, then validate them against
    # `bc_trainer.minibatch_size` in `BC.set_demonstrations()`.
    # Keeping them equal avoids a 1024 vs 256 mismatch in this code path.
    dagger_student_bc = build_bc_trainer(
        bc_mod=bc,
        observation_space=observation_space,
        action_space=action_space,
        rng=np.random.default_rng(args.seed + 1),
        batch_size=args.minibatch_size,
        minibatch_size=args.minibatch_size,
        device=args.device,
    )
    venv = make_vec_env(
        dataset_config,
        seed=args.seed + 10_000,
        dummy_vec_env_cls=dummy_vec_env_cls,
        rollout_info_wrapper_cls=rollout_info_wrapper_cls,
    )
    mpc_oracle_policy_cls = build_mpc_oracle_policy_class(
        base_policy_cls=base_policy_cls,
        flatten_extractor_cls=flatten_extractor_cls,
        torch_mod=torch_mod,
        mpc_cls=casadi_mpc_cls,
    )
    mpc_oracle_policy = mpc_oracle_policy_cls(
        venv=venv,
        observation_space=observation_space,
        action_space=action_space,
        horizon=int(dataset_config.get("horizon", args.oracle_horizon)),
        replan_every=int(dataset_config.get("replan_every", args.oracle_replan_every)),
        plan_dt=dataset_config.get("plan_dt", args.oracle_plan_dt),
    )
    dagger_kwargs: Dict[str, Any] = {}
    if hasattr(dagger, "ExponentialBetaSchedule"):
        dagger_kwargs["beta_schedule"] = dagger.ExponentialBetaSchedule(args.dagger_beta_decay)

    dagger_trainer = dagger.SimpleDAggerTrainer(
        venv=venv,
        scratch_dir=dagger_scratch_dir,
        expert_policy=mpc_oracle_policy,
        expert_trajs=dagger_seed_trajs,
        rng=np.random.default_rng(args.seed + 2),
        bc_trainer=dagger_student_bc,
        **dagger_kwargs,
    )
    train_dagger_compat(
        dagger_trainer=dagger_trainer,
        total_timesteps=args.dagger_total_timesteps,
        round_min_episodes=args.dagger_round_min_episodes,
        round_min_timesteps=args.dagger_round_min_timesteps,
        dagger_bc_epochs=args.dagger_bc_epochs,
    )
    save_policy(dagger_trainer, dagger_policy_path, torch_mod)
    dagger_policy = getattr(dagger_trainer, "policy", dagger_student_bc.policy)
    venv.close()

    bc_eval = evaluate_policy(
        policy=bc_trainer.policy,
        env_config=dataset_config,
        episodes=args.eval_episodes,
        seed=args.eval_seed,
        name="bc",
    )
    dagger_eval = evaluate_policy(
        policy=dagger_policy,
        env_config=dataset_config,
        episodes=args.eval_episodes,
        seed=args.eval_seed,
        name="dagger",
    )

    summary = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "run_dir": str(run_dir),
        "selected_episodes": len(spans),
        "selected_transitions": int(sum(span.length for span in spans)),
        "success_only": not args.include_failures,
        "dagger_initial_trajectories": len(dagger_seed_spans),
        "bc_policy_path": str(bc_policy_path),
        "dagger_policy_path": str(dagger_policy_path),
        "dagger_scratch_dir": str(dagger_scratch_dir),
        "dataset_config": dataset_config,
        "training": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "minibatch_size": args.minibatch_size,
            "bc_epochs": args.bc_epochs,
            "dagger_total_timesteps": args.dagger_total_timesteps,
            "dagger_round_min_episodes": args.dagger_round_min_episodes,
            "dagger_round_min_timesteps": args.dagger_round_min_timesteps,
            "dagger_bc_epochs": args.dagger_bc_epochs,
            "dagger_beta_decay": args.dagger_beta_decay,
            "device": args.device,
            "oracle_horizon": int(dataset_config.get("horizon", args.oracle_horizon)),
            "oracle_replan_every": int(dataset_config.get("replan_every", args.oracle_replan_every)),
            "oracle_plan_dt": dataset_config.get("plan_dt", args.oracle_plan_dt),
        },
        "evaluation": {
            "bc": asdict(bc_eval),
            "dagger": asdict(dagger_eval),
        },
        "notes": {
            "dagger_expert_source": "Actual CasadiREMUSMPC oracle queried on learner rollout states",
            "dagger_seed_data": "Initial expert trajectories loaded from the provided HDF5 dataset",
        },
    }
    save_json(run_dir / "train_summary.json", summary)
    save_json(run_dir / "bc_eval.json", asdict(bc_eval))
    save_json(run_dir / "dagger_eval.json", asdict(dagger_eval))

    print(
        f"BC eval success_rate={bc_eval.success_rate:.3f} "
        f"(goal/coll/oob/to={bc_eval.goal_count}/{bc_eval.collision_count}/"
        f"{bc_eval.out_of_bounds_count}/{bc_eval.timeout_count})",
        flush=True,
    )
    print(
        f"DAgger eval success_rate={dagger_eval.success_rate:.3f} "
        f"(goal/coll/oob/to={dagger_eval.goal_count}/{dagger_eval.collision_count}/"
        f"{dagger_eval.out_of_bounds_count}/{dagger_eval.timeout_count})",
        flush=True,
    )
    print(f"saved policies to {run_dir}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BC and BC+DAgger policies from an expert HDF5 dataset using the imitation library."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing transitions.h5, config.json, and episode_summary.csv.",
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
    parser.add_argument(
        "--include-failures",
        action="store_true",
        help="Include failed trajectories instead of training only on successful ones.",
    )
    parser.add_argument(
        "--max-success-episodes",
        type=int,
        default=None,
        help="Optional cap on the number of selected trajectories.",
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="BC/DAgger BC batch size.")
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=256,
        help="Gradient minibatch size used inside BC.",
    )
    parser.add_argument("--bc-epochs", type=int, default=5, help="Number of BC epochs.")
    parser.add_argument(
        "--dagger-total-timesteps",
        type=int,
        default=100_000,
        help="Environment timesteps to collect during DAgger.",
    )
    parser.add_argument(
        "--dagger-round-min-episodes",
        type=int,
        default=8,
        help="Minimum completed episodes per DAgger aggregation round.",
    )
    parser.add_argument(
        "--dagger-round-min-timesteps",
        type=int,
        default=4_000,
        help="Minimum timesteps per DAgger aggregation round.",
    )
    parser.add_argument(
        "--dagger-bc-epochs",
        type=int,
        default=1,
        help="Number of BC epochs to run after each DAgger aggregation round.",
    )
    parser.add_argument(
        "--dagger-beta-decay",
        type=float,
        default=0.7,
        help="Exponential beta decay for DAgger's expert-action mixture.",
    )
    parser.add_argument(
        "--dagger-initial-trajectories",
        type=int,
        default=512,
        help="Number of expert trajectories to seed into round 0 of DAgger. Use <=0 for all selected trajectories.",
    )
    parser.add_argument(
        "--oracle-horizon",
        type=int,
        default=12,
        help="Fallback MPC horizon when config.json does not specify one.",
    )
    parser.add_argument(
        "--oracle-replan-every",
        type=int,
        default=3,
        help="Fallback MPC replanning interval when config.json does not specify one.",
    )
    parser.add_argument(
        "--oracle-plan-dt",
        type=float,
        default=None,
        help="Fallback MPC planning dt when config.json does not specify one.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes for each trained policy.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=20_000,
        help="Base seed for post-training evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device passed to imitation BC.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_policies(parse_args())
