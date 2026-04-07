import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import gymnasium as gym


from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from REMUSAUVEnv import REMUSAUVEnv


RUN_NAME = "remus_current"
ROOT_DIR = Path("./runs") / RUN_NAME
LOG_DIR = ROOT_DIR / "logs"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
SUMMARY_CSV_PATH = LOG_DIR / "training_summary.csv"
MODEL_PATH = ROOT_DIR / "ppo_remus_current_model"
VECNORM_PATH = ROOT_DIR / "ppo_remus_current_vecnormalize.pkl"
TOTAL_TIMESTEPS = 500_000


class SuccessInfoWrapper(gym.Wrapper):
    """Inject SB3-compatible success information into terminal infos."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        if done:
            info = dict(info)
            info["is_success"] = bool(info.get("event") == "goal")
        return obs, reward, terminated, truncated, info


class PPOWithSummary(PPO):
    """
    PPO subclass that:
      - tracks terminal event counts
      - prints a compact one-line summary at each log step
      - saves the same summary rows to a dedicated CSV file

    Notes:
      - success_rate_window follows SB3 convention: recent mean over ep_success_buffer
      - success_rate_total is cumulative over all completed episodes in this run
    """

    def __init__(
        self,
        *args,
        summary_csv_path: Union[str, os.PathLike],
        reset_summary_csv: bool = True,
        **kwargs,
    ):
        self.summary_csv_path = Path(summary_csv_path)
        self.summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if reset_summary_csv and self.summary_csv_path.exists():
            self.summary_csv_path.unlink()

        self.total_episodes = 0
        self.total_successes = 0
        self.event_counts: Dict[str, int] = {
            "goal": 0,
            "collision": 0,
            "out_of_bounds": 0,
            "timeout": 0,
            "other": 0,
        }

        super().__init__(*args, **kwargs)

    def _update_info_buffer(self, infos, dones: Optional[np.ndarray] = None) -> None:
        super()._update_info_buffer(infos, dones)

        if dones is None:
            dones = np.array([False] * len(infos), dtype=bool)

        for done, info in zip(dones, infos):
            if not bool(done):
                continue

            self.total_episodes += 1
            event = str(info.get("event", "other"))
            if event not in self.event_counts:
                event = "other"
            self.event_counts[event] += 1

            is_success = bool(info.get("is_success", info.get("event") == "goal"))
            if is_success:
                self.total_successes += 1

    @staticmethod
    def _to_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    @staticmethod
    def _fmt(value: Any, digits: int = 3) -> str:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "nan"
        if np.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"

    def _append_summary_row(self, row: Dict[str, Any]) -> None:
        write_header = not self.summary_csv_path.exists()
        with self.summary_csv_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def dump_logs(self, iteration: int = 0) -> None:
        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)

        ep_rew_mean = float("nan")
        ep_len_mean = float("nan")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            ep_rew_mean = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
            ep_len_mean = safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer])
            self.logger.record("rollout/ep_rew_mean", ep_rew_mean)
            self.logger.record("rollout/ep_len_mean", ep_len_mean)

        success_rate_window = float("nan")
        if len(self.ep_success_buffer) > 0:
            success_rate_window = safe_mean(self.ep_success_buffer)
            self.logger.record("rollout/success_rate", success_rate_window)

        success_rate_total = float("nan")
        if self.total_episodes > 0:
            success_rate_total = self.total_successes / self.total_episodes

        if iteration > 0:
            self.logger.record("time/iterations", iteration, exclude="tensorboard")
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        self.logger.record("custom/success_rate_total", 0.0 if np.isnan(success_rate_total) else success_rate_total)
        self.logger.record("custom/episodes_total", self.total_episodes, exclude="tensorboard")
        self.logger.record("custom/successes_total", self.total_successes, exclude="tensorboard")
        self.logger.record("custom/goal_count", self.event_counts["goal"], exclude="tensorboard")
        self.logger.record("custom/collision_count", self.event_counts["collision"], exclude="tensorboard")
        self.logger.record("custom/out_of_bounds_count", self.event_counts["out_of_bounds"], exclude="tensorboard")
        self.logger.record("custom/timeout_count", self.event_counts["timeout"], exclude="tensorboard")
        self.logger.record("custom/other_terminal_count", self.event_counts["other"], exclude="tensorboard")

        metrics = dict(self.logger.name_to_value)

        row = {
            "iteration": int(iteration),
            "total_timesteps": int(self.num_timesteps),
            "time_elapsed_sec": int(time_elapsed),
            "fps": int(fps),
            "episodes_total": int(self.total_episodes),
            "successes_total": int(self.total_successes),
            "success_rate_window": self._to_float(success_rate_window),
            "success_rate_total": self._to_float(success_rate_total),
            "goal_count": int(self.event_counts["goal"]),
            "collision_count": int(self.event_counts["collision"]),
            "out_of_bounds_count": int(self.event_counts["out_of_bounds"]),
            "timeout_count": int(self.event_counts["timeout"]),
            "other_terminal_count": int(self.event_counts["other"]),
            "ep_rew_mean": self._to_float(ep_rew_mean),
            "ep_len_mean": self._to_float(ep_len_mean),
            "approx_kl": self._to_float(metrics.get("train/approx_kl", float("nan"))),
            "clip_fraction": self._to_float(metrics.get("train/clip_fraction", float("nan"))),
            "entropy_loss": self._to_float(metrics.get("train/entropy_loss", float("nan"))),
            "explained_variance": self._to_float(metrics.get("train/explained_variance", float("nan"))),
            "learning_rate": self._to_float(metrics.get("train/learning_rate", float("nan"))),
            "loss": self._to_float(metrics.get("train/loss", float("nan"))),
            "n_updates": int(self._to_float(metrics.get("train/n_updates", 0))),
            "policy_gradient_loss": self._to_float(metrics.get("train/policy_gradient_loss", float("nan"))),
            "std": self._to_float(metrics.get("train/std", float("nan"))),
            "value_loss": self._to_float(metrics.get("train/value_loss", float("nan"))),
        }
        self._append_summary_row(row)

        line = (
            f"iter={row['iteration']:04d} | steps={row['total_timesteps']:>7d} | fps={row['fps']:>3d} | "
            f"ep_rew={self._fmt(row['ep_rew_mean'], 2):>7} | ep_len={self._fmt(row['ep_len_mean'], 1):>6} | "
            f"succ_recent={self._fmt(row['success_rate_window'], 3):>5} | succ_total={self._fmt(row['success_rate_total'], 3):>5} | "
            f"goal/coll/oob/to={row['goal_count']}/{row['collision_count']}/{row['out_of_bounds_count']}/{row['timeout_count']} | "
            f"kl={self._fmt(row['approx_kl'], 4):>6} | clip={self._fmt(row['clip_fraction'], 3):>5} | "
            f"ev={self._fmt(row['explained_variance'], 3):>5} | pg={self._fmt(row['policy_gradient_loss'], 4):>7} | "
            f"vf={self._fmt(row['value_loss'], 4):>7} | std={self._fmt(row['std'], 3):>5}"
        )
        print(line, flush=True)

        self.logger.dump(step=self.num_timesteps)


def build_env(seed: int = 0):
    env = REMUSAUVEnv(seed=seed, current_enabled=True, include_current_in_obs=True)
    env = SuccessInfoWrapper(env)
    env = Monitor(env)
    return env


def make_env(seed: int = 0):
    def _init():
        return build_env(seed=seed)

    return _init


def build_logger(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        return configure(str(log_dir), ["csv", "tensorboard"])
    except Exception:
        return configure(str(log_dir), ["csv"])


if __name__ == "__main__":
    ROOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    raw_env = build_env(seed=0)
    check_env(raw_env, warn=True)
    raw_env.close()

    vec_env = DummyVecEnv([make_env(0)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPOWithSummary(
        "MlpPolicy",
        vec_env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        tensorboard_log=str(LOG_DIR),
        summary_csv_path=SUMMARY_CSV_PATH,
    )
    model.set_logger(build_logger(LOG_DIR))

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="ppo_remus_current",
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback, log_interval=1)
    model.save(str(MODEL_PATH))
    vec_env.save(str(VECNORM_PATH))
    vec_env.close()

    print(f"Summary CSV saved to: {SUMMARY_CSV_PATH}")
    print(f"Model saved to: {MODEL_PATH}.zip")
    print(f"VecNormalize stats saved to: {VECNORM_PATH}")
