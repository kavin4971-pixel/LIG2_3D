from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ============================================================
# Utilities
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


ENV_FILE_CANDIDATES = [
    "REMUSAUVEnv.py",
    "REMUSAUVEnv_sim_only.py",
    "REMUSAUVEnv_random_target.py",
    "REMUSAUVEnv(23).py",
    "REMUSAUVEnv(24).py",
]

RUN_VERSION = (
    "Transformer PPO REMUS diagnostics v5.2 | stage4 >90% eval tuning, "
    "long-current time budget, earlier obstacle guidance"
)


class RunningMeanStd:
    """Numerically stable running mean/std for observation normalization."""

    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            batch_mean = x
            batch_var = np.zeros_like(x)
            batch_count = 1.0
        else:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = float(x.shape[0])
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-8)
        self.count = total_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        y = (x - self.mean.astype(np.float32)) / np.sqrt(self.var.astype(np.float32) + 1e-8)
        return np.clip(y, -clip, clip)

    def state_dict(self) -> Dict[str, np.ndarray | float]:
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": float(self.count),
        }

    def load_state_dict(self, state: Dict[str, np.ndarray | float]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


class ObsNormWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, obs_rms: RunningMeanStd, update_stats: bool = True, clip_obs: float = 10.0):
        super().__init__(env)
        self.obs_rms = obs_rms
        self.update_stats = update_stats
        self.clip_obs = clip_obs

    def observation(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        if self.update_stats:
            self.obs_rms.update(obs)
        return self.obs_rms.normalize(obs, clip=self.clip_obs).astype(np.float32)


@dataclass
class CurriculumStage:
    stage_id: int
    n_obstacles: int
    target_z_min: float
    target_z_max: float
    label: str
    success_gate: Optional[float] = None
    target_distance_min: Optional[float] = None
    target_distance_max: Optional[float] = None
    target_boundary_margin: Optional[float] = None
    max_steps: Optional[int] = None
    current_enabled: bool = True
    current_speed_min: Optional[float] = None
    current_speed_max: Optional[float] = None
    current_vertical_max: Optional[float] = None
    current_osc_amp_xy: Optional[float] = None
    current_osc_amp_z: Optional[float] = None
    obstacle_radius_min: Optional[float] = None
    obstacle_radius_max: Optional[float] = None
    obstacle_speed_min: Optional[float] = None
    obstacle_speed_max: Optional[float] = None
    prefer_path_obstacles: bool = False
    obstacle_path_lateral_jitter: Optional[float] = None
    obstacle_path_depth_jitter: Optional[float] = None
    guidance_cruise_propeller: Optional[float] = None
    guidance_near_propeller: Optional[float] = None
    guidance_slow_radius: Optional[float] = None
    guidance_full_speed_radius: Optional[float] = None
    guidance_propeller_residual_scale: Optional[float] = None
    guidance_surface_residual_scale: Optional[float] = None
    guidance_obstacle_avoidance: bool = False
    guidance_obstacle_avoid_distance: Optional[float] = None
    guidance_obstacle_avoid_margin: Optional[float] = None
    guidance_obstacle_avoid_gain: Optional[float] = None
    guidance_obstacle_slowdown: Optional[float] = None


DEFAULT_CURRICULUM = [
    CurriculumStage(
        stage_id=1,
        n_obstacles=0,
        target_z_min=8.0,
        target_z_max=15.0,
        label="short_2.5D_no_current",
        success_gate=0.60,
        target_distance_min=25.0,
        target_distance_max=45.0,
        target_boundary_margin=5.0,
        current_enabled=False,
    ),
    CurriculumStage(
        stage_id=2,
        n_obstacles=0,
        target_z_min=5.0,
        target_z_max=35.0,
        label="medium_3D_current",
        success_gate=0.70,
        target_distance_min=40.0,
        target_distance_max=80.0,
        target_boundary_margin=5.0,
        current_enabled=True,
    ),
    CurriculumStage(
        stage_id=3,
        n_obstacles=6,
        target_z_min=5.0,
        target_z_max=55.0,
        label="3D_with_obstacles",
        success_gate=0.90,
        target_distance_min=50.0,
        target_distance_max=100.0,
        target_boundary_margin=5.0,
        current_enabled=True,
    ),
    CurriculumStage(
        stage_id=4,
        n_obstacles=12,
        target_z_min=5.0,
        target_z_max=55.0,
        label="strong_current_moving_obstacles",
        success_gate=None,
        target_distance_min=60.0,
        target_distance_max=110.0,
        target_boundary_margin=8.0,
        max_steps=3000,
        current_enabled=True,
        current_speed_min=0.55,
        current_speed_max=0.95,
        current_vertical_max=0.08,
        current_osc_amp_xy=0.18,
        current_osc_amp_z=0.04,
        obstacle_radius_min=1.2,
        obstacle_radius_max=3.5,
        obstacle_speed_min=0.15,
        obstacle_speed_max=0.45,
        prefer_path_obstacles=True,
        obstacle_path_lateral_jitter=9.0,
        obstacle_path_depth_jitter=9.0,
        guidance_cruise_propeller=0.80,
        guidance_near_propeller=0.55,
        guidance_slow_radius=1.35,
        guidance_full_speed_radius=7.0,
        guidance_propeller_residual_scale=0.15,
        guidance_surface_residual_scale=0.35,
        guidance_obstacle_avoidance=True,
        guidance_obstacle_avoid_distance=34.0,
        guidance_obstacle_avoid_margin=9.0,
        guidance_obstacle_avoid_gain=1.55,
        guidance_obstacle_slowdown=0.22,
    ),
]


def _load_env_class_from_path(env_path: Path):
    env_path = env_path.expanduser().resolve()
    if not env_path.exists():
        raise FileNotFoundError(f"Environment file not found: {env_path}")

    module_name = f"remus_env_module_{abs(hash(str(env_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, env_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load environment module from {env_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "REMUSAUVEnv"):
        raise AttributeError(f"{env_path} does not define REMUSAUVEnv")
    return module.REMUSAUVEnv


def load_env_class():
    if "REMUSAUVEnv" in globals():
        return globals()["REMUSAUVEnv"]

    base_dir = Path(__file__).resolve().parent
    for candidate in ENV_FILE_CANDIDATES:
        candidate_path = base_dir / candidate
        if candidate_path.exists():
            return _load_env_class_from_path(candidate_path)

    raise FileNotFoundError(
        "Could not find REMUSAUVEnv in this script or alongside it. "
        f"Looked for: {', '.join(ENV_FILE_CANDIDATES)} in {base_dir}"
    )


# ============================================================
# Reward wrapper: physics env stays unchanged, RL reward is external
# ============================================================


@dataclass
class RewardConfig:
    progress_weight: float = 180.0
    closing_speed_weight: float = 0.04
    alignment_weight: float = 0.008
    step_penalty: float = 0.002
    action_l2_weight: float = 0.002
    action_smooth_weight: float = 0.001
    roll_weight: float = 0.006
    pitch_weight: float = 0.018
    obstacle_weight: float = 1.8
    obstacle_margin: float = 7.0
    orbit_penalty_weight: float = 1.0
    orbit_window: int = 16
    orbit_progress_eps_norm: float = 0.003
    orbit_yaw_eps: float = math.radians(20.0)
    goal_bonus: float = 600.0
    collision_penalty: float = 240.0
    out_of_bounds_penalty: float = 80.0
    timeout_penalty: float = 70.0
    timeout_distance_penalty_weight: float = 12.0


REWARD_COMPONENT_KEYS = [
    "progress_reward",
    "closing_speed_reward",
    "alignment_reward",
    "step_penalty",
    "action_l2_penalty",
    "action_smooth_penalty",
    "roll_penalty",
    "pitch_penalty",
    "obstacle_clearance_penalty",
    "orbit_penalty",
    "goal_bonus",
    "collision_penalty",
    "out_of_bounds_penalty",
    "timeout_penalty",
    "timeout_distance_penalty",
]


class RemusRewardWrapper(gym.Wrapper):
    """
    Adds an external RL reward on top of the simulation-only environment.

    The wrapped REMUSAUVEnv already returns:
      - reward = 0.0
      - info['progress'], info['distance_to_goal'], terminal flags, etc.

    This wrapper uses only observation + info to define the training reward.
    Current is hidden from the policy because the underlying env is instantiated
    with include_current_in_obs=False.
    """

    def __init__(self, env: gym.Env, cfg: RewardConfig):
        super().__init__(env)
        self.cfg = cfg
        self.prev_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        self.yaw_history: Deque[float] = deque(maxlen=cfg.orbit_window)
        self.dist_history: Deque[float] = deque(maxlen=cfg.orbit_window)
        self.initial_distance = 1.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_action.fill(0.0)
        self.yaw_history.clear()
        self.dist_history.clear()
        self.initial_distance = max(float(info.get("mission_distance", np.linalg.norm(obs[12:15]))), 1.0)
        self.yaw_history.append(float(obs[5]))
        self.dist_history.append(float(np.linalg.norm(obs[12:15])))
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        progress_norm = float(info["progress"]) / self.initial_distance
        reward_components: Dict[str, float] = {
            "progress_reward": self.cfg.progress_weight * progress_norm,
            "closing_speed_reward": 0.0,
            "alignment_reward": 0.0,
            "step_penalty": -self.cfg.step_penalty,
            "action_l2_penalty": 0.0,
            "action_smooth_penalty": 0.0,
            "roll_penalty": 0.0,
            "pitch_penalty": 0.0,
            "obstacle_clearance_penalty": 0.0,
            "orbit_penalty": 0.0,
            "goal_bonus": 0.0,
            "collision_penalty": 0.0,
            "out_of_bounds_penalty": 0.0,
            "timeout_penalty": 0.0,
            "timeout_distance_penalty": 0.0,
        }

        roll = float(obs[3])
        pitch = float(obs[4])
        rel_target_body = obs[12:15].astype(np.float32)
        target_dist = float(np.linalg.norm(rel_target_body))
        closing_speed = max(float(info["progress"]) / max(float(getattr(self.unwrapped, "dt", 0.05)), 1e-6), 0.0)
        target_alignment = _finite_float(info.get("target_alignment"), 0.0)

        action = np.asarray(action, dtype=np.float32)
        action_l2 = float(np.dot(action, action))
        action_delta_l2 = float(np.dot(action - self.prev_action, action - self.prev_action))
        reward_components["closing_speed_reward"] = self.cfg.closing_speed_weight * closing_speed
        reward_components["alignment_reward"] = self.cfg.alignment_weight * max(target_alignment, 0.0)
        reward_components["action_l2_penalty"] = -self.cfg.action_l2_weight * action_l2
        reward_components["action_smooth_penalty"] = -self.cfg.action_smooth_weight * action_delta_l2
        reward_components["roll_penalty"] = -self.cfg.roll_weight * abs(roll)
        reward_components["pitch_penalty"] = -self.cfg.pitch_weight * abs(pitch)

        obs_feats = obs[-12:]
        min_clearance = np.inf
        for i in range(0, 12, 4):
            rel = obs_feats[i : i + 3]
            rad = float(obs_feats[i + 3])
            if rad <= 0.0:
                continue
            clear = float(np.linalg.norm(rel) - (0.25 + rad))
            min_clearance = min(min_clearance, clear)
        if np.isfinite(min_clearance) and min_clearance < self.cfg.obstacle_margin:
            reward_components["obstacle_clearance_penalty"] = -self.cfg.obstacle_weight * (
                self.cfg.obstacle_margin - min_clearance
            )

        yaw_now = float(obs[5])
        self.yaw_history.append(yaw_now)
        self.dist_history.append(target_dist)
        if len(self.yaw_history) >= self.cfg.orbit_window:
            yaw_range = float(np.max(self.yaw_history) - np.min(self.yaw_history))
            dist_gain_norm = float(self.dist_history[0] - self.dist_history[-1]) / self.initial_distance
            if yaw_range > self.cfg.orbit_yaw_eps and dist_gain_norm < self.cfg.orbit_progress_eps_norm:
                reward_components["orbit_penalty"] = -self.cfg.orbit_penalty_weight

        if info.get("goal_reached", False):
            reward_components["goal_bonus"] = self.cfg.goal_bonus
        if info.get("collision", False):
            reward_components["collision_penalty"] = -self.cfg.collision_penalty
        if info.get("out_of_bounds", False):
            reward_components["out_of_bounds_penalty"] = -self.cfg.out_of_bounds_penalty
        if truncated and info.get("event") == "timeout":
            reward_components["timeout_penalty"] = -self.cfg.timeout_penalty
            reward_components["timeout_distance_penalty"] = -self.cfg.timeout_distance_penalty_weight * target_dist

        reward = float(sum(reward_components.values()))
        info["reward_components"] = reward_components
        info["progress_norm"] = progress_norm
        info["closing_speed"] = closing_speed
        info["action_l2"] = action_l2
        info["action_delta_l2"] = action_delta_l2
        info["policy_min_obstacle_clearance"] = float(min_clearance) if np.isfinite(min_clearance) else float("nan")
        info["reward_total"] = reward
        self.prev_action = action.copy()
        return obs, reward, terminated, truncated, info


class PursuitGuidanceActionWrapper(gym.Wrapper):
    """
    Treats the policy output as a residual around a simple line-of-sight pursuit
    controller. This gives PPO early successful trajectories instead of asking
    it to discover forward propulsion, yaw steering, and depth steering all from
    sparse terminal rewards.
    """

    def __init__(
        self,
        env: gym.Env,
        cruise_propeller: float = 0.65,
        near_propeller: float = 0.25,
        heading_gain: float = 1.8,
        elevation_gain: float = 1.8,
        propeller_residual_scale: float = 0.20,
        surface_residual_scale: float = 0.45,
        slow_radius: float = 1.8,
        full_speed_radius: float = 8.0,
        obstacle_avoidance: bool = False,
        obstacle_avoid_distance: float = 20.0,
        obstacle_avoid_margin: float = 5.0,
        obstacle_avoid_gain: float = 0.7,
        obstacle_slowdown: float = 0.25,
    ):
        super().__init__(env)
        self.cruise_propeller = float(cruise_propeller)
        self.near_propeller = float(near_propeller)
        self.heading_gain = float(heading_gain)
        self.elevation_gain = float(elevation_gain)
        self.propeller_residual_scale = float(propeller_residual_scale)
        self.surface_residual_scale = float(surface_residual_scale)
        self.slow_radius = float(slow_radius)
        self.full_speed_radius = max(float(full_speed_radius), self.slow_radius + 1e-3)
        self.obstacle_avoidance = bool(obstacle_avoidance)
        self.obstacle_avoid_distance = float(obstacle_avoid_distance)
        self.obstacle_avoid_margin = float(obstacle_avoid_margin)
        self.obstacle_avoid_gain = float(obstacle_avoid_gain)
        self.obstacle_slowdown = float(obstacle_slowdown)
        self.last_obs: Optional[np.ndarray] = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = np.asarray(obs, dtype=np.float32).copy()
        return obs, info

    def _guidance_action(self) -> np.ndarray:
        if self.last_obs is None:
            return np.array([self.cruise_propeller, 0.0, 0.0], dtype=np.float32)

        rel_target_body = np.asarray(self.last_obs[12:15], dtype=np.float32)
        dist = float(np.linalg.norm(rel_target_body))
        bearing = float(np.arctan2(rel_target_body[1], max(float(rel_target_body[0]), 1e-3)))
        horizontal = max(float(np.linalg.norm(rel_target_body[:2])), 1e-3)
        elevation = float(np.arctan2(rel_target_body[2], horizontal))

        speed_blend = np.clip((dist - self.slow_radius) / (self.full_speed_radius - self.slow_radius), 0.0, 1.0)
        propeller = self.near_propeller + (self.cruise_propeller - self.near_propeller) * float(speed_blend)
        rudder = float(np.clip(self.heading_gain * bearing, -0.80, 0.80))
        stern = float(np.clip(self.elevation_gain * elevation, -0.80, 0.80))

        if self.obstacle_avoidance:
            obs_feats = np.asarray(self.last_obs[-12:], dtype=np.float32)
            max_danger = 0.0
            max_slowdown_danger = 0.0
            for i in range(0, 12, 4):
                rel = obs_feats[i : i + 3]
                radius = float(obs_feats[i + 3])
                if radius <= 0.0:
                    continue
                center_distance = float(np.linalg.norm(rel))
                clearance = center_distance - (0.25 + radius)
                forward = float(rel[0])
                if forward < -radius or forward > self.obstacle_avoid_distance:
                    continue
                if clearance >= self.obstacle_avoid_margin:
                    continue

                clearance_danger = (self.obstacle_avoid_margin - clearance) / max(self.obstacle_avoid_margin, 1e-6)
                forward_danger = (self.obstacle_avoid_distance - max(forward, 0.0)) / max(self.obstacle_avoid_distance, 1e-6)
                centerline_scale = 1.0 - np.clip(abs(float(rel[1])) / max(radius + 2.5, 1e-6), 0.0, 1.0)
                danger = float(
                    np.clip(
                        clearance_danger * (0.35 + 0.65 * forward_danger) * (1.0 + 0.45 * centerline_scale),
                        0.0,
                        1.0,
                    )
                )
                max_danger = max(max_danger, danger)

                lateral = float(rel[1])
                vertical = float(rel[2])
                side = -np.sign(lateral) if abs(lateral) > 0.20 else -np.sign(bearing if abs(bearing) > 0.05 else 1.0)
                depth_side = -np.sign(vertical) if abs(vertical) > 0.20 else -np.sign(elevation if abs(elevation) > 0.05 else 1.0)
                rudder += self.obstacle_avoid_gain * danger * float(side) * (0.85 + 0.35 * centerline_scale)
                stern += self.obstacle_avoid_gain * danger * float(depth_side) * (0.75 + 0.25 * centerline_scale)

                close_clearance = min(3.5, self.obstacle_avoid_margin)
                if clearance < close_clearance and forward < 0.65 * self.obstacle_avoid_distance:
                    slow_danger = (close_clearance - clearance) / max(close_clearance, 1e-6)
                    max_slowdown_danger = max(max_slowdown_danger, float(np.clip(slow_danger, 0.0, 1.0)))

            if max_slowdown_danger > 0.0:
                propeller *= 1.0 - self.obstacle_slowdown * max_slowdown_danger
            if dist > 12.0 and max_danger < 0.35:
                propeller = max(propeller, 0.96 * self.cruise_propeller)

        depth = float(self.last_obs[2])
        world_size = float(getattr(self.unwrapped, "world_size", 60.0))
        depth_margin = 6.0
        if depth < depth_margin:
            stern += 0.55 * (depth_margin - depth) / depth_margin
            propeller = max(propeller, self.near_propeller)
        elif depth > world_size - depth_margin:
            stern -= 0.55 * (depth - (world_size - depth_margin)) / depth_margin
            propeller = max(propeller, self.near_propeller)

        return np.array([propeller, rudder, stern], dtype=np.float32)

    def step(self, action):
        policy_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        guidance = self._guidance_action()
        residual_scale = np.array(
            [self.propeller_residual_scale, self.surface_residual_scale, self.surface_residual_scale],
            dtype=np.float32,
        )
        guided_action = np.clip(guidance + residual_scale * policy_action, -1.0, 1.0).astype(np.float32)
        obs, reward, terminated, truncated, info = self.env.step(guided_action)
        self.last_obs = np.asarray(obs, dtype=np.float32).copy()
        info["policy_action"] = policy_action.copy()
        info["guidance_action"] = guidance.copy()
        info["guided_action"] = guided_action.copy()
        return obs, reward, terminated, truncated, info


# ============================================================
# Transformer actor-critic
# ============================================================


class CausalTransformerActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        ff_mult: int = 4,
        dropout: float = 0.1,
        init_log_std: float = -1.0,
        init_propeller_bias: float = 0.75,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.seq_len = seq_len
        self.token_dim = obs_dim + act_dim
        self.d_model = d_model

        self.in_proj = nn.Linear(self.token_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)

        self.actor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, act_dim),
        )
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        if act_dim > 0:
            with torch.no_grad():
                self.actor[-1].bias[0] = float(init_propeller_bias)
        self.critic = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)

    def _forward_features(
        self,
        obs_seq: torch.Tensor,
        prev_action_seq: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([obs_seq, prev_action_seq], dim=-1)
        x = self.in_proj(x) + self.pos_embed[:, : x.shape[1], :]
        x = self.encoder(
            x,
            mask=self._causal_mask(x.shape[1], x.device),
            src_key_padding_mask=~valid_mask.bool(),
        )
        x = self.final_norm(x)

        last_idx = valid_mask.long().sum(dim=1) - 1
        last_idx = torch.clamp(last_idx, min=0)
        features = x[torch.arange(x.size(0), device=x.device), last_idx]
        return features

    def distribution(
        self,
        obs_seq: torch.Tensor,
        prev_action_seq: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[Normal, torch.Tensor]:
        h = self._forward_features(obs_seq, prev_action_seq, valid_mask)
        mean = self.actor(h)
        std = self.log_std.exp().expand_as(mean)
        value = self.critic(h).squeeze(-1)
        return Normal(mean, std), value

    def act(
        self,
        obs_seq: torch.Tensor,
        prev_action_seq: torch.Tensor,
        valid_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.distribution(obs_seq, prev_action_seq, valid_mask)
        raw_action = dist.mean if deterministic else dist.rsample()
        squashed_action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action).sum(-1)
        log_prob -= torch.log(1.0 - squashed_action.pow(2) + 1e-6).sum(-1)
        return squashed_action, log_prob, value

    def evaluate_actions(
        self,
        obs_seq: torch.Tensor,
        prev_action_seq: torch.Tensor,
        valid_mask: torch.Tensor,
        squashed_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.distribution(obs_seq, prev_action_seq, valid_mask)
        clamped = torch.clamp(squashed_action, -0.999999, 0.999999)
        raw_action = 0.5 * (torch.log1p(clamped) - torch.log1p(-clamped))
        log_prob = dist.log_prob(raw_action).sum(-1)
        log_prob -= torch.log(1.0 - clamped.pow(2) + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


# ============================================================
# Sequence context helper
# ============================================================


class SequenceContext:
    def __init__(self, seq_len: int, obs_dim: int, act_dim: int):
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_hist: Deque[np.ndarray] = deque(maxlen=seq_len)
        self.prev_act_hist: Deque[np.ndarray] = deque(maxlen=seq_len)

    def reset(self) -> None:
        self.obs_hist.clear()
        self.prev_act_hist.clear()

    def append(self, obs: np.ndarray, prev_action: np.ndarray) -> None:
        self.obs_hist.append(np.asarray(obs, dtype=np.float32).copy())
        self.prev_act_hist.append(np.asarray(prev_action, dtype=np.float32).copy())

    def tensorize(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_seq = np.zeros((self.seq_len, self.obs_dim), dtype=np.float32)
        prev_act_seq = np.zeros((self.seq_len, self.act_dim), dtype=np.float32)
        valid = np.zeros((self.seq_len,), dtype=np.bool_)

        n = len(self.obs_hist)
        if n > 0:
            obs_seq[-n:] = np.stack(list(self.obs_hist), axis=0)
            prev_act_seq[-n:] = np.stack(list(self.prev_act_hist), axis=0)
            valid[-n:] = True

        obs_tensor = torch.as_tensor(obs_seq, device=device).unsqueeze(0)
        prev_act_tensor = torch.as_tensor(prev_act_seq, device=device).unsqueeze(0)
        valid_tensor = torch.as_tensor(valid, device=device).unsqueeze(0)
        return obs_tensor, prev_act_tensor, valid_tensor


# ============================================================
# PPO rollout buffer
# ============================================================


class RolloutBuffer:
    def __init__(self):
        self.obs_seq: List[np.ndarray] = []
        self.prev_act_seq: List[np.ndarray] = []
        self.valid_mask: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []

    def add(
        self,
        obs_seq: np.ndarray,
        prev_act_seq: np.ndarray,
        valid_mask: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        self.obs_seq.append(obs_seq)
        self.prev_act_seq.append(prev_act_seq)
        self.valid_mask.append(valid_mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(float(done))

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> Dict[str, torch.Tensor]:
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values + [last_value], dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * non_terminal - values[t]
            gae = delta + gamma * gae_lambda * non_terminal * gae
            advantages[t] = gae
        returns = advantages + values[:-1]

        batch = {
            "obs_seq": torch.as_tensor(np.asarray(self.obs_seq), dtype=torch.float32),
            "prev_act_seq": torch.as_tensor(np.asarray(self.prev_act_seq), dtype=torch.float32),
            "valid_mask": torch.as_tensor(np.asarray(self.valid_mask), dtype=torch.bool),
            "actions": torch.as_tensor(np.asarray(self.actions), dtype=torch.float32),
            "old_log_probs": torch.as_tensor(np.asarray(self.log_probs), dtype=torch.float32),
            "advantages": torch.as_tensor(advantages, dtype=torch.float32),
            "returns": torch.as_tensor(returns, dtype=torch.float32),
            "values": torch.as_tensor(np.asarray(self.values), dtype=torch.float32),
        }
        return batch


RC_LOG_FIELDS = []
for _rc_key in REWARD_COMPONENT_KEYS:
    RC_LOG_FIELDS.extend(
        [
            f"rc_window_mean__{_rc_key}",
            f"rc_cumulative_sum__{_rc_key}",
            f"rc_trigger_rate__{_rc_key}",
        ]
    )


EPISODE_DIAGNOSTIC_FIELDS = [
    "global_step",
    "update",
    "mode",
    "eval_kind",
    "episode_index",
    "stage_id",
    "stage_label",
    "n_obstacles",
    "seed",
    "return",
    "episode_length",
    "success",
    "collision",
    "out_of_bounds",
    "timeout",
    "event",
    "mission_distance",
    "target_x",
    "target_y",
    "target_z",
    "target_boundary_margin",
    "target_xy_margin",
    "final_distance",
    "min_distance",
    "final_workspace_margin",
    "min_workspace_margin",
    "mean_workspace_margin",
    "progress_total",
    "path_length",
    "path_ratio",
    "path_efficiency",
    "mean_speed",
    "max_speed",
    "mean_surge_speed",
    "mean_abs_roll",
    "max_abs_roll",
    "mean_abs_pitch",
    "max_abs_pitch",
    "mean_abs_heading_error",
    "max_abs_heading_error",
    "mean_cross_track_error",
    "max_cross_track_error",
    "final_path_progress_fraction",
    "mean_min_obstacle_clearance",
    "min_obstacle_clearance",
    "clearance_breach_steps",
    "mean_action_l2",
    "mean_action_delta_l2",
    "mean_guidance_action_l2",
    "mean_guided_action_l2",
    "mean_propeller_rps",
    "mean_abs_propeller_rps",
    "mean_target_alignment",
    "mean_abs_depth_error",
    "mean_current_speed",
    "max_current_speed",
    "reward_total_check",
] + RC_LOG_FIELDS


TRAIN_METRIC_FIELDS = [
    "global_step",
    "update",
    "stage_id",
    "stage_label",
    "episodes",
    "successes",
    "success_rate",
    "collision_rate",
    "out_of_bounds_rate",
    "timeout_rate",
    "mean_return",
    "mean_len",
    "mean_final_distance",
    "mean_min_distance",
    "mean_path_ratio",
    "mean_path_efficiency",
    "mean_min_obstacle_clearance",
    "min_obstacle_clearance",
    "mean_speed",
    "max_speed_mean",
    "mean_action_l2",
    "mean_action_delta_l2",
    "mean_guided_action_l2",
    "mean_abs_heading_error",
    "max_abs_heading_error_mean",
    "mean_cross_track_error",
    "max_cross_track_error_mean",
    "mean_abs_roll",
    "mean_abs_pitch",
    "mean_target_alignment",
    "mean_current_speed",
    "elapsed_sec",
    "steps_per_sec",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clipfrac",
    "log_std_mean",
    "log_std_min",
    "log_std_max",
    "obs_rms_count",
] + RC_LOG_FIELDS


EVAL_METRIC_FIELDS = [
    "global_step",
    "update",
    "eval_kind",
    "stage_id",
    "stage_label",
    "n_obstacles",
    "eval_episodes",
    "success_rate",
    "collision_rate",
    "out_of_bounds_rate",
    "timeout_rate",
    "eval_return",
    "eval_len",
    "eval_final_distance",
    "eval_min_distance",
    "mean_path_ratio",
    "mean_path_efficiency",
    "mean_min_obstacle_clearance",
    "min_obstacle_clearance",
    "mean_speed",
    "mean_action_l2",
    "mean_guided_action_l2",
    "mean_abs_heading_error",
    "mean_cross_track_error",
    "mean_current_speed",
]


EVAL_DIAGNOSTIC_FIELDS = [
    "global_step",
    "update",
    "eval_kind",
    "stage_id",
    "stage_label",
    "n_obstacles",
    "success_rate",
    "collision_rate",
    "out_of_bounds_rate",
    "timeout_rate",
    "avg_episode_length",
    "avg_path_ratio",
    "avg_path_efficiency",
    "avg_final_distance",
    "avg_min_distance",
    "avg_min_obstacle_clearance",
    "worst_min_obstacle_clearance",
    "clearance_lt_0_rate",
    "clearance_lt_0_5_rate",
    "clearance_lt_1_rate",
    "clearance_lt_2_rate",
    "path_ratio_gt_1_25_rate",
    "path_ratio_gt_1_5_rate",
    "path_efficiency_lt_0_6_rate",
    "heading_abs_gt_45deg_rate",
    "heading_abs_gt_90deg_rate",
    "cross_track_gt_5_rate",
    "cross_track_gt_10_rate",
    "slow_mean_speed_lt_0_25_rate",
    "fast_mean_speed_gt_2_rate",
    "avg_current_speed",
] + RC_LOG_FIELDS


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_float(value: Any, default: float = 0.0) -> float:
    return float(value) if _is_finite_number(value) else default


def _mean(values: List[float], default: float = 0.0) -> float:
    finite = [float(v) for v in values if _is_finite_number(v)]
    return float(np.mean(finite)) if finite else default


def _sum(values: List[float], default: float = 0.0) -> float:
    finite = [float(v) for v in values if _is_finite_number(v)]
    return float(np.sum(finite)) if finite else default


def _min(values: List[float], default: float = 0.0) -> float:
    finite = [float(v) for v in values if _is_finite_number(v)]
    return float(np.min(finite)) if finite else default


def _rate(rows: List[Dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return 100.0 * float(np.mean([1.0 if bool(row.get(key, 0)) else 0.0 for row in rows]))


def _threshold_rate(rows: List[Dict[str, Any]], key: str, threshold: float, op: str) -> float:
    values = [_finite_float(row.get(key), float("nan")) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return 0.0
    if op == "lt":
        hits = [value < threshold for value in values]
    elif op == "gt":
        hits = [value > threshold for value in values]
    else:
        raise ValueError(f"Unsupported threshold op: {op}")
    return 100.0 * float(np.mean(hits))


class CsvLogger:
    def __init__(self, path: Path, fieldnames: List[str], enabled: bool = True):
        self.path = path
        self.fieldnames = fieldnames
        self.enabled = enabled
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _clean_value(value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            value = float(value)
        if isinstance(value, float):
            return value if math.isfinite(value) else ""
        if isinstance(value, (int, bool, str)):
            return value
        if isinstance(value, np.ndarray):
            return json.dumps(value.tolist())
        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value)
        return value

    def write(self, row: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        cleaned = {field: self._clean_value(row.get(field, "")) for field in self.fieldnames}
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(cleaned)


class EpisodeDiagnostics:
    def __init__(self, reset_info: Dict[str, Any], reward_component_keys: List[str]):
        self.reward_component_keys = reward_component_keys
        self.mission_distance = _finite_float(reset_info.get("mission_distance"), 0.0)
        self.target = self._target_array(reset_info.get("target"))
        self.target_boundary_margin = _finite_float(reset_info.get("target_boundary_margin"), 0.0)
        self.target_xy_margin = _finite_float(reset_info.get("target_xy_margin"), float("nan"))
        self.n_obstacles = int(reset_info.get("n_obstacles", 0) or 0)
        self.seed = reset_info.get("seed", "")

        self.ep_return = 0.0
        self.steps = 0
        self.event = "unfinished"
        self.success = False
        self.collision = False
        self.out_of_bounds = False
        self.timeout = False

        self.final_distance = self.mission_distance
        self.min_distance = self.mission_distance if self.mission_distance > 0.0 else float("inf")
        self.final_workspace_margin = float("nan")
        self.min_workspace_margin = float("inf")
        self.progress_total = 0.0
        self.path_length = 0.0
        self.final_path_progress_fraction = 0.0
        self.clearance_breach_steps = 0

        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self.maxes: Dict[str, float] = {}
        self.mins: Dict[str, float] = {}
        self.rc_sums = {key: 0.0 for key in reward_component_keys}
        self.rc_triggers = {key: 0 for key in reward_component_keys}

    @staticmethod
    def _target_array(target: Any) -> np.ndarray:
        try:
            target_arr = np.asarray(target, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            target_arr = np.zeros(0, dtype=np.float64)
        if target_arr.size < 3:
            return np.full(3, np.nan, dtype=np.float64)
        return target_arr[:3].copy()

    def _add_metric(self, name: str, value: Any) -> None:
        if not _is_finite_number(value):
            return
        value_f = float(value)
        self.sums[name] = self.sums.get(name, 0.0) + value_f
        self.counts[name] = self.counts.get(name, 0) + 1
        self.maxes[name] = max(self.maxes.get(name, value_f), value_f)
        self.mins[name] = min(self.mins.get(name, value_f), value_f)

    def _mean_metric(self, name: str, default: float = 0.0) -> float:
        count = self.counts.get(name, 0)
        if count <= 0:
            return default
        return self.sums.get(name, 0.0) / count

    def _max_metric(self, name: str, default: float = 0.0) -> float:
        return self.maxes.get(name, default)

    def _min_metric(self, name: str, default: float = 0.0) -> float:
        return self.mins.get(name, default)

    def record_step(self, action: np.ndarray, reward: float, info: Dict[str, Any]) -> None:
        self.steps += 1
        self.ep_return += float(reward)
        self.event = str(info.get("event") or self.event)
        self.success = self.success or bool(info.get("goal_reached", False))
        self.collision = self.collision or bool(info.get("collision", False))
        self.out_of_bounds = self.out_of_bounds or bool(info.get("out_of_bounds", False))
        self.timeout = self.timeout or (info.get("event") == "timeout")

        distance = _finite_float(info.get("distance_to_goal"), self.final_distance)
        self.final_distance = distance
        self.min_distance = min(self.min_distance, distance)
        self.progress_total += _finite_float(info.get("progress"), 0.0)
        self.path_length += _finite_float(info.get("step_distance"), 0.0)
        workspace_margin = info.get("workspace_margin")
        if _is_finite_number(workspace_margin):
            workspace_margin_f = float(workspace_margin)
            self.final_workspace_margin = workspace_margin_f
            self.min_workspace_margin = min(self.min_workspace_margin, workspace_margin_f)
            self._add_metric("workspace_margin", workspace_margin_f)
        self.final_path_progress_fraction = _finite_float(
            info.get("path_progress_fraction"),
            self.final_path_progress_fraction,
        )

        min_clearance = info.get("min_obstacle_clearance")
        if _is_finite_number(min_clearance) and float(min_clearance) < 0.0:
            self.clearance_breach_steps += 1

        action = np.asarray(action, dtype=np.float32)
        self._add_metric("action_l2", float(np.dot(action, action)))
        guidance_action = info.get("guidance_action")
        if guidance_action is not None:
            guidance_arr = np.asarray(guidance_action, dtype=np.float32)
            self._add_metric("guidance_action_l2", float(np.dot(guidance_arr, guidance_arr)))
        guided_action = info.get("guided_action")
        if guided_action is not None:
            guided_arr = np.asarray(guided_action, dtype=np.float32)
            self._add_metric("guided_action_l2", float(np.dot(guided_arr, guided_arr)))
        self._add_metric("action_delta_l2", info.get("action_delta_l2"))
        self._add_metric("speed", info.get("speed"))
        self._add_metric("surge_speed", info.get("surge_speed"))
        self._add_metric("abs_roll", abs(_finite_float(info.get("roll"), 0.0)))
        self._add_metric("abs_pitch", abs(_finite_float(info.get("pitch"), 0.0)))
        self._add_metric("abs_heading_error", abs(_finite_float(info.get("heading_error"), 0.0)))
        self._add_metric("cross_track_error", info.get("cross_track_error"))
        self._add_metric("min_obstacle_clearance", min_clearance)
        self._add_metric("propeller_rps", info.get("propeller_rps"))
        self._add_metric("abs_propeller_rps", abs(_finite_float(info.get("propeller_rps"), 0.0)))
        self._add_metric("target_alignment", info.get("target_alignment"))
        self._add_metric("abs_depth_error", abs(_finite_float(info.get("target_depth_error"), 0.0)))
        current = info.get("current_inertial")
        if current is not None:
            current_arr = np.asarray(current, dtype=np.float32)
            self._add_metric("current_speed", float(np.linalg.norm(current_arr)))

        reward_components = info.get("reward_components", {})
        for key in self.reward_component_keys:
            value = _finite_float(reward_components.get(key), 0.0)
            self.rc_sums[key] += value
            if abs(value) > 1e-12:
                self.rc_triggers[key] += 1

    def to_row(self) -> Dict[str, Any]:
        path_ratio = self.path_length / max(self.mission_distance, 1e-8)
        straight_progress = max(self.mission_distance - self.final_distance, 0.0)
        path_efficiency = straight_progress / max(self.path_length, 1e-8)
        if self.steps <= 0:
            path_ratio = 0.0
            path_efficiency = 0.0

        row: Dict[str, Any] = {
            "return": self.ep_return,
            "episode_length": self.steps,
            "success": int(self.success),
            "collision": int(self.collision),
            "out_of_bounds": int(self.out_of_bounds),
            "timeout": int(self.timeout),
            "event": self.event,
            "mission_distance": self.mission_distance,
            "target_x": self.target[0],
            "target_y": self.target[1],
            "target_z": self.target[2],
            "target_boundary_margin": self.target_boundary_margin,
            "target_xy_margin": self.target_xy_margin,
            "final_distance": self.final_distance,
            "min_distance": self.min_distance if math.isfinite(self.min_distance) else "",
            "final_workspace_margin": self.final_workspace_margin if math.isfinite(self.final_workspace_margin) else "",
            "min_workspace_margin": self.min_workspace_margin if math.isfinite(self.min_workspace_margin) else "",
            "mean_workspace_margin": self._mean_metric("workspace_margin", float("nan")),
            "progress_total": self.progress_total,
            "path_length": self.path_length,
            "path_ratio": path_ratio,
            "path_efficiency": path_efficiency,
            "mean_speed": self._mean_metric("speed"),
            "max_speed": self._max_metric("speed"),
            "mean_surge_speed": self._mean_metric("surge_speed"),
            "mean_abs_roll": self._mean_metric("abs_roll"),
            "max_abs_roll": self._max_metric("abs_roll"),
            "mean_abs_pitch": self._mean_metric("abs_pitch"),
            "max_abs_pitch": self._max_metric("abs_pitch"),
            "mean_abs_heading_error": self._mean_metric("abs_heading_error"),
            "max_abs_heading_error": self._max_metric("abs_heading_error"),
            "mean_cross_track_error": self._mean_metric("cross_track_error"),
            "max_cross_track_error": self._max_metric("cross_track_error"),
            "final_path_progress_fraction": self.final_path_progress_fraction,
            "mean_min_obstacle_clearance": self._mean_metric("min_obstacle_clearance", float("nan")),
            "min_obstacle_clearance": self._min_metric("min_obstacle_clearance", float("nan")),
            "clearance_breach_steps": self.clearance_breach_steps,
            "mean_action_l2": self._mean_metric("action_l2"),
            "mean_action_delta_l2": self._mean_metric("action_delta_l2"),
            "mean_guidance_action_l2": self._mean_metric("guidance_action_l2"),
            "mean_guided_action_l2": self._mean_metric("guided_action_l2"),
            "mean_propeller_rps": self._mean_metric("propeller_rps"),
            "mean_abs_propeller_rps": self._mean_metric("abs_propeller_rps"),
            "mean_target_alignment": self._mean_metric("target_alignment"),
            "mean_abs_depth_error": self._mean_metric("abs_depth_error"),
            "mean_current_speed": self._mean_metric("current_speed"),
            "max_current_speed": self._max_metric("current_speed"),
            "reward_total_check": sum(self.rc_sums.values()),
            "n_obstacles": self.n_obstacles,
            "seed": self.seed,
        }

        for key in self.reward_component_keys:
            row[f"rc_window_mean__{key}"] = self.rc_sums[key] / max(self.steps, 1)
            row[f"rc_cumulative_sum__{key}"] = self.rc_sums[key]
            row[f"rc_trigger_rate__{key}"] = self.rc_triggers[key] / max(self.steps, 1)
        return row


def summarize_episode_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {
            "episodes": 0,
            "successes": 0,
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "out_of_bounds_rate": 0.0,
            "timeout_rate": 0.0,
            "mean_return": 0.0,
            "mean_len": 0.0,
            "mean_final_distance": 0.0,
            "mean_min_distance": 0.0,
            "mean_path_ratio": 0.0,
            "mean_path_efficiency": 0.0,
            "mean_min_obstacle_clearance": 0.0,
            "min_obstacle_clearance": 0.0,
            "mean_speed": 0.0,
            "max_speed_mean": 0.0,
            "mean_action_l2": 0.0,
            "mean_action_delta_l2": 0.0,
            "mean_guided_action_l2": 0.0,
            "mean_abs_heading_error": 0.0,
            "max_abs_heading_error_mean": 0.0,
            "mean_cross_track_error": 0.0,
            "max_cross_track_error_mean": 0.0,
            "mean_abs_roll": 0.0,
            "mean_abs_pitch": 0.0,
            "mean_target_alignment": 0.0,
            "mean_current_speed": 0.0,
        }

    return {
        "episodes": len(rows),
        "successes": int(sum(int(row.get("success", 0)) for row in rows)),
        "success_rate": _rate(rows, "success"),
        "collision_rate": _rate(rows, "collision"),
        "out_of_bounds_rate": _rate(rows, "out_of_bounds"),
        "timeout_rate": _rate(rows, "timeout"),
        "mean_return": _mean([row.get("return", 0.0) for row in rows]),
        "mean_len": _mean([row.get("episode_length", 0.0) for row in rows]),
        "mean_final_distance": _mean([row.get("final_distance", 0.0) for row in rows]),
        "mean_min_distance": _mean([row.get("min_distance", 0.0) for row in rows]),
        "mean_path_ratio": _mean([row.get("path_ratio", 0.0) for row in rows]),
        "mean_path_efficiency": _mean([row.get("path_efficiency", 0.0) for row in rows]),
        "mean_min_obstacle_clearance": _mean([row.get("mean_min_obstacle_clearance", 0.0) for row in rows]),
        "min_obstacle_clearance": _min([row.get("min_obstacle_clearance", 0.0) for row in rows]),
        "mean_speed": _mean([row.get("mean_speed", 0.0) for row in rows]),
        "max_speed_mean": _mean([row.get("max_speed", 0.0) for row in rows]),
        "mean_action_l2": _mean([row.get("mean_action_l2", 0.0) for row in rows]),
        "mean_action_delta_l2": _mean([row.get("mean_action_delta_l2", 0.0) for row in rows]),
        "mean_guided_action_l2": _mean([row.get("mean_guided_action_l2", 0.0) for row in rows]),
        "mean_abs_heading_error": _mean([row.get("mean_abs_heading_error", 0.0) for row in rows]),
        "max_abs_heading_error_mean": _mean([row.get("max_abs_heading_error", 0.0) for row in rows]),
        "mean_cross_track_error": _mean([row.get("mean_cross_track_error", 0.0) for row in rows]),
        "max_cross_track_error_mean": _mean([row.get("max_cross_track_error", 0.0) for row in rows]),
        "mean_abs_roll": _mean([row.get("mean_abs_roll", 0.0) for row in rows]),
        "mean_abs_pitch": _mean([row.get("mean_abs_pitch", 0.0) for row in rows]),
        "mean_target_alignment": _mean([row.get("mean_target_alignment", 0.0) for row in rows]),
        "mean_current_speed": _mean([row.get("mean_current_speed", 0.0) for row in rows]),
    }


# ============================================================
# Trainer
# ============================================================


@dataclass
class PPOConfig:
    total_steps: int = 300_000
    rollout_steps: int = 4096
    seq_len: int = 32
    gamma: float = 0.995
    gae_lambda: float = 0.97
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    max_grad_norm: float = 0.5
    learning_rate: float = 1e-4
    update_epochs: int = 10
    minibatch_size: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1
    init_log_std: float = -1.0
    init_propeller_bias: float = 0.75
    seed: int = 42
    eval_every_updates: int = 5
    eval_episodes: int = 5
    curriculum_gate_consecutive_evals: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./checkpoints_transformer_remus"
    run_dir: str = ""
    log_csv: bool = True
    run_version: str = RUN_VERSION
    clip_obs: float = 10.0
    use_guided_action_prior: bool = True
    guidance_cruise_propeller: float = 0.65
    guidance_near_propeller: float = 0.25
    guidance_heading_gain: float = 1.8
    guidance_elevation_gain: float = 1.8
    guidance_propeller_residual_scale: float = 0.20
    guidance_surface_residual_scale: float = 0.45
    guidance_slow_radius: float = 1.8
    guidance_full_speed_radius: float = 8.0
    guidance_obstacle_avoid_distance: float = 20.0
    guidance_obstacle_avoid_margin: float = 5.0
    guidance_obstacle_avoid_gain: float = 0.70
    guidance_obstacle_slowdown: float = 0.25
    resume_from: str = ""
    start_stage: int = 0
    plot_stage4: bool = True
    show_stage4_plots: bool = True
    plot_stage4_every_evals: int = 1
    plot_stage4_episode_index: int = -1
    eval_progress_every: int = 100
    max_first_failure_plot_scan_episodes: int = 200


class PPOTrainer:
    def __init__(
        self,
        policy: CausalTransformerActorCritic,
        cfg: PPOConfig,
        reward_cfg: RewardConfig,
        obs_rms: RunningMeanStd,
        curriculum: List[CurriculumStage],
    ):
        self.policy = policy
        self.cfg = cfg
        self.reward_cfg = reward_cfg
        self.obs_rms = obs_rms
        self.curriculum = sorted(curriculum, key=lambda x: x.stage_id)
        self.device = torch.device(cfg.device)
        self.policy.to(self.device)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=cfg.learning_rate)
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if cfg.run_dir:
            self.run_dir = Path(cfg.run_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = self.save_dir / "runs" / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_logger = CsvLogger(self.run_dir / "metrics.csv", TRAIN_METRIC_FIELDS, enabled=cfg.log_csv)
        self.eval_metrics_logger = CsvLogger(self.run_dir / "eval_metrics.csv", EVAL_METRIC_FIELDS, enabled=cfg.log_csv)
        self.eval_diagnostics_logger = CsvLogger(
            self.run_dir / "eval_diagnostics.csv",
            EVAL_DIAGNOSTIC_FIELDS,
            enabled=cfg.log_csv,
        )
        self.train_episode_logger = CsvLogger(
            self.run_dir / "train_episode_diagnostics.csv",
            EPISODE_DIAGNOSTIC_FIELDS,
            enabled=cfg.log_csv,
        )
        self.eval_episode_logger = CsvLogger(
            self.run_dir / "eval_episode_diagnostics.csv",
            EPISODE_DIAGNOSTIC_FIELDS,
            enabled=cfg.log_csv,
        )
        self.reward_component_cumulative = {key: 0.0 for key in REWARD_COMPONENT_KEYS}
        self.stage4_eval_count = 0
        self.current_stage = self.curriculum[0]
        self.stage_gate_hits = 0
        self.env = self._build_env(self.current_stage, seed_offset=0, update_obs_stats=True)
        self.eval_env = self._build_env(self.current_stage, seed_offset=10_000, update_obs_stats=False)
        self.obs_dim = int(np.prod(self.env.observation_space.shape))
        self.act_dim = int(np.prod(self.env.action_space.shape))
        if self.cfg.resume_from:
            self._load_checkpoint(Path(self.cfg.resume_from))
        if self.cfg.start_stage > 0:
            stage = self._stage_by_id(self.cfg.start_stage)
            if stage is None:
                raise ValueError(f"Unknown --start-stage {self.cfg.start_stage}")
            self._switch_to_stage(stage)

    def _build_env(self, stage: CurriculumStage, seed_offset: int, update_obs_stats: bool) -> gym.Env:
        REMUSAUVEnv = load_env_class()
        env = REMUSAUVEnv(
            seed=self.cfg.seed + seed_offset,
            current_enabled=stage.current_enabled,
            include_current_in_obs=False,
            n_obstacles=stage.n_obstacles,
        )
        if hasattr(env, "target_z_min"):
            env.target_z_min = stage.target_z_min
        if hasattr(env, "target_z_max"):
            env.target_z_max = stage.target_z_max
        if stage.target_distance_min is not None and hasattr(env, "min_start_target_distance"):
            env.min_start_target_distance = stage.target_distance_min
        if hasattr(env, "max_start_target_distance"):
            env.max_start_target_distance = stage.target_distance_max
        if stage.target_boundary_margin is not None and hasattr(env, "target_boundary_margin"):
            env.target_boundary_margin = stage.target_boundary_margin
        if stage.max_steps is not None and hasattr(env, "max_steps"):
            env.max_steps = stage.max_steps
        if stage.current_speed_min is not None and hasattr(env, "current_speed_min"):
            env.current_speed_min = stage.current_speed_min
        if stage.current_speed_max is not None and hasattr(env, "current_speed_max"):
            env.current_speed_max = stage.current_speed_max
        if stage.current_vertical_max is not None and hasattr(env, "current_vertical_max"):
            env.current_vertical_max = stage.current_vertical_max
        if stage.current_osc_amp_xy is not None and hasattr(env, "current_osc_amp_max"):
            env.current_osc_amp_max[0] = stage.current_osc_amp_xy
            env.current_osc_amp_max[1] = stage.current_osc_amp_xy
        if stage.current_osc_amp_z is not None and hasattr(env, "current_osc_amp_max"):
            env.current_osc_amp_max[2] = stage.current_osc_amp_z
        if stage.obstacle_radius_min is not None and hasattr(env, "obstacle_radius_min"):
            env.obstacle_radius_min = stage.obstacle_radius_min
        if stage.obstacle_radius_max is not None and hasattr(env, "obstacle_radius_max"):
            env.obstacle_radius_max = stage.obstacle_radius_max
        if stage.obstacle_speed_min is not None and hasattr(env, "obstacle_speed_min"):
            env.obstacle_speed_min = stage.obstacle_speed_min
        if stage.obstacle_speed_max is not None and hasattr(env, "obstacle_speed_max"):
            env.obstacle_speed_max = stage.obstacle_speed_max
        if hasattr(env, "prefer_path_obstacles"):
            env.prefer_path_obstacles = stage.prefer_path_obstacles
        if stage.obstacle_path_lateral_jitter is not None and hasattr(env, "obstacle_path_lateral_jitter"):
            env.obstacle_path_lateral_jitter = stage.obstacle_path_lateral_jitter
        if stage.obstacle_path_depth_jitter is not None and hasattr(env, "obstacle_path_depth_jitter"):
            env.obstacle_path_depth_jitter = stage.obstacle_path_depth_jitter
        if self.cfg.use_guided_action_prior:
            env = PursuitGuidanceActionWrapper(
                env,
                cruise_propeller=stage.guidance_cruise_propeller
                if stage.guidance_cruise_propeller is not None
                else self.cfg.guidance_cruise_propeller,
                near_propeller=stage.guidance_near_propeller
                if stage.guidance_near_propeller is not None
                else self.cfg.guidance_near_propeller,
                heading_gain=self.cfg.guidance_heading_gain,
                elevation_gain=self.cfg.guidance_elevation_gain,
                propeller_residual_scale=stage.guidance_propeller_residual_scale
                if stage.guidance_propeller_residual_scale is not None
                else self.cfg.guidance_propeller_residual_scale,
                surface_residual_scale=stage.guidance_surface_residual_scale
                if stage.guidance_surface_residual_scale is not None
                else self.cfg.guidance_surface_residual_scale,
                slow_radius=stage.guidance_slow_radius
                if stage.guidance_slow_radius is not None
                else self.cfg.guidance_slow_radius,
                full_speed_radius=stage.guidance_full_speed_radius
                if stage.guidance_full_speed_radius is not None
                else self.cfg.guidance_full_speed_radius,
                obstacle_avoidance=stage.guidance_obstacle_avoidance,
                obstacle_avoid_distance=stage.guidance_obstacle_avoid_distance
                if stage.guidance_obstacle_avoid_distance is not None
                else self.cfg.guidance_obstacle_avoid_distance,
                obstacle_avoid_margin=stage.guidance_obstacle_avoid_margin
                if stage.guidance_obstacle_avoid_margin is not None
                else self.cfg.guidance_obstacle_avoid_margin,
                obstacle_avoid_gain=stage.guidance_obstacle_avoid_gain
                if stage.guidance_obstacle_avoid_gain is not None
                else self.cfg.guidance_obstacle_avoid_gain,
                obstacle_slowdown=stage.guidance_obstacle_slowdown
                if stage.guidance_obstacle_slowdown is not None
                else self.cfg.guidance_obstacle_slowdown,
            )
        env = RemusRewardWrapper(env, self.reward_cfg)
        env = ObsNormWrapper(env, self.obs_rms, update_stats=update_obs_stats, clip_obs=self.cfg.clip_obs)
        return env

    def _stage_by_id(self, stage_id: int) -> Optional[CurriculumStage]:
        return next((stage for stage in self.curriculum if stage.stage_id == stage_id), None)

    def _load_checkpoint(self, path: Path) -> None:
        path = path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["model_state_dict"])
        if "obs_rms" in ckpt:
            self.obs_rms.load_state_dict(ckpt["obs_rms"])

        ckpt_stage = ckpt.get("curriculum_stage", {})
        stage_id = int(ckpt_stage.get("stage_id", self.current_stage.stage_id))
        stage = self._stage_by_id(stage_id)
        if stage is not None:
            self.current_stage = stage
            self.stage_gate_hits = 0
            self.env = self._build_env(self.current_stage, seed_offset=self.current_stage.stage_id * 1000, update_obs_stats=True)
            self.eval_env = self._build_env(
                self.current_stage,
                seed_offset=10_000 + self.current_stage.stage_id * 1000,
                update_obs_stats=False,
            )
        print(f"[Checkpoint] Loaded {path} at stage {self.current_stage.stage_id}:{self.current_stage.label}")

    def _switch_to_stage(self, stage: CurriculumStage) -> None:
        self.current_stage = stage
        self.stage_gate_hits = 0
        self.env = self._build_env(self.current_stage, seed_offset=self.current_stage.stage_id * 1000, update_obs_stats=True)
        self.eval_env = self._build_env(self.current_stage, seed_offset=10_000 + self.current_stage.stage_id * 1000, update_obs_stats=False)
        print(f"[Curriculum] Switched to stage {self.current_stage.stage_id}: {self.current_stage.label}")

    def maybe_advance_curriculum(self, eval_success_rate: float) -> None:
        current_idx = next((i for i, stage in enumerate(self.curriculum) if stage.stage_id == self.current_stage.stage_id), None)
        if current_idx is None or current_idx >= len(self.curriculum) - 1:
            return

        gate = self.current_stage.success_gate
        if gate is None:
            return

        if eval_success_rate >= gate:
            self.stage_gate_hits += 1
            print(
                f"[Curriculum] Gate hit {self.stage_gate_hits}/{self.cfg.curriculum_gate_consecutive_evals} "
                f"for stage {self.current_stage.stage_id} (success_rate={eval_success_rate:.2%} >= {gate:.2%})"
            )
        else:
            if self.stage_gate_hits > 0:
                print(
                    f"[Curriculum] Gate reset for stage {self.current_stage.stage_id} "
                    f"(success_rate={eval_success_rate:.2%} < {gate:.2%})"
                )
            self.stage_gate_hits = 0
            return

        if self.stage_gate_hits >= self.cfg.curriculum_gate_consecutive_evals:
            self._switch_to_stage(self.curriculum[current_idx + 1])

    def collect_rollout(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        buffer = RolloutBuffer()
        stats: Dict[str, Any] = {"episodes": 0, "successes": 0, "mean_return": 0.0, "mean_len": 0.0}

        obs, info = self.env.reset()
        episode_diag = EpisodeDiagnostics(info, REWARD_COMPONENT_KEYS)
        ep_return = 0.0
        ep_len = 0
        ep_returns: List[float] = []
        ep_lens: List[int] = []
        episode_rows: List[Dict[str, Any]] = []
        successes = 0

        context = SequenceContext(self.cfg.seq_len, self.obs_dim, self.act_dim)
        prev_action = np.zeros(self.act_dim, dtype=np.float32)
        context.append(obs, prev_action)

        for _ in range(self.cfg.rollout_steps):
            obs_seq_t, prev_act_seq_t, valid_mask_t = context.tensorize(self.device)
            with torch.no_grad():
                action_t, log_prob_t, value_t = self.policy.act(obs_seq_t, prev_act_seq_t, valid_mask_t, deterministic=False)

            action = action_t.squeeze(0).cpu().numpy().astype(np.float32)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            buffer.add(
                obs_seq=obs_seq_t.squeeze(0).cpu().numpy(),
                prev_act_seq=prev_act_seq_t.squeeze(0).cpu().numpy(),
                valid_mask=valid_mask_t.squeeze(0).cpu().numpy(),
                action=action,
                log_prob=float(log_prob_t.item()),
                value=float(value_t.item()),
                reward=float(reward),
                done=done,
            )

            ep_return += float(reward)
            ep_len += 1
            episode_diag.record_step(action, float(reward), info)

            context.append(next_obs, action)
            obs = next_obs

            if done:
                episode_rows.append(episode_diag.to_row())
                ep_returns.append(ep_return)
                ep_lens.append(ep_len)
                stats["episodes"] += 1
                if info.get("goal_reached", False):
                    successes += 1

                obs, info = self.env.reset()
                episode_diag = EpisodeDiagnostics(info, REWARD_COMPONENT_KEYS)
                context.reset()
                prev_action = np.zeros(self.act_dim, dtype=np.float32)
                context.append(obs, prev_action)
                ep_return = 0.0
                ep_len = 0

        obs_seq_t, prev_act_seq_t, valid_mask_t = context.tensorize(self.device)
        with torch.no_grad():
            _, last_value = self.policy.distribution(obs_seq_t, prev_act_seq_t, valid_mask_t)
        last_value_scalar = 0.0 if ep_len == 0 else float(last_value.item())

        if ep_returns:
            stats["mean_return"] = float(np.mean(ep_returns))
            stats["mean_len"] = float(np.mean(ep_lens))
            stats["successes"] = int(successes)
        else:
            stats["mean_return"] = ep_return
            stats["mean_len"] = ep_len
            stats["successes"] = int(successes)
        stats["episode_rows"] = episode_rows

        batch = buffer.compute_returns_and_advantages(last_value_scalar, self.cfg.gamma, self.cfg.gae_lambda)
        return batch, stats

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs_seq = batch["obs_seq"].to(self.device)
        prev_act_seq = batch["prev_act_seq"].to(self.device)
        valid_mask = batch["valid_mask"].to(self.device)
        actions = batch["actions"].to(self.device)
        old_log_probs = batch["old_log_probs"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        returns = batch["returns"].to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        n = obs_seq.shape[0]

        losses = {"policy": [], "value": [], "entropy": [], "approx_kl": [], "clipfrac": []}

        for _ in range(self.cfg.update_epochs):
            idx = torch.randperm(n, device=self.device)
            for start in range(0, n, self.cfg.minibatch_size):
                mb_idx = idx[start : start + self.cfg.minibatch_size]

                new_log_probs, entropy, values = self.policy.evaluate_actions(
                    obs_seq[mb_idx], prev_act_seq[mb_idx], valid_mask[mb_idx], actions[mb_idx]
                )

                ratio = (new_log_probs - old_log_probs[mb_idx]).exp()
                unclipped = ratio * advantages[mb_idx]
                clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * advantages[mb_idx]
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = F.mse_loss(values, returns[mb_idx])
                entropy_loss = entropy.mean()

                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy_loss

                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optim.step()

                with torch.no_grad():
                    approx_kl = (old_log_probs[mb_idx] - new_log_probs).mean().item()
                    clipfrac = (torch.abs(ratio - 1.0) > self.cfg.clip_eps).float().mean().item()
                losses["policy"].append(policy_loss.item())
                losses["value"].append(value_loss.item())
                losses["entropy"].append(entropy_loss.item())
                losses["approx_kl"].append(approx_kl)
                losses["clipfrac"].append(clipfrac)

        return {k: float(np.mean(v)) if v else 0.0 for k, v in losses.items()}

    def _decorate_episode_rows(
        self,
        rows: List[Dict[str, Any]],
        *,
        global_step: int,
        update: int,
        mode: str,
        eval_kind: str = "",
    ) -> List[Dict[str, Any]]:
        decorated_rows = []
        for episode_index, row in enumerate(rows):
            decorated = dict(row)
            decorated.update(
                {
                    "global_step": global_step,
                    "update": update,
                    "mode": mode,
                    "eval_kind": eval_kind,
                    "episode_index": episode_index,
                    "stage_id": self.current_stage.stage_id,
                    "stage_label": self.current_stage.label,
                    "n_obstacles": self.current_stage.n_obstacles,
                }
            )
            decorated_rows.append(decorated)
        return decorated_rows

    def _reward_component_rollup(
        self,
        rows: List[Dict[str, Any]],
        *,
        update_cumulative: bool,
    ) -> Dict[str, float]:
        total_steps = int(sum(int(row.get("episode_length", 0) or 0) for row in rows))
        rollup: Dict[str, float] = {}
        for key in REWARD_COMPONENT_KEYS:
            window_sum = _sum([row.get(f"rc_cumulative_sum__{key}", 0.0) for row in rows])
            trigger_steps = 0.0
            for row in rows:
                steps = _finite_float(row.get("episode_length"), 0.0)
                trigger_steps += _finite_float(row.get(f"rc_trigger_rate__{key}"), 0.0) * steps

            if update_cumulative:
                self.reward_component_cumulative[key] += window_sum
                cumulative_sum = self.reward_component_cumulative[key]
            else:
                cumulative_sum = window_sum

            rollup[f"rc_window_mean__{key}"] = window_sum / max(total_steps, 1)
            rollup[f"rc_cumulative_sum__{key}"] = cumulative_sum
            rollup[f"rc_trigger_rate__{key}"] = trigger_steps / max(total_steps, 1)
        return rollup

    def _write_train_logs(
        self,
        *,
        update: int,
        global_step: int,
        rollout_stats: Dict[str, Any],
        loss_stats: Dict[str, float],
        elapsed_sec: float,
    ) -> None:
        raw_rows = rollout_stats.get("episode_rows", [])
        episode_rows = self._decorate_episode_rows(
            raw_rows,
            global_step=global_step,
            update=update,
            mode="train",
        )
        for row in episode_rows:
            self.train_episode_logger.write(row)

        summary = summarize_episode_rows(episode_rows)
        if not episode_rows:
            summary["mean_return"] = _finite_float(rollout_stats.get("mean_return"), 0.0)
            summary["mean_len"] = _finite_float(rollout_stats.get("mean_len"), 0.0)
            summary["episodes"] = int(rollout_stats.get("episodes", 0) or 0)
            summary["successes"] = int(rollout_stats.get("successes", 0) or 0)

        log_std = self.policy.log_std.detach().cpu().numpy()
        row: Dict[str, Any] = {
            "global_step": global_step,
            "update": update,
            "stage_id": self.current_stage.stage_id,
            "stage_label": self.current_stage.label,
            "elapsed_sec": elapsed_sec,
            "steps_per_sec": global_step / max(elapsed_sec, 1e-8),
            "policy_loss": loss_stats.get("policy", 0.0),
            "value_loss": loss_stats.get("value", 0.0),
            "entropy": loss_stats.get("entropy", 0.0),
            "approx_kl": loss_stats.get("approx_kl", 0.0),
            "clipfrac": loss_stats.get("clipfrac", 0.0),
            "log_std_mean": float(np.mean(log_std)),
            "log_std_min": float(np.min(log_std)),
            "log_std_max": float(np.max(log_std)),
            "obs_rms_count": float(self.obs_rms.count),
            **summary,
            **self._reward_component_rollup(episode_rows, update_cumulative=True),
        }
        self.metrics_logger.write(row)

    def _write_eval_logs(
        self,
        *,
        update: int,
        global_step: int,
        eval_kind: str,
        eval_stats: Dict[str, float],
        raw_episode_rows: List[Dict[str, Any]],
    ) -> None:
        episode_rows = self._decorate_episode_rows(
            raw_episode_rows,
            global_step=global_step,
            update=update,
            mode="eval",
            eval_kind=eval_kind,
        )
        for row in episode_rows:
            self.eval_episode_logger.write(row)

        summary = summarize_episode_rows(episode_rows)
        eval_metric_row = {
            "global_step": global_step,
            "update": update,
            "eval_kind": eval_kind,
            "stage_id": self.current_stage.stage_id,
            "stage_label": self.current_stage.label,
            "n_obstacles": self.current_stage.n_obstacles,
            "eval_episodes": len(episode_rows),
            "success_rate": summary["success_rate"],
            "collision_rate": summary["collision_rate"],
            "out_of_bounds_rate": summary["out_of_bounds_rate"],
            "timeout_rate": summary["timeout_rate"],
            "eval_return": eval_stats.get("eval_return", 0.0),
            "eval_len": eval_stats.get("eval_len", 0.0),
            "eval_final_distance": eval_stats.get("eval_final_distance", 0.0),
            "eval_min_distance": summary["mean_min_distance"],
            "mean_path_ratio": summary["mean_path_ratio"],
            "mean_path_efficiency": summary["mean_path_efficiency"],
            "mean_min_obstacle_clearance": summary["mean_min_obstacle_clearance"],
            "min_obstacle_clearance": summary["min_obstacle_clearance"],
            "mean_speed": summary["mean_speed"],
            "mean_action_l2": summary["mean_action_l2"],
            "mean_guided_action_l2": summary["mean_guided_action_l2"],
            "mean_abs_heading_error": summary["mean_abs_heading_error"],
            "mean_cross_track_error": summary["mean_cross_track_error"],
            "mean_current_speed": summary["mean_current_speed"],
        }
        self.eval_metrics_logger.write(eval_metric_row)

        diag_row = {
            "global_step": global_step,
            "update": update,
            "eval_kind": eval_kind,
            "stage_id": self.current_stage.stage_id,
            "stage_label": self.current_stage.label,
            "n_obstacles": self.current_stage.n_obstacles,
            "success_rate": summary["success_rate"],
            "collision_rate": summary["collision_rate"],
            "out_of_bounds_rate": summary["out_of_bounds_rate"],
            "timeout_rate": summary["timeout_rate"],
            "avg_episode_length": summary["mean_len"],
            "avg_path_ratio": summary["mean_path_ratio"],
            "avg_path_efficiency": summary["mean_path_efficiency"],
            "avg_final_distance": summary["mean_final_distance"],
            "avg_min_distance": summary["mean_min_distance"],
            "avg_min_obstacle_clearance": summary["mean_min_obstacle_clearance"],
            "worst_min_obstacle_clearance": summary["min_obstacle_clearance"],
            "clearance_lt_0_rate": _threshold_rate(episode_rows, "min_obstacle_clearance", 0.0, "lt"),
            "clearance_lt_0_5_rate": _threshold_rate(episode_rows, "min_obstacle_clearance", 0.5, "lt"),
            "clearance_lt_1_rate": _threshold_rate(episode_rows, "min_obstacle_clearance", 1.0, "lt"),
            "clearance_lt_2_rate": _threshold_rate(episode_rows, "min_obstacle_clearance", 2.0, "lt"),
            "path_ratio_gt_1_25_rate": _threshold_rate(episode_rows, "path_ratio", 1.25, "gt"),
            "path_ratio_gt_1_5_rate": _threshold_rate(episode_rows, "path_ratio", 1.5, "gt"),
            "path_efficiency_lt_0_6_rate": _threshold_rate(episode_rows, "path_efficiency", 0.6, "lt"),
            "heading_abs_gt_45deg_rate": _threshold_rate(episode_rows, "mean_abs_heading_error", math.radians(45.0), "gt"),
            "heading_abs_gt_90deg_rate": _threshold_rate(episode_rows, "mean_abs_heading_error", math.radians(90.0), "gt"),
            "cross_track_gt_5_rate": _threshold_rate(episode_rows, "mean_cross_track_error", 5.0, "gt"),
            "cross_track_gt_10_rate": _threshold_rate(episode_rows, "mean_cross_track_error", 10.0, "gt"),
            "slow_mean_speed_lt_0_25_rate": _threshold_rate(episode_rows, "mean_speed", 0.25, "lt"),
            "fast_mean_speed_gt_2_rate": _threshold_rate(episode_rows, "mean_speed", 2.0, "gt"),
            "avg_current_speed": summary["mean_current_speed"],
            **self._reward_component_rollup(episode_rows, update_cumulative=False),
        }
        self.eval_diagnostics_logger.write(diag_row)

    def _plot_stage4_trajectory(
        self,
        trajectory: Dict[str, Any],
        *,
        update: int,
        global_step: int,
        eval_stats: Dict[str, float],
    ) -> Optional[Path]:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
        except Exception as exc:
            print(f"  Stage4 plot skipped: matplotlib unavailable ({exc})")
            return None

        positions = np.asarray(trajectory.get("positions", []), dtype=np.float64)
        if positions.ndim != 2 or positions.shape[0] < 2:
            return None

        start = np.asarray(trajectory.get("start", positions[0]), dtype=np.float64)
        target = np.asarray(trajectory.get("target", positions[-1]), dtype=np.float64)
        obstacle_radii = np.asarray(trajectory.get("obstacle_radii", []), dtype=np.float64)
        obstacle_initial = np.asarray(trajectory.get("obstacle_initial", []), dtype=np.float64)
        obstacle_final = np.asarray(trajectory.get("obstacle_final", []), dtype=np.float64)
        currents = np.asarray(trajectory.get("currents", []), dtype=np.float64)

        plot_dir = self.run_dir / "stage4_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        path = plot_dir / f"stage4_update_{update:04d}_step_{global_step}_ep{trajectory.get('episode_index', 0)}.png"

        fig = plt.figure(figsize=(13.0, 6.2), dpi=140)
        ax3d = fig.add_subplot(1, 2, 1, projection="3d")
        ax2d = fig.add_subplot(1, 2, 2)

        ax3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], color="#1f77b4", linewidth=2.0, label="AUV")
        ax3d.scatter([start[0]], [start[1]], [start[2]], color="#2ca02c", s=45, label="Start")
        ax3d.scatter([target[0]], [target[1]], [target[2]], color="#d62728", s=55, marker="*", label="Target")
        if obstacle_final.size > 0:
            ax3d.scatter(
                obstacle_final[:, 0],
                obstacle_final[:, 1],
                obstacle_final[:, 2],
                s=np.maximum(obstacle_radii, 0.5) * 28.0,
                color="#ff7f0e",
                alpha=0.55,
                label="Moving obstacles",
            )
        ax3d.set_xlabel("x [m]")
        ax3d.set_ylabel("y [m]")
        ax3d.set_zlabel("z depth [m]")
        ax3d.invert_zaxis()
        ax3d.set_title("Stage 4 trajectory (3D)")
        ax3d.legend(loc="upper left")

        ax2d.plot(positions[:, 0], positions[:, 1], color="#1f77b4", linewidth=2.0)
        ax2d.scatter([start[0]], [start[1]], color="#2ca02c", s=45, label="Start")
        ax2d.scatter([target[0]], [target[1]], color="#d62728", s=65, marker="*", label="Target")
        if obstacle_initial.size > 0:
            ax2d.scatter(obstacle_initial[:, 0], obstacle_initial[:, 1], color="#ffbb78", s=18, alpha=0.55, label="Obstacle start")
        if obstacle_final.size > 0:
            ax2d.scatter(obstacle_final[:, 0], obstacle_final[:, 1], color="#ff7f0e", s=25, alpha=0.85, label="Obstacle final")
            for center, radius in zip(obstacle_final, obstacle_radii):
                ax2d.add_patch(Circle((center[0], center[1]), radius + 0.25, fill=False, color="#ff7f0e", alpha=0.30))
        if currents.ndim == 2 and currents.shape[0] > 0:
            mean_current = np.mean(currents[:, :2], axis=0)
            anchor = positions[min(len(positions) // 3, len(positions) - 1), :2]
            ax2d.arrow(
                anchor[0],
                anchor[1],
                mean_current[0] * 8.0,
                mean_current[1] * 8.0,
                width=0.18,
                color="#9467bd",
                alpha=0.8,
                length_includes_head=True,
            )
            ax2d.text(anchor[0], anchor[1], " mean current", color="#9467bd", fontsize=8)
        ax2d.set_aspect("equal", adjustable="box")
        ax2d.grid(True, alpha=0.25)
        ax2d.set_xlabel("x [m]")
        ax2d.set_ylabel("y [m]")
        ax2d.set_title("Top-down path and moving obstacles")
        ax2d.legend(loc="best", fontsize=8)

        fig.suptitle(
            f"Stage 4 | update {update} | success={eval_stats.get('eval_success_rate', 0.0):.0%} | "
            f"return={eval_stats.get('eval_return', 0.0):.1f} | event={trajectory.get('event', '')}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

        print(f"  Stage4 trajectory plot: {path}")
        if self.cfg.show_stage4_plots:
            try:
                if os.name == "nt":
                    os.startfile(str(path))
            except Exception as exc:
                print(f"  Stage4 plot saved but could not be opened automatically: {exc}")
        return path

    @torch.no_grad()
    def evaluate(
        self,
        n_episodes: int = 5,
        capture_trajectory_episode: Optional[int] = None,
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        returns: List[float] = []
        lengths: List[int] = []
        final_distances: List[float] = []
        episode_rows: List[Dict[str, Any]] = []
        captured_trajectory: Optional[Dict[str, Any]] = None
        fallback_trajectory: Optional[Dict[str, Any]] = None
        eval_started_at = time.perf_counter()
        if (
            capture_trajectory_episode is not None
            and capture_trajectory_episode < 0
            and n_episodes > self.cfg.max_first_failure_plot_scan_episodes
        ):
            print(
                "  Large eval: first-failure trajectory scan disabled "
                f"for {n_episodes} episodes; plotting episode 0 instead. "
                "Use --no-stage4-plots for fastest bulk evaluation."
            )
            capture_trajectory_episode = 0
        capture_first_failure = capture_trajectory_episode is not None and capture_trajectory_episode < 0
        successes = 0

        for ep_idx in range(n_episodes):
            eval_seed = self.cfg.seed + 50_000 + ep_idx
            obs, info = self.eval_env.reset(seed=eval_seed)
            capture_this_episode = capture_trajectory_episode is not None and (
                (capture_first_failure and captured_trajectory is None) or ep_idx == capture_trajectory_episode
            )
            episode_trajectory: Optional[Dict[str, Any]] = None
            if capture_this_episode:
                episode_trajectory = {
                    "episode_index": ep_idx,
                    "seed": eval_seed,
                    "start": np.asarray(info.get("start", np.zeros(3)), dtype=np.float64).copy(),
                    "target": np.asarray(info.get("target", np.zeros(3)), dtype=np.float64).copy(),
                    "positions": [np.asarray(info.get("start", np.zeros(3)), dtype=np.float64).copy()],
                    "currents": [np.asarray(info.get("current_inertial", np.zeros(3)), dtype=np.float64).copy()],
                    "obstacle_initial": np.asarray(info.get("obstacle_centers", np.zeros((0, 3))), dtype=np.float64).copy(),
                    "obstacle_final": np.asarray(info.get("obstacle_centers", np.zeros((0, 3))), dtype=np.float64).copy(),
                    "obstacle_radii": np.asarray(info.get("obstacle_radii", np.zeros((0,))), dtype=np.float64).copy(),
                    "event": "",
                }
            episode_diag = EpisodeDiagnostics(info, REWARD_COMPONENT_KEYS)
            context = SequenceContext(self.cfg.seq_len, self.obs_dim, self.act_dim)
            prev_action = np.zeros(self.act_dim, dtype=np.float32)
            context.append(obs, prev_action)

            done = False
            ep_return = 0.0
            ep_len = 0
            last_distance = float(info.get("mission_distance", 0.0))
            while not done:
                obs_seq_t, prev_act_seq_t, valid_mask_t = context.tensorize(self.device)
                action_t, _, _ = self.policy.act(obs_seq_t, prev_act_seq_t, valid_mask_t, deterministic=True)
                action = action_t.squeeze(0).cpu().numpy().astype(np.float32)
                next_obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                ep_return += float(reward)
                ep_len += 1
                episode_diag.record_step(action, float(reward), info)
                last_distance = float(info.get("distance_to_goal", last_distance))
                if episode_trajectory is not None:
                    episode_trajectory["positions"].append(
                        np.asarray(info.get("position", np.zeros(3)), dtype=np.float64).copy()
                    )
                    episode_trajectory["currents"].append(
                        np.asarray(info.get("current_inertial", np.zeros(3)), dtype=np.float64).copy()
                    )
                    episode_trajectory["obstacle_final"] = np.asarray(
                        info.get("obstacle_centers", np.zeros((0, 3))),
                        dtype=np.float64,
                    ).copy()
                    episode_trajectory["event"] = str(info.get("event") or "")
                context.append(next_obs, action)
                obs = next_obs

                if done and info.get("goal_reached", False):
                    successes += 1

            returns.append(ep_return)
            lengths.append(ep_len)
            final_distances.append(last_distance)
            episode_row = episode_diag.to_row()
            episode_rows.append(episode_row)
            if episode_trajectory is not None:
                if fallback_trajectory is None:
                    fallback_trajectory = episode_trajectory
                if capture_first_failure:
                    if not episode_diag.success and captured_trajectory is None:
                        captured_trajectory = episode_trajectory
                else:
                    captured_trajectory = episode_trajectory

            completed = ep_idx + 1
            if self.cfg.eval_progress_every > 0 and (
                completed == 1 or completed == n_episodes or completed % self.cfg.eval_progress_every == 0
            ):
                elapsed = max(time.perf_counter() - eval_started_at, 1e-6)
                episodes_per_sec = completed / elapsed
                remaining = (n_episodes - completed) / max(episodes_per_sec, 1e-9)
                mean_len = float(np.mean(lengths)) if lengths else 0.0
                print(
                    f"  Eval progress {completed}/{n_episodes} "
                    f"success={successes / max(completed, 1):.2%} "
                    f"mean_len={mean_len:.1f} "
                    f"eps/s={episodes_per_sec:.3f} "
                    f"eta={remaining / 3600.0:.2f}h"
                )

        if captured_trajectory is None and fallback_trajectory is not None:
            captured_trajectory = fallback_trajectory

        return (
            {
                "eval_return": float(np.mean(returns)) if returns else 0.0,
                "eval_len": float(np.mean(lengths)) if lengths else 0.0,
                "eval_success_rate": float(successes / max(n_episodes, 1)),
                "eval_final_distance": float(np.mean(final_distances)) if final_distances else 0.0,
            },
            episode_rows,
            captured_trajectory,
        )

    def _save_checkpoint(self, path: Path, update: int, eval_stats: Dict[str, float]) -> None:
        ckpt = {
            "model_state_dict": self.policy.state_dict(),
            "obs_rms": self.obs_rms.state_dict(),
            "ppo_config": asdict(self.cfg),
            "reward_config": asdict(self.reward_cfg),
            "curriculum_stage": asdict(self.current_stage),
            "eval_stats": eval_stats,
            "update": update,
        }
        torch.save(ckpt, path)

    def train(self) -> None:
        total_updates = max(1, math.ceil(self.cfg.total_steps / self.cfg.rollout_steps))
        best_key = (-1.0, float("-inf"))
        global_step = 0
        train_started_at = time.perf_counter()

        summary_path = self.run_dir / "run_config.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_version": self.cfg.run_version,
                    "run_dir": str(self.run_dir),
                    "checkpoint_dir": str(self.save_dir),
                    "ppo_config": asdict(self.cfg),
                    "reward_config": asdict(self.reward_cfg),
                    "curriculum": [asdict(stage) for stage in self.curriculum],
                },
                f,
                indent=2,
            )
        print(f"[Run] version={self.cfg.run_version}")
        print(f"[Run] logs={self.run_dir}")

        for update in range(1, total_updates + 1):
            batch, rollout_stats = self.collect_rollout()
            global_step += self.cfg.rollout_steps
            loss_stats = self.update(batch)
            elapsed_sec = time.perf_counter() - train_started_at
            self._write_train_logs(
                update=update,
                global_step=global_step,
                rollout_stats=rollout_stats,
                loss_stats=loss_stats,
                elapsed_sec=elapsed_sec,
            )

            msg = (
                f"[Update {update:04d}/{total_updates:04d}] "
                f"stage={self.current_stage.stage_id}:{self.current_stage.label} "
                f"step={global_step} "
                f"episodes={rollout_stats['episodes']} "
                f"successes={rollout_stats['successes']} "
                f"mean_return={rollout_stats['mean_return']:.2f} "
                f"mean_len={rollout_stats['mean_len']:.1f} "
                f"policy_loss={loss_stats['policy']:.4f} "
                f"value_loss={loss_stats['value']:.4f} "
                f"entropy={loss_stats['entropy']:.4f}"
            )
            print(msg)

            if update % self.cfg.eval_every_updates == 0 or update == total_updates:
                is_stage4_eval = self.current_stage.stage_id == 4
                capture_stage4_plot = (
                    is_stage4_eval
                    and self.cfg.plot_stage4
                    and self.cfg.plot_stage4_every_evals > 0
                    and self.stage4_eval_count % self.cfg.plot_stage4_every_evals == 0
                )
                capture_ep = self.cfg.plot_stage4_episode_index if capture_stage4_plot else None
                eval_stats, eval_episode_rows, trajectory = self.evaluate(
                    self.cfg.eval_episodes,
                    capture_trajectory_episode=capture_ep,
                )
                self._write_eval_logs(
                    update=update,
                    global_step=global_step,
                    eval_kind="periodic",
                    eval_stats=eval_stats,
                    raw_episode_rows=eval_episode_rows,
                )
                if trajectory is not None:
                    self._plot_stage4_trajectory(
                        trajectory,
                        update=update,
                        global_step=global_step,
                        eval_stats=eval_stats,
                    )
                if is_stage4_eval:
                    self.stage4_eval_count += 1
                print(
                    f"  Eval: return={eval_stats['eval_return']:.2f}, "
                    f"len={eval_stats['eval_len']:.1f}, "
                    f"final_dist={eval_stats['eval_final_distance']:.2f}, "
                    f"success_rate={eval_stats['eval_success_rate']:.2%}"
                )
                ckpt_last = self.save_dir / "transformer_ppo_last.pt"
                self._save_checkpoint(ckpt_last, update, eval_stats)

                current_key = (
                    eval_stats["eval_success_rate"],
                    -eval_stats["eval_final_distance"],
                )
                if current_key > best_key:
                    best_key = current_key
                    ckpt_best = self.save_dir / "transformer_ppo_best.pt"
                    self._save_checkpoint(ckpt_best, update, eval_stats)
                    print(f"  Saved new best model to {ckpt_best}")

                self.maybe_advance_curriculum(eval_stats["eval_success_rate"])


# ============================================================
# CLI
# ============================================================


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transformer PPO for current-blind REMUS AUV navigation")
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--rollout-steps", type=int, default=4096)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--init-propeller-bias", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--save-dir", type=str, default="./checkpoints_transformer_remus")
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--eval-every-updates", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--no-csv-logs", action="store_true")
    parser.add_argument("--no-guided-action-prior", action="store_true")
    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--start-stage", type=int, default=0)
    parser.add_argument("--no-stage4-plots", action="store_true")
    parser.add_argument("--no-show-stage4-plots", action="store_true")
    parser.add_argument("--plot-stage4-every-evals", type=int, default=1)
    parser.add_argument(
        "--plot-stage4-episode-index",
        type=int,
        default=-1,
        help="Stage 4 plot episode index. Use -1 to plot the first failed eval episode, or episode 0 if all succeed.",
    )
    parser.add_argument("--eval-progress-every", type=int, default=100)
    parser.add_argument("--max-first-failure-plot-scan-episodes", type=int, default=200)
    parser.add_argument("--eval-only", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)

    reward_cfg = RewardConfig()
    cfg = PPOConfig(
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        seq_len=args.seq_len,
        learning_rate=args.learning_rate,
        init_propeller_bias=args.init_propeller_bias,
        seed=args.seed,
        device=args.device,
        save_dir=args.save_dir,
        run_dir=args.run_dir,
        log_csv=not args.no_csv_logs,
        eval_every_updates=args.eval_every_updates,
        eval_episodes=args.eval_episodes,
        use_guided_action_prior=not args.no_guided_action_prior,
        resume_from=args.resume_from,
        start_stage=args.start_stage,
        plot_stage4=not args.no_stage4_plots,
        show_stage4_plots=not args.no_show_stage4_plots,
        plot_stage4_every_evals=args.plot_stage4_every_evals,
        plot_stage4_episode_index=args.plot_stage4_episode_index,
        eval_progress_every=args.eval_progress_every,
        max_first_failure_plot_scan_episodes=args.max_first_failure_plot_scan_episodes,
    )

    # Build one temporary env to infer dimensions after wrappers/normalization.
    REMUSAUVEnv = load_env_class()
    tmp_stage = DEFAULT_CURRICULUM[0]
    tmp_env = REMUSAUVEnv(
        seed=args.seed,
        current_enabled=tmp_stage.current_enabled,
        include_current_in_obs=False,
        n_obstacles=tmp_stage.n_obstacles,
    )
    tmp_env.target_z_min = DEFAULT_CURRICULUM[0].target_z_min
    tmp_env.target_z_max = DEFAULT_CURRICULUM[0].target_z_max
    if tmp_stage.target_distance_min is not None and hasattr(tmp_env, "min_start_target_distance"):
        tmp_env.min_start_target_distance = tmp_stage.target_distance_min
    if hasattr(tmp_env, "max_start_target_distance"):
        tmp_env.max_start_target_distance = tmp_stage.target_distance_max
    if tmp_stage.target_boundary_margin is not None and hasattr(tmp_env, "target_boundary_margin"):
        tmp_env.target_boundary_margin = tmp_stage.target_boundary_margin
    if tmp_stage.max_steps is not None and hasattr(tmp_env, "max_steps"):
        tmp_env.max_steps = tmp_stage.max_steps
    tmp_env = RemusRewardWrapper(tmp_env, reward_cfg)
    obs_dim = int(np.prod(tmp_env.observation_space.shape))
    act_dim = int(np.prod(tmp_env.action_space.shape))
    tmp_env.close()

    obs_rms = RunningMeanStd(shape=(obs_dim,))

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

    trainer = PPOTrainer(
        policy=policy,
        cfg=cfg,
        reward_cfg=reward_cfg,
        obs_rms=obs_rms,
        curriculum=DEFAULT_CURRICULUM,
    )
    if args.eval_only:
        capture_ep = args.plot_stage4_episode_index if cfg.plot_stage4 and trainer.current_stage.stage_id == 4 else None
        eval_stats, eval_episode_rows, trajectory = trainer.evaluate(
            cfg.eval_episodes,
            capture_trajectory_episode=capture_ep,
        )
        trainer._write_eval_logs(
            update=0,
            global_step=0,
            eval_kind="eval_only",
            eval_stats=eval_stats,
            raw_episode_rows=eval_episode_rows,
        )
        if trajectory is not None:
            trainer._plot_stage4_trajectory(
                trajectory,
                update=0,
                global_step=0,
                eval_stats=eval_stats,
            )
        print(
            f"Eval-only stage={trainer.current_stage.stage_id}:{trainer.current_stage.label} "
            f"return={eval_stats['eval_return']:.2f}, "
            f"len={eval_stats['eval_len']:.1f}, "
            f"final_dist={eval_stats['eval_final_distance']:.2f}, "
            f"success_rate={eval_stats['eval_success_rate']:.2%}"
        )
        return
    trainer.train()


if __name__ == "__main__":
    main()
