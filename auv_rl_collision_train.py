from __future__ import annotations

"""RL training script for AUV target reaching with hard collision failures.

This file is intentionally self-contained:
- it reuses ``Environment3D`` from ``environment3d.py`` only for the 3D box
  bounds and random point sampling,
- it implements its own spherical obstacle field,
- it treats wall/obstacle contact as immediate failure,
- it provides PPO training and evaluation helpers.

Typical usage
-------------
Train a PPO policy:
    python auv_rl_collision_train.py --mode train

Evaluate a saved PPO policy:
    python auv_rl_collision_train.py --mode eval --model-path runs/auv_ppo/final_model

Run a smoke test with random actions (no SB3 required):
    python auv_rl_collision_train.py --mode random --episodes 3
"""

import argparse
import copy
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from environment3d import Environment3D


# ---------------------------------------------------------------------------
# Optional Gymnasium import.
# The small fallback classes make it possible to import this file and run the
# random smoke test even when Gymnasium is not installed.
# ---------------------------------------------------------------------------
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - fallback path.
    gym = None

    class _FallbackEnvBase:
        metadata: dict[str, Any] = {}

        def reset(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
            raise NotImplementedError

        def step(
            self,
            action: np.ndarray,
        ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            raise NotImplementedError

        def render(self) -> Any:
            return None

        def close(self) -> None:
            return None

    class _FallbackBox:
        def __init__(
            self,
            low: float | np.ndarray,
            high: float | np.ndarray,
            shape: tuple[int, ...],
            dtype: Any = np.float32,
        ) -> None:
            self.shape = tuple(int(v) for v in shape)
            self.dtype = dtype
            self.low = np.full(self.shape, low, dtype=dtype)
            self.high = np.full(self.shape, high, dtype=dtype)

        def sample(self) -> np.ndarray:
            return np.random.uniform(self.low, self.high).astype(self.dtype)

    class _FallbackSpaces:
        Box = _FallbackBox

    spaces = _FallbackSpaces()
    GymEnvBase = _FallbackEnvBase
else:  # pragma: no cover - path used when gymnasium is installed.
    GymEnvBase = gym.Env


Vector3 = np.ndarray


@dataclass
class ObstacleFieldConfig:
    radius: float = 1.35
    count: int | None = None
    complexity: float | None = 0.015
    clearance_padding: float = 0.05
    max_attempts_per_obstacle: int = 1200


@dataclass
class SpawnConfig:
    start_offset_xy: tuple[float, float] = (2.0, 2.0)
    start_height_ratio: float = 0.5
    start_reserved_extra: float = 1.75
    target_boundary_padding: float = 0.35
    min_target_boundary_clearance: float = 0.80
    target_obstacle_padding: float = 0.15
    target_min_distance_from_agent: float = 6.0
    target_max_attempts: int = 2000


@dataclass
class LayoutConfig:
    size: tuple[float, float, float] = (30.0, 30.0, 12.0)
    origin: tuple[float, float, float] = (-15.0, -15.0, 0.0)
    obstacle_field: ObstacleFieldConfig = field(default_factory=ObstacleFieldConfig)
    spawn: SpawnConfig = field(default_factory=SpawnConfig)
    seed: int = 7


@dataclass
class AUVSimConfig:
    # AUV / target geometry.
    auv_radius: float = 0.65
    target_radius: float = 0.42
    capture_radius: float = 0.50

    # Motion dynamics.
    max_speed: float = 6.0
    max_accel: float = 7.5
    dt: float = 0.10
    episode_horizon_steps: int = 500

    # Moving obstacles.
    moving_obstacles: bool = True
    obstacle_speed_ratio_to_auv: float = 0.10
    obstacle_target_keepout: float = 1.0
    obstacle_resolution_passes: int = 2

    # Observation design.
    nearest_obstacles_in_observation: int = 8

    # Reward shaping.
    progress_reward_scale: float = 2.0
    speed_to_goal_reward_scale: float = 0.10
    time_penalty: float = 0.01
    danger_penalty_scale: float = 0.25
    danger_clearance: float = 1.0
    success_reward: float = 150.0
    collision_penalty: float = 150.0
    timeout_penalty: float = 10.0

    # Randomization.
    seed: int = 7

    @property
    def obstacle_speed(self) -> float:
        return self.max_speed * self.obstacle_speed_ratio_to_auv


@dataclass
class PPOTrainConfig:
    total_timesteps: int = 300_000
    n_envs: int = 8
    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 512
    n_epochs: int = 10
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    clip_range: float = 0.2
    max_grad_norm: float = 0.5
    seed: int = 7
    device: str = "auto"
    save_dir: str = "runs/auv_ppo"
    policy_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "net_arch": {
                "pi": [256, 256, 128],
                "vf": [256, 256, 128],
            }
        }
    )


@dataclass
class SphereObstacle:
    center: Vector3
    radius: float
    velocity: Vector3

    def copy(self) -> "SphereObstacle":
        return SphereObstacle(
            center=self.center.copy(),
            radius=float(self.radius),
            velocity=self.velocity.copy(),
        )

    @property
    def volume(self) -> float:
        return sphere_volume(self.radius)


class AUVNavigationRLEnv(GymEnvBase):
    """Headless RL environment for AUV target reaching.

    Rules
    -----
    - action: continuous 3D acceleration in [-1, 1]^3
    - reward: progress to target + small shaping
    - success: reach target capture radius
    - failure: hit wall or any obstacle
    - timeout: exceed step horizon
    """

    metadata = {"render_modes": ["none"], "render_fps": 10}

    def __init__(
        self,
        layout_config: LayoutConfig | None = None,
        sim_config: AUVSimConfig | None = None,
    ) -> None:
        self.layout_config = copy.deepcopy(LayoutConfig() if layout_config is None else layout_config)
        self.sim_config = copy.deepcopy(AUVSimConfig() if sim_config is None else sim_config)

        self._episode_index = 0
        self.rng = np.random.default_rng(self.sim_config.seed)

        self.env: Environment3D | None = None
        self.agent_pos = np.zeros(3, dtype=float)
        self.agent_vel = np.zeros(3, dtype=float)
        self.target_pos = np.zeros(3, dtype=float)
        self.obstacles: list[SphereObstacle] = []
        self.previous_distance = 0.0
        self.step_count = 0
        self.last_event = "reset"
        self.requested_obstacle_count = 0
        self.requested_obstacle_complexity = 0.0

        obs_dim = self._observation_dim()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32)

    # ------------------------------------------------------------------
    # Gym-style API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options

        if gym is not None:
            super_reset = getattr(super(), "reset", None)
            if callable(super_reset):
                super_reset(seed=seed)

        episode_seed = self.sim_config.seed + self._episode_index if seed is None else int(seed)
        self._episode_index += 1
        self.rng = np.random.default_rng(episode_seed)

        self.env = Environment3D.from_size(
            size=self.layout_config.size,
            origin=self.layout_config.origin,
        )
        self.agent_pos = self._default_start_position()
        self.agent_vel = np.zeros(3, dtype=float)
        self.last_event = "running"
        self.step_count = 0

        self.obstacles = self._generate_obstacles()
        self.target_pos = self._sample_target_point()
        self._resolve_dynamic_constraints()

        self.previous_distance = self._distance(self.agent_pos, self.target_pos)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.env is None:
            raise RuntimeError("Environment must be reset before calling step().")

        action_arr = np.asarray(action, dtype=np.float32).reshape(3)
        action_arr = np.clip(action_arr, -1.0, 1.0)
        cfg = self.sim_config
        self.step_count += 1

        if cfg.moving_obstacles and self.obstacles:
            self._update_dynamic_obstacles(cfg.dt)
            if self._agent_hits_any_obstacle(self.agent_pos):
                self.last_event = "obstacle_collision"
                obs = self._get_obs()
                info = self._get_info(collision=True, success=False)
                return obs, -cfg.collision_penalty, True, False, info

        accel_cmd = action_arr.astype(float) * cfg.max_accel
        self.agent_vel = self.agent_vel + accel_cmd * cfg.dt
        self.agent_vel = self._clip_norm(self.agent_vel, cfg.max_speed)

        proposed_pos = self.agent_pos + self.agent_vel * cfg.dt

        if not self._is_sphere_inside_env(proposed_pos, cfg.auv_radius):
            self.last_event = "wall_collision"
            obs = self._get_obs()
            info = self._get_info(collision=True, success=False)
            return obs, -cfg.collision_penalty, True, False, info

        if self._agent_hits_any_obstacle(proposed_pos):
            self.last_event = "obstacle_collision"
            obs = self._get_obs()
            info = self._get_info(collision=True, success=False)
            return obs, -cfg.collision_penalty, True, False, info

        self.agent_pos = proposed_pos
        current_distance = self._distance(self.agent_pos, self.target_pos)

        if current_distance <= cfg.capture_radius:
            self.last_event = "success"
            reward = cfg.success_reward
            reward += cfg.progress_reward_scale * (self.previous_distance - current_distance)
            self.previous_distance = current_distance
            obs = self._get_obs()
            info = self._get_info(collision=False, success=True)
            return obs, float(reward), True, False, info

        reward = self._shaped_reward(current_distance)
        self.previous_distance = current_distance

        truncated = self.step_count >= cfg.episode_horizon_steps
        if truncated:
            self.last_event = "timeout"
            reward -= cfg.timeout_penalty

        obs = self._get_obs()
        info = self._get_info(collision=False, success=False)
        return obs, float(reward), False, truncated, info

    def render(self) -> dict[str, Any]:
        return {
            "agent_pos": self.agent_pos.copy(),
            "agent_vel": self.agent_vel.copy(),
            "target_pos": self.target_pos.copy(),
            "obstacles": [obstacle.copy() for obstacle in self.obstacles],
            "requested_obstacle_count": int(self.requested_obstacle_count),
            "event": self.last_event,
            "step_count": self.step_count,
        }

    def close(self) -> None:
        return None

    # ------------------------------------------------------------------
    # Reset helpers
    # ------------------------------------------------------------------
    def _default_start_position(self) -> Vector3:
        if self.env is None:
            raise RuntimeError("Environment not initialized.")
        spawn = self.layout_config.spawn
        point = np.array(
            [
                self.env.min_bound[0] + spawn.start_offset_xy[0],
                self.env.min_bound[1] + spawn.start_offset_xy[1],
                self.env.min_bound[2] + self.env.size[2] * spawn.start_height_ratio,
            ],
            dtype=float,
        )
        return self.env.clamp(point)

    def _generate_obstacles(self) -> list[SphereObstacle]:
        if self.env is None:
            raise RuntimeError("Environment not initialized.")

        layout = self.layout_config
        field_cfg = layout.obstacle_field
        cfg = self.sim_config

        if field_cfg.count is not None:
            count = max(0, int(field_cfg.count))
            complexity = complexity_from_count(float(np.prod(self.env.size)), field_cfg.radius, count)
        else:
            complexity = 0.0 if field_cfg.complexity is None else float(field_cfg.complexity)
            if not 0.0 <= complexity <= 1.0:
                raise ValueError("Obstacle complexity must be between 0 and 1.")
            count = obstacle_count_from_complexity(float(np.prod(self.env.size)), field_cfg.radius, complexity)

        self.requested_obstacle_count = count
        self.requested_obstacle_complexity = complexity

        start_reserved_radius = cfg.auv_radius + layout.spawn.start_reserved_extra
        obstacle_speed = cfg.obstacle_speed if cfg.moving_obstacles else 0.0

        generated: list[SphereObstacle] = []
        total_attempts = max(1, count * field_cfg.max_attempts_per_obstacle)
        required_start_gap = field_cfg.radius + start_reserved_radius + field_cfg.clearance_padding

        for _ in range(total_attempts):
            if len(generated) >= count:
                break

            center = self._random_point_with_clearance(field_cfg.radius)
            if self._distance(center, self.agent_pos) < required_start_gap:
                continue

            valid = True
            for other in generated:
                required_gap = field_cfg.radius + other.radius + field_cfg.clearance_padding
                if self._distance(center, other.center) < required_gap:
                    valid = False
                    break
            if not valid:
                continue

            generated.append(
                SphereObstacle(
                    center=center,
                    radius=field_cfg.radius,
                    velocity=self._random_unit_vector() * obstacle_speed,
                )
            )

        return generated

    def _sample_target_point(self) -> Vector3:
        if self.env is None:
            raise RuntimeError("Environment not initialized.")

        spawn = self.layout_config.spawn
        cfg = self.sim_config
        boundary_clearance = max(
            cfg.target_radius + spawn.target_boundary_padding,
            spawn.min_target_boundary_clearance,
        )
        obstacle_clearance = cfg.target_radius + spawn.target_obstacle_padding

        for _ in range(spawn.target_max_attempts):
            point = self._random_point_with_clearance(boundary_clearance)
            if self._distance(point, self.agent_pos) < spawn.target_min_distance_from_agent:
                continue
            if not self._is_point_obstacle_free(point, clearance=obstacle_clearance):
                continue
            return point

        fallback = self.env.center.copy()
        fallback[2] = max(fallback[2], self.env.min_bound[2] + spawn.min_target_boundary_clearance)
        return self._push_point_out_of_obstacles(fallback, clearance=obstacle_clearance)

    # ------------------------------------------------------------------
    # Observation / reward helpers
    # ------------------------------------------------------------------
    def _observation_dim(self) -> int:
        k = int(self.sim_config.nearest_obstacles_in_observation)
        base = 3 + 3 + 3 + 1 + 6 + 1  # pos, vel, target rel, target dist, wall margins, min clearance
        per_obstacle = 3 + 3 + 1 + 1 + 1  # rel pos, rel vel, radius, clearance, active flag
        return base + k * per_obstacle

    def _get_obs(self) -> np.ndarray:
        if self.env is None:
            raise RuntimeError("Environment not initialized.")

        env = self.env
        cfg = self.sim_config
        diag = float(np.linalg.norm(env.size))
        diag = max(diag, 1e-6)
        max_obs_speed = max(cfg.obstacle_speed, 1e-6)

        pos_centered = (self.agent_pos - env.center) / np.maximum(env.size / 2.0, 1e-6)
        vel_scaled = self.agent_vel / max(cfg.max_speed, 1e-6)
        target_rel = (self.target_pos - self.agent_pos) / diag
        target_dist = np.array([self._distance(self.agent_pos, self.target_pos) / diag], dtype=float)

        wall_margins = np.array(
            [
                (self.agent_pos[0] - env.min_bound[0]) / env.size[0],
                (env.max_bound[0] - self.agent_pos[0]) / env.size[0],
                (self.agent_pos[1] - env.min_bound[1]) / env.size[1],
                (env.max_bound[1] - self.agent_pos[1]) / env.size[1],
                (self.agent_pos[2] - env.min_bound[2]) / env.size[2],
                (env.max_bound[2] - self.agent_pos[2]) / env.size[2],
            ],
            dtype=float,
        )

        clearances: list[tuple[float, np.ndarray]] = []
        for obstacle in self.obstacles:
            clearance = self._agent_clearance_to_obstacle(self.agent_pos, obstacle)
            features = np.concatenate(
                [
                    (obstacle.center - self.agent_pos) / diag,
                    obstacle.velocity / max_obs_speed,
                    np.array([obstacle.radius / diag], dtype=float),
                    np.array([clearance / diag], dtype=float),
                    np.array([1.0], dtype=float),
                ]
            )
            clearances.append((clearance, features))

        clearances.sort(key=lambda item: item[0])
        blocks: list[np.ndarray] = []
        for _, features in clearances[: cfg.nearest_obstacles_in_observation]:
            blocks.append(features)

        zero_block = np.zeros(3 + 3 + 1 + 1 + 1, dtype=float)
        while len(blocks) < cfg.nearest_obstacles_in_observation:
            blocks.append(zero_block.copy())

        min_clearance = np.array([
            0.0 if not clearances else clearances[0][0] / diag
        ], dtype=float)

        obs = np.concatenate(
            [
                pos_centered,
                vel_scaled,
                target_rel,
                target_dist,
                wall_margins,
                min_clearance,
                *blocks,
            ]
        )
        return obs.astype(np.float32)

    def _shaped_reward(self, current_distance: float) -> float:
        cfg = self.sim_config
        progress = self.previous_distance - current_distance
        reward = cfg.progress_reward_scale * progress

        to_target = self.target_pos - self.agent_pos
        target_norm = float(np.linalg.norm(to_target))
        if target_norm > 1e-8:
            speed_toward_target = float(np.dot(self.agent_vel, to_target / target_norm))
            reward += cfg.speed_to_goal_reward_scale * (speed_toward_target / max(cfg.max_speed, 1e-6))

        reward -= cfg.time_penalty

        nearest_clearance = self._nearest_obstacle_clearance(self.agent_pos)
        if nearest_clearance < cfg.danger_clearance:
            reward -= cfg.danger_penalty_scale * (cfg.danger_clearance - nearest_clearance)

        return float(reward)

    def _get_info(
        self,
        *,
        collision: bool = False,
        success: bool = False,
    ) -> dict[str, Any]:
        return {
            "is_success": bool(success),
            "collision": bool(collision),
            "event": self.last_event,
            "distance_to_target": float(self._distance(self.agent_pos, self.target_pos)),
            "min_obstacle_clearance": float(self._nearest_obstacle_clearance(self.agent_pos)),
            "step_count": int(self.step_count),
            "requested_obstacle_count": int(self.requested_obstacle_count),
            "actual_obstacle_count": int(len(self.obstacles)),
        }

    # ------------------------------------------------------------------
    # Dynamic obstacle updates
    # ------------------------------------------------------------------
    def _update_dynamic_obstacles(self, dt: float) -> None:
        for obstacle in self.obstacles:
            obstacle.center = obstacle.center + obstacle.velocity * dt
        self._resolve_dynamic_constraints()

    def _resolve_dynamic_constraints(self) -> None:
        if self.env is None:
            return

        cfg = self.sim_config
        speed = cfg.obstacle_speed

        for _ in range(cfg.obstacle_resolution_passes):
            # Wall and target keepout handling.
            for obstacle in self.obstacles:
                wall_normal = np.zeros(3, dtype=float)
                lower = self.env.min_bound + obstacle.radius
                upper = self.env.max_bound - obstacle.radius

                for axis in range(3):
                    if obstacle.center[axis] < lower[axis]:
                        obstacle.center[axis] = lower[axis]
                        wall_normal[axis] += 1.0
                    elif obstacle.center[axis] > upper[axis]:
                        obstacle.center[axis] = upper[axis]
                        wall_normal[axis] -= 1.0

                if np.linalg.norm(wall_normal) > 0.0:
                    obstacle.velocity = self._random_hemisphere_direction(wall_normal) * speed

                keepout = obstacle.radius + cfg.target_radius + cfg.obstacle_target_keepout
                delta_target = obstacle.center - self.target_pos
                dist_target = float(np.linalg.norm(delta_target))
                if dist_target < keepout:
                    target_normal = self._safe_normalized(delta_target)
                    if target_normal is None:
                        target_normal = self._random_unit_vector()
                    obstacle.center = self.target_pos + target_normal * keepout
                    obstacle.center = np.clip(
                        obstacle.center,
                        self.env.min_bound + obstacle.radius,
                        self.env.max_bound - obstacle.radius,
                    )
                    obstacle.velocity = self._random_hemisphere_direction(target_normal) * speed

            # Obstacle-obstacle handling.
            for i in range(len(self.obstacles)):
                for j in range(i + 1, len(self.obstacles)):
                    a = self.obstacles[i]
                    b = self.obstacles[j]
                    delta = b.center - a.center
                    dist = float(np.linalg.norm(delta))
                    min_dist = a.radius + b.radius
                    if dist >= min_dist:
                        continue

                    normal = self._safe_normalized(delta)
                    if normal is None:
                        normal = self._random_unit_vector()
                        dist = 0.0

                    overlap = min_dist - dist
                    a.center = a.center - normal * (0.5 * overlap)
                    b.center = b.center + normal * (0.5 * overlap)
                    a.center = np.clip(a.center, self.env.min_bound + a.radius, self.env.max_bound - a.radius)
                    b.center = np.clip(b.center, self.env.min_bound + b.radius, self.env.max_bound - b.radius)
                    a.velocity = self._random_hemisphere_direction(-normal) * speed
                    b.velocity = self._random_hemisphere_direction(normal) * speed

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _random_point_with_clearance(self, clearance: float) -> Vector3:
        if self.env is None:
            raise RuntimeError("Environment not initialized.")
        clearance = float(clearance)
        lower = self.env.min_bound + clearance
        upper = self.env.max_bound - clearance
        if np.any(upper <= lower):
            raise ValueError("Clearance leaves no room inside the environment.")
        return self.rng.uniform(lower, upper)

    def _is_sphere_inside_env(self, center: Vector3, radius: float) -> bool:
        if self.env is None:
            raise RuntimeError("Environment not initialized.")
        radius = float(radius)
        return bool(
            np.all(center - radius >= self.env.min_bound)
            and np.all(center + radius <= self.env.max_bound)
        )

    def _is_point_obstacle_free(self, point: Vector3, clearance: float = 0.0) -> bool:
        clearance = float(clearance)
        for obstacle in self.obstacles:
            if self._distance(point, obstacle.center) <= obstacle.radius + clearance:
                return False
        return True

    def _push_point_out_of_obstacles(self, point: Vector3, clearance: float = 0.0) -> Vector3:
        if self.env is None:
            raise RuntimeError("Environment not initialized.")

        point_arr = self.env.clamp(point)
        for _ in range(4):
            adjusted = False
            for obstacle in self.obstacles:
                delta = point_arr - obstacle.center
                dist = float(np.linalg.norm(delta))
                min_dist = obstacle.radius + clearance
                if dist >= min_dist:
                    continue
                adjusted = True
                normal = self._safe_normalized(delta)
                if normal is None:
                    normal = self._random_unit_vector()
                point_arr = obstacle.center + normal * min_dist
                point_arr = self.env.clamp(point_arr)
            if not adjusted:
                break
        return point_arr

    def _agent_hits_any_obstacle(self, position: Vector3) -> bool:
        return any(self._agent_clearance_to_obstacle(position, obstacle) <= 0.0 for obstacle in self.obstacles)

    def _agent_clearance_to_obstacle(self, position: Vector3, obstacle: SphereObstacle) -> float:
        return float(np.linalg.norm(position - obstacle.center) - (self.sim_config.auv_radius + obstacle.radius))

    def _nearest_obstacle_clearance(self, position: Vector3) -> float:
        if not self.obstacles:
            return float("inf")
        return min(self._agent_clearance_to_obstacle(position, obstacle) for obstacle in self.obstacles)

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------
    def _random_unit_vector(self) -> Vector3:
        vec = self.rng.normal(size=3)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0], dtype=float)
        return vec / norm

    def _random_hemisphere_direction(self, outward_normal: Vector3) -> Vector3:
        normal = self._safe_normalized(outward_normal)
        if normal is None:
            return self._random_unit_vector()
        for _ in range(32):
            candidate = self._random_unit_vector()
            if float(np.dot(candidate, normal)) > 0.0:
                return candidate
        return normal

    @staticmethod
    def _safe_normalized(vec: Vector3) -> Vector3 | None:
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            return None
        return vec / norm

    @staticmethod
    def _clip_norm(vec: Vector3, max_norm: float) -> Vector3:
        norm = float(np.linalg.norm(vec))
        if norm <= max_norm or norm < 1e-12:
            return vec
        return vec * (max_norm / norm)

    @staticmethod
    def _distance(a: Vector3, b: Vector3) -> float:
        return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def sphere_volume(radius: float) -> float:
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("Sphere radius must be > 0.")
    return (4.0 / 3.0) * math.pi * (radius ** 3)


def obstacle_count_from_complexity(env_volume: float, radius: float, complexity: float) -> int:
    if not 0.0 <= complexity <= 1.0:
        raise ValueError("Obstacle complexity must be between 0 and 1.")
    return max(0, int(round((env_volume * complexity) / sphere_volume(radius))))


def complexity_from_count(env_volume: float, radius: float, count: int) -> float:
    count = int(count)
    if count < 0:
        raise ValueError("Obstacle count must be >= 0.")
    if env_volume <= 0.0:
        return 0.0
    return (count * sphere_volume(radius)) / env_volume


def ensure_rl_dependencies() -> None:
    if gym is None:
        raise ImportError("Gymnasium is required. Install it with: pip install gymnasium")

    try:
        import stable_baselines3  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Stable-Baselines3 is required. Install it with: pip install 'stable-baselines3[extra]'"
        ) from exc


def train_agent(
    *,
    layout_config: LayoutConfig | None = None,
    sim_config: AUVSimConfig | None = None,
    train_config: PPOTrainConfig | None = None,
) -> Path:
    ensure_rl_dependencies()

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    layout_cfg = copy.deepcopy(LayoutConfig() if layout_config is None else layout_config)
    sim_cfg = copy.deepcopy(AUVSimConfig() if sim_config is None else sim_config)
    train_cfg = copy.deepcopy(PPOTrainConfig() if train_config is None else train_config)

    save_dir = Path(train_cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = save_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    single_env = AUVNavigationRLEnv(layout_config=layout_cfg, sim_config=sim_cfg)
    check_env(single_env, warn=True, skip_render_check=True)
    single_env.close()

    def make_env(rank: int):
        def _factory() -> AUVNavigationRLEnv:
            env = AUVNavigationRLEnv(layout_config=layout_cfg, sim_config=sim_cfg)
            env.reset(seed=train_cfg.seed + rank)
            return env

        return _factory

    vec_cls = SubprocVecEnv if train_cfg.n_envs > 1 else DummyVecEnv
    vec_env = vec_cls([make_env(i) for i in range(train_cfg.n_envs)])
    vec_env = VecMonitor(vec_env)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=train_cfg.learning_rate,
        n_steps=train_cfg.n_steps,
        batch_size=train_cfg.batch_size,
        n_epochs=train_cfg.n_epochs,
        gamma=train_cfg.gamma,
        gae_lambda=train_cfg.gae_lambda,
        ent_coef=train_cfg.ent_coef,
        vf_coef=train_cfg.vf_coef,
        clip_range=train_cfg.clip_range,
        max_grad_norm=train_cfg.max_grad_norm,
        tensorboard_log=str(tensorboard_dir),
        seed=train_cfg.seed,
        verbose=1,
        device=train_cfg.device,
        policy_kwargs=train_cfg.policy_kwargs,
    )

    metadata = {
        "layout_config": asdict(layout_cfg),
        "sim_config": asdict(sim_cfg),
        "train_config": asdict(train_cfg),
    }
    (save_dir / "training_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    model.learn(total_timesteps=train_cfg.total_timesteps, progress_bar=True)

    model_path = save_dir / "final_model"
    model.save(str(model_path))
    vec_env.close()
    return model_path


def evaluate_agent(
    model_path: str | Path,
    *,
    layout_config: LayoutConfig | None = None,
    sim_config: AUVSimConfig | None = None,
    episodes: int = 20,
    deterministic: bool = True,
) -> dict[str, float]:
    ensure_rl_dependencies()

    from stable_baselines3 import PPO

    layout_cfg = copy.deepcopy(LayoutConfig() if layout_config is None else layout_config)
    sim_cfg = copy.deepcopy(AUVSimConfig() if sim_config is None else sim_config)

    env = AUVNavigationRLEnv(layout_config=layout_cfg, sim_config=sim_cfg)
    model = PPO.load(str(model_path))

    returns: list[float] = []
    lengths: list[int] = []
    successes = 0
    collisions = 0
    timeouts = 0

    for episode in range(episodes):
        obs, _ = env.reset(seed=(sim_cfg.seed + 10_000 + episode))
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1

        returns.append(total_reward)
        lengths.append(steps)
        successes += int(info.get("is_success", False))
        collisions += int(info.get("collision", False))
        timeouts += int(info.get("event") == "timeout")

    env.close()
    return {
        "episodes": float(episodes),
        "success_rate": successes / max(episodes, 1),
        "collision_rate": collisions / max(episodes, 1),
        "timeout_rate": timeouts / max(episodes, 1),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
    }


def random_policy_smoke_test(
    *,
    layout_config: LayoutConfig | None = None,
    sim_config: AUVSimConfig | None = None,
    episodes: int = 3,
) -> dict[str, float]:
    env = AUVNavigationRLEnv(
        layout_config=copy.deepcopy(LayoutConfig() if layout_config is None else layout_config),
        sim_config=copy.deepcopy(AUVSimConfig() if sim_config is None else sim_config),
    )

    returns: list[float] = []
    lengths: list[int] = []
    successes = 0
    collisions = 0

    for episode in range(episodes):
        obs, _ = env.reset(seed=(env.sim_config.seed + episode))
        total_reward = 0.0
        terminated = False
        truncated = False
        steps = 0

        while not terminated and not truncated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1

        del obs
        returns.append(total_reward)
        lengths.append(steps)
        successes += int(info.get("is_success", False))
        collisions += int(info.get("collision", False))

    env.close()
    return {
        "episodes": float(episodes),
        "success_rate": successes / max(episodes, 1),
        "collision_rate": collisions / max(episodes, 1),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "eval", "random"], default="train")
    parser.add_argument("--model-path", type=str, default="", help="Saved PPO model path for eval mode")
    parser.add_argument("--episodes", type=int, default=20, help="Evaluation or smoke-test episodes")
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--save-dir", type=str, default="runs/auv_ppo")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--static-obstacles", action="store_true", help="Disable obstacle motion")
    parser.add_argument("--nearest-obstacles", type=int, default=8)
    parser.add_argument("--obstacle-radius", type=float, default=1.35)
    parser.add_argument("--obstacle-count", type=int, default=-1, help="Use -1 to enable complexity mode")
    parser.add_argument("--obstacle-complexity", type=float, default=0.015)
    parser.add_argument("--env-size", type=float, nargs=3, default=(30.0, 30.0, 12.0))
    parser.add_argument("--env-origin", type=float, nargs=3, default=(-15.0, -15.0, 0.0))
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    obstacle_count = None if int(args.obstacle_count) < 0 else int(args.obstacle_count)
    layout_cfg = LayoutConfig(
        size=tuple(float(v) for v in args.env_size),
        origin=tuple(float(v) for v in args.env_origin),
        obstacle_field=ObstacleFieldConfig(
            radius=float(args.obstacle_radius),
            count=obstacle_count,
            complexity=float(args.obstacle_complexity),
        ),
        seed=int(args.seed),
    )
    sim_cfg = AUVSimConfig(
        moving_obstacles=not args.static_obstacles,
        nearest_obstacles_in_observation=int(args.nearest_obstacles),
        seed=int(args.seed),
    )

    if args.mode == "train":
        train_cfg = PPOTrainConfig(
            total_timesteps=int(args.total_timesteps),
            n_envs=int(args.n_envs),
            save_dir=str(args.save_dir),
            seed=int(args.seed),
            device=str(args.device),
        )
        model_path = train_agent(
            layout_config=layout_cfg,
            sim_config=sim_cfg,
            train_config=train_cfg,
        )
        print(f"Saved PPO model to: {model_path}")
        return

    if args.mode == "eval":
        if not args.model_path:
            raise ValueError("--model-path is required in eval mode.")
        summary = evaluate_agent(
            model_path=args.model_path,
            layout_config=layout_cfg,
            sim_config=sim_cfg,
            episodes=int(args.episodes),
        )
        print(json.dumps(summary, indent=2))
        return

    if args.mode == "random":
        summary = random_policy_smoke_test(
            layout_config=layout_cfg,
            sim_config=sim_cfg,
            episodes=int(args.episodes),
        )
        print(json.dumps(summary, indent=2))
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
