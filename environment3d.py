from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


VectorLike = Iterable[float]
ReservedSphereLike = tuple[VectorLike, float]


@dataclass
class SphereObstacle:
    center: np.ndarray
    radius: float
    velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float).reshape(3)
        self.radius = float(self.radius)
        self.velocity = np.asarray(self.velocity, dtype=float).reshape(3)

        if self.radius <= 0:
            raise ValueError("Obstacle radius must be > 0.")

    @property
    def volume(self) -> float:
        return (4.0 / 3.0) * math.pi * (self.radius ** 3)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    def contains(self, point: VectorLike, margin: float = 0.0) -> bool:
        p = np.asarray(point, dtype=float).reshape(3)
        return bool(np.linalg.norm(p - self.center) <= self.radius + float(margin))

    def signed_distance(self, point: VectorLike) -> float:
        p = np.asarray(point, dtype=float).reshape(3)
        return float(np.linalg.norm(p - self.center) - self.radius)


@dataclass
class ObstacleMotionConfig:
    """Parameters that control obstacle motion after generation."""

    enabled: bool = True
    speed_ratio_to_auv_max: float = 0.10
    target_clearance: float = 1.0
    resolution_passes: int = 2

    def speed_from_auv_max(self, auv_max_speed: float) -> float:
        auv_max_speed = float(auv_max_speed)
        if auv_max_speed < 0:
            raise ValueError("auv_max_speed must be >= 0.")
        return max(0.0, self.speed_ratio_to_auv_max * auv_max_speed)


@dataclass
class ObstacleConfig:
    """Parameters that control obstacle generation for the environment."""

    radius: float = 1.35
    count: int | None = None
    complexity: float | None = 0.015
    clearance_multiplier: float = 2.0
    clearance_padding: float = 0.05
    max_attempts_per_obstacle: int = 1200
    motion: ObstacleMotionConfig = field(default_factory=ObstacleMotionConfig)

    def placement_clearance(self, agent_radius: float) -> float:
        agent_radius = float(agent_radius)
        if agent_radius < 0:
            raise ValueError("agent_radius must be >= 0.")
        return self.clearance_multiplier * agent_radius + self.clearance_padding


@dataclass
class SpawnConfig:
    """Environment-owned defaults for start/goal sampling."""

    start_offset_xy: tuple[float, float] = (2.0, 2.0)
    start_height_ratio: float = 0.5
    start_reserved_extra: float = 1.75
    target_boundary_padding: float = 0.35
    min_target_boundary_clearance: float = 0.80
    target_obstacle_padding: float = 0.15
    target_min_distance_from_agent: float = 6.0
    target_max_attempts: int = 2000

    def start_position(self, env: "Environment3D") -> np.ndarray:
        point = np.array(
            [
                env.min_bound[0] + self.start_offset_xy[0],
                env.min_bound[1] + self.start_offset_xy[1],
                env.min_bound[2] + env.size[2] * self.start_height_ratio,
            ],
            dtype=float,
        )
        return env.clamp(point)

    def start_reserved_radius(self, agent_radius: float) -> float:
        agent_radius = float(agent_radius)
        if agent_radius < 0:
            raise ValueError("agent_radius must be >= 0.")
        return agent_radius + self.start_reserved_extra

    def target_boundary_clearance(self, target_radius: float) -> float:
        target_radius = float(target_radius)
        if target_radius < 0:
            raise ValueError("target_radius must be >= 0.")
        return max(
            target_radius + self.target_boundary_padding,
            self.min_target_boundary_clearance,
        )

    def target_obstacle_clearance(self, target_radius: float) -> float:
        target_radius = float(target_radius)
        if target_radius < 0:
            raise ValueError("target_radius must be >= 0.")
        return target_radius + self.target_obstacle_padding


@dataclass
class EnvironmentVisualConfig:
    """Rendering-related constants for the environment scene."""

    box_line_thickness: float = 2.0
    grid_spacing: float = 2.0


@dataclass
class EnvironmentConfig:
    """Single place where environment-related defaults live.

    Edit this structure instead of scattering size, obstacle, spawn, and
    obstacle-motion tuning inside the Panda3D application code.
    """

    size: tuple[float, float, float] = (30.0, 30.0, 12.0)
    origin: tuple[float, float, float] = (-15.0, -15.0, 0.0)
    random_seed: int | None = 7
    obstacle: ObstacleConfig = field(default_factory=ObstacleConfig)
    spawn: SpawnConfig = field(default_factory=SpawnConfig)
    visual: EnvironmentVisualConfig = field(default_factory=EnvironmentVisualConfig)


DEFAULT_ENVIRONMENT_CONFIG = EnvironmentConfig()


@dataclass
class Environment3D:
    min_bound: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float)
    )
    max_bound: np.ndarray = field(
        default_factory=lambda: np.array([3.0, 3.0, 3.0], dtype=float)
    )
    obstacles: list[SphereObstacle] = field(default_factory=list)
    config: EnvironmentConfig | None = None
    requested_obstacle_count: int = 0
    requested_obstacle_complexity: float = 0.0
    obstacle_mode: str = "complexity"

    def __post_init__(self) -> None:
        self.min_bound = np.asarray(self.min_bound, dtype=float).reshape(3)
        self.max_bound = np.asarray(self.max_bound, dtype=float).reshape(3)
        self.obstacles = [
            obstacle if isinstance(obstacle, SphereObstacle) else SphereObstacle(*obstacle)
            for obstacle in self.obstacles
        ]

        if np.any(self.max_bound <= self.min_bound):
            raise ValueError("Each max bound must be greater than min bound.")

        for idx, obstacle in enumerate(self.obstacles):
            if not self.is_sphere_inside(obstacle.center, obstacle.radius):
                raise ValueError("Obstacle must be fully contained inside the environment.")
            for other in self.obstacles[:idx]:
                gap = np.linalg.norm(obstacle.center - other.center)
                if gap < obstacle.radius + other.radius:
                    raise ValueError("Obstacles must not overlap.")

    @classmethod
    def from_size(
        cls,
        size: VectorLike = (3.0, 3.0, 3.0),
        origin: VectorLike = (0.0, 0.0, 0.0),
    ) -> "Environment3D":
        origin_arr = np.asarray(origin, dtype=float).reshape(3)
        size_arr = np.asarray(size, dtype=float).reshape(3)

        if np.any(size_arr <= 0):
            raise ValueError("Each size value must be > 0.")

        config = EnvironmentConfig(
            size=tuple(map(float, size_arr.tolist())),
            origin=tuple(map(float, origin_arr.tolist())),
        )
        return cls(min_bound=origin_arr, max_bound=origin_arr + size_arr, config=config)

    @classmethod
    def from_config(cls, config: EnvironmentConfig | None = None) -> "Environment3D":
        cfg = copy.deepcopy(DEFAULT_ENVIRONMENT_CONFIG if config is None else config)
        origin = np.asarray(cfg.origin, dtype=float).reshape(3)
        size = np.asarray(cfg.size, dtype=float).reshape(3)

        if np.any(size <= 0):
            raise ValueError("Each size value must be > 0.")

        return cls(min_bound=origin, max_bound=origin + size, config=cfg)

    @classmethod
    def default(cls) -> "Environment3D":
        return cls.from_config(DEFAULT_ENVIRONMENT_CONFIG)

    @property
    def size(self) -> np.ndarray:
        return self.max_bound - self.min_bound

    @property
    def center(self) -> np.ndarray:
        return (self.min_bound + self.max_bound) / 2.0

    @property
    def volume(self) -> float:
        return float(np.prod(self.size))

    @property
    def obstacle_volume(self) -> float:
        return float(sum(obstacle.volume for obstacle in self.obstacles))

    @property
    def obstacle_complexity(self) -> float:
        if self.volume <= 0:
            return 0.0
        return self.obstacle_volume / self.volume

    @property
    def obstacle_radius(self) -> float:
        if self.config is None:
            return 0.0
        return float(self.config.obstacle.radius)

    @property
    def obstacle_motion(self) -> ObstacleMotionConfig:
        if self.config is None:
            return ObstacleMotionConfig(enabled=False)
        return self.config.obstacle.motion

    @property
    def dynamic_obstacles_enabled(self) -> bool:
        return bool(self.obstacle_motion.enabled)

    @property
    def grid_spacing(self) -> float:
        if self.config is None:
            return 2.0
        return float(self.config.visual.grid_spacing)

    @property
    def box_line_thickness(self) -> float:
        if self.config is None:
            return 2.0
        return float(self.config.visual.box_line_thickness)

    @staticmethod
    def sphere_volume(radius: float) -> float:
        radius = float(radius)
        if radius <= 0:
            raise ValueError("Sphere radius must be > 0.")
        return (4.0 / 3.0) * math.pi * (radius ** 3)

    @staticmethod
    def _vector_norm(vector: VectorLike) -> float:
        return float(np.linalg.norm(np.asarray(vector, dtype=float).reshape(3)))

    @staticmethod
    def _safe_unit(
        vector: VectorLike,
        fallback: VectorLike | None = None,
    ) -> np.ndarray:
        vec = np.asarray(vector, dtype=float).reshape(3)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            return vec / norm

        if fallback is None:
            return np.array([1.0, 0.0, 0.0], dtype=float)

        fallback_vec = np.asarray(fallback, dtype=float).reshape(3)
        fallback_norm = float(np.linalg.norm(fallback_vec))
        if fallback_norm > 1e-8:
            return fallback_vec / fallback_norm

        return np.array([1.0, 0.0, 0.0], dtype=float)

    @staticmethod
    def random_unit_vector(
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        rng = rng or np.random.default_rng()
        while True:
            vector = rng.normal(0.0, 1.0, size=3)
            norm = float(np.linalg.norm(vector))
            if norm > 1e-8:
                return vector / norm

    @classmethod
    def random_unit_vector_in_hemisphere(
        cls,
        normal: VectorLike,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        rng = rng or np.random.default_rng()
        surface_normal = cls._safe_unit(normal)

        for _ in range(32):
            direction = cls.random_unit_vector(rng)
            if float(np.dot(direction, surface_normal)) >= 0.0:
                return direction

        direction = cls.random_unit_vector(rng)
        if float(np.dot(direction, surface_normal)) < 0.0:
            direction = -direction
        return cls._safe_unit(direction, fallback=surface_normal)

    def obstacle_count_from_complexity(self, radius: float, complexity: float) -> int:
        radius = float(radius)
        complexity = float(complexity)

        if radius <= 0:
            raise ValueError("Sphere radius must be > 0.")
        if not 0.0 <= complexity <= 1.0:
            raise ValueError("Obstacle complexity must be between 0 and 1.")

        target_volume = complexity * self.volume
        sphere_volume = self.sphere_volume(radius)
        return max(0, int(round(target_volume / sphere_volume)))

    def complexity_from_count(self, radius: float, count: int) -> float:
        count = int(count)
        if count < 0:
            raise ValueError("Obstacle count must be >= 0.")
        if count == 0:
            return 0.0
        return (count * self.sphere_volume(radius)) / self.volume

    def contains(self, point: VectorLike) -> bool:
        p = np.asarray(point, dtype=float).reshape(3)
        return bool(np.all(p >= self.min_bound) and np.all(p <= self.max_bound))

    def clamp(self, point: VectorLike) -> np.ndarray:
        p = np.asarray(point, dtype=float).reshape(3)
        return np.clip(p, self.min_bound, self.max_bound)

    def random_point(
        self,
        rng: np.random.Generator | None = None,
        clearance: float = 0.0,
    ) -> np.ndarray:
        rng = rng or np.random.default_rng()
        clearance = float(clearance)
        if clearance < 0:
            raise ValueError("clearance must be >= 0.")

        lower = self.min_bound + clearance
        upper = self.max_bound - clearance
        if np.any(upper <= lower):
            raise ValueError("clearance leaves no room inside the environment.")

        return rng.uniform(lower, upper)

    def clear_obstacles(self) -> None:
        self.obstacles.clear()

    def make_rng(self) -> np.random.Generator:
        seed = None
        if self.config is not None:
            seed = self.config.random_seed
        return np.random.default_rng(seed)

    def default_start_position(self) -> np.ndarray:
        if self.config is None:
            point = np.array(
                [
                    self.min_bound[0] + 2.0,
                    self.min_bound[1] + 2.0,
                    self.min_bound[2] + self.size[2] * 0.5,
                ],
                dtype=float,
            )
            return self.clamp(point)

        return self.config.spawn.start_position(self)

    def dynamic_obstacle_speed(self, auv_max_speed: float) -> float:
        return self.obstacle_motion.speed_from_auv_max(auv_max_speed)

    def target_obstacle_keepout_radius(self, target_radius: float) -> float:
        target_radius = float(target_radius)
        if target_radius < 0:
            raise ValueError("target_radius must be >= 0.")

        spawn_cfg = self.config.spawn if self.config is not None else SpawnConfig()
        spawn_keepout = spawn_cfg.target_obstacle_clearance(target_radius)
        motion_keepout = target_radius + max(0.0, self.obstacle_motion.target_clearance)
        return max(spawn_keepout, motion_keepout)

    def initialize_dynamic_obstacles(
        self,
        *,
        auv_max_speed: float,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not self.dynamic_obstacles_enabled or not self.obstacles:
            return

        rng = rng or self.make_rng()
        speed = self.dynamic_obstacle_speed(auv_max_speed)
        if speed <= 0.0:
            for obstacle in self.obstacles:
                obstacle.velocity = np.zeros(3, dtype=float)
            return

        for obstacle in self.obstacles:
            direction = self._safe_unit(
                obstacle.velocity,
                fallback=self.random_unit_vector(rng),
            )
            obstacle.velocity = direction * speed

    def apply_configured_obstacles(
        self,
        *,
        auv_radius: float,
        rng: np.random.Generator | None = None,
        start_pos: VectorLike | None = None,
    ) -> list[SphereObstacle]:
        if self.config is None:
            self.requested_obstacle_count = 0
            self.requested_obstacle_complexity = 0.0
            self.obstacle_mode = "count"
            self.clear_obstacles()
            return []

        obstacle_cfg = self.config.obstacle
        spawn_cfg = self.config.spawn

        if obstacle_cfg.count is not None:
            self.obstacle_mode = "count"
            self.requested_obstacle_count = max(0, int(obstacle_cfg.count))
            self.requested_obstacle_complexity = self.complexity_from_count(
                radius=obstacle_cfg.radius,
                count=self.requested_obstacle_count,
            )
        else:
            self.obstacle_mode = "complexity"
            requested_complexity = (
                0.0
                if obstacle_cfg.complexity is None
                else float(obstacle_cfg.complexity)
            )
            if not 0.0 <= requested_complexity <= 1.0:
                raise ValueError("obstacle_complexity must be between 0 and 1.")
            self.requested_obstacle_complexity = requested_complexity
            self.requested_obstacle_count = self.obstacle_count_from_complexity(
                radius=obstacle_cfg.radius,
                complexity=requested_complexity,
            )

        start = (
            self.default_start_position()
            if start_pos is None
            else np.asarray(start_pos, dtype=float).reshape(3)
        )
        reserved_spheres = [
            (start, spawn_cfg.start_reserved_radius(auv_radius)),
        ]

        obstacles = self.generate_random_sphere_obstacles(
            radius=obstacle_cfg.radius,
            count=self.requested_obstacle_count,
            rng=rng,
            clearance=obstacle_cfg.placement_clearance(auv_radius),
            reserved_spheres=reserved_spheres,
            max_attempts_per_obstacle=obstacle_cfg.max_attempts_per_obstacle,
        )
        return obstacles

    def sample_target_point(
        self,
        *,
        current_pos: VectorLike,
        target_radius: float,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        current = np.asarray(current_pos, dtype=float).reshape(3)
        spawn_cfg = self.config.spawn if self.config is not None else SpawnConfig()
        keepout = self.target_obstacle_keepout_radius(target_radius)

        try:
            point = self.random_free_point(
                rng=rng,
                boundary_clearance=spawn_cfg.target_boundary_clearance(target_radius),
                obstacle_clearance=keepout,
                reserved_spheres=[(current, spawn_cfg.target_min_distance_from_agent)],
                max_attempts=spawn_cfg.target_max_attempts,
            )
        except RuntimeError:
            point = self.center.copy()
            point[2] = max(point[2], self.min_bound[2] + spawn_cfg.min_target_boundary_clearance)
            point = self.push_point_out_of_obstacles(
                point,
                clearance=keepout + 0.05,
            )
            point = self.clamp(point)

        return point

    def update_dynamic_obstacles(
        self,
        *,
        dt: float,
        auv_max_speed: float,
        target_center: VectorLike | None = None,
        target_radius: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        dt = float(dt)
        if dt < 0:
            raise ValueError("dt must be >= 0.")
        if not self.dynamic_obstacles_enabled or not self.obstacles:
            return

        rng = rng or self.make_rng()
        speed = self.dynamic_obstacle_speed(auv_max_speed)
        if speed <= 0.0:
            for obstacle in self.obstacles:
                obstacle.velocity = np.zeros(3, dtype=float)
            return

        self.initialize_dynamic_obstacles(auv_max_speed=auv_max_speed, rng=rng)

        target_center_arr = None
        target_keepout = 0.0
        if target_center is not None:
            target_center_arr = np.asarray(target_center, dtype=float).reshape(3)
            target_keepout = self.target_obstacle_keepout_radius(target_radius)

        for obstacle in self.obstacles:
            obstacle.center = obstacle.center + obstacle.velocity * dt

        collision_normals = [np.zeros(3, dtype=float) for _ in self.obstacles]
        resolution_passes = max(1, int(self.obstacle_motion.resolution_passes))

        for _ in range(resolution_passes):
            adjusted_this_pass = False

            for idx, obstacle in enumerate(self.obstacles):
                adjusted, normal = self._resolve_obstacle_with_bounds(obstacle)
                if adjusted:
                    collision_normals[idx] += normal
                    adjusted_this_pass = True

            if target_center_arr is not None and target_keepout > 0.0:
                for idx, obstacle in enumerate(self.obstacles):
                    adjusted, normal = self._resolve_obstacle_with_target_zone(
                        obstacle=obstacle,
                        target_center=target_center_arr,
                        keepout_radius=target_keepout,
                        rng=rng,
                    )
                    if adjusted:
                        collision_normals[idx] += normal
                        adjusted_this_pass = True

            for left_idx in range(len(self.obstacles) - 1):
                for right_idx in range(left_idx + 1, len(self.obstacles)):
                    adjusted, normal = self._resolve_obstacle_pair(
                        self.obstacles[left_idx],
                        self.obstacles[right_idx],
                        rng=rng,
                    )
                    if adjusted:
                        collision_normals[left_idx] += normal
                        collision_normals[right_idx] -= normal
                        adjusted_this_pass = True

            if not adjusted_this_pass:
                break

        for idx, obstacle in enumerate(self.obstacles):
            outward = collision_normals[idx]
            if self._vector_norm(outward) > 1e-8:
                new_direction = self.random_unit_vector_in_hemisphere(outward, rng=rng)
                obstacle.velocity = new_direction * speed
            else:
                obstacle.velocity = self._safe_unit(
                    obstacle.velocity,
                    fallback=self.random_unit_vector(rng),
                ) * speed

    def is_sphere_inside(
        self,
        center: VectorLike,
        radius: float,
        clearance: float = 0.0,
    ) -> bool:
        center = np.asarray(center, dtype=float).reshape(3)
        radius = float(radius)
        clearance = float(clearance)
        margin = radius + clearance
        return bool(
            np.all(center - margin >= self.min_bound)
            and np.all(center + margin <= self.max_bound)
        )

    def is_point_obstacle_free(self, point: VectorLike, clearance: float = 0.0) -> bool:
        point = np.asarray(point, dtype=float).reshape(3)
        clearance = float(clearance)
        return all(not obstacle.contains(point, margin=clearance) for obstacle in self.obstacles)

    def can_place_sphere(
        self,
        center: VectorLike,
        radius: float,
        clearance: float = 0.0,
        other_obstacles: Sequence[SphereObstacle] | None = None,
        reserved_spheres: Sequence[ReservedSphereLike] | None = None,
    ) -> bool:
        center = np.asarray(center, dtype=float).reshape(3)
        radius = float(radius)
        clearance = float(clearance)

        if radius <= 0:
            raise ValueError("Sphere radius must be > 0.")
        if clearance < 0:
            raise ValueError("clearance must be >= 0.")
        if not self.is_sphere_inside(center, radius, clearance=clearance):
            return False

        obstacles = list(self.obstacles if other_obstacles is None else other_obstacles)
        for obstacle in obstacles:
            distance = np.linalg.norm(center - obstacle.center)
            if distance < radius + obstacle.radius + clearance:
                return False

        for reserved_center, reserved_radius in self._normalize_reserved_spheres(reserved_spheres):
            distance = np.linalg.norm(center - reserved_center)
            if distance < radius + reserved_radius + clearance:
                return False

        return True

    def random_free_point(
        self,
        rng: np.random.Generator | None = None,
        boundary_clearance: float = 0.0,
        obstacle_clearance: float = 0.0,
        reserved_spheres: Sequence[ReservedSphereLike] | None = None,
        max_attempts: int = 1000,
    ) -> np.ndarray:
        rng = rng or np.random.default_rng()
        boundary_clearance = float(boundary_clearance)
        obstacle_clearance = float(obstacle_clearance)
        if max_attempts <= 0:
            raise ValueError("max_attempts must be > 0.")

        reserved = self._normalize_reserved_spheres(reserved_spheres)
        for _ in range(max_attempts):
            point = self.random_point(rng=rng, clearance=boundary_clearance)
            if not self.is_point_obstacle_free(point, clearance=obstacle_clearance):
                continue

            overlaps_reserved = False
            for reserved_center, reserved_radius in reserved:
                if np.linalg.norm(point - reserved_center) < reserved_radius:
                    overlaps_reserved = True
                    break
            if overlaps_reserved:
                continue

            return point

        raise RuntimeError("Could not sample a free point in the current environment.")

    def push_point_out_of_obstacles(
        self,
        point: VectorLike,
        clearance: float = 0.0,
        max_passes: int = 3,
    ) -> np.ndarray:
        point_arr = np.asarray(point, dtype=float).reshape(3).copy()
        clearance = float(clearance)
        if clearance < 0:
            raise ValueError("clearance must be >= 0.")
        if max_passes <= 0:
            raise ValueError("max_passes must be > 0.")

        for _ in range(max_passes):
            adjusted = False
            point_arr = self.clamp(point_arr)

            for obstacle in self.obstacles:
                delta = point_arr - obstacle.center
                distance = float(np.linalg.norm(delta))
                min_distance = obstacle.radius + clearance
                if distance >= min_distance:
                    continue

                adjusted = True
                if distance < 1e-8:
                    delta = np.array([1.0, 0.0, 0.0], dtype=float)
                    distance = 1.0
                point_arr = obstacle.center + (delta / distance) * min_distance

            if not adjusted:
                break

        return self.clamp(point_arr)

    def generate_random_sphere_obstacles(
        self,
        *,
        radius: float,
        count: int | None = None,
        complexity: float | None = None,
        rng: np.random.Generator | None = None,
        clearance: float = 0.0,
        reserved_spheres: Sequence[ReservedSphereLike] | None = None,
        max_attempts_per_obstacle: int = 200,
    ) -> list[SphereObstacle]:
        radius = float(radius)
        clearance = float(clearance)
        if radius <= 0:
            raise ValueError("Sphere radius must be > 0.")
        if clearance < 0:
            raise ValueError("clearance must be >= 0.")
        if max_attempts_per_obstacle <= 0:
            raise ValueError("max_attempts_per_obstacle must be > 0.")

        if count is None:
            if complexity is None:
                raise ValueError("Either count or complexity must be provided.")
            count = self.obstacle_count_from_complexity(radius=radius, complexity=complexity)

        count = int(count)
        if count < 0:
            raise ValueError("Obstacle count must be >= 0.")

        self.clear_obstacles()
        if count == 0:
            return []

        rng = rng or np.random.default_rng()
        reserved = self._normalize_reserved_spheres(reserved_spheres)
        generated: list[SphereObstacle] = []
        total_attempts = count * max_attempts_per_obstacle

        for _ in range(total_attempts):
            if len(generated) >= count:
                break

            center = self.random_point(rng=rng, clearance=radius + clearance)
            if not self.can_place_sphere(
                center=center,
                radius=radius,
                clearance=clearance,
                other_obstacles=generated,
                reserved_spheres=reserved,
            ):
                continue

            generated.append(SphereObstacle(center=center, radius=radius))

        self.obstacles = generated
        return list(self.obstacles)

    def _resolve_obstacle_with_bounds(
        self,
        obstacle: SphereObstacle,
    ) -> tuple[bool, np.ndarray]:
        lower = self.min_bound + obstacle.radius
        upper = self.max_bound - obstacle.radius
        normal = np.zeros(3, dtype=float)
        adjusted = False

        for axis in range(3):
            if obstacle.center[axis] < lower[axis]:
                obstacle.center[axis] = lower[axis]
                normal[axis] += 1.0
                adjusted = True
            elif obstacle.center[axis] > upper[axis]:
                obstacle.center[axis] = upper[axis]
                normal[axis] -= 1.0
                adjusted = True

        if not adjusted:
            return False, normal
        return True, self._safe_unit(normal)

    def _resolve_obstacle_with_target_zone(
        self,
        *,
        obstacle: SphereObstacle,
        target_center: np.ndarray,
        keepout_radius: float,
        rng: np.random.Generator,
    ) -> tuple[bool, np.ndarray]:
        delta = obstacle.center - target_center
        distance = float(np.linalg.norm(delta))
        min_distance = obstacle.radius + float(keepout_radius)
        if distance >= min_distance:
            return False, np.zeros(3, dtype=float)

        normal = self._safe_unit(
            delta,
            fallback=obstacle.velocity if obstacle.speed > 1e-8 else self.random_unit_vector(rng),
        )
        obstacle.center = target_center + normal * min_distance
        return True, normal

    def _resolve_obstacle_pair(
        self,
        left: SphereObstacle,
        right: SphereObstacle,
        *,
        rng: np.random.Generator,
    ) -> tuple[bool, np.ndarray]:
        delta = left.center - right.center
        distance = float(np.linalg.norm(delta))
        min_distance = left.radius + right.radius
        if distance >= min_distance:
            return False, np.zeros(3, dtype=float)

        fallback = left.velocity - right.velocity
        if self._vector_norm(fallback) < 1e-8:
            fallback = self.random_unit_vector(rng)
        normal = self._safe_unit(delta, fallback=fallback)

        overlap = min_distance - distance
        correction = normal * (0.5 * overlap + 1e-6)
        left.center = left.center + correction
        right.center = right.center - correction
        return True, normal

    @staticmethod
    def _normalize_reserved_spheres(
        reserved_spheres: Sequence[ReservedSphereLike] | None,
    ) -> list[tuple[np.ndarray, float]]:
        normalized: list[tuple[np.ndarray, float]] = []
        if reserved_spheres is None:
            return normalized

        for center, radius in reserved_spheres:
            c = np.asarray(center, dtype=float).reshape(3)
            r = float(radius)
            if r < 0:
                raise ValueError("Reserved sphere radius must be >= 0.")
            normalized.append((c, r))

        return normalized
