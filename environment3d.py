from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence
import math

import numpy as np


VectorLike = Iterable[float]
ReservedSphereLike = tuple[VectorLike, float]


@dataclass
class SphereObstacle:
    center: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float).reshape(3)
        self.radius = float(self.radius)

        if self.radius <= 0:
            raise ValueError("Obstacle radius must be > 0.")

    @property
    def volume(self) -> float:
        return (4.0 / 3.0) * math.pi * (self.radius ** 3)

    def contains(self, point: VectorLike, margin: float = 0.0) -> bool:
        p = np.asarray(point, dtype=float).reshape(3)
        return bool(np.linalg.norm(p - self.center) <= self.radius + float(margin))

    def signed_distance(self, point: VectorLike) -> float:
        p = np.asarray(point, dtype=float).reshape(3)
        return float(np.linalg.norm(p - self.center) - self.radius)


@dataclass
class Environment3D:
    min_bound: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float)
    )
    max_bound: np.ndarray = field(
        default_factory=lambda: np.array([3.0, 3.0, 3.0], dtype=float)
    )
    obstacles: list[SphereObstacle] = field(default_factory=list)

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
        origin = np.asarray(origin, dtype=float).reshape(3)
        size = np.asarray(size, dtype=float).reshape(3)

        if np.any(size <= 0):
            raise ValueError("Each size value must be > 0.")

        return cls(min_bound=origin, max_bound=origin + size)

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

    @staticmethod
    def sphere_volume(radius: float) -> float:
        radius = float(radius)
        if radius <= 0:
            raise ValueError("Sphere radius must be > 0.")
        return (4.0 / 3.0) * math.pi * (radius ** 3)

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
