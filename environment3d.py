from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
import numpy as np


VectorLike = Iterable[float]


@dataclass
class Environment3D:
    min_bound: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float)
    )
    max_bound: np.ndarray = field(
        default_factory=lambda: np.array([3.0, 3.0, 3.0], dtype=float)
    )

    def __post_init__(self) -> None:
        self.min_bound = np.asarray(self.min_bound, dtype=float).reshape(3)
        self.max_bound = np.asarray(self.max_bound, dtype=float).reshape(3)

        if np.any(self.max_bound <= self.min_bound):
            raise ValueError("Each max bound must be greater than min bound.")

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

    def contains(self, point: VectorLike) -> bool:
        p = np.asarray(point, dtype=float).reshape(3)
        return bool(np.all(p >= self.min_bound) and np.all(p <= self.max_bound))

    def clamp(self, point: VectorLike) -> np.ndarray:
        p = np.asarray(point, dtype=float).reshape(3)
        return np.clip(p, self.min_bound, self.max_bound)

    def random_point(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.uniform(self.min_bound, self.max_bound)