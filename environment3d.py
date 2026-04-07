from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
import math

import numpy as np


VectorLike = Iterable[float]
EARTH_ROTATION_RATE_RAD_PER_SEC = 7.2921159e-5


@dataclass
class CoriolisConfig:
    """Latitude-controlled Coriolis configuration.

    The local coordinate frame used by the physics helpers is ENU-like:
    - X: east-west
    - Y: north-south
    - Z: up-down

    In the Panda3D/RL setup this simply means that the same simulation frame is
    interpreted consistently everywhere. The key user-facing control variable is
    ``latitude_deg``.
    """

    latitude_deg: float | None = 36.0
    enabled: bool = True
    traditional_approximation: bool = True
    strength_scale: float = 1.0
    earth_rotation_rate_rad_per_sec: float = EARTH_ROTATION_RATE_RAD_PER_SEC

    def __post_init__(self) -> None:
        if self.latitude_deg is not None:
            self.latitude_deg = float(self.latitude_deg)
            if not -90.0 <= self.latitude_deg <= 90.0:
                raise ValueError("latitude_deg must be between -90 and 90, or None.")
        self.enabled = bool(self.enabled)
        self.traditional_approximation = bool(self.traditional_approximation)
        self.strength_scale = float(self.strength_scale)
        self.earth_rotation_rate_rad_per_sec = float(self.earth_rotation_rate_rad_per_sec)
        if self.earth_rotation_rate_rad_per_sec <= 0.0:
            raise ValueError("earth_rotation_rate_rad_per_sec must be > 0.")

    @property
    def active(self) -> bool:
        return self.enabled and self.latitude_deg is not None and abs(self.strength_scale) > 0.0

    @property
    def latitude_rad(self) -> float | None:
        if self.latitude_deg is None:
            return None
        return math.radians(self.latitude_deg)

    @property
    def coriolis_parameter(self) -> float:
        if not self.active:
            return 0.0
        phi = float(self.latitude_rad)
        return 2.0 * self.earth_rotation_rate_rad_per_sec * math.sin(phi) * self.strength_scale

    @property
    def max_reference_coriolis_parameter(self) -> float:
        return 2.0 * self.earth_rotation_rate_rad_per_sec

    @property
    def local_rotation_vector(self) -> np.ndarray:
        """Earth rotation vector expressed in local ENU coordinates.

        Ω_local = [0, Ω cos(phi), Ω sin(phi)]
        """
        if not self.active:
            return np.zeros(3, dtype=float)
        phi = float(self.latitude_rad)
        omega = self.earth_rotation_rate_rad_per_sec * self.strength_scale
        return np.array(
            [
                0.0,
                omega * math.cos(phi),
                omega * math.sin(phi),
            ],
            dtype=float,
        )


@dataclass
class Environment3D:
    """Axis-aligned 3D environment with environment-owned physics helpers."""

    min_bound: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float)
    )
    max_bound: np.ndarray = field(
        default_factory=lambda: np.array([3.0, 3.0, 3.0], dtype=float)
    )
    coriolis: CoriolisConfig = field(default_factory=CoriolisConfig)

    def __post_init__(self) -> None:
        self.min_bound = np.asarray(self.min_bound, dtype=float).reshape(3)
        self.max_bound = np.asarray(self.max_bound, dtype=float).reshape(3)
        if not isinstance(self.coriolis, CoriolisConfig):
            self.coriolis = CoriolisConfig(**dict(self.coriolis))  # type: ignore[arg-type]

        if np.any(self.max_bound <= self.min_bound):
            raise ValueError("Each max bound must be greater than min bound.")

    @classmethod
    def from_size(
        cls,
        size: VectorLike = (3.0, 3.0, 3.0),
        origin: VectorLike = (0.0, 0.0, 0.0),
        *,
        coriolis: CoriolisConfig | None = None,
        latitude_deg: float | None = 36.0,
        coriolis_enabled: bool = True,
        coriolis_scale: float = 1.0,
        traditional_approximation: bool = True,
    ) -> "Environment3D":
        origin_arr = np.asarray(origin, dtype=float).reshape(3)
        size_arr = np.asarray(size, dtype=float).reshape(3)

        if np.any(size_arr <= 0):
            raise ValueError("Each size value must be > 0.")

        if coriolis is None:
            coriolis = CoriolisConfig(
                latitude_deg=latitude_deg,
                enabled=coriolis_enabled,
                traditional_approximation=traditional_approximation,
                strength_scale=coriolis_scale,
            )

        return cls(min_bound=origin_arr, max_bound=origin_arr + size_arr, coriolis=coriolis)

    @property
    def size(self) -> np.ndarray:
        return self.max_bound - self.min_bound

    @property
    def center(self) -> np.ndarray:
        return (self.min_bound + self.max_bound) / 2.0

    @property
    def latitude_deg(self) -> float | None:
        return self.coriolis.latitude_deg

    @property
    def coriolis_enabled(self) -> bool:
        return self.coriolis.active

    @property
    def coriolis_parameter(self) -> float:
        return self.coriolis.coriolis_parameter

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
        if clearance < 0.0:
            raise ValueError("clearance must be >= 0.")

        lower = self.min_bound + clearance
        upper = self.max_bound - clearance
        if np.any(upper <= lower):
            raise ValueError("clearance leaves no room inside the environment.")
        return rng.uniform(lower, upper)

    def coriolis_acceleration(self, velocity: VectorLike) -> np.ndarray:
        vel = np.asarray(velocity, dtype=float).reshape(3)
        if not self.coriolis.active:
            return np.zeros(3, dtype=float)

        if self.coriolis.traditional_approximation:
            f = self.coriolis_parameter
            return np.array([f * vel[1], -f * vel[0], 0.0], dtype=float)

        omega_local = self.coriolis.local_rotation_vector
        return -2.0 * np.cross(omega_local, vel)

    def environmental_acceleration(
        self,
        *,
        position: VectorLike | None = None,
        velocity: VectorLike,
    ) -> np.ndarray:
        del position  # reserved for future spatially varying fields.
        return self.coriolis_acceleration(velocity)

    def apply_coriolis_to_velocity(
        self,
        velocity: VectorLike,
        dt: float,
    ) -> np.ndarray:
        """Advance velocity under the Coriolis-only term for one time step.

        This is especially useful for moving obstacles whose speed magnitude is
        kept fixed while their heading slowly precesses because of Coriolis.
        """
        vel = np.asarray(velocity, dtype=float).reshape(3)
        dt = float(dt)
        if abs(dt) < 1e-12 or not self.coriolis.active:
            return vel.copy()

        if self.coriolis.traditional_approximation:
            f = self.coriolis_parameter
            if abs(f) < 1e-12:
                return vel.copy()
            theta = f * dt
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)
            x_new = cos_theta * vel[0] + sin_theta * vel[1]
            y_new = -sin_theta * vel[0] + cos_theta * vel[1]
            return np.array([x_new, y_new, vel[2]], dtype=float)

        omega_local = self.coriolis.local_rotation_vector
        omega_norm = float(np.linalg.norm(omega_local))
        if omega_norm < 1e-12:
            return vel.copy()

        axis = omega_local / omega_norm
        theta = -2.0 * omega_norm * dt
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        cross_term = np.cross(axis, vel)
        parallel = axis * float(np.dot(axis, vel))
        perpendicular = vel - parallel
        rotated = parallel + perpendicular * cos_theta + cross_term * sin_theta
        return np.asarray(rotated, dtype=float)
