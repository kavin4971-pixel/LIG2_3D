import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import gymnasium as gym
from gymnasium import spaces


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def rotation_matrix_body_to_inertial(phi: float, theta: float, psi: float) -> np.ndarray:
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)

    rz = np.array(
        [
            [cps, -sps, 0.0],
            [sps, cps, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    ry = np.array(
        [
            [cth, 0.0, sth],
            [0.0, 1.0, 0.0],
            [-sth, 0.0, cth],
        ]
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cphi, -sphi],
            [0.0, sphi, cphi],
        ]
    )
    return rz @ ry @ rx


def euler_rate_matrix(phi: float, theta: float) -> np.ndarray:
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth = np.cos(theta)
    tth = np.tan(theta)

    return np.array(
        [
            [1.0, sphi * tth, cphi * tth],
            [0.0, cphi, -sphi],
            [0.0, sphi / cth, cphi / cth],
        ]
    )


def rate_limit(current_value: float, desired_value: float, max_rate: float, dt: float) -> float:
    delta = np.clip(desired_value - current_value, -max_rate * dt, max_rate * dt)
    return current_value + delta


def first_order_rate_limited(current_value: float, desired_value: float, time_constant: float, max_rate: float, dt: float) -> float:
    time_constant = max(float(time_constant), 1e-6)
    desired_rate = (desired_value - current_value) / time_constant
    desired_rate = np.clip(desired_rate, -max_rate, max_rate)
    return current_value + dt * desired_rate


@dataclass
class Obstacle:
    center: np.ndarray
    velocity: np.ndarray
    radius: float


class REMUSAUVEnv(gym.Env):
    """
    REMUS-style AUV environment with a fixed start point and randomized target per episode.

    Compared to the baseline version, this variant includes:
      - hull lift / moment terms based on angle of attack and sideslip
      - cross-flow nonlinear maneuvering damping for sway/heave/yaw/pitch
      - full 6x6 added-mass matrix with sway-yaw and heave-pitch coupling
      - first-order actuator lag and propeller RPM state with KT/KQ model
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt: float = 0.05,
        max_steps: int = 2400,
        world_size: float = 60.0,
        n_obstacles: int = 6,
        seed: Optional[int] = None,
        current_enabled: bool = True,
        include_current_in_obs: bool = True,
        fixed_mid_obstacle: bool = False,
        fixed_obstacle_radius: float = 6.0,
        fixed_obstacle_center: Optional[np.ndarray] = None,
    ):
        super().__init__()

        self.dt = dt
        self.max_steps = max_steps
        self.world_size = world_size
        self.n_obstacles = n_obstacles
        self.current_enabled = current_enabled
        self.include_current_in_obs = include_current_in_obs
        self.fixed_mid_obstacle = fixed_mid_obstacle
        self.fixed_obstacle_radius = float(fixed_obstacle_radius)
        self.fixed_obstacle_center = None if fixed_obstacle_center is None else np.asarray(fixed_obstacle_center, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

        # -------------------------------------------------
        # REMUS 100 inspired nominal parameters (simplified)
        # -------------------------------------------------
        self.m = 38.6
        self.g = 9.81
        self.rho = 1025.0
        self.volume = self.m / self.rho  # near-neutral buoyancy assumption

        # Approximate inertia
        self.Ix = 0.18
        self.Iy = 3.2
        self.Iz = 3.2

        # Center of gravity / buoyancy
        self.r_g = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        # Body z is positive downward, so negative z_b places buoyancy above CG.
        # A few cm of separation damps the roll-to-pitch depth coupling that was
        # too strong under sustained propeller reaction torque.
        self.r_b = np.array([0.0, 0.0, -0.03], dtype=np.float64)

        # -------------------------------------------------
        # Vehicle geometry / task scales
        # -------------------------------------------------
        self.auv_length = 1.60
        self.auv_diameter = 0.19
        self.auv_hull_radius = 0.5 * self.auv_diameter
        self.auv_radius = 0.25

        self.goal_radius = 1.25
        self.fixed_start = np.array([-40.0, -40.0, 10.0], dtype=np.float64)
        self.target_z_min = 5.0
        self.target_z_max = 55.0
        self.min_start_target_distance = 30.0
        self.max_start_target_distance: Optional[float] = None
        self.target_boundary_margin = 5.0

        # -------------------------------------------------
        # Full rigid-body / added-mass model
        # -------------------------------------------------
        self.MRB = np.diag([self.m, self.m, self.m, self.Ix, self.Iy, self.Iz]).astype(np.float64)

        # Added-mass couplings: heave-pitch and sway-yaw are the dominant ones.
        self.MA = np.array(
            [
                [8.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 45.0, 0.0, 0.0, 0.0, 4.5],
                [0.0, 0.0, 45.0, 0.0, -4.5, 0.0],
                [0.0, 0.0, 0.0, 0.08, 0.0, 0.0],
                [0.0, 0.0, -4.5, 0.0, 4.6, 0.0],
                [0.0, 4.5, 0.0, 0.0, 0.0, 4.6],
            ],
            dtype=np.float64,
        )
        self.M = self.MRB + self.MA
        self.M_inv = np.linalg.inv(self.M)

        # -------------------------------------------------
        # Residual drag + maneuvering + hull-lift model
        # -------------------------------------------------
        # Residual axial/rotational damping kept modest because cross-flow and
        # hull-lift terms now carry most of the sway/heave/yaw/pitch physics.
        self.D_res_lin = np.diag([9.0, 3.0, 3.0, 0.08, 0.8, 0.8])
        self.D_res_quad = np.array([14.0, 10.0, 10.0, 0.10, 1.0, 1.0], dtype=np.float64)

        self.hull_ref_area = self.auv_length * self.auv_diameter
        self.hull_cd0 = 0.20
        self.hull_cd_alpha2 = 2.4
        self.hull_cd_beta2 = 2.0

        self.hull_cz_alpha = 1.15
        self.hull_cm_alpha = 0.32
        self.hull_cz_q = 0.82
        self.hull_cm_q = 0.48

        self.hull_cy_beta = 1.05
        self.hull_cn_beta = 0.30
        self.hull_cy_r = 0.72
        self.hull_cn_r = 0.42

        self.crossflow_cd = 1.10
        self.crossflow_sections = 17

        # -------------------------------------------------
        # Actuator limits / dynamics
        # -------------------------------------------------
        self.max_rudder = np.deg2rad(25.0)
        self.max_stern_plane = np.deg2rad(25.0)
        self.max_reverse_propeller = 0.25

        self.max_propeller_rps = 35.0
        self.min_propeller_rps = -6.0
        self.propeller_time_constant = 0.45
        self.propeller_rps_rate_limit = 35.0
        self.propeller_deadband = 0.03

        self.rudder_time_constant = 0.18
        self.stern_time_constant = 0.18
        self.rudder_rate_limit = np.deg2rad(45.0)
        self.stern_rate_limit = np.deg2rad(45.0)
        self.control_surface_deadband = np.deg2rad(0.4)

        # Propeller/open-water model
        self.propeller_diameter = 0.18
        self.propeller_wake_fraction = 0.18
        self.propeller_thrust_deduction = 0.06
        self.kt0 = 0.20
        self.kt1 = 0.085
        self.kq0 = 0.028
        self.kq1 = 0.014
        self.propeller_reverse_efficiency = 0.55
        self.max_thrust = 320.0
        # Net reaction torque after motor/stator/hull cancellation. Without this
        # attenuation the small roll angle leaks into pitch/depth too strongly.
        self.propeller_reaction_torque_scale = 0.25
        self.max_propeller_torque = 0.12

        # Control-surface model (dynamic-pressure / lift based)
        self.fin_cl_alpha = 3.0
        self.fin_cd0 = 0.05
        self.fin_cd2 = 1.6
        self.fin_stall_angle = np.deg2rad(18.0)
        self.rudder_area = 0.018
        self.stern_area = 0.020
        self.rudder_arm = 0.75
        self.stern_arm = 0.75
        self.slipstream_gain = 0.55

        # Ocean current profile (sampled per episode, slowly varying)
        self.current_speed_min = 0.05
        self.current_speed_max = 0.35
        self.current_vertical_max = 0.03
        self.current_osc_amp_max = np.array([0.08, 0.08, 0.015], dtype=np.float64)
        self.current_base_inertial = np.zeros(3, dtype=np.float64)
        self.current_osc_amp = np.zeros(3, dtype=np.float64)
        self.current_osc_omega = np.zeros(3, dtype=np.float64)
        self.current_phase = np.zeros(3, dtype=np.float64)
        self.current_inertial = np.zeros(3, dtype=np.float64)

        # Obstacle sampling profile. Later curriculum stages can tighten these
        # around the mission path to create a meaningful moving-obstacle test.
        self.obstacle_radius_min = 0.8
        self.obstacle_radius_max = 2.5
        self.obstacle_speed_min = 0.05
        self.obstacle_speed_max = 0.20
        self.prefer_path_obstacles = False
        self.obstacle_path_lateral_jitter = 12.0
        self.obstacle_path_depth_jitter = 8.0
        self.obstacle_path_fraction_min = 0.15
        self.obstacle_path_fraction_max = 0.85

        # Numerics / safety
        self.velocity_clip = np.array([3.6, 3.0, 3.0, 1.2, 1.2, 1.4], dtype=np.float64)
        self.theta_limit = np.deg2rad(50.0)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        obs_dim = 6 + 6 + 3 + 3 + 12
        if self.include_current_in_obs:
            obs_dim += 3

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.state: Optional[np.ndarray] = None
        self.start: Optional[np.ndarray] = None
        self.target: Optional[np.ndarray] = None
        self.obstacles: List[Obstacle] = []
        self.step_count = 0
        # [propeller_rps, rudder_angle, stern_angle]
        self.actuator_state = np.zeros(3, dtype=np.float64)
        self.last_action = np.zeros(3, dtype=np.float64)

    # -------------------------------------------------
    # Sampling utilities
    # -------------------------------------------------
    def _sample_point(self, z_low: float = 5.0, z_high: float = 25.0) -> np.ndarray:
        return np.array(
            [
                self.rng.uniform(-self.world_size, self.world_size),
                self.rng.uniform(-self.world_size, self.world_size),
                self.rng.uniform(z_low, z_high),
            ],
            dtype=np.float64,
        )

    def _fixed_mid_obstacle(self) -> Obstacle:
        if self.fixed_obstacle_center is not None:
            center = self.fixed_obstacle_center.astype(np.float64).copy()
        else:
            start = self.start if self.start is not None else self.fixed_start
            target = self.target if self.target is not None else np.array(
                [0.5 * self.world_size, 0.5 * self.world_size, self.target_z_max],
                dtype=np.float64,
            )
            center = 0.5 * (start + target)
        center[2] = float(np.clip(center[2], 4.0, self.world_size - 4.0))
        return Obstacle(
            center=center,
            velocity=np.zeros(3, dtype=np.float64),
            radius=float(self.fixed_obstacle_radius),
        )

    def _target_xy_limit(self) -> float:
        margin = max(float(getattr(self, "target_boundary_margin", 0.0)), 0.0)
        margin = min(margin, max(self.world_size - 1.0, 0.0))
        return max(self.world_size - margin, 1.0)

    def _target_has_boundary_margin(self, target: np.ndarray) -> bool:
        return bool(np.all(np.abs(target[:2]) <= self._target_xy_limit()))

    def _target_xy_margin(self, target: np.ndarray) -> float:
        return float(self.world_size - np.max(np.abs(target[:2])))

    def _sample_target(self) -> np.ndarray:
        fallback_target: Optional[np.ndarray] = None
        for _ in range(1000):
            target = np.array(
                [
                    self.rng.uniform(-self.world_size, self.world_size),
                    self.rng.uniform(-self.world_size, self.world_size),
                    self.rng.uniform(self.target_z_min, self.target_z_max),
                ],
                dtype=np.float64,
            )
            distance = float(np.linalg.norm(target - self.fixed_start))
            if not self._target_has_boundary_margin(target):
                continue
            if distance < self.min_start_target_distance:
                continue
            if self.max_start_target_distance is not None and distance > self.max_start_target_distance:
                continue
            if target[0] <= self.fixed_start[0] + 2.0:
                continue
            return target

        for _ in range(1000):
            distance_low = max(float(self.min_start_target_distance), 1.0)
            distance_high = float(self.max_start_target_distance or min(self.world_size * 1.5, distance_low + 40.0))
            distance_high = max(distance_high, distance_low + 1.0)
            distance = self.rng.uniform(distance_low, distance_high)
            yaw = self.rng.uniform(-0.45 * np.pi, 0.45 * np.pi)
            horizontal = max(distance * self.rng.uniform(0.75, 1.0), 1.0)
            target = np.array(
                [
                    self.fixed_start[0] + horizontal * np.cos(yaw),
                    self.fixed_start[1] + horizontal * np.sin(yaw),
                    self.rng.uniform(self.target_z_min, self.target_z_max),
                ],
                dtype=np.float64,
            )
            target[:2] = np.clip(target[:2], -self._target_xy_limit(), self._target_xy_limit())
            target[2] = float(np.clip(target[2], 0.0, self.world_size))
            fallback_target = target.copy()

            distance = float(np.linalg.norm(target - self.fixed_start))
            if distance < self.min_start_target_distance:
                continue
            if self.max_start_target_distance is not None and distance > self.max_start_target_distance:
                continue
            if target[0] <= self.fixed_start[0] + 2.0:
                continue
            return target

        # Fallback: keep the target inside the configured boundary margin even if
        # the distance constraints are unusually tight for a custom workspace.
        if fallback_target is not None:
            return fallback_target
        return np.array(
            [
                min(self.fixed_start[0] + self.min_start_target_distance, self._target_xy_limit()),
                float(np.clip(self.fixed_start[1], -self._target_xy_limit(), self._target_xy_limit())),
                float(np.clip(0.5 * (self.target_z_min + self.target_z_max), 0.0, self.world_size)),
            ],
            dtype=np.float64,
        )

    def _sample_path_obstacle_center(self) -> np.ndarray:
        mission_vec = self.target - self.start
        mission_len = float(np.linalg.norm(mission_vec))
        if mission_len < 1e-8:
            return self._sample_point(z_low=10.0, z_high=60.0)

        direction = mission_vec / mission_len
        frac = self.rng.uniform(self.obstacle_path_fraction_min, self.obstacle_path_fraction_max)
        center = self.start + frac * mission_vec

        lateral = self.rng.normal(size=3)
        lateral[2] = 0.0
        lateral = lateral - np.dot(lateral, direction) * direction
        lateral_norm = float(np.linalg.norm(lateral))
        if lateral_norm < 1e-8:
            lateral = np.array([-direction[1], direction[0], 0.0], dtype=np.float64)
            lateral_norm = float(np.linalg.norm(lateral))
        lateral = lateral / max(lateral_norm, 1e-8)

        center = center + lateral * self.rng.uniform(-self.obstacle_path_lateral_jitter, self.obstacle_path_lateral_jitter)
        center[2] += self.rng.uniform(-self.obstacle_path_depth_jitter, self.obstacle_path_depth_jitter)
        center[:2] = np.clip(center[:2], -self.world_size, self.world_size)
        center[2] = float(np.clip(center[2], 4.0, self.world_size - 4.0))
        return center

    def _generate_obstacles(self) -> List[Obstacle]:
        obstacles: List[Obstacle] = []
        if self.fixed_mid_obstacle:
            obstacles.append(self._fixed_mid_obstacle())
        for _ in range(self.n_obstacles):
            for _ in range(1000):
                center = self._sample_path_obstacle_center() if self.prefer_path_obstacles else self._sample_point(z_low=10.0, z_high=60.0)
                radius = self.rng.uniform(self.obstacle_radius_min, self.obstacle_radius_max)

                if np.linalg.norm(center - self.target) < (radius + self.goal_radius + 1.5):
                    continue
                if np.linalg.norm(center - self.start) < (radius + self.auv_length + 1.5):
                    continue

                valid = True
                for obs in obstacles:
                    if np.linalg.norm(center - obs.center) < (radius + obs.radius + 1.0):
                        valid = False
                        break
                if not valid:
                    continue

                direction = self.rng.normal(size=3)
                direction /= np.linalg.norm(direction) + 1e-8
                speed = self.rng.uniform(self.obstacle_speed_min, self.obstacle_speed_max)
                obstacles.append(Obstacle(center=center, velocity=direction * speed, radius=radius))
                break
        return obstacles

    def _obstacle_arrays(self) -> Dict[str, np.ndarray]:
        if not self.obstacles:
            return {
                "obstacle_centers": np.zeros((0, 3), dtype=np.float64),
                "obstacle_velocities": np.zeros((0, 3), dtype=np.float64),
                "obstacle_radii": np.zeros((0,), dtype=np.float64),
            }
        return {
            "obstacle_centers": np.stack([obs.center.copy() for obs in self.obstacles], axis=0),
            "obstacle_velocities": np.stack([obs.velocity.copy() for obs in self.obstacles], axis=0),
            "obstacle_radii": np.asarray([obs.radius for obs in self.obstacles], dtype=np.float64),
        }

    def _sample_current_profile(self) -> None:
        if not self.current_enabled:
            self.current_base_inertial[:] = 0.0
            self.current_osc_amp[:] = 0.0
            self.current_osc_omega[:] = 0.0
            self.current_phase[:] = 0.0
            self.current_inertial[:] = 0.0
            return

        heading = self.rng.uniform(-np.pi, np.pi)
        speed = self.rng.uniform(self.current_speed_min, self.current_speed_max)
        vertical = self.rng.uniform(-self.current_vertical_max, self.current_vertical_max)

        self.current_base_inertial = np.array(
            [speed * np.cos(heading), speed * np.sin(heading), vertical],
            dtype=np.float64,
        )
        self.current_osc_amp = self.rng.uniform(-self.current_osc_amp_max, self.current_osc_amp_max)
        self.current_osc_omega = self.rng.uniform(0.05, 0.20, size=3)
        self.current_phase = self.rng.uniform(-np.pi, np.pi, size=3)
        self._update_current()

    # -------------------------------------------------
    # Dynamics helpers
    # -------------------------------------------------
    @staticmethod
    def _coriolis_from_mass_matrix(M: np.ndarray, nu: np.ndarray) -> np.ndarray:
        nu1 = nu[:3]
        nu2 = nu[3:]
        M11 = M[:3, :3]
        M12 = M[:3, 3:]
        M21 = M[3:, :3]
        M22 = M[3:, 3:]

        a = M11 @ nu1 + M12 @ nu2
        b = M21 @ nu1 + M22 @ nu2

        C = np.zeros((6, 6), dtype=np.float64)
        C[:3, 3:] = -skew(a)
        C[3:, :3] = -skew(a)
        C[3:, 3:] = -skew(b)
        return C

    def _residual_damping_tau(self, nu_r: np.ndarray) -> np.ndarray:
        return -(self.D_res_lin @ nu_r + self.D_res_quad * np.abs(nu_r) * nu_r)

    def _flow_angles(self, nu_r: np.ndarray) -> tuple[float, float, float, float, float]:
        u_r, v_r, w_r, _, q, r = nu_r
        U = float(np.linalg.norm(nu_r[:3]))
        u_ref = max(abs(u_r), 0.25)
        alpha = np.arctan2(w_r, u_ref)
        beta = np.arctan2(v_r, u_ref)
        q_hat = q * self.auv_length / max(2.0 * U, 0.20)
        r_hat = r * self.auv_length / max(2.0 * U, 0.20)
        return U, alpha, beta, q_hat, r_hat

    def _hull_lift_tau(self, nu_r: np.ndarray) -> np.ndarray:
        U, alpha, beta, q_hat, r_hat = self._flow_angles(nu_r)
        if U < 0.05:
            return np.zeros(6, dtype=np.float64)

        q_dyn = 0.5 * self.rho * (U ** 2)
        tau = np.zeros(6, dtype=np.float64)

        x_drag = -q_dyn * self.hull_ref_area * (
            self.hull_cd0 + self.hull_cd_alpha2 * (alpha ** 2) + self.hull_cd_beta2 * (beta ** 2)
        )
        z_force = -q_dyn * self.hull_ref_area * (self.hull_cz_alpha * alpha + self.hull_cz_q * q_hat)
        m_moment = -q_dyn * self.hull_ref_area * self.auv_length * (
            self.hull_cm_alpha * alpha + self.hull_cm_q * q_hat
        )
        y_force = -q_dyn * self.hull_ref_area * (self.hull_cy_beta * beta + self.hull_cy_r * r_hat)
        n_moment = -q_dyn * self.hull_ref_area * self.auv_length * (
            self.hull_cn_beta * beta + self.hull_cn_r * r_hat
        )

        tau[0] = x_drag
        tau[1] = y_force
        tau[2] = z_force
        tau[4] = m_moment
        tau[5] = n_moment
        return tau

    def _crossflow_tau(self, nu_r: np.ndarray) -> np.ndarray:
        _, v_r, w_r, _, q, r = nu_r
        tau = np.zeros(6, dtype=np.float64)
        dx = self.auv_length / self.crossflow_sections
        x_positions = np.linspace(-0.5 * self.auv_length + 0.5 * dx, 0.5 * self.auv_length - 0.5 * dx, self.crossflow_sections)

        for x in x_positions:
            v_local = v_r + x * r
            w_local = w_r - x * q

            dY = -0.5 * self.rho * self.crossflow_cd * self.auv_diameter * dx * abs(v_local) * v_local
            dZ = -0.5 * self.rho * self.crossflow_cd * self.auv_diameter * dx * abs(w_local) * w_local

            tau[1] += dY
            tau[2] += dZ
            tau[4] += -x * dZ
            tau[5] += x * dY

        return tau

    def _hydrodynamic_tau(self, nu_r: np.ndarray) -> np.ndarray:
        return self._residual_damping_tau(nu_r) + self._hull_lift_tau(nu_r) + self._crossflow_tau(nu_r)

    def _restoring_force(self, eta: np.ndarray) -> np.ndarray:
        _, _, _, phi, theta, _ = eta

        weight = self.m * self.g
        buoyancy = self.rho * self.volume * self.g

        g_eta = np.zeros(6, dtype=np.float64)
        g_eta[0] = (weight - buoyancy) * np.sin(theta)
        g_eta[1] = -(weight - buoyancy) * np.cos(theta) * np.sin(phi)
        g_eta[2] = -(weight - buoyancy) * np.cos(theta) * np.cos(phi)

        xg, yg, zg = self.r_g
        xb, yb, zb = self.r_b

        g_eta[3] = -(yg * weight - yb * buoyancy) * np.cos(theta) * np.cos(phi) + (zg * weight - zb * buoyancy) * np.cos(theta) * np.sin(phi)
        g_eta[4] = (zg * weight - zb * buoyancy) * np.sin(theta) + (xg * weight - xb * buoyancy) * np.cos(theta) * np.cos(phi)
        g_eta[5] = -(xg * weight - xb * buoyancy) * np.cos(theta) * np.sin(phi) - (yg * weight - yb * buoyancy) * np.sin(theta)
        return g_eta

    def _update_current(self) -> None:
        if not self.current_enabled:
            self.current_inertial[:] = 0.0
            return

        t = self.step_count * self.dt
        oscillation = self.current_osc_amp * np.sin(self.current_osc_omega * t + self.current_phase)
        self.current_inertial = self.current_base_inertial + oscillation
        self.current_inertial[2] = np.clip(
            self.current_inertial[2],
            -self.current_vertical_max - abs(self.current_osc_amp_max[2]),
            self.current_vertical_max + abs(self.current_osc_amp_max[2]),
        )

    def _current_body(self, eta: np.ndarray) -> np.ndarray:
        rot = rotation_matrix_body_to_inertial(eta[3], eta[4], eta[5])
        return rot.T @ self.current_inertial

    def _relative_velocity(self, eta: np.ndarray, nu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        current_body = self._current_body(eta)
        nu_r = nu.copy()
        nu_r[:3] -= current_body
        return nu_r, current_body

    def _map_propeller_command_to_rps(self, cmd: float) -> float:
        if abs(cmd) < self.propeller_deadband:
            return 0.0
        if cmd >= 0.0:
            return cmd * self.max_propeller_rps
        return cmd * abs(self.min_propeller_rps)

    def _apply_control_surface_deadband(self, desired_angle: float) -> float:
        if abs(desired_angle) < self.control_surface_deadband:
            return 0.0
        return desired_angle

    def _update_actuators(self, action: np.ndarray) -> None:
        desired = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        desired_propeller_rps = self._map_propeller_command_to_rps(float(np.clip(desired[0], -self.max_reverse_propeller, 1.0)))
        desired_rudder = self._apply_control_surface_deadband(desired[1] * self.max_rudder)
        desired_stern = self._apply_control_surface_deadband(desired[2] * self.max_stern_plane)

        self.actuator_state[0] = np.clip(
            first_order_rate_limited(
                self.actuator_state[0], desired_propeller_rps, self.propeller_time_constant, self.propeller_rps_rate_limit, self.dt
            ),
            self.min_propeller_rps,
            self.max_propeller_rps,
        )
        self.actuator_state[1] = np.clip(
            first_order_rate_limited(
                self.actuator_state[1], desired_rudder, self.rudder_time_constant, self.rudder_rate_limit, self.dt
            ),
            -self.max_rudder,
            self.max_rudder,
        )
        self.actuator_state[2] = np.clip(
            first_order_rate_limited(
                self.actuator_state[2], desired_stern, self.stern_time_constant, self.stern_rate_limit, self.dt
            ),
            -self.max_stern_plane,
            self.max_stern_plane,
        )

    def _lift_coefficient(self, effective_alpha: float) -> float:
        alpha = np.clip(effective_alpha, -self.fin_stall_angle, self.fin_stall_angle)
        return self.fin_cl_alpha * alpha

    def _drag_coefficient(self, effective_alpha: float) -> float:
        alpha = np.clip(effective_alpha, -self.fin_stall_angle, self.fin_stall_angle)
        return self.fin_cd0 + self.fin_cd2 * (alpha ** 2)

    def _propeller_tau(self, nu_r: np.ndarray) -> np.ndarray:
        n = float(self.actuator_state[0])
        if abs(n) < 1e-5:
            return np.zeros(6, dtype=np.float64)

        u_r = float(nu_r[0])
        sign_n = np.sign(n)
        n_abs = abs(n)
        wake_velocity = (1.0 - self.propeller_wake_fraction) * u_r
        advance = wake_velocity / max(n_abs * self.propeller_diameter, 1e-4)

        kt = max(self.kt0 - self.kt1 * advance, 0.02)
        kq = max(self.kq0 - self.kq1 * advance, 0.005)

        thrust = self.rho * (self.propeller_diameter ** 4) * kt * n_abs * n
        torque = self.rho * (self.propeller_diameter ** 5) * kq * n_abs * sign_n

        if n < 0.0:
            thrust *= self.propeller_reverse_efficiency
            torque *= self.propeller_reverse_efficiency

        thrust *= (1.0 - self.propeller_thrust_deduction)
        torque *= self.propeller_reaction_torque_scale
        thrust = float(np.clip(thrust, -self.max_thrust * self.max_reverse_propeller, self.max_thrust))
        torque = float(np.clip(torque, -self.max_propeller_torque, self.max_propeller_torque))

        tau = np.zeros(6, dtype=np.float64)
        tau[0] = thrust
        tau[3] = -torque
        return tau

    def _control_surface_tau(self, nu_r: np.ndarray) -> np.ndarray:
        _, rudder_angle, stern_angle = self.actuator_state
        u_r, v_r, w_r, _, _, _ = nu_r

        slipstream_speed = self.slipstream_gain * abs(self.actuator_state[0]) * self.propeller_diameter

        u_eff_rudder = np.sign(u_r if abs(u_r) > 1e-5 else 1.0) * np.sqrt(u_r ** 2 + slipstream_speed ** 2)
        beta = np.arctan2(v_r, max(abs(u_eff_rudder), 1e-4))
        alpha_rudder = rudder_angle - beta
        q_lat = 0.5 * self.rho * (u_eff_rudder ** 2 + v_r ** 2)
        cl_r = self._lift_coefficient(alpha_rudder)
        cd_r = self._drag_coefficient(alpha_rudder)
        y_rudder = q_lat * self.rudder_area * cl_r
        x_rudder_drag = -q_lat * self.rudder_area * cd_r
        n_rudder = self.rudder_arm * y_rudder

        u_eff_stern = np.sign(u_r if abs(u_r) > 1e-5 else 1.0) * np.sqrt(u_r ** 2 + slipstream_speed ** 2)
        gamma = np.arctan2(w_r, max(abs(u_eff_stern), 1e-4))
        alpha_stern = stern_angle + gamma
        q_vert = 0.5 * self.rho * (u_eff_stern ** 2 + w_r ** 2)
        cl_s = self._lift_coefficient(alpha_stern)
        cd_s = self._drag_coefficient(alpha_stern)
        z_stern = -q_vert * self.stern_area * cl_s
        x_stern_drag = -q_vert * self.stern_area * cd_s
        m_stern = self.stern_arm * z_stern

        tau = np.zeros(6, dtype=np.float64)
        tau[0] = x_rudder_drag + x_stern_drag
        tau[1] = y_rudder
        tau[2] = z_stern
        tau[4] = m_stern
        tau[5] = n_rudder
        return tau

    def _actuation_tau(self, nu_r: np.ndarray) -> np.ndarray:
        return self._propeller_tau(nu_r) + self._control_surface_tau(nu_r)

    def _kinematics(self, eta: np.ndarray, nu: np.ndarray) -> np.ndarray:
        phi, theta, psi = eta[3], eta[4], eta[5]
        u, v, w, p, q, r = nu
        rot = rotation_matrix_body_to_inertial(phi, theta, psi)
        t_mat = euler_rate_matrix(phi, theta)
        pos_dot = rot @ np.array([u, v, w], dtype=np.float64)
        ang_dot = t_mat @ np.array([p, q, r], dtype=np.float64)
        return np.concatenate([pos_dot, ang_dot])

    def _integrate(self, eta: np.ndarray, nu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nu_r, _ = self._relative_velocity(eta, nu)

        c_rb = self._coriolis_from_mass_matrix(self.MRB, nu)
        c_a = self._coriolis_from_mass_matrix(self.MA, nu_r)
        restoring = self._restoring_force(eta)
        tau_hydro = self._hydrodynamic_tau(nu_r)
        tau_act = self._actuation_tau(nu_r)

        rhs = tau_act + tau_hydro - (c_rb @ nu) - (c_a @ nu_r) - restoring
        nu_dot = self.M_inv @ rhs
        nu_next = nu + self.dt * nu_dot
        nu_next = np.clip(nu_next, -self.velocity_clip, self.velocity_clip)

        eta_dot = self._kinematics(eta, nu_next)
        eta_next = eta + self.dt * eta_dot
        eta_next[3] = wrap_angle(eta_next[3])
        eta_next[4] = np.clip(wrap_angle(eta_next[4]), -self.theta_limit, self.theta_limit)
        eta_next[5] = wrap_angle(eta_next[5])
        return eta_next, nu_next

    # -------------------------------------------------
    # Obstacles
    # -------------------------------------------------
    def _update_obstacles(self) -> None:
        for obs in self.obstacles:
            obs.center = obs.center + obs.velocity * self.dt

            for axis in range(3):
                if obs.center[axis] < -self.world_size:
                    obs.center[axis] = -self.world_size
                    obs.velocity[axis] *= -1.0
                elif obs.center[axis] > self.world_size:
                    obs.center[axis] = self.world_size
                    obs.velocity[axis] *= -1.0

            if obs.center[2] < 0.0:
                obs.center[2] = 0.0
                obs.velocity[2] *= -1.0

            vec_to_target = obs.center - self.target
            dist_to_target = np.linalg.norm(vec_to_target)
            min_dist_to_target = obs.radius + self.goal_radius + 0.5
            if dist_to_target < min_dist_to_target:
                normal = vec_to_target / (dist_to_target + 1e-8)
                obs.center = self.target + normal * min_dist_to_target
                obs.velocity = obs.velocity - 2.0 * np.dot(obs.velocity, normal) * normal

        for i in range(len(self.obstacles)):
            for j in range(i + 1, len(self.obstacles)):
                oi = self.obstacles[i]
                oj = self.obstacles[j]
                d_vec = oi.center - oj.center
                dist = np.linalg.norm(d_vec)
                min_dist = oi.radius + oj.radius
                if dist < min_dist:
                    normal = d_vec / (dist + 1e-8)
                    overlap = min_dist - dist
                    oi.center += 0.5 * overlap * normal
                    oj.center -= 0.5 * overlap * normal

                    rand_i = self.rng.normal(size=3)
                    rand_i /= np.linalg.norm(rand_i) + 1e-8
                    rand_j = self.rng.normal(size=3)
                    rand_j /= np.linalg.norm(rand_j) + 1e-8

                    speed_i = np.linalg.norm(oi.velocity)
                    speed_j = np.linalg.norm(oj.velocity)
                    oi.velocity = rand_i * max(speed_i, 0.05)
                    oj.velocity = rand_j * max(speed_j, 0.05)

    # -------------------------------------------------
    # Observation / reward
    # -------------------------------------------------
    def _normalized_actuator_state(self) -> np.ndarray:
        prop_norm = self.actuator_state[0] / (self.max_propeller_rps if self.actuator_state[0] >= 0.0 else abs(self.min_propeller_rps))
        return np.array(
            [
                np.clip(prop_norm, -1.0, 1.0),
                self.actuator_state[1] / self.max_rudder,
                self.actuator_state[2] / self.max_stern_plane,
            ],
            dtype=np.float64,
        )

    def _get_obs(self) -> np.ndarray:
        eta = self.state[:6]
        nu = self.state[6:]
        pos = eta[:3]
        rot = rotation_matrix_body_to_inertial(eta[3], eta[4], eta[5])

        rel_target_body = rot.T @ (self.target - pos)
        current_body = self._current_body(eta)

        dists = []
        for obs in self.obstacles:
            d = np.linalg.norm(obs.center - pos)
            dists.append((d, obs))
        dists.sort(key=lambda item: item[0])

        obs_feat = []
        for _, obs in dists[:3]:
            rel_body = rot.T @ (obs.center - pos)
            obs_feat.extend(rel_body.tolist())
            obs_feat.append(obs.radius)

        while len(obs_feat) < 12:
            obs_feat.append(0.0)

        parts = [eta, nu, rel_target_body]
        if self.include_current_in_obs:
            parts.append(current_body)
        parts.append(self._normalized_actuator_state())
        parts.append(np.array(obs_feat, dtype=np.float64))
        return np.concatenate(parts).astype(np.float32)

    def _check_collision(self, pos: np.ndarray) -> bool:
        for obs in self.obstacles:
            if np.linalg.norm(pos - obs.center) <= (self.auv_radius + obs.radius):
                return True
        return False

    def _out_of_bounds(self, pos: np.ndarray) -> bool:
        return np.any(np.abs(pos[:2]) > self.world_size) or pos[2] < 0.0 or pos[2] > self.world_size

    def _workspace_margin(self, pos: np.ndarray) -> float:
        xy_margin = self.world_size - float(np.max(np.abs(pos[:2])))
        depth_margin = min(float(pos[2]), self.world_size - float(pos[2]))
        return float(min(xy_margin, depth_margin))

    def _progress_metrics(self, prev_pos: np.ndarray, pos: np.ndarray) -> Dict[str, float]:
        dist_prev = float(np.linalg.norm(prev_pos - self.target))
        dist_now = float(np.linalg.norm(pos - self.target))
        return {
            "distance_to_goal": dist_now,
            "progress": dist_prev - dist_now,
        }

    def _nearest_obstacle_metrics(self, pos: np.ndarray) -> Dict[str, float]:
        if not self.obstacles:
            return {
                "nearest_obstacle_distance": float("nan"),
                "nearest_obstacle_radius": float("nan"),
                "min_obstacle_clearance": float("nan"),
            }

        nearest_distance = float("inf")
        nearest_radius = 0.0
        min_clearance = float("inf")
        for obs in self.obstacles:
            center_distance = float(np.linalg.norm(pos - obs.center))
            clearance = center_distance - (self.auv_radius + obs.radius)
            if center_distance < nearest_distance:
                nearest_distance = center_distance
                nearest_radius = float(obs.radius)
            min_clearance = min(min_clearance, clearance)

        return {
            "nearest_obstacle_distance": nearest_distance,
            "nearest_obstacle_radius": nearest_radius,
            "min_obstacle_clearance": float(min_clearance),
        }

    def _line_tracking_metrics(self, pos: np.ndarray) -> Dict[str, float]:
        mission_vec = self.target - self.start
        mission_len = float(np.linalg.norm(mission_vec))
        if mission_len < 1e-8:
            return {
                "cross_track_error": 0.0,
                "path_progress_fraction": 0.0,
            }

        rel = pos - self.start
        progress_unclipped = float(np.dot(rel, mission_vec) / (mission_len ** 2))
        progress_fraction = float(np.clip(progress_unclipped, 0.0, 1.0))
        closest = self.start + progress_fraction * mission_vec
        return {
            "cross_track_error": float(np.linalg.norm(pos - closest)),
            "path_progress_fraction": progress_fraction,
        }

    def _heading_metrics(self, eta: np.ndarray, pos: np.ndarray) -> Dict[str, float]:
        target_vec = self.target - pos
        horizontal_dist = float(np.linalg.norm(target_vec[:2]))
        desired_yaw = float(np.arctan2(target_vec[1], target_vec[0]))
        desired_pitch = float(np.arctan2(target_vec[2], max(horizontal_dist, 1e-8)))
        rot = rotation_matrix_body_to_inertial(eta[3], eta[4], eta[5])
        body_x = rot[:, 0]
        target_norm = float(np.linalg.norm(target_vec))
        alignment = 0.0
        if target_norm > 1e-8:
            alignment = float(np.dot(body_x, target_vec) / target_norm)

        return {
            "target_bearing": desired_yaw,
            "heading_error": float(wrap_angle(desired_yaw - eta[5])),
            "target_elevation": desired_pitch,
            "elevation_error": float(desired_pitch - eta[4]),
            "target_alignment": alignment,
        }

    # -------------------------------------------------
    # Gym API
    # -------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.start = self.fixed_start.copy()
        self.target = self._sample_target()

        eta = np.zeros(6, dtype=np.float64)
        eta[:3] = self.start
        eta[3] = self.rng.uniform(-0.03, 0.03)
        eta[4] = self.rng.uniform(-0.05, 0.05)
        yaw_to_target = np.arctan2(self.target[1] - self.start[1], self.target[0] - self.start[0])
        eta[5] = yaw_to_target

        nu = np.zeros(6, dtype=np.float64)
        self.state = np.concatenate([eta, nu])

        self.step_count = 0
        self.actuator_state[:] = 0.0
        self.last_action[:] = 0.0
        self._sample_current_profile()
        self.obstacles = self._generate_obstacles()

        return self._get_obs(), {
            "start": self.start.copy(),
            "target": self.target.copy(),
            "mission_distance": float(np.linalg.norm(self.target - self.start)),
            "target_boundary_margin": float(self.target_boundary_margin),
            "target_xy_margin": self._target_xy_margin(self.target),
            "n_obstacles": int(len(self.obstacles)),
            "seed": seed if seed is not None else "",
            "current_inertial": self.current_inertial.copy(),
            **self._obstacle_arrays(),
            "fixed_mid_obstacle": self.fixed_mid_obstacle,
            "fixed_obstacle_radius": self.fixed_obstacle_radius if self.fixed_mid_obstacle else 0.0,
            "fixed_obstacle_center": self.obstacles[0].center.copy() if self.fixed_mid_obstacle and len(self.obstacles) > 0 else None,
        }

    def step(self, action: np.ndarray):
        self.step_count += 1
        self._update_current()

        eta = self.state[:6].copy()
        nu = self.state[6:].copy()
        prev_pos = eta[:3].copy()

        clipped_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._update_actuators(clipped_action)
        _, current_body = self._relative_velocity(eta, nu)
        eta_next, nu_next = self._integrate(eta, nu)

        self.state = np.concatenate([eta_next, nu_next])
        self._update_obstacles()

        pos = eta_next[:3]
        metrics = self._progress_metrics(prev_pos, pos)
        step_distance = float(np.linalg.norm(pos - prev_pos))
        obstacle_metrics = self._nearest_obstacle_metrics(pos)
        line_metrics = self._line_tracking_metrics(pos)
        heading_metrics = self._heading_metrics(eta_next, pos)

        terminated = False
        truncated = False
        info: Dict[str, Any] = {
            **metrics,
            **obstacle_metrics,
            **line_metrics,
            **heading_metrics,
            "step": int(self.step_count),
            "sim_time": float(self.step_count * self.dt),
            "step_distance": step_distance,
            "position": pos.copy(),
            "workspace_margin": self._workspace_margin(pos),
            "velocity_body": nu_next.copy(),
            "speed": float(np.linalg.norm(nu_next[:3])),
            "surge_speed": float(nu_next[0]),
            "sway_speed": float(nu_next[1]),
            "heave_speed": float(nu_next[2]),
            "roll": float(eta_next[3]),
            "pitch": float(eta_next[4]),
            "yaw": float(eta_next[5]),
            "depth": float(pos[2]),
            "target_depth_error": float(self.target[2] - pos[2]),
            "n_obstacles": int(len(self.obstacles)),
            "current_inertial": self.current_inertial.copy(),
            "current_body": current_body.copy(),
            **self._obstacle_arrays(),
            "actuator_state": self._normalized_actuator_state().copy(),
            "propeller_rps": float(self.actuator_state[0]),
            "goal_reached": False,
            "collision": False,
            "out_of_bounds": False,
            "event": None,
        }

        if self._check_collision(pos):
            terminated = True
            info["collision"] = True
            info["event"] = "collision"

        if info["distance_to_goal"] <= self.goal_radius:
            terminated = True
            info["goal_reached"] = True
            info["event"] = "goal"

        if self._out_of_bounds(pos):
            terminated = True
            info["out_of_bounds"] = True
            info["event"] = "out_of_bounds"

        if self.step_count >= self.max_steps:
            truncated = True
            if info["event"] is None:
                info["event"] = "timeout"

        self.last_action = clipped_action.copy()
        return self._get_obs(), 0.0, terminated, truncated, info
