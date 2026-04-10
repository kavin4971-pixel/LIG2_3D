import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import gymnasium as gym
from gymnasium import spaces


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


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


@dataclass
class Obstacle:
    center: np.ndarray
    velocity: np.ndarray
    radius: float


class REMUSAUVEnv(gym.Env):
    """
    REMUS-style AUV environment for a fixed terminal-navigation mission.

    Mission setup in this version:
      - start = [-40, -40, 10] m
      - target = [40, 40, 50] m
      - straight-line distance = 120 m

    The vehicle geometry remains REMUS-like, while the workspace is sized for
    local terminal navigation with moving obstacles and current disturbance.
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
    ):
        super().__init__()

        self.dt = dt
        self.max_steps = max_steps
        self.world_size = world_size
        self.n_obstacles = n_obstacles
        self.current_enabled = current_enabled
        self.include_current_in_obs = include_current_in_obs
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
        self.r_b = np.array([0.0, 0.0, -0.01], dtype=np.float64)

        # Mass-like matrices (diagonal approximation)
        self.MRB_diag = np.array([self.m, self.m, self.m, self.Ix, self.Iy, self.Iz], dtype=np.float64)
        self.MA_diag = np.array([8.0, 45.0, 45.0, 0.05, 4.0, 4.0], dtype=np.float64)
        self.M_diag = self.MRB_diag + self.MA_diag
        self.M_inv = np.diag(1.0 / self.M_diag)

        # Relative-flow drag
        self.D_lin = np.diag([10.0, 55.0, 55.0, 0.25, 8.0, 8.0])
        self.D_quad = np.array([18.0, 95.0, 95.0, 0.35, 12.0, 12.0], dtype=np.float64)

        # The 3 m/s operating point needs stronger passive lateral / yaw damping
        # than the original low-thrust setup. Without this retuning, the vehicle
        # becomes directionally underdamped and can spin up even with zero rudder.
        self.D_lin[1] *= 6.0
        self.D_quad[1] *= 6.0
        self.D_lin[5] *= 10.0
        self.D_quad[5] *= 10.0

        # -------------------------------------------------
        # Actuator limits / dynamics
        # -------------------------------------------------
        # Chosen so that the nominal surge equilibrium is close to 3 m/s
        # after accounting for the hull drag model and the zero-lift drag of the
        # stern/rudder surfaces.
        self.max_thrust = 210.0                     # N (forward)
        self.max_reverse_propeller = 0.25          # reverse throttle saturation in normalized units
        self.max_rudder = np.deg2rad(25.0)         # rad
        self.max_stern_plane = np.deg2rad(25.0)    # rad

        self.propeller_rate_limit = 0.80           # normalized command per second
        self.rudder_rate_limit = np.deg2rad(45.0)  # rad/s
        self.stern_rate_limit = np.deg2rad(45.0)   # rad/s

        # Control-surface model (dynamic-pressure/lift based)
        self.fin_cl_alpha = 3.0                    # lift slope [1/rad]
        self.fin_cd0 = 0.05
        self.fin_cd2 = 1.6
        self.fin_stall_angle = np.deg2rad(18.0)
        self.rudder_area = 0.018                   # m^2
        self.stern_area = 0.020                    # m^2
        self.rudder_arm = 0.75                     # m behind CG
        self.stern_arm = 0.75                      # m behind CG

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

        # -------------------------------------------------
        # Vehicle geometry / task scales
        # -------------------------------------------------
        # Realistic REMUS-like hull geometry
        self.auv_length = 1.60                      # m
        self.auv_diameter = 0.19                    # m
        self.auv_hull_radius = 0.5 * self.auv_diameter

        # Spherical approximation used by the environment for collision checks.
        # Slightly inflated above the bare hull radius to account for the fact that
        # we are approximating a 1.6 m long body with a sphere.
        self.auv_radius = 0.25                      # m (effective collision radius)

        # Fixed terminal-navigation mission (120 m straight-line separation).
        self.goal_radius = 1.25                     # m
        self.fixed_start = np.array([-40.0, -40.0, 10.0], dtype=np.float64)
        self.fixed_target = np.array([40.0, 40.0, 50.0], dtype=np.float64)
        self.fixed_mission_distance = float(np.linalg.norm(self.fixed_target - self.fixed_start))

        # Numerics / safety
        self.velocity_clip = np.array([3.6, 3.0, 3.0, 1.4, 1.4, 1.8], dtype=np.float64)
        self.theta_limit = np.deg2rad(50.0)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        obs_dim = 6 + 6 + 3 + 3 + 12  # eta + nu + rel_target_body + actuator_state + nearest obstacles
        if self.include_current_in_obs:
            obs_dim += 3

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.state: Optional[np.ndarray] = None
        self.start: Optional[np.ndarray] = None
        self.target: Optional[np.ndarray] = None
        self.obstacles: List[Obstacle] = []
        self.step_count = 0
        self.actuator_state = np.zeros(3, dtype=np.float64)  # [propeller_norm, rudder_angle, stern_angle]
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


    def _generate_obstacles(self) -> List[Obstacle]:
        obstacles: List[Obstacle] = []
        for _ in range(self.n_obstacles):
            for _ in range(1000):
                center = self._sample_point(z_low=4.0, z_high=28.0)
                radius = self.rng.uniform(0.8, 2.5)

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
                speed = self.rng.uniform(0.05, 0.20)
                obstacles.append(Obstacle(center=center, velocity=direction * speed, radius=radius))
                break
        return obstacles

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
    def _coriolis_matrix_diag(m_diag: np.ndarray, nu: np.ndarray) -> np.ndarray:
        u, v, w, p, q, r = nu
        m1, m2, m3, i1, i2, i3 = m_diag
        return np.array(
            [
                [0.0, 0.0, 0.0, 0.0, m3 * w, -m2 * v],
                [0.0, 0.0, 0.0, -m3 * w, 0.0, m1 * u],
                [0.0, 0.0, 0.0, m2 * v, -m1 * u, 0.0],
                [0.0, m3 * w, -m2 * v, 0.0, i3 * r, -i2 * q],
                [-m3 * w, 0.0, m1 * u, -i3 * r, 0.0, i1 * p],
                [m2 * v, -m1 * u, 0.0, i2 * q, -i1 * p, 0.0],
            ],
            dtype=np.float64,
        )

    def _drag(self, nu_r: np.ndarray) -> np.ndarray:
        return self.D_lin @ nu_r + self.D_quad * np.abs(nu_r) * nu_r

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

    def _update_actuators(self, action: np.ndarray) -> None:
        desired = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        desired_propeller = np.clip(desired[0], -self.max_reverse_propeller, 1.0)
        desired_rudder = desired[1] * self.max_rudder
        desired_stern = desired[2] * self.max_stern_plane

        self.actuator_state[0] = np.clip(
            rate_limit(self.actuator_state[0], desired_propeller, self.propeller_rate_limit, self.dt),
            -self.max_reverse_propeller,
            1.0,
        )
        self.actuator_state[1] = np.clip(
            rate_limit(self.actuator_state[1], desired_rudder, self.rudder_rate_limit, self.dt),
            -self.max_rudder,
            self.max_rudder,
        )
        self.actuator_state[2] = np.clip(
            rate_limit(self.actuator_state[2], desired_stern, self.stern_rate_limit, self.dt),
            -self.max_stern_plane,
            self.max_stern_plane,
        )

    def _lift_coefficient(self, effective_alpha: float) -> float:
        alpha = np.clip(effective_alpha, -self.fin_stall_angle, self.fin_stall_angle)
        return self.fin_cl_alpha * alpha

    def _drag_coefficient(self, effective_alpha: float) -> float:
        alpha = np.clip(effective_alpha, -self.fin_stall_angle, self.fin_stall_angle)
        return self.fin_cd0 + self.fin_cd2 * (alpha ** 2)

    def _control_to_tau(self, nu_r: np.ndarray) -> np.ndarray:
        propeller_cmd, rudder_angle, stern_angle = self.actuator_state
        u_r, v_r, w_r, _, _, _ = nu_r

        x_prop = self.max_thrust * np.sign(propeller_cmd) * (propeller_cmd ** 2)

        beta = np.arctan2(v_r, max(abs(u_r), 1e-4))
        alpha_rudder = rudder_angle - beta
        q_lat = 0.5 * self.rho * (u_r ** 2 + v_r ** 2)
        cl_r = self._lift_coefficient(alpha_rudder)
        cd_r = self._drag_coefficient(alpha_rudder)
        y_rudder = q_lat * self.rudder_area * cl_r
        x_rudder_drag = -q_lat * self.rudder_area * cd_r
        n_rudder = self.rudder_arm * y_rudder

        gamma = np.arctan2(w_r, max(abs(u_r), 1e-4))
        alpha_stern = stern_angle + gamma
        q_vert = 0.5 * self.rho * (u_r ** 2 + w_r ** 2)
        cl_s = self._lift_coefficient(alpha_stern)
        cd_s = self._drag_coefficient(alpha_stern)
        z_stern = -q_vert * self.stern_area * cl_s
        x_stern_drag = -q_vert * self.stern_area * cd_s
        m_stern = self.stern_arm * z_stern

        tau = np.zeros(6, dtype=np.float64)
        tau[0] = x_prop + x_rudder_drag + x_stern_drag
        tau[1] = y_rudder
        tau[2] = z_stern
        tau[4] = m_stern
        tau[5] = n_rudder
        return tau

    def _kinematics(self, eta: np.ndarray, nu: np.ndarray) -> np.ndarray:
        phi, theta, psi = eta[3], eta[4], eta[5]
        u, v, w, p, q, r = nu
        rot = rotation_matrix_body_to_inertial(phi, theta, psi)
        t_mat = euler_rate_matrix(phi, theta)
        pos_dot = rot @ np.array([u, v, w], dtype=np.float64)
        ang_dot = t_mat @ np.array([p, q, r], dtype=np.float64)
        return np.concatenate([pos_dot, ang_dot])

    def _integrate(self, eta: np.ndarray, nu: np.ndarray, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nu_r, _ = self._relative_velocity(eta, nu)

        c_rb = self._coriolis_matrix_diag(self.MRB_diag, nu)
        c_a = self._coriolis_matrix_diag(self.MA_diag, nu_r)
        drag = self._drag(nu_r)
        restoring = self._restoring_force(eta)

        rhs = tau - (c_rb @ nu) - (c_a @ nu_r) - drag - restoring
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
        return np.array(
            [
                self.actuator_state[0],
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

    def _reward(self, prev_pos: np.ndarray, pos: np.ndarray, action: np.ndarray) -> float:
        dist_prev = np.linalg.norm(prev_pos - self.target)
        dist_now = np.linalg.norm(pos - self.target)
        progress = dist_prev - dist_now

        reward = 8.0 * progress
        reward -= 0.004

        actuator_norm = self._normalized_actuator_state()
        reward -= 0.015 * (actuator_norm[0] ** 2)
        reward -= 0.004 * (actuator_norm[1] ** 2 + actuator_norm[2] ** 2)
        reward -= 0.003 * np.sum((np.asarray(action) - self.last_action) ** 2)

        phi, theta = self.state[3], self.state[4]
        reward -= 0.03 * abs(phi)
        reward -= 0.08 * abs(theta)

        min_clearance = np.inf
        for obs in self.obstacles:
            clearance = np.linalg.norm(pos - obs.center) - (self.auv_radius + obs.radius)
            min_clearance = min(min_clearance, clearance)
        if min_clearance < 1.5:
            reward -= 0.4 * (1.5 - min_clearance)

        return float(reward)

    # -------------------------------------------------
    # Gym API
    # -------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.start = self.fixed_start.copy()
        self.target = self.fixed_target.copy()

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
            "mission_distance": self.fixed_mission_distance,
            "current_inertial": self.current_inertial.copy(),
        }

    def step(self, action: np.ndarray):
        self.step_count += 1
        self._update_current()

        eta = self.state[:6].copy()
        nu = self.state[6:].copy()
        prev_pos = eta[:3].copy()

        clipped_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._update_actuators(clipped_action)
        nu_r, current_body = self._relative_velocity(eta, nu)
        tau = self._control_to_tau(nu_r)
        eta_next, nu_next = self._integrate(eta, nu, tau)

        self.state = np.concatenate([eta_next, nu_next])
        self._update_obstacles()

        pos = eta_next[:3]
        reward = self._reward(prev_pos, pos, clipped_action)

        terminated = False
        truncated = False
        info: Dict[str, Any] = {
            "distance_to_goal": float(np.linalg.norm(pos - self.target)),
            "current_inertial": self.current_inertial.copy(),
            "current_body": current_body.copy(),
            "actuator_state": self._normalized_actuator_state().copy(),
        }

        if self._check_collision(pos):
            reward -= 120.0
            terminated = True
            info["event"] = "collision"

        if info["distance_to_goal"] <= self.goal_radius:
            reward += 180.0
            terminated = True
            info["event"] = "goal"

        if self._out_of_bounds(pos):
            reward -= 60.0
            terminated = True
            info["event"] = "out_of_bounds"

        if self.step_count >= self.max_steps:
            truncated = True
            info["event"] = "timeout"

        self.last_action = clipped_action.copy()
        return self._get_obs(), float(reward), terminated, truncated, info
