from __future__ import annotations

import argparse
import csv
import heapq
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from REMUSAUVEnv import (
    REMUSAUVEnv,
    Obstacle,
    rate_limit,
    rotation_matrix_body_to_inertial,
    wrap_angle,
)


DEFAULT_RESULT_ROOT = Path(r"C:\Users\kavin\Desktop\LIG2_result")
RUN_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


@dataclass
class MPCConfig:
    plan_dt: float = 0.25
    horizon: int = 16
    replan_every: int = 4
    num_samples: int = 24
    elite_frac: float = 0.25
    cem_iters: int = 2
    grid_resolution_xy: float = 6.0
    grid_resolution_z: float = 5.0
    planner_prediction_time: float = 2.5
    obstacle_buffer: float = 1.6
    obstacle_influence: float = 9.0
    boundary_margin: float = 3.0
    boundary_influence: float = 8.0
    waypoint_lookahead: float = 30.0
    max_speed_cmd: float = 3.0
    min_speed_cmd: float = 0.80
    tau_speed: float = 0.9
    tau_yaw: float = 0.55
    tau_pitch: float = 0.55
    max_yaw_rate: float = 0.75
    max_pitch_rate: float = 0.45
    max_accel: float = 0.65
    speed_std: float = 0.22
    yaw_std: float = 0.025
    pitch_std: float = 0.045
    terminal_goal_weight: float = 24.0
    stage_goal_weight: float = 0.80
    waypoint_weight: float = 0.90
    obstacle_weight: float = 4.5
    boundary_weight: float = 6.0
    attitude_weight: float = 0.5
    smooth_weight: float = 0.05
    speed_weight: float = 0.18
    collision_penalty: float = 4_000.0
    oob_penalty: float = 3_000.0
    low_clearance_penalty: float = 700.0
    prop_base: float = 0.30
    prop_speed_kp: float = 0.72
    prop_surge_kp: float = 0.45
    yaw_kp: float = 3.80
    yaw_rate_kd: float = 0.55
    sway_kd: float = 0.15
    pitch_kp: float = 1.70
    pitch_rate_kd: float = 0.34
    heave_kd: float = 0.10
    heading_slowdown_gain: float = 0.30
    pitch_slowdown_gain: float = 0.22
    clearance_slowdown_range: float = 6.0
    near_goal_relax_distance: float = 8.0
    warm_start_keep: int = 12
    timeout_speed_buffer: float = 0.25
    closing_speed_weight: float = 4.0
    turn_speed_floor: float = 1.20
    goal_bonus: float = 180.0


@dataclass
class EpisodeResult:
    episode: int
    seed: int
    success: int
    reward: float
    steps: int
    event: str
    distance_to_goal: float


def rotation_matrix_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    return np.array([cp * cy, cp * sy, -sp], dtype=np.float64)


def course_to_angles(direction: np.ndarray) -> Tuple[float, float]:
    d = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(d)
    if norm < 1e-8:
        return 0.0, 0.0
    d = d / norm
    yaw = np.arctan2(d[1], d[0])
    pitch = -np.arctan2(d[2], max(np.linalg.norm(d[:2]), 1e-8))
    return float(wrap_angle(yaw)), float(np.clip(pitch, -0.45, 0.45))


def boundary_margin(pos: np.ndarray, world_size: float) -> float:
    xy_margin = min(world_size - abs(pos[0]), world_size - abs(pos[1]))
    z_margin = min(pos[2], world_size - pos[2])
    return float(min(xy_margin, z_margin))


def obstacle_clearance(pos: np.ndarray, obstacles: Sequence[Obstacle], auv_radius: float) -> float:
    if not obstacles:
        return np.inf
    return float(
        min(np.linalg.norm(pos - obs.center) - (auv_radius + obs.radius) for obs in obstacles)
    )


class GridPlanner3D:
    def __init__(self, env: REMUSAUVEnv, cfg: MPCConfig) -> None:
        self.env = env
        self.cfg = cfg
        self.xy_coords = np.arange(-env.world_size, env.world_size + cfg.grid_resolution_xy, cfg.grid_resolution_xy)
        self.z_coords = np.arange(0.0, env.world_size + cfg.grid_resolution_z, cfg.grid_resolution_z)
        self.shape = (len(self.xy_coords), len(self.xy_coords), len(self.z_coords))
        self.neighbors = self._build_neighbors()

    def _build_neighbors(self) -> List[Tuple[int, int, int, float]]:
        offsets: List[Tuple[int, int, int, float]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    cost = np.linalg.norm(
                        np.array(
                            [
                                dx * self.cfg.grid_resolution_xy,
                                dy * self.cfg.grid_resolution_xy,
                                dz * self.cfg.grid_resolution_z,
                            ],
                            dtype=np.float64,
                        )
                    )
                    offsets.append((dx, dy, dz, float(cost)))
        return offsets

    def clamp_idx(self, idx: Tuple[int, int, int]) -> Tuple[int, int, int]:
        ix = int(np.clip(idx[0], 0, self.shape[0] - 1))
        iy = int(np.clip(idx[1], 0, self.shape[1] - 1))
        iz = int(np.clip(idx[2], 0, self.shape[2] - 1))
        return ix, iy, iz

    def pos_to_idx(self, pos: np.ndarray) -> Tuple[int, int, int]:
        ix = int(np.argmin(np.abs(self.xy_coords - pos[0])))
        iy = int(np.argmin(np.abs(self.xy_coords - pos[1])))
        iz = int(np.argmin(np.abs(self.z_coords - pos[2])))
        return self.clamp_idx((ix, iy, iz))

    def idx_to_pos(self, idx: Tuple[int, int, int]) -> np.ndarray:
        return np.array(
            [self.xy_coords[idx[0]], self.xy_coords[idx[1]], self.z_coords[idx[2]]],
            dtype=np.float64,
        )

    def _occupied(self, pos: np.ndarray, obstacles: Sequence[Obstacle]) -> bool:
        if boundary_margin(pos, self.env.world_size) < self.cfg.boundary_margin:
            return True
        inflated_auv = self.env.auv_radius + self.cfg.obstacle_buffer
        for obs in obstacles:
            predicted_center = obs.center + obs.velocity * self.cfg.planner_prediction_time
            if np.linalg.norm(pos - predicted_center) <= (obs.radius + inflated_auv):
                return True
        return False

    def plan(self, start: np.ndarray, goal: np.ndarray, obstacles: Sequence[Obstacle]) -> List[np.ndarray]:
        start_idx = self.pos_to_idx(start)
        goal_idx = self.pos_to_idx(goal)
        if start_idx == goal_idx:
            return [goal.copy()]

        if self._occupied(self.idx_to_pos(start_idx), obstacles):
            return [goal.copy()]

        open_heap: List[Tuple[float, Tuple[int, int, int]]] = []
        heapq.heappush(open_heap, (0.0, start_idx))
        came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        g_score: Dict[Tuple[int, int, int], float] = {start_idx: 0.0}
        closed: set[Tuple[int, int, int]] = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)

            if current == goal_idx:
                path_indices = [current]
                while current in came_from:
                    current = came_from[current]
                    path_indices.append(current)
                path_indices.reverse()
                return [self.idx_to_pos(idx) for idx in path_indices]

            current_pos = self.idx_to_pos(current)
            for dx, dy, dz, move_cost in self.neighbors:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                if not (0 <= neighbor[0] < self.shape[0] and 0 <= neighbor[1] < self.shape[1] and 0 <= neighbor[2] < self.shape[2]):
                    continue
                if neighbor in closed:
                    continue
                neighbor_pos = self.idx_to_pos(neighbor)
                if self._occupied(neighbor_pos, obstacles):
                    continue

                tentative = g_score[current] + move_cost
                b_margin = boundary_margin(neighbor_pos, self.env.world_size)
                if b_margin < self.cfg.boundary_influence:
                    tentative += 2.0 * (self.cfg.boundary_influence - b_margin) ** 2

                dist_goal = np.linalg.norm(neighbor_pos - goal)
                if dist_goal < np.linalg.norm(current_pos - goal):
                    tentative -= 1.5

                if tentative < g_score.get(neighbor, np.inf):
                    g_score[neighbor] = tentative
                    came_from[neighbor] = current
                    heuristic = np.linalg.norm(neighbor_pos - goal)
                    heapq.heappush(open_heap, (tentative + heuristic, neighbor))

        return [goal.copy()]

    def _segment_clear(self, start: np.ndarray, end: np.ndarray, obstacles: Sequence[Obstacle]) -> bool:
        samples = 10
        inflated_auv = self.env.auv_radius + self.cfg.obstacle_buffer
        for idx in range(1, samples + 1):
            alpha = idx / samples
            pos = (1.0 - alpha) * start + alpha * end
            if boundary_margin(pos, self.env.world_size) < self.cfg.boundary_margin:
                return False
            for obs in obstacles:
                if np.linalg.norm(pos - obs.center) <= (obs.radius + inflated_auv):
                    return False
        return True

    def pick_waypoint(
        self,
        path: Sequence[np.ndarray],
        current_pos: np.ndarray,
        goal: np.ndarray,
        lookahead: Optional[float] = None,
    ) -> np.ndarray:
        if not path:
            return goal.copy()
        lookahead_dist = float(self.cfg.waypoint_lookahead if lookahead is None else lookahead)
        max_straight_dist = 1.8 * lookahead_dist
        for point in reversed(path):
            if np.linalg.norm(point - current_pos) > max_straight_dist:
                continue
            if self._segment_clear(current_pos, point, self.env.obstacles):
                return point.copy()

        cumulative = 0.0
        previous = current_pos
        chosen = path[-1]
        for point in path:
            cumulative += np.linalg.norm(point - previous)
            previous = point
            chosen = point
            if cumulative >= lookahead_dist:
                break
        return chosen.copy()


class PathGuidedSamplingMPC:
    def __init__(self, env: REMUSAUVEnv, config: Optional[MPCConfig] = None, seed: int = 0) -> None:
        self.env = env
        self.cfg = config or MPCConfig()
        self.rng = np.random.default_rng(seed)
        self.planner = GridPlanner3D(env, self.cfg)
        self.cached_actions = np.zeros((self.cfg.horizon, 3), dtype=np.float64)
        self.cached_path: List[np.ndarray] = []
        self.cached_waypoint = np.zeros(3, dtype=np.float64)
        self.last_plan_step = -10_000
        self.plan_substeps = max(1, int(round(self.cfg.plan_dt / self.env.dt)))
        self.plan_dt_effective = self.plan_substeps * self.env.dt

    def reset(self) -> None:
        self.cached_actions.fill(0.0)
        self.cached_path = []
        self.cached_waypoint = np.zeros(3, dtype=np.float64)
        self.last_plan_step = -10_000

    def _time_left(self, step_count: Optional[int] = None) -> float:
        step_idx = self.env.step_count if step_count is None else int(step_count)
        return float(max((self.env.max_steps - step_idx) * self.env.dt, self.env.dt))

    def _required_body_speed(self, pos: np.ndarray, step_count: Optional[int] = None) -> float:
        goal_vec = self.env.target - pos
        goal_dist = float(np.linalg.norm(goal_vec))
        if goal_dist < 1e-8:
            return 0.0
        goal_dir = goal_vec / goal_dist
        current_along_goal = float(np.dot(self._current_from_profile(self.env.step_count if step_count is None else int(step_count)), goal_dir))
        required_ground = goal_dist / self._time_left(step_count)
        return max(0.0, required_ground - current_along_goal + self.cfg.timeout_speed_buffer)

    def _reference_speed(self, pos: np.ndarray, eta: np.ndarray) -> float:
        clear = obstacle_clearance(pos, self.env.obstacles, self.env.auv_radius)
        goal_dist = float(np.linalg.norm(self.env.target - pos))
        wall_margin = boundary_margin(pos, self.env.world_size)

        # For the fixed 120 m mission, cruise near the top speed when the path is
        # open, then taper off only near the terminal zone.
        speed = self.cfg.max_speed_cmd
        speed *= np.clip(goal_dist / 45.0, 0.68, 1.0)
        if clear < self.cfg.clearance_slowdown_range:
            speed *= np.clip(clear / self.cfg.clearance_slowdown_range, 0.55, 1.0)
        if wall_margin < self.cfg.boundary_influence:
            speed *= np.clip(wall_margin / self.cfg.boundary_influence, 0.55, 1.0)
        speed *= 1.0 - 0.05 * abs(eta[4])

        if goal_dist > self.cfg.near_goal_relax_distance and clear > 1.5 and wall_margin > 1.5:
            speed = max(speed, self._required_body_speed(pos, self.env.step_count))
        return float(np.clip(speed, self.cfg.min_speed_cmd, self.cfg.max_speed_cmd))

    def _goal_direction(self, pos: np.ndarray) -> np.ndarray:
        waypoint = self.cached_waypoint if self.cached_path else self.env.target
        direction = waypoint - pos
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = self.env.target - pos
            norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return direction / norm

    def _reference_angles(self, pos: np.ndarray) -> Tuple[float, float]:
        direction = self._goal_direction(pos)
        return course_to_angles(direction)

    def _warm_start_mean(self, nominal_action: np.ndarray) -> np.ndarray:
        mean = np.zeros((self.cfg.horizon, 3), dtype=np.float64)
        mean[:] = nominal_action.reshape(1, 3)
        if self.cached_actions.shape[0] >= 2:
            keep = min(self.cfg.warm_start_keep, self.cfg.horizon - 1, self.cached_actions.shape[0] - 1)
            if keep > 0:
                mean[:keep] = self.cached_actions[1:1 + keep]
                mean[keep:] = self.cached_actions[min(keep, self.cached_actions.shape[0] - 1)]
        return mean

    def _sample_action_sequences(self, mean: np.ndarray) -> np.ndarray:
        num_elites = max(4, int(self.cfg.elite_frac * self.cfg.num_samples))
        std = np.array([0.18, 0.28, 0.22], dtype=np.float64)
        best_samples = None
        best_costs = None
        current_mean = mean.copy()
        current_std = std.copy()

        for _ in range(self.cfg.cem_iters):
            noise = self.rng.normal(
                loc=0.0,
                scale=current_std.reshape(1, 1, 3),
                size=(self.cfg.num_samples, self.cfg.horizon, 3),
            )
            if self.cfg.horizon > 1:
                noise[:, 1:, :] = 0.65 * noise[:, 1:, :] + 0.35 * noise[:, :-1, :]
            samples = current_mean[None, :, :] + noise
            samples[:, :, 0] = np.clip(samples[:, :, 0], 0.25, 1.0)
            samples[:, :, 1] = np.clip(samples[:, :, 1], -1.0, 1.0)
            samples[:, :, 2] = np.clip(samples[:, :, 2], -1.0, 1.0)
            samples[0] = current_mean

            costs = np.asarray([self._evaluate_action_sequence(sample) for sample in samples], dtype=np.float64)
            elite_idx = np.argsort(costs)[:num_elites]
            elites = samples[elite_idx]
            elite_mean = np.mean(elites, axis=0)
            current_mean = 0.35 * current_mean + 0.65 * elite_mean
            current_std = np.std(elites, axis=0).mean(axis=0)
            current_std = np.maximum(current_std, np.array([0.03, 0.08, 0.06], dtype=np.float64))
            best_samples = elites
            best_costs = costs[elite_idx]

        assert best_samples is not None and best_costs is not None
        return best_samples[int(np.argmin(best_costs))]

    def _current_from_profile(self, step_count: int) -> np.ndarray:
        if not self.env.current_enabled:
            return np.zeros(3, dtype=np.float64)
        t = step_count * self.env.dt
        oscillation = self.env.current_osc_amp * np.sin(self.env.current_osc_omega * t + self.env.current_phase)
        current = self.env.current_base_inertial + oscillation
        current[2] = np.clip(
            current[2],
            -self.env.current_vertical_max - abs(self.env.current_osc_amp_max[2]),
            self.env.current_vertical_max + abs(self.env.current_osc_amp_max[2]),
        )
        return current

    def _update_actuators_predict(self, actuator_state: np.ndarray, action: np.ndarray) -> np.ndarray:
        desired = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        desired_propeller = np.clip(desired[0], -self.env.max_reverse_propeller, 1.0)
        desired_rudder = desired[1] * self.env.max_rudder
        desired_stern = desired[2] * self.env.max_stern_plane

        next_state = actuator_state.copy()
        next_state[0] = np.clip(
            rate_limit(next_state[0], desired_propeller, self.env.propeller_rate_limit, self.env.dt),
            -self.env.max_reverse_propeller,
            1.0,
        )
        next_state[1] = np.clip(
            rate_limit(next_state[1], desired_rudder, self.env.rudder_rate_limit, self.env.dt),
            -self.env.max_rudder,
            self.env.max_rudder,
        )
        next_state[2] = np.clip(
            rate_limit(next_state[2], desired_stern, self.env.stern_rate_limit, self.env.dt),
            -self.env.max_stern_plane,
            self.env.max_stern_plane,
        )
        return next_state

    def _control_to_tau_predict(self, actuator_state: np.ndarray, nu_r: np.ndarray) -> np.ndarray:
        propeller_cmd, rudder_angle, stern_angle = actuator_state
        u_r, v_r, w_r, _, _, _ = nu_r

        x_prop = self.env.max_thrust * np.sign(propeller_cmd) * (propeller_cmd ** 2)

        beta = np.arctan2(v_r, max(abs(u_r), 1e-4))
        alpha_rudder = rudder_angle - beta
        q_lat = 0.5 * self.env.rho * (u_r ** 2 + v_r ** 2)
        cl_r = self.env._lift_coefficient(alpha_rudder)
        cd_r = self.env._drag_coefficient(alpha_rudder)
        y_rudder = q_lat * self.env.rudder_area * cl_r
        x_rudder_drag = -q_lat * self.env.rudder_area * cd_r
        n_rudder = self.env.rudder_arm * y_rudder

        gamma = np.arctan2(w_r, max(abs(u_r), 1e-4))
        alpha_stern = stern_angle + gamma
        q_vert = 0.5 * self.env.rho * (u_r ** 2 + w_r ** 2)
        cl_s = self.env._lift_coefficient(alpha_stern)
        cd_s = self.env._drag_coefficient(alpha_stern)
        z_stern = -q_vert * self.env.stern_area * cl_s
        x_stern_drag = -q_vert * self.env.stern_area * cd_s
        m_stern = self.env.stern_arm * z_stern

        tau = np.zeros(6, dtype=np.float64)
        tau[0] = x_prop + x_rudder_drag + x_stern_drag
        tau[1] = y_rudder
        tau[2] = z_stern
        tau[4] = m_stern
        tau[5] = n_rudder
        return tau

    def _update_obstacles_predict(self, obstacles: List[Obstacle], target: np.ndarray) -> None:
        for obs in obstacles:
            obs.center = obs.center + obs.velocity * self.env.dt
            for axis in range(3):
                if obs.center[axis] < -self.env.world_size:
                    obs.center[axis] = -self.env.world_size
                    obs.velocity[axis] *= -1.0
                elif obs.center[axis] > self.env.world_size:
                    obs.center[axis] = self.env.world_size
                    obs.velocity[axis] *= -1.0
            if obs.center[2] < 0.0:
                obs.center[2] = 0.0
                obs.velocity[2] *= -1.0

            vec_to_target = obs.center - target
            dist_to_target = np.linalg.norm(vec_to_target)
            min_dist_to_target = obs.radius + self.env.goal_radius + 0.5
            if dist_to_target < min_dist_to_target:
                normal = vec_to_target / (dist_to_target + 1e-8)
                obs.center = target + normal * min_dist_to_target
                obs.velocity = obs.velocity - 2.0 * np.dot(obs.velocity, normal) * normal

    def _predict_step(
        self,
        eta: np.ndarray,
        nu: np.ndarray,
        actuator_state: np.ndarray,
        step_count: int,
        action: np.ndarray,
        obstacles: List[Obstacle],
        target: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, List[Obstacle], np.ndarray]:
        current_inertial = np.zeros(3, dtype=np.float64)
        for _ in range(self.plan_substeps):
            step_count += 1
            current_inertial = self._current_from_profile(step_count)
            actuator_state = self._update_actuators_predict(actuator_state, action)
            rot = rotation_matrix_body_to_inertial(eta[3], eta[4], eta[5])
            current_body = rot.T @ current_inertial
            nu_r = nu.copy()
            nu_r[:3] -= current_body
            tau = self._control_to_tau_predict(actuator_state, nu_r)

            c_rb = self.env._coriolis_matrix_diag(self.env.MRB_diag, nu)
            c_a = self.env._coriolis_matrix_diag(self.env.MA_diag, nu_r)
            drag = self.env._drag(nu_r)
            restoring = self.env._restoring_force(eta)
            rhs = tau - (c_rb @ nu) - (c_a @ nu_r) - drag - restoring
            nu_dot = self.env.M_inv @ rhs
            nu = np.clip(nu + self.env.dt * nu_dot, -self.env.velocity_clip, self.env.velocity_clip)

            eta_dot = self.env._kinematics(eta, nu)
            eta = eta + self.env.dt * eta_dot
            eta[3] = wrap_angle(eta[3])
            eta[4] = np.clip(wrap_angle(eta[4]), -self.env.theta_limit, self.env.theta_limit)
            eta[5] = wrap_angle(eta[5])

            self._update_obstacles_predict(obstacles, target)
        return eta, nu, actuator_state, step_count, obstacles, current_inertial

    def _evaluate_action_sequence(self, sequence: np.ndarray) -> float:
        eta = self.env.state[:6].copy()
        nu = self.env.state[6:].copy()
        actuator_state = self.env.actuator_state.copy()
        step_count = int(self.env.step_count)
        obstacles = [Obstacle(obs.center.copy(), obs.velocity.copy(), float(obs.radius)) for obs in self.env.obstacles]
        target = self.env.target.copy()
        waypoint = self.cached_waypoint.copy() if self.cached_path else target.copy()
        prev_action = self.env.last_action.copy()
        prev_pos = eta[:3].copy()
        initial_goal_dist = float(np.linalg.norm(prev_pos - target))
        total_cost = 0.0

        for action in sequence:
            eta, nu, actuator_state, step_count, obstacles, _ = self._predict_step(
                eta=eta,
                nu=nu,
                actuator_state=actuator_state,
                step_count=step_count,
                action=action,
                obstacles=obstacles,
                target=target,
            )
            pos = eta[:3]
            dist_wp = float(np.linalg.norm(pos - waypoint))
            dist_goal = float(np.linalg.norm(pos - target))
            progress = float(np.linalg.norm(prev_pos - target) - dist_goal)
            prev_pos = pos.copy()

            total_cost += self.cfg.waypoint_weight * dist_wp
            total_cost += self.cfg.stage_goal_weight * dist_goal
            total_cost -= 18.0 * progress
            desired_yaw, desired_pitch = course_to_angles(waypoint - pos)
            total_cost += self.cfg.attitude_weight * (abs(eta[4] - desired_pitch) + 0.35 * abs(wrap_angle(eta[5] - desired_yaw)))
            total_cost += self.cfg.smooth_weight * np.sum((action - prev_action) ** 2)
            total_cost += 0.12 * (0.95 - float(action[0])) ** 2
            prev_action = np.asarray(action, dtype=np.float64)

            time_left = max((self.env.max_steps - step_count) * self.env.dt, self.plan_dt_effective)
            required_closing = dist_goal / time_left
            closing_speed = max(progress / self.plan_dt_effective, 0.0)
            if dist_goal > self.cfg.near_goal_relax_distance:
                deficit = max(0.0, required_closing - closing_speed)
                total_cost += self.cfg.closing_speed_weight * (deficit ** 2) * self.plan_dt_effective

            clear = obstacle_clearance(pos, obstacles, self.env.auv_radius)
            if clear < self.cfg.obstacle_influence:
                total_cost += self.cfg.obstacle_weight * (self.cfg.obstacle_influence - clear) ** 2
            if clear < 0.75:
                total_cost += self.cfg.low_clearance_penalty * (0.75 - clear) ** 2
            if clear < 0.0:
                total_cost += self.cfg.collision_penalty
                break

            b_margin = boundary_margin(pos, self.env.world_size)
            if b_margin < self.cfg.boundary_influence:
                total_cost += self.cfg.boundary_weight * (self.cfg.boundary_influence - b_margin) ** 2
            if b_margin < 0.0:
                total_cost += self.cfg.oob_penalty
                break

            if dist_goal <= self.env.goal_radius:
                total_cost -= self.cfg.goal_bonus
                break

        terminal_goal = np.linalg.norm(eta[:3] - target)
        total_cost += self.cfg.terminal_goal_weight * terminal_goal
        total_cost -= 6.0 * (initial_goal_dist - terminal_goal)
        return float(total_cost)

    def _low_level_action(self, desired_speed: float, desired_yaw: float, desired_pitch: float) -> np.ndarray:
        eta = self.env.state[:6]
        nu = self.env.state[6:]
        desired_ground = desired_speed * rotation_matrix_yaw_pitch(desired_yaw, desired_pitch)
        yaw_cmd, pitch_cmd = course_to_angles(desired_ground)
        yaw_error = wrap_angle(yaw_cmd - eta[5])
        pitch_cmd = np.clip(pitch_cmd, -0.45, 0.45)

        pitch_error_term = eta[4] - pitch_cmd
        clear = obstacle_clearance(eta[:3], self.env.obstacles, self.env.auv_radius)
        wall_margin = boundary_margin(eta[:3], self.env.world_size)
        goal_dist = float(np.linalg.norm(self.env.target - eta[:3]))

        speed_cap = desired_speed
        speed_cap /= 1.0 + self.cfg.heading_slowdown_gain * abs(yaw_error)
        speed_cap /= 1.0 + self.cfg.pitch_slowdown_gain * abs(pitch_error_term)
        if abs(yaw_error) > 0.95:
            speed_cap *= 0.62
        elif abs(yaw_error) > 0.55:
            speed_cap *= 0.78
        if abs(pitch_error_term) > 0.30:
            speed_cap *= 0.74
        if clear < self.cfg.clearance_slowdown_range:
            speed_cap *= np.clip(clear / self.cfg.clearance_slowdown_range, 0.40, 1.0)
        if wall_margin < self.cfg.boundary_influence:
            speed_cap *= np.clip(wall_margin / self.cfg.boundary_influence, 0.40, 1.0)
        if goal_dist > self.cfg.near_goal_relax_distance and clear > 1.2 and wall_margin > 1.2:
            speed_cap = max(speed_cap, min(self.cfg.turn_speed_floor, desired_speed))
        speed_cap = float(np.clip(speed_cap, self.cfg.min_speed_cmd, self.cfg.max_speed_cmd))

        propeller = (
            self.cfg.prop_base
            + self.cfg.prop_speed_kp * (speed_cap / self.cfg.max_speed_cmd)
            + self.cfg.prop_surge_kp * (speed_cap - nu[0])
        )
        rudder = self.cfg.yaw_kp * yaw_error - self.cfg.yaw_rate_kd * nu[5] - self.cfg.sway_kd * nu[1]
        stern = self.cfg.pitch_kp * pitch_error_term + self.cfg.pitch_rate_kd * nu[4] + self.cfg.heave_kd * nu[2]

        action = np.array(
            [
                np.clip(propeller, -self.env.max_reverse_propeller, 1.0),
                np.clip(rudder, -1.0, 1.0),
                np.clip(stern, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        return action

    def _maybe_replan_path(self) -> None:
        if (self.env.step_count - self.last_plan_step) < self.cfg.replan_every and self.cached_path:
            return
        pos = self.env.state[:3]
        goal_dist = float(np.linalg.norm(self.env.target - pos))
        clear = obstacle_clearance(pos, self.env.obstacles, self.env.auv_radius)
        lookahead = self.cfg.waypoint_lookahead
        if clear > self.cfg.clearance_slowdown_range and goal_dist > 2.0 * self.cfg.waypoint_lookahead:
            lookahead = min(1.45 * self.cfg.waypoint_lookahead, self.cfg.waypoint_lookahead + 0.12 * goal_dist)
        self.cached_path = self.planner.plan(pos, self.env.target, self.env.obstacles)
        self.cached_waypoint = self.planner.pick_waypoint(self.cached_path, pos, self.env.target, lookahead=lookahead)
        self.last_plan_step = int(self.env.step_count)

    def _sampling_needed(self, pos: np.ndarray) -> bool:
        clear = obstacle_clearance(pos, self.env.obstacles, self.env.auv_radius)
        wall_margin = boundary_margin(pos, self.env.world_size)
        return bool(
            clear < (self.cfg.obstacle_influence + 1.0)
            or wall_margin < (self.cfg.boundary_influence + 1.0)
        )

    def act(self) -> np.ndarray:
        self._maybe_replan_path()

        pos = self.env.state[:3]
        eta = self.env.state[:6]
        speed_ref = self._reference_speed(pos, eta)
        ref_yaw, ref_pitch = self._reference_angles(pos)
        nominal_action = self._low_level_action(speed_ref, ref_yaw, ref_pitch).astype(np.float64)

        if not self._sampling_needed(pos):
            self.cached_actions = self._warm_start_mean(nominal_action)
            return np.asarray(nominal_action, dtype=np.float32)

        mean = self._warm_start_mean(nominal_action)
        best_sequence = self._sample_action_sequences(mean)
        self.cached_actions = best_sequence.copy()
        blended = 0.75 * nominal_action + 0.25 * best_sequence[0]
        blended[0] = np.clip(blended[0], -self.env.max_reverse_propeller, 1.0)
        blended[1:] = np.clip(blended[1:], -1.0, 1.0)
        return np.asarray(blended, dtype=np.float32)


def resolve_run_dir(result_root: Path, out_dir: Optional[Path], started_at: datetime, smoke: bool) -> Optional[Path]:
    if smoke:
        return None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    run_dir = result_root / started_at.strftime(RUN_TIMESTAMP_FORMAT)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def evaluate_controller(args: argparse.Namespace) -> Dict[str, Any]:
    started_at = datetime.now()
    run_dir = resolve_run_dir(args.result_root, args.out_dir, started_at, args.smoke)

    config = MPCConfig(
        plan_dt=args.plan_dt,
        horizon=args.horizon,
        replan_every=args.replan_every,
        num_samples=args.num_samples,
        elite_frac=args.elite_frac,
        cem_iters=args.cem_iters,
        grid_resolution_xy=args.grid_resolution_xy,
        grid_resolution_z=args.grid_resolution_z,
        waypoint_lookahead=args.waypoint_lookahead,
    )

    results: List[EpisodeResult] = []
    trajectories: List[Dict[str, Any]] = []

    for episode in range(args.episodes):
        env_seed = args.seed + episode
        env = REMUSAUVEnv(
            dt=args.env_dt,
            max_steps=args.env_max_steps,
            n_obstacles=args.n_obstacles,
            current_enabled=not args.disable_current,
            include_current_in_obs=True,
            seed=env_seed,
        )
        obs, info = env.reset(seed=env_seed)
        controller = PathGuidedSamplingMPC(env, config=config, seed=env_seed)
        controller.reset()

        episode_return = 0.0
        event = "timeout"
        path_log: List[List[float]] = [env.state[:3].astype(float).tolist()]
        obstacle_log: List[List[List[float]]] = []

        for _ in range(env.max_steps):
            action = controller.act()
            obs, reward, terminated, truncated, step_info = env.step(action)
            del obs
            episode_return += float(reward)
            path_log.append(env.state[:3].astype(float).tolist())
            obstacle_log.append([obs_item.center.astype(float).tolist() for obs_item in env.obstacles])
            if terminated or truncated:
                event = str(step_info.get("event", "timeout" if truncated else "other"))
                break

        final_dist = float(np.linalg.norm(env.state[:3] - env.target))
        results.append(
            EpisodeResult(
                episode=episode,
                seed=env_seed,
                success=int(event == "goal"),
                reward=float(episode_return),
                steps=int(env.step_count),
                event=event,
                distance_to_goal=final_dist,
            )
        )

        if run_dir is not None and args.save_trajectories:
            trajectories.append(
                {
                    "episode": episode,
                    "seed": env_seed,
                    "event": event,
                    "path": path_log,
                    "target": env.target.astype(float).tolist(),
                    "start": env.start.astype(float).tolist(),
                    "obstacles": obstacle_log,
                }
            )
        env.close()

    counts = {
        "goal": sum(item.event == "goal" for item in results),
        "collision": sum(item.event == "collision" for item in results),
        "out_of_bounds": sum(item.event == "out_of_bounds" for item in results),
        "timeout": sum(item.event == "timeout" for item in results),
        "other": sum(item.event not in {"goal", "collision", "out_of_bounds", "timeout"} for item in results),
    }

    summary = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "episodes": args.episodes,
        "seed": args.seed,
        "success_rate": counts["goal"] / max(args.episodes, 1),
        "counts": counts,
        "mean_reward": float(np.mean([item.reward for item in results])) if results else 0.0,
        "mean_steps": float(np.mean([item.steps for item in results])) if results else 0.0,
        "mean_final_distance": float(np.mean([item.distance_to_goal for item in results])) if results else 0.0,
        "environment": {
            "env_dt": args.env_dt,
            "env_max_steps": args.env_max_steps,
            "n_obstacles": args.n_obstacles,
            "current_enabled": not args.disable_current,
        },
        "controller": asdict(config),
    }

    if run_dir is not None:
        save_json(run_dir / "summary.json", summary)
        with (run_dir / "episode_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "episode",
                    "seed",
                    "success",
                    "reward",
                    "steps",
                    "event",
                    "distance_to_goal",
                ],
            )
            writer.writeheader()
            for item in results:
                writer.writerow(asdict(item))
        if args.save_trajectories:
            save_json(run_dir / "trajectories.json", {"episodes": trajectories})

    print(
        f"success_rate={summary['success_rate']:.3f} | "
        f"goal/collision/oob/timeout={counts['goal']}/{counts['collision']}/{counts['out_of_bounds']}/{counts['timeout']}",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid path-planning + sampling MPC controller for REMUSAUVEnv."
    )
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed.")
    parser.add_argument("--env-dt", type=float, default=0.05, help="Environment dt.")
    parser.add_argument("--env-max-steps", type=int, default=2400, help="Maximum environment steps.")
    parser.add_argument("--n-obstacles", type=int, default=6, help="Number of dynamic obstacles.")
    parser.add_argument("--disable-current", action="store_true", help="Disable ocean current disturbance.")
    parser.add_argument("--smoke", action="store_true", help="Do not save outputs.")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT, help="Root for timestamped results.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Optional explicit output directory.")
    parser.add_argument("--save-trajectories", action="store_true", help="Save trajectory JSON when not in smoke mode.")
    parser.add_argument("--plan-dt", type=float, default=0.25, help="Internal MPC planning dt.")
    parser.add_argument("--horizon", type=int, default=16, help="Internal MPC horizon.")
    parser.add_argument("--replan-every", type=int, default=4, help="Path replanning period in environment steps.")
    parser.add_argument("--num-samples", type=int, default=24, help="Number of sampled action sequences.")
    parser.add_argument("--elite-frac", type=float, default=0.25, help="Elite fraction for CEM.")
    parser.add_argument("--cem-iters", type=int, default=2, help="Number of CEM updates per MPC step.")
    parser.add_argument("--grid-resolution-xy", type=float, default=6.0, help="Planner XY grid resolution.")
    parser.add_argument("--grid-resolution-z", type=float, default=5.0, help="Planner Z grid resolution.")
    parser.add_argument("--waypoint-lookahead", type=float, default=30.0, help="Distance used to select the path waypoint.")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_controller(parse_args())
