from __future__ import annotations

import argparse
import csv
import heapq
import json
import time
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


def format_vec3(vec: np.ndarray) -> str:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    return f"({arr[0]:6.2f}, {arr[1]:6.2f}, {arr[2]:6.2f})"


def maybe_print_progress(
    *,
    episode: int,
    total_episodes: int,
    env: REMUSAUVEnv,
    episode_return: float,
    step_info: Dict[str, Any],
    log_prefix: str = "",
) -> None:
    pos = env.state[:3]
    nu = env.state[6:]
    dist = float(step_info.get("distance_to_goal", np.linalg.norm(pos - env.target)))
    clearance = obstacle_clearance(pos, env.obstacles, env.auv_radius)
    wall_margin = boundary_margin(pos, env.world_size)
    action = env.last_action
    actuator = step_info.get("actuator_state", env.actuator_state)
    current_body = step_info.get("current_body", np.zeros(3, dtype=np.float64))
    act_time = float(step_info.get("act_time", 0.0))
    mode = str(step_info.get("mode", ""))
    mode_text = f" | mode={mode}" if mode else ""
    print(
        f"{log_prefix}[ep {episode + 1:02d}/{total_episodes:02d} | step {env.step_count:04d}] "
        f"dist={dist:6.2f} m | pos={format_vec3(pos)} | vel=({nu[0]:5.2f},{nu[1]:5.2f},{nu[2]:5.2f}) "
        f"| act=({action[0]:4.2f},{action[1]:4.2f},{action[2]:4.2f}) "
        f"| fin=({float(actuator[0]):4.2f},{float(actuator[1]):4.2f},{float(actuator[2]):4.2f}) "
        f"| clr={clearance:5.2f} | wall={wall_margin:5.2f} | cur={format_vec3(current_body)} "
        f"| t={act_time*1000.0:5.1f} ms{mode_text} | R={episode_return:8.2f}",
        flush=True,
    )


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
    boundary_margin: float = 4.0
    boundary_influence: float = 10.0
    waypoint_lookahead: float = 24.0
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
    obstacle_weight: float = 4.8
    boundary_weight: float = 7.0
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
    current_comp_gain: float = 0.65
    terminal_current_comp_gain: float = 0.78
    wall_current_comp_gain: float = 0.92
    warm_start_keep: int = 12
    timeout_speed_buffer: float = 0.25
    closing_speed_weight: float = 4.0
    turn_speed_floor: float = 1.20
    goal_bonus: float = 180.0
    terminal_capture_distance: float = 14.0
    terminal_sampling_distance: float = 18.0
    terminal_speed_max: float = 1.25
    terminal_min_speed: float = 0.60
    terminal_steer_speed_min: float = 0.65
    terminal_waypoint_lookahead: float = 9.0
    terminal_prop_floor: float = 0.18
    wall_recovery_margin: float = 9.0
    wall_safe_margin: float = 6.0
    wall_steer_speed_min: float = 0.50
    wall_prop_floor: float = 0.10
    wall_outward_speed_weight: float = 42.0
    shortcut_boundary_margin: float = 8.0
    shortcut_clearance_margin: float = 2.0
    shortcut_max_factor: float = 1.18
    bypass_los_margin: float = 1.5
    bypass_clearance_margin: float = 2.5
    bypass_lead_distance: float = 6.0
    bypass_waypoint_reach: float = 3.0
    bypass_goal_stage_weight: float = 0.15
    bypass_waypoint_weight_scale: float = 4.0
    bypass_speed_max: float = 1.55
    bypass_speed_min: float = 0.90
    emergency_reverse_margin: float = 1.00
    emergency_reverse_outward_speed: float = 0.20
    inner_terminal_nominal_distance: float = 4.0
    missed_capture_distance: float = 4.0
    final_capture_distance: float = 1.9
    final_capture_xy_window: float = 0.85
    final_capture_speed_min: float = 0.86
    final_capture_speed_max: float = 1.12
    final_capture_prop_floor: float = 0.32
    final_capture_rudder_limit: float = 0.72
    final_capture_pitch_gain: float = 1.25
    final_capture_stern_bias: float = 0.12
    final_capture_current_comp_gain: float = 0.22
    final_capture_settle_distance: float = 1.35
    final_capture_settle_xy_window: float = 0.60
    final_capture_settle_tangential_speed: float = 0.14
    final_capture_settle_outward_speed: float = 0.05
    near_missed_capture_delta: float = 0.30
    mpc_time_budget_ms: float = 70.0
    safe_num_samples: int = 8
    safe_horizon: int = 8
    safe_cem_iters: int = 1
    medium_num_samples: int = 14
    medium_horizon: int = 10
    medium_cem_iters: int = 1
    danger_num_samples: int = 24
    danger_horizon: int = 16
    danger_cem_iters: int = 2
    early_terminate_ratio: float = 1.15
    early_terminate_abs_margin: float = 120.0
    capture_latch_distance: float = 2.8
    capture_latch_prop_floor: float = 0.30
    capture_latch_speed_min: float = 0.72
    capture_latch_speed_max: float = 1.00
    capture_latch_yaw_gain: float = 1.20
    capture_latch_pitch_gain: float = 1.25
    capture_latch_penalty: float = 18.0
    capture_latch_progress_margin: float = 0.15
    capture_commit_distance: float = 4.50
    capture_commit_speed_min: float = 0.58
    capture_commit_speed_max: float = 0.86
    capture_commit_prop_floor: float = 0.22
    capture_commit_prop_ceiling: float = 0.58
    capture_commit_prop_tangent_gain: float = 0.55
    capture_commit_prop_outward_gain: float = 1.10
    capture_commit_rudder_limit: float = 0.70
    capture_commit_rudder_close_limit: float = 0.48
    capture_commit_xy_hold_window: float = 1.80
    capture_commit_pitch_gain: float = 1.18
    capture_pos_gain: float = 1.00
    capture_tangent_cancel_gain: float = 1.05
    capture_outward_cancel_gain: float = 1.20
    capture_tangential_weight: float = 14.0
    capture_outward_weight: float = 18.0
    capture_prop_ceiling_weight: float = 0.60
    capture_progress_weight: float = 1.20
    capture_settle_distance: float = 1.80
    capture_reverse_outward_trigger: float = 0.10
    capture_reverse_max: float = 0.06
    recapture_reentry_distance: float = 6.5
    recapture_reentry_lateral_scale: float = 2.2
    recapture_reentry_speed_min: float = 0.82
    recapture_reentry_speed_max: float = 1.08
    recapture_reentry_prop_floor: float = 0.24
    recapture_reentry_rudder_limit: float = 0.86
    recapture_anchor_radius: float = 1.75
    recapture_goal_release_distance: float = 3.2
    recapture_rearm_steps: int = 25


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

        # Follow the planned path conservatively by default. The previous version
        # jumped to the farthest line-of-sight point, which often cut corners near
        # the boundary and dynamic obstacles. Here we first choose the nominal
        # along-path waypoint, then allow only modest straight-line shortcutting
        # in clearly safe open water.
        cumulative = 0.0
        previous = current_pos
        chosen_idx = len(path) - 1
        for idx, point in enumerate(path):
            cumulative += np.linalg.norm(point - previous)
            previous = point
            chosen_idx = idx
            if cumulative >= lookahead_dist:
                break

        chosen = path[chosen_idx].copy()
        wall_margin = boundary_margin(current_pos, self.env.world_size)
        clear = obstacle_clearance(current_pos, self.env.obstacles, self.env.auv_radius)
        max_straight_dist = self.cfg.shortcut_max_factor * lookahead_dist
        if wall_margin < self.cfg.shortcut_boundary_margin or clear < (self.cfg.obstacle_influence + self.cfg.shortcut_clearance_margin):
            return chosen

        shortcut_idx = chosen_idx
        for idx in range(chosen_idx + 1, len(path)):
            point = path[idx]
            if np.linalg.norm(point - current_pos) > max_straight_dist:
                break
            if self._segment_clear(current_pos, point, self.env.obstacles):
                shortcut_idx = idx
            else:
                break
        return path[shortcut_idx].copy()


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
        self.best_goal_dist = np.inf
        self.capture_latched = False
        self.recapture_active = False
        self.recapture_anchor = self.env.target.copy()
        self.last_missed_capture_step = -10_000
        self.bypass_active = False
        self.bypass_path: List[np.ndarray] = []
        self.current_mode = ""
        self.last_act_time = 0.0

    def reset(self) -> None:
        self.cached_actions.fill(0.0)
        self.cached_path = []
        self.cached_waypoint = np.zeros(3, dtype=np.float64)
        self.last_plan_step = -10_000
        self.best_goal_dist = np.inf
        self.capture_latched = False
        self.recapture_active = False
        self.recapture_anchor = self.env.target.copy()
        self.last_missed_capture_step = -10_000
        self.bypass_active = False
        self.bypass_path = []
        self.current_mode = ""
        self.last_act_time = 0.0

    def _update_capture_latch(self, pos: np.ndarray, goal_dist: Optional[float] = None) -> None:
        dist = float(np.linalg.norm(self.env.target - pos)) if goal_dist is None else float(goal_dist)
        if dist <= self.cfg.capture_latch_distance or self.best_goal_dist <= self.cfg.capture_latch_distance:
            self.capture_latched = True

    def _capture_latch_mode(self, pos: np.ndarray) -> bool:
        _ = pos
        return bool(self.capture_latched)


    def _safe_unit(self, vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float64)
        norm = float(np.linalg.norm(arr))
        if norm < 1e-8:
            return np.zeros_like(arr)
        return arr / norm

    def _activate_recapture_anchor(self, pos: np.ndarray, eta: Optional[np.ndarray] = None, nu: Optional[np.ndarray] = None) -> None:
        _ = pos, eta, nu
        self.recapture_anchor = self.env.target.copy()
        self.recapture_active = True
        self.last_missed_capture_step = int(self.env.step_count)

    def _update_recapture_state(self, pos: np.ndarray, eta: Optional[np.ndarray] = None, nu: Optional[np.ndarray] = None) -> None:
        goal_dist, _, _, _, _, v_r, _, _ = self._goal_velocity_components(pos, eta, nu)
        if self.recapture_active:
            anchor_dist = float(np.linalg.norm(pos - self.recapture_anchor))
            if self._final_capture_mode(pos, eta):
                self.recapture_active = False
                return
            if anchor_dist <= self.cfg.recapture_anchor_radius:
                self.recapture_active = False
                return
            if goal_dist <= self.cfg.recapture_goal_release_distance and v_r > 0.05:
                self.recapture_active = False
                return
            if self._missed_capture(pos, eta) and (self.env.step_count - self.last_missed_capture_step) >= self.cfg.recapture_rearm_steps:
                self._activate_recapture_anchor(pos, eta, nu)
            return

        if self.capture_latched and self._missed_capture(pos, eta):
            self._activate_recapture_anchor(pos, eta, nu)

    def _navigation_target(self) -> np.ndarray:
        if self.recapture_active:
            return self.env.target.copy()
        if self.cached_path:
            return self.cached_waypoint.copy()
        return self.env.target.copy()

    def _segment_block_distance(self, start: np.ndarray, end: np.ndarray, center: np.ndarray) -> Tuple[float, float, np.ndarray]:
        seg = np.asarray(end - start, dtype=np.float64)
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq < 1e-12:
            closest = np.asarray(start, dtype=np.float64).copy()
            return float(np.linalg.norm(center - closest)), 0.0, closest
        alpha = float(np.clip(np.dot(center - start, seg) / seg_len_sq, 0.0, 1.0))
        closest = start + alpha * seg
        return float(np.linalg.norm(center - closest)), alpha, closest

    def _find_blocking_obstacle(self, start: np.ndarray, end: np.ndarray, obstacles: Sequence[Obstacle]) -> Optional[Obstacle]:
        seg = np.asarray(end - start, dtype=np.float64)
        if float(np.linalg.norm(seg)) < 1e-6:
            return None
        best_obs: Optional[Obstacle] = None
        best_alpha = np.inf
        inflate_extra = self.cfg.obstacle_buffer + self.cfg.bypass_los_margin + self.env.auv_radius
        for obs in obstacles:
            predicted_center = obs.center + obs.velocity * min(self.cfg.planner_prediction_time, 1.5)
            dist, alpha, _ = self._segment_block_distance(start, end, predicted_center)
            blocked_radius = float(obs.radius + inflate_extra)
            if dist <= blocked_radius and alpha < best_alpha:
                best_alpha = alpha
                best_obs = Obstacle(predicted_center.copy(), obs.velocity.copy(), float(obs.radius))
        return best_obs

    def _segment_min_clearance(self, start: np.ndarray, end: np.ndarray, obstacles: Sequence[Obstacle], samples: int = 18) -> float:
        min_clear = np.inf
        for idx in range(samples + 1):
            alpha = idx / max(samples, 1)
            pos = (1.0 - alpha) * start + alpha * end
            b_margin = boundary_margin(pos, self.env.world_size)
            min_clear = min(min_clear, b_margin)
            for obs in obstacles:
                clear = float(np.linalg.norm(pos - obs.center) - (obs.radius + self.env.auv_radius))
                min_clear = min(min_clear, clear)
        return float(min_clear)

    def _build_bypass_path(self, start: np.ndarray, goal: np.ndarray, blocker: Obstacle) -> List[np.ndarray]:
        direction = self._safe_unit(goal - start)
        if float(np.linalg.norm(direction)) < 1e-6:
            return [goal.copy()]
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        lateral = np.cross(direction, up)
        if float(np.linalg.norm(lateral)) < 1e-6:
            lateral = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        lateral = self._safe_unit(lateral)
        vertical = self._safe_unit(np.cross(lateral, direction))

        base_clear = blocker.radius + self.env.auv_radius + self.cfg.obstacle_buffer + self.cfg.bypass_clearance_margin
        lead = self.cfg.bypass_lead_distance + 0.35 * base_clear
        candidates: List[Tuple[float, List[np.ndarray]]] = []
        obstacle_list = self.env.obstacles

        for side in (-1.0, 1.0):
            for vscale in (0.0, 0.45, -0.35):
                offset = side * lateral * (1.05 * base_clear) + vscale * vertical * (0.55 * base_clear)
                pre = blocker.center - direction * lead + offset
                post = blocker.center + direction * lead + offset
                pre[0] = float(np.clip(pre[0], -self.env.world_size + 2.0, self.env.world_size - 2.0))
                pre[1] = float(np.clip(pre[1], -self.env.world_size + 2.0, self.env.world_size - 2.0))
                pre[2] = float(np.clip(pre[2], 2.0, self.env.world_size - 2.0))
                post[0] = float(np.clip(post[0], -self.env.world_size + 2.0, self.env.world_size - 2.0))
                post[1] = float(np.clip(post[1], -self.env.world_size + 2.0, self.env.world_size - 2.0))
                post[2] = float(np.clip(post[2], 2.0, self.env.world_size - 2.0))
                path = [pre, post, goal.copy()]
                segments = [(start, pre), (pre, post), (post, goal)]
                total_len = float(sum(np.linalg.norm(b - a) for a, b in segments))
                min_clear = min(self._segment_min_clearance(a, b, obstacle_list) for a, b in segments)
                blocked_penalty = 0.0
                for a, b in segments:
                    if not self.planner._segment_clear(a, b, obstacle_list):
                        blocked_penalty += 3000.0
                score = total_len + blocked_penalty + 140.0 / max(min_clear + 0.25, 0.25)
                candidates.append((score, path))

        if not candidates:
            return [goal.copy()]
        candidates.sort(key=lambda item: item[0])
        return [p.copy() for p in candidates[0][1]]

    def _mpc_profile(self, pos: np.ndarray) -> Tuple[int, int, int]:
        clear = obstacle_clearance(pos, self.env.obstacles, self.env.auv_radius)
        wall_margin = boundary_margin(pos, self.env.world_size)
        goal_dist = float(np.linalg.norm(self.env.target - pos))
        missed = self._missed_capture(pos)
        if self.bypass_active or self._capture_latch_mode(pos) or goal_dist < self.cfg.terminal_sampling_distance or missed or clear < 6.0 or wall_margin < self.cfg.wall_recovery_margin + 0.5:
            return self.cfg.danger_num_samples, self.cfg.danger_horizon, self.cfg.danger_cem_iters
        if goal_dist < 28.0 or clear < 11.0 or wall_margin < self.cfg.boundary_influence + 2.0:
            return self.cfg.medium_num_samples, self.cfg.medium_horizon, self.cfg.medium_cem_iters
        return self.cfg.safe_num_samples, self.cfg.safe_horizon, self.cfg.safe_cem_iters

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
        goal_dist, horiz_err, vert_err, _, _, v_r, v_t, _ = self._goal_velocity_components(pos, eta, self.env.state[6:])
        wall_margin = boundary_margin(pos, self.env.world_size)
        rel_body = self._goal_relative_body(pos, eta)
        forward_err = float(rel_body[0])
        lateral_vert_err = float(np.linalg.norm(rel_body[1:]))
        missed_capture = self._missed_capture(pos, eta)
        final_capture = self._final_capture_mode(pos, eta)
        capture_commit = self._capture_commit_mode(pos, eta)

        if self.bypass_active:
            speed = 1.18 + 0.05 * min(goal_dist, 12.0)
            if clear < self.cfg.clearance_slowdown_range:
                speed *= np.clip(clear / self.cfg.clearance_slowdown_range, 0.72, 1.0)
            if wall_margin < self.cfg.wall_recovery_margin:
                speed *= np.clip(wall_margin / self.cfg.wall_recovery_margin, 0.70, 1.0)
            return float(np.clip(speed, self.cfg.bypass_speed_min, self.cfg.bypass_speed_max))

        if self.recapture_active:
            anchor_dist = float(np.linalg.norm(self.recapture_anchor - pos))
            speed = 1.02 + 0.05 * min(anchor_dist, 4.0)
            if anchor_dist < (self.cfg.recapture_anchor_radius + 0.8):
                speed = min(speed, 0.98)
            if clear < self.cfg.clearance_slowdown_range:
                speed *= np.clip(clear / self.cfg.clearance_slowdown_range, 0.78, 1.0)
            if wall_margin < self.cfg.wall_recovery_margin:
                speed *= np.clip(wall_margin / self.cfg.wall_recovery_margin, 0.82, 1.0)
            return float(np.clip(speed, self.cfg.recapture_reentry_speed_min, self.cfg.recapture_reentry_speed_max))

        elif final_capture:
            speed = 0.94 + 0.05 * min(abs(vert_err), 2.0) + 0.04 * min(horiz_err, 1.5)
            speed -= 0.10 * v_t
            speed -= 0.16 * max(0.0, -v_r)
            if horiz_err <= self.cfg.final_capture_xy_window and abs(vert_err) > 0.8:
                speed = max(speed, 0.96)
            if clear < self.cfg.clearance_slowdown_range:
                speed *= np.clip(clear / self.cfg.clearance_slowdown_range, 0.78, 1.0)
            if wall_margin < self.cfg.wall_recovery_margin:
                speed *= np.clip(wall_margin / self.cfg.wall_recovery_margin, 0.78, 1.0)
            return float(np.clip(speed, self.cfg.final_capture_speed_min, self.cfg.final_capture_speed_max))

        if capture_commit:
            speed = 0.70 + 0.04 * min(goal_dist, self.cfg.capture_commit_distance) + 0.05 * min(abs(vert_err), 2.5)
            speed -= 0.24 * v_t
            speed -= 0.42 * max(0.0, -v_r)
            if horiz_err < self.cfg.capture_commit_xy_hold_window and abs(vert_err) > 0.9:
                speed = min(speed, 0.76)
            if clear < self.cfg.clearance_slowdown_range:
                speed *= np.clip(clear / self.cfg.clearance_slowdown_range, 0.80, 1.0)
            if wall_margin < self.cfg.wall_recovery_margin:
                speed *= np.clip(wall_margin / self.cfg.wall_recovery_margin, 0.72, 1.0)
            return float(np.clip(speed, self.cfg.capture_commit_speed_min, self.cfg.capture_commit_speed_max))

        if goal_dist <= self.cfg.terminal_capture_distance or missed_capture:
            speed = min(self.cfg.terminal_speed_max, 0.50 + 0.07 * goal_dist + 0.03 * max(forward_err, 0.0))
            if lateral_vert_err > 1.25:
                speed = min(speed, 0.80)
            if forward_err < -0.25:
                speed = min(speed, self.cfg.terminal_steer_speed_min)
            if missed_capture:
                speed = min(speed, 0.84)
            if clear < self.cfg.clearance_slowdown_range:
                speed *= np.clip(clear / self.cfg.clearance_slowdown_range, 0.55, 1.0)
            if wall_margin < self.cfg.wall_recovery_margin:
                speed *= np.clip(wall_margin / self.cfg.wall_recovery_margin, 0.50, 1.0)
            required = self._required_body_speed(pos, self.env.step_count)
            if goal_dist > (2.0 * self.env.goal_radius) and forward_err > 0.10:
                speed = max(speed, min(self.cfg.terminal_speed_max, 0.75 * required))
            min_speed = self.cfg.terminal_min_speed if goal_dist > (2.0 * self.env.goal_radius) else 0.42
            return float(np.clip(speed, min_speed, self.cfg.terminal_speed_max))

        speed = self.cfg.max_speed_cmd
        speed *= np.clip(goal_dist / 45.0, 0.72, 1.0)
        if goal_dist <= self.cfg.terminal_sampling_distance:
            speed = min(speed, self.cfg.terminal_speed_max + 0.05 * max(goal_dist - self.cfg.terminal_capture_distance, 0.0))
        if clear < self.cfg.clearance_slowdown_range:
            speed *= np.clip(clear / self.cfg.clearance_slowdown_range, 0.50, 1.0)
        if wall_margin < max(self.cfg.boundary_influence, self.cfg.wall_recovery_margin):
            speed *= np.clip(wall_margin / max(self.cfg.boundary_influence, self.cfg.wall_recovery_margin), 0.45, 1.0)
        speed *= 1.0 - 0.05 * abs(eta[4])

        if goal_dist > self.cfg.near_goal_relax_distance and clear > 1.5 and wall_margin > 1.5:
            speed = max(speed, self._required_body_speed(pos, self.env.step_count))
        return float(np.clip(speed, self._dynamic_min_speed(pos), self.cfg.max_speed_cmd))

    def _goal_direction(self, pos: np.ndarray) -> np.ndarray:
        waypoint = self._navigation_target()
        direction = waypoint - pos
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = self.env.target - pos
            norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return direction / norm

    def _goal_relative_body(self, pos: np.ndarray, eta: Optional[np.ndarray] = None) -> np.ndarray:
        eta_use = self.env.state[:6] if eta is None else np.asarray(eta, dtype=np.float64)
        rot = rotation_matrix_body_to_inertial(eta_use[3], eta_use[4], eta_use[5])
        return rot.T @ (self.env.target - pos)

    def _goal_error_components(self, pos: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
        rel = self.env.target - pos
        goal_dist = float(np.linalg.norm(rel))
        horiz_err = float(np.linalg.norm(rel[:2]))
        vert_err = float(rel[2])
        return goal_dist, horiz_err, vert_err, rel

    def _goal_velocity_components(
        self,
        pos: np.ndarray,
        eta: Optional[np.ndarray] = None,
        nu: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray, float, float, np.ndarray]:
        eta_use = self.env.state[:6] if eta is None else np.asarray(eta, dtype=np.float64)
        nu_use = self.env.state[6:] if nu is None else np.asarray(nu, dtype=np.float64)
        goal_dist, horiz_err, vert_err, rel = self._goal_error_components(pos)
        if goal_dist > 1e-8:
            r_hat = rel / goal_dist
        else:
            r_hat = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        rot = rotation_matrix_body_to_inertial(eta_use[3], eta_use[4], eta_use[5])
        v_inertial = rot @ nu_use[:3]
        v_r = float(np.dot(v_inertial, r_hat))
        tangential_vec = v_inertial - v_r * r_hat
        v_t = float(np.linalg.norm(tangential_vec))
        return goal_dist, horiz_err, vert_err, rel, v_inertial, v_r, v_t, tangential_vec

    def _capture_commit_mode(self, pos: np.ndarray, eta: Optional[np.ndarray] = None) -> bool:
        if self.recapture_active:
            return False
        goal_dist = float(np.linalg.norm(self.env.target - pos))
        best_dist = self.best_goal_dist if np.isfinite(self.best_goal_dist) else goal_dist
        if goal_dist <= self.cfg.capture_commit_distance or best_dist <= self.cfg.capture_commit_distance:
            clear = obstacle_clearance(pos, self.env.obstacles, self.env.auv_radius)
            wall_margin = boundary_margin(pos, self.env.world_size)
            if clear > 0.2 and wall_margin > 0.0:
                return True
        return False

    def _final_capture_mode(self, pos: np.ndarray, eta: Optional[np.ndarray] = None) -> bool:
        if self.recapture_active:
            return False
        goal_dist, horiz_err, vert_err, _ = self._goal_error_components(pos)
        if goal_dist > self.cfg.final_capture_distance:
            return False
        clear = obstacle_clearance(pos, self.env.obstacles, self.env.auv_radius)
        wall_margin = boundary_margin(pos, self.env.world_size)
        if clear < (self.cfg.obstacle_influence + 0.5):
            return False
        if wall_margin < (self.cfg.wall_recovery_margin + 0.5):
            return False
        if horiz_err <= self.cfg.final_capture_xy_window:
            return True
        return bool(goal_dist <= 1.6 * self.env.goal_radius and abs(vert_err) >= 0.75)

    def _final_settle_mode(self, pos: np.ndarray, eta: Optional[np.ndarray] = None, nu: Optional[np.ndarray] = None) -> bool:
        if not self._final_capture_mode(pos, eta):
            return False
        goal_dist, horiz_err, _vert_err, _rel, _v_inertial, v_r, v_t, _tvec = self._goal_velocity_components(pos, eta, nu)
        return bool(
            goal_dist <= self.cfg.final_capture_settle_distance
            and horiz_err <= self.cfg.final_capture_settle_xy_window
            and v_t <= self.cfg.final_capture_settle_tangential_speed
            and (-v_r) <= self.cfg.final_capture_settle_outward_speed
        )

    def _missed_capture(self, pos: np.ndarray, eta: Optional[np.ndarray] = None) -> bool:
        goal_dist = float(np.linalg.norm(self.env.target - pos))
        if goal_dist > self.cfg.terminal_sampling_distance and not self._capture_latch_mode(pos):
            return False
        best_dist = self.best_goal_dist if np.isfinite(self.best_goal_dist) else goal_dist
        if best_dist > self.cfg.missed_capture_distance and not self._capture_latch_mode(pos):
            return False
        rel_body = self._goal_relative_body(pos, eta)
        if self._capture_latch_mode(pos):
            delta = self.cfg.capture_latch_progress_margin
            forward_gate = 1.50
        else:
            delta = self.cfg.near_missed_capture_delta if best_dist <= self.cfg.final_capture_distance else 0.8
            forward_gate = 1.20 if best_dist <= self.cfg.final_capture_distance else 0.75
        return bool(goal_dist > best_dist + delta and rel_body[0] < forward_gate)

    def _wall_recovery_mode(self, pos: np.ndarray) -> bool:
        return bool(boundary_margin(pos, self.env.world_size) < self.cfg.wall_recovery_margin)

    def _wall_push_vector(self, pos: np.ndarray) -> np.ndarray:
        trigger = max(self.cfg.boundary_influence, self.cfg.wall_recovery_margin)
        push = np.zeros(3, dtype=np.float64)
        xy_margin = self.env.world_size - np.abs(pos[:2])
        for axis in range(2):
            if xy_margin[axis] < trigger:
                strength = (trigger - xy_margin[axis]) / max(trigger, 1e-6)
                push[axis] += -np.sign(pos[axis]) * strength
        if pos[2] < trigger:
            push[2] += (trigger - pos[2]) / max(trigger, 1e-6)
        top_margin = self.env.world_size - pos[2]
        if top_margin < trigger:
            push[2] -= (trigger - top_margin) / max(trigger, 1e-6)
        return push

    def _reference_angles(self, pos: np.ndarray, desired_speed: float) -> Tuple[float, float]:
        direction = self._goal_direction(pos)
        goal_dist, horiz_err, vert_err, rel, _, v_r, v_t, tangential_vec = self._goal_velocity_components(pos)
        final_capture = self._final_capture_mode(pos)
        final_settle = self._final_settle_mode(pos)
        capture_commit = self._capture_commit_mode(pos)
        desired_ground = desired_speed * direction
        current = self._current_from_profile(self.env.step_count)

        if self.recapture_active:
            desired_vec = rel.copy()
            if v_t > 1e-6:
                desired_vec = desired_vec - 0.90 * tangential_vec
            if v_r < 0.0:
                desired_vec = desired_vec + 0.95 * (-v_r) * (rel / max(goal_dist, 1e-6))
            if horiz_err <= self.cfg.capture_commit_xy_hold_window and abs(vert_err) > 0.8:
                desired_vec[:2] *= 0.62
            if self._wall_recovery_mode(pos):
                wall_push = self._wall_push_vector(pos)
                if np.linalg.norm(wall_push) > 1e-8:
                    desired_vec = desired_vec + 0.45 * max(desired_speed, 0.8) * wall_push
            if np.linalg.norm(desired_vec) < 1e-8:
                desired_vec = rel.copy()
            return course_to_angles(desired_vec)

        if final_capture or capture_commit:
            desired_vec = self.cfg.capture_pos_gain * rel.copy()
            tangent_gain = self.cfg.capture_tangent_cancel_gain
            outward_gain = self.cfg.capture_outward_cancel_gain
            if final_capture:
                tangent_gain *= 1.20
                outward_gain *= 1.15
                desired_vec = desired_vec - self.cfg.final_capture_current_comp_gain * current
            if v_t > 1e-6:
                desired_vec = desired_vec - tangent_gain * tangential_vec
            if v_r < 0.0:
                desired_vec = desired_vec + outward_gain * (-v_r) * (rel / max(goal_dist, 1e-6))
            if final_capture and horiz_err <= self.cfg.final_capture_xy_window and abs(vert_err) > 0.8:
                desired_vec[:2] *= 0.78
            if final_settle and horiz_err <= self.cfg.final_capture_settle_xy_window:
                desired_vec[:2] *= 0.60
            if horiz_err <= self.cfg.capture_commit_xy_hold_window and abs(vert_err) > 0.9:
                desired_vec[:2] *= 0.55
            if np.linalg.norm(desired_vec) < 1e-8:
                desired_vec = max(desired_speed, self.cfg.capture_commit_speed_min) * direction
            return course_to_angles(desired_vec)

        if self._wall_recovery_mode(pos):
            comp_gain = self.cfg.wall_current_comp_gain
        elif self._terminal_mode(pos) or self._missed_capture(pos):
            comp_gain = self.cfg.terminal_current_comp_gain
        else:
            comp_gain = self.cfg.current_comp_gain
        if goal_dist <= self.cfg.terminal_sampling_distance:
            comp_gain *= np.clip(goal_dist / max(self.cfg.terminal_sampling_distance, 1e-6), 0.50, 1.0)

        desired_rel = desired_ground - comp_gain * current
        if self._wall_recovery_mode(pos):
            wall_push = self._wall_push_vector(pos)
            if np.linalg.norm(wall_push) > 1e-8:
                desired_rel = desired_rel + 0.85 * max(desired_speed, 0.8) * wall_push
        if goal_dist <= self.cfg.terminal_capture_distance:
            direct = max(desired_speed, self.cfg.terminal_steer_speed_min) * direction
            alpha = np.clip(
                (self.cfg.terminal_capture_distance - goal_dist) / max(self.cfg.terminal_capture_distance, 1e-6),
                0.0,
                0.65,
            )
            desired_rel = (1.0 - alpha) * desired_rel + alpha * direct
        if np.linalg.norm(desired_rel) < 1e-6:
            desired_rel = max(desired_speed, self.cfg.terminal_steer_speed_min) * direction
        return course_to_angles(desired_rel)

    def _outward_speed(self, pos: np.ndarray, vel_inertial: np.ndarray) -> float:
        outward = 0.0
        xy_margin = self.env.world_size - np.abs(pos[:2])
        for axis in range(2):
            if xy_margin[axis] < self.cfg.wall_recovery_margin:
                outward = max(outward, float(np.sign(pos[axis]) * vel_inertial[axis]))
        if pos[2] < self.cfg.wall_recovery_margin:
            outward = max(outward, float(-vel_inertial[2]))
        if (self.env.world_size - pos[2]) < self.cfg.wall_recovery_margin:
            outward = max(outward, float(vel_inertial[2]))
        return max(0.0, outward)

    def _terminal_mode(self, pos: np.ndarray) -> bool:
        return bool(np.linalg.norm(self.env.target - pos) <= self.cfg.terminal_capture_distance)

    def _terminal_waypoint(self, pos: np.ndarray) -> np.ndarray:
        _ = pos
        return self.env.target.copy()

    def _dynamic_min_speed(self, pos: np.ndarray) -> float:
        goal_dist = float(np.linalg.norm(self.env.target - pos))
        wall_margin = boundary_margin(pos, self.env.world_size)
        clear = obstacle_clearance(pos, self.env.obstacles, self.env.auv_radius)
        if self.bypass_active:
            return self.cfg.bypass_speed_min
        if self.recapture_active:
            return max(self.cfg.terminal_steer_speed_min, self.cfg.recapture_reentry_speed_min - 0.10)
        if self._final_capture_mode(pos):
            return self.cfg.final_capture_speed_min
        if self._capture_commit_mode(pos):
            return self.cfg.capture_commit_speed_min
        if self._capture_latch_mode(pos):
            return self.cfg.capture_latch_speed_min
        if goal_dist <= self.cfg.terminal_capture_distance or self._missed_capture(pos):
            if goal_dist <= 2.0 * self.env.goal_radius:
                return 0.40
            return self.cfg.terminal_steer_speed_min
        if wall_margin < self.cfg.wall_recovery_margin or clear < 1.2:
            return self.cfg.wall_steer_speed_min
        return self.cfg.min_speed_cmd

    def _warm_start_mean(self, nominal_action: np.ndarray) -> np.ndarray:
        mean = np.zeros((self.cfg.horizon, 3), dtype=np.float64)
        mean[:] = nominal_action.reshape(1, 3)
        if self.cached_actions.shape[0] >= 2:
            keep = min(self.cfg.warm_start_keep, self.cfg.horizon - 1, self.cached_actions.shape[0] - 1)
            if keep > 0:
                mean[:keep] = self.cached_actions[1:1 + keep]
                mean[keep:] = self.cached_actions[min(keep, self.cached_actions.shape[0] - 1)]
        return mean

    def _sample_action_sequences(self, mean: np.ndarray, nominal_action: Optional[np.ndarray] = None) -> np.ndarray:
        current_pos = self.env.state[:3]
        num_samples, horizon, cem_iters = self._mpc_profile(current_pos)
        num_elites = max(4, int(self.cfg.elite_frac * num_samples))
        current_mean = mean[:horizon].copy()
        current_std = np.array([0.18, 0.28, 0.22], dtype=np.float64)
        best_samples = None
        best_costs = None
        best_cost_global = np.inf
        deadline = time.perf_counter() + 1e-3 * self.cfg.mpc_time_budget_ms

        for _ in range(cem_iters):
            if time.perf_counter() >= deadline:
                break
            noise = self.rng.normal(
                loc=0.0,
                scale=current_std.reshape(1, 1, 3),
                size=(num_samples, horizon, 3),
            )
            if horizon > 1:
                noise[:, 1:, :] = 0.65 * noise[:, 1:, :] + 0.35 * noise[:, :-1, :]
            samples = current_mean[None, :, :] + noise
            wall_margin = boundary_margin(current_pos, self.env.world_size)
            goal_dist = float(np.linalg.norm(self.env.target - current_pos))
            vel_inertial = rotation_matrix_body_to_inertial(self.env.state[3], self.env.state[4], self.env.state[5]) @ self.env.state[6:9]
            outward_speed = self._outward_speed(current_pos, vel_inertial)
            prop_floor = 0.15
            if goal_dist <= self.cfg.terminal_sampling_distance or self._missed_capture(current_pos):
                prop_floor = self.cfg.terminal_prop_floor
            if self._capture_latch_mode(current_pos):
                prop_floor = max(prop_floor, self.cfg.capture_latch_prop_floor)
            if self.recapture_active:
                prop_floor = max(prop_floor, self.cfg.recapture_reentry_prop_floor)
            if wall_margin < self.cfg.wall_recovery_margin:
                prop_floor = max(prop_floor, self.cfg.wall_prop_floor)
            if wall_margin < self.cfg.emergency_reverse_margin and outward_speed > self.cfg.emergency_reverse_outward_speed and goal_dist > self.cfg.terminal_capture_distance:
                prop_floor = -0.08
            samples[:, :, 0] = np.clip(samples[:, :, 0], prop_floor, 1.0)
            samples[:, :, 1] = np.clip(samples[:, :, 1], -1.0, 1.0)
            samples[:, :, 2] = np.clip(samples[:, :, 2], -1.0, 1.0)
            samples[0] = current_mean

            costs = np.full(num_samples, np.inf, dtype=np.float64)
            for i, sample in enumerate(samples):
                if time.perf_counter() >= deadline:
                    break
                costs[i] = self._evaluate_action_sequence(sample, max_cost=best_cost_global)
                if costs[i] < best_cost_global:
                    best_cost_global = float(costs[i])
            valid_idx = np.where(np.isfinite(costs))[0]
            if valid_idx.size == 0:
                break
            elite_idx = valid_idx[np.argsort(costs[valid_idx])[:min(num_elites, valid_idx.size)]]
            elites = samples[elite_idx]
            elite_mean = np.mean(elites, axis=0)
            current_mean = 0.35 * current_mean + 0.65 * elite_mean
            current_std = np.std(elites, axis=0).mean(axis=0)
            current_std = np.maximum(current_std, np.array([0.03, 0.08, 0.06], dtype=np.float64))
            best_samples = elites
            best_costs = costs[elite_idx]

        if best_samples is None or best_costs is None:
            if nominal_action is None:
                best_sequence = current_mean
            else:
                best_sequence = np.tile(np.asarray(nominal_action, dtype=np.float64), (horizon, 1))
        else:
            best_sequence = best_samples[int(np.argmin(best_costs))]

        padded = np.tile(best_sequence[-1], (self.cfg.horizon, 1))
        padded[:best_sequence.shape[0]] = best_sequence
        return padded

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

    def _evaluate_action_sequence(self, sequence: np.ndarray, max_cost: float = np.inf) -> float:
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
        best_rollout_goal = initial_goal_dist
        total_cost = 0.0
        early_cutoff = np.inf if not np.isfinite(max_cost) else (self.cfg.early_terminate_ratio * max_cost + self.cfg.early_terminate_abs_margin)
        capture_rollout = self.capture_latched or initial_goal_dist <= self.cfg.capture_latch_distance

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
            dist_goal, horiz_err, vert_err, _, vel_inertial, v_r, v_t, _ = self._goal_velocity_components(pos, eta, nu)
            progress = float(np.linalg.norm(prev_pos - target) - dist_goal)
            prev_pos = pos.copy()
            best_rollout_goal = min(best_rollout_goal, dist_goal)
            capture_commit = capture_rollout or dist_goal <= self.cfg.capture_commit_distance or best_rollout_goal <= self.cfg.capture_commit_distance

            if self.bypass_active:
                total_cost += self.cfg.bypass_waypoint_weight_scale * self.cfg.waypoint_weight * dist_wp
                total_cost += self.cfg.bypass_goal_stage_weight * self.cfg.stage_goal_weight * dist_goal
            else:
                total_cost += self.cfg.waypoint_weight * dist_wp
                total_cost += self.cfg.stage_goal_weight * dist_goal
            total_cost -= 18.0 * progress
            desired_yaw, desired_pitch = course_to_angles(waypoint - pos)
            total_cost += self.cfg.attitude_weight * (abs(eta[4] - desired_pitch) + 0.35 * abs(wrap_angle(eta[5] - desired_yaw)))
            total_cost += self.cfg.smooth_weight * np.sum((action - prev_action) ** 2)
            prev_action = np.asarray(action, dtype=np.float64)

            time_left = max((self.env.max_steps - step_count) * self.env.dt, self.plan_dt_effective)
            required_closing = dist_goal / time_left
            closing_speed = max(progress / self.plan_dt_effective, 0.0)
            rel_body = self._goal_relative_body(pos, eta)
            forward_err = float(rel_body[0])
            if dist_goal > self.cfg.near_goal_relax_distance and not capture_commit:
                deficit = max(0.0, required_closing - closing_speed)
                total_cost += self.cfg.closing_speed_weight * (deficit ** 2) * self.plan_dt_effective
            else:
                total_cost += 0.20 * max(0.0, self.cfg.terminal_prop_floor - float(action[0])) ** 2
                if dist_goal > self.env.goal_radius:
                    total_cost += 1.8 * max(0.0, 0.18 - closing_speed) ** 2
                if dist_goal < 3.0:
                    total_cost += 0.10 * max(0.0, float(action[0]) - 0.60) ** 2
                if forward_err < 0.0:
                    total_cost += 0.12 * max(0.0, float(action[0]) - 0.45) ** 2
                if best_rollout_goal <= self.cfg.missed_capture_distance and dist_goal > best_rollout_goal + 0.8 and forward_err < 0.75:
                    total_cost += 0.80 * (dist_goal - best_rollout_goal) ** 2
            if capture_commit:
                total_cost += self.cfg.capture_tangential_weight * (v_t ** 2)
                total_cost += self.cfg.capture_outward_weight * (max(0.0, -v_r) ** 2)
                prop_ceiling = np.clip(
                    self.cfg.capture_commit_prop_ceiling
                    - self.cfg.capture_commit_prop_tangent_gain * v_t
                    - self.cfg.capture_commit_prop_outward_gain * max(0.0, -v_r),
                    self.cfg.capture_commit_prop_floor,
                    self.cfg.capture_commit_prop_ceiling,
                )
                total_cost += self.cfg.capture_prop_ceiling_weight * max(0.0, float(action[0]) - prop_ceiling) ** 2
                if dist_goal > best_rollout_goal + 0.35:
                    total_cost += self.cfg.capture_progress_weight * (dist_goal - best_rollout_goal) ** 2
            if capture_rollout and dist_goal > best_rollout_goal + self.cfg.capture_latch_progress_margin:
                total_cost += self.cfg.capture_latch_penalty * (dist_goal - best_rollout_goal) ** 2

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
            outward_speed = self._outward_speed(pos, vel_inertial)
            if b_margin < self.cfg.wall_recovery_margin and outward_speed > 0.0:
                total_cost += self.cfg.wall_outward_speed_weight * (self.cfg.wall_recovery_margin - b_margin) * (outward_speed ** 2)
                wall_prop_cap = 0.25 if outward_speed < 0.05 else 0.12
                total_cost += 0.18 * max(0.0, float(action[0]) - wall_prop_cap) ** 2
            if b_margin < 0.0:
                total_cost += self.cfg.oob_penalty
                break

            if dist_goal <= self.env.goal_radius:
                total_cost -= self.cfg.goal_bonus
                break
            if total_cost > early_cutoff:
                return float(total_cost)

        terminal_goal = np.linalg.norm(eta[:3] - target)
        if capture_rollout:
            total_cost += self.cfg.capture_latch_penalty * terminal_goal
        total_cost += self.cfg.terminal_goal_weight * terminal_goal
        total_cost -= 6.0 * (initial_goal_dist - terminal_goal)
        return float(total_cost)

    def _low_level_action(self, desired_speed: float, desired_yaw: float, desired_pitch: float) -> np.ndarray:
        eta = self.env.state[:6]
        nu = self.env.state[6:]
        yaw_cmd = wrap_angle(desired_yaw)
        pitch_cmd = float(np.clip(desired_pitch, -0.45, 0.45))
        yaw_error = wrap_angle(yaw_cmd - eta[5])

        pitch_error_term = eta[4] - pitch_cmd
        clear = obstacle_clearance(eta[:3], self.env.obstacles, self.env.auv_radius)
        wall_margin = boundary_margin(eta[:3], self.env.world_size)
        goal_dist, horiz_err, vert_err, _, vel_inertial, v_r, v_t, tangential_vec = self._goal_velocity_components(eta[:3], eta, nu)
        rel_body = self._goal_relative_body(eta[:3], eta)
        forward_err = float(rel_body[0])
        lateral_err = float(rel_body[1])
        vertical_err = float(rel_body[2])
        missed_capture = self._missed_capture(eta[:3], eta)
        final_capture = self._final_capture_mode(eta[:3], eta)
        final_settle = self._final_settle_mode(eta[:3], eta, nu)
        capture_commit = self._capture_commit_mode(eta[:3], eta)
        recapture_active = self.recapture_active
        vel_inertial = rotation_matrix_body_to_inertial(eta[3], eta[4], eta[5]) @ nu[:3]
        outward_speed = self._outward_speed(eta[:3], vel_inertial)

        speed_cap = desired_speed
        speed_cap /= 1.0 + self.cfg.heading_slowdown_gain * abs(yaw_error)
        speed_cap /= 1.0 + self.cfg.pitch_slowdown_gain * abs(pitch_error_term)
        if abs(yaw_error) > 0.95:
            speed_cap *= 0.58
        elif abs(yaw_error) > 0.55:
            speed_cap *= 0.74
        if abs(pitch_error_term) > 0.30:
            speed_cap *= 0.72
        if clear < self.cfg.clearance_slowdown_range:
            speed_cap *= np.clip(clear / self.cfg.clearance_slowdown_range, 0.35, 1.0)
        if wall_margin < max(self.cfg.boundary_influence, self.cfg.wall_recovery_margin):
            speed_cap *= np.clip(wall_margin / max(self.cfg.boundary_influence, self.cfg.wall_recovery_margin), 0.35, 1.0)
        if wall_margin < self.cfg.wall_recovery_margin and outward_speed > 0.05:
            speed_cap *= 1.0 / (1.0 + 0.8 * outward_speed)

        if final_capture:
            speed_cap = float(np.clip(desired_speed, self.cfg.final_capture_speed_min, self.cfg.final_capture_speed_max))
            speed_cap -= 0.06 * v_t
            speed_cap -= 0.10 * max(0.0, -v_r)
            speed_cap = float(np.clip(speed_cap, self.cfg.final_capture_speed_min, self.cfg.final_capture_speed_max))
        elif capture_commit:
            speed_cap = float(np.clip(desired_speed, self.cfg.capture_commit_speed_min, self.cfg.capture_commit_speed_max))
            speed_cap -= 0.22 * v_t
            speed_cap -= 0.40 * max(0.0, -v_r)
            if horiz_err < self.cfg.capture_commit_xy_hold_window and abs(vert_err) > 0.9:
                speed_cap = min(speed_cap, 0.76)
            speed_cap = float(np.clip(speed_cap, self.cfg.capture_commit_speed_min, self.cfg.capture_commit_speed_max))
        elif goal_dist <= self.cfg.terminal_capture_distance or missed_capture:
            terminal_cap = desired_speed
            if abs(lateral_err) > 1.25 or abs(vertical_err) > 1.25:
                terminal_cap = min(terminal_cap, 0.80)
            if forward_err < -0.25:
                terminal_cap = min(terminal_cap, self.cfg.terminal_steer_speed_min)
            if missed_capture:
                terminal_cap = min(terminal_cap, 0.82)
            speed_cap = min(speed_cap, terminal_cap)
        if wall_margin < self.cfg.wall_recovery_margin and outward_speed > 0.05:
            speed_cap = min(speed_cap, 0.85)

        min_speed = self._dynamic_min_speed(eta[:3])
        if goal_dist > self.cfg.near_goal_relax_distance and clear > 1.2 and wall_margin > self.cfg.wall_safe_margin:
            speed_cap = max(speed_cap, min(self.cfg.turn_speed_floor, desired_speed))
        speed_cap = float(np.clip(speed_cap, min_speed, self.cfg.max_speed_cmd))

        recover_mode = recapture_active or goal_dist <= self.cfg.terminal_capture_distance or missed_capture or wall_margin < self.cfg.wall_recovery_margin
        yaw_kp = self.cfg.yaw_kp * (1.20 if recover_mode else 1.0)
        pitch_kp = self.cfg.pitch_kp * (1.10 if goal_dist <= self.cfg.terminal_capture_distance else 1.0)
        if capture_commit:
            yaw_kp *= 0.95
            pitch_kp *= self.cfg.capture_commit_pitch_gain
        if final_capture:
            yaw_kp *= 1.05
            if v_t > 0.18 or v_r < -0.05:
                yaw_kp *= 1.10
            pitch_kp *= self.cfg.final_capture_pitch_gain

        propeller = (
            self.cfg.prop_base
            + self.cfg.prop_speed_kp * (speed_cap / self.cfg.max_speed_cmd)
            + self.cfg.prop_surge_kp * (speed_cap - nu[0])
        )
        if goal_dist <= self.cfg.terminal_capture_distance and abs(yaw_error) > 0.45 and not final_capture:
            propeller -= 0.18 * (abs(yaw_error) - 0.45)
        if capture_commit:
            prop_ceiling = np.clip(
                self.cfg.capture_commit_prop_ceiling
                - self.cfg.capture_commit_prop_tangent_gain * v_t
                - self.cfg.capture_commit_prop_outward_gain * max(0.0, -v_r),
                self.cfg.capture_commit_prop_floor,
                self.cfg.capture_commit_prop_ceiling,
            )
            propeller = min(propeller, prop_ceiling)
        if missed_capture and forward_err < 0.25 and not final_capture:
            propeller -= 0.10 * (0.25 - forward_err)
        if wall_margin < self.cfg.wall_recovery_margin + 1.0:
            propeller -= 0.18 * max(0.0, outward_speed)

        prop_floor = -self.env.max_reverse_propeller
        if goal_dist <= self.cfg.terminal_sampling_distance or missed_capture:
            prop_floor = self.cfg.terminal_prop_floor
        if capture_commit:
            prop_floor = max(prop_floor, self.cfg.capture_commit_prop_floor)
        if recapture_active:
            prop_floor = max(prop_floor, self.cfg.recapture_reentry_prop_floor)
            propeller = min(propeller, self.cfg.recapture_reentry_speed_max)
        if final_capture:
            prop_floor = max(prop_floor, self.cfg.final_capture_prop_floor)
            if v_t > 0.15 or v_r < -0.04:
                propeller = max(propeller, self.cfg.final_capture_prop_floor + 0.06)
                propeller = min(propeller, min(0.78, self.cfg.final_capture_speed_max))
        if wall_margin < self.cfg.wall_recovery_margin:
            prop_floor = max(prop_floor, self.cfg.wall_prop_floor)
        if wall_margin < self.cfg.emergency_reverse_margin and outward_speed > self.cfg.emergency_reverse_outward_speed and goal_dist > self.cfg.terminal_capture_distance:
            prop_floor = -0.08
        if capture_commit and goal_dist < self.cfg.capture_settle_distance and v_r < -self.cfg.capture_reverse_outward_trigger and v_t < 0.22:
            prop_floor = min(prop_floor, -self.cfg.capture_reverse_max)
            propeller = min(propeller, 0.0)
        propeller = np.clip(propeller, prop_floor, 1.0)

        rudder = yaw_kp * yaw_error - self.cfg.yaw_rate_kd * nu[5] - self.cfg.sway_kd * nu[1]
        stern = pitch_kp * pitch_error_term + self.cfg.pitch_rate_kd * nu[4] + self.cfg.heave_kd * nu[2]
        if capture_commit:
            rudder = float(np.clip(rudder, -self.cfg.capture_commit_rudder_limit, self.cfg.capture_commit_rudder_limit))
            if horiz_err < self.cfg.capture_commit_xy_hold_window:
                rudder = float(np.clip(rudder, -self.cfg.capture_commit_rudder_close_limit, self.cfg.capture_commit_rudder_close_limit))
            if abs(vert_err) > 0.9:
                stern *= 1.12
        if final_capture:
            rudder_limit = self.cfg.final_capture_rudder_limit if not final_settle else min(self.cfg.final_capture_rudder_limit, 0.38)
            rudder = float(np.clip(rudder, -rudder_limit, rudder_limit))
            if abs(pitch_cmd) > 0.18:
                stern_bias = -np.sign(pitch_cmd) * self.cfg.final_capture_stern_bias
                if np.sign(stern) != np.sign(stern_bias):
                    stern = stern_bias
                else:
                    stern = stern + stern_bias

        rudder_limit = 1.0
        if recapture_active:
            rudder_limit = min(rudder_limit, self.cfg.recapture_reentry_rudder_limit)

        action = np.array(
            [
                propeller,
                np.clip(rudder, -rudder_limit, rudder_limit),
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
        wall_margin = boundary_margin(pos, self.env.world_size)
        if self.recapture_active:
            self.bypass_active = False
            self.cached_path = [self.env.target.copy()]
            self.cached_waypoint = self.env.target.copy()
            self.last_plan_step = int(self.env.step_count)
            return

        blocker = None
        if goal_dist > self.cfg.terminal_capture_distance:
            blocker = self._find_blocking_obstacle(pos, self.env.target, self.env.obstacles)
        if blocker is not None:
            self.bypass_active = True
            self.cached_path = self._build_bypass_path(pos, self.env.target, blocker)
            self.cached_waypoint = self.cached_path[0].copy() if self.cached_path else self.env.target.copy()
            self.last_plan_step = int(self.env.step_count)
            return

        self.bypass_active = False
        if self._capture_commit_mode(pos) or self._capture_latch_mode(pos) or goal_dist <= self.cfg.terminal_capture_distance or self._missed_capture(pos):
            self.cached_path = [self.env.target.copy()]
            self.cached_waypoint = self._terminal_waypoint(pos)
            self.last_plan_step = int(self.env.step_count)
            return

        lookahead = self.cfg.waypoint_lookahead
        if clear > self.cfg.clearance_slowdown_range and goal_dist > 2.0 * self.cfg.waypoint_lookahead and wall_margin > self.cfg.shortcut_boundary_margin:
            lookahead = min(self.cfg.shortcut_max_factor * self.cfg.waypoint_lookahead, self.cfg.waypoint_lookahead + 0.08 * goal_dist)
        if wall_margin < self.cfg.wall_recovery_margin:
            lookahead = min(lookahead, 0.70 * self.cfg.waypoint_lookahead)

        self.cached_path = self.planner.plan(pos, self.env.target, self.env.obstacles)
        self.cached_waypoint = self.planner.pick_waypoint(self.cached_path, pos, self.env.target, lookahead=lookahead)
        self.last_plan_step = int(self.env.step_count)

    def _sampling_needed(self, pos: np.ndarray) -> bool:
        clear = obstacle_clearance(pos, self.env.obstacles, self.env.auv_radius)
        wall_margin = boundary_margin(pos, self.env.world_size)
        goal_dist = float(np.linalg.norm(self.env.target - pos))
        if self.bypass_active:
            return True
        if self.recapture_active and clear > (self.cfg.obstacle_influence + 1.0) and wall_margin > (self.cfg.wall_recovery_margin + 1.0):
            return False
        if self._final_capture_mode(pos):
            return not self._final_settle_mode(pos)
        if self._capture_commit_mode(pos) and clear > (self.cfg.obstacle_influence + 1.0) and wall_margin > (self.cfg.wall_recovery_margin + 1.0):
            return False
        if self._capture_latch_mode(pos):
            return True
        if goal_dist < self.cfg.inner_terminal_nominal_distance and clear > (self.cfg.obstacle_influence + 1.0) and wall_margin > (self.cfg.wall_recovery_margin + 1.0):
            return False
        return bool(
            clear < (self.cfg.obstacle_influence + 1.0)
            or wall_margin < max(self.cfg.boundary_influence + 1.0, self.cfg.wall_recovery_margin)
            or goal_dist < self.cfg.terminal_sampling_distance
        )

    def act(self) -> np.ndarray:
        t0 = time.perf_counter()
        pos = self.env.state[:3]
        goal_dist_now = float(np.linalg.norm(self.env.target - pos))
        self.best_goal_dist = min(self.best_goal_dist, goal_dist_now)
        self._update_capture_latch(pos, goal_dist_now)
        self._update_recapture_state(pos, self.env.state[:6], self.env.state[6:])
        self._maybe_replan_path()

        pos = self.env.state[:3]
        eta = self.env.state[:6]
        if self.bypass_active:
            self.current_mode = "BYPASS"
        elif self.recapture_active:
            self.current_mode = "RECAP"
        elif self._final_capture_mode(pos, eta):
            self.current_mode = "FINAL"
        elif self._capture_commit_mode(pos, eta):
            self.current_mode = "CAPT"
        elif self._wall_recovery_mode(pos):
            self.current_mode = "WALL"
        else:
            self.current_mode = "CRUISE"
        speed_ref = self._reference_speed(pos, eta)
        ref_yaw, ref_pitch = self._reference_angles(pos, speed_ref)
        nominal_action = self._low_level_action(speed_ref, ref_yaw, ref_pitch).astype(np.float64)

        if not self._sampling_needed(pos):
            self.cached_actions = self._warm_start_mean(nominal_action)
            self.last_act_time = time.perf_counter() - t0
            return np.asarray(nominal_action, dtype=np.float32)

        mean = self._warm_start_mean(nominal_action)
        best_sequence = self._sample_action_sequences(mean, nominal_action)
        self.cached_actions = best_sequence.copy()

        if self.bypass_active:
            blend_nominal = 0.90
        elif self.recapture_active:
            blend_nominal = 0.92
        elif self._capture_commit_mode(pos):
            blend_nominal = 0.85
        elif self._terminal_mode(pos) or self._missed_capture(pos):
            blend_nominal = 0.65
        elif boundary_margin(pos, self.env.world_size) < self.cfg.wall_recovery_margin:
            blend_nominal = 0.60
        else:
            blend_nominal = 0.75
        blended = blend_nominal * nominal_action + (1.0 - blend_nominal) * best_sequence[0]
        blended[0] = np.clip(blended[0], -self.env.max_reverse_propeller, 1.0)
        blended[1:] = np.clip(blended[1:], -1.0, 1.0)
        self.last_act_time = time.perf_counter() - t0
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
            fixed_mid_obstacle=args.fixed_mid_obstacle,
            fixed_obstacle_radius=args.fixed_obstacle_radius,
            seed=env_seed,
        )
        obs, info = env.reset(seed=env_seed)
        controller = PathGuidedSamplingMPC(env, config=config, seed=env_seed)
        controller.reset()

        episode_return = 0.0
        event = "timeout"
        path_log: List[List[float]] = [env.state[:3].astype(float).tolist()]
        obstacle_log: List[List[List[float]]] = []

        if not args.quiet:
            fixed_obs_text = ""
            if bool(info.get('fixed_mid_obstacle', False)):
                fixed_center = info.get('fixed_obstacle_center', None)
                fixed_radius = float(info.get('fixed_obstacle_radius', 0.0))
                if fixed_center is not None:
                    fixed_obs_text = f" | fixed_obs=center={format_vec3(np.asarray(fixed_center))}, r={fixed_radius:.2f}"
                else:
                    fixed_obs_text = f" | fixed_obs=r={fixed_radius:.2f}"
            print(
                f"\n=== Episode {episode + 1}/{args.episodes} | seed={env_seed} ===\n"
                f"start={format_vec3(env.start)} | target={format_vec3(env.target)} "
                f"| init_dist={np.linalg.norm(env.target - env.start):.2f} m "
                f"| current={format_vec3(info.get('current_inertial', np.zeros(3)))}"
                f"{fixed_obs_text}",
                flush=True,
            )

        for _ in range(env.max_steps):
            action = controller.act()
            obs, reward, terminated, truncated, step_info = env.step(action)
            step_info['mode'] = controller.current_mode
            del obs
            episode_return += float(reward)
            path_log.append(env.state[:3].astype(float).tolist())
            obstacle_log.append([obs_item.center.astype(float).tolist() for obs_item in env.obstacles])

            if (not args.quiet) and args.log_every > 0 and (env.step_count == 1 or env.step_count % args.log_every == 0):
                maybe_print_progress(
                    episode=episode,
                    total_episodes=args.episodes,
                    env=env,
                    episode_return=episode_return,
                    step_info=step_info,
                )

            if terminated or truncated:
                event = str(step_info.get("event", "timeout" if truncated else "other"))
                if not args.quiet:
                    maybe_print_progress(
                        episode=episode,
                        total_episodes=args.episodes,
                        env=env,
                        episode_return=episode_return,
                        step_info=step_info,
                        log_prefix=f"[{event.upper()}] ",
                    )
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

        if not args.quiet:
            print(
                f"--- Episode {episode + 1}/{args.episodes} done | event={event} | steps={env.step_count} "
                f"| final_dist={final_dist:.2f} m | return={episode_return:.2f} ---",
                flush=True,
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
            "fixed_mid_obstacle": args.fixed_mid_obstacle,
            "fixed_obstacle_radius": args.fixed_obstacle_radius if args.fixed_mid_obstacle else 0.0,
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
    parser.add_argument("--env-max-steps", type=int, default=3600, help="Maximum environment steps.")
    parser.add_argument("--n-obstacles", type=int, default=6, help="Number of dynamic obstacles.")
    parser.add_argument("--fixed-mid-obstacle", action="store_true", help="Place one fixed obstacle at the midpoint between start and target.")
    parser.add_argument("--fixed-obstacle-radius", type=float, default=6.0, help="Radius of the fixed midpoint obstacle.")
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
    parser.add_argument("--waypoint-lookahead", type=float, default=26.0, help="Distance used to select the path waypoint.")
    parser.add_argument("--log-every", type=int, default=100, help="Print progress every N environment steps. Set 0 to disable periodic step logs.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-episode progress logs and only print the final summary.")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_controller(parse_args())
