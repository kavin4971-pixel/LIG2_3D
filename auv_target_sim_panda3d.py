from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock
from panda3d.core import (
    AmbientLight,
    CardMaker,
    DirectionalLight,
    LineSegs,
    TextNode,
    TransparencyAttrib,
    Vec3,
    WindowProperties,
)

from auv_rl_collision_train import (
    AUVNavigationRLEnv,
    AUVSimConfig,
    LayoutConfig,
    ObstacleFieldConfig,
    SpawnConfig,
)


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def layout_config_from_dict(data: dict[str, Any] | None) -> LayoutConfig:
    merged = asdict(LayoutConfig())
    if data:
        _deep_update(merged, data)
    return LayoutConfig(
        size=tuple(float(v) for v in merged["size"]),
        origin=tuple(float(v) for v in merged["origin"]),
        obstacle_field=ObstacleFieldConfig(**merged["obstacle_field"]),
        spawn=SpawnConfig(**merged["spawn"]),
        seed=int(merged["seed"]),
    )


def sim_config_from_dict(data: dict[str, Any] | None) -> AUVSimConfig:
    merged = asdict(AUVSimConfig())
    if data:
        _deep_update(merged, data)
    return AUVSimConfig(**merged)


def resolve_model_path(model_path: str | Path) -> Path:
    raw = Path(model_path)
    candidates = [raw]
    if raw.suffix != ".zip":
        candidates.insert(0, Path(f"{raw}.zip"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_config_path(model_path: Path, config_path: str | Path | None) -> Path | None:
    if config_path is None:
        sibling = model_path.parent / "training_config.json"
        return sibling if sibling.exists() else None
    path = Path(config_path)
    return path if path.exists() else None


def load_training_configs(
    model_path: Path,
    config_path: str | Path | None,
) -> tuple[LayoutConfig, AUVSimConfig, Path | None]:
    resolved_config = resolve_config_path(model_path, config_path)
    if resolved_config is None:
        return LayoutConfig(), AUVSimConfig(), None

    payload = json.loads(resolved_config.read_text(encoding="utf-8"))
    layout_cfg = layout_config_from_dict(payload.get("layout_config"))
    sim_cfg = sim_config_from_dict(payload.get("sim_config"))
    return layout_cfg, sim_cfg, resolved_config


def load_ppo_model(model_path: Path, device: str = "auto"):
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover - runtime dependency.
        raise RuntimeError(
            "stable-baselines3 is required to run PPO inference. "
            "Install it with: pip install 'stable-baselines3[extra]'"
        ) from exc

    return PPO.load(str(model_path), env=None, device=device)


class AUVTargetSim(ShowBase):
    """Panda3D viewer that runs a trained PPO policy inside the RL environment.

    The key design choice is to reuse ``AUVNavigationRLEnv`` from the training
    script directly. That keeps the Panda3D demo and the RL environment aligned:
    same observation vector, same obstacle motion rules, and same hard-failure
    collision logic.
    """

    def __init__(
        self,
        *,
        model_path: Path,
        layout_config: LayoutConfig,
        sim_config: AUVSimConfig,
        deterministic_policy: bool = True,
        auto_reset_delay: float = 1.25,
        device: str = "auto",
        base_seed: int | None = None,
    ) -> None:
        super().__init__()

        self.model_path = Path(model_path)
        self.model = load_ppo_model(self.model_path, device=device)
        self.layout_config = layout_config
        self.sim_config = sim_config
        self.deterministic_policy = bool(deterministic_policy)
        self.auto_reset_delay = float(auto_reset_delay)
        self.base_seed = int(sim_config.seed if base_seed is None else base_seed)

        self.rl_env = AUVNavigationRLEnv(
            layout_config=self.layout_config,
            sim_config=self.sim_config,
        )

        # Runtime state.
        self.current_obs = np.zeros(self.rl_env.observation_space.shape, dtype=np.float32)
        self.last_info: dict[str, Any] = {}
        self.last_reward = 0.0
        self.last_action = np.zeros(3, dtype=float)
        self.episode_done = False
        self.paused = False
        self.reset_timer = 0.0
        self.episode_number = 0
        self.success_count = 0
        self.failure_count = 0
        self.timeout_count = 0
        self._sim_accumulator = 0.0
        self._max_steps_per_frame = 4

        # Trail state.
        self.trail_points: list[tuple[float, float, float]] = []
        self.max_trail_points = 350
        self.trail_np = self.render.attachNewNode("auv-trail")

        # Visual obstacle nodes.
        self.obstacle_root_np = self.render.attachNewNode("obstacles")
        self.obstacle_nodes: list[Any] = []

        self._configure_window()
        self.disableMouse()
        self.setBackgroundColor(0.01, 0.06, 0.12, 1.0)
        self.camLens.setFov(75)
        self.camLens.setNearFar(0.1, 1000)

        self._setup_lights()
        self._build_environment()
        self._build_auv()
        self._build_target()
        self._build_ui()
        self._bind_keys()

        self.reset_episode()
        self._reset_camera()
        self.taskMgr.add(self._update, "update-auv-ppo")

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------
    @property
    def min_bound(self) -> np.ndarray:
        return np.asarray(self.layout_config.origin, dtype=float)

    @property
    def max_bound(self) -> np.ndarray:
        return self.min_bound + np.asarray(self.layout_config.size, dtype=float)

    @property
    def env_volume(self) -> float:
        return float(np.prod(np.asarray(self.layout_config.size, dtype=float)))

    @property
    def actual_obstacle_complexity(self) -> float:
        if self.env_volume <= 0.0:
            return 0.0
        total = 0.0
        for obstacle in self.rl_env.obstacles:
            total += (4.0 / 3.0) * math.pi * (float(obstacle.radius) ** 3)
        return total / self.env_volume

    def _configure_window(self) -> None:
        props = WindowProperties()
        props.setTitle("AUV Target Simulation - PPO Policy Viewer")
        props.setSize(1440, 840)
        self.win.requestProperties(props)

    def _setup_lights(self) -> None:
        ambient = AmbientLight("ambient")
        ambient.setColor((0.45, 0.50, 0.55, 1.0))
        ambient_np = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_np)

        sun = DirectionalLight("sun")
        sun.setColor((0.88, 0.88, 0.82, 1.0))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(-35, -55, 0)
        self.render.setLight(sun_np)

    def _build_environment(self) -> None:
        min_b = self.min_bound
        max_b = self.max_bound

        box_np = self._make_wire_box(min_b, max_b, thickness=2.0)
        box_np.reparentTo(self.render)
        box_np.setColor(0.25, 0.80, 1.00, 1.0)

        grid_np = self._make_floor_grid(
            min_x=float(min_b[0]),
            max_x=float(max_b[0]),
            min_y=float(min_b[1]),
            max_y=float(max_b[1]),
            z=float(min_b[2]),
            spacing=2.0,
        )
        grid_np.reparentTo(self.render)
        grid_np.setColor(0.10, 0.28, 0.38, 1.0)

        surface = self._make_water_surface(
            min_x=float(min_b[0]),
            max_x=float(max_b[0]),
            min_y=float(min_b[1]),
            max_y=float(max_b[1]),
            z=float(max_b[2]),
        )
        surface.reparentTo(self.render)

    def _build_auv(self) -> None:
        self.auv_np = self.render.attachNewNode("auv")

        self.auv_body = self.loader.loadModel("models/box")
        self.auv_body.reparentTo(self.auv_np)
        self.auv_body.setScale(0.45, 1.10, 0.25)
        self.auv_body.setColor(0.95, 0.78, 0.18, 1.0)

        self.auv_nose = self.loader.loadModel("models/misc/sphere")
        self.auv_nose.reparentTo(self.auv_np)
        self.auv_nose.setScale(0.18)
        self.auv_nose.setPos(0.0, 1.20, 0.0)
        self.auv_nose.setColor(0.98, 0.30, 0.22, 1.0)

        fin = self.loader.loadModel("models/box")
        fin.reparentTo(self.auv_np)
        fin.setScale(0.60, 0.06, 0.02)
        fin.setPos(0.0, -0.25, 0.18)
        fin.setColor(0.20, 0.25, 0.30, 1.0)

        left_tail = self.loader.loadModel("models/box")
        left_tail.reparentTo(self.auv_np)
        left_tail.setScale(0.02, 0.18, 0.14)
        left_tail.setPos(-0.22, -1.05, 0.0)
        left_tail.setColor(0.20, 0.25, 0.30, 1.0)

        right_tail = self.loader.loadModel("models/box")
        right_tail.reparentTo(self.auv_np)
        right_tail.setScale(0.02, 0.18, 0.14)
        right_tail.setPos(0.22, -1.05, 0.0)
        right_tail.setColor(0.20, 0.25, 0.30, 1.0)

    def _build_target(self) -> None:
        self.target_np = self.loader.loadModel("models/misc/sphere")
        self.target_np.reparentTo(self.render)
        self.target_np.setScale(self.sim_config.target_radius)
        self.target_np.setColor(1.00, 0.15, 0.15, 1.0)

        self.target_keepout_np = self.loader.loadModel("models/misc/sphere")
        self.target_keepout_np.reparentTo(self.render)
        self.target_keepout_np.setScale(
            self.sim_config.target_radius + self.sim_config.obstacle_target_keepout
        )
        self.target_keepout_np.setColor(1.0, 0.15, 0.15, 0.08)
        self.target_keepout_np.setTransparency(TransparencyAttrib.MAlpha)

        self.target_ring_np = self.render.attachNewNode("target-ring")

    def _build_ui(self) -> None:
        self.hud_text = OnscreenText(
            text="",
            pos=(-1.30, 0.93),
            scale=0.040,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft,
            mayChange=True,
        )
        self.help_text = OnscreenText(
            text="[SPACE] pause/resume   [R] reset episode   [D] deterministic on/off   [C] reset camera   [ESC] quit",
            pos=(-1.30, 0.84),
            scale=0.032,
            fg=(0.75, 0.90, 1.00, 1.0),
            align=TextNode.ALeft,
            mayChange=True,
        )

    def _bind_keys(self) -> None:
        self.accept("space", self._toggle_pause)
        self.accept("r", self.reset_episode)
        self.accept("d", self._toggle_deterministic)
        self.accept("c", self._reset_camera)
        self.accept("escape", self.userExit)

    def _make_wire_box(
        self,
        min_bound: Iterable[float],
        max_bound: Iterable[float],
        thickness: float = 1.0,
    ):
        min_x, min_y, min_z = map(float, min_bound)
        max_x, max_y, max_z = map(float, max_bound)

        pts = [
            (min_x, min_y, min_z),
            (max_x, min_y, min_z),
            (max_x, max_y, min_z),
            (min_x, max_y, min_z),
            (min_x, min_y, max_z),
            (max_x, min_y, max_z),
            (max_x, max_y, max_z),
            (min_x, max_y, max_z),
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        segs = LineSegs("env-box")
        segs.setThickness(thickness)
        for i, j in edges:
            segs.moveTo(*pts[i])
            segs.drawTo(*pts[j])
        return self.render.attachNewNode(segs.create())

    def _make_floor_grid(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        z: float,
        spacing: float,
    ):
        segs = LineSegs("floor-grid")
        segs.setThickness(1.0)

        x = min_x
        while x <= max_x + 1e-6:
            segs.moveTo(x, min_y, z)
            segs.drawTo(x, max_y, z)
            x += spacing

        y = min_y
        while y <= max_y + 1e-6:
            segs.moveTo(min_x, y, z)
            segs.drawTo(max_x, y, z)
            y += spacing

        return self.render.attachNewNode(segs.create())

    def _make_water_surface(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        z: float,
    ):
        card = CardMaker("water-surface")
        card.setFrame(min_x, max_x, min_y, max_y)
        np_card = self.render.attachNewNode(card.generate())
        np_card.setP(-90)
        np_card.setPos(0, 0, z)
        np_card.setColor(0.10, 0.35, 0.60, 0.18)
        np_card.setTransparency(TransparencyAttrib.MAlpha)
        np_card.setTwoSided(True)
        return np_card

    # ------------------------------------------------------------------
    # Episode control / environment sync
    # ------------------------------------------------------------------
    def reset_episode(self) -> None:
        self.episode_number += 1
        seed = self.base_seed + self.episode_number - 1
        self.current_obs, self.last_info = self.rl_env.reset(seed=seed)
        self.last_reward = 0.0
        self.last_action = np.zeros(3, dtype=float)
        self.episode_done = False
        self.reset_timer = 0.0
        self._sim_accumulator = 0.0

        self.trail_points.clear()
        self._rebuild_trail()
        self._sync_scene_from_env(rebuild_obstacles=True)
        self._append_trail_point(force=True)
        self._set_auv_visual_state("running")

    def _sync_scene_from_env(self, *, rebuild_obstacles: bool = False) -> None:
        agent_pos = self.rl_env.agent_pos
        target_pos = self.rl_env.target_pos
        self.auv_np.setPos(*agent_pos.tolist())
        self.target_np.setPos(*target_pos.tolist())
        self.target_keepout_np.setPos(*target_pos.tolist())

        velocity = self.rl_env.agent_vel
        speed = float(np.linalg.norm(velocity))
        if speed > 1e-6:
            self.auv_np.lookAt(
                self.auv_np.getPos(self.render)
                + Vec3(float(velocity[0]), float(velocity[1]), float(velocity[2]))
            )

        if rebuild_obstacles or len(self.obstacle_nodes) != len(self.rl_env.obstacles):
            self._rebuild_obstacle_visuals()

        for node, obstacle in zip(self.obstacle_nodes, self.rl_env.obstacles):
            node.setPos(*obstacle.center.tolist())

    def _rebuild_obstacle_visuals(self) -> None:
        self.obstacle_root_np.removeNode()
        self.obstacle_root_np = self.render.attachNewNode("obstacles")
        self.obstacle_nodes = []

        if not self.rl_env.obstacles:
            return

        template = self.loader.loadModel("models/misc/sphere")
        template.setTransparency(TransparencyAttrib.MAlpha)

        for idx, obstacle in enumerate(self.rl_env.obstacles):
            sphere = template.copyTo(self.obstacle_root_np)
            sphere.setName(f"obstacle-{idx}")
            sphere.setScale(obstacle.radius)
            sphere.setPos(*obstacle.center.tolist())
            sphere.setColor(0.90, 0.42, 0.18, 0.42)
            sphere.setTransparency(TransparencyAttrib.MAlpha)
            self.obstacle_nodes.append(sphere)

        template.removeNode()

    def _step_policy_once(self) -> None:
        action, _state = self.model.predict(
            self.current_obs,
            deterministic=self.deterministic_policy,
        )
        self.last_action = np.asarray(action, dtype=float).reshape(3)

        obs, reward, terminated, truncated, info = self.rl_env.step(self.last_action)
        self.current_obs = obs
        self.last_reward = float(reward)
        self.last_info = info
        self._sync_scene_from_env(rebuild_obstacles=False)
        self._append_trail_point(force=True)

        if terminated or truncated:
            self.episode_done = True
            self.reset_timer = 0.0
            event = str(info.get("event", "done"))
            if info.get("is_success", False):
                self.success_count += 1
                self._set_auv_visual_state("success")
            elif event == "timeout":
                self.timeout_count += 1
                self._set_auv_visual_state("timeout")
            else:
                self.failure_count += 1
                self._set_auv_visual_state("failure")
        else:
            self._set_auv_visual_state("running")

    # ------------------------------------------------------------------
    # Simulation update
    # ------------------------------------------------------------------
    def _update(self, task):
        frame_dt = min(globalClock.getDt(), 0.25)

        if not self.paused:
            if self.episode_done:
                self.reset_timer += frame_dt
                if self.auto_reset_delay >= 0.0 and self.reset_timer >= self.auto_reset_delay:
                    self.reset_episode()
            else:
                self._sim_accumulator += frame_dt
                sim_dt = max(float(self.sim_config.dt), 1e-6)
                steps = 0
                while (
                    self._sim_accumulator >= sim_dt
                    and steps < self._max_steps_per_frame
                    and not self.episode_done
                ):
                    self._step_policy_once()
                    self._sim_accumulator -= sim_dt
                    steps += 1

                if steps >= self._max_steps_per_frame and self._sim_accumulator > sim_dt:
                    self._sim_accumulator = sim_dt

        self._update_target_ring(task.time)
        self._update_camera(frame_dt)
        self._update_hud()
        return task.cont

    # ------------------------------------------------------------------
    # Camera / visuals / HUD
    # ------------------------------------------------------------------
    def _set_auv_visual_state(self, mode: str) -> None:
        if mode == "success":
            self.auv_body.setColor(0.22, 0.92, 0.35, 1.0)
            self.auv_nose.setColor(0.10, 0.55, 0.20, 1.0)
        elif mode == "failure":
            self.auv_body.setColor(0.95, 0.25, 0.22, 1.0)
            self.auv_nose.setColor(0.65, 0.10, 0.10, 1.0)
        elif mode == "timeout":
            self.auv_body.setColor(0.92, 0.60, 0.16, 1.0)
            self.auv_nose.setColor(0.65, 0.35, 0.10, 1.0)
        else:
            self.auv_body.setColor(0.95, 0.78, 0.18, 1.0)
            self.auv_nose.setColor(0.98, 0.30, 0.22, 1.0)

    def _update_camera(self, dt: float) -> None:
        auv_pos = self.auv_np.getPos(self.render)
        forward = self.auv_np.getQuat(self.render).getForward()
        desired_cam_pos = auv_pos - forward * 8.0 + Vec3(0, 0, 3.0)

        current = self.camera.getPos(self.render)
        blend = min(1.0, dt * 3.0)
        new_cam_pos = current * (1.0 - blend) + desired_cam_pos * blend
        self.camera.setPos(self.render, new_cam_pos)
        self.camera.lookAt(auv_pos + Vec3(0, 0, 0.4))

    def _reset_camera(self) -> None:
        auv_pos = self.auv_np.getPos(self.render)
        self.camera.setPos(auv_pos + Vec3(0, -12, 5))
        self.camera.lookAt(auv_pos)

    def _update_target_ring(self, t: float) -> None:
        self.target_ring_np.removeNode()
        self.target_ring_np = self.render.attachNewNode("target-ring")

        radius = float(self.sim_config.capture_radius) + 0.14 + 0.06 * math.sin(t * 4.0)
        z = self.target_np.getZ()
        center = self.target_np.getPos(self.render)

        segs = LineSegs("ring")
        segs.setThickness(2.0)
        segs.setColor(1.0, 0.35, 0.25, 1.0)

        slices = 40
        for i in range(slices + 1):
            theta = 2.0 * math.pi * (i / slices)
            x = center.getX() + radius * math.cos(theta)
            y = center.getY() + radius * math.sin(theta)
            if i == 0:
                segs.moveTo(x, y, z)
            else:
                segs.drawTo(x, y, z)

        self.target_ring_np.attachNewNode(segs.create())

    def _update_hud(self) -> None:
        agent = self.rl_env.agent_pos
        target = self.rl_env.target_pos
        speed = float(np.linalg.norm(self.rl_env.agent_vel))
        distance = float(np.linalg.norm(target - agent))
        min_clearance = float(self.last_info.get("min_obstacle_clearance", float("inf")))
        event = str(self.last_info.get("event", "running"))
        status = self._status_label(event)
        requested_complexity = 100.0 * float(self.rl_env.requested_obstacle_complexity)
        actual_complexity = 100.0 * self.actual_obstacle_complexity
        min_clearance_text = "inf" if math.isinf(min_clearance) else f"{min_clearance:5.2f} m"
        reset_text = (
            "-"
            if (not self.episode_done or self.auto_reset_delay < 0.0)
            else f"{max(0.0, self.auto_reset_delay - self.reset_timer):4.1f} s"
        )

        self.hud_text.setText(
            "\n".join(
                [
                    f"State     : {status}",
                    f"Episode   : {self.episode_number:4d}   Success/Fail/Timeout = {self.success_count:3d}/{self.failure_count:3d}/{self.timeout_count:3d}",
                    f"Policy    : {'DETERMINISTIC' if self.deterministic_policy else 'STOCHASTIC'}   Step dt = {self.sim_config.dt:4.2f} s   Auto-reset = {reset_text}",
                    f"Model     : {self.model_path.name}",
                    f"AUV Pos   : ({agent[0]:6.2f}, {agent[1]:6.2f}, {agent[2]:5.2f})",
                    f"Target    : ({target[0]:6.2f}, {target[1]:6.2f}, {target[2]:5.2f})",
                    f"Distance  : {distance:5.2f} m   Speed = {speed:5.2f} m/s   MinClr = {min_clearance_text}",
                    f"Action    : ({self.last_action[0]:+5.2f}, {self.last_action[1]:+5.2f}, {self.last_action[2]:+5.2f})   Reward = {self.last_reward:+7.3f}",
                    f"Obstacles : {len(self.rl_env.obstacles):2d}/{self.rl_env.requested_obstacle_count:2d}   ObsVel = {self.sim_config.obstacle_speed:4.2f} m/s",
                    f"ObsComp   : {actual_complexity:5.2f}% / {requested_complexity:5.2f}%",
                ]
            )
        )

        self.help_text.setText(
            "[SPACE] pause/resume   [R] reset episode   [D] deterministic on/off   [C] reset camera   [ESC] quit"
        )

    @staticmethod
    def _status_label(event: str) -> str:
        event = event.strip().lower()
        mapping = {
            "running": "RUNNING",
            "success": "SUCCESS",
            "wall_collision": "FAILED (WALL COLLISION)",
            "obstacle_collision": "FAILED (OBSTACLE COLLISION)",
            "timeout": "TIMEOUT",
            "reset": "RESET",
        }
        return mapping.get(event, event.replace("_", " ").upper())

    # ------------------------------------------------------------------
    # Trail helpers / controls
    # ------------------------------------------------------------------
    def _append_trail_point(self, force: bool = False) -> None:
        del force
        pos = self.auv_np.getPos(self.render)
        self.trail_points.append((pos.getX(), pos.getY(), pos.getZ()))
        if len(self.trail_points) > self.max_trail_points:
            self.trail_points.pop(0)
        self._rebuild_trail()

    def _rebuild_trail(self) -> None:
        self.trail_np.removeNode()
        self.trail_np = self.render.attachNewNode("auv-trail")

        if len(self.trail_points) < 2:
            return

        segs = LineSegs("trail")
        segs.setThickness(2.0)
        segs.setColor(0.30, 0.95, 1.00, 1.0)

        for idx, point in enumerate(self.trail_points):
            if idx == 0:
                segs.moveTo(*point)
            else:
                segs.drawTo(*point)

        self.trail_np.attachNewNode(segs.create())

    def _toggle_pause(self) -> None:
        self.paused = not self.paused

    def _toggle_deterministic(self) -> None:
        self.deterministic_policy = not self.deterministic_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained PPO policy in Panda3D using the same environment dynamics as training.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="runs/auv_ppo/final_model",
        help="Path to the PPO model saved by auv_rl_collision_train.py (with or without .zip).",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional path to training_config.json. If omitted, a sibling file next to the model is used when available.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic action sampling instead of deterministic inference.",
    )
    parser.add_argument(
        "--auto-reset-delay",
        type=float,
        default=1.25,
        help="Seconds to wait before auto-reset after success/failure. Use a negative value to disable auto-reset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device passed to stable-baselines3 when loading the model (e.g. auto, cpu, cuda).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed for episode resets. Defaults to the training sim_config seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"PPO model not found: {model_path}. Train one first with auv_rl_collision_train.py."
        )

    layout_cfg, sim_cfg, used_config = load_training_configs(model_path, args.config_path)
    app = AUVTargetSim(
        model_path=model_path,
        layout_config=layout_cfg,
        sim_config=sim_cfg,
        deterministic_policy=not args.stochastic,
        auto_reset_delay=args.auto_reset_delay,
        device=args.device,
        base_seed=args.seed,
    )

    if used_config is not None:
        print(f"[viewer] Loaded training config: {used_config}")
    else:
        print("[viewer] No training_config.json found. Falling back to default LayoutConfig/AUVSimConfig.")
    print(f"[viewer] Loaded PPO model: {model_path}")
    app.run()


if __name__ == "__main__":
    main()
