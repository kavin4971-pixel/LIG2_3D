from __future__ import annotations

import math
from typing import Iterable

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

from environment3d import Environment3D


class AUVTargetSim(ShowBase):
    """Simple Panda3D baseline for an AUV reaching a target in a 3D volume.

    Environment sizing, obstacle policies, spawn policies, and obstacle-motion
    policies are centralized in ``environment3d.py``. This script keeps only
    AUV-/guidance-related parameters plus Panda3D scene logic.

    Coordinate convention:
    - X: left/right
    - Y: forward/backward (Panda3D forward axis)
    - Z: up/down
    """

    def __init__(self) -> None:
        super().__init__()

        # AUV / guidance tuning parameters.
        self.capture_radius = 0.50
        self.max_speed = 6.0
        self.max_accel = 7.5
        self.slowdown_radius = 7.0
        self.paused = False
        self.arrived = False

        # Simple collision proxy sizes.
        self.auv_radius = 0.65
        self.target_radius = 0.42

        # Environment is configured inside environment3d.py.
        self.env = Environment3D.default()
        self.rng = self.env.make_rng()
        self.start_pos = self.env.default_start_position()
        self.env.apply_configured_obstacles(
            auv_radius=self.auv_radius,
            rng=self.rng,
            start_pos=self.start_pos,
        )
        self.env.initialize_dynamic_obstacles(
            auv_max_speed=self.max_speed,
            rng=self.rng,
        )

        # State.
        self.velocity = Vec3(0, 0, 0)
        self.trail_points: list[tuple[float, float, float]] = []
        self.max_trail_points = 300
        self.trail_update_interval = 0.10
        self._trail_timer = 0.0
        self.trail_np = self.render.attachNewNode("auv-trail")
        self.obstacle_nodes: list = []

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

        self.auv_np.setPos(*self.start_pos.tolist())
        self._append_trail_point(force=True)
        self.randomize_target()
        self._reset_camera()

        self.taskMgr.add(self._update, "update-auv")

    # ---------------------------------------------------------------------
    # Scene setup
    # ---------------------------------------------------------------------
    def _configure_window(self) -> None:
        props = WindowProperties()
        props.setTitle("AUV Target Simulation - Panda3D")
        props.setSize(1280, 720)
        self.win.requestProperties(props)

    def _setup_lights(self) -> None:
        ambient = AmbientLight("ambient")
        ambient.setColor((0.45, 0.50, 0.55, 1.0))
        ambient_np = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_np)

        sun = DirectionalLight("sun")
        sun.setColor((0.85, 0.85, 0.80, 1.0))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(-35, -50, 0)
        self.render.setLight(sun_np)

    def _build_environment(self) -> None:
        min_b = self.env.min_bound
        max_b = self.env.max_bound

        box_np = self._make_wire_box(
            min_b,
            max_b,
            thickness=self.env.box_line_thickness,
        )
        box_np.reparentTo(self.render)
        box_np.setColor(0.25, 0.80, 1.00, 1.0)

        grid_np = self._make_floor_grid(
            min_x=float(min_b[0]),
            max_x=float(max_b[0]),
            min_y=float(min_b[1]),
            max_y=float(max_b[1]),
            z=float(min_b[2]),
            spacing=self.env.grid_spacing,
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

        self._rebuild_obstacle_visuals()

    def _build_auv(self) -> None:
        self.auv_np = self.render.attachNewNode("auv")

        body = self.loader.loadModel("models/box")
        body.reparentTo(self.auv_np)
        body.setScale(0.45, 1.10, 0.25)
        body.setColor(0.95, 0.78, 0.18, 1.0)

        nose = self.loader.loadModel("models/misc/sphere")
        nose.reparentTo(self.auv_np)
        nose.setScale(0.18)
        nose.setPos(0.0, 1.20, 0.0)
        nose.setColor(0.98, 0.30, 0.22, 1.0)

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
        self.target_np.setScale(self.target_radius)
        self.target_np.setColor(1.00, 0.15, 0.15, 1.0)

        self.target_ring_np = self.render.attachNewNode("target-ring")

    def _build_ui(self) -> None:
        self.hud_text = OnscreenText(
            text="",
            pos=(-1.30, 0.92),
            scale=0.045,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft,
            mayChange=True,
        )
        self.help_text = OnscreenText(
            text="[SPACE] pause/resume   [R] random target   [C] reset camera   [ESC] quit",
            pos=(-1.30, 0.84),
            scale=0.035,
            fg=(0.75, 0.90, 1.00, 1.0),
            align=TextNode.ALeft,
        )

    def _bind_keys(self) -> None:
        self.accept("space", self._toggle_pause)
        self.accept("r", self.randomize_target)
        self.accept("c", self._reset_camera)
        self.accept("escape", self.userExit)

    def _rebuild_obstacle_visuals(self) -> None:
        if hasattr(self, "obstacle_root_np"):
            self.obstacle_root_np.removeNode()

        self.obstacle_root_np = self.render.attachNewNode("obstacles")
        self.obstacle_nodes = []
        if not self.env.obstacles:
            return

        template = self.loader.loadModel("models/misc/sphere")
        template.setTransparency(TransparencyAttrib.MAlpha)

        for idx, obstacle in enumerate(self.env.obstacles):
            sphere = template.copyTo(self.obstacle_root_np)
            sphere.setName(f"obstacle-{idx}")
            sphere.setScale(obstacle.radius)
            sphere.setPos(*obstacle.center.tolist())
            sphere.setColor(0.90, 0.42, 0.18, 0.42)
            sphere.setTransparency(TransparencyAttrib.MAlpha)
            self.obstacle_nodes.append(sphere)

        template.removeNode()

    def _sync_obstacle_visuals(self) -> None:
        if len(self.obstacle_nodes) != len(self.env.obstacles):
            self._rebuild_obstacle_visuals()
            return

        for node, obstacle in zip(self.obstacle_nodes, self.env.obstacles):
            node.setPos(*obstacle.center.tolist())

    # ---------------------------------------------------------------------
    # Environment geometry helpers
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # Simulation update
    # ---------------------------------------------------------------------
    def _update(self, task):
        dt = min(globalClock.getDt(), 0.05)

        if not self.paused and not self.arrived:
            self._update_obstacles(dt)
            self._update_guidance(dt)
            self._append_trail_point()

        self._update_target_ring(task.time)
        self._update_camera(dt)
        self._update_hud()
        return task.cont

    def _update_obstacles(self, dt: float) -> None:
        target = self.target_np.getPos(self.render)
        target_np = np.array([target.getX(), target.getY(), target.getZ()], dtype=float)
        self.env.update_dynamic_obstacles(
            dt=dt,
            auv_max_speed=self.max_speed,
            target_center=target_np,
            target_radius=self.target_radius,
            rng=self.rng,
        )
        self._sync_obstacle_visuals()

    def _update_guidance(self, dt: float) -> None:
        pos = self.auv_np.getPos(self.render)
        target = self.target_np.getPos(self.render)
        to_target = target - pos
        distance = to_target.length()

        if distance <= self.capture_radius:
            self.arrived = True
            self.velocity = Vec3(0, 0, 0)
            return

        desired_speed = self.max_speed
        if distance < self.slowdown_radius:
            desired_speed = max(1.0, self.max_speed * (distance / self.slowdown_radius))

        desired_velocity = Vec3(0, 0, 0)
        if distance > 1e-6:
            desired_velocity = to_target.normalized() * desired_speed

        steering = desired_velocity - self.velocity
        if steering.length() > self.max_accel:
            steering.normalize()
            steering *= self.max_accel

        self.velocity += steering * dt
        if self.velocity.length() > self.max_speed:
            self.velocity.normalize()
            self.velocity *= self.max_speed

        new_pos = pos + self.velocity * dt
        new_pos = self._clamp_and_resolve(new_pos)
        self.auv_np.setPos(self.render, new_pos)

        if self.velocity.lengthSquared() > 1e-5:
            self.auv_np.lookAt(self.auv_np.getPos(self.render) + self.velocity)

    def _clamp_and_resolve(self, pos: Vec3) -> Vec3:
        point = np.array([pos.getX(), pos.getY(), pos.getZ()], dtype=float)
        velocity = np.array(
            [self.velocity.getX(), self.velocity.getY(), self.velocity.getZ()],
            dtype=float,
        )

        clamped = self.env.clamp(point)
        if not np.allclose(point, clamped):
            if point[0] != clamped[0]:
                velocity[0] *= -0.2
            if point[1] != clamped[1]:
                velocity[1] *= -0.2
            if point[2] != clamped[2]:
                velocity[2] *= -0.2
            point = clamped

        for _ in range(3):
            adjusted = False
            point = self.env.clamp(point)

            for obstacle in self.env.obstacles:
                delta = point - obstacle.center
                min_distance = obstacle.radius + self.auv_radius
                distance = float(np.linalg.norm(delta))

                if distance >= min_distance:
                    continue

                adjusted = True
                if distance < 1e-8:
                    speed_norm = float(np.linalg.norm(velocity))
                    if speed_norm > 1e-8:
                        normal = velocity / speed_norm
                    else:
                        normal = np.array([1.0, 0.0, 0.0], dtype=float)
                else:
                    normal = delta / distance

                point = obstacle.center + normal * min_distance

                inward_speed = float(np.dot(velocity, normal))
                if inward_speed < 0.0:
                    velocity = velocity - inward_speed * normal
                    velocity *= 0.75

            if not adjusted:
                break

        point = self.env.clamp(point)
        self.velocity = Vec3(float(velocity[0]), float(velocity[1]), float(velocity[2]))
        return Vec3(float(point[0]), float(point[1]), float(point[2]))

    # ---------------------------------------------------------------------
    # Camera / UI / target visuals
    # ---------------------------------------------------------------------
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

    def _update_hud(self) -> None:
        pos = self.auv_np.getPos(self.render)
        target = self.target_np.getPos(self.render)
        distance = (target - pos).length()
        speed = self.velocity.length()
        obstacle_speed = self.env.dynamic_obstacle_speed(self.max_speed)
        status = "ARRIVED" if self.arrived else ("PAUSED" if self.paused else "RUNNING")

        target_complexity_pct = 100.0 * self.env.requested_obstacle_complexity
        actual_complexity_pct = 100.0 * self.env.obstacle_complexity
        mode_suffix = "manual count" if self.env.obstacle_mode == "count" else "auto complexity"

        self.hud_text.setText(
            "\n".join(
                [
                    f"State   : {status}",
                    f"AUV Pos : ({pos.getX():6.2f}, {pos.getY():6.2f}, {pos.getZ():5.2f})",
                    f"Target  : ({target.getX():6.2f}, {target.getY():6.2f}, {target.getZ():5.2f})",
                    f"Distance: {distance:5.2f} m",
                    f"Speed   : {speed:5.2f} m/s",
                    f"ObsVel  : {obstacle_speed:5.2f} m/s",
                    (
                        f"Obsts   : {len(self.env.obstacles):2d}/{self.env.requested_obstacle_count:2d}  "
                        f"(r={self.env.obstacle_radius:4.2f} m)"
                    ),
                    f"ObsComp : {actual_complexity_pct:5.2f}% / {target_complexity_pct:5.2f}%  [{mode_suffix}]",
                ]
            )
        )

    def _update_target_ring(self, t: float) -> None:
        self.target_ring_np.removeNode()
        self.target_ring_np = self.render.attachNewNode("target-ring")

        radius = 0.65 + 0.10 * math.sin(t * 4.0)
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

    # ---------------------------------------------------------------------
    # Trail / control helpers
    # ---------------------------------------------------------------------
    def _append_trail_point(self, force: bool = False) -> None:
        dt = globalClock.getDt() if not force else self.trail_update_interval
        self._trail_timer += dt
        if not force and self._trail_timer < self.trail_update_interval:
            return
        self._trail_timer = 0.0

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

    def randomize_target(self) -> None:
        current = self.auv_np.getPos(self.render)
        current_np = np.array([current.getX(), current.getY(), current.getZ()], dtype=float)
        point = self.env.sample_target_point(
            current_pos=current_np,
            target_radius=self.target_radius,
            rng=self.rng,
        )

        candidate = Vec3(float(point[0]), float(point[1]), float(point[2]))
        self.target_np.setPos(self.render, candidate)
        self.env.update_dynamic_obstacles(
            dt=0.0,
            auv_max_speed=self.max_speed,
            target_center=point,
            target_radius=self.target_radius,
            rng=self.rng,
        )
        self._sync_obstacle_visuals()
        self.arrived = False

    def _toggle_pause(self) -> None:
        if self.arrived:
            return
        self.paused = not self.paused


if __name__ == "__main__":
    app = AUVTargetSim()
    app.run()
