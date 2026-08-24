"""Typed teleoperation and MuJoCo overlays for the Manager-Based A2Arm task.

The module deliberately contains no environment monkey-patching.  Keyboard state
is converted to :class:`A2ArmTeleopCommand` and installed through the public task
command-term API before each policy step.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.tasks.locomotion.a2arm.state import A2ArmPosForceState, A2ArmTeleopCommand
from unilab.utils.rotation import np_quat_apply, np_yaw_quat

KEY_SPACE = ord(" ")
KEY_BACKSPACE = 259
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265
KEY_PAGE_UP, KEY_PAGE_DOWN = 266, 267


@dataclass
class TeleopState:
    """Mutable keyboard state with the training force profile's timing."""

    velocity_low: np.ndarray
    velocity_high: np.ndarray
    sphere_low: np.ndarray
    sphere_high: np.ndarray
    ee_init: np.ndarray
    ee_ramp: int = 25
    ee_hold: int = 25
    base_ramp: int = 25
    base_hold: int = 50
    impulse_ee_n: float = 15.0
    impulse_base_n: float = 20.0
    velocity: np.ndarray = field(init=False)
    ee_sphere: np.ndarray = field(init=False)
    ee_force: np.ndarray = field(init=False)
    base_force: np.ndarray = field(init=False)
    hold_mode: bool = False
    _ee_target: np.ndarray = field(init=False, repr=False)
    _base_target: np.ndarray = field(init=False, repr=False)
    _ee_step: int = field(default=-1, init=False, repr=False)
    _base_step: int = field(default=-1, init=False, repr=False)
    _ee_held: bool = field(default=False, init=False, repr=False)
    _base_held: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.velocity_low = np.asarray(self.velocity_low, dtype=np.float64).reshape(3)
        self.velocity_high = np.asarray(self.velocity_high, dtype=np.float64).reshape(3)
        self.sphere_low = np.asarray(self.sphere_low, dtype=np.float64).reshape(3)
        self.sphere_high = np.asarray(self.sphere_high, dtype=np.float64).reshape(3)
        self.ee_init = np.asarray(self.ee_init, dtype=np.float64).reshape(3).copy()
        self.velocity = np.zeros(3, dtype=np.float64)
        self.ee_sphere = self.ee_init.copy()
        self.ee_force = np.zeros(3, dtype=np.float64)
        self.base_force = np.zeros(3, dtype=np.float64)
        self._ee_target = np.zeros(3, dtype=np.float64)
        self._base_target = np.zeros(3, dtype=np.float64)

    def reset(self) -> None:
        self.velocity.fill(0.0)
        self.ee_sphere[:] = self.ee_init
        self.clear_forces()

    def nudge_velocity(self, axis: int, delta: float) -> None:
        self.velocity[axis] = np.clip(
            self.velocity[axis] + float(delta), self.velocity_low[axis], self.velocity_high[axis]
        )

    def zero_velocity(self) -> None:
        self.velocity.fill(0.0)

    def nudge_sphere(self, axis: int, delta: float) -> None:
        self.ee_sphere[axis] = np.clip(
            self.ee_sphere[axis] + float(delta), self.sphere_low[axis], self.sphere_high[axis]
        )

    def reset_sphere(self) -> None:
        self.ee_sphere[:] = self.ee_init

    def push_ee(self, axis: int, sign: float) -> None:
        self._ee_target.fill(0.0)
        self._ee_target[axis] = float(sign) * self.impulse_ee_n
        self._ee_step = 0
        self._ee_held = self.hold_mode

    def push_base(self, axis: int, sign: float) -> None:
        self._base_target.fill(0.0)
        self._base_target[axis] = float(sign) * self.impulse_base_n
        self._base_step = 0
        self._base_held = self.hold_mode

    def toggle_hold(self) -> None:
        self.hold_mode = not self.hold_mode
        if not self.hold_mode:
            self.release_held()

    def release_held(self) -> None:
        if self._ee_held:
            self._ee_step = max(1, self.ee_ramp) + self.ee_hold
            self._ee_held = False
        if self._base_held:
            self._base_step = max(1, self.base_ramp) + self.base_hold
            self._base_held = False

    def clear_forces(self) -> None:
        self.ee_force.fill(0.0)
        self.base_force.fill(0.0)
        self._ee_target.fill(0.0)
        self._base_target.fill(0.0)
        self._ee_step = self._base_step = -1
        self._ee_held = self._base_held = False

    @staticmethod
    def _trapezoid_frac(step: int, ramp: int, hold: int) -> float:
        ramp = max(1, int(ramp))
        total = 2 * ramp + int(hold)
        if step < ramp:
            return step / ramp
        if step < ramp + hold:
            return 1.0
        return float(np.clip((total - step) / ramp, 0.0, 1.0))

    def _advance_one(self, *, body: str) -> None:
        if body == "ee":
            step, held, ramp, hold = self._ee_step, self._ee_held, self.ee_ramp, self.ee_hold
            target, output = self._ee_target, self.ee_force
        else:
            step, held, ramp, hold = (
                self._base_step,
                self._base_held,
                self.base_ramp,
                self.base_hold,
            )
            target, output = self._base_target, self.base_force
        if step < 0:
            return
        if held:
            output[:] = target * min(1.0, step / max(1, ramp))
        else:
            output[:] = target * self._trapezoid_frac(step, ramp, hold)
            if step > 2 * max(1, ramp) + hold:
                output.fill(0.0)
                step = -1
        step += 1 if step >= 0 else 0
        if body == "ee":
            self._ee_step = step
        else:
            self._base_step = step

    def advance_forces(self) -> None:
        self._advance_one(body="ee")
        self._advance_one(body="base")

    def as_command(self, num_envs: int = 1) -> A2ArmTeleopCommand:
        count = int(num_envs)
        if count <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        return A2ArmTeleopCommand(
            velocity=np.broadcast_to(self.velocity, (count, 3)),
            ee_sphere=np.broadcast_to(self.ee_sphere, (count, 3)),
            ee_force=np.broadcast_to(self.ee_force, (count, 3)),
            base_force=np.broadcast_to(self.base_force, (count, 3)),
        )

    def describe(self) -> str:
        return (
            f"vel={np.round(self.velocity, 2)} ee={np.round(self.ee_sphere, 2)} "
            f"F_ee={np.round(self.ee_force, 1)} F_base={np.round(self.base_force, 1)} "
            f"hold={'ON' if self.hold_mode else 'off'}"
        )


def make_teleop_from_state(state: A2ArmPosForceState) -> TeleopState:
    """Build keyboard limits from the public typed command configuration."""
    cfg = state.cfg
    return TeleopState(
        velocity_low=np.asarray([item[0] for item in cfg.velocity_ranges]),
        velocity_high=np.asarray([item[1] for item in cfg.velocity_ranges]),
        sphere_low=np.asarray([cfg.goal_radius_range[0], cfg.goal_pitch_range[0], cfg.goal_yaw_range[0]]),
        sphere_high=np.asarray([cfg.goal_radius_range[1], cfg.goal_pitch_range[1], cfg.goal_yaw_range[1]]),
        ee_init=np.asarray(cfg.goal_start, dtype=np.float64),
        ee_ramp=max(1, int(cfg.force_duration[0])),
        ee_hold=int(cfg.gripper_settling),
        base_ramp=max(1, int(cfg.force_duration[0])),
        base_hold=int(cfg.base_settling),
        impulse_ee_n=max(abs(float(v)) for v in cfg.max_push_force_gripper_ext),
        impulse_base_n=max(abs(float(v)) for v in cfg.max_push_force_base_ext),
    )


def install_teleop_override(env: Any, teleop: TeleopState) -> None:
    """Install the current teleop payload through the public command manager."""
    state: A2ArmPosForceState = env.command_manager.get_term("task_state")
    state.set_teleop_override(teleop.as_command(env.num_envs))


def clear_teleop_override(env: Any) -> None:
    state: A2ArmPosForceState = env.command_manager.get_term("task_state")
    state.clear_teleop_override()


def make_key_callback(
    teleop: TeleopState,
    *,
    on_pause: Callable[[], None],
    on_reset: Callable[[], None],
    on_toggle_range: Callable[[], None],
) -> Callable[[int], None]:
    """Return the passive-viewer callback with the legacy key bindings."""

    def callback(keycode: int) -> None:
        if keycode == KEY_SPACE:
            on_pause()
        elif keycode == KEY_BACKSPACE:
            on_reset()
        elif keycode in (ord("G"), ord("g")):
            on_toggle_range()
        elif keycode in (ord("F"), ord("f")):
            teleop.clear_forces()
        elif keycode in (ord("W"), ord("w")):
            teleop.nudge_velocity(0, +0.1)
        elif keycode in (ord("S"), ord("s")):
            teleop.nudge_velocity(0, -0.1)
        elif keycode in (ord("A"), ord("a")):
            teleop.nudge_velocity(1, +0.1)
        elif keycode in (ord("D"), ord("d")):
            teleop.nudge_velocity(1, -0.1)
        elif keycode in (ord("Q"), ord("q")):
            teleop.nudge_velocity(2, +0.1)
        elif keycode in (ord("E"), ord("e")):
            teleop.nudge_velocity(2, -0.1)
        elif keycode in (ord("Z"), ord("z")):
            teleop.zero_velocity()
        elif keycode in (ord("H"), ord("h")):
            teleop.toggle_hold()
        elif keycode in (ord("U"), ord("u")):
            teleop.nudge_sphere(0, +0.02)
        elif keycode in (ord("J"), ord("j")):
            teleop.nudge_sphere(0, -0.02)
        elif keycode in (ord("I"), ord("i")):
            teleop.nudge_sphere(1, +0.05)
        elif keycode in (ord("K"), ord("k")):
            teleop.nudge_sphere(1, -0.05)
        elif keycode in (ord("O"), ord("o")):
            teleop.nudge_sphere(2, +0.05)
        elif keycode in (ord("L"), ord("l")):
            teleop.nudge_sphere(2, -0.05)
        elif keycode in (ord("P"), ord("p")):
            teleop.reset_sphere()
        elif keycode == KEY_UP:
            teleop.push_ee(0, +1.0)
        elif keycode == KEY_DOWN:
            teleop.push_ee(0, -1.0)
        elif keycode == KEY_LEFT:
            teleop.push_ee(1, +1.0)
        elif keycode == KEY_RIGHT:
            teleop.push_ee(1, -1.0)
        elif keycode == KEY_PAGE_UP:
            teleop.push_ee(2, +1.0)
        elif keycode == KEY_PAGE_DOWN:
            teleop.push_ee(2, -1.0)
        elif keycode in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6")):
            value = keycode - ord("1")
            teleop.push_base(value // 2, 1.0 if value % 2 == 0 else -1.0)

    return callback


def _add_sphere(scene: Any, pos: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    import mujoco

    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, 0.0, 0.0]),
        np.asarray(pos, dtype=np.float64),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _add_arrow(scene: Any, p0: np.ndarray, vec: np.ndarray, scale: float, width: float, rgba: np.ndarray) -> None:
    import mujoco

    if np.linalg.norm(vec) < 1e-6 or scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        width,
        np.asarray(p0, dtype=np.float64),
        np.asarray(p0, dtype=np.float64) + np.asarray(vec, dtype=np.float64) * scale,
    )
    scene.ngeom += 1


def _add_line(scene: Any, p0: np.ndarray, p1: np.ndarray, width: float, rgba: np.ndarray) -> None:
    import mujoco

    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        width,
        np.asarray(p0, dtype=np.float64),
        np.asarray(p1, dtype=np.float64),
    )
    scene.ngeom += 1


def _sphere_point(center: np.ndarray, yaw_quat: np.ndarray, sphere: np.ndarray) -> np.ndarray:
    length, pitch, yaw = np.asarray(sphere, dtype=np.float64)
    cart = np.asarray(
        [length * np.cos(pitch) * np.cos(yaw), length * np.cos(pitch) * np.sin(yaw), length * np.sin(pitch)]
    )
    return np.asarray(center + np_quat_apply(yaw_quat[None, :], cart[None, :])[0], dtype=np.float64)


def _draw_sample_range(viewer: Any, state: A2ArmPosForceState) -> None:
    scene = viewer.user_scn
    cfg = state.cfg
    center = np.asarray(state.goal_center_world()[0], dtype=np.float64)
    yaw_quat = np.asarray(np_yaw_quat(state.root_quat_world)[0], dtype=np.float64)
    radii = (float(cfg.goal_radius_range[0]), float(cfg.goal_radius_range[1]))
    pitches = np.linspace(float(cfg.goal_pitch_range[0]), float(cfg.goal_pitch_range[1]), 7)
    yaws = np.linspace(float(cfg.goal_yaw_range[0]), float(cfg.goal_yaw_range[1]), 9)
    shell = np.asarray([0.2, 0.5, 1.0, 0.5], dtype=np.float32)
    edge = np.asarray([1.0, 0.85, 0.1, 0.7], dtype=np.float32)
    for radius in radii:
        for pitch in pitches:
            points = [_sphere_point(center, yaw_quat, (radius, pitch, yaw)) for yaw in yaws]
            for p0, p1 in zip(points, points[1:]):
                _add_line(scene, p0, p1, 0.006, shell)
        for yaw in yaws:
            points = [_sphere_point(center, yaw_quat, (radius, pitch, yaw)) for pitch in pitches]
            for p0, p1 in zip(points, points[1:]):
                _add_line(scene, p0, p1, 0.006, shell)
    for pitch in (pitches[0], pitches[-1]):
        for yaw in (yaws[0], yaws[-1]):
            _add_line(
                scene,
                _sphere_point(center, yaw_quat, (radii[0], pitch, yaw)),
                _sphere_point(center, yaw_quat, (radii[1], pitch, yaw)),
                0.006,
                edge,
            )
    _add_sphere(scene, center, 0.02, np.asarray([0.1, 0.9, 0.9, 0.9]))


def draw_markers(viewer: Any, env: Any, *, show_range: bool = True) -> None:
    """Draw goal, measured EE, and currently applied force overlays."""
    scene = viewer.user_scn
    scene.ngeom = 0
    state: A2ArmPosForceState = env.command_manager.get_term("task_state")
    robot = env.scene["robot"]
    if show_range:
        _draw_sample_range(viewer, state)
    goal = np.asarray(state.current_goal_world[0], dtype=np.float64)
    _add_sphere(scene, goal, 0.04, np.asarray([0.1, 0.9, 0.2, 0.9]))
    ee = np.asarray(state.ee_world_pos()[0], dtype=np.float64)
    _add_sphere(scene, ee, 0.03, np.asarray([1.0, 0.6, 0.1, 0.9]))
    _add_arrow(scene, ee, state.force_ee_world[0], 0.01, 0.012, np.asarray([1.0, 0.1, 0.1, 0.95]))
    _add_arrow(
        scene,
        np.asarray(robot.data.root_link_pos_w[0], dtype=np.float64),
        state.force_base_world[0],
        0.01,
        0.015,
        np.asarray([1.0, 0.1, 0.8, 0.95]),
    )


def print_legend() -> None:
    print(
        "[play] Base: W/S vx, A/D vy, Q/E yaw, Z stop | EE: U/J radius, I/K pitch, O/L yaw, P reset\n"
        "[play] EE push: arrows/PageUp/PageDown | base push: 1..6 | H hold, F clear, Backspace reset, Space pause"
    )


__all__ = [
    "TeleopState",
    "make_teleop_from_state",
    "install_teleop_override",
    "clear_teleop_override",
    "make_key_callback",
    "draw_markers",
    "print_legend",
]
