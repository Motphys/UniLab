"""Task-owned A2Arm command and lifecycle state.

The state owner keeps the legacy command/force/gait cadence while exposing the
typed hooks consumed by Manager-Based action, observation, reward, and
termination terms.  Hooks are keyed by the control-step token so the standard
manager pipeline cannot advance a stateful signal twice in one step.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from unilab.managers import CommandTerm, CommandTermCfg
from unilab.utils.rotation import np_quat_apply, np_quat_apply_inverse, np_yaw_quat

from .constants import (
    ACTOR_STEP_DIM,
    ARM_JOINT_NAMES,
    CMD_BASE_FORCE,
    CMD_EE_FORCE,
    CMD_EE_POS,
    CMD_VEL,
    NUM_ACTIONS,
    NUM_COMMANDS,
    NUM_LEG,
)

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


def sphere2cart(sphere: np.ndarray) -> np.ndarray:
    length, pitch, yaw = np.moveaxis(np.asarray(sphere), -1, 0)
    return np.stack(
        [
            length * np.cos(pitch) * np.cos(yaw),
            length * np.cos(pitch) * np.sin(yaw),
            length * np.sin(pitch),
        ],
        axis=-1,
    )


def cart2sphere(cart: np.ndarray) -> np.ndarray:
    cart = np.asarray(cart)
    xy = np.linalg.norm(cart[..., :2], axis=-1)
    return np.stack(
        [
            np.linalg.norm(cart, axis=-1),
            np.arctan2(cart[..., 2], xy),
            np.arctan2(cart[..., 1], cart[..., 0]),
        ],
        axis=-1,
    )


class ForceSchedule:
    """Per-environment ramp/hold/release force episode scheduler."""

    def __init__(
        self,
        num_envs: int,
        mag_range: Sequence[float],
        interval_range: Sequence[int],
        duration_range: Sequence[int],
        probability: float,
        dtype: np.dtype[Any],
        *,
        z_scale: float = 1.0,
        settling: int = 0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._mag = tuple(float(v) for v in mag_range)
        self._interval = tuple(int(v) for v in interval_range)
        self._duration = tuple(int(v) for v in duration_range)
        self._probability = float(probability)
        self._dtype = dtype
        self._z_scale = float(z_scale)
        self._settling = max(0, int(settling))
        self._rng = rng
        self.current = np.zeros((num_envs, 3), dtype=dtype)
        self._target = np.zeros_like(self.current)
        self._active = np.zeros(num_envs, dtype=bool)
        self._elapsed = np.zeros(num_envs, dtype=np.int32)
        self._ramp = np.ones(num_envs, dtype=np.int32)
        self._hold = np.zeros(num_envs, dtype=np.int32)
        self._timer = np.zeros(num_envs, dtype=np.int32)
        self.reset(np.arange(num_envs, dtype=np.int32))

    def reset(self, env_ids: np.ndarray) -> None:
        ids = np.asarray(env_ids, dtype=np.intp)
        self.current[ids] = 0.0
        self._target[ids] = 0.0
        self._active[ids] = False
        self._elapsed[ids] = 0
        self._timer[ids] = self._randint(self._interval[0], self._interval[1] + 1, len(ids))

    def _uniform(self, low: float = 0.0, high: float = 1.0, size: Any = None) -> np.ndarray:
        if self._rng is None:
            return np.random.uniform(low, high, size=size)
        return self._rng.uniform(low, high, size=size)

    def _randint(self, low: int, high: int, size: Any) -> np.ndarray:
        if self._rng is None:
            return np.random.randint(low, high, size=size)
        return self._rng.integers(low, high, size=size)

    def _start(self, ids: np.ndarray) -> None:
        fire = self._uniform(size=len(ids)) < self._probability
        firing = ids[fire]
        if len(firing):
            target = self._uniform(self._mag[0], self._mag[1], size=(len(firing), 3))
            target[:, 2] *= self._z_scale
            self._target[firing] = target.astype(self._dtype)
            self._ramp[firing] = np.maximum(
                1, self._randint(self._duration[0], self._duration[1] + 1, size=len(firing))
            )
            self._hold[firing] = self._settling
            self._elapsed[firing] = 0
            self._active[firing] = True
        idle = ids[~fire]
        if len(idle):
            self._timer[idle] = self._randint(self._interval[0], self._interval[1] + 1, len(idle))

    def step(self, enabled: bool) -> np.ndarray:
        if not enabled:
            self.current.fill(0.0)
            return self.current
        idle = ~self._active
        self._timer[idle] -= 1
        due = np.flatnonzero(idle & (self._timer <= 0)).astype(np.int32)
        if len(due):
            self._start(due)
        active = np.flatnonzero(self._active).astype(np.int32)
        if len(active):
            elapsed = self._elapsed[active].astype(self._dtype)
            ramp = self._ramp[active].astype(self._dtype)
            hold = self._hold[active].astype(self._dtype)
            total = 2.0 * ramp + hold
            frac = np.ones_like(elapsed)
            rising = elapsed < ramp
            frac[rising] = elapsed[rising] / ramp[rising]
            falling = elapsed >= ramp + hold
            frac[falling] = np.clip((total[falling] - elapsed[falling]) / ramp[falling], 0.0, 1.0)
            self.current[active] = self._target[active] * frac[:, None]
            self._elapsed[active] += 1
            done = self._elapsed[active] >= total.astype(np.int32)
            done_ids = active[done]
            if len(done_ids):
                self._active[done_ids] = False
                self.current[done_ids] = 0.0
                self._target[done_ids] = 0.0
                self._timer[done_ids] = self._randint(
                    self._interval[0], self._interval[1] + 1, len(done_ids)
                )
        return self.current


@dataclass(frozen=True)
class A2ArmTeleopCommand:
    """Typed batch of interactive overrides consumed by the task command term."""

    velocity: np.ndarray
    ee_sphere: np.ndarray
    ee_force: np.ndarray
    base_force: np.ndarray

    def __post_init__(self) -> None:
        for name in ("velocity", "ee_sphere", "ee_force", "base_force"):
            value = np.asarray(getattr(self, name), dtype=np.float32).copy()
            if value.ndim == 1:
                value = value[None, :]
            if value.ndim != 2 or value.shape[1] != 3:
                raise ValueError(f"teleop {name} must have shape (num_envs, 3), got {value.shape}")
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(kw_only=True)
class A2ArmPosForceCommandCfg(CommandTermCfg):
    """All task command, trajectory, gait, and force schedule parameters."""

    entity_name: str = "robot"
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    num_actions: int = NUM_ACTIONS
    ctrl_dt: float = 0.02
    force_start_step: int = 192000
    force_curriculum_scales: tuple[float, ...] = ()
    force_curriculum_stage_steps: int = 1
    velocity_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-0.6, 0.6),
        (-0.4, 0.4),
        (-0.6, 0.6),
    )
    velocity_clip: tuple[float, float, float] = (0.1, 0.1, 0.2)
    zero_velocity_probability: float = 0.3
    zero_velocity_probability_after_force: float = 0.8
    command_resample_steps: int = 250
    gripper_force_kp: float = 300.0
    base_force_kd: float = 200.0
    goal_sphere_center: tuple[float, float, float] = (0.2, 0.0, 0.735)
    goal_start: tuple[float, float, float] = (0.4494, 0.8115, 0.0)
    goal_end: tuple[float, float, float] = (0.4494, 0.0, 0.0)
    goal_radius_range: tuple[float, float] = (0.28, 0.60)
    goal_pitch_range: tuple[float, float] = (-np.pi / 3.0, 0.4 * np.pi)
    goal_yaw_range: tuple[float, float] = (-0.6 * np.pi, 0.6 * np.pi)
    goal_traj_time_range: tuple[float, float] = (1.0, 3.0)
    goal_hold_time_range: tuple[float, float] = (0.5, 2.0)
    goal_collision_upper: tuple[float, float, float] = (0.25, 0.2, -0.15)
    goal_collision_lower: tuple[float, float, float] = (-0.7, -0.2, -0.8)
    goal_underground_limit: float = -0.7
    goal_collision_samples: int = 10
    goal_resample_attempts: int = 10
    gait_cycle_time: float = 0.70
    gait_target_scale: float = 0.18
    gait_target_threshold: float = 0.5
    max_push_force_gripper_cmd: tuple[float, float] = (-30.0, 30.0)
    max_push_force_gripper_ext: tuple[float, float] = (-30.0, 30.0)
    max_push_force_base_cmd: tuple[float, float] = (-60.0, 60.0)
    max_push_force_base_ext: tuple[float, float] = (-50.0, 50.0)
    gripper_interval_cmd: tuple[int, int] = (250, 500)
    gripper_interval_ext: tuple[int, int] = (300, 600)
    base_interval_cmd: tuple[int, int] = (250, 500)
    base_interval_ext: tuple[int, int] = (400, 700)
    force_duration: tuple[int, int] = (25, 75)
    gripper_settling: int = 25
    base_settling: int = 50
    gripper_probability_cmd: float = 0.8
    gripper_probability_ext: float = 0.8
    base_probability_cmd: float = 0.8
    base_probability_ext: float = 0.8
    force_z_gripper_cmd_scale: float = 0.33
    force_z_gripper_ext_scale: float = 0.33
    force_z_base_ext_scale: float = 0.05
    root_yaw_range: float = np.pi / 2.0
    randomize_base_mass: bool = True
    added_mass_range: tuple[float, float] = (0.0, 4.0)
    random_com: bool = True
    com_offset_x: tuple[float, float] = (-0.08, 0.08)
    com_offset_y: tuple[float, float] = (-0.08, 0.08)
    com_offset_z: tuple[float, float] = (-0.08, 0.08)
    randomize_foot_friction: bool = True
    foot_friction_range: tuple[float, float] = (0.5, 1.8)
    randomize_gripper_mass: bool = True
    gripper_added_mass_range: tuple[float, float] = (0.0, 0.10)
    velocity_push: bool = True
    push_interval: int = 400
    max_push_vel_xy: float = 0.3
    velocity_push_standing_scale: float = 1.0
    soft_dof_pos_limit: float = 0.9

    def build(self, env: ManagerBasedRlEnv) -> A2ArmPosForceState:
        return A2ArmPosForceState(self, env)


class A2ArmPosForceState(CommandTerm):
    """Shared task state with explicit pre-physics and post-physics hooks."""

    cfg: A2ArmPosForceCommandCfg

    def __init__(self, cfg: A2ArmPosForceCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._dtype = np.dtype(np.float32)
        self._command = np.zeros((self.num_envs, NUM_COMMANDS), dtype=self._dtype)
        self._last_control_token: int | None = None
        self._last_transition_token: int | None = None
        self._command_timer = np.zeros(self.num_envs, dtype=np.int32)
        self._gait_phase = np.zeros(self.num_envs, dtype=self._dtype)
        self._stance_mask = np.zeros((self.num_envs, 4), dtype=self._dtype)
        self._reference_dof_pos = np.zeros((self.num_envs, NUM_LEG), dtype=self._dtype)
        self._foot_contact = np.zeros((self.num_envs, 4), dtype=bool)
        self._last_contact = np.zeros((self.num_envs, 4), dtype=bool)
        self._feet_air_time = np.zeros((self.num_envs, 4), dtype=self._dtype)
        self._first_contact = np.zeros((self.num_envs, 4), dtype=bool)
        self._air_time_snapshot = np.zeros((self.num_envs, 4), dtype=self._dtype)
        self._goal_sphere = np.zeros((self.num_envs, 3), dtype=self._dtype)
        self._goal_start = np.zeros_like(self._goal_sphere)
        self._goal_target = np.zeros_like(self._goal_sphere)
        self._goal_cart = np.zeros_like(self._goal_sphere)
        self._goal_world = np.zeros_like(self._goal_sphere)
        self._goal_timer = np.zeros(self.num_envs, dtype=np.int32)
        self._goal_traj_steps = np.ones(self.num_envs, dtype=np.int32)
        self._goal_total_steps = np.ones(self.num_envs, dtype=np.int32)
        self._last_dof_vel = np.zeros((self.num_envs, NUM_ACTIONS), dtype=self._dtype)
        self._pending_dof_vel = np.zeros_like(self._last_dof_vel)
        self._pending_dof_vel_token: int | None = None
        self._teleop_enabled = False
        self._teleop_velocity = np.zeros((self.num_envs, 3), dtype=self._dtype)
        self._teleop_ee_sphere = np.zeros((self.num_envs, 3), dtype=self._dtype)
        self._teleop_ee_force = np.zeros((self.num_envs, 3), dtype=self._dtype)
        self._teleop_base_force = np.zeros((self.num_envs, 3), dtype=self._dtype)
        self._dr_friction = np.zeros((self.num_envs, 1), dtype=self._dtype)
        self._dr_base_mass = np.zeros((self.num_envs, 1), dtype=self._dtype)
        self._dr_base_com = np.zeros((self.num_envs, 3), dtype=self._dtype)
        self._dr_gripper_mass = np.zeros((self.num_envs, 1), dtype=self._dtype)
        self._sched_gripper_cmd = ForceSchedule(
            self.num_envs,
            cfg.max_push_force_gripper_cmd,
            cfg.gripper_interval_cmd,
            cfg.force_duration,
            cfg.gripper_probability_cmd,
            self._dtype,
            z_scale=cfg.force_z_gripper_cmd_scale,
            settling=cfg.gripper_settling,
            rng=self._env.rng,
        )
        self._sched_gripper_ext = ForceSchedule(
            self.num_envs,
            cfg.max_push_force_gripper_ext,
            cfg.gripper_interval_ext,
            cfg.force_duration,
            cfg.gripper_probability_ext,
            self._dtype,
            z_scale=cfg.force_z_gripper_ext_scale,
            settling=cfg.gripper_settling,
            rng=self._env.rng,
        )
        self._sched_base_cmd = ForceSchedule(
            self.num_envs,
            cfg.max_push_force_base_cmd,
            cfg.base_interval_cmd,
            cfg.force_duration,
            cfg.base_probability_cmd,
            self._dtype,
            z_scale=0.0,
            settling=cfg.base_settling,
            rng=self._env.rng,
        )
        self._sched_base_ext = ForceSchedule(
            self.num_envs,
            cfg.max_push_force_base_ext,
            cfg.base_interval_ext,
            cfg.force_duration,
            cfg.base_probability_ext,
            self._dtype,
            z_scale=cfg.force_z_base_ext_scale,
            settling=cfg.base_settling,
            rng=self._env.rng,
        )
        self._force_ee_cmd = np.zeros((self.num_envs, 3), dtype=self._dtype)
        self._force_ee_world = np.zeros_like(self._force_ee_cmd)
        self._force_base_cmd = np.zeros_like(self._force_ee_cmd)
        self._force_base_world = np.zeros_like(self._force_ee_cmd)
        self._sensor_view = env.scene.bind_sensor_data(
            (
                "endpoint_pos",
                "endpoint_quat",
                "armbasepoint_world_pos",
                "armbasepoint_world_quat",
                "FL_foot_contact",
                "FR_foot_contact",
                "RL_foot_contact",
                "RR_foot_contact",
            )
        )
        self._foot_contact_view = env.scene.bind_sensor_data(
            ("FL_foot_contact", "FR_foot_contact", "RL_foot_contact", "RR_foot_contact")
        )
        self._foot_force_view = env.scene.bind_sensor_data(
            ("FL_foot_force", "FR_foot_force", "RL_foot_force", "RR_foot_force")
        )
        self._foot_vel_view = env.scene.bind_sensor_data(
            ("FL_global_linvel", "FR_global_linvel", "RL_global_linvel", "RR_global_linvel")
        )
        self._foot_pos_view = env.scene.bind_sensor_data(("FL_pos", "FR_pos", "RL_pos", "RR_pos"))
        self._thigh_pos_view = env.scene.bind_sensor_data(
            ("FL_thigh_pos", "FR_thigh_pos", "RL_thigh_pos", "RR_thigh_pos")
        )
        self._undesired_contact_view = env.scene.bind_sensor_data(
            (
                "base1_contact",
                "base2_contact",
                "base3_contact",
                "FL_thigh_contact",
                "FR_thigh_contact",
                "RL_thigh_contact",
                "RR_thigh_contact",
                "FL_calf_contact1",
                "FR_calf_contact1",
                "RL_calf_contact1",
                "RR_calf_contact1",
                "FL_calf_contact2",
                "FR_calf_contact2",
                "RL_calf_contact2",
                "RR_calf_contact2",
            )
        )
        self._entity = env.scene[cfg.entity_name]
        self._ee_entity = env.scene["end_effector"]
        hard_limits = np.asarray(self._entity.data.soft_joint_pos_limits, dtype=self._dtype)
        midpoint = (hard_limits[:, 0] + hard_limits[:, 1]) * 0.5
        half_range = (hard_limits[:, 1] - hard_limits[:, 0]) * 0.5 * float(cfg.soft_dof_pos_limit)
        self._soft_dof_pos_limits = np.stack((midpoint - half_range, midpoint + half_range), axis=1)
        self._base_mass_binding = self._entity.bind_body_mass_write(
            [0], term_name="a2arm_base_mass"
        )
        self._base_com_binding = self._entity.bind_body_ipos_write([0], term_name="a2arm_base_com")
        self._foot_friction_binding = self._entity.bind_geom_friction_write(
            [0, 1, 2, 3], term_name="a2arm_foot_friction"
        )
        self._gripper_mass_binding = self._ee_entity.bind_body_mass_write(
            [0], term_name="a2arm_gripper_mass"
        )
        self._ee_entity.bind_body_force(term_name="a2arm_gripper_force")
        self._entity.bind_body_force(term_name="a2arm_base_force")
        # Manager terms are constructed before the reset lifecycle opens its
        # ResetStateTransaction.  Initialize only task-owned memory here; all
        # Entity state writes happen from ``reset()`` when CommandManager.reset
        # is called inside ManagerBasedRlEnv.reset's scoped transaction.
        all_ids = np.arange(self.num_envs, dtype=np.int32)
        self._reference_dof_pos[:] = self._entity.data.default_joint_pos[:, :NUM_LEG]
        for schedule in (
            self._sched_gripper_cmd,
            self._sched_gripper_ext,
            self._sched_base_cmd,
            self._sched_base_ext,
        ):
            schedule.reset(all_ids)
        self._initialize_goals(all_ids)

    @property
    def command(self) -> np.ndarray:
        return self._command

    @property
    def force_ee_command(self) -> np.ndarray:
        return self._force_ee_cmd

    @property
    def force_ee_world(self) -> np.ndarray:
        return self._force_ee_world

    @property
    def force_base_command(self) -> np.ndarray:
        return self._force_base_cmd

    @property
    def force_base_world(self) -> np.ndarray:
        return self._force_base_world

    @property
    def gait_phase(self) -> np.ndarray:
        return np.column_stack((self._gait_phase, np.remainder(self._gait_phase + 0.5, 1.0)))

    @property
    def stance_mask(self) -> np.ndarray:
        return self._stance_mask

    @property
    def reference_dof_pos(self) -> np.ndarray:
        return self._reference_dof_pos

    @property
    def foot_contact(self) -> np.ndarray:
        return self._foot_contact

    @property
    def first_contact(self) -> np.ndarray:
        return self._first_contact

    @property
    def air_time_snapshot(self) -> np.ndarray:
        return self._air_time_snapshot

    @property
    def current_goal_sphere(self) -> np.ndarray:
        return self._goal_sphere

    @property
    def current_goal_world(self) -> np.ndarray:
        return self._goal_world

    @property
    def last_dof_vel(self) -> np.ndarray:
        return self._last_dof_vel

    @property
    def soft_dof_pos_limits(self) -> np.ndarray:
        return self._soft_dof_pos_limits

    @property
    def dr_friction(self) -> np.ndarray:
        return self._dr_friction

    @property
    def dr_base_mass(self) -> np.ndarray:
        return self._dr_base_mass

    @property
    def dr_base_com(self) -> np.ndarray:
        return self._dr_base_com

    @property
    def dr_gripper_mass(self) -> np.ndarray:
        return self._dr_gripper_mass

    def goal_center_world(self) -> np.ndarray:
        """Return the legacy yaw-aligned, terrain-relative sphere center."""
        root_pos = self._entity.data.root_link_pos_w
        yaw = np_yaw_quat(self._entity.data.root_link_quat_w)
        center_offset = np.broadcast_to(
            np.asarray(self.cfg.goal_sphere_center, dtype=self._dtype),
            (self.num_envs, 3),
        )
        center = np.zeros((self.num_envs, 3), dtype=self._dtype)
        center[:, :2] = root_pos[:, :2]
        return center + np_quat_apply(yaw, center_offset)

    @property
    def root_quat_world(self) -> np.ndarray:
        """Current root orientation exposed for typed playback overlays."""
        return self._entity.data.root_link_quat_w

    @property
    def root_pos_world(self) -> np.ndarray:
        """Current root position exposed for typed playback overlays."""
        return self._entity.data.root_link_pos_w

    def ee_world_pos(self) -> np.ndarray:
        sensor = self._sensor_view.read()
        endpoint_pos = sensor[:, 0:3]
        arm_base_pos = sensor[:, 7:10]
        arm_base_quat = sensor[:, 10:14]
        return arm_base_pos + np_quat_apply(arm_base_quat, endpoint_pos)

    def foot_force_vec(self) -> np.ndarray:
        return self._foot_force_view.read().reshape(self.num_envs, 4, 3)

    def foot_vel(self) -> np.ndarray:
        return self._foot_vel_view.read().reshape(self.num_envs, 4, 3)

    def foot_pos(self) -> np.ndarray:
        return self._foot_pos_view.read().reshape(self.num_envs, 4, 3)

    def thigh_pos(self) -> np.ndarray:
        return self._thigh_pos_view.read().reshape(self.num_envs, 4, 3)

    def undesired_contacts(self) -> np.ndarray:
        return self._undesired_contact_view.read().reshape(self.num_envs, 15)

    def _resample_velocity(self, ids: np.ndarray) -> None:
        cfg = self.cfg
        for col, bounds in enumerate(cfg.velocity_ranges):
            self._command[ids, col] = self._env.rng.uniform(bounds[0], bounds[1], size=len(ids))
        probability = (
            cfg.zero_velocity_probability_after_force
            if self._env.step_counter >= cfg.force_start_step
            else cfg.zero_velocity_probability
        )
        zero = self._env.rng.uniform(size=len(ids)) < probability
        self._command[ids[zero], :3] = 0.0
        clips = np.asarray(cfg.velocity_clip)
        small = np.all(np.abs(self._command[ids, :3]) < clips, axis=1)
        self._command[ids[small], :3] = 0.0

    def _resample_goals(self, ids: np.ndarray) -> None:
        cfg = self.cfg
        n = len(ids)
        if n == 0:
            return
        starts = self._goal_start[ids]
        candidates = np.broadcast_to(np.asarray(cfg.goal_end, dtype=self._dtype), (n, 3)).copy()
        remaining = np.arange(n, dtype=np.int32)
        for _ in range(max(1, int(cfg.goal_resample_attempts))):
            count = len(remaining)
            candidate = np.column_stack(
                (
                    self._env.rng.uniform(*cfg.goal_radius_range, size=count),
                    self._env.rng.uniform(*cfg.goal_pitch_range, size=count),
                    self._env.rng.uniform(*cfg.goal_yaw_range, size=count),
                )
            ).astype(self._dtype)
            candidates[remaining] = candidate
            unsafe = self._collision_check(starts[remaining], candidate)
            remaining = remaining[unsafe]
            if len(remaining) == 0:
                break
        self._goal_target[ids] = candidates

    def _initialize_goals(self, ids: np.ndarray) -> None:
        cfg = self.cfg
        self._goal_start[ids] = cfg.goal_start
        self._goal_sphere[ids] = self._goal_start[ids]
        self._resample_goals(ids)
        traj = self._env.rng.uniform(*cfg.goal_traj_time_range, size=len(ids))
        hold = self._env.rng.uniform(*cfg.goal_hold_time_range, size=len(ids))
        self._goal_traj_steps[ids] = np.maximum(1, np.rint(traj / cfg.ctrl_dt).astype(np.int32))
        self._goal_total_steps[ids] = self._goal_traj_steps[ids] + np.maximum(
            0, np.rint(hold / cfg.ctrl_dt).astype(np.int32)
        )
        self._goal_timer[ids] = 0
        self._command[ids, CMD_EE_POS] = self._goal_sphere[ids]

    def _collision_check(self, starts: np.ndarray, goals: np.ndarray) -> np.ndarray:
        n = max(2, int(self.cfg.goal_collision_samples))
        t = np.linspace(0.0, 1.0, n, dtype=self._dtype)
        path = starts[:, None, :] + (goals - starts)[:, None, :] * t[None, :, None]
        path = sphere2cart(path.reshape(-1, 3)).reshape(len(starts), n, 3)
        upper = np.asarray(self.cfg.goal_collision_upper, dtype=self._dtype)
        lower = np.asarray(self.cfg.goal_collision_lower, dtype=self._dtype)
        inside = np.all(path < upper, axis=2) & np.all(path > lower, axis=2)
        underground = np.any(path[..., 2] < float(self.cfg.goal_underground_limit), axis=1)
        return np.any(inside, axis=1) | underground

    def prepare_control_step(self, step_token: int) -> None:
        if self._last_control_token == int(step_token):
            return
        self._last_control_token = int(step_token)
        if self._teleop_enabled:
            self._force_ee_cmd.fill(0.0)
            self._force_base_cmd.fill(0.0)
            self._force_ee_world[:] = self._teleop_ee_force
            self._force_base_world[:] = self._teleop_base_force
            self._command[:, CMD_EE_FORCE] = 0.0
            self._command[:, CMD_BASE_FORCE] = 0.0
            self._ee_entity.apply_body_force(
                self._force_ee_world, term_name="a2arm_teleop_ee_force"
            )
            self._entity.apply_body_force(
                self._force_base_world, term_name="a2arm_teleop_base_force"
            )
            return
        enabled = int(step_token) >= self.cfg.force_start_step
        scale = 1.0
        if enabled and self.cfg.force_curriculum_scales:
            index = min(
                (int(step_token) - self.cfg.force_start_step)
                // max(1, self.cfg.force_curriculum_stage_steps),
                len(self.cfg.force_curriculum_scales) - 1,
            )
            scale = float(self.cfg.force_curriculum_scales[index])
        self._force_ee_cmd[:] = self._sched_gripper_cmd.step(enabled) * scale
        self._force_ee_world[:] = self._sched_gripper_ext.step(enabled) * scale
        self._force_base_cmd[:] = self._sched_base_cmd.step(enabled) * scale
        self._force_base_world[:] = self._sched_base_ext.step(enabled) * scale
        self._command[:, CMD_EE_FORCE] = self._force_ee_cmd
        self._command[:, CMD_BASE_FORCE] = self._force_base_cmd
        if enabled and np.any(self._force_ee_world):
            self._ee_entity.apply_body_force(self._force_ee_world, term_name="a2arm_gripper_force")
        if enabled and np.any(self._force_base_world):
            self._entity.apply_body_force(self._force_base_world, term_name="a2arm_base_force")
        self._maybe_apply_velocity_push(int(step_token))

    def _maybe_apply_velocity_push(self, step_token: int) -> None:
        cfg = self.cfg
        if not cfg.velocity_push or cfg.push_interval <= 0 or step_token <= 0:
            return
        if step_token % int(cfg.push_interval) != 0:
            return
        current = self._entity.data.root_link_lin_vel_w
        target = current.copy()
        target[:, :2] = self._env.rng.uniform(
            -float(cfg.max_push_vel_xy), float(cfg.max_push_vel_xy), size=(self.num_envs, 2)
        )
        moving = (
            (np.abs(self._command[:, 0]) > cfg.velocity_clip[0])
            | (np.abs(self._command[:, 1]) > cfg.velocity_clip[1])
            | (np.abs(self._command[:, 2]) > cfg.velocity_clip[2])
        )
        if float(cfg.velocity_push_standing_scale) != 1.0:
            target[~moving, :2] = current[~moving, :2] + (
                target[~moving, :2] - current[~moving, :2]
            ) * float(cfg.velocity_push_standing_scale)
        self._entity.apply_root_linear_velocity_delta_to_sim(
            target - current, term_name="a2arm_velocity_push"
        )

    def prepare_transition(self, step_token: int) -> None:
        if self._last_transition_token == int(step_token):
            return
        if self._pending_dof_vel_token is not None:
            self._last_dof_vel[:] = self._pending_dof_vel
        self._last_transition_token = int(step_token)
        if self._teleop_enabled:
            self._command[:, CMD_VEL] = self._teleop_velocity
            self._command[:, CMD_EE_POS] = self._teleop_ee_sphere
            self._goal_sphere[:] = self._teleop_ee_sphere
        else:
            self._command_timer += 1
            due = self._command_timer >= self.cfg.command_resample_steps
            if np.any(due):
                due_ids = np.flatnonzero(due).astype(np.int32)
                self._resample_velocity(due_ids)
                self._command_timer[due_ids] = 0
        moving = (
            (np.abs(self._command[:, 0]) > self.cfg.velocity_clip[0])
            | (np.abs(self._command[:, 1]) > self.cfg.velocity_clip[1])
            | (np.abs(self._command[:, 2]) > self.cfg.velocity_clip[2])
        )
        self._gait_phase[:] = np.remainder(
            self._gait_phase + float(self.cfg.ctrl_dt) / self.cfg.gait_cycle_time, 1.0
        )
        self._gait_phase[~moving] = 0.0
        sin_pos = np.sin(2.0 * np.pi * self._gait_phase)
        left = sin_pos + self.cfg.gait_target_threshold
        right = sin_pos - self.cfg.gait_target_threshold
        self._stance_mask.fill(0.0)
        self._stance_mask[:, 0] = left >= 0.0
        self._stance_mask[:, 3] = left >= 0.0
        self._stance_mask[:, 1] = right < 0.0
        self._stance_mask[:, 2] = right < 0.0
        default = self._entity.data.default_joint_pos[:, :NUM_LEG]
        self._reference_dof_pos[:] = default
        scale_1 = self.cfg.gait_target_scale / (1.0 - self.cfg.gait_target_threshold)
        scale_2 = 2.0 * scale_1
        left_neg = np.where(left > 0.0, 0.0, left)
        right_neg = np.where(right < 0.0, 0.0, right)
        for thigh, calf in ((1, 2), (10, 11)):
            self._reference_dof_pos[:, thigh] -= left_neg * scale_1
            self._reference_dof_pos[:, calf] += left_neg * scale_2
        for thigh, calf in ((4, 5), (7, 8)):
            self._reference_dof_pos[:, thigh] += right_neg * scale_1
            self._reference_dof_pos[:, calf] -= right_neg * scale_2
        if not self._teleop_enabled:
            interpolation = np.clip(
                self._goal_timer.astype(self._dtype) / np.maximum(self._goal_traj_steps, 1),
                0.0,
                1.0,
            )
            self._goal_sphere[:] = (
                self._goal_start + (self._goal_target - self._goal_start) * interpolation[:, None]
            )
            done = self._goal_timer >= self._goal_total_steps
            if np.any(done):
                done_ids = np.flatnonzero(done).astype(np.int32)
                self._goal_start[done_ids] = self._goal_target[done_ids]
                self._resample_goals(done_ids)
                traj = self._env.rng.uniform(*self.cfg.goal_traj_time_range, size=len(done_ids))
                hold = self._env.rng.uniform(*self.cfg.goal_hold_time_range, size=len(done_ids))
                self._goal_traj_steps[done_ids] = np.maximum(
                    1, np.rint(traj / self.cfg.ctrl_dt).astype(np.int32)
                )
                self._goal_total_steps[done_ids] = self._goal_traj_steps[done_ids] + np.maximum(
                    0, np.rint(hold / self.cfg.ctrl_dt).astype(np.int32)
                )
                self._goal_timer[done_ids] = 0
            self._goal_timer += 1
        self._command[:, CMD_EE_POS] = self._goal_sphere
        self._goal_cart[:] = sphere2cart(self._goal_sphere)
        self._goal_world[:] = self.goal_center_world() + np_quat_apply(
            np_yaw_quat(self._entity.data.root_link_quat_w), self._goal_cart
        )
        contact = self._read_foot_contact()
        contact_filt = contact | self._last_contact
        self._first_contact[:] = (self._feet_air_time > 0.0) & contact_filt
        self._feet_air_time += float(self.cfg.ctrl_dt)
        self._air_time_snapshot[:] = self._feet_air_time
        self._feet_air_time *= ~contact_filt
        self._last_contact[:] = contact
        self._foot_contact[:] = contact
        self._pending_dof_vel[:] = self._entity.data.joint_vel
        self._pending_dof_vel_token = int(step_token)

    def finalize_transition(self, step_token: int) -> None:
        """Retain current velocity for the next step's acceleration penalty."""
        if self._pending_dof_vel_token == int(step_token):
            return

    def _read_foot_contact(self) -> np.ndarray:
        return self._foot_contact_view.read() > 0.5

    def set_teleop_override(
        self,
        command: A2ArmTeleopCommand | None = None,
        *,
        velocity: np.ndarray | None = None,
        ee_sphere: np.ndarray | None = None,
        ee_force: np.ndarray | None = None,
        base_force: np.ndarray | None = None,
    ) -> None:
        """Install a typed interactive override without patching env methods."""
        if command is not None:
            if any(value is not None for value in (velocity, ee_sphere, ee_force, base_force)):
                raise ValueError("pass either command or individual teleop arrays, not both")
        else:
            if any(value is None for value in (velocity, ee_sphere, ee_force, base_force)):
                raise ValueError("all teleop arrays are required when command is omitted")
            command = A2ArmTeleopCommand(
                velocity=velocity,  # type: ignore[arg-type]
                ee_sphere=ee_sphere,  # type: ignore[arg-type]
                ee_force=ee_force,  # type: ignore[arg-type]
                base_force=base_force,  # type: ignore[arg-type]
            )
        values = {
            "velocity": np.asarray(command.velocity, dtype=self._dtype),
            "ee_sphere": np.asarray(command.ee_sphere, dtype=self._dtype),
            "ee_force": np.asarray(command.ee_force, dtype=self._dtype),
            "base_force": np.asarray(command.base_force, dtype=self._dtype),
        }
        for name, value in values.items():
            if value.shape == (3,):
                value = value[None, :]
                values[name] = value
            if value.shape != (self.num_envs, 3):
                raise ValueError(f"teleop {name} must have shape ({self.num_envs}, 3)")
        self._teleop_velocity[:] = values["velocity"]
        self._teleop_ee_sphere[:] = values["ee_sphere"]
        self._teleop_ee_force[:] = values["ee_force"]
        self._teleop_base_force[:] = values["base_force"]
        self._teleop_enabled = True

    def clear_teleop_override(self) -> None:
        self._teleop_enabled = False
        self._teleop_velocity.fill(0.0)
        self._teleop_ee_force.fill(0.0)
        self._teleop_base_force.fill(0.0)
        self._force_ee_cmd.fill(0.0)
        self._force_ee_world.fill(0.0)
        self._force_base_cmd.fill(0.0)
        self._force_base_world.fill(0.0)
        self._command[:, CMD_EE_FORCE] = 0.0
        self._command[:, CMD_BASE_FORCE] = 0.0
        self._resample_velocity(np.arange(self.num_envs, dtype=np.int32))

    def reset(self, env_ids: np.ndarray | slice | None = None) -> dict[str, float]:
        ids = (
            np.arange(self.num_envs, dtype=np.int32)
            if env_ids is None or isinstance(env_ids, slice)
            else np.asarray(env_ids, dtype=np.int32)
        )
        # Reset root and DoF state inside the manager-owned transaction.  This
        # is the legacy spawn distribution expressed through Entity's public
        # reset contract; no backend model/data object is touched here.
        root = np.array(self._entity.data.default_root_state[ids], copy=True)
        root[:, 0:2] += self._env.rng.uniform(-0.5, 0.5, size=(len(ids), 2))
        yaw = self._env.rng.uniform(
            -self.cfg.root_yaw_range, self.cfg.root_yaw_range, size=len(ids)
        )
        root[:, 3:7] = np.column_stack(
            (np.cos(yaw / 2.0), np.zeros(len(ids)), np.zeros(len(ids)), np.sin(yaw / 2.0))
        )
        root[:, 7:13] = self._env.rng.uniform(-0.5, 0.5, size=(len(ids), 6))
        self._entity.write_root_link_pose_to_sim(root[:, :7], env_ids=ids)
        self._entity.write_root_link_velocity_to_sim(root[:, 7:13], env_ids=ids)
        q = np.array(self._entity.data.default_joint_pos[ids], copy=True)
        q[:, :NUM_LEG] *= self._env.rng.uniform(0.5, 1.5, size=(len(ids), NUM_LEG))
        q[:, NUM_LEG:] += self._env.rng.uniform(-0.3, 0.3, size=(len(ids), NUM_ACTIONS - NUM_LEG))
        self._entity.write_joint_state_to_sim(q, np.zeros_like(q), env_ids=ids)
        self._command[ids] = 0.0
        # Legacy reset samples a fresh velocity command and a random phase in
        # the fixed resampling interval; preserving both avoids a standing-only
        # startup and keeps command cadence distribution unchanged.
        self._resample_velocity(ids)
        self._command_timer[ids] = self._env.rng.integers(
            0, max(1, int(self.cfg.command_resample_steps)), size=len(ids)
        )
        self._gait_phase[ids] = 0.0
        self._stance_mask[ids] = 0.0
        self._reference_dof_pos[ids] = self._entity.data.default_joint_pos[ids, :NUM_LEG]
        self._foot_contact[ids] = False
        self._last_contact[ids] = False
        self._feet_air_time[ids] = 0.0
        self._first_contact[ids] = False
        self._air_time_snapshot[ids] = 0.0
        self._initialize_goals(ids)
        self._last_dof_vel[ids] = 0.0
        self._pending_dof_vel[ids] = 0.0
        self._pending_dof_vel_token = None
        self._apply_reset_randomization(ids)
        for schedule in (
            self._sched_gripper_cmd,
            self._sched_gripper_ext,
            self._sched_base_cmd,
            self._sched_base_ext,
        ):
            schedule.reset(ids)
        self._force_ee_cmd[ids] = 0.0
        self._force_ee_world[ids] = 0.0
        self._force_base_cmd[ids] = 0.0
        self._force_base_world[ids] = 0.0
        return {}

    def _resample_command(self, env_ids: np.ndarray) -> None:
        del env_ids

    def _update_metrics(self, env_ids: np.ndarray | None = None) -> None:
        del env_ids

    def _update_command(self, env_ids: np.ndarray | None) -> None:
        del env_ids

    def _apply_reset_randomization(self, ids: np.ndarray) -> None:
        cfg = self.cfg
        n = len(ids)
        if n == 0:
            return
        base_mass_default = self._base_mass_binding[1]
        gripper_mass_default = self._gripper_mass_binding[1]
        if cfg.randomize_base_mass:
            delta = self._env.rng.uniform(*cfg.added_mass_range, size=n).astype(self._dtype)
            self._dr_base_mass[ids, 0] = delta
            self._entity.write_body_mass_to_sim(
                base_mass_default[None, :] + delta[:, None], [0], ids, term_name="a2arm_base_mass"
            )
        else:
            self._dr_base_mass[ids] = 0.0
        if cfg.random_com:
            offset = np.column_stack(
                [
                    self._env.rng.uniform(*cfg.com_offset_x, size=n),
                    self._env.rng.uniform(*cfg.com_offset_y, size=n),
                    self._env.rng.uniform(*cfg.com_offset_z, size=n),
                ]
            ).astype(self._dtype)
            self._dr_base_com[ids] = offset
            self._entity.write_body_ipos_to_sim(
                self._base_com_binding[1][None, :, :] + offset[:, None, :],
                [0],
                ids,
                term_name="a2arm_base_com",
            )
        else:
            self._dr_base_com[ids] = 0.0
        if cfg.randomize_gripper_mass:
            delta = self._env.rng.uniform(*cfg.gripper_added_mass_range, size=n).astype(self._dtype)
            self._dr_gripper_mass[ids, 0] = delta
            self._ee_entity.write_body_mass_to_sim(
                gripper_mass_default[None, :] + delta[:, None],
                [0],
                ids,
                term_name="a2arm_gripper_mass",
            )
        else:
            self._dr_gripper_mass[ids] = 0.0
        if cfg.randomize_foot_friction:
            friction = np.broadcast_to(self._foot_friction_binding[1], (n, 4, 3)).copy()
            samples = self._env.rng.uniform(*cfg.foot_friction_range, size=n).astype(self._dtype)
            friction[:, :, 0] = samples[:, None]
            self._entity.write_geom_friction_to_sim(
                friction, [0, 1, 2, 3], ids, term_name="a2arm_foot_friction"
            )
            self._dr_friction[ids, 0] = samples
        else:
            self._dr_friction[ids] = 0.0


__all__ = [
    "A2ArmPosForceCommandCfg",
    "A2ArmPosForceState",
    "ForceSchedule",
    "cart2sphere",
    "sphere2cart",
]
