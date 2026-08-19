"""Hydra-owned Manager-Based terms for the Go2 footstand task.

The task keeps its historical NumPy observation, action, reward, termination,
and reset semantics while using only the public manager/entity facade.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from unilab.base import registry
from unilab.dtype_config import get_global_dtype
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env
from unilab.managers import ActionTerm, ActionTermCfg, ManagerTermBase, ManagerTermBaseCfg
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.utils.rotation import np_quat_apply, np_quat_apply_inverse

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv, ManagerSensorView
    from unilab.managers.action_manager import ActionManager
    from unilab.managers.termination_manager import TerminationManager

    class _FootstandEnv(ManagerBasedRlEnv, Protocol):
        @property
        def common_step_counter(self) -> int: ...

        @property
        def action_manager(self) -> ActionManager: ...

        @property
        def termination_manager(self) -> TerminationManager: ...


NUM_ACTIONS = 12
FRAME_OBS_DIM = 45
PRIVILEGED_OBS_DIM = 49

_WORLD_GRAVITY = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
_BODY_FORWARD = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
_TARGET_HEIGHT = 0.53
_CONTACT_THRESHOLD = 0.1
_STAND_HEIGHT_FRACTION = 0.8
_STAND_ORIENTATION_THRESHOLD = 0.5

_FRONT_FEET = np.asarray([0, 1], dtype=np.intp)
_REAR_FEET = np.asarray([2, 3], dtype=np.intp)
_FRONT_LEGS = np.arange(0, 6, dtype=np.intp)
_REAR_LEGS = np.arange(6, 12, dtype=np.intp)
_REAR_HIPS = np.asarray([6, 9], dtype=np.intp)
_REAR_LEFT = np.asarray([6, 7, 8], dtype=np.intp)
_REAR_RIGHT = np.asarray([9, 10, 11], dtype=np.intp)
_REAR_MIRROR_SIGNS = np.asarray([-1.0, 1.0, 1.0], dtype=np.float32)
_FRONT_LEG_TARGET = np.asarray([0.0, 1.82, -1.16, 0.0, 1.82, -1.16], dtype=np.float32)

_TRACKED_BODY_NAMES = (
    "FL_thigh",
    "FR_thigh",
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
)
_FRONT_LEFT_BODY_INDICES = np.asarray([0, 2], dtype=np.intp)
_FRONT_RIGHT_BODY_INDICES = np.asarray([1, 3], dtype=np.intp)
_KNEE_BODY_INDICES = np.asarray([2, 3, 4, 5], dtype=np.intp)

_SENSOR_SPECS = (
    ("local_linvel", 3),
    ("gyro", 3),
    ("upvector", 3),
    ("global_position", 3),
    ("accelerometer", 3),
    ("global_angvel", 3),
    ("FL_foot_contact", 1),
    ("FR_foot_contact", 1),
    ("RL_foot_contact", 1),
    ("RR_foot_contact", 1),
    ("FL_pos", 3),
    ("FR_pos", 3),
    ("RL_pos", 3),
    ("RR_pos", 3),
    ("base1_contact", 1),
    ("base2_contact", 1),
    ("base3_contact", 1),
    ("RL_hip_contact", 1),
    ("RR_hip_contact", 1),
    ("RL_thigh_contact", 1),
    ("RR_thigh_contact", 1),
    ("RL_calf_contact1", 1),
    ("RL_calf_contact2", 1),
    ("RR_calf_contact1", 1),
    ("RR_calf_contact2", 1),
    ("FL_hip_contact", 1),
    ("FR_hip_contact", 1),
    ("FL_thigh_contact", 1),
    ("FR_thigh_contact", 1),
    ("FL_calf_contact1", 1),
    ("FL_calf_contact2", 1),
    ("FR_calf_contact1", 1),
    ("FR_calf_contact2", 1),
)
_FOOT_CONTACT_NAMES = tuple(name for name, _ in _SENSOR_SPECS[6:10])
_FOOT_POSITION_NAMES = tuple(name for name, _ in _SENSOR_SPECS[10:14])
_TERMINATION_CONTACT_NAMES = tuple(name for name, _ in _SENSOR_SPECS[14:25])
_PENALTY_CONTACT_NAMES = tuple(name for name, _ in _SENSOR_SPECS[25:33])


def _real(
    term: str,
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{term} {name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{term} {name} must be finite")
    if minimum is not None and (result <= minimum if strict_minimum else result < minimum):
        relation = "greater than" if strict_minimum else "at least"
        raise ValueError(f"{term} {name} must be {relation} {minimum}")
    return result


def _pair(
    term: str,
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{term} {name} must be a two-value range")
    if len(value) != 2:
        raise ValueError(f"{term} {name} must contain two values")
    lower = _real(term, f"{name}[0]", value[0], minimum=minimum)
    upper = _real(term, f"{name}[1]", value[1], minimum=minimum)
    if lower > upper:
        raise ValueError(f"{term} {name} lower bound {lower} exceeds upper bound {upper}")
    return lower, upper


def _name(term: str, field: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{term} {field} must be a non-empty string")
    return value


def _env_ids(env: ManagerBasedRlEnv, env_ids: np.ndarray | slice | None) -> np.ndarray:
    if env_ids is None:
        return np.arange(env.num_envs, dtype=np.int32)
    if isinstance(env_ids, slice):
        return np.arange(env.num_envs, dtype=np.int32)[env_ids]
    return np.asarray(env_ids, dtype=np.int32)


@dataclass(kw_only=True)
class FootstandIncrementalActionCfg(ActionTermCfg):
    """Incremental position action in the historical actuator/policy order."""

    actuator_names: tuple[str, ...] | list[str]
    joint_names: tuple[str, ...] | list[str]
    joint_position_limits: tuple[tuple[float, float], ...] | list[list[float]]
    action_scale: float = 0.3
    clip_actions: float = 1.0
    kp: float = 35.0
    kd: float = 0.5
    simulate_action_latency: bool = False

    def build(self, env: ManagerBasedRlEnv) -> FootstandIncrementalAction:
        return FootstandIncrementalAction(self, env)


class FootstandIncrementalAction(ActionTerm):
    """Integrate clipped policy deltas and write position targets each substep."""

    cfg: FootstandIncrementalActionCfg
    _entity: Entity

    def __init__(self, cfg: FootstandIncrementalActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        term = type(self).__name__
        if cfg.clip is not None:
            raise NotImplementedError(f"{term} does not support actuator-name clip")
        for field_name, patterns in (
            ("actuator_names", cfg.actuator_names),
            ("joint_names", cfg.joint_names),
        ):
            if isinstance(patterns, (str, bytes)) or not isinstance(patterns, (tuple, list)):
                raise TypeError(f"{term} {field_name} must be an ordered sequence of patterns")
            if len(patterns) != NUM_ACTIONS:
                raise ValueError(f"{term} requires {NUM_ACTIONS} ordered {field_name} patterns")
        self._scale = _real(term, "action_scale", cfg.action_scale, minimum=0.0)
        self._clip_actions = _real(
            term, "clip_actions", cfg.clip_actions, minimum=0.0, strict_minimum=True
        )
        self._kp = _real(term, "kp", cfg.kp, minimum=0.0)
        self._kd = _real(term, "kd", cfg.kd, minimum=0.0)
        if not isinstance(cfg.simulate_action_latency, bool):
            raise TypeError(f"{term} simulate_action_latency must be bool")

        actuator_ids: list[int] = []
        joint_ids: list[int] = []
        actuator_names: list[str] = []
        joint_names: list[str] = []
        for actuator_pattern, joint_pattern in zip(
            cfg.actuator_names, cfg.joint_names, strict=True
        ):
            if not isinstance(actuator_pattern, str) or not actuator_pattern:
                raise ValueError(f"{term} actuator patterns must be non-empty strings")
            if not isinstance(joint_pattern, str) or not joint_pattern:
                raise ValueError(f"{term} joint patterns must be non-empty strings")
            found_actuator_ids, found_actuator_names = self._entity.find_actuators(
                (actuator_pattern,), preserve_order=True
            )
            found_joint_ids, found_joint_names = self._entity.find_joints(
                (joint_pattern,), preserve_order=True
            )
            if len(found_actuator_ids) != 1 or len(found_joint_ids) != 1:
                raise ValueError(
                    f"{term} patterns actuator={actuator_pattern!r}, joint={joint_pattern!r} "
                    "must each resolve exactly once; "
                    f"got actuators={found_actuator_names}, joints={found_joint_names}"
                )
            actuator_ids.append(found_actuator_ids[0])
            joint_ids.append(found_joint_ids[0])
            actuator_names.extend(found_actuator_names)
            joint_names.extend(found_joint_names)
        if len(set(actuator_ids)) != NUM_ACTIONS or len(set(joint_ids)) != NUM_ACTIONS:
            raise ValueError(f"{term} actuator-to-joint mapping must be one-to-one")
        if set(joint_ids) != set(range(self._entity.num_joints)):
            raise ValueError(f"{term} must control every declared Go2 joint exactly once")

        self._actuator_ids = np.asarray(actuator_ids, dtype=np.intp)
        self._joint_ids = np.asarray(joint_ids, dtype=np.intp)
        self._actuator_ids.setflags(write=False)
        self._joint_ids.setflags(write=False)
        self._actuator_names = tuple(actuator_names)
        self._joint_names = tuple(joint_names)

        try:
            selected_ranges = np.asarray(cfg.joint_position_limits, dtype=get_global_dtype())
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{term} joint_position_limits must be numeric") from exc
        if selected_ranges.shape != (NUM_ACTIONS, 2):
            raise ValueError(f"{term} joint_position_limits must have shape ({NUM_ACTIONS}, 2)")
        if not np.isfinite(selected_ranges).all():
            raise ValueError(f"{term} joint_position_limits must be finite")
        if np.any(selected_ranges[:, 0] >= selected_ranges[:, 1]):
            raise ValueError(f"{term} joint_position_limits must have lower < upper")
        self._target_lower = np.asarray(selected_ranges[:, 0], dtype=get_global_dtype())
        self._target_upper = np.asarray(selected_ranges[:, 1], dtype=get_global_dtype())
        self._joint_lower = np.empty((NUM_ACTIONS,), dtype=get_global_dtype())
        self._joint_upper = np.empty_like(self._joint_lower)
        self._joint_lower[self._joint_ids] = self._target_lower
        self._joint_upper[self._joint_ids] = self._target_upper

        dtype = get_global_dtype()
        shape = (env.num_envs, NUM_ACTIONS)
        self._raw_action = np.zeros(shape, dtype=dtype)
        self._previous_raw_action = np.zeros_like(self._raw_action)
        self._target = np.asarray(
            self._entity.data.joint_pos[:, self._joint_ids], dtype=dtype
        ).copy()
        self._state = FootstandState(cast("_FootstandEnv", env), self)

    @property
    def action_dim(self) -> int:
        return NUM_ACTIONS

    @property
    def raw_action(self) -> np.ndarray:
        return self._raw_action

    @property
    def previous_raw_action(self) -> np.ndarray:
        return self._previous_raw_action

    @property
    def target(self) -> np.ndarray:
        return self._target

    @property
    def joint_ids(self) -> np.ndarray:
        return self._joint_ids

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._joint_names

    @property
    def actuator_names(self) -> tuple[str, ...]:
        return self._actuator_names

    @property
    def joint_lower(self) -> np.ndarray:
        return self._joint_lower

    @property
    def joint_upper(self) -> np.ndarray:
        return self._joint_upper

    @property
    def state(self) -> FootstandState:
        return self._state

    @property
    def entity(self) -> Entity:
        return self._entity

    @property
    def estimated_torque(self) -> np.ndarray:
        return self._state.torques

    def process_actions(self, actions: np.ndarray) -> None:
        if not isinstance(actions, np.ndarray):
            raise TypeError(f"expected np.ndarray actions, got {type(actions).__name__}")
        if actions.shape != self._raw_action.shape:
            raise ValueError(f"expected action shape {self._raw_action.shape}, got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("received NaN or Inf actions")
        self._previous_raw_action[:] = self._raw_action
        np.clip(actions, -self._clip_actions, self._clip_actions, out=self._raw_action)
        executed = (
            self._previous_raw_action if self.cfg.simulate_action_latency else self._raw_action
        )
        self._target += self._scale * executed
        np.clip(self._target, self._target_lower, self._target_upper, out=self._target)

    def apply_actions(self) -> None:
        self._entity.set_joint_position_target(self._target, joint_ids=self._joint_ids)

    def estimate_torque(
        self, joint_pos: np.ndarray, joint_vel: np.ndarray, out: np.ndarray
    ) -> None:
        out.fill(0.0)
        selected_pos = joint_pos[:, self._joint_ids]
        selected_vel = joint_vel[:, self._joint_ids]
        out[:, self._joint_ids] = self._kp * (self._target - selected_pos) - self._kd * selected_vel

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = _env_ids(self._env, env_ids)
        self._raw_action[ids] = 0.0
        self._previous_raw_action[ids] = 0.0
        joint_pos = self._entity.data.joint_pos
        self._target[ids] = joint_pos[ids][:, self._joint_ids]
        self._state.reset(ids)


class FootstandState:
    """One per-control-step snapshot shared by termination, reward, and observations."""

    def __init__(self, env: _FootstandEnv, action: FootstandIncrementalAction):
        self._env = env
        self._action = action
        self._entity = action.entity
        names = tuple(name for name, _ in _SENSOR_SPECS)
        self._sensor_view: ManagerSensorView = env.scene.bind_sensor_data(names)
        expected_dims = tuple(width for _, width in _SENSOR_SPECS)
        if self._sensor_view.dimensions != expected_dims:
            raise ValueError(
                "Footstand named-sensor dimensions differ from the task contract: "
                f"expected={expected_dims}, got={self._sensor_view.dimensions}"
            )
        offsets = np.cumsum((0, *expected_dims), dtype=np.intp)
        self._sensor_slices = {
            name: slice(int(offsets[index]), int(offsets[index + 1]))
            for index, (name, _) in enumerate(_SENSOR_SPECS)
        }
        tracked_ids, tracked_names = self._entity.find_bodies(
            _TRACKED_BODY_NAMES, preserve_order=True
        )
        if tuple(tracked_names) != _TRACKED_BODY_NAMES:
            raise ValueError(
                f"Footstand tracked body order differs from the task contract: {tracked_names}"
            )
        self._tracked_body_ids = np.asarray(tracked_ids, dtype=np.intp)
        self._tracked_body_ids.setflags(write=False)

        dtype = get_global_dtype()
        num_envs = env.num_envs
        self.linvel = np.zeros((num_envs, 3), dtype=dtype)
        self.gyro = np.zeros_like(self.linvel)
        self.gravity = np.broadcast_to(_WORLD_GRAVITY, (num_envs, 3)).astype(dtype, copy=True)
        self.upvector = -self.gravity.copy()
        self.accelerometer = np.zeros_like(self.linvel)
        self.global_angvel = np.zeros_like(self.linvel)
        self.root_pos = np.zeros_like(self.linvel)
        self.root_quat = np.zeros((num_envs, 4), dtype=dtype)
        self.root_quat[:, 0] = 1.0
        self.root_linvel_w = np.zeros_like(self.linvel)
        self.root_angvel_w = np.zeros_like(self.linvel)
        self.joint_pos = np.asarray(self._entity.data.default_joint_pos, dtype=dtype).copy()
        self.joint_vel = np.zeros_like(self.joint_pos)
        self.qacc = np.zeros_like(self.joint_pos)
        self.torques = np.zeros_like(self.joint_pos)
        self.height = np.zeros((num_envs,), dtype=dtype)
        self.orientation = np.zeros((num_envs,), dtype=dtype)
        self.foot_contact = np.zeros((num_envs, 4), dtype=np.bool_)
        self.foot_pos = np.zeros((num_envs, 4, 3), dtype=dtype)
        self.termination_contact = np.zeros((num_envs,), dtype=np.bool_)
        self.penalty_contact = np.zeros((num_envs,), dtype=np.bool_)
        self.tracked_body_pos = np.zeros((num_envs, len(_TRACKED_BODY_NAMES), 3), dtype=dtype)
        self.rear_speed = np.zeros((num_envs, 2), dtype=dtype)
        self.rear_anchor_drift = np.zeros((num_envs, 2), dtype=dtype)
        self.rear_anchor_contact = np.zeros((num_envs, 2), dtype=np.bool_)
        self._last_foot_pos = np.zeros_like(self.foot_pos)
        self._rear_anchor_pos = np.zeros((num_envs, 2, 2), dtype=dtype)
        self._last_counter = int(env.common_step_counter)

    @property
    def last_counter(self) -> int:
        return self._last_counter

    @property
    def default_joint_pos(self) -> np.ndarray:
        return self._entity.data.default_joint_pos

    @property
    def action(self) -> FootstandIncrementalAction:
        return self._action

    def _sensor(self, values: np.ndarray, name: str) -> np.ndarray:
        return values[:, self._sensor_slices[name]]

    def _capture(self) -> dict[str, np.ndarray]:
        dtype = get_global_dtype()
        sensors = np.asarray(self._sensor_view.read(), dtype=dtype)
        root_quat = np.asarray(self._entity.data.root_link_quat_w, dtype=dtype)
        gravity_w = np.broadcast_to(_WORLD_GRAVITY, (self._env.num_envs, 3))
        gravity = np.asarray(np_quat_apply_inverse(root_quat, gravity_w), dtype=dtype)
        forward_w = np_quat_apply(
            root_quat, np.broadcast_to(_BODY_FORWARD, (self._env.num_envs, 3))
        )
        orientation = np.asarray(np.square(0.5 * forward_w[:, 2] + 0.5), dtype=dtype)
        foot_contact = (
            np.concatenate([self._sensor(sensors, name) for name in _FOOT_CONTACT_NAMES], axis=1)
            > _CONTACT_THRESHOLD
        )
        foot_pos = np.stack([self._sensor(sensors, name) for name in _FOOT_POSITION_NAMES], axis=1)
        termination_contact = np.any(
            np.concatenate(
                [self._sensor(sensors, name) for name in _TERMINATION_CONTACT_NAMES], axis=1
            ),
            axis=1,
        )
        penalty_contact = np.any(
            np.concatenate(
                [self._sensor(sensors, name) for name in _PENALTY_CONTACT_NAMES], axis=1
            ),
            axis=1,
        )
        return {
            "linvel": self._sensor(sensors, "local_linvel"),
            "gyro": self._sensor(sensors, "gyro"),
            "gravity": gravity,
            "upvector": self._sensor(sensors, "upvector"),
            "accelerometer": self._sensor(sensors, "accelerometer"),
            "global_angvel": self._sensor(sensors, "global_angvel"),
            "root_pos": np.asarray(self._entity.data.root_link_pos_w, dtype=dtype),
            "root_quat": root_quat,
            "root_linvel_w": np.asarray(self._entity.data.root_link_lin_vel_w, dtype=dtype),
            "root_angvel_w": np.asarray(self._entity.data.root_link_ang_vel_w, dtype=dtype),
            "joint_pos": np.asarray(self._entity.data.joint_pos, dtype=dtype),
            "joint_vel": np.asarray(self._entity.data.joint_vel, dtype=dtype),
            "height": self._sensor(sensors, "global_position")[:, 2],
            "orientation": orientation,
            "foot_contact": foot_contact,
            "foot_pos": foot_pos,
            "termination_contact": termination_contact,
            "penalty_contact": penalty_contact,
            "tracked_body_pos": np.asarray(
                self._entity.data.body_link_pos_w[:, self._tracked_body_ids], dtype=dtype
            ),
        }

    def reset(self, env_ids: np.ndarray) -> None:
        values = self._capture()
        self.linvel[env_ids] = values["linvel"][env_ids]
        self.gyro[env_ids] = values["gyro"][env_ids]
        self.gravity[env_ids] = values["gravity"][env_ids]
        self.upvector[env_ids] = values["upvector"][env_ids]
        self.accelerometer[env_ids] = values["accelerometer"][env_ids]
        self.global_angvel[env_ids] = values["global_angvel"][env_ids]
        self.root_pos[env_ids] = values["root_pos"][env_ids]
        self.root_quat[env_ids] = values["root_quat"][env_ids]
        self.root_linvel_w[env_ids] = values["root_linvel_w"][env_ids]
        self.root_angvel_w[env_ids] = values["root_angvel_w"][env_ids]
        self.joint_pos[env_ids] = values["joint_pos"][env_ids]
        self.joint_vel[env_ids] = values["joint_vel"][env_ids]
        self.height[env_ids] = values["height"][env_ids]
        self.orientation[env_ids] = values["orientation"][env_ids]
        self.foot_contact[env_ids] = values["foot_contact"][env_ids]
        self.foot_pos[env_ids] = values["foot_pos"][env_ids]
        self.termination_contact[env_ids] = values["termination_contact"][env_ids]
        self.penalty_contact[env_ids] = values["penalty_contact"][env_ids]
        self.tracked_body_pos[env_ids] = values["tracked_body_pos"][env_ids]
        self.qacc[env_ids] = 0.0
        torque = np.empty_like(self.torques)
        self._action.estimate_torque(self.joint_pos, self.joint_vel, torque)
        self.torques[env_ids] = torque[env_ids]
        self._last_foot_pos[env_ids] = self.foot_pos[env_ids]
        self.rear_speed[env_ids] = 0.0
        self._rear_anchor_pos[env_ids] = self.foot_pos[env_ids][:, _REAR_FEET, :2]
        self.rear_anchor_contact[env_ids] = False
        self.rear_anchor_drift[env_ids] = 0.0
        self._last_counter = int(self._env.common_step_counter)

    def snapshot(self, env: _FootstandEnv) -> FootstandState:
        counter = int(env.common_step_counter)
        if counter == self._last_counter:
            return self
        if counter != self._last_counter + 1:
            raise RuntimeError(
                "FootstandState missed a control-step update: "
                f"last={self._last_counter}, current={counter}"
            )
        values = self._capture()
        new_joint_vel = values["joint_vel"]
        np.subtract(new_joint_vel, self.joint_vel, out=self.qacc)
        self.qacc /= env.step_dt

        new_foot_pos = values["foot_pos"]
        rear_delta = new_foot_pos[:, _REAR_FEET, :2] - self._last_foot_pos[:, _REAR_FEET, :2]
        self.rear_speed[:] = np.linalg.norm(rear_delta / env.step_dt, axis=2)

        standing = (values["height"] >= _TARGET_HEIGHT * _STAND_HEIGHT_FRACTION) & (
            values["orientation"] >= _STAND_ORIENTATION_THRESHOLD
        )
        anchor_contact = values["foot_contact"][:, _REAR_FEET] & standing[:, None]
        rear_xy = new_foot_pos[:, _REAR_FEET, :2]
        new_contact = anchor_contact & ~self.rear_anchor_contact
        self._rear_anchor_pos[new_contact] = rear_xy[new_contact]
        self.rear_anchor_contact[:] = anchor_contact
        self.rear_anchor_drift[:] = np.linalg.norm(rear_xy - self._rear_anchor_pos, axis=2)

        self.linvel[:] = values["linvel"]
        self.gyro[:] = values["gyro"]
        self.gravity[:] = values["gravity"]
        self.upvector[:] = values["upvector"]
        self.accelerometer[:] = values["accelerometer"]
        self.global_angvel[:] = values["global_angvel"]
        self.root_pos[:] = values["root_pos"]
        self.root_quat[:] = values["root_quat"]
        self.root_linvel_w[:] = values["root_linvel_w"]
        self.root_angvel_w[:] = values["root_angvel_w"]
        self.joint_pos[:] = values["joint_pos"]
        self.joint_vel[:] = values["joint_vel"]
        self.height[:] = values["height"]
        self.orientation[:] = values["orientation"]
        self.foot_contact[:] = values["foot_contact"]
        self.foot_pos[:] = values["foot_pos"]
        self.termination_contact[:] = values["termination_contact"]
        self.penalty_contact[:] = values["penalty_contact"]
        self.tracked_body_pos[:] = values["tracked_body_pos"]
        self._action.estimate_torque(self.joint_pos, self.joint_vel, self.torques)
        self._last_foot_pos[:] = self.foot_pos
        self._last_counter = counter
        return self

    def frame(self, env: _FootstandEnv) -> np.ndarray:
        self.snapshot(env)
        return np.concatenate(
            (
                self.linvel,
                self.gyro,
                self.gravity,
                self.joint_pos - self.default_joint_pos,
                self.joint_vel,
                self._action.previous_raw_action,
            ),
            axis=1,
            dtype=get_global_dtype(),
        )

    def privileged(self, env: _FootstandEnv) -> np.ndarray:
        self.snapshot(env)
        return np.concatenate(
            (
                self.gyro,
                self.accelerometer,
                self.linvel,
                self.global_angvel,
                self.joint_pos,
                self.joint_vel,
                self.torques,
                self.height[:, None],
            ),
            axis=1,
            dtype=get_global_dtype(),
        )


def _action(env: _FootstandEnv, action_name: str) -> FootstandIncrementalAction:
    name = _name("Footstand manager term", "action_name", action_name)
    try:
        action = env.action_manager.get_term(name)
    except KeyError as exc:
        raise KeyError(f"Footstand action term {name!r} is unavailable") from exc
    if not isinstance(action, FootstandIncrementalAction):
        raise TypeError(
            f"Footstand action term {name!r} must be FootstandIncrementalAction, "
            f"got {type(action).__name__}"
        )
    return action


def frame_observation(env: _FootstandEnv, action_name: str) -> np.ndarray:
    return _action(env, action_name).state.frame(env)


def privileged_observation(env: _FootstandEnv, action_name: str) -> np.ndarray:
    return _action(env, action_name).state.privileged(env)


class FootstandTermination(ManagerTermBase):
    """Aggregate the historical non-timeout termination state before rewards."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        task_env = cast("_FootstandEnv", env)
        term = type(self).__name__
        allowed = {
            "action_name",
            "grace_steps",
            "height_fraction",
            "orientation_threshold",
            "energy_threshold",
        }
        unknown = sorted(set(cfg.params) - allowed)
        if unknown:
            raise TypeError(f"{term} received unsupported parameters: {unknown}")
        self._state = _action(
            task_env, _name(term, "action_name", cfg.params.get("action_name"))
        ).state
        grace = cfg.params.get("grace_steps")
        if isinstance(grace, (bool, np.bool_)) or not isinstance(grace, (int, np.integer)):
            raise TypeError(f"{term} grace_steps must be an integer")
        if int(grace) < 0:
            raise ValueError(f"{term} grace_steps must be non-negative")
        self._grace_steps = int(grace)
        self._height_fraction = _real(
            term, "height_fraction", cfg.params.get("height_fraction"), minimum=0.0
        )
        self._orientation_threshold = _real(
            term, "orientation_threshold", cfg.params.get("orientation_threshold"), minimum=0.0
        )
        self._energy_threshold = _real(
            term, "energy_threshold", cfg.params.get("energy_threshold"), minimum=0.0
        )
        self.terminated = np.zeros(env.num_envs, dtype=np.bool_)
        self._last_counter = int(task_env.common_step_counter)

    @property
    def state(self) -> FootstandState:
        return self._state

    @property
    def last_counter(self) -> int:
        return self._last_counter

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        self.terminated[_env_ids(self._env, env_ids)] = False
        self._last_counter = int(cast("_FootstandEnv", self._env).common_step_counter)

    def __call__(self, env: _FootstandEnv, **params: Any) -> np.ndarray:
        del params
        state = self._state.snapshot(env)
        previous_steps = np.maximum(env.episode_length_buf - 1, 0)
        grace_elapsed = previous_steps >= self._grace_steps
        low_height = state.height < _TARGET_HEIGHT * self._height_fraction
        bad_orientation = state.orientation < self._orientation_threshold
        pose_failure = grace_elapsed & (low_height | bad_orientation)
        energy = np.sum(np.abs(state.torques) * np.abs(state.joint_vel), axis=1)
        energy_failure = energy > self._energy_threshold
        upside_down = state.upvector[:, 2] < -0.25
        self.terminated[:] = np.logical_or.reduce(
            (state.termination_contact, upside_down, energy_failure, pose_failure)
        )
        self._last_counter = int(env.common_step_counter)
        return self.terminated


def _termination(env: _FootstandEnv, state_term_name: str) -> FootstandTermination:
    name = _name("Footstand reward", "state_term_name", state_term_name)
    state_term = env.termination_manager.get_term_cfg(name).func
    if not isinstance(state_term, FootstandTermination):
        raise TypeError(
            f"Footstand termination term {name!r} must be FootstandTermination, "
            f"got {type(state_term).__name__}"
        )
    if state_term.last_counter != int(env.common_step_counter):
        raise RuntimeError(
            f"Footstand termination state {name!r} was not computed for control step "
            f"{env.common_step_counter}"
        )
    return state_term


class FootstandReward(ManagerTermBase):
    """Historical positive-clipped reward aggregate backed by one state snapshot."""

    _REWARD_NAMES = frozenset(
        {
            "height",
            "contact",
            "orientation",
            "oritentation",
            "action_rate",
            "termination",
            "dof_pos_limits",
            "torques",
            "pose",
            "penalty_contact",
            "tar",
            "rear_feet_contact",
            "both_rear_feet_contact",
            "rear_foot_slip",
            "rear_foot_anchor",
            "front_feet_air",
            "balanced_footstand",
            "rear_leg_symmetry",
            "rear_leg_splay",
            "front_leg_motion",
            "front_feet_crossing",
            "front_leg_crossing",
            "upright_stability",
            "knee_clearance",
            "stay_still",
            "energy",
            "dof_acc",
        }
    )

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        task_env = cast("_FootstandEnv", env)
        term = type(self).__name__
        allowed = {
            "state_term_name",
            "scales",
            "soft_joint_pos_limit_factor",
            "knee_height_target",
            "front_feet_min_separation",
            "front_feet_side_margin",
            "rear_hip_abduction_margin",
            "rear_foot_slip_deadband",
            "rear_foot_anchor_radius",
        }
        unknown = sorted(set(cfg.params) - allowed)
        if unknown:
            raise TypeError(f"{term} received unsupported parameters: {unknown}")
        self._state_term_name = _name(term, "state_term_name", cfg.params.get("state_term_name"))
        scales = cfg.params.get("scales")
        if not isinstance(scales, dict) or not scales:
            raise TypeError(f"{term} scales must be a non-empty mapping")
        unknown_rewards = sorted(set(scales) - self._REWARD_NAMES)
        if unknown_rewards:
            raise ValueError(f"{term} scales contains unknown rewards: {unknown_rewards}")
        self._scales = {
            name: _real(term, f"scales.{name}", value) for name, value in scales.items()
        }
        self._soft_limit_factor = _real(
            term,
            "soft_joint_pos_limit_factor",
            cfg.params.get("soft_joint_pos_limit_factor"),
            minimum=0.0,
        )
        self._knee_height_target = _real(
            term, "knee_height_target", cfg.params.get("knee_height_target"), minimum=0.0
        )
        self._front_min_separation = _real(
            term,
            "front_feet_min_separation",
            cfg.params.get("front_feet_min_separation"),
            minimum=0.0,
        )
        self._front_side_margin = _real(
            term,
            "front_feet_side_margin",
            cfg.params.get("front_feet_side_margin"),
            minimum=0.0,
        )
        self._rear_hip_margin = _real(
            term,
            "rear_hip_abduction_margin",
            cfg.params.get("rear_hip_abduction_margin"),
            minimum=0.0,
        )
        self._rear_slip_deadband = _real(
            term,
            "rear_foot_slip_deadband",
            cfg.params.get("rear_foot_slip_deadband"),
            minimum=0.0,
        )
        self._rear_anchor_radius = _real(
            term,
            "rear_foot_anchor_radius",
            cfg.params.get("rear_foot_anchor_radius"),
            minimum=0.0,
            strict_minimum=True,
        )
        state_term = _termination(task_env, self._state_term_name)
        action = state_term.state.action
        centers = (action.joint_lower + action.joint_upper) / 2.0
        widths = action.joint_upper - action.joint_lower
        self._soft_lower = centers - 0.5 * widths * self._soft_limit_factor
        self._soft_upper = centers + 0.5 * widths * self._soft_limit_factor

    @staticmethod
    def _standing(state: FootstandState) -> np.ndarray:
        return (
            (state.height >= _TARGET_HEIGHT * _STAND_HEIGHT_FRACTION)
            & (state.orientation >= _STAND_ORIENTATION_THRESHOLD)
        ).astype(get_global_dtype(), copy=False)

    def _value(
        self,
        name: str,
        state_term: FootstandTermination,
        state: FootstandState,
    ) -> np.ndarray:
        dtype = get_global_dtype()
        standing = self._standing(state)
        default = state.default_joint_pos
        if name == "height":
            return np.asarray(np.exp(-np.abs(_TARGET_HEIGHT - state.height) / 0.1), dtype=dtype)
        if name == "contact":
            return np.any(state.foot_contact[:, _FRONT_FEET], axis=1).astype(dtype)
        if name in ("orientation", "oritentation"):
            return state.orientation
        if name == "action_rate":
            action = state.action
            return np.sum(np.square(action.raw_action - action.previous_raw_action), axis=1)
        if name == "termination":
            return state_term.terminated.astype(dtype)
        if name == "dof_pos_limits":
            below = np.clip(self._soft_lower - state.joint_pos, 0.0, None)
            above = np.clip(state.joint_pos - self._soft_upper, 0.0, None)
            return np.sum(below + above, axis=1)
        if name == "torques":
            return np.sum(np.square(state.torques), axis=1)
        if name == "pose":
            return np.sum(
                np.square(state.joint_pos[:, _REAR_LEGS] - default[:, _REAR_LEGS]), axis=1
            )
        if name == "penalty_contact":
            return state.penalty_contact.astype(dtype)
        if name == "tar":
            error = np.sum(np.square(state.joint_pos[:, _FRONT_LEGS] - _FRONT_LEG_TARGET), axis=1)
            height_mask = (state.height >= _TARGET_HEIGHT * _STAND_HEIGHT_FRACTION).astype(dtype)
            return np.asarray(np.exp(-error) * height_mask, dtype=dtype)
        if name == "rear_feet_contact":
            return np.mean(state.foot_contact[:, _REAR_FEET], axis=1, dtype=dtype)
        if name == "both_rear_feet_contact":
            return np.all(state.foot_contact[:, _REAR_FEET], axis=1).astype(dtype)
        if name == "rear_foot_slip":
            slip = np.square(np.clip(state.rear_speed - self._rear_slip_deadband, 0.0, None))
            slip *= state.foot_contact[:, _REAR_FEET]
            return np.mean(slip, axis=1, dtype=dtype)
        if name == "rear_foot_anchor":
            drift = np.square(
                np.clip(state.rear_anchor_drift - self._rear_anchor_radius, 0.0, None)
                / self._rear_anchor_radius
            )
            drift *= state.rear_anchor_contact
            return np.mean(drift, axis=1, dtype=dtype)
        if name == "front_feet_air":
            return (~np.any(state.foot_contact[:, _FRONT_FEET], axis=1)).astype(dtype)
        if name == "balanced_footstand":
            support = np.all(state.foot_contact[:, _REAR_FEET], axis=1)
            support &= ~np.any(state.foot_contact[:, _FRONT_FEET], axis=1)
            return support.astype(dtype) * standing
        if name == "rear_leg_symmetry":
            mirrored = state.joint_pos[:, _REAR_RIGHT] * _REAR_MIRROR_SIGNS
            cost = np.mean(np.square(state.joint_pos[:, _REAR_LEFT] - mirrored), axis=1)
            return cost * (1.0 - standing)
        if name == "rear_leg_splay":
            error = state.joint_pos[:, _REAR_HIPS] - default[:, _REAR_HIPS]
            splay = np.clip(np.abs(error) - self._rear_hip_margin, 0.0, None)
            return np.mean(np.square(splay), axis=1) * standing
        if name == "front_leg_motion":
            return np.mean(np.square(state.joint_vel[:, _FRONT_LEGS]), axis=1) * standing
        if name in ("front_feet_crossing", "front_leg_crossing"):
            return self._front_crossing(state)
        if name == "upright_stability":
            cost = np.sum(np.square(state.root_linvel_w), axis=1)
            cost += 0.25 * np.sum(np.square(state.root_angvel_w), axis=1)
            return cost * standing
        if name == "knee_clearance":
            target = max(self._knee_height_target, 1.0e-6)
            height = state.tracked_body_pos[:, _KNEE_BODY_INDICES, 2]
            return np.mean(np.square(np.clip(target - height, 0.0, None) / target), axis=1)
        if name == "stay_still":
            return np.sum(np.square(state.root_linvel_w[:, :2]), axis=1) + np.square(
                state.root_angvel_w[:, 2]
            )
        if name == "energy":
            return np.sum(np.abs(state.joint_vel) * np.abs(state.torques), axis=1)
        if name == "dof_acc":
            return np.sum(np.square(state.qacc), axis=1)
        raise RuntimeError(f"Footstand reward dispatch is incomplete for {name!r}")

    def _front_crossing(self, state: FootstandState) -> np.ndarray:
        left = np.concatenate(
            (
                state.foot_pos[:, [0], :],
                state.tracked_body_pos[:, _FRONT_LEFT_BODY_INDICES, :],
            ),
            axis=1,
        )
        right = np.concatenate(
            (
                state.foot_pos[:, [1], :],
                state.tracked_body_pos[:, _FRONT_RIGHT_BODY_INDICES, :],
            ),
            axis=1,
        )
        points = np.concatenate((left, right), axis=1)
        relative = (points - state.root_pos[:, None, :]).reshape(-1, 3)
        quaternions = np.repeat(state.root_quat, points.shape[1], axis=0)
        body_points = np_quat_apply_inverse(quaternions, relative).reshape(
            state.root_pos.shape[0], points.shape[1], 3
        )
        left_y = body_points[:, : left.shape[1], 1]
        right_y = body_points[:, left.shape[1] :, 1]
        left_error = np.clip(self._front_side_margin - left_y, 0.0, None)
        right_error = np.clip(right_y + self._front_side_margin, 0.0, None)
        separation_error = np.clip(self._front_min_separation - (left_y - right_y), 0.0, None)
        return np.mean(
            np.square(left_error) + np.square(right_error) + np.square(separation_error), axis=1
        )

    def __call__(self, env: _FootstandEnv, **params: Any) -> np.ndarray:
        del params
        state_term = _termination(env, self._state_term_name)
        state = state_term.state
        reward = np.zeros((env.num_envs,), dtype=get_global_dtype())
        for name, scale in self._scales.items():
            if scale != 0.0:
                reward += scale * self._value(name, state_term, state)
        max_rate = 10000.0 / env.step_dt
        return np.clip(reward, 0.0, max_rate)


class FootstandJointReset(ManagerTermBase):
    """Reset all Go2 joints to the home pose plus a uniform offset."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        term = type(self).__name__
        if set(cfg.params) != {"asset_cfg", "position_offset_range"}:
            raise ValueError(
                f"{term} requires exactly asset_cfg and position_offset_range parameters"
            )
        asset_cfg = cfg.params["asset_cfg"]
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(f"{term} asset_cfg must be SceneEntityCfg")
        self._entity = cast("Entity", env.scene[asset_cfg.name])
        self._joint_ids = asset_cfg.joint_ids
        selected = self._entity.data.default_joint_pos[:, self._joint_ids]
        if selected.shape != (env.num_envs, NUM_ACTIONS):
            raise ValueError(f"{term} requires exactly {NUM_ACTIONS} selected joints")
        self._range = _pair(term, "position_offset_range", cfg.params["position_offset_range"])

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | None,
        **params: Any,
    ) -> None:
        del params
        ids = _env_ids(env, env_ids)
        position = np.array(self._entity.data.default_joint_pos[ids][:, self._joint_ids], copy=True)
        position += env.rng.uniform(*self._range, size=position.shape)
        velocity = np.array(self._entity.data.default_joint_vel[ids][:, self._joint_ids], copy=True)
        self._entity.write_joint_state_to_sim(
            np.asarray(position, dtype=get_global_dtype()),
            np.asarray(velocity, dtype=get_global_dtype()),
            joint_ids=self._joint_ids,
            env_ids=ids,
        )


class FootstandMassRandomization(ManagerTermBase):
    """Compose all-link mass scaling and torso additive mass in one reset write."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        term = type(self).__name__
        allowed = {
            "asset_cfg",
            "torso_body_name",
            "link_mass_scale_range",
            "torso_added_mass_range",
        }
        if set(cfg.params) != allowed:
            raise ValueError(f"{term} requires parameters {sorted(allowed)}")
        asset_cfg = cfg.params["asset_cfg"]
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(f"{term} asset_cfg must be SceneEntityCfg")
        self._entity = cast("Entity", env.scene[asset_cfg.name])
        self._body_ids, self._default_mass = self._entity.bind_body_mass_write(
            asset_cfg.body_ids, term_name="footstand_mass"
        )
        torso_name = _name(term, "torso_body_name", cfg.params["torso_body_name"])
        torso_ids, _ = self._entity.find_bodies((torso_name,))
        if len(torso_ids) != 1:
            raise ValueError(f"{term} torso_body_name must resolve exactly one body")
        selected = np.flatnonzero(self._body_ids == torso_ids[0])
        if selected.size != 1:
            raise ValueError(f"{term} torso body must be included in asset_cfg")
        self._torso_index = int(selected[0])
        self._scale_range = _pair(
            term, "link_mass_scale_range", cfg.params["link_mass_scale_range"], minimum=0.0
        )
        self._added_range = _pair(
            term, "torso_added_mass_range", cfg.params["torso_added_mass_range"]
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | None,
        **params: Any,
    ) -> None:
        del params
        ids = _env_ids(env, env_ids)
        scale = env.rng.uniform(*self._scale_range, size=(ids.size, self._default_mass.size))
        mass = self._default_mass[None, :] * scale
        mass[:, self._torso_index] += env.rng.uniform(*self._added_range, size=ids.size)
        if np.any(mass <= 0.0):
            raise ValueError("FootstandMassRandomization produced a non-positive body mass")
        self._entity.write_body_mass_to_sim(
            mass,
            body_ids=self._body_ids,
            env_ids=ids,
            term_name="footstand_mass",
        )


registry.register_env_config("Go2FootStand", ManagerBasedRlEnvCfg)
registry.register_env("Go2FootStand", make_manager_based_rl_env, sim_backend="mujoco")
registry.register_env("Go2FootStand", make_manager_based_rl_env, sim_backend="motrix")
registry.register_env("Go2FootStand", make_manager_based_rl_env, sim_backend="drake")


__all__ = [
    "FRAME_OBS_DIM",
    "NUM_ACTIONS",
    "PRIVILEGED_OBS_DIM",
    "FootstandIncrementalAction",
    "FootstandIncrementalActionCfg",
    "FootstandJointReset",
    "FootstandMassRandomization",
    "FootstandReward",
    "FootstandState",
    "FootstandTermination",
    "frame_observation",
    "privileged_observation",
]
