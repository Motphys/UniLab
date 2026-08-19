"""Manager-Based terms owned by the Go2W flat task."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.envs.mdp.commands.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from unilab.managers import ActionTerm, ActionTermCfg
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.tasks.locomotion.go2w.base import (
    NUM_GO2W_ACTIONS,
    NUM_LEG_ACTIONS,
    NUM_WHEEL_ACTIONS,
    compute_go2w_motor_ctrl,
)

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


_HIP_INDICES = np.asarray([0, 3, 6, 9], dtype=np.intp)
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _real(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and (result <= minimum if strict_minimum else result < minimum):
        relation = "greater than" if strict_minimum else "at least"
        raise ValueError(f"{label} must be {relation} {minimum}")
    return result


def _range(value: Any, *, label: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{label} must be a two-value range")
    lower = _real(value[0], label=f"{label} lower", minimum=0.0)
    upper = _real(value[1], label=f"{label} upper", minimum=0.0)
    if lower > upper:
        raise ValueError(f"{label} lower {lower} exceeds upper {upper}")
    return lower, upper


@dataclass(kw_only=True)
class Go2WMixedActionCfg(ActionTermCfg):
    """Configure the Go2W leg-position and wheel-velocity motor action."""

    actuator_names: tuple[str, ...] | list[str]
    leg_action_scale: float = 0.25
    hip_action_scale: float | None = None
    wheel_action_scale: float = 10.0
    leg_kp: float = 35.0
    leg_kd: float = 0.5
    wheel_kd: float = 0.5
    clip_actions: float = 1.0
    simulate_action_latency: bool = False

    def build(self, env: ManagerBasedRlEnv) -> Go2WMixedAction:
        return Go2WMixedAction(self, env)


class Go2WMixedAction(ActionTerm):
    """Convert one community action term into Go2W motor torques per substep."""

    cfg: Go2WMixedActionCfg
    _entity: Entity

    def __init__(self, cfg: Go2WMixedActionCfg, env: ManagerBasedRlEnv):
        self._validate_cfg(cfg)
        super().__init__(cfg=cfg, env=env)
        actuator_ids, actuator_names = self._entity.find_actuators(cfg.actuator_names)
        joint_ids, joint_names = self._entity.find_joints_by_actuator_names(cfg.actuator_names)
        if len(actuator_ids) != NUM_GO2W_ACTIONS or len(joint_ids) != NUM_GO2W_ACTIONS:
            raise ValueError(
                "Go2WMixedAction requires exactly "
                f"{NUM_GO2W_ACTIONS} actuators and target joints; received "
                f"actuators={actuator_names}, joints={joint_names}"
            )
        self._actuator_ids = np.asarray(actuator_ids, dtype=np.intp)
        self._joint_ids = np.asarray(joint_ids, dtype=np.intp)
        self._actuator_ids.setflags(write=False)
        self._joint_ids.setflags(write=False)

        dtype = get_global_dtype()
        shape = (self.num_envs, NUM_GO2W_ACTIONS)
        self._raw_action = np.zeros(shape, dtype=dtype)
        self._previous_raw_action = np.zeros_like(self._raw_action)
        self._processed_action = np.zeros_like(self._raw_action)
        self._motor_torque = np.zeros_like(self._raw_action)

        self._leg_action_scale = np.full(
            (NUM_LEG_ACTIONS,), float(cfg.leg_action_scale), dtype=dtype
        )
        if cfg.hip_action_scale is not None:
            self._leg_action_scale[_HIP_INDICES] = float(cfg.hip_action_scale)
        self._base_leg_kp = np.full((NUM_LEG_ACTIONS,), float(cfg.leg_kp), dtype=dtype)
        self._base_leg_kd = np.full((NUM_LEG_ACTIONS,), float(cfg.leg_kd), dtype=dtype)
        self._leg_kp = np.broadcast_to(self._base_leg_kp, (self.num_envs, NUM_LEG_ACTIONS)).copy()
        self._leg_kd = np.broadcast_to(self._base_leg_kd, (self.num_envs, NUM_LEG_ACTIONS)).copy()
        self._wheel_kd = np.full(
            (self.num_envs, NUM_WHEEL_ACTIONS), float(cfg.wheel_kd), dtype=dtype
        )

        ctrl_range = self._entity.data.actuator_ctrl_range[self._actuator_ids]
        expected_range_shape = (NUM_GO2W_ACTIONS, 2)
        if ctrl_range.shape != expected_range_shape:
            raise ValueError(
                "Go2WMixedAction actuator control range must have shape "
                f"{expected_range_shape}, got {ctrl_range.shape}"
            )
        self._ctrl_lower = np.asarray(ctrl_range[:, 0], dtype=dtype)
        self._ctrl_upper = np.asarray(ctrl_range[:, 1], dtype=dtype)

    @staticmethod
    def _validate_cfg(cfg: Go2WMixedActionCfg) -> None:
        if not isinstance(cfg.simulate_action_latency, bool):
            raise TypeError("Go2WMixedActionCfg simulate_action_latency must be bool")
        for name, value in (
            ("leg_action_scale", cfg.leg_action_scale),
            ("wheel_action_scale", cfg.wheel_action_scale),
            ("leg_kp", cfg.leg_kp),
            ("leg_kd", cfg.leg_kd),
            ("wheel_kd", cfg.wheel_kd),
            ("clip_actions", cfg.clip_actions),
        ):
            _real(
                value,
                label=f"Go2WMixedActionCfg {name}",
                minimum=0.0,
                strict_minimum=name == "clip_actions",
            )
        if cfg.hip_action_scale is not None:
            _real(
                cfg.hip_action_scale,
                label="Go2WMixedActionCfg hip_action_scale",
                minimum=0.0,
            )

    @property
    def action_dim(self) -> int:
        return NUM_GO2W_ACTIONS

    @property
    def raw_action(self) -> np.ndarray:
        """Clipped policy action, matching the legacy observable action buffer."""
        return self._raw_action

    @property
    def previous_raw_action(self) -> np.ndarray:
        return self._previous_raw_action

    @property
    def processed_action(self) -> np.ndarray:
        return self._processed_action

    @property
    def motor_torque(self) -> np.ndarray:
        return self._motor_torque

    @property
    def leg_kp(self) -> np.ndarray:
        return self._leg_kp

    @property
    def leg_kd(self) -> np.ndarray:
        return self._leg_kd

    def process_actions(self, actions: np.ndarray) -> None:
        if not isinstance(actions, np.ndarray):
            raise TypeError(f"Go2WMixedAction expected np.ndarray, got {type(actions).__name__}")
        if actions.shape != self._raw_action.shape:
            raise ValueError(
                f"Go2WMixedAction expected action shape {self._raw_action.shape}, "
                f"got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("Go2WMixedAction received NaN or Inf actions")

        self._previous_raw_action[:] = self._raw_action
        np.clip(
            actions,
            -float(self.cfg.clip_actions),
            float(self.cfg.clip_actions),
            out=self._raw_action,
        )
        executed = (
            self._previous_raw_action if self.cfg.simulate_action_latency else self._raw_action
        )
        np.multiply(
            executed[:, :NUM_LEG_ACTIONS],
            self._leg_action_scale,
            out=self._processed_action[:, :NUM_LEG_ACTIONS],
        )
        self._processed_action[:, :NUM_LEG_ACTIONS] += self._entity.data.default_joint_pos[
            :, self._joint_ids[:NUM_LEG_ACTIONS]
        ]
        np.multiply(
            executed[:, NUM_LEG_ACTIONS:],
            float(self.cfg.wheel_action_scale),
            out=self._processed_action[:, NUM_LEG_ACTIONS:],
        )

    def apply_actions(self) -> None:
        joint_pos = self._entity.data.joint_pos[:, self._joint_ids]
        joint_vel = self._entity.data.joint_vel[:, self._joint_ids]
        compute_go2w_motor_ctrl(
            self._processed_action,
            joint_pos,
            joint_vel,
            self._leg_kp,
            self._leg_kd,
            self._wheel_kd,
            self._ctrl_lower,
            self._ctrl_upper,
            self._motor_torque,
        )
        self._entity.data.write_ctrl(self._motor_torque, actuator_ids=self._actuator_ids)

    def set_motor_gain_multipliers(
        self,
        env_ids: np.ndarray,
        kp_multiplier: np.ndarray,
        kd_multiplier: np.ndarray,
    ) -> None:
        expected = (len(env_ids), 1)
        if kp_multiplier.shape != expected or kd_multiplier.shape != expected:
            raise ValueError(
                "Go2WMixedAction motor gain multipliers must have shape "
                f"{expected}, got kp={kp_multiplier.shape}, kd={kd_multiplier.shape}"
            )
        self._leg_kp[env_ids] = self._base_leg_kp * kp_multiplier
        self._leg_kd[env_ids] = self._base_leg_kd * kd_multiplier

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_action[env_ids] = 0.0
        self._previous_raw_action[env_ids] = 0.0
        self._processed_action[env_ids] = 0.0
        self._motor_torque[env_ids] = 0.0


@dataclass(kw_only=True)
class Go2WVelocityCommandCfg(UniformVelocityCommandCfg):
    """Velocity command with the legacy Go2W planar dead zone."""

    planar_dead_zone: float = 0.2

    def build(self, env: ManagerBasedRlEnv) -> Go2WVelocityCommand:
        return Go2WVelocityCommand(self, env)


class Go2WVelocityCommand(UniformVelocityCommand):
    cfg: Go2WVelocityCommandCfg  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(self, cfg: Go2WVelocityCommandCfg, env: ManagerBasedRlEnv):
        self._planar_dead_zone = _real(
            cfg.planar_dead_zone,
            label="Go2WVelocityCommandCfg planar_dead_zone",
            minimum=0.0,
        )
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: np.ndarray) -> None:
        super()._resample_command(env_ids)
        planar = self.vel_command_b[env_ids, :2]
        moving = np.linalg.norm(planar, axis=1) > self._planar_dead_zone
        self.vel_command_b[env_ids, :2] = planar * moving[:, None]


def _action(env: ManagerBasedRlEnv, action_name: str) -> Go2WMixedAction:
    if not isinstance(action_name, str) or not action_name:
        raise ValueError("Go2W manager term action_name must be a non-empty string")
    try:
        term = env.action_manager.get_term(action_name)
    except KeyError as exc:
        raise KeyError(f"Go2W action term '{action_name}' is unavailable") from exc
    if not isinstance(term, Go2WMixedAction):
        raise TypeError(
            f"Go2W action term '{action_name}' must be Go2WMixedAction, got {type(term).__name__}"
        )
    return term


def randomize_motor_gains(
    env: ManagerBasedRlEnv,
    env_ids: np.ndarray | None,
    action_name: str,
    kp_multiplier_range: tuple[float, float] | list[float],
    kd_multiplier_range: tuple[float, float] | list[float],
) -> None:
    """Sample owner-level motor gains without mutating backend actuator models."""
    ids = (
        np.arange(env.num_envs, dtype=np.int32)
        if env_ids is None
        else np.asarray(env_ids, dtype=np.int32)
    )
    kp_range = _range(kp_multiplier_range, label="randomize_motor_gains kp_multiplier_range")
    kd_range = _range(kd_multiplier_range, label="randomize_motor_gains kd_multiplier_range")
    shape = (len(ids), 1)
    kp = env.rng.uniform(*kp_range, size=shape).astype(get_global_dtype(), copy=False)
    kd = env.rng.uniform(*kd_range, size=shape).astype(get_global_dtype(), copy=False)
    _action(env, action_name).set_motor_gain_multipliers(ids, kp, kd)


def motor_torque(env: ManagerBasedRlEnv, action_name: str) -> np.ndarray:
    return _action(env, action_name).motor_torque


def motor_torque_l2(env: ManagerBasedRlEnv, action_name: str) -> np.ndarray:
    return np.sum(np.square(motor_torque(env, action_name)), axis=1)


def clipped_action_rate_l2(env: ManagerBasedRlEnv, action_name: str) -> np.ndarray:
    action = _action(env, action_name)
    return np.sum(np.square(action.raw_action - action.previous_raw_action), axis=1)


def upward_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    asset = cast("Entity", env.scene[asset_cfg.name])
    return np.square(1.0 - asset.data.projected_gravity_b[:, 2])


def constant_alive(env: ManagerBasedRlEnv) -> np.ndarray:
    return np.ones((env.num_envs,), dtype=get_global_dtype())


__all__ = [
    "Go2WMixedAction",
    "Go2WMixedActionCfg",
    "Go2WVelocityCommand",
    "Go2WVelocityCommandCfg",
    "clipped_action_rate_l2",
    "constant_alive",
    "motor_torque",
    "motor_torque_l2",
    "randomize_motor_gains",
    "upward_l2",
]
