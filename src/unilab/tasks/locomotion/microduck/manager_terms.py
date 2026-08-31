"""MicroDuck-specific Manager-Based command, event, and reward terms."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from unilab.managers import (
    CommandTerm,
    CommandTermCfg,
    ManagerTermBase,
    ManagerTermBaseCfg,
    SceneEntityCfg,
)
from unilab.tasks.locomotion.common.sensor_terms import SensorTermBase

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_LINEAR_VELOCITY_SENSOR = ("local_linvel",)
_GYRO_SENSOR = ("gyro",)
_UPVECTOR_SENSOR = ("upvector",)
_FOOT_CONTACT_SENSORS = ("left_foot_contact", "right_foot_contact")


def _finite_real(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and (result < minimum or (strict_minimum and result == minimum)):
        relation = "greater than" if strict_minimum else "at least"
        raise ValueError(f"{label} must be {relation} {minimum}")
    return result


def _ratio(value: Any, *, label: str) -> float:
    result = _finite_real(value, label=label)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{label} must be within [0, 1]")
    return result


def _state(term: str, capability: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=get_global_dtype())
    if array.shape != shape:
        raise ValueError(f"{term} {capability} must have shape {shape}, received {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{term} {capability} contains NaN or Inf")
    return array


def _command(env: ManagerBasedRlEnv, term: str, command_name: str) -> np.ndarray:
    if not isinstance(command_name, str) or not command_name:
        raise ValueError(f"{term} command_name must be a non-empty string")
    try:
        command = env.command_manager.get_command(command_name)
    except KeyError as exc:
        raise KeyError(f"{term} command '{command_name}' is not configured") from exc
    array = np.asarray(command, dtype=get_global_dtype())
    if array.ndim != 2 or array.shape[0] != env.num_envs:
        raise ValueError(
            f"{term} command '{command_name}' must have shape ({env.num_envs}, width), "
            f"received {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{term} command '{command_name}' contains NaN or Inf")
    return array


def _asset_selection(
    cfg: ManagerTermBaseCfg,
    env: ManagerBasedRlEnv,
    *,
    term: str,
) -> tuple[Entity, np.ndarray]:
    asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    if not isinstance(asset_cfg, SceneEntityCfg):
        raise TypeError(f"{term} asset_cfg must be SceneEntityCfg")
    entity = cast("Entity", env.scene[asset_cfg.name])
    ids = np.asarray(np.arange(entity.num_joints, dtype=np.intp)[asset_cfg.joint_ids])
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError(f"{term} asset_cfg must select at least one joint")
    ids.setflags(write=False)
    return entity, ids


@dataclass(kw_only=True)
class UniformVectorCommandCfg(CommandTermCfg):
    """Uniformly sample a fixed-width command vector from per-axis ranges."""

    ranges: tuple[tuple[float, float], ...] | list[list[float]]

    def build(self, env: ManagerBasedRlEnv) -> UniformVectorCommand:
        return UniformVectorCommand(self, env)


class UniformVectorCommand(CommandTerm):
    """Task-local vector command with no runtime scene dependency."""

    cfg: UniformVectorCommandCfg

    def __init__(self, cfg: UniformVectorCommandCfg, env: ManagerBasedRlEnv):
        ranges = np.asarray(cfg.ranges, dtype=np.float64)
        if ranges.ndim != 2 or ranges.shape[0] == 0 or ranges.shape[1] != 2:
            raise ValueError("UniformVectorCommandCfg ranges must have shape (N, 2)")
        if not np.isfinite(ranges).all() or np.any(ranges[:, 0] > ranges[:, 1]):
            raise ValueError("UniformVectorCommandCfg ranges must be finite ordered pairs")
        self._ranges = ranges
        super().__init__(cfg, env)
        self._command = np.zeros((self.num_envs, ranges.shape[0]), dtype=get_global_dtype())

    @property
    def command(self) -> np.ndarray:
        return self._command

    def _update_metrics(self, env_ids: np.ndarray | None = None) -> None:
        del env_ids

    def _resample_command(self, env_ids: np.ndarray) -> None:
        self._command[env_ids] = self._env.rng.uniform(
            self._ranges[:, 0],
            self._ranges[:, 1],
            size=(len(env_ids), self._ranges.shape[0]),
        )

    def _update_command(self, env_ids: np.ndarray | None) -> None:
        del env_ids


@dataclass(kw_only=True)
class MicroduckVelocityCommandCfg(UniformVelocityCommandCfg):
    """Velocity command with MicroDuck's turn-in-place sampling branch."""

    turn_in_place_fraction: float = 0.0
    turn_in_place_ang_min: float = 0.4

    def build(self, env: ManagerBasedRlEnv) -> MicroduckVelocityCommand:
        return MicroduckVelocityCommand(self, env)


class MicroduckVelocityCommand(UniformVelocityCommand):
    def __init__(self, cfg: MicroduckVelocityCommandCfg, env: ManagerBasedRlEnv):
        self._turn_fraction = _ratio(
            cfg.turn_in_place_fraction,
            label="MicroduckVelocityCommand turn_in_place_fraction",
        )
        self._turn_ang_min = _ratio(
            cfg.turn_in_place_ang_min,
            label="MicroduckVelocityCommand turn_in_place_ang_min",
        )
        self._turn_ang_max = max(abs(value) for value in cfg.ranges.ang_vel_z)
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: np.ndarray) -> None:
        super()._resample_command(env_ids)
        if self._turn_fraction == 0.0 or len(env_ids) == 0:
            return
        selected = self._env.rng.uniform(0.0, 1.0, len(env_ids)) < self._turn_fraction
        turn_ids = env_ids[selected]
        if len(turn_ids) == 0:
            return
        self.vel_command_b[turn_ids, :2] = 0.0
        min_abs = self._turn_ang_min * self._turn_ang_max
        signs = np.where(self._env.rng.uniform(0.0, 1.0, len(turn_ids)) < 0.5, -1.0, 1.0)
        self.vel_command_b[turn_ids, 2] = signs * self._env.rng.uniform(
            min_abs,
            self._turn_ang_max,
            len(turn_ids),
        )


class _LinearVelocityTerm(SensorTermBase):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._view = self._bind(_LINEAR_VELOCITY_SENSOR)
        if self._view.dimensions != (3,):
            raise ValueError(f"{self.name} local_linvel sensor must expose 3 values")

    def _value(self, env: ManagerBasedRlEnv) -> np.ndarray:
        return _state(
            self.name,
            "local linear velocity",
            self._read(self._view, self.name),
            (env.num_envs, 3),
        )


class _GyroTerm(SensorTermBase):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._view = self._bind(_GYRO_SENSOR)
        if self._view.dimensions != (3,):
            raise ValueError(f"{self.name} gyro sensor must expose 3 values")

    def _value(self, env: ManagerBasedRlEnv) -> np.ndarray:
        return _state(
            self.name,
            "angular velocity",
            self._read(self._view, self.name),
            (env.num_envs, 3),
        )


class _UpvectorTerm(SensorTermBase):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._view = self._bind(_UPVECTOR_SENSOR)
        if self._view.dimensions != (3,):
            raise ValueError(f"{self.name} upvector sensor must expose 3 values")

    def _value(self, env: ManagerBasedRlEnv) -> np.ndarray:
        return _state(
            self.name,
            "upvector",
            self._read(self._view, self.name),
            (env.num_envs, 3),
        )


class track_lin_vel(_LinearVelocityTerm):
    _allowed_params = frozenset({"tracking_sigma", "command_name"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._sigma = _finite_real(
            cfg.params.get("tracking_sigma", 0.25),
            label=f"{self.name} tracking_sigma",
            minimum=0.0,
            strict_minimum=True,
        )
        self._command_name = cfg.params.get("command_name", "twist")

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        error = np.sum(
            np.square(
                _command(env, self.name, self._command_name)[:, :2] - self._value(env)[:, :2]
            ),
            axis=1,
        )
        return np.asarray(np.exp(-error / self._sigma), dtype=get_global_dtype())


class track_ang_vel(_GyroTerm):
    _allowed_params = frozenset({"tracking_sigma", "command_name"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._sigma = _finite_real(
            cfg.params.get("tracking_sigma", 0.25),
            label=f"{self.name} tracking_sigma",
            minimum=0.0,
            strict_minimum=True,
        )
        self._command_name = cfg.params.get("command_name", "twist")

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        error = np.square(
            _command(env, self.name, self._command_name)[:, 2] - self._value(env)[:, 2]
        )
        return np.asarray(np.exp(-error / self._sigma), dtype=get_global_dtype())


class lin_vel_z(_LinearVelocityTerm):
    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        return np.asarray(np.square(self._value(env)[:, 2]), dtype=get_global_dtype())


class ang_vel_xy(_GyroTerm):
    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        return np.asarray(
            np.sum(np.square(self._value(env)[:, :2]), axis=1),
            dtype=get_global_dtype(),
        )


class orientation(_UpvectorTerm):
    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        return np.asarray(
            np.sum(np.square(self._value(env)[:, :2]), axis=1),
            dtype=get_global_dtype(),
        )


def base_height(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    target = _finite_real(target_height, label="base_height target_height")
    entity = cast("Entity", env.scene[asset_cfg.name])
    height = _state(
        "base_height",
        "root position",
        entity.data.root_link_pos_w,
        (env.num_envs, 3),
    )[:, 2]
    return np.asarray(np.square(height - target), dtype=get_global_dtype())


def alive(env: ManagerBasedRlEnv) -> np.ndarray:
    return np.ones((env.num_envs,), dtype=get_global_dtype())


class _JointCommandTerm(ManagerTermBase):
    _allowed_params = frozenset({"asset_cfg", "command_name"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        unexpected = set(cfg.params) - self._allowed_params
        if unexpected:
            raise TypeError(f"{self.name} received unsupported parameters: {sorted(unexpected)}")
        self._entity, self._joint_ids = _asset_selection(cfg, env, term=self.name)
        command_name = cfg.params.get("command_name")
        if not isinstance(command_name, str) or not command_name:
            raise ValueError(f"{self.name} command_name must be a non-empty string")
        self._command_name = command_name

    def _joint_error(self, env: ManagerBasedRlEnv) -> np.ndarray:
        actual = self._entity.data.joint_pos[:, self._joint_ids]
        default = self._entity.data.default_joint_pos[:, self._joint_ids]
        command = _command(env, self.name, self._command_name)
        if command.shape[1] != self._joint_ids.size:
            raise ValueError(
                f"{self.name} command width {command.shape[1]} does not match "
                f"selected joint count {self._joint_ids.size}"
            )
        return actual - default - command


class head_pose_tracking(_JointCommandTerm):
    _allowed_params = frozenset({"asset_cfg", "command_name", "std"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._std = _finite_real(
            cfg.params.get("std", 0.5),
            label=f"{self.name} std",
            minimum=0.0,
            strict_minimum=True,
        )

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        per_joint = np.exp(-np.square(self._joint_error(env) / self._std))
        return np.asarray(np.mean(per_joint, axis=1), dtype=get_global_dtype())


class head_pose_bias(_JointCommandTerm):
    _allowed_params = frozenset({"asset_cfg", "command_name", "tau_s"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        tau = _finite_real(
            cfg.params.get("tau_s", 1.0),
            label=f"{self.name} tau_s",
            minimum=0.0,
            strict_minimum=True,
        )
        self._alpha = min(1.0, env.step_dt / tau)
        self._ema = np.zeros((env.num_envs, self._joint_ids.size), dtype=get_global_dtype())

    def reset(self, env_ids: np.ndarray | slice | None) -> None:
        self._ema[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        fresh = env.episode_length_buf <= 1
        self._ema[fresh] = 0.0
        self._ema *= 1.0 - self._alpha
        self._ema += self._alpha * self._joint_error(env)
        return np.asarray(-np.mean(np.abs(self._ema), axis=1), dtype=get_global_dtype())


class leg_pose(_JointCommandTerm):
    _allowed_params = frozenset({"asset_cfg", "command_name", "std_standing"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._std = _finite_real(
            cfg.params.get("std_standing", 0.1),
            label=f"{self.name} std_standing",
            minimum=0.0,
            strict_minimum=True,
        )

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        command = _command(env, self.name, self._command_name)
        moving = np.linalg.norm(command[:, :2], axis=1) + np.abs(command[:, 2]) > 0.01
        actual = self._entity.data.joint_pos[:, self._joint_ids]
        default = self._entity.data.default_joint_pos[:, self._joint_ids]
        reward = np.mean(np.exp(-np.square((actual - default) / self._std)), axis=1)
        return np.asarray(np.where(moving, 0.0, reward), dtype=get_global_dtype())


class _FootContactTerm(SensorTermBase):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._view = self._bind(_FOOT_CONTACT_SENSORS)
        if len(self._view.dimensions) != 2 or any(
            dimension < 1 for dimension in self._view.dimensions
        ):
            raise ValueError(
                f"{self.name} foot contact sensors must each expose at least one value; "
                f"received {self._view.dimensions}"
            )
        self._contact_columns = np.asarray(
            (0, self._view.dimensions[0]),
            dtype=np.intp,
        )
        self._contact_columns.setflags(write=False)

    def _contact(self, env: ManagerBasedRlEnv) -> np.ndarray:
        values = _state(
            self.name,
            "foot contact",
            self._read(self._view, self.name),
            (env.num_envs, sum(self._view.dimensions)),
        )
        # MuJoCo ``data=force`` contact sensors expose a vector while
        # ``data=found`` sensors expose a scalar. The legacy MicroDuck contract
        # consumes column zero from each named sensor; cache those columns here.
        return values[:, self._contact_columns] > 0.1


class foot_air_time_biped(_FootContactTerm):
    _allowed_params = frozenset({"threshold", "command_threshold", "command_name"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._threshold = _finite_real(
            cfg.params.get("threshold", 0.3),
            label=f"{self.name} threshold",
            minimum=0.0,
        )
        self._command_threshold = _finite_real(
            cfg.params.get("command_threshold", 0.01),
            label=f"{self.name} command_threshold",
            minimum=0.0,
        )
        self._command_name = cfg.params.get("command_name", "twist")
        self._air_time = np.zeros((env.num_envs, 2), dtype=get_global_dtype())
        self._contact_time = np.zeros_like(self._air_time)

    def reset(self, env_ids: np.ndarray | slice | None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._air_time[ids] = 0.0
        self._contact_time[ids] = 0.0

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        contact = self._contact(env)
        self._air_time[contact] = 0.0
        self._air_time[~contact] += env.step_dt
        self._contact_time[~contact] = 0.0
        self._contact_time[contact] += env.step_dt
        in_mode_time = np.where(contact, self._contact_time, self._air_time)
        single_stance = np.sum(contact, axis=1) == 1
        masked = np.where(single_stance[:, None], in_mode_time, 0.0)
        reward = np.minimum(np.min(masked, axis=1), self._threshold)
        command = _command(env, self.name, self._command_name)
        moving = (
            np.linalg.norm(command[:, :2], axis=1) + np.abs(command[:, 2]) > self._command_threshold
        )
        return np.asarray(reward * moving, dtype=get_global_dtype())


class flight_phase(_FootContactTerm):
    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        return np.asarray(~np.any(self._contact(env), axis=1), dtype=get_global_dtype())


class randomize_encoder_bias(ManagerTermBase):
    """Sample per-reset joint encoder calibration bias through Entity data."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self._entity, self._joint_ids = _asset_selection(cfg, env, term=self.name)
        bounds = np.asarray(cfg.params.get("bias_range"), dtype=np.float64)
        if bounds.shape != (2,) or not np.isfinite(bounds).all():
            raise ValueError("randomize_encoder_bias bias_range must be a finite pair")
        if bounds[0] > bounds[1]:
            raise ValueError("randomize_encoder_bias bias_range is reversed")
        self._range = (float(bounds[0]), float(bounds[1]))

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | None,
        bias_range: tuple[float, float],
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> None:
        del bias_range, asset_cfg
        ids = np.arange(env.num_envs, dtype=np.intp) if env_ids is None else env_ids
        self._entity.data.encoder_bias[np.ix_(ids, self._joint_ids)] = env.rng.uniform(
            *self._range,
            size=(len(ids), len(self._joint_ids)),
        )


__all__ = [
    "MicroduckVelocityCommand",
    "MicroduckVelocityCommandCfg",
    "UniformVectorCommand",
    "UniformVectorCommandCfg",
    "alive",
    "ang_vel_xy",
    "base_height",
    "flight_phase",
    "foot_air_time_biped",
    "head_pose_bias",
    "head_pose_tracking",
    "leg_pose",
    "lin_vel_z",
    "orientation",
    "randomize_encoder_bias",
    "track_ang_vel",
    "track_lin_vel",
]
