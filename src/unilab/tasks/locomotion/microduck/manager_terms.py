"""MicroDuck-specific Manager-Based command and reward terms.

Generic named-sensor reward terms (velocity tracking, vertical/angular velocity
and orientation penalties) come from ``tasks.locomotion.common.sensor_reward_terms``;
``base_height_l2``/``alive`` come from ``tasks.locomotion.common.manager_terms``;
``randomize_encoder_bias`` comes from ``unilab.envs.mdp``.  Only terms with
MicroDuck-specific semantics live here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from unilab.managers import (
    ManagerTermBase,
    ManagerTermBaseCfg,
    SceneEntityCfg,
)
from unilab.tasks.locomotion.common.manager_terms import SensorTermBase
from unilab.utils.rotation import np_wrap_to_pi

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
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


class GroundPickPhaseCommand(UniformVelocityCommand):
    """Continuous cyclic task phase encoded as ``[cos, sin, 0]``."""

    def __init__(self, cfg: GroundPickPhaseCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._period = _finite_real(
            cfg.period,
            label="GroundPickPhaseCommand period",
            minimum=0.0,
            strict_minimum=True,
        )
        if not isinstance(cfg.randomize_phase, bool):
            raise TypeError("GroundPickPhaseCommand randomize_phase must be bool")
        self._randomize_phase = cfg.randomize_phase
        self._phase = np.zeros(self.num_envs, dtype=get_global_dtype())

    @property
    def phase(self) -> np.ndarray:
        return self._phase

    def reset(self, env_ids: np.ndarray | slice | None) -> dict[str, float]:
        ids = (
            np.arange(self.num_envs, dtype=np.int32)
            if env_ids is None
            else (
                np.arange(self.num_envs, dtype=np.int32)[env_ids]
                if isinstance(env_ids, slice)
                else np.asarray(env_ids, dtype=np.int32)
            )
        )
        self.time_left[ids] = 0.0
        self.command_counter[ids] = 0
        self.metrics.clear()
        if self._randomize_phase:
            self._phase[ids] = self._env.rng.uniform(0.0, 1.0, size=len(ids))
        else:
            self._phase[ids] = 0.0
        self._update_command(ids)
        return {}

    def compute(self, dt: float | np.ndarray, env_ids: np.ndarray | None = None) -> None:
        # This is a continuous signal, so the sampled-command timer is not part
        # of its lifecycle.
        if env_ids is not None:
            if isinstance(dt, np.ndarray):
                raise ValueError("GroundPickPhaseCommand reset compute expects scalar dt")
            if not np.isfinite(dt):
                raise ValueError("GroundPickPhaseCommand received non-finite dt")
            self._update_command(np.asarray(env_ids, dtype=np.int32))
            return
        delta = np.asarray(dt, dtype=get_global_dtype())
        if not np.isfinite(delta).all():
            raise ValueError("GroundPickPhaseCommand received non-finite dt")
        if delta.ndim == 0:
            self._phase[:] = (self._phase + float(delta) / self._period) % 1.0
        elif delta.shape == (self.num_envs,):
            self._phase[:] = (self._phase + delta / self._period) % 1.0
        else:
            raise ValueError(
                f"GroundPickPhaseCommand dt must be scalar or ({self.num_envs},), got {delta.shape}"
            )
        self._update_command(None)

    def _resample_command(self, env_ids: np.ndarray) -> None:
        del env_ids

    def _update_command(self, env_ids: np.ndarray | None = None) -> None:
        del env_ids
        self.vel_command_b[:, 0] = np.cos(2.0 * math.pi * self._phase)
        self.vel_command_b[:, 1] = np.sin(2.0 * math.pi * self._phase)
        self.vel_command_b[:, 2] = 0.0

    def _update_metrics(self, env_ids: np.ndarray | None = None) -> None:
        del env_ids


@dataclass(kw_only=True)
class GroundPickPhaseCommandCfg(UniformVelocityCommandCfg):
    # Retained because this task term specializes the shared velocity command
    # config selected by the owner overlay.
    turn_in_place_fraction: float = 0.0
    turn_in_place_ang_min: float = 0.4
    period: float = 4.0
    randomize_phase: bool = True

    def build(self, env: ManagerBasedRlEnv) -> GroundPickPhaseCommand:
        return GroundPickPhaseCommand(self, env)


class SitStandCommand(UniformVelocityCommand):
    """Binary posture command with a bounded-rate internal target blend."""

    def __init__(self, cfg: SitStandCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._sit_prob = _ratio(cfg.sit_prob, label="SitStandCommand sit_prob")
        self._ramp_s = _finite_real(
            cfg.ramp_s,
            label="SitStandCommand ramp_s",
            minimum=0.0,
            strict_minimum=True,
        )
        self._sit_z = _finite_real(cfg.sit_z, label="SitStandCommand sit_z")
        self._stand_z = _finite_real(cfg.stand_z, label="SitStandCommand stand_z")
        if self._sit_z >= self._stand_z:
            raise ValueError("SitStandCommand sit_z must be below stand_z")
        self._alpha = np.zeros(self.num_envs, dtype=get_global_dtype())
        self._robot = cast("Entity", env.scene[cfg.entity_name])

    @property
    def alpha(self) -> np.ndarray:
        return self._alpha

    def _resample_command(self, env_ids: np.ndarray) -> None:
        self.vel_command_b[env_ids] = 0.0
        sit = self._env.rng.uniform(0.0, 1.0, size=len(env_ids)) < self._sit_prob
        self.vel_command_b[env_ids, 0] = sit.astype(get_global_dtype())

    def _update_command(self, env_ids: np.ndarray | None = None) -> None:
        del env_ids

    def compute(self, dt: float | np.ndarray, env_ids: np.ndarray | None = None) -> None:
        super().compute(dt, env_ids)
        if env_ids is not None:
            ids = np.asarray(env_ids, dtype=np.int32)
            fresh = self._env.episode_length_buf[ids] <= 1
            if np.any(fresh):
                self._alpha[ids[fresh]] = self._alpha_from_height(ids[fresh])
            return
        dt_values = np.asarray(dt, dtype=get_global_dtype())
        if dt_values.ndim == 0:
            step = np.full(self.num_envs, float(dt_values), dtype=get_global_dtype())
        elif dt_values.shape == (self.num_envs,):
            step = dt_values
        else:
            raise ValueError(
                f"SitStandCommand dt must be scalar or ({self.num_envs},), got {dt_values.shape}"
            )
        fresh = self._env.episode_length_buf <= 1
        if np.any(fresh):
            self._alpha[fresh] = self._alpha_from_height(np.flatnonzero(fresh))
        target = self.vel_command_b[:, 0]
        self._alpha += np.clip(target - self._alpha, -step / self._ramp_s, step / self._ramp_s)

    def _alpha_from_height(self, ids: np.ndarray) -> np.ndarray:
        height = self._robot.data.root_link_pos_w[ids, 2]
        return np.clip(
            (self._stand_z - height) / max(self._stand_z - self._sit_z, 1e-6),
            0.0,
            1.0,
        ).astype(get_global_dtype(), copy=False)

    def _update_metrics(self, env_ids: np.ndarray | None = None) -> None:
        del env_ids


@dataclass(kw_only=True)
class SitStandCommandCfg(UniformVelocityCommandCfg):
    # Retained because this task term specializes the shared velocity command
    # config selected by the owner overlay.
    turn_in_place_fraction: float = 0.0
    turn_in_place_ang_min: float = 0.4
    sit_prob: float = 0.5
    ramp_s: float = 2.0
    sit_z: float = 0.060
    stand_z: float = 0.115

    def build(self, env: ManagerBasedRlEnv) -> SitStandCommand:
        return SitStandCommand(self, env)


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
    _allowed_params = frozenset(
        {
            "asset_cfg",
            "command_name",
            "tau_s",
            "gate_height_low",
            "gate_height_high",
            "gate_tilt_full_deg",
            "gate_tilt_zero_deg",
        }
    )

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
        gate_height_low = cfg.params.get("gate_height_low")
        if gate_height_low is None:
            self._gate: tuple[float, float, float, float] | None = None
        else:
            # Upright gate for recovery envs (upstream head_pose_bias_penalty):
            # smoothstep in root height and tilt, multiplied into the EMA input
            # and the output so the bias clock starts at ~0 when standing up.
            self._gate = (
                _finite_real(gate_height_low, label=f"{self.name} gate_height_low"),
                _finite_real(
                    cfg.params.get("gate_height_high", 0.11),
                    label=f"{self.name} gate_height_high",
                ),
                _finite_real(
                    cfg.params.get("gate_tilt_full_deg", 20.0),
                    label=f"{self.name} gate_tilt_full_deg",
                ),
                _finite_real(
                    cfg.params.get("gate_tilt_zero_deg", 45.0),
                    label=f"{self.name} gate_tilt_zero_deg",
                ),
            )

    def reset(self, env_ids: np.ndarray | slice | None) -> None:
        self._ema[slice(None) if env_ids is None else env_ids] = 0.0

    def _upright_gate(self, env: ManagerBasedRlEnv) -> np.ndarray:
        assert self._gate is not None
        height_low, height_high, tilt_full_deg, tilt_zero_deg = self._gate
        z = np.nan_to_num(
            self._entity.data.root_link_pos_w[:, 2] - np.asarray(env.scene.env_origins)[:, 2],
            nan=0.0,
        )
        t = np.clip((z - height_low) / max(height_high - height_low, 1e-6), 0.0, 1.0)
        gate = t * t * (3.0 - 2.0 * t)
        quat = self._entity.data.root_link_quat_w
        cos_tilt = 1.0 - 2.0 * (np.square(quat[:, 1]) + np.square(quat[:, 2]))
        tilt_deg = np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0)))
        st = np.clip(
            (tilt_zero_deg - tilt_deg) / max(tilt_zero_deg - tilt_full_deg, 1e-6),
            0.0,
            1.0,
        )
        return gate * (st * st * (3.0 - 2.0 * st))

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        fresh = env.episode_length_buf <= 1
        self._ema[fresh] = 0.0
        error = self._joint_error(env)
        gate = self._upright_gate(env) if self._gate is not None else None
        if gate is not None:
            error = error * gate[:, None]
        self._ema *= 1.0 - self._alpha
        self._ema += self._alpha * error
        output = -np.mean(np.abs(self._ema), axis=1)
        if gate is not None:
            output = output * gate
        return np.asarray(output, dtype=get_global_dtype())


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


def posture_height_tracking(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    sit_height: float = 0.060,
    stand_height: float = 0.115,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    """Track the slewed height target exposed by :class:`SitStandCommand`."""
    if std <= 0.0 or not np.isfinite(std):
        raise ValueError("posture_height_tracking std must be finite and positive")
    if not isinstance(asset_cfg, SceneEntityCfg):
        raise TypeError("posture_height_tracking asset_cfg must be SceneEntityCfg")
    command_term = env.command_manager.get_term(command_name)
    if not isinstance(command_term, SitStandCommand):
        raise TypeError(
            f"posture_height_tracking requires a SitStandCommand, got {type(command_term).__name__}"
        )
    alpha = command_term.alpha
    target = (1.0 - alpha) * stand_height + alpha * sit_height
    actual = _state(
        "posture_height_tracking",
        "root position",
        cast("Entity", env.scene[asset_cfg.name]).data.root_link_pos_w,
        (env.num_envs, 3),
    )[:, 2]
    return np.asarray(np.exp(-np.square((actual - target) / std)), dtype=get_global_dtype())


def phase_height_tracking(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    sit_height: float = 0.060,
    stand_height: float = 0.115,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    """Track a cyclic stand-to-sit height target encoded by phase sine."""
    if std <= 0.0 or not np.isfinite(std):
        raise ValueError("phase_height_tracking std must be finite and positive")
    if not isinstance(asset_cfg, SceneEntityCfg):
        raise TypeError("phase_height_tracking asset_cfg must be SceneEntityCfg")
    command = _command(env, "phase_height_tracking", command_name)
    if command.shape[1] < 2:
        raise ValueError("phase_height_tracking command must contain cosine and sine")
    midpoint = 0.5 * (stand_height + sit_height)
    amplitude = 0.5 * (stand_height - sit_height)
    target = midpoint - amplitude * command[:, 1]
    actual = _state(
        "phase_height_tracking",
        "root position",
        cast("Entity", env.scene[asset_cfg.name]).data.root_link_pos_w,
        (env.num_envs, 3),
    )[:, 2]
    return np.asarray(np.exp(-np.square((actual - target) / std)), dtype=get_global_dtype())


def body_pose_tracking(
    env: ManagerBasedRlEnv,
    command_name: str = "body_pose",
    nominal_height: float = 0.095,
    xy_std: float = 0.05,
    z_std: float = 0.02,
    angle_std: float = math.radians(15),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    """Mean of six per-axis Gaussians tracking the commanded 6D body pose delta.

    Mirrors upstream ``body_pose_tracking_6d``: the command is
    ``[x, y, z, roll, pitch, yaw]`` as deltas from the nominal standing pose
    (xy from the env origin, z from ``nominal_height``, angles from upright).
    The velocity task keeps this term at weight zero so the body_pose command
    and observation channels stay alive for tasks that raise the weight.
    """
    _finite_real(nominal_height, label="body_pose_tracking nominal_height")
    for label, value in (("xy_std", xy_std), ("z_std", z_std), ("angle_std", angle_std)):
        _finite_real(
            value,
            label=f"body_pose_tracking {label}",
            minimum=0.0,
            strict_minimum=True,
        )
    if not isinstance(asset_cfg, SceneEntityCfg):
        raise TypeError("body_pose_tracking asset_cfg must be SceneEntityCfg")
    command = _command(env, "body_pose_tracking", command_name)
    if command.shape[1] != 6:
        raise ValueError(
            f"body_pose_tracking command '{command_name}' must have width 6, "
            f"received shape {command.shape}"
        )
    asset = cast("Entity", env.scene[asset_cfg.name])
    position = _state(
        "body_pose_tracking",
        "root position",
        asset.data.root_link_pos_w - np.asarray(env.scene.env_origins),
        (env.num_envs, 3),
    )
    quat = _state(
        "body_pose_tracking",
        "root quaternion",
        asset.data.root_link_quat_w,
        (env.num_envs, 4),
    )
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    errors = (
        ((position[:, 0] - command[:, 0]) / xy_std) ** 2,
        ((position[:, 1] - command[:, 1]) / xy_std) ** 2,
        ((position[:, 2] - nominal_height - command[:, 2]) / z_std) ** 2,
        ((roll - command[:, 3]) / angle_std) ** 2,
        ((pitch - command[:, 4]) / angle_std) ** 2,
        (np_wrap_to_pi(yaw - command[:, 5]) / angle_std) ** 2,
    )
    return np.asarray(np.mean(np.exp(-np.stack(errors, axis=1)), axis=1), dtype=get_global_dtype())


__all__ = [
    "GroundPickPhaseCommand",
    "GroundPickPhaseCommandCfg",
    "MicroduckVelocityCommand",
    "MicroduckVelocityCommandCfg",
    "SitStandCommand",
    "SitStandCommandCfg",
    "body_pose_tracking",
    "flight_phase",
    "foot_air_time_biped",
    "head_pose_bias",
    "head_pose_tracking",
    "leg_pose",
    "phase_height_tracking",
    "posture_height_tracking",
]
