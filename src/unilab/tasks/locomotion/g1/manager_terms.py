"""Task-owned Manager-Based terms for G1 biped locomotion.

The equations come from UniLab's legacy G1Walk joystick tasks.  The adaptation
uses community ``func + params`` terms, NumPy, cached cold-path sensor bindings,
and the base-owned entity facade; hot paths never parse assets or probe backend
privates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast
from weakref import WeakKeyDictionary

import numpy as np
import torch

from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.base.curriculum import EpisodeLengthTracker
from unilab.dtype_config import get_global_dtype
from unilab.envs.manager_based_rl_env import (
    ManagerBasedRlEnv as _ConcreteManagerBasedRlEnv,
)
from unilab.envs.manager_based_rl_env import (
    ManagerBasedRlEnvCfg,
    _resolve_backend_entity_contract,
)
from unilab.envs.mdp.commands.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.tasks.locomotion.common.manager_terms import SensorTermBase

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv

    class _RewardTermCfgView(Protocol):
        weight: float

    class _RewardManagerView(Protocol):
        @property
        def active_terms(self) -> list[str]: ...

        def get_term_cfg(self, term_name: str) -> _RewardTermCfgView: ...

    class _G1Env(ManagerBasedRlEnv, Protocol):
        @property
        def common_step_counter(self) -> int: ...

        @property
        def reset_buf(self) -> np.ndarray: ...

        @property
        def reward_manager(self) -> _RewardManagerView: ...


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_LEFT_FOOT_CONTACT_SENSORS = tuple(f"left_foot_contact_{index}" for index in range(4))
_RIGHT_FOOT_CONTACT_SENSORS = tuple(f"right_foot_contact_{index}" for index in range(4))
_FOOT_CONTACT_SENSORS = _LEFT_FOOT_CONTACT_SENSORS + _RIGHT_FOOT_CONTACT_SENSORS
_FOOT_POS_SENSORS = ("left_foot_pos", "right_foot_pos")
_FOOT_QUAT_SENSORS = ("left_foot_quat", "right_foot_quat")
_GAIT_INIT_MODES = ("offset_phase", "independent")


def _real(
    term: str,
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{term} {name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{term} {name} must be finite")
    if minimum is not None and (result <= minimum if strict_minimum else result < minimum):
        relation = "greater than" if strict_minimum else "at least"
        raise ValueError(f"{term} {name} must be {relation} {minimum}")
    return result


def _weights(term: str, name: str, value: Any) -> np.ndarray:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list, np.ndarray)):
        raise TypeError(f"{term} {name} must be a sequence of real numbers")
    result = np.asarray(
        [_real(term, f"{name}[{index}]", item) for index, item in enumerate(value)],
        dtype=get_global_dtype(),
    )
    if result.ndim != 1 or result.shape[0] == 0:
        raise ValueError(f"{term} {name} must be a non-empty one-dimensional sequence")
    return result


def _square(value: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(value, torch.Tensor):
        return torch.square(value)
    return np.square(value)


def _exp(value: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(value, torch.Tensor):
        return torch.exp(value)
    return np.exp(value)


def _values(value: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float32)
    return np.asarray(value, dtype=get_global_dtype())


def _bool_values(value: np.ndarray | torch.Tensor | Any) -> np.ndarray | torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float32)
    return np.asarray(value, dtype=get_global_dtype())


def _array_like(value: np.ndarray | torch.Tensor | bool | np.bool_) -> np.ndarray | torch.Tensor:
    if isinstance(value, (bool, np.bool_)):
        return np.asarray(value)
    return value


def _state(term: str, capability: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{term} {capability} must be an np.ndarray")
    if value.shape != shape:
        raise ValueError(f"{term} {capability} must have shape {shape}, got {value.shape}")
    if not np.isfinite(value).all():
        env_ids = np.flatnonzero(~np.isfinite(value).reshape(shape[0], -1).all(axis=1)).tolist()
        raise ValueError(f"{term} {capability} contains NaN or Inf for environments {env_ids[:10]}")
    return value


def _command(env: ManagerBasedRlEnv, term: str, command_name: str) -> np.ndarray:
    if not isinstance(command_name, str) or not command_name:
        raise ValueError(f"{term} command_name must be a non-empty string")
    try:
        command = env.command_manager.get_command(command_name)
    except KeyError as exc:
        raise KeyError(f"{term} command capability '{command_name}' is unavailable") from exc
    if command is None:
        raise KeyError(f"{term} command capability '{command_name}' is unavailable")
    return _state(term, f"command '{command_name}'", command, (env.num_envs, 3))


def _asset(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> Entity:
    return cast("Entity", env.scene[asset_cfg.name])


class _SensorTerm(SensorTermBase):
    """G1 manager terms share the locomotion-wide named-sensor binding."""

    def _read_state_tensor(
        self,
        env: ManagerBasedRlEnv,
        capability: str,
        binding: tuple[Any, tuple[str, ...]],
        shape: tuple[int, ...],
    ) -> np.ndarray | torch.Tensor:
        view, sensor_names = binding
        read_plan = getattr(env.scene, "_tensor_read_plan", None)
        if read_plan is not None and read_plan.device.type == "cuda":
            owner_sensors = set(read_plan.sensor_names.get("robot", ()))
            if not set(sensor_names).issubset(owner_sensors):
                # Device-resident adapters currently publish only the audited
                # named IMU subset. Body-frame task sensors still use their
                # validated host cache until carrier migration completes.
                read_plan = None
        if read_plan is None:
            return _state(self.name, capability, self._read(view, self.name), shape)
        entity = env.scene["robot"]
        views = read_plan.sensor_tensor_views(entity, sensor_names).values
        if len(sensor_names) == 1:
            values = views[sensor_names[0]]
            if tuple(values.shape) != shape:
                raise ValueError(
                    f"{self.name} {capability} must have shape {shape}, got {tuple(values.shape)}"
                )
            if not bool(torch.isfinite(values).all()):
                raise ValueError(f"{self.name} {capability} contains NaN or Inf")
            return values
        values = torch.cat([views[name] for name in sensor_names], dim=1)
        if tuple(values.shape) != shape:
            raise ValueError(
                f"{self.name} {capability} must have shape {shape}, got {tuple(values.shape)}"
            )
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"{self.name} {capability} contains NaN or Inf")
        return values


# ---------------------------------------------------------------------------
# Gait phase state (shared by the observation term and the gait reward terms)
# ---------------------------------------------------------------------------


def compute_feet_phase_height_targets(
    gait_phase: np.ndarray | torch.Tensor, swing_height: float
) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """Cubic-Bézier per-foot height targets, ported from the legacy G1 task."""

    def cubic_bezier_height(
        phi: np.ndarray | torch.Tensor, swing_height: float
    ) -> np.ndarray | torch.Tensor:
        if isinstance(phi, torch.Tensor):
            # Keep the legacy branch structure while bounding only the half
            # of the domain it selects; ``where`` preserves the original values.
            phi_normalized = torch.remainder(phi + math.pi, 2 * math.pi) - math.pi
            x = (phi_normalized + math.pi) / (2 * math.pi)
            stance_t = (2 * x).clamp(max=1.0)
            stance = swing_height * (stance_t**3 + 3 * stance_t**2 * (1 - stance_t))
            swing_t = (2 * x - 1).clamp(min=0.0)
            swing = swing_height - swing_height * (swing_t**3 + 3 * swing_t**2 * (1 - swing_t))
            return torch.where(x <= 0.5, stance, swing)
        phi_normalized = np.fmod(phi + np.pi, 2 * np.pi) - np.pi
        x = (phi_normalized + np.pi) / (2 * np.pi)

        def cubic_bezier_interpolation(
            y_start: np.ndarray, y_end: np.ndarray, t: np.ndarray
        ) -> np.ndarray:
            y_diff = y_end - y_start
            bezier = t**3 + 3 * (t**2 * (1 - t))
            return np.asarray(y_start + y_diff * bezier, dtype=get_global_dtype())

        stance = cubic_bezier_interpolation(np.zeros_like(x), np.full_like(x, swing_height), 2 * x)
        swing = cubic_bezier_interpolation(
            np.full_like(x, swing_height), np.zeros_like(x), 2 * x - 1
        )
        return np.where(x <= 0.5, stance, swing)

    left_target = cubic_bezier_height(gait_phase[:, 0], swing_height)
    right_target = cubic_bezier_height(gait_phase[:, 1], swing_height)
    return left_target, right_target


def compute_feet_phase_contact_targets(
    gait_phase: np.ndarray | torch.Tensor, swing_height: float
) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """Expected per-foot contact flags derived from the Bézier height targets."""
    left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
    contact_height_threshold = swing_height * 0.5
    return left_target <= contact_height_threshold, right_target <= contact_height_threshold


@dataclass
class _G1GaitContext:
    phase: np.ndarray | torch.Tensor  # (num_envs, 2), radians in [0, 2*pi)
    delta: float  # 2*pi*frequency*ctrl_dt
    frequency: float
    init_mode: str
    last_counter: int


_GAIT_CONTEXTS: WeakKeyDictionary[Any, _G1GaitContext] = WeakKeyDictionary()


def _gait_context(env: _G1Env, term: str, frequency: float, init_mode: str) -> _G1GaitContext:
    context = _GAIT_CONTEXTS.get(env)
    if context is None:
        delta = float(
            2.0
            * math.pi
            * frequency
            * _real(term, "step_dt", env.step_dt, minimum=0.0, strict_minimum=True)
        )
        device = torch.device(env.device)
        phase = (
            torch.zeros((env.num_envs, 2), dtype=torch.float32, device=device)
            if device.type == "cuda"
            else np.zeros((env.num_envs, 2), dtype=get_global_dtype())
        )
        context = _G1GaitContext(
            phase=phase,
            delta=delta,
            frequency=frequency,
            init_mode=init_mode,
            last_counter=int(env.common_step_counter),
        )
        _GAIT_CONTEXTS[env] = context
        return context
    if context.frequency != frequency or context.init_mode != init_mode:
        raise ValueError(
            f"{term} gait terms on one env must share frequency and init_mode; "
            f"got frequency {context.frequency} vs {frequency}, "
            f"init_mode {context.init_mode!r} vs {init_mode!r}"
        )
    return context


def _advance_gait(env: _G1Env, context: _G1GaitContext) -> np.ndarray | torch.Tensor:
    counter = env.common_step_counter
    if isinstance(counter, (bool, np.bool_)) or not isinstance(counter, (int, np.integer)):
        raise TypeError("G1 gait terms require an integer common_step_counter")
    counter = int(counter)
    if counter < context.last_counter:
        raise ValueError("G1 gait terms common_step_counter cannot move backwards")
    two_pi = 2.0 * np.pi
    if isinstance(context.phase, torch.Tensor):
        steps = counter - context.last_counter
        if steps:
            phase = torch.remainder(context.phase + steps * context.delta, two_pi)
            context.phase = phase
        context.last_counter = counter
        return context.phase
    for _ in range(counter - context.last_counter):
        context.phase = np.asarray(
            np.fmod(context.phase + context.delta, two_pi), dtype=get_global_dtype()
        )
    context.last_counter = counter
    return context.phase


def _resample_gait(env: _G1Env, context: _G1GaitContext, env_ids: np.ndarray) -> None:
    ids = np.asarray(env_ids, dtype=np.intp).reshape(-1)
    count = len(ids)
    if count == 0:
        return
    if context.init_mode == "independent":
        left = env.rng.uniform(0.0, 2.0 * np.pi, size=(count,))
        right = env.rng.uniform(0.0, 2.0 * np.pi, size=(count,))
    else:
        left = env.rng.uniform(0.0, 2.0 * np.pi, size=(count,))
        right = left + np.pi
    samples = np.column_stack([left, right]).astype(get_global_dtype(), copy=False)
    if isinstance(context.phase, torch.Tensor):
        context.phase[torch.as_tensor(ids, device=context.phase.device)] = torch.from_numpy(
            samples
        ).to(device=context.phase.device, dtype=context.phase.dtype)
    else:
        context.phase[ids] = samples


class G1GaitPhase(ManagerTermBase):
    """Two-foot gait phase observation in radians, with per-reset phase sampling."""

    _allowed_params: ClassVar[frozenset[str]] = frozenset({"frequency", "init_mode"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: _G1Env):
        super().__init__(env)
        unexpected = set(cfg.params) - self._allowed_params
        if unexpected:
            raise TypeError(f"{self.name} received unsupported parameters: {sorted(unexpected)}")
        frequency = _real(self.name, "frequency", cfg.params.get("frequency", 1.5), minimum=0.0)
        init_mode = cfg.params.get("init_mode", "offset_phase")
        if init_mode not in _GAIT_INIT_MODES:
            raise ValueError(f"{self.name} init_mode must be one of {_GAIT_INIT_MODES}")
        self._context = _gait_context(env, self.name, frequency, init_mode)

    def __call__(self, env: _G1Env, **params: Any) -> np.ndarray | torch.Tensor:
        del params
        phase = _advance_gait(env, self._context)
        if isinstance(phase, torch.Tensor):
            return phase
        return phase.copy()

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        if env_ids is None:
            ids = np.arange(self.num_envs, dtype=np.intp)
        elif isinstance(env_ids, slice):
            ids = np.arange(self.num_envs, dtype=np.intp)[env_ids]
        else:
            ids = np.asarray(env_ids, dtype=np.intp).reshape(-1)
        _resample_gait(cast("_G1Env", self._env), self._context, ids)


class _GaitRewardTerm(_SensorTerm):
    """Shared gait-context, forward-speed gate, and foot-position binding."""

    _allowed_params = frozenset(
        {
            "frequency",
            "init_mode",
            "swing_height",
            "tracking_sigma",
            "min_forward_speed",
            "command_name",
        }
    )

    def __init__(self, cfg: ManagerTermBaseCfg, env: _G1Env):
        super().__init__(cfg, env)
        frequency = _real(self.name, "frequency", cfg.params.get("frequency", 1.5), minimum=0.0)
        init_mode = cfg.params.get("init_mode", "offset_phase")
        if init_mode not in _GAIT_INIT_MODES:
            raise ValueError(f"{self.name} init_mode must be one of {_GAIT_INIT_MODES}")
        self._swing_height = _real(
            self.name, "swing_height", cfg.params.get("swing_height", 0.09), minimum=0.0
        )
        self._tracking_sigma = _real(
            self.name,
            "tracking_sigma",
            cfg.params.get("tracking_sigma", 0.008),
            minimum=0.0,
            strict_minimum=True,
        )
        self._min_forward_speed = _real(
            self.name,
            "min_forward_speed",
            cfg.params.get("min_forward_speed", 0.0),
            minimum=0.0,
        )
        command_name = cfg.params.get("command_name", "twist")
        if not isinstance(command_name, str) or not command_name:
            raise ValueError(f"{self.name} command_name must be a non-empty string")
        self._command_name = command_name
        self._context = _gait_context(env, self.name, frequency, init_mode)
        self._feet_pos = self._bind(_FOOT_POS_SENSORS)
        if self._feet_pos.dimensions != (3, 3):
            raise ValueError(
                f"{self.name} foot position sensors must expose 3-D xyz; received "
                f"{self._feet_pos.dimensions} on backend '{self._feet_pos.backend_type}'"
            )
        self._linvel = self._bind(("pelvis_local_linvel",))

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        return (*_FOOT_POS_SENSORS, "pelvis_local_linvel")

    def _targets(self, env: _G1Env) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        phase = _advance_gait(env, self._context)
        return compute_feet_phase_height_targets(phase, self._swing_height)

    def _foot_heights(
        self, env: _G1Env
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        values = self._read_state_tensor(
            env, "foot position", (self._feet_pos, _FOOT_POS_SENSORS), (self.num_envs, 6)
        )
        return values[:, 2], values[:, 5]

    def _gate(self, env: ManagerBasedRlEnv) -> np.ndarray | torch.Tensor:
        linvel = self._read_state_tensor(
            env,
            "pelvis linear velocity",
            (self._linvel, ("pelvis_local_linvel",)),
            (env.num_envs, 3),
        )
        if isinstance(linvel, torch.Tensor):
            forward_speed = torch.clamp(linvel[:, 0], min=0.0)
            return (forward_speed >= self._min_forward_speed).to(dtype=torch.float32)
        forward_speed = np.maximum(linvel[:, 0], 0.0)
        return np.asarray(forward_speed >= self._min_forward_speed, dtype=get_global_dtype())


class feet_phase(_GaitRewardTerm):
    """Reward gait phase tracking by encouraging the expected swing-foot height."""

    def __call__(self, env: _G1Env, **params: Any) -> np.ndarray | torch.Tensor:
        del params
        left_target, right_target = self._targets(env)
        left_height, right_height = self._foot_heights(env)
        if isinstance(left_height, torch.Tensor) or isinstance(left_target, torch.Tensor):
            source = left_height if isinstance(left_height, torch.Tensor) else left_target
            device = source.device
            left_height = torch.as_tensor(left_height, device=device, dtype=torch.float32)
            right_height = torch.as_tensor(right_height, device=device, dtype=torch.float32)
            left_target = torch.as_tensor(left_target, device=device, dtype=torch.float32)
            right_target = torch.as_tensor(right_target, device=device, dtype=torch.float32)
        error = _square(left_height - left_target) + _square(right_height - right_target)
        reward = _exp(-error / self._tracking_sigma)
        return _values(reward * self._gate(env))


class feet_phase_contrast(_GaitRewardTerm):
    """Reward left/right foot-height contrast against the gait-phase targets."""

    def __call__(self, env: _G1Env, **params: Any) -> np.ndarray | torch.Tensor:
        del params
        left_target, right_target = self._targets(env)
        left_height, right_height = self._foot_heights(env)
        error = _square((left_height - right_height) - (left_target - right_target))
        reward = _exp(-error / self._tracking_sigma)
        return _values(reward * self._gate(env))


class _FootContactTerm(_GaitRewardTerm):
    """Adds the aggregated per-foot contact binding shared by contact gait terms."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: _G1Env):
        super().__init__(cfg, env)
        self._contacts = self._bind(_FOOT_CONTACT_SENSORS)
        if self._contacts.dimensions != (1,) * 8:
            raise ValueError(
                f"{self.name} foot contact sensors must each expose 1-D found; received "
                f"{self._contacts.dimensions} on backend '{self._contacts.backend_type}'"
            )

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        return (*super().tensor_sensor_names, *_FOOT_CONTACT_SENSORS)

    def _contact_pair(
        self, env: ManagerBasedRlEnv
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        values = self._read_state_tensor(
            env, "foot contact", (self._contacts, _FOOT_CONTACT_SENSORS), (self.num_envs, 8)
        )
        if isinstance(values, torch.Tensor):
            left = torch.any(values[:, :4] > 0.5, dim=1)
            right = torch.any(values[:, 4:] > 0.5, dim=1)
            return left, right
        left = np.any(values[:, :4] > 0.5, axis=1)
        right = np.any(values[:, 4:] > 0.5, axis=1)
        return left, right


class feet_phase_contact(_FootContactTerm):
    """Reward foot contact matching the expected stance phase of the gait."""

    def __call__(self, env: _G1Env, **params: Any) -> np.ndarray | torch.Tensor:
        del params
        phase = _advance_gait(env, self._context)
        left_target, right_target = compute_feet_phase_contact_targets(phase, self._swing_height)
        left_contact, right_contact = self._contact_pair(env)
        left_values: np.ndarray | torch.Tensor = left_contact
        right_values: np.ndarray | torch.Tensor = right_contact
        left_match = _bool_values(_array_like(left_values) == left_target)
        right_match = _bool_values(_array_like(right_values) == right_target)
        return _values(0.5 * (left_match + right_match) * self._gate(env))


class feet_double_stance(_FootContactTerm):
    """Penalize double-stance contact while a forward command is active."""

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray | torch.Tensor:
        del params
        command = _command(env, self.name, self._command_name)
        left_contact, right_contact = self._contact_pair(env)
        double_stance = np.asarray(
            np.logical_and(left_contact, right_contact), dtype=get_global_dtype()
        )
        forward_mask = np.asarray(np.maximum(command[:, 0], 0.0) > 1.0e-6, dtype=get_global_dtype())
        return np.asarray(double_stance * forward_mask, dtype=get_global_dtype())


class feet_air_time(_FootContactTerm):
    """Count feet whose current air time sits inside the rewarded window."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: _G1Env):
        super().__init__(cfg, env)
        self._air_time = np.zeros((env.num_envs, 2), dtype=get_global_dtype())

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        self._air_time[env_ids if env_ids is not None else slice(None)] = 0.0

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray | torch.Tensor:
        del params
        left_contact, right_contact = self._contact_pair(env)
        contact = np.column_stack([left_contact, right_contact])
        step_dt = _real(self.name, "step_dt", env.step_dt, minimum=0.0, strict_minimum=True)
        self._air_time = np.where(contact, 0.0, self._air_time + step_dt).astype(
            get_global_dtype(), copy=False
        )
        in_range = (self._air_time > 0.05) & (self._air_time < 0.5)
        return np.asarray(
            np.sum(in_range.astype(get_global_dtype()), axis=1), dtype=get_global_dtype()
        )


# ---------------------------------------------------------------------------
# Velocity / orientation reward terms (sensor-bound, legacy equations)
# ---------------------------------------------------------------------------


class _LinVelTerm(_SensorTerm):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._linvel = self._bind(("pelvis_local_linvel",))

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        return ("pelvis_local_linvel",)

    def _read_linvel(self, env: ManagerBasedRlEnv) -> np.ndarray | torch.Tensor:
        return self._read_state_tensor(
            env,
            "pelvis linear velocity",
            (self._linvel, ("pelvis_local_linvel",)),
            (env.num_envs, 3),
        )


class _GyroTerm(_SensorTerm):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._gyro = self._bind(("torso_gyro",))

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        return ("torso_gyro",)

    def _read_gyro(self, env: ManagerBasedRlEnv) -> np.ndarray | torch.Tensor:
        return self._read_state_tensor(
            env, "torso gyro", (self._gyro, ("torso_gyro",)), (env.num_envs, 3)
        )


class _UpvectorTerm(_SensorTerm):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._upvector = self._bind(("torso_upvector",))

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        return ("torso_upvector",)

    def _read_upvector(self, env: ManagerBasedRlEnv) -> np.ndarray | torch.Tensor:
        return self._read_state_tensor(
            env, "torso upvector", (self._upvector, ("torso_upvector",)), (env.num_envs, 3)
        )


class forward_progress(_LinVelTerm):
    """Reward forward progress relative to commanded speed."""

    _allowed_params = frozenset({"command_name"})

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray | torch.Tensor:
        command = _command(env, self.name, _term_command_name(self.name, params))
        linvel = self._read_linvel(env)
        commanded_speed = np.maximum(command[:, 0], 1e-6)
        forward_speed = np.maximum(linvel[:, 0], 0.0)
        return np.asarray(
            np.minimum(forward_speed / commanded_speed, 1.0), dtype=get_global_dtype()
        )


def _term_command_name(term: str, params: dict[str, Any]) -> str:
    name = params.get("command_name", "twist")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{term} command_name must be a non-empty string")
    return name


class under_speed(_LinVelTerm):
    """Penalty for being below commanded forward speed."""

    _allowed_params = frozenset({"command_name"})

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray | torch.Tensor:
        command = _command(env, self.name, _term_command_name(self.name, params))
        linvel = self._read_linvel(env)
        commanded_speed = np.maximum(command[:, 0], 1e-6)
        forward_speed = np.maximum(linvel[:, 0], 0.0)
        gap = np.maximum(command[:, 0] - forward_speed, 0.0)
        return np.asarray(gap / commanded_speed, dtype=get_global_dtype())


class g1_tilt_exceeded(_UpvectorTerm):
    """Terminate when the base tilt from upright exceeds ``max_tilt_deg``."""

    _allowed_params = frozenset({"max_tilt_deg"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        max_tilt_deg = _real(self.name, "max_tilt_deg", cfg.params.get("max_tilt_deg"), minimum=0.0)
        self._max_tilt_rad = math.radians(max_tilt_deg)

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray | torch.Tensor:
        del params
        upvector = self._read_upvector(env)
        if isinstance(upvector, torch.Tensor):
            z = torch.clamp(upvector[:, 2], min=-1.0, max=1.0)
            return torch.acos(z) > self._max_tilt_rad
        tilt = np.arccos(np.clip(upvector[:, 2], -1.0, 1.0))
        return np.asarray(tilt > self._max_tilt_rad, dtype=np.bool_)


class penalty_feet_ori(_SensorTerm):
    """Penalty for non-flat foot orientations (roll/pitch quaternion rows)."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._quats = self._bind(_FOOT_QUAT_SENSORS)
        if self._quats.dimensions != (4, 4):
            raise ValueError(
                f"{self.name} foot quaternion sensors must expose 4-D quats; received "
                f"{self._quats.dimensions} on backend '{self._quats.backend_type}'"
            )

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        return _FOOT_QUAT_SENSORS

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray | torch.Tensor:
        del params
        values = self._read_state_tensor(
            env,
            "foot quaternion",
            (self._quats, _FOOT_QUAT_SENSORS),
            (env.num_envs, 8),
        )
        penalty = _square(values[:, 1]) + _square(values[:, 2])
        penalty = penalty + _square(values[:, 5]) + _square(values[:, 6])
        return _values(penalty)


class penalty_close_feet_xy(_SensorTerm):
    """Penalty for feet closer than ``threshold`` in the horizontal plane."""

    _allowed_params = frozenset({"threshold"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._threshold = _real(
            self.name, "threshold", cfg.params.get("threshold", 0.15), minimum=0.0
        )
        self._feet_pos = self._bind(_FOOT_POS_SENSORS)
        if self._feet_pos.dimensions != (3, 3):
            raise ValueError(
                f"{self.name} foot position sensors must expose 3-D xyz; received "
                f"{self._feet_pos.dimensions} on backend '{self._feet_pos.backend_type}'"
            )

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        return _FOOT_POS_SENSORS

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray | torch.Tensor:
        del params
        values = self._read_state_tensor(
            env,
            "foot position",
            (self._feet_pos, _FOOT_POS_SENSORS),
            (env.num_envs, 6),
        )
        feet_delta = values[:, :2] - values[:, 3:5]
        if isinstance(values, torch.Tensor):
            feet_dist = torch.linalg.vector_norm(feet_delta, dim=1)
            feet_gap = torch.square(feet_dist - self._threshold)
            return torch.where(
                feet_dist < self._threshold,
                feet_gap,
                torch.zeros_like(feet_dist),
            ).to(dtype=torch.float32)
        feet_dist = np.linalg.norm(feet_delta, axis=1)
        return np.asarray(
            np.where(feet_dist < self._threshold, _square(feet_dist - self._threshold), 0.0),
            dtype=get_global_dtype(),
        )


# ---------------------------------------------------------------------------
# Entity-based reward terms (legacy equations, no sensor binding)
# ---------------------------------------------------------------------------


def weighted_pose(
    env: ManagerBasedRlEnv,
    pose_weights: Any,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray | torch.Tensor:
    """Weighted L2 penalty for joint position deviation from the default pose."""
    asset = _asset(env, asset_cfg)
    read_plan = getattr(env.scene, "_tensor_read_plan", None)
    if read_plan is None:
        position = asset.data.joint_pos[:, asset_cfg.joint_ids]
        default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    else:
        state = read_plan.joint_tensor_view(asset)
        position = state.joint_pos[:, asset_cfg.joint_ids]
        default = asset.data.default_joint_pos_torch(env.device)[:, asset_cfg.joint_ids]
    weights = _weights("weighted_pose", "pose_weights", pose_weights)
    if weights.shape[0] != position.shape[1]:
        raise ValueError(
            f"weighted_pose pose_weights length {weights.shape[0]} does not match "
            f"joint count {position.shape[1]}"
        )
    if isinstance(position, torch.Tensor):
        weights_torch = torch.as_tensor(
            weights, device=position.device, dtype=position.dtype
        ).expand_as(position)
        diff = position - default
    else:
        diff = _state("weighted_pose", "joint position", position, position.shape) - _state(
            "weighted_pose", "default joint position", default, position.shape
        )
        weights_torch = weights
    return _values(torch_or_np_sum(weights_torch * _square(diff), axis=1))


def torch_or_np_sum(value: np.ndarray | torch.Tensor, *, axis: int) -> np.ndarray | torch.Tensor:
    if isinstance(value, torch.Tensor):
        return torch.sum(value, dim=axis)
    return np.sum(value, axis=axis)


def upper_body_pose(
    env: ManagerBasedRlEnv,
    pose_weights: Any,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray | torch.Tensor:
    """Weighted L2 pose penalty with the twelve leg joints zeroed out."""
    weights = _weights("upper_body_pose", "pose_weights", pose_weights)
    if weights.shape[0] < 12:
        raise ValueError("upper_body_pose pose_weights must cover at least the twelve leg joints")
    weights = weights.copy()
    weights[:12] = 0.0
    return weighted_pose(env, weights, asset_cfg=asset_cfg)


# ---------------------------------------------------------------------------
# Velocity command term (legacy dead zone + standing zeroing semantics)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class G1VelocityCommandCfg(UniformVelocityCommandCfg):
    """G1 velocity command with the legacy planar-norm dead zone."""

    planar_dead_zone: float = 0.2

    def build(self, env: ManagerBasedRlEnv) -> G1VelocityCommand:
        return G1VelocityCommand(self, env)


class G1VelocityCommand(UniformVelocityCommand):
    cfg: G1VelocityCommandCfg  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(self, cfg: G1VelocityCommandCfg, env: ManagerBasedRlEnv):
        self._planar_dead_zone = _real(
            "G1VelocityCommand", "planar_dead_zone", cfg.planar_dead_zone, minimum=0.0
        )
        if cfg.heading_command:
            raise NotImplementedError(
                "G1VelocityCommand capability 'heading command' is unavailable in the "
                "Manager-Based runtime; the legacy G1 heading-channel zeroing has no "
                "production owner and fails closed instead of falling back"
            )
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: np.ndarray) -> None:
        super()._resample_command(env_ids)
        planar = self.vel_command_b[env_ids, :2]
        moving = np.linalg.norm(planar, axis=1) > self._planar_dead_zone
        self.vel_command_b[env_ids, :2] = planar * moving[:, None]


# ---------------------------------------------------------------------------
# Penalty curriculum (ports the legacy G1 penalty-curriculum semantics)
# ---------------------------------------------------------------------------


class G1PenaltyCurriculum(ManagerTermBase):
    """Scale negative-weight reward terms by average episode length.

    Ports the legacy G1 penalty curriculum: penalty weights start at
    ``initial_scale`` of their configured value and relax toward ``max_scale``
    as the tracked average episode length crosses the configured thresholds.
    """

    _allowed_params: ClassVar[frozenset[str]] = frozenset(
        {
            "initial_scale",
            "min_scale",
            "max_scale",
            "level_down_threshold",
            "level_up_threshold",
            "degree",
        }
    )

    def __init__(self, cfg: ManagerTermBaseCfg, env: _G1Env):
        super().__init__(env)
        unexpected = set(cfg.params) - self._allowed_params
        if unexpected:
            raise TypeError(f"{self.name} received unsupported parameters: {sorted(unexpected)}")
        self._min_scale = _real(
            self.name, "min_scale", cfg.params.get("min_scale", 0.5), minimum=0.0
        )
        self._max_scale = _real(
            self.name,
            "max_scale",
            cfg.params.get("max_scale", 1.0),
            minimum=self._min_scale,
        )
        self._current_scale = _real(
            self.name,
            "initial_scale",
            cfg.params.get("initial_scale", 0.5),
            minimum=self._min_scale,
        )
        if self._current_scale > self._max_scale:
            raise ValueError(f"{self.name} initial_scale must be at most max_scale")
        self._level_down_threshold = _real(
            self.name,
            "level_down_threshold",
            cfg.params.get("level_down_threshold", 150.0),
            minimum=0.0,
        )
        self._level_up_threshold = _real(
            self.name,
            "level_up_threshold",
            cfg.params.get("level_up_threshold", 750.0),
            minimum=0.0,
        )
        self._degree = _real(self.name, "degree", cfg.params.get("degree", 0.001), minimum=0.0)
        self._tracker = EpisodeLengthTracker(env.num_envs)
        self._original_weights: dict[str, float] = {}
        for name in env.reward_manager.active_terms:
            weight = float(env.reward_manager.get_term_cfg(name).weight)
            if weight < 0.0:
                self._original_weights[name] = weight
        self._apply_scale()

    def _apply_scale(self) -> None:
        env = cast("_G1Env", self._env)
        for name, original in self._original_weights.items():
            env.reward_manager.get_term_cfg(name).weight = original * self._current_scale

    def __call__(
        self,
        env: _G1Env,
        env_ids: np.ndarray | slice | None,
        **params: Any,
    ) -> dict[str, float]:
        del params
        ids = (
            np.arange(env.num_envs, dtype=np.intp)
            if env_ids is None
            else np.arange(env.num_envs, dtype=np.intp)[env_ids]
            if isinstance(env_ids, slice)
            else np.asarray(env_ids, dtype=np.intp).reshape(-1)
        )
        done_ids = ids[env.reset_buf[ids]]
        if len(done_ids) > 0:
            self._tracker.update(env.episode_length_buf[done_ids].astype(np.float64))
            average = self._tracker.average_length
            if average < self._level_down_threshold:
                self._current_scale *= 1.0 - self._degree
            elif average > self._level_up_threshold:
                self._current_scale *= 1.0 + self._degree
            self._current_scale = float(
                np.clip(self._current_scale, self._min_scale, self._max_scale)
            )
            self._apply_scale()
        return {
            "average_episode_length": float(self._tracker.average_length),
            "penalty_scale": float(self._current_scale),
        }


# ---------------------------------------------------------------------------
# G1 Manager-Based env: Registry-owned production runtime on the single
# lifecycle
# ---------------------------------------------------------------------------


class G1WalkManagerBasedEnv(_ConcreteManagerBasedRlEnv):
    """Manager-Based G1 walk runtime."""


def make_g1_walk_env(
    cfg: ManagerBasedRlEnvCfg,
    num_envs: int = 1,
    backend_type: str = "mujoco",
) -> G1WalkManagerBasedEnv:
    """Construct the Registry-owned G1 Manager-Based production runtime."""
    if not isinstance(cfg, ManagerBasedRlEnvCfg):
        raise TypeError(
            f"make_g1_walk_env expected ManagerBasedRlEnvCfg, received {type(cfg).__name__}"
        )
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError(f"make_g1_walk_env num_envs must be a positive integer, got {num_envs!r}")
    if not isinstance(backend_type, str) or not backend_type:
        raise ValueError(
            f"make_g1_walk_env backend_type must be a non-empty string, got {backend_type!r}"
        )

    cfg.validate()
    assert cfg.scene is not None
    base_name, body_state_requested, tracked_body_names = _resolve_backend_entity_contract(cfg)
    backend_kwargs = env_backend_kwargs(cfg)
    backend_kwargs["base_name"] = base_name
    if backend_type == "mujoco" and tracked_body_names is not None:
        backend_kwargs["tracked_body_names"] = tracked_body_names

    backend = create_backend(
        backend_type,
        cfg.scene,
        num_envs,
        cfg.sim_dt,
        body_state_required=body_state_requested,
        **backend_kwargs,
    )
    try:
        return G1WalkManagerBasedEnv(cfg, backend, num_envs)
    except Exception:
        backend.cleanup_scene_assets()
        raise


__all__ = [
    "G1GaitPhase",
    "G1PenaltyCurriculum",
    "G1VelocityCommand",
    "G1VelocityCommandCfg",
    "G1WalkManagerBasedEnv",
    "compute_feet_phase_contact_targets",
    "compute_feet_phase_height_targets",
    "feet_air_time",
    "feet_double_stance",
    "feet_phase",
    "feet_phase_contact",
    "feet_phase_contrast",
    "forward_progress",
    "g1_tilt_exceeded",
    "make_g1_walk_env",
    "penalty_close_feet_xy",
    "penalty_feet_ori",
    "under_speed",
    "upper_body_pose",
    "weighted_pose",
]
