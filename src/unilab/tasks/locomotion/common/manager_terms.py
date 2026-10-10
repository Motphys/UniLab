"""Shared Manager-Based terms and term bases for the locomotion families.

The equations come from UniLab's quadruped joystick tasks and are reused
by quadruped and biped owners alike.  The adaptation uses community
``func + params`` terms, NumPy, and the base-owned sensor facade.  Reward terms
that read named XML sensors live in ``sensor_reward_terms.py`` and build on the
``SensorTermBase`` cold-path binding contract defined here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast

import numpy as np
import torch

from unilab.dtype_config import get_global_dtype
from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from unilab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv, ManagerSensorView

    class _GaitEnv(ManagerBasedRlEnv, Protocol):
        @property
        def common_step_counter(self) -> int: ...


_OFFSETS = (0.0, 0.5, 0.5, 0.0)
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _real(
    term: str,
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
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
    if maximum is not None and result > maximum:
        raise ValueError(f"{term} {name} must be at most {maximum}")
    return result


def _offsets(term: str, value: Any) -> np.ndarray:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list, np.ndarray)):
        raise TypeError(f"{term} phase_offsets must be a sequence of four real numbers")
    if isinstance(value, np.ndarray) and value.ndim != 1:
        raise ValueError(f"{term} phase_offsets must be one-dimensional, got {value.shape}")
    items = list(value)
    if len(items) != 4:
        raise ValueError(f"{term} phase_offsets must contain 4 values, got {len(items)}")
    result = np.asarray(
        [_real(term, f"phase_offsets[{index}]", item) for index, item in enumerate(items)],
        dtype=get_global_dtype(),
    )
    result.setflags(write=False)
    return result


def _names(term: str, value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{term} sensor_names must be a sequence of four strings")
    names = tuple(value)
    if len(names) != 4:
        raise ValueError(f"{term} sensor_names must contain 4 names, got {len(names)}")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{term} sensor_names must contain non-empty strings")
    if len(set(names)) != 4:
        raise ValueError(f"{term} sensor_names must be unique: {names}")
    return names


def _state(term: str, capability: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{term} {capability} must be an np.ndarray")
    if value.shape != shape:
        raise ValueError(f"{term} {capability} must have shape {shape}, got {value.shape}")
    if not np.isfinite(value).all():
        env_ids = np.flatnonzero(~np.isfinite(value).reshape(shape[0], -1).all(axis=1)).tolist()
        raise ValueError(f"{term} {capability} contains NaN or Inf for environments {env_ids[:10]}")
    return value


def _command(env: ManagerBasedRlEnv, term: str, command_name: str) -> np.ndarray | torch.Tensor:
    if not isinstance(command_name, str) or not command_name:
        raise ValueError(f"{term} command_name must be a non-empty string")
    try:
        command = env.command_manager.get_command(command_name)
    except KeyError as exc:
        raise KeyError(f"{term} command capability '{command_name}' is unavailable") from exc
    if command is None:
        raise KeyError(f"{term} command capability '{command_name}' is unavailable")
    if isinstance(command, torch.Tensor):
        return command
    return _state(term, f"command '{command_name}'", command, (env.num_envs, 3))


def _asset(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> Entity:
    return cast("Entity", env.scene[asset_cfg.name])


class SensorTermBase(ManagerTermBase):
    """Cold-path named-sensor binding shared by locomotion manager terms."""

    _allowed_params: ClassVar[frozenset[str]] = frozenset()

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        return ()

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        unexpected = set(cfg.params) - self._allowed_params
        if unexpected:
            raise TypeError(f"{self.name} received unsupported parameters: {sorted(unexpected)}")

    def _bind(self, sensor_names: tuple[str, ...]) -> ManagerSensorView:
        try:
            return self._env.scene.bind_sensor_data(sensor_names)
        except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise type(exc)(
                f"Manager term '{self.name}' named-sensor capability could not be "
                f"materialized for {sensor_names}: {exc}"
            ) from exc

    @staticmethod
    def _read(view: ManagerSensorView, term: str) -> np.ndarray:
        try:
            return view.read()
        except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise type(exc)(
                f"Manager term '{term}' named-sensor capability failed on "
                f"backend '{view.backend_type}': {exc}"
            ) from exc


def track_lin_vel_xy_exp(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray | torch.Tensor:
    """Track commanded planar velocity with the legacy independent exponential kernel."""
    scale = _real("track_lin_vel_xy_exp", "std", std, minimum=0.0, strict_minimum=True)
    actual = _state(
        "track_lin_vel_xy_exp",
        "root linear velocity",
        _asset(env, asset_cfg).data.root_link_lin_vel_b,
        (env.num_envs, 3),
    )
    command = _command(env, "track_lin_vel_xy_exp", command_name)
    if isinstance(command, torch.Tensor):
        actual_t = (
            actual
            if isinstance(actual, torch.Tensor)
            else torch.as_tensor(actual, device=command.device, dtype=command.dtype)
        )
        actual_tensor = cast(torch.Tensor, actual_t)
        delta_t = command[:, :2] - actual_tensor[:, :2]
        error = torch.sum(delta_t.square(), dim=1)
    else:
        error = np.sum(np.square(command[:, :2] - actual[:, :2]), axis=1)
    return _exp_scaled(error, scale)


def track_ang_vel_z_exp(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray | torch.Tensor:
    """Track commanded yaw velocity without folding roll/pitch into the kernel."""
    scale = _real("track_ang_vel_z_exp", "std", std, minimum=0.0, strict_minimum=True)
    actual = _state(
        "track_ang_vel_z_exp",
        "root angular velocity",
        _asset(env, asset_cfg).data.root_link_ang_vel_b,
        (env.num_envs, 3),
    )
    command = _command(env, "track_ang_vel_z_exp", command_name)
    if isinstance(command, torch.Tensor):
        actual_t = (
            actual
            if isinstance(actual, torch.Tensor)
            else torch.as_tensor(actual, device=command.device, dtype=command.dtype)
        )
        actual_tensor = cast(torch.Tensor, actual_t)
        error = (command[:, 2] - actual_tensor[:, 2]).square()
    else:
        error = np.square(command[:, 2] - actual[:, 2])
    return _exp_scaled(error, scale)


def _exp_scaled(error: np.ndarray | torch.Tensor, scale: float):
    if isinstance(error, torch.Tensor):
        return torch.exp(-error / (scale * scale))
    return torch.as_tensor(np.asarray(np.exp(-error / (scale * scale)), dtype=get_global_dtype()))


def lin_vel_z_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize vertical root velocity independently from planar tracking."""
    velocity = _state(
        "lin_vel_z_l2",
        "root linear velocity",
        _asset(env, asset_cfg).data.root_link_lin_vel_b,
        (env.num_envs, 3),
    )
    return torch.as_tensor(np.asarray(np.square(velocity[:, 2]), dtype=get_global_dtype()))


def ang_vel_xy_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize root roll/pitch angular velocity independently from yaw tracking."""
    velocity = _state(
        "ang_vel_xy_l2",
        "root angular velocity",
        _asset(env, asset_cfg).data.root_link_ang_vel_b,
        (env.num_envs, 3),
    )
    return torch.as_tensor(
        np.asarray(np.sum(np.square(velocity[:, :2]), axis=1), dtype=get_global_dtype())
    )


def base_height_l2(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize world-frame root height error for the flat-ground pilot."""
    target = _real("base_height_l2", "target_height", target_height)
    position = _state(
        "base_height_l2",
        "root position",
        _asset(env, asset_cfg).data.root_link_pos_w,
        (env.num_envs, 3),
    )
    return torch.as_tensor(np.asarray(np.square(position[:, 2] - target), dtype=get_global_dtype()))


def alive(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Constant reward for every step, unconditional as in the legacy tasks."""
    return torch.ones((env.num_envs,), dtype=torch.float32)


def joint_deviation_l1(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize selected joint displacement from the default pose with an L1 kernel."""
    asset = _asset(env, asset_cfg)
    position = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    if (
        not isinstance(position, np.ndarray)
        or position.ndim != 2
        or position.shape[0] != env.num_envs
    ):
        shape = getattr(position, "shape", None)
        raise ValueError(
            f"joint_deviation_l1 joint position must be 2-D with leading dimension {env.num_envs}, got {shape}"
        )
    default = _state("joint_deviation_l1", "default joint position", default, position.shape)
    position = _state("joint_deviation_l1", "joint position", position, position.shape)
    return torch.as_tensor(np.asarray(np.sum(np.abs(position - default), axis=1)))


def stand_still_l1(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize selected joint deviation only below the commanded-motion threshold."""
    threshold = _real(
        "stand_still_l1",
        "command_threshold",
        command_threshold,
        minimum=0.0,
    )
    stopped = np.linalg.norm(_command(env, "stand_still_l1", command_name), axis=1) < threshold
    return joint_deviation_l1(env, asset_cfg=asset_cfg) * torch.as_tensor(
        stopped, dtype=torch.float32
    )


class _GaitTerm(ManagerTermBase):
    _allowed_params: ClassVar[frozenset[str]] = frozenset(
        {"frequency", "phase_offsets", "command_name", "command_threshold"}
    )

    def __init__(self, cfg: ManagerTermBaseCfg, env: _GaitEnv):
        super().__init__(env)
        unexpected = set(cfg.params) - self._allowed_params
        if unexpected:
            raise TypeError(f"{self.name} received unsupported parameters: {sorted(unexpected)}")
        self._frequency = _real(
            self.name, "frequency", cfg.params.get("frequency", 2.0), minimum=0.0
        )
        self._device = torch.device(getattr(env, "device", "cpu"))
        self._offsets = torch.tensor(
            _offsets(self.name, cfg.params.get("phase_offsets", _OFFSETS)),
            dtype=torch.float32,
            device=self._device,
        )
        self._step_dt = _real(self.name, "step_dt", env.step_dt, minimum=0.0, strict_minimum=True)
        command_name = cfg.params.get("command_name")
        if command_name is not None and (not isinstance(command_name, str) or not command_name):
            raise ValueError(f"{self.name} command_name must be a non-empty string or None")
        self._command_name = command_name
        self._command_threshold = _real(
            self.name,
            "command_threshold",
            cfg.params.get("command_threshold", 0.0),
            minimum=0.0,
        )
        self._phase_value = torch.zeros(env.num_envs, dtype=torch.float32, device=self._device)
        self._last_counter = 0
        self._advance_to(env, self._counter(env))

    def _counter(self, env: _GaitEnv) -> int:
        counter = env.common_step_counter
        if isinstance(counter, (bool, np.bool_)) or not isinstance(counter, (int, np.integer)):
            raise TypeError(f"{self.name} common_step_counter must be an integer")
        if counter < 0:
            raise ValueError(f"{self.name} common_step_counter must be non-negative")
        return int(counter)

    def _moving(self, env: _GaitEnv) -> torch.Tensor:
        if self._command_name is None:
            return torch.ones(env.num_envs, dtype=torch.bool, device=self._device)
        command = torch.as_tensor(_command(env, self.name, self._command_name), device=self._device)
        return torch.linalg.vector_norm(command, dim=1) > self._command_threshold

    def _advance_to(self, env: _GaitEnv, counter: int) -> None:
        delta = counter - self._last_counter
        if delta < 0:
            raise ValueError(f"{self.name} common_step_counter cannot move backwards")
        increment = self._step_dt * self._frequency
        if delta == 1:  # Hot path: preserve the legacy float32 iterative phase exactly.
            self._phase_value = torch.fmod(self._phase_value + increment * self._moving(env), 1.0)
        elif delta > 1:  # Cold catch-up for a term constructed or inspected between steps.
            for _ in range(delta):
                self._phase_value = torch.fmod(
                    self._phase_value + increment * self._moving(env),
                    1.0,
                )
        self._last_counter = counter

    def _phase(self, env: _GaitEnv) -> torch.Tensor:
        self._advance_to(env, self._counter(env))
        return torch.remainder(self._phase_value[:, None] + self._offsets[None, :], 1.0)


class quadruped_gait_phase(_GaitTerm):
    """Four-foot phase observation with the legacy diagonal-trot ordering."""

    def __call__(self, env: _GaitEnv, **params: Any) -> torch.Tensor:
        del params
        return self._phase(env)


class _FootSensorTerm(_GaitTerm):
    def __init__(self, cfg: ManagerTermBaseCfg, env: _GaitEnv):
        super().__init__(cfg, env)
        sensor_names = _names(self.name, cfg.params.get("sensor_names"))
        self._sensor_names = sensor_names
        try:
            self._view = env.scene.bind_sensor_data(sensor_names)
        except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise type(exc)(
                f"Manager term '{self.name}' named-foot-sensor capability could not be "
                f"materialized for {sensor_names}: {exc}"
            ) from exc

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        return self._sensor_names

    def _read(self) -> torch.Tensor:
        read_plan = getattr(self._env.scene, "_tensor_read_plan", None)
        if read_plan is not None:
            values = read_plan.sensor_tensor_views("robot", self._sensor_names).values
            return torch.cat([values[name] for name in self._sensor_names], dim=1)
        try:
            return torch.as_tensor(self._view.read(), device=self._device)
        except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise type(exc)(
                f"Manager term '{self.name}' named-foot-sensor capability failed on "
                f"backend '{self._view.backend_type}': {exc}"
            ) from exc


class feet_phase_contact(_FootSensorTerm):
    """Reward foot contact matching the configured stance portion of gait phase."""

    _allowed_params = _GaitTerm._allowed_params | {
        "sensor_names",
        "contact_threshold",
        "stance_threshold",
    }

    def __init__(self, cfg: ManagerTermBaseCfg, env: _GaitEnv):
        super().__init__(cfg, env)
        self._contact_threshold = _real(
            self.name, "contact_threshold", cfg.params.get("contact_threshold", 0.1), minimum=0.0
        )
        self._stance_threshold = _real(
            self.name,
            "stance_threshold",
            cfg.params.get("stance_threshold", 0.6),
            minimum=0.0,
            maximum=1.0,
        )
        if any(width not in (1, 3) for width in self._view.dimensions):
            raise ValueError(
                f"{self.name} contact sensors must each expose 1-D found or 3-D force; "
                f"received {self._view.dimensions} on backend '{self._view.backend_type}'"
            )
        starts = np.cumsum((0, *self._view.dimensions[:-1]), dtype=np.int64)
        # MuJoCo ``<contact data="force">`` sensors report the force in the
        # contact frame whose first axis is the contact normal, so the gating
        # component is column 0 for both 1-D ``found`` and 3-D ``force``.
        # ``reduce="netforce"`` variants report world-frame force instead and
        # are outside this contract.
        self._columns = starts

    def __call__(self, env: _GaitEnv, **params: Any) -> torch.Tensor:
        del params
        contact = self._read()[:, self._columns] > self._contact_threshold
        expected = self._phase(env) < self._stance_threshold
        if self._frequency < 1.0e-8:
            expected.fill_(True)
        elif self._command_name is not None:
            expected |= ~self._moving(env)[:, None]
        return (contact == expected).to(torch.float32).mean(dim=1)


class feet_phase_swing_height(_FootSensorTerm):
    """Reward foot height near a target during the configured swing phase."""

    _allowed_params = _GaitTerm._allowed_params | {
        "sensor_names",
        "target_height",
        "kernel",
        "swing_start",
    }

    def __init__(self, cfg: ManagerTermBaseCfg, env: _GaitEnv):
        super().__init__(cfg, env)
        self._target = _real(
            self.name, "target_height", cfg.params.get("target_height", 0.1), minimum=0.0
        )
        self._kernel = _real(
            self.name,
            "kernel",
            cfg.params.get("kernel", 0.01),
            minimum=0.0,
            strict_minimum=True,
        )
        self._swing_start = _real(
            self.name,
            "swing_start",
            cfg.params.get("swing_start", 0.6),
            minimum=0.0,
            maximum=1.0,
        )
        if self._view.dimensions != (3, 3, 3, 3):
            raise ValueError(
                f"{self.name} position sensors must each expose 3-D xyz; received "
                f"{self._view.dimensions} on backend '{self._view.backend_type}'"
            )

    def __call__(self, env: _GaitEnv, **params: Any) -> torch.Tensor:
        del params
        heights = self._read()[:, (2, 5, 8, 11)]
        swing = self._phase(env) >= self._swing_start
        if self._command_name is not None:
            swing &= self._moving(env)[:, None]
        reward = torch.exp(-(heights - self._target).square() / self._kernel) * swing
        return reward.mean(dim=1).to(torch.float32)


class feet_air_while_standing(ManagerTermBase):
    """Count feet without contact while the configured velocity command is standing."""

    _allowed_params = frozenset(
        {"sensor_names", "command_name", "command_threshold", "contact_threshold"}
    )

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        unexpected = set(cfg.params) - self._allowed_params
        if unexpected:
            raise TypeError(f"{self.name} received unsupported parameters: {sorted(unexpected)}")
        sensor_names = _names(self.name, cfg.params.get("sensor_names"))
        command_name = cfg.params.get("command_name")
        if not isinstance(command_name, str) or not command_name:
            raise ValueError(f"{self.name} command_name must be a non-empty string")
        self._command_name = command_name
        self._command_threshold = _real(
            self.name,
            "command_threshold",
            cfg.params.get("command_threshold", 0.1),
            minimum=0.0,
        )
        self._contact_threshold = _real(
            self.name,
            "contact_threshold",
            cfg.params.get("contact_threshold", 0.1),
            minimum=0.0,
        )
        try:
            self._view = env.scene.bind_sensor_data(sensor_names)
        except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise type(exc)(
                f"Manager term '{self.name}' named-foot-sensor capability could not be "
                f"materialized for {sensor_names}: {exc}"
            ) from exc
        if any(width not in (1, 3) for width in self._view.dimensions):
            raise ValueError(
                f"{self.name} contact sensors must each expose 1-D found or 3-D force; "
                f"received {self._view.dimensions} on backend '{self._view.backend_type}'"
            )
        starts = np.cumsum((0, *self._view.dimensions[:-1]), dtype=np.int64)
        # MuJoCo ``<contact data="force">`` sensors report the force in the
        # contact frame whose first axis is the contact normal, so the gating
        # component is column 0 for both 1-D ``found`` and 3-D ``force``.
        # ``reduce="netforce"`` variants report world-frame force instead and
        # are outside this contract.
        self._columns = starts

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        try:
            values = self._view.read()
        except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise type(exc)(
                f"Manager term '{self.name}' named-foot-sensor capability failed on "
                f"backend '{self._view.backend_type}': {exc}"
            ) from exc
        contact = values[:, self._columns] > self._contact_threshold
        standing = (
            np.linalg.norm(_command(env, self.name, self._command_name), axis=1)
            <= self._command_threshold
        )
        return torch.as_tensor(np.asarray(np.sum(~contact, axis=1) * standing), dtype=torch.float32)


__all__ = [
    "SensorTermBase",
    "alive",
    "ang_vel_xy_l2",
    "base_height_l2",
    "feet_air_while_standing",
    "feet_phase_contact",
    "feet_phase_swing_height",
    "joint_deviation_l1",
    "lin_vel_z_l2",
    "quadruped_gait_phase",
    "stand_still_l1",
    "track_ang_vel_z_exp",
    "track_lin_vel_xy_exp",
]
