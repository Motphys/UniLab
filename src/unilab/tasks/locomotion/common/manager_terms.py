"""Shared Manager-Based terms and term bases for the locomotion families.

The equations come from UniLab's quadruped joystick tasks and are reused
by quadruped and biped owners alike.  The adaptation uses community
``func + params`` terms, Torch state carriers, and packed named-sensor reads.
Reward terms that read named XML sensors live in ``sensor_reward_terms.py``
and build on the ``SensorTermBase`` cold-path binding contract defined here.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast

import torch

from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from unilab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers.types import ManagerBasedRlEnv, ManagerSensorView

    class _GaitEnv(ManagerBasedRlEnv, Protocol):
        @property
        def common_step_counter(self) -> int: ...


_OFFSETS = (0.0, 0.5, 0.5, 0.0)
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def require_real(
    term: str,
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{term} {name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{term} {name} must be finite")
    if minimum is not None and (result <= minimum if strict_minimum else result < minimum):
        relation = "greater than" if strict_minimum else "at least"
        raise ValueError(f"{term} {name} must be {relation} {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{term} {name} must be at most {maximum}")
    return result


def _offsets(term: str, value: Any) -> torch.Tensor:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{term} phase_offsets must be a sequence of four real numbers")
    items = list(value)
    if len(items) != 4:
        raise ValueError(f"{term} phase_offsets must contain 4 values, got {len(items)}")
    return torch.tensor(
        [require_real(term, f"phase_offsets[{index}]", item) for index, item in enumerate(items)],
        dtype=torch.float32,
    )


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


def require_command(env: ManagerBasedRlEnv, term: str, command_name: str) -> torch.Tensor:
    if not isinstance(command_name, str) or not command_name:
        raise ValueError(f"{term} command_name must be a non-empty string")
    try:
        command = env.command_manager.get_command(command_name)
    except KeyError as exc:
        raise KeyError(f"{term} command capability '{command_name}' is unavailable") from exc
    if command is None:
        raise KeyError(f"{term} command capability '{command_name}' is unavailable")
    return _torch_state(command)


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

    def reset(self, env_ids: torch.Tensor) -> None:
        """Row boundary for stateless sensor terms."""
        del env_ids


class base_height_l2(ManagerTermBase):
    """Penalize world-frame root height error for the flat-ground pilot."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self._target_height = require_real(
            self.name,
            "target_height",
            cfg.params["target_height"],
        )
        asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(f"{self.name} asset_cfg must be a SceneEntityCfg")
        self._entity_name = asset_cfg.name
        entity_cfg = env.cfg.scene.entities.get(self._entity_name) if env.cfg.scene else None
        root = getattr(entity_cfg, "root_body_name", None) if entity_cfg is not None else None
        if not isinstance(root, str) or not root:
            raise ValueError(f"{self.name} requires an entity root_body_name")
        self._root_body_name = root

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return (self._root_body_name,)

    @property
    def entity_name(self) -> str:
        return self._entity_name

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        target_height: float,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        del target_height, asset_cfg
        read_plan = getattr(env.scene, "read_plan", None)
        if read_plan is None:
            raise NotImplementedError(
                f"{self.name} requires the Manager packed root-body read plan"
            )
        body = read_plan.body_tensor_view(self._entity_name, (self._root_body_name,))
        return (body.pos_w[:, 0, 2] - self._target_height).square()


def alive(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Constant reward for every step, unconditional as in the legacy tasks."""
    device = torch.device(env.device)
    return torch.ones((env.num_envs,), dtype=torch.float32, device=device)


class joint_deviation_l1(ManagerTermBase):
    """Penalize selected joint displacement from the default pose with an L1 kernel."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(f"{self.name} asset_cfg must be a SceneEntityCfg")
        self._entity_name = asset_cfg.name
        self._joint_ids = asset_cfg.joint_ids

    @property
    def tensor_state_entity(self) -> str:
        return self._entity_name

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str = "twist",
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        del asset_cfg
        read_plan = getattr(env.scene, "read_plan", None)
        if read_plan is None:
            raise NotImplementedError(
                f"{self.name} requires the Manager packed joint-state read plan"
            )
        position = read_plan.joint_tensor_view(self._entity_name).joint_pos[self._joint_ids]
        default = env.scene[self._entity_name].data.default_joint_pos(env.device)[
            :, self._joint_ids
        ]
        return (position - default).abs().sum(dim=1)


class stand_still_l1(joint_deviation_l1):
    """Penalize selected joint deviation only below the commanded-motion threshold."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._command_name = cfg.params["command_name"]
        if not isinstance(self._command_name, str) or not self._command_name:
            raise ValueError(f"{self.name} command_name must be a non-empty string")
        self._command_threshold = require_real(
            self.name,
            "command_threshold",
            cfg.params.get("command_threshold", 0.1),
            minimum=0.0,
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str = "twist",
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        del command_name, command_threshold, asset_cfg
        command = require_command(env, self.name, self._command_name)
        if not isinstance(command, torch.Tensor):
            command = _torch_state(command)
        stopped = torch.linalg.vector_norm(command, dim=1) < self._command_threshold
        return super().__call__(env) * stopped.to(dtype=torch.float32)


class _GaitTerm(ManagerTermBase):
    _allowed_params: ClassVar[frozenset[str]] = frozenset(
        {"frequency", "phase_offsets", "command_name", "command_threshold"}
    )

    def __init__(self, cfg: ManagerTermBaseCfg, env: _GaitEnv):
        super().__init__(env)
        unexpected = set(cfg.params) - self._allowed_params
        if unexpected:
            raise TypeError(f"{self.name} received unsupported parameters: {sorted(unexpected)}")
        self._frequency = require_real(
            self.name, "frequency", cfg.params.get("frequency", 2.0), minimum=0.0
        )
        self._device = torch.device(env.device)
        self._offsets = _offsets(self.name, cfg.params.get("phase_offsets", _OFFSETS)).to(
            device=self._device
        )
        self._step_dt = require_real(
            self.name, "step_dt", env.step_dt, minimum=0.0, strict_minimum=True
        )
        command_name = cfg.params.get("command_name")
        if command_name is not None and (not isinstance(command_name, str) or not command_name):
            raise ValueError(f"{self.name} command_name must be a non-empty string or None")
        self._command_name = command_name
        self._command_threshold = require_real(
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
        if isinstance(counter, bool) or not isinstance(counter, int):
            raise TypeError(f"{self.name} common_step_counter must be an integer")
        if counter < 0:
            raise ValueError(f"{self.name} common_step_counter must be non-negative")
        return int(counter)

    def _moving(self, env: _GaitEnv) -> torch.Tensor:
        if self._command_name is None:
            return torch.ones(env.num_envs, dtype=torch.bool, device=self._device)
        command = torch.as_tensor(
            require_command(env, self.name, self._command_name), device=self._device
        )
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
        read_plan = getattr(self._env.scene, "read_plan", None)
        sensor_names = getattr(read_plan, "sensor_names", {}).get("robot", ())
        if read_plan is None or not set(self._sensor_names).issubset(set(sensor_names)):
            raise NotImplementedError(
                f"Manager term '{self.name}' requires its named sensors "
                f"{self._sensor_names} in the packed Torch read plan"
            )
        values = read_plan.sensor_tensor_views("robot", self._sensor_names).values
        return torch.cat([values[name] for name in self._sensor_names], dim=1)


class feet_phase_contact(_FootSensorTerm):
    """Reward foot contact matching the configured stance portion of gait phase."""

    _allowed_params = _GaitTerm._allowed_params | {
        "sensor_names",
        "contact_threshold",
        "stance_threshold",
    }

    def __init__(self, cfg: ManagerTermBaseCfg, env: _GaitEnv):
        super().__init__(cfg, env)
        self._contact_threshold = require_real(
            self.name, "contact_threshold", cfg.params.get("contact_threshold", 0.1), minimum=0.0
        )
        self._stance_threshold = require_real(
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
        starts = torch.cumsum(
            torch.tensor((0, *self._view.dimensions[:-1]), dtype=torch.int64), dim=0
        )
        # MuJoCo ``<contact data="force">`` sensors report the force in the
        # contact frame whose first axis is the contact normal, so the gating
        # component is column 0 for both 1-D ``found`` and 3-D ``force``.
        # ``reduce="netforce"`` variants report world-frame force instead and
        # are outside this contract.
        self._columns = starts

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        return self._sensor_names

    def _read_contact(self) -> torch.Tensor:
        read_plan = getattr(self._env.scene, "read_plan", None)
        sensor_names = getattr(read_plan, "sensor_names", {}).get("robot", ())
        if read_plan is None or not set(self._sensor_names).issubset(set(sensor_names)):
            raise NotImplementedError(
                f"Manager term '{self.name}' requires its named sensors "
                f"{self._sensor_names} in the packed Torch read plan"
            )
        values = read_plan.sensor_tensor_views("robot", self._sensor_names).values
        return torch.cat([values[name] for name in self._sensor_names], dim=1)

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
        self._target = require_real(
            self.name, "target_height", cfg.params.get("target_height", 0.1), minimum=0.0
        )
        self._kernel = require_real(
            self.name,
            "kernel",
            cfg.params.get("kernel", 0.01),
            minimum=0.0,
            strict_minimum=True,
        )
        self._swing_start = require_real(
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
        self._sensor_names = sensor_names
        command_name = cfg.params.get("command_name")
        if not isinstance(command_name, str) or not command_name:
            raise ValueError(f"{self.name} command_name must be a non-empty string")
        self._command_name = command_name
        self._command_threshold = require_real(
            self.name,
            "command_threshold",
            cfg.params.get("command_threshold", 0.1),
            minimum=0.0,
        )
        self._contact_threshold = require_real(
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
        starts = torch.cumsum(
            torch.tensor((0, *self._view.dimensions[:-1]), dtype=torch.int64), dim=0
        )
        # MuJoCo ``<contact data="force">`` sensors report the force in the
        # contact frame whose first axis is the contact normal, so the gating
        # component is column 0 for both 1-D ``found`` and 3-D ``force``.
        # ``reduce="netforce"`` variants report world-frame force instead and
        # are outside this contract.
        self._columns = starts

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        return self._sensor_names

    def _read_contact(self) -> torch.Tensor:
        read_plan = getattr(self._env.scene, "read_plan", None)
        sensor_names = getattr(read_plan, "sensor_names", {}).get("robot", ())
        if read_plan is None or not set(self._sensor_names).issubset(set(sensor_names)):
            raise NotImplementedError(
                f"Manager term '{self.name}' requires its named sensors "
                f"{self._sensor_names} in the packed Torch read plan"
            )
        values = read_plan.sensor_tensor_views("robot", self._sensor_names).values
        return torch.cat([values[name] for name in self._sensor_names], dim=1)

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        contact = self._read_contact()[:, self._columns] > self._contact_threshold
        standing = (
            torch.linalg.vector_norm(_command_tensor(env, self.name, self._command_name), dim=1)
            <= self._command_threshold
        )
        air = contact.logical_not().sum(dim=1)
        return (air * standing).to(dtype=torch.float32)


def _command_tensor(
    env: ManagerBasedRlEnv,
    term: str,
    command_name: str,
) -> torch.Tensor:
    return require_command(env, term, command_name).to(dtype=torch.float32)


def _torch_state(value: torch.Tensor, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return value.to(dtype=dtype)


__all__ = [
    "SensorTermBase",
    "alive",
    "base_height_l2",
    "feet_air_while_standing",
    "feet_phase_contact",
    "feet_phase_swing_height",
    "joint_deviation_l1",
    "quadruped_gait_phase",
    "stand_still_l1",
]
