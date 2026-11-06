# Derived from mujocolab/mjlab v1.6.0 (0fb8a681),
# src/mjlab/tasks/velocity/mdp/rewards.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for Torch, named-sensor contact groups, and the
# base-owned entity facade; Apache-2.0.
"""mjlab-aligned gait-quality terms shared by locomotion families.

These are the window-form ``feet_air_time`` and the clearance / swing-height /
slip / angular-momentum reward terms that mjlab ships in its velocity task,
plus the privileged per-foot observation terms (``foot_height`` /
``foot_air_time`` / ``foot_contact`` / ``foot_contact_forces``) its velocity
critic consumes.  mjlab reads feet through a ``ContactSensor`` (primaries with
per-slot any-reduce) and a ``TerrainHeightSensor``; the UniLab port binds one
named contact sensor group per foot through the ``SensorTermBase`` cold-path
contract (any sensor in the group above the threshold means foot contact,
mirroring the slot reduce) and reads foot heights/velocities from
``body_link_pos_w`` / ``body_link_lin_vel_w`` through ``asset_cfg.body_ids``,
which is exact on flat terrain without raycast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch

from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from unilab.managers.scene_entity_config import SceneEntityCfg

from .manager_terms import SensorTermBase, require_command, require_real

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers.types import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _torch_state(value: torch.Tensor, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return value.to(dtype=dtype)


def _sensor_groups(term: str, value: Any) -> tuple[tuple[str, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{term} sensor_groups must be a sequence of sensor-name sequences")
    groups: list[tuple[str, ...]] = []
    for index, group in enumerate(value):
        if isinstance(group, (str, bytes)) or not isinstance(group, (tuple, list)) or not group:
            raise ValueError(f"{term} sensor_groups[{index}] must be a non-empty name sequence")
        names = tuple(group)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError(f"{term} sensor_groups[{index}] must contain non-empty strings")
        groups.append(names)
    if not groups:
        raise ValueError(f"{term} sensor_groups must declare at least one foot")
    flat = [name for group in groups for name in group]
    if len(set(flat)) != len(flat):
        raise ValueError(f"{term} sensor_groups sensor names must be unique: {flat}")
    return tuple(groups)


def _command_gate(
    env: ManagerBasedRlEnv,
    term: str,
    command_name: str | None,
    command_threshold: float,
) -> torch.Tensor | None:
    """mjlab command gate: planar norm plus yaw magnitude above the threshold."""
    if command_name is None:
        return None
    command = require_command(env, term, command_name)
    command_tensor = _torch_state(command)
    total = torch.linalg.vector_norm(command_tensor[:, :2], dim=1) + torch.abs(command_tensor[:, 2])
    return total > command_threshold


def _command_name_param(term: str, value: Any, *, required: bool) -> str | None:
    if value is None:
        if required:
            raise ValueError(f"{term} command_name must be a non-empty string")
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{term} command_name must be a non-empty string")
    return value


def _selected_body_names(asset: Entity, asset_cfg: SceneEntityCfg) -> tuple[str, ...]:
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, list):
        return tuple(asset.body_names[index] for index in body_ids)
    return tuple(asset.body_names[body_ids])


def _feet_body_tensor_view(
    term: str,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    body_names: tuple[str, ...],
) -> Any:
    read_plan = getattr(env.scene, "read_plan", None)
    if read_plan is None:
        raise NotImplementedError(f"{term} requires the Manager packed foot-body read plan")
    entity = env.scene[asset_cfg.name]
    body = read_plan.body_tensor_view(entity, body_names)
    expected = (env.num_envs, len(body_names), 3)
    if tuple(body.pos_w.shape) != expected or tuple(body.lin_vel_w.shape) != expected:
        raise ValueError(f"{term} foot body state must have shape {expected}")
    return body


class _FootContactTerm(SensorTermBase):
    """Cold-path binding of per-foot contact sensor groups (mjlab slot reduce)."""

    _allowed_params: ClassVar[frozenset[str]] = frozenset({"sensor_groups", "contact_threshold"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._groups = _sensor_groups(self.name, cfg.params.get("sensor_groups"))
        self._contact_threshold = require_real(
            self.name, "contact_threshold", cfg.params.get("contact_threshold", 0.0), minimum=0.0
        )
        flat = tuple(name for group in self._groups for name in group)
        self._view = self._bind(flat)
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
        self._flat_width = int(sum(self._view.dimensions))
        # Per-foot column groups in the flattened sensor layout.
        self._columns: list[torch.Tensor] = []
        device = torch.device(env.device)
        for group in self._groups:
            self._columns.append(starts[: len(group)].to(device=device))
            starts = starts[len(group) :]
        if len({len(group) for group in self._groups}) == 1:
            self._group_columns = torch.stack(self._columns, dim=0)
        else:
            self._group_columns = None

    @property
    def num_feet(self) -> int:
        return len(self._groups)

    @property
    def entity_name(self) -> str:
        return getattr(self, "_entity_name", "robot")

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        flat = tuple(name for group in self._groups for name in group)
        if not flat:
            return ()
        return flat

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return getattr(self, "_body_names", ())

    def _contact_values(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        read_plan = getattr(env.scene, "read_plan", None)
        names = self.tensor_sensor_names
        if read_plan is None or not set(names).issubset(
            set(read_plan.sensor_names.get("robot", ()))
        ):
            raise NotImplementedError(
                f"Manager term '{self.name}' requires its named sensors "
                f"{names} in the packed Torch read plan"
            )
        entity = env.scene["robot"]
        values = read_plan.sensor_tensor_views(entity, names).values
        ordered = torch.cat([values[name] for name in names], dim=1)
        if tuple(ordered.shape) != (env.num_envs, self._flat_width):
            raise ValueError(
                f"{self.name} foot contact sensors must have shape "
                f"{(env.num_envs, self._flat_width)}; got {tuple(ordered.shape)}"
            )
        return ordered > self._contact_threshold

    def _contact(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        return self._contact_feet(env)

    def _contact_feet(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        contact = self._contact_values(env)
        device = contact.device
        if self._group_columns is not None:
            return contact[:, self._group_columns.to(device)].any(dim=2)
        device_columns = [columns.to(device=device) for columns in self._columns]
        feet = [contact[:, columns].any(dim=1) for columns in device_columns]
        return torch.stack(feet, dim=1)


class feet_air_time(_FootContactTerm):
    """Count feet whose current air time sits inside the rewarded window.

    Mirrors mjlab ``feet_air_time``: air time accumulates ``env.step_dt`` while
    the foot has no contact and resets to zero on contact; a foot scores while
    ``threshold_min < air_time < threshold_max``.  When ``command_name`` is set
    the count is gated by the command gate.
    """

    _allowed_params = _FootContactTerm._allowed_params | {
        "threshold_min",
        "threshold_max",
        "command_name",
        "command_threshold",
    }

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._threshold_min = require_real(
            self.name, "threshold_min", cfg.params.get("threshold_min", 0.05), minimum=0.0
        )
        self._threshold_max = require_real(
            self.name,
            "threshold_max",
            cfg.params.get("threshold_max", 0.5),
            minimum=0.0,
            strict_minimum=True,
        )
        if self._threshold_min >= self._threshold_max:
            raise ValueError(
                f"{self.name} threshold_min must be below threshold_max, got "
                f"{self._threshold_min} >= {self._threshold_max}"
            )
        self._command_name = _command_name_param(
            self.name, cfg.params.get("command_name"), required=False
        )
        self._command_threshold = require_real(
            self.name, "command_threshold", cfg.params.get("command_threshold", 0.5), minimum=0.0
        )
        self._air_time = torch.zeros(
            (env.num_envs, self.num_feet), dtype=torch.float32, device=env.device
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        if not isinstance(env_ids, torch.Tensor):
            raise TypeError(f"{self.name} reset env_ids must be a torch.Tensor")
        self._air_time[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        contact = self._contact_feet(env)
        step_dt = require_real(self.name, "step_dt", env.step_dt, minimum=0.0, strict_minimum=True)
        self._air_time = torch.where(
            contact, torch.zeros_like(self._air_time), self._air_time + step_dt
        )
        reward = torch.sum(
            (self._air_time > self._threshold_min) & (self._air_time < self._threshold_max),
            dim=1,
        ).to(dtype=torch.float32)
        gate = _command_gate(env, self.name, self._command_name, self._command_threshold)
        if gate is not None:
            return reward * gate.to(device=reward.device, dtype=reward.dtype)
        return reward


class feet_swing_height(_FootContactTerm):
    """Penalize peak-swing-height deviation from the target, evaluated at landing.

    Mirrors mjlab ``feet_swing_height``: the peak foot height accumulates while
    the foot is in the air; on first contact the squared relative error
    ``(peak / target_height - 1)^2`` is charged and the peak resets.
    """

    _allowed_params = _FootContactTerm._allowed_params | {
        "target_height",
        "command_name",
        "command_threshold",
        "asset_cfg",
    }

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._target = require_real(
            self.name,
            "target_height",
            cfg.params.get("target_height"),
            minimum=0.0,
            strict_minimum=True,
        )
        self._command_name = _command_name_param(
            self.name, cfg.params.get("command_name"), required=True
        )
        self._command_threshold = require_real(
            self.name, "command_threshold", cfg.params.get("command_threshold"), minimum=0.0
        )
        asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(f"{self.name} asset_cfg must be a SceneEntityCfg")
        asset = cast("Entity", env.scene[asset_cfg.name])
        body_names = _selected_body_names(asset, asset_cfg)
        if len(body_names) != self.num_feet:
            raise ValueError(
                f"{self.name} asset_cfg selects {len(body_names)} foot bodies but "
                f"sensor_groups declares {self.num_feet} feet"
            )
        self._asset_cfg = asset_cfg
        self._entity_name = asset_cfg.name
        self._body_names = body_names
        self._peak_heights = torch.zeros(
            (env.num_envs, self.num_feet), dtype=torch.float32, device=env.device
        )
        self._was_in_air = torch.zeros(
            (env.num_envs, self.num_feet), dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        if not isinstance(env_ids, torch.Tensor):
            raise TypeError(f"{self.name} reset env_ids must be a torch.Tensor")
        self._peak_heights[env_ids] = 0.0
        self._was_in_air[env_ids] = False

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        contact = self._contact(env)
        body = _feet_body_tensor_view(self.name, env, self._asset_cfg, self._body_names)
        positions = body.pos_w.to(device=contact.device, dtype=torch.float32)
        in_air = ~contact
        self._peak_heights = torch.where(
            in_air, torch.maximum(self._peak_heights, positions[:, :, 2]), self._peak_heights
        )
        first_contact = contact & self._was_in_air
        gate = _command_gate(env, self.name, self._command_name, self._command_threshold)
        error = self._peak_heights / self._target - 1.0
        cost = torch.sum(torch.square(error) * first_contact, dim=1)
        if gate is not None:
            cost = cost * gate.to(device=cost.device, dtype=cost.dtype)
        self._peak_heights = torch.where(
            first_contact, torch.zeros_like(self._peak_heights), self._peak_heights
        )
        self._was_in_air = in_air
        return cost.to(dtype=torch.float32)


class feet_slip(_FootContactTerm):
    """Penalize foot sliding: squared xy velocity while the foot is in contact."""

    _allowed_params = _FootContactTerm._allowed_params | {
        "command_name",
        "command_threshold",
        "asset_cfg",
    }

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._command_name = _command_name_param(
            self.name, cfg.params.get("command_name"), required=True
        )
        self._command_threshold = require_real(
            self.name, "command_threshold", cfg.params.get("command_threshold", 0.01), minimum=0.0
        )
        asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(f"{self.name} asset_cfg must be a SceneEntityCfg")
        asset = cast("Entity", env.scene[asset_cfg.name])
        body_names = _selected_body_names(asset, asset_cfg)
        if len(body_names) != self.num_feet:
            raise ValueError(
                f"{self.name} asset_cfg selects {len(body_names)} foot bodies but "
                f"sensor_groups declares {self.num_feet} feet"
            )
        self._asset_cfg = asset_cfg
        self._entity_name = asset_cfg.name
        self._body_names = body_names

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        contact = self._contact(env)
        body = _feet_body_tensor_view(self.name, env, self._asset_cfg, self._body_names)
        velocities = body.lin_vel_w.to(device=contact.device, dtype=torch.float32)
        vel_xy_norm_sq = torch.sum(torch.square(velocities[:, :, :2]), dim=2)
        gate = _command_gate(env, self.name, self._command_name, self._command_threshold)
        cost = torch.sum(vel_xy_norm_sq * contact, dim=1)
        if gate is not None:
            cost = cost * gate.to(device=cost.device, dtype=cost.dtype)
        return cost.to(dtype=torch.float32)


class feet_clearance(ManagerTermBase):
    """Penalize clearance-height deviation weighted by planar foot velocity.

    Mirrors mjlab ``feet_clearance`` with foot heights taken from world-frame
    foot body z (exact on flat terrain) instead of a terrain height sensor.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self._target_height = require_real(
            self.name,
            "target_height",
            cfg.params["target_height"],
        )
        self._command_name = _command_name_param(
            self.name,
            cfg.params.get("command_name"),
            required=False,
        )
        self._command_threshold = require_real(
            self.name,
            "command_threshold",
            cfg.params.get("command_threshold", 0.01),
            minimum=0.0,
        )
        asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(f"{self.name} asset_cfg must be a SceneEntityCfg")
        asset = cast("Entity", env.scene[asset_cfg.name])
        self._asset_cfg = asset_cfg
        self._entity_name = asset_cfg.name
        self._body_names = _selected_body_names(asset, asset_cfg)

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return self._body_names

    @property
    def entity_name(self) -> str:
        return self._entity_name

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        target_height: float = 0.1,
        command_name: str | None = None,
        command_threshold: float = 0.01,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        del target_height, command_name, command_threshold, asset_cfg
        body = _feet_body_tensor_view(self.name, env, self._asset_cfg, self._body_names)
        delta = (body.pos_w[:, :, 2] - self._target_height).abs()
        vel_norm = torch.linalg.vector_norm(body.lin_vel_w[:, :, :2], dim=2)
        cost = (delta * vel_norm).sum(dim=1)
        gate = _command_gate(env, self.name, self._command_name, self._command_threshold)
        if gate is not None:
            gate_tensor = _torch_state(gate, dtype=torch.bool)
            cost = cost * gate_tensor
        return cost.to(dtype=torch.float32)


class self_collision_cost(_FootContactTerm):
    """Count monitored self-collision geom pairs reporting contact.

    Mirrors mjlab ``self_collision_cost`` in its found-count form: each sensor
    group binds one contact sensor watching a geom pair, and the cost is the
    number of pairs with an active contact.  mjlab's ``force_threshold`` only
    applies when the sensor records force history, which the upstream config
    does not enable, so it is not ported.
    """

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        return torch.sum(self._contact(env), dim=1).to(dtype=torch.float32)


class angular_momentum_penalty(SensorTermBase):
    """Penalize whole-body angular momentum read from a named vec3 sensor."""

    _allowed_params: ClassVar[frozenset[str]] = frozenset({"sensor_name"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        sensor_name = cfg.params.get("sensor_name")
        if not isinstance(sensor_name, str) or not sensor_name:
            raise ValueError(f"{self.name} sensor_name must be a non-empty string")
        self._sensor_name = sensor_name
        self._sensor = self._bind((sensor_name,))
        if self._sensor.dimensions != (3,):
            raise ValueError(
                f"{self.name} sensor '{sensor_name}' must expose 3 values; received "
                f"{self._sensor.dimensions} on backend '{self._sensor.backend_type}'"
            )

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        return (self._sensor_name,)

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        return self._read_tensor_values(env).square().sum(dim=1)

    def _read_tensor_values(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        read_plan = getattr(env.scene, "read_plan", None)
        names = self.tensor_sensor_names
        if read_plan is None or not set(names).issubset(
            set(read_plan.sensor_names.get("robot", ()))
        ):
            raise NotImplementedError(
                f"Manager term '{self.name}' requires its named sensors "
                f"{names} in the packed Torch read plan"
            )
        entity = env.scene["robot"]
        value = read_plan.sensor_tensor_views(entity, self.tensor_sensor_names).values[
            self._sensor_name
        ]
        if tuple(value.shape) != (env.num_envs, 3):
            raise ValueError(
                f"{self.name} sensor '{self._sensor_name}' must have shape "
                f"(num_envs, 3); got {tuple(value.shape)}"
            )
        return value.to(dtype=torch.float32)


# Privileged per-foot observations (mjlab velocity critic terms).  These mirror
# mjlab's ``foot_height`` / ``foot_air_time`` / ``foot_contact`` /
# ``foot_contact_forces`` observation terms: heights come from world-frame foot
# body z (exact on flat terrain without a raycast sensor), contact state from
# the same named contact-sensor groups as the gait rewards.


class foot_height(ManagerTermBase):
    """Per-foot vertical clearance above the terrain, shape ``(num_envs, num_feet)``.

    mjlab reads this from a ``TerrainHeightSensor`` ring around each foot; the
    UniLab port returns the world-frame foot body z, which is identical on flat
    terrain.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(f"{self.name} asset_cfg must be a SceneEntityCfg")
        asset = cast("Entity", env.scene[asset_cfg.name])
        self._asset_cfg = asset_cfg
        self._entity_name = asset_cfg.name
        self._body_names = _selected_body_names(asset, asset_cfg)

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return self._body_names

    @property
    def entity_name(self) -> str:
        return self._entity_name

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        del asset_cfg
        body = _feet_body_tensor_view(self.name, env, self._asset_cfg, self._body_names)
        return body.pos_w[:, :, 2]


class foot_air_time(_FootContactTerm):
    """Current per-foot air time in seconds, shape ``(num_envs, num_feet)``.

    Mirrors mjlab ``foot_air_time`` (``ContactSensor.data.current_air_time``):
    the timer accumulates ``env.step_dt`` while the foot has no contact and
    resets to zero on contact.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._air_time = torch.zeros(
            (env.num_envs, self.num_feet), dtype=torch.float32, device=env.device
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        if not isinstance(env_ids, torch.Tensor):
            raise TypeError(f"{self.name} reset env_ids must be a torch.Tensor")
        self._air_time[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        contact = self._contact_feet(env)
        step_dt = require_real(self.name, "step_dt", env.step_dt, minimum=0.0, strict_minimum=True)
        self._air_time = torch.where(
            contact, torch.zeros_like(self._air_time), self._air_time + step_dt
        )
        return self._air_time.clone()


class foot_contact(_FootContactTerm):
    """Per-foot contact flag as float, shape ``(num_envs, num_feet)``.

    Mirrors mjlab ``foot_contact`` (``ContactSensor.data.found > 0``).
    """

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        return self._contact(env).to(dtype=torch.float32)


class foot_contact_forces(_FootContactTerm):
    """Log-compressed per-foot contact forces, shape ``(num_envs, 3 * num_feet)``.

    Mirrors mjlab ``foot_contact_forces``: the per-foot 3-D contact forces are
    flattened in sensor-group order and compressed as
    ``sign(f) * log1p(|f|)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        if any(width != 3 for width in self._view.dimensions):
            raise ValueError(
                f"{self.name} contact sensors must each expose 3-D force; "
                f"received {self._view.dimensions} on backend '{self._view.backend_type}'"
            )

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        forces = self._force_values(env)
        return forces.sign() * forces.abs().log1p()

    def _force_values(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        read_plan = getattr(env.scene, "read_plan", None)
        names = self.tensor_sensor_names
        if read_plan is None or not set(names).issubset(
            set(read_plan.sensor_names.get("robot", ()))
        ):
            raise NotImplementedError(
                f"Manager term '{self.name}' requires its named sensors "
                f"{names} in the packed Torch read plan"
            )
        entity = env.scene["robot"]
        values = read_plan.sensor_tensor_views(entity, names).values
        forces = torch.cat([values[name] for name in names], dim=1)
        if tuple(forces.shape) != (env.num_envs, self._flat_width):
            raise ValueError(
                f"{self.name} foot contact forces must have shape "
                f"{(env.num_envs, self._flat_width)}; got {tuple(forces.shape)}"
            )
        return forces.to(dtype=torch.float32)


__all__ = [
    "angular_momentum_penalty",
    "feet_air_time",
    "feet_clearance",
    "feet_slip",
    "feet_swing_height",
    "foot_air_time",
    "foot_contact",
    "foot_contact_forces",
    "foot_height",
    "self_collision_cost",
]
