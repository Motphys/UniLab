# Derived from mujocolab/mjlab v1.6.0 (0fb8a681),
# src/mjlab/tasks/velocity/mdp/rewards.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy, named-sensor contact groups, and the
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

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.managers.manager_base import ManagerTermBaseCfg
from unilab.managers.scene_entity_config import SceneEntityCfg

from .manager_terms import SensorTermBase, _command, _real, _state

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


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
) -> np.ndarray | None:
    """mjlab command gate: planar norm plus yaw magnitude above the threshold."""
    if command_name is None:
        return None
    command = _command(env, term, command_name)
    total = np.linalg.norm(command[:, :2], axis=1) + np.abs(command[:, 2])
    return np.asarray(total > command_threshold)


def _command_name_param(term: str, value: Any, *, required: bool) -> str | None:
    if value is None:
        if required:
            raise ValueError(f"{term} command_name must be a non-empty string")
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{term} command_name must be a non-empty string")
    return value


def _num_selected_bodies(asset: Entity, asset_cfg: SceneEntityCfg) -> int:
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, list):
        return len(body_ids)
    return len(range(*body_ids.indices(asset.num_bodies)))


def _feet_pos_vel(
    term: str, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, num_feet: int
) -> tuple[np.ndarray, np.ndarray]:
    asset = cast("Entity", env.scene[asset_cfg.name])
    positions = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    velocities = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]
    shape = (env.num_envs, num_feet, 3)
    positions = _state(term, "foot body position", positions, shape)
    velocities = _state(term, "foot body velocity", velocities, shape)
    return positions, velocities


class _FootContactTerm(SensorTermBase):
    """Cold-path binding of per-foot contact sensor groups (mjlab slot reduce)."""

    _allowed_params: ClassVar[frozenset[str]] = frozenset({"sensor_groups", "contact_threshold"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._groups = _sensor_groups(self.name, cfg.params.get("sensor_groups"))
        self._contact_threshold = _real(
            self.name, "contact_threshold", cfg.params.get("contact_threshold", 0.0), minimum=0.0
        )
        flat = tuple(name for group in self._groups for name in group)
        self._view = self._bind(flat)
        if any(width not in (1, 3) for width in self._view.dimensions):
            raise ValueError(
                f"{self.name} contact sensors must each expose 1-D found or 3-D force; "
                f"received {self._view.dimensions} on backend '{self._view.backend_type}'"
            )
        starts = np.cumsum((0, *self._view.dimensions[:-1]), dtype=np.int64)
        # MuJoCo ``<contact data="force">`` sensors report the force in the
        # contact frame whose first axis is the contact normal, so the gating
        # component is column 0 for both 1-D ``found`` and 3-D ``force``.
        columns = starts
        self._flat_width = int(sum(self._view.dimensions))
        # Per-foot column groups in the flattened sensor layout.
        self._columns: list[np.ndarray] = []
        offset = 0
        for group in self._groups:
            self._columns.append(columns[offset : offset + len(group)])
            offset += len(group)

    @property
    def num_feet(self) -> int:
        return len(self._groups)

    def _contact(self, env: ManagerBasedRlEnv) -> np.ndarray:
        values = _state(
            self.name,
            "foot contact",
            self._read(self._view, self.name),
            (env.num_envs, self._flat_width),
        )
        contact = values > self._contact_threshold
        per_foot = [contact[:, columns].any(axis=1) for columns in self._columns]
        return np.stack(per_foot, axis=1)


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
        self._threshold_min = _real(
            self.name, "threshold_min", cfg.params.get("threshold_min", 0.05), minimum=0.0
        )
        self._threshold_max = _real(
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
        self._command_threshold = _real(
            self.name, "command_threshold", cfg.params.get("command_threshold", 0.5), minimum=0.0
        )
        self._air_time = np.zeros((env.num_envs, self.num_feet), dtype=get_global_dtype())

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        self._air_time[env_ids if env_ids is not None else slice(None)] = 0.0

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        contact = self._contact(env)
        step_dt = _real(self.name, "step_dt", env.step_dt, minimum=0.0, strict_minimum=True)
        self._air_time = np.where(contact, 0.0, self._air_time + step_dt).astype(
            get_global_dtype(), copy=False
        )
        in_range = (self._air_time > self._threshold_min) & (self._air_time < self._threshold_max)
        reward = np.sum(in_range.astype(get_global_dtype()), axis=1)
        gate = _command_gate(env, self.name, self._command_name, self._command_threshold)
        if gate is not None:
            reward = reward * gate
        return np.asarray(reward, dtype=get_global_dtype())


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
        self._target = _real(
            self.name,
            "target_height",
            cfg.params.get("target_height"),
            minimum=0.0,
            strict_minimum=True,
        )
        self._command_name = _command_name_param(
            self.name, cfg.params.get("command_name"), required=True
        )
        self._command_threshold = _real(
            self.name, "command_threshold", cfg.params.get("command_threshold"), minimum=0.0
        )
        asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(f"{self.name} asset_cfg must be a SceneEntityCfg")
        asset = cast("Entity", env.scene[asset_cfg.name])
        num_bodies = _num_selected_bodies(asset, asset_cfg)
        if num_bodies != self.num_feet:
            raise ValueError(
                f"{self.name} asset_cfg selects {num_bodies} foot bodies but "
                f"sensor_groups declares {self.num_feet} feet"
            )
        self._asset_cfg = asset_cfg
        self._peak_heights = np.zeros((env.num_envs, self.num_feet), dtype=get_global_dtype())
        self._was_in_air = np.zeros((env.num_envs, self.num_feet), dtype=np.bool_)

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        rows = env_ids if env_ids is not None else slice(None)
        self._peak_heights[rows] = 0.0
        self._was_in_air[rows] = False

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        contact = self._contact(env)
        positions, _ = _feet_pos_vel(self.name, env, self._asset_cfg, self.num_feet)
        in_air = ~contact
        self._peak_heights = np.where(
            in_air, np.maximum(self._peak_heights, positions[:, :, 2]), self._peak_heights
        )
        first_contact = contact & self._was_in_air
        gate = _command_gate(env, self.name, self._command_name, self._command_threshold)
        error = self._peak_heights / self._target - 1.0
        cost = np.sum(np.square(error) * first_contact, axis=1)
        if gate is not None:
            cost = cost * gate
        self._peak_heights = np.where(first_contact, 0.0, self._peak_heights)
        self._was_in_air = in_air
        return np.asarray(cost, dtype=get_global_dtype())


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
        self._command_threshold = _real(
            self.name, "command_threshold", cfg.params.get("command_threshold", 0.01), minimum=0.0
        )
        asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(f"{self.name} asset_cfg must be a SceneEntityCfg")
        asset = cast("Entity", env.scene[asset_cfg.name])
        num_bodies = _num_selected_bodies(asset, asset_cfg)
        if num_bodies != self.num_feet:
            raise ValueError(
                f"{self.name} asset_cfg selects {num_bodies} foot bodies but "
                f"sensor_groups declares {self.num_feet} feet"
            )
        self._asset_cfg = asset_cfg

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        contact = self._contact(env)
        _, velocities = _feet_pos_vel(self.name, env, self._asset_cfg, self.num_feet)
        vel_xy_norm_sq = np.sum(np.square(velocities[:, :, :2]), axis=2)
        gate = _command_gate(env, self.name, self._command_name, self._command_threshold)
        cost = np.sum(vel_xy_norm_sq * contact, axis=1)
        if gate is not None:
            cost = cost * gate
        return np.asarray(cost, dtype=get_global_dtype())


def feet_clearance(
    env: ManagerBasedRlEnv,
    target_height: float,
    command_name: str | None = None,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    """Penalize clearance-height deviation weighted by planar foot velocity.

    Mirrors mjlab ``feet_clearance`` with foot heights taken from world-frame
    foot body z (exact on flat terrain) instead of a terrain height sensor.
    """
    target = _real("feet_clearance", "target_height", target_height)
    threshold = _real("feet_clearance", "command_threshold", command_threshold, minimum=0.0)
    name = _command_name_param("feet_clearance", command_name, required=False)
    asset = cast("Entity", env.scene[asset_cfg.name])
    positions = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    velocities = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]
    if not isinstance(positions, np.ndarray) or positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError(
            f"feet_clearance foot body position must have shape ({env.num_envs}, num_feet, 3), "
            f"got {getattr(positions, 'shape', None)}"
        )
    positions = _state("feet_clearance", "foot body position", positions, positions.shape)
    velocities = _state("feet_clearance", "foot body velocity", velocities, positions.shape)
    delta = np.abs(positions[:, :, 2] - target)
    vel_norm = np.linalg.norm(velocities[:, :, :2], axis=2)
    cost = np.sum(delta * vel_norm, axis=1)
    gate = _command_gate(env, "feet_clearance", name, threshold)
    if gate is not None:
        cost = cost * gate
    return np.asarray(cost, dtype=get_global_dtype())


class self_collision_cost(_FootContactTerm):
    """Count monitored self-collision geom pairs reporting contact.

    Mirrors mjlab ``self_collision_cost`` in its found-count form: each sensor
    group binds one contact sensor watching a geom pair, and the cost is the
    number of pairs with an active contact.  mjlab's ``force_threshold`` only
    applies when the sensor records force history, which the upstream config
    does not enable, so it is not ported.
    """

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        return np.asarray(np.sum(self._contact(env), axis=1), dtype=get_global_dtype())


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

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        angmom = _state(
            self.name,
            f"sensor '{self._sensor_name}'",
            self._read(self._sensor, self.name),
            (env.num_envs, 3),
        )
        return np.asarray(np.sum(np.square(angmom), axis=1), dtype=get_global_dtype())


# Privileged per-foot observations (mjlab velocity critic terms).  These mirror
# mjlab's ``foot_height`` / ``foot_air_time`` / ``foot_contact`` /
# ``foot_contact_forces`` observation terms: heights come from world-frame foot
# body z (exact on flat terrain without a raycast sensor), contact state from
# the same named contact-sensor groups as the gait rewards.


def foot_height(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    """Per-foot vertical clearance above the terrain, shape ``(num_envs, num_feet)``.

    mjlab reads this from a ``TerrainHeightSensor`` ring around each foot; the
    UniLab port returns the world-frame foot body z, which is identical on flat
    terrain.
    """
    asset = cast("Entity", env.scene[asset_cfg.name])
    positions = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    if not isinstance(positions, np.ndarray) or positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError(
            f"foot_height foot body position must have shape ({env.num_envs}, num_feet, 3), "
            f"got {getattr(positions, 'shape', None)}"
        )
    positions = _state("foot_height", "foot body position", positions, positions.shape)
    return np.asarray(positions[:, :, 2], dtype=get_global_dtype())


class foot_air_time(_FootContactTerm):
    """Current per-foot air time in seconds, shape ``(num_envs, num_feet)``.

    Mirrors mjlab ``foot_air_time`` (``ContactSensor.data.current_air_time``):
    the timer accumulates ``env.step_dt`` while the foot has no contact and
    resets to zero on contact.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._air_time = np.zeros((env.num_envs, self.num_feet), dtype=get_global_dtype())

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        self._air_time[env_ids if env_ids is not None else slice(None)] = 0.0

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        contact = self._contact(env)
        step_dt = _real(self.name, "step_dt", env.step_dt, minimum=0.0, strict_minimum=True)
        self._air_time = np.where(contact, 0.0, self._air_time + step_dt).astype(
            get_global_dtype(), copy=False
        )
        return self._air_time.copy()


class foot_contact(_FootContactTerm):
    """Per-foot contact flag as float, shape ``(num_envs, num_feet)``.

    Mirrors mjlab ``foot_contact`` (``ContactSensor.data.found > 0``).
    """

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        return np.asarray(self._contact(env), dtype=get_global_dtype())


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

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        values = _state(
            self.name,
            "foot contact forces",
            self._read(self._view, self.name),
            (env.num_envs, self._flat_width),
        )
        return np.asarray(np.sign(values) * np.log1p(np.abs(values)), dtype=get_global_dtype())


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
