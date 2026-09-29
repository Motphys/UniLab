"""Base-owned NumPy scene/entity facade for manager terms.

The facade deliberately describes partitions of an already materialized UniLab scene.
It is not a second scene composer: all name resolution and state reads go through the
public :class:`~unisim.backend.base.SimBackend` contract.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NoReturn, cast

import numpy as np
import torch
from unisim.backend.base import (
    BackendRootStateLayout,
    BackendSensorView,
    HostBridgeTransferPlan,
    SimBackend,
    TensorDataPlane,
    TensorExecution,
    TensorIOSpec,
    TensorLifecycleCapabilities,
    TensorProcessTopology,
    tensor_device_matches,
)
from unisim.dr.types import IntervalRandomizationPlan

from unilab.utils.rotation import np_quat_apply, np_quat_apply_inverse, np_yaw_from_quat

if TYPE_CHECKING:
    from unilab.base.reset_state import ResetStateTransaction
    from unilab.base.scene import SceneCfg


NamesCfg = tuple[str, ...] | list[str] | None
BodyStateCopyFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]


@dataclass(frozen=True)
class EntityCfg:
    """Declare one logical entity inside an existing backend scene.

    Names are explicit because UniLab keeps scene composition in task-owned XML and
    backend adapters.  ``None`` means that the namespace is not exposed by this
    entity; an empty sequence means that it is exposed but contains no elements.
    """

    root_body_name: str | None = None
    joint_names: NamesCfg = None
    body_names: NamesCfg = None
    geom_names: NamesCfg = None
    site_names: NamesCfg = None
    actuator_names: NamesCfg = None
    physical_entity: str | None = None


def _normalize_names(entity_name: str, kind: str, names: NamesCfg) -> tuple[str, ...] | None:
    if names is None:
        return None
    if isinstance(names, str):
        raise TypeError(
            f"Entity '{entity_name}' {kind} names must be a sequence of strings, not a scalar"
        )
    invalid = [value for value in names if not isinstance(value, str)]
    if invalid:
        raise TypeError(f"Entity '{entity_name}' {kind} names must be strings; got {invalid}")
    normalized = tuple(names)
    if any(not name for name in normalized):
        raise ValueError(f"Entity '{entity_name}' {kind} names must be non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Entity '{entity_name}' {kind} names must be unique: {normalized}")
    return normalized


def _readonly_ids(values: np.ndarray | Sequence[int], *, expected: int, label: str) -> np.ndarray:
    raw_ids = np.asarray(values)
    if not np.issubdtype(raw_ids.dtype, np.integer) or np.issubdtype(raw_ids.dtype, np.bool_):
        raise TypeError(f"{label} resolver must return integer IDs, got dtype {raw_ids.dtype}")
    ids = np.asarray(raw_ids, dtype=np.int32)
    if ids.shape != (expected,):
        raise ValueError(f"{label} resolver returned shape {ids.shape}, expected ({expected},)")
    if np.any(ids < 0):
        raise ValueError(f"{label} resolver returned negative IDs: {ids.tolist()}")
    if np.unique(ids).size != ids.size:
        raise ValueError(f"{label} resolver returned duplicate IDs: {ids.tolist()}")
    ids = np.array(ids, copy=True, dtype=np.int32)
    ids.setflags(write=False)
    return ids


def _as_column_index(ids: np.ndarray) -> slice | np.ndarray:
    """Use a slice for contiguous columns and advanced indexing otherwise."""
    if ids.size:
        start = int(ids[0])
        if np.array_equal(ids, np.arange(start, start + ids.size, dtype=ids.dtype)):
            return slice(start, start + ids.size)
    index = np.asarray(ids, dtype=np.intp).copy()
    index.setflags(write=False)
    return index


def _readonly_array(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values)
    if result.flags.writeable:
        result = result.copy()
    result.setflags(write=False)
    return result


# Matching semantics derived from mujocolab/mjlab v1.6.0 (0fb8a681),
# src/mjlab/utils/lab_api/string.py. Copyright 2025, The mjlab Developers;
# adapted for the UniLab NumPy facade under Apache-2.0.
def _resolve_matching_names(
    keys: str | Sequence[str], names: Sequence[str], preserve_order: bool
) -> tuple[list[int], list[str]]:
    """Pinned mjlab-compatible full-regex matching over cached entity names."""
    patterns = (keys,) if isinstance(keys, str) else tuple(keys)
    matches: list[tuple[int, int, str]] = []
    matched_by: list[str | None] = [None] * len(names)
    per_pattern: list[list[str]] = [[] for _ in patterns]

    for name_index, candidate in enumerate(names):
        for pattern_index, pattern in enumerate(patterns):
            try:
                matched = re.fullmatch(pattern, candidate) is not None
            except re.error as exc:
                raise ValueError(f"Invalid entity selector regex {pattern!r}: {exc}") from exc
            if not matched:
                continue
            if matched_by[name_index] is not None:
                raise ValueError(
                    f"Multiple matches for '{candidate}': "
                    f"'{matched_by[name_index]}' and '{pattern}'!"
                )
            matched_by[name_index] = pattern
            matches.append((pattern_index, name_index, candidate))
            per_pattern[pattern_index].append(candidate)

    if any(not values for values in per_pattern):
        rendered = ", ".join(
            f"{pattern!r}: {values}" for pattern, values in zip(patterns, per_pattern)
        )
        raise ValueError(
            "Not all entity selector regular expressions matched; "
            f"matches=({rendered}), available={list(names)}"
        )

    if preserve_order:
        matches.sort(key=lambda item: item[0])
    return [item[1] for item in matches], [item[2] for item in matches]


_StateReadKey = tuple[str, tuple[int, ...] | None]


@dataclass(frozen=True)
class EntityTensorStateView:
    """Entity-scoped Torch qpos/qvel-derived joint state for one read phase."""

    joint_pos: torch.Tensor
    joint_vel: torch.Tensor


@dataclass(frozen=True)
class EntityTensorSensorViews:
    """Validated named tensor sensors in the declared request order."""

    names: tuple[str, ...]
    values: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


@dataclass(frozen=True)
class EntityTensorBodyStateView:
    """Entity-ordered world-frame Torch body state for one read phase."""

    body_names: tuple[str, ...]
    pos_w: torch.Tensor
    quat_w: torch.Tensor
    lin_vel_w: torch.Tensor
    ang_vel_w: torch.Tensor


_TENSOR_BODY_SENSOR_FIELDS: tuple[tuple[str, str], ...] = (
    ("pos_w", "track_pos_w_"),
    ("quat_w", "track_quat_w_"),
    ("lin_vel_w", "track_linvel_w_"),
    ("ang_vel_w", "track_angvel_w_"),
)
_TENSOR_SENSOR_WIDTHS = {"pos_w": 3, "quat_w": 4, "lin_vel_w": 3, "ang_vel_w": 3}


@dataclass(frozen=True)
class SceneTensorReadSpec:
    """Declare one entity's named state reads for the scene tensor phase.

    ``sensor_names`` are explicit backend names. ``body_names`` are entity-local
    names; the scene owner expands them into the canonical ``track_*`` sensor
    contract. Requests are explicit so the packed packet never grows from an
    implicit EntityCfg namespace that is only needed by a cold-path consumer.
    """

    entity: str
    sensor_names: tuple[str, ...] = ()
    body_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.entity, str) or not self.entity:
            raise TypeError("Scene tensor read entity must be a non-empty string")
        if isinstance(self.sensor_names, (str, bytes)):
            raise TypeError("Scene tensor read sensor_names must be a sequence of strings")
        if isinstance(self.body_names, (str, bytes)):
            raise TypeError("Scene tensor read body_names must be a sequence of strings")
        sensors = tuple(self.sensor_names)
        bodies = tuple(self.body_names)
        if not sensors and not bodies:
            # Generic joint observations request the entity's packed qpos/qvel
            # packet without named sensors or body-projected state. This is a
            # state-only request and remains valid for a runtime read phase.
            pass
        for label, names in (("sensor", sensors), ("body", bodies)):
            if any(not isinstance(name, str) or not name for name in names):
                raise TypeError(f"Scene tensor read {label} names must be non-empty strings")
            if len(set(names)) != len(names):
                raise ValueError(f"Scene tensor read {label} names must be unique: {names}")
        object.__setattr__(self, "sensor_names", sensors)
        object.__setattr__(self, "body_names", bodies)


class SceneTensorReadPlan:
    """Phase-scoped packed tensor reads for one ``EntityScene``.

    The plan is the Manager read boundary. A phase calls ``refresh()`` (or
    ``refresh_selected()`` after a selected reset) once and then slices entity
    views from the resulting packet. Host-bridge execution therefore performs
    one packed H2D transfer per read phase regardless of the number of terms,
    entities, sensors, or bodies.
    """

    def __init__(
        self,
        *,
        scene: "EntityScene",
        device: torch.device,
        entities: Mapping[str, "Entity"],
        sensor_names: Mapping[str, tuple[str, ...]],
        body_names: Mapping[str, tuple[str, ...]],
        host_plan: HostBridgeTransferPlan | None,
    ) -> None:
        self._scene = scene
        self._device = torch.device(device)
        self._entities = MappingProxyType(dict(entities))
        self._sensor_names = MappingProxyType(dict(sensor_names))
        self._body_names = MappingProxyType(dict(body_names))
        self._host_plan = host_plan
        self._packet_names = self._aggregate_packet_names()
        self._packet_sensor_owners: dict[str, str] = {}
        self._packet_sensor_owners.update(
            (name, entity_name)
            for entity_name, names in self._sensor_names.items()
            for name in names
        )
        for entity_name, body_names_value in self._body_names.items():
            for body_name in body_names_value:
                for _, prefix in _TENSOR_BODY_SENSOR_FIELDS:
                    self._packet_sensor_owners.setdefault(prefix + body_name, entity_name)
        self._packet: Mapping[str, torch.Tensor] = MappingProxyType({})
        self._refreshed = False
        self._closed = False

    @property
    def scene(self) -> "EntityScene":
        return self._scene

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def entities(self) -> Mapping[str, "Entity"]:
        return self._entities

    @property
    def sensor_names(self) -> Mapping[str, tuple[str, ...]]:
        return self._sensor_names

    @property
    def body_names(self) -> Mapping[str, tuple[str, ...]]:
        return self._body_names

    @property
    def host_plan(self) -> HostBridgeTransferPlan | None:
        return self._host_plan

    @property
    def ready(self) -> bool:
        """Return whether the current packet is available to phase readers."""
        return self._refreshed and not self._closed

    @property
    def packet_names(self) -> tuple[str, ...]:
        """Return the immutable backend sensor names compiled into this plan."""
        return self._packet_names

    def refresh(self) -> None:
        """Publish one full-batch packet for the current read phase."""
        self._require_not_closed()
        packet = self._read_packet(self._host_plan.read_state_sensors if self._host_plan else None)
        self._publish_packet(packet)

    def refresh_selected(self) -> None:
        """Publish the post-selected-reset packet for the current read phase.

        Host bridges use the backend's selected-row boundary so reset rebuild
        does not trigger a second full read. Device-resident backends refresh
        their stable public views through the same normal read path.
        """
        self._require_not_closed()
        reader = (
            self._host_plan.read_selected_state_sensors if self._host_plan is not None else None
        )
        packet = self._read_packet(reader)
        self._publish_packet(packet)

    def invalidate(self) -> None:
        """Drop phase values after an in-phase simulation mutation."""
        self._require_not_closed()
        self._packet = MappingProxyType({})
        self._refreshed = False

    def joint_tensor_view(self, entity: "str | Entity") -> EntityTensorStateView:
        """Return one entity's validated joint state from the phase packet."""
        owner = self._require_entity(entity)
        names = owner._joint_names
        if names is None:
            raise owner._capability_error(
                "scene joint tensor state", "joint_names were not declared in EntityCfg"
            )
        try:
            qpos_ids = self._scene._backend.get_joint_state_qpos_indices(names)
            qvel_ids = self._scene._backend.get_joint_state_qvel_indices(names)
        except (AttributeError, NotImplementedError) as exc:
            raise owner._capability_error("scene joint tensor state layout", str(exc)) from exc
        expected = (len(names),)
        if qpos_ids.shape != expected or qvel_ids.shape != expected:
            raise ValueError(
                f"Entity '{owner.name}' scene tensor joint-state layout on backend "
                f"'{owner._backend_type}' returned shapes {qpos_ids.shape} and "
                f"{qvel_ids.shape}; expected {expected}"
            )
        if np.any(qpos_ids < 0) or np.any(qvel_ids < 0):
            raise ValueError(
                f"Entity '{owner.name}' scene tensor joint-state layout contains negative columns"
            )

        result: dict[str, torch.Tensor] = {}
        expected_width = len(names)
        for field, state_field, columns in (
            ("joint_pos", "qpos", qpos_ids),
            ("joint_vel", "qvel", qvel_ids),
        ):
            raw = self._packet[state_field]
            expected_shape = (self._scene._backend.num_envs, expected_width)
            if raw.ndim != 2 or raw.shape[0] != expected_shape[0]:
                raise ValueError(
                    f"Entity '{owner.name}' tensor {field} view on backend "
                    f"'{owner._backend_type}' has shape {tuple(raw.shape)}; expected "
                    f"{expected_shape} after entity column selection"
                )
            if int(np.max(columns)) >= raw.shape[1]:
                raise ValueError(
                    f"Entity '{owner.name}' tensor {field} columns exceed backend "
                    f"width {raw.shape[1]}"
                )
            selected = raw[:, np.asarray(columns, dtype=np.intp)]
            if tuple(selected.shape) != expected_shape:
                raise ValueError(
                    f"Entity '{owner.name}' tensor {field} selected shape "
                    f"{tuple(selected.shape)} does not match {expected_shape}"
                )
            if not bool(torch.isfinite(selected).all()):
                raise ValueError(f"Entity '{owner.name}' tensor {field} has NaN or Inf")
            result[field] = selected
        return EntityTensorStateView(**result)

    def sensor_tensor_views(
        self, entity: "str | Entity", names: Sequence[str]
    ) -> EntityTensorSensorViews:
        """Return explicit named sensors from the phase packet."""
        owner = self._require_entity(entity)
        if isinstance(names, (str, bytes)):
            raise TypeError(
                f"Entity '{owner.name}' scene tensor sensor names must be a sequence of strings"
            )
        sensor_names = tuple(names)
        if not sensor_names:
            raise ValueError(f"Entity '{owner.name}' scene tensor sensor request is empty")
        if any(not isinstance(name, str) or not name for name in sensor_names):
            raise TypeError(
                f"Entity '{owner.name}' scene tensor sensor names must be non-empty strings"
            )
        if len(set(sensor_names)) != len(sensor_names):
            raise ValueError(
                f"Entity '{owner.name}' scene tensor sensor names must be unique: {sensor_names}"
            )
        missing = [name for name in sensor_names if name not in self._packet]
        if missing:
            raise ValueError(
                f"Entity '{owner.name}' scene tensor sensors {missing} were not compiled "
                f"into the packed read; compiled={list(self._packet_names)}"
            )
        values = {
            name: Entity._validate_tensor_sensor(
                self._packet[name],
                entity_name=owner.name,
                backend_type=owner._backend_type,
                sensor_name=name,
                expected_width=None,
                num_envs=self._scene._backend.num_envs,
                device=self._device,
            )
            for name in sensor_names
        }
        return EntityTensorSensorViews(names=sensor_names, values=values)

    def body_tensor_view(
        self, entity: "str | Entity", body_names: Sequence[str] | None = None
    ) -> EntityTensorBodyStateView:
        """Return entity-local body state from canonical phase sensors."""
        owner = self._require_entity(entity)
        if owner._body_names is None:
            raise owner._capability_error(
                "scene body tensor state", "body_names were not declared in EntityCfg"
            )
        requested = self._scene._normalize_tensor_body_request(
            owner, tuple(body_names) if body_names is not None else owner._body_names
        )
        missing = [name for name in requested if name not in set(self._body_names[owner.name])]
        if missing:
            raise ValueError(
                f"Entity '{owner.name}' scene tensor bodies {missing} were not compiled "
                f"into the packed read; compiled={list(self._body_names[owner.name])}"
            )

        fields: dict[str, torch.Tensor] = {}
        for field, prefix in _TENSOR_BODY_SENSOR_FIELDS:
            stacked = [
                Entity._validate_tensor_sensor(
                    self._packet[prefix + body_name],
                    entity_name=owner.name,
                    backend_type=owner._backend_type,
                    sensor_name=prefix + body_name,
                    expected_width=_TENSOR_SENSOR_WIDTHS[field],
                    num_envs=self._scene._backend.num_envs,
                    device=self._device,
                )
                for body_name in requested
            ]
            fields[field] = torch.stack(stacked, dim=1)
        return EntityTensorBodyStateView(
            body_names=requested,
            pos_w=fields["pos_w"],
            quat_w=fields["quat_w"],
            lin_vel_w=fields["lin_vel_w"],
            ang_vel_w=fields["ang_vel_w"],
        )

    @property
    def transfer_stats(self) -> dict[str, int]:
        self._require_open()
        if self._host_plan is None:
            return {}
        return dict(self._host_plan.transfer_stats)

    @property
    def last_timing(self) -> dict[str, float]:
        self._require_open()
        if self._host_plan is None:
            return {}
        return dict(self._host_plan.last_timing)

    def close(self) -> None:
        """Release the backend-owned packed read plan."""
        if self._closed:
            return
        try:
            if self._host_plan is not None:
                self._host_plan.close()
        finally:
            self._packet = MappingProxyType({})
            self._refreshed = False
            self._closed = True

    def _aggregate_packet_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for entity_names in self._sensor_names.values():
            names.extend(entity_names)
        for entity_names in self._body_names.values():
            for body_name in entity_names:
                names.extend(prefix + body_name for _, prefix in _TENSOR_BODY_SENSOR_FIELDS)
        return tuple(dict.fromkeys(names))

    def _read_packet(
        self, reader: Callable[[], Mapping[str, Any]] | None
    ) -> dict[str, torch.Tensor]:
        backend = self._scene._backend
        if reader is not None:
            try:
                raw = reader()
            except (AttributeError, KeyError, TypeError, ValueError, NotImplementedError) as exc:
                raise type(exc)(
                    f"Manager scene packed tensor read on backend '{backend.backend_type}': {exc}"
                ) from exc
        else:
            try:
                raw = backend.get_state_views(("qpos", "qvel"), device=self._device)
                raw = dict(raw)
                for name in self._packet_names:
                    raw[name] = backend.get_sensor_view(name, device=self._device)
            except (AttributeError, KeyError, TypeError, ValueError, NotImplementedError) as exc:
                raise type(exc)(
                    "Manager scene device-resident tensor read on backend "
                    f"'{backend.backend_type}': {exc}"
                ) from exc
        if not isinstance(raw, Mapping):
            raise TypeError(
                "Manager scene tensor read packet on backend "
                f"'{backend.backend_type}' is {type(raw).__name__}, expected a mapping"
            )
        return dict(raw)

    def _publish_packet(self, packet: dict[str, torch.Tensor]) -> None:
        backend = self._scene._backend
        expected_keys = {"qpos", "qvel", *self._packet_names}
        missing = sorted(expected_keys - set(packet))
        if missing:
            raise ValueError(
                "Manager scene tensor read packet on backend "
                f"'{backend.backend_type}' is missing {missing}"
            )
        num_envs = backend.num_envs
        for field in ("qpos", "qvel"):
            value = packet[field]
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Scene tensor {field} view on backend '{backend.backend_type}' is "
                    f"{type(value).__name__}, expected torch.Tensor"
                )
            if value.ndim != 2 or value.shape[0] != num_envs or value.shape[1] < 1:
                raise ValueError(
                    f"Scene tensor {field} view on backend '{backend.backend_type}' has "
                    f"shape {tuple(value.shape)}; expected ({num_envs}, width)"
                )
            if value.dtype != torch.float32 or value.device != self._device:
                raise TypeError(
                    f"Scene tensor {field} view must be float32 on {self._device}; "
                    f"got {value.dtype} on {value.device}"
                )
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Scene tensor {field} view has NaN or Inf")
        for name in self._packet_names:
            owner = self._packet_sensor_owners[name]
            packet[name] = Entity._validate_tensor_sensor(
                packet[name],
                entity_name=owner,
                backend_type=backend.backend_type,
                sensor_name=name,
                expected_width=None,
                num_envs=num_envs,
                device=self._device,
            )
        self._packet = MappingProxyType(packet)
        self._refreshed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Scene tensor read plan is closed")
        self._require_refreshed()

    def _require_not_closed(self) -> None:
        if self._closed:
            raise RuntimeError("Scene tensor read plan is closed")

    def _require_refreshed(self) -> None:
        if not self._refreshed:
            raise RuntimeError("Scene tensor read plan must be refreshed before the read phase")

    def _require_entity(self, entity: "str | Entity") -> "Entity":
        self._require_open()
        name = entity if isinstance(entity, str) else entity.name
        try:
            return self._entities[name]
        except KeyError as exc:
            raise KeyError(
                f"Scene tensor read entity '{name}' was not compiled into this plan; "
                f"compiled={list(self._entities)}"
            ) from exc


class _EntityStateReadCache:
    """Update-phase cache shared by every entity bound to one backend scene."""

    def __init__(self) -> None:
        self._active = False
        self._values: dict[_StateReadKey, np.ndarray] = {}

    def get(self, key: _StateReadKey) -> np.ndarray | None:
        if not self._active:
            return None
        return self._values.get(key)

    def put(self, key: _StateReadKey, value: np.ndarray) -> None:
        if self._active:
            self._values[key] = value

    def invalidate(self) -> None:
        self._values.clear()

    @contextmanager
    def scoped(self) -> Iterator[None]:
        if self._active:
            raise RuntimeError("Entity state-read cache phase is already active")
        self._active = True
        self._values.clear()
        try:
            yield
        finally:
            self._values.clear()
            self._active = False


def _state_selector_key(ids: np.ndarray | None) -> tuple[int, ...] | None:
    """Freeze a cold-path backend selector into a cheap hot-path cache key."""
    if ids is None:
        return None
    return tuple(int(value) for value in ids)


class EntityData:
    """Hot-path NumPy state surface backed by cached backend IDs.

    Control writes use the environment's Torch control tensor. NumPy callers
    remain supported for Manager terms that have not yet crossed that tensor
    boundary; both paths publish into the same actuator layout without touching
    backend-private state.
    """

    def __init__(
        self,
        backend: SimBackend,
        *,
        root_body_ids: np.ndarray | None,
        joint_pos_ids: np.ndarray | None,
        joint_vel_ids: np.ndarray | None,
        default_root_state: np.ndarray | None,
        default_root_state_error: str | None,
        default_joint_pos: np.ndarray | None,
        default_joint_vel: np.ndarray | None,
        soft_joint_pos_limits: np.ndarray | None,
        gravity_vec_w: np.ndarray | None,
        body_ids: np.ndarray | None,
        actuator_ids: np.ndarray | None,
        actuator_ctrl_range: np.ndarray | None,
        control_buffer: np.ndarray | torch.Tensor | None,
        entity_name: str,
        backend_type: str,
        state_read_cache: _EntityStateReadCache,
    ) -> None:
        self._backend = backend
        self._entity_name = entity_name
        self._backend_type = backend_type
        self._root_body_ids = root_body_ids
        self._root_body_state_key = _state_selector_key(root_body_ids)
        self._joint_pos_index = None if joint_pos_ids is None else _as_column_index(joint_pos_ids)
        self._joint_vel_index = None if joint_vel_ids is None else _as_column_index(joint_vel_ids)
        self._default_root_state = default_root_state
        self._default_root_state_error = default_root_state_error
        self._default_joint_pos = default_joint_pos
        self._default_joint_vel = default_joint_vel
        self._default_joint_pos_tensor: torch.Tensor | None = None
        self._default_joint_vel_tensor: torch.Tensor | None = None
        self._soft_joint_pos_limits = soft_joint_pos_limits
        self._gravity_vec_w = gravity_vec_w
        self._encoder_bias = (
            None
            if default_joint_pos is None
            else np.zeros(default_joint_pos.shape, dtype=default_joint_pos.dtype)
        )
        self._body_ids = body_ids
        self._body_state_key = _state_selector_key(body_ids)
        self._actuator_ids = actuator_ids
        self._actuator_ctrl_range = actuator_ctrl_range
        self._control_buffer = control_buffer
        self._state_read_cache = state_read_cache

    def _cached_getter(
        self,
        method: str,
        fn: Any,
        *args: Any,
        selector: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        key = (method, selector)
        cached = self._state_read_cache.get(key)
        if cached is not None:
            return cached
        value = fn(*args)
        self._state_read_cache.put(key, value)
        return value

    def _require(self, value, capability: str):
        if value is None:
            raise NotImplementedError(
                f"Entity '{self._entity_name}' data capability '{capability}' is unavailable "
                f"on backend '{self._backend_type}': it was not materialized"
            )
        return value

    @property
    def root_link_pos_w(self) -> np.ndarray:
        ids = self._require(self._root_body_ids, "root body state")
        return self._cached_getter(
            "body_pos_w",
            self._backend.get_body_pos_w,
            ids,
            selector=self._root_body_state_key,
        )[:, 0]

    @property
    def root_link_quat_w(self) -> np.ndarray:
        ids = self._require(self._root_body_ids, "root body state")
        return self._cached_getter(
            "body_quat_w",
            self._backend.get_body_quat_w,
            ids,
            selector=self._root_body_state_key,
        )[:, 0]

    @property
    def root_link_lin_vel_w(self) -> np.ndarray:
        ids = self._require(self._root_body_ids, "root body state")
        return self._cached_getter(
            "body_lin_vel_w",
            self._backend.get_body_lin_vel_w,
            ids,
            selector=self._root_body_state_key,
        )[:, 0]

    @property
    def root_link_ang_vel_w(self) -> np.ndarray:
        ids = self._require(self._root_body_ids, "root body state")
        return self._cached_getter(
            "body_ang_vel_w",
            self._backend.get_body_ang_vel_w,
            ids,
            selector=self._root_body_state_key,
        )[:, 0]

    @property
    def root_link_lin_vel_b(self) -> np.ndarray:
        ids = self._require(self._root_body_ids, "root body state")
        return self._cached_getter(
            "body_lin_vel_b",
            self._backend.get_body_lin_vel_b,
            ids,
            selector=self._root_body_state_key,
        )[:, 0]

    @property
    def root_link_ang_vel_b(self) -> np.ndarray:
        ids = self._require(self._root_body_ids, "root body state")
        return self._cached_getter(
            "body_ang_vel_b",
            self._backend.get_body_ang_vel_b,
            ids,
            selector=self._root_body_state_key,
        )[:, 0]

    @property
    def heading_w(self) -> np.ndarray:
        """Root yaw in the world frame, derived from the backend quaternion view."""
        return np_yaw_from_quat(self.root_link_quat_w)

    @property
    def projected_gravity_b(self) -> np.ndarray:
        """Unit gravity vector projected into the root link frame."""
        gravity = self._require(self._gravity_vec_w, "projected gravity")
        return np_quat_apply_inverse(self.root_link_quat_w, gravity)

    @property
    def gravity_vec_w(self) -> np.ndarray:
        """Read-only world-frame unit gravity vector for every environment."""
        return self._require(self._gravity_vec_w, "world-frame gravity")

    @property
    def root_link_pose_w(self) -> np.ndarray:
        return np.concatenate((self.root_link_pos_w, self.root_link_quat_w), axis=-1)

    @property
    def root_link_vel_w(self) -> np.ndarray:
        return np.concatenate((self.root_link_lin_vel_w, self.root_link_ang_vel_w), axis=-1)

    @property
    def default_root_state(self) -> np.ndarray:
        """Read-only 13-D community root state for every environment."""
        if self._default_root_state is None:
            detail = self._default_root_state_error or "root_body_name was not declared"
            raise NotImplementedError(
                f"Entity '{self._entity_name}' data capability 'default root state' is "
                f"unavailable on backend '{self._backend_type}': {detail}"
            )
        return self._default_root_state

    @property
    def joint_pos(self) -> np.ndarray:
        index = self._require(self._joint_pos_index, "joint position")
        return self._cached_getter("dof_pos", self._backend.get_dof_pos)[:, index]

    @property
    def joint_vel(self) -> np.ndarray:
        index = self._require(self._joint_vel_index, "joint velocity")
        return self._cached_getter("dof_vel", self._backend.get_dof_vel)[:, index]

    @property
    def joint_pos_biased(self) -> np.ndarray:
        """Joint positions with the manager-owned encoder bias applied."""
        return self.joint_pos + self.encoder_bias

    @property
    def default_joint_pos(self) -> np.ndarray:
        """Read-only per-environment default joint positions."""
        return self._require(self._default_joint_pos, "default joint position")

    def default_joint_pos_torch(self, device: str | torch.device) -> torch.Tensor:
        """Return the cold-path default joint positions as a cached Torch tensor."""
        resolved = torch.device(device)
        cached = self._default_joint_pos_tensor
        if cached is None or cached.device != resolved:
            cached = torch.from_numpy(np.ascontiguousarray(self.default_joint_pos)).to(
                device=resolved, dtype=torch.float32
            )
            self._default_joint_pos_tensor = cached
        return cached

    @property
    def default_joint_vel(self) -> np.ndarray:
        """Read-only zero default velocities from the UniLab reset contract."""
        return self._require(self._default_joint_vel, "default joint velocity")

    def default_joint_vel_torch(self, device: str | torch.device) -> torch.Tensor:
        """Return the cold-path default joint velocities as a cached Torch tensor."""
        resolved = torch.device(device)
        cached = self._default_joint_vel_tensor
        if cached is None or cached.device != resolved:
            cached = torch.from_numpy(np.ascontiguousarray(self.default_joint_vel)).to(
                device=resolved, dtype=torch.float32
            )
            self._default_joint_vel_tensor = cached
        return cached

    @property
    def soft_joint_pos_limits(self) -> np.ndarray:
        """Read-only joint position limits in the declared entity joint order."""
        return self._require(self._soft_joint_pos_limits, "joint position limits")

    @property
    def encoder_bias(self) -> np.ndarray:
        """Mutable per-environment joint encoder bias used by position actions."""
        return self._require(self._encoder_bias, "joint encoder bias")

    @property
    def body_link_pos_w(self) -> np.ndarray:
        ids = self._require(self._body_ids, "body state")
        return self._cached_getter(
            "body_pos_w",
            self._backend.get_body_pos_w,
            ids,
            selector=self._body_state_key,
        )

    @property
    def body_link_quat_w(self) -> np.ndarray:
        ids = self._require(self._body_ids, "body state")
        return self._cached_getter(
            "body_quat_w",
            self._backend.get_body_quat_w,
            ids,
            selector=self._body_state_key,
        )

    @property
    def body_link_lin_vel_w(self) -> np.ndarray:
        ids = self._require(self._body_ids, "body state")
        return self._cached_getter(
            "body_lin_vel_w",
            self._backend.get_body_lin_vel_w,
            ids,
            selector=self._body_state_key,
        )

    @property
    def body_link_ang_vel_w(self) -> np.ndarray:
        ids = self._require(self._body_ids, "body state")
        return self._cached_getter(
            "body_ang_vel_w",
            self._backend.get_body_ang_vel_w,
            ids,
            selector=self._body_state_key,
        )

    @property
    def body_link_pose_w(self) -> np.ndarray:
        return np.concatenate((self.body_link_pos_w, self.body_link_quat_w), axis=-1)

    def body_link_pos_w_rows(self, env_ids: np.ndarray) -> np.ndarray:
        """Row-scoped variant of body_link_pos_w for partial-reset rebuilds."""
        ids = self._require(self._body_ids, "body state")
        return self._backend.get_body_pose_w_rows(env_ids, ids)[0]

    def body_link_quat_w_rows(self, env_ids: np.ndarray) -> np.ndarray:
        """Row-scoped variant of body_link_quat_w for partial-reset rebuilds."""
        ids = self._require(self._body_ids, "body state")
        return self._backend.get_body_pose_w_rows(env_ids, ids)[1]

    def body_link_lin_vel_w_rows(self, env_ids: np.ndarray) -> np.ndarray:
        """Row-scoped variant of body_link_lin_vel_w for partial-reset rebuilds."""
        ids = self._require(self._body_ids, "body state")
        return self._backend.get_body_lin_vel_w_rows(env_ids, ids)

    def body_link_ang_vel_w_rows(self, env_ids: np.ndarray) -> np.ndarray:
        """Row-scoped variant of body_link_ang_vel_w for partial-reset rebuilds."""
        ids = self._require(self._body_ids, "body state")
        return self._backend.get_body_ang_vel_w_rows(env_ids, ids)

    @property
    def body_link_vel_w(self) -> np.ndarray:
        return np.concatenate((self.body_link_lin_vel_w, self.body_link_ang_vel_w), axis=-1)

    @property
    def actuator_ctrl_range(self) -> np.ndarray:
        return self._require(self._actuator_ctrl_range, "actuator control range")

    def write_ctrl(
        self,
        values: np.ndarray | torch.Tensor,
        env_ids: np.ndarray | slice | None = None,
        *,
        actuator_ids: np.ndarray | Sequence[int] | slice | None = None,
    ) -> None:
        """Write entity-local actuator controls into the env-owned control buffer.

        This is an in-memory scene write, analogous to the pinned manager runtime's
        entity target buffers.  Physics remains owned by ``TorchEnv``/``SimBackend``;
        this method never steps or calls a backend-private API.
        """
        entity_actuator_ids = self._require(self._actuator_ids, "actuator control write")
        control = self._require(self._control_buffer, "actuator control write")
        if not isinstance(values, np.ndarray):
            if not isinstance(values, torch.Tensor):
                raise TypeError(
                    f"Entity '{self._entity_name}' write_ctrl expected np.ndarray or "
                    f"torch.Tensor, received {type(values).__name__}"
                )
            tensor_control = self._require_tensor_control_buffer()
            self._write_tensor_ctrl(
                values,
                tensor_control,
                env_ids=env_ids,
                actuator_ids=actuator_ids,
            )
            return
        row_index: np.ndarray | slice
        if env_ids is None:
            row_index = slice(None)
            row_count = control.shape[0]
        elif isinstance(env_ids, slice):
            row_index = env_ids
            row_count = len(range(*env_ids.indices(control.shape[0])))
        else:
            raw_ids = np.asarray(env_ids)
            if (
                raw_ids.ndim != 1
                or not np.issubdtype(raw_ids.dtype, np.integer)
                or np.issubdtype(raw_ids.dtype, np.bool_)
            ):
                raise TypeError(
                    f"Entity '{self._entity_name}' write_ctrl env_ids must be a 1-D "
                    f"integer array or slice, got shape={raw_ids.shape}, dtype={raw_ids.dtype}"
                )
            row_index = np.asarray(raw_ids, dtype=np.intp)
            if np.any(row_index < 0) or np.any(row_index >= control.shape[0]):
                raise IndexError(
                    f"Entity '{self._entity_name}' write_ctrl env_ids out of range for "
                    f"{control.shape[0]} environments: {row_index.tolist()}"
                )
            if np.unique(row_index).size != row_index.size:
                raise ValueError(
                    f"Entity '{self._entity_name}' write_ctrl env_ids contain duplicates: "
                    f"{row_index.tolist()}"
                )
            row_count = len(row_index)

        if actuator_ids is None:
            selected_actuator_ids = entity_actuator_ids
        elif isinstance(actuator_ids, slice):
            selected_actuator_ids = entity_actuator_ids[actuator_ids]
        else:
            raw_actuator_ids = np.asarray(actuator_ids)
            if (
                raw_actuator_ids.ndim != 1
                or not np.issubdtype(raw_actuator_ids.dtype, np.integer)
                or np.issubdtype(raw_actuator_ids.dtype, np.bool_)
            ):
                raise TypeError(
                    f"Entity '{self._entity_name}' write_ctrl actuator_ids must be a 1-D "
                    "integer array or slice"
                )
            local_actuator_ids = np.asarray(raw_actuator_ids, dtype=np.intp)
            if np.any(local_actuator_ids < 0) or np.any(
                local_actuator_ids >= len(entity_actuator_ids)
            ):
                raise IndexError(
                    f"Entity '{self._entity_name}' write_ctrl actuator_ids out of range for "
                    f"{len(entity_actuator_ids)} entity actuators: {local_actuator_ids.tolist()}"
                )
            if np.unique(local_actuator_ids).size != local_actuator_ids.size:
                raise ValueError(
                    f"Entity '{self._entity_name}' write_ctrl actuator_ids contain duplicates: "
                    f"{local_actuator_ids.tolist()}"
                )
            selected_actuator_ids = entity_actuator_ids[local_actuator_ids]

        actuator_index = _as_column_index(np.asarray(selected_actuator_ids, dtype=np.int32))
        actuator_count = len(selected_actuator_ids)
        expected = (row_count, actuator_count)
        if values.shape != expected:
            raise ValueError(
                f"Entity '{self._entity_name}' write_ctrl expected shape {expected}, "
                f"received {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"Entity '{self._entity_name}' write_ctrl received NaN or Inf")

        if isinstance(control, torch.Tensor):
            self._write_tensor_ctrl(
                torch.as_tensor(values, dtype=torch.float32, device=control.device),
                control,
                env_ids=env_ids,
                actuator_ids=actuator_ids,
            )
        elif isinstance(row_index, slice) or isinstance(actuator_index, slice):
            control[row_index, actuator_index] = values
        else:
            control[row_index[:, None], actuator_index[None, :]] = values

    def _require_tensor_control_buffer(self) -> torch.Tensor:
        control = self._require(self._control_buffer, "actuator control write")
        if not isinstance(control, torch.Tensor):
            raise TypeError(
                f"Entity '{self._entity_name}' tensor write_ctrl requires a Torch control buffer"
            )
        return control

    def _write_tensor_ctrl(
        self,
        values: torch.Tensor,
        control: torch.Tensor,
        *,
        env_ids: np.ndarray | slice | None,
        actuator_ids: np.ndarray | Sequence[int] | slice | None,
    ) -> None:
        entity_actuator_ids = self._require(self._actuator_ids, "actuator control write")
        if values.dtype != torch.float32:
            raise TypeError(
                f"Entity '{self._entity_name}' tensor write_ctrl must be float32, "
                f"got {values.dtype}"
            )
        if values.device != control.device:
            raise TypeError(
                f"Entity '{self._entity_name}' tensor write_ctrl must live on "
                f"{control.device}, got {values.device}"
            )
        if not values.is_contiguous():
            raise TypeError(f"Entity '{self._entity_name}' tensor write_ctrl must be contiguous")

        row_index: torch.Tensor | slice
        if env_ids is None:
            row_index = slice(None)
            row_count = control.shape[0]
        elif isinstance(env_ids, slice):
            row_index = env_ids
            row_count = len(range(*env_ids.indices(control.shape[0])))
        else:
            raw_ids = np.asarray(env_ids)
            if (
                raw_ids.ndim != 1
                or not np.issubdtype(raw_ids.dtype, np.integer)
                or np.issubdtype(raw_ids.dtype, np.bool_)
            ):
                raise TypeError(
                    f"Entity '{self._entity_name}' tensor write_ctrl env_ids must be a "
                    f"1-D integer array or slice, got shape={raw_ids.shape}, "
                    f"dtype={raw_ids.dtype}"
                )
            host_ids = np.asarray(raw_ids, dtype=np.int64)
            if np.any(host_ids < 0) or np.any(host_ids >= control.shape[0]):
                raise IndexError(
                    f"Entity '{self._entity_name}' tensor write_ctrl env_ids are out of "
                    f"range for {control.shape[0]} environments: {host_ids.tolist()}"
                )
            if np.unique(host_ids).size != host_ids.size:
                raise ValueError(
                    f"Entity '{self._entity_name}' tensor write_ctrl env_ids contain "
                    f"duplicates: {host_ids.tolist()}"
                )
            row_index = torch.from_numpy(host_ids).to(device=control.device)
            row_count = int(host_ids.size)

        if actuator_ids is None:
            selected_actuator_ids = entity_actuator_ids
        elif isinstance(actuator_ids, slice):
            selected_actuator_ids = entity_actuator_ids[actuator_ids]
        else:
            raw_actuator_ids = np.asarray(actuator_ids)
            if (
                raw_actuator_ids.ndim != 1
                or not np.issubdtype(raw_actuator_ids.dtype, np.integer)
                or np.issubdtype(raw_actuator_ids.dtype, np.bool_)
            ):
                raise TypeError(
                    f"Entity '{self._entity_name}' tensor write_ctrl actuator_ids must be "
                    "a 1-D integer array or slice"
                )
            local_actuator_ids = np.asarray(raw_actuator_ids, dtype=np.intp)
            if np.any(local_actuator_ids < 0) or np.any(
                local_actuator_ids >= len(entity_actuator_ids)
            ):
                raise IndexError(
                    f"Entity '{self._entity_name}' tensor write_ctrl actuator_ids are out "
                    f"of range for {len(entity_actuator_ids)}: {local_actuator_ids.tolist()}"
                )
            if np.unique(local_actuator_ids).size != local_actuator_ids.size:
                raise ValueError(
                    f"Entity '{self._entity_name}' tensor write_ctrl actuator_ids contain "
                    f"duplicates: {local_actuator_ids.tolist()}"
                )
            selected_actuator_ids = entity_actuator_ids[local_actuator_ids]

        column_index = torch.from_numpy(np.asarray(selected_actuator_ids, dtype=np.int64)).to(
            device=control.device
        )
        expected = (row_count, int(column_index.numel()))
        if tuple(values.shape) != expected:
            raise ValueError(
                f"Entity '{self._entity_name}' tensor write_ctrl expected shape "
                f"{expected}, received {tuple(values.shape)}"
            )
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"Entity '{self._entity_name}' tensor write_ctrl got NaN or Inf")

        if isinstance(row_index, slice):
            control[:, column_index] = values
        else:
            control[row_index[:, None], column_index[None, :]] = values


class Entity:
    """Logical entity with cached local-to-backend mappings."""

    def __init__(
        self,
        name: str,
        cfg: EntityCfg,
        backend: SimBackend,
        control_buffer: np.ndarray | torch.Tensor | None = None,
        reset_state: ResetStateTransaction | None = None,
        *,
        default_qpos: np.ndarray | None = None,
        state_read_cache: _EntityStateReadCache | None = None,
    ) -> None:
        if not name:
            raise ValueError("Entity name must be a non-empty string")
        self.name = name
        self._backend_type = backend.backend_type
        self._backend = backend
        self._body_tensor_layouts: dict[tuple[str, ...], np.ndarray] = {}
        self._reset_state = reset_state
        self._reset_root_layout: BackendRootStateLayout | None = None
        self._reset_root_layout_error: str | None = None
        self._reset_joint_qpos_ids: np.ndarray | None = None
        self._reset_joint_qvel_ids: np.ndarray | None = None
        self._joint_model_dof_ids: np.ndarray | None = None
        self._motion_body_ids: np.ndarray | None = None
        self._mocap_body_name: str | None = None
        self._physical_entity: str | None = None
        self._entity_defaults: dict[str, np.ndarray] | None = None
        if reset_state is not None and reset_state.scene_layout is not None:
            layout = reset_state.scene_layout
            physical = cfg.physical_entity
            if physical is None and cfg.root_body_name is not None and "/" in cfg.root_body_name:
                physical = cfg.root_body_name.split("/", 1)[0]
            if physical is None:
                raise ValueError(f"Entity '{name}' requires physical_entity in a composed scene")
            owner = layout.get_entity(physical)
            expected_root = physical + "/" + owner.root_body
            if cfg.root_body_name is not None and cfg.root_body_name != expected_root:
                raise ValueError(
                    f"Entity '{name}' root_body_name must name physical root {expected_root!r}; "
                    f"got {cfg.root_body_name!r}"
                )
            if any(j.kind not in ("hinge", "slide") for j in owner.joints):
                raise NotImplementedError("UniLab entity consumer currently supports scalar joints")
            self._physical_entity = physical
            self._entity_defaults = dict(backend.get_entity_default_state(physical))

        self._joint_names = _normalize_names(name, "joint", cfg.joint_names)
        self._body_names = _normalize_names(name, "body", cfg.body_names)
        self._geom_names = _normalize_names(name, "geom", cfg.geom_names)
        self._site_names = _normalize_names(name, "site", cfg.site_names)
        self._actuator_names = _normalize_names(name, "actuator", cfg.actuator_names)
        if self._physical_entity is not None:
            prefix = self._physical_entity + "/"
            selected = [
                value
                for values in (self._joint_names, self._body_names, self._actuator_names)
                if values
                for value in values
            ]
            if cfg.root_body_name:
                selected.append(cfg.root_body_name)
            if any(not value.startswith(prefix) for value in selected):
                raise ValueError("mapped logical selectors must belong to their physical entity")

        root_body_ids = None
        if cfg.root_body_name is not None:
            if not isinstance(cfg.root_body_name, str) or not cfg.root_body_name:
                raise TypeError(f"Entity '{self.name}' root_body_name must be a non-empty string")
            root_body_ids = self._resolve_ids(
                "root body",
                (cfg.root_body_name,),
                backend.get_body_ids,
            )
        self._root_body_ids = root_body_ids

        joint_pos_ids = joint_vel_ids = None
        if self._joint_names is not None:
            joint_pos_ids = self._resolve_ids(
                "joint position index",
                self._joint_names,
                backend.get_joint_dof_pos_indices,
            )
            joint_vel_ids = self._resolve_ids(
                "joint velocity index",
                self._joint_names,
                backend.get_joint_dof_vel_indices,
            )
        self._joint_dof_ids = (joint_pos_ids, joint_vel_ids)

        body_ids = None
        if self._body_names is not None:
            body_ids = self._resolve_ids("body", self._body_names, backend.get_body_ids)
        self._body_ids = body_ids

        self._geom_ids = None
        if self._geom_names is not None:
            self._geom_ids = self._resolve_enumerated_ids(
                "geom", self._geom_names, backend.get_geom_names
            )

        self._site_ids = None
        if self._site_names is not None:
            self._site_ids = self._resolve_ids("site", self._site_names, backend.get_site_ids)

        actuator_ids = None
        if self._actuator_names is not None:
            actuator_ids = self._resolve_enumerated_ids(
                "actuator", self._actuator_names, backend.get_actuator_names
            )
        self._actuator_ids = actuator_ids

        # Mapped DEVICE_RESIDENT backends deliberately release their legacy
        # host state slots after materialization.  Their immutable public
        # layout already validates the selected addresses without a cold D2H.
        cold_layout_readonly = (
            self._entity_defaults is not None
            and backend.get_tensor_capabilities().execution is TensorExecution.DEVICE_RESIDENT
        )
        if cold_layout_readonly:
            assert reset_state is not None and reset_state.scene_layout is not None
            self._validate_mapped_state_addresses(
                reset_state.scene_layout,
                self._physical_entity,
                joint_pos_ids,
                joint_vel_ids,
                root_body_ids,
                body_ids,
            )
        else:
            self._validate_joint_state(backend, joint_pos_ids, joint_vel_ids)
            self._validate_body_state(backend, root_body_ids, body_ids)
        (
            self._reset_root_layout,
            default_root_state,
            self._reset_root_layout_error,
        ) = self._materialize_root_state(backend, cfg.root_body_name, default_qpos)
        default_joint_pos = self._materialize_default_joint_pos(
            backend,
            joint_pos_ids,
            default_qpos,
        )

        default_joint_vel = self._materialize_default_joint_vel(backend, joint_vel_ids)
        soft_joint_pos_limits = self._materialize_soft_joint_pos_limits(backend, joint_pos_ids)
        gravity_vec_w = self._materialize_gravity_vector(
            backend, root_body_ids, entity_defaults=self._entity_defaults
        )
        actuator_ctrl_range = self._materialize_actuator_ctrl_range(backend, actuator_ids)
        (
            self._actuator_target_joint_names,
            self._joint_to_actuator_local,
        ) = self._materialize_joint_actuator_mapping(backend, actuator_ids)
        if control_buffer is not None:
            expected_control_shape = (backend.num_envs, backend.num_actuators)
            if control_buffer.shape != expected_control_shape:
                raise ValueError(
                    f"Entity '{self.name}' control buffer has shape {control_buffer.shape}; "
                    f"expected {expected_control_shape} on backend '{self._backend_type}'"
                )
            if isinstance(control_buffer, torch.Tensor):
                if control_buffer.dtype != torch.float32:
                    raise TypeError(
                        f"Entity '{self.name}' control buffer must be float32 Torch, "
                        f"got {control_buffer.dtype}"
                    )
            elif not np.issubdtype(control_buffer.dtype, np.floating):
                raise TypeError(
                    f"Entity '{self.name}' control buffer must have floating dtype, "
                    f"got {control_buffer.dtype}"
                )

        self.data = EntityData(
            backend,
            root_body_ids=root_body_ids,
            joint_pos_ids=joint_pos_ids,
            joint_vel_ids=joint_vel_ids,
            default_root_state=default_root_state,
            default_root_state_error=self._reset_root_layout_error,
            default_joint_pos=default_joint_pos,
            default_joint_vel=default_joint_vel,
            soft_joint_pos_limits=soft_joint_pos_limits,
            gravity_vec_w=gravity_vec_w,
            body_ids=body_ids,
            actuator_ids=actuator_ids,
            actuator_ctrl_range=actuator_ctrl_range,
            control_buffer=control_buffer,
            entity_name=self.name,
            backend_type=self._backend_type,
            state_read_cache=(
                state_read_cache if state_read_cache is not None else _EntityStateReadCache()
            ),
        )

    @property
    def motion_body_ids(self) -> np.ndarray:
        """Motion-dataset body columns for the declared entity body order."""
        if self._body_names is None:
            raise self._capability_error(
                "motion body IDs",
                "body_names were not declared in EntityCfg",
            )
        if self._motion_body_ids is None:
            if self._physical_entity is not None:
                # Motion datasets use MuJoCo world-inclusive body numbering.
                # Mapped public body IDs exclude the unowned world row, so the
                # immutable layout gives the same mapping without coupling the
                # cold command materialization to backend-specific getters.
                assert self._reset_state is not None and self._reset_state.scene_layout is not None
                assert self._body_names is not None
                owner = self._reset_state.scene_layout.get_entity(self._physical_entity)
                local_motion_body_ids = {
                    local_name: index for index, local_name in enumerate(owner.body_names)
                }
                self._motion_body_ids = np.asarray(
                    [local_motion_body_ids[name.split("/", 1)[1]] + 1 for name in self._body_names],
                    dtype=np.int32,
                )
            else:
                self._motion_body_ids = self._resolve_ids(
                    "motion body",
                    self._body_names,
                    self._backend.get_motion_body_ids,
                )
            self._motion_body_ids.setflags(write=False)
        return self._motion_body_ids

    def _capability_error(self, capability: str, detail: str) -> NotImplementedError:
        return NotImplementedError(
            f"Entity '{self.name}' capability '{capability}' is unavailable on "
            f"backend '{self._backend_type}': {detail}"
        )

    def joint_tensor_view(self, device: torch.device) -> EntityTensorStateView:
        """Read entity joint state as validated Torch tensors on ``device``.

        The backend owns the transfer topology: host bridges return packed
        copies and device-resident adapters return stable live views.  This
        facade resolves entity joint columns and validates layout, dtype,
        device, and finite values without exposing the backend.
        """
        names = self._joint_names
        if names is None:
            raise self._capability_error(
                "joint tensor state", "joint_names were not declared in EntityCfg"
            )
        if self._joint_dof_ids[0] is None or self._joint_dof_ids[1] is None:
            raise self._capability_error(
                "joint tensor state", "joint position/velocity state was not materialized"
            )
        try:
            qpos_ids = self._backend.get_joint_state_qpos_indices(names)
            qvel_ids = self._backend.get_joint_state_qvel_indices(names)
        except (AttributeError, NotImplementedError) as exc:
            raise self._capability_error("joint tensor state layout", str(exc)) from exc
        expected = (len(names),)
        if qpos_ids.shape != expected or qvel_ids.shape != expected:
            raise ValueError(
                f"Entity '{self.name}' tensor joint-state layout on backend "
                f"'{self._backend_type}' returned shapes {qpos_ids.shape} and "
                f"{qvel_ids.shape}; expected {expected}"
            )
        if np.any(qpos_ids < 0) or np.any(qvel_ids < 0):
            raise ValueError(
                f"Entity '{self.name}' tensor joint-state layout contains negative columns"
            )

        capabilities = self._backend.get_tensor_capabilities()
        state_fields = ("qpos", "qvel")
        if capabilities.execution not in {
            TensorExecution.HOST_BRIDGE,
            TensorExecution.DEVICE_RESIDENT,
        }:
            raise self._capability_error(
                "joint tensor state", f"tensor execution is {capabilities.execution}"
            )
        if not capabilities.state_views or not set(state_fields).issubset(
            capabilities.state_fields
        ):
            raise self._capability_error("joint tensor state", "qpos/qvel views are unavailable")
        if not tensor_device_matches(capabilities.torch_devices, device):
            raise ValueError(
                f"Entity '{self.name}' tensor joint-state device {device} is "
                f"unsupported on backend '{self._backend_type}'; "
                f"accepted={capabilities.torch_devices}"
            )
        try:
            views = self._backend.get_state_views(state_fields, device=device)
        except (AttributeError, KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise type(exc)(
                f"Entity '{self.name}' tensor joint-state read on backend "
                f"'{self._backend_type}': {exc}"
            ) from exc

        result: dict[str, torch.Tensor] = {}
        expected_width = len(names)
        for field, state_field, columns in (
            ("joint_pos", "qpos", qpos_ids),
            ("joint_vel", "qvel", qvel_ids),
        ):
            raw = views[state_field]
            if not isinstance(raw, torch.Tensor):
                raise TypeError(
                    f"Entity '{self.name}' tensor {field} view on backend "
                    f"'{self._backend_type}' is {type(raw).__name__}, expected torch.Tensor"
                )
            expected_shape = (self._backend.num_envs, expected_width)
            if raw.ndim != 2 or raw.shape[0] != expected_shape[0]:
                raise ValueError(
                    f"Entity '{self.name}' tensor {field} view on backend "
                    f"'{self._backend_type}' has shape {tuple(raw.shape)}; expected "
                    f"{expected_shape} after entity column selection"
                )
            if int(np.max(columns)) >= raw.shape[1]:
                raise ValueError(
                    f"Entity '{self.name}' tensor {field} columns exceed backend "
                    f"width {raw.shape[1]}"
                )
            if raw.dtype != torch.float32 or raw.device != torch.device(device):
                raise TypeError(
                    f"Entity '{self.name}' tensor {field} view must be float32 "
                    f"on {device}; got {raw.dtype} on {raw.device}"
                )
            selected = raw[:, np.asarray(columns, dtype=np.intp)]
            if tuple(selected.shape) != expected_shape:
                raise ValueError(
                    f"Entity '{self.name}' tensor {field} selected shape "
                    f"{tuple(selected.shape)} does not match {expected_shape}"
                )
            if not bool(torch.isfinite(selected).all()):
                raise ValueError(f"Entity '{self.name}' tensor {field} has NaN or Inf")
            result[field] = selected
        return EntityTensorStateView(**result)

    def _require_tensor_sensors(
        self, capability: str, device: str | torch.device
    ) -> tuple[TensorLifecycleCapabilities, torch.device]:
        resolved_device = torch.device(device)
        capabilities = self._backend.get_tensor_capabilities()
        if capabilities.execution not in {
            TensorExecution.HOST_BRIDGE,
            TensorExecution.DEVICE_RESIDENT,
        }:
            raise self._capability_error(
                capability, f"tensor execution is {capabilities.execution}"
            )
        if not capabilities.sensor_views:
            raise self._capability_error(capability, "named sensor views are unavailable")
        if not tensor_device_matches(capabilities.torch_devices, resolved_device):
            raise ValueError(
                f"Entity '{self.name}' tensor {capability} device {resolved_device} is "
                f"unsupported on backend '{self._backend_type}'; "
                f"accepted={capabilities.torch_devices}"
            )
        return capabilities, resolved_device

    @staticmethod
    def _validate_tensor_sensor(
        value: Any,
        *,
        entity_name: str,
        backend_type: str,
        sensor_name: str,
        expected_width: int | None,
        num_envs: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Entity '{entity_name}' tensor sensor '{sensor_name}' on backend "
                f"'{backend_type}' is {type(value).__name__}, expected torch.Tensor"
            )
        if (
            value.ndim != 2
            or value.shape[0] != num_envs
            or (expected_width is not None and value.shape[1] != expected_width)
        ):
            rendered = (
                f"({num_envs}, {expected_width})"
                if expected_width is not None
                else f"({num_envs}, width)"
            )
            raise ValueError(
                f"Entity '{entity_name}' tensor sensor '{sensor_name}' on backend "
                f"'{backend_type}' has shape {tuple(value.shape)}; expected {rendered}"
            )
        if value.dtype != torch.float32:
            raise TypeError(
                f"Entity '{entity_name}' tensor sensor '{sensor_name}' must be "
                f"float32; got {value.dtype}"
            )
        if value.device != device:
            raise TypeError(
                f"Entity '{entity_name}' tensor sensor '{sensor_name}' must live on "
                f"{device}; got {value.device}"
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Entity '{entity_name}' tensor sensor '{sensor_name}' has NaN or Inf")
        return value

    def _sensor_tensor_view(
        self,
        sensor_name: str,
        *,
        capability: str,
        device: torch.device,
        expected_width: int | None,
        context: str,
    ) -> torch.Tensor:
        try:
            raw = self._backend.get_sensor_view(sensor_name, device=device)
        except (AttributeError, KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise type(exc)(
                f"Entity '{self.name}' {capability} '{context}' tensor sensor "
                f"'{sensor_name}' read on backend '{self._backend_type}': {exc}"
            ) from exc
        return self._validate_tensor_sensor(
            raw,
            entity_name=self.name,
            backend_type=self._backend_type,
            sensor_name=sensor_name,
            expected_width=expected_width,
            num_envs=self._backend.num_envs,
            device=device,
        )

    def sensor_tensor_views(
        self,
        device: str | torch.device,
        names: Sequence[str],
    ) -> EntityTensorSensorViews:
        """Read explicit named backend sensors as validated Torch tensors.

        Requests are explicit strings (not patterns), so callers resolve regex
        selectors on their own cold path. This narrow direct path is for
        single-name diagnostic or device-resident reads. Host-bridge Manager
        execution must aggregate the same names into one packed read plan; it
        must not call this method per term.
        """
        if isinstance(names, (str, bytes)):
            raise TypeError(
                f"Entity '{self.name}' tensor sensor names must be a sequence of strings"
            )
        sensor_names = tuple(names)
        if not sensor_names:
            raise ValueError(f"Entity '{self.name}' tensor sensor request is empty")
        if any(not isinstance(name, str) or not name for name in sensor_names):
            raise TypeError(f"Entity '{self.name}' tensor sensor names must be non-empty strings")
        if len(set(sensor_names)) != len(sensor_names):
            raise ValueError(
                f"Entity '{self.name}' tensor sensor names must be unique: {sensor_names}"
            )
        capabilities, resolved_device = self._require_tensor_sensors("sensor views", device)
        if capabilities.execution is TensorExecution.HOST_BRIDGE and len(sensor_names) > 1:
            raise NotImplementedError(
                f"Entity '{self.name}' multi-name host-bridge sensor reads require a "
                "scene-owned packed host-bridge plan; direct per-name reads would "
                "create scattered H2D transfers"
            )
        values = {
            name: self._sensor_tensor_view(
                name,
                capability="sensor views",
                device=resolved_device,
                expected_width=None,
                context="request",
            )
            for name in sensor_names
        }
        return EntityTensorSensorViews(names=sensor_names, values=values)

    def body_tensor_view(
        self,
        device: str | torch.device,
        body_names: Sequence[str] | None = None,
    ) -> EntityTensorBodyStateView:
        """Read explicit entity bodies as validated world-frame Torch state.

        Body tensor reads consume the canonical backend ``track_*`` sensor
        contract through the public ``SimBackend`` boundary. The direct API is
        available for device-resident adapters with stable live views. CPU
        host-bridge execution must aggregate the canonical names into one packed
        scene plan before Manager term use.
        """
        if self._body_names is None:
            raise self._capability_error(
                "body tensor state", "body_names were not declared in EntityCfg"
            )
        declared = self._body_names
        if body_names is None:
            requested = declared
        elif isinstance(body_names, (str, bytes)):
            raise TypeError(f"Entity '{self.name}' tensor body names must be a sequence of strings")
        else:
            requested = tuple(body_names)
            if any(not isinstance(name, str) or not name for name in requested):
                raise TypeError(f"Entity '{self.name}' tensor body names must be non-empty strings")
            if len(set(requested)) != len(requested):
                raise ValueError(
                    f"Entity '{self.name}' tensor body names must be unique: {requested}"
                )
            missing = [name for name in requested if name not in set(declared)]
            if missing:
                raise ValueError(
                    f"Entity '{self.name}' tensor body names {missing} are not declared; "
                    f"available={list(declared)}"
                )
        if not requested:
            raise ValueError(f"Entity '{self.name}' tensor body request selected no bodies")

        layout = self._body_tensor_layouts.get(requested)
        if layout is None:
            layout = self._resolve_ids("body tensor state", requested, self._backend.get_body_ids)
            self._body_tensor_layouts[requested] = layout

        capabilities, resolved_device = self._require_tensor_sensors("body tensor state", device)
        if capabilities.execution is TensorExecution.HOST_BRIDGE:
            raise NotImplementedError(
                f"Entity '{self.name}' host-bridge body tensor reads require a "
                "scene-owned packed host-bridge plan; direct per-field/per-body reads "
                "would create scattered H2D transfers"
            )
        del layout

        fields: dict[str, torch.Tensor] = {}
        for field, prefix in _TENSOR_BODY_SENSOR_FIELDS:
            stacked = [
                self._sensor_tensor_view(
                    prefix + body_name,
                    capability="body tensor state",
                    device=resolved_device,
                    expected_width=_TENSOR_SENSOR_WIDTHS[field],
                    context=f"body '{body_name}' {field}",
                )
                for body_name in requested
            ]
            fields[field] = torch.stack(stacked, dim=1)
        return EntityTensorBodyStateView(
            body_names=requested,
            pos_w=fields["pos_w"],
            quat_w=fields["quat_w"],
            lin_vel_w=fields["lin_vel_w"],
            ang_vel_w=fields["ang_vel_w"],
        )

    def _resolve_ids(self, capability: str, names: tuple[str, ...], resolver) -> np.ndarray:
        try:
            values = resolver(names)
        except NotImplementedError as exc:
            raise self._capability_error(capability, str(exc)) from exc
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Entity '{self.name}' could not resolve {capability} names {list(names)} "
                f"on backend '{self._backend_type}': {exc}"
            ) from exc
        return _readonly_ids(
            values,
            expected=len(names),
            label=f"Entity '{self.name}' {capability}",
        )

    def _resolve_enumerated_ids(
        self, capability: str, names: tuple[str, ...], resolver
    ) -> np.ndarray:
        try:
            all_names = tuple(resolver())
        except NotImplementedError as exc:
            raise self._capability_error(capability, str(exc)) from exc
        invalid = [value for value in all_names if not isinstance(value, str)]
        if invalid:
            raise TypeError(
                f"Entity '{self.name}' {capability} name resolver on backend "
                f"'{self._backend_type}' returned non-string names: {invalid}"
            )
        nonempty_names = [value for value in all_names if value]
        if len(set(nonempty_names)) != len(nonempty_names):
            raise ValueError(
                f"Entity '{self.name}' {capability} name resolver on backend "
                f"'{self._backend_type}' returned duplicate names"
            )
        ids_by_name = {value: index for index, value in enumerate(all_names) if value}
        missing = [value for value in names if value not in ids_by_name]
        if missing:
            raise ValueError(
                f"Entity '{self.name}' could not resolve {capability} names {missing} on "
                f"backend '{self._backend_type}'; available={list(all_names)}"
            )
        return _readonly_ids(
            np.asarray([ids_by_name[value] for value in names], dtype=np.int32),
            expected=len(names),
            label=f"Entity '{self.name}' {capability}",
        )

    def _read_state(self, capability: str, getter, *args) -> np.ndarray:
        try:
            return np.asarray(getter(*args))
        except (AttributeError, NotImplementedError) as exc:
            raise self._capability_error(capability, str(exc)) from exc

    def _validate_joint_state(
        self,
        backend: SimBackend,
        pos_ids: np.ndarray | None,
        vel_ids: np.ndarray | None,
    ) -> None:
        for capability, getter, ids in (
            ("joint position state", backend.get_dof_pos, pos_ids),
            ("joint velocity state", backend.get_dof_vel, vel_ids),
        ):
            if ids is None:
                continue
            value = self._read_state(capability, getter)
            if value.ndim != 2 or value.shape[0] != backend.num_envs:
                raise ValueError(
                    f"Entity '{self.name}' capability '{capability}' on backend "
                    f"'{self._backend_type}' returned shape {value.shape}; expected "
                    f"({backend.num_envs}, num_dof)"
                )
            if ids.size and int(np.max(ids)) >= value.shape[1]:
                raise ValueError(
                    f"Entity '{self.name}' capability '{capability}' resolved index "
                    f"{int(np.max(ids))}, but backend '{self._backend_type}' returned "
                    f"only {value.shape[1]} columns"
                )

    def _validate_mapped_state_addresses(
        self,
        layout: Any,
        physical_entity: str | None,
        pos_ids: np.ndarray | None,
        vel_ids: np.ndarray | None,
        root_body_ids: np.ndarray | None,
        body_ids: np.ndarray | None,
    ) -> None:
        """Validate mapped selections against immutable public layout addresses.

        DEVICE_RESIDENT mapped backends intentionally close their legacy host
        state getters after materialization.  Entity construction is cold and
        contract-only here: selected column and body addresses were already
        resolved through public name APIs, so validate their bounds against the
        immutable scene layout rather than copying live CUDA state to host.
        """
        assert physical_entity is not None
        owner = layout.get_entity(physical_entity)
        if pos_ids is not None and pos_ids.size and int(np.max(pos_ids)) >= layout.nq:
            raise ValueError(
                f"Entity '{self.name}' joint position index {int(np.max(pos_ids))} "
                f"exceeds mapped qpos width {layout.nq}"
            )
        if vel_ids is not None and vel_ids.size and int(np.max(vel_ids)) >= layout.nv:
            raise ValueError(
                f"Entity '{self.name}' joint velocity index {int(np.max(vel_ids))} "
                f"exceeds mapped qvel width {layout.nv}"
            )
        for label, ids in (("root body", root_body_ids), ("body", body_ids)):
            if ids is not None and ids.size and int(np.max(ids)) >= layout.nbody:
                raise ValueError(
                    f"Entity '{self.name}' {label} id {int(np.max(ids))} exceeds "
                    f"mapped body count {layout.nbody}"
                )
        expected_root = owner.body_ids[owner.body_names.index(owner.root_body)]
        if root_body_ids is not None and len(root_body_ids) != 1:
            raise ValueError(f"Entity '{self.name}' root body selection must contain one row")
        if root_body_ids is not None and int(root_body_ids[0]) != expected_root:
            raise ValueError(
                f"Entity '{self.name}' selected root body {int(root_body_ids[0])}; "
                f"mapped layout declares {expected_root}"
            )

    def _validate_body_state(
        self,
        backend: SimBackend,
        root_body_ids: np.ndarray | None,
        body_ids: np.ndarray | None,
    ) -> None:
        arrays = [values for values in (root_body_ids, body_ids) if values is not None]
        if not arrays:
            return
        validation_ids = np.unique(np.concatenate(arrays)).astype(np.int32, copy=False)
        for capability, getter, width in (
            ("body position state", backend.get_body_pos_w, 3),
            ("body quaternion state", backend.get_body_quat_w, 4),
            ("body linear velocity state", backend.get_body_lin_vel_w, 3),
            ("body angular velocity state", backend.get_body_ang_vel_w, 3),
        ):
            value = self._read_state(capability, getter, validation_ids)
            expected = (backend.num_envs, len(validation_ids), width)
            if value.shape != expected:
                raise ValueError(
                    f"Entity '{self.name}' capability '{capability}' on backend "
                    f"'{self._backend_type}' returned shape {value.shape}; expected {expected}"
                )
            if not np.isfinite(value).all():
                raise ValueError(
                    f"Entity '{self.name}' capability '{capability}' on backend "
                    f"'{self._backend_type}' returned NaN or Inf"
                )
        if root_body_ids is None:
            return
        for capability, getter in (
            ("body-frame linear velocity state", backend.get_body_lin_vel_b),
            ("body-frame angular velocity state", backend.get_body_ang_vel_b),
        ):
            value = self._read_state(capability, getter, root_body_ids)
            expected = (backend.num_envs, len(root_body_ids), 3)
            if value.shape != expected:
                raise ValueError(
                    f"Entity '{self.name}' capability '{capability}' on backend "
                    f"'{self._backend_type}' returned shape {value.shape}; expected {expected}"
                )
            if not np.isfinite(value).all():
                raise ValueError(
                    f"Entity '{self.name}' capability '{capability}' on backend "
                    f"'{self._backend_type}' returned NaN or Inf"
                )

    def _materialize_actuator_ctrl_range(
        self, backend: SimBackend, actuator_ids: np.ndarray | None
    ) -> np.ndarray | None:
        if actuator_ids is None:
            return None
        ranges = self._read_state("actuator control range", backend.get_actuator_ctrl_range)
        expected = (backend.num_actuators, 2)
        if ranges.shape != expected:
            raise ValueError(
                f"Entity '{self.name}' capability 'actuator control range' on backend "
                f"'{self._backend_type}' returned shape {ranges.shape}; expected {expected}"
            )
        selected = np.array(ranges[_as_column_index(actuator_ids)], copy=True)
        selected.setflags(write=False)
        return selected

    def _materialize_default_joint_pos(
        self,
        backend: SimBackend,
        joint_pos_ids: np.ndarray | None,
        default_qpos: np.ndarray | None,
    ) -> np.ndarray | None:
        if joint_pos_ids is None:
            return None
        if self._entity_defaults is not None:
            return self._selected_entity_default_joints("joint_positions")
        current = self._read_state("joint position state", backend.get_dof_pos)
        if default_qpos is None:
            defaults = self._read_state("default joint position", backend.get_default_dof_pos)
            if defaults.shape != current.shape[1:]:
                raise ValueError(
                    f"Entity '{self.name}' capability 'default joint position' on backend "
                    f"'{self._backend_type}' returned shape {defaults.shape}; expected "
                    f"{current.shape[1:]} to match get_dof_pos()"
                )
            selected = np.asarray(defaults[_as_column_index(joint_pos_ids)])
        else:
            assert self._joint_names is not None
            defaults = self._validate_root_default_vector(default_qpos, "selected default qpos")
            try:
                state_qpos_ids = backend.get_joint_state_qpos_indices(self._joint_names)
            except (AttributeError, NotImplementedError) as exc:
                raise self._capability_error("default joint-state layout", str(exc)) from exc
            resolved_qpos_ids = _readonly_ids(
                state_qpos_ids,
                expected=len(self._joint_names),
                label=f"Entity '{self.name}' default qpos",
            )
            if resolved_qpos_ids.size and int(np.max(resolved_qpos_ids)) >= defaults.size:
                raise ValueError(
                    f"Entity '{self.name}' default qpos layout exceeds backend "
                    f"'{self._backend_type}' width {defaults.size}: {resolved_qpos_ids.tolist()}"
                )
            selected = np.asarray(defaults[_as_column_index(resolved_qpos_ids)])
            self._reset_joint_qpos_ids = resolved_qpos_ids
        materialized = np.broadcast_to(
            selected,
            (backend.num_envs, len(joint_pos_ids)),
        ).astype(current.dtype, copy=True)
        materialized.setflags(write=False)
        return materialized

    def _materialize_soft_joint_pos_limits(
        self,
        backend: SimBackend,
        joint_pos_ids: np.ndarray | None,
    ) -> np.ndarray | None:
        if joint_pos_ids is None:
            return None
        if self._physical_entity is not None:
            result = np.asarray(backend.get_joint_range(names=self._joint_names)).copy()
            result.setflags(write=False)
            return result
        try:
            raw_ranges = backend.get_joint_range()
        except (AttributeError, NotImplementedError) as exc:
            raise self._capability_error("joint position limits", str(exc)) from exc
        if raw_ranges is None:
            return None
        ranges = np.asarray(raw_ranges)
        if ranges.ndim != 2 or ranges.shape[1] != 2:
            raise ValueError(
                f"Entity '{self.name}' capability 'joint position limits' on backend "
                f"'{self._backend_type}' returned shape {ranges.shape}; expected (num_dof, 2)"
            )
        if joint_pos_ids.size and int(np.max(joint_pos_ids)) >= ranges.shape[0]:
            raise ValueError(
                f"Entity '{self.name}' capability 'joint position limits' resolved index "
                f"{int(np.max(joint_pos_ids))}, but backend '{self._backend_type}' returned "
                f"only {ranges.shape[0]} rows"
            )
        selected = np.array(ranges[_as_column_index(joint_pos_ids)], copy=True)
        selected.setflags(write=False)
        return selected

    def _materialize_root_state(
        self,
        backend: SimBackend,
        root_body_name: str | None,
        default_qpos: np.ndarray | None,
    ) -> tuple[BackendRootStateLayout | None, np.ndarray | None, str | None]:
        if root_body_name is None:
            return None, None, "root_body_name was not declared in EntityCfg"
        if self._entity_defaults is not None:
            defaults = self._entity_defaults
            value = np.concatenate((defaults["root_pose"], defaults["root_velocity"]), axis=1)
            value.setflags(write=False)
            return None, value, None
        try:
            layout = backend.get_root_state_layout(root_body_name)
        except (AttributeError, NotImplementedError) as exc:
            return None, None, str(exc)
        if not isinstance(layout, BackendRootStateLayout):
            raise TypeError(
                f"Entity '{self.name}' capability 'root-state layout' on backend "
                f"'{self._backend_type}' must return BackendRootStateLayout, got "
                f"{type(layout).__name__}"
            )
        try:
            qpos = backend.get_default_qpos() if default_qpos is None else default_qpos
            qvel = backend.get_init_qvel()
        except (AttributeError, NotImplementedError) as exc:
            return None, None, str(exc)
        qpos_default = self._validate_root_default_vector(qpos, "default qpos")
        qvel_default = self._validate_root_default_vector(qvel, "initial qvel")
        qpos_indices = np.asarray(layout.qpos_indices, dtype=np.intp)
        qvel_indices = np.asarray(layout.qvel_indices, dtype=np.intp)
        if np.any(qpos_indices >= qpos_default.size):
            raise ValueError(
                f"Entity '{self.name}' root qpos layout exceeds backend "
                f"'{self._backend_type}' width {qpos_default.size}: {qpos_indices.tolist()}"
            )
        if np.any(qvel_indices >= qvel_default.size):
            raise ValueError(
                f"Entity '{self.name}' root qvel layout exceeds backend "
                f"'{self._backend_type}' width {qvel_default.size}: {qvel_indices.tolist()}"
            )

        pose = np.asarray(qpos_default[qpos_indices])
        quaternion = pose[3:7]
        norm = float(np.linalg.norm(quaternion))
        if not np.isclose(norm, 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError(
                f"Entity '{self.name}' default root quaternion on backend "
                f"'{self._backend_type}' must be unit length; norm={norm}"
            )
        generalized_velocity = np.asarray(qvel_default[qvel_indices])
        velocity_w = np.array(generalized_velocity, copy=True)
        velocity_w[3:6] = np_quat_apply(quaternion, generalized_velocity[3:6])
        root_state = np.concatenate((pose, velocity_w))
        materialized = np.broadcast_to(root_state, (backend.num_envs, 13)).copy()
        materialized.setflags(write=False)
        return layout, materialized, None

    def _validate_root_default_vector(self, value: np.ndarray, capability: str) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"Entity '{self.name}' capability '{capability}' on backend "
                f"'{self._backend_type}' must return np.ndarray, got {type(value).__name__}"
            )
        if value.ndim != 1 or not np.issubdtype(value.dtype, np.floating):
            raise TypeError(
                f"Entity '{self.name}' capability '{capability}' on backend "
                f"'{self._backend_type}' must be a 1-D floating array; got "
                f"shape={value.shape}, dtype={value.dtype}"
            )
        if not np.isfinite(value).all():
            raise ValueError(
                f"Entity '{self.name}' capability '{capability}' on backend "
                f"'{self._backend_type}' returned NaN or Inf"
            )
        return value

    def _materialize_default_joint_vel(
        self, backend: SimBackend, joint_vel_ids: np.ndarray | None
    ) -> np.ndarray | None:
        if joint_vel_ids is None:
            return None
        if self._entity_defaults is not None:
            return self._selected_entity_default_joints("joint_velocities")
        current = self._read_state("joint velocity state", backend.get_dof_vel)
        materialized = np.zeros(
            (backend.num_envs, len(joint_vel_ids)),
            dtype=current.dtype,
        )
        materialized.setflags(write=False)
        return materialized

    def _selected_entity_default_joints(self, field: str) -> np.ndarray:
        assert self._reset_state is not None and self._reset_state.scene_layout is not None
        assert self._physical_entity is not None and self._entity_defaults is not None
        owner = self._reset_state.scene_layout.get_entity(self._physical_entity)
        names = [joint.name for joint in owner.joints]
        selected = [names.index(name.split("/", 1)[1]) for name in self._joint_names or ()]
        result = self._entity_defaults[field][:, selected].copy()
        result.setflags(write=False)
        return result

    def _materialize_gravity_vector(
        self,
        backend: SimBackend,
        root_body_ids: np.ndarray | None,
        *,
        entity_defaults: Mapping[str, np.ndarray] | None = None,
    ) -> np.ndarray | None:
        if root_body_ids is None:
            return None
        if entity_defaults is not None:
            quat = np.asarray(entity_defaults["root_pose"][:, 3:7])
        else:
            quat = self._read_state(
                "root body quaternion state", backend.get_body_quat_w, root_body_ids
            )
        gravity = np.zeros((backend.num_envs, 3), dtype=quat.dtype)
        gravity[:, 2] = -1.0
        gravity.setflags(write=False)
        return gravity

    def _materialize_joint_actuator_mapping(
        self, backend: SimBackend, actuator_ids: np.ndarray | None
    ) -> tuple[tuple[str, ...] | None, np.ndarray | None]:
        if actuator_ids is None or self._joint_names is None:
            return None, None
        try:
            all_target_names = tuple(backend.get_actuator_joint_names())
        except NotImplementedError as exc:
            raise self._capability_error("actuator target joint", str(exc)) from exc
        if len(all_target_names) != backend.num_actuators:
            raise ValueError(
                f"Entity '{self.name}' capability 'actuator target joint' on backend "
                f"'{self._backend_type}' returned {len(all_target_names)} names for "
                f"{backend.num_actuators} actuators"
            )
        target_names = tuple(all_target_names[int(index)] for index in actuator_ids)
        if any(not isinstance(name, str) or not name for name in target_names):
            raise ValueError(
                f"Entity '{self.name}' actuator target joint names must be non-empty strings; "
                f"got {target_names}"
            )
        if len(set(target_names)) != len(target_names):
            raise ValueError(
                f"Entity '{self.name}' actuator target joints must be unique for position "
                f"control; got {target_names}"
            )
        joint_index_by_name = {name: index for index, name in enumerate(self._joint_names)}
        missing = [name for name in target_names if name not in joint_index_by_name]
        if missing:
            raise ValueError(
                f"Entity '{self.name}' actuators target joints outside its declared joint "
                f"partition on backend '{self._backend_type}': {missing}"
            )
        joint_to_actuator = np.full(len(self._joint_names), -1, dtype=np.int32)
        for actuator_local_id, joint_name in enumerate(target_names):
            joint_to_actuator[joint_index_by_name[joint_name]] = actuator_local_id
        joint_to_actuator.setflags(write=False)
        return target_names, joint_to_actuator

    def _require_names(self, kind: str, names: tuple[str, ...] | None) -> tuple[str, ...]:
        if names is None:
            raise self._capability_error(kind, "the namespace was not declared in EntityCfg")
        return names

    def _unsupported_names(self, kind: str) -> NoReturn:
        raise self._capability_error(kind, "SimBackend does not declare this namespace")

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._require_names("joint", self._joint_names)

    @property
    def body_names(self) -> tuple[str, ...]:
        return self._require_names("body", self._body_names)

    @property
    def geom_names(self) -> tuple[str, ...]:
        return self._require_names("geom", self._geom_names)

    @property
    def site_names(self) -> tuple[str, ...]:
        return self._require_names("site", self._site_names)

    @property
    def actuator_names(self) -> tuple[str, ...]:
        return self._require_names("actuator", self._actuator_names)

    @property
    def tendon_names(self) -> tuple[str, ...]:
        return self._unsupported_names("tendon")

    @property
    def camera_names(self) -> tuple[str, ...]:
        return self._unsupported_names("camera")

    @property
    def light_names(self) -> tuple[str, ...]:
        return self._unsupported_names("light")

    @property
    def material_names(self) -> tuple[str, ...]:
        return self._unsupported_names("material")

    @property
    def texture_names(self) -> tuple[str, ...]:
        return self._unsupported_names("texture")

    @property
    def pair_names(self) -> tuple[str, ...]:
        return self._unsupported_names("pair")

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def num_bodies(self) -> int:
        return len(self.body_names)

    @property
    def num_geoms(self) -> int:
        return len(self.geom_names)

    @property
    def num_sites(self) -> int:
        return len(self.site_names)

    @property
    def num_actuators(self) -> int:
        return len(self.actuator_names)

    @property
    def num_tendons(self) -> int:
        return len(self.tendon_names)

    @property
    def num_cameras(self) -> int:
        return len(self.camera_names)

    @property
    def num_lights(self) -> int:
        return len(self.light_names)

    @property
    def num_materials(self) -> int:
        return len(self.material_names)

    @property
    def num_textures(self) -> int:
        return len(self.texture_names)

    @property
    def num_pairs(self) -> int:
        return len(self.pair_names)

    def _find(
        self,
        kind: str,
        names: tuple[str, ...] | None,
        keys: str | Sequence[str],
        preserve_order: bool,
    ) -> tuple[list[int], list[str]]:
        return _resolve_matching_names(keys, self._require_names(kind, names), preserve_order)

    def find_joints(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        return self._find("joint", self._joint_names, keys, preserve_order)

    def find_joints_by_actuator_names(
        self, keys: str | Sequence[str]
    ) -> tuple[list[int], list[str]]:
        """Resolve actuator-target joint patterns in natural entity joint order."""
        target_names = self._actuator_target_joint_names
        if target_names is None:
            raise self._capability_error(
                "actuator target joint",
                "joint_names and actuator_names must both be declared in EntityCfg",
            )
        target_set = set(target_names)
        natural_ids = [index for index, name in enumerate(self.joint_names) if name in target_set]
        natural_names = [self.joint_names[index] for index in natural_ids]
        matched_ids, matched_names = _resolve_matching_names(keys, natural_names, False)
        return [natural_ids[index] for index in matched_ids], matched_names

    def set_joint_position_target(
        self,
        target: np.ndarray,
        joint_ids: np.ndarray | Sequence[int] | slice | None = None,
        env_ids: np.ndarray | slice | None = None,
    ) -> None:
        """Map entity-local joint targets to the env-owned actuator control buffer."""
        joint_to_actuator = self._joint_to_actuator_local
        if joint_to_actuator is None:
            raise self._capability_error(
                "joint position target",
                "joint-to-actuator metadata was not materialized",
            )
        local_joint_ids = self._normalize_local_joint_ids(
            joint_ids,
            capability="joint position target",
        )
        actuator_ids = joint_to_actuator[local_joint_ids]
        if np.any(actuator_ids < 0):
            passive_names = [
                self.joint_names[int(index)] for index in local_joint_ids[actuator_ids < 0]
            ]
            raise NotImplementedError(
                f"Entity '{self.name}' capability 'joint position target' is unavailable "
                f"for passive joints on backend '{self._backend_type}': {passive_names}"
            )
        self.data.write_ctrl(target, env_ids, actuator_ids=actuator_ids)

    def _set_joint_control_target(
        self,
        target: np.ndarray,
        joint_ids: np.ndarray | Sequence[int] | slice | None,
        env_ids: np.ndarray | slice | None,
        *,
        capability: str,
    ) -> None:
        """Map a joint target to the entity-owned actuator control buffer.

        The action term selects the semantic transmission while this helper
        owns only the stable joint-to-actuator mapping and validation. The
        backend remains responsible for interpreting each actuator's control
        signal, and no backend-private model access is needed in the manager
        hot path.
        """
        joint_to_actuator = self._joint_to_actuator_local
        if joint_to_actuator is None:
            raise self._capability_error(
                capability, "joint-to-actuator metadata was not materialized"
            )
        local_joint_ids = self._normalize_local_joint_ids(joint_ids, capability=capability)
        actuator_ids = joint_to_actuator[local_joint_ids]
        if np.any(actuator_ids < 0):
            passive_names = [
                self.joint_names[int(index)] for index in local_joint_ids[actuator_ids < 0]
            ]
            raise NotImplementedError(
                f"Entity '{self.name}' capability '{capability}' is unavailable "
                f"for passive joints on backend '{self._backend_type}': {passive_names}"
            )
        self.data.write_ctrl(target, env_ids, actuator_ids=actuator_ids)

    def set_joint_velocity_target(
        self,
        target: np.ndarray,
        joint_ids: np.ndarray | Sequence[int] | slice | None = None,
        env_ids: np.ndarray | slice | None = None,
    ) -> None:
        """Map joint velocity targets to the entity actuator control buffer."""
        self._set_joint_control_target(
            target,
            joint_ids,
            env_ids,
            capability="joint velocity target",
        )

    def set_joint_effort_target(
        self,
        target: np.ndarray,
        joint_ids: np.ndarray | Sequence[int] | slice | None = None,
        env_ids: np.ndarray | slice | None = None,
    ) -> None:
        """Map joint effort (torque) targets to the actuator control buffer."""
        self._set_joint_control_target(
            target,
            joint_ids,
            env_ids,
            capability="joint effort target",
        )

    def bind_body_state_copy(
        self,
        body_ids: np.ndarray | Sequence[int] | slice | None = None,
    ) -> BodyStateCopyFn:
        """Bind entity-local body columns to the backend copy contract on the cold path."""
        if self._body_ids is None:
            raise self._capability_error(
                "body-state copy",
                "body_names were not declared in EntityCfg",
            )
        local_ids = self._normalize_local_body_ids(body_ids, capability="body-state copy")
        if local_ids.size == 0:
            raise ValueError(f"Entity '{self.name}' body-state copy selected no bodies")
        backend_ids = np.array(self._body_ids[local_ids], copy=True, dtype=np.int32)
        backend_ids.setflags(write=False)
        return partial(self._backend.copy_body_state_w, backend_ids)

    def write_root_state_to_sim(
        self,
        root_state: np.ndarray,
        env_ids: np.ndarray | slice | None = None,
    ) -> None:
        """Stage a 13-D world-frame root state in the active reset transaction."""
        if self._physical_entity is not None:
            if root_state.ndim != 2 or root_state.shape[1] != 13:
                raise ValueError("root_state must have 13 columns")
            self._stage_entity_write(
                env_ids, root_pose=root_state[:, :7], root_velocity=root_state[:, 7:]
            )
            return
        reset_state, layout = self._require_root_state_write()
        resolved_env_ids = self._normalize_reset_env_ids(env_ids)
        reset_state.write_root_state(
            resolved_env_ids,
            layout,
            root_state,
            term_name=f"{self.name}.write_root_state_to_sim",
        )

    def bind_actuator_gain_write(
        self,
        actuator_ids: np.ndarray | Sequence[int] | slice | None = None,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bind selected actuator columns and immutable gain defaults on the cold path."""
        if self._reset_state is None:
            raise self._capability_error(
                "reset actuator-gain write",
                "EntityScene was materialized without an env-owned reset transaction",
            )
        if self._actuator_ids is None:
            raise self._capability_error(
                "reset actuator-gain write",
                "actuator_names were not declared in EntityCfg",
            )
        local_ids = self._normalize_local_actuator_ids(
            actuator_ids,
            capability="reset actuator-gain write",
        )
        if local_ids.size == 0:
            raise ValueError(
                f"Entity '{self.name}' reset actuator-gain write selected no actuators"
            )
        backend_ids = self._actuator_ids[local_ids]
        _, default_kp, default_kd = self._reset_state.bind_actuator_gain_write(
            backend_ids,
            term_name=f"{term_name}:{self.name}",
        )
        bound_local_ids = np.array(local_ids, copy=True)
        bound_local_ids.setflags(write=False)
        return bound_local_ids, default_kp, default_kd

    def write_actuator_gains_to_sim(
        self,
        kp: np.ndarray,
        kd: np.ndarray,
        actuator_ids: np.ndarray | Sequence[int] | slice | None = None,
        env_ids: np.ndarray | slice | None = None,
        *,
        term_name: str = "pd_gains",
    ) -> None:
        """Stage entity-local actuator gains in the active reset transaction."""
        if self._reset_state is None or self._actuator_ids is None:
            raise self._capability_error(
                "reset actuator-gain write",
                "actuator metadata or the env-owned reset transaction was not materialized",
            )
        local_ids = self._normalize_local_actuator_ids(
            actuator_ids,
            capability="reset actuator-gain write",
        )
        resolved_env_ids = self._normalize_reset_env_ids(env_ids)
        self._reset_state.write_actuator_gains(
            resolved_env_ids,
            self._actuator_ids[local_ids],
            kp,
            kd,
            term_name=f"{term_name}:{self.name}",
        )

    def bind_geom_size_write(
        self,
        geom_ids: np.ndarray | Sequence[int] | slice | None = None,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bind entity-local geom_size columns and immutable defaults."""
        transaction, local_ids, model_ids = self._reset_model_field_ids(
            "geom",
            geom_ids,
            term_name=term_name,
        )
        _, defaults = transaction.bind_geom_size_write(
            model_ids,
            term_name=f"{term_name}:{self.name}",
        )
        return self._readonly_local_binding(local_ids, defaults)

    def write_geom_size_to_sim(
        self,
        values: np.ndarray,
        geom_ids: np.ndarray | Sequence[int] | slice | None = None,
        env_ids: np.ndarray | slice | None = None,
        *,
        term_name: str = "geom_size",
    ) -> None:
        """Stage geom_size values through the reset transaction."""
        transaction, _, model_ids = self._reset_model_field_ids(
            "geom",
            geom_ids,
            term_name=term_name,
        )
        transaction.write_geom_size(
            self._normalize_reset_env_ids(env_ids),
            model_ids,
            values,
            term_name=f"{term_name}:{self.name}",
        )

    def bind_geom_solref_write(
        self,
        geom_ids: np.ndarray | Sequence[int] | slice | None = None,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bind entity-local geom_solref columns and immutable defaults."""
        transaction, local_ids, model_ids = self._reset_model_field_ids(
            "geom",
            geom_ids,
            term_name=term_name,
        )
        _, defaults = transaction.bind_geom_solref_write(
            model_ids,
            term_name=f"{term_name}:{self.name}",
        )
        return self._readonly_local_binding(local_ids, defaults)

    def write_geom_solref_to_sim(
        self,
        values: np.ndarray,
        geom_ids: np.ndarray | Sequence[int] | slice | None = None,
        env_ids: np.ndarray | slice | None = None,
        *,
        term_name: str = "geom_solref",
    ) -> None:
        """Stage geom_solref values through the reset transaction."""
        transaction, _, model_ids = self._reset_model_field_ids(
            "geom",
            geom_ids,
            term_name=term_name,
        )
        transaction.write_geom_solref(
            self._normalize_reset_env_ids(env_ids),
            model_ids,
            values,
            term_name=f"{term_name}:{self.name}",
        )

    def bind_geom_solimp_write(
        self,
        geom_ids: np.ndarray | Sequence[int] | slice | None = None,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bind entity-local geom_solimp columns and immutable defaults."""
        transaction, local_ids, model_ids = self._reset_model_field_ids(
            "geom",
            geom_ids,
            term_name=term_name,
        )
        _, defaults = transaction.bind_geom_solimp_write(
            model_ids,
            term_name=f"{term_name}:{self.name}",
        )
        return self._readonly_local_binding(local_ids, defaults)

    def write_geom_solimp_to_sim(
        self,
        values: np.ndarray,
        geom_ids: np.ndarray | Sequence[int] | slice | None = None,
        env_ids: np.ndarray | slice | None = None,
        *,
        term_name: str = "geom_solimp",
    ) -> None:
        """Stage geom_solimp values through the reset transaction."""
        transaction, _, model_ids = self._reset_model_field_ids(
            "geom",
            geom_ids,
            term_name=term_name,
        )
        transaction.write_geom_solimp(
            self._normalize_reset_env_ids(env_ids),
            model_ids,
            values,
            term_name=f"{term_name}:{self.name}",
        )

    def bind_joint_damping_write(
        self,
        joint_ids: np.ndarray | Sequence[int] | slice | None = None,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bind entity-local joint_damping columns and immutable defaults."""
        transaction, local_ids, model_ids = self._reset_model_field_ids(
            "joint",
            joint_ids,
            term_name=term_name,
        )
        _, defaults = transaction.bind_dof_damping_write(
            model_ids,
            term_name=f"{term_name}:{self.name}",
        )
        return self._readonly_local_binding(local_ids, defaults)

    def write_joint_damping_to_sim(
        self,
        values: np.ndarray,
        joint_ids: np.ndarray | Sequence[int] | slice | None = None,
        env_ids: np.ndarray | slice | None = None,
        *,
        term_name: str = "joint_damping",
    ) -> None:
        """Stage joint_damping values through the reset transaction."""
        transaction, _, model_ids = self._reset_model_field_ids(
            "joint",
            joint_ids,
            term_name=term_name,
        )
        transaction.write_dof_damping(
            self._normalize_reset_env_ids(env_ids),
            model_ids,
            values,
            term_name=f"{term_name}:{self.name}",
        )

    def bind_joint_frictionloss_write(
        self,
        joint_ids: np.ndarray | Sequence[int] | slice | None = None,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bind entity-local joint_frictionloss columns and immutable defaults."""
        transaction, local_ids, model_ids = self._reset_model_field_ids(
            "joint",
            joint_ids,
            term_name=term_name,
        )
        _, defaults = transaction.bind_dof_frictionloss_write(
            model_ids,
            term_name=f"{term_name}:{self.name}",
        )
        return self._readonly_local_binding(local_ids, defaults)

    def write_joint_frictionloss_to_sim(
        self,
        values: np.ndarray,
        joint_ids: np.ndarray | Sequence[int] | slice | None = None,
        env_ids: np.ndarray | slice | None = None,
        *,
        term_name: str = "joint_frictionloss",
    ) -> None:
        """Stage joint_frictionloss values through the reset transaction."""
        transaction, _, model_ids = self._reset_model_field_ids(
            "joint",
            joint_ids,
            term_name=term_name,
        )
        transaction.write_dof_frictionloss(
            self._normalize_reset_env_ids(env_ids),
            model_ids,
            values,
            term_name=f"{term_name}:{self.name}",
        )

    def _reset_model_field_ids(
        self,
        kind: str,
        ids: np.ndarray | Sequence[int] | slice | None,
        *,
        term_name: str,
    ) -> tuple[ResetStateTransaction, np.ndarray, np.ndarray]:
        if self._reset_state is None:
            raise self._capability_error(term_name, "no env-owned reset transaction")
        if kind == "geom":
            if self._geom_ids is None:
                raise self._capability_error(term_name, "geom names were not declared")
            local_ids = self._normalize_local_geom_ids(ids, capability=term_name)
            model_ids = self._geom_ids[local_ids]
        else:
            local_ids = self._normalize_local_joint_ids(ids, capability=term_name)
            model_ids = self._materialize_joint_model_dof_ids()[local_ids]
        if local_ids.size == 0:
            raise ValueError(f"Entity '{self.name}' {term_name} selected no {kind}s")
        return self._reset_state, local_ids, model_ids

    def bind_mocap_pose_write(self, body_name: str, *, term_name: str) -> np.ndarray:
        """Bind an explicitly named mocap body, independently of the floating root."""
        if self._reset_state is None:
            raise self._capability_error(term_name, "no env-owned reset transaction")
        if body_name not in self.body_names:
            raise ValueError(f"Entity '{self.name}' does not expose mocap body {body_name!r}")
        if self._mocap_body_name is not None and self._mocap_body_name != body_name:
            raise ValueError(
                f"Entity '{self.name}' already bound mocap body {self._mocap_body_name!r}"
            )
        binding = self._reset_state.bind_mocap_pose(body_name)
        self._mocap_body_name = body_name
        return binding.default_pose.copy()

    def read_mocap_pose(self) -> np.ndarray:
        """Read full-batch mocap poses, including pending reset writes."""
        if self._reset_state is None or self._mocap_body_name is None:
            raise RuntimeError(f"Entity '{self.name}' must bind its mocap pose before reading")
        return self._reset_state.read_mocap_pose(self._mocap_body_name)

    def write_mocap_pose_to_sim(
        self,
        poses: np.ndarray,
        env_ids: np.ndarray | slice | None = None,
        *,
        term_name: str = "mocap_pose",
    ) -> None:
        """Stage poses to commit after the ordinary reset state upload."""
        if self._reset_state is None or self._mocap_body_name is None:
            raise RuntimeError(f"Entity '{self.name}' must bind its mocap pose before writing")
        self._reset_state.write_mocap_pose(
            self._mocap_body_name,
            self._normalize_reset_env_ids(env_ids),
            poses,
            term_name=f"{term_name}:{self.name}",
        )

    def bind_joint_armature_write(
        self,
        joint_ids: np.ndarray | Sequence[int] | slice | None = None,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bind entity-local joints and immutable default DOF armatures."""
        if self._reset_state is None:
            raise self._capability_error(
                "reset joint-armature write",
                "EntityScene was materialized without an env-owned reset transaction",
            )
        model_dof_ids = self._materialize_joint_model_dof_ids()
        local_ids = self._normalize_local_joint_ids(
            joint_ids,
            capability="reset joint-armature write",
        )
        if local_ids.size == 0:
            raise ValueError(f"Entity '{self.name}' reset joint-armature write selected no joints")
        _, defaults = self._reset_state.bind_dof_armature_write(
            model_dof_ids[local_ids],
            term_name=f"{term_name}:{self.name}",
        )
        return self._readonly_local_binding(local_ids, defaults)

    def write_joint_armature_to_sim(
        self,
        values: np.ndarray,
        joint_ids: np.ndarray | Sequence[int] | slice | None = None,
        env_ids: np.ndarray | slice | None = None,
        *,
        term_name: str = "joint_armature",
    ) -> None:
        """Stage selected entity joint armatures in the active reset transaction."""
        if self._reset_state is None:
            raise self._capability_error(
                "reset joint-armature write",
                "EntityScene was materialized without an env-owned reset transaction",
            )
        model_dof_ids = self._materialize_joint_model_dof_ids()
        local_ids = self._normalize_local_joint_ids(
            joint_ids,
            capability="reset joint-armature write",
        )
        self._reset_state.write_dof_armature(
            self._normalize_reset_env_ids(env_ids),
            model_dof_ids[local_ids],
            values,
            term_name=f"{term_name}:{self.name}",
        )

    def bind_geom_friction_write(
        self,
        geom_ids: np.ndarray | Sequence[int] | slice | None = None,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bind entity-local geoms and immutable default friction vectors."""
        if self._reset_state is None or self._geom_ids is None:
            raise self._capability_error(
                "reset geom-friction write",
                "geom metadata or the env-owned reset transaction was not materialized",
            )
        local_ids = self._normalize_local_geom_ids(
            geom_ids,
            capability="reset geom-friction write",
        )
        if local_ids.size == 0:
            raise ValueError(f"Entity '{self.name}' reset geom-friction write selected no geoms")
        _, defaults = self._reset_state.bind_geom_friction_write(
            self._geom_ids[local_ids],
            term_name=f"{term_name}:{self.name}",
        )
        return self._readonly_local_binding(local_ids, defaults)

    def write_geom_friction_to_sim(
        self,
        values: np.ndarray,
        geom_ids: np.ndarray | Sequence[int] | slice | None = None,
        env_ids: np.ndarray | slice | None = None,
        *,
        term_name: str = "geom_friction",
    ) -> None:
        """Stage selected entity geom friction in the active reset transaction."""
        if self._reset_state is None or self._geom_ids is None:
            raise self._capability_error(
                "reset geom-friction write",
                "geom metadata or the env-owned reset transaction was not materialized",
            )
        local_ids = self._normalize_local_geom_ids(
            geom_ids,
            capability="reset geom-friction write",
        )
        self._reset_state.write_geom_friction(
            self._normalize_reset_env_ids(env_ids),
            self._geom_ids[local_ids],
            values,
            term_name=f"{term_name}:{self.name}",
        )

    def bind_body_mass_write(
        self,
        body_ids: np.ndarray | Sequence[int] | slice | None = None,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bind entity-local body columns and immutable default masses."""
        reset_state, local_ids, backend_ids = self._bind_body_randomization(
            body_ids,
            capability="reset body-mass write",
        )
        _, defaults = reset_state.bind_body_mass_write(
            backend_ids,
            term_name=f"{term_name}:{self.name}",
        )
        return self._readonly_local_binding(local_ids, defaults)

    def write_body_mass_to_sim(
        self,
        values: np.ndarray,
        body_ids: np.ndarray | Sequence[int] | slice | None = None,
        env_ids: np.ndarray | slice | None = None,
        *,
        term_name: str = "randomize_rigid_body_mass",
    ) -> None:
        """Stage selected entity body masses in the active reset transaction."""
        reset_state, _, backend_ids = self._bind_body_randomization(
            body_ids,
            capability="reset body-mass write",
        )
        reset_state.write_body_mass(
            self._normalize_reset_env_ids(env_ids),
            backend_ids,
            values,
            term_name=f"{term_name}:{self.name}",
        )

    def bind_body_ipos_write(
        self,
        body_ids: np.ndarray | Sequence[int] | slice | None = None,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bind entity-local body columns and immutable inertial positions."""
        reset_state, local_ids, backend_ids = self._bind_body_randomization(
            body_ids,
            capability="reset body-ipos write",
        )
        _, defaults = reset_state.bind_body_ipos_write(
            backend_ids,
            term_name=f"{term_name}:{self.name}",
        )
        return self._readonly_local_binding(local_ids, defaults)

    def write_body_ipos_to_sim(
        self,
        values: np.ndarray,
        body_ids: np.ndarray | Sequence[int] | slice | None = None,
        env_ids: np.ndarray | slice | None = None,
        *,
        term_name: str = "randomize_rigid_body_com",
    ) -> None:
        """Stage selected entity body inertial positions in the reset transaction."""
        reset_state, _, backend_ids = self._bind_body_randomization(
            body_ids,
            capability="reset body-ipos write",
        )
        reset_state.write_body_ipos(
            self._normalize_reset_env_ids(env_ids),
            backend_ids,
            values,
            term_name=f"{term_name}:{self.name}",
        )

    def bind_body_inertia_write(
        self,
        body_ids: np.ndarray | Sequence[int] | slice | None = None,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bind entity-local body columns and authoritative default inertias.

        The backend returns either canonical or per-environment default rows;
        entity-local columns are selected without compiling a model in UniLab.
        """
        reset_state, local_ids, backend_ids = self._bind_body_randomization(
            body_ids,
            capability="reset body-inertia write",
        )
        _, defaults = reset_state.bind_body_inertia_write(
            backend_ids,
            term_name=f"{term_name}:{self.name}",
        )
        return self._readonly_local_binding(local_ids, defaults)

    def write_body_inertia_to_sim(
        self,
        values: np.ndarray,
        body_ids: np.ndarray | Sequence[int] | slice | None = None,
        env_ids: np.ndarray | slice | None = None,
        *,
        term_name: str = "randomize_body_mass_inertia",
    ) -> None:
        """Stage selected entity body principal inertias in the reset transaction."""
        reset_state, _, backend_ids = self._bind_body_randomization(
            body_ids,
            capability="reset body-inertia write",
        )
        reset_state.write_body_inertia(
            self._normalize_reset_env_ids(env_ids),
            backend_ids,
            values,
            term_name=f"{term_name}:{self.name}",
        )

    def bind_root_linear_velocity_delta(self, *, term_name: str) -> None:
        """Validate the interval root-velocity capability on the cold path."""
        self._bind_root_velocity_delta(angular=False, term_name=term_name)

    def bind_root_angular_velocity_delta(self, *, term_name: str) -> None:
        """Validate the interval root angular-velocity capability on the cold path."""
        self._bind_root_velocity_delta(angular=True, term_name=term_name)

    def _bind_root_velocity_delta(self, *, angular: bool, term_name: str) -> None:
        if self._root_body_ids is None:
            raise self._capability_error(
                "interval root velocity delta",
                "root_body_name was not declared in EntityCfg",
            )
        try:
            capabilities = self._backend.get_dr_capabilities()
        except (AttributeError, NotImplementedError) as exc:
            raise self._capability_error("interval root velocity delta", str(exc)) from exc
        supported = (
            capabilities.supports_interval_body_angular_velocity_delta
            if angular
            else capabilities.supports_interval_body_velocity_delta
        )
        if not supported:
            raise self._capability_error(
                "interval root velocity delta",
                f"EventManager term '{term_name}' requested an unsupported backend capability",
            )

    def apply_root_linear_velocity_delta_to_sim(
        self,
        values: np.ndarray,
        env_ids: np.ndarray | slice | None = None,
        *,
        term_name: str = "push_by_setting_velocity",
    ) -> None:
        """Dispatch a cached root linear-velocity delta through the formal interval plan."""
        self.apply_root_velocity_delta_to_sim(
            values,
            None,
            env_ids=env_ids,
            term_name=term_name,
        )

    def apply_root_velocity_delta_to_sim(
        self,
        linear_delta: np.ndarray | None,
        angular_delta: np.ndarray | None,
        env_ids: np.ndarray | slice | None = None,
        *,
        term_name: str = "push_by_setting_velocity",
    ) -> None:
        """Dispatch world-frame root linear/angular velocity deltas in one interval plan."""
        if linear_delta is None and angular_delta is None:
            return
        if self._root_body_ids is None:
            raise self._capability_error(
                "interval root velocity delta",
                "root_body_name was not declared in EntityCfg",
            )
        ids = self._normalize_reset_env_ids(env_ids)
        linear_plan = self._validate_root_velocity_delta(
            linear_delta,
            ids,
            term_name=term_name,
        )
        angular_plan = self._validate_root_velocity_delta(
            angular_delta,
            ids,
            term_name=term_name,
        )
        try:
            self._backend.apply_interval_randomization(
                IntervalRandomizationPlan(
                    body_ids=self._root_body_ids,
                    body_linear_velocity_delta=linear_plan,
                    body_angular_velocity_delta=angular_plan,
                )
            )
        except NotImplementedError as exc:
            raise self._capability_error(
                "interval root velocity delta",
                f"EventManager term '{term_name}': {exc}",
            ) from exc

    def _validate_root_velocity_delta(
        self,
        values: np.ndarray | None,
        ids: np.ndarray,
        *,
        term_name: str,
    ) -> np.ndarray | None:
        if values is None:
            return None
        if not isinstance(values, np.ndarray):
            raise TypeError(
                f"EventManager term '{term_name}' root velocity delta must be np.ndarray, "
                f"got {type(values).__name__}"
            )
        expected = (ids.size, 3)
        if values.shape != expected:
            raise ValueError(
                f"EventManager term '{term_name}' root velocity delta has shape "
                f"{values.shape}; expected {expected}"
            )
        if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
            raise ValueError(
                f"EventManager term '{term_name}' root velocity delta must be finite floating data"
            )
        assert self._root_body_ids is not None
        delta = np.zeros(
            (self._backend.num_envs, len(self._root_body_ids), 3),
            dtype=values.dtype,
        )
        delta[ids, 0, :] = values
        return delta

    def bind_body_wrench(
        self,
        body_ids: np.ndarray | Sequence[int] | slice | None = None,
        *,
        torque: bool,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resolve entity-local and backend body columns for interval wrench writes.

        Returns readonly ``(local_ids, backend_ids)``; ``local_ids`` index the
        entity's body-state views (e.g. ``data.body_link_quat_w``) and
        ``backend_ids`` are the columns accepted by
        :meth:`apply_body_wrench_to_sim`.
        """
        if self._body_ids is None:
            raise self._capability_error(
                "interval body wrench",
                "body_names were not declared in EntityCfg",
            )
        local_ids = self._normalize_local_body_ids(body_ids, capability="interval body wrench")
        if local_ids.size == 0:
            raise ValueError(f"Entity '{self.name}' interval body wrench selected no bodies")
        try:
            capabilities = self._backend.get_dr_capabilities()
        except (AttributeError, NotImplementedError) as exc:
            raise self._capability_error("interval body wrench", str(exc)) from exc
        if not capabilities.supports_interval_body_force:
            raise self._capability_error(
                "interval body wrench",
                f"EventManager term '{term_name}' requested an unsupported backend capability",
            )
        if torque and not capabilities.supports_interval_body_torque:
            raise self._capability_error(
                "interval body wrench",
                f"EventManager term '{term_name}' requested an unsupported backend "
                "capability: interval body torque",
            )
        return self._readonly_local_binding(local_ids, self._body_ids[local_ids])

    def apply_body_wrench_to_sim(
        self,
        forces: np.ndarray,
        torques: np.ndarray | None,
        body_ids: np.ndarray,
        env_ids: np.ndarray | slice | None = None,
        *,
        term_name: str,
    ) -> None:
        """Dispatch world-frame body forces/torques through the formal interval plan.

        ``body_ids`` are the immutable backend columns returned by
        :meth:`bind_body_wrench`; ``forces``/``torques`` are world-frame rows
        for ``env_ids`` staged for the upcoming step.
        """
        ids = self._normalize_reset_env_ids(env_ids)
        force_values = self._validate_body_wrench_values(
            forces,
            ids,
            body_ids,
            label="force",
            term_name=term_name,
        )
        torque_values = self._validate_body_wrench_values(
            torques,
            ids,
            body_ids,
            label="torque",
            term_name=term_name,
        )
        try:
            self._backend.apply_interval_randomization(
                IntervalRandomizationPlan(
                    body_ids=body_ids,
                    body_force=force_values,
                    body_torque=torque_values,
                )
            )
        except NotImplementedError as exc:
            raise self._capability_error(
                "interval body wrench",
                f"EventManager term '{term_name}': {exc}",
            ) from exc

    def _validate_body_wrench_values(
        self,
        values: np.ndarray | None,
        ids: np.ndarray,
        body_ids: np.ndarray,
        *,
        label: str,
        term_name: str,
    ) -> np.ndarray | None:
        if values is None:
            return None
        if not isinstance(values, np.ndarray):
            raise TypeError(
                f"EventManager term '{term_name}' body {label} must be np.ndarray, "
                f"got {type(values).__name__}"
            )
        expected = (ids.size, body_ids.size, 3)
        if values.shape != expected:
            raise ValueError(
                f"EventManager term '{term_name}' body {label} has shape {values.shape}; "
                f"expected {expected}"
            )
        if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
            raise ValueError(
                f"EventManager term '{term_name}' body {label} must be finite floating data"
            )
        full = np.zeros((self._backend.num_envs, body_ids.size, 3), dtype=values.dtype)
        full[ids] = values
        return full

    def write_root_link_pose_to_sim(
        self,
        root_pose: np.ndarray,
        env_ids: np.ndarray | slice | None = None,
    ) -> None:
        """Stage world position and wxyz root orientation during reset."""
        if self._physical_entity is not None:
            self._stage_entity_write(env_ids, root_pose=root_pose)
            return
        reset_state, layout = self._require_root_state_write()
        resolved_env_ids = self._normalize_reset_env_ids(env_ids)
        reset_state.write_root_pose(
            resolved_env_ids,
            layout,
            root_pose,
            term_name=f"{self.name}.write_root_link_pose_to_sim",
        )

    def write_root_link_velocity_to_sim(
        self,
        root_velocity: np.ndarray,
        env_ids: np.ndarray | slice | None = None,
    ) -> None:
        """Stage world linear/angular root velocity during reset."""
        if self._physical_entity is not None:
            self._stage_entity_write(env_ids, root_velocity=root_velocity)
            return
        reset_state, layout = self._require_root_state_write()
        resolved_env_ids = self._normalize_reset_env_ids(env_ids)
        reset_state.write_root_velocity(
            resolved_env_ids,
            layout,
            root_velocity,
            term_name=f"{self.name}.write_root_link_velocity_to_sim",
        )

    def read_reset_root_pose(
        self,
        env_ids: np.ndarray | slice | None = None,
    ) -> np.ndarray:
        """Read the world position and wxyz root orientation staged in the reset.

        Returns a detached ``(len(env_ids), 7)`` copy of the pose currently
        staged in the active reset transaction (backend default for rows no
        term has written yet), so a later reset term can build on an earlier
        term's root placement.
        """
        if self._physical_entity is not None:
            assert self._reset_state is not None
            return self._reset_state.read_entity_root_pose(
                self._physical_entity, self._normalize_reset_env_ids(env_ids)
            )
        reset_state, layout = self._require_root_state_write()
        resolved_env_ids = self._normalize_reset_env_ids(env_ids)
        return reset_state.read_root_pose(
            resolved_env_ids,
            layout,
            term_name=f"{self.name}.read_reset_root_pose",
        )

    def _require_root_state_write(
        self,
    ) -> tuple[ResetStateTransaction, BackendRootStateLayout]:
        if self._reset_state is None:
            raise self._capability_error(
                "reset root-state write",
                "EntityScene was materialized without an env-owned reset transaction",
            )
        if self._reset_root_layout is None:
            detail = self._reset_root_layout_error or "root-state layout was not materialized"
            raise self._capability_error("reset root-state layout", detail)
        return self._reset_state, self._reset_root_layout

    def write_joint_state_to_sim(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        joint_ids: np.ndarray | Sequence[int] | slice | None = None,
        env_ids: np.ndarray | slice | None = None,
    ) -> None:
        """Stage community-style joint state writes in the active reset transaction."""
        if self._reset_state is None:
            raise self._capability_error(
                "reset joint-state write",
                "EntityScene was materialized without an env-owned reset transaction",
            )
        if self._joint_names is None:
            raise self._capability_error(
                "reset joint-state write",
                "joint_names were not declared in EntityCfg",
            )
        local_joint_ids = self._normalize_local_joint_ids(
            joint_ids,
            capability="reset joint-state write",
        )
        resolved_env_ids = self._normalize_reset_env_ids(env_ids)
        if self._physical_entity is not None:
            names = tuple(self._joint_names[int(i)].split("/", 1)[1] for i in local_joint_ids)
            self._stage_entity_write(
                resolved_env_ids,
                joint_positions=position,
                joint_velocities=velocity,
                joint_names=names,
            )
            return
        self._materialize_reset_joint_indices()
        assert self._reset_joint_qpos_ids is not None
        assert self._reset_joint_qvel_ids is not None
        self._reset_state.write_joint_state(
            resolved_env_ids,
            self._reset_joint_qpos_ids[local_joint_ids],
            self._reset_joint_qvel_ids[local_joint_ids],
            position,
            velocity,
            term_name=f"{self.name}.write_joint_state_to_sim",
        )

    def _stage_entity_write(self, env_ids, **fields) -> None:
        assert self._reset_state is not None and self._physical_entity is not None
        self._reset_state.write_entity_state(
            self._physical_entity,
            self._normalize_reset_env_ids(env_ids),
            term_name=f"{self.name}.entity_state",
            **fields,
        )

    def _materialize_reset_joint_indices(self) -> None:
        if self._reset_joint_qpos_ids is not None and self._reset_joint_qvel_ids is not None:
            return
        assert self._joint_names is not None
        try:
            qpos_ids = (
                self._reset_joint_qpos_ids
                if self._reset_joint_qpos_ids is not None
                else self._backend.get_joint_state_qpos_indices(self._joint_names)
            )
            qvel_ids = self._backend.get_joint_state_qvel_indices(self._joint_names)
        except (AttributeError, NotImplementedError) as exc:
            raise self._capability_error("reset joint-state layout", str(exc)) from exc
        self._reset_joint_qpos_ids = _readonly_ids(
            qpos_ids,
            expected=len(self._joint_names),
            label=f"Entity '{self.name}' reset qpos",
        )
        self._reset_joint_qvel_ids = _readonly_ids(
            qvel_ids,
            expected=len(self._joint_names),
            label=f"Entity '{self.name}' reset qvel",
        )

    def _bind_body_randomization(
        self,
        body_ids: np.ndarray | Sequence[int] | slice | None,
        *,
        capability: str,
    ) -> tuple[ResetStateTransaction, np.ndarray, np.ndarray]:
        if self._reset_state is None:
            raise self._capability_error(
                capability,
                "EntityScene was materialized without an env-owned reset transaction",
            )
        if self._body_ids is None:
            raise self._capability_error(
                capability,
                "body_names were not declared in EntityCfg",
            )
        local_ids = self._normalize_local_body_ids(body_ids, capability=capability)
        if local_ids.size == 0:
            raise ValueError(f"Entity '{self.name}' {capability} selected no bodies")
        return self._reset_state, local_ids, self._body_ids[local_ids]

    def _readonly_local_binding(
        self,
        local_ids: np.ndarray,
        defaults: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _readonly_array(local_ids), _readonly_array(defaults)

    def _materialize_joint_model_dof_ids(self) -> np.ndarray:
        """Resolve full model DOF addresses once for reset-time model fields."""
        cached = self._joint_model_dof_ids
        if cached is not None:
            return cached
        if self._joint_names is None:
            raise self._capability_error(
                "reset joint-armature write",
                "joint_names were not declared in EntityCfg",
            )
        try:
            values = self._backend.get_joint_dof_indices(self._joint_names)
        except (AttributeError, NotImplementedError) as exc:
            raise self._capability_error("reset joint-armature write", str(exc)) from exc
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Entity '{self.name}' could not resolve joint model DOF names "
                f"{list(self._joint_names)} on backend '{self._backend_type}': {exc}"
            ) from exc
        resolved = _readonly_ids(
            values,
            expected=len(self._joint_names),
            label=f"Entity '{self.name}' joint model DOF",
        )
        self._joint_model_dof_ids = resolved
        return resolved

    def _normalize_local_body_ids(
        self,
        body_ids: np.ndarray | Sequence[int] | slice | None,
        *,
        capability: str,
    ) -> np.ndarray:
        if body_ids is None:
            ids = np.arange(self.num_bodies, dtype=np.intp)
        elif isinstance(body_ids, slice):
            ids = np.arange(self.num_bodies, dtype=np.intp)[body_ids]
        else:
            raw = np.asarray(body_ids)
            if (
                raw.ndim != 1
                or not np.issubdtype(raw.dtype, np.integer)
                or np.issubdtype(raw.dtype, np.bool_)
            ):
                raise TypeError(
                    f"Entity '{self.name}' {capability} body_ids must be a 1-D integer "
                    "array or slice"
                )
            ids = np.asarray(raw, dtype=np.intp)
        if np.any(ids < 0) or np.any(ids >= self.num_bodies):
            raise IndexError(
                f"Entity '{self.name}' {capability} body_ids out of range for "
                f"{self.num_bodies} bodies: {ids.tolist()}"
            )
        if np.unique(ids).size != ids.size:
            raise ValueError(
                f"Entity '{self.name}' {capability} body_ids contain duplicates: {ids.tolist()}"
            )
        return ids

    def _normalize_local_joint_ids(
        self,
        joint_ids: np.ndarray | Sequence[int] | slice | None,
        *,
        capability: str,
    ) -> np.ndarray:
        if joint_ids is None:
            ids = np.arange(self.num_joints, dtype=np.intp)
        elif isinstance(joint_ids, slice):
            ids = np.arange(self.num_joints, dtype=np.intp)[joint_ids]
        else:
            raw = np.asarray(joint_ids)
            if (
                raw.ndim != 1
                or not np.issubdtype(raw.dtype, np.integer)
                or np.issubdtype(raw.dtype, np.bool_)
            ):
                raise TypeError(
                    f"Entity '{self.name}' {capability} joint_ids must be a 1-D integer "
                    "array or slice"
                )
            ids = np.asarray(raw, dtype=np.intp)
        if np.any(ids < 0) or np.any(ids >= self.num_joints):
            raise IndexError(
                f"Entity '{self.name}' {capability} joint_ids out of range for "
                f"{self.num_joints} joints: {ids.tolist()}"
            )
        if np.unique(ids).size != ids.size:
            raise ValueError(
                f"Entity '{self.name}' {capability} joint_ids contain duplicates: {ids.tolist()}"
            )
        return ids

    def _normalize_local_geom_ids(
        self,
        geom_ids: np.ndarray | Sequence[int] | slice | None,
        *,
        capability: str,
    ) -> np.ndarray:
        if geom_ids is None:
            ids = np.arange(self.num_geoms, dtype=np.intp)
        elif isinstance(geom_ids, slice):
            ids = np.arange(self.num_geoms, dtype=np.intp)[geom_ids]
        else:
            raw = np.asarray(geom_ids)
            if (
                raw.ndim != 1
                or not np.issubdtype(raw.dtype, np.integer)
                or np.issubdtype(raw.dtype, np.bool_)
            ):
                raise TypeError(
                    f"Entity '{self.name}' {capability} geom_ids must be a 1-D integer "
                    "array or slice"
                )
            ids = np.asarray(raw, dtype=np.intp)
        if np.any(ids < 0) or np.any(ids >= self.num_geoms):
            raise IndexError(
                f"Entity '{self.name}' {capability} geom_ids out of range for "
                f"{self.num_geoms} geoms: {ids.tolist()}"
            )
        if np.unique(ids).size != ids.size:
            raise ValueError(
                f"Entity '{self.name}' {capability} geom_ids contain duplicates: {ids.tolist()}"
            )
        return ids

    def _normalize_local_actuator_ids(
        self,
        actuator_ids: np.ndarray | Sequence[int] | slice | None,
        *,
        capability: str,
    ) -> np.ndarray:
        if actuator_ids is None:
            ids = np.arange(self.num_actuators, dtype=np.intp)
        elif isinstance(actuator_ids, slice):
            ids = np.arange(self.num_actuators, dtype=np.intp)[actuator_ids]
        else:
            raw = np.asarray(actuator_ids)
            if (
                raw.ndim != 1
                or not np.issubdtype(raw.dtype, np.integer)
                or np.issubdtype(raw.dtype, np.bool_)
            ):
                raise TypeError(
                    f"Entity '{self.name}' {capability} actuator_ids must be a 1-D "
                    "integer array or slice"
                )
            ids = np.asarray(raw, dtype=np.intp)
        if np.any(ids < 0) or np.any(ids >= self.num_actuators):
            raise IndexError(
                f"Entity '{self.name}' {capability} actuator_ids out of range for "
                f"{self.num_actuators} actuators: {ids.tolist()}"
            )
        if np.unique(ids).size != ids.size:
            raise ValueError(
                f"Entity '{self.name}' {capability} actuator_ids contain duplicates: {ids.tolist()}"
            )
        return ids

    def _normalize_reset_env_ids(self, env_ids: np.ndarray | slice | None) -> np.ndarray:
        if env_ids is None:
            return np.arange(self._backend.num_envs, dtype=np.int32)
        if isinstance(env_ids, slice):
            return np.arange(self._backend.num_envs, dtype=np.int32)[env_ids]
        return env_ids

    def find_bodies(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        return self._find("body", self._body_names, keys, preserve_order)

    def find_geoms(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        return self._find("geom", self._geom_names, keys, preserve_order)

    def find_sites(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        return self._find("site", self._site_names, keys, preserve_order)

    def find_actuators(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        return self._find("actuator", self._actuator_names, keys, preserve_order)

    def find_tendons(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        del keys, preserve_order
        return self._unsupported_names("tendon")

    def find_cameras(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        del keys, preserve_order
        return self._unsupported_names("camera")

    def find_lights(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        del keys, preserve_order
        return self._unsupported_names("light")

    def find_materials(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        del keys, preserve_order
        return self._unsupported_names("material")

    def find_textures(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        del keys, preserve_order
        return self._unsupported_names("texture")

    def find_pairs(
        self, keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        del keys, preserve_order
        return self._unsupported_names("pair")


class EntityScene(Mapping[str, Entity]):
    """Read-only name-addressable collection of backend-bound entities."""

    def __init__(
        self,
        entities: Mapping[str, EntityCfg],
        backend: SimBackend,
        control_buffer: np.ndarray | torch.Tensor | None = None,
        *,
        reset_state: ResetStateTransaction | None = None,
        default_qpos: np.ndarray | None = None,
    ) -> None:
        self._backend = backend
        self._state_read_cache = _EntityStateReadCache()
        self._tensor_read_plan: SceneTensorReadPlan | None = None
        materialized: dict[str, Entity] = {}
        for name, cfg in entities.items():
            if not isinstance(name, str) or not name:
                raise TypeError(f"Scene entity names must be non-empty strings; got {name!r}")
            if not isinstance(cfg, EntityCfg):
                raise TypeError(
                    f"Scene entity '{name}' must be EntityCfg, got {type(cfg).__name__}"
                )
            materialized[name] = Entity(
                name,
                cfg,
                backend,
                control_buffer,
                reset_state,
                default_qpos=default_qpos,
                state_read_cache=self._state_read_cache,
            )
        self._entities = MappingProxyType(materialized)
        self._reset_state = reset_state
        env_origins = np.zeros((backend.num_envs, 3), dtype=np.float32)
        env_origins.setflags(write=False)
        self._env_origins = env_origins

    @classmethod
    def from_scene_cfg(
        cls,
        cfg: SceneCfg,
        backend: SimBackend,
        control_buffer: np.ndarray | torch.Tensor | None = None,
        *,
        reset_state: ResetStateTransaction | None = None,
        default_qpos: np.ndarray | None = None,
    ) -> EntityScene:
        return cls(
            cast(Mapping[str, EntityCfg], cfg.entities),
            backend,
            control_buffer,
            reset_state=reset_state,
            default_qpos=default_qpos,
        )

    @property
    def entities(self) -> Mapping[str, Entity]:
        """Pinned community-style read-only entity mapping."""
        return self._entities

    @property
    def env_origins(self) -> np.ndarray:
        """Read-only per-environment origins; flat UniLab scenes default to zero."""
        return self._env_origins

    def compile_tensor_reads(
        self,
        device: str | torch.device,
        specs: Sequence[SceneTensorReadSpec],
    ) -> "SceneTensorReadPlan":
        """Compile the scene's only packed tensor read layout for one device.

        ``HOST_BRIDGE`` execution aggregates every entity's sensors and canonical
        body views into one public ``TensorIOSpec``, so a read phase performs one
        H2D transfer. ``DEVICE_RESIDENT`` execution keeps stable backend views but
        exposes the same phase API; manager terms therefore do not branch on the
        backend topology.
        """
        if isinstance(specs, SceneTensorReadSpec):
            raise TypeError("Scene tensor read specs must be a sequence of SceneTensorReadSpec")
        normalized = tuple(specs)
        if not normalized:
            raise ValueError("Scene tensor read request is empty")
        if any(not isinstance(spec, SceneTensorReadSpec) for spec in normalized):
            raise TypeError("Scene tensor read specs must be SceneTensorReadSpec instances")

        entities: dict[str, Entity] = {}
        per_entity_sensors: dict[str, tuple[str, ...]] = {}
        per_entity_bodies: dict[str, tuple[str, ...]] = {}
        request_owners: dict[str, str] = {}
        for spec in normalized:
            try:
                entity = self._entities[spec.entity]
            except KeyError as exc:
                raise KeyError(
                    f"Scene tensor read entity '{spec.entity}' not found; "
                    f"available={list(self._entities)}"
                ) from exc
            entities[spec.entity] = entity
            sensors = per_entity_sensors.setdefault(spec.entity, ())
            bodies = per_entity_bodies.setdefault(spec.entity, ())
            for name in spec.sensor_names:
                owner = request_owners.get(name)
                if owner is not None and owner != spec.entity:
                    raise ValueError(
                        f"Scene tensor sensor '{name}' is requested by entities "
                        f"'{owner}' and '{spec.entity}'"
                    )
                request_owners[name] = spec.entity
            per_entity_sensors[spec.entity] = (*sensors, *spec.sensor_names)
            per_entity_bodies[spec.entity] = (*bodies, *spec.body_names)

        expanded_bodies: dict[str, tuple[str, ...]] = {}
        for entity_name, body_names in per_entity_bodies.items():
            if not body_names:
                expanded_bodies[entity_name] = ()
                continue
            expanded_bodies[entity_name] = self._normalize_tensor_body_request(
                entities[entity_name], body_names
            )

        resolved_device = torch.device(device)
        if resolved_device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA scene tensor reads requested but CUDA is unavailable")
            if resolved_device.index is None:
                resolved_device = torch.device("cuda", index=torch.cuda.current_device())

        capabilities = self._backend.get_tensor_capabilities()
        self._validate_tensor_read_capabilities(capabilities, resolved_device)
        sensor_names = self._aggregate_tensor_read_names(
            entities, per_entity_sensors, expanded_bodies
        )

        host_plan: HostBridgeTransferPlan | None = None
        if capabilities.execution is TensorExecution.HOST_BRIDGE:
            spec = TensorIOSpec(
                state_fields=("qpos", "qvel"),
                sensor_names=sensor_names,
                device=resolved_device,
            )
            try:
                host_plan = self._backend.compile_host_bridge_io(spec)
            except (AttributeError, KeyError, TypeError, ValueError, NotImplementedError) as exc:
                raise type(exc)(
                    "Manager scene packed tensor read compilation on backend "
                    f"'{self._backend.backend_type}': {exc}"
                ) from exc

        return SceneTensorReadPlan(
            scene=self,
            device=resolved_device,
            entities=entities,
            sensor_names=per_entity_sensors,
            body_names=expanded_bodies,
            host_plan=host_plan,
        )

    @staticmethod
    def _normalize_tensor_body_request(
        entity: "Entity", requested: tuple[str, ...]
    ) -> tuple[str, ...]:
        declared = entity.body_names
        missing = [name for name in requested if name not in set(declared)]
        if missing:
            raise ValueError(
                f"Entity '{entity.name}' tensor body names {missing} are not declared; "
                f"available={list(declared)}"
            )
        return tuple(dict.fromkeys(requested))

    @staticmethod
    def _validate_tensor_read_capabilities(
        capabilities: TensorLifecycleCapabilities, device: torch.device
    ) -> None:
        backend_type = "backend"
        if capabilities.execution not in {
            TensorExecution.HOST_BRIDGE,
            TensorExecution.DEVICE_RESIDENT,
        }:
            raise NotImplementedError(
                f"Manager scene tensor reads are unavailable on {backend_type}: "
                f"tensor execution is {capabilities.execution}"
            )
        if not capabilities.state_views or not {"qpos", "qvel"}.issubset(capabilities.state_fields):
            raise NotImplementedError(
                "Manager scene tensor reads require public qpos/qvel state views"
            )
        if not capabilities.sensor_views:
            raise NotImplementedError(
                "Manager scene tensor reads require public named sensor views"
            )
        if capabilities.execution is TensorExecution.HOST_BRIDGE and not (
            capabilities.packed_host_bridge
            and capabilities.process_topology is TensorProcessTopology.IN_PROCESS
            and capabilities.data_plane is TensorDataPlane.HOST_BRIDGE
        ):
            raise NotImplementedError(
                "Manager scene HOST_BRIDGE tensor reads require an in-process packed host bridge"
            )
        if not tensor_device_matches(capabilities.torch_devices, device):
            raise ValueError(
                f"Manager scene tensor read device {device} is unsupported; "
                f"accepted={capabilities.torch_devices}"
            )

    @staticmethod
    def _aggregate_tensor_read_names(
        entities: Mapping[str, "Entity"],
        sensor_names: Mapping[str, tuple[str, ...]],
        body_names: Mapping[str, tuple[str, ...]],
    ) -> tuple[str, ...]:
        aggregate: list[str] = []

        def add(name: str) -> None:
            if name not in seen:
                seen.add(name)
                aggregate.append(name)

        seen: set[str] = set()
        for entity_name in entities:
            for name in sensor_names[entity_name]:
                add(name)
        for entity_name in entities:
            for body_name in body_names[entity_name]:
                for _, prefix in _TENSOR_BODY_SENSOR_FIELDS:
                    add(prefix + body_name)
        return tuple(aggregate)

    @contextmanager
    def _scoped_state_reads(self) -> Iterator[None]:
        """Internal ManagerBasedRlEnv boundary for one stable update phase."""
        with self._state_read_cache.scoped():
            yield

    def _invalidate_state_reads(self) -> None:
        """Discard cached backend state after an in-phase simulation mutation."""
        self._state_read_cache.invalidate()
        if self._tensor_read_plan is not None:
            self._tensor_read_plan.invalidate()

    def reset_to_default(self, env_ids: np.ndarray, *, term_name: str) -> None:
        """Stage a full-scene default state in the active reset transaction."""
        if self._reset_state is None:
            raise NotImplementedError(
                f"EventManager term '{term_name}' reset-state capability is unavailable: "
                "EntityScene was materialized without an env-owned reset transaction"
            )
        if np.any(self._env_origins):
            raise NotImplementedError(
                f"EventManager term '{term_name}' cannot apply non-zero env_origins without "
                "a formal backend root-state layout"
            )
        self._reset_state.reset_to_default(env_ids, term_name=term_name)

    def bind_gravity_write(self, *, term_name: str) -> np.ndarray:
        """Bind immutable backend gravity for a reset event on the cold path."""
        if self._reset_state is None:
            raise NotImplementedError(
                f"EventManager term '{term_name}' gravity capability is unavailable: "
                "EntityScene was materialized without an env-owned reset transaction"
            )
        return self._reset_state.bind_gravity_write(term_name=term_name)

    def write_gravity_to_sim(
        self,
        values: np.ndarray,
        env_ids: np.ndarray,
        *,
        term_name: str,
    ) -> None:
        """Stage gravity values in the exactly-once reset transaction."""
        if self._reset_state is None:
            raise NotImplementedError(
                f"EventManager term '{term_name}' gravity capability is unavailable: "
                "EntityScene was materialized without an env-owned reset transaction"
            )
        self._reset_state.write_gravity(env_ids, values, term_name=term_name)

    def bind_sensor_data(self, names: Sequence[str]) -> BackendSensorView:
        """Bind existing backend sensors for a manager term on the cold path.

        The returned view owns the backend-specific reader.  Terms retain that
        view and only call :meth:`BackendSensorView.read` while stepping, so the
        scene facade never exposes a backend model, data object, or native handle.
        """
        try:
            return self._backend.bind_sensor_data(names)
        except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise type(exc)(
                "Manager scene named-sensor capability on backend "
                f"'{self._backend.backend_type}': {exc}"
            ) from exc

    def __getitem__(self, name: str) -> Entity:
        try:
            return self._entities[name]
        except KeyError as exc:
            raise KeyError(
                f"Scene entity '{name}' not found; available={list(self._entities)}"
            ) from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self._entities)

    def __len__(self) -> int:
        return len(self._entities)


__all__ = [
    "Entity",
    "EntityCfg",
    "EntityData",
    "EntityScene",
    "EntityTensorBodyStateView",
    "EntityTensorSensorViews",
    "EntityTensorStateView",
    "SceneTensorReadPlan",
    "SceneTensorReadSpec",
]
