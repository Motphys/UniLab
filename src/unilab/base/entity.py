"""Base-owned NumPy scene/entity facade for manager terms.

The facade deliberately describes partitions of an already materialized UniLab scene.
It is not a second scene composer: all name resolution and state reads go through the
public :class:`~unilab.base.backend.base.SimBackend` contract.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, NoReturn

import numpy as np

from unilab.base.backend.base import BackendRootStateLayout, BackendSensorView, SimBackend
from unilab.dr.types import IntervalRandomizationPlan
from unilab.utils.rotation import np_quat_apply, np_quat_apply_inverse, np_yaw_from_quat

if TYPE_CHECKING:
    from unilab.base.reset_state import ResetStateTransaction
    from unilab.base.scene import SceneCfg


NamesCfg = tuple[str, ...] | list[str] | None


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


class EntityData:
    """Hot-path NumPy state surface backed by cached backend IDs."""

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
        gravity_vec_w: np.ndarray | None,
        body_ids: np.ndarray | None,
        actuator_ids: np.ndarray | None,
        actuator_ctrl_range: np.ndarray | None,
        control_buffer: np.ndarray | None,
        entity_name: str,
        backend_type: str,
    ) -> None:
        self._backend = backend
        self._entity_name = entity_name
        self._backend_type = backend_type
        self._root_body_ids = root_body_ids
        self._joint_pos_index = None if joint_pos_ids is None else _as_column_index(joint_pos_ids)
        self._joint_vel_index = None if joint_vel_ids is None else _as_column_index(joint_vel_ids)
        self._default_root_state = default_root_state
        self._default_root_state_error = default_root_state_error
        self._default_joint_pos = default_joint_pos
        self._default_joint_vel = default_joint_vel
        self._gravity_vec_w = gravity_vec_w
        self._encoder_bias = (
            None
            if default_joint_pos is None
            else np.zeros(default_joint_pos.shape, dtype=default_joint_pos.dtype)
        )
        self._body_ids = body_ids
        self._actuator_ids = actuator_ids
        self._actuator_ctrl_range = actuator_ctrl_range
        self._control_buffer = control_buffer

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
        return self._backend.get_body_pos_w(ids)[:, 0]

    @property
    def root_link_quat_w(self) -> np.ndarray:
        ids = self._require(self._root_body_ids, "root body state")
        return self._backend.get_body_quat_w(ids)[:, 0]

    @property
    def root_link_lin_vel_w(self) -> np.ndarray:
        ids = self._require(self._root_body_ids, "root body state")
        return self._backend.get_body_lin_vel_w(ids)[:, 0]

    @property
    def root_link_ang_vel_w(self) -> np.ndarray:
        ids = self._require(self._root_body_ids, "root body state")
        return self._backend.get_body_ang_vel_w(ids)[:, 0]

    @property
    def root_link_lin_vel_b(self) -> np.ndarray:
        ids = self._require(self._root_body_ids, "root body state")
        return self._backend.get_body_lin_vel_b(ids)[:, 0]

    @property
    def root_link_ang_vel_b(self) -> np.ndarray:
        ids = self._require(self._root_body_ids, "root body state")
        return self._backend.get_body_ang_vel_b(ids)[:, 0]

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
        return self._backend.get_dof_pos()[:, index]

    @property
    def joint_vel(self) -> np.ndarray:
        index = self._require(self._joint_vel_index, "joint velocity")
        return self._backend.get_dof_vel()[:, index]

    @property
    def joint_pos_biased(self) -> np.ndarray:
        """Joint positions with the manager-owned encoder bias applied."""
        return self.joint_pos + self.encoder_bias

    @property
    def default_joint_pos(self) -> np.ndarray:
        """Read-only per-environment default joint positions."""
        return self._require(self._default_joint_pos, "default joint position")

    @property
    def default_joint_vel(self) -> np.ndarray:
        """Read-only zero default velocities from the UniLab reset contract."""
        return self._require(self._default_joint_vel, "default joint velocity")

    @property
    def encoder_bias(self) -> np.ndarray:
        """Mutable per-environment joint encoder bias used by position actions."""
        return self._require(self._encoder_bias, "joint encoder bias")

    @property
    def body_link_pos_w(self) -> np.ndarray:
        ids = self._require(self._body_ids, "body state")
        return self._backend.get_body_pos_w(ids)

    @property
    def body_link_quat_w(self) -> np.ndarray:
        ids = self._require(self._body_ids, "body state")
        return self._backend.get_body_quat_w(ids)

    @property
    def body_link_lin_vel_w(self) -> np.ndarray:
        ids = self._require(self._body_ids, "body state")
        return self._backend.get_body_lin_vel_w(ids)

    @property
    def body_link_ang_vel_w(self) -> np.ndarray:
        ids = self._require(self._body_ids, "body state")
        return self._backend.get_body_ang_vel_w(ids)

    @property
    def body_link_pose_w(self) -> np.ndarray:
        return np.concatenate((self.body_link_pos_w, self.body_link_quat_w), axis=-1)

    @property
    def body_link_vel_w(self) -> np.ndarray:
        return np.concatenate((self.body_link_lin_vel_w, self.body_link_ang_vel_w), axis=-1)

    @property
    def actuator_ctrl_range(self) -> np.ndarray:
        return self._require(self._actuator_ctrl_range, "actuator control range")

    def write_ctrl(
        self,
        values: np.ndarray,
        env_ids: np.ndarray | slice | None = None,
        *,
        actuator_ids: np.ndarray | Sequence[int] | slice | None = None,
    ) -> None:
        """Write entity-local actuator controls into the env-owned control buffer.

        This is an in-memory scene write, analogous to the pinned manager runtime's
        entity target buffers.  Physics remains owned by ``NpEnv``/``SimBackend``;
        this method never steps or calls a backend-private API.
        """
        entity_actuator_ids = self._require(self._actuator_ids, "actuator control write")
        control = self._require(self._control_buffer, "actuator control write")
        if not isinstance(values, np.ndarray):
            raise TypeError(
                f"Entity '{self._entity_name}' write_ctrl expected np.ndarray, "
                f"received {type(values).__name__}"
            )
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

        if isinstance(row_index, slice) or isinstance(actuator_index, slice):
            control[row_index, actuator_index] = values
        else:
            control[row_index[:, None], actuator_index[None, :]] = values


class Entity:
    """Logical entity with cached local-to-backend mappings."""

    def __init__(
        self,
        name: str,
        cfg: EntityCfg,
        backend: SimBackend,
        control_buffer: np.ndarray | None = None,
        reset_state: ResetStateTransaction | None = None,
        *,
        default_qpos: np.ndarray | None = None,
    ) -> None:
        if not name:
            raise ValueError("Entity name must be a non-empty string")
        self.name = name
        self._backend_type = backend.backend_type
        self._backend = backend
        self._reset_state = reset_state
        self._reset_root_layout: BackendRootStateLayout | None = None
        self._reset_root_layout_error: str | None = None
        self._reset_joint_qpos_ids: np.ndarray | None = None
        self._reset_joint_qvel_ids: np.ndarray | None = None

        self._joint_names = _normalize_names(name, "joint", cfg.joint_names)
        self._body_names = _normalize_names(name, "body", cfg.body_names)
        self._geom_names = _normalize_names(name, "geom", cfg.geom_names)
        self._site_names = _normalize_names(name, "site", cfg.site_names)
        self._actuator_names = _normalize_names(name, "actuator", cfg.actuator_names)

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
        gravity_vec_w = self._materialize_gravity_vector(backend, root_body_ids)
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
            if not np.issubdtype(control_buffer.dtype, np.floating):
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
            gravity_vec_w=gravity_vec_w,
            body_ids=body_ids,
            actuator_ids=actuator_ids,
            actuator_ctrl_range=actuator_ctrl_range,
            control_buffer=control_buffer,
            entity_name=self.name,
            backend_type=self._backend_type,
        )

    def _capability_error(self, capability: str, detail: str) -> NotImplementedError:
        return NotImplementedError(
            f"Entity '{self.name}' capability '{capability}' is unavailable on "
            f"backend '{self._backend_type}': {detail}"
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
            [ids_by_name[value] for value in names],
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

    def _materialize_root_state(
        self,
        backend: SimBackend,
        root_body_name: str | None,
        default_qpos: np.ndarray | None,
    ) -> tuple[BackendRootStateLayout | None, np.ndarray | None, str | None]:
        if root_body_name is None:
            return None, None, "root_body_name was not declared in EntityCfg"
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
        current = self._read_state("joint velocity state", backend.get_dof_vel)
        materialized = np.zeros(
            (backend.num_envs, len(joint_vel_ids)),
            dtype=current.dtype,
        )
        materialized.setflags(write=False)
        return materialized

    def _materialize_gravity_vector(
        self, backend: SimBackend, root_body_ids: np.ndarray | None
    ) -> np.ndarray | None:
        if root_body_ids is None:
            return None
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

    def write_root_state_to_sim(
        self,
        root_state: np.ndarray,
        env_ids: np.ndarray | slice | None = None,
    ) -> None:
        """Stage a 13-D world-frame root state in the active reset transaction."""
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

    def bind_root_linear_velocity_delta(self, *, term_name: str) -> None:
        """Validate the interval root-velocity capability on the cold path."""
        if self._root_body_ids is None:
            raise self._capability_error(
                "interval root velocity delta",
                "root_body_name was not declared in EntityCfg",
            )
        try:
            capabilities = self._backend.get_dr_capabilities()
        except (AttributeError, NotImplementedError) as exc:
            raise self._capability_error("interval root velocity delta", str(exc)) from exc
        if not capabilities.supports_interval_body_velocity_delta:
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
        if self._root_body_ids is None:
            raise self._capability_error(
                "interval root velocity delta",
                "root_body_name was not declared in EntityCfg",
            )
        ids = self._normalize_reset_env_ids(env_ids)
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
        delta = np.zeros(
            (self._backend.num_envs, len(self._root_body_ids), 3),
            dtype=values.dtype,
        )
        delta[ids, 0, :] = values
        try:
            self._backend.apply_interval_randomization(
                IntervalRandomizationPlan(
                    body_ids=self._root_body_ids,
                    body_linear_velocity_delta=delta,
                )
            )
        except NotImplementedError as exc:
            raise self._capability_error(
                "interval root velocity delta",
                f"EventManager term '{term_name}': {exc}",
            ) from exc

    def write_root_link_pose_to_sim(
        self,
        root_pose: np.ndarray,
        env_ids: np.ndarray | slice | None = None,
    ) -> None:
        """Stage world position and wxyz root orientation during reset."""
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
        reset_state, layout = self._require_root_state_write()
        resolved_env_ids = self._normalize_reset_env_ids(env_ids)
        reset_state.write_root_velocity(
            resolved_env_ids,
            layout,
            root_velocity,
            term_name=f"{self.name}.write_root_link_velocity_to_sim",
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
        bound_ids = np.array(local_ids, copy=True)
        bound_ids.setflags(write=False)
        bound_defaults = np.array(defaults, copy=True)
        bound_defaults.setflags(write=False)
        return bound_ids, bound_defaults

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
        control_buffer: np.ndarray | None = None,
        *,
        reset_state: ResetStateTransaction | None = None,
        default_qpos: np.ndarray | None = None,
    ) -> None:
        self._backend = backend
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
        control_buffer: np.ndarray | None = None,
        *,
        reset_state: ResetStateTransaction | None = None,
        default_qpos: np.ndarray | None = None,
    ) -> EntityScene:
        return cls(
            cfg.entities,
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


__all__ = ["Entity", "EntityCfg", "EntityData", "EntityScene"]
