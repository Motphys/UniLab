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

from unilab.base.backend.base import SimBackend

if TYPE_CHECKING:
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
        body_ids: np.ndarray | None,
        actuator_ctrl_range: np.ndarray | None,
        entity_name: str,
        backend_type: str,
    ) -> None:
        self._backend = backend
        self._entity_name = entity_name
        self._backend_type = backend_type
        self._root_body_ids = root_body_ids
        self._joint_pos_index = None if joint_pos_ids is None else _as_column_index(joint_pos_ids)
        self._joint_vel_index = None if joint_vel_ids is None else _as_column_index(joint_vel_ids)
        self._body_ids = body_ids
        self._actuator_ctrl_range = actuator_ctrl_range

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
    def root_link_pose_w(self) -> np.ndarray:
        return np.concatenate((self.root_link_pos_w, self.root_link_quat_w), axis=-1)

    @property
    def root_link_vel_w(self) -> np.ndarray:
        return np.concatenate((self.root_link_lin_vel_w, self.root_link_ang_vel_w), axis=-1)

    @property
    def joint_pos(self) -> np.ndarray:
        index = self._require(self._joint_pos_index, "joint position")
        return self._backend.get_dof_pos()[:, index]

    @property
    def joint_vel(self) -> np.ndarray:
        index = self._require(self._joint_vel_index, "joint velocity")
        return self._backend.get_dof_vel()[:, index]

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


class Entity:
    """Logical entity with cached local-to-backend mappings."""

    def __init__(self, name: str, cfg: EntityCfg, backend: SimBackend) -> None:
        if not name:
            raise ValueError("Entity name must be a non-empty string")
        self.name = name
        self._backend_type = backend.backend_type

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

        self._validate_joint_state(backend, joint_pos_ids, joint_vel_ids)
        self._validate_body_state(backend, root_body_ids, body_ids)
        actuator_ctrl_range = self._materialize_actuator_ctrl_range(backend, actuator_ids)

        self.data = EntityData(
            backend,
            root_body_ids=root_body_ids,
            joint_pos_ids=joint_pos_ids,
            joint_vel_ids=joint_vel_ids,
            body_ids=body_ids,
            actuator_ctrl_range=actuator_ctrl_range,
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

    def __init__(self, entities: Mapping[str, EntityCfg], backend: SimBackend) -> None:
        materialized: dict[str, Entity] = {}
        for name, cfg in entities.items():
            if not isinstance(name, str) or not name:
                raise TypeError(f"Scene entity names must be non-empty strings; got {name!r}")
            if not isinstance(cfg, EntityCfg):
                raise TypeError(
                    f"Scene entity '{name}' must be EntityCfg, got {type(cfg).__name__}"
                )
            materialized[name] = Entity(name, cfg, backend)
        self._entities = MappingProxyType(materialized)

    @classmethod
    def from_scene_cfg(cls, cfg: SceneCfg, backend: SimBackend) -> EntityScene:
        return cls(cfg.entities, backend)

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
