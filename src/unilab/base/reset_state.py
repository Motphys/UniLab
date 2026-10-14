"""Base-owned reset-state transaction for Manager-Based event terms.

The transaction composes every reset-mode state and model-field write inside
the backend-owned packed transaction (``SimBackend.begin_packed_reset`` →
``unisim.backend.packed_reset.TorchPackedResetComposer``) and commits through
that composer exactly once per lifecycle. It deliberately knows nothing about
task configuration, IPC, runners, or backend-private state.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from unisim.backend.base import (
    BackendMocapPoseBinding,
    BackendRootStateLayout,
    SimBackend,
)
from unisim.dr.types import (
    RESET_TERM_BODY_INERTIA,
    RESET_TERM_BODY_IPOS,
    RESET_TERM_BODY_MASS,
    RESET_TERM_DOF_ARMATURE,
    RESET_TERM_DOF_DAMPING,
    RESET_TERM_DOF_FRICTIONLOSS,
    RESET_TERM_GEOM_FRICTION,
    RESET_TERM_GEOM_SIZE,
    RESET_TERM_GEOM_SOLIMP,
    RESET_TERM_GEOM_SOLREF,
    RESET_TERM_GRAVITY,
    RESET_TERM_KD,
    RESET_TERM_KP,
)
from unisim.entities import EntityStatePatch, SceneResetRequest
from unisim.scene_layout import CompiledSceneLayout

from unilab.base.rotation import quat_apply_inverse

if TYPE_CHECKING:
    from unisim.backend.packed_reset import TorchPackedResetComposer

_RANDOMIZATION_TERM_TAILS: dict[str, tuple[int, ...]] = {
    RESET_TERM_BODY_INERTIA: (3,),
    RESET_TERM_BODY_MASS: (),
    RESET_TERM_BODY_IPOS: (3,),
    RESET_TERM_DOF_ARMATURE: (),
    RESET_TERM_DOF_DAMPING: (),
    RESET_TERM_DOF_FRICTIONLOSS: (),
    RESET_TERM_GEOM_FRICTION: (3,),
    RESET_TERM_GEOM_SIZE: (3,),
    RESET_TERM_GEOM_SOLIMP: (5,),
    RESET_TERM_GEOM_SOLREF: (2,),
    RESET_TERM_GRAVITY: (),
    RESET_TERM_KD: (),
    RESET_TERM_KP: (),
}


def _randomization_term_tail(field: str) -> tuple[int, ...]:
    try:
        return _RANDOMIZATION_TERM_TAILS[field]
    except KeyError as exc:
        raise ValueError(f"unknown reset randomization term {field!r}") from exc


def _readonly_array(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values)
    if result.flags.writeable:
        result = result.copy()
    result.setflags(write=False)
    return result


class ResetStateTransaction:
    """Reusable, fail-closed transaction for reset-mode state mutation."""

    def __init__(
        self,
        backend: SimBackend,
        *,
        default_qpos: np.ndarray | None = None,
        scene_layout: CompiledSceneLayout | None = None,
    ) -> None:
        self._backend = backend
        self._num_envs = backend.num_envs
        self._selected_default_qpos = default_qpos
        self.scene_layout = scene_layout
        if scene_layout is not None and any(
            joint.kind not in ("hinge", "slide")
            for entity in scene_layout.entities
            for joint in entity.joints
        ):
            raise NotImplementedError("mapped manager reset currently supports scalar joints")
        self._entity_values: dict[str, dict[str, np.ndarray]] = {}
        self._entity_fields: dict[str, set[str]] = {}
        self._entity_joints: dict[str, set[str]] = {}
        self._entity_joint_fields: dict[str, dict[str, set[str]]] = {}
        self._entity_layouts = (
            {}
            if scene_layout is None
            else {entity.name: entity for entity in scene_layout.entities}
        )
        self._entity_joint_columns = {
            name: {joint.name: index for index, joint in enumerate(entity.joints)}
            for name, entity in self._entity_layouts.items()
        }
        self._entity_rows: tuple[int, ...] | None = None
        self._entity_row_index: dict[int, int] = {}
        self._restore_entity_controls = False
        self._active = False
        self._tensor_active = False
        self._tensor_active_mask: torch.Tensor | None = None
        self._tensor_has_writes = False
        self._tensor_owner_subset = False
        self._mask_count = torch.zeros((), dtype=torch.int64)
        self._mask_outside_rows = torch.zeros((), dtype=torch.bool)
        self._begin_rows: torch.Tensor | None = None
        self._begin_ids: np.ndarray | None = None
        self._host_has_writes = False
        self._default_qpos: np.ndarray | None = None
        self._default_qvel: np.ndarray | None = None
        self._default_kp: np.ndarray | None = None
        self._default_kd: np.ndarray | None = None
        self._randomization_defaults: dict[str, np.ndarray] = {}
        self._startup_randomization_fields: set[str] = set()
        self._requesting_terms: set[str] = set()
        self._last_commit_had_writes = False
        self._last_set_state_timing_ms: dict[str, float] = {}
        self._mocap_bindings: dict[str, BackendMocapPoseBinding] = {}
        self._mocap_pending: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._packed_reset_device: torch.device | None = None
        self._packed_reset_transaction: TorchPackedResetComposer | None = None
        self._packed_state_default_installed = False
        self._owner_tensor_write_mask: torch.Tensor | None = None
        self._tensor_write_mask: torch.Tensor | None = None
        self._tensor_root_columns: (
            tuple[tuple[int, ...], tuple[int, ...], torch.Tensor, torch.Tensor] | None
        ) = None
        self._tensor_joint_columns: (
            tuple[tuple[int, ...], tuple[int, ...], torch.Tensor, torch.Tensor] | None
        ) = None
        self._motion_payload_valid: torch.Tensor | None = None
        self._tensor_root_payload_valid: torch.Tensor | None = None
        self._deferred_payload_validation = False
        self._deferred_mask_validation = False
        self._commit_verdict_published = False
        self._completed_tensor_has_writes = False

    @property
    def active(self) -> bool:
        """Whether a reset lifecycle currently owns the transaction."""
        return self._active or self._tensor_active

    @property
    def motion_payload_valid(self) -> torch.Tensor:
        """Latest fused motion-reset payload verdict."""
        return self._require_motion_payload_valid()

    @property
    def tensor_root_payload_valid(self) -> torch.Tensor:
        """Latest fused tensor root-reset payload verdict."""
        return self._require_tensor_root_payload_valid()

    @property
    def deferred_payload_validation(self) -> bool:
        """Whether fused reset payload verdicts may defer publication."""
        return self._deferred_payload_validation

    def set_deferred_payload_validation(self, enabled: bool) -> None:
        """Opt in to completion-boundary fused reset payload publication."""
        if not isinstance(enabled, bool):
            raise TypeError(f"deferred payload validation must be bool, got {type(enabled)}")
        self._deferred_payload_validation = enabled

    @property
    def deferred_mask_validation(self) -> bool:
        """Whether reset mask scope/row-count verdicts may defer publication."""
        return self._deferred_mask_validation

    def set_deferred_mask_validation(self, enabled: bool) -> None:
        """Opt in to completion-boundary reset mask verdict publication."""
        if not isinstance(enabled, bool):
            raise TypeError(f"deferred mask validation must be bool, got {type(enabled)}")
        if self.active:
            raise RuntimeError("deferred mask validation cannot change during an active reset")
        self._deferred_mask_validation = enabled

    def publish_mask_validation_verdicts(self) -> None:
        """Fail closed on deferred reset-scope and row-count verdicts."""
        if not self._deferred_mask_validation:
            return
        if bool(self._mask_outside_rows):
            raise ValueError("tensor reset mask attempted a write outside the active mask")
        if self._commit_verdict_published:
            return
        if not self._last_commit_had_writes:
            return
        if bool(self._mask_count.item()):
            self._commit_verdict_published = True
            return
        if self._completed_tensor_has_writes:
            self._commit_verdict_published = True
            return
        raise ValueError("fixed-width tensor reset commit has no selected rows")

    @property
    def tensor_active(self) -> bool:
        """Whether the active reset composes device tensor rows without host staging."""
        return self._tensor_active

    def _require_motion_payload_valid(self) -> torch.Tensor:
        if self._motion_payload_valid is None:
            return torch.ones((), dtype=torch.bool)
        return self._motion_payload_valid

    def _require_tensor_root_payload_valid(self) -> torch.Tensor:
        if self._tensor_root_payload_valid is None:
            return torch.ones((), dtype=torch.bool)
        return self._tensor_root_payload_valid

    def publish_payload_validation_verdicts(self, *, term_name: str) -> None:
        """Fail closed on fused reset payload verdicts at a completion boundary."""
        if not self._deferred_payload_validation:
            return
        if bool(self.tensor_root_payload_valid) and bool(self.motion_payload_valid):
            return
        raise ValueError(
            f"EventManager term '{term_name}' tensor reset payload contains NaN/Inf "
            "or a non-unit quaternion"
        )

    @property
    def last_commit_had_writes(self) -> bool:
        """Whether the most recent scoped commit submitted rows to the backend."""
        return self._last_commit_had_writes

    @property
    def last_set_state_timing_ms(self) -> dict[str, float]:
        """Sub-timings from the most recent commit's set_state call.

        Always includes ``dr_reset_set_state_ms`` (outer wall-clock around the
        backend call); backend-reported ``set_state_*_ms`` and packed-composer
        ``packed_reset_*_ms`` sub-keys are merged in when present. Empty when
        the last commit had no staged writes.
        """
        return self._last_set_state_timing_ms

    @contextmanager
    def scoped_startup(self, env_ids: torch.Tensor) -> Iterator[ResetStateTransaction]:
        """Begin the all-row startup transaction and commit after all terms succeed."""
        ids = self._validate_ids(env_ids, capability="startup")
        expected_ids = np.arange(self._num_envs, dtype=np.int64)
        if not np.array_equal(ids, expected_ids):
            raise ValueError("startup reset-state transaction must include every environment row")
        self.begin(env_ids)
        try:
            yield self
        except BaseException:
            self.abort()
            raise
        else:
            self.commit()

    def declare_packed_reset_device(self, device: torch.device) -> None:
        """Declare the packed reset device and fail closed without the capability."""
        resolved = torch.device(device)
        if resolved.type == "cuda" and resolved.index is None:
            resolved = torch.device("cuda", index=torch.cuda.current_device())
        if self._packed_reset_transaction is not None and self._packed_reset_device == resolved:
            return
        capabilities = self._backend.get_tensor_capabilities()
        if not getattr(capabilities, "packed_reset", False):
            raise NotImplementedError(
                "backend does not declare the packed_reset capability required by the "
                "packed reset contract: SimBackend.begin_packed_reset"
            )
        begin_packed_reset: Any = getattr(self._backend, "begin_packed_reset", None)
        if not callable(begin_packed_reset):
            raise NotImplementedError(
                "backend declares packed_reset without SimBackend.begin_packed_reset"
            )
        self._packed_reset_device = resolved
        self._owner_tensor_write_mask = torch.zeros(
            self._num_envs, dtype=torch.bool, device=resolved
        )
        self._packed_reset_transaction = cast(
            "TorchPackedResetComposer", begin_packed_reset(device=resolved)
        )
        self._packed_state_default_installed = False

    @contextmanager
    def scoped_device_event_tensor(
        self, reset_mask: torch.Tensor
    ) -> Iterator[ResetStateTransaction]:
        """Begin a fixed-width owner reset and commit it."""
        self.begin_tensor(reset_mask)
        try:
            yield self
        except BaseException:
            self.abort()
            raise
        else:
            self.commit_device_tensor()

    @contextmanager
    def scoped_device_owner_tensor(
        self, reset_mask: torch.Tensor
    ) -> Iterator[ResetStateTransaction]:
        """Begin a full-width step-owner tensor transaction."""
        self.begin_tensor(reset_mask, owner_subset=True)
        try:
            yield self
        except BaseException:
            self.abort()
            raise
        else:
            self.commit_device_tensor()

    def begin(self, env_ids: torch.Tensor) -> None:
        """Open a transaction for the concrete reset environment IDs."""
        if self._active or self._tensor_active:
            raise RuntimeError("ManagerBased reset-state transaction is already active")
        ids = self._validate_ids(env_ids, capability="begin")
        transaction = self._require_packed_transaction()
        device = self._packed_reset_device
        assert device is not None
        self._ensure_packed_state_default("begin")
        rows = torch.as_tensor(ids, dtype=torch.int64, device=device)
        transaction.begin_state(rows)
        self._begin_rows = rows
        self._begin_ids = ids
        self._host_has_writes = False
        self._requesting_terms.clear()
        self._last_commit_had_writes = False
        self._last_set_state_timing_ms = {}
        self._entity_values.clear()
        self._entity_fields.clear()
        self._entity_joints.clear()
        self._entity_joint_fields.clear()
        self._entity_rows = None
        self._entity_row_index.clear()
        self._restore_entity_controls = False
        self._active = True

    def begin_tensor(
        self,
        reset_mask: torch.Tensor,
        *,
        owner_subset: bool = False,
    ) -> None:
        """Open a fixed-width packed reset transaction selected by a bool mask."""
        if self._active or self._tensor_active:
            raise RuntimeError("ManagerBased reset-state transaction is already active")
        device = self._packed_reset_device
        if device is None:
            raise RuntimeError(
                "tensor reset transaction requires its selected device to be declared"
            )
        if not isinstance(reset_mask, torch.Tensor):
            raise TypeError("fixed-width tensor reset requires a Torch bool mask")
        if reset_mask.shape != (self._num_envs,) or reset_mask.dtype != torch.bool:
            raise ValueError(
                f"fixed-width tensor reset mask must have shape {(self._num_envs,)} and bool dtype"
            )
        if reset_mask.device != device:
            raise ValueError(
                "fixed-width tensor reset mask must live on the declared reset device "
                f"{device}; got {reset_mask.device}"
            )
        transaction = self._require_packed_transaction()
        self._ensure_packed_state_default("begin_tensor")
        transaction.begin_state_mask(reset_mask)
        self._tensor_active_mask = reset_mask
        if owner_subset:
            write_mask = self._owner_tensor_write_mask
            assert write_mask is not None
            write_mask.zero_()
            self._tensor_write_mask = write_mask
        else:
            self._tensor_write_mask = None
        self._tensor_has_writes = False
        self._tensor_owner_subset = owner_subset
        self._requesting_terms.clear()
        self._last_commit_had_writes = False
        self._commit_verdict_published = False
        self._tensor_active = True

    def write_entity_state(
        self,
        entity: str,
        env_ids: torch.Tensor,
        *,
        term_name: str,
        root_pose: np.ndarray | None = None,
        root_velocity: np.ndarray | None = None,
        joint_positions: np.ndarray | None = None,
        joint_velocities: np.ndarray | None = None,
        joint_names: tuple[str, ...] = (),
    ) -> None:
        """Stage one public physical-entity patch in the current manager transaction."""
        self._require_active()
        if self.scene_layout is None:
            raise NotImplementedError("physical entity patches require a compiled scene layout")
        ids = self._validate_ids(env_ids, capability=term_name)
        begin_ids = self._begin_ids
        if begin_ids is None or not np.isin(ids, begin_ids).all():
            raise ValueError("entity write is outside the active reset")
        if not len(ids):
            return
        patch = EntityStatePatch(
            entity, root_pose, root_velocity, joint_positions, joint_velocities, joint_names
        )
        request = SceneResetRequest(tuple(int(i) for i in ids), (patch,))
        self.scene_layout.validate_reset(request, num_envs=self._num_envs)
        rows = tuple(sorted(request.env_ids))
        if self._entity_rows is not None and self._entity_rows != rows:
            raise NotImplementedError(
                "one entity transaction requires the same selected env rows for every patch"
            )
        self._entity_rows = rows
        self._entity_row_index = {value: index for index, value in enumerate(rows)}
        incoming_rows = {value: index for index, value in enumerate(request.env_ids)}
        order = [incoming_rows[i] for i in rows]
        owner = self._entity_layouts[entity]
        if entity not in self._entity_values:
            self._entity_values[entity] = {}
            self._entity_fields[entity] = set()
            self._entity_joints[entity] = set()
            self._entity_joint_fields[entity] = {}
        values = self._entity_values[entity]
        selected_joints = joint_names or tuple(joint.name for joint in owner.joints)
        for field in ("root_pose", "root_velocity", "joint_positions", "joint_velocities"):
            incoming = getattr(patch, field)
            if incoming is None:
                continue
            if field.startswith("joint"):
                columns = [self._entity_joint_columns[entity][name] for name in selected_joints]
                if field not in values:
                    values[field] = np.empty((len(rows), len(owner.joints)), dtype=np.float64)
                values[field][:, columns] = incoming[order]
                self._entity_joints[entity].update(selected_joints)
                self._entity_joint_fields[entity].setdefault(field, set()).update(selected_joints)
            else:
                values[field] = incoming[order]
            self._entity_fields[entity].add(field)
        self._requesting_terms.add(term_name)

    def read_entity_root_pose(self, entity: str, env_ids: torch.Tensor) -> np.ndarray:
        """Read staged root pose or a detached current public snapshot."""
        self._require_active()
        ids = self._validate_ids(env_ids, capability="read_entity_root_pose")
        begin_ids = self._begin_ids
        if begin_ids is None or not np.isin(ids, begin_ids).all():
            raise ValueError("entity state read is outside the active reset")
        if "root_pose" in self._entity_values.get(entity, {}):
            rows = [self._entity_row_index[int(i)] for i in ids]
            return self._entity_values[entity]["root_pose"][rows].copy()
        return self._backend.get_entity_state(entity)["root_pose"][ids].copy()

    def _commit_entities(self) -> None:
        assert self.scene_layout is not None
        if self._entity_rows is None:
            return
        if self._host_has_writes or self._mocap_pending:
            raise NotImplementedError("cannot mix entity patches with packed state/DR writes")
        patches = []
        for name, values in self._entity_values.items():
            entity = self._entity_layouts[name]
            joint_names = tuple(
                j.name for j in entity.joints if j.name in self._entity_joints[name]
            )
            columns = [self._entity_joint_columns[name][n] for n in joint_names]
            current = None
            for field, written in self._entity_joint_fields[name].items():
                missing = self._entity_joints[name] - written
                if missing:
                    if current is None:
                        current = self._backend.get_entity_state(name)
                    absent = [self._entity_joint_columns[name][joint] for joint in missing]
                    values[field][:, absent] = current[field][np.ix_(self._entity_rows, absent)]
            fields = {
                field: values[field][:, columns] if field.startswith("joint") else values[field]
                for field in self._entity_fields[name]
            }
            patches.append(EntityStatePatch(name, joint_names=joint_names, **fields))
        request = SceneResetRequest(
            self._entity_rows,
            tuple(patches),
            restore_default_controls=self._restore_entity_controls,
        )
        self.scene_layout.validate_reset(request, num_envs=self._num_envs)
        started = time.perf_counter()
        self._backend.reset_entities(request)
        self._last_commit_had_writes = True
        self._last_set_state_timing_ms = {
            "dr_reset_set_state_ms": (time.perf_counter() - started) * 1000
        }

    def bind_geom_size_write(
        self,
        column_ids: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bind immutable geom_size defaults through the declared backend capability."""
        default = self._materialize_randomization_default(
            RESET_TERM_GEOM_SIZE,
            expected_tail=(3,),
            term_name=term_name,
        )
        columns = self._validate_columns(
            column_ids,
            width=self._randomization_default_width(default, field=RESET_TERM_GEOM_SIZE),
            capability="geom_size IDs",
            term_name=term_name,
        )
        selected = self._select_randomization_default_columns(
            default,
            columns,
            field=RESET_TERM_GEOM_SIZE,
        )
        return self._readonly_binding(columns, selected)

    def write_geom_size(
        self,
        env_ids: torch.Tensor,
        column_ids: np.ndarray,
        values: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage selected geom_size values in the active reset."""
        self._write_selected_randomization(
            RESET_TERM_GEOM_SIZE,
            env_ids,
            column_ids,
            values,
            value_tail=(3,),
            term_name=term_name,
        )

    def bind_geom_solref_write(
        self,
        column_ids: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bind immutable geom_solref defaults through the declared backend capability."""
        default = self._materialize_randomization_default(
            RESET_TERM_GEOM_SOLREF,
            expected_tail=(2,),
            term_name=term_name,
        )
        columns = self._validate_columns(
            column_ids,
            width=self._randomization_default_width(default, field=RESET_TERM_GEOM_SOLREF),
            capability="geom_solref IDs",
            term_name=term_name,
        )
        selected = self._select_randomization_default_columns(
            default,
            columns,
            field=RESET_TERM_GEOM_SOLREF,
        )
        return self._readonly_binding(columns, selected)

    def write_geom_solref(
        self,
        env_ids: torch.Tensor,
        column_ids: np.ndarray,
        values: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage selected geom_solref values in the active reset."""
        self._write_selected_randomization(
            RESET_TERM_GEOM_SOLREF,
            env_ids,
            column_ids,
            values,
            value_tail=(2,),
            term_name=term_name,
        )

    def bind_geom_solimp_write(
        self,
        column_ids: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bind immutable geom_solimp defaults through the declared backend capability."""
        default = self._materialize_randomization_default(
            RESET_TERM_GEOM_SOLIMP,
            expected_tail=(5,),
            term_name=term_name,
        )
        columns = self._validate_columns(
            column_ids,
            width=self._randomization_default_width(default, field=RESET_TERM_GEOM_SOLIMP),
            capability="geom_solimp IDs",
            term_name=term_name,
        )
        selected = self._select_randomization_default_columns(
            default,
            columns,
            field=RESET_TERM_GEOM_SOLIMP,
        )
        return self._readonly_binding(columns, selected)

    def write_geom_solimp(
        self,
        env_ids: torch.Tensor,
        column_ids: np.ndarray,
        values: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage selected geom_solimp values in the active reset."""
        self._write_selected_randomization(
            RESET_TERM_GEOM_SOLIMP,
            env_ids,
            column_ids,
            values,
            value_tail=(5,),
            term_name=term_name,
        )

    def bind_dof_damping_write(
        self,
        column_ids: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bind immutable dof_damping defaults through the declared backend capability."""
        default = self._materialize_randomization_default(
            RESET_TERM_DOF_DAMPING,
            expected_tail=(),
            term_name=term_name,
        )
        columns = self._validate_columns(
            column_ids,
            width=self._randomization_default_width(default, field=RESET_TERM_DOF_DAMPING),
            capability="dof_damping IDs",
            term_name=term_name,
        )
        selected = self._select_randomization_default_columns(
            default,
            columns,
            field=RESET_TERM_DOF_DAMPING,
        )
        return self._readonly_binding(columns, selected)

    def write_dof_damping(
        self,
        env_ids: torch.Tensor,
        column_ids: np.ndarray,
        values: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage selected dof_damping values in the active reset."""
        self._write_selected_randomization(
            RESET_TERM_DOF_DAMPING,
            env_ids,
            column_ids,
            values,
            value_tail=(),
            term_name=term_name,
        )

    def bind_dof_frictionloss_write(
        self,
        column_ids: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bind immutable dof_frictionloss defaults through the declared backend capability."""
        default = self._materialize_randomization_default(
            RESET_TERM_DOF_FRICTIONLOSS,
            expected_tail=(),
            term_name=term_name,
        )
        columns = self._validate_columns(
            column_ids,
            width=self._randomization_default_width(
                default,
                field=RESET_TERM_DOF_FRICTIONLOSS,
            ),
            capability="dof_frictionloss IDs",
            term_name=term_name,
        )
        selected = self._select_randomization_default_columns(
            default,
            columns,
            field=RESET_TERM_DOF_FRICTIONLOSS,
        )
        return self._readonly_binding(columns, selected)

    def write_dof_frictionloss(
        self,
        env_ids: torch.Tensor,
        column_ids: np.ndarray,
        values: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage selected dof_frictionloss values in the active reset."""
        self._write_selected_randomization(
            RESET_TERM_DOF_FRICTIONLOSS,
            env_ids,
            column_ids,
            values,
            value_tail=(),
            term_name=term_name,
        )

    def bind_body_mass_write(
        self,
        body_ids: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Bind body-mass columns and immutable backend defaults on the cold path."""
        default = self._materialize_randomization_default(
            RESET_TERM_BODY_MASS,
            expected_tail=(),
            term_name=term_name,
        )
        columns = self._validate_columns(
            body_ids,
            width=self._randomization_default_width(default, field=RESET_TERM_BODY_MASS),
            capability="body mass IDs",
            term_name=term_name,
        )
        selected = self._select_randomization_default_columns(
            default,
            columns,
            field=RESET_TERM_BODY_MASS,
        )
        return columns, self._torch_default_table(selected)

    def bind_body_ipos_write(
        self,
        body_ids: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Bind body inertial-position columns and immutable backend defaults."""
        default = self._materialize_randomization_default(
            RESET_TERM_BODY_IPOS,
            expected_tail=(3,),
            term_name=term_name,
        )
        columns = self._validate_columns(
            body_ids,
            width=self._randomization_default_width(default, field=RESET_TERM_BODY_IPOS),
            capability="body ipos IDs",
            term_name=term_name,
        )
        selected = self._select_randomization_default_columns(
            default,
            columns,
            field=RESET_TERM_BODY_IPOS,
        )
        return columns, self._torch_default_table(selected)

    def bind_body_inertia_write(
        self,
        body_ids: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Bind body-inertia columns and authoritative backend defaults.

        UniSim returns either a canonical ``(nbody, 3)`` table or a per-world
        ``(num_envs, nbody, 3)`` table. No caller-side model compilation or
        body-order cross-check is needed.
        """
        inertia = self._materialize_randomization_default(
            RESET_TERM_BODY_INERTIA,
            expected_tail=(3,),
            term_name=term_name,
        )
        if np.any(inertia < 0.0):
            raise ValueError(
                f"EventManager term '{term_name}' default body_inertia contains negative values"
            )
        columns = self._validate_columns(
            body_ids,
            width=self._randomization_default_width(default=inertia, field=RESET_TERM_BODY_INERTIA),
            capability="body inertia IDs",
            term_name=term_name,
        )
        selected = self._select_randomization_default_columns(
            inertia,
            columns,
            field=RESET_TERM_BODY_INERTIA,
        )
        return columns, self._torch_default_table(selected)

    def bind_gravity_write(self, *, term_name: str) -> np.ndarray:
        """Bind the immutable backend gravity vector on the cold path."""
        return self._materialize_randomization_default(
            RESET_TERM_GRAVITY,
            expected_tail=(),
            term_name=term_name,
        )

    def bind_gravity_write_torch(self, *, term_name: str, device: torch.device) -> torch.Tensor:
        """Bind the immutable backend gravity vector as a Torch table."""
        default = self.bind_gravity_write(term_name=term_name)
        return torch.from_numpy(np.array(default, copy=True)).to(device=device, dtype=torch.float64)

    def bind_dof_armature_write(
        self,
        dof_ids: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Bind DOF-armature columns and immutable backend defaults."""
        default = self._materialize_randomization_default(
            RESET_TERM_DOF_ARMATURE,
            expected_tail=(),
            term_name=term_name,
        )
        columns = self._validate_columns(
            dof_ids,
            width=self._randomization_default_width(default, field=RESET_TERM_DOF_ARMATURE),
            capability="DOF armature IDs",
            term_name=term_name,
        )
        selected = self._select_randomization_default_columns(
            default,
            columns,
            field=RESET_TERM_DOF_ARMATURE,
        )
        return columns, self._torch_default_table(selected)

    def bind_geom_friction_write(
        self,
        geom_ids: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Bind geom-friction rows and immutable backend defaults."""
        default = self._materialize_randomization_default(
            RESET_TERM_GEOM_FRICTION,
            expected_tail=(3,),
            term_name=term_name,
        )
        columns = self._validate_columns(
            geom_ids,
            width=self._randomization_default_width(default, field=RESET_TERM_GEOM_FRICTION),
            capability="geom friction IDs",
            term_name=term_name,
        )
        selected = self._select_randomization_default_columns(
            default,
            columns,
            field=RESET_TERM_GEOM_FRICTION,
        )
        return columns, self._torch_default_table(selected)

    def _torch_default_table(self, values: np.ndarray) -> torch.Tensor:
        """Materialize an immutable backend default as a Torch table."""
        return torch.from_numpy(np.array(values, dtype=np.float64, copy=True)).to(
            dtype=torch.float64
        )

    def write_body_mass(
        self,
        env_ids: torch.Tensor,
        body_ids: np.ndarray,
        values: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage selected body masses in the exactly-once reset payload."""
        self._write_selected_randomization(
            RESET_TERM_BODY_MASS,
            env_ids,
            body_ids,
            values,
            value_tail=(),
            term_name=term_name,
        )

    def write_body_ipos(
        self,
        env_ids: torch.Tensor,
        body_ids: np.ndarray,
        values: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage selected body inertial positions in the reset payload."""
        self._write_selected_randomization(
            RESET_TERM_BODY_IPOS,
            env_ids,
            body_ids,
            values,
            value_tail=(3,),
            term_name=term_name,
        )

    def write_body_inertia(
        self,
        env_ids: torch.Tensor,
        body_ids: np.ndarray,
        values: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage selected body principal-inertia diagonals in the reset payload."""
        self._write_selected_randomization(
            RESET_TERM_BODY_INERTIA,
            env_ids,
            body_ids,
            values,
            value_tail=(3,),
            term_name=term_name,
        )

    def write_gravity(
        self,
        env_ids: torch.Tensor,
        values: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage per-environment gravity vectors in the reset payload."""
        ids = self._prepare_state_write(
            env_ids,
            capability="gravity",
            term_name=term_name,
        )
        self._require_randomization_default(RESET_TERM_GRAVITY, term_name)
        gravity = self._validate_values(
            values,
            shape=(ids.size, 3),
            capability="gravity",
            term_name=term_name,
        )
        self._require_packed_transaction().write_model_field(
            RESET_TERM_GRAVITY,
            self._host_rows(ids),
            self._packed_values(gravity),
        )
        self._host_has_writes = True

    def write_dof_armature(
        self,
        env_ids: torch.Tensor,
        dof_ids: np.ndarray,
        values: np.ndarray | torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage selected DOF armatures in the reset payload."""
        self._write_selected_randomization(
            RESET_TERM_DOF_ARMATURE,
            env_ids,
            dof_ids,
            values,
            value_tail=(),
            term_name=term_name,
        )

    def write_geom_friction(
        self,
        env_ids: torch.Tensor,
        geom_ids: np.ndarray,
        values: np.ndarray | torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage selected three-axis geom friction in the reset payload."""
        self._write_selected_randomization(
            RESET_TERM_GEOM_FRICTION,
            env_ids,
            geom_ids,
            values,
            value_tail=(3,),
            term_name=term_name,
        )

    def bind_actuator_gain_write(
        self,
        actuator_ids: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Resolve gain mutation capability and immutable defaults on the cold path."""
        columns = self._validate_columns(
            actuator_ids,
            width=self._backend.num_actuators,
            capability="actuator IDs",
            term_name=term_name,
        )
        self._materialize_default_actuator_gains(term_name)
        assert self._default_kp is not None
        assert self._default_kd is not None
        selected_kp = self._torch_default_table(
            self._select_randomization_default_columns(
                self._default_kp,
                columns,
                field=RESET_TERM_KP,
            )
        )
        selected_kd = self._torch_default_table(
            self._select_randomization_default_columns(
                self._default_kd,
                columns,
                field=RESET_TERM_KD,
            )
        )
        return columns, selected_kp, selected_kd

    def write_actuator_gains(
        self,
        env_ids: torch.Tensor,
        actuator_ids: np.ndarray,
        kp: torch.Tensor,
        kd: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage selected per-environment actuator gains in the reset transaction."""
        ids = self._prepare_state_write(
            env_ids,
            capability="actuator-gain",
            term_name=term_name,
        )
        columns = self._validate_columns(
            actuator_ids,
            width=self._backend.num_actuators,
            capability="actuator IDs",
            term_name=term_name,
        )
        self._materialize_default_actuator_gains(term_name)
        gains_shape = (ids.size, columns.size)
        kp_values = self._validate_values(
            kp,
            shape=gains_shape,
            capability="actuator kp",
            term_name=term_name,
        )
        kd_values = self._validate_values(
            kd,
            shape=gains_shape,
            capability="actuator kd",
            term_name=term_name,
        )
        transaction = self._require_packed_transaction()
        rows = self._host_rows(ids)
        transaction.write_model_field(
            RESET_TERM_KP, rows, self._packed_values(kp_values), columns=columns
        )
        transaction.write_model_field(
            RESET_TERM_KD, rows, self._packed_values(kd_values), columns=columns
        )
        self._host_has_writes = True

    def write_randomization_tensor_mask(
        self,
        field: str,
        reset_mask: torch.Tensor,
        columns: np.ndarray,
        values: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage a packed transaction's full-width final DR table by mask."""
        effective_mask = self._tensor_mask_scope(
            reset_mask, capability="write_randomization_tensor_mask"
        )
        transaction = self._require_packed_transaction()
        tail = _randomization_term_tail(field)
        base = self._materialize_tensor_randomization_default(field, term_name)
        resolved = self._validate_columns(
            columns,
            width=int(base.shape[1]),
            capability=f"{field} column IDs",
            term_name=term_name,
        )
        expected_shape = (self._num_envs, resolved.size, *tail)
        if not isinstance(values, torch.Tensor):
            raise TypeError(
                f"EventManager term '{term_name}' tensor {field} must be torch.Tensor, "
                f"got {type(values).__name__}"
            )
        if tuple(values.shape) != expected_shape:
            raise ValueError(
                f"EventManager term '{term_name}' tensor mask {field} must have shape "
                f"{expected_shape}; got {tuple(values.shape)}"
            )
        if values.dtype != torch.float32 or not values.is_contiguous():
            raise TypeError(
                f"EventManager term '{term_name}' tensor mask {field} must be contiguous float32"
            )
        device = self._packed_reset_device
        assert device is not None
        if values.device != device:
            raise ValueError(
                f"EventManager term '{term_name}' tensor mask {field} must live on "
                f"{device}; got {values.device}"
            )
        transaction.write_model_field_mask(
            field,
            effective_mask,
            values,
            columns=resolved,
        )
        self._tensor_has_writes = True

    def _materialize_tensor_randomization_default(self, field: str, term_name: str) -> torch.Tensor:
        """Build the immutable per-env device base for one DR field on the cold path."""
        device = self._packed_reset_device
        if device is None:
            raise RuntimeError("device tensor reset requires its selected device to be declared")
        if field == RESET_TERM_KP:
            self._materialize_default_actuator_gains(term_name)
            default = self._default_kp
        elif field == RESET_TERM_KD:
            self._materialize_default_actuator_gains(term_name)
            default = self._default_kd
        else:
            default = self._require_randomization_default(field, term_name)
        assert default is not None
        table = torch.as_tensor(np.array(default, dtype=np.float32, copy=True), device=device)
        if self._default_is_per_env(default, field=field):
            return table
        return table.unsqueeze(0).expand(self._num_envs, *table.shape).contiguous()

    def reset_to_default(self, reset_mask: torch.Tensor, *, term_name: str) -> None:
        """Stage backend default qpos/qvel for the active full-width mask."""
        self._require_active()
        active_mask = self._tensor_active_mask
        if active_mask is None:
            raise RuntimeError("fixed-width default reset requires an active mask transaction")
        if self._tensor_owner_subset:
            if self._tensor_mask_has_outside_rows(reset_mask, active_mask):
                raise ValueError(
                    f"EventManager term '{term_name}' attempted default reset outside "
                    "the active mask"
                )
            self._record_tensor_write_mask(reset_mask)
        elif active_mask is not reset_mask:
            raise ValueError(
                f"EventManager term '{term_name}' attempted default reset outside the active mask"
            )
        if self.scene_layout is not None:
            raise NotImplementedError(
                "mask reset does not support mapped entity default-state restoration"
            )
        if self._mask_is_empty(reset_mask):
            return
        self._requesting_terms.add(term_name)
        self._require_packed_transaction().begin_state_mask(reset_mask)
        self._tensor_has_writes = True

    def write_joint_state(
        self,
        env_ids: torch.Tensor,
        qpos_indices: np.ndarray,
        qvel_indices: np.ndarray,
        position: torch.Tensor,
        velocity: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage selected joint position and velocity columns in the reset batch."""
        self._require_active()
        ids = self._validate_ids(env_ids, capability="write_joint_state")
        begin_ids = self._begin_ids
        assert begin_ids is not None
        outside = ids[~np.isin(ids, begin_ids)]
        if outside.size:
            raise ValueError(
                f"EventManager term '{term_name}' attempted joint-state mutation outside "
                f"the active reset: {outside.tolist()}"
            )
        self._requesting_terms.add(term_name)
        self._materialize_default_state(term_name)
        assert self._default_qpos is not None
        assert self._default_qvel is not None

        pos_columns = self._validate_columns(
            qpos_indices,
            width=self._default_qpos.size,
            capability="qpos indices",
            term_name=term_name,
        )
        vel_columns = self._validate_columns(
            qvel_indices,
            width=self._default_qvel.size,
            capability="qvel indices",
            term_name=term_name,
        )
        if pos_columns.size != vel_columns.size:
            raise ValueError(
                f"EventManager term '{term_name}' joint-state qpos/qvel index counts differ: "
                f"{pos_columns.size} != {vel_columns.size}"
            )
        positions = self._validate_values(
            position,
            shape=(ids.size, pos_columns.size),
            capability="joint position",
            term_name=term_name,
        )
        velocities = self._validate_values(
            velocity,
            shape=(ids.size, vel_columns.size),
            capability="joint velocity",
            term_name=term_name,
        )

        self._require_packed_transaction().write_state(
            self._host_rows(ids),
            qpos_columns=pos_columns,
            qpos_values=self._packed_values(positions),
            qvel_columns=vel_columns,
            qvel_values=self._packed_values(velocities),
        )
        self._host_has_writes = True

    def write_joint_state_mask(
        self,
        reset_mask: torch.Tensor,
        qpos_indices: np.ndarray,
        qvel_indices: np.ndarray,
        position: torch.Tensor,
        velocity: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage full-width joint values for the active reset mask."""
        self._require_active()
        active_mask = self._tensor_active_mask
        if active_mask is None:
            raise RuntimeError("ManagerBased mask reset transaction is not active")
        if self._tensor_owner_subset:
            if self._tensor_mask_has_outside_rows(reset_mask, active_mask):
                raise ValueError(
                    f"EventManager term '{term_name}' attempted joint-state mutation outside "
                    "the active mask"
                )
        elif active_mask is not reset_mask:
            raise ValueError(
                f"EventManager term '{term_name}' attempted joint-state mutation outside "
                "the active mask"
            )
        self._requesting_terms.add(term_name)
        assert self._default_qpos is not None
        assert self._default_qvel is not None
        pos_columns = self._validate_columns(
            qpos_indices,
            width=self._default_qpos.size,
            capability="qpos indices",
            term_name=term_name,
        )
        vel_columns = self._validate_columns(
            qvel_indices,
            width=self._default_qvel.size,
            capability="qvel indices",
            term_name=term_name,
        )
        if pos_columns.size != vel_columns.size:
            raise ValueError(
                f"EventManager term '{term_name}' joint-state qpos/qvel index counts differ: "
                f"{pos_columns.size} != {vel_columns.size}"
            )
        positions = self._validate_tensor_values(
            position,
            shape=(self._num_envs, pos_columns.size),
            capability="joint position",
            term_name=term_name,
        )
        velocities = self._validate_tensor_values(
            velocity,
            shape=(self._num_envs, vel_columns.size),
            capability="joint velocity",
            term_name=term_name,
        )
        transaction = self._require_packed_transaction()
        device = self._packed_reset_device
        assert device is not None
        if self._tensor_owner_subset:
            self._record_tensor_write_mask(reset_mask)
            if self._mask_is_empty(reset_mask):
                return
        transaction.write_state_mask(
            reset_mask,
            qpos_columns=pos_columns,
            qpos_values=positions.to(device=device, dtype=torch.float32).contiguous(),
            qvel_columns=vel_columns,
            qvel_values=velocities.to(device=device, dtype=torch.float32).contiguous(),
        )
        self._tensor_has_writes = True

    def write_root_state(
        self,
        env_ids: torch.Tensor,
        layout: BackendRootStateLayout,
        root_state: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage a community 13-D world-frame root state."""
        self._require_active()
        ids = self._validate_ids(env_ids, capability="write_root_state")
        values = self._validate_values(
            root_state,
            shape=(ids.size, 13),
            capability="root state",
            term_name=term_name,
        )
        self.write_root_pose(
            env_ids,
            layout,
            torch.from_numpy(values[:, :7]),
            term_name=term_name,
        )
        self.write_root_velocity(
            env_ids,
            layout,
            torch.from_numpy(values[:, 7:]),
            term_name=term_name,
        )

    def _validate_motion_tensor_payload(
        self,
        *,
        root_state: torch.Tensor,
        joint_position: torch.Tensor,
        joint_velocity: torch.Tensor,
        term_name: str,
    ) -> None:
        """Validate the fused motion payload through one device synchronization."""
        quat_norm = torch.linalg.vector_norm(root_state[:, 3:7], dim=-1)
        valid_quat = torch.isclose(
            quat_norm,
            torch.ones_like(quat_norm),
            rtol=1e-5,
            atol=1e-6,
        )
        valid = (
            torch.isfinite(root_state).all()
            & torch.isfinite(joint_position).all()
            & torch.isfinite(joint_velocity).all()
            & valid_quat.all()
        )
        # Avoid a Python bool conversion of the aggregate: scalar conversion is
        # itself synchronized and this fused reset path already has one device
        # reduction. Compare against a device-resident true scalar so the
        # success path remains non-synchronizing; invalid diagnostics below are
        # still allowed to synchronize and identify the offending payload.
        if self._motion_payload_valid is None or (
            self._motion_payload_valid.device != valid.device
        ):
            self._motion_payload_valid = torch.empty_like(valid)
        self._motion_payload_valid.copy_(valid)
        if self._deferred_payload_validation:
            return
        if torch.equal(valid, torch.ones_like(valid)):
            return
        payloads = (
            ("root state", root_state),
            ("joint position", joint_position),
            ("joint velocity", joint_velocity),
        )
        for name, values in payloads:
            if not bool(torch.isfinite(values).all()):
                raise ValueError(
                    f"EventManager term '{term_name}' tensor motion {name} contains NaN or Inf"
                )
        self._validate_tensor_root_quaternions(root_state[:, 3:7], term_name=term_name)

    def write_root_state_tensor(
        self,
        reset_mask: torch.Tensor,
        layout: BackendRootStateLayout,
        root_state: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage a full-width root-state payload selected by a bool mask."""
        effective_mask = self._tensor_mask_scope(
            reset_mask, capability="write_root_state_tensor_mask"
        )
        if root_state.shape != (self._num_envs, 13):
            raise ValueError(
                f"EventManager term '{term_name}' tensor mask root state must have shape "
                f"{(self._num_envs, 13)}; got {tuple(root_state.shape)}"
            )
        if root_state.dtype != torch.float32 or not root_state.is_contiguous():
            raise TypeError(
                f"EventManager term '{term_name}' tensor mask root state must be contiguous float32"
            )
        if root_state.device != self._packed_reset_device:
            raise ValueError(
                f"EventManager term '{term_name}' tensor mask root state must live on "
                f"{self._packed_reset_device}; got {root_state.device}"
            )
        self._validate_tensor_root_payload(root_state, term_name=term_name)
        qpos_columns_t, qvel_columns_t = self._tensor_root_columns_for_layout(
            layout, term_name=term_name
        )
        transaction = self._require_packed_transaction()
        transaction.write_state_mask(
            effective_mask,
            qpos_columns=qpos_columns_t,
            qpos_values=root_state[:, :7].contiguous(),
            qvel_columns=qvel_columns_t,
            qvel_values=root_state[:, 7:13].contiguous(),
        )
        self._tensor_has_writes = True

    def _require_tensor_mask_write(
        self, reset_mask: torch.Tensor, *, capability: str
    ) -> torch.Tensor:
        if not self._tensor_active or self._tensor_active_mask is None:
            raise RuntimeError(
                f"ManagerBased tensor mask {capability} requires an active mask reset"
            )
        if self._tensor_owner_subset:
            active_mask = self._tensor_active_mask
            if self._tensor_mask_has_outside_rows(reset_mask, active_mask):
                raise ValueError(
                    f"ManagerBased tensor mask {capability} attempted a write outside "
                    "the active mask"
                )
        elif reset_mask is not self._tensor_active_mask:
            raise ValueError(f"ManagerBased tensor mask {capability} must use the transaction mask")
        self._requesting_terms.add(capability)
        return self._tensor_active_mask

    def _tensor_mask_scope(self, reset_mask: torch.Tensor, *, capability: str) -> torch.Tensor:
        """Resolve the fixed-width scope selected by a mask."""
        self._require_tensor_mask_write(reset_mask, capability=capability)
        if self._tensor_owner_subset:
            active_mask = self._tensor_active_mask
            assert active_mask is not None
            self._record_tensor_write_mask(reset_mask)
            return reset_mask & active_mask
        return reset_mask

    def _tensor_mask_has_outside_rows(
        self, reset_mask: torch.Tensor, active_mask: torch.Tensor
    ) -> bool:
        outside = (reset_mask & ~active_mask).any()
        self._mask_outside_rows.copy_(outside)
        if self._deferred_mask_validation:
            return False
        return bool(outside)

    def _mask_is_empty(self, reset_mask: torch.Tensor) -> bool:
        self._mask_count.copy_(reset_mask.sum())
        if self._deferred_mask_validation:
            return False
        return bool(self._mask_count.item() == 0)

    def _commit_has_rows(self, commit_mask: torch.Tensor) -> bool:
        return not self._mask_is_empty(commit_mask)

    def _record_tensor_write_mask(self, reset_mask: torch.Tensor) -> None:
        """Extend the packed owner-subset carrier with one actual write mask."""
        write_mask = self._tensor_write_mask
        if write_mask is None:
            if not self._tensor_owner_subset:
                return
            raise RuntimeError("owner tensor reset has no packed actual-write mask")
        write_mask.logical_or_(reset_mask)

    def write_motion_state_tensor(
        self,
        reset_mask: torch.Tensor,
        layout: BackendRootStateLayout,
        qpos_indices: np.ndarray,
        qvel_indices: np.ndarray,
        root_state: torch.Tensor,
        joint_position: torch.Tensor,
        joint_velocity: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage one full-width motion root and joint payload by bool mask."""
        effective_mask = self._tensor_mask_scope(
            reset_mask, capability="write_motion_state_tensor_mask"
        )
        qpos_columns, qvel_columns = self._tensor_joint_columns_for_indices(
            qpos_indices, qvel_indices, term_name=term_name
        )
        width = qpos_columns.numel()
        expected = (self._num_envs, width)
        if root_state.shape != (self._num_envs, 13):
            raise ValueError(
                f"EventManager term '{term_name}' tensor mask motion root state must "
                f"have shape {(self._num_envs, 13)}; got {tuple(root_state.shape)}"
            )
        if joint_position.shape != expected:
            raise ValueError(
                f"EventManager term '{term_name}' tensor mask motion joint position must "
                f"have shape {expected}; got {tuple(joint_position.shape)}"
            )
        if joint_velocity.shape != joint_position.shape:
            raise ValueError(
                f"EventManager term '{term_name}' tensor mask motion joint velocity must "
                f"have shape {tuple(joint_position.shape)}"
            )
        for name, values in (
            ("root state", root_state),
            ("joint position", joint_position),
            ("joint velocity", joint_velocity),
        ):
            if values.dtype != torch.float32 or not values.is_contiguous():
                raise TypeError(
                    f"EventManager term '{term_name}' tensor mask motion {name} must "
                    "be contiguous float32"
                )
        self._validate_motion_tensor_payload(
            root_state=root_state,
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            term_name=term_name,
        )
        root_qpos_columns, root_qvel_columns = self._tensor_root_columns_for_layout(
            layout, term_name=term_name
        )
        transaction = self._require_packed_transaction()
        if self._tensor_owner_subset:
            # An owner-subset motion write publishes only its requested rows.
            # With no requested rows it remains a full-width dispatch side
            # effect (RNG/packet/carrier work) and must not claim a backend
            # state commit.
            self._mask_count.copy_(effective_mask.sum())
            return
        transaction.write_state_mask(
            effective_mask,
            qpos_columns=root_qpos_columns,
            qpos_values=root_state[:, :7].contiguous(),
            qvel_columns=root_qvel_columns,
            qvel_values=root_state[:, 7:13].contiguous(),
        )
        transaction.write_state_mask(
            effective_mask,
            qpos_columns=qpos_columns,
            qpos_values=joint_position,
            qvel_columns=qvel_columns,
            qvel_values=joint_velocity,
        )
        self._tensor_has_writes = True

    def write_root_pose(
        self,
        env_ids: torch.Tensor,
        layout: BackendRootStateLayout,
        pose: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage world position and wxyz orientation for one floating root."""
        ids = self._prepare_state_write(env_ids, capability="root-pose", term_name=term_name)
        positions = self._validate_values(
            pose,
            shape=(ids.size, 7),
            capability="root pose",
            term_name=term_name,
        )
        self._validate_quaternions(positions[:, 3:7], term_name=term_name)
        qpos_columns, _ = self._validate_root_layout(layout, term_name=term_name)
        self._require_packed_transaction().write_state(
            self._host_rows(ids),
            qpos_columns=qpos_columns,
            qpos_values=self._packed_values(positions),
        )
        self._host_has_writes = True

    def write_root_velocity(
        self,
        env_ids: torch.Tensor,
        layout: BackendRootStateLayout,
        velocity_w: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage world-frame root velocity in generalized qvel columns.

        The public generalized-state contract stores free-root linear velocity
        in world coordinates and angular velocity in root-body coordinates.
        The conversion uses the pose already staged in this transaction.
        """
        ids = self._prepare_state_write(env_ids, capability="root-velocity", term_name=term_name)
        velocities = self._validate_values(
            velocity_w,
            shape=(ids.size, 6),
            capability="root velocity",
            term_name=term_name,
        )
        qpos_columns, qvel_columns = self._validate_root_layout(layout, term_name=term_name)
        if ids.size:
            transaction = self._require_packed_transaction()
            device = self._packed_reset_device
            assert device is not None
            rows = self._host_rows(ids)
            qpos, _ = transaction.state_views()
            quaternion_columns = torch.as_tensor(
                qpos_columns[3:7], dtype=torch.int64, device=device
            )
            quaternions = qpos.index_select(0, rows).index_select(1, quaternion_columns)
            norms = torch.linalg.vector_norm(quaternions, dim=-1)
            invalid = ~torch.isclose(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-6)
            if bool(invalid.any()):
                raise ValueError(
                    f"EventManager term '{term_name}' root quaternion must be unit length; "
                    f"norms={norms[invalid].detach().cpu().tolist()}"
                )
            velocity_tensor = torch.as_tensor(
                np.ascontiguousarray(velocities, dtype=np.float32), device=device
            )
            encoded = velocity_tensor.clone()
            encoded[:, 3:6] = quat_apply_inverse(quaternions, velocity_tensor[:, 3:6])
            transaction.write_state(rows, qvel_columns=qvel_columns, qvel_values=encoded)
        self._host_has_writes = True

    def read_root_pose(
        self,
        env_ids: torch.Tensor,
        layout: BackendRootStateLayout,
        *,
        term_name: str,
    ) -> np.ndarray:
        """Read the world position and wxyz orientation staged for one floating root.

        Returns a detached ``(len(env_ids), 7)`` copy of the pose currently
        staged in this transaction. Rows no term has written yet hold the
        backend default pose, so a later reset term can build on an earlier
        term's root placement without re-deriving it.
        """
        ids = self._prepare_state_write(env_ids, capability="root-pose", term_name=term_name)
        qpos_columns, _ = self._validate_root_layout(layout, term_name=term_name)
        transaction = self._require_packed_transaction()
        device = self._packed_reset_device
        assert device is not None
        rows = self._host_rows(ids)
        qpos, _ = transaction.state_views()
        columns = torch.as_tensor(qpos_columns, dtype=torch.int64, device=device)
        staged = qpos.index_select(0, rows).index_select(1, columns)
        return np.array(staged.detach().cpu().numpy(), copy=True)

    def bind_mocap_pose(self, body_name: str) -> BackendMocapPoseBinding:
        """Resolve a mocap body once, without exposing a native backend handle."""
        if body_name not in self._mocap_bindings:
            if self._active:
                raise RuntimeError("Mocap bodies must be bound before reset starts")
            binding = self._backend.bind_mocap_pose(body_name)
            if binding.num_envs != self._num_envs or binding.body_name != body_name:
                raise ValueError("Backend mocap binding does not match the requested body/batch")
            self._mocap_bindings[body_name] = binding
        return self._mocap_bindings[body_name]

    def read_mocap_pose(self, body_name: str) -> np.ndarray:
        """Read current poses with this transaction's pending rows overlaid."""
        poses = self._mocap_bindings[body_name].read()
        pending = self._mocap_pending.get(body_name)
        if pending is not None:
            rows, values = pending
            poses[rows.detach().cpu().numpy()] = values.detach().cpu().numpy()
        return poses

    def write_mocap_pose(
        self, body_name: str, env_ids: torch.Tensor, poses: np.ndarray, *, term_name: str
    ) -> None:
        """Stage poses; upload only after the ordinary reset state has committed."""
        self._require_active()
        if body_name not in self._mocap_bindings:
            raise RuntimeError(f"Mocap body {body_name!r} must be bound before writing")
        ids = self._validate_ids(env_ids, capability="mocap pose")
        if self._tensor_active:
            active_mask = self._tensor_active_mask
            assert active_mask is not None
            requested_mask = torch.zeros_like(active_mask)
            requested_mask[ids] = True
            outside = self._tensor_mask_has_outside_rows(requested_mask, active_mask)
        else:
            begin_ids = self._begin_ids
            assert begin_ids is not None
            outside = not np.isin(ids, begin_ids).all()
        if outside:
            raise ValueError(f"{term_name}: mocap mutation outside the active reset")
        values = self._validate_values(
            poses, shape=(ids.size, 7), capability="mocap pose", term_name=term_name
        )
        self._validate_quaternions(values[:, 3:], term_name=term_name)
        capabilities = self._backend.get_tensor_capabilities()
        if not getattr(capabilities, "packed_reset_mocap", False):
            raise NotImplementedError(
                f"EventManager term '{term_name}' mocap mutation requires a backend "
                "declaring 'packed_reset_mocap'"
            )
        transaction = self._require_packed_transaction()
        device = self._packed_reset_device
        assert device is not None
        rows_tensor = torch.as_tensor(ids, dtype=torch.int64, device=device)
        poses_tensor = torch.as_tensor(
            np.ascontiguousarray(values, dtype=np.float32), device=device
        )
        transaction.write_mocap_pose(body_name, rows_tensor, poses_tensor)
        previous = self._mocap_pending.get(body_name)
        if previous is None:
            self._mocap_pending[body_name] = (rows_tensor, poses_tensor)
        else:
            previous_rows, previous_poses = previous
            keep = ~torch.isin(previous_rows, rows_tensor)
            self._mocap_pending[body_name] = (
                torch.cat((previous_rows[keep], rows_tensor)),
                torch.cat((previous_poses[keep], poses_tensor)),
            )
        self._requesting_terms.add(term_name)

    def commit(self) -> dict | None:
        """Commit all staged rows through the packed composer boundary."""
        self._require_active()
        if self.scene_layout is not None:
            try:
                self._commit_entities()
                if self._host_has_writes or self._mocap_pending:
                    raise NotImplementedError(
                        "mapped scene DR writes need a public entity transaction"
                    )
                return None
            finally:
                self._finish()
        has_writes = self._host_has_writes
        mocap_staged = bool(self._mocap_pending)
        self._last_commit_had_writes = has_writes or mocap_staged
        try:
            transaction = self._require_packed_transaction()
            rows = self._begin_rows
            assert rows is not None
            try:
                if has_writes and rows.numel():
                    started = time.perf_counter()
                    result = transaction.commit(rows)
                    self._last_set_state_timing_ms = {
                        "dr_reset_set_state_ms": (time.perf_counter() - started) * 1000.0
                    }
                    self._merge_backend_timing(result)
                    return cast(dict | None, result)
                if not mocap_staged:
                    return None
                started = time.perf_counter()
                result = transaction.commit_mocap_poses()
                self._last_set_state_timing_ms = {
                    "dr_reset_set_state_ms": (time.perf_counter() - started) * 1000.0
                }
                self._merge_backend_timing(result)
                return None
            except (
                AttributeError,
                NotImplementedError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                terms = ", ".join(sorted(self._requesting_terms))
                raise NotImplementedError(
                    "EventManager reset-state capability "
                    "'SimBackend.begin_packed_reset' is unavailable for term(s) "
                    f"[{terms}] on backend '{self._backend.backend_type}': {exc}"
                ) from exc
        finally:
            self._finish()

    def commit_device_tensor(self) -> dict | None:
        """Commit the fixed-width reset through the packed composer boundary."""
        self._require_active()
        mocap_staged = bool(self._mocap_pending)
        has_writes = self._tensor_has_writes or self._host_has_writes
        self._last_commit_had_writes = has_writes or mocap_staged
        if self._tensor_active_mask is None:
            raise RuntimeError("fixed-width tensor reset is not active")
        try:
            transaction = self._require_packed_transaction()
            if not has_writes:
                if not mocap_staged:
                    return None
                started = time.perf_counter()
                result = transaction.commit_mocap_poses()
                self._last_set_state_timing_ms = {
                    "dr_reset_set_state_ms": (time.perf_counter() - started) * 1000.0
                }
                self._merge_backend_timing(result)
                return None
            if self.scene_layout is not None:
                raise NotImplementedError(
                    "fixed-width tensor reset supports scalar qpos/qvel rows only; mapped "
                    "entity writes remain an explicit migration boundary"
                )
            if self._tensor_owner_subset:
                write_mask = self._tensor_write_mask
                assert write_mask is not None
                commit_mask = write_mask
            else:
                commit_mask = self._tensor_active_mask
                assert commit_mask is not None
            if not self._commit_has_rows(commit_mask):
                return None
            try:
                started = time.perf_counter()
                result = transaction.commit_mask(commit_mask)
                self._last_set_state_timing_ms = {
                    "dr_reset_set_state_ms": (time.perf_counter() - started) * 1000.0
                }
                self._merge_backend_timing(result)
                return cast(dict | None, result)
            except (
                AttributeError,
                NotImplementedError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                terms = ", ".join(sorted(self._requesting_terms))
                raise NotImplementedError(
                    "EventManager reset-state capability "
                    "'SimBackend.begin_packed_reset' is unavailable for term(s) "
                    f"[{terms}] on backend '{self._backend.backend_type}': {exc}"
                ) from exc
        finally:
            self._finish()

    def abort(self) -> None:
        """Discard staged rows without touching the backend."""
        if self._active or self._tensor_active:
            self._finish()

    def _merge_backend_timing(self, result: Any) -> None:
        """Merge backend-reported commit sub-timings into the last timing table."""
        if isinstance(result, dict):
            backend_timing = result.get("timing")
            if isinstance(backend_timing, dict):
                self._last_set_state_timing_ms.update(backend_timing)

    @property
    def last_set_state_completion(self) -> Any | None:
        """Return the public receipt from the most recent backend reset commit."""
        return self._last_set_state_timing_ms.get("graph_completion")

    def _require_packed_transaction(self) -> TorchPackedResetComposer:
        """Return the declared packed composer or fail closed."""
        transaction = self._packed_reset_transaction
        if transaction is None:
            raise NotImplementedError(
                "backend does not declare the packed_reset capability required by the "
                "packed reset contract: SimBackend.begin_packed_reset"
            )
        return transaction

    def _ensure_packed_state_default(self, term_name: str) -> None:
        """Install the immutable one-row default state on the packed composer once."""
        if self._packed_state_default_installed:
            return
        transaction = self._packed_reset_transaction
        device = self._packed_reset_device
        if transaction is None or device is None:
            raise RuntimeError(
                "packed reset transaction requires declare_packed_reset_device before begin"
            )
        self._materialize_default_state(term_name)
        assert self._default_qpos is not None
        assert self._default_qvel is not None
        public_widths = self._backend.get_public_state_widths()
        if self._default_qpos.size != int(public_widths.nq):
            self._default_qpos = self._validate_state_vector(
                self._backend.get_default_qpos(), "default qpos", term_name
            )
        if self._default_qvel.size != int(public_widths.nv):
            self._default_qvel = self._validate_state_vector(
                self._backend.get_init_qvel(), "initial qvel", term_name
            )
        transaction.set_state_default(
            torch.as_tensor(
                np.array(self._default_qpos, dtype=np.float32, copy=True), device=device
            ),
            torch.as_tensor(
                np.array(self._default_qvel, dtype=np.float32, copy=True), device=device
            ),
        )
        self._packed_state_default_installed = True

    def _host_rows(self, ids: np.ndarray) -> torch.Tensor:
        """Convert validated host env IDs to packed-composer device rows."""
        device = self._packed_reset_device
        assert device is not None
        return torch.as_tensor(ids, dtype=torch.int64, device=device)

    def _packed_values(self, values: np.ndarray) -> torch.Tensor:
        """Move validated host values into the packed composer as float32."""
        device = self._packed_reset_device
        assert device is not None
        return torch.as_tensor(np.ascontiguousarray(values, dtype=np.float32), device=device)

    def _materialize_default_state(self, term_name: str) -> None:
        if self._default_qpos is not None:
            return
        qpos = self._selected_default_qpos
        if qpos is None:
            try:
                qpos = self._backend.get_default_qpos()
            except (AttributeError, NotImplementedError) as exc:
                raise self._capability_error(term_name, "default qpos", exc) from exc
        try:
            qvel = self._backend.get_init_qvel()
        except (AttributeError, NotImplementedError) as exc:
            raise self._capability_error(term_name, "initial qvel", exc) from exc

        default_qpos = self._validate_state_vector(qpos, "default qpos", term_name)
        default_qvel = self._validate_state_vector(qvel, "initial qvel", term_name)
        self._default_qpos = default_qpos
        self._default_qvel = default_qvel

    def _validate_tensor_root_payload(self, root_state: torch.Tensor, *, term_name: str) -> None:
        """Validate finite values and quaternion norms through one device comparison."""
        quat_norm = torch.linalg.vector_norm(root_state[:, 3:7], dim=-1)
        valid_quat = torch.isclose(
            quat_norm,
            torch.ones_like(quat_norm),
            rtol=1e-5,
            atol=1e-6,
        )
        valid = torch.isfinite(root_state).all() & valid_quat.all()
        if self._tensor_root_payload_valid is None or (
            self._tensor_root_payload_valid.device != valid.device
        ):
            self._tensor_root_payload_valid = torch.empty_like(valid)
        self._tensor_root_payload_valid.copy_(valid)
        if self._deferred_payload_validation:
            return
        if torch.equal(valid, torch.ones_like(valid)):
            return
        if not bool(torch.isfinite(root_state).all()):
            raise ValueError(
                f"EventManager term '{term_name}' tensor root state contains NaN or Inf"
            )
        self._validate_tensor_root_quaternions(root_state[:, 3:7], term_name=term_name)

    def _validate_tensor_root_quaternions(self, values: torch.Tensor, *, term_name: str) -> None:
        norms = torch.linalg.vector_norm(values, dim=-1)
        if not bool(torch.isclose(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-6).all()):
            raise ValueError(
                f"EventManager term '{term_name}' tensor root quaternion must be unit length"
            )

    def _materialize_default_actuator_gains(self, term_name: str) -> None:
        if self._default_kp is not None:
            return
        try:
            capabilities = self._backend.get_dr_capabilities()
        except (AttributeError, NotImplementedError) as exc:
            raise self._capability_error(term_name, "actuator gain randomization", exc) from exc
        required = frozenset((RESET_TERM_KP, RESET_TERM_KD))
        unsupported = capabilities.get_unsupported_reset_terms(required)
        if unsupported:
            detail = ", ".join(sorted(unsupported))
            raise self._capability_error(
                term_name,
                "actuator gain randomization",
                NotImplementedError(f"unsupported reset payload fields: {detail}"),
            )
        default_kp = self._fetch_reset_term_default(
            RESET_TERM_KP,
            expected_tail=(),
            term_name=term_name,
        )
        default_kd = self._fetch_reset_term_default(
            RESET_TERM_KD,
            expected_tail=(),
            term_name=term_name,
        )
        for name, default in ((RESET_TERM_KP, default_kp), (RESET_TERM_KD, default_kd)):
            width = self._randomization_default_width(default, field=name)
            if width != self._backend.num_actuators:
                raise ValueError(
                    f"EventManager term '{term_name}' default actuator {name} on backend "
                    f"'{self._backend.backend_type}' has model width {width}; expected "
                    f"{self._backend.num_actuators}"
                )
        self._default_kp = default_kp
        self._default_kd = default_kd

    def _materialize_randomization_default(
        self,
        field: str,
        *,
        expected_tail: tuple[int, ...],
        term_name: str,
    ) -> np.ndarray:
        cached = self._randomization_defaults.get(field)
        if cached is not None:
            return cached
        try:
            capabilities = self._backend.get_dr_capabilities()
        except (AttributeError, NotImplementedError) as exc:
            raise self._capability_error(term_name, f"{field} randomization", exc) from exc
        unsupported = capabilities.get_unsupported_reset_terms(frozenset((field,)))
        if unsupported:
            raise self._capability_error(
                term_name,
                f"{field} randomization",
                NotImplementedError(f"unsupported reset payload field: {field}"),
            )
        default = self._fetch_reset_term_default(
            field,
            expected_tail=expected_tail,
            term_name=term_name,
        )
        self._randomization_defaults[field] = default
        return default

    def _fetch_reset_term_default(
        self,
        field: str,
        *,
        expected_tail: tuple[int, ...],
        term_name: str,
    ) -> np.ndarray:
        try:
            value = self._backend.get_reset_term_default(field)
        except (AttributeError, NotImplementedError) as exc:
            raise self._capability_error(term_name, f"default {field}", exc) from exc
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"EventManager term '{term_name}' capability 'default {field}' on backend "
                f"'{self._backend.backend_type}' must return np.ndarray, got "
                f"{type(value).__name__}"
            )
        canonical_ndim = 1 + len(expected_tail)
        canonical_shape = value.ndim == canonical_ndim
        per_env_shape = value.ndim == canonical_ndim + 1 and value.shape[0] == self._num_envs
        if field == RESET_TERM_GRAVITY:
            canonical_shape = value.shape == (3,)
            per_env_shape = value.shape == (self._num_envs, 3)
        if not canonical_shape and not per_env_shape:
            raise ValueError(
                f"EventManager term '{term_name}' capability 'default {field}' on backend "
                f"'{self._backend.backend_type}' returned shape {value.shape}; expected "
                f"a canonical {canonical_ndim}-D table or a per-environment "
                f"({self._num_envs}, *canonical) table"
            )
        tail_slice = 2 if per_env_shape else 1
        if value.shape[tail_slice:] != expected_tail:
            raise ValueError(
                f"EventManager term '{term_name}' capability 'default {field}' on backend "
                f"'{self._backend.backend_type}' returned shape {value.shape}; expected tail "
                f"{expected_tail}"
            )
        if not np.issubdtype(value.dtype, np.floating):
            raise TypeError(
                f"EventManager term '{term_name}' capability 'default {field}' on backend "
                f"'{self._backend.backend_type}' must be floating, got {value.dtype}"
            )
        if not np.isfinite(value).all():
            raise ValueError(
                f"EventManager term '{term_name}' capability 'default {field}' on backend "
                f"'{self._backend.backend_type}' returned NaN or Inf"
            )
        default = np.array(value, copy=True)
        default.setflags(write=False)
        return default

    def _require_randomization_default(self, field: str, term_name: str) -> np.ndarray:
        try:
            return self._randomization_defaults[field]
        except KeyError as exc:
            raise RuntimeError(
                f"EventManager term '{term_name}' must bind reset field '{field}' "
                "during manager construction before writing it"
            ) from exc

    def _default_is_per_env(
        self,
        default: np.ndarray,
        *,
        field: str,
    ) -> bool:
        canonical_ndim = 1 + len(_randomization_term_tail(field))
        if default.ndim == canonical_ndim:
            return False
        if default.ndim == canonical_ndim + 1 and default.shape[0] == self._num_envs:
            return True
        raise RuntimeError(
            f"Reset transaction cached an invalid '{field}' default table with shape "
            f"{default.shape}"
        )

    def _randomization_default_width(
        self,
        default: np.ndarray,
        *,
        field: str,
    ) -> int:
        axis = 1 if self._default_is_per_env(default, field=field) else 0
        return int(default.shape[axis])

    def _select_randomization_default_columns(
        self,
        default: np.ndarray,
        columns: np.ndarray,
        *,
        field: str,
    ) -> np.ndarray:
        if self._default_is_per_env(default, field=field):
            return default[:, columns]
        return default[columns]

    def record_startup_randomization(
        self,
        field: str,
        env_ids: torch.Tensor,
        column_ids: np.ndarray,
        values: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Re-stage full-width startup rows so the commit promotes the baseline.

        The backend remains authoritative for committed state. The packed
        composer promotes every staged field's committed rows into its reset
        baseline when the public backend commit succeeds, so a later mask reset
        rebuilds unwritten model columns from the startup values instead of the
        immutable construction defaults. The promotion is deferred until the
        commit succeeds, preserving the reset transaction's all-or-nothing
        contract.
        """
        if field not in self._randomization_defaults:
            # Validation and immutable-default binding are owned by the
            # corresponding write performed by the startup event.
            raise RuntimeError(
                f"EventManager term '{term_name}' must bind reset field '{field}' "
                "before recording startup randomization"
            )
        ids = self._validate_ids(env_ids, capability="startup randomization")
        if ids.size != self._num_envs or not np.array_equal(ids, np.arange(self._num_envs)):
            raise ValueError(
                f"EventManager term '{term_name}' startup randomization field '{field}' "
                f"requires all {self._num_envs} environment rows; got {ids.tolist()}"
            )
        tail = _randomization_term_tail(field)
        self._write_selected_randomization(
            field,
            env_ids,
            column_ids,
            values,
            value_tail=tail,
            term_name=term_name,
        )
        self._startup_randomization_fields.add(field)

    def _readonly_binding(
        self,
        columns: np.ndarray,
        selected_default: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _readonly_array(columns), _readonly_array(selected_default)

    def _write_selected_randomization(
        self,
        field: str,
        env_ids: torch.Tensor,
        column_ids: np.ndarray,
        values: np.ndarray | torch.Tensor,
        *,
        value_tail: tuple[int, ...],
        term_name: str,
    ) -> None:
        ids = self._prepare_state_write(
            env_ids,
            capability=field,
            term_name=term_name,
        )
        default = self._require_randomization_default(field, term_name)
        columns = self._validate_columns(
            column_ids,
            width=self._randomization_default_width(default, field=field),
            capability=f"{field} column IDs",
            term_name=term_name,
        )
        selected = self._validate_values(
            values,
            shape=(ids.size, columns.size, *value_tail),
            capability=field,
            term_name=term_name,
        )
        self._require_packed_transaction().write_model_field(
            field,
            self._host_rows(ids),
            self._packed_values(selected),
            columns=columns,
        )
        self._host_has_writes = True

    def _prepare_state_write(
        self,
        env_ids: torch.Tensor,
        *,
        capability: str,
        term_name: str,
    ) -> np.ndarray:
        self._require_active()
        if self._tensor_active:
            return self._prepare_tensor_randomization_write(
                env_ids, capability=capability, term_name=term_name
            )
        ids = self._validate_ids(env_ids, capability=f"write_{capability}")
        begin_ids = self._begin_ids
        assert begin_ids is not None
        outside = ids[~np.isin(ids, begin_ids)]
        if outside.size:
            raise ValueError(
                f"EventManager term '{term_name}' attempted {capability} mutation outside "
                f"the active reset: {outside.tolist()}"
            )
        self._requesting_terms.add(term_name)
        self._ensure_packed_state_default(term_name)
        return ids

    def _prepare_tensor_randomization_write(
        self,
        env_ids: torch.Tensor,
        *,
        capability: str,
        term_name: str,
    ) -> np.ndarray:
        """Validate a host DR write against the active packed reset rows."""
        if self._tensor_owner_subset:
            raise RuntimeError(
                f"EventManager term '{term_name}' reset randomization is not part of "
                "an owner-subset tensor transaction"
            )
        ids = self._validate_ids(env_ids, capability=f"write_{capability}")
        active_mask = self._tensor_active_mask
        assert active_mask is not None
        requested_mask = torch.zeros_like(active_mask)
        requested_mask[ids] = True
        if self._tensor_mask_has_outside_rows(requested_mask, active_mask):
            raise ValueError(
                f"EventManager term '{term_name}' attempted {capability} mutation "
                "outside the active tensor reset"
            )
        self._requesting_terms.add(term_name)
        return ids

    def _validate_root_layout(
        self,
        layout: BackendRootStateLayout,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(layout, BackendRootStateLayout):
            raise TypeError(
                f"EventManager term '{term_name}' root-state layout must be "
                f"BackendRootStateLayout, got {type(layout).__name__}"
            )
        assert self._default_qpos is not None
        assert self._default_qvel is not None
        qpos_columns = self._validate_columns(
            np.asarray(layout.qpos_indices, dtype=np.intp),
            width=self._default_qpos.size,
            capability="root qpos indices",
            term_name=term_name,
        )
        qvel_columns = self._validate_columns(
            np.asarray(layout.qvel_indices, dtype=np.intp),
            width=self._default_qvel.size,
            capability="root qvel indices",
            term_name=term_name,
        )
        return qpos_columns, qvel_columns

    def _tensor_root_columns_for_layout(
        self, layout: BackendRootStateLayout, *, term_name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qpos_columns, qvel_columns = self._validate_root_layout(layout, term_name=term_name)
        key = (tuple(int(i) for i in qpos_columns), tuple(int(i) for i in qvel_columns))
        cached = self._tensor_root_columns
        device = self._packed_reset_device
        assert device is not None
        if cached is not None and (cached[0], cached[1]) == key:
            return cached[2], cached[3]
        qpos_tensor = torch.as_tensor(qpos_columns, dtype=torch.int64, device=device)
        qvel_tensor = torch.as_tensor(qvel_columns, dtype=torch.int64, device=device)
        self._tensor_root_columns = (*key, qpos_tensor, qvel_tensor)
        return qpos_tensor, qvel_tensor

    def _tensor_joint_columns_for_indices(
        self,
        qpos_indices: np.ndarray,
        qvel_indices: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._default_qpos is not None and self._default_qvel is not None
        qpos_columns = self._validate_columns(
            np.asarray(qpos_indices, dtype=np.intp),
            width=self._default_qpos.size,
            capability="tensor joint qpos indices",
            term_name=term_name,
        )
        qvel_columns = self._validate_columns(
            np.asarray(qvel_indices, dtype=np.intp),
            width=self._default_qvel.size,
            capability="tensor joint qvel indices",
            term_name=term_name,
        )
        key = (
            tuple(int(index) for index in qpos_columns),
            tuple(int(index) for index in qvel_columns),
        )
        cached = self._tensor_joint_columns
        device = self._packed_reset_device
        assert device is not None
        if cached is not None and (cached[0], cached[1]) == key:
            return cached[2], cached[3]
        qpos_tensor = torch.as_tensor(qpos_columns, dtype=torch.int64, device=device)
        qvel_tensor = torch.as_tensor(qvel_columns, dtype=torch.int64, device=device)
        self._tensor_joint_columns = (*key, qpos_tensor, qvel_tensor)
        return qpos_tensor, qvel_tensor

    def _validate_quaternions(self, values: np.ndarray, *, term_name: str) -> None:
        norms = np.linalg.norm(values, axis=1)
        invalid = ~np.isclose(norms, 1.0, rtol=1e-5, atol=1e-6)
        if np.any(invalid):
            raise ValueError(
                f"EventManager term '{term_name}' root quaternion must be unit length; "
                f"norms={norms[invalid].tolist()}"
            )

    def _validate_state_vector(
        self,
        value: np.ndarray,
        capability: str,
        term_name: str,
    ) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"EventManager term '{term_name}' capability '{capability}' on backend "
                f"'{self._backend.backend_type}' must return np.ndarray, got "
                f"{type(value).__name__}"
            )
        if value.ndim != 1:
            raise ValueError(
                f"EventManager term '{term_name}' capability '{capability}' on backend "
                f"'{self._backend.backend_type}' returned shape {value.shape}; expected 1-D"
            )
        if not np.issubdtype(value.dtype, np.floating):
            raise TypeError(
                f"EventManager term '{term_name}' capability '{capability}' on backend "
                f"'{self._backend.backend_type}' must be floating, got {value.dtype}"
            )
        if not np.isfinite(value).all():
            raise ValueError(
                f"EventManager term '{term_name}' capability '{capability}' on backend "
                f"'{self._backend.backend_type}' returned NaN or Inf"
            )
        result = np.array(value, copy=True)
        result.setflags(write=False)
        return result

    def _validate_ids(self, env_ids: torch.Tensor, *, capability: str) -> np.ndarray:
        if not isinstance(env_ids, torch.Tensor):
            raise TypeError(
                f"ManagerBased reset-state {capability} env_ids must be torch.Tensor, "
                f"got {type(env_ids).__name__}"
            )
        if env_ids.ndim != 1 or env_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError(
                f"ManagerBased reset-state {capability} env_ids must be a 1-D integer "
                f"tensor, got shape={tuple(env_ids.shape)}, dtype={env_ids.dtype}"
            )
        rows = env_ids.to(dtype=torch.int64)
        if rows.numel() and not bool(((rows >= 0) & (rows < self._num_envs)).all()):
            raise IndexError(
                f"ManagerBased reset-state {capability} env_ids out of range for "
                f"{self._num_envs} environments: {rows.tolist()}"
            )
        ids = rows.detach().cpu().numpy().astype(np.int32, copy=False)
        # Duplicate check via bincount instead of np.unique: identical semantics
        # (ids are already range-checked above) but avoids the sort — ~30x faster
        # at num_envs=4096 and ~4x at typical partial-reset widths. This runs on
        # every reset-state write (~6x per env step), so the sort cost was
        # measurable in the collector host phase (issue #1352).
        if ids.size > 1 and np.bincount(ids, minlength=self._num_envs).max() > 1:
            raise ValueError(
                f"ManagerBased reset-state {capability} env_ids contain duplicates: {ids.tolist()}"
            )
        return ids

    def _validate_columns(
        self,
        values: np.ndarray,
        *,
        width: int,
        capability: str,
        term_name: str,
    ) -> np.ndarray:
        if not isinstance(values, np.ndarray):
            raise TypeError(
                f"EventManager term '{term_name}' {capability} must be np.ndarray, "
                f"got {type(values).__name__}"
            )
        if (
            values.ndim != 1
            or not np.issubdtype(values.dtype, np.integer)
            or np.issubdtype(values.dtype, np.bool_)
        ):
            raise TypeError(
                f"EventManager term '{term_name}' {capability} must be a 1-D integer array"
            )
        columns = np.asarray(values, dtype=np.intp)
        if np.any(columns < 0) or np.any(columns >= width):
            raise IndexError(
                f"EventManager term '{term_name}' {capability} out of range for width "
                f"{width}: {columns.tolist()}"
            )
        if np.unique(columns).size != columns.size:
            raise ValueError(
                f"EventManager term '{term_name}' {capability} contain duplicates: "
                f"{columns.tolist()}"
            )
        return columns

    def _validate_tensor_values(
        self,
        values: torch.Tensor,
        *,
        shape: tuple[int, ...],
        capability: str,
        term_name: str,
    ) -> torch.Tensor:
        if not isinstance(values, torch.Tensor):
            raise TypeError(
                f"EventManager term '{term_name}' {capability} must be torch.Tensor, "
                f"got {type(values).__name__}"
            )
        if tuple(values.shape) != shape:
            raise ValueError(
                f"EventManager term '{term_name}' {capability} has shape "
                f"{tuple(values.shape)}; expected {shape}"
            )
        if not values.is_floating_point():
            raise TypeError(f"EventManager term '{term_name}' {capability} must be floating")
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"EventManager term '{term_name}' {capability} contains NaN or Inf")
        return values.contiguous()

    def _validate_values(
        self,
        values: np.ndarray | torch.Tensor,
        *,
        shape: tuple[int, ...],
        capability: str,
        term_name: str,
    ) -> np.ndarray:
        if isinstance(values, torch.Tensor):
            if values.device.type != "cpu":
                raise TypeError(
                    f"EventManager term '{term_name}' {capability} host write requires a "
                    f"CPU Torch tensor; got {values.device}"
                )
            if not values.is_contiguous():
                values = values.contiguous()
            values = values.numpy()
        if not isinstance(values, np.ndarray):
            raise TypeError(
                f"EventManager term '{term_name}' {capability} must be np.ndarray or a CPU "
                f"torch.Tensor, got {type(values).__name__}"
            )
        if values.shape != shape:
            raise ValueError(
                f"EventManager term '{term_name}' {capability} has shape {values.shape}; "
                f"expected {shape}"
            )
        if not np.issubdtype(values.dtype, np.floating):
            raise TypeError(
                f"EventManager term '{term_name}' {capability} must be floating, got {values.dtype}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"EventManager term '{term_name}' {capability} contains NaN or Inf")
        return values

    def _capability_error(
        self,
        term_name: str,
        capability: str,
        exc: BaseException,
    ) -> NotImplementedError:
        return NotImplementedError(
            f"EventManager term '{term_name}' reset-state capability '{capability}' is "
            f"unavailable on backend '{self._backend.backend_type}': {exc}"
        )

    def _require_active(self) -> None:
        if not (self._active or self._tensor_active):
            raise RuntimeError("ManagerBased reset-state mutation requires an active reset event")

    def _finish(self) -> None:
        if self._packed_reset_transaction is not None:
            self._packed_reset_transaction.discard()
        self._active = False
        self._begin_rows = None
        self._begin_ids = None
        self._host_has_writes = False
        self._startup_randomization_fields.clear()
        self._mocap_pending.clear()
        self._requesting_terms.clear()
        self._tensor_active = False
        self._tensor_active_mask = None
        self._mask_count.zero_()
        self._mask_outside_rows.zero_()
        self._tensor_has_writes = False
        self._completed_tensor_has_writes = self._last_commit_had_writes
        self._tensor_owner_subset = False
        self._tensor_write_mask = None
        self._commit_verdict_published = False


__all__ = ["ResetStateTransaction"]
