"""MuJoCo-owned host reference implementation of the typed batch contract."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from unilab.base.backend.batch import (
    BACKEND_BATCH_CONTRACT_VERSION,
    BackendBatchContractError,
    BackendBatchCounters,
    BackendBatchDiagnostics,
    BackendIORequirements,
    BackendReadResult,
    BackendTiming,
    BoundBackendPlan,
    BoundStatePlan,
    BufferLayout,
    BufferView,
    ExecutionProfile,
    PhysicalUnit,
    ReferenceFrame,
    RowSelection,
    StateBatch,
    StateBatchLease,
    StateBatchPhase,
    StateEntityKind,
    StateFieldKind,
    StateFieldSpec,
)
from unilab.base.backend.mutation import (
    BoundMutationPlan,
    BoundMutationSpec,
    MutationContractError,
    MutationEntityKind,
    MutationFieldKind,
    MutationTargetKind,
)
from unilab.base.backend.mutation_batch import (
    BoundMutationValueBufferGroup,
    BoundMutationValueBuffers,
    BoundMutationValueWindow,
    TypedBackendMutationBatch,
)

if TYPE_CHECKING:
    from .backend import MuJoCoBackend

_STATE_FINGERPRINT_PREFIX = "mujoco-host-state-v1"
_PLAN_FINGERPRINT_PREFIX = "mujoco-host-batch-v1"
_ROOT_RESET_VALUE_SHAPES = {
    "state.root.position": (1, 3),
    "state.root.orientation": (1, 4),
    "state.root.linear_velocity": (1, 3),
    "state.root.angular_velocity": (1, 3),
}
_DOF_RESET_TARGETS = frozenset({"state.dof.position", "state.dof.angular_velocity"})


@dataclass(frozen=True)
class _PreparedResetSlot:
    """Cold-bound translation from a semantic reset field to state columns."""

    field_index: int
    state_offset: int
    width: int
    dof_columns: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class _PreparedResetBufferGroup:
    """Backend-compiled vectorized assignment for homogeneous reset fields."""

    destination_columns: np.ndarray = field(repr=False, compare=False)
    row_values: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True)
class _PreparedResetBufferSet:
    """Strongly-owned cache entry for one manager mutation buffer set."""

    owner: BoundMutationValueBuffers = field(repr=False, compare=False)
    individual: tuple[tuple[_PreparedResetSlot, np.ndarray], ...] = field(
        repr=False,
        compare=False,
    )
    groups: tuple[_PreparedResetBufferGroup, ...] = field(repr=False, compare=False)


def _step_allocations(dtype: str) -> int:
    """Count pool outputs plus the remaining float64 physics-state coercion.

    ``BatchEnvPool`` consumes float64 trajectories.  The typed host plan owns
    a cold-allocated float64 control trajectory, so control is no longer
    cast/allocated by the pool on every step.  The backend state cache still
    uses the public plan dtype and therefore needs one conversion when that
    dtype is float32.
    """

    input_conversions = 0 if dtype == "float64" else 1
    return 2 + input_conversions


def _readonly_view(array: np.ndarray) -> np.ndarray:
    view = array.view()
    view.flags.writeable = False
    return view


@dataclass
class _MuJoCoStateSource:
    spec: StateFieldSpec
    source: np.ndarray
    gather_indices: np.ndarray | None = None
    gather_axis: int = 1
    _full: np.ndarray = field(init=False, repr=False)
    _copy_source: bool = field(init=False, repr=False)
    _all_view: np.ndarray = field(init=False, repr=False)
    _all_descriptor: BufferView = field(init=False, repr=False)
    _selected: np.ndarray = field(init=False, repr=False)
    _selected_descriptors: dict[int, BufferView] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        num_envs = int(self.source.shape[0])
        expected_shape = (num_envs, *self.spec.buffer.row_shape)
        if self.gather_indices is None:
            if self.source.shape != expected_shape:
                raise BackendBatchContractError(
                    f"MuJoCo source for {self.spec.key!r} has shape {self.source.shape}, "
                    f"expected {expected_shape}"
                )
            self._copy_source = not self.source.flags.c_contiguous
            self._full = (
                np.empty(expected_shape, dtype=self.spec.buffer.dtype)
                if self._copy_source
                else self.source
            )
        else:
            self._copy_source = False
            self._full = np.empty(expected_shape, dtype=self.spec.buffer.dtype)
        self._all_view = _readonly_view(self._full)
        self._all_descriptor = BufferView(
            handle=self._all_view,
            shape=expected_shape,
            contract=self.spec.buffer,
        )
        self._selected = np.empty(expected_shape, dtype=self.spec.buffer.dtype)

    def materialize(
        self,
        rows: RowSelection,
        row_ids: np.ndarray | None,
    ) -> BufferView:
        if self.gather_indices is not None:
            np.take(
                self.source,
                self.gather_indices,
                axis=self.gather_axis,
                out=self._full,
            )
        elif self._copy_source:
            np.copyto(self._full, self.source)
        if rows.is_all:
            return self._all_descriptor
        else:
            assert row_ids is not None
            np.take(self._full, row_ids, axis=0, out=self._selected[: rows.count])
            descriptor = self._selected_descriptors.get(rows.count)
            if descriptor is None:
                handle = _readonly_view(self._selected[: rows.count])
                descriptor = BufferView(
                    handle=handle,
                    shape=(rows.count, *self.spec.buffer.row_shape),
                    contract=self.spec.buffer,
                )
                self._selected_descriptors[rows.count] = descriptor
            return descriptor


@dataclass
class _MuJoCoHostBatchPlan:
    public_plan: BoundBackendPlan
    sources: tuple[_MuJoCoStateSource, ...]
    lease: StateBatchLease
    _control_trajectory: np.ndarray = field(init=False, repr=False)
    _row_ids: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        control = self.public_plan.control
        self._control_trajectory = np.empty(
            (
                self.public_plan.num_envs,
                control.physics_substeps_per_control,
                *control.buffer.row_shape,
            ),
            # mujoco_uni always consumes float64 controls.  Keep this storage
            # native from the cold path onward so ``BatchEnvPool.step`` does
            # not allocate/cast the full trajectory every control barrier.
            dtype=np.float64,
        )
        self._row_ids = np.empty(self.public_plan.num_envs, dtype=np.intp)

    def stage_control(self, control: np.ndarray) -> np.ndarray:
        np.copyto(self._control_trajectory, control[:, None, ...], casting="unsafe")
        return self._control_trajectory

    @property
    def step_allocations(self) -> int:
        """Per-step numeric storage allocated by the pool after plan binding."""
        return _step_allocations(self.public_plan.control.buffer.dtype)

    def materialize(self, rows: RowSelection, phase: StateBatchPhase) -> BackendReadResult:
        self.lease.invalidate()
        start = time.perf_counter()
        row_ids = None
        if not rows.is_all:
            assert rows.indices is not None
            self._row_ids[: rows.count] = rows.indices
            row_ids = self._row_ids[: rows.count]
        descriptors = tuple(source.materialize(rows, row_ids) for source in self.sources)
        state = StateBatch(
            plan=self.public_plan,
            rows=rows,
            phase=phase,
            descriptors=descriptors,
            lease=self.lease,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return BackendReadResult(
            state=state,
            diagnostics=BackendBatchDiagnostics(
                counters=BackendBatchCounters(
                    state_materializations=1,
                    instrumentation_complete=True,
                ),
                timings=(BackendTiming("state_materialize", elapsed_ms),),
            ),
        )


def _compile_prepared_reset_slots(
    plan: BoundMutationPlan,
    *,
    qpos_state_offset: int,
    qvel_state_offset: int,
    root_qpos_dim: int,
    root_qvel_dim: int,
) -> tuple[_PreparedResetSlot, ...] | None:
    """Compile a complete state-only mutation plan into direct reset slots.

    This is deliberately a capability of the backend-owned mutation binding,
    not a G1 special case.  A cold-bound value-buffer window can only be used
    when *every* mutation field is a supported simulation-state field, which
    preserves the typed sub-batch's exact coverage semantics.  Other plans
    retain the ordinary per-value validation path below.
    """

    slots: list[_PreparedResetSlot] = []
    root_layout = {
        "state.root.position": (qpos_state_offset, 3),
        "state.root.orientation": (qpos_state_offset + 3, 4),
        "state.root.linear_velocity": (qvel_state_offset, 3),
        "state.root.angular_velocity": (qvel_state_offset + 3, 3),
    }
    for field_index, spec in enumerate(plan.specs):
        if spec.target.target_kind is not MutationTargetKind.SIMULATION_STATE:
            return None
        target_key = spec.target.target_key
        root = root_layout.get(target_key)
        if root is not None:
            expected_shape = _ROOT_RESET_VALUE_SHAPES[target_key]
            if (
                spec.target.entity_kind is not MutationEntityKind.BODY
                or len(spec.target.entity_ids) != 1
                or spec.value_buffer.row_shape != expected_shape
                or (root_qpos_dim, root_qvel_dim) != (7, 6)
            ):
                return None
            state_offset, width = root
            slots.append(
                _PreparedResetSlot(
                    field_index=field_index,
                    state_offset=state_offset,
                    width=width,
                )
            )
            continue
        if target_key not in _DOF_RESET_TARGETS:
            return None
        if (
            spec.target.entity_kind is not MutationEntityKind.DOF
            or not spec.target.entity_ids
            or spec.value_buffer.row_shape != (len(spec.target.entity_ids), 1)
        ):
            return None
        state_offset = (
            qpos_state_offset + root_qpos_dim
            if target_key == "state.dof.position"
            else qvel_state_offset + root_qvel_dim
        )
        slots.append(
            _PreparedResetSlot(
                field_index=field_index,
                state_offset=state_offset,
                width=1,
                dof_columns=np.asarray(spec.target.entity_ids, dtype=np.intp),
            )
        )
    return tuple(slots)


@dataclass
class _MuJoCoHostMutationPlan:
    """MuJoCo-owned cold-path staging for one typed mutation plan.

    The public ``BoundMutationPlan`` deliberately does not expose raw MuJoCo
    field layout.  This runtime companion owns the only ``xfrc_applied`` and
    raw reset-state translations, and allocates its control, wrench, reset,
    and row-selection scratch on the cold path rather than during physics.

    The public mutation contract deliberately distinguishes a floating root
    body from joint DoFs.  MuJoCo's full-physics state keeps the first free
    root at fixed qpos/qvel prefixes (7/6); translating those semantic root
    targets here keeps that private layout out of manager code and preserves
    the same compiled reset envelope across the independent ``mujoco`` and
    ``mjwarp`` backends.
    """

    public_plan: BoundMutationPlan
    num_bodies: int
    state_size: int
    state_dtype: str
    qpos_state_offset: int
    qvel_state_offset: int
    root_qpos_dim: int
    root_qvel_dim: int
    _staged_xfrc: np.ndarray = field(init=False, repr=False)
    _reset_state: np.ndarray = field(init=False, repr=False)
    _reset_source_state: np.ndarray | None = field(init=False, repr=False)
    _reset_env_ids: np.ndarray = field(init=False, repr=False)
    _reset_row_ids: np.ndarray = field(init=False, repr=False)
    _staged_reset_rows: RowSelection | None = field(default=None, init=False, repr=False)
    _control_trajectories: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    _prepared_reset_slots: tuple[_PreparedResetSlot, ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _prepared_buffer_sets: dict[
        int,
        _PreparedResetBufferSet,
    ] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_bodies <= 0:
            raise BackendBatchContractError("MuJoCo mutation binding requires at least one body")
        self._staged_xfrc = np.empty(
            (self.public_plan.num_envs, 6 * self.num_bodies),
            dtype=np.float64,
        )
        if self.state_size <= 0:
            raise BackendBatchContractError(
                "MuJoCo mutation binding requires a positive state size"
            )
        # ``BatchEnvPool.reset`` normalizes ``initial_state`` to native
        # float64.  Keep this backend-private staging array native too: the
        # public state cache may intentionally be float32, but passing that
        # cache dtype here would allocate/cast one full reset state on every
        # autoreset.  Input mutation values are copied into this scratch with
        # ordinary widening semantics; only the pool input representation is
        # changed, not the public plan or the cached physics trajectory.
        self._reset_state = np.empty(
            (self.public_plan.num_envs, self.state_size),
            dtype=np.float64,
        )
        source_dtype = np.dtype(self.state_dtype)
        self._reset_source_state = (
            None
            if source_dtype == np.dtype(np.float64)
            else np.empty(
                (self.public_plan.num_envs, self.state_size),
                dtype=source_dtype,
            )
        )
        self._reset_env_ids = np.empty(self.public_plan.num_envs, dtype=np.int32)
        self._reset_row_ids = np.empty(self.public_plan.num_envs, dtype=np.intp)
        for row_id in range(self.public_plan.num_envs):
            self._reset_env_ids[row_id] = row_id
            self._reset_row_ids[row_id] = row_id
        self._prepared_reset_slots = _compile_prepared_reset_slots(
            self.public_plan,
            qpos_state_offset=self.qpos_state_offset,
            qvel_state_offset=self.qvel_state_offset,
            root_qpos_dim=self.root_qpos_dim,
            root_qvel_dim=self.root_qvel_dim,
        )

    def register_batch_plan(self, plan: BoundBackendPlan) -> None:
        """Allocate the plan-paired pool control trajectory on the cold path."""
        if plan.num_envs != self.public_plan.num_envs:
            raise BackendBatchContractError(
                "MuJoCo mutation and state plans must use the same row universe"
            )
        if plan.execution_profile is not ExecutionProfile.HOST_NUMPY:
            raise BackendBatchContractError("MuJoCo typed mutations only support host_numpy plans")
        key = plan.fingerprint
        existing = self._control_trajectories.get(key)
        expected_shape = (
            plan.num_envs,
            plan.control.physics_substeps_per_control,
            plan.control.buffer.row_shape[0] + 6 * self.num_bodies,
        )
        if existing is not None:
            if existing.shape != expected_shape or existing.dtype != np.dtype(np.float64):
                raise BackendBatchContractError(
                    "MuJoCo mutation plan has an incompatible registered control trajectory"
                )
            return
        self._control_trajectories[key] = np.empty(
            expected_shape,
            # mujoco_uni consumes the raw combined ctrl/xfrc state vector in
            # MuJoCo's native scalar type.  Manager control may be float32,
            # but the cold-path trajectory must retain the float64 xfrc lane.
            dtype=np.float64,
        )

    def require_registered_batch_plan(self, plan: BoundBackendPlan) -> None:
        try:
            trajectory = self._control_trajectories[plan.fingerprint]
        except KeyError as exc:
            raise BackendBatchContractError(
                "MuJoCo typed mutation plan was not cold-path paired with this state/control plan"
            ) from exc
        if trajectory.shape[0] != plan.num_envs:
            raise BackendBatchContractError(
                "MuJoCo mutation trajectory row universe does not match the state/control plan"
            )

    @staticmethod
    def _require_value_handle(value, rows: RowSelection) -> np.ndarray:
        handle = value.buffer.handle
        if not isinstance(handle, np.ndarray):
            raise BackendBatchContractError(
                "MuJoCo typed mutation values require numpy host buffer handles"
            )
        expected_shape = (rows.count, *value.spec.value_buffer.row_shape)
        if handle.shape != expected_shape:
            raise BackendBatchContractError(
                "MuJoCo typed mutation value handle shape does not match its bound contract"
            )
        if handle.dtype.name != value.spec.value_buffer.dtype:
            raise BackendBatchContractError(
                "MuJoCo typed mutation value handle dtype does not match its bound contract"
            )
        if not handle.flags.c_contiguous:
            raise BackendBatchContractError("MuJoCo typed mutation values must be C-contiguous")
        return handle

    def stage_external_force(self, batch: TypedBackendMutationBatch) -> bool:
        """Validate and stage selected-row force values without exposing raw fields."""
        self.public_plan.require_compatible(batch.plan)
        if batch.rows.universe_size != self.public_plan.num_envs:
            raise BackendBatchContractError(
                "MuJoCo typed mutation rows do not match the backend row universe"
            )
        if (
            batch.model.values
            or batch.state.values
            or batch.state.bound_buffer_window is not None
            or batch.task_state.values
        ):
            raise BackendBatchContractError(
                "MuJoCo host mutation commit only supports external force values"
            )
        if not batch.wrench.values:
            return False

        staged_values: list[tuple[BoundMutationSpec, np.ndarray]] = []
        for value in batch.wrench.values:
            spec = value.spec
            if (
                spec.target.target_key != "wrench.body.force"
                or len(spec.target.entity_ids) == 0
                or spec.value_buffer.row_shape[-1] != 3
            ):
                raise MutationContractError(
                    "MuJoCo typed mutation plan contains an unsupported external wrench target"
                )
            staged_values.append((spec, self._require_value_handle(value, batch.rows)))

        self._staged_xfrc.fill(0.0)
        if batch.rows.indices is None:
            row_ids = tuple(range(batch.rows.universe_size))
        else:
            row_ids = batch.rows.indices
        for spec, values in staged_values:
            for body_offset, body_id in enumerate(spec.target.entity_ids):
                force_slice = slice(6 * body_id, 6 * body_id + 3)
                for row_offset, row_id in enumerate(row_ids):
                    self._staged_xfrc[row_id, force_slice] = values[row_offset, body_offset]
        return True

    def stage_control_with_wrench(
        self,
        plan: BoundBackendPlan,
        control: np.ndarray,
    ) -> np.ndarray:
        self.require_registered_batch_plan(plan)
        trajectory = self._control_trajectories[plan.fingerprint]
        control_width = plan.control.buffer.row_shape[0]
        np.copyto(trajectory[..., :control_width], control[:, None, ...])
        np.copyto(trajectory[..., control_width:], self._staged_xfrc[:, None, ...])
        return trajectory

    def clear_staged_external_force(self) -> None:
        """Enforce the only supported persistence mode: one physics step."""
        self._staged_xfrc.fill(0.0)

    def _stage_reset_rows(self, rows: RowSelection) -> np.ndarray:
        """Record the selected-row mapping in cold-allocated scratch buffers."""
        if rows.universe_size != self.public_plan.num_envs:
            raise BackendBatchContractError(
                "MuJoCo typed mutation rows do not match the backend row universe"
            )
        if not rows.is_all:
            assert rows.indices is not None
            self._reset_env_ids[: rows.count] = rows.indices
            self._reset_row_ids[: rows.count] = rows.indices
        self._staged_reset_rows = rows
        return self._reset_row_ids[: rows.count]

    def _require_staged_reset_rows(self, rows: RowSelection) -> None:
        if self._staged_reset_rows != rows:
            raise BackendBatchContractError(
                "MuJoCo typed reset rows were not staged by this mutation commit"
            )

    def reset_env_ids(self, rows: RowSelection) -> np.ndarray:
        """Return pool-facing selected IDs after a matching reset state was staged."""
        self._require_staged_reset_rows(rows)
        return self._reset_env_ids[: rows.count]

    def reset_row_ids(self, rows: RowSelection) -> np.ndarray:
        """Return cache-facing selected IDs after a matching reset state was staged."""
        self._require_staged_reset_rows(rows)
        return self._reset_row_ids[: rows.count]

    def _stage_native_physics_state(
        self,
        *,
        physics_state: np.ndarray,
        row_ids: np.ndarray,
        rows: RowSelection,
    ) -> None:
        """Copy selected public cache rows into the native pool scratch.

        ``np.take`` rejects a float32 source with float64 ``out`` under its
        safe-casting rule.  For the public float32 profile, use one additional
        *cold-allocated* source scratch before widening into the native pool
        staging array.  This preserves the no-warm-allocation contract while
        ensuring ``BatchEnvPool.reset`` receives already-native memory.
        """

        count = rows.count
        if rows.is_all:
            np.copyto(self._reset_state, physics_state, casting="unsafe")
            return
        source = self._reset_source_state
        if source is None:
            np.take(physics_state, row_ids, axis=0, out=self._reset_state[:count])
            return
        np.take(physics_state, row_ids, axis=0, out=source[:count])
        np.copyto(self._reset_state[:count], source[:count], casting="unsafe")

    def _stage_prepared_reset_state(
        self,
        *,
        batch: TypedBackendMutationBatch,
        physics_state: np.ndarray,
        window: BoundMutationValueWindow,
    ) -> np.ndarray:
        """Commit a cold-bound complete state window without per-field wrappers.

        The typed envelope has already validated that the window belongs to
        this plan, covers every field exactly once, and is exclusively a
        simulation-state sub-batch.  This backend-owned fast path still
        rechecks the public owner/row identity and uses only the immutable
        cold-bound slot map; it never infers selectors or reaches into a task
        kernel.
        """

        window.plan.require_compatible(self.public_plan)
        if window.rows != batch.rows:
            raise BackendBatchContractError(
                "MuJoCo prepared reset window rows do not match the mutation envelope"
            )
        slots = self._prepared_reset_slots
        if slots is None:
            raise BackendBatchContractError(
                "MuJoCo prepared reset window is unsupported by this mutation plan"
            )
        if len(slots) != len(window.buffers.buffers):
            raise BackendBatchContractError(
                "MuJoCo prepared reset window field coverage differs from the bound plan"
            )

        row_ids = self._stage_reset_rows(batch.rows)
        count = batch.rows.count
        self._stage_native_physics_state(
            physics_state=physics_state,
            row_ids=row_ids,
            rows=batch.rows,
        )

        buffer_set = window.buffers
        buffer_set_id = id(buffer_set)
        prepared = self._prepared_buffer_sets.get(buffer_set_id)
        if prepared is None:
            # A manager-owned fixed-capacity set is validated once when it is
            # constructed and once when this backend first consumes it.  Keep
            # a strong object reference beside the id so Python id reuse can
            # never select a foreign set.  Subsequent reset windows only
            # change row mapping; their numeric addresses and metadata are a
            # cold-bound contract.
            prepared = self._compile_prepared_buffer_set(window=window, slots=slots)
            self._prepared_buffer_sets[buffer_set_id] = prepared
        else:
            if prepared.owner is not buffer_set:  # pragma: no cover - strong-ref invariant.
                raise BackendBatchContractError(
                    "MuJoCo prepared reset buffer identity cache is inconsistent"
                )

        for group in prepared.groups:
            self._reset_state[:count, group.destination_columns] = group.row_values[:count]
        for slot, values in prepared.individual:
            if slot.dof_columns is None:
                np.copyto(
                    self._reset_state[:count, slot.state_offset : slot.state_offset + slot.width],
                    values[:count, 0, :],
                    casting="unsafe",
                )
            else:
                self._reset_state[:count, slot.state_offset + slot.dof_columns] = values[
                    :count, :, 0
                ]
        return self._reset_state[:count]

    @staticmethod
    def _compile_prepared_buffer_set(
        *,
        window: BoundMutationValueWindow,
        slots: tuple[_PreparedResetSlot, ...],
    ) -> _PreparedResetBufferSet:
        """Compile supported group hints while retaining canonical fallback."""

        values_by_field = {slot.field_index: window.buffer_at(slot.field_index) for slot in slots}
        slots_by_field = {slot.field_index: slot for slot in slots}
        grouped_fields: set[int] = set()
        prepared_groups: list[_PreparedResetBufferGroup] = []
        for group in window.buffers.groups:
            compiled = _MuJoCoHostMutationPlan._compile_prepared_buffer_group(
                group=group,
                slots_by_field=slots_by_field,
            )
            if compiled is None:
                continue
            prepared_groups.append(compiled)
            grouped_fields.update(group.field_indices)
        individual = tuple(
            (slot, values_by_field[slot.field_index])
            for slot in slots
            if slot.field_index not in grouped_fields
        )
        return _PreparedResetBufferSet(
            owner=window.buffers,
            individual=individual,
            groups=tuple(prepared_groups),
        )

    @staticmethod
    def _compile_prepared_buffer_group(
        *,
        group: BoundMutationValueBufferGroup,
        slots_by_field: dict[int, _PreparedResetSlot],
    ) -> _PreparedResetBufferGroup | None:
        """Translate a singleton-DoF group to one field-major column write."""

        slots = tuple(slots_by_field.get(field_index) for field_index in group.field_indices)
        if any(slot is None for slot in slots) or group.buffer.shape[2:] != (1, 1):
            return None
        typed_slots = tuple(slot for slot in slots if slot is not None)
        columns: list[int] = []
        for slot in typed_slots:
            dof_columns = slot.dof_columns
            if slot.width != 1 or dof_columns is None or dof_columns.shape != (1,):
                return None
            columns.append(slot.state_offset + int(dof_columns[0]))
        destination_columns = np.asarray(
            columns,
            dtype=np.intp,
        )
        if len(set(int(column) for column in destination_columns)) != len(destination_columns):
            return None
        return _PreparedResetBufferGroup(
            destination_columns=destination_columns,
            row_values=group.buffer[:, :, 0, 0].T,
        )

    def stage_reset_state(
        self,
        batch: TypedBackendMutationBatch,
        physics_state: np.ndarray,
    ) -> np.ndarray:
        """Patch bound floating-root and hinge values into reset-state scratch."""
        self.public_plan.require_compatible(batch.plan)
        if batch.rows.universe_size != self.public_plan.num_envs:
            raise BackendBatchContractError(
                "MuJoCo typed mutation rows do not match the backend row universe"
            )
        if batch.model.values or batch.wrench.values or batch.task_state.values:
            raise BackendBatchContractError(
                "MuJoCo host reset commit only supports simulation-state values"
            )
        if physics_state.shape != (self.public_plan.num_envs, self.state_size):
            raise BackendBatchContractError(
                "MuJoCo typed reset source state does not match the bound mutation plan"
            )

        prepared_window = batch.state.bound_buffer_window
        if prepared_window is not None:
            return self._stage_prepared_reset_state(
                batch=batch,
                physics_state=physics_state,
                window=prepared_window,
            )
        if not batch.state.values:
            raise BackendBatchContractError("MuJoCo typed reset requires at least one state value")

        staged_values: list[tuple[BoundMutationSpec, np.ndarray]] = []
        for value in batch.state.values:
            spec = value.spec
            if spec.target.target_kind is not MutationTargetKind.SIMULATION_STATE:
                raise MutationContractError(
                    "MuJoCo typed mutation plan contains an unsupported reset-state target"
                )
            target_key = spec.target.target_key
            expected_shape = _ROOT_RESET_VALUE_SHAPES.get(target_key)
            if expected_shape is not None:
                if (
                    spec.target.entity_kind is not MutationEntityKind.BODY
                    or len(spec.target.entity_ids) != 1
                    or spec.value_buffer.row_shape != expected_shape
                ):
                    raise MutationContractError(
                        "MuJoCo typed root reset target has an invalid bound layout"
                    )
            elif target_key in _DOF_RESET_TARGETS:
                if (
                    spec.target.entity_kind is not MutationEntityKind.DOF
                    or not spec.target.entity_ids
                    or spec.value_buffer.row_shape[-1] != 1
                ):
                    raise MutationContractError(
                        "MuJoCo typed DoF reset target has an invalid bound layout"
                    )
            else:
                raise MutationContractError(
                    "MuJoCo typed mutation plan contains an unsupported reset-state target"
                )
            staged_values.append((spec, self._require_value_handle(value, batch.rows)))

        row_ids = self._stage_reset_rows(batch.rows)
        self._stage_native_physics_state(
            physics_state=physics_state,
            row_ids=row_ids,
            rows=batch.rows,
        )

        for spec, values in staged_values:
            target_key = spec.target.target_key
            if target_key == "state.root.position":
                state_offset = self.qpos_state_offset
                for row_offset in range(batch.rows.count):
                    self._reset_state[row_offset, state_offset : state_offset + 3] = values[
                        row_offset, 0, :
                    ]
            elif target_key == "state.root.orientation":
                state_offset = self.qpos_state_offset + 3
                for row_offset in range(batch.rows.count):
                    self._reset_state[row_offset, state_offset : state_offset + 4] = values[
                        row_offset, 0, :
                    ]
            elif target_key == "state.root.linear_velocity":
                state_offset = self.qvel_state_offset
                for row_offset in range(batch.rows.count):
                    self._reset_state[row_offset, state_offset : state_offset + 3] = values[
                        row_offset, 0, :
                    ]
            elif target_key == "state.root.angular_velocity":
                state_offset = self.qvel_state_offset + 3
                for row_offset in range(batch.rows.count):
                    self._reset_state[row_offset, state_offset : state_offset + 3] = values[
                        row_offset, 0, :
                    ]
            elif target_key == "state.dof.position":
                state_offset = self.qpos_state_offset + self.root_qpos_dim
                for dof_offset, dof_id in enumerate(spec.target.entity_ids):
                    for row_offset in range(batch.rows.count):
                        self._reset_state[row_offset, state_offset + dof_id] = values[
                            row_offset, dof_offset, 0
                        ]
            elif target_key == "state.dof.angular_velocity":
                state_offset = self.qvel_state_offset + self.root_qvel_dim
                for dof_offset, dof_id in enumerate(spec.target.entity_ids):
                    for row_offset in range(batch.rows.count):
                        self._reset_state[row_offset, state_offset + dof_id] = values[
                            row_offset, dof_offset, 0
                        ]
            else:
                raise MutationContractError(
                    "MuJoCo typed reset target has an unsupported field kind"
                )
        return self._reset_state[: batch.rows.count]


def _require_field_contract(
    backend: MuJoCoBackend,
    spec: StateFieldSpec,
    *,
    row_shape: tuple[int, ...],
    frame: ReferenceFrame | None = None,
    unit: PhysicalUnit | None = None,
) -> None:
    expected_dtype = np.dtype(backend._np_dtype).name
    if spec.buffer.row_shape != row_shape:
        raise BackendBatchContractError(
            f"MuJoCo field {spec.key!r} requires row_shape {row_shape}, got {spec.buffer.row_shape}"
        )
    if spec.buffer.dtype != expected_dtype:
        raise BackendBatchContractError(
            f"MuJoCo field {spec.key!r} requires dtype {expected_dtype}, got {spec.buffer.dtype}"
        )
    if spec.buffer.layout is not BufferLayout.C_CONTIGUOUS:
        raise BackendBatchContractError(
            f"MuJoCo host field {spec.key!r} requires c_contiguous layout"
        )
    if frame is not None and spec.frame is not frame:
        raise BackendBatchContractError(
            f"MuJoCo field {spec.key!r} requires frame {frame.value}, got {spec.frame.value}"
        )
    if unit is not None and spec.unit is not unit:
        raise BackendBatchContractError(
            f"MuJoCo field {spec.key!r} requires unit {unit.value}, got {spec.unit.value}"
        )


def _bind_root_source(backend: MuJoCoBackend, spec: StateFieldSpec) -> _MuJoCoStateSource:
    if (backend._root_qpos_dim, backend._root_qvel_dim) != (7, 6):
        raise BackendBatchContractError(
            "MuJoCo typed root fields require a free-joint root cache; "
            "fixed-base body state must use a bound body field"
        )
    identity = spec.identity
    if identity.entity_ids != (backend._base_body_id,):
        raise BackendBatchContractError(
            f"MuJoCo root field {spec.key!r} must bind base body id {backend._base_body_id}"
        )
    sources = {
        StateFieldKind.POSITION: (
            backend._base_pos_view,
            (3,),
            ReferenceFrame.WORLD,
            PhysicalUnit.METER,
        ),
        StateFieldKind.ORIENTATION: (
            backend._base_quat_view,
            (4,),
            ReferenceFrame.WORLD,
            PhysicalUnit.QUATERNION,
        ),
        StateFieldKind.LINEAR_VELOCITY: (
            backend._base_lin_vel_view,
            (3,),
            ReferenceFrame.WORLD,
            PhysicalUnit.METER_PER_SECOND,
        ),
        StateFieldKind.ANGULAR_VELOCITY: (
            backend._base_ang_vel_view,
            (3,),
            ReferenceFrame.WORLD,
            PhysicalUnit.RADIAN_PER_SECOND,
        ),
    }
    try:
        source, row_shape, frame, unit = sources[identity.field_kind]
    except KeyError as exc:
        raise BackendBatchContractError(
            f"unsupported MuJoCo root field kind {identity.field_kind.value!r}"
        ) from exc
    _require_field_contract(
        backend,
        spec,
        row_shape=row_shape,
        frame=frame,
        unit=unit,
    )
    return _MuJoCoStateSource(spec=spec, source=source)


def _bind_dof_source(backend: MuJoCoBackend, spec: StateFieldSpec) -> _MuJoCoStateSource:
    identity = spec.identity
    indices = np.asarray(identity.entity_ids, dtype=np.intp)
    if identity.field_kind is StateFieldKind.POSITION:
        source = backend._dof_pos_view
        if np.any(indices >= source.shape[1]):
            raise BackendBatchContractError(
                f"MuJoCo DOF field {spec.key!r} contains an out-of-range id"
            )
        position_types: list[int | None] = [None] * source.shape[1]
        for joint_id in range(int(backend._model.njnt)):
            joint_type = int(backend._model.jnt_type[joint_id])
            if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            start = int(backend._model.jnt_qposadr[joint_id]) - backend._root_qpos_dim
            width = 4 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1
            for offset in range(width):
                if 0 <= start + offset < len(position_types):
                    position_types[start + offset] = joint_type
        selected_types = {position_types[int(index)] for index in indices}
        if selected_types == {int(mujoco.mjtJoint.mjJNT_HINGE)}:
            unit = PhysicalUnit.RADIAN
        elif selected_types == {int(mujoco.mjtJoint.mjJNT_SLIDE)}:
            unit = PhysicalUnit.METER
        else:
            raise BackendBatchContractError(
                f"MuJoCo DOF position field {spec.key!r} must select homogeneous "
                "hinge or slide coordinates; ball/mixed fields require a separate semantic"
            )
    elif identity.field_kind in {
        StateFieldKind.ANGULAR_VELOCITY,
        StateFieldKind.LINEAR_VELOCITY,
    }:
        source = backend._dof_vel_view
        if np.any(indices >= source.shape[1]):
            raise BackendBatchContractError(
                f"MuJoCo DOF field {spec.key!r} contains an out-of-range id"
            )
        velocity_types: list[int | None] = [None] * source.shape[1]
        for joint_id in range(int(backend._model.njnt)):
            joint_type = int(backend._model.jnt_type[joint_id])
            if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            start = int(backend._model.jnt_dofadr[joint_id]) - backend._root_qvel_dim
            width = 3 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1
            for offset in range(width):
                if 0 <= start + offset < len(velocity_types):
                    velocity_types[start + offset] = joint_type
        selected_types = {velocity_types[int(index)] for index in indices}
        angular_types = {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_BALL),
        }
        if identity.field_kind is StateFieldKind.ANGULAR_VELOCITY:
            if not selected_types or not selected_types.issubset(angular_types):
                raise BackendBatchContractError(
                    f"MuJoCo angular DOF field {spec.key!r} contains a non-angular coordinate"
                )
            unit = PhysicalUnit.RADIAN_PER_SECOND
        else:
            if selected_types != {int(mujoco.mjtJoint.mjJNT_SLIDE)}:
                raise BackendBatchContractError(
                    f"MuJoCo linear DOF field {spec.key!r} contains a non-linear coordinate"
                )
            unit = PhysicalUnit.METER_PER_SECOND
    else:
        raise BackendBatchContractError(
            f"unsupported MuJoCo DOF field kind {identity.field_kind.value!r}"
        )
    _require_field_contract(
        backend,
        spec,
        row_shape=(len(indices),),
        frame=ReferenceFrame.JOINT,
        unit=unit,
    )
    return _MuJoCoStateSource(
        spec=spec,
        source=source,
        gather_indices=indices,
    )


def _bind_sensor_source(backend: MuJoCoBackend, spec: StateFieldSpec) -> _MuJoCoStateSource:
    identity = spec.identity
    if identity.field_kind is not StateFieldKind.VALUE:
        raise BackendBatchContractError("MuJoCo sensor fields only support value semantics")
    sensor_columns: list[int] = []
    sensor_contracts = {
        int(mujoco.mjtSensor.mjSENS_VELOCIMETER): (
            ReferenceFrame.SENSOR,
            PhysicalUnit.METER_PER_SECOND,
        ),
        int(mujoco.mjtSensor.mjSENS_GYRO): (
            ReferenceFrame.SENSOR,
            PhysicalUnit.RADIAN_PER_SECOND,
        ),
        int(mujoco.mjtSensor.mjSENS_CONTACT): (
            ReferenceFrame.SENSOR,
            PhysicalUnit.NEWTON,
        ),
    }
    frame_sensor_units = {
        int(mujoco.mjtSensor.mjSENS_FRAMEZAXIS): PhysicalUnit.UNITLESS,
        int(mujoco.mjtSensor.mjSENS_FRAMEPOS): PhysicalUnit.METER,
        int(mujoco.mjtSensor.mjSENS_FRAMEQUAT): PhysicalUnit.QUATERNION,
    }
    expected_contracts: set[tuple[ReferenceFrame, PhysicalUnit]] = set()
    for sensor_id in identity.entity_ids:
        if sensor_id >= int(backend._model.nsensor):
            raise BackendBatchContractError(
                f"MuJoCo sensor field {spec.key!r} contains an out-of-range id"
            )
        start = int(backend._model.sensor_adr[sensor_id])
        dim = int(backend._model.sensor_dim[sensor_id])
        sensor_columns.extend(range(start, start + dim))
        sensor_type = int(backend._model.sensor_type[sensor_id])
        if sensor_type in frame_sensor_units:
            reference_id = int(backend._model.sensor_refid[sensor_id])
            if reference_id < 0:
                reference_frame = ReferenceFrame.WORLD
            else:
                reference_type = int(backend._model.sensor_reftype[sensor_id])
                body_reference_types = {
                    int(mujoco.mjtObj.mjOBJ_BODY),
                    int(mujoco.mjtObj.mjOBJ_XBODY),
                }
                if (
                    reference_type not in body_reference_types
                    or reference_id != backend._base_body_id
                ):
                    raise BackendBatchContractError(
                        f"MuJoCo sensor field {spec.key!r} has an unsupported reference frame"
                    )
                reference_frame = ReferenceFrame.BASE
            expected_contracts.add((reference_frame, frame_sensor_units[sensor_type]))
            continue
        try:
            expected_contracts.add(sensor_contracts[sensor_type])
        except KeyError as exc:
            raise BackendBatchContractError(
                f"MuJoCo sensor type {sensor_type} has no typed state contract"
            ) from exc
    if expected_contracts != {(spec.frame, spec.unit)}:
        expected = ", ".join(
            f"{frame.value}/{unit.value}"
            for frame, unit in sorted(
                expected_contracts,
                key=lambda item: (item[0].value, item[1].value),
            )
        )
        raise BackendBatchContractError(
            f"MuJoCo sensor field {spec.key!r} requires homogeneous frame/unit {expected}"
        )
    indices = np.asarray(sensor_columns, dtype=np.intp)
    _require_field_contract(backend, spec, row_shape=(len(indices),))
    return _MuJoCoStateSource(
        spec=spec,
        source=backend._sensor_data,
        gather_indices=indices,
    )


def _bind_body_source(backend: MuJoCoBackend, spec: StateFieldSpec) -> _MuJoCoStateSource:
    if not backend.add_body_sensors:
        raise BackendBatchContractError(
            "MuJoCo body batch fields require cold-path add_body_sensors materialization"
        )
    identity = spec.identity
    body_ids = np.asarray(identity.entity_ids, dtype=np.intp)
    if np.any(body_ids >= int(backend._model.nbody)):
        raise BackendBatchContractError(
            f"MuJoCo body field {spec.key!r} contains an out-of-range id"
        )
    mapped = backend._body_id_to_tracked_idx[body_ids]
    if np.any(mapped < 0):
        raise BackendBatchContractError(
            f"MuJoCo body field {spec.key!r} references a body without a bound tracking cache"
        )
    sources: dict[
        tuple[StateFieldKind, ReferenceFrame],
        tuple[np.ndarray, int, PhysicalUnit],
    ] = {
        (StateFieldKind.POSITION, ReferenceFrame.WORLD): (
            backend._tracked_pos_w_all,
            3,
            PhysicalUnit.METER,
        ),
        (StateFieldKind.ORIENTATION, ReferenceFrame.WORLD): (
            backend._tracked_quat_w_all,
            4,
            PhysicalUnit.QUATERNION,
        ),
        (StateFieldKind.LINEAR_VELOCITY, ReferenceFrame.WORLD): (
            backend._tracked_linvel_w_all,
            3,
            PhysicalUnit.METER_PER_SECOND,
        ),
        (StateFieldKind.ANGULAR_VELOCITY, ReferenceFrame.WORLD): (
            backend._tracked_angvel_w_all,
            3,
            PhysicalUnit.RADIAN_PER_SECOND,
        ),
        (StateFieldKind.POSITION, ReferenceFrame.BASE): (
            backend._tracked_pos_b_all,
            3,
            PhysicalUnit.METER,
        ),
        (StateFieldKind.ORIENTATION, ReferenceFrame.BASE): (
            backend._tracked_quat_b_all,
            4,
            PhysicalUnit.QUATERNION,
        ),
        (StateFieldKind.LINEAR_VELOCITY, ReferenceFrame.BASE): (
            backend._tracked_linvel_b_all,
            3,
            PhysicalUnit.METER_PER_SECOND,
        ),
        (StateFieldKind.ANGULAR_VELOCITY, ReferenceFrame.BASE): (
            backend._tracked_angvel_b_all,
            3,
            PhysicalUnit.RADIAN_PER_SECOND,
        ),
    }
    try:
        source, width, unit = sources[(identity.field_kind, spec.frame)]
    except KeyError as exc:
        raise BackendBatchContractError(
            f"unsupported MuJoCo body field/frame combination for {spec.key!r}"
        ) from exc
    _require_field_contract(
        backend,
        spec,
        row_shape=(len(mapped), width),
        frame=spec.frame,
        unit=unit,
    )
    return _MuJoCoStateSource(
        spec=spec,
        source=source,
        gather_indices=np.asarray(mapped, dtype=np.intp),
    )


def _bind_state_source(backend: MuJoCoBackend, spec: StateFieldSpec) -> _MuJoCoStateSource:
    binders = {
        StateEntityKind.ROOT: _bind_root_source,
        StateEntityKind.DOF: _bind_dof_source,
        StateEntityKind.SENSOR: _bind_sensor_source,
        StateEntityKind.BODY: _bind_body_source,
    }
    return binders[spec.identity.entity_kind](backend, spec)


def _binding_payloads(
    backend: MuJoCoBackend,
    requirements: BackendIORequirements,
) -> tuple[dict[str, Any], dict[str, Any]]:
    def _names(object_type: mujoco.mjtObj, count: int) -> tuple[str, ...]:
        return tuple(
            mujoco.mj_id2name(backend._model, object_type, index) or f"#{index}"
            for index in range(count)
        )

    def _buffer_payload(buffer: Any) -> dict[str, Any]:
        return {
            "row_shape": buffer.row_shape,
            "dtype": buffer.dtype,
            "layout": buffer.layout.value,
            "placement": {
                "memory_space": buffer.placement.memory_space.value,
                "device_type": buffer.placement.device_type,
                "device_index": buffer.placement.device_index,
            },
            "owner": buffer.owner.value,
            "mutability": buffer.mutability.value,
            "lifetime": buffer.lifetime.value,
            "dlpack_exportable": buffer.dlpack_exportable,
            "address_stable": buffer.address_stable,
        }

    fields = []
    for spec in requirements.state_fields:
        fields.append(
            {
                "semantic_key": spec.key,
                "entity_kind": spec.identity.entity_kind.value,
                "field_kind": spec.identity.field_kind.value,
                "entity_ids": spec.identity.entity_ids,
                "frame": spec.frame.value,
                "unit": spec.unit.value,
                "buffer": _buffer_payload(spec.buffer),
            }
        )
    state_payload = {
        "contract_version": requirements.contract_version,
        "backend_type": backend.backend_type,
        "model_dims": {
            "nq": int(backend._model.nq),
            "nv": int(backend._model.nv),
            "nu": int(backend._model.nu),
            "nsensor": int(backend._model.nsensor),
        },
        "model_semantics": {
            "body_names": _names(mujoco.mjtObj.mjOBJ_BODY, int(backend._model.nbody)),
            "joint_names": _names(mujoco.mjtObj.mjOBJ_JOINT, int(backend._model.njnt)),
            "actuator_names": _names(
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                int(backend._model.nu),
            ),
            "sensor_names": _names(mujoco.mjtObj.mjOBJ_SENSOR, int(backend._model.nsensor)),
            "joint_type": tuple(int(value) for value in backend._model.jnt_type),
            "joint_qposadr": tuple(int(value) for value in backend._model.jnt_qposadr),
            "joint_dofadr": tuple(int(value) for value in backend._model.jnt_dofadr),
            "sensor_type": tuple(int(value) for value in backend._model.sensor_type),
            "sensor_dim": tuple(int(value) for value in backend._model.sensor_dim),
            "sensor_adr": tuple(int(value) for value in backend._model.sensor_adr),
            "sensor_objtype": tuple(int(value) for value in backend._model.sensor_objtype),
            "sensor_objid": tuple(int(value) for value in backend._model.sensor_objid),
            "sensor_reftype": tuple(int(value) for value in backend._model.sensor_reftype),
            "sensor_refid": tuple(int(value) for value in backend._model.sensor_refid),
            "timestep": float(backend._model.opt.timestep),
            "dtype": np.dtype(backend._np_dtype).name,
        },
        "execution_profile": requirements.execution_profile.value,
        "fields": fields,
    }
    plan_payload = {
        "state": state_payload,
        "control": {
            "semantic_key": requirements.control.semantic_key,
            "buffer": _buffer_payload(requirements.control.buffer),
            "cadence": requirements.control.physics_substeps_per_control,
        },
        "hot_path_budget": (
            None
            if requirements.hot_path_budget is None
            else dict(requirements.hot_path_budget.items())
        ),
        "reset_hot_path_budget": (
            None
            if requirements.reset_hot_path_budget is None
            else dict(requirements.reset_hot_path_budget.items())
        ),
    }
    return state_payload, plan_payload


def _payload_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _bind_mujoco_host_batch(
    backend: MuJoCoBackend,
    requirements: BackendIORequirements,
    *,
    backend_instance_id: str,
) -> _MuJoCoHostBatchPlan:
    if requirements.execution_profile is not ExecutionProfile.HOST_NUMPY:
        raise BackendBatchContractError("MuJoCo reference batches only support host_numpy")
    if backend._pre_step_control_fn is not None:
        raise BackendBatchContractError(
            "MuJoCo managed host batches do not support pre-step control callbacks"
        )
    expected_dtype = np.dtype(backend._np_dtype).name
    control = requirements.control
    if control.buffer.row_shape != (int(backend._model.nu),):
        raise BackendBatchContractError(
            f"MuJoCo control requires row_shape {(int(backend._model.nu),)}, "
            f"got {control.buffer.row_shape}"
        )
    if control.buffer.dtype != expected_dtype:
        raise BackendBatchContractError(
            f"MuJoCo control requires dtype {expected_dtype}, got {control.buffer.dtype}"
        )
    if control.buffer.layout is not BufferLayout.C_CONTIGUOUS:
        raise BackendBatchContractError("MuJoCo host control requires c_contiguous layout")
    if requirements.hot_path_budget is not None:
        BackendBatchCounters(
            allocations=_step_allocations(control.buffer.dtype),
            state_materializations=1,
            instrumentation_complete=True,
        ).require_within(requirements.hot_path_budget)

    sources = tuple(_bind_state_source(backend, spec) for spec in requirements.state_fields)
    state_payload, plan_payload = _binding_payloads(backend, requirements)
    state_digest = _payload_digest(state_payload)
    plan_digest = _payload_digest(plan_payload)
    state_plan = BoundStatePlan(
        backend_type=backend.backend_type,
        backend_instance_id=backend_instance_id,
        num_envs=backend.num_envs,
        fields=requirements.state_fields,
        execution_profile=requirements.execution_profile,
        fingerprint=f"{_STATE_FINGERPRINT_PREFIX}:{state_digest}",
    )
    public_plan = BoundBackendPlan(
        state=state_plan,
        control=control,
        execution_profile=requirements.execution_profile,
        fingerprint=f"{_PLAN_FINGERPRINT_PREFIX}:{plan_digest}",
        hot_path_budget=requirements.hot_path_budget,
        reset_hot_path_budget=requirements.reset_hot_path_budget,
        contract_version=BACKEND_BATCH_CONTRACT_VERSION,
    )
    return _MuJoCoHostBatchPlan(
        public_plan=public_plan,
        sources=sources,
        lease=StateBatchLease(backend_instance_id),
    )
