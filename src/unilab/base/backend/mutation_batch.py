"""Runtime batch envelopes for a cold-path bound mutation plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .batch import (
    BufferLifetime,
    BufferMutability,
    BufferOwner,
    BufferView,
    MemorySpace,
    RowSelection,
)
from .device import DeviceCompletion, DeviceTensorView, require_device_tensor_view
from .mutation import (
    BoundMutationPlan,
    BoundMutationSpec,
    MutationContractError,
    MutationTargetKind,
)


@dataclass(frozen=True)
class MutationValueBatch:
    plan: BoundMutationPlan
    field_index: int
    rows: RowSelection
    buffer: BufferView

    def __post_init__(self) -> None:
        if not isinstance(self.plan, BoundMutationPlan):
            raise MutationContractError("mutation value plan must be a BoundMutationPlan")
        if isinstance(self.field_index, bool) or not isinstance(self.field_index, int):
            raise MutationContractError("mutation field_index must be an integer")
        if self.field_index < 0:
            raise MutationContractError("mutation field_index is not bound")
        try:
            spec = self.plan.specs[self.field_index]
        except IndexError as exc:
            raise MutationContractError("mutation field_index is not bound") from exc
        if not isinstance(self.rows, RowSelection):
            raise MutationContractError("mutation rows must be a RowSelection")
        if self.rows.universe_size != self.plan.num_envs:
            raise MutationContractError("mutation row universe does not match bound plan")
        if not isinstance(self.buffer, BufferView):
            raise MutationContractError("mutation value buffer must be a BufferView")
        if self.buffer.contract != spec.value_buffer:
            raise MutationContractError("mutation value metadata does not match bound field")
        expected_shape = (self.rows.count, *spec.value_buffer.row_shape)
        if self.buffer.shape != expected_shape:
            raise MutationContractError(
                f"mutation value requires shape {expected_shape}, got {self.buffer.shape}"
            )

    @property
    def spec(self) -> BoundMutationSpec:
        return self.plan.specs[self.field_index]


@dataclass(frozen=True)
class BoundMutationValueBufferGroup:
    """Homogeneous field-major storage hint for cold-bound mutation values.

    ``field_indices`` maps the leading axis of ``buffer`` to canonical fields
    in a :class:`BoundMutationValueBuffers` set.  Backends may use this hint to
    commit homogeneous fields in one vectorized operation.  The canonical
    per-field buffers remain authoritative, so an unsupported group can always
    fall back without dropping a mutation field.
    """

    field_indices: tuple[int, ...]
    buffer: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.field_indices, tuple)
            or not self.field_indices
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in self.field_indices
            )
        ):
            raise MutationContractError(
                "bound mutation value buffer group requires non-negative field indices"
            )
        if len(set(self.field_indices)) != len(self.field_indices):
            raise MutationContractError(
                "bound mutation value buffer group contains duplicate field indices"
            )
        if not isinstance(self.buffer, np.ndarray):
            raise MutationContractError("bound mutation value buffer group must use a numpy array")


def _same_array_view(left: np.ndarray, right: np.ndarray) -> bool:
    """Return whether two arrays describe the exact same numeric view."""

    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.strides == right.strides
        and int(left.__array_interface__["data"][0]) == int(right.__array_interface__["data"][0])
    )


@dataclass(frozen=True)
class BoundMutationValueBuffers:
    """Cold-bound manager buffers for a complete typed mutation plan.

    A normal :class:`MutationValueBatch` intentionally carries a freshly
    shaped ``BufferView`` for each value at every barrier.  That is the right
    general API for sparse, heterogeneous mutation batches, but it creates a
    substantial amount of descriptor/validation work for a task which always
    writes every field of one already-bound plan (for example G1's complete
    reset state).

    This class is the explicit, opt-in alternative for that narrow case.  It
    validates fixed-capacity manager-owned arrays *once* on the cold path.
    A later :meth:`window` only supplies the row mapping; values always occupy
    the leading ``rows.count`` entries of each stable backing array.  The
    enclosing typed sub-batch still checks plan ownership, exact rows, target
    kind, and canonical field coverage, so this does not introduce an
    untyped dictionary or relax backend ownership.
    """

    plan: BoundMutationPlan
    buffers: tuple[np.ndarray, ...] = field(repr=False, compare=False)
    groups: tuple[BoundMutationValueBufferGroup, ...] = field(
        default=(),
        repr=False,
        compare=False,
    )
    _field_indices: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.plan, BoundMutationPlan):
            raise MutationContractError("bound mutation value buffers require a BoundMutationPlan")
        if not isinstance(self.buffers, tuple) or len(self.buffers) != len(self.plan.specs):
            raise MutationContractError(
                "bound mutation value buffers must contain one array per bound mutation field"
            )
        for index, (spec, buffer) in enumerate(zip(self.plan.specs, self.buffers, strict=True)):
            if not isinstance(buffer, np.ndarray):
                raise MutationContractError(
                    f"bound mutation value buffer {index} must be a numpy array"
                )
            expected_shape = (self.plan.num_envs, *spec.value_buffer.row_shape)
            if buffer.shape != expected_shape:
                raise MutationContractError(
                    f"bound mutation value buffer {index} requires shape {expected_shape}, "
                    f"got {buffer.shape}"
                )
            if buffer.dtype.name != spec.value_buffer.dtype:
                raise MutationContractError(
                    f"bound mutation value buffer {index} dtype does not match its field contract"
                )
            if not buffer.flags.c_contiguous:
                raise MutationContractError(
                    f"bound mutation value buffer {index} must be C-contiguous"
                )
            if not buffer.flags.writeable:
                raise MutationContractError(
                    f"bound mutation value buffer {index} must remain manager-writeable"
                )
        if not isinstance(self.groups, tuple) or any(
            not isinstance(group, BoundMutationValueBufferGroup) for group in self.groups
        ):
            raise MutationContractError(
                "bound mutation value buffer groups must be typed buffer groups"
            )
        grouped_fields: set[int] = set()
        for group_index, group in enumerate(self.groups):
            overlap = grouped_fields.intersection(group.field_indices)
            if overlap:
                raise MutationContractError(
                    "bound mutation value buffer groups overlap canonical fields: "
                    + ", ".join(str(index) for index in sorted(overlap))
                )
            try:
                specs = tuple(self.plan.specs[index] for index in group.field_indices)
                field_buffers = tuple(self.buffers[index] for index in group.field_indices)
            except IndexError as exc:
                raise MutationContractError(
                    f"bound mutation value buffer group {group_index} references an unbound field"
                ) from exc
            first_contract = specs[0].value_buffer
            if any(spec.value_buffer != first_contract for spec in specs[1:]):
                raise MutationContractError(
                    f"bound mutation value buffer group {group_index} fields are not homogeneous"
                )
            expected_shape = (
                len(group.field_indices),
                self.plan.num_envs,
                *first_contract.row_shape,
            )
            if group.buffer.shape != expected_shape:
                raise MutationContractError(
                    f"bound mutation value buffer group {group_index} requires shape "
                    f"{expected_shape}, got {group.buffer.shape}"
                )
            if group.buffer.dtype.name != first_contract.dtype:
                raise MutationContractError(
                    f"bound mutation value buffer group {group_index} dtype does not match "
                    "its field contracts"
                )
            if not group.buffer.flags.c_contiguous or not group.buffer.flags.writeable:
                raise MutationContractError(
                    f"bound mutation value buffer group {group_index} must be C-contiguous "
                    "and manager-writeable"
                )
            for field_offset, field_buffer in enumerate(field_buffers):
                if not _same_array_view(field_buffer, group.buffer[field_offset]):
                    raise MutationContractError(
                        f"bound mutation value buffer group {group_index} field slice does not "
                        "match its canonical buffer"
                    )
            grouped_fields.update(group.field_indices)
        object.__setattr__(self, "_field_indices", tuple(range(len(self.buffers))))

    def window(self, rows: RowSelection) -> "BoundMutationValueWindow":
        """Bind a runtime row selection without allocating per-field views."""

        return BoundMutationValueWindow(buffers=self, rows=rows)


@dataclass(frozen=True)
class BoundMutationValueWindow:
    """Row-scoped use of a cold-bound :class:`BoundMutationValueBuffers` set."""

    buffers: BoundMutationValueBuffers = field(repr=False, compare=False)
    rows: RowSelection

    def __post_init__(self) -> None:
        if not isinstance(self.buffers, BoundMutationValueBuffers):
            raise MutationContractError(
                "bound mutation value window requires cold-bound mutation value buffers"
            )
        if not isinstance(self.rows, RowSelection):
            raise MutationContractError("bound mutation value window rows must be a RowSelection")
        if self.rows.universe_size != self.buffers.plan.num_envs:
            raise MutationContractError(
                "bound mutation value window row universe does not match its bound plan"
            )

    @property
    def plan(self) -> BoundMutationPlan:
        """Expose the immutable plan identity needed by the typed envelope."""

        return self.buffers.plan

    @property
    def field_indices(self) -> tuple[int, ...]:
        """The canonical, complete field set represented by this window."""

        return self.buffers._field_indices

    def buffer_at(self, field_index: int) -> np.ndarray:
        """Return stable full-capacity storage; consumers use ``rows.count``.

        Returning the backing storage rather than a prefix view is deliberate:
        it avoids constructing one temporary ndarray view per field per reset.
        Backend consumers already receive ``rows`` and must only read the
        leading local rows of this manager-owned staging contract.
        """

        if isinstance(field_index, bool) or not isinstance(field_index, int) or field_index < 0:
            raise MutationContractError("bound mutation field_index must be an integer")
        try:
            buffer = self.buffers.buffers[field_index]
            spec = self.plan.specs[field_index]
        except IndexError as exc:
            raise MutationContractError("bound mutation field_index is not bound") from exc
        expected_shape = (self.plan.num_envs, *spec.value_buffer.row_shape)
        if (
            buffer.shape != expected_shape
            or buffer.dtype.name != spec.value_buffer.dtype
            or not buffer.flags.c_contiguous
            or not buffer.flags.writeable
        ):
            raise MutationContractError(
                "cold-bound mutation value buffer changed after materialization"
            )
        return buffer


def _validate_values(values: tuple[MutationValueBatch, ...], name: str) -> None:
    if not isinstance(values, tuple) or any(
        not isinstance(value, MutationValueBatch) for value in values
    ):
        raise MutationContractError(f"{name} values must be typed mutation values")


@dataclass(frozen=True)
class ModelParameterMutationBatch:
    values: tuple[MutationValueBatch, ...] = ()

    def __post_init__(self) -> None:
        _validate_values(self.values, self.__class__.__name__)


@dataclass(frozen=True)
class SimulationStateMutationBatch:
    values: tuple[MutationValueBatch, ...] = ()
    bound_buffer_window: BoundMutationValueWindow | None = None

    def __post_init__(self) -> None:
        _validate_values(self.values, self.__class__.__name__)
        if self.bound_buffer_window is not None:
            if not isinstance(self.bound_buffer_window, BoundMutationValueWindow):
                raise MutationContractError(
                    "SimulationStateMutationBatch bound_buffer_window must be a "
                    "BoundMutationValueWindow"
                )
            if self.values:
                raise MutationContractError(
                    "SimulationStateMutationBatch cannot mix value descriptors with "
                    "cold-bound buffers"
                )


@dataclass(frozen=True)
class ExternalWrenchMutationBatch:
    values: tuple[MutationValueBatch, ...] = ()

    def __post_init__(self) -> None:
        _validate_values(self.values, self.__class__.__name__)


@dataclass(frozen=True)
class TaskStateMutationBatch:
    values: tuple[MutationValueBatch, ...] = ()

    def __post_init__(self) -> None:
        _validate_values(self.values, self.__class__.__name__)


_SUB_BATCH_KINDS = (
    (ModelParameterMutationBatch, MutationTargetKind.MODEL_PARAMETER),
    (SimulationStateMutationBatch, MutationTargetKind.SIMULATION_STATE),
    (ExternalWrenchMutationBatch, MutationTargetKind.EXTERNAL_WRENCH),
    (TaskStateMutationBatch, MutationTargetKind.TASK_STATE),
)


def _validate_sub_batch(
    batch: Any,
    batch_type: type[Any],
    expected_kind: MutationTargetKind,
    plan: BoundMutationPlan,
    rows: RowSelection,
) -> tuple[int, ...]:
    if not isinstance(batch, batch_type):
        raise MutationContractError(f"mutation envelope requires {batch_type.__name__}")
    if isinstance(batch, SimulationStateMutationBatch) and batch.bound_buffer_window is not None:
        window = batch.bound_buffer_window
        window.plan.require_compatible(plan)
        if window.rows != rows:
            raise MutationContractError(
                "bound mutation value window rows do not match the envelope"
            )
        bound_indices = window.field_indices
        for field_index in bound_indices:
            spec = plan.specs[field_index]
            if spec.target.target_kind is not expected_kind:
                raise MutationContractError(
                    f"mutation term {spec.term_key!r} is in the wrong typed sub-batch"
                )
        return bound_indices
    value_indices: list[int] = []
    for value in batch.values:
        plan.require_compatible(value.plan)
        if value.rows != rows:
            raise MutationContractError("mutation value rows do not match the envelope")
        if value.spec.target.target_kind is not expected_kind:
            raise MutationContractError(
                f"mutation term {value.spec.term_key!r} is in the wrong typed sub-batch"
            )
        value_indices.append(value.field_index)
    if len(set(value_indices)) != len(value_indices):
        raise MutationContractError(f"{batch_type.__name__} contains duplicate mutation fields")
    return tuple(value_indices)


@dataclass(frozen=True)
class TypedBackendMutationBatch:
    plan: BoundMutationPlan
    rows: RowSelection
    model: ModelParameterMutationBatch = field(default_factory=ModelParameterMutationBatch)
    state: SimulationStateMutationBatch = field(default_factory=SimulationStateMutationBatch)
    wrench: ExternalWrenchMutationBatch = field(default_factory=ExternalWrenchMutationBatch)
    task_state: TaskStateMutationBatch = field(default_factory=TaskStateMutationBatch)

    def __post_init__(self) -> None:
        if not isinstance(self.plan, BoundMutationPlan):
            raise MutationContractError("mutation envelope plan must be a BoundMutationPlan")
        if not isinstance(self.rows, RowSelection):
            raise MutationContractError("mutation envelope rows must be a RowSelection")
        if self.rows.universe_size != self.plan.num_envs:
            raise MutationContractError("mutation envelope row universe does not match bound plan")
        all_indices: list[int] = []
        for batch, (batch_type, expected_kind) in zip(
            (self.model, self.state, self.wrench, self.task_state),
            _SUB_BATCH_KINDS,
            strict=True,
        ):
            all_indices.extend(
                _validate_sub_batch(batch, batch_type, expected_kind, self.plan, self.rows)
            )
        if len(set(all_indices)) != len(all_indices):
            raise MutationContractError("mutation field appears in multiple typed sub-batches")

    @property
    def plan_fingerprint(self) -> str:
        return self.plan.fingerprint


@dataclass(frozen=True)
class DeviceResetMutationBatch:
    """One all-world device reset envelope with a CUDA-active row mask.

    The public batch API keeps :class:`RowSelection` host-visible and therefore
    cannot represent a dynamically computed CUDA list of reset rows without a
    device-to-host synchronization.  This envelope deliberately uses
    ``RowSelection.all`` for descriptor shape/ownership and carries the exact
    selected rows as an explicit CUDA bool mask.  The backend applies state
    values only where that mask is true and returns an all-world reset state;
    no Python index extraction is permitted on the hot path.

    Simulation-state and Model-parameter values may share this lifecycle
    barrier.  They must use the same CUDA placement, producer lease, epoch,
    and completion event so a backend can commit both categories before one
    reset forward without extracting selected rows on the host.  Wrench and
    task-state scheduling remain separate lifecycle contracts.
    """

    plan: BoundMutationPlan
    rows: RowSelection
    mutation: TypedBackendMutationBatch
    active_mask: BufferView

    def __post_init__(self) -> None:
        if not isinstance(self.plan, BoundMutationPlan):
            raise MutationContractError("device reset mutation plan must be a BoundMutationPlan")
        if not isinstance(self.rows, RowSelection):
            raise MutationContractError("device reset rows must be a RowSelection")
        if self.rows.universe_size != self.plan.num_envs or not self.rows.is_all:
            raise MutationContractError(
                "device reset mutation requires RowSelection.all for its bound row universe"
            )
        if not isinstance(self.mutation, TypedBackendMutationBatch):
            raise MutationContractError(
                "device reset mutation requires a TypedBackendMutationBatch"
            )
        self.plan.require_compatible(self.mutation.plan)
        if self.mutation.rows != self.rows:
            raise MutationContractError("device reset mutation rows differ from its envelope")
        if self.mutation.wrench.values or self.mutation.task_state.values:
            raise MutationContractError(
                "device reset mutation only supports simulation-state and Model values"
            )
        if self.mutation.state.bound_buffer_window is not None:
            raise MutationContractError(
                "device reset mutation does not support cold-bound host state buffers"
            )
        reset_values = (*self.mutation.model.values, *self.mutation.state.values)
        if not reset_values:
            raise MutationContractError(
                "device reset mutation requires at least one simulation-state or Model value"
            )
        if not isinstance(self.active_mask, BufferView):
            raise MutationContractError("device reset active mask must be a BufferView")
        if self.active_mask.shape != (self.plan.num_envs,):
            raise MutationContractError(
                "device reset active mask must have one boolean entry per bound world"
            )
        contract = self.active_mask.contract
        if (
            contract.row_shape != ()
            or contract.dtype != "bool"
            or contract.placement.memory_space is not MemorySpace.DEVICE
            or contract.placement.device_type != "cuda"
            or contract.owner is not BufferOwner.MANAGER
            or contract.mutability is not BufferMutability.READ_ONLY
            or contract.lifetime is not BufferLifetime.UNTIL_COMMIT
            or not contract.dlpack_exportable
            or not contract.address_stable
        ):
            raise MutationContractError(
                "device reset active mask must be a stable manager-owned CUDA bool commit buffer"
            )
        mask = require_device_tensor_view(
            self.active_mask.handle,
            contract=contract,
            require_completion=True,
        )
        completion = mask.require_completion()
        for value in reset_values:
            value_contract = value.spec.value_buffer
            if value_contract.placement != contract.placement:
                raise MutationContractError(
                    "device reset values and active mask must use one CUDA placement"
                )
            device_value = require_device_tensor_view(
                value.buffer.handle,
                contract=value_contract,
                require_completion=True,
            )
            value_completion = device_value.require_completion()
            if (
                value_completion.placement != completion.placement
                or value_completion.owner_id != completion.owner_id
                or value_completion.epoch != completion.epoch
                or value_completion.event is not completion.event
            ):
                raise MutationContractError(
                    "device reset values must share the active-mask completion event"
                )

    @property
    def plan_fingerprint(self) -> str:
        return self.plan.fingerprint

    @property
    def completion(self) -> DeviceCompletion:
        """Return the single event after every reset value and mask is ready."""

        mask = self.active_mask.handle
        assert isinstance(mask, DeviceTensorView)  # validated in __post_init__.
        return mask.require_completion()


__all__ = [
    "BoundMutationValueBufferGroup",
    "BoundMutationValueBuffers",
    "BoundMutationValueWindow",
    "DeviceResetMutationBatch",
    "ExternalWrenchMutationBatch",
    "ModelParameterMutationBatch",
    "MutationValueBatch",
    "SimulationStateMutationBatch",
    "TaskStateMutationBatch",
    "TypedBackendMutationBatch",
]
