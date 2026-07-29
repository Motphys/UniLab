"""Runtime batch envelopes for a cold-path bound mutation plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .batch import BufferView, RowSelection
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


__all__ = [
    "BoundMutationValueBuffers",
    "BoundMutationValueWindow",
    "ExternalWrenchMutationBatch",
    "ModelParameterMutationBatch",
    "MutationValueBatch",
    "SimulationStateMutationBatch",
    "TaskStateMutationBatch",
    "TypedBackendMutationBatch",
]
