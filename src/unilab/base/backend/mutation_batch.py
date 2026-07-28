"""Runtime batch envelopes for a cold-path bound mutation plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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

    def __post_init__(self) -> None:
        _validate_values(self.values, self.__class__.__name__)


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
    indices: list[int] = []
    for value in batch.values:
        plan.require_compatible(value.plan)
        if value.rows != rows:
            raise MutationContractError("mutation value rows do not match the envelope")
        if value.spec.target.target_kind is not expected_kind:
            raise MutationContractError(
                f"mutation term {value.spec.term_key!r} is in the wrong typed sub-batch"
            )
        indices.append(value.field_index)
    if len(set(indices)) != len(indices):
        raise MutationContractError(f"{batch_type.__name__} contains duplicate mutation fields")
    return tuple(indices)


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
    "ExternalWrenchMutationBatch",
    "ModelParameterMutationBatch",
    "MutationValueBatch",
    "SimulationStateMutationBatch",
    "TaskStateMutationBatch",
    "TypedBackendMutationBatch",
]
