"""Typed batch contracts shared by simulation backends and managed tasks."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import Any, Protocol

import numpy as np

BACKEND_BATCH_CONTRACT_VERSION = "backend-batch-contract-v1"


class BackendBatchContractError(ValueError):
    """Raised when a typed backend batch violates its bound contract."""


class StaleStateBatchError(RuntimeError):
    """Raised when a borrowed state view outlives a backend mutation barrier."""


class MemorySpace(str, Enum):
    HOST = "host"
    DEVICE = "device"


class ExecutionProfile(str, Enum):
    HOST_NUMPY = "host_numpy"


class ControlImplementation(str, Enum):
    CONTROL_STEP_CONSTANT = "control_step_constant"
    HOST_SUBSTEP_CALLBACK = "host_substep_callback"


class BufferOwner(str, Enum):
    BACKEND = "backend"
    RUNTIME = "runtime"
    MANAGER = "manager"
    RUNNER = "runner"


class BufferMutability(str, Enum):
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"


class BufferLifetime(str, Enum):
    """Validity horizon for a typed buffer view.

    ``BORROWED_UNTIL_MUTATION`` is a lease, not a promise that read-only
    materialization preserves prior views. A backend may treat every later
    materialization as a lease barrier so that it can safely reuse scratch
    storage; consumers must regard the previously borrowed view as stale.
    """

    BORROWED_UNTIL_MUTATION = "borrowed_until_mutation"
    UNTIL_STEP_COMPLETE = "until_step_complete"
    UNTIL_COMMIT = "until_commit"
    PLAN = "plan"


class BufferLayout(str, Enum):
    C_CONTIGUOUS = "c_contiguous"


class ReferenceFrame(str, Enum):
    NONE = "none"
    WORLD = "world"
    BASE = "base"
    BODY = "body"
    SENSOR = "sensor"
    JOINT = "joint"


class StateEntityKind(str, Enum):
    ROOT = "root"
    JOINT = "joint"
    DOF = "dof"
    SENSOR = "sensor"
    BODY = "body"
    SITE = "site"


class StateFieldKind(str, Enum):
    POSITION = "position"
    ORIENTATION = "orientation"
    LINEAR_VELOCITY = "linear_velocity"
    ANGULAR_VELOCITY = "angular_velocity"
    VALUE = "value"


class PhysicalUnit(str, Enum):
    UNITLESS = "1"
    METER = "m"
    METER_PER_SECOND = "m/s"
    METER_PER_SECOND_SQUARED = "m/s^2"
    RADIAN = "rad"
    RADIAN_PER_SECOND = "rad/s"
    RADIAN_PER_SECOND_SQUARED = "rad/s^2"
    NEWTON = "N"
    NEWTON_METER = "N*m"
    KILOGRAM = "kg"
    KILOGRAM_METER_SQUARED = "kg*m^2"
    SECOND = "s"
    QUATERNION = "quaternion"


class StateBatchPhase(str, Enum):
    CURRENT = "current"
    TERMINAL = "terminal"
    RESET = "reset"


def _non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BackendBatchContractError(f"{name} must be a non-empty string")
    return value.strip()


def _enum(value: Any, enum_type: type[Enum], name: str) -> None:
    if not isinstance(value, enum_type):
        choices = ", ".join(item.value for item in enum_type)
        raise BackendBatchContractError(f"{name} must be one of: {choices}")


def _shape(value: tuple[int, ...], name: str) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise BackendBatchContractError(f"{name} must be a tuple")
    if any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in value):
        raise BackendBatchContractError(f"{name} dimensions must be positive integers")
    return value


def _count(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BackendBatchContractError(f"{name} must be an integer >= {minimum}")
    return value


_BACKEND_BATCH_COUNTER_FIELDS = (
    "host_to_device_transfers",
    "device_to_host_transfers",
    "host_to_device_bytes",
    "device_to_host_bytes",
    "global_synchronizations",
    "allocations",
    "state_materializations",
    "dynamic_getter_calls",
    "selector_resolutions",
    "asset_metadata_reads",
    "registry_lookups",
)


@dataclass(frozen=True)
class BufferPlacement:
    """Explicit memory-space identity; it is never inferred from a buffer handle."""

    memory_space: MemorySpace
    device_type: str
    device_index: int | None = None

    def __post_init__(self) -> None:
        _enum(self.memory_space, MemorySpace, "memory_space")
        device_type = _non_empty(self.device_type, "device_type").lower()
        object.__setattr__(self, "device_type", device_type)
        if self.memory_space is MemorySpace.HOST:
            if device_type != "cpu" or self.device_index is not None:
                raise BackendBatchContractError(
                    "host placement requires device_type='cpu' and no device_index"
                )
            return
        if device_type == "cpu":
            raise BackendBatchContractError("device placement cannot use device_type='cpu'")
        if self.device_index is None:
            raise BackendBatchContractError("device placement requires a device_index")
        _count(self.device_index, "device_index")

    @classmethod
    def host(cls) -> BufferPlacement:
        return cls(memory_space=MemorySpace.HOST, device_type="cpu")

    @classmethod
    def device(cls, device_type: str, device_index: int) -> BufferPlacement:
        return cls(
            memory_space=MemorySpace.DEVICE,
            device_type=device_type,
            device_index=device_index,
        )


@dataclass(frozen=True)
class BufferContract:
    """Static metadata for one row-major or backend-native batch buffer."""

    row_shape: tuple[int, ...]
    dtype: str
    layout: BufferLayout
    placement: BufferPlacement
    owner: BufferOwner
    mutability: BufferMutability
    lifetime: BufferLifetime
    dlpack_exportable: bool
    address_stable: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "row_shape", _shape(self.row_shape, "row_shape"))
        try:
            dtype = np.dtype(_non_empty(self.dtype, "dtype")).name
        except TypeError as exc:
            raise BackendBatchContractError(f"invalid dtype {self.dtype!r}") from exc
        object.__setattr__(self, "dtype", dtype)
        _enum(self.layout, BufferLayout, "layout")
        if not isinstance(self.placement, BufferPlacement):
            raise BackendBatchContractError("placement must be a BufferPlacement")
        _enum(self.owner, BufferOwner, "owner")
        _enum(self.mutability, BufferMutability, "mutability")
        _enum(self.lifetime, BufferLifetime, "lifetime")
        if not isinstance(self.dlpack_exportable, bool):
            raise BackendBatchContractError("dlpack_exportable must be a bool")
        if not isinstance(self.address_stable, bool):
            raise BackendBatchContractError("address_stable must be a bool")


@dataclass(frozen=True)
class BoundFieldIdentity:
    """Selector-free backend field identity resolved on the cold path."""

    entity_kind: StateEntityKind
    field_kind: StateFieldKind
    entity_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _enum(self.entity_kind, StateEntityKind, "entity_kind")
        _enum(self.field_kind, StateFieldKind, "field_kind")
        if not isinstance(self.entity_ids, tuple):
            raise BackendBatchContractError("entity_ids must be a tuple of cold-path bound ids")
        if not self.entity_ids:
            raise BackendBatchContractError("entity_ids must contain at least one bound id")
        if any(
            isinstance(entity_id, bool) or not isinstance(entity_id, int) or entity_id < 0
            for entity_id in self.entity_ids
        ):
            raise BackendBatchContractError("entity_ids must contain non-negative integers")
        if len(set(self.entity_ids)) != len(self.entity_ids):
            raise BackendBatchContractError("entity_ids must not contain duplicates")


@dataclass(frozen=True)
class StateFieldSpec:
    semantic_key: str
    identity: BoundFieldIdentity
    frame: ReferenceFrame
    unit: PhysicalUnit
    buffer: BufferContract

    def __post_init__(self) -> None:
        object.__setattr__(self, "semantic_key", _non_empty(self.semantic_key, "semantic_key"))
        if not isinstance(self.identity, BoundFieldIdentity):
            raise BackendBatchContractError("identity must be a BoundFieldIdentity")
        _enum(self.frame, ReferenceFrame, "frame")
        _enum(self.unit, PhysicalUnit, "unit")
        if not isinstance(self.buffer, BufferContract):
            raise BackendBatchContractError("buffer must be a BufferContract")
        if self.buffer.owner is not BufferOwner.BACKEND:
            raise BackendBatchContractError("state buffers must be backend-owned")
        if self.buffer.mutability is not BufferMutability.READ_ONLY:
            raise BackendBatchContractError("state buffers must be read-only")
        if self.buffer.lifetime is not BufferLifetime.BORROWED_UNTIL_MUTATION:
            raise BackendBatchContractError(
                "state buffers must use borrowed_until_mutation lifetime"
            )
        if not self.buffer.address_stable:
            raise BackendBatchContractError("state buffer addresses must be plan-stable")

    @property
    def key(self) -> str:
        return self.semantic_key


@dataclass(frozen=True)
class ControlSpec:
    semantic_key: str
    buffer: BufferContract
    physics_substeps_per_control: int = 1
    implementation: ControlImplementation = ControlImplementation.CONTROL_STEP_CONSTANT

    def __post_init__(self) -> None:
        object.__setattr__(self, "semantic_key", _non_empty(self.semantic_key, "semantic_key"))
        if not isinstance(self.buffer, BufferContract):
            raise BackendBatchContractError("buffer must be a BufferContract")
        if self.buffer.owner not in {BufferOwner.MANAGER, BufferOwner.RUNNER}:
            raise BackendBatchContractError("control buffers must be manager- or runner-owned")
        if self.buffer.mutability is not BufferMutability.READ_ONLY:
            raise BackendBatchContractError("control buffers must be read-only to the backend")
        if self.buffer.lifetime is not BufferLifetime.UNTIL_STEP_COMPLETE:
            raise BackendBatchContractError("control buffers must live until step completion")
        if not self.buffer.address_stable:
            raise BackendBatchContractError("control buffer addresses must be plan-stable")
        _count(
            self.physics_substeps_per_control,
            "physics_substeps_per_control",
            minimum=1,
        )
        _enum(self.implementation, ControlImplementation, "control implementation")


def _validate_control_requirements(
    execution_profile: ExecutionProfile,
    fields: tuple[StateFieldSpec, ...],
    control: ControlSpec,
) -> None:
    del execution_profile, fields
    if control.implementation is ControlImplementation.HOST_SUBSTEP_CALLBACK:
        raise BackendBatchContractError("managed typed batch plans reject host substep callbacks")


def _validate_profile(
    execution_profile: ExecutionProfile,
    fields: tuple[StateFieldSpec, ...],
    control: ControlSpec,
) -> None:
    _enum(execution_profile, ExecutionProfile, "execution_profile")
    expected_space = MemorySpace.HOST
    buffers = [field.buffer for field in fields]
    buffers.append(control.buffer)
    if any(buffer.placement.memory_space is not expected_space for buffer in buffers):
        raise BackendBatchContractError(
            f"{execution_profile.value} requires every state/control buffer in "
            f"{expected_space.value} memory"
        )
    placements = {buffer.placement for buffer in buffers}
    if len(placements) != 1:
        raise BackendBatchContractError(
            f"{execution_profile.value} requires one shared state/control placement"
        )


def _validate_state_profile(
    execution_profile: ExecutionProfile,
    fields: tuple[StateFieldSpec, ...],
) -> None:
    _enum(execution_profile, ExecutionProfile, "execution_profile")
    expected_space = MemorySpace.HOST
    if any(field.buffer.placement.memory_space is not expected_space for field in fields):
        raise BackendBatchContractError(
            f"{execution_profile.value} requires every state buffer in "
            f"{expected_space.value} memory"
        )
    if len({field.buffer.placement for field in fields}) != 1:
        raise BackendBatchContractError(
            f"{execution_profile.value} requires one shared state placement"
        )


def _validate_fields(fields: tuple[StateFieldSpec, ...]) -> None:
    if not isinstance(fields, tuple) or not fields:
        raise BackendBatchContractError("state_fields must be a non-empty tuple")
    if any(not isinstance(spec, StateFieldSpec) for spec in fields):
        raise BackendBatchContractError("state_fields must contain only StateFieldSpec values")
    keys = [spec.key for spec in fields]
    if len(set(keys)) != len(keys):
        raise BackendBatchContractError("state field semantic keys must be unique")
    bound_fields = [(spec.identity, spec.frame) for spec in fields]
    if len(set(bound_fields)) != len(bound_fields):
        raise BackendBatchContractError("bound state field identities and frames must be unique")


@dataclass(frozen=True)
class BackendIORequirements:
    state_fields: tuple[StateFieldSpec, ...]
    control: ControlSpec
    execution_profile: ExecutionProfile
    contract_version: str = BACKEND_BATCH_CONTRACT_VERSION
    hot_path_budget: BackendBatchCounterBudget | None = None
    reset_hot_path_budget: BackendBatchCounterBudget | None = None

    def __post_init__(self) -> None:
        _validate_fields(self.state_fields)
        if not isinstance(self.control, ControlSpec):
            raise BackendBatchContractError("control must be a ControlSpec")
        _validate_profile(self.execution_profile, self.state_fields, self.control)
        _validate_control_requirements(
            self.execution_profile,
            self.state_fields,
            self.control,
        )
        for name, budget in (
            ("hot_path_budget", self.hot_path_budget),
            ("reset_hot_path_budget", self.reset_hot_path_budget),
        ):
            if budget is not None and not isinstance(budget, BackendBatchCounterBudget):
                raise BackendBatchContractError(
                    f"{name} must be a BackendBatchCounterBudget or None"
                )
        if self.contract_version != BACKEND_BATCH_CONTRACT_VERSION:
            raise BackendBatchContractError(
                f"unsupported backend batch contract version {self.contract_version!r}"
            )


@dataclass(frozen=True)
class BoundStatePlan:
    backend_type: str
    backend_instance_id: str
    num_envs: int
    fields: tuple[StateFieldSpec, ...]
    execution_profile: ExecutionProfile
    fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend_type", _non_empty(self.backend_type, "backend_type"))
        object.__setattr__(
            self,
            "backend_instance_id",
            _non_empty(self.backend_instance_id, "backend_instance_id"),
        )
        _count(self.num_envs, "num_envs", minimum=1)
        _validate_fields(self.fields)
        _validate_state_profile(self.execution_profile, self.fields)
        object.__setattr__(self, "fingerprint", _non_empty(self.fingerprint, "fingerprint"))

    def field_index(self, semantic_key: str) -> int:
        """Cold-path/diagnostic lookup; compiled hot paths retain the returned index."""
        for index, spec in enumerate(self.fields):
            if spec.key == semantic_key:
                return index
        raise BackendBatchContractError(f"state field {semantic_key!r} is not bound")


@dataclass(frozen=True)
class BoundBackendPlan:
    state: BoundStatePlan
    control: ControlSpec
    execution_profile: ExecutionProfile
    fingerprint: str
    contract_version: str = BACKEND_BATCH_CONTRACT_VERSION
    hot_path_budget: BackendBatchCounterBudget | None = None
    reset_hot_path_budget: BackendBatchCounterBudget | None = None
    reset_requires_mutation_batch: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.state, BoundStatePlan):
            raise BackendBatchContractError("state must be a BoundStatePlan")
        if not isinstance(self.control, ControlSpec):
            raise BackendBatchContractError("control must be a ControlSpec")
        _enum(self.execution_profile, ExecutionProfile, "execution_profile")
        if self.state.execution_profile is not self.execution_profile:
            raise BackendBatchContractError("state and backend execution profiles must match")
        _validate_profile(self.execution_profile, self.state.fields, self.control)
        _validate_control_requirements(self.execution_profile, self.state.fields, self.control)
        object.__setattr__(self, "fingerprint", _non_empty(self.fingerprint, "fingerprint"))
        for name, budget in (
            ("hot_path_budget", self.hot_path_budget),
            ("reset_hot_path_budget", self.reset_hot_path_budget),
        ):
            if budget is not None and not isinstance(budget, BackendBatchCounterBudget):
                raise BackendBatchContractError(
                    f"{name} must be a BackendBatchCounterBudget or None"
                )
        if not isinstance(self.reset_requires_mutation_batch, bool):
            raise BackendBatchContractError("reset_requires_mutation_batch must be a bool")
        if self.contract_version != BACKEND_BATCH_CONTRACT_VERSION:
            raise BackendBatchContractError(
                f"unsupported backend batch contract version {self.contract_version!r}"
            )

    @property
    def backend_type(self) -> str:
        return self.state.backend_type

    @property
    def backend_instance_id(self) -> str:
        return self.state.backend_instance_id

    @property
    def num_envs(self) -> int:
        return self.state.num_envs

    def require_owner(self, *, backend_type: str, backend_instance_id: str) -> None:
        if self.backend_type != backend_type or self.backend_instance_id != backend_instance_id:
            raise BackendBatchContractError(
                "bound plan belongs to a different backend type or instance"
            )

    def require_compatible(self, other: BoundBackendPlan) -> None:
        if not isinstance(other, BoundBackendPlan):
            raise BackendBatchContractError("batch plan must be a BoundBackendPlan")
        if self != other:
            raise BackendBatchContractError(
                "batch was built from a different backend plan or fingerprint"
            )


@dataclass(frozen=True)
class RowSelection:
    """Explicit row mapping. Selected row order is preserved and semantically significant."""

    universe_size: int
    indices: tuple[int, ...] | None

    def __post_init__(self) -> None:
        _count(self.universe_size, "universe_size", minimum=1)
        if self.indices is None:
            return
        if not isinstance(self.indices, tuple):
            raise BackendBatchContractError("selected row indices must be a tuple")
        if not self.indices:
            raise BackendBatchContractError("selected row indices must not be empty")
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= self.universe_size
            for index in self.indices
        ):
            raise BackendBatchContractError("selected row index is outside the row universe")
        if len(set(self.indices)) != len(self.indices):
            raise BackendBatchContractError("selected row indices must be unique")

    @classmethod
    def all(cls, universe_size: int) -> RowSelection:
        return cls(universe_size=universe_size, indices=None)

    @classmethod
    def selected(cls, universe_size: int, indices: tuple[int, ...]) -> RowSelection:
        return cls(universe_size=universe_size, indices=indices)

    @property
    def count(self) -> int:
        return self.universe_size if self.indices is None else len(self.indices)

    @property
    def is_all(self) -> bool:
        return self.indices is None


@dataclass(frozen=True)
class BufferView:
    """Backend-asserted descriptor around an opaque host or device buffer handle."""

    handle: object = field(repr=False, compare=False)
    shape: tuple[int, ...]
    contract: BufferContract

    def __post_init__(self) -> None:
        if self.handle is None:
            raise BackendBatchContractError("buffer handle cannot be None")
        object.__setattr__(self, "shape", _shape(self.shape, "buffer shape"))
        if not isinstance(self.contract, BufferContract):
            raise BackendBatchContractError("buffer view contract must be a BufferContract")


class StateBatchLease:
    """Backend-owned generation clock used to invalidate borrowed state views."""

    __slots__ = ("_generation", "owner_id")

    def __init__(self, owner_id: str) -> None:
        self.owner_id = _non_empty(owner_id, "lease owner_id")
        self._generation = 0

    @property
    def generation(self) -> int:
        return self._generation

    def invalidate(self) -> None:
        self._generation += 1

    def assert_valid(self, generation: int) -> None:
        if generation != self._generation:
            raise StaleStateBatchError(
                "state batch is stale because the backend crossed a mutation barrier"
            )


class BorrowedBufferView:
    """A buffer view whose metadata and handle access are guarded by a state lease."""

    __slots__ = ("_generation", "_lease", "_view")

    def __init__(
        self,
        view: BufferView,
        lease: StateBatchLease,
        generation: int,
    ) -> None:
        self._view = view
        self._lease = lease
        self._generation = generation

    def _assert_valid(self) -> None:
        self._lease.assert_valid(self._generation)

    @property
    def handle(self) -> object:
        self._assert_valid()
        return self._view.handle

    @property
    def shape(self) -> tuple[int, ...]:
        self._assert_valid()
        return self._view.shape

    @property
    def contract(self) -> BufferContract:
        self._assert_valid()
        return self._view.contract


@dataclass(frozen=True, eq=False)
class StateBatch:
    plan: BoundBackendPlan
    rows: RowSelection
    phase: StateBatchPhase
    descriptors: InitVar[tuple[BufferView, ...]]
    lease: InitVar[StateBatchLease]
    _generation: int = field(init=False, repr=False)
    _lease: StateBatchLease = field(init=False, repr=False)
    _borrowed_buffers: tuple[BorrowedBufferView, ...] = field(init=False, repr=False)

    def __post_init__(
        self,
        descriptors: tuple[BufferView, ...],
        lease: StateBatchLease,
    ) -> None:
        if not isinstance(self.plan, BoundBackendPlan):
            raise BackendBatchContractError("plan must be a BoundBackendPlan")
        if not isinstance(self.rows, RowSelection):
            raise BackendBatchContractError("rows must be a RowSelection")
        if self.rows.universe_size != self.plan.num_envs:
            raise BackendBatchContractError("row universe does not match bound plan num_envs")
        _enum(self.phase, StateBatchPhase, "state batch phase")
        if not isinstance(descriptors, tuple):
            raise BackendBatchContractError("state buffers must be a tuple")
        if len(descriptors) != len(self.plan.state.fields):
            raise BackendBatchContractError("state buffer count does not match bound state fields")
        if not isinstance(lease, StateBatchLease):
            raise BackendBatchContractError("lease must be a StateBatchLease")
        if lease.owner_id != self.plan.backend_instance_id:
            raise BackendBatchContractError("state lease belongs to a different backend instance")
        for spec, view in zip(self.plan.state.fields, descriptors, strict=True):
            if not isinstance(view, BufferView):
                raise BackendBatchContractError("state buffers must contain BufferView values")
            if view.contract != spec.buffer:
                raise BackendBatchContractError(
                    f"state buffer metadata does not match field {spec.key!r}"
                )
            expected_shape = (self.rows.count, *spec.buffer.row_shape)
            if view.shape != expected_shape:
                raise BackendBatchContractError(
                    f"state field {spec.key!r} requires shape {expected_shape}, got {view.shape}"
                )
        generation = lease.generation
        object.__setattr__(self, "_generation", generation)
        object.__setattr__(self, "_lease", lease)
        object.__setattr__(
            self,
            "_borrowed_buffers",
            tuple(BorrowedBufferView(view, lease, generation) for view in descriptors),
        )

    def assert_valid(self) -> None:
        self._lease.assert_valid(self._generation)

    def buffer_at(self, field_index: int) -> BorrowedBufferView:
        self.assert_valid()
        if isinstance(field_index, bool) or not isinstance(field_index, int):
            raise BackendBatchContractError("field_index must be an integer")
        if field_index < 0:
            raise BackendBatchContractError(f"field index {field_index} is not bound")
        try:
            return self._borrowed_buffers[field_index]
        except IndexError as exc:
            raise BackendBatchContractError(f"field index {field_index} is not bound") from exc

    def buffer(self, semantic_key: str) -> BorrowedBufferView:
        return self.buffer_at(self.plan.state.field_index(semantic_key))


@dataclass(frozen=True)
class ControlBatch:
    plan: BoundBackendPlan
    rows: RowSelection
    buffer: BufferView

    def __post_init__(self) -> None:
        if not isinstance(self.plan, BoundBackendPlan):
            raise BackendBatchContractError("plan must be a BoundBackendPlan")
        if not isinstance(self.rows, RowSelection):
            raise BackendBatchContractError("rows must be a RowSelection")
        if self.rows.universe_size != self.plan.num_envs:
            raise BackendBatchContractError("row universe does not match bound plan num_envs")
        if not isinstance(self.buffer, BufferView):
            raise BackendBatchContractError("control buffer must be a BufferView")
        if self.buffer.contract != self.plan.control.buffer:
            raise BackendBatchContractError("control buffer metadata does not match bound plan")
        expected_shape = (self.rows.count, *self.plan.control.buffer.row_shape)
        if self.buffer.shape != expected_shape:
            raise BackendBatchContractError(
                f"control buffer requires shape {expected_shape}, got {self.buffer.shape}"
            )


class BackendMutationBatch(Protocol):
    """Typed extension point implemented by the Phase 1 mutation contract."""

    @property
    def plan_fingerprint(self) -> str: ...

    @property
    def rows(self) -> RowSelection: ...


@dataclass(frozen=True)
class BackendBatchCounters:
    """Per-barrier counters for runtime-owned numeric work and dynamic access.

    ``allocations`` counts numeric storage or scratch allocated after binding. It
    intentionally excludes Python result/descriptor wrapper objects, which need
    a separate allocation-stability profiler at the executor/runtime layer.
    Dynamic counters cover legacy fine-grained getters, selector resolution,
    asset/model metadata I/O, and term/backend registry lookup; a backend's
    lookup of an already-bound plan handle is not a registry lookup.
    """

    host_to_device_transfers: int = 0
    device_to_host_transfers: int = 0
    host_to_device_bytes: int = 0
    device_to_host_bytes: int = 0
    global_synchronizations: int = 0
    allocations: int = 0
    state_materializations: int = 0
    dynamic_getter_calls: int = 0
    selector_resolutions: int = 0
    asset_metadata_reads: int = 0
    registry_lookups: int = 0
    instrumentation_complete: bool = False

    def __post_init__(self) -> None:
        for name in _BACKEND_BATCH_COUNTER_FIELDS:
            _count(getattr(self, name), name)
        if not isinstance(self.instrumentation_complete, bool):
            raise BackendBatchContractError("instrumentation_complete must be a bool")

    def require_within(self, budget: BackendBatchCounterBudget) -> None:
        if not isinstance(budget, BackendBatchCounterBudget):
            raise BackendBatchContractError("budget must be a BackendBatchCounterBudget")
        if not self.instrumentation_complete:
            raise BackendHotPathViolationError(
                instrumentation_complete=False,
                violations=(),
            )
        violations = tuple(
            BackendCounterViolation(
                counter=name,
                actual=getattr(self, name),
                limit=getattr(budget, name),
            )
            for name in _BACKEND_BATCH_COUNTER_FIELDS
            if getattr(self, name) > getattr(budget, name)
        )
        if violations:
            raise BackendHotPathViolationError(
                instrumentation_complete=True,
                violations=violations,
            )


@dataclass(frozen=True)
class BackendBatchCounterBudget:
    """Maximum permitted per-barrier counts for one bound managed path."""

    host_to_device_transfers: int = 0
    device_to_host_transfers: int = 0
    host_to_device_bytes: int = 0
    device_to_host_bytes: int = 0
    global_synchronizations: int = 0
    allocations: int = 0
    state_materializations: int = 0
    dynamic_getter_calls: int = 0
    selector_resolutions: int = 0
    asset_metadata_reads: int = 0
    registry_lookups: int = 0

    def __post_init__(self) -> None:
        for name in _BACKEND_BATCH_COUNTER_FIELDS:
            _count(getattr(self, name), name)

    def items(self) -> tuple[tuple[str, int], ...]:
        """Return canonical cold-path serialization entries."""
        return tuple((name, getattr(self, name)) for name in _BACKEND_BATCH_COUNTER_FIELDS)


@dataclass(frozen=True)
class BackendCounterViolation:
    counter: str
    actual: int
    limit: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "counter", _non_empty(self.counter, "counter"))
        if self.counter not in _BACKEND_BATCH_COUNTER_FIELDS:
            raise BackendBatchContractError(f"unknown backend counter {self.counter!r}")
        _count(self.actual, "actual")
        _count(self.limit, "limit")
        if self.actual <= self.limit:
            raise BackendBatchContractError("counter violation actual value must exceed its limit")


class BackendHotPathViolationError(BackendBatchContractError):
    def __init__(
        self,
        *,
        instrumentation_complete: bool,
        violations: tuple[BackendCounterViolation, ...],
    ) -> None:
        if not isinstance(instrumentation_complete, bool):
            raise BackendBatchContractError("instrumentation_complete must be a bool")
        if not isinstance(violations, tuple) or any(
            not isinstance(violation, BackendCounterViolation) for violation in violations
        ):
            raise BackendBatchContractError(
                "violations must be a tuple of BackendCounterViolation values"
            )
        names = tuple(violation.counter for violation in violations)
        canonical_names = tuple(name for name in _BACKEND_BATCH_COUNTER_FIELDS if name in names)
        if names != canonical_names:
            raise BackendBatchContractError(
                "counter violations must be unique and in canonical order"
            )
        self.instrumentation_complete = instrumentation_complete
        self.violations = violations
        if not instrumentation_complete:
            if violations:
                raise BackendBatchContractError(
                    "incomplete instrumentation cannot report counter violations"
                )
            message = "managed hot-path instrumentation is incomplete"
        else:
            if not violations:
                raise BackendBatchContractError(
                    "complete instrumentation errors require counter violations"
                )
            details = "; ".join(
                f"{item.counter}: actual={item.actual}, limit={item.limit}" for item in violations
            )
            message = f"managed hot-path counter budget exceeded: {details}"
        super().__init__(message)


@dataclass(frozen=True)
class BackendTiming:
    phase: str
    milliseconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", _non_empty(self.phase, "timing phase"))
        if isinstance(self.milliseconds, bool) or not isinstance(self.milliseconds, (int, float)):
            raise BackendBatchContractError("timing milliseconds must be numeric")
        milliseconds = float(self.milliseconds)
        if not np.isfinite(milliseconds) or milliseconds < 0:
            raise BackendBatchContractError("timing milliseconds must be finite and non-negative")
        object.__setattr__(self, "milliseconds", milliseconds)


@dataclass(frozen=True)
class BackendBatchDiagnostics:
    counters: BackendBatchCounters = field(default_factory=BackendBatchCounters)
    timings: tuple[BackendTiming, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.counters, BackendBatchCounters):
            raise BackendBatchContractError("counters must be BackendBatchCounters")
        if not isinstance(self.timings, tuple) or any(
            not isinstance(timing, BackendTiming) for timing in self.timings
        ):
            raise BackendBatchContractError("timings must contain BackendTiming values")
        phases = [timing.phase for timing in self.timings]
        if len(set(phases)) != len(phases):
            raise BackendBatchContractError("timing phases must be unique")


@dataclass(frozen=True)
class BackendReadResult:
    state: StateBatch
    diagnostics: BackendBatchDiagnostics = field(default_factory=BackendBatchDiagnostics)

    def __post_init__(self) -> None:
        if not isinstance(self.state, StateBatch):
            raise BackendBatchContractError("state must be a StateBatch")
        self.state.assert_valid()
        if not isinstance(self.diagnostics, BackendBatchDiagnostics):
            raise BackendBatchContractError("diagnostics must be BackendBatchDiagnostics")
        _validate_result_budget(self.state, self.diagnostics)


def _validate_result_budget(
    state: StateBatch,
    diagnostics: BackendBatchDiagnostics,
    *,
    reset: bool = False,
) -> None:
    budget = state.plan.hot_path_budget
    if reset and state.plan.reset_hot_path_budget is not None:
        budget = state.plan.reset_hot_path_budget
    if budget is not None:
        diagnostics.counters.require_within(budget)


@dataclass(frozen=True)
class BackendStepResult:
    terminal_state: StateBatch
    diagnostics: BackendBatchDiagnostics = field(default_factory=BackendBatchDiagnostics)

    def __post_init__(self) -> None:
        if not isinstance(self.terminal_state, StateBatch):
            raise BackendBatchContractError("terminal_state must be a StateBatch")
        if self.terminal_state.phase is not StateBatchPhase.TERMINAL:
            raise BackendBatchContractError("step results require terminal state semantics")
        self.terminal_state.assert_valid()
        if not isinstance(self.diagnostics, BackendBatchDiagnostics):
            raise BackendBatchContractError("diagnostics must be BackendBatchDiagnostics")
        _validate_result_budget(self.terminal_state, self.diagnostics)


@dataclass(frozen=True)
class BackendResetResult:
    reset_state: StateBatch
    diagnostics: BackendBatchDiagnostics = field(default_factory=BackendBatchDiagnostics)

    def __post_init__(self) -> None:
        if not isinstance(self.reset_state, StateBatch):
            raise BackendBatchContractError("reset_state must be a StateBatch")
        if self.reset_state.phase is not StateBatchPhase.RESET:
            raise BackendBatchContractError("reset results require reset state semantics")
        self.reset_state.assert_valid()
        if not isinstance(self.diagnostics, BackendBatchDiagnostics):
            raise BackendBatchContractError("diagnostics must be BackendBatchDiagnostics")
        _validate_result_budget(self.reset_state, self.diagnostics, reset=True)


__all__ = [
    "BACKEND_BATCH_CONTRACT_VERSION",
    "BackendBatchContractError",
    "BackendBatchCounterBudget",
    "BackendBatchCounters",
    "BackendBatchDiagnostics",
    "BackendCounterViolation",
    "BackendHotPathViolationError",
    "BackendIORequirements",
    "BackendMutationBatch",
    "BackendReadResult",
    "BackendResetResult",
    "BackendStepResult",
    "BackendTiming",
    "BorrowedBufferView",
    "BoundBackendPlan",
    "BoundFieldIdentity",
    "BoundStatePlan",
    "BufferContract",
    "BufferLayout",
    "BufferLifetime",
    "BufferMutability",
    "BufferOwner",
    "BufferPlacement",
    "BufferView",
    "ControlBatch",
    "ControlImplementation",
    "ControlSpec",
    "ExecutionProfile",
    "MemorySpace",
    "PhysicalUnit",
    "ReferenceFrame",
    "RowSelection",
    "StaleStateBatchError",
    "StateBatch",
    "StateBatchLease",
    "StateBatchPhase",
    "StateEntityKind",
    "StateFieldKind",
    "StateFieldSpec",
]
