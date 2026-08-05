"""Reference managed-task lifecycle over the public typed backend contract.

The runtime deliberately owns lifecycle ordering but not task math or backend
storage.  A cold-bound :class:`ManagedTaskKernel` receives only borrowed
``StateBatch`` views and runtime-owned buffers; it never receives an env,
backend, selector, registry, asset, or model object.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np

from unilab.base.backend.base import SimBackend
from unilab.base.backend.batch import (
    BackendBatchContractError,
    BackendBatchCounters,
    BackendBatchDiagnostics,
    BackendMutationBatch,
    BackendResetResult,
    BackendStepResult,
    BoundBackendPlan,
    BufferView,
    ControlBatch,
    ExecutionProfile,
    RowSelection,
    StaleStateBatchError,
    StateBatch,
    StateBatchPhase,
)
from unilab.base.backend.mutation import BoundMutationPlan

from .entities import ManagerContractError
from .fingerprint import managed_policy_abi_snapshot, validate_compiled_plan_fingerprints
from .plan import CompiledTaskPlan


class ManagedRuntimeError(ManagerContractError):
    """Raised when a runtime/kernel/backend lifecycle contract is violated."""


class ManagedLifecyclePhase(str, Enum):
    """Canonical lifecycle events emitted by the optional diagnostic trace."""

    INITIAL_RESET_REQUEST = "initial_reset_request"
    ACTION = "action"
    PRE_PHYSICS = "pre_physics"
    PHYSICS = "physics"
    TERMINATION = "termination"
    REWARD = "reward"
    METRIC = "metric"
    TERMINAL_OBSERVATION = "terminal_observation"
    TIMEOUT = "timeout"
    FINAL_OBSERVATION = "final_observation"
    AUTORESET = "autoreset"
    RESET_REQUEST = "reset_request"
    RESET_BACKEND = "reset_backend"
    TASK_STATE_RESET = "task_state_reset"
    OBSERVATION = "observation"
    COMPLETE = "complete"


@dataclass(frozen=True)
class ManagedLifecycleEvent:
    """One cold-diagnostic lifecycle event; ``rows=None`` means all worlds."""

    phase: ManagedLifecyclePhase
    rows: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.phase, ManagedLifecyclePhase):
            raise ManagedRuntimeError("lifecycle phase must be a ManagedLifecyclePhase")
        if self.rows is not None:
            if not isinstance(self.rows, tuple) or not self.rows:
                raise ManagedRuntimeError("lifecycle event rows must be a non-empty tuple or None")
            if any(
                isinstance(row, bool) or not isinstance(row, int) or row < 0 for row in self.rows
            ):
                raise ManagedRuntimeError("lifecycle event rows must be non-negative integers")
            if len(set(self.rows)) != len(self.rows):
                raise ManagedRuntimeError("lifecycle event rows must be unique")


@dataclass(frozen=True)
class ManagedResetRequest:
    """A typed reset request prepared by a cold-bound task kernel.

    ``kernel_state`` is opaque to the backend and is handed back only to the
    same kernel after the backend has committed its reset barrier.  It lets a
    task carry sampled command/history values without exposing raw physics
    fields to manager code.
    """

    rows: RowSelection
    mutation_batch: BackendMutationBatch | None = None
    kernel_state: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.rows, RowSelection):
            raise ManagedRuntimeError("managed reset rows must be a RowSelection")


@dataclass(frozen=True)
class ManagedMetric:
    """A task metric produced after terminal reward computation."""

    key: str
    value: float | np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ManagedRuntimeError("managed metric key must be a non-empty string")
        if isinstance(self.value, bool) or not isinstance(self.value, (float, np.ndarray)):
            raise ManagedRuntimeError("managed metric value must be a float or numpy array")
        if isinstance(self.value, float) and not np.isfinite(self.value):
            raise ManagedRuntimeError("managed metric float value must be finite")
        if isinstance(self.value, np.ndarray) and not np.isfinite(self.value).all():
            raise ManagedRuntimeError("managed metric array value must be finite")


@dataclass(frozen=True)
class ManagedRuntimeBuffer:
    """One explicitly registered manager/executor-owned numeric buffer.

    This is a diagnostic-only cold/warm instrumentation descriptor.  It makes
    the allocation boundary explicit without exposing a backend buffer or
    counting short-lived Python result wrappers as numeric allocations.  A
    provider must return the same canonical names and ndarray identities for
    the life of a warmed runtime.
    """

    name: str
    array: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ManagedRuntimeError("managed runtime buffer name must be non-empty")
        if not isinstance(self.array, np.ndarray):
            raise ManagedRuntimeError("managed runtime buffer must be a numpy array")
        if self.array.ndim == 0 or not self.array.flags.c_contiguous:
            raise ManagedRuntimeError(
                "managed runtime buffer must be a non-scalar C-contiguous numpy array"
            )


@dataclass(frozen=True)
class ManagedRuntimeBufferAddress:
    """Stable public identity for one registered numeric buffer."""

    name: str
    address: int
    shape: tuple[int, ...]
    dtype: str
    nbytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ManagedRuntimeError("managed runtime buffer address name must be non-empty")
        if isinstance(self.address, bool) or not isinstance(self.address, int) or self.address <= 0:
            raise ManagedRuntimeError("managed runtime buffer address must be positive")
        if (
            not isinstance(self.shape, tuple)
            or not self.shape
            or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in self.shape
            )
        ):
            raise ManagedRuntimeError("managed runtime buffer address shape is invalid")
        try:
            dtype = np.dtype(self.dtype).name
        except TypeError as exc:
            raise ManagedRuntimeError("managed runtime buffer address dtype is invalid") from exc
        object.__setattr__(self, "dtype", dtype)
        if isinstance(self.nbytes, bool) or not isinstance(self.nbytes, int) or self.nbytes <= 0:
            raise ManagedRuntimeError("managed runtime buffer address nbytes must be positive")


@dataclass(frozen=True)
class ManagedRuntimeStabilityDiagnostics:
    """Public warm-path buffer and backend-counter observation snapshot.

    ``warm_numeric_allocations`` counts only registration-visible numeric
    buffer replacement/addition after the post-reset baseline.  It does not
    claim to count Python dataclass/dict/descriptor heap churn; those objects
    are intentionally outside the typed numeric-buffer ownership contract.
    """

    buffers: tuple[ManagedRuntimeBufferAddress, ...]
    state_buffers: tuple[ManagedRuntimeBufferAddress, ...]
    warm_numeric_allocations: int
    address_churn: int
    observations: int
    backend_step_counters: BackendBatchCounters | None
    backend_reset_counters: BackendBatchCounters | None
    instrumentation_complete: bool

    def __post_init__(self) -> None:
        for label, values in (("buffers", self.buffers), ("state_buffers", self.state_buffers)):
            if not isinstance(values, tuple) or any(
                not isinstance(value, ManagedRuntimeBufferAddress) for value in values
            ):
                raise ManagedRuntimeError(f"managed runtime stability {label} is invalid")
            names = tuple(value.name for value in values)
            if names != tuple(sorted(names)) or len(set(names)) != len(names):
                raise ManagedRuntimeError(
                    f"managed runtime stability {label} names must be canonical and unique"
                )
        for label, value in (
            ("warm_numeric_allocations", self.warm_numeric_allocations),
            ("address_churn", self.address_churn),
            ("observations", self.observations),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ManagedRuntimeError(f"managed runtime stability {label} must be non-negative")
        for label, counters in (
            ("backend_step_counters", self.backend_step_counters),
            ("backend_reset_counters", self.backend_reset_counters),
        ):
            if counters is not None and not isinstance(counters, BackendBatchCounters):
                raise ManagedRuntimeError(f"managed runtime stability {label} is invalid")
        if not isinstance(self.instrumentation_complete, bool):
            raise ManagedRuntimeError(
                "managed runtime stability instrumentation_complete must be a bool"
            )


@runtime_checkable
class ManagedRuntimeBufferProvider(Protocol):
    """Explicit task-owned buffer registration for warm-path instrumentation."""

    def managed_runtime_buffers(
        self, *, task_state: object
    ) -> tuple[ManagedRuntimeBuffer, ...]: ...


def _buffer_address(buffer: ManagedRuntimeBuffer) -> ManagedRuntimeBufferAddress:
    """Snapshot one C-contiguous ndarray without retaining its mutable handle."""

    array = buffer.array
    address = int(array.__array_interface__["data"][0])
    return ManagedRuntimeBufferAddress(
        name=buffer.name,
        address=address,
        shape=tuple(int(dim) for dim in array.shape),
        dtype=array.dtype.name,
        nbytes=int(array.nbytes),
    )


class _ManagedRuntimeStabilityMonitor:
    """Fail-closed monitor for registered numeric ownership and state views."""

    def __init__(self, *, require_complete_backend_instrumentation: bool) -> None:
        self._require_complete_backend_instrumentation = require_complete_backend_instrumentation
        self._baseline_buffers: tuple[ManagedRuntimeBufferAddress, ...] | None = None
        self._state_buffers: dict[str, ManagedRuntimeBufferAddress] = {}
        self._warm_numeric_allocations = 0
        self._address_churn = 0
        self._observations = 0
        self._backend_step_counters: BackendBatchCounters | None = None
        self._backend_reset_counters: BackendBatchCounters | None = None

    @staticmethod
    def _canonical(
        buffers: tuple[ManagedRuntimeBuffer, ...], *, context: str
    ) -> tuple[ManagedRuntimeBufferAddress, ...]:
        addresses = tuple(
            sorted((_buffer_address(buffer) for buffer in buffers), key=lambda item: item.name)
        )
        names = tuple(item.name for item in addresses)
        if len(set(names)) != len(names):
            raise ManagedRuntimeError(f"{context} registered duplicate managed runtime buffers")
        return addresses

    def arm(self, buffers: tuple[ManagedRuntimeBuffer, ...]) -> None:
        if self._baseline_buffers is not None:
            raise ManagedRuntimeError("managed runtime stability monitor may only arm once")
        baseline = self._canonical(buffers, context="managed runtime stability baseline")
        if not baseline:
            raise ManagedRuntimeError("managed runtime stability baseline must register buffers")
        self._baseline_buffers = baseline

    def observe_buffers(self, buffers: tuple[ManagedRuntimeBuffer, ...]) -> None:
        if self._baseline_buffers is None:
            raise ManagedRuntimeError("managed runtime stability monitor has not been armed")
        actual = self._canonical(buffers, context="managed runtime stability observation")
        if actual == self._baseline_buffers:
            self._observations += 1
            return
        self._address_churn += 1
        baseline_by_name = {item.name: item for item in self._baseline_buffers}
        actual_by_name = {item.name: item for item in actual}
        added_or_removed = tuple(sorted(set(baseline_by_name) ^ set(actual_by_name)))
        changed = tuple(
            name
            for name in sorted(set(baseline_by_name) & set(actual_by_name))
            if baseline_by_name[name] != actual_by_name[name]
        )
        self._warm_numeric_allocations += len(added_or_removed) + len(changed)
        raise ManagedRuntimeError(
            "managed warm buffer stability violated: "
            f"added_or_removed={added_or_removed!r}, changed={changed!r}"
        )

    def observe_state(self, state: StateBatch) -> None:
        state.assert_valid()
        observed: list[ManagedRuntimeBufferAddress] = []
        for field_index, state_field in enumerate(state.plan.state.fields):
            handle = state.buffer_at(field_index).handle
            if not isinstance(handle, np.ndarray):
                raise ManagedRuntimeError(
                    "host managed stability instrumentation requires numpy StateBatch buffers"
                )
            row_mode = "all" if state.rows.is_all else "selected"
            name = f"{state.phase.value}.{row_mode}.rows={state.rows.count}.{state_field.key}"
            observed.append(_buffer_address(ManagedRuntimeBuffer(name=name, array=handle)))
        for address in observed:
            previous = self._state_buffers.get(address.name)
            if previous is None:
                self._state_buffers[address.name] = address
            elif previous != address:
                self._address_churn += 1
                raise ManagedRuntimeError(
                    f"managed backend StateBatch address changed after warmup: {address.name!r}"
                )

    def record_backend(self, *, phase: str, diagnostics: BackendBatchDiagnostics) -> None:
        if not isinstance(diagnostics, BackendBatchDiagnostics):
            raise ManagedRuntimeError("managed backend diagnostics are invalid")
        counters = diagnostics.counters
        if self._require_complete_backend_instrumentation and not counters.instrumentation_complete:
            raise ManagedRuntimeError(
                "managed warm stability requires complete backend batch instrumentation"
            )
        if phase == "step":
            self._backend_step_counters = counters
        elif phase == "reset":
            self._backend_reset_counters = counters
        else:  # pragma: no cover - internal fixed call sites.
            raise ManagedRuntimeError(f"unknown managed backend diagnostic phase {phase!r}")

    def snapshot(self) -> ManagedRuntimeStabilityDiagnostics:
        if self._baseline_buffers is None:
            raise ManagedRuntimeError("managed runtime stability monitor has not been armed")
        observed_counters = tuple(
            counters
            for counters in (self._backend_step_counters, self._backend_reset_counters)
            if counters is not None
        )
        complete = bool(observed_counters) and all(
            counters.instrumentation_complete for counters in observed_counters
        )
        return ManagedRuntimeStabilityDiagnostics(
            buffers=self._baseline_buffers,
            state_buffers=tuple(sorted(self._state_buffers.values(), key=lambda item: item.name)),
            warm_numeric_allocations=self._warm_numeric_allocations,
            address_churn=self._address_churn,
            observations=self._observations,
            backend_step_counters=self._backend_step_counters,
            backend_reset_counters=self._backend_reset_counters,
            instrumentation_complete=complete,
        )


@dataclass(frozen=True)
class ManagedKernelBinding:
    """Cold-bound public metadata available to an injected task kernel.

    The binding deliberately contains plan-level identities and, when needed,
    the public typed mutation plan.  It never exposes a backend instance,
    env, selector, registry, asset, model, or mutable runtime buffer.
    """

    task_fingerprint: str
    policy_abi_fingerprint: str
    num_envs: int
    dtype: str
    execution_profile: ExecutionProfile
    state_field_indices: tuple[tuple[str, int], ...]
    observation_buffer_indices: tuple[tuple[str, int], ...]
    mutation_plan: BoundMutationPlan | None
    event_mutation_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("task_fingerprint", self.task_fingerprint),
            ("policy_abi_fingerprint", self.policy_abi_fingerprint),
            ("dtype", self.dtype),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ManagedRuntimeError(f"managed kernel binding {name} must be non-empty")
        if (
            isinstance(self.num_envs, bool)
            or not isinstance(self.num_envs, int)
            or self.num_envs <= 0
        ):
            raise ManagedRuntimeError("managed kernel binding num_envs must be positive")
        if not isinstance(self.execution_profile, ExecutionProfile):
            raise ManagedRuntimeError("managed kernel binding execution_profile is invalid")
        if not isinstance(self.state_field_indices, tuple):
            raise ManagedRuntimeError("managed kernel binding state indices must be a tuple")
        expected_indices = tuple(range(len(self.state_field_indices)))
        keys: list[str] = []
        for key, index in self.state_field_indices:
            if not isinstance(key, str) or not key.strip():
                raise ManagedRuntimeError("managed kernel binding state key must be non-empty")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ManagedRuntimeError("managed kernel binding state index must be non-negative")
            keys.append(key)
        if tuple(index for _, index in self.state_field_indices) != expected_indices:
            raise ManagedRuntimeError("managed kernel binding state indices must be contiguous")
        if tuple(keys) != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise ManagedRuntimeError("managed kernel binding state keys must be canonical")
        if (
            not isinstance(self.observation_buffer_indices, tuple)
            or not self.observation_buffer_indices
        ):
            raise ManagedRuntimeError("managed kernel binding requires observation buffers")
        observation_keys: list[str] = []
        for key, index in self.observation_buffer_indices:
            if not isinstance(key, str) or not key.strip():
                raise ManagedRuntimeError(
                    "managed kernel binding observation buffer key is invalid"
                )
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ManagedRuntimeError(
                    "managed kernel binding observation buffer index must be non-negative"
                )
            observation_keys.append(key)
        if tuple(index for _, index in self.observation_buffer_indices) != tuple(
            range(len(self.observation_buffer_indices))
        ):
            raise ManagedRuntimeError(
                "managed kernel binding observation buffer indices must be contiguous"
            )
        if len(set(observation_keys)) != len(observation_keys):
            raise ManagedRuntimeError(
                "managed kernel binding observation buffer keys must be unique"
            )
        if self.mutation_plan is not None and not isinstance(self.mutation_plan, BoundMutationPlan):
            raise ManagedRuntimeError("managed kernel binding mutation_plan is invalid")
        if (
            not isinstance(self.event_mutation_indices, tuple)
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in self.event_mutation_indices
            )
            or self.event_mutation_indices != tuple(sorted(set(self.event_mutation_indices)))
        ):
            raise ManagedRuntimeError(
                "managed kernel binding Event mutation indices must be canonical"
            )
        if self.event_mutation_indices:
            if self.mutation_plan is None or any(
                index >= len(self.mutation_plan.specs) for index in self.event_mutation_indices
            ):
                raise ManagedRuntimeError(
                    "managed kernel binding Event mutation index is not bound"
                )


class ManagedTaskKernel(Protocol):
    """Static task math ABI used by :class:`ManagedReferenceRuntime`.

    Implementations are constructed on the cold path and receive no concrete
    backend/env argument.  Every state read must come from the supplied typed
    batch, and every task write must target the supplied runtime-owned buffers.
    """

    executor_key: str

    def bind(self, *, binding: ManagedKernelBinding) -> None:
        """Capture immutable public metadata exactly once on the cold path."""

    def create_task_state(self, *, num_envs: int, dtype: np.dtype[Any]) -> object: ...

    def apply_action(
        self,
        *,
        actions: np.ndarray,
        task_state: object,
        control_out: np.ndarray,
    ) -> None: ...

    def build_pre_physics_mutation(
        self,
        *,
        task_state: object,
    ) -> BackendMutationBatch | None: ...

    def evaluate_termination(
        self,
        *,
        state: StateBatch,
        task_state: object,
        terminated_out: np.ndarray,
    ) -> None: ...

    def evaluate_reward(
        self,
        *,
        state: StateBatch,
        task_state: object,
        reward_out: np.ndarray,
    ) -> None: ...

    def evaluate_metrics(
        self,
        *,
        state: StateBatch,
        task_state: object,
        terminated: np.ndarray,
    ) -> tuple[ManagedMetric, ...]: ...

    def write_observations(
        self,
        *,
        state: StateBatch,
        task_state: object,
        observation_buffers: tuple[np.ndarray, ...],
    ) -> None: ...

    def prepare_reset(
        self,
        *,
        rows: RowSelection,
        task_state: object,
    ) -> ManagedResetRequest: ...

    def complete_reset(
        self,
        *,
        request: ManagedResetRequest,
        state: StateBatch,
        task_state: object,
    ) -> None: ...


def _require_c_contiguous(array: np.ndarray, *, name: str) -> None:
    if not array.flags.c_contiguous:
        raise ManagedRuntimeError(f"{name} must be C-contiguous")


def _require_full_array(
    array: object,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    name: str,
) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise ManagedRuntimeError(f"{name} must be a numpy array")
    if array.shape != shape:
        raise ManagedRuntimeError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype != dtype:
        raise ManagedRuntimeError(f"{name} must have dtype {dtype.name}, got {array.dtype.name}")
    _require_c_contiguous(array, name=name)
    return array


@dataclass
class ManagedEnvState:
    """Manager-owned lifecycle state returned by ``init_state()``/``step()``.

    The fields intentionally mirror the env-layer ``NpEnvState`` contract
    (dict observations, vector reward/termination flags, info mapping), but
    the type is owned by the manager layer so the runtime never depends on
    the env contract.  Conversion to ``NpEnvState`` is only allowed at env
    adapter boundaries.
    """

    obs: dict[str, np.ndarray]
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict[str, Any]
    final_observation: dict[str, np.ndarray] | None = None

    def replace(self, **updates: Any) -> "ManagedEnvState":
        return dataclasses.replace(self, **updates)


class ManagedReferenceRuntime:
    """Host reference runtime with one canonical terminal/autoreset lifecycle.

    It is intentionally a correctness executor.  Fused executors consume the
    same ``CompiledTaskPlan`` and preserve the lifecycle semantics proven here
    rather than copying NpEnv control flow.
    """

    def __init__(
        self,
        *,
        backend: SimBackend,
        plan: CompiledTaskPlan,
        kernel: ManagedTaskKernel,
        max_episode_steps: int | None,
        autoreset: bool = True,
        record_lifecycle: bool = False,
        validate_observation_terms: bool = False,
        stability_buffer_provider: ManagedRuntimeBufferProvider | None = None,
        require_complete_backend_instrumentation: bool = False,
    ) -> None:
        if not isinstance(backend, SimBackend):
            raise ManagedRuntimeError("managed runtime requires a SimBackend")
        if not isinstance(plan, CompiledTaskPlan):
            raise ManagedRuntimeError("managed runtime requires a CompiledTaskPlan")
        validate_compiled_plan_fingerprints(plan)
        if plan.backend_io.execution_profile is not ExecutionProfile.HOST_NUMPY:
            raise ManagedRuntimeError("managed reference runtime only supports host_numpy plans")
        if not isinstance(autoreset, bool):
            raise ManagedRuntimeError("managed autoreset must be a bool")
        if not isinstance(record_lifecycle, bool):
            raise ManagedRuntimeError("record_lifecycle must be a bool")
        if not isinstance(validate_observation_terms, bool):
            raise ManagedRuntimeError("validate_observation_terms must be a bool")
        if stability_buffer_provider is not None and not isinstance(
            stability_buffer_provider, ManagedRuntimeBufferProvider
        ):
            raise ManagedRuntimeError(
                "managed stability buffer provider must implement ManagedRuntimeBufferProvider"
            )
        if stability_buffer_provider is not None and stability_buffer_provider is not kernel:
            raise ManagedRuntimeError(
                "managed stability buffer provider must be the task kernel owner"
            )
        if not isinstance(require_complete_backend_instrumentation, bool):
            raise ManagedRuntimeError("require_complete_backend_instrumentation must be a bool")
        if require_complete_backend_instrumentation and stability_buffer_provider is None:
            raise ManagedRuntimeError(
                "complete backend instrumentation requires a managed stability buffer provider"
            )
        if max_episode_steps is not None and (
            isinstance(max_episode_steps, bool)
            or not isinstance(max_episode_steps, int)
            or max_episode_steps <= 0
        ):
            raise ManagedRuntimeError("max_episode_steps must be a positive integer or None")
        self._validate_kernel(kernel, plan)
        self._backend = backend
        self._plan = plan
        self._kernel = kernel
        self._autoreset = autoreset
        self._record_lifecycle = record_lifecycle
        self._validate_observation_terms_enabled = validate_observation_terms
        self._max_episode_steps = max_episode_steps
        self._stability_buffer_provider = stability_buffer_provider
        self._stability_monitor = (
            None
            if stability_buffer_provider is None
            else _ManagedRuntimeStabilityMonitor(
                require_complete_backend_instrumentation=require_complete_backend_instrumentation
            )
        )

        bound = backend.bind_task_io(plan.backend_io)
        if not isinstance(bound, BoundBackendPlan):
            raise ManagedRuntimeError("backend bind_task_io must return a BoundBackendPlan")
        if bound.backend_type != backend.backend_type:
            raise ManagedRuntimeError("backend bound plan type differs from backend identity")
        if bound.execution_profile is not plan.backend_io.execution_profile:
            raise ManagedRuntimeError(
                "backend bound plan execution profile differs from compiled plan"
            )
        if bound.num_envs != backend.num_envs:
            raise ManagedRuntimeError(
                "backend bound plan row universe differs from backend num_envs"
            )
        if (
            bound.state.fields != plan.backend_io.state_fields
            or bound.control != plan.backend_io.control
        ):
            raise ManagedRuntimeError(
                "backend bound plan I/O differs from compiled task requirements"
            )
        self._bound_plan = bound
        self._mutation_plan = self._bind_mutation_plan()
        if bound.reset_requires_mutation_batch and self._mutation_plan is None:
            raise ManagedRuntimeError("backend reset contract requires a bound mutation plan")
        self._num_envs = bound.num_envs
        self._dtype = np.dtype(bound.control.buffer.dtype)
        self._kernel_binding = ManagedKernelBinding(
            task_fingerprint=plan.fingerprint,
            policy_abi_fingerprint=plan.policy_abi.fingerprint,
            num_envs=self._num_envs,
            dtype=self._dtype.name,
            execution_profile=bound.execution_profile,
            state_field_indices=tuple(
                (field.key, index) for index, field in enumerate(bound.state.fields)
            ),
            observation_buffer_indices=tuple(
                (group.key, index) for index, group in enumerate(plan.policy_abi.observation_groups)
            ),
            mutation_plan=self._mutation_plan,
            event_mutation_indices=tuple(event.mutation_index for event in plan.mutation_events),
        )
        self._bind_kernel()
        self._control = np.empty(
            (self._num_envs, *bound.control.buffer.row_shape), dtype=self._dtype
        )
        # Full-row control has a plan-stable address and contract.  Bind its
        # typed envelope once on the cold path; only the numeric contents are
        # manager-written before each backend barrier.
        self._all_rows = RowSelection.all(self._num_envs)
        self._control_batch = ControlBatch(
            plan=self._bound_plan,
            rows=self._all_rows,
            buffer=BufferView(
                handle=self._control,
                shape=self._control.shape,
                contract=self._bound_plan.control.buffer,
            ),
        )
        self._observation_keys, self._observation_buffers = self._allocate_observations()
        if validate_observation_terms:
            unsupported = tuple(
                group.key
                for group in plan.policy_abi.observation_groups
                if not np.issubdtype(np.dtype(group.dtype), np.floating)
            )
            if unsupported:
                raise ManagedRuntimeError(
                    "observation term validation requires floating-point groups; "
                    f"unsupported={unsupported}"
                )
        self._obs = dict(zip(self._observation_keys, self._observation_buffers, strict=True))
        self._reward = np.zeros((self._num_envs,), dtype=self._dtype)
        self._terminated = np.zeros((self._num_envs,), dtype=bool)
        self._truncated = np.zeros((self._num_envs,), dtype=bool)
        self._steps = np.zeros((self._num_envs,), dtype=np.uint32)
        self._final_observation_scratch = {
            key: np.empty_like(value) for key, value in self._obs.items()
        }
        self._compat_final_observation = {
            key: np.empty_like(value) for key, value in self._obs.items()
        }
        self._final_observation_mask = np.zeros((self._num_envs,), dtype=bool)
        self._info: dict[str, Any] = {
            "steps": self._steps,
            "log": {},
            "final_observation": self._compat_final_observation,
            "_final_observation": self._final_observation_mask,
        }
        self._state: ManagedEnvState | None = None
        self._task_state: object | None = None
        self._last_trace: tuple[ManagedLifecycleEvent, ...] = ()
        self._trace_events: list[ManagedLifecycleEvent] = []
        self._last_step_diagnostics: BackendBatchDiagnostics | None = None
        self._last_reset_diagnostics: BackendBatchDiagnostics | None = None

    @staticmethod
    def _validate_kernel(kernel: ManagedTaskKernel, plan: CompiledTaskPlan) -> None:
        executor_key = getattr(kernel, "executor_key", None)
        if not isinstance(executor_key, str) or executor_key != plan.executor_key:
            raise ManagedRuntimeError(
                "managed kernel executor_key does not match the compiled task plan"
            )
        # This cold-path depth check only sees instance attributes.  It cannot
        # detect class attributes or closure captures; the public kernel
        # protocol remains the owning backend-isolation contract.
        for name in ("backend", "env", "_backend", "_env"):
            if hasattr(kernel, name):
                raise ManagedRuntimeError(
                    f"managed kernel must not retain forbidden {name!r} owner reference"
                )

    def _bind_mutation_plan(self) -> BoundMutationPlan | None:
        if not self._plan.mutation_specs:
            return None
        bound = self._backend.bind_mutation_plan(self._plan.mutation_specs)
        if not isinstance(bound, BoundMutationPlan):
            raise ManagedRuntimeError("backend bind_mutation_plan must return a BoundMutationPlan")
        try:
            bound.require_owner(
                backend_type=self._bound_plan.backend_type,
                backend_instance_id=self._bound_plan.backend_instance_id,
            )
        except BackendBatchContractError as exc:
            raise ManagedRuntimeError(
                "bound mutation plan owner differs from the bound backend plan"
            ) from exc
        if bound.num_envs != self._bound_plan.num_envs:
            raise ManagedRuntimeError("bound mutation plan row universe differs from state plan")
        return bound

    def _bind_kernel(self) -> None:
        """Inject immutable public binding metadata before any lifecycle call."""

        bind = getattr(self._kernel, "bind", None)
        if not callable(bind):
            raise ManagedRuntimeError("managed kernel must expose a cold-path bind method")
        result = bind(binding=self._kernel_binding)
        if result is not None:
            raise ManagedRuntimeError("managed kernel bind method must return None")
        self._validate_kernel(self._kernel, self._plan)

    def _allocate_observations(self) -> tuple[tuple[str, ...], tuple[np.ndarray, ...]]:
        channel_contracts = {channel.key: channel.buffer for channel in self._plan.output_channels}
        keys: list[str] = []
        buffers: list[np.ndarray] = []
        for group in self._plan.policy_abi.observation_groups:
            channel_key = f"obs:{group.key}"
            try:
                contract = channel_contracts[channel_key]
            except KeyError as exc:
                raise ManagedRuntimeError(
                    f"compiled plan lacks runtime output channel {channel_key!r}"
                ) from exc
            if contract.row_shape != (group.width,) or contract.dtype != group.dtype:
                raise ManagedRuntimeError(
                    f"compiled observation channel {channel_key!r} disagrees with policy ABI"
                )
            if contract.placement != self._bound_plan.control.buffer.placement:
                raise ManagedRuntimeError(
                    f"compiled observation channel {channel_key!r} placement disagrees with control"
                )
            keys.append(group.key)
            buffers.append(np.empty((self._num_envs, group.width), dtype=contract.dtype))
        return tuple(keys), tuple(buffers)

    @property
    def plan(self) -> CompiledTaskPlan:
        return self._plan

    @property
    def policy_abi_snapshot(self) -> dict[str, Any]:
        """Return a fresh semantic ABI extension for experiment/sim2sim metadata.

        This is a cold/diagnostic accessor over immutable plan metadata only;
        it neither reads backend state nor exposes backend-local selector
        bindings.
        """

        return managed_policy_abi_snapshot(self._plan)

    @property
    def bound_plan(self) -> BoundBackendPlan:
        return self._bound_plan

    @property
    def kernel_binding(self) -> ManagedKernelBinding:
        return self._kernel_binding

    @property
    def state(self) -> ManagedEnvState | None:
        return self._state

    @property
    def task_state(self) -> object | None:
        """Expose kernel-owned state for diagnostics; callers must not mutate it."""
        return self._task_state

    @property
    def last_trace(self) -> tuple[ManagedLifecycleEvent, ...]:
        return self._last_trace

    @property
    def last_step_diagnostics(self) -> BackendBatchDiagnostics | None:
        """Most recent typed backend step diagnostics, if a step has run."""

        return self._last_step_diagnostics

    @property
    def last_reset_diagnostics(self) -> BackendBatchDiagnostics | None:
        """Most recent typed backend reset diagnostics, including initial reset."""

        return self._last_reset_diagnostics

    @property
    def stability_diagnostics(self) -> ManagedRuntimeStabilityDiagnostics | None:
        """Return the explicit warm-buffer diagnostic snapshot when enabled.

        The optional monitor is deliberately opt-in so production execution is
        not burdened with diagnostic descriptor construction.  A caller that
        requests it before ``init_state`` receives an explicit lifecycle
        error rather than a partial baseline.
        """

        if self._stability_monitor is None:
            return None
        return self._stability_monitor.snapshot()

    def _manager_runtime_buffers(self) -> tuple[ManagedRuntimeBuffer, ...]:
        """Return the core runtime-owned arrays in canonical diagnostic form."""

        buffers: list[ManagedRuntimeBuffer] = [
            ManagedRuntimeBuffer("runtime.control", self._control),
            ManagedRuntimeBuffer("runtime.reward", self._reward),
            ManagedRuntimeBuffer("runtime.terminated", self._terminated),
            ManagedRuntimeBuffer("runtime.truncated", self._truncated),
            ManagedRuntimeBuffer("runtime.steps", self._steps),
            ManagedRuntimeBuffer("runtime.final_observation_mask", self._final_observation_mask),
        ]
        for key in self._observation_keys:
            buffers.append(ManagedRuntimeBuffer(f"runtime.observation.{key}", self._obs[key]))
            buffers.append(
                ManagedRuntimeBuffer(
                    f"runtime.final_observation.{key}", self._final_observation_scratch[key]
                )
            )
            buffers.append(
                ManagedRuntimeBuffer(
                    f"runtime.compat_final_observation.{key}",
                    self._compat_final_observation[key],
                )
            )
        return tuple(buffers)

    def _stability_buffers(self) -> tuple[ManagedRuntimeBuffer, ...]:
        provider = self._stability_buffer_provider
        task_state = self._task_state
        if provider is None or task_state is None:
            raise ManagedRuntimeError(
                "managed runtime stability buffers are unavailable before init"
            )
        task_buffers = provider.managed_runtime_buffers(task_state=task_state)
        if not isinstance(task_buffers, tuple) or any(
            not isinstance(buffer, ManagedRuntimeBuffer) for buffer in task_buffers
        ):
            raise ManagedRuntimeError(
                "managed stability buffer provider must return a tuple of ManagedRuntimeBuffer"
            )
        return (*self._manager_runtime_buffers(), *task_buffers)

    def _arm_stability_monitor(self) -> None:
        if self._stability_monitor is not None:
            self._stability_monitor.arm(self._stability_buffers())

    def _observe_stability_buffers(self) -> None:
        if self._stability_monitor is not None:
            self._stability_monitor.observe_buffers(self._stability_buffers())

    def _observe_stability_state(self, state: StateBatch) -> None:
        if self._stability_monitor is not None:
            self._stability_monitor.observe_state(state)

    def _record_backend_diagnostics(
        self, *, phase: str, diagnostics: BackendBatchDiagnostics
    ) -> None:
        if phase == "step":
            self._last_step_diagnostics = diagnostics
        elif phase == "reset":
            self._last_reset_diagnostics = diagnostics
        else:  # pragma: no cover - internal fixed call sites.
            raise ManagedRuntimeError(f"unknown managed backend diagnostic phase {phase!r}")
        if self._stability_monitor is not None:
            self._stability_monitor.record_backend(phase=phase, diagnostics=diagnostics)

    def _begin_trace(self) -> None:
        self._trace_events.clear()

    def _trace(self, phase: ManagedLifecyclePhase, rows: RowSelection | None = None) -> None:
        if not self._record_lifecycle:
            return
        self._trace_events.append(
            ManagedLifecycleEvent(
                phase=phase,
                rows=None if rows is None or rows.is_all else rows.indices,
            )
        )

    def _finish_trace(self) -> None:
        self._last_trace = tuple(self._trace_events)

    def _require_task_state(self) -> object:
        if self._task_state is None:
            raise ManagedRuntimeError("managed task state has not been initialized")
        return self._task_state

    def _validate_mutation(
        self,
        mutation: BackendMutationBatch | None,
        *,
        rows: RowSelection,
        context: str,
        required: bool,
    ) -> None:
        if self._mutation_plan is None:
            if mutation is not None:
                raise ManagedRuntimeError(
                    f"{context} returned a mutation without a bound mutation plan"
                )
            return
        if mutation is None:
            if required:
                raise ManagedRuntimeError(
                    f"{context} requires a typed mutation batch for the bound mutation plan"
                )
            return
        try:
            mutation_fingerprint = mutation.plan_fingerprint
            mutation_rows = mutation.rows
        except AttributeError as exc:
            raise ManagedRuntimeError(f"{context} mutation is not a typed batch envelope") from exc
        if not isinstance(mutation_fingerprint, str) or not isinstance(mutation_rows, RowSelection):
            raise ManagedRuntimeError(f"{context} mutation is not a typed batch envelope")
        if mutation_fingerprint != self._mutation_plan.fingerprint:
            raise ManagedRuntimeError(
                f"{context} mutation batch belongs to a different bound mutation plan"
            )
        if mutation_rows != rows:
            raise ManagedRuntimeError(f"{context} mutation rows do not match lifecycle rows")

    def _validate_state(
        self, state: StateBatch, *, phase: StateBatchPhase, rows: RowSelection
    ) -> None:
        if not isinstance(state, StateBatch):
            raise ManagedRuntimeError("backend lifecycle result must contain a StateBatch")
        try:
            state.plan.require_compatible(self._bound_plan)
        except BackendBatchContractError as exc:
            raise ManagedRuntimeError(
                "backend lifecycle state belongs to a different bound backend plan"
            ) from exc
        if state.phase is not phase:
            raise ManagedRuntimeError(
                f"backend lifecycle state must have phase {phase.value}, got {state.phase.value}"
            )
        if state.rows != rows:
            raise ManagedRuntimeError("backend lifecycle state rows do not match request rows")
        state.assert_valid()

    def _validate_observation_buffers(self) -> None:
        for key, buffer, group in zip(
            self._observation_keys,
            self._observation_buffers,
            self._plan.policy_abi.observation_groups,
            strict=True,
        ):
            if key != group.key:  # pragma: no cover - constructor-owned invariant
                raise ManagedRuntimeError(
                    "managed observation buffer order differs from policy ABI"
                )
            _require_full_array(
                buffer,
                shape=(self._num_envs, group.width),
                dtype=np.dtype(group.dtype),
                name=f"managed observation buffer {key!r}",
            )
            if not np.isfinite(buffer).all():
                raise ManagedRuntimeError(
                    f"managed observation buffer {key!r} contains non-finite values"
                )

    @staticmethod
    def _row_indices(rows: RowSelection) -> slice | np.ndarray:
        if rows.is_all:
            return slice(None)
        assert rows.indices is not None
        return np.asarray(rows.indices, dtype=np.intp)

    def _poison_observation_terms(self, rows: RowSelection) -> None:
        if not self._validate_observation_terms_enabled:
            return
        indices = self._row_indices(rows)
        for buffer, group in zip(
            self._observation_buffers,
            self._plan.policy_abi.observation_groups,
            strict=True,
        ):
            for observation in group.outputs:
                output = observation.output
                buffer[indices, output.start : output.stop] = np.nan

    def _validate_observation_term_writes(self, rows: RowSelection) -> None:
        if not self._validate_observation_terms_enabled:
            return
        indices = self._row_indices(rows)
        unwritten: list[str] = []
        for buffer, group in zip(
            self._observation_buffers,
            self._plan.policy_abi.observation_groups,
            strict=True,
        ):
            for observation in group.outputs:
                output = observation.output
                if np.isnan(buffer[indices, output.start : output.stop]).any():
                    unwritten.append(observation.semantic_key)
        if unwritten:
            raise ManagedRuntimeError(
                "managed observation kernel left declared term outputs unwritten; "
                f"semantic_keys={tuple(unwritten)}"
            )

    def _apply_metrics(self, metrics: tuple[ManagedMetric, ...]) -> None:
        if not isinstance(metrics, tuple) or any(
            not isinstance(metric, ManagedMetric) for metric in metrics
        ):
            raise ManagedRuntimeError("managed kernel metrics must be a tuple of ManagedMetric")
        keys = tuple(metric.key for metric in metrics)
        if len(set(keys)) != len(keys):
            raise ManagedRuntimeError("managed kernel metrics must use unique keys")
        log = self._info["log"]
        if not isinstance(log, dict):  # pragma: no cover - internal invariant
            raise ManagedRuntimeError("managed runtime log buffer is corrupted")
        log.clear()
        for metric in metrics:
            log[metric.key] = (
                metric.value.copy() if isinstance(metric.value, np.ndarray) else metric.value
            )

    def _clear_final_observation(self) -> None:
        self._final_observation_mask.fill(False)
        if self._state is not None:
            self._state = self._state.replace(final_observation=None)

    def _capture_final_observation(self, rows: RowSelection) -> None:
        if rows.is_all:
            indices = slice(None)
        else:
            assert rows.indices is not None
            indices = np.asarray(rows.indices, dtype=np.intp)
        for key, observation in self._obs.items():
            self._final_observation_scratch[key][indices] = observation[indices]
            self._compat_final_observation[key][indices] = observation[indices]
        self._final_observation_mask.fill(False)
        if rows.is_all:
            self._final_observation_mask.fill(True)
        else:
            self._final_observation_mask[indices] = True
        assert self._state is not None
        self._state = self._state.replace(final_observation=self._final_observation_scratch)

    def _reset_rows(
        self,
        rows: RowSelection,
        *,
        initial: bool,
        terminal_state: StateBatch | None = None,
    ) -> None:
        task_state = self._require_task_state()
        self._trace(
            ManagedLifecyclePhase.INITIAL_RESET_REQUEST
            if initial
            else ManagedLifecyclePhase.RESET_REQUEST,
            rows,
        )
        request = self._kernel.prepare_reset(rows=rows, task_state=task_state)
        if not isinstance(request, ManagedResetRequest):
            raise ManagedRuntimeError(
                "managed kernel prepare_reset must return ManagedResetRequest"
            )
        if request.rows != rows:
            raise ManagedRuntimeError("managed reset request rows do not match lifecycle rows")
        self._validate_mutation(
            request.mutation_batch,
            rows=rows,
            context="managed reset",
            required=self._mutation_plan is not None,
        )
        self._trace(ManagedLifecyclePhase.RESET_BACKEND, rows)
        reset_result = self._backend.reset_batch(
            self._bound_plan,
            rows,
            mutation_batch=request.mutation_batch,
        )
        if not isinstance(reset_result, BackendResetResult):
            raise ManagedRuntimeError("backend reset_batch must return a BackendResetResult")
        self._record_backend_diagnostics(phase="reset", diagnostics=reset_result.diagnostics)
        if terminal_state is not None:
            self._require_stale_after_reset(terminal_state)
        self._validate_state(reset_result.reset_state, phase=StateBatchPhase.RESET, rows=rows)
        self._observe_stability_state(reset_result.reset_state)
        self._trace(ManagedLifecyclePhase.TASK_STATE_RESET, rows)
        self._kernel.complete_reset(
            request=request,
            state=reset_result.reset_state,
            task_state=task_state,
        )
        reset_result.reset_state.assert_valid()
        self._trace(ManagedLifecyclePhase.OBSERVATION, rows)
        self._poison_observation_terms(rows)
        self._kernel.write_observations(
            state=reset_result.reset_state,
            task_state=task_state,
            observation_buffers=self._observation_buffers,
        )
        reset_result.reset_state.assert_valid()
        self._validate_observation_term_writes(rows)
        self._validate_observation_buffers()

    @staticmethod
    def _require_stale_after_reset(terminal: StateBatch) -> None:
        """Verify that a reset barrier invalidated the terminal borrowed view."""

        try:
            terminal.assert_valid()
        except StaleStateBatchError:
            return
        raise ManagedRuntimeError(
            "backend reset_batch did not invalidate the terminal StateBatch lease"
        )

    def init_state(self) -> ManagedEnvState:
        self._begin_trace()
        self._task_state = self._kernel.create_task_state(
            num_envs=self._num_envs, dtype=self._dtype
        )
        if self._task_state is None:
            raise ManagedRuntimeError("managed kernel create_task_state must not return None")
        self._reward.fill(0.0)
        self._terminated.fill(False)
        self._truncated.fill(False)
        self._steps.fill(0)
        self._info["log"].clear()
        self._state = ManagedEnvState(
            obs=self._obs,
            reward=self._reward,
            terminated=self._terminated,
            truncated=self._truncated,
            info=self._info,
            final_observation=None,
        )
        self._clear_final_observation()
        self._reset_rows(self._all_rows, initial=True)
        self._clear_final_observation()
        self._arm_stability_monitor()
        self._trace(ManagedLifecyclePhase.COMPLETE)
        self._finish_trace()
        return self._state

    def step(self, actions: np.ndarray) -> ManagedEnvState:
        if self._state is None:
            raise ManagedRuntimeError("managed runtime step requires init_state() first")
        assert self._state is not None
        task_state = self._require_task_state()
        self._observe_stability_buffers()
        self._begin_trace()
        actions = _require_full_array(
            actions,
            shape=(self._num_envs, self._bound_plan.control.buffer.row_shape[0]),
            dtype=self._dtype,
            name="managed actions",
        )
        if not np.isfinite(actions).all():
            raise ManagedRuntimeError("managed actions contain non-finite values")
        self._clear_final_observation()
        self._truncated.fill(False)

        self._trace(ManagedLifecyclePhase.ACTION)
        self._kernel.apply_action(
            actions=actions,
            task_state=task_state,
            control_out=self._control,
        )
        _require_full_array(
            self._control,
            shape=(self._num_envs, *self._bound_plan.control.buffer.row_shape),
            dtype=self._dtype,
            name="managed control buffer",
        )
        if not np.isfinite(self._control).all():
            raise ManagedRuntimeError("managed control buffer contains non-finite values")
        self._trace(ManagedLifecyclePhase.PRE_PHYSICS)
        mutation = self._kernel.build_pre_physics_mutation(task_state=task_state)
        self._validate_mutation(
            mutation,
            rows=self._all_rows,
            context="managed pre-physics",
            required=False,
        )
        self._trace(ManagedLifecyclePhase.PHYSICS)
        step_result = self._backend.step_batch(
            self._bound_plan,
            self._control_batch,
            mutation_batch=mutation,
            nsteps=self._bound_plan.control.physics_substeps_per_control,
        )
        if not isinstance(step_result, BackendStepResult):
            raise ManagedRuntimeError("backend step_batch must return a BackendStepResult")
        self._record_backend_diagnostics(phase="step", diagnostics=step_result.diagnostics)
        terminal = step_result.terminal_state
        self._validate_state(
            terminal,
            phase=StateBatchPhase.TERMINAL,
            rows=self._all_rows,
        )
        self._observe_stability_state(terminal)

        self._trace(ManagedLifecyclePhase.TERMINATION)
        self._terminated.fill(False)
        self._kernel.evaluate_termination(
            state=terminal,
            task_state=task_state,
            terminated_out=self._terminated,
        )
        _require_full_array(
            self._terminated,
            shape=(self._num_envs,),
            dtype=np.dtype(bool),
            name="managed terminated buffer",
        )
        self._trace(ManagedLifecyclePhase.REWARD)
        self._reward.fill(0.0)
        self._kernel.evaluate_reward(
            state=terminal,
            task_state=task_state,
            reward_out=self._reward,
        )
        _require_full_array(
            self._reward,
            shape=(self._num_envs,),
            dtype=self._dtype,
            name="managed reward buffer",
        )
        if not np.isfinite(self._reward).all():
            raise ManagedRuntimeError("managed reward buffer contains non-finite values")
        self._trace(ManagedLifecyclePhase.METRIC)
        self._apply_metrics(
            self._kernel.evaluate_metrics(
                state=terminal,
                task_state=task_state,
                terminated=self._terminated,
            )
        )
        terminal.assert_valid()
        self._trace(ManagedLifecyclePhase.TERMINAL_OBSERVATION)
        self._poison_observation_terms(self._all_rows)
        self._kernel.write_observations(
            state=terminal,
            task_state=task_state,
            observation_buffers=self._observation_buffers,
        )
        terminal.assert_valid()
        self._validate_observation_term_writes(self._all_rows)
        self._validate_observation_buffers()

        self._steps += 1
        self._trace(ManagedLifecyclePhase.TIMEOUT)
        if self._max_episode_steps is not None:
            np.greater_equal(self._steps, self._max_episode_steps, out=self._truncated)
        done = self._terminated | self._truncated
        if self._autoreset and bool(np.any(done)):
            done_rows = RowSelection.selected(
                self._num_envs,
                tuple(int(row) for row in np.flatnonzero(done)),
            )
            self._trace(ManagedLifecyclePhase.FINAL_OBSERVATION, done_rows)
            self._capture_final_observation(done_rows)
            self._trace(ManagedLifecyclePhase.AUTORESET, done_rows)
            self._steps[np.asarray(done_rows.indices, dtype=np.intp)] = 0
            self._reset_rows(done_rows, initial=False, terminal_state=terminal)
        self._observe_stability_buffers()
        self._trace(ManagedLifecyclePhase.COMPLETE)
        self._finish_trace()
        return self._state


__all__ = [
    "ManagedEnvState",
    "ManagedLifecycleEvent",
    "ManagedLifecyclePhase",
    "ManagedKernelBinding",
    "ManagedMetric",
    "ManagedReferenceRuntime",
    "ManagedResetRequest",
    "ManagedRuntimeBuffer",
    "ManagedRuntimeBufferAddress",
    "ManagedRuntimeBufferProvider",
    "ManagedRuntimeError",
    "ManagedRuntimeStabilityDiagnostics",
    "ManagedTaskKernel",
]
