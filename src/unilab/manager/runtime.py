"""Reference managed-task lifecycle over the public typed backend contract.

The runtime deliberately owns lifecycle ordering but not task math or backend
storage.  A cold-bound :class:`ManagedTaskKernel` receives only borrowed
``StateBatch`` views and runtime-owned buffers; it never receives an env,
backend, selector, registry, asset, or model object.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import numpy as np

from unilab.base.backend.base import SimBackend
from unilab.base.backend.batch import (
    BackendBatchContractError,
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
from unilab.base.np_env import NpEnvState

from .entities import ManagerContractError
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


class ManagedReferenceRuntime:
    """Host reference runtime with one canonical terminal/autoreset lifecycle.

    It is intentionally a correctness executor.  Future fused/device
    executors consume the same ``CompiledTaskPlan`` and preserve the lifecycle
    semantics proven here rather than copying NpEnv control flow.
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
    ) -> None:
        if not isinstance(backend, SimBackend):
            raise ManagedRuntimeError("managed runtime requires a SimBackend")
        if not isinstance(plan, CompiledTaskPlan):
            raise ManagedRuntimeError("managed runtime requires a CompiledTaskPlan")
        if plan.backend_io.execution_profile is not ExecutionProfile.HOST_NUMPY:
            raise ManagedRuntimeError("managed reference runtime only supports host_numpy plans")
        if not isinstance(autoreset, bool):
            raise ManagedRuntimeError("managed autoreset must be a bool")
        if not isinstance(record_lifecycle, bool):
            raise ManagedRuntimeError("record_lifecycle must be a bool")
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
        self._max_episode_steps = max_episode_steps

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
        )
        self._bind_kernel()
        self._control = np.empty(
            (self._num_envs, *bound.control.buffer.row_shape), dtype=self._dtype
        )
        self._observation_keys, self._observation_buffers = self._allocate_observations()
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
        self._state: NpEnvState | None = None
        self._task_state: object | None = None
        self._last_trace: tuple[ManagedLifecycleEvent, ...] = ()
        self._trace_events: list[ManagedLifecycleEvent] = []

    @staticmethod
    def _validate_kernel(kernel: ManagedTaskKernel, plan: CompiledTaskPlan) -> None:
        executor_key = getattr(kernel, "executor_key", None)
        if not isinstance(executor_key, str) or executor_key != plan.executor_key:
            raise ManagedRuntimeError(
                "managed kernel executor_key does not match the compiled task plan"
            )
        # This is a cold-path structural guard, not runtime capability probing.
        # A kernel must never close over an env/backend and bypass StateBatch.
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
    def bound_plan(self) -> BoundBackendPlan:
        return self._bound_plan

    @property
    def kernel_binding(self) -> ManagedKernelBinding:
        return self._kernel_binding

    @property
    def state(self) -> NpEnvState | None:
        return self._state

    @property
    def task_state(self) -> object | None:
        """Expose kernel-owned state for diagnostics; callers must not mutate it."""
        return self._task_state

    @property
    def last_trace(self) -> tuple[ManagedLifecycleEvent, ...]:
        return self._last_trace

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
        for key, buffer in zip(self._observation_keys, self._observation_buffers, strict=True):
            _require_full_array(
                buffer,
                shape=(self._num_envs, buffer.shape[1]),
                dtype=buffer.dtype,
                name=f"managed observation buffer {key!r}",
            )
            if not np.isfinite(buffer).all():
                raise ManagedRuntimeError(
                    f"managed observation buffer {key!r} contains non-finite values"
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
        if terminal_state is not None:
            self._require_stale_after_reset(terminal_state)
        self._validate_state(reset_result.reset_state, phase=StateBatchPhase.RESET, rows=rows)
        self._trace(ManagedLifecyclePhase.TASK_STATE_RESET, rows)
        self._kernel.complete_reset(
            request=request,
            state=reset_result.reset_state,
            task_state=task_state,
        )
        reset_result.reset_state.assert_valid()
        self._trace(ManagedLifecyclePhase.OBSERVATION, rows)
        self._kernel.write_observations(
            state=reset_result.reset_state,
            task_state=task_state,
            observation_buffers=self._observation_buffers,
        )
        reset_result.reset_state.assert_valid()
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

    def init_state(self) -> NpEnvState:
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
        self._state = NpEnvState(
            obs=self._obs,
            reward=self._reward,
            terminated=self._terminated,
            truncated=self._truncated,
            info=self._info,
            final_observation=None,
        )
        self._clear_final_observation()
        self._reset_rows(RowSelection.all(self._num_envs), initial=True)
        self._clear_final_observation()
        self._trace(ManagedLifecyclePhase.COMPLETE)
        self._finish_trace()
        return self._state

    def step(self, actions: np.ndarray) -> NpEnvState:
        if self._state is None:
            self.init_state()
        assert self._state is not None
        task_state = self._require_task_state()
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
            rows=RowSelection.all(self._num_envs),
            context="managed pre-physics",
            required=False,
        )
        control = ControlBatch(
            plan=self._bound_plan,
            rows=RowSelection.all(self._num_envs),
            buffer=BufferView(
                handle=self._control,
                shape=self._control.shape,
                contract=self._bound_plan.control.buffer,
            ),
        )
        self._trace(ManagedLifecyclePhase.PHYSICS)
        step_result = self._backend.step_batch(
            self._bound_plan,
            control,
            mutation_batch=mutation,
            nsteps=self._bound_plan.control.physics_substeps_per_control,
        )
        if not isinstance(step_result, BackendStepResult):
            raise ManagedRuntimeError("backend step_batch must return a BackendStepResult")
        terminal = step_result.terminal_state
        all_rows = RowSelection.all(self._num_envs)
        self._validate_state(terminal, phase=StateBatchPhase.TERMINAL, rows=all_rows)

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
        self._kernel.write_observations(
            state=terminal,
            task_state=task_state,
            observation_buffers=self._observation_buffers,
        )
        terminal.assert_valid()
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
        self._trace(ManagedLifecyclePhase.COMPLETE)
        self._finish_trace()
        return self._state


__all__ = [
    "ManagedLifecycleEvent",
    "ManagedLifecyclePhase",
    "ManagedKernelBinding",
    "ManagedMetric",
    "ManagedReferenceRuntime",
    "ManagedResetRequest",
    "ManagedRuntimeError",
    "ManagedTaskKernel",
]
