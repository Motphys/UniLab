"""CUDA-resident managed-task lifecycle over the public typed backend ABI.

This module is deliberately separate from :mod:`unilab.manager.runtime`.
``ManagedReferenceRuntime`` owns the NumPy compatibility contract, whereas
``DeviceManagedRuntime`` never creates ``NpEnvState`` or materializes a task
value on the host.  The only host-side objects in its warm path are immutable
typed descriptors and optional lifecycle diagnostics; action, physics state,
termination, reward, observations, reset membership, and final-observation
data remain CUDA tensors joined by explicit events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, cast
from weakref import WeakKeyDictionary

import torch

from unilab.base.backend import (
    BackendBatchContractError,
    BackendBatchDiagnostics,
    BackendCompletionEvent,
    BackendResetResult,
    BackendStepResult,
    BoundBackendPlan,
    BufferContract,
    BufferLayout,
    BufferLifetime,
    BufferMutability,
    BufferOwner,
    BufferPlacement,
    BufferView,
    ControlBatch,
    DeviceBufferContractError,
    DeviceBufferLease,
    DeviceCompletion,
    DeviceResetMutationBatch,
    DeviceTensorView,
    ExecutionProfile,
    MemorySpace,
    MutationValueBatch,
    RowSelection,
    SimulationStateMutationBatch,
    StateBatch,
    StateBatchPhase,
    TypedBackendMutationBatch,
    require_device_tensor_view,
)
from unilab.base.backend.base import SimBackend
from unilab.base.backend.mutation import BoundMutationPlan

from .fingerprint import managed_policy_abi_snapshot
from .plan import CompiledTaskPlan
from .runtime import ManagedKernelBinding, ManagedLifecyclePhase, ManagedRuntimeError


class DeviceManagedRuntimeError(ManagedRuntimeError):
    """Raised when a device-resident manager lifecycle contract is violated."""


@dataclass(frozen=True)
class DeviceTransitionBuffer:
    """One named runtime-owned tensor exported by a device transition."""

    key: str
    view: DeviceTensorView = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise DeviceManagedRuntimeError("device transition buffer key must be non-empty")
        if not isinstance(self.view, DeviceTensorView):
            raise DeviceManagedRuntimeError("device transition buffer requires a DeviceTensorView")


@dataclass(frozen=True)
class DeviceTransition:
    """One device-only terminal/autoreset transition.

    ``terminal_observations`` are from the state immediately after physics;
    ``observations`` are the next policy observations after the masked reset
    barrier; and ``final_observations`` are meaningful exactly where
    ``final_observation_mask`` is true.  Consumers must wait on
    :attr:`completion` (or an individual view) on their own CUDA stream.
    """

    plan_fingerprint: str
    observations: tuple[DeviceTransitionBuffer, ...]
    terminal_observations: tuple[DeviceTransitionBuffer, ...]
    final_observations: tuple[DeviceTransitionBuffer, ...]
    reward: DeviceTensorView = field(repr=False, compare=False)
    terminated: DeviceTensorView = field(repr=False, compare=False)
    truncated: DeviceTensorView = field(repr=False, compare=False)
    final_observation_mask: DeviceTensorView = field(repr=False, compare=False)
    completion: DeviceCompletion = field(repr=False, compare=False)
    trace: tuple[ManagedLifecyclePhase, ...] = ()
    step_diagnostics: BackendBatchDiagnostics | None = None
    reset_diagnostics: BackendBatchDiagnostics | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plan_fingerprint, str) or not self.plan_fingerprint.strip():
            raise DeviceManagedRuntimeError("device transition plan_fingerprint must be non-empty")
        for label, buffers in (
            ("observations", self.observations),
            ("terminal_observations", self.terminal_observations),
            ("final_observations", self.final_observations),
        ):
            if not isinstance(buffers, tuple) or not buffers:
                raise DeviceManagedRuntimeError(f"device transition {label} must be non-empty")
            if any(not isinstance(buffer, DeviceTransitionBuffer) for buffer in buffers):
                raise DeviceManagedRuntimeError(
                    f"device transition {label} must contain typed buffers"
                )
            keys = tuple(buffer.key for buffer in buffers)
            if len(set(keys)) != len(keys):
                raise DeviceManagedRuntimeError(f"device transition {label} has duplicate keys")
        expected_keys = tuple(buffer.key for buffer in self.observations)
        if (
            tuple(buffer.key for buffer in self.terminal_observations) != expected_keys
            or tuple(buffer.key for buffer in self.final_observations) != expected_keys
        ):
            raise DeviceManagedRuntimeError(
                "device transition observation key layouts must be identical"
            )
        if not isinstance(self.completion, DeviceCompletion):
            raise DeviceManagedRuntimeError("device transition completion is invalid")
        all_views = (
            *(buffer.view for buffer in self.observations),
            *(buffer.view for buffer in self.terminal_observations),
            *(buffer.view for buffer in self.final_observations),
            self.reward,
            self.terminated,
            self.truncated,
            self.final_observation_mask,
        )
        for view in all_views:
            if not isinstance(view, DeviceTensorView):
                raise DeviceManagedRuntimeError("device transition contains an invalid view")
            producer = view.require_completion()
            if (
                producer.placement != self.completion.placement
                or producer.owner_id != self.completion.owner_id
                or producer.epoch != self.completion.epoch
                or producer.event is not self.completion.event
            ):
                raise DeviceManagedRuntimeError(
                    "device transition outputs must share one runtime completion event"
                )
        if not isinstance(self.trace, tuple) or any(
            not isinstance(phase, ManagedLifecyclePhase) for phase in self.trace
        ):
            raise DeviceManagedRuntimeError("device transition lifecycle trace is invalid")
        for diagnostics in (self.step_diagnostics, self.reset_diagnostics):
            if diagnostics is not None and not isinstance(diagnostics, BackendBatchDiagnostics):
                raise DeviceManagedRuntimeError("device transition diagnostics are invalid")

    def observation(self, key: str) -> DeviceTensorView:
        """Return a next-policy observation without materializing it on host."""

        for buffer in self.observations:
            if buffer.key == key:
                return buffer.view
        raise DeviceManagedRuntimeError(f"device transition has no observation group {key!r}")

    def terminal_observation(self, key: str) -> DeviceTensorView:
        for buffer in self.terminal_observations:
            if buffer.key == key:
                return buffer.view
        raise DeviceManagedRuntimeError(
            f"device transition has no terminal observation group {key!r}"
        )

    def final_observation(self, key: str) -> DeviceTensorView:
        for buffer in self.final_observations:
            if buffer.key == key:
                return buffer.view
        raise DeviceManagedRuntimeError(f"device transition has no final observation group {key!r}")


@dataclass(frozen=True)
class DeviceResetPayload:
    """Kernel-owned CUDA reset staging consumed by the generic runtime envelope.

    The values are ordered exactly like ``BoundMutationPlan.specs``.  The
    runtime attaches leases/completion events and constructs the public
    :class:`DeviceResetMutationBatch`; task code never has to manufacture a
    backend envelope or access a raw physics tensor.
    """

    active_mask: torch.Tensor = field(repr=False, compare=False)
    values: tuple[torch.Tensor, ...] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.active_mask, torch.Tensor):
            raise DeviceManagedRuntimeError("device reset payload active_mask must be a tensor")
        if not isinstance(self.values, tuple) or not self.values:
            raise DeviceManagedRuntimeError("device reset payload requires value tensors")
        if any(not isinstance(value, torch.Tensor) for value in self.values):
            raise DeviceManagedRuntimeError("device reset payload values must contain only tensors")


class DeviceManagedTaskKernel(Protocol):
    """Torch CUDA task-math ABI consumed by :class:`DeviceManagedRuntime`."""

    executor_key: str

    def bind(self, *, binding: ManagedKernelBinding) -> None:
        """Capture cold-bound public plan metadata exactly once."""

        ...

    def create_task_state(
        self,
        *,
        num_envs: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> object:
        """Allocate task-owned CUDA state on the cold path."""

        ...

    def apply_action(
        self,
        *,
        actions: torch.Tensor,
        task_state: object,
        control_out: torch.Tensor,
    ) -> None:
        """Write backend control from policy action on the task CUDA stream."""

        ...

    def evaluate_terminal(
        self,
        *,
        state: StateBatch,
        task_state: object,
        reward_out: torch.Tensor,
        terminated_out: torch.Tensor,
        terminal_observation_buffers: tuple[torch.Tensor, ...],
    ) -> None:
        """Compute terminal termination/reward/observation tensors on CUDA."""

        ...

    def prepare_reset(
        self,
        *,
        active_mask: torch.Tensor,
        task_state: object,
    ) -> DeviceResetPayload:
        """Write manager-owned reset staging for an all-world CUDA mask."""

        ...

    def complete_reset(
        self,
        *,
        active_mask: torch.Tensor,
        state: StateBatch,
        task_state: object,
        observation_buffers: tuple[torch.Tensor, ...],
    ) -> None:
        """Update manager state and write next observations after reset physics."""

        ...


def _torch_dtype(dtype: str) -> torch.dtype:
    mapping = {"float32": torch.float32, "float64": torch.float64}
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise DeviceManagedRuntimeError(f"unsupported device runtime dtype {dtype!r}") from exc


def _runtime_contract(
    *, placement: BufferPlacement, row_shape: tuple[int, ...], dtype: str
) -> BufferContract:
    return BufferContract(
        row_shape=row_shape,
        dtype=dtype,
        layout=BufferLayout.C_CONTIGUOUS,
        placement=placement,
        owner=BufferOwner.RUNTIME,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.PLAN,
        dlpack_exportable=True,
        address_stable=True,
    )


def _mask_contract(*, placement: BufferPlacement) -> BufferContract:
    return BufferContract(
        row_shape=(),
        dtype="bool",
        layout=BufferLayout.C_CONTIGUOUS,
        placement=placement,
        owner=BufferOwner.MANAGER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.UNTIL_COMMIT,
        dlpack_exportable=True,
        address_stable=True,
    )


class DeviceManagedRuntime:
    """Canonical all-CUDA terminal/autoreset lifecycle for one compiled task.

    A reset is intentionally submitted for every transition, including an
    all-false mask.  That avoids a CPU ``any(done)`` branch and keeps the
    task-stream -> physics-stream dependency explicit.  The ``mjwarp`` reset
    implementation treats an all-false mask as a semantic no-op; later graph
    work may optimize that barrier only if it preserves this contract.
    """

    def __init__(
        self,
        *,
        backend: SimBackend,
        plan: CompiledTaskPlan,
        kernel: DeviceManagedTaskKernel,
        max_episode_steps: int | None,
        record_lifecycle: bool = False,
    ) -> None:
        if not isinstance(backend, SimBackend):
            raise DeviceManagedRuntimeError("device managed runtime requires a SimBackend")
        if not isinstance(plan, CompiledTaskPlan):
            raise DeviceManagedRuntimeError("device managed runtime requires a CompiledTaskPlan")
        if plan.backend_io.execution_profile is not ExecutionProfile.DEVICE_RESIDENT:
            raise DeviceManagedRuntimeError(
                "device managed runtime only supports device_resident plans"
            )
        if not isinstance(record_lifecycle, bool):
            raise DeviceManagedRuntimeError("record_lifecycle must be a bool")
        if max_episode_steps is not None and (
            isinstance(max_episode_steps, bool)
            or not isinstance(max_episode_steps, int)
            or max_episode_steps <= 0
        ):
            raise DeviceManagedRuntimeError("max_episode_steps must be a positive integer or None")
        self._validate_kernel(kernel, plan)

        bound = backend.bind_task_io(plan.backend_io)
        self._validate_bound_plan(backend=backend, plan=plan, bound=bound)
        if not plan.mutation_specs:
            raise DeviceManagedRuntimeError(
                "device managed runtime requires typed simulation-state reset mutations"
            )
        mutation_plan = backend.bind_mutation_plan(plan.mutation_specs)
        self._validate_mutation_plan(bound=bound, mutation_plan=mutation_plan)

        placement = bound.control.buffer.placement
        if (
            placement.memory_space is not MemorySpace.DEVICE
            or placement.device_type != "cuda"
            or placement.device_index is None
        ):
            raise DeviceManagedRuntimeError("device managed runtime requires one CUDA placement")
        self._backend = backend
        self._plan = plan
        self._kernel = kernel
        self._bound_plan = bound
        self._mutation_plan = mutation_plan
        self._placement = placement
        self._device = torch.device(f"cuda:{placement.device_index}")
        self._dtype = _torch_dtype(bound.control.buffer.dtype)
        if self._dtype is not torch.float32:
            raise DeviceManagedRuntimeError("mjwarp device managed runtime requires float32")
        self._num_envs = bound.num_envs
        self._all_rows = RowSelection.all(self._num_envs)
        self._max_episode_steps = max_episode_steps
        self._record_lifecycle = record_lifecycle
        self._trace_events: list[ManagedLifecyclePhase] = []
        self._last_trace: tuple[ManagedLifecyclePhase, ...] = ()
        self._last_step_diagnostics: BackendBatchDiagnostics | None = None
        self._last_reset_diagnostics: BackendBatchDiagnostics | None = None

        self._task_stream = cast(torch.cuda.Stream, torch.cuda.Stream(device=self._device))
        self._control_lease = DeviceBufferLease(
            f"{bound.backend_instance_id}:device-runtime-control"
        )
        self._reset_lease = DeviceBufferLease(f"{bound.backend_instance_id}:device-manager-reset")
        self._output_lease = DeviceBufferLease(f"{bound.backend_instance_id}:device-runtime-output")
        self._consumed_action_epochs: WeakKeyDictionary[DeviceBufferLease, int] = (
            WeakKeyDictionary()
        )

        binding = ManagedKernelBinding(
            task_fingerprint=plan.fingerprint,
            policy_abi_fingerprint=plan.policy_abi.fingerprint,
            num_envs=self._num_envs,
            dtype=bound.control.buffer.dtype,
            execution_profile=ExecutionProfile.DEVICE_RESIDENT,
            state_field_indices=tuple(
                (state_field.key, index) for index, state_field in enumerate(bound.state.fields)
            ),
            observation_buffer_indices=tuple(
                (group.key, index) for index, group in enumerate(plan.policy_abi.observation_groups)
            ),
            mutation_plan=mutation_plan,
        )
        kernel.bind(binding=binding)
        self._validate_kernel(kernel, plan)
        self._kernel_binding = binding
        self._task_state = kernel.create_task_state(
            num_envs=self._num_envs,
            dtype=self._dtype,
            device=self._device,
        )
        if self._task_state is None:
            raise DeviceManagedRuntimeError("device managed kernel create_task_state returned None")

        self._control = torch.empty(
            (self._num_envs, *bound.control.buffer.row_shape),
            dtype=self._dtype,
            device=self._device,
        )
        self._reward_contract = _runtime_contract(
            placement=placement, row_shape=(), dtype=bound.control.buffer.dtype
        )
        self._bool_contract = _runtime_contract(placement=placement, row_shape=(), dtype="bool")
        self._reward = torch.empty((self._num_envs,), dtype=self._dtype, device=self._device)
        self._terminated = torch.empty((self._num_envs,), dtype=torch.bool, device=self._device)
        self._truncated = torch.empty((self._num_envs,), dtype=torch.bool, device=self._device)
        self._done = torch.empty((self._num_envs,), dtype=torch.bool, device=self._device)
        self._final_observation_mask = torch.empty(
            (self._num_envs,), dtype=torch.bool, device=self._device
        )
        self._episode_steps = torch.empty((self._num_envs,), dtype=torch.int64, device=self._device)
        self._observation_keys, self._observation_contracts = self._observation_layout(plan)
        self._observations = tuple(
            torch.empty(
                (self._num_envs, *contract.row_shape), dtype=self._dtype, device=self._device
            )
            for contract in self._observation_contracts
        )
        self._terminal_observations = tuple(torch.empty_like(value) for value in self._observations)
        self._final_observations = tuple(torch.empty_like(value) for value in self._observations)
        self._initialized = False

    @staticmethod
    def _validate_kernel(kernel: DeviceManagedTaskKernel, plan: CompiledTaskPlan) -> None:
        try:
            executor_key = kernel.executor_key
        except AttributeError as exc:
            raise DeviceManagedRuntimeError("device managed kernel has no executor_key") from exc
        if not isinstance(executor_key, str) or executor_key != plan.executor_key:
            raise DeviceManagedRuntimeError(
                "device managed kernel executor_key does not match the compiled task plan"
            )
        # This is a cold-path structural check.  Device task math may retain
        # tensors/configuration, but never a backend/env owner it could use to
        # bypass StateBatch.
        for forbidden in ("backend", "env", "_backend", "_env", "model", "_model"):
            if forbidden in vars(kernel):
                raise DeviceManagedRuntimeError(
                    f"device managed kernel must not retain forbidden {forbidden!r} owner reference"
                )

    @staticmethod
    def _validate_bound_plan(
        *, backend: SimBackend, plan: CompiledTaskPlan, bound: BoundBackendPlan
    ) -> None:
        if not isinstance(bound, BoundBackendPlan):
            raise DeviceManagedRuntimeError("backend bind_task_io must return BoundBackendPlan")
        if (
            bound.backend_type != backend.backend_type
            or bound.execution_profile is not ExecutionProfile.DEVICE_RESIDENT
            or bound.num_envs != backend.num_envs
        ):
            raise DeviceManagedRuntimeError(
                "device backend bound plan has an incompatible identity"
            )
        if (
            bound.state.fields != plan.backend_io.state_fields
            or bound.control != plan.backend_io.control
        ):
            raise DeviceManagedRuntimeError(
                "device backend bound plan I/O differs from the compiled task plan"
            )

    @staticmethod
    def _validate_mutation_plan(
        *, bound: BoundBackendPlan, mutation_plan: BoundMutationPlan
    ) -> None:
        if not isinstance(mutation_plan, BoundMutationPlan):
            raise DeviceManagedRuntimeError(
                "backend bind_mutation_plan must return BoundMutationPlan"
            )
        try:
            mutation_plan.require_owner(
                backend_type=bound.backend_type,
                backend_instance_id=bound.backend_instance_id,
            )
        except BackendBatchContractError as exc:
            raise DeviceManagedRuntimeError(
                "device mutation plan owner differs from its bound backend plan"
            ) from exc
        if mutation_plan.num_envs != bound.num_envs:
            raise DeviceManagedRuntimeError(
                "device mutation plan row universe differs from backend"
            )

    @staticmethod
    def _observation_layout(
        plan: CompiledTaskPlan,
    ) -> tuple[tuple[str, ...], tuple[BufferContract, ...]]:
        channels = {channel.key: channel.buffer for channel in plan.output_channels}
        keys: list[str] = []
        contracts: list[BufferContract] = []
        for group in plan.policy_abi.observation_groups:
            try:
                contract = channels[f"obs:{group.key}"]
            except KeyError as exc:
                raise DeviceManagedRuntimeError(
                    f"compiled device plan lacks observation channel obs:{group.key}"
                ) from exc
            if (
                contract.row_shape != (group.width,)
                or contract.dtype != group.dtype
                or contract.placement.memory_space is not MemorySpace.DEVICE
                or not contract.dlpack_exportable
            ):
                raise DeviceManagedRuntimeError(
                    f"compiled device observation channel obs:{group.key} is incompatible"
                )
            keys.append(group.key)
            contracts.append(contract)
        return tuple(keys), tuple(contracts)

    @property
    def plan(self) -> CompiledTaskPlan:
        return self._plan

    @property
    def bound_plan(self) -> BoundBackendPlan:
        return self._bound_plan

    @property
    def policy_abi_snapshot(self) -> dict[str, Any]:
        return managed_policy_abi_snapshot(self._plan)

    @property
    def kernel_binding(self) -> ManagedKernelBinding:
        return self._kernel_binding

    @property
    def last_trace(self) -> tuple[ManagedLifecyclePhase, ...]:
        return self._last_trace

    @property
    def last_step_diagnostics(self) -> BackendBatchDiagnostics | None:
        return self._last_step_diagnostics

    @property
    def last_reset_diagnostics(self) -> BackendBatchDiagnostics | None:
        return self._last_reset_diagnostics

    def _begin_trace(self) -> None:
        self._trace_events.clear()

    def _trace(self, phase: ManagedLifecyclePhase) -> None:
        if self._record_lifecycle:
            self._trace_events.append(phase)

    def _finish_trace(self) -> tuple[ManagedLifecyclePhase, ...]:
        self._last_trace = tuple(self._trace_events)
        return self._last_trace

    def _require_backend_completion(
        self,
        *,
        diagnostics: BackendBatchDiagnostics,
        state: StateBatch,
    ) -> DeviceCompletion:
        completion = diagnostics.completion_event
        if not isinstance(completion, BackendCompletionEvent):
            raise DeviceManagedRuntimeError(
                "device backend lifecycle result lacks an explicit completion event"
            )
        if (
            completion.backend_type != self._bound_plan.backend_type
            or completion.placement != self._placement
        ):
            raise DeviceManagedRuntimeError("device backend completion identity is incompatible")
        if not isinstance(completion.handle, DeviceCompletion):
            raise DeviceManagedRuntimeError(
                "device backend completion handle is not a DeviceCompletion"
            )
        handle = completion.handle
        if handle.placement != self._placement:
            raise DeviceManagedRuntimeError("device backend completion placement is incompatible")
        if handle.owner_id != self._bound_plan.backend_instance_id:
            raise DeviceManagedRuntimeError(
                "device backend completion owner differs from the bound backend instance"
            )
        for field_index, state_field in enumerate(self._bound_plan.state.fields):
            raw_view = state.buffer_at(field_index).handle
            view = require_device_tensor_view(
                raw_view,
                contract=state_field.buffer,
                require_completion=True,
            )
            producer = view.require_completion()
            if (
                producer.placement != handle.placement
                or producer.owner_id != handle.owner_id
                or producer.epoch != handle.epoch
                or producer.event is not handle.event
            ):
                raise DeviceManagedRuntimeError(
                    f"device state field {state_field.key!r} completion differs from lifecycle diagnostics"
                )
        return handle

    def _validate_state(self, state: StateBatch, *, phase: StateBatchPhase) -> None:
        if not isinstance(state, StateBatch):
            raise DeviceManagedRuntimeError("device backend lifecycle result lacks StateBatch")
        try:
            state.plan.require_compatible(self._bound_plan)
        except BackendBatchContractError as exc:
            raise DeviceManagedRuntimeError(
                "device StateBatch belongs to another backend plan"
            ) from exc
        if state.phase is not phase or state.rows != self._all_rows:
            raise DeviceManagedRuntimeError("device StateBatch phase or rows are incompatible")
        state.assert_valid()

    def _require_action(self, action: DeviceTensorView) -> torch.Tensor:
        view = require_device_tensor_view(
            action,
            contract=self._bound_plan.control.buffer,
            require_completion=True,
        )
        expected = (self._num_envs, *self._bound_plan.control.buffer.row_shape)
        if view.shape != expected:
            raise DeviceManagedRuntimeError(
                f"device actions require shape {expected}, got {view.shape}"
            )
        previous = self._consumed_action_epochs.get(view.lease)
        if previous is not None and view.epoch <= previous:
            raise DeviceBufferContractError(
                "device policy action lease epoch was already consumed; publish a new event"
            )
        self._consumed_action_epochs[view.lease] = view.epoch
        return view.torch()

    def _build_reset_batch(self, payload: DeviceResetPayload) -> DeviceResetMutationBatch:
        if not isinstance(payload, DeviceResetPayload):
            raise DeviceManagedRuntimeError(
                "device kernel prepare_reset returned an invalid payload"
            )
        mask = payload.active_mask
        if (
            mask.device != self._device
            or mask.dtype is not torch.bool
            or tuple(mask.shape) != (self._num_envs,)
            or not mask.is_contiguous()
        ):
            raise DeviceManagedRuntimeError(
                "device reset active mask must be a contiguous CUDA bool all-world buffer"
            )
        if len(payload.values) != len(self._mutation_plan.specs):
            raise DeviceManagedRuntimeError(
                "device reset payload must provide every bound mutation value exactly once"
            )
        completion = DeviceCompletion.record(
            placement=self._placement,
            owner_id=self._reset_lease.owner_id,
            epoch=self._reset_lease.epoch,
            stream=self._task_stream,
        )
        mask_view = DeviceTensorView(
            tensor_handle=mask,
            contract=_mask_contract(placement=self._placement),
            lease=self._reset_lease,
            completion=completion,
        )
        values: list[Any] = []
        for field_index, (spec, tensor) in enumerate(
            zip(self._mutation_plan.specs, payload.values, strict=True)
        ):
            expected_shape = (self._num_envs, *spec.value_buffer.row_shape)
            if (
                tensor.device != self._device
                or tensor.dtype is not self._dtype
                or tuple(tensor.shape) != expected_shape
                or not tensor.is_contiguous()
            ):
                raise DeviceManagedRuntimeError(
                    f"device reset value {field_index} differs from its cold-bound mutation contract"
                )
            view = DeviceTensorView(
                tensor_handle=tensor,
                contract=spec.value_buffer,
                lease=self._reset_lease,
                completion=completion,
            )
            values.append(
                MutationValueBatch(
                    plan=self._mutation_plan,
                    field_index=field_index,
                    rows=self._all_rows,
                    buffer=BufferView(
                        handle=view,
                        shape=expected_shape,
                        contract=spec.value_buffer,
                    ),
                )
            )
        mutation = TypedBackendMutationBatch(
            plan=self._mutation_plan,
            rows=self._all_rows,
            state=SimulationStateMutationBatch(tuple(values)),
        )
        return DeviceResetMutationBatch(
            plan=self._mutation_plan,
            rows=self._all_rows,
            mutation=mutation,
            active_mask=BufferView(
                handle=mask_view,
                shape=(self._num_envs,),
                contract=_mask_contract(placement=self._placement),
            ),
        )

    def _control_batch(self) -> ControlBatch:
        completion = DeviceCompletion.record(
            placement=self._placement,
            owner_id=self._control_lease.owner_id,
            epoch=self._control_lease.epoch,
            stream=self._task_stream,
        )
        view = DeviceTensorView(
            tensor_handle=self._control,
            contract=self._bound_plan.control.buffer,
            lease=self._control_lease,
            completion=completion,
        )
        return ControlBatch(
            plan=self._bound_plan,
            rows=self._all_rows,
            buffer=BufferView(
                handle=view,
                shape=tuple(int(value) for value in self._control.shape),
                contract=self._bound_plan.control.buffer,
            ),
        )

    def _output_completion(self) -> DeviceCompletion:
        return DeviceCompletion.record(
            placement=self._placement,
            owner_id=self._output_lease.owner_id,
            epoch=self._output_lease.epoch,
            stream=self._task_stream,
        )

    def _transition_view(
        self, tensor: torch.Tensor, contract: BufferContract, completion: DeviceCompletion
    ) -> DeviceTensorView:
        return DeviceTensorView(
            tensor_handle=tensor,
            contract=contract,
            lease=self._output_lease,
            completion=completion,
        )

    def _build_transition(
        self,
        *,
        completion: DeviceCompletion,
        step_diagnostics: BackendBatchDiagnostics | None,
        reset_diagnostics: BackendBatchDiagnostics | None,
    ) -> DeviceTransition:
        def named(values: tuple[torch.Tensor, ...]) -> tuple[DeviceTransitionBuffer, ...]:
            return tuple(
                DeviceTransitionBuffer(
                    key=key,
                    view=self._transition_view(value, contract, completion),
                )
                for key, contract, value in zip(
                    self._observation_keys,
                    self._observation_contracts,
                    values,
                    strict=True,
                )
            )

        return DeviceTransition(
            plan_fingerprint=self._plan.fingerprint,
            observations=named(self._observations),
            terminal_observations=named(self._terminal_observations),
            final_observations=named(self._final_observations),
            reward=self._transition_view(self._reward, self._reward_contract, completion),
            terminated=self._transition_view(self._terminated, self._bool_contract, completion),
            truncated=self._transition_view(self._truncated, self._bool_contract, completion),
            final_observation_mask=self._transition_view(
                self._final_observation_mask,
                self._bool_contract,
                completion,
            ),
            completion=completion,
            trace=self._finish_trace(),
            step_diagnostics=step_diagnostics,
            reset_diagnostics=reset_diagnostics,
        )

    def reset(self) -> DeviceTransition:
        """Run an initial all-world CUDA reset and return device observations."""

        self._last_step_diagnostics = None
        self._output_lease.invalidate()
        self._reset_lease.invalidate()
        self._begin_trace()
        self._trace(ManagedLifecyclePhase.INITIAL_RESET_REQUEST)
        with torch.cuda.stream(self._task_stream):
            all_mask = self._done
            all_mask.fill_(True)
            payload = self._kernel.prepare_reset(active_mask=all_mask, task_state=self._task_state)
            reset_batch = self._build_reset_batch(payload)
        self._trace(ManagedLifecyclePhase.RESET_BACKEND)
        result = self._backend.reset_batch(
            self._bound_plan,
            self._all_rows,
            mutation_batch=reset_batch,
        )
        if not isinstance(result, BackendResetResult):
            raise DeviceManagedRuntimeError(
                "device backend reset_batch must return BackendResetResult"
            )
        self._validate_state(result.reset_state, phase=StateBatchPhase.RESET)
        reset_completion = self._require_backend_completion(
            diagnostics=result.diagnostics,
            state=result.reset_state,
        )
        self._last_reset_diagnostics = result.diagnostics
        self._trace(ManagedLifecyclePhase.TASK_STATE_RESET)
        with torch.cuda.stream(self._task_stream):
            reset_completion.wait(self._task_stream)
            self._kernel.complete_reset(
                active_mask=payload.active_mask,
                state=result.reset_state,
                task_state=self._task_state,
                observation_buffers=self._observations,
            )
            for terminal, observation in zip(
                self._terminal_observations, self._observations, strict=True
            ):
                terminal.copy_(observation, non_blocking=True)
            self._reward.zero_()
            self._terminated.zero_()
            self._truncated.zero_()
            self._final_observation_mask.zero_()
            self._episode_steps.zero_()
            self._trace(ManagedLifecyclePhase.OBSERVATION)
            completion = self._output_completion()
        self._initialized = True
        self._trace(ManagedLifecyclePhase.COMPLETE)
        return self._build_transition(
            completion=completion,
            step_diagnostics=None,
            reset_diagnostics=result.diagnostics,
        )

    def step(self, actions: DeviceTensorView) -> DeviceTransition:
        """Advance one all-device terminal/autoreset transition."""

        if not self._initialized:
            raise DeviceManagedRuntimeError("device managed runtime requires reset() before step()")
        action = self._require_action(actions)
        action_completion = actions.require_completion()
        self._output_lease.invalidate()
        self._control_lease.invalidate()
        self._begin_trace()
        self._trace(ManagedLifecyclePhase.ACTION)
        with torch.cuda.stream(self._task_stream):
            action_completion.wait(self._task_stream)
            self._kernel.apply_action(
                actions=action,
                task_state=self._task_state,
                control_out=self._control,
            )
            control_batch = self._control_batch()
        self._trace(ManagedLifecyclePhase.PRE_PHYSICS)
        self._trace(ManagedLifecyclePhase.PHYSICS)
        step_result = self._backend.step_batch(
            self._bound_plan,
            control_batch,
            nsteps=self._bound_plan.control.physics_substeps_per_control,
        )
        if not isinstance(step_result, BackendStepResult):
            raise DeviceManagedRuntimeError(
                "device backend step_batch must return BackendStepResult"
            )
        terminal = step_result.terminal_state
        self._validate_state(terminal, phase=StateBatchPhase.TERMINAL)
        step_completion = self._require_backend_completion(
            diagnostics=step_result.diagnostics,
            state=terminal,
        )
        self._last_step_diagnostics = step_result.diagnostics

        self._reset_lease.invalidate()
        self._trace(ManagedLifecyclePhase.TERMINATION)
        with torch.cuda.stream(self._task_stream):
            step_completion.wait(self._task_stream)
            self._reward.zero_()
            self._terminated.zero_()
            self._kernel.evaluate_terminal(
                state=terminal,
                task_state=self._task_state,
                reward_out=self._reward,
                terminated_out=self._terminated,
                terminal_observation_buffers=self._terminal_observations,
            )
            self._trace(ManagedLifecyclePhase.REWARD)
            self._trace(ManagedLifecyclePhase.METRIC)
            self._trace(ManagedLifecyclePhase.TERMINAL_OBSERVATION)
            self._episode_steps.add_(1)
            self._truncated.zero_()
            if self._max_episode_steps is not None:
                torch.ge(self._episode_steps, self._max_episode_steps, out=self._truncated)
            self._trace(ManagedLifecyclePhase.TIMEOUT)
            torch.logical_or(self._terminated, self._truncated, out=self._done)
            self._final_observation_mask.copy_(self._done, non_blocking=True)
            for final, terminal_observation in zip(
                self._final_observations, self._terminal_observations, strict=True
            ):
                torch.where(self._done[:, None], terminal_observation, final, out=final)
            # The trace deliberately records a masked barrier even when no
            # row is done.  Looking at ``any(done)`` on the host would violate
            # the device-resident contract; the final-observation mask is the
            # explicit device-side record of actual membership.
            self._trace(ManagedLifecyclePhase.FINAL_OBSERVATION)
            self._trace(ManagedLifecyclePhase.AUTORESET)
            self._trace(ManagedLifecyclePhase.RESET_REQUEST)
            payload = self._kernel.prepare_reset(
                active_mask=self._done, task_state=self._task_state
            )
            reset_batch = self._build_reset_batch(payload)
        self._trace(ManagedLifecyclePhase.RESET_BACKEND)
        reset_result = self._backend.reset_batch(
            self._bound_plan,
            self._all_rows,
            mutation_batch=reset_batch,
        )
        if not isinstance(reset_result, BackendResetResult):
            raise DeviceManagedRuntimeError(
                "device backend reset_batch must return BackendResetResult"
            )
        self._validate_state(reset_result.reset_state, phase=StateBatchPhase.RESET)
        reset_completion = self._require_backend_completion(
            diagnostics=reset_result.diagnostics,
            state=reset_result.reset_state,
        )
        self._last_reset_diagnostics = reset_result.diagnostics
        self._trace(ManagedLifecyclePhase.TASK_STATE_RESET)
        with torch.cuda.stream(self._task_stream):
            reset_completion.wait(self._task_stream)
            self._kernel.complete_reset(
                active_mask=payload.active_mask,
                state=reset_result.reset_state,
                task_state=self._task_state,
                observation_buffers=self._observations,
            )
            self._episode_steps.masked_fill_(payload.active_mask, 0)
            self._trace(ManagedLifecyclePhase.OBSERVATION)
            completion = self._output_completion()
        self._trace(ManagedLifecyclePhase.COMPLETE)
        return self._build_transition(
            completion=completion,
            step_diagnostics=step_result.diagnostics,
            reset_diagnostics=reset_result.diagnostics,
        )


__all__ = [
    "DeviceManagedRuntime",
    "DeviceManagedRuntimeError",
    "DeviceManagedTaskKernel",
    "DeviceResetPayload",
    "DeviceTransition",
    "DeviceTransitionBuffer",
]
