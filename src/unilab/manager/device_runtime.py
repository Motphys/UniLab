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

from dataclasses import dataclass, field, replace
from typing import Any, Protocol, cast, runtime_checkable
from weakref import WeakKeyDictionary

import torch

from unilab.base.backend import (
    BackendBatchContractError,
    BackendBatchDiagnostics,
    BackendCompletionEvent,
    BackendMutationPerformanceDiagnostics,
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
    DeviceGraphDiagnostics,
    DevicePhaseTimingError,
    DeviceResetMutationBatch,
    DeviceResetPhaseTimingDiagnostics,
    DeviceResetPhaseTimingSampleToken,
    DeviceResetPhaseTimingSession,
    DeviceResetPhaseTimingTrace,
    DeviceTensorView,
    ExecutionProfile,
    MemorySpace,
    ModelParameterMutationBatch,
    MutationValueBatch,
    RowSelection,
    SimulationStateMutationBatch,
    StateBatch,
    StateBatchPhase,
    TypedBackendMutationBatch,
    require_device_tensor_view,
)
from unilab.base.backend.base import SimBackend
from unilab.base.backend.mutation import (
    BoundMutationPlan,
    MutationEntityKind,
    MutationTargetKind,
)
from unilab.dr.keyed_rng import (
    KeyedRandomSpec,
    KeyedRandomStream,
    KeyedRandomTrafficDiagnostics,
)

from .fingerprint import managed_policy_abi_snapshot, validate_compiled_plan_fingerprints
from .plan import CompiledMutationEvent, CompiledTaskPlan
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
    metrics: tuple[DeviceTransitionBuffer, ...]
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
        if not isinstance(self.metrics, tuple) or any(
            not isinstance(buffer, DeviceTransitionBuffer) for buffer in self.metrics
        ):
            raise DeviceManagedRuntimeError("device transition metrics must contain typed buffers")
        metric_keys = tuple(buffer.key for buffer in self.metrics)
        if len(set(metric_keys)) != len(metric_keys):
            raise DeviceManagedRuntimeError("device transition metrics have duplicate keys")
        if not isinstance(self.completion, DeviceCompletion):
            raise DeviceManagedRuntimeError("device transition completion is invalid")
        all_views = (
            *(buffer.view for buffer in self.observations),
            *(buffer.view for buffer in self.terminal_observations),
            *(buffer.view for buffer in self.final_observations),
            *(buffer.view for buffer in self.metrics),
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

    def metric(self, key: str) -> DeviceTensorView:
        """Return one task metric without materializing it on the host."""

        for buffer in self.metrics:
            if buffer.key == key:
                return buffer.view
        raise DeviceManagedRuntimeError(f"device transition has no metric {key!r}")


@dataclass(frozen=True)
class DeviceResetValue:
    """One deterministic kernel-owned reset field in canonical plan coordinates."""

    field_index: int
    tensor: torch.Tensor = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.field_index, bool)
            or not isinstance(self.field_index, int)
            or self.field_index < 0
        ):
            raise DeviceManagedRuntimeError("device reset field_index must be non-negative")
        if not isinstance(self.tensor, torch.Tensor):
            raise DeviceManagedRuntimeError("device reset value must contain a tensor")


@dataclass(frozen=True)
class DeviceResetPayload:
    """Sparse deterministic CUDA reset staging supplied by a task kernel.

    Random Event fields are manager-owned and therefore must not appear here.
    The runtime validates exact deterministic coverage, samples its keyed
    streams, and constructs one public :class:`DeviceResetMutationBatch`.
    """

    active_mask: torch.Tensor = field(repr=False, compare=False)
    values: tuple[DeviceResetValue, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.active_mask, torch.Tensor):
            raise DeviceManagedRuntimeError("device reset payload active_mask must be a tensor")
        if not isinstance(self.values, tuple) or any(
            not isinstance(value, DeviceResetValue) for value in self.values
        ):
            raise DeviceManagedRuntimeError(
                "device reset payload values must contain DeviceResetValue descriptors"
            )


@dataclass(frozen=True)
class DeviceMutationEventBinding:
    """Backend-bound shape and placement for one compiled random Event."""

    event: CompiledMutationEvent
    value_contract: BufferContract
    random_spec: KeyedRandomSpec

    def __post_init__(self) -> None:
        if not isinstance(self.event, CompiledMutationEvent):
            raise DeviceManagedRuntimeError("device Event binding requires a compiled Event")
        if not isinstance(self.value_contract, BufferContract):
            raise DeviceManagedRuntimeError("device Event binding requires a value contract")
        if not isinstance(self.random_spec, KeyedRandomSpec):
            raise DeviceManagedRuntimeError("device Event binding requires a keyed random spec")
        if (
            self.random_spec.term_key != self.event.term_key
            or self.random_spec.term_version != self.event.term_version
            or self.random_spec.row_shape != self.value_contract.row_shape
            or self.random_spec.distribution is not self.event.distribution
            or self.random_spec.parameters != self.event.parameters
            or self.random_spec.correlation is not self.event.correlation
            or self.random_spec.algorithm != self.event.algorithm
        ):
            raise DeviceManagedRuntimeError(
                "device Event binding differs from its compiled random semantics"
            )


@dataclass(frozen=True)
class DeviceRuntimeBuffer:
    """One explicitly registered runtime/executor-owned CUDA buffer."""

    name: str
    tensor: torch.Tensor = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise DeviceManagedRuntimeError("device runtime buffer name must be non-empty")
        if not isinstance(self.tensor, torch.Tensor) or not self.tensor.is_cuda:
            raise DeviceManagedRuntimeError("device runtime buffer must be a CUDA tensor")
        if self.tensor.ndim == 0 or not self.tensor.is_contiguous() or self.tensor.numel() == 0:
            raise DeviceManagedRuntimeError(
                "device runtime buffer must be non-empty, non-scalar, and contiguous"
            )


@dataclass(frozen=True)
class DeviceRuntimeBufferAddress:
    """Host-visible identity metadata for one registered CUDA allocation."""

    name: str
    address: int
    shape: tuple[int, ...]
    dtype: str
    device: str
    nbytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise DeviceManagedRuntimeError("device buffer address name must be non-empty")
        if isinstance(self.address, bool) or not isinstance(self.address, int) or self.address <= 0:
            raise DeviceManagedRuntimeError("device buffer address must be positive")
        if (
            not isinstance(self.shape, tuple)
            or not self.shape
            or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in self.shape
            )
        ):
            raise DeviceManagedRuntimeError("device buffer address shape is invalid")
        if not isinstance(self.dtype, str) or not self.dtype.strip():
            raise DeviceManagedRuntimeError("device buffer address dtype must be non-empty")
        if not isinstance(self.device, str) or not self.device.startswith("cuda:"):
            raise DeviceManagedRuntimeError("device buffer address requires a CUDA device")
        if isinstance(self.nbytes, bool) or not isinstance(self.nbytes, int) or self.nbytes <= 0:
            raise DeviceManagedRuntimeError("device buffer address nbytes must be positive")


@dataclass(frozen=True)
class DeviceRuntimeStateEpoch:
    """Latest lease/completion epoch observed for one backend state pack."""

    name: str
    phase: StateBatchPhase
    lease_epoch: int
    completion_epoch: int
    owner_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise DeviceManagedRuntimeError("device state epoch name must be non-empty")
        if not isinstance(self.phase, StateBatchPhase):
            raise DeviceManagedRuntimeError("device state epoch phase is invalid")
        for label, value in (
            ("lease_epoch", self.lease_epoch),
            ("completion_epoch", self.completion_epoch),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DeviceManagedRuntimeError(f"device state {label} must be non-negative")
        if self.lease_epoch != self.completion_epoch:
            raise DeviceManagedRuntimeError("device state lease/completion epochs must match")
        if not isinstance(self.owner_id, str) or not self.owner_id.strip():
            raise DeviceManagedRuntimeError("device state epoch owner must be non-empty")


@dataclass(frozen=True)
class DeviceRuntimeTrafficDiagnostics:
    """Cumulative public accounting for device lifecycle barriers."""

    policy_steps: int = 0
    step_barriers: int = 0
    reset_barriers: int = 0
    host_to_device_transfers: int = 0
    device_to_host_transfers: int = 0
    host_to_device_bytes: int = 0
    device_to_host_bytes: int = 0
    global_synchronizations: int = 0
    backend_allocations: int = 0
    state_materializations: int = 0
    dynamic_getter_calls: int = 0
    selector_resolutions: int = 0
    asset_metadata_reads: int = 0
    registry_lookups: int = 0
    instrumentation_complete: bool = True

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if name == "instrumentation_complete":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DeviceManagedRuntimeError(
                    f"device traffic diagnostic {name} must be non-negative"
                )
        if not isinstance(self.instrumentation_complete, bool):
            raise DeviceManagedRuntimeError(
                "device traffic instrumentation_complete must be a bool"
            )


@dataclass(frozen=True)
class DeviceRuntimeStabilityDiagnostics:
    """Opt-in registered-buffer, state-pack, epoch, and traffic snapshot."""

    buffers: tuple[DeviceRuntimeBufferAddress, ...]
    state_buffers: tuple[DeviceRuntimeBufferAddress, ...]
    state_epochs: tuple[DeviceRuntimeStateEpoch, ...]
    warm_numeric_allocations: int
    address_churn: int
    observations: int
    output_epoch: int
    control_epoch: int
    reset_epoch: int
    traffic: DeviceRuntimeTrafficDiagnostics
    graph: DeviceGraphDiagnostics
    instrumentation_complete: bool

    def __post_init__(self) -> None:
        for label, values in (("buffers", self.buffers), ("state_buffers", self.state_buffers)):
            if not isinstance(values, tuple) or any(
                not isinstance(value, DeviceRuntimeBufferAddress) for value in values
            ):
                raise DeviceManagedRuntimeError(f"device stability {label} is invalid")
            names = tuple(value.name for value in values)
            if names != tuple(sorted(names)) or len(set(names)) != len(names):
                raise DeviceManagedRuntimeError(
                    f"device stability {label} names must be canonical and unique"
                )
        if not isinstance(self.state_epochs, tuple) or any(
            not isinstance(value, DeviceRuntimeStateEpoch) for value in self.state_epochs
        ):
            raise DeviceManagedRuntimeError("device stability state_epochs is invalid")
        epoch_names = tuple(value.name for value in self.state_epochs)
        if epoch_names != tuple(sorted(epoch_names)) or len(set(epoch_names)) != len(epoch_names):
            raise DeviceManagedRuntimeError(
                "device stability state epoch names must be canonical and unique"
            )
        for label, value in (
            ("warm_numeric_allocations", self.warm_numeric_allocations),
            ("address_churn", self.address_churn),
            ("observations", self.observations),
            ("output_epoch", self.output_epoch),
            ("control_epoch", self.control_epoch),
            ("reset_epoch", self.reset_epoch),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DeviceManagedRuntimeError(f"device stability {label} must be non-negative")
        if not isinstance(self.traffic, DeviceRuntimeTrafficDiagnostics):
            raise DeviceManagedRuntimeError("device stability traffic is invalid")
        if not isinstance(self.graph, DeviceGraphDiagnostics):
            raise DeviceManagedRuntimeError("device stability graph diagnostics are invalid")
        if (
            not self.graph.instrumentation_complete
            or self.graph.eager_fallback_count != 0
            or not self.graph.active_keys
        ):
            raise DeviceManagedRuntimeError(
                "device stability requires complete graph-only backend diagnostics"
            )
        if not isinstance(self.instrumentation_complete, bool):
            raise DeviceManagedRuntimeError(
                "device stability instrumentation_complete must be a bool"
            )


@runtime_checkable
class DeviceRuntimeBufferProvider(Protocol):
    """Explicit executor-owned CUDA buffer registration for diagnostics."""

    def device_runtime_buffers(self, *, task_state: object) -> tuple[DeviceRuntimeBuffer, ...]: ...


def _device_buffer_address(buffer: DeviceRuntimeBuffer) -> DeviceRuntimeBufferAddress:
    tensor = buffer.tensor
    return DeviceRuntimeBufferAddress(
        name=buffer.name,
        address=int(tensor.data_ptr()),
        shape=tuple(int(dim) for dim in tensor.shape),
        dtype=str(tensor.dtype).removeprefix("torch."),
        device=str(tensor.device),
        nbytes=int(tensor.numel() * tensor.element_size()),
    )


class _DeviceRuntimeStabilityMonitor:
    """Fail-closed monitor over explicitly registered CUDA allocations."""

    def __init__(self) -> None:
        self._baseline_buffers: tuple[DeviceRuntimeBufferAddress, ...] | None = None
        self._state_buffers: dict[str, DeviceRuntimeBufferAddress] = {}
        self._state_epochs: dict[str, DeviceRuntimeStateEpoch] = {}
        self._warm_numeric_allocations = 0
        self._address_churn = 0
        self._observations = 0

    @staticmethod
    def _canonical(
        buffers: tuple[DeviceRuntimeBuffer, ...], *, context: str
    ) -> tuple[DeviceRuntimeBufferAddress, ...]:
        addresses = tuple(
            sorted(
                (_device_buffer_address(buffer) for buffer in buffers), key=lambda item: item.name
            )
        )
        names = tuple(item.name for item in addresses)
        if not addresses or len(set(names)) != len(names):
            raise DeviceManagedRuntimeError(
                f"{context} requires non-empty, uniquely named device buffers"
            )
        return addresses

    def arm(self, buffers: tuple[DeviceRuntimeBuffer, ...]) -> None:
        if self._baseline_buffers is not None:
            raise DeviceManagedRuntimeError("device stability monitor may only arm once")
        self._baseline_buffers = self._canonical(
            buffers, context="device runtime stability baseline"
        )

    def observe_buffers(self, buffers: tuple[DeviceRuntimeBuffer, ...]) -> None:
        if self._baseline_buffers is None:
            raise DeviceManagedRuntimeError("device stability monitor has not been armed")
        actual = self._canonical(buffers, context="device runtime stability observation")
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
        raise DeviceManagedRuntimeError(
            "device warm buffer stability violated: "
            f"added_or_removed={added_or_removed!r}, changed={changed!r}"
        )

    def observe_state(self, state: StateBatch) -> None:
        state.assert_valid()
        for field_index, state_field in enumerate(state.plan.state.fields):
            view = require_device_tensor_view(
                state.buffer_at(field_index).handle,
                contract=state_field.buffer,
                require_completion=True,
            )
            name = f"backend.state.{state_field.key}"
            address = _device_buffer_address(DeviceRuntimeBuffer(name=name, tensor=view.torch()))
            previous = self._state_buffers.get(name)
            if previous is None:
                self._state_buffers[name] = address
            elif previous != address:
                self._address_churn += 1
                raise DeviceManagedRuntimeError(
                    f"device backend StateBatch address changed after warmup: {name!r}"
                )
            completion = view.require_completion()
            self._state_epochs[name] = DeviceRuntimeStateEpoch(
                name=name,
                phase=state.phase,
                lease_epoch=view.epoch,
                completion_epoch=completion.epoch,
                owner_id=completion.owner_id,
            )

    def snapshot(
        self,
        *,
        output_epoch: int,
        control_epoch: int,
        reset_epoch: int,
        traffic: DeviceRuntimeTrafficDiagnostics,
        graph: DeviceGraphDiagnostics,
    ) -> DeviceRuntimeStabilityDiagnostics:
        if self._baseline_buffers is None:
            raise DeviceManagedRuntimeError("device stability monitor has not been armed")
        complete = traffic.instrumentation_complete and bool(self._state_buffers)
        return DeviceRuntimeStabilityDiagnostics(
            buffers=self._baseline_buffers,
            state_buffers=tuple(sorted(self._state_buffers.values(), key=lambda item: item.name)),
            state_epochs=tuple(sorted(self._state_epochs.values(), key=lambda item: item.name)),
            warm_numeric_allocations=self._warm_numeric_allocations,
            address_churn=self._address_churn,
            observations=self._observations,
            output_epoch=output_epoch,
            control_epoch=control_epoch,
            reset_epoch=reset_epoch,
            traffic=traffic,
            graph=graph,
            instrumentation_complete=complete,
        )


class DeviceManagedTaskKernel(Protocol):
    """Torch CUDA task-math ABI consumed by :class:`DeviceManagedRuntime`."""

    executor_key: str
    metric_keys: tuple[str, ...]

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
        metric_buffers: tuple[torch.Tensor, ...],
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
        run_seed: int = 0,
        record_lifecycle: bool = False,
        stability_buffer_provider: DeviceRuntimeBufferProvider | None = None,
    ) -> None:
        if not isinstance(backend, SimBackend):
            raise DeviceManagedRuntimeError("device managed runtime requires a SimBackend")
        if not isinstance(plan, CompiledTaskPlan):
            raise DeviceManagedRuntimeError("device managed runtime requires a CompiledTaskPlan")
        validate_compiled_plan_fingerprints(plan)
        if plan.backend_io.execution_profile is not ExecutionProfile.DEVICE_RESIDENT:
            raise DeviceManagedRuntimeError(
                "device managed runtime only supports device_resident plans"
            )
        if not isinstance(record_lifecycle, bool):
            raise DeviceManagedRuntimeError("record_lifecycle must be a bool")
        if stability_buffer_provider is not None and not isinstance(
            stability_buffer_provider, DeviceRuntimeBufferProvider
        ):
            raise DeviceManagedRuntimeError(
                "device stability provider must implement DeviceRuntimeBufferProvider"
            )
        if stability_buffer_provider is not None and stability_buffer_provider is not kernel:
            raise DeviceManagedRuntimeError(
                "device stability provider must be the task kernel owner"
            )
        if max_episode_steps is not None and (
            isinstance(max_episode_steps, bool)
            or not isinstance(max_episode_steps, int)
            or max_episode_steps <= 0
        ):
            raise DeviceManagedRuntimeError("max_episode_steps must be a positive integer or None")
        if isinstance(run_seed, bool) or not isinstance(run_seed, int):
            raise DeviceManagedRuntimeError("device managed runtime run_seed must be an integer")
        metric_keys = self._validate_kernel(kernel, plan)

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
        self._reset_phase_timing_session: DeviceResetPhaseTimingSession | None = None
        if self._dtype is not torch.float32:
            raise DeviceManagedRuntimeError("mjwarp device managed runtime requires float32")
        self._num_envs = bound.num_envs
        event_bindings: list[DeviceMutationEventBinding] = []
        event_streams: list[KeyedRandomStream] = []
        for event in plan.mutation_events:
            try:
                spec = mutation_plan.specs[event.mutation_index]
            except IndexError as exc:  # pragma: no cover - plan validation guards this.
                raise DeviceManagedRuntimeError(
                    "compiled Event mutation is absent from the bound backend plan"
                ) from exc
            source = plan.mutation_specs[event.mutation_index]
            if source.target.entity_kind is MutationEntityKind.GLOBAL:
                expected_entity_count = 0
                expected_row_shape = source.value_template.row_shape
            else:
                selector = source.target.selector_spec
                if (
                    selector is None or not selector.entity_ids
                ):  # pragma: no cover - plan invariant.
                    raise DeviceManagedRuntimeError(
                        f"compiled Event {event.term_key!r} has no selector binding"
                    )
                expected_entity_count = len(selector.entity_ids)
                expected_row_shape = (expected_entity_count, *source.value_template.row_shape)
            expected_value_contract = replace(
                source.value_template,
                row_shape=expected_row_shape,
            )
            if (
                spec.term_key != event.term_key
                or spec.target.target_key != source.target.target_key
                or spec.target.target_kind is not source.target.target_kind
                or spec.target.entity_kind is not source.target.entity_kind
                or spec.target.field_kind is not source.target.field_kind
                or len(spec.target.entity_ids) != expected_entity_count
                or spec.trigger is not event.trigger
                or spec.commit_phase is not event.commit_phase
                or spec.operation is not source.operation
                or spec.baseline is not source.baseline
                or spec.persistence is not source.persistence
                or spec.recompute is not source.recompute
                or spec.target.target_kind
                not in {
                    MutationTargetKind.MODEL_PARAMETER,
                    MutationTargetKind.SIMULATION_STATE,
                }
                or spec.value_buffer != expected_value_contract
                or spec.value_buffer.placement != placement
            ):
                raise DeviceManagedRuntimeError(
                    f"bound mutation Event {event.term_key!r} is incompatible with its plan"
                )
            random_spec = KeyedRandomSpec(
                term_key=event.term_key,
                term_version=event.term_version,
                row_shape=spec.value_buffer.row_shape,
                distribution=event.distribution,
                correlation=event.correlation,
                parameters=event.parameters,
                algorithm=event.algorithm,
            )
            event_bindings.append(
                DeviceMutationEventBinding(
                    event=event,
                    value_contract=spec.value_buffer,
                    random_spec=random_spec,
                )
            )
            event_streams.append(
                KeyedRandomStream(
                    random_spec,
                    run_seed=run_seed,
                    num_envs=self._num_envs,
                    device=self._device,
                    dtype=_torch_dtype(spec.value_buffer.dtype),
                )
            )
        self._event_bindings = tuple(event_bindings)
        self._event_streams = tuple(event_streams)
        self._event_mutation_indices = tuple(
            binding.event.mutation_index for binding in self._event_bindings
        )
        unsupported_targets = tuple(
            spec.term_key
            for spec in mutation_plan.specs
            if spec.target.target_kind
            not in {
                MutationTargetKind.MODEL_PARAMETER,
                MutationTargetKind.SIMULATION_STATE,
            }
        )
        if unsupported_targets:
            raise DeviceManagedRuntimeError(
                "device reset runtime does not support mutation targets for: "
                + ", ".join(unsupported_targets)
            )
        self._model_mutation_indices = tuple(
            index
            for index, spec in enumerate(mutation_plan.specs)
            if spec.target.target_kind is MutationTargetKind.MODEL_PARAMETER
        )
        self._state_mutation_indices = tuple(
            index
            for index, spec in enumerate(mutation_plan.specs)
            if spec.target.target_kind is MutationTargetKind.SIMULATION_STATE
        )
        event_index_set = frozenset(self._event_mutation_indices)
        self._deterministic_mutation_indices = tuple(
            index for index in range(len(mutation_plan.specs)) if index not in event_index_set
        )
        self._all_rows = RowSelection.all(self._num_envs)
        self._max_episode_steps = max_episode_steps
        self._record_lifecycle = record_lifecycle
        self._stability_buffer_provider = stability_buffer_provider
        self._stability_monitor = (
            None if stability_buffer_provider is None else _DeviceRuntimeStabilityMonitor()
        )
        self._traffic = DeviceRuntimeTrafficDiagnostics()
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
        self._control_event = cast(torch.cuda.Event, torch.cuda.Event(enable_timing=False))
        self._reset_event = cast(torch.cuda.Event, torch.cuda.Event(enable_timing=False))
        self._output_event = cast(torch.cuda.Event, torch.cuda.Event(enable_timing=False))
        # RSL-RL initializes random episode lengths on its policy/default
        # stream.  The runtime consumes the resulting buffer on
        # ``_task_stream``.  Keep a dedicated event for this low-frequency
        # handoff rather than relying on CUDA default-stream semantics.
        self._episode_length_input_event = cast(
            torch.cuda.Event, torch.cuda.Event(enable_timing=False)
        )
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
            event_mutation_indices=self._event_mutation_indices,
        )
        kernel.bind(binding=binding)
        if self._validate_kernel(kernel, plan) != metric_keys:
            raise DeviceManagedRuntimeError("device managed kernel metric keys changed during bind")
        self._kernel_binding = binding
        self._metric_keys = metric_keys
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
        self._metrics = tuple(
            torch.empty((self._num_envs,), dtype=self._dtype, device=self._device)
            for _ in self._metric_keys
        )
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
        # Construction may run on a policy/default stream while every warm
        # lifecycle operation runs on ``_task_stream``.  Publish all cold-path
        # tensor initialization explicitly before the first reset instead of
        # relying on CUDA default-stream ordering.
        self._cold_init_event = cast(torch.cuda.Event, torch.cuda.Event(enable_timing=False))
        cold_init_stream = torch.cuda.current_stream(self._device)
        if cold_init_stream != self._task_stream:
            self._cold_init_event.record(cold_init_stream)
            self._task_stream.wait_event(cast(Any, self._cold_init_event))
        self._initialized = False

    @staticmethod
    def _validate_kernel(
        kernel: DeviceManagedTaskKernel, plan: CompiledTaskPlan
    ) -> tuple[str, ...]:
        try:
            executor_key = kernel.executor_key
        except AttributeError as exc:
            raise DeviceManagedRuntimeError("device managed kernel has no executor_key") from exc
        if not isinstance(executor_key, str) or executor_key != plan.executor_key:
            raise DeviceManagedRuntimeError(
                "device managed kernel executor_key does not match the compiled task plan"
            )
        try:
            metric_keys = kernel.metric_keys
        except AttributeError as exc:
            raise DeviceManagedRuntimeError("device managed kernel has no metric_keys") from exc
        if (
            not isinstance(metric_keys, tuple)
            or any(not isinstance(key, str) or not key.strip() for key in metric_keys)
            or len(set(metric_keys)) != len(metric_keys)
        ):
            raise DeviceManagedRuntimeError(
                "device managed kernel metric_keys must be unique non-empty strings"
            )
        # This cold-path depth check inspects instance attributes only.  It
        # intentionally does not claim to detect class attributes or closure
        # captures; task implementations remain responsible for the protocol.
        for forbidden in ("backend", "env", "_backend", "_env", "model", "_model"):
            if forbidden in vars(kernel):
                raise DeviceManagedRuntimeError(
                    f"device managed kernel must not retain forbidden {forbidden!r} owner reference"
                )
        return metric_keys

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
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def metric_keys(self) -> tuple[str, ...]:
        """Return the cold-bound task metric layout in transition order."""

        return self._metric_keys

    @property
    def event_bindings(self) -> tuple[DeviceMutationEventBinding, ...]:
        """Return immutable backend-bound Event metadata without exposing samplers."""

        return self._event_bindings

    def capture_event_trigger_counts(self) -> tuple[tuple[str, Any], ...]:
        """Materialize per-world counters at an explicit diagnostic boundary."""

        if self._initialized:
            torch.cuda.current_stream(self._device).wait_event(cast(Any, self._output_event))
        return tuple(
            (binding.event.term_key, stream.capture_trigger_counts())
            for binding, stream in zip(self._event_bindings, self._event_streams, strict=True)
        )

    @property
    def event_traffic_diagnostics(
        self,
    ) -> tuple[tuple[str, KeyedRandomTrafficDiagnostics], ...]:
        """Return allocation/transfer counters without materializing Event data."""

        return tuple(
            (binding.event.term_key, stream.traffic_diagnostics)
            for binding, stream in zip(self._event_bindings, self._event_streams, strict=True)
        )

    @property
    def control_contract(self) -> BufferContract:
        """Return the runner-owned action contract frozen by the bound plan."""

        return self._bound_plan.control.buffer

    @property
    def episode_length_buffer(self) -> torch.Tensor:
        """Return manager-owned episode steps for the RSL-RL VecEnv ABI.

        Callers may read this tensor to construct an initial schedule.  A
        replacement schedule must be submitted through
        :meth:`set_episode_length_buffer` so its producer stream is ordered
        before the task stream.
        """

        return self._episode_steps

    def set_episode_length_buffer(self, values: torch.Tensor) -> None:
        """Submit a cold-path initial episode schedule to stable runtime storage.

        ``values`` can have been produced on any stream of the runtime CUDA
        device.  The explicit event prevents the task stream from observing a
        partially written schedule when upstream RSL-RL randomizes episode
        lengths on its policy/default stream.
        """

        if (
            not isinstance(values, torch.Tensor)
            or values.device != self._device
            or values.dtype is not torch.int64
            or tuple(values.shape) != (self._num_envs,)
            or not values.is_contiguous()
        ):
            raise DeviceManagedRuntimeError(
                "device episode-length initialization differs from the runtime contract"
            )
        producer_stream = torch.cuda.current_stream(self._device)
        with torch.cuda.stream(self._task_stream):
            if producer_stream != self._task_stream:
                self._episode_length_input_event.record(producer_stream)
                # PyTorch's stubs expose two internal Event aliases here;
                # runtime validation above still guarantees a CUDA event.
                self._task_stream.wait_event(cast(Any, self._episode_length_input_event))
            self._episode_steps.copy_(values, non_blocking=True)
            # The caller may release this temporary tensor as soon as the
            # setter returns. Keep its allocator storage alive until the task
            # stream has consumed the asynchronous copy.
            values.record_stream(self._task_stream)

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

    @property
    def task_state(self) -> object:
        """Expose the kernel-owned state only for explicit diagnostics."""

        return self._task_state

    @property
    def traffic_diagnostics(self) -> DeviceRuntimeTrafficDiagnostics:
        """Return cumulative typed counters without materializing device data."""

        return self._traffic

    @property
    def reset_phase_timing_diagnostics(
        self,
    ) -> DeviceResetPhaseTimingDiagnostics | None:
        """Return host-only timing lifecycle counters without querying CUDA."""

        session = self._reset_phase_timing_session
        return None if session is None else session.diagnostics

    def begin_reset_phase_timing(self, *, capacity: int) -> None:
        """Preallocate and prime an explicit reset timing window."""

        if self._reset_phase_timing_session is not None:
            raise DeviceManagedRuntimeError("a reset phase timing window is already active")
        try:
            session = self._backend.create_reset_phase_timing_session(capacity=capacity)
        except (BackendBatchContractError, DevicePhaseTimingError, NotImplementedError) as exc:
            raise DeviceManagedRuntimeError(
                f"backend could not create reset phase timing: {exc}"
            ) from exc
        if not isinstance(session, DeviceResetPhaseTimingSession):
            raise DeviceManagedRuntimeError(
                "backend returned an invalid reset phase timing session"
            )
        if (
            session.backend_type != self._bound_plan.backend_type
            or session.backend_instance_id != self._bound_plan.backend_instance_id
            or session.placement != self._placement
        ):
            raise DeviceManagedRuntimeError(
                "reset phase timing session belongs to another bound backend plan"
            )
        self._reset_phase_timing_session = session

    def materialize_reset_phase_timings(self) -> DeviceResetPhaseTimingTrace:
        """Close the active timing window with one low-frequency event wait."""

        session = self._reset_phase_timing_session
        if session is None:
            raise DeviceManagedRuntimeError("no reset phase timing window is active")
        try:
            trace = session.materialize()
        except DevicePhaseTimingError as exc:
            raise DeviceManagedRuntimeError(
                f"could not materialize reset phase timing: {exc}"
            ) from exc
        self._reset_phase_timing_session = None
        return trace

    def capture_performance_diagnostics(self) -> BackendMutationPerformanceDiagnostics:
        """Capture plan-scoped backend evidence outside the device hot path."""

        try:
            diagnostics = self._backend.get_mutation_performance_diagnostics(self._mutation_plan)
        except (BackendBatchContractError, NotImplementedError) as exc:
            raise DeviceManagedRuntimeError(
                f"backend could not capture mutation performance diagnostics: {exc}"
            ) from exc
        if not isinstance(diagnostics, BackendMutationPerformanceDiagnostics):
            raise DeviceManagedRuntimeError(
                "backend returned invalid mutation performance diagnostics"
            )
        if (
            diagnostics.backend_type != self._bound_plan.backend_type
            or diagnostics.backend_instance_id != self._bound_plan.backend_instance_id
            or diagnostics.mutation_plan_fingerprint != self._mutation_plan.fingerprint
        ):
            raise DeviceManagedRuntimeError(
                "backend mutation performance diagnostics have a different owner"
            )
        return diagnostics

    @property
    def stability_diagnostics(self) -> DeviceRuntimeStabilityDiagnostics | None:
        """Return the opt-in registered-buffer snapshot after initial reset."""

        if self._stability_monitor is None:
            return None
        return self._stability_monitor.snapshot(
            output_epoch=self._output_lease.epoch,
            control_epoch=self._control_lease.epoch,
            reset_epoch=self._reset_lease.epoch,
            traffic=self._traffic,
            graph=self._backend.get_device_graph_diagnostics(verify_storage=True),
        )

    def _runtime_buffers(self) -> tuple[DeviceRuntimeBuffer, ...]:
        buffers: list[DeviceRuntimeBuffer] = [
            DeviceRuntimeBuffer("runtime.control", self._control),
            DeviceRuntimeBuffer("runtime.reward", self._reward),
            DeviceRuntimeBuffer("runtime.terminated", self._terminated),
            DeviceRuntimeBuffer("runtime.truncated", self._truncated),
            DeviceRuntimeBuffer("runtime.done", self._done),
            DeviceRuntimeBuffer("runtime.final_observation_mask", self._final_observation_mask),
            DeviceRuntimeBuffer("runtime.episode_steps", self._episode_steps),
        ]
        buffers.extend(
            DeviceRuntimeBuffer(f"runtime.metric.{key}", value)
            for key, value in zip(self._metric_keys, self._metrics, strict=True)
        )
        for key, observation, terminal, final in zip(
            self._observation_keys,
            self._observations,
            self._terminal_observations,
            self._final_observations,
            strict=True,
        ):
            buffers.extend(
                (
                    DeviceRuntimeBuffer(f"runtime.observation.{key}", observation),
                    DeviceRuntimeBuffer(f"runtime.terminal_observation.{key}", terminal),
                    DeviceRuntimeBuffer(f"runtime.final_observation.{key}", final),
                )
            )
        for binding, stream in zip(self._event_bindings, self._event_streams, strict=True):
            prefix = f"runtime.event.{binding.event.term_key}"
            buffers.extend(
                DeviceRuntimeBuffer(f"{prefix}.{name}", tensor)
                for name, tensor in stream.named_buffers
            )
        return tuple(buffers)

    def _stability_buffers(self) -> tuple[DeviceRuntimeBuffer, ...]:
        provider = self._stability_buffer_provider
        if provider is None:
            raise DeviceManagedRuntimeError("device stability buffers are not enabled")
        task_buffers = provider.device_runtime_buffers(task_state=self._task_state)
        if not isinstance(task_buffers, tuple) or any(
            not isinstance(buffer, DeviceRuntimeBuffer) for buffer in task_buffers
        ):
            raise DeviceManagedRuntimeError(
                "device stability provider must return DeviceRuntimeBuffer values"
            )
        return (*self._runtime_buffers(), *task_buffers)

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
        if not isinstance(diagnostics, BackendBatchDiagnostics):
            raise DeviceManagedRuntimeError("device backend diagnostics are invalid")
        counters = diagnostics.counters
        if not counters.instrumentation_complete:
            raise DeviceManagedRuntimeError(
                "device lifecycle requires complete backend instrumentation"
            )
        forbidden = {
            "host_to_device_transfers": counters.host_to_device_transfers,
            "device_to_host_transfers": counters.device_to_host_transfers,
            "host_to_device_bytes": counters.host_to_device_bytes,
            "device_to_host_bytes": counters.device_to_host_bytes,
            "global_synchronizations": counters.global_synchronizations,
            "allocations": counters.allocations,
            "dynamic_getter_calls": counters.dynamic_getter_calls,
            "selector_resolutions": counters.selector_resolutions,
            "asset_metadata_reads": counters.asset_metadata_reads,
            "registry_lookups": counters.registry_lookups,
        }
        violations = tuple(name for name, value in forbidden.items() if value != 0)
        if violations or counters.state_materializations != 1:
            raise DeviceManagedRuntimeError(
                "device backend barrier violates the zero-host-roundtrip budget: "
                f"phase={phase!r}, violations={violations!r}, "
                f"state_materializations={counters.state_materializations}"
            )
        if phase == "step":
            self._last_step_diagnostics = diagnostics
            step_increment = 1
            reset_increment = 0
        elif phase == "reset":
            self._last_reset_diagnostics = diagnostics
            step_increment = 0
            reset_increment = 1
        else:  # pragma: no cover - internal fixed call sites.
            raise DeviceManagedRuntimeError(f"unknown device diagnostic phase {phase!r}")
        current = self._traffic
        self._traffic = DeviceRuntimeTrafficDiagnostics(
            policy_steps=current.policy_steps,
            step_barriers=current.step_barriers + step_increment,
            reset_barriers=current.reset_barriers + reset_increment,
            host_to_device_transfers=(
                current.host_to_device_transfers + counters.host_to_device_transfers
            ),
            device_to_host_transfers=(
                current.device_to_host_transfers + counters.device_to_host_transfers
            ),
            host_to_device_bytes=current.host_to_device_bytes + counters.host_to_device_bytes,
            device_to_host_bytes=current.device_to_host_bytes + counters.device_to_host_bytes,
            global_synchronizations=(
                current.global_synchronizations + counters.global_synchronizations
            ),
            backend_allocations=current.backend_allocations + counters.allocations,
            state_materializations=(
                current.state_materializations + counters.state_materializations
            ),
            dynamic_getter_calls=current.dynamic_getter_calls + counters.dynamic_getter_calls,
            selector_resolutions=current.selector_resolutions + counters.selector_resolutions,
            asset_metadata_reads=current.asset_metadata_reads + counters.asset_metadata_reads,
            registry_lookups=current.registry_lookups + counters.registry_lookups,
            instrumentation_complete=(
                current.instrumentation_complete and counters.instrumentation_complete
            ),
        )

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

    def _build_reset_batch(
        self, payload: DeviceResetPayload
    ) -> tuple[DeviceResetMutationBatch, DeviceResetPhaseTimingSampleToken | None]:
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
        payload_indices = tuple(value.field_index for value in payload.values)
        if len(set(payload_indices)) != len(payload_indices):
            raise DeviceManagedRuntimeError(
                "device reset payload contains a duplicate deterministic field"
            )
        if any(index >= len(self._mutation_plan.specs) for index in payload_indices):
            raise DeviceManagedRuntimeError(
                "device reset payload references an unbound mutation field"
            )
        if any(index in self._event_mutation_indices for index in payload_indices):
            raise DeviceManagedRuntimeError(
                "device task kernel must not provide a manager-owned Event field"
            )
        if payload_indices != self._deterministic_mutation_indices:
            raise DeviceManagedRuntimeError(
                "device reset payload must provide every deterministic field once in "
                "canonical mutation order"
            )
        tensors: list[torch.Tensor | None] = [None] * len(self._mutation_plan.specs)
        for descriptor in payload.values:
            spec = self._mutation_plan.specs[descriptor.field_index]
            tensor = descriptor.tensor
            expected_shape = (self._num_envs, *spec.value_buffer.row_shape)
            if (
                tensor.device != self._device
                or tensor.dtype is not _torch_dtype(spec.value_buffer.dtype)
                or tuple(tensor.shape) != expected_shape
                or not tensor.is_contiguous()
            ):
                raise DeviceManagedRuntimeError(
                    f"device reset value {descriptor.field_index} differs from its "
                    "cold-bound mutation contract"
                )
            tensors[descriptor.field_index] = tensor
        phase_timing = self._reset_phase_timing_session
        phase_timing_token: DeviceResetPhaseTimingSampleToken | None = None
        try:
            if phase_timing is not None:
                phase_timing_token = phase_timing.begin_sample(self._task_stream)
            for binding, stream in zip(self._event_bindings, self._event_streams, strict=True):
                sampled = stream.sample(mask).values
                expected_shape = (self._num_envs, *binding.value_contract.row_shape)
                if (
                    sampled.device != self._device
                    or sampled.dtype is not _torch_dtype(binding.value_contract.dtype)
                    or tuple(sampled.shape) != expected_shape
                    or not sampled.is_contiguous()
                ):
                    raise DeviceManagedRuntimeError(
                        f"manager Event {binding.event.term_key!r} produced an invalid value buffer"
                    )
                tensors[binding.event.mutation_index] = sampled
            if phase_timing is not None:
                assert phase_timing_token is not None
                phase_timing.end_mutation_sample(phase_timing_token, self._task_stream)
        except DevicePhaseTimingError as exc:
            raise DeviceManagedRuntimeError(f"reset phase timing failed: {exc}") from exc
        if any(tensor is None for tensor in tensors):
            raise DeviceManagedRuntimeError(
                "device reset values do not cover the complete bound mutation plan"
            )
        completion = DeviceCompletion.record(
            placement=self._placement,
            owner_id=self._reset_lease.owner_id,
            epoch=self._reset_lease.epoch,
            stream=self._task_stream,
            event=self._reset_event,
        )
        mask_view = DeviceTensorView(
            tensor_handle=mask,
            contract=_mask_contract(placement=self._placement),
            lease=self._reset_lease,
            completion=completion,
        )
        values: list[MutationValueBatch] = []
        for field_index, (spec, optional_tensor) in enumerate(
            zip(self._mutation_plan.specs, tensors, strict=True)
        ):
            assert optional_tensor is not None
            tensor = optional_tensor
            expected_shape = (self._num_envs, *spec.value_buffer.row_shape)
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
            model=ModelParameterMutationBatch(
                tuple(values[index] for index in self._model_mutation_indices)
            ),
            state=SimulationStateMutationBatch(
                tuple(values[index] for index in self._state_mutation_indices)
            ),
        )
        batch = DeviceResetMutationBatch(
            plan=self._mutation_plan,
            rows=self._all_rows,
            mutation=mutation,
            active_mask=BufferView(
                handle=mask_view,
                shape=(self._num_envs,),
                contract=_mask_contract(placement=self._placement),
            ),
        )
        return batch, phase_timing_token

    def _submit_reset_batch(
        self,
        batch: DeviceResetMutationBatch,
        phase_timing: DeviceResetPhaseTimingSampleToken | None,
    ) -> BackendResetResult:
        if phase_timing is None:
            return self._backend.reset_batch(
                self._bound_plan,
                self._all_rows,
                mutation_batch=batch,
            )
        return self._backend.reset_batch(
            self._bound_plan,
            self._all_rows,
            mutation_batch=batch,
            phase_timing=phase_timing,
        )

    def _control_batch(self) -> ControlBatch:
        completion = DeviceCompletion.record(
            placement=self._placement,
            owner_id=self._control_lease.owner_id,
            epoch=self._control_lease.epoch,
            stream=self._task_stream,
            event=self._control_event,
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
            event=self._output_event,
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

        metrics = tuple(
            DeviceTransitionBuffer(
                key=key,
                view=self._transition_view(value, self._reward_contract, completion),
            )
            for key, value in zip(self._metric_keys, self._metrics, strict=True)
        )

        return DeviceTransition(
            plan_fingerprint=self._plan.fingerprint,
            observations=named(self._observations),
            terminal_observations=named(self._terminal_observations),
            final_observations=named(self._final_observations),
            metrics=metrics,
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

        first_reset = not self._initialized
        if not first_reset:
            self._observe_stability_buffers()
        self._last_step_diagnostics = None
        self._output_lease.invalidate()
        self._reset_lease.invalidate()
        self._begin_trace()
        self._trace(ManagedLifecyclePhase.INITIAL_RESET_REQUEST)
        with torch.cuda.stream(self._task_stream):
            all_mask = self._done
            all_mask.fill_(True)
            payload = self._kernel.prepare_reset(active_mask=all_mask, task_state=self._task_state)
            reset_batch, phase_timing = self._build_reset_batch(payload)
        self._trace(ManagedLifecyclePhase.RESET_BACKEND)
        result = self._submit_reset_batch(reset_batch, phase_timing)
        if not isinstance(result, BackendResetResult):
            raise DeviceManagedRuntimeError(
                "device backend reset_batch must return BackendResetResult"
            )
        self._validate_state(result.reset_state, phase=StateBatchPhase.RESET)
        reset_completion = self._require_backend_completion(
            diagnostics=result.diagnostics,
            state=result.reset_state,
        )
        self._record_backend_diagnostics(phase="reset", diagnostics=result.diagnostics)
        self._observe_stability_state(result.reset_state)
        self._trace(ManagedLifecyclePhase.TASK_STATE_RESET)
        with torch.cuda.stream(self._task_stream):
            reset_completion.wait(self._task_stream)
            self._kernel.complete_reset(
                active_mask=payload.active_mask,
                state=result.reset_state,
                task_state=self._task_state,
                observation_buffers=self._observations,
            )
            for terminal, final, observation in zip(
                self._terminal_observations,
                self._final_observations,
                self._observations,
                strict=True,
            ):
                terminal.copy_(observation, non_blocking=True)
                final.copy_(observation, non_blocking=True)
            self._reward.zero_()
            for metric in self._metrics:
                metric.zero_()
            self._terminated.zero_()
            self._truncated.zero_()
            self._final_observation_mask.zero_()
            self._episode_steps.zero_()
            self._trace(ManagedLifecyclePhase.OBSERVATION)
            completion = self._output_completion()
        if first_reset:
            self._arm_stability_monitor()
        else:
            self._observe_stability_buffers()
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
        self._observe_stability_buffers()
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
        self._record_backend_diagnostics(phase="step", diagnostics=step_result.diagnostics)
        self._observe_stability_state(terminal)

        self._reset_lease.invalidate()
        self._trace(ManagedLifecyclePhase.TERMINATION)
        with torch.cuda.stream(self._task_stream):
            step_completion.wait(self._task_stream)
            self._reward.zero_()
            for metric in self._metrics:
                metric.zero_()
            self._terminated.zero_()
            self._kernel.evaluate_terminal(
                state=terminal,
                task_state=self._task_state,
                reward_out=self._reward,
                metric_buffers=self._metrics,
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
                # RSL-RL evaluates timeout bootstrap values as one static batch
                # and applies the timeout mask afterwards.  Every row must
                # therefore contain a finite terminal observation; leaving
                # non-done rows stale would let ``0 * NaN`` poison rewards.
                final.copy_(terminal_observation, non_blocking=True)
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
            reset_batch, phase_timing = self._build_reset_batch(payload)
        self._trace(ManagedLifecyclePhase.RESET_BACKEND)
        reset_result = self._submit_reset_batch(reset_batch, phase_timing)
        if not isinstance(reset_result, BackendResetResult):
            raise DeviceManagedRuntimeError(
                "device backend reset_batch must return BackendResetResult"
            )
        self._validate_state(reset_result.reset_state, phase=StateBatchPhase.RESET)
        reset_completion = self._require_backend_completion(
            diagnostics=reset_result.diagnostics,
            state=reset_result.reset_state,
        )
        self._record_backend_diagnostics(phase="reset", diagnostics=reset_result.diagnostics)
        self._observe_stability_state(reset_result.reset_state)
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
        self._observe_stability_buffers()
        self._traffic = replace(
            self._traffic,
            policy_steps=self._traffic.policy_steps + 1,
        )
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
    "DeviceMutationEventBinding",
    "DeviceResetPayload",
    "DeviceResetValue",
    "DeviceRuntimeBuffer",
    "DeviceRuntimeBufferAddress",
    "DeviceRuntimeBufferProvider",
    "DeviceRuntimeStabilityDiagnostics",
    "DeviceRuntimeStateEpoch",
    "DeviceRuntimeTrafficDiagnostics",
    "DeviceTransition",
    "DeviceTransitionBuffer",
]
