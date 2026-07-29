"""Long-loop CUDA allocation, address, and instrumentation gates for Issue 705."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
from tests.training.device_runtime_harness import forbid_host_roundtrip, runtime_harness
from tests.training.test_device_lifecycle import _force_root_state

from unilab.base.backend import (
    BackendBatchCounters,
    BackendCompletionEvent,
    BackendStepResult,
    BufferView,
    DeviceBufferLease,
    DeviceCompletion,
    DeviceTensorView,
    StateBatch,
    StateBatchLease,
)
from unilab.base.backend.mjwarp.backend import (
    MjwarpBackend,
    MjwarpDeviceCapacityDiagnostics,
)
from unilab.manager import DeviceManagedRuntimeError, DeviceRuntimeStabilityDiagnostics

pytestmark = pytest.mark.slow


def _diagnostics(runtime: Any) -> DeviceRuntimeStabilityDiagnostics:
    value = runtime.stability_diagnostics
    assert isinstance(value, DeviceRuntimeStabilityDiagnostics)
    assert value.instrumentation_complete
    assert value.warm_numeric_allocations == 0
    assert value.address_churn == 0
    assert value.traffic.host_to_device_transfers == 0
    assert value.traffic.device_to_host_transfers == 0
    assert value.traffic.global_synchronizations == 0
    assert value.traffic.backend_allocations == 0
    assert value.graph.instrumentation_complete
    assert value.graph.active_keys
    assert value.graph.eager_fallback_count == 0
    assert value.graph.recapture_count == 0
    assert value.graph.stale_rejection_count == 0
    return value


@pytest.mark.parametrize(
    ("num_envs", "episode_limit", "rollout_steps"),
    (
        (128, 10_000, 96),
        (128, 1, 96),
        (4096, 10_000, 32),
        (4096, 1, 32),
    ),
)
def test_long_rollout_memory_and_addresses_are_stable(
    num_envs: int,
    episode_limit: int,
    rollout_steps: int,
) -> None:
    """No/all/sparse reset execution keeps every registered allocation stable."""

    with runtime_harness(
        num_envs=num_envs,
        seed=0,
        max_episode_steps=episode_limit,
    ) as harness:
        for _ in range(8):
            harness.step(0.01)

        # Exercise sparse autoreset once before allocator baselining.  The
        # state injection itself is an explicit public setup barrier; only the
        # subsequent runtime transition belongs to the measured path.
        if num_envs == 128 and episode_limit > 1:
            forced = _force_root_state(
                cast(Any, harness),
                rows=(1, num_envs - 1),
                height=0.1,
                producer_stream=harness.producer_stream,
                after=harness.transition.completion,
            )
            sparse = harness.step(after=forced)
            sparse.completion.event.synchronize()
            terminated = sparse.terminated.torch().cpu()
            assert torch.equal(
                torch.nonzero(terminated, as_tuple=False).flatten(),
                torch.tensor((1, num_envs - 1), dtype=torch.int64),
            )
            for _ in range(4):
                harness.step()

        harness.wait()
        baseline = _diagnostics(harness.runtime)
        baseline_buffers = baseline.buffers
        baseline_state_buffers = baseline.state_buffers
        baseline_graph_buffers = baseline.graph.storage_buffers
        baseline_graph_keys = baseline.graph.active_keys
        baseline_graph_launches = baseline.graph.launch_count
        baseline_epochs = {item.name: item.lease_epoch for item in baseline.state_epochs}
        action_pointer = int(harness.action.data_ptr())
        torch.cuda.reset_peak_memory_stats(harness.device)
        allocated = [int(torch.cuda.memory_allocated(harness.device))]
        reserved = [int(torch.cuda.memory_reserved(harness.device))]

        chunk_size = 16
        completed = 0
        while completed < rollout_steps:
            count = min(chunk_size, rollout_steps - completed)
            with forbid_host_roundtrip(harness.backend):
                for index in range(count):
                    harness.step(0.0025 * float((completed + index) % 3))
                    assert int(harness.action.data_ptr()) == action_pointer
            harness.wait()
            allocated.append(int(torch.cuda.memory_allocated(harness.device)))
            reserved.append(int(torch.cuda.memory_reserved(harness.device)))
            diagnostics = _diagnostics(harness.runtime)
            assert diagnostics.buffers == baseline_buffers
            assert diagnostics.state_buffers == baseline_state_buffers
            assert diagnostics.graph.storage_buffers == baseline_graph_buffers
            assert diagnostics.graph.active_keys == baseline_graph_keys
            completed += count

        diagnostics = _diagnostics(harness.runtime)
        assert diagnostics.traffic.policy_steps - baseline.traffic.policy_steps == rollout_steps
        assert diagnostics.traffic.step_barriers - baseline.traffic.step_barriers == rollout_steps
        assert diagnostics.traffic.reset_barriers - baseline.traffic.reset_barriers == rollout_steps
        assert (
            diagnostics.traffic.state_materializations - baseline.traffic.state_materializations
            == 2 * rollout_steps
        )
        assert diagnostics.observations - baseline.observations == 2 * rollout_steps
        assert diagnostics.graph.capture_count == baseline.graph.capture_count
        assert diagnostics.graph.launch_count - baseline_graph_launches == 3 * rollout_steps
        assert all(
            item.lease_epoch > baseline_epochs[item.name] for item in diagnostics.state_epochs
        )

        # Event-scoped sampling is deliberately outside each measured chunk.
        # After all lifecycle variants are warm, the caching allocator must
        # neither retain new live tensors nor keep growing its reserve.
        # Delayed cleanup from an earlier fixture may lower live/reserved
        # memory during this case. Only positive growth above this runtime's
        # warmed baseline is a leak signal.
        assert max(allocated) - allocated[0] <= 4 * 1024 * 1024
        assert max(reserved) - reserved[0] <= 64 * 1024 * 1024
        properties = torch.cuda.get_device_properties(harness.device)
        assert torch.cuda.max_memory_reserved(harness.device) / int(properties.total_memory) < 0.8


def test_g1_device_owner_capacity_budget_has_real_cuda_headroom() -> None:
    """The explicit G1 owner budget covers every sampled G1 device state.

    Capacity reads deliberately synchronize and copy small Warp counters, so
    the test-only wrapper samples immediately after each public ``step_batch``
    return and before ``DeviceManagedRuntime`` can invoke its masked reset.
    The wrapper is not installed in the runtime/runner path. The initialized
    reset state is sampled as well. Sampling every subsequent physics barrier
    matters because autoreset may otherwise clear an overflow bit before a
    final-only diagnostic could observe it.
    """

    with runtime_harness(
        num_envs=128,
        seed=0,
        max_episode_steps=10_000,
        mjwarp_nconmax=128,
        mjwarp_njmax=256,
    ) as harness:
        backend = harness.backend
        assert isinstance(backend, MjwarpBackend)
        samples: list[MjwarpDeviceCapacityDiagnostics] = [backend.get_device_capacity_diagnostics()]
        original_step_batch = backend.step_batch

        def record_step_capacity(*args: Any, **kwargs: Any) -> BackendStepResult:
            result = original_step_batch(*args, **kwargs)
            assert isinstance(result, BackendStepResult)
            samples.append(backend.get_device_capacity_diagnostics())
            return result

        with patch.object(backend, "step_batch", side_effect=record_step_capacity):
            for step in range(96):
                harness.step(0.15 * float((step % 7) - 3))
        harness.wait()

    assert len(samples) == 97
    assert all(sample.nconmax_per_world == 128 for sample in samples)
    assert all(sample.njmax_per_world == 256 for sample in samples)
    assert all(sample.global_contact_capacity == 128 * 128 for sample in samples)
    peak_contacts = max(sample.global_contact_count for sample in samples)
    peak_constraints = max(sample.max_constraints_per_world for sample in samples)
    # Keep a material allocation margin, not merely a non-overflowing final
    # state. Capacity/profile changes require an explicit owner re-evaluation.
    assert peak_contacts <= samples[0].global_contact_capacity // 2
    assert peak_constraints <= samples[0].njmax_per_world * 3 // 4
    assert all(sample.overflow_world_count == 0 for sample in samples)
    assert all(sample.overflow_mask == 0 for sample in samples)


def _copied_device_state(result: BackendStepResult) -> BackendStepResult:
    original_completion = result.diagnostics.completion_event
    assert isinstance(original_completion, BackendCompletionEvent)
    handle = original_completion.handle
    assert isinstance(handle, DeviceCompletion)
    stream = torch.cuda.current_stream(torch.device(f"cuda:{handle.placement.device_index}"))
    handle.wait(stream)
    owner_id = result.terminal_state.plan.backend_instance_id
    lease = DeviceBufferLease(owner_id)
    event = cast(torch.cuda.Event, torch.cuda.Event(enable_timing=False))
    tensors: list[torch.Tensor] = []
    with torch.cuda.stream(stream):
        for index in range(len(result.terminal_state.plan.state.fields)):
            source = result.terminal_state.buffer_at(index).handle
            assert isinstance(source, DeviceTensorView)
            tensors.append(source.torch().clone())
        event.record(stream)
    completion = DeviceCompletion(
        placement=handle.placement,
        owner_id=owner_id,
        epoch=lease.epoch,
        event=event,
    )
    descriptors = tuple(
        BufferView(
            handle=DeviceTensorView(
                tensor_handle=tensor,
                contract=field.buffer,
                lease=lease,
                completion=completion,
            ),
            shape=tuple(int(dim) for dim in tensor.shape),
            contract=field.buffer,
        )
        for tensor, field in zip(
            tensors,
            result.terminal_state.plan.state.fields,
            strict=True,
        )
    )
    state = StateBatch(
        plan=result.terminal_state.plan,
        rows=result.terminal_state.rows,
        phase=result.terminal_state.phase,
        descriptors=descriptors,
        lease=StateBatchLease(owner_id),
    )
    diagnostics = replace(
        result.diagnostics,
        completion_event=BackendCompletionEvent(
            backend_type=result.terminal_state.plan.backend_type,
            placement=handle.placement,
            handle=completion,
        ),
    )
    return replace(result, terminal_state=state, diagnostics=diagnostics)


def test_device_stability_instrumentation_fails_closed() -> None:
    """Buffer churn, address churn, missing counters, and transfers are rejected."""

    with runtime_harness(num_envs=8, seed=1, max_episode_steps=100) as harness:
        task = harness.runtime.task_state
        task.commands = torch.empty_like(task.commands)  # type: ignore[attr-defined]
        with pytest.raises(DeviceManagedRuntimeError, match="warm buffer stability violated"):
            harness.step()

    with runtime_harness(num_envs=8, seed=2, max_episode_steps=100) as harness:
        harness.step()
        original_step = harness.backend.step_batch

        def copied_step(*args: Any, **kwargs: Any) -> BackendStepResult:
            result = original_step(*args, **kwargs)
            assert isinstance(result, BackendStepResult)
            return _copied_device_state(result)

        with patch.object(harness.backend, "step_batch", side_effect=copied_step):
            with pytest.raises(DeviceManagedRuntimeError, match="StateBatch address changed"):
                harness.step()

    for counters, expected in (
        (BackendBatchCounters(instrumentation_complete=False), "complete backend instrumentation"),
        (
            BackendBatchCounters(
                host_to_device_transfers=1,
                host_to_device_bytes=4,
                state_materializations=1,
                instrumentation_complete=True,
            ),
            "zero-host-roundtrip budget",
        ),
    ):
        with runtime_harness(num_envs=8, seed=3, max_episode_steps=100) as harness:
            original_step = harness.backend.step_batch

            def corrupt_step(*args: Any, **kwargs: Any) -> BackendStepResult:
                result = original_step(*args, **kwargs)
                assert isinstance(result, BackendStepResult)
                return replace(
                    result,
                    diagnostics=replace(result.diagnostics, counters=counters),
                )

            with patch.object(harness.backend, "step_batch", side_effect=corrupt_step):
                with pytest.raises(DeviceManagedRuntimeError, match=expected):
                    harness.step()
