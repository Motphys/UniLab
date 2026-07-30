"""Real-CUDA contract tests for manager-owned reset Event composition."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from tests.training.device_runtime_harness import runtime_harness

from unilab.base.backend import (
    BufferPlacement,
    DeviceResetMutationBatch,
    DeviceTensorView,
    MutationOperation,
    MutationTargetKind,
)
from unilab.manager import (
    DeviceManagedRuntime,
    DeviceManagedRuntimeError,
    DeviceResetPayload,
    DeviceResetValue,
    ManagerContractError,
)

pytestmark = pytest.mark.slow


def _event_runtime():
    return runtime_harness(
        num_envs=4,
        seed=821,
        max_episode_steps=128,
        randomize_kp=True,
        randomize_kd=True,
    )


def _event_counts(runtime: DeviceManagedRuntime) -> dict[str, np.ndarray]:
    return dict(runtime.capture_event_trigger_counts())


def test_runtime_composes_sparse_kernel_and_event_values_by_target_kind() -> None:
    with _event_runtime() as harness:
        runtime = harness.runtime
        mutation_plan = runtime.kernel_binding.mutation_plan
        assert mutation_plan is not None
        event_indices = runtime.kernel_binding.event_mutation_indices
        deterministic_indices = tuple(
            index for index in range(len(mutation_plan.specs)) if index not in event_indices
        )

        reset_values = runtime.task_state.reset_values  # type: ignore[attr-defined]
        assert tuple(value.field_index for value in reset_values) == deterministic_indices
        assert all(isinstance(value, DeviceResetValue) for value in reset_values)

        captured: list[DeviceResetMutationBatch] = []
        original_reset = harness.backend.reset_batch

        def capture_reset(plan, rows, *, mutation_batch=None):
            assert isinstance(mutation_batch, DeviceResetMutationBatch)
            captured.append(mutation_batch)
            return original_reset(plan, rows, mutation_batch=mutation_batch)

        with patch.object(harness.backend, "reset_batch", side_effect=capture_reset):
            transition = runtime.reset()
        transition.completion.event.synchronize()  # Explicit test oracle boundary.

        assert len(captured) == 1
        batch = captured[0]
        model_values = batch.mutation.model.values
        state_values = batch.mutation.state.values
        assert tuple(value.field_index for value in model_values) == event_indices
        assert tuple(value.field_index for value in state_values) == deterministic_indices
        assert all(
            value.spec.target.target_kind is MutationTargetKind.MODEL_PARAMETER
            for value in model_values
        )
        assert all(
            value.spec.target.target_kind is MutationTargetKind.SIMULATION_STATE
            for value in state_values
        )

        handles = [batch.active_mask.handle]
        handles.extend(value.buffer.handle for value in (*model_values, *state_values))
        assert all(isinstance(handle, DeviceTensorView) for handle in handles)
        typed_handles = [handle for handle in handles if isinstance(handle, DeviceTensorView)]
        producer = typed_handles[0].require_completion()
        assert all(handle.lease is typed_handles[0].lease for handle in typed_handles)
        assert all(handle.epoch == producer.epoch for handle in typed_handles)
        assert all(handle.require_completion().event is producer.event for handle in typed_handles)

        for binding in runtime.event_bindings:
            value = next(
                item for item in model_values if item.field_index == binding.event.mutation_index
            )
            tensor = value.buffer.handle
            assert isinstance(tensor, DeviceTensorView)
            assert tensor.shape == (runtime.num_envs, *binding.value_contract.row_shape)


def test_runtime_rejects_malformed_sparse_payload_before_backend_commit() -> None:
    with _event_runtime() as harness:
        runtime = harness.runtime
        kernel = runtime._kernel
        original_prepare = kernel.prepare_reset
        event_index = runtime.kernel_binding.event_mutation_indices[0]
        counts_before = _event_counts(runtime)

        def malformed_payload(kind: str):
            def prepare(*, active_mask, task_state):
                payload = original_prepare(active_mask=active_mask, task_state=task_state)
                values = payload.values
                first = values[0]
                if kind == "duplicate":
                    changed = (first, *values)
                elif kind == "missing":
                    changed = values[:-1]
                elif kind == "event":
                    changed = (*values, DeviceResetValue(event_index, first.tensor))
                elif kind == "shape":
                    changed = (
                        replace(
                            first,
                            tensor=torch.empty(
                                (runtime.num_envs,),
                                dtype=first.tensor.dtype,
                                device=runtime.device,
                            ),
                        ),
                        *values[1:],
                    )
                elif kind == "dtype":
                    changed = (
                        replace(first, tensor=torch.empty_like(first.tensor, dtype=torch.float64)),
                        *values[1:],
                    )
                elif kind == "device":
                    changed = (
                        replace(
                            first,
                            tensor=torch.empty(
                                tuple(first.tensor.shape), dtype=first.tensor.dtype, device="cpu"
                            ),
                        ),
                        *values[1:],
                    )
                else:  # pragma: no cover - fixed local cases.
                    raise AssertionError(kind)
                return DeviceResetPayload(active_mask=payload.active_mask, values=changed)

            return prepare

        cases = (
            ("duplicate", "duplicate deterministic field"),
            ("missing", "provide every deterministic field"),
            ("event", "must not provide a manager-owned Event field"),
            ("shape", "differs from its cold-bound mutation contract"),
            ("dtype", "differs from its cold-bound mutation contract"),
            ("device", "differs from its cold-bound mutation contract"),
        )
        for kind, message in cases:
            with (
                patch.object(kernel, "prepare_reset", side_effect=malformed_payload(kind)),
                patch.object(
                    harness.backend,
                    "reset_batch",
                    side_effect=AssertionError("malformed payload reached backend"),
                ),
                pytest.raises(DeviceManagedRuntimeError, match=message),
            ):
                runtime.reset()

        counts_after = _event_counts(runtime)
        assert counts_after.keys() == counts_before.keys()
        for key in counts_before:
            np.testing.assert_array_equal(counts_after[key], counts_before[key])


def test_runtime_rejects_forged_event_plan_before_backend_bind() -> None:
    with _event_runtime() as harness:
        runtime = harness.runtime
        event = replace(runtime.plan.mutation_events[0], parameters=(0.8, 1.2))
        forged = replace(
            runtime.plan,
            mutation_events=(event, *runtime.plan.mutation_events[1:]),
        )

        with (
            patch.object(
                harness.backend,
                "bind_task_io",
                side_effect=AssertionError("forged plan reached backend bind"),
            ),
            pytest.raises(ManagerContractError, match="fingerprints do not match"),
        ):
            DeviceManagedRuntime(
                backend=harness.backend,
                plan=forged,
                kernel=runtime._kernel,
                max_episode_steps=128,
                run_seed=821,
            )


def test_runtime_rejects_incompatible_bound_event_before_kernel_bind() -> None:
    with _event_runtime() as harness:
        runtime = harness.runtime
        mutation_plan = runtime.kernel_binding.mutation_plan
        assert mutation_plan is not None
        event_index = runtime.kernel_binding.event_mutation_indices[0]
        original = mutation_plan.specs[event_index]
        device_index = runtime.device.index
        assert device_index is not None
        incompatible_specs = (
            replace(original, operation=MutationOperation.SET),
            replace(
                original,
                value_buffer=replace(original.value_buffer, row_shape=(1,)),
            ),
            replace(
                original,
                value_buffer=replace(original.value_buffer, dtype="float64"),
            ),
            replace(
                original,
                value_buffer=replace(
                    original.value_buffer,
                    placement=BufferPlacement.device("cuda", device_index + 1),
                ),
            ),
        )

        for incompatible in incompatible_specs:
            specs = list(mutation_plan.specs)
            specs[event_index] = incompatible
            forged = replace(mutation_plan, specs=tuple(specs))
            with (
                patch.object(harness.backend, "bind_mutation_plan", return_value=forged),
                patch.object(
                    runtime._kernel,
                    "bind",
                    side_effect=AssertionError("incompatible Event reached kernel bind"),
                ),
                pytest.raises(DeviceManagedRuntimeError, match="incompatible with its plan"),
            ):
                DeviceManagedRuntime(
                    backend=harness.backend,
                    plan=runtime.plan,
                    kernel=runtime._kernel,
                    max_episode_steps=128,
                    run_seed=821,
                )
