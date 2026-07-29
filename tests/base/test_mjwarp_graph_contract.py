"""CUDA graph identity, storage generation, and fail-closed gates for mjwarp."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast
from unittest.mock import patch

import pytest

from tests.training.device_runtime_harness import runtime_harness
from unilab.base.backend import (
    BackendBatchContractError,
    DeviceGraphBufferAddress,
    DeviceGraphCaptureKey,
    DeviceGraphContractError,
    DeviceGraphDiagnostics,
    DeviceGraphExecutionMode,
)
from unilab.base.backend.mjwarp.dependencies import load_mjwarp_dependencies

pytestmark = pytest.mark.slow


def _backend(harness: Any) -> Any:
    return cast(Any, harness.backend)


def test_graph_key_change_recaptures_or_fails_closed() -> None:
    """Plan/cadence/generation changes cannot reuse an old graph bundle."""

    with runtime_harness(num_envs=8, seed=0, max_episode_steps=100) as harness:
        backend = _backend(harness)
        initial = backend.get_device_graph_diagnostics(verify_storage=True)
        assert initial.execution_mode is DeviceGraphExecutionMode.CUDA_GRAPH
        assert initial.instrumentation_complete
        assert initial.capture_count == 1
        assert initial.recapture_count == 0
        assert initial.eager_fallback_count == 0
        assert len(initial.active_keys) == 1
        key = initial.active_keys[0]
        assert key.plan_fingerprint == harness.runtime.bound_plan.fingerprint
        assert key.num_envs == 8
        assert key.state_dtype == "float32"
        assert key.control_dtype == "float32"
        assert (
            key.physics_substeps == harness.runtime.bound_plan.control.physics_substeps_per_control
        )

        # Rebinding the identical contract is idempotent and cannot hide a
        # second capture or a changed storage generation.
        rebound = backend.bind_task_io(harness.runtime.plan.backend_io)
        assert rebound is harness.runtime.bound_plan
        idempotent = backend.get_device_graph_diagnostics(verify_storage=True)
        assert idempotent.capture_count == initial.capture_count
        assert idempotent.active_keys == initial.active_keys

        # A different control cadence produces a different compiled plan and
        # must receive a separate complete graph key.
        control = replace(
            harness.runtime.plan.backend_io.control,
            physics_substeps_per_control=key.physics_substeps + 1,
        )
        changed_requirements = replace(harness.runtime.plan.backend_io, control=control)
        changed_plan = backend.bind_task_io(changed_requirements)
        changed = backend.get_device_graph_diagnostics(verify_storage=True)
        assert changed.capture_count == initial.capture_count + 1
        assert len(changed.active_keys) == 2
        assert changed_plan.fingerprint != key.plan_fingerprint
        assert {item.physics_substeps for item in changed.active_keys} == {
            key.physics_substeps,
            key.physics_substeps + 1,
        }

        # Hot-path checks use the owner-managed generation, not a dynamic
        # storage scan. A stale generation therefore rejects before replay.
        backend._device_graph_storage_generation += 1
        with pytest.raises(BackendBatchContractError, match="stale graph generation"):
            harness.step()
        rejected = backend.get_device_graph_diagnostics()
        assert rejected.stale_rejection_count == changed.stale_rejection_count + 1
        assert rejected.launch_count == changed.launch_count


def test_model_storage_replacement_recaptures_every_active_key() -> None:
    """Owner-visible model field replacement advances generation and recaptures."""

    with runtime_harness(num_envs=8, seed=1, max_episode_steps=100) as harness:
        backend = _backend(harness)
        harness.wait()
        before = backend.get_device_graph_diagnostics(verify_storage=True)
        original = backend._device_model.geom_friction
        backend._device_model.geom_friction = backend._warp.clone(original)
        backend._warp.synchronize()

        backend._recapture_device_graphs_after_storage_change()
        after = backend.get_device_graph_diagnostics(verify_storage=True)
        assert after.storage_generation == before.storage_generation + 1
        assert after.storage_fingerprint != before.storage_fingerprint
        assert after.storage_buffers != before.storage_buffers
        assert after.capture_count == before.capture_count + len(before.active_keys)
        assert after.recapture_count == before.recapture_count + len(before.active_keys)
        assert all(
            key.storage_generation == after.storage_generation
            and key.storage_fingerprint == after.storage_fingerprint
            for key in after.active_keys
        )

        transition = harness.step(0.01)
        transition.completion.event.synchronize()
        executed = backend.get_device_graph_diagnostics(verify_storage=True)
        assert executed.launch_count == after.launch_count + 3
        assert executed.stale_rejection_count == 0


def test_graph_dtype_and_batch_key_mismatch_fail_closed() -> None:
    """Unsupported dtype and a tampered batch identity cannot reach replay."""

    with runtime_harness(num_envs=8, seed=4, max_episode_steps=100) as harness:
        backend = _backend(harness)
        requirements = harness.runtime.plan.backend_io
        bad_buffer = replace(requirements.control.buffer, dtype="float64")
        bad_control = replace(requirements.control, buffer=bad_buffer)
        with pytest.raises(BackendBatchContractError, match="requires C-contiguous float32"):
            backend.bind_task_io(replace(requirements, control=bad_control))
        after_dtype_rejection = backend.get_device_graph_diagnostics(verify_storage=True)
        assert after_dtype_rejection.capture_count == 1

        fingerprint = harness.runtime.bound_plan.fingerprint
        bundle = backend._device_graph_bundles[fingerprint]
        backend._device_graph_bundles[fingerprint] = replace(
            bundle,
            key=replace(bundle.key, num_envs=bundle.key.num_envs + 1),
        )
        with pytest.raises(BackendBatchContractError, match="changed capture key"):
            harness.step()
        rejected = backend.get_device_graph_diagnostics()
        assert rejected.launch_count == after_dtype_rejection.launch_count
        assert rejected.stale_rejection_count == 1
        assert rejected.active_keys == ()


def test_unannounced_or_device_abi_storage_replacement_is_rejected() -> None:
    """A stale pointer cannot pass diagnostics or be repaired as model-only DR."""

    with runtime_harness(num_envs=8, seed=2, max_episode_steps=100) as harness:
        backend = _backend(harness)
        harness.wait()
        original = backend._device_data.qpos
        backend._device_data.qpos = backend._warp.clone(original)
        backend._warp.synchronize()

        with pytest.raises(BackendBatchContractError, match="addresses changed"):
            backend.get_device_graph_diagnostics(verify_storage=True)
        with pytest.raises(BackendBatchContractError, match="ABI storage replacement"):
            backend._recapture_device_graphs_after_storage_change()
        before_launches = backend.get_device_graph_diagnostics().launch_count
        with pytest.raises(BackendBatchContractError, match="graph storage is stale"):
            harness.step()
        diagnostics = backend.get_device_graph_diagnostics()
        assert diagnostics.launch_count == before_launches
        assert diagnostics.stale_rejection_count >= 3


@pytest.mark.parametrize(
    ("method", "return_value"),
    (
        ("get_cuda_driver_version", (12, 3)),
        ("is_mempool_enabled", False),
    ),
)
def test_device_profile_rejects_unsupported_graph_runtime(
    method: str, return_value: object
) -> None:
    """Zero-roundtrip device execution has no eager fallback."""

    warp = load_mjwarp_dependencies().warp
    with patch.object(warp, method, return_value=return_value):
        with pytest.raises(BackendBatchContractError, match="eager physics"):
            with runtime_harness(num_envs=2, seed=3, max_episode_steps=100):
                pass


def test_graph_diagnostics_contract_rejects_inconsistent_identity() -> None:
    key = DeviceGraphCaptureKey(
        backend_type="mjwarp",
        plan_fingerprint="plan",
        num_envs=8,
        state_dtype="float32",
        control_dtype="float32",
        physics_substeps=4,
        storage_generation=0,
        storage_fingerprint="storage",
    )
    with pytest.raises(DeviceGraphContractError, match="match the diagnostics"):
        DeviceGraphDiagnostics(
            backend_type="mjwarp",
            execution_mode=DeviceGraphExecutionMode.CUDA_GRAPH,
            active_keys=(replace(key, storage_generation=1),),
            storage_buffers=(
                DeviceGraphBufferAddress(
                    name="model.field",
                    address=1,
                    shape=(1,),
                    dtype="float32",
                    device="cuda:0",
                ),
            ),
            storage_generation=0,
            storage_fingerprint="storage",
            capture_count=1,
            launch_count=0,
            recapture_count=0,
            stale_rejection_count=0,
            eager_fallback_count=0,
            storage_verification_count=1,
            instrumentation_complete=True,
        )
