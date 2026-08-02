"""Backend-neutral mutation performance diagnostics contract tests."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest

from unilab.base.backend import (
    BACKEND_PERFORMANCE_DIAGNOSTICS_VERSION,
    BackendDeviceLifecycleDiagnostics,
    BackendModelFieldDiagnostics,
    BackendModelMaterializationDiagnostics,
    BackendMutationPerformanceDiagnostics,
    BackendPerformanceDiagnosticsError,
    DeviceGraphBufferAddress,
    DeviceGraphDiagnostics,
    DeviceGraphExecutionMode,
)
from unilab.base.backend.base import SimBackend
from unilab.base.backend.mujoco.backend import MuJoCoBackend


def _graph(*, launches: int = 5) -> DeviceGraphDiagnostics:
    return DeviceGraphDiagnostics(
        backend_type="mjwarp",
        execution_mode=DeviceGraphExecutionMode.CUDA_GRAPH,
        active_keys=(),
        storage_buffers=(
            DeviceGraphBufferAddress(
                name="data.qpos",
                address=4096,
                shape=(8, 7),
                dtype="float32",
                device="cuda:0",
            ),
        ),
        storage_generation=2,
        storage_fingerprint="storage-v2",
        capture_count=1,
        launch_count=launches,
        recapture_count=0,
        stale_rejection_count=0,
        eager_fallback_count=0,
        storage_verification_count=1,
        instrumentation_complete=True,
    )


def _model_diagnostics() -> BackendMutationPerformanceDiagnostics:
    fields = (
        BackendModelFieldDiagnostics(
            field_name="actuator_acc0",
            role="derived",
            materialized_shape=(8, 4),
            materialized_address=8192,
            model_bytes=128,
            replaced=True,
            compiled_default_shape=(4,),
            per_world_default_shape=None,
        ),
        BackendModelFieldDiagnostics(
            field_name="dof_armature",
            role="direct",
            materialized_shape=(8, 4),
            materialized_address=12288,
            model_bytes=128,
            replaced=True,
            compiled_default_shape=(4,),
            per_world_default_shape=None,
        ),
    )
    return BackendMutationPerformanceDiagnostics(
        backend_type="mjwarp",
        backend_instance_id="mjwarp:1",
        mutation_plan_fingerprint="mutation-v1",
        model_targets=("joint.armature",),
        recompute_kind="set_const_0",
        direct_fields=("dof_armature",),
        derived_fields=("actuator_acc0",),
        recompute_capture_count=1,
        recompute_launch_count=2,
        materialization=BackendModelMaterializationDiagnostics(
            receipt_fingerprint="receipt-v1",
            num_worlds=8,
            fields=fields,
            expanded_model_bytes=256,
            baseline_bytes=16,
            storage_generation=2,
            storage_fingerprint="storage-v2",
        ),
        lifecycle=BackendDeviceLifecycleDiagnostics(
            runtime_barriers=3,
            step_graph_launches=1,
            reset_graph_launches=2,
            forward_graph_launches=2,
            state_refreshes=3,
            invalid_model_sample_rows=4,
        ),
        graph=_graph(),
    )


def test_mutation_performance_contract_accepts_consistent_model_and_state_evidence() -> None:
    model = _model_diagnostics()
    assert model.contract_version == BACKEND_PERFORMANCE_DIAGNOSTICS_VERSION
    assert model.lifecycle.invalid_model_sample_rows == 4
    assert model.materialization is not None
    assert model.materialization.expanded_model_bytes == sum(
        field.model_bytes for field in model.materialization.fields if field.replaced
    )

    state_only = BackendMutationPerformanceDiagnostics(
        backend_type="mjwarp",
        backend_instance_id="mjwarp:2",
        mutation_plan_fingerprint="mutation-state-v1",
        model_targets=(),
        recompute_kind="none",
        direct_fields=(),
        derived_fields=(),
        recompute_capture_count=0,
        recompute_launch_count=0,
        materialization=None,
        lifecycle=BackendDeviceLifecycleDiagnostics(
            runtime_barriers=0,
            step_graph_launches=0,
            reset_graph_launches=0,
            forward_graph_launches=0,
            state_refreshes=0,
            invalid_model_sample_rows=0,
        ),
        graph=_graph(launches=0),
    )
    assert state_only.materialization is None


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (
            lambda value: replace(
                value,
                lifecycle=replace(value.lifecycle, forward_graph_launches=3),
            ),
            "semantic lifecycle graph launches",
        ),
        (
            lambda value: replace(
                value,
                materialization=replace(
                    value.materialization,
                    storage_fingerprint="foreign-storage",
                ),
            ),
            "storage identities differ",
        ),
        (
            lambda value: replace(value, recompute_capture_count=0),
            "requires a captured graph",
        ),
    ),
)
def test_mutation_performance_contract_rejects_inconsistent_nested_evidence(
    mutate: Any,
    match: str,
) -> None:
    with pytest.raises(BackendPerformanceDiagnosticsError, match=match):
        mutate(_model_diagnostics())


def test_default_and_mujoco_backend_performance_diagnostics_fail_closed() -> None:
    assert "get_mutation_performance_diagnostics" not in MuJoCoBackend.__dict__
    dummy = cast(SimBackend, object())
    with pytest.raises(NotImplementedError, match="does not expose"):
        SimBackend.get_mutation_performance_diagnostics(dummy, cast(Any, None))
