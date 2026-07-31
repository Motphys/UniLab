"""Atomic Model-storage graph recapture acceptance for production ``mjwarp``."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
from tests.dr.mjwarp_model_mutation_support import (
    PlanKey,
    ResetBatchBuffers,
    bind_model_plan,
    control_batch,
    model_mutation_runtime,
    state_tensor,
    wait_result,
)

from unilab.base.backend import (
    BackendBatchContractError,
    BackendIORequirements,
    MutationGraphInvalidation,
    MutationOperation,
    RowSelection,
)
from unilab.base.backend.mjwarp import (
    MjwarpModelFieldRole,
    MjwarpModelInvalidationOutcome,
    MjwarpModelMaterializationReceipt,
)
from unilab.base.backend.mjwarp.recompute import (
    MjwarpModelRecomputeKind,
    recompute_derived_fields,
)

pytestmark = pytest.mark.slow

_SELECTED = (1, 5)
_PAIRED_CONTROLS = (0, 4)
_DIRECT_FIELD = "dof_armature"
_RECOMPUTE_KIND = MjwarpModelRecomputeKind.SET_CONST_0


def _bind_second_cadence(runtime: Any) -> Any:
    bound = runtime.plan
    requirements = BackendIORequirements(
        state_fields=bound.state.fields,
        control=replace(
            bound.control,
            physics_substeps_per_control=bound.control.physics_substeps_per_control + 1,
        ),
        execution_profile=bound.execution_profile,
        hot_path_budget=bound.hot_path_budget,
        reset_hot_path_budget=bound.reset_hot_path_budget,
    )
    return runtime.backend.bind_task_io(requirements)


def _launch_old_forward_graphs(runtime: Any, bundles: dict[str, Any]) -> None:
    backend = runtime.backend
    bridge = backend._ensure_device_bridge()
    with (
        torch.cuda.stream(bridge.physics_stream),
        backend._warp.ScopedStream(bridge.warp_physics_stream),
    ):
        for bundle in bundles.values():
            backend._warp.capture_launch(bundle.forward_graph)
    backend._warp.synchronize_device()


@pytest.mark.parametrize("num_envs", (32, 128))
def test_field_expansion_invalidates_and_recaptures_all_graph_consumers(
    num_envs: int,
) -> None:
    """Recapture, rollback, pointer identity, and physics effect form one oracle."""

    key = PlanKey("joint.armature", MutationOperation.SET, mixed_state=True)
    with model_mutation_runtime(num_envs=num_envs, plan_keys=()) as runtime:
        backend = runtime.backend
        second_plan = _bind_second_cadence(runtime)
        before = backend.get_device_graph_diagnostics(verify_storage=True)
        assert len(before.active_keys) == 2
        assert second_plan.fingerprint in {item.plan_fingerprint for item in before.active_keys}
        old_bundles = dict(backend._device_graph_bundles)
        old_arrays = {
            name: getattr(backend._device_model, name)
            for name in (_DIRECT_FIELD, *recompute_derived_fields(_RECOMPUTE_KIND))
        }
        old_addresses = {name: int(value.ptr or 0) for name, value in old_arrays.items()}
        old_bridge = backend._warp.to_torch(old_arrays[_DIRECT_FIELD])
        backend._model_bridge_cache[_DIRECT_FIELD] = old_bridge
        assert int(old_bridge.data_ptr()) == old_addresses[_DIRECT_FIELD]

        original_capture = backend._capture_device_graph_bundle
        capture_observations: list[tuple[tuple[int, ...], tuple[bool, ...], int]] = []

        def fail_second_capture(capture_key: Any, *, bound: Any, recapture: bool) -> Any:
            assert recapture
            current_arrays = tuple(getattr(backend._device_model, name) for name in old_arrays)
            addresses = tuple(int(value.ptr or 0) for value in current_arrays)
            replacements = tuple(
                current is not old_arrays[name]
                for name, current in zip(old_arrays, current_arrays, strict=True)
            )
            bridge_address = int(backend._model_bridge_cache[_DIRECT_FIELD].data_ptr())
            capture_observations.append((addresses, replacements, bridge_address))
            assert backend._model_materialization_receipt is None
            if len(capture_observations) == 2:
                raise RuntimeError("injected second graph capture failure")
            return original_capture(capture_key, bound=bound, recapture=recapture)

        with patch.object(
            backend,
            "_capture_device_graph_bundle",
            side_effect=fail_second_capture,
        ):
            with pytest.raises(BackendBatchContractError, match="transaction rolled back"):
                bind_model_plan(runtime, key)

        assert len(capture_observations) == 2
        for addresses, replacements, bridge_address in capture_observations:
            assert all(replacements)
            for name, address in zip(old_arrays, addresses, strict=True):
                if old_addresses[name]:
                    assert address != old_addresses[name]
            assert bridge_address == addresses[tuple(old_arrays).index(_DIRECT_FIELD)]
        assert backend._model_materialization_receipt is None
        assert backend._model_default_baselines == {}
        assert backend._expanded_model_fields == frozenset()
        assert backend._model_bridge_cache == {_DIRECT_FIELD: old_bridge}
        assert backend._model_bridge_generation == 0
        assert backend._model_sensor_context is None
        assert backend._model_sensor_generation == 0
        assert not backend._model_materialization_poisoned
        assert not backend._device_graph_storage_poisoned
        for name, original in old_arrays.items():
            assert getattr(backend._device_model, name) is original
        for fingerprint, old_bundle in old_bundles.items():
            assert backend._device_graph_bundles[fingerprint] is old_bundle

        rolled_back = backend.get_device_graph_diagnostics(verify_storage=True)
        assert (
            replace(
                rolled_back,
                storage_verification_count=before.storage_verification_count,
            )
            == before
        )
        _launch_old_forward_graphs(runtime, old_bundles)

        plan = bind_model_plan(runtime, key)
        assert backend._model_materialization_receipt is not None
        receipt = cast(
            MjwarpModelMaterializationReceipt,
            backend._model_materialization_receipt,
        )
        receipt.verify_fingerprint()
        fields = {item.field_name: item for item in receipt.fields}
        assert fields[_DIRECT_FIELD].role is MjwarpModelFieldRole.DIRECT
        assert {
            name for name, item in fields.items() if item.role is MjwarpModelFieldRole.DERIVED
        } == set(recompute_derived_fields(_RECOMPUTE_KIND))
        for name, original in old_arrays.items():
            current = getattr(backend._device_model, name)
            assert current is not original
            assert tuple(current.shape)[0] == num_envs
            if old_addresses[name]:
                assert int(current.ptr or 0) != old_addresses[name]
            assert fields[name].materialized_address == int(current.ptr or 0)
        new_bridge = backend._model_bridge_cache[_DIRECT_FIELD]
        assert new_bridge is not old_bridge
        assert int(new_bridge.data_ptr()) == int(backend._device_model.dof_armature.ptr or 0)
        assert int(old_bridge.data_ptr()) == old_addresses[_DIRECT_FIELD]

        invalidations = {item.consumer: item for item in receipt.invalidations}
        for consumer in (
            MutationGraphInvalidation.RESET_GRAPH,
            MutationGraphInvalidation.FORWARD_GRAPH,
            MutationGraphInvalidation.STEP_GRAPH,
        ):
            assert invalidations[consumer].outcome is MjwarpModelInvalidationOutcome.REBUILT
            assert invalidations[consumer].affected_count == len(old_bundles)
        bridge_invalidation = invalidations[MutationGraphInvalidation.MODEL_BRIDGE_CACHE]
        assert bridge_invalidation.outcome is MjwarpModelInvalidationOutcome.REBUILT
        assert bridge_invalidation.affected_count == 1
        for consumer in (
            MutationGraphInvalidation.SENSOR_CONTEXT,
            MutationGraphInvalidation.SENSE_GRAPH,
        ):
            assert invalidations[consumer].outcome is MjwarpModelInvalidationOutcome.NOT_PRESENT
            assert invalidations[consumer].affected_count == 0

        recaptured = backend.get_device_graph_diagnostics(verify_storage=True)
        assert recaptured.storage_generation == before.storage_generation + 1
        assert recaptured.storage_fingerprint != before.storage_fingerprint
        assert recaptured.capture_count == before.capture_count + len(old_bundles)
        assert recaptured.recapture_count == before.recapture_count + len(old_bundles)
        assert recaptured.eager_fallback_count == 0
        assert receipt.graph_plan_fingerprints_before == tuple(sorted(old_bundles))
        assert receipt.graph_plan_fingerprints_after == tuple(sorted(old_bundles))
        assert receipt.storage_generation_after == recaptured.storage_generation
        assert receipt.storage_fingerprint_after == recaptured.storage_fingerprint
        assert receipt.model_bridge_generation_after == (receipt.model_bridge_generation_before + 1)
        assert receipt.sensor_generation_after == receipt.sensor_generation_before + 1
        for fingerprint, old_bundle in old_bundles.items():
            new_bundle = backend._device_graph_bundles[fingerprint]
            assert new_bundle is not old_bundle
            assert new_bundle.key.storage_generation == recaptured.storage_generation
            assert new_bundle.key.storage_fingerprint == recaptured.storage_fingerprint

        baseline = runtime.model_default(_DIRECT_FIELD).clone()
        qpos, _ = runtime.set_uniform_state(target_position=0.15, target_velocity=0.0)
        buffers = ResetBatchBuffers(runtime, plan)
        buffers.active_mask[list(_SELECTED)] = True
        buffers.values["model.value"][list(_SELECTED), 0, 0] = 0.15
        buffers.values["state.position"][list(_SELECTED), 0, 0] = 0.15
        buffers.values["state.velocity"][list(_SELECTED), 0, 0] = 0.0
        reset = backend.reset_batch(
            runtime.plan,
            RowSelection.all(runtime.num_envs),
            mutation_batch=buffers.publish(),
        )
        wait_result(reset)
        immediate_position = state_tensor(reset.reset_state, "dof.position")
        immediate_velocity = state_tensor(reset.reset_state, "dof.angular_velocity")
        torch.testing.assert_close(
            immediate_position[list(_SELECTED), runtime.dof_position_index],
            immediate_position[list(_PAIRED_CONTROLS), runtime.dof_position_index],
        )
        torch.testing.assert_close(
            immediate_velocity[list(_SELECTED), runtime.dof_velocity_index],
            immediate_velocity[list(_PAIRED_CONTROLS), runtime.dof_velocity_index],
        )
        torch.testing.assert_close(runtime.model_default(_DIRECT_FIELD), baseline)
        direct = runtime.model_field(_DIRECT_FIELD)
        complement = sorted(set(range(num_envs)).difference(_SELECTED))
        torch.testing.assert_close(
            direct[list(_SELECTED), runtime.raw_qvel_index],
            torch.full((len(_SELECTED),), 0.15, device=runtime.device),
        )
        torch.testing.assert_close(
            direct[complement, runtime.raw_qvel_index],
            baseline[0, runtime.raw_qvel_index].expand(len(complement)),
        )
        derived = runtime.model_field("dof_invweight0")[:, runtime.raw_qvel_index]
        selected_derived = derived[list(_SELECTED)]
        control_derived = derived[list(_PAIRED_CONTROLS)]
        assert bool(torch.isfinite(selected_derived).all())
        torch.testing.assert_close(control_derived[0], control_derived[1])
        assert float(torch.max(torch.abs(selected_derived - control_derived))) > 1.0e-7

        control = runtime.position_hold_control(qpos)
        control[:, runtime.actuator_id] += 0.25
        stepped = backend.step_batch(
            runtime.plan,
            control_batch(runtime, control, owner="graph-mutation-effect"),
            nsteps=1,
        )
        wait_result(stepped)
        velocity = state_tensor(stepped.terminal_state, "dof.angular_velocity")
        selected_velocity = velocity[list(_SELECTED), runtime.dof_velocity_index]
        control_velocity = velocity[list(_PAIRED_CONTROLS), runtime.dof_velocity_index]
        assert bool(torch.isfinite(selected_velocity).all())
        assert bool(torch.isfinite(control_velocity).all())
        assert float(torch.max(torch.abs(selected_velocity - control_velocity))) > 1.0e-5

        before_stale = backend.get_device_graph_diagnostics(verify_storage=True)
        backend._device_model.dof_armature = backend._warp.clone(backend._device_model.dof_armature)
        backend._warp.synchronize_device()
        with pytest.raises(BackendBatchContractError, match="addresses changed"):
            backend.get_device_graph_diagnostics(verify_storage=True)
        with pytest.raises(BackendBatchContractError, match="graph storage is stale"):
            backend.step_batch(
                runtime.plan,
                control_batch(runtime, control, owner="graph-mutation-stale"),
                nsteps=1,
            )
        stale = backend.get_device_graph_diagnostics()
        assert stale.launch_count == before_stale.launch_count
        assert stale.stale_rejection_count == before_stale.stale_rejection_count + 2
        assert stale.eager_fallback_count == 0
        assert stale.active_keys == ()
