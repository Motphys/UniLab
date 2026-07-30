"""Atomic per-world Model storage materialization for the ``mjwarp`` owner."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np
import pytest
import torch
from tests.manager.test_g1_reference_differential import _cfg
from tests.training.device_runtime_harness import forbid_host_roundtrip, require_cuda

from unilab.base.backend import (
    BackendBatchContractError,
    DeviceBufferLease,
    DeviceCompletion,
    DeviceTensorView,
    MutationGraphInvalidation,
)
from unilab.base.backend.mjwarp import (
    MJWARP_MODEL_INVALIDATIONS,
    MjwarpModelFieldRole,
    MjwarpModelInvalidationOutcome,
    MjwarpModelMaterializationContractError,
    MjwarpModelMaterializationRequest,
)
from unilab.base.backend.mjwarp.backend import MjwarpBackend as ProductionMjwarpBackend
from unilab.envs.locomotion.g1.managed_device import create_g1_managed_device_runtime
from unilab.manager import DeviceManagedRuntime

pytestmark = pytest.mark.slow

_NUM_WORLDS = 4
_DIRECT_FIELD = "dof_armature"
_DERIVED_FIELDS = (
    "actuator_acc0",
    "body_invweight0",
    "dof_invweight0",
    "tendon_invweight0",
    "tendon_length0",
)


@contextmanager
def _owner(*, bind_runtime: bool) -> Iterator[tuple[Any, DeviceManagedRuntime | None]]:
    require_cuda()
    cfg = _cfg(
        max_episode_seconds=None,
        observation_noise_level=0.0,
        observation_noise_seed=None,
    )
    cfg.mjwarp_nconmax = 128
    cfg.mjwarp_njmax = 256
    assert cfg.scene is not None
    backend = ProductionMjwarpBackend(
        deepcopy(cfg.scene),
        _NUM_WORLDS,
        cfg.sim_dt,
        base_name=cfg.asset.base_name,
        nconmax=cfg.mjwarp_nconmax,
        njmax=cfg.mjwarp_njmax,
    )
    assert type(backend) is ProductionMjwarpBackend
    runtime = None
    try:
        if bind_runtime:
            runtime = create_g1_managed_device_runtime(
                backend=backend,
                cfg=cfg,
                reset_seed=7,
                max_episode_steps=100,
            )
        yield backend, runtime
    finally:
        backend.cleanup_scene_assets()


def _request(
    *,
    direct_fields: tuple[str, ...] = (_DIRECT_FIELD,),
    derived_fields: tuple[str, ...] = _DERIVED_FIELDS,
    per_world_default_fields: tuple[str, ...] = (),
) -> MjwarpModelMaterializationRequest:
    return MjwarpModelMaterializationRequest(
        num_worlds=_NUM_WORLDS,
        direct_fields=direct_fields,
        derived_fields=derived_fields,
        per_world_default_fields=per_world_default_fields,
    )


def _bind_second_cadence(backend: Any, runtime: DeviceManagedRuntime) -> Any:
    requirements = runtime.plan.backend_io
    changed_control = replace(
        requirements.control,
        physics_substeps_per_control=requirements.control.physics_substeps_per_control + 1,
    )
    return backend.bind_task_io(replace(requirements, control=changed_control))


def _run_reset_and_step(runtime: DeviceManagedRuntime) -> None:
    initial = runtime.reset()
    initial.completion.event.synchronize()
    contract = runtime.control_contract
    action = torch.zeros(
        (runtime.num_envs, *contract.row_shape),
        dtype=torch.float32,
        device=runtime.device,
    )
    lease = DeviceBufferLease("mjwarp-materialization-policy-action")
    completion = DeviceCompletion.record(
        placement=contract.placement,
        owner_id=lease.owner_id,
        epoch=lease.epoch,
    )
    transition = runtime.step(
        DeviceTensorView(
            tensor_handle=action,
            contract=contract,
            lease=lease,
            completion=completion,
        )
    )
    transition.completion.event.synchronize()


def _field_map(receipt: Any) -> dict[str, Any]:
    return {item.field_name: item for item in receipt.fields}


def _invalidation_map(receipt: Any) -> dict[MutationGraphInvalidation, Any]:
    return {item.consumer: item for item in receipt.invalidations}


def _assert_unpublished(backend: Any, originals: dict[str, Any]) -> None:
    assert backend._model_materialization_receipt is None
    assert backend._model_default_baselines == {}
    assert backend._expanded_model_fields == frozenset()
    assert backend._model_bridge_generation == 0
    assert backend._model_sensor_generation == 0
    assert not backend._model_materialization_poisoned
    for name, original in originals.items():
        assert getattr(backend._device_model, name) is original
    backend._verify_device_graph_storage()


def test_request_contract_is_canonical_fingerprinted_and_exported() -> None:
    request = _request()
    same = _request()
    assert request == same
    assert request.fingerprint == same.fingerprint
    assert request.all_fields == tuple(sorted((_DIRECT_FIELD, *_DERIVED_FIELDS)))
    request.verify_fingerprint()

    with pytest.raises(MjwarpModelMaterializationContractError, match="canonical and unique"):
        _request(direct_fields=("dof_damping", "dof_armature"), derived_fields=())
    with pytest.raises(MjwarpModelMaterializationContractError, match="overlap"):
        _request(direct_fields=("dof_armature",), derived_fields=("dof_armature",))
    with pytest.raises(MjwarpModelMaterializationContractError, match="must be direct fields"):
        _request(per_world_default_fields=("dof_damping",))

    object.__setattr__(request, "fingerprint", "mjwarp-model-materialization-v1:tampered")
    with pytest.raises(MjwarpModelMaterializationContractError, match="fingerprint"):
        request.verify_fingerprint()


def test_direct_and_derived_fields_materialize_atomically() -> None:
    """Real CUDA proves content, graph identity, idempotency, and next-step use."""

    with _owner(bind_runtime=True) as (backend, runtime):
        assert runtime is not None
        second_plan = _bind_second_cadence(backend, runtime)
        before = backend.get_device_graph_diagnostics(verify_storage=True)
        assert len(before.active_keys) == 2
        assert second_plan.fingerprint in {key.plan_fingerprint for key in before.active_keys}
        old_bundles = dict(backend._device_graph_bundles)

        request = _request()
        original_arrays = {
            name: getattr(backend._device_model, name) for name in request.all_fields
        }
        original_values = {
            name: np.ascontiguousarray(value.numpy()).copy()
            for name, value in original_arrays.items()
        }
        receipt = backend._materialize_model_fields(request)
        receipt.verify_fingerprint()

        fields = _field_map(receipt)
        assert tuple(fields) == request.all_fields
        expected_bytes = 0
        for name in request.all_fields:
            source = original_arrays[name]
            actual = getattr(backend._device_model, name)
            expected = np.ascontiguousarray(
                np.broadcast_to(
                    original_values[name],
                    (_NUM_WORLDS, *original_values[name].shape[1:]),
                )
            )
            np.testing.assert_array_equal(actual.numpy(), expected)
            assert actual is not source
            assert tuple(actual.shape) == (_NUM_WORLDS, *tuple(source.shape[1:]))
            assert fields[name].materialized_shape == tuple(actual.shape)
            assert fields[name].replaced
            assert fields[name].model_bytes == expected.nbytes
            assert fields[name].role is (
                MjwarpModelFieldRole.DIRECT
                if name == _DIRECT_FIELD
                else MjwarpModelFieldRole.DERIVED
            )
            if int(source.ptr or 0):
                assert int(actual.ptr or 0) != int(source.ptr or 0)
            expected_bytes += expected.nbytes
        assert receipt.expanded_model_bytes == expected_bytes == 2576
        compiled_default = backend._get_model_default_baseline(
            _DIRECT_FIELD,
            per_world=False,
        )
        assert receipt.baseline_bytes == compiled_default.numpy().nbytes == 140

        invalidations = _invalidation_map(receipt)
        assert tuple(item.consumer for item in receipt.invalidations) == MJWARP_MODEL_INVALIDATIONS
        for consumer in (
            MutationGraphInvalidation.STEP_GRAPH,
            MutationGraphInvalidation.FORWARD_GRAPH,
            MutationGraphInvalidation.RESET_GRAPH,
        ):
            assert invalidations[consumer].outcome is MjwarpModelInvalidationOutcome.REBUILT
            assert invalidations[consumer].affected_count == 2
        for consumer in (
            MutationGraphInvalidation.MODEL_BRIDGE_CACHE,
            MutationGraphInvalidation.SENSOR_CONTEXT,
            MutationGraphInvalidation.SENSE_GRAPH,
        ):
            assert invalidations[consumer].outcome is MjwarpModelInvalidationOutcome.NOT_PRESENT
            assert invalidations[consumer].affected_count == 0

        after = backend.get_device_graph_diagnostics(verify_storage=True)
        assert receipt.storage_generation_before == before.storage_generation
        assert receipt.storage_generation_after == after.storage_generation
        assert after.storage_generation == before.storage_generation + 1
        assert after.capture_count == before.capture_count + len(before.active_keys)
        assert after.recapture_count == before.recapture_count + len(before.active_keys)
        assert set(backend._device_graph_bundles) == set(old_bundles)
        for fingerprint, old_bundle in old_bundles.items():
            new_bundle = backend._device_graph_bundles[fingerprint]
            assert new_bundle is not old_bundle
            assert new_bundle.key.storage_generation == after.storage_generation
            assert new_bundle.key.storage_fingerprint == after.storage_fingerprint

        assert backend._materialize_model_fields(_request()) is receipt
        with pytest.raises(BackendBatchContractError, match="conflicting request"):
            backend._materialize_model_fields(
                _request(direct_fields=("dof_damping",), derived_fields=())
            )

        with forbid_host_roundtrip(backend):
            _run_reset_and_step(runtime)
        executed = backend.get_device_graph_diagnostics(verify_storage=True)
        assert executed.stale_rejection_count == 0
        assert executed.launch_count > after.launch_count
        assert backend._materialize_model_fields(_request()) is receipt

        tampered = replace(receipt)
        object.__setattr__(tampered, "fingerprint", "mjwarp-model-materialization-v1-receipt:bad")
        with pytest.raises(MjwarpModelMaterializationContractError, match="fingerprint"):
            tampered.verify_fingerprint()

        backend._device_model.dof_armature = backend._warp.clone(backend._device_model.dof_armature)
        backend._warp.synchronize_device()
        with pytest.raises(BackendBatchContractError, match="addresses changed"):
            backend._materialize_model_fields(_request())
        assert backend._device_graph_storage_poisoned
        with pytest.raises(BackendBatchContractError, match="graph storage is stale"):
            runtime.reset()


def test_compiled_and_per_world_baselines_are_isolated_from_model_writes() -> None:
    """A variant baseline remains distinct from the compiled CPU default."""

    with _owner(bind_runtime=False) as (backend, _):
        direct_source = backend._device_model.dof_damping
        compiled_source = np.ascontiguousarray(direct_source.numpy())
        variants = np.ascontiguousarray(
            np.broadcast_to(
                compiled_source,
                (_NUM_WORLDS, *compiled_source.shape[1:]),
            )
        ).copy()
        variants[1] += np.float32(0.25)
        variants[3] += np.float32(0.75)
        per_world = backend._allocate_model_array(
            "variant.dof_damping",
            direct_source,
            variants,
        )
        derived_source = backend._device_model.dof_invweight0
        compiled_derived = np.ascontiguousarray(derived_source.numpy())
        derived_variants = np.ascontiguousarray(
            np.broadcast_to(
                compiled_derived,
                (_NUM_WORLDS, *compiled_derived.shape[1:]),
            )
        ).copy()
        derived_variants[1] += np.float32(1.0)
        per_world_derived = backend._allocate_model_array(
            "variant.dof_invweight0",
            derived_source,
            derived_variants,
        )
        backend._warp.synchronize_device()
        backend._device_model.dof_damping = per_world
        backend._device_model.dof_invweight0 = per_world_derived
        backend._recapture_device_graphs_after_storage_change()

        backend._expanded_model_fields = frozenset({"dof_damping", "dof_invweight0"})
        old_bridge = backend._get_model_field_bridge("dof_damping")
        old_cache = backend._model_bridge_cache
        request = _request(
            direct_fields=("dof_damping",),
            derived_fields=("dof_invweight0",),
            per_world_default_fields=("dof_damping",),
        )
        generation_before = backend._device_graph_storage_generation
        receipt = backend._materialize_model_fields(request)
        fields = _field_map(receipt)
        assert not fields["dof_damping"].replaced
        assert not fields["dof_invweight0"].replaced
        assert fields["dof_damping"].per_world_default_shape == variants.shape
        np.testing.assert_array_equal(
            backend._device_model.dof_invweight0.numpy(),
            derived_variants,
        )
        assert receipt.expanded_model_bytes == 0
        assert backend._device_graph_storage_generation == generation_before
        assert backend._model_bridge_cache is old_cache
        retained_bridge = backend._model_bridge_cache["dof_damping"]
        assert retained_bridge is old_bridge
        assert retained_bridge.data_ptr() == int(per_world.ptr)

        bridge_invalidation = _invalidation_map(receipt)[
            MutationGraphInvalidation.MODEL_BRIDGE_CACHE
        ]
        assert bridge_invalidation.outcome is MjwarpModelInvalidationOutcome.UNCHANGED
        assert bridge_invalidation.affected_count == 0

        compiled_baseline = backend._get_model_default_baseline(
            "dof_damping",
            per_world=False,
        ).numpy()
        per_world_baseline = backend._get_model_default_baseline(
            "dof_damping",
            per_world=True,
        ).numpy()
        np.testing.assert_array_equal(
            compiled_baseline[0],
            np.asarray(backend._cpu_model.dof_damping, dtype=compiled_baseline.dtype),
        )
        np.testing.assert_array_equal(per_world_baseline, variants)
        compiled_snapshot = compiled_baseline.copy()
        per_world_snapshot = per_world_baseline.copy()

        retained_bridge[2].add_(torch.tensor(9.0, device=retained_bridge.device))
        backend._warp.synchronize_device()
        model_after = backend._device_model.dof_damping.numpy()
        assert not np.array_equal(model_after[2], variants[2])
        np.testing.assert_array_equal(model_after[[0, 1, 3]], variants[[0, 1, 3]])
        np.testing.assert_array_equal(
            backend._get_model_default_baseline(
                "dof_damping",
                per_world=False,
            ).numpy(),
            compiled_snapshot,
        )
        np.testing.assert_array_equal(
            backend._get_model_default_baseline(
                "dof_damping",
                per_world=True,
            ).numpy(),
            per_world_snapshot,
        )


def test_invalid_fields_and_request_identity_fail_before_publication() -> None:
    with _owner(bind_runtime=False) as (backend, _):
        tampered = _request(derived_fields=())
        object.__setattr__(tampered, "fingerprint", "tampered")
        with pytest.raises(BackendBatchContractError, match="request identity is corrupt"):
            backend._materialize_model_fields(tampered)

        with pytest.raises(BackendBatchContractError, match="world count"):
            backend._materialize_model_fields(
                MjwarpModelMaterializationRequest(
                    num_worlds=_NUM_WORLDS + 1,
                    direct_fields=("dof_armature",),
                )
            )
        with pytest.raises(BackendBatchContractError, match="has no field"):
            backend._materialize_model_fields(
                _request(direct_fields=("missing_field",), derived_fields=())
            )

        backend._device_model.non_array_field = 1
        with pytest.raises(BackendBatchContractError, match="not Warp array storage"):
            backend._materialize_model_fields(
                _request(direct_fields=("non_array_field",), derived_fields=())
            )

        source = backend._device_model.dof_damping
        backend._device_model.invalid_shape_field = backend._warp.zeros(
            shape=(2, *tuple(source.shape[1:])),
            dtype=source.dtype,
            device=source.device,
        )
        backend._device_graph_storage_buffers = backend._snapshot_device_graph_storage()
        backend._device_graph_storage_fingerprint = backend._graph_storage_fingerprint(
            backend._device_graph_storage_buffers
        )
        with pytest.raises(BackendBatchContractError, match="illegal world dimension"):
            backend._materialize_model_fields(
                _request(direct_fields=("invalid_shape_field",), derived_fields=())
            )
        assert backend._model_materialization_receipt is None
        assert backend._expanded_model_fields == frozenset()


def test_prepublication_failures_leave_no_partial_model_state() -> None:
    with _owner(bind_runtime=False) as (backend, _):
        originals = {
            name: getattr(backend._device_model, name) for name in ("dof_armature", "dof_damping")
        }
        two_fields = _request(
            direct_fields=("dof_armature", "dof_damping"),
            derived_fields=(),
        )
        original_empty = backend._warp.empty
        allocation_count = 0

        def fail_on_second_field(*args: Any, **kwargs: Any) -> Any:
            nonlocal allocation_count
            allocation_count += 1
            if allocation_count == 3:
                raise RuntimeError("injected second-field allocation failure")
            return original_empty(*args, **kwargs)

        with patch.object(backend._warp, "empty", side_effect=fail_on_second_field):
            with pytest.raises(BackendBatchContractError, match="allocate staged model field"):
                backend._materialize_model_fields(two_fields)
        _assert_unpublished(backend, originals)

        one_field = _request(derived_fields=())
        with patch.object(
            backend,
            "_snapshot_compiled_model_default",
            side_effect=RuntimeError("injected baseline failure"),
        ):
            with pytest.raises(BackendBatchContractError, match="before publication"):
                backend._materialize_model_fields(one_field)
        _assert_unpublished(backend, originals)

        cached_bridge = backend._warp.to_torch(originals["dof_armature"])
        backend._model_bridge_cache["dof_armature"] = cached_bridge
        with patch.object(
            backend._warp,
            "to_torch",
            side_effect=RuntimeError("injected bridge rebuild failure"),
        ):
            with pytest.raises(BackendBatchContractError, match="rebuild model bridge"):
                backend._materialize_model_fields(one_field)
        assert backend._model_bridge_cache == {"dof_armature": cached_bridge}
        assert getattr(backend._device_model, "dof_armature") is originals["dof_armature"]
        assert backend._model_materialization_receipt is None

        with patch.object(
            backend,
            "_prepare_model_sensor_context",
            side_effect=RuntimeError("injected sensor rebuild failure"),
        ):
            with pytest.raises(BackendBatchContractError, match="before publication"):
                backend._materialize_model_fields(one_field)
        assert backend._model_bridge_cache == {"dof_armature": cached_bridge}
        assert getattr(backend._device_model, "dof_armature") is originals["dof_armature"]
        assert backend._model_materialization_receipt is None

        receipt = backend._materialize_model_fields(one_field)
        assert receipt.request_fingerprint == one_field.fingerprint
        assert (
            _invalidation_map(receipt)[MutationGraphInvalidation.MODEL_BRIDGE_CACHE].outcome
            is MjwarpModelInvalidationOutcome.REBUILT
        )


def test_graph_capture_failure_restores_old_graphs_and_allows_retry() -> None:
    with _owner(bind_runtime=True) as (backend, runtime):
        assert runtime is not None
        _bind_second_cadence(backend, runtime)
        before = backend.get_device_graph_diagnostics(verify_storage=True)
        old_bundles = dict(backend._device_graph_bundles)
        original_field = backend._device_model.dof_damping
        request = _request(direct_fields=("dof_damping",), derived_fields=())
        original_capture = backend._capture_device_graph_bundle
        capture_count = 0

        def fail_second_capture(key: Any, *, recapture: bool) -> Any:
            nonlocal capture_count
            capture_count += 1
            if capture_count == 2:
                raise RuntimeError("injected graph capture failure")
            return original_capture(key, recapture=recapture)

        with patch.object(
            backend,
            "_capture_device_graph_bundle",
            side_effect=fail_second_capture,
        ):
            with pytest.raises(BackendBatchContractError, match="transaction rolled back"):
                backend._materialize_model_fields(request)

        assert backend._device_model.dof_damping is original_field
        assert backend._model_materialization_receipt is None
        assert not backend._model_materialization_poisoned
        assert not backend._device_graph_storage_poisoned
        assert set(backend._device_graph_bundles) == set(old_bundles)
        for fingerprint, old_bundle in old_bundles.items():
            assert backend._device_graph_bundles[fingerprint] is old_bundle
        restored = backend.get_device_graph_diagnostics(verify_storage=True)
        assert restored.storage_generation == before.storage_generation
        assert restored.storage_fingerprint == before.storage_fingerprint
        assert restored.active_keys == before.active_keys
        assert restored.capture_count == before.capture_count
        assert restored.recapture_count == before.recapture_count

        old_bundle = backend._require_device_graph_bundle(
            plan=runtime.bound_plan,
            nsteps=runtime.bound_plan.control.physics_substeps_per_control,
        )
        bridge = backend._ensure_device_bridge()
        with (
            torch.cuda.stream(bridge.physics_stream),
            backend._warp.ScopedStream(bridge.warp_physics_stream),
        ):
            backend._warp.capture_launch(old_bundle.forward_graph)
        backend._warp.synchronize_device()

        receipt = backend._materialize_model_fields(request)
        assert receipt.storage_generation_after == before.storage_generation + 1
        _run_reset_and_step(runtime)
        executed = backend.get_device_graph_diagnostics(verify_storage=True)
        assert executed.stale_rejection_count == before.stale_rejection_count


def test_unrecoverable_rollback_permanently_poisons_owner() -> None:
    with _owner(bind_runtime=False) as (backend, _):
        request = _request(derived_fields=())
        original = backend._device_model.dof_armature
        with (
            patch(
                "unilab.base.backend.mjwarp.backend.MjwarpModelMaterializationReceipt",
                side_effect=RuntimeError("injected receipt failure"),
            ),
            patch.object(
                backend,
                "_restore_device_graph_state",
                side_effect=RuntimeError("injected rollback failure"),
            ),
        ):
            with pytest.raises(BackendBatchContractError, match="permanently poisoned"):
                backend._materialize_model_fields(request)

        assert backend._device_model.dof_armature is original
        assert backend._model_materialization_poisoned
        assert backend._device_graph_storage_poisoned
        assert backend._device_graph_bundles == {}
        with pytest.raises(BackendBatchContractError, match="permanently poisoned"):
            backend._materialize_model_fields(request)


def test_first_materialization_after_runtime_barrier_is_rejected() -> None:
    with _owner(bind_runtime=True) as (backend, runtime):
        assert runtime is not None
        initial = runtime.reset()
        initial.completion.event.synchronize()
        assert backend._runtime_barrier_count == 1
        before = backend.get_device_graph_diagnostics(verify_storage=True)

        with pytest.raises(BackendBatchContractError, match="before the first runtime"):
            backend._materialize_model_fields(_request(derived_fields=()))

        after = backend.get_device_graph_diagnostics(verify_storage=True)
        assert backend._model_materialization_receipt is None
        assert backend._expanded_model_fields == frozenset()
        assert after.storage_generation == before.storage_generation
        assert after.storage_fingerprint == before.storage_fingerprint
        assert after.active_keys == before.active_keys
