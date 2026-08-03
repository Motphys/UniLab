"""Production G1 Event DR semantics and physics-effect tests on real CUDA."""

from __future__ import annotations

from contextlib import ExitStack

import numpy as np
import pytest
import torch
from tests.dr.mjwarp_model_mutation_support import reset_device_state
from tests.training.device_runtime_harness import (
    DeviceRuntimeHarness,
    forbid_host_roundtrip,
    runtime_harness,
)

from unilab.base.backend import DeviceTensorView
from unilab.base.backend.mjwarp.backend import MjwarpBackend
from unilab.dr.keyed_rng import keyed_random_reference

pytestmark = pytest.mark.slow

_NUM_ENVS = 8
_SEED = 705821
_MAX_EPISODE_STEPS = 8
_SELECTED = (1, 3, 6)
_COMPLEMENT = tuple(index for index in range(_NUM_ENVS) if index not in _SELECTED)


def _pd_snapshot(harness: DeviceRuntimeHarness) -> dict[str, np.ndarray]:
    backend = harness.backend
    assert isinstance(backend, MjwarpBackend)
    gain = backend._get_model_field_bridge("actuator_gainprm").detach().cpu().numpy().copy()
    bias = backend._get_model_field_bridge("actuator_biasprm").detach().cpu().numpy().copy()
    default_gain = (
        backend._get_model_default_bridge("actuator_gainprm").detach().cpu().numpy().copy()
    )
    default_bias = (
        backend._get_model_default_bridge("actuator_biasprm").detach().cpu().numpy().copy()
    )
    assert np.all(default_gain[:, :, 0] > 0.0)
    assert np.all(default_bias[:, :, 2] < 0.0)
    kp = gain[:, :, 0] / np.broadcast_to(default_gain[:, :, 0], gain[:, :, 0].shape)
    kd = bias[:, :, 2] / np.broadcast_to(default_bias[:, :, 2], bias[:, :, 2].shape)
    return {
        "gain": gain,
        "bias": bias,
        "default_gain": default_gain,
        "default_bias": default_bias,
        "kp": kp,
        "kd": kd,
    }


def _armature_snapshot(harness: DeviceRuntimeHarness) -> tuple[np.ndarray, np.ndarray]:
    backend = harness.backend
    assert isinstance(backend, MjwarpBackend)
    mutation_plan = harness.runtime.kernel_binding.mutation_plan
    assert mutation_plan is not None
    binding = next(
        item
        for item in harness.runtime.event_bindings
        if item.event.term_key == "g1_randomize_dof_armature"
    )
    spec = mutation_plan.specs[binding.event.mutation_index]
    raw_ids = np.asarray(
        tuple(backend._root_qvel_dim + entity_id for entity_id in spec.target.entity_ids),
        dtype=np.intp,
    )
    armature = backend._get_model_field_bridge("dof_armature").detach().cpu().numpy().copy()
    default = backend._get_model_default_bridge("dof_armature").detach().cpu().numpy().copy()
    selected = armature[:, raw_ids]
    selected_default = np.broadcast_to(default[:, raw_ids], selected.shape)
    assert np.all(selected_default > 0.0)
    return selected, selected_default


def _gravity_compensation_snapshot(harness: DeviceRuntimeHarness) -> np.ndarray:
    backend = harness.backend
    assert isinstance(backend, MjwarpBackend)
    mutation_plan = harness.runtime.kernel_binding.mutation_plan
    assert mutation_plan is not None
    binding = next(
        item
        for item in harness.runtime.event_bindings
        if item.event.term_key == "g1_randomize_body_gravity_compensation"
    )
    spec = mutation_plan.specs[binding.event.mutation_index]
    assert spec.target.entity_ids == (1, 16)
    gravity_compensation = (
        backend._get_model_field_bridge("body_gravcomp").detach().cpu().numpy().copy()
    )
    return gravity_compensation[:, np.asarray(spec.target.entity_ids, dtype=np.intp)]


def _expected_event_values(
    harness: DeviceRuntimeHarness,
    *,
    trigger_counts: np.ndarray,
) -> dict[str, np.ndarray]:
    env_ids = np.arange(harness.runtime.num_envs, dtype=np.int64)
    return {
        binding.event.term_key: keyed_random_reference(
            binding.random_spec,
            run_seed=_SEED,
            env_ids=env_ids,
            trigger_counts=trigger_counts,
        )[..., 0]
        for binding in harness.runtime.event_bindings
    }


def _set_episode_steps(harness: DeviceRuntimeHarness, selected_value: int, other: int) -> None:
    values = torch.full(
        (_NUM_ENVS,),
        other,
        dtype=torch.int64,
        device=harness.device,
    )
    values[list(_SELECTED)] = selected_value
    harness.runtime.set_episode_length_buffer(values)


def _assert_finite_transition(harness: DeviceRuntimeHarness) -> None:
    transition = harness.transition
    views = [transition.reward]
    views.extend(buffer.view for buffer in transition.observations)
    views.extend(buffer.view for buffer in transition.terminal_observations)
    views.extend(buffer.view for buffer in transition.final_observations)
    for view in views:
        assert bool(torch.isfinite(view.torch()).all().item())


def test_g1_reset_events_are_keyed_partial_and_stable() -> None:
    with runtime_harness(
        num_envs=_NUM_ENVS,
        seed=_SEED,
        max_episode_steps=_MAX_EPISODE_STEPS,
        randomize_kp=True,
        randomize_kd=True,
        kp_multiplier_range=(0.7, 1.3),
        kd_multiplier_range=(0.8, 1.2),
    ) as harness:
        harness.wait()
        runtime = harness.runtime
        initial = _pd_snapshot(harness)
        counts = dict(runtime.capture_event_trigger_counts())
        assert all(
            np.array_equal(value, np.ones(_NUM_ENVS, dtype=np.int64)) for value in counts.values()
        )

        expected = _expected_event_values(
            harness,
            trigger_counts=np.zeros(_NUM_ENVS, dtype=np.int64),
        )
        np.testing.assert_allclose(initial["kp"], expected["g1_randomize_kp"], rtol=1e-6)
        np.testing.assert_allclose(initial["kd"], expected["g1_randomize_kd"], rtol=1e-6)
        np.testing.assert_allclose(
            initial["kp"],
            np.broadcast_to(initial["kp"][:, :1], initial["kp"].shape),
            rtol=3e-7,
        )
        np.testing.assert_allclose(
            initial["kd"],
            np.broadcast_to(initial["kd"][:, :1], initial["kd"].shape),
            rtol=3e-7,
        )
        assert np.unique(initial["kp"][:, 0]).size > 1
        assert np.unique(initial["kd"][:, 0]).size > 1
        assert not np.allclose(initial["kp"][:, 0], initial["kd"][:, 0])
        assert np.all((initial["kp"] >= 0.7) & (initial["kp"] <= 1.3))
        assert np.all((initial["kd"] >= 0.8) & (initial["kd"] <= 1.2))
        np.testing.assert_allclose(initial["gain"][:, :, 0], -initial["bias"][:, :, 1])

        stability_before = runtime.stability_diagnostics
        assert stability_before is not None
        event_addresses_before = {
            item.name: item.address
            for item in stability_before.buffers
            if item.name.startswith("runtime.event.")
        }
        assert event_addresses_before

        # A normal policy step submits an all-false CUDA reset mask.  Event
        # streams must retain both samples and counters without a host branch.
        with forbid_host_roundtrip(harness.backend):
            harness.step(0.0)
        harness.wait()
        mask = harness.transition.final_observation_mask.torch().cpu().numpy().copy()
        assert not mask.any()
        all_false = _pd_snapshot(harness)
        np.testing.assert_array_equal(all_false["gain"], initial["gain"])
        np.testing.assert_array_equal(all_false["bias"], initial["bias"])
        for key, value in runtime.capture_event_trigger_counts():
            np.testing.assert_array_equal(value, counts[key])

        # Force only selected rows to timeout through the public device-owned
        # episode-length buffer, then repeat to prove DEFAULT never accumulates.
        _set_episode_steps(harness, _MAX_EPISODE_STEPS - 1, 1)
        with forbid_host_roundtrip(harness.backend):
            harness.step(0.0)
        harness.wait()
        first_mask = harness.transition.final_observation_mask.torch().cpu().numpy().copy()
        np.testing.assert_array_equal(
            first_mask,
            np.asarray([index in _SELECTED for index in range(_NUM_ENVS)]),
        )
        first = _pd_snapshot(harness)
        first_counts = dict(runtime.capture_event_trigger_counts())
        expected_counts = np.ones(_NUM_ENVS, dtype=np.int64)
        expected_counts[list(_SELECTED)] = 2
        for value in first_counts.values():
            np.testing.assert_array_equal(value, expected_counts)
        first_expected = _expected_event_values(
            harness,
            trigger_counts=expected_counts - 1,
        )
        np.testing.assert_allclose(first["kp"], first_expected["g1_randomize_kp"], rtol=1e-6)
        np.testing.assert_allclose(first["kd"], first_expected["g1_randomize_kd"], rtol=1e-6)
        np.testing.assert_array_equal(
            first["gain"][list(_COMPLEMENT)], initial["gain"][list(_COMPLEMENT)]
        )
        np.testing.assert_array_equal(
            first["bias"][list(_COMPLEMENT)], initial["bias"][list(_COMPLEMENT)]
        )

        _set_episode_steps(harness, _MAX_EPISODE_STEPS - 1, 2)
        with forbid_host_roundtrip(harness.backend):
            harness.step(0.0)
        harness.wait()
        second = _pd_snapshot(harness)
        second_counts = dict(runtime.capture_event_trigger_counts())
        expected_counts[list(_SELECTED)] = 3
        for value in second_counts.values():
            np.testing.assert_array_equal(value, expected_counts)
        second_expected = _expected_event_values(
            harness,
            trigger_counts=expected_counts - 1,
        )
        np.testing.assert_allclose(second["kp"], second_expected["g1_randomize_kp"], rtol=1e-6)
        np.testing.assert_allclose(second["kd"], second_expected["g1_randomize_kd"], rtol=1e-6)
        np.testing.assert_array_equal(
            second["gain"][list(_COMPLEMENT)], initial["gain"][list(_COMPLEMENT)]
        )
        np.testing.assert_array_equal(
            second["bias"][list(_COMPLEMENT)], initial["bias"][list(_COMPLEMENT)]
        )
        np.testing.assert_allclose(second["gain"][:, :, 0], -second["bias"][:, :, 1])
        _assert_finite_transition(harness)

        for _, traffic in runtime.event_traffic_diagnostics:
            assert traffic.host_to_device_transfers == 0
            assert traffic.device_to_host_transfers == 0
            assert traffic.global_synchronizations == 0
            assert traffic.sample_allocations == 0
        traffic = runtime.traffic_diagnostics
        assert traffic.host_to_device_transfers == 0
        assert traffic.device_to_host_transfers == 0
        assert traffic.global_synchronizations == 0
        assert traffic.backend_allocations == 0

        stability_after = runtime.stability_diagnostics
        assert stability_after is not None
        event_addresses_after = {
            item.name: item.address
            for item in stability_after.buffers
            if item.name.startswith("runtime.event.")
        }
        assert event_addresses_after == event_addresses_before
        assert stability_after.warm_numeric_allocations == 0
        assert stability_after.address_churn == 0
        assert stability_after.instrumentation_complete


def test_g1_armature_event_recomputes_selected_worlds_on_device() -> None:
    with runtime_harness(
        num_envs=_NUM_ENVS,
        seed=_SEED,
        max_episode_steps=_MAX_EPISODE_STEPS,
        randomize_dof_armature=True,
        dof_armature_multiplier_range=(0.6, 1.4),
    ) as harness:
        harness.wait()
        backend = harness.backend
        assert isinstance(backend, MjwarpBackend)
        initial, default = _armature_snapshot(harness)
        expected = _expected_event_values(
            harness,
            trigger_counts=np.zeros(_NUM_ENVS, dtype=np.int64),
        )["g1_randomize_dof_armature"]
        np.testing.assert_allclose(initial / default, expected, rtol=1e-6)
        np.testing.assert_allclose(
            initial / default,
            np.broadcast_to((initial / default)[:, :1], initial.shape),
            rtol=3e-7,
        )

        mutation_plan = harness.runtime.kernel_binding.mutation_plan
        assert mutation_plan is not None
        before = backend.get_model_recompute_diagnostics(mutation_plan)
        assert before is not None
        assert before.kind.value == "set_const_0"
        addresses_before = harness.runtime.stability_diagnostics
        assert addresses_before is not None

        _set_episode_steps(harness, _MAX_EPISODE_STEPS - 1, 1)
        with forbid_host_roundtrip(backend):
            harness.step(0.0)
        harness.wait()
        after_values, after_default = _armature_snapshot(harness)
        counts = dict(harness.runtime.capture_event_trigger_counts())["g1_randomize_dof_armature"]
        expected_counts = np.ones(_NUM_ENVS, dtype=np.int64)
        expected_counts[list(_SELECTED)] = 2
        np.testing.assert_array_equal(counts, expected_counts)
        expected_after = _expected_event_values(
            harness,
            trigger_counts=expected_counts - 1,
        )["g1_randomize_dof_armature"]
        np.testing.assert_allclose(after_values / after_default, expected_after, rtol=1e-6)
        np.testing.assert_array_equal(after_values[list(_COMPLEMENT)], initial[list(_COMPLEMENT)])

        after = backend.get_model_recompute_diagnostics(mutation_plan)
        assert after is not None
        assert after.launch_count == before.launch_count + 1
        traffic = harness.runtime.traffic_diagnostics
        assert traffic.host_to_device_transfers == 0
        assert traffic.device_to_host_transfers == 0
        assert traffic.global_synchronizations == 0
        assert traffic.backend_allocations == 0
        addresses_after = harness.runtime.stability_diagnostics
        assert addresses_after is not None
        assert addresses_after.buffers == addresses_before.buffers
        assert addresses_after.warm_numeric_allocations == 0
        assert addresses_after.address_churn == 0


def test_g1_mixed_tier_c_events_join_one_strongest_recompute() -> None:
    with runtime_harness(
        num_envs=_NUM_ENVS,
        seed=_SEED,
        max_episode_steps=_MAX_EPISODE_STEPS,
        randomize_dof_armature=True,
        randomize_body_gravity_compensation=True,
        dof_armature_multiplier_range=(0.6, 1.4),
        body_gravity_compensation_range=(-0.2, 0.4),
    ) as harness:
        harness.wait()
        backend = harness.backend
        assert isinstance(backend, MjwarpBackend)
        mutation_plan = harness.runtime.kernel_binding.mutation_plan
        assert mutation_plan is not None
        terms = {binding.event.term_key for binding in harness.runtime.event_bindings}
        assert {
            "g1_randomize_dof_armature",
            "g1_randomize_body_gravity_compensation",
        }.issubset(terms)

        diagnostics = backend.get_model_recompute_diagnostics(mutation_plan)
        assert diagnostics is not None
        assert diagnostics.kind.value == "set_const"
        assert diagnostics.direct_fields == ("body_gravcomp", "dof_armature")
        assert diagnostics.derived_fields == (
            "actuator_acc0",
            "body_invweight0",
            "body_subtreemass",
            "dof_invweight0",
            "tendon_invweight0",
            "tendon_length0",
        )
        launch_count = diagnostics.launch_count

        initial_armature, initial_armature_default = _armature_snapshot(harness)
        initial_gravity_compensation = _gravity_compensation_snapshot(harness)
        initial_expected = _expected_event_values(
            harness,
            trigger_counts=np.zeros(_NUM_ENVS, dtype=np.int64),
        )
        np.testing.assert_allclose(
            initial_armature / initial_armature_default,
            initial_expected["g1_randomize_dof_armature"],
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            initial_gravity_compensation,
            initial_expected["g1_randomize_body_gravity_compensation"],
            rtol=1e-6,
            atol=1e-8,
        )

        for other, expected_count in ((1, 2), (2, 3)):
            before_armature, _ = _armature_snapshot(harness)
            before_gravity_compensation = _gravity_compensation_snapshot(harness)
            _set_episode_steps(harness, _MAX_EPISODE_STEPS - 1, other)
            with forbid_host_roundtrip(backend):
                harness.step(0.0)
            harness.wait()

            mask = harness.transition.final_observation_mask.torch().cpu().numpy().copy()
            np.testing.assert_array_equal(
                mask,
                np.asarray([index in _SELECTED for index in range(_NUM_ENVS)]),
            )
            after_armature, after_armature_default = _armature_snapshot(harness)
            after_gravity_compensation = _gravity_compensation_snapshot(harness)
            np.testing.assert_array_equal(
                after_armature[list(_COMPLEMENT)],
                before_armature[list(_COMPLEMENT)],
            )
            np.testing.assert_array_equal(
                after_gravity_compensation[list(_COMPLEMENT)],
                before_gravity_compensation[list(_COMPLEMENT)],
            )
            assert not np.array_equal(
                after_armature[list(_SELECTED)],
                before_armature[list(_SELECTED)],
            )
            assert not np.array_equal(
                after_gravity_compensation[list(_SELECTED)],
                before_gravity_compensation[list(_SELECTED)],
            )

            trigger_counts = np.ones(_NUM_ENVS, dtype=np.int64)
            trigger_counts[list(_SELECTED)] = expected_count
            expected = _expected_event_values(
                harness,
                trigger_counts=trigger_counts - 1,
            )
            np.testing.assert_allclose(
                after_armature / after_armature_default,
                expected["g1_randomize_dof_armature"],
                rtol=1e-6,
            )
            np.testing.assert_allclose(
                after_gravity_compensation,
                expected["g1_randomize_body_gravity_compensation"],
                rtol=1e-6,
                atol=1e-8,
            )
            for term_key in (
                "g1_randomize_dof_armature",
                "g1_randomize_body_gravity_compensation",
            ):
                counts = dict(harness.runtime.capture_event_trigger_counts())[term_key]
                np.testing.assert_array_equal(counts, trigger_counts)

            diagnostics = backend.get_model_recompute_diagnostics(mutation_plan)
            assert diagnostics is not None
            launch_count += 1
            assert diagnostics.kind.value == "set_const"
            assert diagnostics.launch_count == launch_count

        traffic = harness.runtime.traffic_diagnostics
        assert traffic.host_to_device_transfers == 0
        assert traffic.device_to_host_transfers == 0
        assert traffic.global_synchronizations == 0
        assert traffic.backend_allocations == 0


def _set_uniform_state(harness: DeviceRuntimeHarness) -> None:
    qpos = np.tile(
        harness.backend.get_keyframe_qpos("stand"),
        (harness.backend.num_envs, 1),
    ).astype(np.float32)
    qvel = np.zeros(
        (harness.backend.num_envs, harness.backend.get_init_qvel().size),
        dtype=np.float32,
    )
    assert isinstance(harness.backend, MjwarpBackend)
    reset_device_state(
        backend=harness.backend,
        plan=harness.runtime.bound_plan,
        placement=harness.placement,
        base_name="pelvis",
        qpos=qpos,
        qvel=qvel,
    )


def _qpos_snapshot(harness: DeviceRuntimeHarness) -> np.ndarray:
    backend = harness.backend
    assert isinstance(backend, MjwarpBackend)
    return backend._ensure_device_bridge().qpos.detach().cpu().numpy().copy()


def test_partial_event_changes_only_selected_world_physics_not_control() -> None:
    with ExitStack() as stack:
        changed = stack.enter_context(
            runtime_harness(
                num_envs=_NUM_ENVS,
                seed=_SEED,
                max_episode_steps=16,
                randomize_kp=True,
                randomize_kd=True,
                kp_multiplier_range=(0.4, 1.6),
                kd_multiplier_range=(0.5, 1.5),
            )
        )
        unchanged = stack.enter_context(
            runtime_harness(
                num_envs=_NUM_ENVS,
                seed=_SEED,
                max_episode_steps=16,
                randomize_kp=True,
                randomize_kd=True,
                kp_multiplier_range=(0.4, 1.6),
                kd_multiplier_range=(0.5, 1.5),
            )
        )
        changed.wait()
        unchanged.wait()
        before_changed = _pd_snapshot(changed)
        before_unchanged = _pd_snapshot(unchanged)
        np.testing.assert_array_equal(before_changed["gain"], before_unchanged["gain"])
        np.testing.assert_array_equal(before_changed["bias"], before_unchanged["bias"])

        _set_uniform_state(changed)
        _set_uniform_state(unchanged)
        _set_episode_steps(changed, 15, 0)
        unchanged.runtime.set_episode_length_buffer(
            torch.zeros(_NUM_ENVS, dtype=torch.int64, device=unchanged.device)
        )
        with forbid_host_roundtrip(changed.backend):
            changed.step(0.0)
        with forbid_host_roundtrip(unchanged.backend):
            unchanged.step(0.0)
        changed.wait()
        unchanged.wait()
        expected_mask = np.asarray([index in _SELECTED for index in range(_NUM_ENVS)])
        np.testing.assert_array_equal(
            changed.transition.final_observation_mask.torch().cpu().numpy(),
            expected_mask,
        )
        assert not bool(unchanged.transition.final_observation_mask.torch().any().item())

        after_changed = _pd_snapshot(changed)
        after_unchanged = _pd_snapshot(unchanged)
        np.testing.assert_array_equal(
            after_changed["gain"][list(_COMPLEMENT)],
            after_unchanged["gain"][list(_COMPLEMENT)],
        )
        np.testing.assert_array_equal(
            after_changed["bias"][list(_COMPLEMENT)],
            after_unchanged["bias"][list(_COMPLEMENT)],
        )
        assert not np.allclose(
            after_changed["gain"][list(_SELECTED)],
            after_unchanged["gain"][list(_SELECTED)],
        )
        assert not np.allclose(
            after_changed["bias"][list(_SELECTED)],
            after_unchanged["bias"][list(_SELECTED)],
        )

        # Re-establish identical physics state after the selective reset.  The
        # next control is identical, so only the changed Model rows may diverge.
        _set_uniform_state(changed)
        _set_uniform_state(unchanged)
        changed.step(0.35)
        unchanged.step(0.35)
        changed.wait()
        unchanged.wait()
        np.testing.assert_array_equal(
            changed.runtime._control.detach().cpu().numpy(),
            unchanged.runtime._control.detach().cpu().numpy(),
        )
        changed_qpos = _qpos_snapshot(changed)
        unchanged_qpos = _qpos_snapshot(unchanged)
        complement_error = float(
            np.max(np.abs(changed_qpos[list(_COMPLEMENT)] - unchanged_qpos[list(_COMPLEMENT)]))
        )
        selected_error = float(
            np.max(np.abs(changed_qpos[list(_SELECTED)] - unchanged_qpos[list(_SELECTED)]))
        )
        assert complement_error <= 2e-6
        assert selected_error > max(1e-5, 10.0 * complement_error)
