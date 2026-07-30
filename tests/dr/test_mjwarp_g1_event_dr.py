"""Production G1 Event DR semantics and physics-effect tests on real CUDA."""

from __future__ import annotations

from contextlib import ExitStack

import numpy as np
import pytest
import torch
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


def _set_uniform_state(harness: DeviceRuntimeHarness) -> None:
    qpos = np.tile(
        harness.backend.get_keyframe_qpos("stand"),
        (harness.backend.num_envs, 1),
    ).astype(np.float32)
    qvel = np.zeros(
        (harness.backend.num_envs, harness.backend.get_init_qvel().size),
        dtype=np.float32,
    )
    harness.backend.set_state(
        np.arange(harness.backend.num_envs, dtype=np.int32),
        qpos,
        qvel,
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
