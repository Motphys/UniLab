"""Public performance evidence for production G1 mjwarp DR profiles."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
from tests.training.device_runtime_harness import forbid_host_roundtrip, runtime_harness

pytestmark = pytest.mark.slow


@dataclass(frozen=True)
class _Profile:
    profile_id: str
    runtime_kwargs: dict[str, bool]
    model_targets: tuple[str, ...]
    direct_fields: tuple[str, ...]
    derived_fields: tuple[str, ...]
    recompute_kind: str


_PROFILES = (
    _Profile("disabled", {}, (), (), (), "none"),
    _Profile(
        "tier_b_pd",
        {"randomize_kp": True, "randomize_kd": True},
        ("actuator.pd_damping", "actuator.pd_stiffness"),
        ("actuator_biasprm", "actuator_gainprm"),
        (),
        "none",
    ),
    _Profile(
        "tier_c_armature",
        {"randomize_dof_armature": True},
        ("joint.armature",),
        ("dof_armature",),
        (
            "actuator_acc0",
            "body_invweight0",
            "dof_invweight0",
            "tendon_invweight0",
            "tendon_length0",
        ),
        "set_const_0",
    ),
    _Profile(
        "tier_c_mixed",
        {
            "randomize_dof_armature": True,
            "randomize_body_gravity_compensation": True,
        },
        ("body.gravity_compensation", "joint.armature"),
        ("body_gravcomp", "dof_armature"),
        (
            "actuator_acc0",
            "body_invweight0",
            "body_subtreemass",
            "dof_invweight0",
            "tendon_invweight0",
            "tendon_length0",
        ),
        "set_const",
    ),
)


@pytest.mark.parametrize("profile", _PROFILES, ids=lambda profile: profile.profile_id)
def test_public_diagnostics_cover_receipt_bytes_and_lifecycle_deltas(
    profile: _Profile,
) -> None:
    with runtime_harness(
        num_envs=8,
        seed=705829,
        max_episode_steps=16,
        **profile.runtime_kwargs,
    ) as harness:
        harness.wait()
        before = harness.runtime.capture_performance_diagnostics()
        assert before.model_targets == profile.model_targets
        assert before.direct_fields == profile.direct_fields
        assert before.derived_fields == profile.derived_fields
        assert before.recompute_kind == profile.recompute_kind
        assert before.instrumentation_complete
        assert before.lifecycle.instrumentation_complete
        assert before.lifecycle.invalid_model_sample_rows == 0
        assert before.graph.instrumentation_complete
        assert before.graph.eager_fallback_count == 0
        assert before.graph.stale_rejection_count == 0

        materialization = before.materialization
        if profile.model_targets:
            assert materialization is not None
            assert materialization.num_worlds == harness.runtime.num_envs
            assert (
                tuple(
                    field.field_name for field in materialization.fields if field.role == "direct"
                )
                == profile.direct_fields
            )
            assert (
                tuple(
                    field.field_name for field in materialization.fields if field.role == "derived"
                )
                == profile.derived_fields
            )
            assert materialization.expanded_model_bytes == sum(
                field.model_bytes for field in materialization.fields if field.replaced
            )
            assert materialization.baseline_bytes == sum(
                math.prod(field.compiled_default_shape) * 4
                for field in materialization.fields
                if field.role == "direct"
            )
            assert all(
                field.materialized_shape[0] == harness.runtime.num_envs
                and (field.model_bytes == 0 or field.materialized_address > 0)
                for field in materialization.fields
            )
            assert materialization.storage_generation == before.graph.storage_generation
            assert materialization.storage_fingerprint == before.graph.storage_fingerprint
        else:
            assert materialization is None

        stable_before = harness.runtime.stability_diagnostics
        assert stable_before is not None
        with forbid_host_roundtrip(harness.backend):
            harness.step(0.0)
        harness.wait()
        after = harness.runtime.capture_performance_diagnostics()
        stable_after = harness.runtime.stability_diagnostics
        assert stable_after is not None

        assert after.lifecycle.runtime_barriers == before.lifecycle.runtime_barriers + 2
        assert after.lifecycle.step_graph_launches == before.lifecycle.step_graph_launches + 1
        assert after.lifecycle.reset_graph_launches == before.lifecycle.reset_graph_launches + 1
        assert after.lifecycle.forward_graph_launches == before.lifecycle.forward_graph_launches + 1
        assert after.lifecycle.state_refreshes == before.lifecycle.state_refreshes + 2
        assert after.lifecycle.invalid_model_sample_rows == 0
        assert after.graph.launch_count == before.graph.launch_count + 3
        expected_recompute_delta = int(profile.recompute_kind != "none")
        assert (
            after.recompute_launch_count == before.recompute_launch_count + expected_recompute_delta
        )
        assert after.graph.storage_buffers == before.graph.storage_buffers
        assert stable_after.buffers == stable_before.buffers
        assert stable_after.address_churn == 0
        assert stable_after.warm_numeric_allocations == 0
        traffic = harness.runtime.traffic_diagnostics
        assert traffic.host_to_device_transfers == 0
        assert traffic.device_to_host_transfers == 0
        assert traffic.global_synchronizations == 0
        assert traffic.backend_allocations == 0


def test_runtime_surfaces_invalid_model_sample_rows_at_cold_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with runtime_harness(
        num_envs=8,
        seed=705871,
        max_episode_steps=1,
        randomize_dof_armature=True,
    ) as harness:
        harness.wait()
        stream_index = next(
            index
            for index, binding in enumerate(harness.runtime.event_bindings)
            if binding.event.term_key == "g1_randomize_dof_armature"
        )
        stream = harness.runtime._event_streams[stream_index]
        sample_candidate = stream._sample_candidate

        def sample_with_invalid_row() -> None:
            sample_candidate()
            stream._candidate[0].fill_(float("nan"))

        monkeypatch.setattr(stream, "_sample_candidate", sample_with_invalid_row)
        before = harness.runtime.capture_performance_diagnostics()
        with forbid_host_roundtrip(harness.backend):
            harness.step(0.0)
        harness.wait()
        after = harness.runtime.capture_performance_diagnostics()

        assert (
            after.lifecycle.invalid_model_sample_rows
            == before.lifecycle.invalid_model_sample_rows + 1
        )
