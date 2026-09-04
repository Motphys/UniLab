from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from unisim.dr.types import (
    RESET_TERM_BASE_MASS,
    RESET_TERM_BODY_MASS,
    RESET_TERM_DOF_ARMATURE,
    RESET_TERM_GEOM_FRICTION,
    RESET_TERM_GRAVITY,
    RESET_TERM_KP,
)

from unilab.dr import (
    INTERVAL_TERM_BODY_FORCE,
    INTERVAL_TERM_PUSH,
    DomainRandomizationCapabilities,
    DomainRandomizationManager,
    DomainRandomizationProvider,
    IntervalRandomizationPlan,
    IntervalTermOp,
    ResetPlan,
    ResetRandomizationPayload,
)
from unilab.dr.dr_utils import build_common_reset_randomization


def test_capabilities_filter_reset_payload_drops_unsupported_terms():
    capabilities = DomainRandomizationCapabilities(
        supported_reset_terms=frozenset({RESET_TERM_BASE_MASS})
    )
    payload = ResetRandomizationPayload(
        base_mass_delta=np.array([0.25]),
        gravity=np.array([[0.0, 0.0, -3.71]]),
        kp=np.array([[12.0, 12.0]]),
    )

    filtered, unsupported = capabilities.filter_reset_payload(payload)

    assert unsupported == frozenset({RESET_TERM_GRAVITY, RESET_TERM_KP})
    assert filtered is not None
    assert filtered.base_mass_delta is not None
    np.testing.assert_allclose(filtered.base_mass_delta, np.array([0.25]))
    assert filtered.gravity is None
    assert filtered.kp is None


def test_build_common_reset_randomization_samples_gravity_vector():
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            domain_rand=SimpleNamespace(
                randomize_gravity=True,
                gravity_range=[[-1.0, -2.0, -10.5], [1.0, 2.0, -8.5]],
            )
        )
    )

    payload = build_common_reset_randomization(env, num_reset=8)

    assert payload is not None
    assert payload.gravity is not None
    assert payload.gravity.shape == (8, 3)
    assert payload.requested_terms() == frozenset({RESET_TERM_GRAVITY})
    assert np.all(payload.gravity[:, 0] >= -1.0)
    assert np.all(payload.gravity[:, 0] <= 1.0)
    assert np.all(payload.gravity[:, 1] >= -2.0)
    assert np.all(payload.gravity[:, 1] <= 2.0)
    assert np.all(payload.gravity[:, 2] >= -10.5)
    assert np.all(payload.gravity[:, 2] <= -8.5)


def test_build_common_reset_randomization_samples_mass_ground_friction_and_armature():
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            domain_rand=SimpleNamespace(
                randomize_body_mass=True,
                body_mass_multiplier_range=[0.5, 0.5],
                randomize_ground_friction=True,
                ground_friction_multiplier_range=[2.0, 2.0],
                randomize_dof_armature=True,
                dof_armature_multiplier_range=[3.0, 3.0],
            )
        )
    )
    base_body_mass = np.asarray([0.0, 10.0, 2.0, 0.5], dtype=np.float64)
    base_geom_friction = np.asarray([[1.0, 0.005, 0.0001], [0.8, 0.004, 0.0002]], dtype=np.float64)
    base_dof_armature = np.asarray([0.0, 0.01, 0.02, 0.0], dtype=np.float64)

    payload = build_common_reset_randomization(
        env,
        num_reset=3,
        base_body_mass=base_body_mass,
        base_geom_friction=base_geom_friction,
        ground_geom_id=1,
        base_dof_armature=base_dof_armature,
    )

    assert payload is not None
    assert payload.requested_terms() == frozenset(
        {RESET_TERM_BODY_MASS, RESET_TERM_GEOM_FRICTION, RESET_TERM_DOF_ARMATURE}
    )
    assert payload.body_mass is not None
    np.testing.assert_allclose(payload.body_mass[:, 0], 0.0)
    np.testing.assert_allclose(
        payload.body_mass[:, 1:], np.broadcast_to(base_body_mass[1:] * 0.5, (3, 3))
    )
    assert payload.geom_friction is not None
    expected_friction = np.broadcast_to(base_geom_friction, (3, 2, 3)).copy()
    expected_friction[:, 1, 0] *= 2.0
    np.testing.assert_allclose(payload.geom_friction, expected_friction)
    assert payload.dof_armature is not None
    expected_armature = np.broadcast_to(base_dof_armature, (3, 4)).copy()
    expected_armature[:, [1, 2]] *= 3.0
    np.testing.assert_allclose(payload.dof_armature, expected_armature)


@dataclass
class _FakeBackend:
    capabilities: DomainRandomizationCapabilities
    backend_type: str = "motrix"

    def __post_init__(self) -> None:
        self.last_randomization: ResetRandomizationPayload | None = None
        self.interval_plans: list[IntervalRandomizationPlan] = []

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        return self.capabilities

    def set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: ResetRandomizationPayload | None = None,
    ) -> None:
        self.last_randomization = randomization

    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        self.interval_plans.append(plan)


@dataclass
class _FakeTimedBackend:
    """Backend that reports set_state sub-timings via the extended contract."""

    capabilities: DomainRandomizationCapabilities
    timing: dict[str, float]
    backend_type: str = "motrix"

    def __post_init__(self) -> None:
        self.last_randomization: ResetRandomizationPayload | None = None
        self.call_count = 0

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        return self.capabilities

    def set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: ResetRandomizationPayload | None = None,
    ) -> dict:
        self.last_randomization = randomization
        self.call_count += 1
        return {"timing": dict(self.timing)}


class _FakeProvider(DomainRandomizationProvider):
    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        return None

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        return ResetPlan(
            env_ids=env_ids,
            qpos=np.zeros((len(env_ids), 8), dtype=np.float32),
            qvel=np.zeros((len(env_ids), 7), dtype=np.float32),
            info_updates={"commands": np.zeros((len(env_ids), 3), dtype=np.float32)},
            randomization=ResetRandomizationPayload(
                base_mass_delta=np.full((len(env_ids),), 0.1, dtype=np.float32),
                kp=np.full((len(env_ids), 2), 5.0, dtype=np.float32),
            ),
        )

    def build_reset_observation(
        self, env: Any, env_ids: np.ndarray, info_updates: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        return {"obs": np.zeros((len(env_ids), 1), dtype=np.float32)}


def test_manager_skips_unsupported_reset_terms_with_warning(caplog):
    backend = _FakeBackend(
        capabilities=DomainRandomizationCapabilities(
            supported_reset_terms=frozenset({RESET_TERM_BASE_MASS})
        )
    )
    env = SimpleNamespace(_backend=backend)
    manager = DomainRandomizationManager(env, _FakeProvider())

    with caplog.at_level(logging.WARNING):
        obs, info = manager.reset(np.array([0, 1], dtype=np.int32))

    assert obs["obs"].shape == (2, 1)
    assert info["commands"].shape == (2, 3)
    assert backend.last_randomization is not None
    assert backend.last_randomization.base_mass_delta is not None
    np.testing.assert_allclose(backend.last_randomization.base_mass_delta, np.array([0.1, 0.1]))
    assert backend.last_randomization.kp is None
    assert (
        "motrix backend does not support reset randomization terms: kp; skipping them."
        in caplog.text
    )


def test_manager_keeps_supported_reset_terms_without_warning(caplog):
    backend = _FakeBackend(
        capabilities=DomainRandomizationCapabilities(
            supported_reset_terms=frozenset({RESET_TERM_BASE_MASS, RESET_TERM_KP})
        )
    )
    env = SimpleNamespace(_backend=backend)
    manager = DomainRandomizationManager(env, _FakeProvider())

    with caplog.at_level(logging.WARNING):
        obs, info = manager.reset(np.array([0, 1], dtype=np.int32))

    assert obs["obs"].shape == (2, 1)
    assert info["commands"].shape == (2, 3)
    assert backend.last_randomization is not None
    assert backend.last_randomization.base_mass_delta is not None
    assert backend.last_randomization.kp is not None
    assert "skipping them" not in caplog.text


def test_manager_merges_backend_set_state_sub_timings():
    """Backend-reported ``{"timing": {...}}`` from set_state flows into
    ``last_reset_timing_ms`` next to ``dr_reset_set_state_ms``."""
    reported = {
        "set_state_mask_ms": 0.12,
        "set_state_data_slice_ms": 0.34,
        "set_state_forward_kinematic_ms": 2.1,
        "set_state_internal_gap_ms": 0.05,
    }
    backend = _FakeTimedBackend(
        capabilities=DomainRandomizationCapabilities(
            supported_reset_terms=frozenset({RESET_TERM_BASE_MASS, RESET_TERM_KP})
        ),
        timing=reported,
    )
    env = SimpleNamespace(_backend=backend)
    manager = DomainRandomizationManager(env, _FakeProvider())

    obs, _ = manager.reset(np.array([0, 1, 2], dtype=np.int32))

    assert obs["obs"].shape == (3, 1)
    assert backend.call_count == 1
    timings = manager.last_reset_timing_ms
    # Outer wall-clock measurement still present.
    assert "dr_reset_set_state_ms" in timings
    # Every reported backend sub-key is merged in.
    for key, expected in reported.items():
        assert key in timings
        assert timings[key] == pytest.approx(expected)


def test_manager_tolerates_missing_or_malformed_backend_timing():
    """Backends may return ``None`` (unchanged behavior) or a dict with
    non-numeric values; the manager must not crash and must not add spurious
    sub-keys in either case."""
    plain_backend = _FakeBackend(
        capabilities=DomainRandomizationCapabilities(
            supported_reset_terms=frozenset({RESET_TERM_BASE_MASS, RESET_TERM_KP})
        )
    )
    env = SimpleNamespace(_backend=plain_backend)
    manager = DomainRandomizationManager(env, _FakeProvider())
    manager.reset(np.array([0], dtype=np.int32))
    plain_keys = set(manager.last_reset_timing_ms)
    assert "dr_reset_set_state_ms" in plain_keys
    assert not any(k.startswith("set_state_") for k in plain_keys)

    malformed_backend = _FakeTimedBackend(
        capabilities=DomainRandomizationCapabilities(
            supported_reset_terms=frozenset({RESET_TERM_BASE_MASS, RESET_TERM_KP})
        ),
        timing={"set_state_mask_ms": "not a number", "set_state_data_slice_ms": 0.5},
    )
    env = SimpleNamespace(_backend=malformed_backend)
    manager = DomainRandomizationManager(env, _FakeProvider())
    manager.reset(np.array([0], dtype=np.int32))
    timings = manager.last_reset_timing_ms
    # Malformed value dropped, well-formed one merged.
    assert "set_state_mask_ms" not in timings
    assert timings["set_state_data_slice_ms"] == pytest.approx(0.5)


class _FakeIntervalProvider(_FakeProvider):
    def __init__(self, plan: IntervalRandomizationPlan | None) -> None:
        self._plan = plan

    def build_interval_randomization_plan(
        self, env: Any, step_counter: int
    ) -> IntervalRandomizationPlan | None:
        return self._plan


def _interval_manager(
    plan: IntervalRandomizationPlan | None,
    capabilities: DomainRandomizationCapabilities,
) -> tuple[DomainRandomizationManager, _FakeBackend]:
    backend = _FakeBackend(capabilities=capabilities)
    env = SimpleNamespace(_backend=backend)
    manager = DomainRandomizationManager(env, _FakeIntervalProvider(plan))
    return manager, backend


def test_manager_dispatches_custom_interval_term_without_manager_change():
    """A backend-owned custom term flows through the generic manager dispatch:
    declaring it in ``supported_interval_terms`` is enough, no manager edit."""
    plan = IntervalRandomizationPlan(
        ops=(IntervalTermOp("custom_shake", np.zeros((2, 3), dtype=np.float64)),)
    )
    capabilities = DomainRandomizationCapabilities(
        supported_interval_terms=frozenset({"custom_shake"})
    )
    manager, backend = _interval_manager(plan, capabilities)

    manager.apply_interval_randomization_if_due(step_counter=10)

    assert backend.interval_plans == [plan]


def test_manager_rejects_interval_term_missing_from_capabilities():
    plan = IntervalRandomizationPlan(
        ops=(IntervalTermOp("custom_shake", np.zeros((2, 3), dtype=np.float64)),)
    )
    manager, backend = _interval_manager(plan, DomainRandomizationCapabilities())

    with pytest.raises(NotImplementedError) as excinfo:
        manager.apply_interval_randomization_if_due(step_counter=10)

    assert "custom_shake" in str(excinfo.value)
    assert backend.backend_type in str(excinfo.value)
    assert backend.interval_plans == []


def test_manager_dispatches_legacy_fields_plan_via_capability_bools():
    """Legacy-field plans are still adapted through ``iter_ops()`` and checked
    against the deprecated legacy capability bools."""
    plan = IntervalRandomizationPlan(
        push_perturbation_limit=np.asarray([10.0, 10.0, 5.0]),
        body_ids=np.asarray([3], dtype=np.int32),
        body_force=np.zeros((4, 1, 3), dtype=np.float64),
    )
    capabilities = DomainRandomizationCapabilities(
        supports_interval_push=True,
        supports_interval_body_force=True,
    )
    manager, backend = _interval_manager(plan, capabilities)

    manager.apply_interval_randomization_if_due(step_counter=10)

    assert backend.interval_plans == [plan]


def test_manager_skips_none_and_empty_interval_plans():
    capabilities = DomainRandomizationCapabilities(
        supported_interval_terms=frozenset({INTERVAL_TERM_PUSH})
    )
    none_manager, none_backend = _interval_manager(None, capabilities)
    none_manager.apply_interval_randomization_if_due(step_counter=10)
    assert none_backend.interval_plans == []

    empty_manager, empty_backend = _interval_manager(IntervalRandomizationPlan(), capabilities)
    empty_manager.apply_interval_randomization_if_due(step_counter=10)
    assert empty_backend.interval_plans == []


def test_manager_dispatches_mixed_legacy_and_ops_plan():
    plan = IntervalRandomizationPlan(
        push_perturbation_limit=np.asarray([10.0, 10.0, 5.0]),
        ops=(IntervalTermOp("custom_shake", np.zeros((2, 3), dtype=np.float64)),),
    )
    capabilities = DomainRandomizationCapabilities(
        supports_interval_push=True,
        supported_interval_terms=frozenset({"custom_shake"}),
    )
    manager, backend = _interval_manager(plan, capabilities)

    manager.apply_interval_randomization_if_due(step_counter=10)

    assert backend.interval_plans == [plan]


def test_manager_detects_unsupported_terms_from_both_representations():
    capabilities = DomainRandomizationCapabilities(
        supports_interval_push=True,
        supported_interval_terms=frozenset({"custom_shake"}),
    )
    # Legacy-derived term missing from capabilities.
    legacy_plan = IntervalRandomizationPlan(
        body_ids=np.asarray([0], dtype=np.int32),
        body_torque=np.zeros((2, 1, 3), dtype=np.float64),
    )
    manager, backend = _interval_manager(legacy_plan, capabilities)
    with pytest.raises(NotImplementedError, match="body_torque"):
        manager.apply_interval_randomization_if_due(step_counter=10)
    assert backend.interval_plans == []

    # Explicit op term missing from capabilities.
    ops_plan = IntervalRandomizationPlan(
        push_perturbation_limit=np.asarray([1.0, 1.0, 1.0]),
        ops=(IntervalTermOp("custom_twist", np.zeros((2, 3), dtype=np.float64)),),
    )
    manager, backend = _interval_manager(ops_plan, capabilities)
    with pytest.raises(NotImplementedError, match="custom_twist"):
        manager.apply_interval_randomization_if_due(step_counter=10)
    assert backend.interval_plans == []


def test_manager_dispatches_multi_op_plan_in_one_backend_call():
    plan = IntervalRandomizationPlan(
        ops=(
            IntervalTermOp(INTERVAL_TERM_PUSH, np.asarray([10.0, 10.0, 5.0])),
            IntervalTermOp(
                INTERVAL_TERM_BODY_FORCE,
                np.zeros((4, 1, 3), dtype=np.float64),
                body_ids=np.asarray([3], dtype=np.int32),
            ),
        )
    )
    capabilities = DomainRandomizationCapabilities(
        supported_interval_terms=frozenset({INTERVAL_TERM_PUSH, INTERVAL_TERM_BODY_FORCE})
    )
    manager, backend = _interval_manager(plan, capabilities)

    manager.apply_interval_randomization_if_due(step_counter=10)

    assert backend.interval_plans == [plan]


def test_interval_plan_with_ops_pickle_round_trip():
    plan = IntervalRandomizationPlan(
        ops=(
            IntervalTermOp(INTERVAL_TERM_PUSH, np.asarray([10.0, 10.0, 5.0])),
            IntervalTermOp(
                "custom_shake",
                np.ones((2, 3), dtype=np.float64),
                body_ids=np.asarray([1, 2], dtype=np.int32),
            ),
        )
    )

    restored = pickle.loads(pickle.dumps(plan, protocol=4))

    assert [op.term for op in restored.ops] == [INTERVAL_TERM_PUSH, "custom_shake"]
    np.testing.assert_array_equal(restored.ops[0].payload, plan.ops[0].payload)
    np.testing.assert_array_equal(restored.ops[1].payload, plan.ops[1].payload)
    assert restored.ops[0].body_ids is None
    assert restored.ops[1].body_ids is not None
    np.testing.assert_array_equal(restored.ops[1].body_ids, plan.ops[1].body_ids)
