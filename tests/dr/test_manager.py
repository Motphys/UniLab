from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from unilab.dr import (
    DomainRandomizationCapabilities,
    DomainRandomizationManager,
    DomainRandomizationProvider,
    ResetPlan,
    ResetRandomizationPayload,
)
from unilab.dr.dr_utils import build_common_reset_randomization
from unilab.dr.types import RESET_TERM_BASE_MASS, RESET_TERM_GRAVITY, RESET_TERM_KP


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


@dataclass
class _FakeBackend:
    capabilities: DomainRandomizationCapabilities
    backend_type: str = "motrix"

    def __post_init__(self) -> None:
        self.last_randomization: ResetRandomizationPayload | None = None

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
