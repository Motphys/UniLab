from __future__ import annotations

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
from unilab.dr.types import RESET_TERM_BASE_MASS, RESET_TERM_KP


def test_capabilities_filter_reset_payload_drops_unsupported_terms():
    capabilities = DomainRandomizationCapabilities(
        supported_reset_terms=frozenset({RESET_TERM_BASE_MASS})
    )
    payload = ResetRandomizationPayload(
        base_mass_delta=np.array([0.25]),
        kp=np.array([[12.0, 12.0]]),
    )

    filtered, unsupported = capabilities.filter_reset_payload(payload)

    assert unsupported == frozenset({RESET_TERM_KP})
    assert filtered is not None
    assert filtered.base_mass_delta is not None
    np.testing.assert_allclose(filtered.base_mass_delta, np.array([0.25]))
    assert filtered.kp is None


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
