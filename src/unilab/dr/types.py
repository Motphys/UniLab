from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

RESET_TERM_BASE_COM = "base_com_offset"
RESET_TERM_BASE_MASS = "base_mass_delta"
RESET_TERM_BODY_IQUAT = "body_iquat"
RESET_TERM_BODY_INERTIA = "body_inertia"
RESET_TERM_KP = "kp"
RESET_TERM_KD = "kd"


@dataclass(frozen=True)
class DomainRandomizationCapabilities:
    supported_reset_terms: frozenset[str] = field(default_factory=frozenset)
    supports_interval_push: bool = False

    def supports_reset_term(self, term: str) -> bool:
        return term in self.supported_reset_terms

    def get_unsupported_reset_terms(self, requested_terms: frozenset[str]) -> frozenset[str]:
        return frozenset(term for term in requested_terms if not self.supports_reset_term(term))

    def filter_reset_payload(
        self, payload: ResetRandomizationPayload
    ) -> tuple[ResetRandomizationPayload | None, frozenset[str]]:
        unsupported = self.get_unsupported_reset_terms(payload.requested_terms())
        if not unsupported:
            return payload, frozenset()

        filtered = ResetRandomizationPayload(
            base_mass_delta=(
                payload.base_mass_delta if self.supports_reset_term(RESET_TERM_BASE_MASS) else None
            ),
            base_com_offset=(
                payload.base_com_offset if self.supports_reset_term(RESET_TERM_BASE_COM) else None
            ),
            body_iquat=(
                payload.body_iquat if self.supports_reset_term(RESET_TERM_BODY_IQUAT) else None
            ),
            body_inertia=(
                payload.body_inertia if self.supports_reset_term(RESET_TERM_BODY_INERTIA) else None
            ),
            kp=payload.kp if self.supports_reset_term(RESET_TERM_KP) else None,
            kd=payload.kd if self.supports_reset_term(RESET_TERM_KD) else None,
        )
        return (None if filtered.is_empty() else filtered), unsupported


@dataclass
class ResetRandomizationPayload:
    base_mass_delta: np.ndarray | None = None
    base_com_offset: np.ndarray | None = None
    body_iquat: np.ndarray | None = None
    body_inertia: np.ndarray | None = None
    kp: np.ndarray | None = None
    kd: np.ndarray | None = None

    def requested_terms(self) -> frozenset[str]:
        terms: set[str] = set()
        if self.base_mass_delta is not None:
            terms.add(RESET_TERM_BASE_MASS)
        if self.base_com_offset is not None:
            terms.add(RESET_TERM_BASE_COM)
        if self.body_iquat is not None:
            terms.add(RESET_TERM_BODY_IQUAT)
        if self.body_inertia is not None:
            terms.add(RESET_TERM_BODY_INERTIA)
        if self.kp is not None:
            terms.add(RESET_TERM_KP)
        if self.kd is not None:
            terms.add(RESET_TERM_KD)
        return frozenset(terms)

    def is_empty(self) -> bool:
        return not self.requested_terms()


@dataclass
class IntervalRandomizationPlan:
    push_perturbation_limit: np.ndarray | None = None

    def is_empty(self) -> bool:
        return self.push_perturbation_limit is None


@dataclass
class ResetPlan:
    env_ids: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    info_updates: dict[str, Any]
    randomization: ResetRandomizationPayload | None = None
