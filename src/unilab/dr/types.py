from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

RESET_TERM_BASE_COM = "base_com_offset"
RESET_TERM_BASE_MASS = "base_mass_delta"


@dataclass(frozen=True)
class DomainRandomizationCapabilities:
    supported_reset_terms: frozenset[str] = field(default_factory=frozenset)
    supports_interval_push: bool = False


@dataclass
class ResetRandomizationPayload:
    base_mass_delta: np.ndarray | None = None
    base_com_offset: np.ndarray | None = None

    def requested_terms(self) -> frozenset[str]:
        terms: set[str] = set()
        if self.base_mass_delta is not None:
            terms.add(RESET_TERM_BASE_MASS)
        if self.base_com_offset is not None:
            terms.add(RESET_TERM_BASE_COM)
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
