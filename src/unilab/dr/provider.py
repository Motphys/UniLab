from __future__ import annotations

import abc
from typing import Any

import numpy as np

from .types import (
    DomainRandomizationCapabilities,
    InitRandomizationPlan,
    IntervalRandomizationPlan,
    ResetPlan,
)


class DomainRandomizationProvider(abc.ABC):
    @abc.abstractmethod
    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        pass

    def build_init_randomization_plan(self, env: Any) -> InitRandomizationPlan | None:
        return None

    @abc.abstractmethod
    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        pass

    @abc.abstractmethod
    def build_reset_observation(
        self, env: Any, env_ids: np.ndarray, info_updates: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        pass

    def build_interval_randomization_plan(
        self, env: Any, step_counter: int
    ) -> IntervalRandomizationPlan | None:
        return None
