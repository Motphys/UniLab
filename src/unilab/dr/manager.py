from __future__ import annotations

from typing import Any

import numpy as np

from .provider import DomainRandomizationProvider
from .types import DomainRandomizationCapabilities


class DomainRandomizationManager:
    def __init__(self, env: Any, provider: DomainRandomizationProvider):
        self._env = env
        self._provider = provider
        self._capabilities: DomainRandomizationCapabilities = env._backend.get_dr_capabilities()
        self._provider.validate(env, self._capabilities)

    def reset(self, env_ids: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        plan = self._provider.build_reset_plan(self._env, env_ids)
        payload = plan.randomization
        if payload is not None:
            unsupported = payload.requested_terms() - self._capabilities.supported_reset_terms
            if unsupported:
                terms = ", ".join(sorted(unsupported))
                raise NotImplementedError(
                    f"{self._env._backend.backend_type} backend does not support reset randomization terms: {terms}"
                )
        self._env._backend.set_state(
            plan.env_ids,
            plan.qpos,
            plan.qvel,
            randomization=plan.randomization,
        )
        obs = self._provider.build_reset_observation(self._env, plan.env_ids, plan.info_updates)
        return obs, plan.info_updates

    def apply_interval_randomization_if_due(self, step_counter: int) -> None:
        plan = self._provider.build_interval_randomization_plan(self._env, step_counter)
        if plan is None or plan.is_empty():
            return
        if (
            plan.push_perturbation_limit is not None
            and not self._capabilities.supports_interval_push
        ):
            raise NotImplementedError(
                f"{self._env._backend.backend_type} backend does not support interval push"
            )
        self._env._backend.apply_interval_randomization(plan)
