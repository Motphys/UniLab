from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .provider import DomainRandomizationProvider
from .types import DomainRandomizationCapabilities

logger = logging.getLogger(__name__)


class DomainRandomizationManager:
    def __init__(self, env: Any, provider: DomainRandomizationProvider):
        self._env = env
        self._provider = provider
        self._capabilities: DomainRandomizationCapabilities = env._backend.get_dr_capabilities()
        self._warned_reset_terms: frozenset[str] = frozenset()
        self._provider.validate(env, self._capabilities)

    def apply_init_randomization(self) -> bool:
        plan = self._provider.build_init_randomization_plan(self._env)
        if plan is None or plan.is_empty():
            return False
        self._env._backend.apply_init_randomization(plan)
        return True

    def reset(self, env_ids: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        plan = self._provider.build_reset_plan(self._env, env_ids)
        payload = plan.randomization
        if payload is not None:
            payload, unsupported = self._capabilities.filter_reset_payload(payload)
            if unsupported:
                self._log_unsupported_reset_terms(unsupported)
        self._env._backend.set_state(
            plan.env_ids,
            plan.qpos,
            plan.qvel,
            randomization=payload,
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
        if (
            plan.body_linear_velocity_delta is not None
            and not self._capabilities.supports_interval_body_velocity_delta
        ):
            raise NotImplementedError(
                f"{self._env._backend.backend_type} backend does not support interval body velocity perturbation"
            )
        if plan.body_force is not None and not self._capabilities.supports_interval_body_force:
            raise NotImplementedError(
                f"{self._env._backend.backend_type} backend does not support interval body force perturbation"
            )
        self._env._backend.apply_interval_randomization(plan)

    def _log_unsupported_reset_terms(self, unsupported: frozenset[str]) -> None:
        new_terms = frozenset(term for term in unsupported if term not in self._warned_reset_terms)
        if not new_terms:
            return
        self._warned_reset_terms |= new_terms
        logger.warning(
            "%s backend does not support reset randomization terms: %s; skipping them.",
            self._env._backend.backend_type,
            ", ".join(sorted(new_terms)),
        )
