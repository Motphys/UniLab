from __future__ import annotations

import logging
import time
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
        self._last_reset_timing_ms: dict[str, float] = {}
        self._provider.validate(env, self._capabilities)

    @property
    def last_reset_timing_ms(self) -> dict[str, float]:
        return dict(self._last_reset_timing_ms)

    def apply_init_randomization(self) -> bool:
        plan = self._provider.build_init_randomization_plan(self._env)
        if plan is None or plan.is_empty():
            return False
        self._env._backend.apply_init_randomization(plan)
        return True

    def reset(self, env_ids: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        reset_t0 = time.perf_counter()
        self._last_reset_timing_ms = {}

        t0 = time.perf_counter()
        plan = self._provider.build_reset_plan(self._env, env_ids)
        plan_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        payload = plan.randomization
        if payload is not None:
            payload, unsupported = self._capabilities.filter_reset_payload(payload)
            if unsupported:
                self._log_unsupported_reset_terms(unsupported)
        payload_filter_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._env._backend.set_state(
            plan.env_ids,
            plan.qpos,
            plan.qvel,
            randomization=payload,
        )
        set_state_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        obs = self._provider.build_reset_observation(self._env, plan.env_ids, plan.info_updates)
        build_observation_ms = (time.perf_counter() - t0) * 1000.0

        total_ms = (time.perf_counter() - reset_t0) * 1000.0
        measured_ms = plan_ms + payload_filter_ms + set_state_ms + build_observation_ms
        timing = {
            "dr_reset_total_ms": total_ms,
            "dr_reset_plan_ms": plan_ms,
            "dr_reset_payload_filter_ms": payload_filter_ms,
            "dr_reset_set_state_ms": set_state_ms,
            "dr_reset_build_observation_ms": build_observation_ms,
            "dr_reset_internal_gap_ms": total_ms - measured_ms,
        }
        timing.update(self._env._backend.last_set_state_timing_ms)
        provider_timing = getattr(self._provider, "last_reset_observation_timing_ms", {})
        if isinstance(provider_timing, dict):
            timing.update(provider_timing)
        self._last_reset_timing_ms = timing
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
        logging.warning(
            "%s backend does not support reset randomization terms: %s; skipping them.",
            self._env._backend.backend_type,
            ", ".join(sorted(new_terms)),
        )
