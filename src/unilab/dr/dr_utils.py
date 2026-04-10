from __future__ import annotations

from typing import Any

import numpy as np

from unilab.base.dtype_config import get_global_dtype
from unilab.dr.types import (
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    ResetRandomizationPayload,
)


def zero_actions(num_reset: int, num_action: int) -> np.ndarray:
    return np.zeros((num_reset, num_action), dtype=get_global_dtype())


def build_common_reset_randomization(env: Any, num_reset: int) -> ResetRandomizationPayload | None:
    domain_rand = getattr(env.cfg, "domain_rand", None)
    if domain_rand is None:
        return None

    payload = ResetRandomizationPayload()
    if getattr(domain_rand, "randomize_base_mass", False):
        low, high = domain_rand.added_mass_range
        payload.base_mass_delta = np.random.uniform(low, high, size=(num_reset,))

    if getattr(domain_rand, "random_com", False):
        low, high = domain_rand.com_offset_x
        base_com_offset = np.zeros((num_reset, 3), dtype=np.float64)
        base_com_offset[:, 0] = np.random.uniform(low, high, size=(num_reset,))
        payload.base_com_offset = base_com_offset

    num_actuators = getattr(env, "_num_action", None)
    if num_actuators is not None and getattr(domain_rand, "randomize_kp", False):
        low, high = domain_rand.kp_multiplier_range
        kp_multiplier = np.random.uniform(low, high, size=(num_reset, 1))
        base_kp = float(env.cfg.control_config.Kp)
        payload.kp = np.broadcast_to(base_kp * kp_multiplier, (num_reset, num_actuators)).copy()

    if num_actuators is not None and getattr(domain_rand, "randomize_kd", False):
        low, high = domain_rand.kd_multiplier_range
        kd_multiplier = np.random.uniform(low, high, size=(num_reset, 1))
        base_kd = float(env.cfg.control_config.Kd)
        payload.kd = np.broadcast_to(base_kd * kd_multiplier, (num_reset, num_actuators)).copy()

    return None if payload.is_empty() else payload


def validate_common_reset_randomization(
    env: Any, capabilities: DomainRandomizationCapabilities
) -> frozenset[str]:
    payload = build_common_reset_randomization(env, num_reset=1)
    if payload is None:
        return frozenset()
    return capabilities.get_unsupported_reset_terms(payload.requested_terms())


def build_interval_push_plan(env: Any, step_counter: int) -> IntervalRandomizationPlan | None:
    domain_rand = getattr(env.cfg, "domain_rand", None)
    if domain_rand is None or not getattr(domain_rand, "push_robots", False):
        return None
    if step_counter % domain_rand.push_interval != 0:
        return None
    return IntervalRandomizationPlan(
        push_perturbation_limit=np.asarray(domain_rand.max_force, dtype=np.float64)
    )


def validate_interval_push_support(env: Any, capabilities: DomainRandomizationCapabilities) -> None:
    domain_rand = getattr(env.cfg, "domain_rand", None)
    if domain_rand is None or not getattr(domain_rand, "push_robots", False):
        return
    if not capabilities.supports_interval_push:
        raise NotImplementedError(
            f"{env._backend.backend_type} backend does not support interval push"
        )
