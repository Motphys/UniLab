from __future__ import annotations

from typing import Any

import numpy as np

from unilab.dr.types import (
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    ResetRandomizationPayload,
)
from unilab.dtype_config import get_global_dtype


def zero_actions(num_reset: int, num_action: int) -> np.ndarray:
    return np.zeros((num_reset, num_action), dtype=get_global_dtype())


def _coerce_range(name: str, values: Any) -> tuple[float, float]:
    bounds = np.asarray(values, dtype=np.float64)
    if bounds.shape != (2,):
        raise ValueError(f"domain_rand.{name} must have shape (2,), got {bounds.shape}")
    low = float(bounds[0])
    high = float(bounds[1])
    if high < low:
        raise ValueError(f"domain_rand.{name} high must be >= low")
    return low, high


def build_common_reset_randomization(
    env: Any,
    num_reset: int,
    *,
    base_kp: np.ndarray | None = None,
    base_kd: np.ndarray | None = None,
    base_body_mass: np.ndarray | None = None,
    base_geom_friction: np.ndarray | None = None,
    ground_geom_id: int | None = None,
    base_dof_armature: np.ndarray | None = None,
) -> ResetRandomizationPayload | None:
    domain_rand = getattr(env.cfg, "domain_rand", None)
    if domain_rand is None:
        return None

    payload = ResetRandomizationPayload()
    if getattr(domain_rand, "randomize_base_mass", False):
        low, high = domain_rand.added_mass_range
        payload.base_mass_delta = np.random.uniform(low, high, size=(num_reset,))

    if getattr(domain_rand, "randomize_body_mass", False):
        if base_body_mass is None:
            raise ValueError("body mass randomization requires a cached base body-mass table")
        body_mass_template = np.asarray(base_body_mass, dtype=np.float64)
        if body_mass_template.ndim != 1:
            raise ValueError(
                f"base_body_mass must have shape (nbody,), got {body_mass_template.shape}"
            )
        low, high = _coerce_range(
            "body_mass_multiplier_range", domain_rand.body_mass_multiplier_range
        )
        multipliers = np.random.uniform(
            low=low, high=high, size=(num_reset, body_mass_template.size)
        )
        body_mass = np.broadcast_to(body_mass_template, multipliers.shape).copy()
        randomized = body_mass_template > 0.0
        body_mass[:, randomized] *= multipliers[:, randomized]
        payload.body_mass = body_mass

    if getattr(domain_rand, "random_com", False):
        low, high = domain_rand.com_offset_x
        base_com_offset = np.zeros((num_reset, 3), dtype=np.float64)
        base_com_offset[:, 0] = np.random.uniform(low, high, size=(num_reset,))
        payload.base_com_offset = base_com_offset

    if getattr(domain_rand, "randomize_gravity", False):
        gravity_range = np.asarray(domain_rand.gravity_range, dtype=np.float64)
        if gravity_range.shape != (2, 3):
            raise ValueError(
                f"domain_rand.gravity_range must have shape (2, 3), got {gravity_range.shape}"
            )
        low = np.minimum(gravity_range[0], gravity_range[1])
        high = np.maximum(gravity_range[0], gravity_range[1])
        payload.gravity = np.random.uniform(low=low, high=high, size=(num_reset, 3))

    if getattr(domain_rand, "randomize_ground_friction", False):
        if base_geom_friction is None or ground_geom_id is None:
            raise ValueError(
                "ground friction randomization requires cached geom friction and ground geom id"
            )
        geom_friction_template = np.asarray(base_geom_friction, dtype=np.float64)
        if geom_friction_template.ndim != 2 or geom_friction_template.shape[1] != 3:
            raise ValueError(
                f"base_geom_friction must have shape (ngeom, 3), got {geom_friction_template.shape}"
            )
        ground_id = int(ground_geom_id)
        if ground_id < 0 or ground_id >= geom_friction_template.shape[0]:
            raise ValueError(
                f"ground_geom_id must be in [0, {geom_friction_template.shape[0]}), got {ground_id}"
            )
        low, high = _coerce_range(
            "ground_friction_multiplier_range",
            domain_rand.ground_friction_multiplier_range,
        )
        geom_friction = np.broadcast_to(
            geom_friction_template, (num_reset, *geom_friction_template.shape)
        ).copy()
        geom_friction[:, ground_id, 0] = geom_friction_template[ground_id, 0] * np.random.uniform(
            low=low, high=high, size=(num_reset,)
        )
        payload.geom_friction = geom_friction

    if getattr(domain_rand, "randomize_dof_armature", False):
        if base_dof_armature is None:
            raise ValueError("dof armature randomization requires a cached dof-armature table")
        dof_armature_template = np.asarray(base_dof_armature, dtype=np.float64)
        if dof_armature_template.ndim != 1:
            raise ValueError(
                f"base_dof_armature must have shape (nv,), got {dof_armature_template.shape}"
            )
        low, high = _coerce_range(
            "dof_armature_multiplier_range", domain_rand.dof_armature_multiplier_range
        )
        dof_armature = np.broadcast_to(
            dof_armature_template, (num_reset, dof_armature_template.size)
        ).copy()
        randomized = dof_armature_template > 0.0
        dof_armature[:, randomized] *= np.random.uniform(
            low=low, high=high, size=(num_reset, int(np.count_nonzero(randomized)))
        )
        payload.dof_armature = dof_armature

    num_actuators = getattr(env, "_num_action", None)
    need_kp = num_actuators is not None and getattr(domain_rand, "randomize_kp", False)
    need_kd = num_actuators is not None and getattr(domain_rand, "randomize_kd", False)

    if need_kp or need_kd:
        assert num_actuators is not None

        if need_kp:
            kp = (
                base_kp
                if base_kp is not None
                else np.full(num_actuators, float(env.cfg.control_config.Kp))
            )
            low, high = domain_rand.kp_multiplier_range
            payload.kp = (kp * np.random.uniform(low, high, (num_reset, 1))).astype(np.float64)

        if need_kd:
            kd = (
                base_kd
                if base_kd is not None
                else np.full(num_actuators, float(env.cfg.control_config.Kd))
            )
            low, high = domain_rand.kd_multiplier_range
            payload.kd = (kd * np.random.uniform(low, high, (num_reset, 1))).astype(np.float64)

    return None if payload.is_empty() else payload


def validate_common_reset_randomization(
    env: Any,
    capabilities: DomainRandomizationCapabilities,
    *,
    base_kp: np.ndarray | None = None,
    base_kd: np.ndarray | None = None,
    base_body_mass: np.ndarray | None = None,
    base_geom_friction: np.ndarray | None = None,
    ground_geom_id: int | None = None,
    base_dof_armature: np.ndarray | None = None,
) -> frozenset[str]:
    payload = build_common_reset_randomization(
        env,
        num_reset=1,
        base_kp=base_kp,
        base_kd=base_kd,
        base_body_mass=base_body_mass,
        base_geom_friction=base_geom_friction,
        ground_geom_id=ground_geom_id,
        base_dof_armature=base_dof_armature,
    )
    if payload is None:
        return frozenset()
    return capabilities.get_unsupported_reset_terms(payload.requested_terms())


def build_interval_push_plan(env: Any, step_counter: int) -> IntervalRandomizationPlan | None:
    domain_rand = getattr(env.cfg, "domain_rand", None)
    if domain_rand is None or not getattr(domain_rand, "push_robots", False):
        return None
    if step_counter % domain_rand.push_interval != 0:
        return None
    return IntervalRandomizationPlan(push_perturbation_limit=domain_rand.max_force)


def validate_interval_push_support(env: Any, capabilities: DomainRandomizationCapabilities) -> None:
    domain_rand = getattr(env.cfg, "domain_rand", None)
    if domain_rand is None or not getattr(domain_rand, "push_robots", False):
        return
    if not capabilities.supports_interval_push:
        raise NotImplementedError(
            f"{env._backend.backend_type} backend does not support interval push"
        )
    force_limit = np.asarray(domain_rand.max_force, dtype=np.float64)
    if force_limit.shape != (3,):
        raise ValueError(f"domain_rand.max_force must have shape (3,), got {force_limit.shape}")
