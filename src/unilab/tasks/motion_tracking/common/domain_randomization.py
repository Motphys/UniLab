"""Domain-randomization reset/observation provider for motion tracking."""

from __future__ import annotations

import time
from typing import Any, cast

import numpy as np

from unilab.dr import (
    DomainRandomizationCapabilities,
    DomainRandomizationProvider,
    IntervalRandomizationPlan,
    ResetPlan,
)
from unilab.dr.dr_utils import (
    build_common_reset_randomization,
    build_interval_push_plan,
    validate_common_reset_randomization,
    validate_interval_push_support,
    zero_actions,
)
from unilab.dr.types import RESET_TERM_GEOM_FRICTION, ResetRandomizationPayload
from unilab.dtype_config import get_global_dtype
from unilab.envs.motion_tracking.common.reset import build_motion_reference_state


class MotionTrackingDomainRandomizationProvider(DomainRandomizationProvider):
    def __init__(
        self,
        *,
        base_kp: np.ndarray | None = None,
        base_kd: np.ndarray | None = None,
        base_geom_friction: np.ndarray | None = None,
        foot_geom_ids: np.ndarray | None = None,
    ) -> None:
        self._base_kp = base_kp
        self._base_kd = base_kd
        self._base_geom_friction = base_geom_friction
        self._foot_geom_ids = foot_geom_ids
        self._last_reset_observation_timing_ms: dict[str, float] = {}

    @property
    def last_reset_observation_timing_ms(self) -> dict[str, float]:
        return dict(self._last_reset_observation_timing_ms)

    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        validate_common_reset_randomization(
            env, capabilities, base_kp=self._base_kp, base_kd=self._base_kd
        )
        validate_interval_push_support(env, capabilities)
        if getattr(env.cfg.domain_rand, "randomize_geom_friction", False):
            if not capabilities.supports_reset_term(RESET_TERM_GEOM_FRICTION):
                raise NotImplementedError(
                    f"{env._backend.backend_type} backend does not support "
                    "geom-friction reset randomization"
                )
            if (
                self._base_geom_friction is None
                or self._foot_geom_ids is None
                or self._foot_geom_ids.size == 0
            ):
                raise ValueError("randomize_geom_friction=True but provider has no foot geom IDs")

    def build_interval_randomization_plan(
        self, env: Any, step_counter: int
    ) -> IntervalRandomizationPlan | None:
        return build_interval_push_plan(env, step_counter)

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        motion_frames = env.motion_sampler.sample_frames(env_ids)
        motion_data = env.motion_loader.get_motion_at_frame(motion_frames)
        qpos, qvel = build_motion_reference_state(env, env_ids, motion_data)

        info_updates = {
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
        }
        randomization = build_common_reset_randomization(
            env, num_reset, base_kp=self._base_kp, base_kd=self._base_kd
        )

        dr_cfg = env.cfg.domain_rand
        if getattr(dr_cfg, "randomize_geom_friction", False):
            assert self._base_geom_friction is not None
            assert self._foot_geom_ids is not None
            payload = randomization or ResetRandomizationPayload()
            low, high = dr_cfg.friction_range
            scale = np.random.uniform(low, high, size=(num_reset, 1)).astype(np.float64)
            geom_friction = np.broadcast_to(
                self._base_geom_friction,
                (num_reset, *self._base_geom_friction.shape),
            ).copy()
            geom_friction[:, self._foot_geom_ids, 0] = scale * np.ones(
                (1, self._foot_geom_ids.size)
            )
            payload.geom_friction = geom_friction
            randomization = payload

        if getattr(dr_cfg, "randomize_joint_default_pos", False):
            low, high = dr_cfg.joint_default_pos_range
            info_updates["default_dof_pos_bias"] = np.random.uniform(
                low, high, size=(num_reset, env._num_action)
            ).astype(get_global_dtype())

        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=randomization,
        )

    def build_reset_observation(
        self, env: Any, env_ids: np.ndarray, info_updates: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        obs_t0 = time.perf_counter()

        t0 = time.perf_counter()
        motion_data = env.motion_loader.get_motion_at_frame(
            env.motion_sampler.current_frames[env_ids]
        )
        motion_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        linvel = env.get_local_linvel()[env_ids]
        linvel_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        gyro = env.get_gyro()[env_ids]
        gyro_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        dof_pos = env.get_dof_pos()[env_ids]
        dof_pos_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        dof_vel = env.get_dof_vel()[env_ids]
        dof_vel_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        robot_body_pos_w, robot_body_quat_w = env._backend.get_body_pose_w_rows(
            env_ids, env.body_ids
        )
        body_pose_ms = (time.perf_counter() - t0) * 1000.0

        obs_info = dict(info_updates)
        default_dof_pos_bias = info_updates.get("default_dof_pos_bias")
        if isinstance(default_dof_pos_bias, np.ndarray):
            obs_info["default_dof_pos_bias"] = default_dof_pos_bias
        obs_info["env_ids"] = env_ids

        t0 = time.perf_counter()
        obs = cast(
            dict[str, np.ndarray],
            env._compute_obs(
                obs_info,
                motion_data,
                linvel,
                gyro,
                dof_pos,
                dof_vel,
                robot_body_pos_w,
                robot_body_quat_w,
            ),
        )
        compute_obs_ms = (time.perf_counter() - t0) * 1000.0

        total_ms = (time.perf_counter() - obs_t0) * 1000.0
        getters_ms = linvel_ms + gyro_ms + dof_pos_ms + dof_vel_ms + body_pose_ms
        self._last_reset_observation_timing_ms = {
            "dr_reset_observation_getters_ms": getters_ms,
            "dr_reset_obs_get_motion_ms": motion_ms,
            "dr_reset_obs_get_local_linvel_ms": linvel_ms,
            "dr_reset_obs_get_gyro_ms": gyro_ms,
            "dr_reset_obs_get_dof_pos_ms": dof_pos_ms,
            "dr_reset_obs_get_dof_vel_ms": dof_vel_ms,
            "dr_reset_obs_get_body_pose_ms": body_pose_ms,
            "dr_reset_observation_compute_obs_ms": compute_obs_ms,
            "dr_reset_observation_internal_gap_ms": (
                total_ms - motion_ms - getters_ms - compute_obs_ms
            ),
        }
        return obs
