"""Shared DomainRandomizationProvider for locomotion environments.

Implements the common reset/interval randomization logic shared by
G1, Go1, and Go2 joystick environments.  Subclasses override hooks
to provide robot-specific behaviour.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from unilab.base.dtype_config import get_global_dtype
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
from unilab.utils.math_utils import np_quat_mul, np_yaw_to_quat


class LocomotionDRProvider(DomainRandomizationProvider):
    """Base DR provider for locomotion joystick environments.

    Shared logic:
    - ``validate``, ``build_interval_randomization_plan``, ``_sample_commands``
    - ``build_reset_plan`` (template with hooks)
    - ``build_reset_observation`` (template with hook)

    Override these hooks in subclasses:
    - ``_get_qvel_limit`` — default ``0.5``
    - ``_build_extra_info_updates`` — default empty dict
    - ``_compute_reset_obs`` — must be implemented per robot
    """

    # ── shared methods ───────────────────────────────────────────────

    def _get_base_actuator_gains(self, env: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return (base_kp, base_kd) for per-joint kp/kd domain randomization.

        Override to provide per-joint gains cached at init time.
        Returns ``(None, None)`` by default, which falls back to scalar
        ``ControlConfig.Kp`` / ``ControlConfig.Kd``.
        """
        return None, None

    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        base_kp, base_kd = self._get_base_actuator_gains(env)
        validate_common_reset_randomization(env, capabilities, base_kp=base_kp, base_kd=base_kd)
        validate_interval_push_support(env, capabilities)

    def build_interval_randomization_plan(
        self, env: Any, step_counter: int
    ) -> IntervalRandomizationPlan | None:
        return build_interval_push_plan(env, step_counter)

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        low = np.asarray(env.cfg.commands.vel_limit[0], dtype=get_global_dtype())
        high = np.asarray(env.cfg.commands.vel_limit[1], dtype=get_global_dtype())
        return np.asarray(
            np.random.uniform(low=low, high=high, size=(num_reset, 3)), dtype=get_global_dtype()
        )

    # ── template: build_reset_plan ───────────────────────────────────

    def _get_qvel_limit(self, env: Any) -> float:
        """Return the base qvel reset limit.  Override for configurable limits."""
        return 0.5

    def _build_extra_info_updates(self, env: Any, num_reset: int) -> dict[str, np.ndarray]:
        """Return additional info_updates entries (e.g. gait_phase for G1)."""
        return {}

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))
        qpos[:, 0:2] += np.random.uniform(-0.5, 0.5, (num_reset, 2))
        yaw = np.random.uniform(-np.pi, np.pi, (num_reset,))
        qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], np_yaw_to_quat(yaw))
        limit = self._get_qvel_limit(env)
        qvel[:, 0:6] = np.asarray(
            np.random.uniform(-limit, limit, size=(num_reset, 6)), dtype=get_global_dtype()
        )
        info_updates: dict[str, Any] = {
            "commands": self._sample_commands(env, num_reset),
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
        }
        info_updates.update(self._build_extra_info_updates(env, num_reset))
        base_kp, base_kd = self._get_base_actuator_gains(env)
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=build_common_reset_randomization(
                env, num_reset, base_kp=base_kp, base_kd=base_kd
            ),
        )

    # ── template: build_reset_observation ────────────────────────────

    def _compute_reset_obs(
        self,
        env: Any,
        env_ids: np.ndarray,
        info_updates: dict[str, Any],
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compute reset observations.  Override per robot to pass correct args to _compute_obs."""
        raise NotImplementedError

    def build_reset_observation(
        self, env: Any, env_ids: np.ndarray, info_updates: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        linvel = env.get_local_linvel()[env_ids]
        gyro = env.get_gyro()[env_ids]
        gravity = env._backend.get_sensor_data("upvector")[env_ids]
        dof_pos = env.get_dof_pos()[env_ids]
        dof_vel = env.get_dof_vel()[env_ids]
        return cast(
            dict[str, np.ndarray],
            self._compute_reset_obs(
                env, env_ids, info_updates, linvel, gyro, gravity, dof_pos, dof_vel
            ),
        )
