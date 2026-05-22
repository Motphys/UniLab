"""G1 Motion Tracking SAC Environment — thin SAC wrapper over G1MotionTrackingEnv.

Differences from the PPO base:
- Critic observations additionally include ``base_lin_vel`` (3 dims),
  matching holosoma's asymmetric actor-critic design for WBT.
- Registered under a separate name so it can be paired with FastSAC
  configs without affecting the PPO motion-tracking pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unilab.base import registry
from unilab.dtype_config import get_global_dtype

from .tracking import G1MotionTrackingCfg, G1MotionTrackingEnv


@registry.envcfg("G1MotionTrackingSAC")
@dataclass
class G1MotionTrackingSACCfg(G1MotionTrackingCfg):
    """Config for SAC-based motion tracking (identical fields, separate registry entry)."""


@registry.env("G1MotionTrackingSAC", sim_backend="mujoco")
@registry.env("G1MotionTrackingSAC", sim_backend="motrix")
class G1MotionTrackingSACEnv(G1MotionTrackingEnv):
    """G1 Motion Tracking environment for FastSAC training.

    Extends the PPO motion-tracking environment with ``base_lin_vel``
    appended to the critic observation, matching holosoma's asymmetric
    actor-critic WBT design.

    The motrix backend is registered for sim2sim eval/playback only — checkpoints
    trained on mujoco can be replayed via motrix's native renderer through
    ``eval --sim motrix``.
    """

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        spec = super().obs_groups_spec
        # Append base_lin_vel (3) to critic observations.
        return {**spec, "critic": spec["critic"] + 3}

    def _compute_obs(
        self,
        info: dict,
        motion_data,
        linvel: np.ndarray,
        gyro: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        robot_body_pos_w: np.ndarray,
        robot_body_quat_w: np.ndarray,
    ) -> dict[str, np.ndarray]:
        obs = super()._compute_obs(  # pyright: ignore[reportAttributeAccessIssue]
            info,
            motion_data,
            linvel,
            gyro,
            dof_pos,
            dof_vel,
            robot_body_pos_w,
            robot_body_quat_w,
        )
        # Append base_lin_vel to critic observations.
        obs["critic"] = np.concatenate([obs["critic"], linvel], axis=1, dtype=get_global_dtype())  # type: ignore[call-overload]
        return obs
