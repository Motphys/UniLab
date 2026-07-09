"""G1 Motion Tracking profiles — thin registry subclasses over the shared engine.

The robot-agnostic engine and owner modules now live in
:mod:`unilab.envs.motion_tracking.common`. This module keeps the G1 registry
entries (``G1MotionTracking`` / ``G1MotionTrackingDeploy``) and re-exports the
historical ``G1*`` / ``Domain_Rand`` / ``_build_motion_reference_state`` symbol
names so existing subclasses and tests keep importing them from ``.tracking``.
"""

from __future__ import annotations

from dataclasses import dataclass

from unilab.base import registry

from ..common.config import (
    Domain_Rand,
    DomainRand,
    MotionTrackingCfg,
    MotionTrackingDeployEnvCfg,
    PoseRandomization,
    VelocityRandomization,
    _zero_pose_randomization,
    _zero_velocity_randomization,
)
from ..common.domain_randomization import MotionTrackingDomainRandomizationProvider
from ..common.reset import build_motion_reference_state
from ..common.rewards import RewardConfig
from ..common.tracking import MotionTrackingDeployEnv, MotionTrackingEnv

# ── backward-compatible aliases (historical G1* symbol names) ────────
G1MotionTrackingCfg = MotionTrackingCfg
G1MotionTrackingDomainRandomizationProvider = MotionTrackingDomainRandomizationProvider
_build_motion_reference_state = build_motion_reference_state


@registry.envcfg("G1MotionTracking")
@dataclass
class G1MotionTrackingEnvCfg(MotionTrackingCfg):
    """Registered configuration for G1 motion tracking."""

    pass


@registry.envcfg("G1MotionTrackingDeploy")
@dataclass
class G1MotionTrackingDeployEnvCfg(MotionTrackingDeployEnvCfg):
    """Registered deploy configuration for G1 motion tracking."""

    pass


@registry.env("G1MotionTracking", sim_backend="mujoco")
@registry.env("G1MotionTracking", sim_backend="motrix")
class G1MotionTrackingEnv(MotionTrackingEnv):
    """G1 Motion Tracking Environment."""

    _cfg: MotionTrackingCfg


@registry.env("G1MotionTrackingDeploy", sim_backend="mujoco")
@registry.env("G1MotionTrackingDeploy", sim_backend="motrix")
class G1MotionTrackingDeployEnv(MotionTrackingDeployEnv):
    """Deploy-oriented G1 motion tracking env with unitree_rl_lab mimic actor inputs."""

    _cfg: MotionTrackingDeployEnvCfg


__all__ = [
    "DomainRand",
    "Domain_Rand",
    "G1MotionTrackingCfg",
    "G1MotionTrackingDeployEnv",
    "G1MotionTrackingDeployEnvCfg",
    "G1MotionTrackingDomainRandomizationProvider",
    "G1MotionTrackingEnv",
    "G1MotionTrackingEnvCfg",
    "MotionTrackingCfg",
    "MotionTrackingDeployEnv",
    "MotionTrackingDeployEnvCfg",
    "MotionTrackingDomainRandomizationProvider",
    "MotionTrackingEnv",
    "PoseRandomization",
    "RewardConfig",
    "VelocityRandomization",
    "_build_motion_reference_state",
    "_zero_pose_randomization",
    "_zero_velocity_randomization",
]
