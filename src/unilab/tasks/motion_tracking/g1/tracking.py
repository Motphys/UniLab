"""G1 Motion Tracking profiles — thin registry subclasses over the shared engine.

The robot-agnostic engine and owner modules live in
:mod:`unilab.tasks.motion_tracking.common`. This module keeps the G1 registry
entries (``G1MotionTracking`` / ``G1MotionTrackingDeploy``) and re-exports the
historical ``G1*`` / ``Domain_Rand`` / ``_build_motion_reference_state`` symbol
names so existing subclasses and tests keep importing them from ``.tracking``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.envs.motion_tracking.common.reset import build_motion_reference_state
from unilab.envs.motion_tracking.common.rewards import RewardConfig

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
from ..common.domain_randomization import (
    MotionTrackingDomainRandomizationProvider,
)
from ..common.tracking import (
    MotionTrackingDeployEnv,
    MotionTrackingEnv,
)

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


@dataclass
class G1MotionTracking23DofCfg(G1MotionTrackingCfg):
    motion_file: str | list[str] = str(
        ASSETS_ROOT_PATH / "motions" / "g1" / "dance1_subject2_part_23dof.npz"
    )
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat_23dof.xml")
        )
    )
    body_names: tuple[str, ...] = (
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_roll_rubber_hand",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_roll_rubber_hand",
    )
    ee_body_names: tuple[str, ...] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_roll_rubber_hand",
        "right_wrist_roll_rubber_hand",
    )


@registry.envcfg("G1MotionTracking23Dof")
@dataclass
class G1MotionTracking23DofEnvCfg(G1MotionTracking23DofCfg):
    pass


@registry.envcfg("G1MotionTracking23DofDeploy")
@dataclass
class G1MotionTracking23DofDeployEnvCfg(G1MotionTracking23DofCfg):
    pass


@registry.env("G1MotionTracking", sim_backend="drake")
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


# --- 23-DoF env registrations (same env classes, 23-DoF configs) ---
registry.register_env("G1MotionTracking23Dof", G1MotionTrackingEnv, sim_backend="mujoco")
registry.register_env("G1MotionTracking23Dof", G1MotionTrackingEnv, sim_backend="motrix")
registry.register_env(
    "G1MotionTracking23DofDeploy", G1MotionTrackingDeployEnv, sim_backend="mujoco"
)
registry.register_env(
    "G1MotionTracking23DofDeploy", G1MotionTrackingDeployEnv, sim_backend="motrix"
)


__all__ = [
    "DomainRand",
    "Domain_Rand",
    "G1MotionTracking23DofCfg",
    "G1MotionTracking23DofDeployEnvCfg",
    "G1MotionTracking23DofEnvCfg",
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
