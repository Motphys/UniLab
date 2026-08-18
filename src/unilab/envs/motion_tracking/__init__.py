"""Motion tracking environments."""

__unilab_registry_modules__ = ("unilab.envs.motion_tracking.g1",)

from .g1 import (
    G1MotionTracking23DofCfg,
    G1MotionTracking23DofDeployEnvCfg,
    G1MotionTracking23DofEnvCfg,
    G1MotionTrackingCfg,
    G1MotionTrackingEnv,
    G1MotionTrackingEnvCfg,
)

__all__ = [
    "G1MotionTrackingCfg",
    "G1MotionTrackingEnv",
    "G1MotionTrackingEnvCfg",
    # 23-DoF variants (from parent UniLab/, under testing)
    "G1MotionTracking23DofCfg",
    "G1MotionTracking23DofDeployEnvCfg",
    "G1MotionTracking23DofEnvCfg",
]
