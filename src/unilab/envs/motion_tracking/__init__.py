"""Motion tracking environments."""

__unilab_registry_modules__ = ("unilab.envs.motion_tracking.g1",)

from .g1 import (
    G1ClimbTrackingCfg,
    G1ClimbTrackingEnv,
    G1ClimbTrackingEnvCfg,
    G1FlipTracking23DofCfg,
    G1FlipTracking23DofEnvCfg,
    G1FlipTrackingCfg,
    G1FlipTrackingEnv,
    G1FlipTrackingEnvCfg,
    G1MotionTracking23DofCfg,
    G1MotionTracking23DofDeployEnvCfg,
    G1MotionTracking23DofEnvCfg,
    G1MotionTrackingCfg,
    G1MotionTrackingEnv,
    G1MotionTrackingEnvCfg,
    G1WallFlipTracking23DofCfg,
    G1WallFlipTracking23DofEnvCfg,
    G1WallFlipTrackingCfg,
    G1WallFlipTrackingEnv,
    G1WallFlipTrackingEnvCfg,
)

__all__ = [
    "G1MotionTrackingCfg",
    "G1MotionTrackingEnv",
    "G1MotionTrackingEnvCfg",
    "G1FlipTrackingCfg",
    "G1FlipTrackingEnv",
    "G1FlipTrackingEnvCfg",
    "G1WallFlipTrackingCfg",
    "G1WallFlipTrackingEnv",
    "G1WallFlipTrackingEnvCfg",
    "G1ClimbTrackingCfg",
    "G1ClimbTrackingEnv",
    "G1ClimbTrackingEnvCfg",
    # 23-DoF variants (from parent UniLab/, under testing)
    "G1MotionTracking23DofCfg",
    "G1MotionTracking23DofDeployEnvCfg",
    "G1MotionTracking23DofEnvCfg",
    "G1FlipTracking23DofCfg",
    "G1FlipTracking23DofEnvCfg",
    "G1WallFlipTracking23DofCfg",
    "G1WallFlipTracking23DofEnvCfg",
]
