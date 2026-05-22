"""Motion tracking environments for Unitree G1."""

from .box_tracking import G1BoxTrackingCfg, G1BoxTrackingEnv, G1BoxTrackingEnvCfg
from .flip_tracking import (
    G1ClimbTrackingCfg,
    G1ClimbTrackingEnv,
    G1ClimbTrackingEnvCfg,
    G1FlipTrackingCfg,
    G1FlipTrackingEnv,
    G1FlipTrackingEnvCfg,
    G1WallFlipTrackingCfg,
    G1WallFlipTrackingEnv,
    G1WallFlipTrackingEnvCfg,
)
from .motion_box_loader import BoxMotionData, BoxMotionLoader
from .tracking import (
    G1MotionTrackingCfg,
    G1MotionTrackingDeployEnv,
    G1MotionTrackingDeployEnvCfg,
    G1MotionTrackingEnv,
    G1MotionTrackingEnvCfg,
)
from .tracking_sac import G1MotionTrackingSACCfg, G1MotionTrackingSACEnv

__all__ = [
    "G1MotionTrackingCfg",
    "G1MotionTrackingDeployEnv",
    "G1MotionTrackingDeployEnvCfg",
    "G1MotionTrackingEnv",
    "G1MotionTrackingEnvCfg",
    "G1MotionTrackingSACCfg",
    "G1MotionTrackingSACEnv",
    "G1FlipTrackingCfg",
    "G1FlipTrackingEnv",
    "G1FlipTrackingEnvCfg",
    "G1WallFlipTrackingCfg",
    "G1WallFlipTrackingEnv",
    "G1WallFlipTrackingEnvCfg",
    "G1ClimbTrackingCfg",
    "G1ClimbTrackingEnv",
    "G1ClimbTrackingEnvCfg",
    "G1BoxTrackingCfg",
    "G1BoxTrackingEnv",
    "G1BoxTrackingEnvCfg",
    "BoxMotionData",
    "BoxMotionLoader",
]
