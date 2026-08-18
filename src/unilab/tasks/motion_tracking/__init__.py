"""Motion-tracking task packages."""

from .g1 import (
    BoxMotionData,
    BoxMotionLoader,
    G1BoxTracking23DofCfg,
    G1BoxTracking23DofEnvCfg,
    G1BoxTrackingCfg,
    G1BoxTrackingEnv,
    G1BoxTrackingEnvCfg,
)
from .x2 import (
    X2MotionTrackingCfg,
    X2WallFlipTrackingCfg,
    X2WallFlipTrackingEnv,
    X2WallFlipTrackingEnvCfg,
)

__all__ = [
    "BoxMotionData",
    "BoxMotionLoader",
    "G1BoxTracking23DofCfg",
    "G1BoxTracking23DofEnvCfg",
    "G1BoxTrackingCfg",
    "G1BoxTrackingEnv",
    "G1BoxTrackingEnvCfg",
    "X2MotionTrackingCfg",
    "X2WallFlipTrackingCfg",
    "X2WallFlipTrackingEnv",
    "X2WallFlipTrackingEnvCfg",
]
