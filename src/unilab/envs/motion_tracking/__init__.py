"""Motion tracking environments."""

__unilab_registry_modules__ = ("unilab.envs.motion_tracking.g1",)

from .g1 import (
    G1FlipTrackingCfg,
    G1FlipTrackingEnv,
    G1FlipTrackingEnvCfg,
    G1MotionTrackingCfg,
    G1MotionTrackingEnv,
    G1MotionTrackingEnvCfg,
)

__all__ = [
    "G1MotionTrackingCfg",
    "G1MotionTrackingEnv",
    "G1MotionTrackingEnvCfg",
    "G1FlipTrackingCfg",
    "G1FlipTrackingEnv",
    "G1FlipTrackingEnvCfg",
]
