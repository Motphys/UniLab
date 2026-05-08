"""Motion tracking environments."""

__unilab_registry_modules__ = ("unilab.envs.motion_tracking.g1",)

from .g1 import (
    G1FlipTrackingCfg,
    G1FlipTrackingEnv,
    G1FlipTrackingEnvCfg,
    G1MotionTrackingCfg,
    G1MotionTrackingEnv,
    G1MotionTrackingEnvCfg,
    G1MotionTrackingSACCfg,
    G1MotionTrackingSACEnv,
    G1WallFlipTrackingCfg,
    G1WallFlipTrackingEnv,
    G1WallFlipTrackingEnvCfg,
)

__all__ = [
    "G1MotionTrackingCfg",
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
]
