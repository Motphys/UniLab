"""Flip-specialized G1 motion tracking environment.

This keeps the generic G1MotionTracking defaults backward-compatible while
providing a dedicated registry task for flip-focused datasets/profiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg

from .tracking import (
    G1MotionTrackingCfg,
    G1MotionTrackingEnv,
    PoseRandomization,
    VelocityRandomization,
)


def _zero_pose_randomization() -> PoseRandomization:
    return PoseRandomization(
        x=(0.0, 0.0),
        y=(0.0, 0.0),
        z=(0.0, 0.0),
        roll=(0.0, 0.0),
        pitch=(0.0, 0.0),
        yaw=(0.0, 0.0),
    )


def _zero_velocity_randomization() -> VelocityRandomization:
    return VelocityRandomization(
        x=(0.0, 0.0),
        y=(0.0, 0.0),
        z=(0.0, 0.0),
        roll=(0.0, 0.0),
        pitch=(0.0, 0.0),
        yaw=(0.0, 0.0),
    )


@dataclass
class G1FlipTrackingCfg(G1MotionTrackingCfg):
    """Config profile for flip tracking clips."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
        )
    )
    motion_file: str | list[str] = str(
        ASSETS_ROOT_PATH / "motions" / "g1" / "flip_360_001__A304.npz"
    )
    pose_randomization: PoseRandomization = field(default_factory=_zero_pose_randomization)
    velocity_randomization: VelocityRandomization = field(
        default_factory=_zero_velocity_randomization
    )
    joint_position_range: tuple[float, float] = (0.0, 0.0)
    truncate_on_clip_end: bool = True
    # Some flip clips include large anchor orientation deviations.
    anchor_ori_threshold: float = 1e9


@registry.envcfg("G1FlipTracking")
@dataclass
class G1FlipTrackingEnvCfg(G1FlipTrackingCfg):
    """Registered configuration for G1 flip tracking."""

    pass


@registry.env("G1FlipTracking", sim_backend="mujoco")
@registry.env("G1FlipTracking", sim_backend="motrix")
class G1FlipTrackingEnv(G1MotionTrackingEnv):
    """G1 flip-tracking environment implementation."""

    _cfg: G1FlipTrackingCfg


@dataclass
class G1WallFlipTrackingCfg(G1FlipTrackingCfg):
    """Config profile for wall-assisted G1 flip tracking clips."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat_with_wall.xml")
        )
    )
    motion_file: str | list[str] = str(
        ASSETS_ROOT_PATH / "motions" / "g1" / "flip_from_wall_104__A304.npz"
    )
    sampling_mode: Literal["start", "clip_start", "uniform", "adaptive"] = "start"
    anchor_pos_z_threshold: float = 0.5
    ee_body_pos_z_threshold: float = 0.5


@registry.envcfg("G1WallFlipTracking")
@dataclass
class G1WallFlipTrackingEnvCfg(G1WallFlipTrackingCfg):
    """Registered configuration for G1 wall flip tracking."""

    pass


@registry.env("G1WallFlipTracking", sim_backend="mujoco")
@registry.env("G1WallFlipTracking", sim_backend="motrix")
class G1WallFlipTrackingEnv(G1MotionTrackingEnv):
    """G1 wall flip-tracking environment implementation."""

    _cfg: G1WallFlipTrackingCfg
