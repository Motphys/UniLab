"""X2 motion-tracking profiles backed by the shared humanoid tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.g1.base import Sensor
from unilab.envs.motion_tracking.g1.flip_tracking import (
    _zero_pose_randomization,
    _zero_velocity_randomization,
)
from unilab.envs.motion_tracking.g1.tracking import (
    G1MotionTrackingDeployEnv,
    G1MotionTrackingDeployEnvCfg,
    PoseRandomization,
    VelocityRandomization,
)


@dataclass
class X2MotionTrackingCfg(G1MotionTrackingDeployEnvCfg):
    """Base X2 motion-tracking config profile."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "x2" / "scene_flat.xml")
        )
    )
    motion_file: str | list[str] = str(
        ASSETS_ROOT_PATH / "motions" / "x2" / "tictacflip_6-3_g1format.npz"
    )
    sensor: Sensor = field(
        default_factory=lambda: Sensor(
            local_linvel="body-linear-vel",
            gyro="body-angular-velocity",
            upvector="body-orientation",
        )
    )
    anchor_body_name: str = "torso_link"
    body_names: tuple[str, ...] = (
        "pelvis",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
        "waist_yaw_link",
        "waist_pitch_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "left_wrist_pitch_link",
        "left_wrist_roll_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
        "right_wrist_pitch_link",
        "right_wrist_roll_link",
    )
    ee_body_names: tuple[str, ...] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )


@dataclass
class X2WallFlipTrackingCfg(X2MotionTrackingCfg):
    """Config profile for wall-assisted X2 flip tracking."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "x2" / "scene_flat_with_wall.xml"),
            # Render-only twin: wall as a worldbody geom so the offline grid
            # renderer replicates it under every env cell (matches G1's per-env
            # wall view). Physics still uses the <body> wall in model_file.
            visual_model_file=str(
                ASSETS_ROOT_PATH / "robots" / "x2" / "scene_flat_with_wall_visual.xml"
            ),
        )
    )
    pose_randomization: PoseRandomization = field(default_factory=_zero_pose_randomization)
    velocity_randomization: VelocityRandomization = field(
        default_factory=_zero_velocity_randomization
    )
    joint_position_range: tuple[float, float] = (0.0, 0.0)
    sampling_mode: Literal["start", "clip_start", "uniform", "adaptive", "mixed"] = "adaptive"
    truncate_on_clip_end: bool = False
    terminate_on_undesired_contacts: bool = True
    anchor_ori_threshold: float = 1e9
    anchor_pos_z_threshold: float = 0.5
    ee_body_pos_z_threshold: float = 0.5


@registry.envcfg("X2WallFlipTracking")
@dataclass
class X2WallFlipTrackingEnvCfg(X2WallFlipTrackingCfg):
    """Registered configuration for X2 wall flip tracking."""

    pass


@registry.env("X2WallFlipTracking", sim_backend="mujoco")
class X2WallFlipTrackingEnv(G1MotionTrackingDeployEnv):
    """X2 wall flip-tracking environment implementation."""

    _cfg: X2WallFlipTrackingCfg
    _keyframe_name = "home"

    def __init__(self, cfg: X2WallFlipTrackingCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        # X2 meshes are hosted on Hugging Face, not committed to git. Ensure they
        # exist locally before the scene XML is parsed by the simulation backend.
        from unilab.assets.hub import resolve_robot_asset_dir

        resolve_robot_asset_dir("robots/x2/meshes", marker="pelvis.STL")
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
