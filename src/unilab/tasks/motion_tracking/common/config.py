"""Configuration dataclasses for the robot-agnostic motion-tracking engine.

The default field values are kept at the historical G1 profile so that the
registered ``G1MotionTracking`` / ``G1MotionTrackingDeploy`` configs remain
field-identical to the pre-refactor monolith. Per-robot profiles (g1/x2)
override these via subclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.scene import SceneCfg
from unilab.tasks.motion_tracking.common.g1_base import G1BaseCfg

from .rewards import RewardConfig


@dataclass
class PoseRandomization:
    """Pose randomization ranges for reset."""

    x: tuple[float, float] = (-0.05, 0.05)
    y: tuple[float, float] = (-0.05, 0.05)
    z: tuple[float, float] = (-0.01, 0.01)
    roll: tuple[float, float] = (-0.1, 0.1)
    pitch: tuple[float, float] = (-0.1, 0.1)
    yaw: tuple[float, float] = (-0.2, 0.2)


@dataclass
class VelocityRandomization:
    """Velocity randomization ranges for reset."""

    x: tuple[float, float] = (-0.5, 0.5)
    y: tuple[float, float] = (-0.5, 0.5)
    z: tuple[float, float] = (-0.2, 0.2)
    roll: tuple[float, float] = (-0.52, 0.52)
    pitch: tuple[float, float] = (-0.52, 0.52)
    yaw: tuple[float, float] = (-0.78, 0.78)


@dataclass
class DomainRand:
    """Domain randomization config required by motrix backend hooks."""

    randomize_base_mass: bool = False
    added_mass_range: list[float] = field(default_factory=lambda: [-1.5, 1.5])

    random_com: bool = False
    com_offset_x: list[float] = field(default_factory=lambda: [-0.05, 0.05])
    com_offset_y: list[float] = field(default_factory=lambda: [-0.05, 0.05])
    com_offset_z: list[float] = field(default_factory=lambda: [-0.05, 0.05])

    randomize_gravity: bool = False
    gravity_range: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0, -9.81], [0.0, 0.0, -9.81]]
    )

    push_robots: bool = False
    push_interval: int = 750
    max_force: list[float] = field(default_factory=lambda: [1.0, 1.0, 0.5])
    push_body_name: str | None = None

    randomize_kp: bool = False
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])

    randomize_kd: bool = False
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])

    randomize_geom_friction: bool = False
    friction_range: list[float] = field(default_factory=lambda: [0.3, 1.2])
    friction_geom_pattern: str = r"^(left|right)_foot[1-7]_collision$"

    randomize_joint_default_pos: bool = False
    joint_default_pos_range: list[float] = field(default_factory=lambda: [-0.01, 0.01])


# Backward-compatible alias for the historical (snake_case) class name.
Domain_Rand = DomainRand


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
class MotionTrackingCfg(G1BaseCfg):
    """Configuration for the motion tracking environment."""

    scene: SceneCfg = field(  # pyright: ignore[reportIncompatibleVariableOverride]
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
        )
    )
    # Kept at the historical single-clip default for backward compatibility.
    motion_file: str | list[str] = str(
        ASSETS_ROOT_PATH / "motions" / "g1" / "dance1_subject2_part.npz"
    )
    # motion_file: str | list[str] = str(ASSETS_ROOT_PATH / "motions" / "g1" / "gangnam_style.npz")
    # motion_file: str | list[str] = str(ASSETS_ROOT_PATH / "motions" / "g1" / "fight1_subject5_from_csv.npz") #LAFAN
    # motion_file: str | list[str] = str(ASSETS_ROOT_PATH / "motions" / "g1" / "dance_basic_slide_180_R_loop_001__A322_M.npz") #LAFAN
    # motion_file: str | list[str] = str(ASSETS_ROOT_PATH / "motions" / "g1" / "playing_violin_R_003__A327_from_csv.npz") #Seed
    anchor_body_name: str = "torso_link"
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
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    )
    sampling_mode: Literal["start", "clip_start", "uniform", "adaptive", "mixed"] = "adaptive"
    sampling_start_ratio: float = 0.0
    truncate_on_clip_end: bool = False
    max_episode_seconds: float = 10.0  # pyright: ignore[reportIncompatibleVariableOverride]
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    pose_randomization: PoseRandomization = field(default_factory=PoseRandomization)
    velocity_randomization: VelocityRandomization = field(default_factory=VelocityRandomization)
    domain_rand: DomainRand = field(default_factory=DomainRand)
    joint_position_range: tuple[float, float] = (-0.1, 0.1)
    # Termination thresholds
    anchor_pos_z_threshold: float = 0.25
    anchor_ori_threshold: float = 0.8
    ee_body_pos_z_threshold: float = 0.25
    ee_body_names: tuple[str, ...] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    undesired_contact_z_threshold: float = 0.05
    terminate_on_undesired_contacts: bool = False


@dataclass
class MotionTrackingDeployEnvCfg(MotionTrackingCfg):
    """Base deploy configuration for motion tracking."""

    pass
