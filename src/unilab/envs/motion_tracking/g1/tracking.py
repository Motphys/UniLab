"""G1 Motion Tracking Environment - Motion imitation task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import (
    DomainRandomizationCapabilities,
    DomainRandomizationProvider,
    IntervalRandomizationPlan,
    ResetPlan,
)
from unilab.dr.types import RESET_TERM_GEOM_FRICTION, ResetRandomizationPayload
from unilab.dr.dr_utils import (
    build_common_reset_randomization,
    build_interval_push_plan,
    validate_common_reset_randomization,
    validate_interval_push_support,
    zero_actions,
)
from unilab.dr.types import RESET_TERM_GEOM_FRICTION, ResetRandomizationPayload
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.math import np_sample_uniform
from unilab.envs.common.rotation import (
    np_quat_apply,
    np_quat_from_euler_xyz,
    np_quat_inv,
    np_quat_mul,
)
from unilab.envs.locomotion.g1.base import G1BaseCfg, G1BaseEnv

from .motion_loader import MotionData, MotionLoader, MotionSampler


@dataclass
class RewardConfig:
    """Reward configuration for motion tracking."""

    scales: dict[str, float] = field(
        default_factory=lambda: {
            "motion_global_root_pos": 0.5,
            "motion_global_root_ori": 0.5,
            "motion_body_pos": 1.0,
            "motion_body_ori": 1.0,
            "motion_body_lin_vel": 1.0,
            "motion_body_ang_vel": 1.0,
            "motion_ee_body_pos_z": 0.0,
            "motion_joint_pos": 0.0,
            "motion_joint_vel": 0.0,
            "action_rate_l2": -0.1,
            "joint_limit": -10.0,
        }
    )
    # Standard deviations for exponential rewards
    std_root_pos: float = 0.3
    std_root_ori: float = 0.4
    std_body_pos: float = 0.3
    std_body_ori: float = 0.4
    std_body_lin_vel: float = 1.0
    std_body_ang_vel: float = 3.14
    std_joint_pos: float = 0.2
    std_joint_vel: float = 1.0


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
class Domain_Rand:
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


@dataclass
class G1MotionTrackingCfg(G1BaseCfg):
    """Configuration for G1 motion tracking environment."""

    scene: SceneCfg = field(
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
    max_episode_seconds: float = 10.0
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    pose_randomization: PoseRandomization = field(default_factory=PoseRandomization)
    velocity_randomization: VelocityRandomization = field(default_factory=VelocityRandomization)
    domain_rand: Domain_Rand = field(default_factory=Domain_Rand)
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


@registry.envcfg("G1MotionTracking")
@dataclass
class G1MotionTrackingEnvCfg(G1MotionTrackingCfg):
    """Registered configuration for G1 motion tracking."""

    pass


@registry.envcfg("G1MotionTrackingDeploy")
@dataclass
class G1MotionTrackingDeployEnvCfg(G1MotionTrackingCfg):
    """Registered deploy configuration for G1 motion tracking."""

    pass


def _build_motion_reference_state(
    env: Any, env_ids: np.ndarray, motion_data: MotionData
) -> tuple[np.ndarray, np.ndarray]:
    dtype = get_global_dtype()
    num_reset = len(env_ids)

    root_pos = motion_data.body_pos_w[:, 0].copy()
    root_ori = motion_data.body_quat_w[:, 0].copy()
    root_lin_vel = motion_data.body_lin_vel_w[:, 0].copy()
    root_ang_vel = motion_data.body_ang_vel_w[:, 0].copy()
    joint_pos = motion_data.joint_pos.copy()
    joint_vel = motion_data.joint_vel.copy()

    pose_rand = env.cfg.pose_randomization
    pose_ranges = [
        (pose_rand.x[0], pose_rand.x[1]),
        (pose_rand.y[0], pose_rand.y[1]),
        (pose_rand.z[0], pose_rand.z[1]),
        (pose_rand.roll[0], pose_rand.roll[1]),
        (pose_rand.pitch[0], pose_rand.pitch[1]),
        (pose_rand.yaw[0], pose_rand.yaw[1]),
    ]
    pose_samples = np.array(
        [[np.random.uniform(low, high) for low, high in pose_ranges] for _ in range(num_reset)],
        dtype=dtype,
    )
    root_pos += pose_samples[:, 0:3]
    root_ori = np_quat_mul(
        np_quat_from_euler_xyz(pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]),
        root_ori,
    )

    vel_rand = env.cfg.velocity_randomization
    vel_ranges = [
        (vel_rand.x[0], vel_rand.x[1]),
        (vel_rand.y[0], vel_rand.y[1]),
        (vel_rand.z[0], vel_rand.z[1]),
        (vel_rand.roll[0], vel_rand.roll[1]),
        (vel_rand.pitch[0], vel_rand.pitch[1]),
        (vel_rand.yaw[0], vel_rand.yaw[1]),
    ]
    vel_samples = np.array(
        [[np.random.uniform(low, high) for low, high in vel_ranges] for _ in range(num_reset)],
        dtype=dtype,
    )
    root_lin_vel += vel_samples[:, :3]
    root_ang_vel += vel_samples[:, 3:]

    joint_pos += np_sample_uniform(
        env.cfg.joint_position_range[0],
        env.cfg.joint_position_range[1],
        joint_pos.shape,
        dtype=np.float32,
    )
    joint_range = env._get_joint_range()
    if joint_range is not None:
        joint_pos = np.clip(joint_pos, joint_range[:, 0], joint_range[:, 1])

    qpos = np.tile(env._init_qpos, (num_reset, 1))
    qvel = np.tile(env._init_qvel, (num_reset, 1))
    qpos[:, 0:3] = root_pos
    qpos[:, 3:7] = root_ori
    qpos[:, 7:] = joint_pos

    qvel[:, 0:3] = root_lin_vel
    qvel[:, 3:6] = np_quat_apply(np_quat_inv(root_ori), root_ang_vel)
    qvel[:, 6:] = joint_vel
    return qpos, qvel


def _gravity_z_in_body_from_quat_w(quat_w: np.ndarray) -> np.ndarray:
    """Z component of world gravity ``[0, 0, -1]`` expressed in body frame."""
    return 2.0 * (quat_w[:, 1] * quat_w[:, 1] + quat_w[:, 2] * quat_w[:, 2]) - 1.0


def _write_motion_anchor_transform(
    robot_anchor_pos_w: np.ndarray,
    robot_anchor_quat_w: np.ndarray,
    anchor_pos_w: np.ndarray,
    anchor_quat_w: np.ndarray,
    out_pos: np.ndarray,
    out_ori6: np.ndarray,
) -> None:
    aw = robot_anchor_quat_w[:, 0]
    ax = robot_anchor_quat_w[:, 1]
    ay = robot_anchor_quat_w[:, 2]
    az = robot_anchor_quat_w[:, 3]

    vx = anchor_pos_w[:, 0] - robot_anchor_pos_w[:, 0]
    vy = anchor_pos_w[:, 1] - robot_anchor_pos_w[:, 1]
    vz = anchor_pos_w[:, 2] - robot_anchor_pos_w[:, 2]

    qx = -ax
    qy = -ay
    qz = -az
    tx = 2 * (qy * vz - qz * vy)
    ty = 2 * (qz * vx - qx * vz)
    tz = 2 * (qx * vy - qy * vx)
    out_pos[:, 0] = vx + aw * tx + qy * tz - qz * ty
    out_pos[:, 1] = vy + aw * ty + qz * tx - qx * tz
    out_pos[:, 2] = vz + aw * tz + qx * ty - qy * tx

    bw = anchor_quat_w[:, 0]
    bx = anchor_quat_w[:, 1]
    by = anchor_quat_w[:, 2]
    bz = anchor_quat_w[:, 3]
    rw = aw * bw + ax * bx + ay * by + az * bz
    rx = aw * bx - ax * bw - ay * bz + az * by
    ry = aw * by + ax * bz - ay * bw - az * bx
    rz = aw * bz - ax * by + ay * bx - az * bw

    xx = rx * rx
    yy = ry * ry
    zz = rz * rz
    xy = rx * ry
    xz = rx * rz
    yz = ry * rz
    wx = rw * rx
    wy = rw * ry
    wz = rw * rz
    out_ori6[:, 0] = 1 - 2 * (yy + zz)
    out_ori6[:, 1] = 2 * (xy - wz)
    out_ori6[:, 2] = 2 * (xy + wz)
    out_ori6[:, 3] = 1 - 2 * (xx + zz)
    out_ori6[:, 4] = 2 * (xz - wy)
    out_ori6[:, 5] = 2 * (yz + wx)


class G1MotionTrackingDomainRandomizationProvider(DomainRandomizationProvider):
    def __init__(
        self,
        *,
        base_kp: np.ndarray | None = None,
        base_kd: np.ndarray | None = None,
        base_geom_friction: np.ndarray | None = None,
        foot_geom_ids: np.ndarray | None = None,
    ) -> None:
        self._base_kp = base_kp
        self._base_kd = base_kd
        self._base_geom_friction = base_geom_friction
        self._foot_geom_ids = foot_geom_ids

    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        validate_common_reset_randomization(
            env, capabilities, base_kp=self._base_kp, base_kd=self._base_kd
        )
        validate_interval_push_support(env, capabilities)
        if getattr(env.cfg.domain_rand, "randomize_geom_friction", False):
            if not capabilities.supports_reset_term(RESET_TERM_GEOM_FRICTION):
                raise NotImplementedError(
                    f"{env._backend.backend_type} backend does not support "
                    "geom-friction reset randomization"
                )
            if (
                self._base_geom_friction is None
                or self._foot_geom_ids is None
                or self._foot_geom_ids.size == 0
            ):
                raise ValueError("randomize_geom_friction=True but provider has no foot geom IDs")

    def build_interval_randomization_plan(
        self, env: Any, step_counter: int
    ) -> IntervalRandomizationPlan | None:
        return build_interval_push_plan(env, step_counter)

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        motion_frames = env.motion_sampler.sample_frames(env_ids)
        motion_data = env.motion_loader.get_motion_at_frame(motion_frames)
        qpos, qvel = _build_motion_reference_state(env, env_ids, motion_data)

        info_updates = {
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
        }
        randomization = build_common_reset_randomization(
            env, num_reset, base_kp=self._base_kp, base_kd=self._base_kd
        )

        dr_cfg = env.cfg.domain_rand
        if getattr(dr_cfg, "randomize_geom_friction", False):
            assert self._base_geom_friction is not None
            assert self._foot_geom_ids is not None
            payload = randomization or ResetRandomizationPayload()
            low, high = dr_cfg.friction_range
            scale = np.random.uniform(low, high, size=(num_reset, 1)).astype(np.float64)
            geom_friction = np.broadcast_to(
                self._base_geom_friction,
                (num_reset, *self._base_geom_friction.shape),
            ).copy()
            geom_friction[:, self._foot_geom_ids, 0] = scale * np.ones(
                (1, self._foot_geom_ids.size)
            )
            payload.geom_friction = geom_friction
            randomization = payload

        if getattr(dr_cfg, "randomize_joint_default_pos", False):
            low, high = dr_cfg.joint_default_pos_range
            info_updates["default_dof_pos_bias"] = np.random.uniform(
                low, high, size=(num_reset, env._num_action)
            ).astype(get_global_dtype())

        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=randomization,
        )

    def build_reset_observation(
        self, env: Any, env_ids: np.ndarray, info_updates: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        motion_data = env.motion_loader.get_motion_at_frame(
            env.motion_sampler.current_frames[env_ids]
        )
        linvel = env.get_local_linvel()[env_ids]
        gyro = env.get_gyro()[env_ids]
        dof_pos = env.get_dof_pos()[env_ids]
        dof_vel = env.get_dof_vel()[env_ids]
        all_pos_w, all_quat_w = env._get_body_pose_w()
        obs_info = dict(info_updates)
        default_dof_pos_bias = info_updates.get("default_dof_pos_bias")
        if isinstance(default_dof_pos_bias, np.ndarray):
            obs_info["default_dof_pos_bias"] = default_dof_pos_bias
        obs_info["env_ids"] = env_ids
        return cast(
            dict[str, np.ndarray],
            env._compute_obs(
                obs_info,
                motion_data,
                linvel,
                gyro,
                dof_pos,
                dof_vel,
                all_pos_w[env_ids],
                all_quat_w[env_ids],
            ),
        )


@registry.env("G1MotionTracking", sim_backend="mujoco")
@registry.env("G1MotionTracking", sim_backend="motrix")
class G1MotionTrackingEnv(G1BaseEnv):
    """G1 Motion Tracking Environment."""

    _cfg: G1MotionTrackingCfg

    def __init__(self, cfg: G1MotionTrackingCfg, num_envs=1, backend_type="mujoco"):
        if not cfg.motion_file:
            raise ValueError("motion_file must be specified in config")

        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            add_body_sensors=True,
            motrix_max_iterations=cfg.motrix_max_iterations,
        )
        super().__init__(cfg, backend, num_envs)

        # Resolve body IDs for backend querying and motion-file indexing.
        self.body_ids = self._backend.get_body_ids(cfg.body_names)
        motion_body_ids = self._backend.get_motion_body_ids(cfg.body_names)

        self.anchor_body_idx = cfg.body_names.index(cfg.anchor_body_name)

        # Get end-effector body indices for termination
        self.ee_body_indices = np.array(
            [cfg.body_names.index(name) for name in cfg.ee_body_names], dtype=np.int32
        )
        self._has_ee_body_indices = bool(self.ee_body_indices.size)

        # Get non-EE body indices for undesired contact penalty
        ee_set = set(cfg.ee_body_names)
        self.undesired_contact_body_indices = np.array(
            [i for i, name in enumerate(cfg.body_names) if name not in ee_set],
            dtype=np.int32,
        )
        self._has_undesired_contact_body_indices = bool(self.undesired_contact_body_indices.size)

        # Load motion data
        self.motion_loader = MotionLoader(cfg.motion_file, body_indices=motion_body_ids)
        self.motion_sampler = MotionSampler(
            self.motion_loader,
            mode=cfg.sampling_mode,
            num_envs=num_envs,
            start_ratio=cfg.sampling_start_ratio,
        )
        needs_kp_kd = cfg.domain_rand.randomize_kp or cfg.domain_rand.randomize_kd
        needs_friction = getattr(cfg.domain_rand, "randomize_geom_friction", False)
        base_kp = base_kd = None
        if needs_kp_kd:
            base_kp, base_kd = backend.get_actuator_gains()
        base_geom_friction = None
        foot_geom_ids = None
        if needs_friction:
            import re as _re

            base_geom_friction = backend.get_geom_friction()
            geom_names = backend.get_geom_names()
            pattern = _re.compile(cfg.domain_rand.friction_geom_pattern)
            foot_geom_ids = np.asarray(
                [i for i, name in enumerate(geom_names) if name and pattern.match(name)],
                dtype=np.int64,
            )
            if foot_geom_ids.size == 0:
                raise ValueError(
                    "friction_geom_pattern "
                    f"'{cfg.domain_rand.friction_geom_pattern}' did not match any geom"
                )
        dr_provider = G1MotionTrackingDomainRandomizationProvider(
            base_kp=base_kp,
            base_kd=base_kd,
            base_geom_friction=base_geom_friction,
            foot_geom_ids=foot_geom_ids,
        )
        self._init_domain_randomization(dr_provider)

        dtype = get_global_dtype()
        n_body = len(cfg.body_names)
        self._n_motion_bodies = n_body
        self._actor_obs_width = self._actor_obs_dim(self._num_action)
        self._critic_base_obs_width = self._critic_base_obs_dim(self._num_action)
        self._critic_obs_width = self._critic_base_obs_width + n_body * 9
        self._copy_body_state_w = self._backend.copy_body_state_w

        # Buffers for relative body transforms
        self.body_pos_relative_w = np.zeros((num_envs, n_body, 3), dtype=dtype)
        self.body_quat_relative_w = np.zeros((num_envs, n_body, 4), dtype=dtype)
        self.body_quat_relative_w[:, :, 0] = 1.0  # Initialize to identity quaternion
        self._motion_data_buffer = (
            self.motion_loader.make_motion_data_buffer(num_envs)
            if hasattr(self.motion_loader, "make_motion_data_buffer")
            else None
        )
        self._zero_actions = np.zeros((num_envs, self._num_action), dtype=dtype)
        self._joint_range = self._backend.get_joint_range()
        if self._joint_range is not None:
            self._joint_range = np.asarray(self._joint_range, dtype=dtype)
            self._joint_lower = self._joint_range[:, 0]
            self._joint_upper = self._joint_range[:, 1]
        else:
            self._joint_lower = None
            self._joint_upper = None
        self._delta_pos_w = np.empty((num_envs, 3), dtype=dtype)
        self._delta_ori_w = np.empty((num_envs, 4), dtype=dtype)
        self._motion_anchor_pos_b = np.empty((num_envs, 3), dtype=dtype)
        self._motion_anchor_ori_b = np.empty((num_envs, 6), dtype=dtype)
        self._motion_command = np.empty((num_envs, self._num_action * 2), dtype=dtype)
        self._joint_pos_rel = np.empty((num_envs, self._num_action), dtype=dtype)
        self._robot_body_pos_w = np.empty((num_envs, n_body, 3), dtype=dtype)
        self._robot_body_quat_w = np.empty((num_envs, n_body, 4), dtype=dtype)
        self._robot_body_lin_vel_w = np.empty((num_envs, n_body, 3), dtype=dtype)
        self._robot_body_ang_vel_w = np.empty((num_envs, n_body, 3), dtype=dtype)
        self._quat_error_w = np.empty((num_envs, n_body), dtype=dtype)
        self._quat_error_x = np.empty((num_envs, n_body), dtype=dtype)
        self._body_vec_error = np.empty((num_envs, n_body, 3), dtype=dtype)
        self._body_vec_tmp = np.empty((num_envs, n_body, 3), dtype=dtype)
        self._joint_error = np.empty((num_envs, self._num_action), dtype=dtype)
        self._joint_error_upper = np.empty((num_envs, self._num_action), dtype=dtype)
        self._env_error = np.empty((num_envs,), dtype=dtype)
        self._env_error2 = np.empty((num_envs,), dtype=dtype)
        self._reward_term = np.empty((num_envs,), dtype=dtype)
        self._weighted_reward = np.empty((num_envs,), dtype=dtype)
        self._terminated = np.empty((num_envs,), dtype=bool)
        self._env_bool = np.empty((num_envs,), dtype=bool)
        self._ee_pos_error_z = np.empty((num_envs, self.ee_body_indices.size), dtype=dtype)
        self._ee_terminated = np.empty((num_envs, self.ee_body_indices.size), dtype=bool)
        self._undesired_contact_mask = np.empty(
            (num_envs, self.undesired_contact_body_indices.size), dtype=bool
        )

        self._enable_reward_log = True
        self._init_reward_functions()
        self._active_reward_fns = {
            name: reward_fn
            for name, reward_fn in self._reward_fns.items()
            if self._reward_term_is_active(name)
        }
        self._clip_end_truncated = np.zeros((num_envs,), dtype=bool)

    def _effective_default_angles(self, env_ids: np.ndarray | None = None) -> np.ndarray:
        """Return default_angles with per-episode joint-default-pos bias applied."""
        state = getattr(self, "_state", None)
        if state is not None:
            bias = state.info.get("default_dof_pos_bias")
            if bias is not None:
                if env_ids is not None:
                    return self.default_angles + bias[env_ids]
                return self.default_angles + bias
        return self.default_angles

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions
        exec_actions = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else actions
        )
        bias = state.info.get("default_dof_pos_bias")
        base = self.default_angles + bias if bias is not None else self.default_angles
        ctrl: np.ndarray = exec_actions * self._cfg.control_config.action_scale + base
        return ctrl

    def _resample_reference_state(self, env_ids: np.ndarray) -> None:
        motion_frames = self.motion_sampler.sample_frames(env_ids)
        motion_data = self.motion_loader.get_motion_at_frame(motion_frames)
        qpos, qvel = _build_motion_reference_state(self, env_ids, motion_data)
        self._backend.set_state(env_ids, qpos, qvel)

    def _refresh_observation_rows(
        self, obs: dict[str, np.ndarray], info: dict, env_ids: np.ndarray
    ) -> None:
        motion_data = self.motion_loader.get_motion_at_frame(
            self.motion_sampler.current_frames[env_ids]
        )
        row_ids = np.asarray(env_ids, dtype=np.intp)
        linvel = self._backend.get_sensor_data_rows(self._cfg.sensor.local_linvel, row_ids)
        gyro = self._backend.get_sensor_data_rows(self._cfg.sensor.gyro, row_ids)
        dof_pos = self.get_dof_pos()[row_ids]
        dof_vel = self.get_dof_vel()[row_ids]
        robot_body_pos_w, robot_body_quat_w = self._backend.get_body_pose_w_rows(
            row_ids, self.body_ids
        )

        obs_info: dict[str, Any] = {}
        current_actions = info.get("current_actions")
        if isinstance(current_actions, np.ndarray):
            obs_info["current_actions"] = current_actions[env_ids]
        obs_info["env_ids"] = env_ids

        refreshed_obs = self._compute_obs(
            obs_info,
            motion_data,
            linvel,
            gyro,
            dof_pos,
            dof_vel,
            robot_body_pos_w,
            robot_body_quat_w,
        )
        for key, value in refreshed_obs.items():
            if value.shape[0] == len(env_ids):
                obs[key][env_ids] = value
            else:
                obs[key][env_ids] = value[env_ids]

    def _get_body_pose_w(self) -> tuple[np.ndarray, np.ndarray]:
        return self._backend.get_body_pose_w(self.body_ids)

    def _get_body_state_w(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        copy_body_state_w = self._copy_body_state_w
        if copy_body_state_w is not None:
            return copy_body_state_w(
                self.body_ids,
                self._robot_body_pos_w,
                self._robot_body_quat_w,
                self._robot_body_lin_vel_w,
                self._robot_body_ang_vel_w,
            )
        robot_body_pos_w, robot_body_quat_w = self._get_body_pose_w()
        robot_body_lin_vel_w, robot_body_ang_vel_w = self._backend.get_body_vel_w(self.body_ids)
        return (
            robot_body_pos_w,
            robot_body_quat_w,
            robot_body_lin_vel_w,
            robot_body_ang_vel_w,
        )

    def _get_joint_range(self) -> np.ndarray | None:
        return self._joint_range

    def _get_current_motion(self) -> MotionData:
        if self._motion_data_buffer is None:
            return self.motion_sampler.get_current_motion()
        return self.motion_sampler.get_current_motion(self._motion_data_buffer)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # Actor: command(2n) + motion_anchor_pos_b(3) + motion_anchor_ori_b(6)
        #        + linvel(3) + gyro(3) + joint_pos(n) + joint_vel(n) + actions(n)
        # Critic mirrors BeyondMimic physical terms without actor observation noise:
        #        command, motion anchor, robot body pos/ori, linvel, gyro, joints, actions.
        n = self._num_action
        actor_width = getattr(self, "_actor_obs_width", self._actor_obs_dim(n))
        critic_width = getattr(
            self,
            "_critic_obs_width",
            self._critic_base_obs_dim(n) + len(self._cfg.body_names) * 9,
        )
        return {"obs": actor_width, "critic": critic_width}

    def _actor_obs_dim(self, n: int) -> int:
        return 3 + 6 + 3 + 3 + n * 5

    def _critic_base_obs_dim(self, n: int) -> int:
        return 3 + 6 + 3 + 3 + n * 5

    def _build_actor_obs(
        self,
        *,
        command: np.ndarray,
        motion_anchor_pos_b: np.ndarray,
        motion_anchor_ori_b: np.ndarray,
        noisy_linvel: np.ndarray,
        noisy_gyro: np.ndarray,
        noisy_joint_pos_rel: np.ndarray,
        noisy_dof_vel: np.ndarray,
        last_actions: np.ndarray,
    ) -> np.ndarray:
        num_envs = command.shape[0]
        n_action = noisy_joint_pos_rel.shape[1]
        actor_obs = np.empty((num_envs, self._actor_obs_dim(n_action)), dtype=get_global_dtype())
        offset = 0
        actor_obs[:, offset : offset + command.shape[1]] = command
        offset += command.shape[1]
        actor_obs[:, offset : offset + 3] = motion_anchor_pos_b
        offset += 3
        actor_obs[:, offset : offset + 6] = motion_anchor_ori_b
        offset += 6
        actor_obs[:, offset : offset + 3] = noisy_linvel
        offset += 3
        actor_obs[:, offset : offset + 3] = noisy_gyro
        offset += 3
        actor_obs[:, offset : offset + n_action] = noisy_joint_pos_rel
        offset += n_action
        actor_obs[:, offset : offset + n_action] = noisy_dof_vel
        offset += n_action
        actor_obs[:, offset : offset + n_action] = last_actions
        return actor_obs

    def _init_reward_functions(self):
        self._reward_fns = {
            "motion_global_root_pos": self._reward_motion_global_root_pos,
            "motion_global_root_ori": self._reward_motion_global_root_ori,
            "motion_body_pos": self._reward_motion_body_pos,
            "motion_body_ori": self._reward_motion_body_ori,
            "motion_body_lin_vel": self._reward_motion_body_lin_vel,
            "motion_body_ang_vel": self._reward_motion_body_ang_vel,
            "motion_ee_body_pos_z": self._reward_motion_ee_body_pos_z,
            "motion_joint_pos": self._reward_motion_joint_pos,
            "motion_joint_vel": self._reward_motion_joint_vel,
            "action_rate_l2": self._reward_action_rate_l2,
            "joint_limit": self._reward_joint_limit,
            "undesired_contacts": self._reward_undesired_contacts,
        }

    def _reward_term_is_active(self, name: str) -> bool:
        if name == "joint_limit":
            return self._joint_lower is not None and self._joint_upper is not None
        if name == "undesired_contacts":
            return self._has_undesired_contact_body_indices
        if name == "motion_ee_body_pos_z":
            return self._has_ee_body_indices
        return True

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._clip_end_truncated.fill(False)

        # Get current motion data
        motion_data = self._get_current_motion()

        # Get robot state
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()

        # Get body states
        (
            robot_body_pos_w,
            robot_body_quat_w,
            robot_body_lin_vel_w,
            robot_body_ang_vel_w,
        ) = self._get_body_state_w()

        # Compute relative body transforms (for observations and rewards)
        self._update_relative_transforms(motion_data, robot_body_pos_w, robot_body_quat_w)

        # Compute terminations
        terminated = self._compute_terminations(motion_data, robot_body_pos_w, robot_body_quat_w)

        # Update failure statistics for adaptive sampling
        self.motion_sampler.update_failure_stats(terminated)

        # Compute reward
        reward = self._compute_reward(
            state.info,
            motion_data,
            robot_body_pos_w,
            robot_body_quat_w,
            robot_body_lin_vel_w,
            robot_body_ang_vel_w,
            dof_pos,
            dof_vel,
        )

        # Compute observations
        obs = self._compute_obs(
            state.info,
            motion_data,
            linvel,
            gyro,
            dof_pos,
            dof_vel,
            robot_body_pos_w,
            robot_body_quat_w,
        )

        # Advance motion frames
        done_env_ids = self.motion_sampler.step()
        if len(done_env_ids) > 0:
            if self._cfg.truncate_on_clip_end:
                self._clip_end_truncated[done_env_ids] = True
            else:
                # Match BeyondMimic: clip boundaries are command resampling points, not
                # episode boundaries; sync the simulated robot to the new reference.
                resample_env_ids = done_env_ids[~terminated[done_env_ids]]
                if len(resample_env_ids) > 0:
                    self._resample_reference_state(resample_env_ids)
                    self._refresh_observation_rows(obs, state.info, resample_env_ids)

        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_truncated(self, state: NpEnvState) -> np.ndarray:
        truncated = super()._compute_truncated(state)
        clip_end_only = getattr(self, "_env_bool", None)
        if clip_end_only is None or clip_end_only.shape != (self._num_envs,):
            clip_end_only = np.empty((self._num_envs,), dtype=bool)
            self._env_bool = clip_end_only
        np.logical_not(state.terminated, out=clip_end_only)
        np.logical_and(self._clip_end_truncated, clip_end_only, out=clip_end_only)
        np.logical_or(truncated, clip_end_only, out=truncated)
        return truncated

    def _update_relative_transforms(
        self, motion_data, robot_body_pos_w: np.ndarray, robot_body_quat_w: np.ndarray
    ):
        """Update relative body transforms for tracking."""
        # Get anchor states
        anchor_pos_w = motion_data.body_pos_w[:, self.anchor_body_idx]
        anchor_quat_w = motion_data.body_quat_w[:, self.anchor_body_idx]
        robot_anchor_pos_w = robot_body_pos_w[:, self.anchor_body_idx]
        robot_anchor_quat_w = robot_body_quat_w[:, self.anchor_body_idx]

        # Compute delta transform: keep robot's XY position, use motion's Z height
        # and apply yaw-only rotation difference.
        delta_pos_w = self._delta_pos_w
        delta_pos_w[:] = robot_anchor_pos_w
        delta_pos_w[:, 2] = anchor_pos_w[:, 2]

        # Compute yaw-only rotation difference, equivalent to
        # np_yaw_quat(np_quat_mul(robot_anchor_quat_w, np_quat_inv(anchor_quat_w))).
        delta_ori_w = self._delta_ori_w
        rw, rx, ry, rz = (
            robot_anchor_quat_w[:, 0],
            robot_anchor_quat_w[:, 1],
            robot_anchor_quat_w[:, 2],
            robot_anchor_quat_w[:, 3],
        )
        aw, ax, ay, az = (
            anchor_quat_w[:, 0],
            anchor_quat_w[:, 1],
            anchor_quat_w[:, 2],
            anchor_quat_w[:, 3],
        )
        qw = rw * aw + rx * ax + ry * ay + rz * az
        qx = -rw * ax + rx * aw - ry * az + rz * ay
        qy = -rw * ay + rx * az + ry * aw - rz * ax
        qz = -rw * az - rx * ay + ry * ax + rz * aw
        half_yaw = 0.5 * np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        np.cos(half_yaw, out=delta_ori_w[:, 0])
        delta_ori_w[:, 1:3] = 0.0
        np.sin(half_yaw, out=delta_ori_w[:, 3])

        dw1 = delta_ori_w[:, 0]
        dz1 = delta_ori_w[:, 3]
        dw = dw1[:, None]
        dz = dz1[:, None]
        mw = motion_data.body_quat_w[..., 0]
        mx = motion_data.body_quat_w[..., 1]
        my = motion_data.body_quat_w[..., 2]
        mz = motion_data.body_quat_w[..., 3]
        out_quat = self.body_quat_relative_w
        out_quat[..., 0] = dw * mw
        out_quat[..., 0] -= dz * mz
        out_quat[..., 1] = dw * mx
        out_quat[..., 1] -= dz * my
        out_quat[..., 2] = dw * my
        out_quat[..., 2] += dz * mx
        out_quat[..., 3] = dw * mz
        out_quat[..., 3] += dz * mw

        rel_pos = self._body_vec_error
        vx = rel_pos[..., 0]
        vy = rel_pos[..., 1]
        vz = rel_pos[..., 2]
        np.subtract(motion_data.body_pos_w[..., 0], anchor_pos_w[:, None, 0], out=vx)
        np.subtract(motion_data.body_pos_w[..., 1], anchor_pos_w[:, None, 1], out=vy)
        np.subtract(motion_data.body_pos_w[..., 2], anchor_pos_w[:, None, 2], out=vz)

        yaw_cross = self._env_error
        yaw_z2 = self._reward_term
        np.multiply(dw1, dz1, out=yaw_cross)
        yaw_cross *= 2.0
        np.square(dz1, out=yaw_z2)
        yaw_z2 *= 2.0
        yaw_cross_2d = yaw_cross[:, None]
        yaw_z2_2d = yaw_z2[:, None]

        out_pos = self.body_pos_relative_w
        out_pos[..., 0] = vx
        out_pos[..., 0] -= yaw_cross_2d * vy
        out_pos[..., 0] -= yaw_z2_2d * vx
        out_pos[..., 0] += delta_pos_w[:, None, 0]
        out_pos[..., 1] = vy
        out_pos[..., 1] += yaw_cross_2d * vx
        out_pos[..., 1] -= yaw_z2_2d * vy
        out_pos[..., 1] += delta_pos_w[:, None, 1]
        out_pos[..., 2] = vz
        out_pos[..., 2] += delta_pos_w[:, None, 2]

    def _compute_terminations(
        self,
        motion_data,
        robot_body_pos_w: np.ndarray,
        robot_body_quat_w: np.ndarray,
    ) -> np.ndarray:
        """Compute termination conditions."""
        terminated = self._terminated
        terminated.fill(False)

        # Anchor position error (Z-axis only)
        anchor_pos_w = motion_data.body_pos_w[:, self.anchor_body_idx]
        robot_anchor_pos_w = robot_body_pos_w[:, self.anchor_body_idx]
        np.subtract(anchor_pos_w[:, 2], robot_anchor_pos_w[:, 2], out=self._env_error)
        np.abs(self._env_error, out=self._env_error)
        np.greater(self._env_error, self._cfg.anchor_pos_z_threshold, out=self._env_bool)
        terminated |= self._env_bool

        # Anchor orientation error (gravity direction). The gravity-z difference
        # is bounded by 2 for unit quaternions, so huge thresholds disable this
        # termination without doing the per-step math.
        if self._cfg.anchor_ori_threshold < 2.0:
            anchor_quat_w = motion_data.body_quat_w[:, self.anchor_body_idx]
            robot_anchor_quat_w = robot_body_quat_w[:, self.anchor_body_idx]
            motion_gravity_z_b = _gravity_z_in_body_from_quat_w(anchor_quat_w)
            robot_gravity_z_b = _gravity_z_in_body_from_quat_w(robot_anchor_quat_w)
            np.subtract(motion_gravity_z_b, robot_gravity_z_b, out=self._env_error)
            np.abs(self._env_error, out=self._env_error)
            np.greater(self._env_error, self._cfg.anchor_ori_threshold, out=self._env_bool)
            terminated |= self._env_bool

        # End-effector position error (Z-axis only)
        if self._has_ee_body_indices:
            np.subtract(
                self.body_pos_relative_w[:, self.ee_body_indices, 2],
                robot_body_pos_w[:, self.ee_body_indices, 2],
                out=self._ee_pos_error_z,
            )
            np.abs(self._ee_pos_error_z, out=self._ee_pos_error_z)
            np.greater(
                self._ee_pos_error_z,
                self._cfg.ee_body_pos_z_threshold,
                out=self._ee_terminated,
            )
            np.logical_or.reduce(self._ee_terminated, axis=1, out=self._env_bool)
            terminated |= self._env_bool

        if self._cfg.terminate_on_undesired_contacts and self._has_undesired_contact_body_indices:
            body_z = robot_body_pos_w[:, self.undesired_contact_body_indices, 2]
            np.less(
                body_z,
                self._cfg.undesired_contact_z_threshold,
                out=self._undesired_contact_mask,
            )
            np.logical_or.reduce(self._undesired_contact_mask, axis=-1, out=self._env_bool)
            terminated |= self._env_bool

        return terminated

    def _write_body_pos_in_anchor_frame(
        self,
        anchor_pos: np.ndarray,
        anchor_quat: np.ndarray,
        body_pos: np.ndarray,
        out: np.ndarray,
    ) -> None:
        aw = anchor_quat[:, None, 0]
        ax = anchor_quat[:, None, 1]
        ay = anchor_quat[:, None, 2]
        az = anchor_quat[:, None, 3]

        num_envs, n_body = body_pos.shape[:2]
        rel_pos = self._body_vec_error[:num_envs, :n_body]

        vx = rel_pos[..., 0]
        vy = rel_pos[..., 1]
        vz = rel_pos[..., 2]
        np.subtract(body_pos[..., 0], anchor_pos[:, None, 0], out=vx)
        np.subtract(body_pos[..., 1], anchor_pos[:, None, 1], out=vy)
        np.subtract(body_pos[..., 2], anchor_pos[:, None, 2], out=vz)

        tx = 2 * (az * vy - ay * vz)
        ty = 2 * (ax * vz - az * vx)
        tz = 2 * (ay * vx - ax * vy)

        out[..., 0] = vx + aw * tx + az * ty - ay * tz
        out[..., 1] = vy + aw * ty + ax * tz - az * tx
        out[..., 2] = vz + aw * tz + ay * tx - ax * ty

    def _write_body_ori6_in_anchor_frame(
        self,
        anchor_quat: np.ndarray,
        body_quat: np.ndarray,
        out: np.ndarray,
    ) -> None:
        aw = anchor_quat[:, None, 0]
        ax = anchor_quat[:, None, 1]
        ay = anchor_quat[:, None, 2]
        az = anchor_quat[:, None, 3]
        bw = body_quat[..., 0]
        bx = body_quat[..., 1]
        by = body_quat[..., 2]
        bz = body_quat[..., 3]

        rw = aw * bw + ax * bx + ay * by + az * bz
        rx = aw * bx - ax * bw - ay * bz + az * by
        ry = aw * by + ax * bz - ay * bw - az * bx
        rz = aw * bz - ax * by + ay * bx - az * bw

        out[..., 0] = 1 - 2 * (ry * ry + rz * rz)
        out[..., 1] = 2 * (rx * ry - rw * rz)
        out[..., 2] = 2 * (rx * ry + rw * rz)
        out[..., 3] = 1 - 2 * (rx * rx + rz * rz)
        out[..., 4] = 2 * (rx * rz - rw * ry)
        out[..., 5] = 2 * (ry * rz + rw * rx)

    def _compute_obs(
        self,
        info: dict,
        motion_data,
        linvel: np.ndarray,
        gyro: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        robot_body_pos_w: np.ndarray,
        robot_body_quat_w: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compute observations as dict with actor and critic groups."""
        num_envs = linvel.shape[0]
        dtype = get_global_dtype()
        n_action = dof_pos.shape[1]
        n_body = self._n_motion_bodies

        # Get anchor states
        anchor_pos_w = motion_data.body_pos_w[:, self.anchor_body_idx]
        anchor_quat_w = motion_data.body_quat_w[:, self.anchor_body_idx]
        robot_anchor_pos_w = robot_body_pos_w[:, self.anchor_body_idx]
        robot_anchor_quat_w = robot_body_quat_w[:, self.anchor_body_idx]

        # Motion anchor pose in robot frame
        if num_envs == self._num_envs:
            motion_anchor_pos_b = self._motion_anchor_pos_b
            motion_anchor_ori_b = self._motion_anchor_ori_b
            joint_pos_rel = self._joint_pos_rel
            zero_actions = self._zero_actions
        else:
            motion_anchor_pos_b = np.empty((num_envs, 3), dtype=dtype)
            motion_anchor_ori_b = np.empty((num_envs, 6), dtype=dtype)
            joint_pos_rel = np.empty((num_envs, n_action), dtype=dtype)
            zero_actions = np.zeros((num_envs, n_action), dtype=dtype)
        _write_motion_anchor_transform(
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            anchor_pos_w,
            anchor_quat_w,
            motion_anchor_pos_b,
            motion_anchor_ori_b,
        )

        # Joint positions and velocities
        bias = info.get("default_dof_pos_bias")
        effective_default = self.default_angles + bias if bias is not None else self.default_angles
        np.subtract(dof_pos, effective_default, out=joint_pos_rel)
        last_actions = info.get("current_actions")
        if not isinstance(last_actions, np.ndarray):
            last_actions = zero_actions

        if num_envs == self._num_envs:
            command = self._motion_command
        else:
            command = np.empty((num_envs, n_action * 2), dtype=dtype)
        command[:, :n_action] = motion_data.joint_pos
        command[:, n_action : n_action * 2] = motion_data.joint_vel

        # Per-step observation noise on sensor channels (actor only).
        # Critic uses the clean originals — asymmetric actor–critic contract.
        noise_cfg = self._cfg.noise_config
        noise_enabled = noise_cfg.level > 0.0
        if noise_enabled:
            linvel_actor = self._obs_noise(linvel, noise_cfg.scale_linvel)
            gyro_actor = self._obs_noise(gyro, noise_cfg.scale_gyro)
            joint_pos_actor = self._obs_noise(joint_pos_rel, noise_cfg.scale_joint_angle)
            dof_vel_actor = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)
        else:
            linvel_actor = linvel
            gyro_actor = gyro
            joint_pos_actor = joint_pos_rel
            dof_vel_actor = dof_vel

        # Actor observations (noisy proprioception)
        actor_obs = self._build_actor_obs(
            command=command,
            motion_anchor_pos_b=motion_anchor_pos_b,
            motion_anchor_ori_b=motion_anchor_ori_b,
            noisy_linvel=linvel_actor,
            noisy_gyro=gyro_actor,
            noisy_joint_pos_rel=joint_pos_actor,
            noisy_dof_vel=dof_vel_actor,
            last_actions=last_actions,
        )

        # Critic observations (clean proprioception + privileged body transforms)
        critic_obs = np.empty((num_envs, self._critic_obs_width), dtype=dtype)
        offset = 0
        critic_obs[:, offset : offset + command.shape[1]] = command
        offset += command.shape[1]
        critic_obs[:, offset : offset + 3] = motion_anchor_pos_b
        offset += 3
        critic_obs[:, offset : offset + 6] = motion_anchor_ori_b
        offset += 6
        critic_obs[:, offset : offset + 3] = linvel
        offset += 3
        critic_obs[:, offset : offset + 3] = gyro
        offset += 3
        critic_obs[:, offset : offset + n_action] = joint_pos_rel
        offset += n_action
        critic_obs[:, offset : offset + n_action] = dof_vel
        offset += n_action
        critic_obs[:, offset : offset + n_action] = last_actions
        offset += n_action
        robot_body_pos_b = critic_obs[:, offset : offset + n_body * 3].reshape(num_envs, n_body, 3)
        self._write_body_pos_in_anchor_frame(
            robot_anchor_pos_w, robot_anchor_quat_w, robot_body_pos_w, robot_body_pos_b
        )
        offset += n_body * 3
        robot_body_ori_b = critic_obs[:, offset : offset + n_body * 6].reshape(num_envs, n_body, 6)
        self._write_body_ori6_in_anchor_frame(
            robot_anchor_quat_w, robot_body_quat_w, robot_body_ori_b
        )
        return {"obs": actor_obs, "critic": critic_obs}

    def _compute_reward(
        self,
        info: dict,
        motion_data,
        robot_body_pos_w: np.ndarray,
        robot_body_quat_w: np.ndarray,
        robot_body_lin_vel_w: np.ndarray,
        robot_body_ang_vel_w: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> np.ndarray:
        """Compute reward."""
        reward = self._env_error2
        reward.fill(0.0)
        cfg = self._cfg.reward_config

        step_count = info.get("steps")
        should_log = self._enable_reward_log and (
            int(step_count[0]) % 4 == 0 if isinstance(step_count, np.ndarray) else True
        )
        log = {} if should_log else info.get("log", {})

        # Store motion and robot states in info for reward functions
        info["motion_data"] = motion_data
        info["robot_body_pos_w"] = robot_body_pos_w
        info["robot_body_quat_w"] = robot_body_quat_w
        info["robot_body_lin_vel_w"] = robot_body_lin_vel_w
        info["robot_body_ang_vel_w"] = robot_body_ang_vel_w
        info["reward_ref_body_pos_w"] = self.body_pos_relative_w
        info["reward_ref_body_quat_w"] = self.body_quat_relative_w
        info["anchor_body_idx"] = self.anchor_body_idx
        info["dof_pos"] = dof_pos
        info["dof_vel"] = dof_vel

        for name, scale in cfg.scales.items():
            if scale == 0:
                continue
            reward_fn = self._active_reward_fns.get(name)
            if reward_fn is None:
                if should_log and name in self._reward_fns:
                    log[f"reward/{name}"] = 0.0
                continue
            rew = reward_fn(info)
            weighted_rew = self._weighted_reward
            np.multiply(rew, scale, out=weighted_rew)
            reward += weighted_rew
            if should_log:
                log[f"reward/{name}"] = float(np.sum(weighted_rew) / weighted_rew.size)

        info["log"] = log
        reward *= self._cfg.ctrl_dt
        return reward

    def _mean_body_xyz_squared_error(self, reference: np.ndarray, actual: np.ndarray) -> np.ndarray:
        vec_error = self._body_vec_error
        env_error = self._env_error
        tmp_error = self._reward_term
        np.subtract(reference[..., 0], actual[..., 0], out=vec_error[..., 0])
        np.square(vec_error[..., 0], out=vec_error[..., 0])
        np.sum(vec_error[..., 0], axis=1, out=env_error)
        np.subtract(reference[..., 1], actual[..., 1], out=vec_error[..., 1])
        np.square(vec_error[..., 1], out=vec_error[..., 1])
        np.sum(vec_error[..., 1], axis=1, out=tmp_error)
        env_error += tmp_error
        np.subtract(reference[..., 2], actual[..., 2], out=vec_error[..., 2])
        np.square(vec_error[..., 2], out=vec_error[..., 2])
        np.sum(vec_error[..., 2], axis=1, out=tmp_error)
        env_error += tmp_error
        env_error /= reference.shape[1]
        return env_error

    def _quat_error_magnitude_squared_body(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        rel_w = self._quat_error_w
        rel_x = self._quat_error_x
        # Motion/backend quaternions are unit quaternions, so the relative
        # rotation angle only needs abs(dot(q1, q2)).
        np.multiply(q1[..., 0], q2[..., 0], out=rel_w)
        np.multiply(q1[..., 1], q2[..., 1], out=rel_x)
        rel_w += rel_x
        np.multiply(q1[..., 2], q2[..., 2], out=rel_x)
        rel_w += rel_x
        np.multiply(q1[..., 3], q2[..., 3], out=rel_x)
        rel_w += rel_x
        np.abs(rel_w, out=rel_w)
        np.clip(rel_w, 0.0, 1.0, out=rel_w)
        np.arccos(rel_w, out=rel_x)
        rel_x *= 2.0
        np.square(rel_x, out=rel_x)
        return rel_x

    def _exp_reward_from_error(self, error: np.ndarray, std: float) -> np.ndarray:
        out = self._reward_term
        np.divide(error, -(std**2), out=out)
        np.exp(out, out=out)
        return out

    # Reward functions
    def _reward_motion_global_root_pos(self, info: dict) -> np.ndarray:
        motion_data = info["motion_data"]
        robot_body_pos_w = info["robot_body_pos_w"]
        anchor_pos_w = motion_data.body_pos_w[:, self.anchor_body_idx]
        robot_anchor_pos_w = robot_body_pos_w[:, self.anchor_body_idx]
        error = self._env_error
        np.subtract(anchor_pos_w[:, 0], robot_anchor_pos_w[:, 0], out=error)
        np.square(error, out=error)
        np.subtract(anchor_pos_w[:, 1], robot_anchor_pos_w[:, 1], out=self._reward_term)
        np.square(self._reward_term, out=self._reward_term)
        error += self._reward_term
        np.subtract(anchor_pos_w[:, 2], robot_anchor_pos_w[:, 2], out=self._reward_term)
        np.square(self._reward_term, out=self._reward_term)
        error += self._reward_term
        return self._exp_reward_from_error(error, self._cfg.reward_config.std_root_pos)

    def _reward_motion_global_root_ori(self, info: dict) -> np.ndarray:
        motion_data = info["motion_data"]
        robot_body_quat_w = info["robot_body_quat_w"]
        anchor_quat_w = motion_data.body_quat_w[:, self.anchor_body_idx]
        robot_anchor_quat_w = robot_body_quat_w[:, self.anchor_body_idx]
        np.multiply(anchor_quat_w[:, 0], robot_anchor_quat_w[:, 0], out=self._env_error)
        np.multiply(anchor_quat_w[:, 1], robot_anchor_quat_w[:, 1], out=self._reward_term)
        self._env_error += self._reward_term
        np.multiply(anchor_quat_w[:, 2], robot_anchor_quat_w[:, 2], out=self._reward_term)
        self._env_error += self._reward_term
        np.multiply(anchor_quat_w[:, 3], robot_anchor_quat_w[:, 3], out=self._reward_term)
        self._env_error += self._reward_term
        np.abs(self._env_error, out=self._env_error)
        np.clip(self._env_error, 0.0, 1.0, out=self._env_error)
        np.arccos(self._env_error, out=self._env_error)
        self._env_error *= 2.0
        np.square(self._env_error, out=self._env_error)
        return self._exp_reward_from_error(self._env_error, self._cfg.reward_config.std_root_ori)

    def _reward_motion_body_pos(self, info: dict) -> np.ndarray:
        robot_body_pos_w = info["robot_body_pos_w"]
        error = self._mean_body_xyz_squared_error(self.body_pos_relative_w, robot_body_pos_w)
        return self._exp_reward_from_error(error, self._cfg.reward_config.std_body_pos)

    def _reward_motion_body_ori(self, info: dict) -> np.ndarray:
        robot_body_quat_w = info["robot_body_quat_w"]
        error = self._quat_error_magnitude_squared_body(
            self.body_quat_relative_w, robot_body_quat_w
        )
        np.sum(error, axis=-1, out=self._env_error)
        self._env_error /= error.shape[1]
        return self._exp_reward_from_error(self._env_error, self._cfg.reward_config.std_body_ori)

    def _reward_motion_body_lin_vel(self, info: dict) -> np.ndarray:
        motion_data = info["motion_data"]
        robot_body_lin_vel_w = info["robot_body_lin_vel_w"]
        error = self._mean_body_xyz_squared_error(motion_data.body_lin_vel_w, robot_body_lin_vel_w)
        return self._exp_reward_from_error(error, self._cfg.reward_config.std_body_lin_vel)

    def _reward_motion_body_ang_vel(self, info: dict) -> np.ndarray:
        motion_data = info["motion_data"]
        robot_body_ang_vel_w = info["robot_body_ang_vel_w"]
        error = self._mean_body_xyz_squared_error(motion_data.body_ang_vel_w, robot_body_ang_vel_w)
        return self._exp_reward_from_error(error, self._cfg.reward_config.std_body_ang_vel)

    def _reward_motion_ee_body_pos_z(self, info: dict) -> np.ndarray:
        robot_body_pos_w = info["robot_body_pos_w"]
        np.subtract(
            self.body_pos_relative_w[:, self.ee_body_indices, 2],
            robot_body_pos_w[:, self.ee_body_indices, 2],
            out=self._ee_pos_error_z,
        )
        np.square(self._ee_pos_error_z, out=self._ee_pos_error_z)
        np.sum(self._ee_pos_error_z, axis=-1, out=self._env_error)
        self._env_error /= self._ee_pos_error_z.shape[1]
        return self._exp_reward_from_error(self._env_error, self._cfg.reward_config.std_body_pos)

    def _reward_motion_joint_pos(self, info: dict) -> np.ndarray:
        motion_data = info["motion_data"]
        dof_pos = info["dof_pos"]
        np.subtract(motion_data.joint_pos, dof_pos, out=self._joint_error)
        np.square(self._joint_error, out=self._joint_error)
        np.sum(self._joint_error, axis=1, out=self._env_error)
        self._env_error /= dof_pos.shape[1]
        return self._exp_reward_from_error(self._env_error, self._cfg.reward_config.std_joint_pos)

    def _reward_motion_joint_vel(self, info: dict) -> np.ndarray:
        motion_data = info["motion_data"]
        dof_vel = info["dof_vel"]
        np.subtract(motion_data.joint_vel, dof_vel, out=self._joint_error)
        np.square(self._joint_error, out=self._joint_error)
        np.sum(self._joint_error, axis=1, out=self._env_error)
        self._env_error /= dof_vel.shape[1]
        return self._exp_reward_from_error(self._env_error, self._cfg.reward_config.std_joint_vel)

    def _reward_undesired_contacts(self, info: dict) -> np.ndarray:
        robot_body_pos_w = info["robot_body_pos_w"]
        body_z = robot_body_pos_w[:, self.undesired_contact_body_indices, 2]
        np.less(
            body_z,
            self._cfg.undesired_contact_z_threshold,
            out=self._undesired_contact_mask,
        )
        np.sum(self._undesired_contact_mask, axis=-1, out=self._env_error)
        return self._env_error

    def _reward_action_rate_l2(self, info: dict) -> np.ndarray:
        np.subtract(info["current_actions"], info["last_actions"], out=self._joint_error)
        np.square(self._joint_error, out=self._joint_error)
        np.sum(self._joint_error, axis=1, out=self._env_error)
        return self._env_error

    def _reward_joint_limit(self, info: dict) -> np.ndarray:
        dof_pos = info["dof_pos"]
        lower = self._joint_lower
        upper = self._joint_upper
        if lower is None or upper is None:
            self._reward_term.fill(0.0)
            return self._reward_term

        # Compute violation
        np.subtract(lower, dof_pos, out=self._joint_error)
        np.maximum(self._joint_error, 0, out=self._joint_error)
        np.subtract(dof_pos, upper, out=self._joint_error_upper)
        np.maximum(self._joint_error_upper, 0, out=self._joint_error_upper)
        self._joint_error += self._joint_error_upper
        np.square(self._joint_error, out=self._joint_error)
        np.sum(self._joint_error, axis=1, out=self._reward_term)
        return self._reward_term


@registry.env("G1MotionTrackingDeploy", sim_backend="mujoco")
@registry.env("G1MotionTrackingDeploy", sim_backend="motrix")
class G1MotionTrackingDeployEnv(G1MotionTrackingEnv):
    """Deploy-oriented G1 motion tracking env with unitree_rl_lab mimic actor inputs."""

    _cfg: G1MotionTrackingDeployEnvCfg

    def _actor_obs_dim(self, n: int) -> int:
        # unitree_rl_lab mimic deploy actor input:
        # motion_command(2n), motion_anchor_ori_b(6), gyro(3), joints, actions.
        return 6 + 3 + n * 5

    def _build_actor_obs(
        self,
        *,
        command: np.ndarray,
        motion_anchor_pos_b: np.ndarray,
        motion_anchor_ori_b: np.ndarray,
        noisy_linvel: np.ndarray,
        noisy_gyro: np.ndarray,
        noisy_joint_pos_rel: np.ndarray,
        noisy_dof_vel: np.ndarray,
        last_actions: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate(
            [
                command,
                motion_anchor_ori_b,
                noisy_gyro,
                noisy_joint_pos_rel,
                noisy_dof_vel,
                last_actions,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
