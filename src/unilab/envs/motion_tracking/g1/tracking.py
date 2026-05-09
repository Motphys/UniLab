"""G1 Motion Tracking Environment - Motion imitation task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.dr import (
    DomainRandomizationCapabilities,
    DomainRandomizationProvider,
    IntervalRandomizationPlan,
    ResetPlan,
)
from unilab.dr.dr_utils import (
    build_common_reset_randomization,
    build_interval_push_plan,
    validate_common_reset_randomization,
    validate_interval_push_support,
    zero_actions,
)
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.math import np_sample_uniform
from unilab.envs.common.rotation import (
    np_matrix_from_quat,
    np_quat_apply,
    np_quat_error_magnitude,
    np_quat_from_euler_xyz,
    np_quat_inv,
    np_quat_mul,
    np_subtract_frame_transforms,
    np_yaw_quat,
)
from unilab.envs.locomotion.g1.base import G1BaseCfg, G1BaseEnv

from .motion_loader import MotionLoader, MotionSampler


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
            "motion_joint_pos": 0.0,
            "motion_joint_vel": 0.0,
            "action_rate_l2": -0.1,
            "joint_limit": -10.0,
            "undesired_contacts": -0.1,
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


@dataclass
class G1MotionTrackingCfg(G1BaseCfg):
    """Configuration for G1 motion tracking environment."""

    model_file: str = str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
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
    sampling_mode: Literal["start", "clip_start", "uniform", "adaptive"] = "adaptive"
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


@registry.envcfg("G1MotionTracking")
@dataclass
class G1MotionTrackingEnvCfg(G1MotionTrackingCfg):
    """Registered configuration for G1 motion tracking."""

    pass


class G1MotionTrackingDomainRandomizationProvider(DomainRandomizationProvider):
    def __init__(
        self, *, base_kp: np.ndarray | None = None, base_kd: np.ndarray | None = None
    ) -> None:
        self._base_kp = base_kp
        self._base_kd = base_kd

    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        validate_common_reset_randomization(
            env, capabilities, base_kp=self._base_kp, base_kd=self._base_kd
        )
        validate_interval_push_support(env, capabilities)

    def build_interval_randomization_plan(
        self, env: Any, step_counter: int
    ) -> IntervalRandomizationPlan | None:
        return build_interval_push_plan(env, step_counter)

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        dtype = get_global_dtype()
        num_reset = len(env_ids)

        motion_frames = env.motion_sampler.sample_frames(env_ids)
        motion_data = env.motion_loader.get_motion_at_frame(motion_frames)

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

        info_updates = {
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
        }
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=build_common_reset_randomization(
                env, num_reset, base_kp=self._base_kp, base_kd=self._base_kd
            ),
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
        return cast(
            dict[str, np.ndarray],
            env._compute_obs(
                info_updates,
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
            cfg.model_file,
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

        # Get non-EE body indices for undesired contact penalty
        ee_set = set(cfg.ee_body_names)
        self.undesired_contact_body_indices = np.array(
            [i for i, name in enumerate(cfg.body_names) if name not in ee_set],
            dtype=np.int32,
        )

        # Load motion data
        self.motion_loader = MotionLoader(cfg.motion_file, body_indices=motion_body_ids)
        self.motion_sampler = MotionSampler(
            self.motion_loader, mode=cfg.sampling_mode, num_envs=num_envs
        )
        if cfg.domain_rand.randomize_kp or cfg.domain_rand.randomize_kd:
            base_kp, base_kd = backend.get_actuator_gains()
            dr_provider = G1MotionTrackingDomainRandomizationProvider(
                base_kp=base_kp, base_kd=base_kd
            )
        else:
            dr_provider = G1MotionTrackingDomainRandomizationProvider()
        self._init_domain_randomization(dr_provider)

        # Buffers for relative body transforms
        self.body_pos_relative_w = np.zeros(
            (num_envs, len(cfg.body_names), 3), dtype=get_global_dtype()
        )
        self.body_quat_relative_w = np.zeros(
            (num_envs, len(cfg.body_names), 4), dtype=get_global_dtype()
        )
        self.body_quat_relative_w[:, :, 0] = 1.0  # Initialize to identity quaternion

        self._enable_reward_log = True
        self._init_reward_functions()
        self._clip_end_truncated = np.zeros((num_envs,), dtype=bool)

    def _get_body_pose_w(self) -> tuple[np.ndarray, np.ndarray]:
        return self._backend.get_body_pos_w(self.body_ids), self._backend.get_body_quat_w(
            self.body_ids
        )

    def _get_joint_range(self) -> np.ndarray | None:
        return self._backend.get_joint_range()  # type: ignore[no-any-return]

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # Actor: command(2n) + motion_anchor_pos_b(3) + motion_anchor_ori_b(6)
        #        + linvel(3) + gyro(3) + joint_pos(n) + joint_vel(n) + actions(n)
        n = self._num_action
        actor_dim = 3 + 6 + 3 + 3 + n * 5
        critic_extra_dim = len(self._cfg.body_names) * 9
        return {"obs": actor_dim, "critic": actor_dim + critic_extra_dim}

    def _init_reward_functions(self):
        self._reward_fns = {
            "motion_global_root_pos": self._reward_motion_global_root_pos,
            "motion_global_root_ori": self._reward_motion_global_root_ori,
            "motion_body_pos": self._reward_motion_body_pos,
            "motion_body_ori": self._reward_motion_body_ori,
            "motion_body_lin_vel": self._reward_motion_body_lin_vel,
            "motion_body_ang_vel": self._reward_motion_body_ang_vel,
            "motion_joint_pos": self._reward_motion_joint_pos,
            "motion_joint_vel": self._reward_motion_joint_vel,
            "action_rate_l2": self._reward_action_rate_l2,
            "joint_limit": self._reward_joint_limit,
            "undesired_contacts": self._reward_undesired_contacts,
        }

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._clip_end_truncated.fill(False)

        # Get current motion data
        motion_data = self.motion_sampler.get_current_motion()

        # Get robot state
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()

        # Get body states
        robot_body_pos_w, robot_body_quat_w = self._get_body_pose_w()
        robot_body_lin_vel_w = self._backend.get_body_lin_vel_w(self.body_ids)
        robot_body_ang_vel_w = self._backend.get_body_ang_vel_w(self.body_ids)

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
            self._clip_end_truncated[done_env_ids] = True

        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_truncated(self, state: NpEnvState) -> np.ndarray:
        truncated = super()._compute_truncated(state)
        clip_end_only = np.logical_and(self._clip_end_truncated, ~state.terminated)
        np.logical_or(truncated, clip_end_only, out=truncated)
        return truncated

    def _update_relative_transforms(
        self, motion_data, robot_body_pos_w: np.ndarray, robot_body_quat_w: np.ndarray
    ):
        """Update relative body transforms for tracking."""
        n_env = robot_body_pos_w.shape[0]
        n_body = len(self._cfg.body_names)

        # Get anchor states
        anchor_pos_w = motion_data.body_pos_w[:, self.anchor_body_idx]
        anchor_quat_w = motion_data.body_quat_w[:, self.anchor_body_idx]
        robot_anchor_pos_w = robot_body_pos_w[:, self.anchor_body_idx]
        robot_anchor_quat_w = robot_body_quat_w[:, self.anchor_body_idx]

        # Compute delta transform: keep robot's XY position, use motion's Z height
        # and apply yaw-only rotation difference
        delta_pos_w = robot_anchor_pos_w.copy()
        delta_pos_w[:, 2] = anchor_pos_w[:, 2]

        # Compute yaw-only rotation difference
        quat_diff = np_quat_mul(robot_anchor_quat_w, np_quat_inv(anchor_quat_w))
        delta_ori_w = np_yaw_quat(quat_diff)

        # Vectorized: transform all bodies at once using reshape trick
        # Flatten (N, B, 4) -> (N*B, 4) for quat ops, then reshape back
        delta_ori_tiled = np.tile(delta_ori_w, (1, n_body)).reshape(n_env * n_body, 4)
        motion_quat_flat = motion_data.body_quat_w.reshape(n_env * n_body, 4)
        self.body_quat_relative_w[:] = np_quat_mul(delta_ori_tiled, motion_quat_flat).reshape(
            n_env, n_body, 4
        )

        rel_pos_all = motion_data.body_pos_w - anchor_pos_w[:, None, :]  # (N, B, 3)
        rel_pos_flat = rel_pos_all.reshape(n_env * n_body, 3)
        rotated = np_quat_apply(delta_ori_tiled, rel_pos_flat).reshape(n_env, n_body, 3)
        self.body_pos_relative_w[:] = delta_pos_w[:, None, :] + rotated

    def _compute_terminations(
        self,
        motion_data,
        robot_body_pos_w: np.ndarray,
        robot_body_quat_w: np.ndarray,
    ) -> np.ndarray:
        """Compute termination conditions."""
        terminated = np.zeros(self._num_envs, dtype=bool)

        # Anchor position error (Z-axis only)
        anchor_pos_w = motion_data.body_pos_w[:, self.anchor_body_idx]
        robot_anchor_pos_w = robot_body_pos_w[:, self.anchor_body_idx]
        anchor_pos_error_z = np.abs(anchor_pos_w[:, 2] - robot_anchor_pos_w[:, 2])
        terminated |= anchor_pos_error_z > self._cfg.anchor_pos_z_threshold

        # Anchor orientation error (gravity direction)
        anchor_quat_w = motion_data.body_quat_w[:, self.anchor_body_idx]
        robot_anchor_quat_w = robot_body_quat_w[:, self.anchor_body_idx]
        gravity_vec = np.broadcast_to(
            np.array([[0, 0, -1]], dtype=get_global_dtype()),
            (anchor_quat_w.shape[0], 3),
        ).copy()
        motion_gravity_b = np_quat_apply(np_quat_inv(anchor_quat_w), gravity_vec)
        robot_gravity_b = np_quat_apply(np_quat_inv(robot_anchor_quat_w), gravity_vec)
        gravity_error = np.abs(motion_gravity_b[:, 2] - robot_gravity_b[:, 2])
        terminated |= gravity_error > self._cfg.anchor_ori_threshold

        # End-effector position error (Z-axis only)
        for ee_idx in self.ee_body_indices:
            ee_pos_error_z = np.abs(
                self.body_pos_relative_w[:, ee_idx, 2] - robot_body_pos_w[:, ee_idx, 2]
            )
            terminated |= ee_pos_error_z > self._cfg.ee_body_pos_z_threshold

        return terminated

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
        command = np.concatenate(
            [motion_data.joint_pos, motion_data.joint_vel],
            axis=1,
            dtype=get_global_dtype(),
        )

        # Get anchor states
        anchor_pos_w = motion_data.body_pos_w[:, self.anchor_body_idx]
        anchor_quat_w = motion_data.body_quat_w[:, self.anchor_body_idx]
        robot_anchor_pos_w = robot_body_pos_w[:, self.anchor_body_idx]
        robot_anchor_quat_w = robot_body_quat_w[:, self.anchor_body_idx]

        # Motion anchor position in robot frame
        motion_anchor_pos_b, _ = np_subtract_frame_transforms(
            robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w
        )

        # Motion anchor orientation in robot frame (as rotation matrix first 2 columns)
        _, motion_anchor_ori_rel = np_subtract_frame_transforms(
            robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w
        )
        motion_anchor_ori_mat = np_matrix_from_quat(motion_anchor_ori_rel)
        motion_anchor_ori_b = motion_anchor_ori_mat[:, :, :2].reshape(num_envs, 6)

        # Joint positions and velocities
        joint_pos_rel = dof_pos - self.default_angles
        last_actions = info.get("current_actions", np.zeros_like(joint_pos_rel))

        # Per-step observation noise on sensor channels
        noise_cfg = self._cfg.noise_config
        linvel = self._obs_noise(linvel, noise_cfg.scale_linvel)
        gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        joint_pos_rel = self._obs_noise(joint_pos_rel, noise_cfg.scale_joint_angle)
        dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)

        # Actor observations
        actor_obs = np.concatenate(
            [
                command,
                motion_anchor_pos_b,
                motion_anchor_ori_b,
                linvel,
                gyro,
                joint_pos_rel,
                dof_vel,
                last_actions,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )

        # Robot body positions in robot anchor frame (critic-only) — vectorized
        n_body = len(self._cfg.body_names)
        # Flatten (N, B, *) -> (N*B, *) for batched frame transform
        anchor_pos_tiled = np.tile(robot_anchor_pos_w, (1, n_body)).reshape(num_envs * n_body, 3)
        anchor_quat_tiled = np.tile(robot_anchor_quat_w, (1, n_body)).reshape(num_envs * n_body, 4)
        body_pos_flat = robot_body_pos_w.reshape(num_envs * n_body, 3)
        body_quat_flat = robot_body_quat_w.reshape(num_envs * n_body, 4)

        pos_b_flat, ori_b_flat = np_subtract_frame_transforms(
            anchor_pos_tiled, anchor_quat_tiled, body_pos_flat, body_quat_flat
        )
        robot_body_pos_b = pos_b_flat.reshape(num_envs, n_body, 3)
        ori_mat = np_matrix_from_quat(ori_b_flat)  # (N*B, 3, 3)
        robot_body_ori_b = ori_mat[:, :, :2].reshape(num_envs, n_body, 6)

        critic_extra = np.concatenate(
            [
                robot_body_pos_b.reshape(num_envs, -1),
                robot_body_ori_b.reshape(num_envs, -1),
            ],
            axis=1,
            dtype=get_global_dtype(),
        )

        critic_obs = np.concatenate([actor_obs, critic_extra], axis=1, dtype=get_global_dtype())
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
        dtype = get_global_dtype()
        reward = np.zeros((self._num_envs,), dtype=dtype)
        cfg = self._cfg.reward_config

        step_count = info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
        should_log = self._enable_reward_log and (int(step_count[0]) % 4 == 0)
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
            if scale == 0 or name not in self._reward_fns:
                continue
            rew = self._reward_fns[name](info)
            weighted_rew = rew * scale
            reward += weighted_rew
            if should_log:
                log[f"reward/{name}"] = float(np.mean(weighted_rew))

        info["log"] = log
        return reward * self._cfg.ctrl_dt

    # Reward functions
    def _reward_motion_global_root_pos(self, info: dict) -> np.ndarray:
        motion_data = info["motion_data"]
        robot_body_pos_w = info["robot_body_pos_w"]
        anchor_pos_w = motion_data.body_pos_w[:, self.anchor_body_idx]
        robot_anchor_pos_w = robot_body_pos_w[:, self.anchor_body_idx]
        error = np.sum(np.square(anchor_pos_w - robot_anchor_pos_w), axis=-1)
        return np.asarray(
            np.exp(-error / self._cfg.reward_config.std_root_pos**2), dtype=get_global_dtype()
        )

    def _reward_motion_global_root_ori(self, info: dict) -> np.ndarray:
        motion_data = info["motion_data"]
        robot_body_quat_w = info["robot_body_quat_w"]
        anchor_quat_w = motion_data.body_quat_w[:, self.anchor_body_idx]
        robot_anchor_quat_w = robot_body_quat_w[:, self.anchor_body_idx]
        error = np_quat_error_magnitude(anchor_quat_w, robot_anchor_quat_w) ** 2
        return np.exp(-error / self._cfg.reward_config.std_root_ori**2)

    def _reward_motion_body_pos(self, info: dict) -> np.ndarray:
        robot_body_pos_w = info["robot_body_pos_w"]
        error = np.sum(np.square(self.body_pos_relative_w - robot_body_pos_w), axis=-1)
        return np.asarray(
            np.exp(-error.mean(-1) / self._cfg.reward_config.std_body_pos**2),
            dtype=get_global_dtype(),
        )

    def _reward_motion_body_ori(self, info: dict) -> np.ndarray:
        robot_body_quat_w = info["robot_body_quat_w"]
        # np_quat_error_magnitude only supports (N, 4) — flatten body dim first
        n_env, n_body = robot_body_quat_w.shape[:2]
        ref_flat = self.body_quat_relative_w.reshape(n_env * n_body, 4)
        rob_flat = robot_body_quat_w.reshape(n_env * n_body, 4)
        error = np_quat_error_magnitude(ref_flat, rob_flat) ** 2
        error = error.reshape(n_env, n_body)
        return np.asarray(
            np.exp(-error.mean(-1) / self._cfg.reward_config.std_body_ori**2),
            dtype=get_global_dtype(),
        )

    def _reward_motion_body_lin_vel(self, info: dict) -> np.ndarray:
        motion_data = info["motion_data"]
        robot_body_lin_vel_w = info["robot_body_lin_vel_w"]
        error = np.sum(np.square(motion_data.body_lin_vel_w - robot_body_lin_vel_w), axis=-1)
        return np.asarray(
            np.exp(-error.mean(-1) / self._cfg.reward_config.std_body_lin_vel**2),
            dtype=get_global_dtype(),
        )

    def _reward_motion_body_ang_vel(self, info: dict) -> np.ndarray:
        motion_data = info["motion_data"]
        robot_body_ang_vel_w = info["robot_body_ang_vel_w"]
        error = np.sum(np.square(motion_data.body_ang_vel_w - robot_body_ang_vel_w), axis=-1)
        return np.asarray(
            np.exp(-error.mean(-1) / self._cfg.reward_config.std_body_ang_vel**2),
            dtype=get_global_dtype(),
        )

    def _reward_motion_joint_pos(self, info: dict) -> np.ndarray:
        motion_data = info["motion_data"]
        dof_pos = info["dof_pos"]
        error = np.mean(np.square(motion_data.joint_pos - dof_pos), axis=-1)
        return np.asarray(
            np.exp(-error / self._cfg.reward_config.std_joint_pos**2), dtype=get_global_dtype()
        )

    def _reward_motion_joint_vel(self, info: dict) -> np.ndarray:
        motion_data = info["motion_data"]
        dof_vel = info["dof_vel"]
        error = np.mean(np.square(motion_data.joint_vel - dof_vel), axis=-1)
        return np.asarray(
            np.exp(-error / self._cfg.reward_config.std_joint_vel**2), dtype=get_global_dtype()
        )

    def _reward_undesired_contacts(self, info: dict) -> np.ndarray:
        robot_body_pos_w = info["robot_body_pos_w"]
        body_z = robot_body_pos_w[:, self.undesired_contact_body_indices, 2]
        is_contact = body_z < self._cfg.undesired_contact_z_threshold
        return np.asarray(is_contact.sum(axis=-1), dtype=get_global_dtype())

    def _reward_action_rate_l2(self, info: dict) -> np.ndarray:
        action_diff = info["current_actions"] - info["last_actions"]
        return np.asarray(np.sum(np.square(action_diff), axis=1), dtype=get_global_dtype())

    def _reward_joint_limit(self, info: dict) -> np.ndarray:
        dof_pos = info["dof_pos"]
        joint_range = self._get_joint_range()
        if joint_range is None:
            return np.zeros((self._num_envs,), dtype=get_global_dtype())
        lower = joint_range[:, 0]
        upper = joint_range[:, 1]

        # Compute violation
        lower_violation = np.maximum(0, lower - dof_pos)
        upper_violation = np.maximum(0, dof_pos - upper)
        violation = lower_violation + upper_violation
        return np.asarray(np.sum(np.square(violation), axis=1), dtype=get_global_dtype())
