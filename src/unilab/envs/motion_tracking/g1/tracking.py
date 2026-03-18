"""G1 Motion Tracking Environment - Motion imitation task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import mujoco
import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.dtype_config import get_global_dtype
from unilab.base.np_env import NpEnvState
from unilab.envs.locomotion.g1.base import G1BaseCfg, G1BaseEnv
from unilab.utils.math_utils import (
    np_matrix_from_quat,
    np_quat_apply,
    np_quat_error_magnitude,
    np_quat_from_euler_xyz,
    np_quat_inv,
    np_quat_mul,
    np_sample_uniform,
    np_subtract_frame_transforms,
    np_yaw_quat,
)

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
class G1MotionTrackingCfg(G1BaseCfg):
    """Configuration for G1 motion tracking environment."""

    model_file: str = str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
    # motion_file: str = str(ASSETS_ROOT_PATH / "motions" / "g1" / "gangnam_style.npz")
    motion_file: str = str(ASSETS_ROOT_PATH / "motions" / "g1" / "dance1_subject2_part.npz")
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
    sampling_mode: Literal["start", "uniform", "adaptive"] = "adaptive"
    log_action_scale: bool = False
    max_episode_seconds: float = 20.0
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    pose_randomization: PoseRandomization = field(default_factory=PoseRandomization)
    velocity_randomization: VelocityRandomization = field(default_factory=VelocityRandomization)
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


@registry.envcfg("G1MotionTracking")
@dataclass
class G1MotionTrackingEnvCfg(G1MotionTrackingCfg):
    """Registered configuration for G1 motion tracking."""

    pass


@registry.env("G1MotionTracking", sim_backend="mujoco")
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
            add_body_sensors=True,
        )
        super().__init__(cfg, backend, num_envs)
        self._apply_mjlab_g1_action_scale()
        self._log_action_scale_diagnostics()

        # Get body IDs from MuJoCo model
        self.body_ids = np.array(
            [
                mujoco.mj_name2id(self._backend.model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in cfg.body_names
            ],
            dtype=np.int32,
        )
        self.anchor_body_idx = cfg.body_names.index(cfg.anchor_body_name)

        # Get end-effector body indices for termination
        self.ee_body_indices = np.array(
            [cfg.body_names.index(name) for name in cfg.ee_body_names], dtype=np.int32
        )

        # Load motion data
        self.motion_loader = MotionLoader(cfg.motion_file, body_indices=self.body_ids)
        self.motion_sampler = MotionSampler(
            self.motion_loader, mode=cfg.sampling_mode, num_envs=num_envs
        )

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

    def _apply_mjlab_g1_action_scale(self) -> None:
        """Set per-joint action scale to match mjlab's normalization."""
        model = self._backend.model
        nu = int(model.nu)

        base_scale = self._cfg.control_config.action_scale
        if isinstance(base_scale, np.ndarray):
            action_scale = base_scale.astype(get_global_dtype(), copy=True)
        else:
            action_scale = np.full((nu,), float(base_scale), dtype=get_global_dtype())

        if hasattr(model, "actuator_gainprm"):
            effort_limit = np.zeros((nu,), dtype=get_global_dtype())
            if hasattr(model, "actuator_forcerange"):
                effort_limit = np.max(np.abs(model.actuator_forcerange), axis=1)

            # Fallback: derive actuator effort limits from joint force ranges when
            # actuator forcerange is not explicitly defined in XML.
            if np.all(effort_limit <= 0.0) and hasattr(model, "actuator_trnid") and hasattr(
                model, "jnt_actfrcrange"
            ):
                joint_ids = model.actuator_trnid[:, 0].astype(np.int32)
                effort_limit = np.max(np.abs(model.jnt_actfrcrange[joint_ids]), axis=1)

            stiffness = model.actuator_gainprm[:, 0]
            valid = (effort_limit > 0.0) & (np.abs(stiffness) > 1e-6)
            action_scale[valid] = 0.25 * effort_limit[valid] / stiffness[valid]

        self._cfg.control_config.action_scale = action_scale

    def _log_action_scale_diagnostics(self) -> None:
        """Log action-scale diagnostics to help detect control-mapping issues."""
        if not self._cfg.log_action_scale:
            return

        model = self._backend.model
        action_scale = self._cfg.control_config.action_scale
        if not isinstance(action_scale, np.ndarray):
            action_scale = np.full((int(model.nu),), float(action_scale), dtype=get_global_dtype())

        unique_scale = np.unique(np.round(action_scale, 6))
        print(f"[G1MotionTracking] action_scale unique: {unique_scale.tolist()}")

        preview_count = min(8, int(model.nu))
        preview = []
        for i in range(preview_count):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            preview.append(f"{name}:{float(action_scale[i]):.6f}")
        print("[G1MotionTracking] action_scale preview:", ", ".join(preview))

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # Actor: command(2n) + motion_anchor_pos_b(3) + motion_anchor_ori_b(6)
        #        + linvel(3) + gyro(3) + joint_pos(n) + joint_vel(n) + actions(n)
        n = self._num_action
        actor_dim = 3 + 6 + 3 + 3 + n * 5
        # Privileged: body_pos_b(num_bodies*3) + body_ori_b(num_bodies*6)
        privileged_dim = len(self._cfg.body_names) * 9
        return {"obs": actor_dim, "privileged": privileged_dim}

    def _init_reward_functions(self):
        self._reward_fns = {
            "motion_global_root_pos": self._reward_motion_global_root_pos,
            "motion_global_root_ori": self._reward_motion_global_root_ori,
            "motion_body_pos": self._reward_motion_body_pos,
            "motion_body_ori": self._reward_motion_body_ori,
            "motion_body_lin_vel": self._reward_motion_body_lin_vel,
            "motion_body_ang_vel": self._reward_motion_body_ang_vel,
            "action_rate_l2": self._reward_action_rate_l2,
            "joint_limit": self._reward_joint_limit,
        }

    def update_state(self, state: NpEnvState) -> NpEnvState:
        # Get current motion data
        motion_data = self.motion_sampler.get_current_motion()

        # Get robot state
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()

        # Get body states
        robot_body_pos_w = self._backend.get_body_pos_w(self.body_ids)
        robot_body_quat_w = self._backend.get_body_quat_w(self.body_ids)
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
            # Resample motion for environments that reached end
            self.motion_sampler.sample_frames(done_env_ids)

        return state.replace(obs=obs, reward=reward, terminated=terminated)

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
        # and apply yaw-only rotation difference
        delta_pos_w = robot_anchor_pos_w.copy()
        delta_pos_w[:, 2] = anchor_pos_w[:, 2]

        # Compute yaw-only rotation difference
        quat_diff = np_quat_mul(robot_anchor_quat_w, np_quat_inv(anchor_quat_w))
        delta_ori_w = np_yaw_quat(quat_diff)

        # Transform motion body states to robot-relative coordinates
        for i in range(len(self._cfg.body_names)):
            self.body_quat_relative_w[:, i] = np_quat_mul(
                delta_ori_w, motion_data.body_quat_w[:, i]
            )
            rel_pos = motion_data.body_pos_w[:, i] - anchor_pos_w
            self.body_pos_relative_w[:, i] = delta_pos_w + np_quat_apply(delta_ori_w, rel_pos)

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
        """Compute observations as dict with actor and privileged groups."""
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

        # Robot body positions in robot anchor frame (privileged)
        robot_body_pos_b = np.zeros(
            (num_envs, len(self._cfg.body_names), 3), dtype=get_global_dtype()
        )
        robot_body_ori_b = np.zeros(
            (num_envs, len(self._cfg.body_names), 6), dtype=get_global_dtype()
        )
        for i in range(len(self._cfg.body_names)):
            pos_b, ori_b = np_subtract_frame_transforms(
                robot_anchor_pos_w,
                robot_anchor_quat_w,
                robot_body_pos_w[:, i],
                robot_body_quat_w[:, i],
            )
            robot_body_pos_b[:, i] = pos_b
            ori_mat = np_matrix_from_quat(ori_b)
            robot_body_ori_b[:, i] = ori_mat[:, :, :2].reshape(num_envs, 6)

        privileged_obs = np.concatenate(
            [
                robot_body_pos_b.reshape(num_envs, -1),
                robot_body_ori_b.reshape(num_envs, -1),
            ],
            axis=1,
            dtype=get_global_dtype(),
        )

        return {"obs": actor_obs, "privileged": privileged_obs}

    def _compute_reward(
        self,
        info: dict,
        motion_data,
        robot_body_pos_w: np.ndarray,
        robot_body_quat_w: np.ndarray,
        robot_body_lin_vel_w: np.ndarray,
        robot_body_ang_vel_w: np.ndarray,
        dof_pos: np.ndarray,
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
        info["dof_pos"] = dof_pos

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
        return np.exp(-error / self._cfg.reward_config.std_root_pos**2)

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
        return np.exp(-error.mean(-1) / self._cfg.reward_config.std_body_pos**2)

    def _reward_motion_body_ori(self, info: dict) -> np.ndarray:
        robot_body_quat_w = info["robot_body_quat_w"]
        error = np_quat_error_magnitude(self.body_quat_relative_w, robot_body_quat_w) ** 2
        return np.exp(-error.mean(-1) / self._cfg.reward_config.std_body_ori**2)

    def _reward_motion_body_lin_vel(self, info: dict) -> np.ndarray:
        motion_data = info["motion_data"]
        robot_body_lin_vel_w = info["robot_body_lin_vel_w"]
        error = np.sum(np.square(motion_data.body_lin_vel_w - robot_body_lin_vel_w), axis=-1)
        return np.exp(-error.mean(-1) / self._cfg.reward_config.std_body_lin_vel**2)

    def _reward_motion_body_ang_vel(self, info: dict) -> np.ndarray:
        motion_data = info["motion_data"]
        robot_body_ang_vel_w = info["robot_body_ang_vel_w"]
        error = np.sum(np.square(motion_data.body_ang_vel_w - robot_body_ang_vel_w), axis=-1)
        return np.exp(-error.mean(-1) / self._cfg.reward_config.std_body_ang_vel**2)

    def _reward_action_rate_l2(self, info: dict) -> np.ndarray:
        action_diff = info["current_actions"] - info["last_actions"]
        return np.sum(np.square(action_diff), axis=1)

    def _reward_joint_limit(self, info: dict) -> np.ndarray:
        dof_pos = info["dof_pos"]
        # Get joint limits from model
        model = self._backend.model
        joint_range = model.jnt_range[1:]  # Skip free joint (1 joint, not 7 qpos)
        lower = joint_range[:, 0]
        upper = joint_range[:, 1]

        # Compute violation
        lower_violation = np.maximum(0, lower - dof_pos)
        upper_violation = np.maximum(0, dof_pos - upper)
        violation = lower_violation + upper_violation
        return np.sum(np.square(violation), axis=1)

    def reset(self, env_indices: np.ndarray):
        """Reset specified environments."""
        dtype = get_global_dtype()
        num_reset = len(env_indices)

        # Sample motion frames
        motion_frames = self.motion_sampler.sample_frames(env_indices)
        motion_data = self.motion_loader.get_motion_at_frame(motion_frames)

        # Get anchor body state from motion
        root_pos = motion_data.body_pos_w[:, 0].copy()
        root_ori = motion_data.body_quat_w[:, 0].copy()
        root_lin_vel = motion_data.body_lin_vel_w[:, 0].copy()
        root_ang_vel = motion_data.body_ang_vel_w[:, 0].copy()
        joint_pos = motion_data.joint_pos.copy()
        joint_vel = motion_data.joint_vel.copy()

        # Apply pose randomization
        pose_rand = self._cfg.pose_randomization
        range_list = [
            (pose_rand.x[0], pose_rand.x[1]),
            (pose_rand.y[0], pose_rand.y[1]),
            (pose_rand.z[0], pose_rand.z[1]),
            (pose_rand.roll[0], pose_rand.roll[1]),
            (pose_rand.pitch[0], pose_rand.pitch[1]),
            (pose_rand.yaw[0], pose_rand.yaw[1]),
        ]
        rand_samples = np.array(
            [[np.random.uniform(r[0], r[1]) for r in range_list] for _ in range(num_reset)],
            dtype=dtype,
        )
        root_pos += rand_samples[:, 0:3]
        ori_delta = np_quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
        )
        root_ori = np_quat_mul(ori_delta, root_ori)

        # Apply velocity randomization
        vel_rand = self._cfg.velocity_randomization
        range_list = [
            (vel_rand.x[0], vel_rand.x[1]),
            (vel_rand.y[0], vel_rand.y[1]),
            (vel_rand.z[0], vel_rand.z[1]),
            (vel_rand.roll[0], vel_rand.roll[1]),
            (vel_rand.pitch[0], vel_rand.pitch[1]),
            (vel_rand.yaw[0], vel_rand.yaw[1]),
        ]
        rand_samples = np.array(
            [[np.random.uniform(r[0], r[1]) for r in range_list] for _ in range(num_reset)],
            dtype=dtype,
        )
        root_lin_vel += rand_samples[:, :3]
        root_ang_vel += rand_samples[:, 3:]

        # Apply joint position randomization
        joint_pos += np_sample_uniform(
            self._cfg.joint_position_range[0],
            self._cfg.joint_position_range[1],
            joint_pos.shape,
            dtype=dtype,
        )

        # Clip joint positions to limits
        model = self._backend.model
        joint_range = model.jnt_range[1:]  # Skip free joint (1 joint, not 7 qpos)
        joint_pos = np.clip(joint_pos, joint_range[:, 0], joint_range[:, 1])

        # Construct qpos and qvel
        qpos = np.tile(self._init_qpos, (num_reset, 1))
        qvel = np.tile(self._init_qvel, (num_reset, 1))

        qpos[:, 0:3] = root_pos
        qpos[:, 3:7] = root_ori
        qpos[:, 7:] = joint_pos

        qvel[:, 0:3] = root_lin_vel
        qvel[:, 3:6] = root_ang_vel
        qvel[:, 6:] = joint_vel

        # Set state
        self._backend.set_state(env_indices, qpos, qvel)

        # Initialize info
        info = {
            "current_actions": np.zeros((num_reset, self._num_action), dtype=dtype),
            "last_actions": np.zeros((num_reset, self._num_action), dtype=dtype),
        }

        # Compute initial observations — slice motion data to match env_indices
        motion_data = self.motion_loader.get_motion_at_frame(
            self.motion_sampler.current_frames[env_indices]
        )
        linvel = self.get_local_linvel()[env_indices]
        gyro = self.get_gyro()[env_indices]
        dof_pos = self.get_dof_pos()[env_indices]
        dof_vel = self.get_dof_vel()[env_indices]
        robot_body_pos_w = self._backend.get_body_pos_w(self.body_ids)[env_indices]
        robot_body_quat_w = self._backend.get_body_quat_w(self.body_ids)[env_indices]

        obs = self._compute_obs(
            info, motion_data, linvel, gyro, dof_pos, dof_vel, robot_body_pos_w, robot_body_quat_w
        )

        return obs, info
