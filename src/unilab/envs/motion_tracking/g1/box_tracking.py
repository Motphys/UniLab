"""G1 box tracking environment with object-aware motion imitation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.dr import ResetPlan
from unilab.dr.dr_utils import build_common_reset_randomization, zero_actions
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
)

from .motion_box_loader import BoxMotionData, BoxMotionLoader
from .tracking import (
    G1MotionTrackingCfg,
    G1MotionTrackingDomainRandomizationProvider,
    G1MotionTrackingEnv,
    RewardConfig,
)


@dataclass
class BoxRewardConfig(RewardConfig):
    """Reward config extended with object-tracking terms."""

    scales: dict[str, float] = field(
        default_factory=lambda: {
            **RewardConfig().scales,
            "undesired_contacts": -0.1,
            "object_global_ref_position_error_exp": 1.0,
            "object_global_ref_orientation_error_exp": 1.0,
        }
    )
    std_object_pos: float = 0.3
    std_object_ori: float = 0.4


@dataclass
class G1BoxTrackingCfg(G1MotionTrackingCfg):
    """Configuration for the G1 large-box tracking task."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat_with_largebox.xml")
        )
    )
    motion_file: str | list[str] = str(
        ASSETS_ROOT_PATH / "motions" / "g1" / "sub3_largebox_003_boxconverted.npz"
    )
    object_body_name: str = "largebox"
    object_pos_threshold: float = 0.25
    object_ori_threshold: float = 0.8
    reward_config: BoxRewardConfig = field(default_factory=BoxRewardConfig)


@registry.envcfg("G1BoxTracking")
@dataclass
class G1BoxTrackingEnvCfg(G1BoxTrackingCfg):
    """Registered config for G1 box tracking."""

    pass


def _build_box_motion_reference_state(
    env: Any, env_ids: np.ndarray, motion_data: BoxMotionData
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
    qpos[:, 7 : 7 + joint_pos.shape[1]] = joint_pos

    qvel[:, 0:3] = root_lin_vel
    qvel[:, 3:6] = np_quat_apply(np_quat_inv(root_ori), root_ang_vel)
    qvel[:, 6 : 6 + joint_vel.shape[1]] = joint_vel

    if motion_data.object_pos_w is not None:
        qpos[:, env._obj_pos_slice] = motion_data.object_pos_w
        qpos[:, env._obj_quat_slice] = motion_data.object_quat_w
        qvel[:, env._obj_lin_vel_slice] = motion_data.object_lin_vel_w
        qvel[:, env._obj_ang_vel_slice] = motion_data.object_ang_vel_w

    return qpos, qvel


class G1BoxTrackingDomainRandomizationProvider(G1MotionTrackingDomainRandomizationProvider):
    """Reset provider that restores both robot and object state from motion data."""

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        motion_frames = env.motion_sampler.sample_frames(env_ids)
        motion_data = cast(BoxMotionData, env.motion_loader.get_motion_at_frame(motion_frames))
        qpos, qvel = _build_box_motion_reference_state(env, env_ids, motion_data)

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


@registry.env("G1BoxTracking", sim_backend="mujoco")
@registry.env("G1BoxTracking", sim_backend="motrix")
class G1BoxTrackingEnv(G1MotionTrackingEnv):
    """Motion tracking env extended with large-box state and rewards."""

    _cfg: G1BoxTrackingCfg

    def __init__(self, cfg: G1BoxTrackingCfg, num_envs=1, backend_type="mujoco"):
        super().__init__(cfg, num_envs, backend_type)

        motion_body_ids = self._backend.get_motion_body_ids(cfg.body_names)
        self.motion_loader = BoxMotionLoader(cfg.motion_file, body_indices=motion_body_ids)
        self.motion_sampler = type(self.motion_sampler)(
            self.motion_loader, mode=cfg.sampling_mode, num_envs=num_envs
        )

        if cfg.domain_rand.randomize_kp or cfg.domain_rand.randomize_kd:
            base_kp, base_kd = self._backend.get_actuator_gains()
            dr_provider = G1BoxTrackingDomainRandomizationProvider(base_kp=base_kp, base_kd=base_kd)
        else:
            dr_provider = G1BoxTrackingDomainRandomizationProvider()
        self._init_domain_randomization(dr_provider)

        self._object_body_ids = self._backend.get_body_ids([cfg.object_body_name])

        nq = self._init_qpos.shape[0]
        self._obj_pos_slice = slice(nq - 7, nq - 4)
        self._obj_quat_slice = slice(nq - 4, nq)
        nv = self._init_qvel.shape[0]
        self._obj_lin_vel_slice = slice(nv - 6, nv - 3)
        self._obj_ang_vel_slice = slice(nv - 3, nv)

        if not self.motion_loader.has_object:
            raise ValueError(
                f"Motion file '{cfg.motion_file}' does not contain object data. "
                "Expected keys: object_pos_w, object_quat_w, object_lin_vel_w, object_ang_vel_w"
            )

    def _get_joint_range(self) -> np.ndarray | None:
        joint_range = super()._get_joint_range()
        if joint_range is not None and joint_range.shape[0] > self.motion_loader.num_joints:
            joint_range = joint_range[: self.motion_loader.num_joints]
        return joint_range

    def _resample_reference_state(self, env_ids: np.ndarray) -> None:
        motion_frames = self.motion_sampler.sample_frames(env_ids)
        motion_data = cast(BoxMotionData, self.motion_loader.get_motion_at_frame(motion_frames))
        qpos, qvel = _build_box_motion_reference_state(self, env_ids, motion_data)
        self._backend.set_state(env_ids, qpos, qvel)

    def get_dof_pos(self) -> np.ndarray:
        dof_pos = super().get_dof_pos()
        return dof_pos[:, : self.motion_loader.num_joints]

    def get_dof_vel(self) -> np.ndarray:
        dof_vel = super().get_dof_vel()
        return dof_vel[:, : self.motion_loader.num_joints]

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        spec = super().obs_groups_spec
        return {**spec, "critic": spec["critic"] + 12}

    def _actor_obs_dim(self, n: int) -> int:
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

    def _init_reward_functions(self):
        super()._init_reward_functions()
        self._reward_fns["object_global_ref_position_error_exp"] = self._reward_object_position
        self._reward_fns["object_global_ref_orientation_error_exp"] = (
            self._reward_object_orientation
        )

    def _compute_terminations(
        self,
        motion_data: BoxMotionData,
        robot_body_pos_w: np.ndarray,
        robot_body_quat_w: np.ndarray,
    ) -> np.ndarray:
        terminated = super()._compute_terminations(motion_data, robot_body_pos_w, robot_body_quat_w)

        if motion_data.object_pos_w is not None:
            obj_pos_w = self._backend.get_body_pos_w(self._object_body_ids)[:, 0, :]
            obj_pos_error = np.linalg.norm(obj_pos_w - motion_data.object_pos_w, axis=-1)
            terminated |= obj_pos_error > self._cfg.object_pos_threshold

        if motion_data.object_quat_w is not None:
            obj_quat_w = self._backend.get_body_quat_w(self._object_body_ids)[:, 0, :]
            obj_ori_error = np_quat_error_magnitude(obj_quat_w, motion_data.object_quat_w)
            terminated |= obj_ori_error > self._cfg.object_ori_threshold

        return terminated

    def _compute_obs(
        self,
        info: dict,
        motion_data: BoxMotionData,
        linvel: np.ndarray,
        gyro: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        robot_body_pos_w: np.ndarray,
        robot_body_quat_w: np.ndarray,
    ) -> dict[str, np.ndarray]:
        obs = super()._compute_obs(
            info, motion_data, linvel, gyro, dof_pos, dof_vel, robot_body_pos_w, robot_body_quat_w
        )

        num_envs = linvel.shape[0]
        obj_pos_w = self._backend.get_body_pos_w(self._object_body_ids)[:num_envs, 0, :]
        obj_quat_w = self._backend.get_body_quat_w(self._object_body_ids)[:num_envs, 0, :]
        obj_lin_vel_w = self._backend.get_body_lin_vel_w(self._object_body_ids)[:num_envs, 0, :]

        anchor_pos_w = robot_body_pos_w[:, self.anchor_body_idx]
        anchor_quat_w = robot_body_quat_w[:, self.anchor_body_idx]

        obj_pos_b, obj_ori_rel = np_subtract_frame_transforms(
            anchor_pos_w, anchor_quat_w, obj_pos_w, obj_quat_w
        )
        obj_ori_mat = np_matrix_from_quat(obj_ori_rel)
        obj_ori_b = obj_ori_mat[:, :, :2].reshape(num_envs, 6)
        obj_lin_vel_b = np_quat_apply(np_quat_inv(anchor_quat_w), obj_lin_vel_w)

        object_obs = np.concatenate(
            [obj_pos_b, obj_ori_b, obj_lin_vel_b],
            axis=1,
            dtype=get_global_dtype(),
        )
        obs["critic"] = np.concatenate(
            [obs["critic"], object_obs], axis=1, dtype=get_global_dtype()
        )
        return obs

    def _reward_object_position(self, info: dict) -> np.ndarray:
        motion_data: BoxMotionData = info["motion_data"]
        if motion_data.object_pos_w is None:
            return np.zeros((self._num_envs,), dtype=get_global_dtype())
        obj_pos_w = self._backend.get_body_pos_w(self._object_body_ids)[:, 0, :]
        error = np.sum(np.square(obj_pos_w - motion_data.object_pos_w), axis=-1)
        return np.asarray(
            np.exp(-error / self._cfg.reward_config.std_object_pos**2), dtype=get_global_dtype()
        )

    def _reward_object_orientation(self, info: dict) -> np.ndarray:
        motion_data: BoxMotionData = info["motion_data"]
        if motion_data.object_quat_w is None:
            return np.zeros((self._num_envs,), dtype=get_global_dtype())
        obj_quat_w = self._backend.get_body_quat_w(self._object_body_ids)[:, 0, :]
        error = np_quat_error_magnitude(obj_quat_w, motion_data.object_quat_w) ** 2
        return np.asarray(
            np.exp(-error / self._cfg.reward_config.std_object_ori**2), dtype=get_global_dtype()
        )
