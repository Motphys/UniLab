"""Shared reset-state construction for motion tracking."""

from __future__ import annotations

from typing import Any

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.utils.geometry import np_sample_uniform
from unilab.utils.rotation import np_quat_apply, np_quat_from_euler_xyz, np_quat_inv, np_quat_mul

from .motion_loader import MotionData


def build_motion_reference_state(
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
