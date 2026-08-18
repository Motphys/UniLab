"""Shared termination computation for motion tracking."""

from __future__ import annotations

from typing import Any

import numpy as np

from unilab.utils.geometry import np_gravity_z_in_body_from_quat


def compute_terminations(
    env: Any,
    motion_data: Any,
    robot_body_pos_w: np.ndarray,
    robot_body_quat_w: np.ndarray,
) -> np.ndarray:
    """Compute termination conditions (writes into ``env._terminated``)."""
    terminated = env._terminated
    terminated.fill(False)

    # Anchor position error (Z-axis only)
    anchor_pos_w = motion_data.body_pos_w[:, env.anchor_body_idx]
    robot_anchor_pos_w = robot_body_pos_w[:, env.anchor_body_idx]
    np.subtract(anchor_pos_w[:, 2], robot_anchor_pos_w[:, 2], out=env._env_error)
    np.abs(env._env_error, out=env._env_error)
    np.greater(env._env_error, env._cfg.anchor_pos_z_threshold, out=env._env_bool)
    terminated |= env._env_bool

    # Anchor orientation error (gravity direction). The gravity-z difference
    # is bounded by 2 for unit quaternions, so huge thresholds disable this
    # termination without doing the per-step math.
    if env._cfg.anchor_ori_threshold < 2.0:
        anchor_quat_w = motion_data.body_quat_w[:, env.anchor_body_idx]
        robot_anchor_quat_w = robot_body_quat_w[:, env.anchor_body_idx]
        motion_gravity_z_b = np_gravity_z_in_body_from_quat(anchor_quat_w)
        robot_gravity_z_b = np_gravity_z_in_body_from_quat(robot_anchor_quat_w)
        np.subtract(motion_gravity_z_b, robot_gravity_z_b, out=env._env_error)
        np.abs(env._env_error, out=env._env_error)
        np.greater(env._env_error, env._cfg.anchor_ori_threshold, out=env._env_bool)
        terminated |= env._env_bool

    # End-effector position error (Z-axis only)
    if env._has_ee_body_indices:
        np.subtract(
            env.body_pos_relative_w[:, env.ee_body_indices, 2],
            robot_body_pos_w[:, env.ee_body_indices, 2],
            out=env._ee_pos_error_z,
        )
        np.abs(env._ee_pos_error_z, out=env._ee_pos_error_z)
        np.greater(
            env._ee_pos_error_z,
            env._cfg.ee_body_pos_z_threshold,
            out=env._ee_terminated,
        )
        np.logical_or.reduce(env._ee_terminated, axis=1, out=env._env_bool)
        terminated |= env._env_bool

    if env._cfg.terminate_on_undesired_contacts and env._has_undesired_contact_body_indices:
        body_z = robot_body_pos_w[:, env.undesired_contact_body_indices, 2]
        np.less(
            body_z,
            env._cfg.undesired_contact_z_threshold,
            out=env._undesired_contact_mask,
        )
        np.logical_or.reduce(env._undesired_contact_mask, axis=-1, out=env._env_bool)
        terminated |= env._env_bool

    return terminated
