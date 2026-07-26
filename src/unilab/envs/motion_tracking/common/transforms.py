"""Relative body-transform computation for motion tracking.

Fills the environment's ``body_pos_relative_w`` / ``body_quat_relative_w``
reference buffers each step. The op order and in-place ``out=`` usage are
load-bearing for numba parity and must not change.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def update_relative_transforms(
    env: Any,
    motion_data: Any,
    robot_body_pos_w: np.ndarray,
    robot_body_quat_w: np.ndarray,
) -> None:
    """Update relative body transforms for tracking."""
    # Get anchor states
    anchor_pos_w = motion_data.body_pos_w[:, env.anchor_body_idx]
    anchor_quat_w = motion_data.body_quat_w[:, env.anchor_body_idx]
    robot_anchor_pos_w = robot_body_pos_w[:, env.anchor_body_idx]
    robot_anchor_quat_w = robot_body_quat_w[:, env.anchor_body_idx]

    # Compute delta transform: keep robot's XY position, use motion's Z height
    # and apply yaw-only rotation difference.
    delta_pos_w = env._delta_pos_w
    delta_pos_w[:] = robot_anchor_pos_w
    delta_pos_w[:, 2] = anchor_pos_w[:, 2]

    # Compute yaw-only rotation difference, equivalent to
    # np_yaw_quat(np_quat_mul(robot_anchor_quat_w, np_quat_inv(anchor_quat_w))).
    delta_ori_w = env._delta_ori_w
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
    out_quat = env.body_quat_relative_w
    out_quat[..., 0] = dw * mw
    out_quat[..., 0] -= dz * mz
    out_quat[..., 1] = dw * mx
    out_quat[..., 1] -= dz * my
    out_quat[..., 2] = dw * my
    out_quat[..., 2] += dz * mx
    out_quat[..., 3] = dw * mz
    out_quat[..., 3] += dz * mw

    rel_pos = env._body_vec_error
    vx = rel_pos[..., 0]
    vy = rel_pos[..., 1]
    vz = rel_pos[..., 2]
    np.subtract(motion_data.body_pos_w[..., 0], anchor_pos_w[:, None, 0], out=vx)
    np.subtract(motion_data.body_pos_w[..., 1], anchor_pos_w[:, None, 1], out=vy)
    np.subtract(motion_data.body_pos_w[..., 2], anchor_pos_w[:, None, 2], out=vz)

    yaw_cross = env._env_error
    yaw_z2 = env._reward_term
    np.multiply(dw1, dz1, out=yaw_cross)
    yaw_cross *= 2.0
    np.square(dz1, out=yaw_z2)
    yaw_z2 *= 2.0
    yaw_cross_2d = yaw_cross[:, None]
    yaw_z2_2d = yaw_z2[:, None]

    out_pos = env.body_pos_relative_w
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
