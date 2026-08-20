"""In-place body-frame writers shared by motion manager terms."""

from __future__ import annotations

import numpy as np


def write_body_pos_in_anchor_frame(
    anchor_pos: np.ndarray,
    anchor_quat: np.ndarray,
    body_pos: np.ndarray,
    out: np.ndarray,
    *,
    body_vec_error: np.ndarray,
) -> None:
    aw = anchor_quat[:, None, 0]
    ax = anchor_quat[:, None, 1]
    ay = anchor_quat[:, None, 2]
    az = anchor_quat[:, None, 3]

    num_envs, num_bodies = body_pos.shape[:2]
    rel_pos = body_vec_error[:num_envs, :num_bodies]
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


def write_body_ori6_in_anchor_frame(
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


__all__ = ["write_body_ori6_in_anchor_frame", "write_body_pos_in_anchor_frame"]
