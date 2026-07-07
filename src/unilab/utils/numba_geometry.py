"""Reusable Numba geometry helpers.

The helpers in this module are intentionally low level: quaternion math,
body-frame vector transforms, and compact rotation representations. Task-owned
Numba kernels can compose them without moving reward or observation layout
knowledge into shared utilities.
"""

from __future__ import annotations

import math
from typing import Any

try:  # pragma: no cover - exercised when optional numba dependency is installed
    from numba import njit

    NUMBA_GEOMETRY_AVAILABLE = True
except Exception:  # pragma: no cover
    njit = None  # type: ignore[assignment]
    NUMBA_GEOMETRY_AVAILABLE = False


def _missing_numba(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("numba_geometry helpers require numba to be installed")


if NUMBA_GEOMETRY_AVAILABLE:

    def _dev(fn):
        return njit(inline="always", fastmath=True, cache=True, nogil=True)(fn)

    @_dev
    def quat_angle_sq_at(q1, q2, item_idx, i):
        dot = (
            q1[i, item_idx, 0] * q2[i, item_idx, 0]
            + q1[i, item_idx, 1] * q2[i, item_idx, 1]
            + q1[i, item_idx, 2] * q2[i, item_idx, 2]
            + q1[i, item_idx, 3] * q2[i, item_idx, 3]
        )
        dot = abs(dot)
        if dot > 1.0:
            dot = 1.0
        angle = 2.0 * math.acos(dot)
        return angle * angle

    @_dev
    def quat_gravity_z_at(quat, item_idx, i):
        return (
            2.0
            * (
                quat[i, item_idx, 1] * quat[i, item_idx, 1]
                + quat[i, item_idx, 2] * quat[i, item_idx, 2]
            )
            - 1.0
        )

    @_dev
    def quat_yaw_from_components(qw, qx, qy, qz):
        return math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

    @_dev
    def rotate_vec_by_inv_quat_components(qw, qx, qy, qz, vx, vy, vz, out, i, offset):
        ix = -qx
        iy = -qy
        iz = -qz
        tx = 2.0 * (iy * vz - iz * vy)
        ty = 2.0 * (iz * vx - ix * vz)
        tz = 2.0 * (ix * vy - iy * vx)
        out[i, offset] = vx + qw * tx + iy * tz - iz * ty
        out[i, offset + 1] = vy + qw * ty + iz * tx - ix * tz
        out[i, offset + 2] = vz + qw * tz + ix * ty - iy * tx

    @_dev
    def write_quat_first_two_matrix_cols_from_components(qw, qx, qy, qz, out, i, offset):
        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        wx = qw * qx
        wy = qw * qy
        wz = qw * qz
        out[i, offset] = 1.0 - 2.0 * (yy + zz)
        out[i, offset + 1] = 2.0 * (xy - wz)
        out[i, offset + 2] = 2.0 * (xy + wz)
        out[i, offset + 3] = 1.0 - 2.0 * (xx + zz)
        out[i, offset + 4] = 2.0 * (xz - wy)
        out[i, offset + 5] = 2.0 * (yz + wx)

    @_dev
    def write_yaw_aligned_body_transforms_at(
        source_body_pos_w,
        source_body_quat_w,
        target_body_pos_w,
        target_body_quat_w,
        anchor,
        n_body,
        out_body_pos_w,
        out_body_quat_w,
        i,
    ):
        anchor_px = source_body_pos_w[i, anchor, 0]
        anchor_py = source_body_pos_w[i, anchor, 1]
        anchor_pz = source_body_pos_w[i, anchor, 2]
        anchor_qw = source_body_quat_w[i, anchor, 0]
        anchor_qx = source_body_quat_w[i, anchor, 1]
        anchor_qy = source_body_quat_w[i, anchor, 2]
        anchor_qz = source_body_quat_w[i, anchor, 3]

        target_anchor_px = target_body_pos_w[i, anchor, 0]
        target_anchor_py = target_body_pos_w[i, anchor, 1]
        target_anchor_qw = target_body_quat_w[i, anchor, 0]
        target_anchor_qx = target_body_quat_w[i, anchor, 1]
        target_anchor_qy = target_body_quat_w[i, anchor, 2]
        target_anchor_qz = target_body_quat_w[i, anchor, 3]

        qw = (
            target_anchor_qw * anchor_qw
            + target_anchor_qx * anchor_qx
            + target_anchor_qy * anchor_qy
            + target_anchor_qz * anchor_qz
        )
        qx = (
            -target_anchor_qw * anchor_qx
            + target_anchor_qx * anchor_qw
            - target_anchor_qy * anchor_qz
            + target_anchor_qz * anchor_qy
        )
        qy = (
            -target_anchor_qw * anchor_qy
            + target_anchor_qx * anchor_qz
            + target_anchor_qy * anchor_qw
            - target_anchor_qz * anchor_qx
        )
        qz = (
            -target_anchor_qw * anchor_qz
            - target_anchor_qx * anchor_qy
            + target_anchor_qy * anchor_qx
            + target_anchor_qz * anchor_qw
        )
        half_yaw = 0.5 * quat_yaw_from_components(qw, qx, qy, qz)
        delta_qw = math.cos(half_yaw)
        delta_qz = math.sin(half_yaw)
        yaw_cross = 2.0 * delta_qw * delta_qz
        yaw_z2 = 2.0 * delta_qz * delta_qz

        for body_idx in range(n_body):
            mw = source_body_quat_w[i, body_idx, 0]
            mx = source_body_quat_w[i, body_idx, 1]
            my = source_body_quat_w[i, body_idx, 2]
            mz = source_body_quat_w[i, body_idx, 3]
            out_body_quat_w[i, body_idx, 0] = delta_qw * mw - delta_qz * mz
            out_body_quat_w[i, body_idx, 1] = delta_qw * mx - delta_qz * my
            out_body_quat_w[i, body_idx, 2] = delta_qw * my + delta_qz * mx
            out_body_quat_w[i, body_idx, 3] = delta_qw * mz + delta_qz * mw

            vx = source_body_pos_w[i, body_idx, 0] - anchor_px
            vy = source_body_pos_w[i, body_idx, 1] - anchor_py
            vz = source_body_pos_w[i, body_idx, 2] - anchor_pz
            out_body_pos_w[i, body_idx, 0] = (
                vx - yaw_cross * vy - yaw_z2 * vx + target_anchor_px
            )
            out_body_pos_w[i, body_idx, 1] = (
                vy + yaw_cross * vx - yaw_z2 * vy + target_anchor_py
            )
            out_body_pos_w[i, body_idx, 2] = vz + anchor_pz

    @_dev
    def write_relative_anchor_transform_at(
        source_body_pos_w,
        source_body_quat_w,
        target_body_pos_w,
        target_body_quat_w,
        anchor,
        out_pos_b,
        out_rot6d_b,
        i,
    ):
        anchor_px = source_body_pos_w[i, anchor, 0]
        anchor_py = source_body_pos_w[i, anchor, 1]
        anchor_pz = source_body_pos_w[i, anchor, 2]
        anchor_qw = source_body_quat_w[i, anchor, 0]
        anchor_qx = source_body_quat_w[i, anchor, 1]
        anchor_qy = source_body_quat_w[i, anchor, 2]
        anchor_qz = source_body_quat_w[i, anchor, 3]

        target_anchor_px = target_body_pos_w[i, anchor, 0]
        target_anchor_py = target_body_pos_w[i, anchor, 1]
        target_anchor_pz = target_body_pos_w[i, anchor, 2]
        target_anchor_qw = target_body_quat_w[i, anchor, 0]
        target_anchor_qx = target_body_quat_w[i, anchor, 1]
        target_anchor_qy = target_body_quat_w[i, anchor, 2]
        target_anchor_qz = target_body_quat_w[i, anchor, 3]

        rotate_vec_by_inv_quat_components(
            target_anchor_qw,
            target_anchor_qx,
            target_anchor_qy,
            target_anchor_qz,
            anchor_px - target_anchor_px,
            anchor_py - target_anchor_py,
            anchor_pz - target_anchor_pz,
            out_pos_b,
            i,
            0,
        )

        rw = (
            target_anchor_qw * anchor_qw
            + target_anchor_qx * anchor_qx
            + target_anchor_qy * anchor_qy
            + target_anchor_qz * anchor_qz
        )
        rx = (
            target_anchor_qw * anchor_qx
            - target_anchor_qx * anchor_qw
            - target_anchor_qy * anchor_qz
            + target_anchor_qz * anchor_qy
        )
        ry = (
            target_anchor_qw * anchor_qy
            + target_anchor_qx * anchor_qz
            - target_anchor_qy * anchor_qw
            - target_anchor_qz * anchor_qx
        )
        rz = (
            target_anchor_qw * anchor_qz
            - target_anchor_qx * anchor_qy
            + target_anchor_qy * anchor_qx
            - target_anchor_qz * anchor_qw
        )
        write_quat_first_two_matrix_cols_from_components(rw, rx, ry, rz, out_rot6d_b, i, 0)

    @_dev
    def write_body_pos_relative_to_anchor_at(body_pos_w, anchor_quat_w, anchor, n_body, out, i, offset):
        anchor_px = body_pos_w[i, anchor, 0]
        anchor_py = body_pos_w[i, anchor, 1]
        anchor_pz = body_pos_w[i, anchor, 2]
        aw = anchor_quat_w[i, anchor, 0]
        ax = anchor_quat_w[i, anchor, 1]
        ay = anchor_quat_w[i, anchor, 2]
        az = anchor_quat_w[i, anchor, 3]

        for body_idx in range(n_body):
            rotate_vec_by_inv_quat_components(
                aw,
                ax,
                ay,
                az,
                body_pos_w[i, body_idx, 0] - anchor_px,
                body_pos_w[i, body_idx, 1] - anchor_py,
                body_pos_w[i, body_idx, 2] - anchor_pz,
                out,
                i,
                offset + body_idx * 3,
            )

    @_dev
    def write_body_quat_relative_6d_to_anchor_at(body_quat_w, anchor, n_body, out, i, offset):
        aw = body_quat_w[i, anchor, 0]
        ax = body_quat_w[i, anchor, 1]
        ay = body_quat_w[i, anchor, 2]
        az = body_quat_w[i, anchor, 3]
        for body_idx in range(n_body):
            bw = body_quat_w[i, body_idx, 0]
            bx = body_quat_w[i, body_idx, 1]
            by = body_quat_w[i, body_idx, 2]
            bz = body_quat_w[i, body_idx, 3]
            rw = aw * bw + ax * bx + ay * by + az * bz
            rx = aw * bx - ax * bw - ay * bz + az * by
            ry = aw * by + ax * bz - ay * bw - az * bx
            rz = aw * bz - ax * by + ay * bx - az * bw
            write_quat_first_two_matrix_cols_from_components(
                rw, rx, ry, rz, out, i, offset + body_idx * 6
            )

else:
    quat_angle_sq_at = _missing_numba
    quat_gravity_z_at = _missing_numba
    quat_yaw_from_components = _missing_numba
    rotate_vec_by_inv_quat_components = _missing_numba
    write_quat_first_two_matrix_cols_from_components = _missing_numba
    write_yaw_aligned_body_transforms_at = _missing_numba
    write_relative_anchor_transform_at = _missing_numba
    write_body_pos_relative_to_anchor_at = _missing_numba
    write_body_quat_relative_6d_to_anchor_at = _missing_numba
