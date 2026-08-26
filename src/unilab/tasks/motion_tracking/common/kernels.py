"""Parallel CPU kernels for the fixed motion-tracking hot-term set."""

from __future__ import annotations

import os
from math import atan2, cos, sin, sqrt

import numpy as np
from numba import config, njit, prange, set_num_threads

_DEFAULT_MOTION_KERNEL_THREADS = 8
_runtime_configured = False

# The OpenMP layer's worker wakeups dominate these short kernels after a long
# backend physics phase. Workqueue has stable sub-millisecond dispatch here.
# Respect an application-provided NUMBA_THREADING_LAYER selection.
if "NUMBA_THREADING_LAYER" not in os.environ:
    setattr(config, "THREADING_LAYER", "workqueue")


def configure_motion_kernel_runtime() -> None:
    """Initialize the task-local Numba worker mask once on the cold path."""
    global _runtime_configured
    if _runtime_configured:
        return
    if "NUMBA_NUM_THREADS" not in os.environ:
        available_threads = int(getattr(config, "NUMBA_DEFAULT_NUM_THREADS", os.cpu_count() or 1))
        set_num_threads(min(_DEFAULT_MOTION_KERNEL_THREADS, available_threads))
    _runtime_configured = True


@njit(inline="always")
def _quat_error_squared(
    q1_w: float,
    q1_x: float,
    q1_y: float,
    q1_z: float,
    q2_w: float,
    q2_x: float,
    q2_y: float,
    q2_z: float,
) -> float:
    """Return the squared shortest-path angular error for two quaternions."""
    rel_w = abs(q2_w * q1_w + q2_x * q1_x + q2_y * q1_y + q2_z * q1_z)
    rel_x = -q2_w * q1_x + q2_x * q1_w - q2_y * q1_z + q2_z * q1_y
    rel_y = -q2_w * q1_y + q2_x * q1_z + q2_y * q1_w - q2_z * q1_x
    rel_z = -q2_w * q1_z - q2_x * q1_y + q2_y * q1_x + q2_z * q1_w
    xyz_norm = sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z)
    clipped_w = min(max(rel_w, -1.0), 1.0)
    angle = 2.0 * atan2(xyz_norm, clipped_w)
    return angle * angle


@njit(inline="always")
def _quat_mul_components(
    w1: float,
    x1: float,
    y1: float,
    z1: float,
    w2: float,
    x2: float,
    y2: float,
    z2: float,
) -> tuple[float, float, float, float]:
    """Hamilton product for scalar quaternion components."""
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


@njit(inline="always")
def _quat_apply_inverse_components(
    w: float,
    x: float,
    y: float,
    z: float,
    vx: float,
    vy: float,
    vz: float,
) -> tuple[float, float, float]:
    """Rotate one vector by the inverse of a unit quaternion."""
    qx = -x
    qy = -y
    qz = -z
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + w * tx + qy * tz - qz * ty,
        vy + w * ty + qz * tx - qx * tz,
        vz + w * tz + qx * ty - qy * tx,
    )


@njit(inline="always")
def _quat_to_rot6d_components(
    w: float,
    x: float,
    y: float,
    z: float,
) -> tuple[float, float, float, float, float, float]:
    """Flatten the first two rotation-matrix columns."""
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return (
        1.0 - 2.0 * (yy + zz),
        2.0 * (xy - wz),
        2.0 * (xy + wz),
        1.0 - 2.0 * (xx + zz),
        2.0 * (xz - wy),
        2.0 * (yz + wx),
    )


@njit(cache=True, nogil=True, parallel=True)
def update_motion_relative_state_kernel(
    env_ids: np.ndarray,
    anchor_body_idx: int,
    motion_body_pos_local_w: np.ndarray,
    motion_body_pos_w: np.ndarray,
    motion_body_quat_w: np.ndarray,
    robot_body_pos_w: np.ndarray,
    robot_body_quat_w: np.ndarray,
    body_pos_relative_w: np.ndarray,
    body_quat_relative_w: np.ndarray,
    motion_anchor_pos_b: np.ndarray,
    motion_anchor_ori_b: np.ndarray,
    robot_body_pos_b: np.ndarray,
    robot_body_ori_b: np.ndarray,
) -> None:
    """Write all MotionCommand relative transforms for selected rows."""
    num_bodies = motion_body_pos_local_w.shape[1]
    for row in prange(env_ids.shape[0]):
        env_idx = env_ids[row]
        motion_anchor_x = motion_body_pos_local_w[env_idx, anchor_body_idx, 0]
        motion_anchor_y = motion_body_pos_local_w[env_idx, anchor_body_idx, 1]
        motion_anchor_z = motion_body_pos_local_w[env_idx, anchor_body_idx, 2]
        motion_anchor_w = motion_body_quat_w[env_idx, anchor_body_idx, 0]
        motion_anchor_qx = motion_body_quat_w[env_idx, anchor_body_idx, 1]
        motion_anchor_qy = motion_body_quat_w[env_idx, anchor_body_idx, 2]
        motion_anchor_qz = motion_body_quat_w[env_idx, anchor_body_idx, 3]
        robot_anchor_x = robot_body_pos_w[env_idx, anchor_body_idx, 0]
        robot_anchor_y = robot_body_pos_w[env_idx, anchor_body_idx, 1]
        robot_anchor_z = robot_body_pos_w[env_idx, anchor_body_idx, 2]
        robot_anchor_w = robot_body_quat_w[env_idx, anchor_body_idx, 0]
        robot_anchor_qx = robot_body_quat_w[env_idx, anchor_body_idx, 1]
        robot_anchor_qy = robot_body_quat_w[env_idx, anchor_body_idx, 2]
        robot_anchor_qz = robot_body_quat_w[env_idx, anchor_body_idx, 3]

        yaw_w, yaw_x, yaw_y, yaw_z = _quat_mul_components(
            robot_anchor_w,
            robot_anchor_qx,
            robot_anchor_qy,
            robot_anchor_qz,
            motion_anchor_w,
            -motion_anchor_qx,
            -motion_anchor_qy,
            -motion_anchor_qz,
        )
        half_yaw = 0.5 * atan2(
            2.0 * (yaw_w * yaw_z + yaw_x * yaw_y),
            1.0 - 2.0 * (yaw_y * yaw_y + yaw_z * yaw_z),
        )
        delta_w = cos(half_yaw)
        delta_z = sin(half_yaw)
        yaw_cross = 2.0 * delta_w * delta_z
        yaw_z2 = 2.0 * delta_z * delta_z

        anchor_vx = motion_body_pos_w[env_idx, anchor_body_idx, 0] - robot_anchor_x
        anchor_vy = motion_body_pos_w[env_idx, anchor_body_idx, 1] - robot_anchor_y
        anchor_vz = motion_body_pos_w[env_idx, anchor_body_idx, 2] - robot_anchor_z
        anchor_pos_x, anchor_pos_y, anchor_pos_z = _quat_apply_inverse_components(
            robot_anchor_w,
            robot_anchor_qx,
            robot_anchor_qy,
            robot_anchor_qz,
            anchor_vx,
            anchor_vy,
            anchor_vz,
        )
        motion_anchor_pos_b[env_idx, 0] = anchor_pos_x
        motion_anchor_pos_b[env_idx, 1] = anchor_pos_y
        motion_anchor_pos_b[env_idx, 2] = anchor_pos_z
        rel_w, rel_x, rel_y, rel_z = _quat_mul_components(
            robot_anchor_w,
            -robot_anchor_qx,
            -robot_anchor_qy,
            -robot_anchor_qz,
            motion_anchor_w,
            motion_anchor_qx,
            motion_anchor_qy,
            motion_anchor_qz,
        )
        anchor_rot6d = _quat_to_rot6d_components(rel_w, rel_x, rel_y, rel_z)
        for component in range(6):
            motion_anchor_ori_b[env_idx, component] = anchor_rot6d[component]

        for body_idx in range(num_bodies):
            body_motion_w = motion_body_quat_w[env_idx, body_idx, 0]
            body_motion_x = motion_body_quat_w[env_idx, body_idx, 1]
            body_motion_y = motion_body_quat_w[env_idx, body_idx, 2]
            body_motion_z = motion_body_quat_w[env_idx, body_idx, 3]
            out_w, out_x, out_y, out_z = _quat_mul_components(
                delta_w,
                0.0,
                0.0,
                delta_z,
                body_motion_w,
                body_motion_x,
                body_motion_y,
                body_motion_z,
            )
            body_quat_relative_w[env_idx, body_idx, 0] = out_w
            body_quat_relative_w[env_idx, body_idx, 1] = out_x
            body_quat_relative_w[env_idx, body_idx, 2] = out_y
            body_quat_relative_w[env_idx, body_idx, 3] = out_z

            vx = motion_body_pos_local_w[env_idx, body_idx, 0] - motion_anchor_x
            vy = motion_body_pos_local_w[env_idx, body_idx, 1] - motion_anchor_y
            vz = motion_body_pos_local_w[env_idx, body_idx, 2] - motion_anchor_z
            relative_x = vx
            relative_x -= yaw_cross * vy
            relative_x -= yaw_z2 * vx
            relative_x += robot_anchor_x
            relative_y = vy
            relative_y += yaw_cross * vx
            relative_y -= yaw_z2 * vy
            relative_y += robot_anchor_y
            body_pos_relative_w[env_idx, body_idx, 0] = relative_x
            body_pos_relative_w[env_idx, body_idx, 1] = relative_y
            body_pos_relative_w[env_idx, body_idx, 2] = vz + motion_anchor_z

            robot_vx = robot_body_pos_w[env_idx, body_idx, 0] - robot_anchor_x
            robot_vy = robot_body_pos_w[env_idx, body_idx, 1] - robot_anchor_y
            robot_vz = robot_body_pos_w[env_idx, body_idx, 2] - robot_anchor_z
            robot_pos_x, robot_pos_y, robot_pos_z = _quat_apply_inverse_components(
                robot_anchor_w,
                robot_anchor_qx,
                robot_anchor_qy,
                robot_anchor_qz,
                robot_vx,
                robot_vy,
                robot_vz,
            )
            robot_body_pos_b[env_idx, body_idx, 0] = robot_pos_x
            robot_body_pos_b[env_idx, body_idx, 1] = robot_pos_y
            robot_body_pos_b[env_idx, body_idx, 2] = robot_pos_z
            body_robot_w = robot_body_quat_w[env_idx, body_idx, 0]
            body_robot_x = robot_body_quat_w[env_idx, body_idx, 1]
            body_robot_y = robot_body_quat_w[env_idx, body_idx, 2]
            body_robot_z = robot_body_quat_w[env_idx, body_idx, 3]
            robot_rel_w, robot_rel_x, robot_rel_y, robot_rel_z = _quat_mul_components(
                robot_anchor_w,
                -robot_anchor_qx,
                -robot_anchor_qy,
                -robot_anchor_qz,
                body_robot_w,
                body_robot_x,
                body_robot_y,
                body_robot_z,
            )
            robot_rot6d = _quat_to_rot6d_components(
                robot_rel_w,
                robot_rel_x,
                robot_rel_y,
                robot_rel_z,
            )
            for component in range(6):
                robot_body_ori_b[env_idx, body_idx, component] = robot_rot6d[component]


@njit(cache=True, nogil=True, parallel=True)
def update_object_relative_state_kernel(
    env_ids: np.ndarray,
    robot_anchor_pos_w: np.ndarray,
    robot_anchor_quat_w: np.ndarray,
    object_pos_w: np.ndarray,
    object_quat_w: np.ndarray,
    object_lin_vel_w: np.ndarray,
    object_state_b: np.ndarray,
) -> None:
    """Write BoxMotionCommand object pose and velocity for selected rows."""
    for row in prange(env_ids.shape[0]):
        env_idx = env_ids[row]
        anchor_x = robot_anchor_pos_w[env_idx, 0]
        anchor_y = robot_anchor_pos_w[env_idx, 1]
        anchor_z = robot_anchor_pos_w[env_idx, 2]
        anchor_w = robot_anchor_quat_w[env_idx, 0]
        anchor_qx = robot_anchor_quat_w[env_idx, 1]
        anchor_qy = robot_anchor_quat_w[env_idx, 2]
        anchor_qz = robot_anchor_quat_w[env_idx, 3]
        pos_x, pos_y, pos_z = _quat_apply_inverse_components(
            anchor_w,
            anchor_qx,
            anchor_qy,
            anchor_qz,
            object_pos_w[env_idx, 0] - anchor_x,
            object_pos_w[env_idx, 1] - anchor_y,
            object_pos_w[env_idx, 2] - anchor_z,
        )
        object_state_b[env_idx, 0] = pos_x
        object_state_b[env_idx, 1] = pos_y
        object_state_b[env_idx, 2] = pos_z
        rel_w, rel_x, rel_y, rel_z = _quat_mul_components(
            anchor_w,
            -anchor_qx,
            -anchor_qy,
            -anchor_qz,
            object_quat_w[env_idx, 0],
            object_quat_w[env_idx, 1],
            object_quat_w[env_idx, 2],
            object_quat_w[env_idx, 3],
        )
        rot6d = _quat_to_rot6d_components(rel_w, rel_x, rel_y, rel_z)
        for component in range(6):
            object_state_b[env_idx, 3 + component] = rot6d[component]
        vel_x, vel_y, vel_z = _quat_apply_inverse_components(
            anchor_w,
            anchor_qx,
            anchor_qy,
            anchor_qz,
            object_lin_vel_w[env_idx, 0],
            object_lin_vel_w[env_idx, 1],
            object_lin_vel_w[env_idx, 2],
        )
        object_state_b[env_idx, 9] = vel_x
        object_state_b[env_idx, 10] = vel_y
        object_state_b[env_idx, 11] = vel_z


@njit(cache=True, nogil=True, parallel=True)
def update_motion_metrics_kernel(
    env_ids: np.ndarray,
    anchor_body_idx: int,
    motion_body_pos_w: np.ndarray,
    robot_body_pos_w: np.ndarray,
    motion_body_quat_w: np.ndarray,
    robot_body_quat_w: np.ndarray,
    motion_body_lin_vel_w: np.ndarray,
    robot_body_lin_vel_w: np.ndarray,
    motion_body_ang_vel_w: np.ndarray,
    robot_body_ang_vel_w: np.ndarray,
    body_pos_relative_w: np.ndarray,
    body_quat_relative_w: np.ndarray,
    motion_joint_pos: np.ndarray,
    robot_joint_pos: np.ndarray,
    motion_joint_vel: np.ndarray,
    robot_joint_vel: np.ndarray,
    error_anchor_pos: np.ndarray,
    error_anchor_rot: np.ndarray,
    error_anchor_lin_vel: np.ndarray,
    error_anchor_ang_vel: np.ndarray,
    error_body_pos: np.ndarray,
    error_body_rot: np.ndarray,
    error_body_lin_vel: np.ndarray,
    error_body_ang_vel: np.ndarray,
    error_joint_pos: np.ndarray,
    error_joint_vel: np.ndarray,
) -> None:
    """Write all MotionCommand error metrics for the selected environment rows.

    ``env_ids`` is a cold-path-owned all-row index buffer for normal steps and
    the reset row list for partial resets.  Keeping one kernel for both paths
    avoids a second production mathematical implementation while preserving the
    manager's row-scoped reset contract.
    """
    num_bodies = motion_body_pos_w.shape[1]
    num_joints = motion_joint_pos.shape[1]
    for row in prange(env_ids.shape[0]):
        env_idx = env_ids[row]
        anchor_dx = (
            motion_body_pos_w[env_idx, anchor_body_idx, 0]
            - robot_body_pos_w[env_idx, anchor_body_idx, 0]
        )
        anchor_dy = (
            motion_body_pos_w[env_idx, anchor_body_idx, 1]
            - robot_body_pos_w[env_idx, anchor_body_idx, 1]
        )
        anchor_dz = (
            motion_body_pos_w[env_idx, anchor_body_idx, 2]
            - robot_body_pos_w[env_idx, anchor_body_idx, 2]
        )
        error_anchor_pos[env_idx] = sqrt(
            anchor_dx * anchor_dx + anchor_dy * anchor_dy + anchor_dz * anchor_dz
        )
        error_anchor_rot[env_idx] = sqrt(
            _quat_error_squared(
                motion_body_quat_w[env_idx, anchor_body_idx, 0],
                motion_body_quat_w[env_idx, anchor_body_idx, 1],
                motion_body_quat_w[env_idx, anchor_body_idx, 2],
                motion_body_quat_w[env_idx, anchor_body_idx, 3],
                robot_body_quat_w[env_idx, anchor_body_idx, 0],
                robot_body_quat_w[env_idx, anchor_body_idx, 1],
                robot_body_quat_w[env_idx, anchor_body_idx, 2],
                robot_body_quat_w[env_idx, anchor_body_idx, 3],
            )
        )

        anchor_lin_sq = 0.0
        anchor_ang_sq = 0.0
        for component in range(3):
            lin_delta = (
                motion_body_lin_vel_w[env_idx, anchor_body_idx, component]
                - robot_body_lin_vel_w[env_idx, anchor_body_idx, component]
            )
            ang_delta = (
                motion_body_ang_vel_w[env_idx, anchor_body_idx, component]
                - robot_body_ang_vel_w[env_idx, anchor_body_idx, component]
            )
            anchor_lin_sq += lin_delta * lin_delta
            anchor_ang_sq += ang_delta * ang_delta
        error_anchor_lin_vel[env_idx] = sqrt(anchor_lin_sq)
        error_anchor_ang_vel[env_idx] = sqrt(anchor_ang_sq)

        body_pos_sum = 0.0
        body_rot_sum = 0.0
        body_lin_sum = 0.0
        body_ang_sum = 0.0
        for body_idx in range(num_bodies):
            pos_sq = 0.0
            lin_sq = 0.0
            ang_sq = 0.0
            for component in range(3):
                pos_delta = (
                    body_pos_relative_w[env_idx, body_idx, component]
                    - robot_body_pos_w[env_idx, body_idx, component]
                )
                lin_delta = (
                    motion_body_lin_vel_w[env_idx, body_idx, component]
                    - robot_body_lin_vel_w[env_idx, body_idx, component]
                )
                ang_delta = (
                    motion_body_ang_vel_w[env_idx, body_idx, component]
                    - robot_body_ang_vel_w[env_idx, body_idx, component]
                )
                pos_sq += pos_delta * pos_delta
                lin_sq += lin_delta * lin_delta
                ang_sq += ang_delta * ang_delta
            body_pos_sum += sqrt(pos_sq)
            body_lin_sum += sqrt(lin_sq)
            body_ang_sum += sqrt(ang_sq)
            body_rot_sum += sqrt(
                _quat_error_squared(
                    body_quat_relative_w[env_idx, body_idx, 0],
                    body_quat_relative_w[env_idx, body_idx, 1],
                    body_quat_relative_w[env_idx, body_idx, 2],
                    body_quat_relative_w[env_idx, body_idx, 3],
                    robot_body_quat_w[env_idx, body_idx, 0],
                    robot_body_quat_w[env_idx, body_idx, 1],
                    robot_body_quat_w[env_idx, body_idx, 2],
                    robot_body_quat_w[env_idx, body_idx, 3],
                )
            )
        if num_bodies == 0:
            error_body_pos[env_idx] = np.nan
            error_body_rot[env_idx] = np.nan
            error_body_lin_vel[env_idx] = np.nan
            error_body_ang_vel[env_idx] = np.nan
        else:
            inv_bodies = 1.0 / num_bodies
            error_body_pos[env_idx] = body_pos_sum * inv_bodies
            error_body_rot[env_idx] = body_rot_sum * inv_bodies
            error_body_lin_vel[env_idx] = body_lin_sum * inv_bodies
            error_body_ang_vel[env_idx] = body_ang_sum * inv_bodies

        joint_pos_sq = 0.0
        joint_vel_sq = 0.0
        for joint_idx in range(num_joints):
            pos_delta = motion_joint_pos[env_idx, joint_idx] - robot_joint_pos[env_idx, joint_idx]
            vel_delta = motion_joint_vel[env_idx, joint_idx] - robot_joint_vel[env_idx, joint_idx]
            joint_pos_sq += pos_delta * pos_delta
            joint_vel_sq += vel_delta * vel_delta
        error_joint_pos[env_idx] = sqrt(joint_pos_sq)
        error_joint_vel[env_idx] = sqrt(joint_vel_sq)


@njit(cache=True, nogil=True, parallel=True)
def termination_anchor_pos_kernel(
    motion_body_pos_w: np.ndarray,
    robot_body_pos_w: np.ndarray,
    anchor_body_idx: int,
    threshold: float,
    out: np.ndarray,
) -> None:
    """Write the per-environment anchor-height termination mask."""
    for env_idx in prange(motion_body_pos_w.shape[0]):
        error = abs(
            motion_body_pos_w[env_idx, anchor_body_idx, 2]
            - robot_body_pos_w[env_idx, anchor_body_idx, 2]
        )
        out[env_idx] = error > threshold


@njit(cache=True, nogil=True, parallel=True)
def reward_motion_body_pos_kernel(
    reference: np.ndarray,
    actual: np.ndarray,
    body_ids: np.ndarray,
    std: float,
    out: np.ndarray,
) -> None:
    """Write the relative body-position exponential reward."""
    num_bodies = body_ids.shape[0]
    if num_bodies == 0:
        out[:] = np.nan
        return
    denominator = -(num_bodies * std * std)
    for env_idx in prange(reference.shape[0]):
        error = reference[env_idx, body_ids[0], 0] * 0
        for body_offset in range(num_bodies):
            body_idx = body_ids[body_offset]
            dx = reference[env_idx, body_idx, 0] - actual[env_idx, body_idx, 0]
            dy = reference[env_idx, body_idx, 1] - actual[env_idx, body_idx, 1]
            dz = reference[env_idx, body_idx, 2] - actual[env_idx, body_idx, 2]
            error += dx * dx + dy * dy + dz * dz
        out[env_idx] = np.exp(error / denominator)


@njit(cache=True, nogil=True, parallel=True)
def reward_motion_body_ori_kernel(
    reference: np.ndarray,
    actual: np.ndarray,
    body_ids: np.ndarray,
    std: float,
    out: np.ndarray,
) -> None:
    """Write the relative body-orientation exponential reward."""
    num_bodies = body_ids.shape[0]
    if num_bodies == 0:
        out[:] = np.nan
        return
    denominator = -(num_bodies * std * std)
    for env_idx in prange(reference.shape[0]):
        error = reference[env_idx, body_ids[0], 0] * 0
        for body_offset in range(num_bodies):
            body_idx = body_ids[body_offset]
            w1 = reference[env_idx, body_idx, 0]
            x1 = reference[env_idx, body_idx, 1]
            y1 = reference[env_idx, body_idx, 2]
            z1 = reference[env_idx, body_idx, 3]
            w2 = actual[env_idx, body_idx, 0]
            x2 = actual[env_idx, body_idx, 1]
            y2 = actual[env_idx, body_idx, 2]
            z2 = actual[env_idx, body_idx, 3]

            error += _quat_error_squared(w1, x1, y1, z1, w2, x2, y2, z2)
        out[env_idx] = np.exp(error / denominator)


@njit(cache=True, nogil=True, parallel=True)
def reward_motion_body_lin_vel_kernel(
    reference: np.ndarray,
    actual: np.ndarray,
    body_ids: np.ndarray,
    std: float,
    out: np.ndarray,
) -> None:
    """Write the global body-linear-velocity exponential reward."""
    num_bodies = body_ids.shape[0]
    if num_bodies == 0:
        out[:] = np.nan
        return
    denominator = -(num_bodies * std * std)
    for env_idx in prange(reference.shape[0]):
        error = reference[env_idx, body_ids[0], 0] * 0
        for body_offset in range(num_bodies):
            body_idx = body_ids[body_offset]
            dx = reference[env_idx, body_idx, 0] - actual[env_idx, body_idx, 0]
            dy = reference[env_idx, body_idx, 1] - actual[env_idx, body_idx, 1]
            dz = reference[env_idx, body_idx, 2] - actual[env_idx, body_idx, 2]
            error += dx * dx + dy * dy + dz * dz
        out[env_idx] = np.exp(error / denominator)


@njit(cache=True, nogil=True, parallel=True)
def reward_motion_body_ang_vel_kernel(
    reference: np.ndarray,
    actual: np.ndarray,
    body_ids: np.ndarray,
    std: float,
    out: np.ndarray,
) -> None:
    """Write the global body-angular-velocity exponential reward."""
    num_bodies = body_ids.shape[0]
    if num_bodies == 0:
        out[:] = np.nan
        return
    denominator = -(num_bodies * std * std)
    for env_idx in prange(reference.shape[0]):
        error = reference[env_idx, body_ids[0], 0] * 0
        for body_offset in range(num_bodies):
            body_idx = body_ids[body_offset]
            dx = reference[env_idx, body_idx, 0] - actual[env_idx, body_idx, 0]
            dy = reference[env_idx, body_idx, 1] - actual[env_idx, body_idx, 1]
            dz = reference[env_idx, body_idx, 2] - actual[env_idx, body_idx, 2]
            error += dx * dx + dy * dy + dz * dz
        out[env_idx] = np.exp(error / denominator)


__all__ = [
    "configure_motion_kernel_runtime",
    "reward_motion_body_ang_vel_kernel",
    "reward_motion_body_lin_vel_kernel",
    "reward_motion_body_ori_kernel",
    "reward_motion_body_pos_kernel",
    "termination_anchor_pos_kernel",
    "update_motion_metrics_kernel",
    "update_motion_relative_state_kernel",
    "update_object_relative_state_kernel",
]
