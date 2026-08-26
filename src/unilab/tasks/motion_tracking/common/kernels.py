"""Parallel CPU kernels for the fixed motion-tracking hot-term set."""

from __future__ import annotations

import os
from math import atan2, sqrt

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
]
