"""Parallel CPU kernels for the fixed motion-tracking hot-term set."""

from __future__ import annotations

import os

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

            # Relative rotation actual * conjugate(reference), matching
            # np_quat_error_magnitude_squared_batched without materializing arrays.
            rel_w = abs(w2 * w1 + x2 * x1 + y2 * y1 + z2 * z1)
            rel_x = -w2 * x1 + x2 * w1 - y2 * z1 + z2 * y1
            rel_y = -w2 * y1 + x2 * z1 + y2 * w1 - z2 * x1
            rel_z = -w2 * z1 - x2 * y1 + y2 * x1 + z2 * w1
            xyz_norm = np.sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z)
            clipped_w = min(max(rel_w, -1.0), 1.0)
            angle = 2.0 * np.arctan2(xyz_norm, clipped_w)
            error += angle * angle
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
]
