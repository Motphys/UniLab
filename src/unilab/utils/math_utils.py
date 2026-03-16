from __future__ import annotations

import importlib

import numpy as np


def _require_mlx_core():
    """Import MLX lazily so non-MLX workflows don't crash at module import time."""
    try:
        return importlib.import_module("mlx.core")
    except Exception as exc:
        raise RuntimeError(
            "MLX backend is unavailable. Use NumPy helpers (np_quat_mul/np_yaw_to_quat) in non-MLX paths."
        ) from exc


def quat_mul(q1, q2):
    """
    Multiply two quaternions.
    """
    mx = _require_mlx_core()
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return mx.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=1,
    )


def axis_angle_to_quat(axis, angle):
    """
    Convert axis-angle to quaternion.
    """
    mx = _require_mlx_core()
    half_angle = angle / 2
    c = mx.cos(half_angle)
    s = mx.sin(half_angle)
    return mx.stack([c, axis[:, 0] * s, axis[:, 1] * s, axis[:, 2] * s], axis=1)


def np_quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternion batches in NumPy, shape (N, 4)."""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=1,
    )


def np_quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of a batch of unit quaternions (N, 4), w-first."""
    conj = q.copy()
    conj[:, 1:] *= -1
    return conj  # type: ignore[no-any-return]


def np_quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion batch (N, 4), w-first, to axis-angle vectors (N, 3).

    Adapted from PyTorch3D. Uses atan2 + Taylor expansion for numerical
    stability near zero rotation.
    """
    xyz = q[:, 1:]  # (N, 3) imaginary part
    w = q[:, 0:1]  # (N, 1) real part
    norms = np.linalg.norm(xyz, axis=-1, keepdims=True)  # (N, 1)
    half_angle = np.arctan2(norms, w)  # (N, 1)
    angle = 2.0 * half_angle  # (N, 1)
    small = np.abs(angle) < 1e-6  # (N, 1)
    safe_angle = np.where(small, 1.0, angle)
    sin_half_over_angle = np.where(
        small,
        0.5 - angle**2 / 48.0,
        np.sin(half_angle) / safe_angle,
    )
    return np.asarray(xyz / sin_half_over_angle)  # type: ignore[no-any-return]


def np_yaw_to_quat(yaw: np.ndarray) -> np.ndarray:
    """Convert yaw batch (N,) to quaternion batch (N, 4) in NumPy."""
    half = 0.5 * yaw
    return np.stack(
        [
            np.cos(half),
            np.zeros_like(half),
            np.zeros_like(half),
            np.sin(half),
        ],
        axis=1,
    )
