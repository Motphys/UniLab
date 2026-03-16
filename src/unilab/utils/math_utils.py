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
