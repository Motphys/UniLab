from __future__ import annotations

import importlib


def _require_mlx_core():
    """Import MLX lazily so non-MLX workflows don't crash at module import time."""
    try:
        return importlib.import_module("mlx.core")
    except Exception as exc:
        raise RuntimeError(
            "MLX backend is unavailable. Install the MLX extra to use MLX rotation helpers."
        ) from exc


def quat_mul(q1, q2):
    """Multiply two MLX quaternion batches."""
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
    """Convert MLX axis-angle batches to quaternions."""
    mx = _require_mlx_core()
    half_angle = angle / 2
    c = mx.cos(half_angle)
    s = mx.sin(half_angle)
    return mx.stack([c, axis[:, 0] * s, axis[:, 1] * s, axis[:, 2] * s], axis=1)
