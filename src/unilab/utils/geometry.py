"""Standalone numpy helpers for coordinate frames and geometry.

Reusable pure-numpy helpers for rotations, coordinate-frame transforms,
and geometric conversions. Kept dtype-agnostic and side-effect-free so
that env / task / backend code can compose them without carrying task
policy inside a shared module. Complements the vectorized quaternion
primitives in :mod:`unilab.utils.rotation`.
"""

from __future__ import annotations

import numpy as np

from unilab.utils.rotation import (
    np_quat_canonicalize,
    np_quat_conjugate,
    np_quat_inv,
    np_quat_mul,
    np_quat_to_axis_angle,
)


def np_normalize_axis(axis: np.ndarray | tuple[float, ...] | list[float]) -> np.ndarray:
    """Return a unit-length copy of a rotation axis vector. Raises on zero norm."""
    axis = np.asarray(axis)
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        raise ValueError(f"axis must be non-zero, got {axis!r}")
    return axis / norm


def np_roll_pitch_from_quat(quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Roll and pitch (rad) from a w-first quaternion, computed via rotation-matrix rows.

    Supports either ``(4,)`` or ``(..., 4)`` inputs. Returned arrays match
    the caller's leading shape and dtype (no upcast).
    """
    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(r21, r22)
    pitch = np.arctan2(-r20, np.sqrt(np.clip(r21 * r21 + r22 * r22, 0.0, None)))
    return roll, pitch


def np_quat_angular_velocity_from_pair(
    quat: np.ndarray, prev_quat: np.ndarray, dt: float
) -> np.ndarray:
    """Angular velocity from two consecutive quaternions via axis-angle diff / dt."""
    rel = np_quat_mul(quat, np_quat_conjugate(prev_quat))
    return np_quat_to_axis_angle(rel) / dt
