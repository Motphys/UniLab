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


def np_sample_uniform(
    lower: float | np.ndarray,
    upper: float | np.ndarray,
    size: tuple[int, ...],
    dtype=np.float32,
) -> np.ndarray:
    """Sample uniformly from ``[lower, upper]`` and cast to ``dtype``."""
    return np.random.uniform(lower, upper, size).astype(dtype)


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


def np_gravity_z_in_body_from_quat(quat_w: np.ndarray) -> np.ndarray:
    """Z component of world gravity ``[0, 0, -1]`` expressed in body frame.

    Equivalent to ``np_quat_apply_inverse(quat_w, [0, 0, -1])[..., 2]`` but
    computed directly from quaternion components to skip the intermediate.
    """
    return 2.0 * (quat_w[..., 1] * quat_w[..., 1] + quat_w[..., 2] * quat_w[..., 2]) - 1.0


def np_quat_angular_velocity_from_pair(
    quat: np.ndarray, prev_quat: np.ndarray, dt: float
) -> np.ndarray:
    """Angular velocity from two consecutive quaternions via axis-angle diff / dt."""
    rel = np_quat_mul(quat, np_quat_conjugate(prev_quat))
    return np_quat_to_axis_angle(rel) / dt


def np_sample_uniform_quaternion(num_samples: int) -> np.ndarray:
    """Sample uniformly random unit quaternions (w-first) via Shoemake (1992).

    Returns an ``(num_samples, 4)`` array in float64 (leaves any downstream
    dtype conversion to the caller).
    """
    u1 = np.random.rand(num_samples)
    u2 = np.random.rand(num_samples) * 2.0 * np.pi
    u3 = np.random.rand(num_samples) * 2.0 * np.pi

    r1 = np.sqrt(1.0 - u1)
    r2 = np.sqrt(u1)
    q1 = r1 * np.sin(u2)
    q2 = r1 * np.cos(u2)
    q3 = r2 * np.sin(u3)
    q4 = r2 * np.cos(u3)

    return np.stack([q4, q1, q2, q3], axis=1)
