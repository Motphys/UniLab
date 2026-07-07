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


def np_quat_normalize(q: np.ndarray) -> np.ndarray:
    """L2-normalize quaternion(s), clamping tiny norms to 1e-8 to avoid divide-by-zero."""
    q = np.asarray(q)
    if q.ndim == 1:
        norm = float(np.linalg.norm(q))
        return q / max(norm, 1.0e-8)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(norm, 1.0e-8, None)


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


def np_quat_orientation_error_local(goal_quat: np.ndarray, curr_quat: np.ndarray) -> np.ndarray:
    """Local-frame orientation error as the signed xyz of the relative quaternion.

    Both inputs are re-normalized to unit length. Returns the imaginary
    part of ``goal * inv(curr)`` after canonicalization, giving a
    signed-axis-scaled error suitable for PD-style feedback terms.
    """
    goal = np_quat_normalize(goal_quat)
    curr = np_quat_normalize(curr_quat)
    if goal.ndim == 1:
        goal = goal[None, :]
    if curr.ndim == 1:
        curr = curr[None, :]
    rel = np_quat_mul(goal, np_quat_inv(curr))
    rel = np_quat_canonicalize(rel)
    sign = np.where(rel[:, 0:1] < 0.0, -1.0, 1.0)
    return rel[:, 1:] * sign


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


def np_spherical_to_cartesian(sphere: np.ndarray) -> np.ndarray:
    """Convert ``(..., 3)[l, phi, theta]`` to ``(..., 3)[x, y, z]``.

    Uses the go2-arm spherical convention: ``phi`` sweeps in the x-z plane
    from the positive x axis, and ``theta`` measures elevation toward
    positive y.
    """
    length = sphere[..., 0]
    phi = sphere[..., 1]
    theta = sphere[..., 2]
    x = length * np.cos(phi) * np.cos(theta)
    y = length * np.sin(theta)
    z = length * np.sin(phi) * np.cos(theta)
    return np.stack([x, y, z], axis=-1)


def np_cartesian_to_spherical(cart: np.ndarray) -> np.ndarray:
    """Convert ``(..., 3)[x, y, z]`` to ``(..., 3)[l, phi, theta]`` (inverse of
    :func:`np_spherical_to_cartesian`)."""
    cart = np.asarray(cart)
    l_sq = np.sum(cart**2, axis=-1, keepdims=True)
    length = np.sqrt(np.maximum(l_sq, 1e-12))
    phi = np.arctan2(cart[..., 2:3], cart[..., 0:1])
    theta = np.arcsin(np.clip(cart[..., 1:2] / length, -1.0, 1.0))
    return np.concatenate([length, phi, theta], axis=-1)


def np_write_relative_anchor_transform_pos_rot6d(
    source_anchor_pos_w: np.ndarray,
    source_anchor_quat_w: np.ndarray,
    target_anchor_pos_w: np.ndarray,
    target_anchor_quat_w: np.ndarray,
    out_pos: np.ndarray,
    out_rot6d: np.ndarray,
) -> None:
    """Fused frame-transform + 6D rotation flatten, writing in place.

    Computes the position of ``target_anchor`` in ``source_anchor``'s frame
    (written to ``out_pos`` of shape ``(N, 3)``) and the relative rotation
    ``conj(source_anchor) * target_anchor`` expressed as the flattened
    first two columns of its rotation matrix (written to ``out_rot6d`` of
    shape ``(N, 6)``). No intermediate quaternion arrays are allocated.

    This is the numpy analogue of the numba kernel
    :func:`unilab.utils.numba_geometry.write_relative_anchor_transform_at`.
    """
    aw = source_anchor_quat_w[:, 0]
    ax = source_anchor_quat_w[:, 1]
    ay = source_anchor_quat_w[:, 2]
    az = source_anchor_quat_w[:, 3]

    vx = target_anchor_pos_w[:, 0] - source_anchor_pos_w[:, 0]
    vy = target_anchor_pos_w[:, 1] - source_anchor_pos_w[:, 1]
    vz = target_anchor_pos_w[:, 2] - source_anchor_pos_w[:, 2]

    qx = -ax
    qy = -ay
    qz = -az
    tx = 2 * (qy * vz - qz * vy)
    ty = 2 * (qz * vx - qx * vz)
    tz = 2 * (qx * vy - qy * vx)
    out_pos[:, 0] = vx + aw * tx + qy * tz - qz * ty
    out_pos[:, 1] = vy + aw * ty + qz * tx - qx * tz
    out_pos[:, 2] = vz + aw * tz + qx * ty - qy * tx

    bw = target_anchor_quat_w[:, 0]
    bx = target_anchor_quat_w[:, 1]
    by = target_anchor_quat_w[:, 2]
    bz = target_anchor_quat_w[:, 3]
    rw = aw * bw + ax * bx + ay * by + az * bz
    rx = aw * bx - ax * bw - ay * bz + az * by
    ry = aw * by + ax * bz - ay * bw - az * bx
    rz = aw * bz - ax * by + ay * bx - az * bw

    xx = rx * rx
    yy = ry * ry
    zz = rz * rz
    xy = rx * ry
    xz = rx * rz
    yz = ry * rz
    wx = rw * rx
    wy = rw * ry
    wz = rw * rz
    out_rot6d[:, 0] = 1 - 2 * (yy + zz)
    out_rot6d[:, 1] = 2 * (xy - wz)
    out_rot6d[:, 2] = 2 * (xy + wz)
    out_rot6d[:, 3] = 1 - 2 * (xx + zz)
    out_rot6d[:, 4] = 2 * (xz - wy)
    out_rot6d[:, 5] = 2 * (yz + wx)
