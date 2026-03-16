"""Math utilities for motion tracking - quaternion and rotation operations."""

from __future__ import annotations

import numpy as np


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (wxyz format).

    Args:
        q1: First quaternion(s) (N, 4) or (4,)
        q2: Second quaternion(s) (N, 4) or (4,)

    Returns:
        Product quaternion(s) (N, 4) or (4,)
    """
    if q1.ndim == 1:
        q1 = q1[None, :]
        q2 = q2[None, :]
        squeeze = True
    else:
        squeeze = False

    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    result = np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=1,
    )

    return result[0] if squeeze else result


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Compute quaternion conjugate (wxyz format).

    Args:
        q: Quaternion(s) (N, 4) or (4,)

    Returns:
        Conjugate quaternion(s) (N, 4) or (4,)
    """
    if q.ndim == 1:
        return np.array([q[0], -q[1], -q[2], -q[3]])
    else:
        result = q.copy()
        result[:, 1:] *= -1
        return result


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Compute quaternion inverse (wxyz format).

    For unit quaternions, inverse equals conjugate.

    Args:
        q: Quaternion(s) (N, 4) or (4,)

    Returns:
        Inverse quaternion(s) (N, 4) or (4,)
    """
    return quat_conjugate(q)


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) by quaternion(s) (wxyz format).

    Args:
        q: Quaternion(s) (N, 4) or (4,)
        v: Vector(s) (N, 3) or (3,)

    Returns:
        Rotated vector(s) (N, 3) or (3,)
    """
    if q.ndim == 1:
        q = q[None, :]
        v = v[None, :]
        squeeze = True
    else:
        squeeze = False

    # Convert to (w, x, y, z)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]

    # Compute rotation using quaternion formula
    # v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
    t = 2 * np.stack(
        [
            y * vz - z * vy,
            z * vx - x * vz,
            x * vy - y * vx,
        ],
        axis=1,
    )

    t += 2 * w[:, None] * v

    result = v + np.stack(
        [
            y * t[:, 2] - z * t[:, 1],
            z * t[:, 0] - x * t[:, 2],
            x * t[:, 1] - y * t[:, 0],
        ],
        axis=1,
    )

    return result[0] if squeeze else result


def quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) by inverse quaternion(s) (wxyz format).

    Args:
        q: Quaternion(s) (N, 4) or (4,)
        v: Vector(s) (N, 3) or (3,)

    Returns:
        Rotated vector(s) (N, 3) or (3,)
    """
    return quat_apply(quat_inv(q), v)


def quat_error_magnitude(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Compute angular error magnitude between quaternions.

    Args:
        q1: First quaternion(s) (N, 4) or (4,)
        q2: Second quaternion(s) (N, 4) or (4,)

    Returns:
        Error magnitude(s) (N,) or scalar
    """
    if q1.ndim == 1:
        q1 = q1[None, :]
        q2 = q2[None, :]
        squeeze = True
    else:
        squeeze = False

    # Compute relative quaternion
    q_rel = quat_mul(q2, quat_inv(q1))

    # Error magnitude is 2 * arcsin(||xyz||)
    xyz_norm = np.linalg.norm(q_rel[:, 1:], axis=1)
    error = 2 * np.arcsin(np.clip(xyz_norm, -1, 1))

    return error[0] if squeeze else error


def quat_from_euler_xyz(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Convert Euler angles (XYZ order) to quaternion (wxyz format).

    Args:
        roll: Roll angle(s) (N,) or scalar
        pitch: Pitch angle(s) (N,) or scalar
        yaw: Yaw angle(s) (N,) or scalar

    Returns:
        Quaternion(s) (N, 4) or (4,)
    """
    roll = np.atleast_1d(roll)
    pitch = np.atleast_1d(pitch)
    yaw = np.atleast_1d(yaw)
    squeeze = roll.shape[0] == 1

    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    result = np.stack([w, x, y, z], axis=1)
    return result[0] if squeeze else result


def yaw_quat(q: np.ndarray) -> np.ndarray:
    """Extract yaw-only quaternion from full quaternion (wxyz format).

    Args:
        q: Quaternion(s) (N, 4) or (4,)

    Returns:
        Yaw-only quaternion(s) (N, 4) or (4,)
    """
    if q.ndim == 1:
        q = q[None, :]
        squeeze = True
    else:
        squeeze = False

    # Extract yaw angle
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    # Create yaw-only quaternion
    half_yaw = yaw * 0.5
    result = np.stack(
        [
            np.cos(half_yaw),
            np.zeros_like(half_yaw),
            np.zeros_like(half_yaw),
            np.sin(half_yaw),
        ],
        axis=1,
    )

    return result[0] if squeeze else result


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion(s) to rotation matrix (wxyz format).

    Args:
        q: Quaternion(s) (N, 4) or (4,)

    Returns:
        Rotation matrix (N, 3, 3) or (3, 3)
    """
    if q.ndim == 1:
        q = q[None, :]
        squeeze = True
    else:
        squeeze = False

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Compute rotation matrix elements
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    result = np.stack(
        [
            np.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], axis=1),
            np.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], axis=1),
            np.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], axis=1),
        ],
        axis=1,
    )

    return result[0] if squeeze else result


def subtract_frame_transforms(
    pos1: np.ndarray, quat1: np.ndarray, pos2: np.ndarray, quat2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute relative transform from frame 1 to frame 2.

    Args:
        pos1: Position of frame 1 (N, 3) or (3,)
        quat1: Orientation of frame 1 (N, 4) or (4,) in wxyz format
        pos2: Position of frame 2 (N, 3) or (3,)
        quat2: Orientation of frame 2 (N, 4) or (4,) in wxyz format

    Returns:
        Tuple of (relative_pos, relative_quat) in frame 1 coordinates
    """
    # Relative position in frame 1 coordinates
    rel_pos = quat_apply_inverse(quat1, pos2 - pos1)

    # Relative orientation
    rel_quat = quat_mul(quat_inv(quat1), quat2)

    return rel_pos, rel_quat


def sample_uniform(
    lower: float | np.ndarray,
    upper: float | np.ndarray,
    size: tuple[int, ...],
    dtype=np.float32,
) -> np.ndarray:
    """Sample uniformly from [lower, upper].

    Args:
        lower: Lower bound(s)
        upper: Upper bound(s)
        size: Output shape
        dtype: Output dtype

    Returns:
        Sampled values
    """
    return np.random.uniform(lower, upper, size).astype(dtype)
