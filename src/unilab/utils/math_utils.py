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
    """Multiply quaternions in NumPy, supports (N, 4) and (4,) inputs."""
    q1_was_1d = q1.ndim == 1
    q2_was_1d = q2.ndim == 1

    if q1_was_1d:
        q1 = q1[None, :]
    if q2_was_1d:
        q2 = q2[None, :]

    if q1.shape[0] == 1 and q2.shape[0] > 1:
        q1 = np.broadcast_to(q1, q2.shape)
    elif q2.shape[0] == 1 and q1.shape[0] > 1:
        q2 = np.broadcast_to(q2, q1.shape)

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
    return result[0] if q1_was_1d and q2_was_1d else result


def np_quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of unit quaternions (N, 4) or (4,), w-first."""
    if q.ndim == 1:
        return np.array([q[0], -q[1], -q[2], -q[3]])
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


def np_quat_inv(q: np.ndarray) -> np.ndarray:
    """Inverse of unit quaternions (N, 4) or (4,), w-first."""
    return np_quat_conjugate(q)


def np_quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) by quaternion(s), supports batched/scalar inputs."""
    q_was_1d = q.ndim == 1
    v_was_1d = v.ndim == 1

    if q_was_1d:
        q = q[None, :]
    if v_was_1d:
        v = v[None, :]

    if q.shape[0] == 1 and v.shape[0] > 1:
        q = np.broadcast_to(q, (v.shape[0], 4))
    elif v.shape[0] == 1 and q.shape[0] > 1:
        v = np.broadcast_to(v, (q.shape[0], 3))

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]

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

    return result[0] if q_was_1d and v_was_1d else result


def np_quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) by inverse quaternion(s)."""
    return np_quat_apply(np_quat_inv(q), v)


def np_quat_error_magnitude(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Angular error magnitude between quaternions (N,) or scalar."""
    q1_was_1d = q1.ndim == 1
    q2_was_1d = q2.ndim == 1

    if q1_was_1d:
        q1 = q1[None, :]
    if q2_was_1d:
        q2 = q2[None, :]

    if q1.shape[0] == 1 and q2.shape[0] > 1:
        q1 = np.broadcast_to(q1, q2.shape)
    elif q2.shape[0] == 1 and q1.shape[0] > 1:
        q2 = np.broadcast_to(q2, q1.shape)

    # Relative rotation from q1 to q2.
    q_rel = np_quat_mul(q2, np_quat_inv(q1))

    # Match mjlab shortest-arc convention:
    # q and -q represent the same orientation, so force w >= 0 to avoid
    # large-angle artifacts when quaternion signs flip.
    sign = np.where(q_rel[:, 0:1] < 0.0, -1.0, 1.0)
    q_rel = q_rel * sign

    # Use atan2-based angle extraction for better numerical behavior.
    xyz_norm = np.linalg.norm(q_rel[:, 1:], axis=1)
    w = np.clip(q_rel[:, 0], -1.0, 1.0)
    error = 2.0 * np.arctan2(xyz_norm, w)
    return error[0] if q1_was_1d and q2_was_1d else error


def np_quat_from_euler_xyz(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Convert Euler angles (XYZ) to quaternions (N, 4) or (4,), w-first."""
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


def np_yaw_quat(q: np.ndarray) -> np.ndarray:
    """Extract yaw-only quaternion from full quaternion(s), w-first."""
    q_was_1d = q.ndim == 1
    if q_was_1d:
        q = q[None, :]

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

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

    return result[0] if q_was_1d else result


def np_matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion(s) to rotation matrix (N, 3, 3) or (3, 3), w-first."""
    q_was_1d = q.ndim == 1
    if q_was_1d:
        q = q[None, :]

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

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

    return result[0] if q_was_1d else result


def np_subtract_frame_transforms(
    pos1: np.ndarray, quat1: np.ndarray, pos2: np.ndarray, quat2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute relative transform from frame 1 to frame 2 in frame-1 coordinates."""
    rel_pos = np_quat_apply_inverse(quat1, pos2 - pos1)
    rel_quat = np_quat_mul(np_quat_inv(quat1), quat2)
    return rel_pos, rel_quat


def np_sample_uniform(
    lower: float | np.ndarray,
    upper: float | np.ndarray,
    size: tuple[int, ...],
    dtype=np.float32,
) -> np.ndarray:
    """Sample uniformly from [lower, upper] with output dtype."""
    return np.random.uniform(lower, upper, size).astype(dtype)
