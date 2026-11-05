"""NumPy rotation primitives owned by :mod:`unisim.math` (ADR-0022)."""

from __future__ import annotations

from unisim.math.rotation_numpy import (
    matrix_first_two_cols_from_quat,
    matrix_from_quat,
    quat_angular_velocity,
    quat_apply,
    quat_apply_inverse,
    quat_canonicalize,
    quat_conjugate,
    quat_ensure_continuity,
    quat_error_magnitude,
    quat_error_magnitude_squared,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
    quat_to_axis_angle,
    subtract_anchor_frame_transforms,
    subtract_frame_transforms,
    wrap_to_pi,
    yaw_from_quat,
    yaw_quat,
    yaw_to_quat,
)

__all__ = [
    "matrix_first_two_cols_from_quat",
    "matrix_from_quat",
    "quat_angular_velocity",
    "quat_apply",
    "quat_apply_inverse",
    "quat_canonicalize",
    "quat_conjugate",
    "quat_ensure_continuity",
    "quat_error_magnitude",
    "quat_error_magnitude_squared",
    "quat_from_angle_axis",
    "quat_from_euler_xyz",
    "quat_mul",
    "quat_to_axis_angle",
    "subtract_anchor_frame_transforms",
    "subtract_frame_transforms",
    "wrap_to_pi",
    "yaw_from_quat",
    "yaw_quat",
    "yaw_to_quat",
]
