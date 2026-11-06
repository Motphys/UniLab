"""Rotation and quaternion math primitives re-exported from unisim (ADR-0022).

unisim is the sole owner of backend-neutral math primitives; this module is a
thin re-export so UniLab consumers share one import location.  Quaternions are
``wxyz``, Euler angles are XYZ, and batched inputs broadcast with trailing
``(..., 4)`` / ``(..., 3)`` dimensions.  Names without a prefix are the
TorchScript-compiled Torch operators; NumPy counterparts carry the ``np_``
prefix.

The Torch names resolve through ``unisim.utils.math.__getattr__`` (they are
compiled lazily), so they are forwarded as attribute assignments to stay
visible to static analysis.
"""

from __future__ import annotations

from unisim.utils import math as _unisim_math
from unisim.utils.math import (
    np_matrix_first_two_cols_from_quat,
    np_matrix_from_quat,
    np_quat_angular_velocity,
    np_quat_apply,
    np_quat_apply_inverse,
    np_quat_canonicalize,
    np_quat_conjugate,
    np_quat_ensure_continuity,
    np_quat_error_magnitude,
    np_quat_error_magnitude_squared,
    np_quat_from_angle_axis,
    np_quat_from_euler_xyz,
    np_quat_mul,
    np_quat_to_axis_angle,
    np_subtract_anchor_frame_transforms,
    np_subtract_frame_transforms,
    np_wrap_to_pi,
    np_yaw_from_quat,
    np_yaw_quat,
    np_yaw_to_quat,
)

axis_angle_from_quat = _unisim_math.axis_angle_from_quat
matrix_first_two_cols_from_quat = _unisim_math.matrix_first_two_cols_from_quat
matrix_from_quat = _unisim_math.matrix_from_quat
quat_angular_velocity = _unisim_math.quat_angular_velocity
quat_apply = _unisim_math.quat_apply
quat_apply_inverse = _unisim_math.quat_apply_inverse
quat_canonicalize = _unisim_math.quat_canonicalize
quat_conjugate = _unisim_math.quat_conjugate
quat_ensure_continuity = _unisim_math.quat_ensure_continuity
quat_error_magnitude = _unisim_math.quat_error_magnitude
quat_from_angle_axis = _unisim_math.quat_from_angle_axis
quat_from_euler_xyz = _unisim_math.quat_from_euler_xyz
quat_mul = _unisim_math.quat_mul
quat_roll_pitch = _unisim_math.quat_roll_pitch
quat_to_rot6 = _unisim_math.quat_to_rot6
wrap_to_pi = _unisim_math.wrap_to_pi
yaw_from_quat = _unisim_math.yaw_from_quat
yaw_quat = _unisim_math.yaw_quat
yaw_to_quat = _unisim_math.yaw_to_quat

__all__ = [
    "axis_angle_from_quat",
    "matrix_first_two_cols_from_quat",
    "matrix_from_quat",
    "np_matrix_first_two_cols_from_quat",
    "np_matrix_from_quat",
    "np_quat_angular_velocity",
    "np_quat_apply",
    "np_quat_apply_inverse",
    "np_quat_canonicalize",
    "np_quat_conjugate",
    "np_quat_ensure_continuity",
    "np_quat_error_magnitude",
    "np_quat_error_magnitude_squared",
    "np_quat_from_angle_axis",
    "np_quat_from_euler_xyz",
    "np_quat_mul",
    "np_quat_to_axis_angle",
    "np_subtract_anchor_frame_transforms",
    "np_subtract_frame_transforms",
    "np_wrap_to_pi",
    "np_yaw_from_quat",
    "np_yaw_quat",
    "np_yaw_to_quat",
    "quat_angular_velocity",
    "quat_apply",
    "quat_apply_inverse",
    "quat_canonicalize",
    "quat_conjugate",
    "quat_ensure_continuity",
    "quat_error_magnitude",
    "quat_from_angle_axis",
    "quat_from_euler_xyz",
    "quat_mul",
    "quat_roll_pitch",
    "quat_to_rot6",
    "wrap_to_pi",
    "yaw_from_quat",
    "yaw_quat",
    "yaw_to_quat",
]
