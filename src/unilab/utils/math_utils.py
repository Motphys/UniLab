from __future__ import annotations

import warnings

from unilab.algos.mlx.common.rotation import axis_angle_to_quat, quat_mul
from unilab.envs.common.math import np_sample_uniform
from unilab.envs.common.rotation import (
    np_matrix_from_quat,
    np_quat_angular_velocity,
    np_quat_apply,
    np_quat_apply_inverse,
    np_quat_canonicalize,
    np_quat_conjugate,
    np_quat_ensure_continuity,
    np_quat_error_magnitude,
    np_quat_from_euler_xyz,
    np_quat_inv,
    np_quat_mul,
    np_quat_to_axis_angle,
    np_subtract_frame_transforms,
    np_yaw_quat,
    np_yaw_to_quat,
)

warnings.warn(
    "`unilab.utils.math_utils` is deprecated and will be removed in 0.2.0; "
    "use `unilab.envs.common.rotation`, `unilab.envs.common.math`, or "
    "`unilab.algos.mlx.common.rotation` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "axis_angle_to_quat",
    "np_matrix_from_quat",
    "np_quat_angular_velocity",
    "np_quat_apply",
    "np_quat_apply_inverse",
    "np_quat_canonicalize",
    "np_quat_conjugate",
    "np_quat_ensure_continuity",
    "np_quat_error_magnitude",
    "np_quat_from_euler_xyz",
    "np_quat_inv",
    "np_quat_mul",
    "np_quat_to_axis_angle",
    "np_sample_uniform",
    "np_subtract_frame_transforms",
    "np_yaw_quat",
    "np_yaw_to_quat",
    "quat_mul",
]
