"""Rotation and quaternion math primitives (ADR-0022).

unisim is the sole owner of backend-neutral math primitives; this package is
a thin re-export so UniLab consumers share one import location.  Quaternions
are ``wxyz``, Euler angles are XYZ, and batched inputs broadcast with
trailing ``(..., 4)`` / ``(..., 3)`` dimensions.  ``rotation_numpy`` holds
the NumPy primitives and ``rotation_tensor`` the TorchScript-compiled Torch
counterparts; the module path selects the backend and function names carry
no backend prefix.
"""

from __future__ import annotations

from . import rotation_numpy, rotation_tensor

__all__ = ["rotation_numpy", "rotation_tensor"]
