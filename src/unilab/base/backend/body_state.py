"""Shared host-copy kernel for backend-owned body-state caches."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, nogil=True, parallel=True)
def copy_selected_body_state(
    source_pos: np.ndarray,
    source_quat: np.ndarray,
    source_lin_vel: np.ndarray,
    source_ang_vel: np.ndarray,
    selected_ids: np.ndarray,
    out_pos: np.ndarray,
    out_quat: np.ndarray,
    out_lin_vel: np.ndarray,
    out_ang_vel: np.ndarray,
) -> None:
    """Copy selected cache columns into four caller-owned state buffers."""
    for env_idx in prange(source_pos.shape[0]):
        for output_idx in range(selected_ids.shape[0]):
            source_idx = selected_ids[output_idx]
            out_pos[env_idx, output_idx, 0] = source_pos[env_idx, source_idx, 0]
            out_pos[env_idx, output_idx, 1] = source_pos[env_idx, source_idx, 1]
            out_pos[env_idx, output_idx, 2] = source_pos[env_idx, source_idx, 2]
            out_quat[env_idx, output_idx, 0] = source_quat[env_idx, source_idx, 0]
            out_quat[env_idx, output_idx, 1] = source_quat[env_idx, source_idx, 1]
            out_quat[env_idx, output_idx, 2] = source_quat[env_idx, source_idx, 2]
            out_quat[env_idx, output_idx, 3] = source_quat[env_idx, source_idx, 3]
            out_lin_vel[env_idx, output_idx, 0] = source_lin_vel[env_idx, source_idx, 0]
            out_lin_vel[env_idx, output_idx, 1] = source_lin_vel[env_idx, source_idx, 1]
            out_lin_vel[env_idx, output_idx, 2] = source_lin_vel[env_idx, source_idx, 2]
            out_ang_vel[env_idx, output_idx, 0] = source_ang_vel[env_idx, source_idx, 0]
            out_ang_vel[env_idx, output_idx, 1] = source_ang_vel[env_idx, source_idx, 1]
            out_ang_vel[env_idx, output_idx, 2] = source_ang_vel[env_idx, source_idx, 2]


__all__ = ["copy_selected_body_state"]
