"""Temporary Motrix packed-cache body-state copy kernel."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


# TODO(#1305, #1308): Delete this kernel after MotrixSim exposes the native
# selected-body fused world-state API tracked by the convergence issue.
@njit(cache=True, nogil=True, parallel=True)
def copy_selected_motrix_body_state(
    link_poses: np.ndarray,
    link_velocities: np.ndarray,
    body_ids: np.ndarray,
    out_pos: np.ndarray,
    out_quat: np.ndarray,
    out_lin_vel: np.ndarray,
    out_ang_vel: np.ndarray,
) -> None:
    """Copy selected xyzw pose and packed velocity cache columns as wxyz state."""
    for env_idx in prange(link_poses.shape[0]):
        for output_idx in range(body_ids.shape[0]):
            body_idx = body_ids[output_idx]
            out_pos[env_idx, output_idx, 0] = link_poses[env_idx, body_idx, 0]
            out_pos[env_idx, output_idx, 1] = link_poses[env_idx, body_idx, 1]
            out_pos[env_idx, output_idx, 2] = link_poses[env_idx, body_idx, 2]
            out_quat[env_idx, output_idx, 0] = link_poses[env_idx, body_idx, 6]
            out_quat[env_idx, output_idx, 1] = link_poses[env_idx, body_idx, 3]
            out_quat[env_idx, output_idx, 2] = link_poses[env_idx, body_idx, 4]
            out_quat[env_idx, output_idx, 3] = link_poses[env_idx, body_idx, 5]
            out_lin_vel[env_idx, output_idx, 0] = link_velocities[env_idx, body_idx, 0]
            out_lin_vel[env_idx, output_idx, 1] = link_velocities[env_idx, body_idx, 1]
            out_lin_vel[env_idx, output_idx, 2] = link_velocities[env_idx, body_idx, 2]
            out_ang_vel[env_idx, output_idx, 0] = link_velocities[env_idx, body_idx, 3]
            out_ang_vel[env_idx, output_idx, 1] = link_velocities[env_idx, body_idx, 4]
            out_ang_vel[env_idx, output_idx, 2] = link_velocities[env_idx, body_idx, 5]


__all__ = ["copy_selected_motrix_body_state"]
