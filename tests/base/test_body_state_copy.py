"""Focused parity tests for shared and temporary body-state copy kernels."""

from __future__ import annotations

import numpy as np

from unilab.base.backend.body_state import copy_selected_body_state
from unilab.base.backend.motrix.body_state import copy_selected_motrix_body_state


def _outputs(num_envs: int, num_selected: int) -> tuple[np.ndarray, ...]:
    shape = (num_envs, num_selected, 3)
    return (
        np.empty(shape, dtype=np.float32),
        np.empty((num_envs, num_selected, 4), dtype=np.float32),
        np.empty(shape, dtype=np.float32),
        np.empty(shape, dtype=np.float32),
    )


def test_shared_body_state_copy_kernel_preserves_selection_and_outputs() -> None:
    rng = np.random.default_rng(7)
    num_envs, num_bodies = 257, 9
    pos = rng.standard_normal((num_envs, num_bodies, 3), dtype=np.float32)
    quat = rng.standard_normal((num_envs, num_bodies, 4), dtype=np.float32)
    lin_vel = rng.standard_normal((num_envs, num_bodies, 3), dtype=np.float32)
    ang_vel = rng.standard_normal((num_envs, num_bodies, 3), dtype=np.float32)
    selected = np.asarray([7, 1, 5], dtype=np.intp)
    out_pos, out_quat, out_lin_vel, out_ang_vel = _outputs(num_envs, len(selected))

    copy_selected_body_state(
        pos,
        quat,
        lin_vel,
        ang_vel,
        selected,
        out_pos,
        out_quat,
        out_lin_vel,
        out_ang_vel,
    )

    np.testing.assert_array_equal(out_pos, pos[:, selected])
    np.testing.assert_array_equal(out_quat, quat[:, selected])
    np.testing.assert_array_equal(out_lin_vel, lin_vel[:, selected])
    np.testing.assert_array_equal(out_ang_vel, ang_vel[:, selected])
    assert copy_selected_body_state.targetoptions["nopython"] is True
    assert copy_selected_body_state.targetoptions["nogil"] is True
    assert copy_selected_body_state.targetoptions["parallel"] is True
    assert copy_selected_body_state.signatures


def test_temporary_motrix_copy_kernel_converts_xyzw_and_packed_velocity() -> None:
    rng = np.random.default_rng(11)
    num_envs, num_bodies = 257, 9
    poses = rng.standard_normal((num_envs, num_bodies, 7), dtype=np.float32)
    velocities = rng.standard_normal((num_envs, num_bodies, 6), dtype=np.float32)
    selected = np.asarray([8, 2, 4], dtype=np.int32)
    out_pos, out_quat, out_lin_vel, out_ang_vel = _outputs(num_envs, len(selected))

    copy_selected_motrix_body_state(
        poses,
        velocities,
        selected,
        out_pos,
        out_quat,
        out_lin_vel,
        out_ang_vel,
    )

    np.testing.assert_array_equal(out_pos, poses[:, selected, :3])
    np.testing.assert_array_equal(out_quat[..., 0], poses[:, selected, 6])
    np.testing.assert_array_equal(out_quat[..., 1:], poses[:, selected, 3:6])
    np.testing.assert_array_equal(out_lin_vel, velocities[:, selected, :3])
    np.testing.assert_array_equal(out_ang_vel, velocities[:, selected, 3:])
    assert copy_selected_motrix_body_state.targetoptions["nopython"] is True
    assert copy_selected_motrix_body_state.targetoptions["nogil"] is True
    assert copy_selected_motrix_body_state.targetoptions["parallel"] is True
    assert copy_selected_motrix_body_state.signatures
