"""Focused parity tests for the shared body-state copy kernel."""

from __future__ import annotations

import numpy as np
from unisim.backend.body_state import copy_selected_body_state


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
    # The shared helper intentionally stays NumPy-only so the core package has
    # no mandatory Numba dependency. Backend-specific compiled kernels belong
    # to their optional adapter extras.
    assert not hasattr(copy_selected_body_state, "targetoptions")
