from __future__ import annotations

import numpy as np
import pytest

from unilab.envs.common.rotation import (
    np_matrix_first_two_cols_from_quat,
    np_subtract_frame_transforms,
)
from unilab.utils.numba_geometry import (
    NUMBA_GEOMETRY_AVAILABLE,
    quat_angle_sq_at,
    write_relative_anchor_transform_at,
)


@pytest.mark.skipif(not NUMBA_GEOMETRY_AVAILABLE, reason="numba is optional")
def test_numba_geometry_quat_angle_sq_matches_expected_angles():
    from numba import njit

    @njit
    def run(q1, q2):
        return quat_angle_sq_at(q1, q2, 0, 0)

    q1 = np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float64)
    q2 = np.array([[[0.0, 1.0, 0.0, 0.0]]], dtype=np.float64)

    assert run(q1, q1) == pytest.approx(0.0)
    assert run(q1, q2) == pytest.approx(np.pi * np.pi)


@pytest.mark.skipif(not NUMBA_GEOMETRY_AVAILABLE, reason="numba is optional")
def test_numba_geometry_relative_anchor_transform_matches_numpy_rotation_helpers():
    from numba import njit

    @njit
    def run(source_pos, source_quat, target_pos, target_quat, out_pos, out_rot6d):
        write_relative_anchor_transform_at(
            source_pos,
            source_quat,
            target_pos,
            target_quat,
            0,
            out_pos,
            out_rot6d,
            0,
        )

    source_pos = np.array([[[0.7, -0.2, 1.3]]], dtype=np.float64)
    source_quat = np.array([[[0.9238795325, 0.0, 0.3826834324, 0.0]]], dtype=np.float64)
    target_pos = np.array([[[-0.1, 0.4, 0.8]]], dtype=np.float64)
    target_quat = np.array([[[0.9659258263, 0.0, 0.0, 0.2588190451]]], dtype=np.float64)
    out_pos = np.empty((1, 3), dtype=np.float64)
    out_rot6d = np.empty((1, 6), dtype=np.float64)

    run(source_pos, source_quat, target_pos, target_quat, out_pos, out_rot6d)

    expected_pos, expected_quat = np_subtract_frame_transforms(
        target_pos[0, 0],
        target_quat[0, 0],
        source_pos[0, 0],
        source_quat[0, 0],
    )
    expected_rot6d = np_matrix_first_two_cols_from_quat(expected_quat)

    np.testing.assert_allclose(out_pos[0], expected_pos, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(out_rot6d[0], expected_rot6d, rtol=1e-9, atol=1e-9)
