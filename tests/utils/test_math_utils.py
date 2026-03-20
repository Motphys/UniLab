"""Tests for quaternion helpers in unilab.utils.math_utils."""

from __future__ import annotations

import numpy as np

from unilab.utils.math_utils import np_quat_error_magnitude


def _quat_from_axis_angle_z(angle_rad: float) -> np.ndarray:
    half = 0.5 * angle_rad
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def test_quat_error_invariant_to_sign_flip() -> None:
    """q and -q represent the same orientation, so error must be zero."""
    q = _quat_from_axis_angle_z(np.deg2rad(170.0))
    neg_q = -q

    err = np_quat_error_magnitude(q, neg_q)
    assert np.isclose(err, 0.0, atol=1e-7)


def test_quat_error_uses_shortest_arc() -> None:
    """Quaternion sign should not change the measured angular distance."""
    q_ref = _quat_from_axis_angle_z(0.0)
    q_target = _quat_from_axis_angle_z(np.deg2rad(170.0))
    q_target_neg = -q_target

    err_pos = np_quat_error_magnitude(q_ref, q_target)
    err_neg = np_quat_error_magnitude(q_ref, q_target_neg)

    expected = np.deg2rad(170.0)
    assert np.isclose(err_pos, expected, atol=1e-7)
    assert np.isclose(err_neg, expected, atol=1e-7)


def test_quat_error_vectorized_batch() -> None:
    q_ref = np.stack([_quat_from_axis_angle_z(0.0), _quat_from_axis_angle_z(0.0)], axis=0)
    q_target = np.stack(
        [
            _quat_from_axis_angle_z(np.deg2rad(30.0)),
            -_quat_from_axis_angle_z(np.deg2rad(45.0)),
        ],
        axis=0,
    )

    err = np_quat_error_magnitude(q_ref, q_target)

    np.testing.assert_allclose(err, np.deg2rad([30.0, 45.0]), atol=1e-7)
