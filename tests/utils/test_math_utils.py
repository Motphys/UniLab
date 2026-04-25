"""Tests for quaternion helpers in unilab.envs.common.rotation."""

from __future__ import annotations

import numpy as np

from unilab.envs.common.rotation import (
    np_quat_angular_velocity,
    np_quat_ensure_continuity,
    np_quat_error_magnitude,
    np_quat_to_axis_angle,
)


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


def test_quat_to_axis_angle_invariant_to_sign_flip() -> None:
    q = _quat_from_axis_angle_z(np.deg2rad(170.0))
    axis_angle = np_quat_to_axis_angle(q[None, :])[0]
    axis_angle_neg = np_quat_to_axis_angle((-q)[None, :])[0]

    np.testing.assert_allclose(axis_angle, axis_angle_neg, atol=1e-7)


def test_quat_ensure_continuity_flips_sequence_signs() -> None:
    q = np.stack(
        [
            _quat_from_axis_angle_z(0.0),
            _quat_from_axis_angle_z(np.deg2rad(10.0)),
            -_quat_from_axis_angle_z(np.deg2rad(20.0)),
            -_quat_from_axis_angle_z(np.deg2rad(30.0)),
        ],
        axis=0,
    )

    continuous = np_quat_ensure_continuity(q)
    dots = np.sum(continuous[:-1] * continuous[1:], axis=1)

    assert np.all(dots >= 0.0)


def test_quat_angular_velocity_ignores_sign_flip_spikes() -> None:
    dt = 0.1
    angles = np.arange(5, dtype=np.float64) * dt
    q = np.stack([_quat_from_axis_angle_z(angle) for angle in angles], axis=0)
    q[2:] *= -1.0

    angvel = np_quat_angular_velocity(q, dt)
    expected = np.tile(np.array([0.0, 0.0, 1.0]), (q.shape[0], 1))

    np.testing.assert_allclose(angvel, expected, atol=1e-6)
