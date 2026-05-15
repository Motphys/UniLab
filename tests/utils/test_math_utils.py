"""Tests for quaternion helpers in unilab.envs.common.rotation."""

from __future__ import annotations

import numpy as np

from unilab.envs.common.rotation import (
    np_matrix_first_two_cols_from_quat,
    np_matrix_from_quat,
    np_quat_angular_velocity,
    np_quat_apply,
    np_quat_apply_batched,
    np_quat_ensure_continuity,
    np_quat_error_magnitude,
    np_quat_error_magnitude_batched,
    np_quat_error_magnitude_squared_batched,
    np_quat_from_euler_xyz,
    np_quat_mul,
    np_quat_mul_batched,
    np_quat_to_axis_angle,
    np_subtract_anchor_frame_transforms,
    np_subtract_frame_transforms,
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


def test_batched_quaternion_helpers_match_flattened_helpers() -> None:
    num_envs = 3
    num_bodies = 4

    anchor_quat = np_quat_from_euler_xyz(
        np.linspace(-0.2, 0.3, num_envs),
        np.linspace(0.1, -0.25, num_envs),
        np.linspace(0.4, -0.1, num_envs),
    )
    body_quat = np_quat_from_euler_xyz(
        np.linspace(-0.3, 0.4, num_envs * num_bodies),
        np.linspace(0.2, -0.15, num_envs * num_bodies),
        np.linspace(-0.5, 0.25, num_envs * num_bodies),
    ).reshape(num_envs, num_bodies, 4)
    vectors = np.linspace(-0.6, 0.7, num_envs * num_bodies * 3).reshape(
        num_envs, num_bodies, 3
    )

    anchor_quat_tiled = np.tile(anchor_quat, (1, num_bodies)).reshape(
        num_envs * num_bodies, 4
    )
    body_quat_flat = body_quat.reshape(num_envs * num_bodies, 4)
    vectors_flat = vectors.reshape(num_envs * num_bodies, 3)

    expected_mul = np_quat_mul(anchor_quat_tiled, body_quat_flat).reshape(
        num_envs, num_bodies, 4
    )
    expected_apply = np_quat_apply(anchor_quat_tiled, vectors_flat).reshape(
        num_envs, num_bodies, 3
    )
    expected_error = np_quat_error_magnitude(anchor_quat_tiled, body_quat_flat).reshape(
        num_envs, num_bodies
    )
    expected_matrix_cols = np_matrix_from_quat(body_quat_flat)[:, :, :2].reshape(
        num_envs, num_bodies, 6
    )

    np.testing.assert_allclose(
        np_quat_mul_batched(anchor_quat[:, None, :], body_quat),
        expected_mul,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np_quat_apply_batched(anchor_quat[:, None, :], vectors),
        expected_apply,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np_quat_error_magnitude_batched(anchor_quat[:, None, :], body_quat),
        expected_error,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np_quat_error_magnitude_squared_batched(anchor_quat[:, None, :], body_quat),
        expected_error * expected_error,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np_matrix_first_two_cols_from_quat(body_quat),
        expected_matrix_cols,
        atol=1e-12,
    )


def test_anchor_frame_transform_matches_flattened_path() -> None:
    num_envs = 3
    num_bodies = 4

    anchor_pos = np.linspace(-0.2, 0.4, num_envs * 3).reshape(num_envs, 3)
    body_pos = np.linspace(-0.5, 0.7, num_envs * num_bodies * 3).reshape(
        num_envs, num_bodies, 3
    )
    anchor_quat = np_quat_from_euler_xyz(
        np.linspace(0.0, 0.2, num_envs),
        np.linspace(-0.1, 0.1, num_envs),
        np.linspace(0.3, -0.2, num_envs),
    )
    body_quat = np_quat_from_euler_xyz(
        np.linspace(-0.25, 0.35, num_envs * num_bodies),
        np.linspace(0.15, -0.2, num_envs * num_bodies),
        np.linspace(-0.4, 0.3, num_envs * num_bodies),
    ).reshape(num_envs, num_bodies, 4)

    anchor_pos_tiled = np.tile(anchor_pos, (1, num_bodies)).reshape(
        num_envs * num_bodies, 3
    )
    anchor_quat_tiled = np.tile(anchor_quat, (1, num_bodies)).reshape(
        num_envs * num_bodies, 4
    )
    expected_pos, expected_quat = np_subtract_frame_transforms(
        anchor_pos_tiled,
        anchor_quat_tiled,
        body_pos.reshape(num_envs * num_bodies, 3),
        body_quat.reshape(num_envs * num_bodies, 4),
    )

    actual_pos, actual_quat = np_subtract_anchor_frame_transforms(
        anchor_pos,
        anchor_quat,
        body_pos,
        body_quat,
    )

    np.testing.assert_allclose(actual_pos, expected_pos.reshape(num_envs, num_bodies, 3))
    np.testing.assert_allclose(actual_quat, expected_quat.reshape(num_envs, num_bodies, 4))
