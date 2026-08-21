"""Tests for the shared CSV-motion interpolation helpers in the library layer.

The reference functions below are verbatim copies of the implementations that
used to live in ``scripts/motion/csv_to_npz.py`` and
``scripts/motion/x2_csv_to_tracking_npz.py`` (see issue #1244). The library
helpers must reproduce them exactly so converted NPZ products stay
bit-identical.
"""

from __future__ import annotations

import numpy as np

from unilab.tasks.motion_tracking.common.motion_loader import (
    compute_motion_velocities,
    interpolate_motion,
    quat_slerp,
)
from unilab.utils.rotation import np_quat_angular_velocity, np_quat_ensure_continuity


def _reference_quat_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Former scripts/motion/csv_to_npz.py::quat_slerp (float32 path)."""
    dot = np.dot(q1, q2)
    if dot < 0:
        q2 = -q2
        dot = -dot

    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    theta = np.arccos(np.clip(dot, -1, 1))
    sin_theta = np.sin(theta)

    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta

    return w1 * q1 + w2 * q2


def _reference_x2_quat_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Former scripts/motion/x2_csv_to_tracking_npz.py::_quat_slerp (float64 path)."""
    q1 = q1.astype(np.float64, copy=False)
    q2 = q2.astype(np.float64, copy=False)
    dot = float(np.dot(q1, q2))
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return (result / np.linalg.norm(result)).astype(np.float32)

    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    w1 = np.sin((1.0 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    return (w1 * q1 + w2 * q2).astype(np.float32)


def _reference_interpolate(
    base_poss_input: np.ndarray,
    base_rots_input: np.ndarray,
    dof_poss_input: np.ndarray,
    input_fps: int,
    output_fps: int,
) -> dict[str, np.ndarray]:
    """Former scripts/motion MotionLoader._interpolate_motion + _compute_velocities."""
    input_dt = 1.0 / input_fps
    output_dt = 1.0 / output_fps
    input_frames = base_poss_input.shape[0]
    duration = (input_frames - 1) * input_dt

    times = np.arange(0, duration, output_dt, dtype=np.float32)
    output_frames = times.shape[0]

    phase = times / duration
    index_0 = np.floor(phase * (input_frames - 1)).astype(np.int32)
    index_1 = np.minimum(index_0 + 1, input_frames - 1)
    blend = phase * (input_frames - 1) - index_0

    base_poss = (
        base_poss_input[index_0] * (1 - blend[:, None]) + base_poss_input[index_1] * blend[:, None]
    )

    base_rots = np.zeros((output_frames, 4), dtype=np.float32)
    for i in range(output_frames):
        base_rots[i] = _reference_quat_slerp(
            base_rots_input[index_0[i]], base_rots_input[index_1[i]], blend[i]
        )
    base_rots = np_quat_ensure_continuity(base_rots)

    dof_poss = (
        dof_poss_input[index_0] * (1 - blend[:, None]) + dof_poss_input[index_1] * blend[:, None]
    )

    base_lin_vels = np.gradient(base_poss, output_dt, axis=0)
    dof_vels = np.gradient(dof_poss, output_dt, axis=0)
    base_ang_vels = np_quat_angular_velocity(base_rots, output_dt)

    return {
        "output_frames": output_frames,
        "base_poss": base_poss,
        "base_rots": base_rots,
        "dof_poss": dof_poss,
        "base_lin_vels": base_lin_vels,
        "base_ang_vels": base_ang_vels,
        "dof_vels": dof_vels,
    }


def _random_quats(rng: np.random.Generator, n: int, dtype=np.float32) -> np.ndarray:
    quats = rng.standard_normal((n, 4)).astype(dtype)
    return quats / np.linalg.norm(quats, axis=1, keepdims=True)


def test_quat_slerp_matches_former_script_float32() -> None:
    rng = np.random.default_rng(0)
    quats = _random_quats(rng, 32)
    for i in range(0, 30, 2):
        for t in (0.0, 0.13, 0.5, 0.97, 1.0):
            expected = _reference_quat_slerp(quats[i], quats[i + 1], np.float32(t))
            actual = quat_slerp(quats[i], quats[i + 1], np.float32(t))
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)


def test_quat_slerp_matches_former_script_near_identical() -> None:
    rng = np.random.default_rng(1)
    q1 = _random_quats(rng, 1)[0]
    q2 = q1 + rng.standard_normal(4).astype(np.float32) * 1e-4
    q2 = q2 / np.linalg.norm(q2)
    expected = _reference_quat_slerp(q1, q2, np.float32(0.3))
    np.testing.assert_array_equal(quat_slerp(q1, q2, np.float32(0.3)), expected)


def test_quat_slerp_float64_path_matches_former_x2_script() -> None:
    rng = np.random.default_rng(2)
    quats = _random_quats(rng, 8)
    for i in range(0, 6, 2):
        for t in (0.0, 0.42, 1.0):
            expected = _reference_x2_quat_slerp(quats[i], quats[i + 1], t)
            actual = quat_slerp(
                quats[i].astype(np.float64), quats[i + 1].astype(np.float64), t
            ).astype(np.float32)
            np.testing.assert_array_equal(actual, expected)


def test_quat_slerp_takes_shortest_path() -> None:
    rng = np.random.default_rng(3)
    q1, q2 = _random_quats(rng, 2)
    direct = quat_slerp(q1, q2, 0.5)
    flipped = quat_slerp(q1, -q2, 0.5)
    np.testing.assert_allclose(direct, flipped, atol=1e-6)


def test_interpolate_motion_matches_former_script() -> None:
    rng = np.random.default_rng(4)
    input_frames = 61
    base_poss_input = rng.standard_normal((input_frames, 3)).astype(np.float32)
    base_rots_input = np_quat_ensure_continuity(_random_quats(rng, input_frames))
    dof_poss_input = rng.standard_normal((input_frames, 29)).astype(np.float32)

    for input_fps, output_fps in ((30, 50), (120, 50), (50, 50), (120, 30)):
        expected = _reference_interpolate(
            base_poss_input, base_rots_input, dof_poss_input, input_fps, output_fps
        )
        actual = interpolate_motion(
            base_poss_input,
            base_rots_input,
            dof_poss_input,
            input_fps=input_fps,
            output_fps=output_fps,
        )
        assert actual.output_frames == expected["output_frames"]
        for key in (
            "base_poss",
            "base_rots",
            "dof_poss",
            "base_lin_vels",
            "base_ang_vels",
            "dof_vels",
        ):
            np.testing.assert_array_equal(getattr(actual, key), expected[key])


def test_compute_motion_velocities_matches_former_script() -> None:
    rng = np.random.default_rng(5)
    frames = 40
    base_poss = rng.standard_normal((frames, 3)).astype(np.float32)
    base_rots = np_quat_ensure_continuity(_random_quats(rng, frames))
    dof_poss = rng.standard_normal((frames, 7)).astype(np.float32)
    dt = 1.0 / 50

    base_lin_vels, base_ang_vels, dof_vels = compute_motion_velocities(
        base_poss, base_rots, dof_poss, dt
    )
    np.testing.assert_array_equal(base_lin_vels, np.gradient(base_poss, dt, axis=0))
    np.testing.assert_array_equal(dof_vels, np.gradient(dof_poss, dt, axis=0))
    np.testing.assert_array_equal(base_ang_vels, np_quat_angular_velocity(base_rots, dt))
