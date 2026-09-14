"""Tests for the shared CSV-motion interpolation helpers in the library layer."""

from __future__ import annotations

import numpy as np

from unilab.tasks.motion_tracking.common.motion_loader import quat_slerp


def _random_quats(rng: np.random.Generator, n: int, dtype=np.float32) -> np.ndarray:
    quats = rng.standard_normal((n, 4)).astype(dtype)
    return quats / np.linalg.norm(quats, axis=1, keepdims=True)


def test_quat_slerp_takes_shortest_path() -> None:
    rng = np.random.default_rng(3)
    q1, q2 = _random_quats(rng, 2)
    direct = quat_slerp(q1, q2, 0.5)
    flipped = quat_slerp(q1, -q2, 0.5)
    np.testing.assert_allclose(direct, flipped, atol=1e-6)
