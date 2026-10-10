"""Cold height-scan sampling offsets for rough locomotion tasks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DEFAULT_SCAN_POINTS_X: tuple[float, ...] = (
    -0.8,
    -0.7,
    -0.6,
    -0.5,
    -0.4,
    -0.3,
    -0.2,
    -0.1,
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
)
DEFAULT_SCAN_POINTS_Y: tuple[float, ...] = (
    -0.5,
    -0.4,
    -0.3,
    -0.2,
    -0.1,
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
)


def height_scan_offsets(points_x: Sequence[float], points_y: Sequence[float]) -> np.ndarray:
    """Build a contiguous (P, 2) array of (x, y) sampling offsets in body frame."""
    x_grid, y_grid = np.meshgrid(
        np.asarray(points_x, dtype=np.float64),
        np.asarray(points_y, dtype=np.float64),
        indexing="ij",
    )
    offsets = np.stack([x_grid.reshape(-1), y_grid.reshape(-1)], axis=1)
    return np.ascontiguousarray(offsets, dtype=np.float64)
