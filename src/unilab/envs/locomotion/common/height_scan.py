"""Shared height-scan and terrain-bound helpers for rough locomotion envs.

These functions and the ``HeightScanConfig`` dataclass are consumed by Go2,
Go2W, Go1, and G1 rough environments — anywhere the policy / critic ingests
a forward-looking height grid sampled from a procedural heightfield.

Each helper mirrors the original implementation that lived inline in
``go2/rough.py`` so that behavior is bit-for-bit identical; the only change
is the host: env classes import these instead of redefining them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.dtype_config import get_global_dtype

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


@dataclass
class HeightScanConfig:
    enabled: bool = True
    hfield_name: str = "terrain_hfield"
    geom_name: str = "floor"
    measured_points_x: list[float] = field(default_factory=lambda: list(DEFAULT_SCAN_POINTS_X))
    measured_points_y: list[float] = field(default_factory=lambda: list(DEFAULT_SCAN_POINTS_Y))
    vertical_offset: float = 0.5
    scale: float = 5.0


def height_scan_offsets(points_x: Sequence[float], points_y: Sequence[float]) -> np.ndarray:
    """Build a contiguous (P, 2) array of (x, y) sampling offsets in body frame."""
    x_grid, y_grid = np.meshgrid(
        np.asarray(points_x, dtype=np.float64),
        np.asarray(points_y, dtype=np.float64),
        indexing="ij",
    )
    offsets = np.stack([x_grid.reshape(-1), y_grid.reshape(-1)], axis=1)
    return np.ascontiguousarray(offsets, dtype=np.float64)


def configured_height_scan_dim(scan_cfg: HeightScanConfig) -> int:
    return len(scan_cfg.measured_points_x) * len(scan_cfg.measured_points_y)


def init_height_scan_sensor(env: Any, scan_cfg: HeightScanConfig, base_body_name: str) -> None:
    """Wire a yaw-aligned heightfield scanner onto ``env``.

    Sets the following attributes on ``env``:
        _height_scan_dim, _height_scan_hfield_geom_id, _height_scan_frame_body_id,
        _height_scan_offsets, _height_scan_sensor.
    """
    env._height_scan_dim = configured_height_scan_dim(scan_cfg)
    if env._height_scan_dim <= 0:
        raise ValueError("terrain_scan measured points must be non-empty")

    env._height_scan_hfield_geom_id = None
    env._height_scan_frame_body_id = None
    env._height_scan_offsets = None
    env._height_scan_sensor = None
    if not scan_cfg.enabled:
        return

    env._height_scan_hfield_geom_id = env._backend.get_geom_id(scan_cfg.geom_name)
    env._height_scan_frame_body_id = env._backend.get_body_id(base_body_name)
    env._height_scan_offsets = height_scan_offsets(
        scan_cfg.measured_points_x,
        scan_cfg.measured_points_y,
    )
    env._height_scan_sensor = env._backend.create_hfield_scanner(
        hfield_geom_id=env._height_scan_hfield_geom_id,
        offsets=env._height_scan_offsets,
        frame_body_id=env._height_scan_frame_body_id,
        alignment="yaw",
        output="height",
    )


def raw_height_scan_obs(env: Any, num_obs: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (raw_heights (N, P), base_pos (N, 3)) or (None, None) if sensor not ready."""
    if (
        env._height_scan_hfield_geom_id is None
        or env._height_scan_frame_body_id is None
        or env._height_scan_offsets is None
        or env._height_scan_sensor is None
    ):
        return None, None

    base_pos = np.asarray(env._backend.get_base_pos(), dtype=get_global_dtype())
    if base_pos.shape[0] != num_obs:
        return None, None

    raw_heights = env._height_scan_sensor.scan()
    if raw_heights.shape != (num_obs, env._height_scan_dim):
        return None, None
    return np.asarray(raw_heights, dtype=get_global_dtype()), base_pos


def height_scan_obs(env: Any, scan_cfg: HeightScanConfig, num_obs: int) -> np.ndarray:
    """Clipped, scaled height observation matching the Go2 rough format."""
    raw_heights, base_pos = raw_height_scan_obs(env, num_obs)
    if raw_heights is None or base_pos is None:
        return np.zeros((num_obs, env._height_scan_dim), dtype=get_global_dtype())
    heights = np.clip(base_pos[:, 2:3] - scan_cfg.vertical_offset - raw_heights, -1.0, 1.0)
    return np.asarray(heights * scan_cfg.scale, dtype=get_global_dtype())


def base_height_from_scan(env: Any, num_obs: int | None = None) -> np.ndarray:
    """Estimate base-relative height by averaging the heightfield samples below the body."""
    if num_obs is None:
        num_obs = int(np.asarray(env._backend.get_base_pos()).shape[0])
    raw_heights, base_pos = raw_height_scan_obs(env, num_obs)
    if raw_heights is None or base_pos is None:
        base_pos = np.asarray(env._backend.get_base_pos(), dtype=get_global_dtype())
        if base_pos.shape[0] != num_obs:
            return np.zeros((num_obs,), dtype=get_global_dtype())
        return np.asarray(base_pos[:, 2], dtype=get_global_dtype())
    return np.asarray(np.mean(base_pos[:, 2:3] - raw_heights, axis=1), dtype=get_global_dtype())


def terrain_num_cols(terrain_cfg: Any) -> int:
    if terrain_cfg.curriculum:
        return len(terrain_cfg.sub_terrains)
    return int(terrain_cfg.num_cols)


def terrain_out_of_bounds(env: Any, terrain_cfg: Any, distance_buffer: float) -> np.ndarray:
    """Boolean mask: True when the body's (x, y) is outside the terrain footprint."""
    if terrain_cfg is None:
        return np.zeros((env._num_envs,), dtype=bool)

    size_x, size_y = terrain_cfg.size
    num_cols = terrain_num_cols(terrain_cfg)
    map_width = terrain_cfg.num_rows * float(size_x) + 2.0 * float(terrain_cfg.border_width)
    map_height = num_cols * float(size_y) + 2.0 * float(terrain_cfg.border_width)
    base_pos = np.asarray(env._backend.get_base_pos(), dtype=get_global_dtype())
    if base_pos.shape[0] != env._num_envs:
        return np.zeros((env._num_envs,), dtype=bool)
    x_out = np.abs(base_pos[:, 0]) > 0.5 * map_width - distance_buffer
    y_out = np.abs(base_pos[:, 1]) > 0.5 * map_height - distance_buffer
    return np.asarray(x_out | y_out, dtype=bool)
