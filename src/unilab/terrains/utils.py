"""Utility functions for terrain generation.

References:
    IsaacLab terrain utilities:
    https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/terrains/trimesh/utils.py
"""

from __future__ import annotations

import numpy as np

from unilab.terrains.terrain_generator import FlatPatchSamplingCfg  # noqa: F401


def bilinear_zoom_2d(arr: np.ndarray, zoom_factors: float | tuple[float, float]) -> np.ndarray:
    """Resize a 2D array with bilinear interpolation."""
    arr = np.asarray(arr, dtype=np.float64)
    if np.isscalar(zoom_factors):
        zy = zx = float(zoom_factors)  # type: ignore[arg-type]
    else:
        zy, zx = float(zoom_factors[0]), float(zoom_factors[1])  # type: ignore[index]
    h, w = arr.shape
    new_h = max(int(round(h * zy)), 1)
    new_w = max(int(round(w * zx)), 1)

    y_in = np.arange(new_h, dtype=np.float64) / zy
    x_in = np.arange(new_w, dtype=np.float64) / zx

    y0 = np.floor(y_in).astype(np.intp)
    x0 = np.floor(x_in).astype(np.intp)

    wy = (y_in - y0).reshape(-1, 1)
    wx = (x_in - x0).reshape(1, -1)

    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y0 + 1, 0, h - 1)
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x0 + 1, 0, w - 1)

    a = arr[y0c[:, None], x0c[None, :]]
    b = arr[y0c[:, None], x1c[None, :]]
    c = arr[y1c[:, None], x0c[None, :]]
    d = arr[y1c[:, None], x1c[None, :]]

    return a * (1 - wy) * (1 - wx) + b * (1 - wy) * wx + c * wy * (1 - wx) + d * wy * wx


def bilinear_resample_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    x_new: np.ndarray,
    y_new: np.ndarray,
) -> np.ndarray:
    """Resample a regular grid ``z(x, y)`` at the cartesian product of new coords."""
    z = np.asarray(z, dtype=np.float64)
    nx, ny = z.shape

    ix = np.clip(np.searchsorted(x, x_new) - 1, 0, nx - 2)
    iy = np.clip(np.searchsorted(y, y_new) - 1, 0, ny - 2)

    wx = ((x_new - x[ix]) / (x[ix + 1] - x[ix])).reshape(-1, 1)
    wy = ((y_new - y[iy]) / (y[iy + 1] - y[iy])).reshape(1, -1)

    z00 = z[ix[:, None], iy[None, :]]
    z01 = z[ix[:, None], (iy + 1)[None, :]]
    z10 = z[(ix + 1)[:, None], iy[None, :]]
    z11 = z[(ix + 1)[:, None], (iy + 1)[None, :]]

    return z00 * (1 - wx) * (1 - wy) + z01 * (1 - wx) * wy + z10 * wx * (1 - wy) + z11 * wx * wy


def _morphological_filter(
    arr: np.ndarray,
    footprint: np.ndarray,
    fill_value: float,
    op: np.ufunc,
) -> np.ndarray:
    """Apply a max/min filter under an arbitrary footprint with constant padding."""
    h, w = arr.shape
    fh, fw = footprint.shape
    pad_h, pad_w = fh // 2, fw // 2
    padded = np.pad(
        arr, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=fill_value
    )
    result = np.full_like(arr, fill_value)
    for di in range(fh):
        for dj in range(fw):
            if footprint[di, dj]:
                op(result, padded[di : di + h, dj : dj + w], out=result)
    return result


def max_filter_with_footprint(
    arr: np.ndarray, footprint: np.ndarray, fill_value: float = -np.inf
) -> np.ndarray:
    """Maximum filter with arbitrary footprint, padding out-of-bounds with ``fill_value``."""
    return _morphological_filter(arr, footprint, fill_value, np.maximum)


def min_filter_with_footprint(
    arr: np.ndarray, footprint: np.ndarray, fill_value: float = np.inf
) -> np.ndarray:
    """Minimum filter with arbitrary footprint, padding out-of-bounds with ``fill_value``."""
    return _morphological_filter(arr, footprint, fill_value, np.minimum)


def find_flat_patches_from_heightfield(
    heights: np.ndarray,
    horizontal_scale: float,
    z_offset: float,
    cfg: FlatPatchSamplingCfg,
    rng: np.random.Generator,
) -> np.ndarray:
    """Find flat patches on a heightfield surface using morphological filtering."""
    # Optionally upsample to a finer grid for higher-precision boundary detection.
    if cfg.grid_resolution is not None and cfg.grid_resolution < horizontal_scale:
        zoom_factor = horizontal_scale / cfg.grid_resolution
        heights = bilinear_zoom_2d(heights, zoom_factor)
        horizontal_scale = cfg.grid_resolution

    num_rows, num_cols = heights.shape

    # Build circular footprint.
    radius_pixels = int(np.ceil(cfg.patch_radius / horizontal_scale))
    y_grid, x_grid = np.ogrid[
        -radius_pixels : radius_pixels + 1, -radius_pixels : radius_pixels + 1
    ]
    footprint = (x_grid**2 + y_grid**2) <= radius_pixels**2

    # Morphological max/min filter to find height variation within footprint.
    # Use constant padding so edge pixels where the footprint extends outside
    # the data are not incorrectly marked flat (default 'reflect' hides edges).
    max_h = max_filter_with_footprint(heights, footprint, fill_value=-np.inf)
    min_h = min_filter_with_footprint(heights, footprint, fill_value=np.inf)
    valid_mask = (max_h - min_h) <= cfg.max_height_diff

    # Exclude pixels whose footprint would extend outside the array. This
    # ensures the full patch circle lies within the heightfield bounds.
    valid_mask[:radius_pixels, :] = False
    valid_mask[-radius_pixels:, :] = False
    valid_mask[:, :radius_pixels] = False
    valid_mask[:, -radius_pixels:] = False

    # Apply spatial range constraints.
    # MuJoCo hfield convention: columns map to the x-axis, rows map to the
    # y-axis (see engine_ray.c vertex: {dx*c - size[0], dy*r - size[1], ...}).
    x_coords = np.arange(num_cols) * horizontal_scale
    y_coords = np.arange(num_rows) * horizontal_scale

    x_valid = (x_coords >= cfg.x_range[0]) & (x_coords <= cfg.x_range[1])
    y_valid = (y_coords >= cfg.y_range[0]) & (y_coords <= cfg.y_range[1])
    valid_mask &= y_valid[:, None] & x_valid[None, :]

    # Apply z range constraint.
    z_values = heights + z_offset
    z_valid = (z_values >= cfg.z_range[0]) & (z_values <= cfg.z_range[1])
    valid_mask &= z_valid

    valid_indices = np.argwhere(valid_mask)

    if len(valid_indices) == 0:
        # Fallback: return sub-terrain center repeated.
        center_x = num_cols * horizontal_scale / 2.0
        center_y = num_rows * horizontal_scale / 2.0
        center_row = min(num_rows // 2, num_rows - 1)
        center_col = min(num_cols // 2, num_cols - 1)
        center_z = heights[center_row, center_col] + z_offset
        return np.tile([center_x, center_y, center_z], (cfg.num_patches, 1))

    replace = len(valid_indices) < cfg.num_patches
    chosen = rng.choice(len(valid_indices), size=cfg.num_patches, replace=replace)
    selected = valid_indices[chosen]

    x = selected[:, 1] * horizontal_scale
    y = selected[:, 0] * horizontal_scale
    z = heights[selected[:, 0], selected[:, 1]] + z_offset

    return np.stack([x, y, z], axis=-1)


def compute_env_origins_grid(num_envs: int, env_spacing: float) -> np.ndarray:
    """Return ``(num_envs, 3)`` reset XY offsets on a ⌈√N⌉×⌈√N⌉ grid centered at origin."""
    if num_envs <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    if env_spacing <= 0.0:
        return np.zeros((num_envs, 3), dtype=np.float64)
    side = int(np.ceil(np.sqrt(num_envs)))
    ii, jj = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    flat_i = ii.flatten()[:num_envs].astype(np.float64)
    flat_j = jj.flatten()[:num_envs].astype(np.float64)
    origins = np.zeros((num_envs, 3), dtype=np.float64)
    origins[:, 0] = (flat_i - (side - 1) / 2.0) * env_spacing
    origins[:, 1] = (flat_j - (side - 1) / 2.0) * env_spacing
    return origins
