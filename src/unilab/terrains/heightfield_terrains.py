"""Terrains composed of heightfields.

This module provides terrain generation functionality using heightfields,
adapted from the IsaacLab terrain generation system.

References:
    IsaacLab mesh terrain implementation:
    https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/terrains/height_field/hf_terrains.py
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from unilab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainHeightField,
    TerrainOutput,
)
from unilab.terrains.utils import (
    bilinear_resample_grid,
    find_flat_patches_from_heightfield,
)

_MIN_BASE_THICKNESS = 0.01


def _build_heightfield(
    noise: np.ndarray,
    *,
    size: tuple[float, float],
    horizontal_scale: float,
    vertical_scale: float,
    base_thickness_ratio: float,
    z_offset_fn: Callable[[float], float] = lambda max_h: 0.0,
) -> TerrainHeightField:
    elevation_min = int(np.min(noise))
    elevation_max = int(np.max(noise))
    elevation_range = elevation_max - elevation_min if elevation_max != elevation_min else 1
    max_physical_height = elevation_range * vertical_scale
    base_thickness = max(max_physical_height * base_thickness_ratio, _MIN_BASE_THICKNESS)
    z_offset = z_offset_fn(max_physical_height)
    return TerrainHeightField(
        noise=noise.copy(),
        size=size,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        elevation_min=elevation_min,
        elevation_max=elevation_max,
        max_physical_height=max_physical_height,
        base_thickness=base_thickness,
        z_offset=z_offset,
    )


def _compute_flat_patches(
    noise: np.ndarray,
    vertical_scale: float,
    horizontal_scale: float,
    z_offset: float,
    flat_patch_sampling: dict | None,
    rng: np.random.Generator,
) -> dict[str, np.ndarray] | None:
    """Compute flat patches for a heightfield terrain if configured."""
    if flat_patch_sampling is None:
        return None
    physical_heights = (noise.astype(np.float64) - noise.min()) * vertical_scale
    flat_patches: dict[str, np.ndarray] = {}
    for name, patch_cfg in flat_patch_sampling.items():
        flat_patches[name] = find_flat_patches_from_heightfield(
            heights=physical_heights,
            horizontal_scale=horizontal_scale,
            z_offset=z_offset,
            cfg=patch_cfg,
            rng=rng,
        )
    return flat_patches


def _make_terrain_output(
    noise: np.ndarray,
    *,
    size: tuple[float, float],
    horizontal_scale: float,
    vertical_scale: float,
    base_thickness_ratio: float,
    origin: np.ndarray,
    flat_patch_sampling: dict | None,
    rng: np.random.Generator,
    z_offset_fn: Callable[[float], float] = lambda max_h: 0.0,
) -> TerrainOutput:
    terrain_hfield = _build_heightfield(
        noise,
        size=size,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        base_thickness_ratio=base_thickness_ratio,
        z_offset_fn=z_offset_fn,
    )

    flat_patches = _compute_flat_patches(
        noise,
        vertical_scale,
        horizontal_scale,
        terrain_hfield.z_offset,
        flat_patch_sampling,
        rng,
    )
    return TerrainOutput(
        origin=origin,
        heightfield=terrain_hfield,
        flat_patches=flat_patches,
    )


@dataclass(kw_only=True)
class HfPyramidSlopedTerrainCfg(SubTerrainCfg):
    slope_range: tuple[float, float]
    """Range of slope gradients (rise / run), interpolated by difficulty."""
    platform_width: float = 1.0
    """Side length of the flat square platform at the terrain center, in meters."""
    inverted: bool = False
    """If True, the pyramid is inverted so the platform is at the bottom."""
    border_width: float = 0.0
    """Width of the flat border around the terrain edges, in meters. Must be >=
    horizontal_scale if non-zero."""
    horizontal_scale: float = 0.1
    """Heightfield grid resolution along x and y, in meters per cell."""
    vertical_scale: float = 0.005
    """Heightfield height resolution, in meters per integer unit of the noise array."""
    base_thickness_ratio: float = 1.0
    """Ratio of the heightfield base thickness to its maximum surface height."""

    def function(self, difficulty: float, rng: np.random.Generator) -> TerrainOutput:
        if self.inverted:
            slope = -self.slope_range[0] - difficulty * (self.slope_range[1] - self.slope_range[0])
        else:
            slope = self.slope_range[0] + difficulty * (self.slope_range[1] - self.slope_range[0])

        if self.border_width > 0 and self.border_width < self.horizontal_scale:
            raise ValueError(
                f"Border width ({self.border_width}) must be >= horizontal scale "
                f"({self.horizontal_scale})"
            )

        border_pixels = int(self.border_width / self.horizontal_scale)
        width_pixels = int(self.size[0] / self.horizontal_scale)
        length_pixels = int(self.size[1] / self.horizontal_scale)

        inner_width_pixels = width_pixels - 2 * border_pixels
        inner_length_pixels = length_pixels - 2 * border_pixels

        noise = np.zeros((width_pixels, length_pixels), dtype=np.int16)

        if border_pixels > 0:
            height_max = int(
                slope * (inner_width_pixels * self.horizontal_scale) / 2 / self.vertical_scale
            )

            center_x = int(inner_width_pixels / 2)
            center_y = int(inner_length_pixels / 2)

            x = np.arange(0, inner_width_pixels)
            y = np.arange(0, inner_length_pixels)
            xx, yy = np.meshgrid(x, y, sparse=True)

            xx = (center_x - np.abs(center_x - xx)) / center_x
            yy = (center_y - np.abs(center_y - yy)) / center_y

            xx = xx.reshape(inner_width_pixels, 1)
            yy = yy.reshape(1, inner_length_pixels)

            hf_raw = height_max * xx * yy

            platform_width = int(self.platform_width / self.horizontal_scale / 2)
            x_pf = inner_width_pixels // 2 - platform_width
            y_pf = inner_length_pixels // 2 - platform_width
            z_pf = hf_raw[x_pf, y_pf] if x_pf >= 0 and y_pf >= 0 else 0
            hf_raw = np.clip(hf_raw, min(0, z_pf), max(0, z_pf))

            noise[
                border_pixels : -border_pixels if border_pixels else width_pixels,
                border_pixels : -border_pixels if border_pixels else length_pixels,
            ] = np.rint(hf_raw).astype(np.int16)
        else:
            height_max = int(slope * self.size[0] / 2 / self.vertical_scale)

            center_x = int(width_pixels / 2)
            center_y = int(length_pixels / 2)

            x = np.arange(0, width_pixels)
            y = np.arange(0, length_pixels)
            xx, yy = np.meshgrid(x, y, sparse=True)

            xx = (center_x - np.abs(center_x - xx)) / center_x
            yy = (center_y - np.abs(center_y - yy)) / center_y

            xx = xx.reshape(width_pixels, 1)
            yy = yy.reshape(1, length_pixels)

            hf_raw = height_max * xx * yy

            platform_width = int(self.platform_width / self.horizontal_scale / 2)
            x_pf = width_pixels // 2 - platform_width
            y_pf = length_pixels // 2 - platform_width
            z_pf = hf_raw[x_pf, y_pf]
            hf_raw = np.clip(hf_raw, min(0, z_pf), max(0, z_pf))

            noise = np.rint(hf_raw).astype(np.int16)

        z_offset_fn = (lambda max_h: -max_h) if self.inverted else (lambda max_h: 0.0)
        terrain_hfield = _build_heightfield(
            noise,
            size=self.size,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_thickness_ratio=self.base_thickness_ratio,
            z_offset_fn=z_offset_fn,
        )

        spawn_height = (
            terrain_hfield.z_offset if self.inverted else terrain_hfield.max_physical_height
        )
        origin = np.array([self.size[0] / 2, self.size[1] / 2, spawn_height])

        return _make_terrain_output(
            noise,
            size=self.size,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_thickness_ratio=self.base_thickness_ratio,
            origin=origin,
            flat_patch_sampling=self.flat_patch_sampling,
            rng=rng,
            z_offset_fn=z_offset_fn,
        )


@dataclass(kw_only=True)
class HfRandomUniformTerrainCfg(SubTerrainCfg):
    noise_range: tuple[float, float]
    """Min and max height noise, in meters."""
    noise_step: float = 0.005
    """Height quantization step, in meters. Sampled heights are multiples of this
    value within noise_range."""
    downsampled_scale: float | None = None
    """Spacing between randomly sampled height points before interpolation, in
    meters. If None, uses horizontal_scale. Must be >= horizontal_scale."""
    horizontal_scale: float = 0.1
    """Heightfield grid resolution along x and y, in meters per cell."""
    vertical_scale: float = 0.005
    """Heightfield height resolution, in meters per integer unit of the noise array."""
    base_thickness_ratio: float = 1.0
    """Ratio of the heightfield base thickness to its maximum surface height."""
    border_width: float = 0.0
    """Width of the flat border around the terrain edges, in meters. Must be >=
    horizontal_scale if non-zero."""

    def function(self, difficulty: float, rng: np.random.Generator) -> TerrainOutput:
        del difficulty  # Unused.

        if self.border_width > 0 and self.border_width < self.horizontal_scale:
            raise ValueError(
                f"Border width ({self.border_width}) must be >= horizontal scale "
                f"({self.horizontal_scale})"
            )

        if self.downsampled_scale is None:
            downsampled_scale = self.horizontal_scale
        elif self.downsampled_scale < self.horizontal_scale:
            raise ValueError(
                f"Downsampled scale must be >= horizontal scale: "
                f"{self.downsampled_scale} < {self.horizontal_scale}"
            )
        else:
            downsampled_scale = self.downsampled_scale

        border_pixels = int(self.border_width / self.horizontal_scale)
        width_pixels = int(self.size[0] / self.horizontal_scale)
        length_pixels = int(self.size[1] / self.horizontal_scale)

        noise = np.zeros((width_pixels, length_pixels), dtype=np.int16)

        if border_pixels > 0:
            inner_width_pixels = width_pixels - 2 * border_pixels
            inner_length_pixels = length_pixels - 2 * border_pixels
            inner_size = (
                inner_width_pixels * self.horizontal_scale,
                inner_length_pixels * self.horizontal_scale,
            )

            width_downsampled = int(inner_size[0] / downsampled_scale)
            length_downsampled = int(inner_size[1] / downsampled_scale)

            height_min = int(self.noise_range[0] / self.vertical_scale)
            height_max = int(self.noise_range[1] / self.vertical_scale)
            height_step = int(self.noise_step / self.vertical_scale)

            height_range = np.arange(height_min, height_max + height_step, height_step)
            height_field_downsampled = rng.choice(
                height_range, size=(width_downsampled, length_downsampled)
            )

            x = np.linspace(0, inner_size[0], width_downsampled)
            y = np.linspace(0, inner_size[1], length_downsampled)

            x_upsampled = np.linspace(0, inner_size[0], inner_width_pixels)
            y_upsampled = np.linspace(0, inner_size[1], inner_length_pixels)
            z_upsampled = bilinear_resample_grid(
                x, y, height_field_downsampled, x_upsampled, y_upsampled
            )

            noise[
                border_pixels : -border_pixels if border_pixels else width_pixels,
                border_pixels : -border_pixels if border_pixels else length_pixels,
            ] = np.rint(z_upsampled).astype(np.int16)
        else:
            width_downsampled = int(self.size[0] / downsampled_scale)
            length_downsampled = int(self.size[1] / downsampled_scale)
            height_min = int(self.noise_range[0] / self.vertical_scale)
            height_max = int(self.noise_range[1] / self.vertical_scale)
            height_step = int(self.noise_step / self.vertical_scale)

            height_range = np.arange(height_min, height_max + height_step, height_step)
            height_field_downsampled = rng.choice(
                height_range, size=(width_downsampled, length_downsampled)
            )

            x = np.linspace(0, self.size[0], width_downsampled)
            y = np.linspace(0, self.size[1], length_downsampled)

            x_upsampled = np.linspace(0, self.size[0], width_pixels)
            y_upsampled = np.linspace(0, self.size[1], length_pixels)
            z_upsampled = bilinear_resample_grid(
                x, y, height_field_downsampled, x_upsampled, y_upsampled
            )
            noise = np.rint(z_upsampled).astype(np.int16)

        spawn_height = (self.noise_range[0] + self.noise_range[1]) / 2
        origin = np.array([self.size[0] / 2, self.size[1] / 2, spawn_height])

        return _make_terrain_output(
            noise,
            size=self.size,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_thickness_ratio=self.base_thickness_ratio,
            origin=origin,
            flat_patch_sampling=self.flat_patch_sampling,
            rng=rng,
        )


@dataclass(kw_only=True)
class HfWaveTerrainCfg(SubTerrainCfg):
    amplitude_range: tuple[float, float]
    """Min and max wave amplitude, in meters. Interpolated by difficulty."""
    num_waves: int = 1
    """Number of complete wave cycles along the terrain length."""
    horizontal_scale: float = 0.1
    """Heightfield grid resolution along x and y, in meters per cell."""
    vertical_scale: float = 0.005
    """Heightfield height resolution, in meters per integer unit of the noise array."""
    base_thickness_ratio: float = 0.25
    """Ratio of the heightfield base thickness to its maximum surface height."""
    border_width: float = 0.0
    """Width of the flat border around the terrain edges, in meters. Must be >=
    horizontal_scale if non-zero."""

    def function(self, difficulty: float, rng: np.random.Generator) -> TerrainOutput:
        if self.num_waves <= 0:
            raise ValueError(f"Number of waves must be positive. Got: {self.num_waves}")

        if self.border_width > 0 and self.border_width < self.horizontal_scale:
            raise ValueError(
                f"Border width ({self.border_width}) must be >= horizontal scale "
                f"({self.horizontal_scale})"
            )

        amplitude = self.amplitude_range[0] + difficulty * (
            self.amplitude_range[1] - self.amplitude_range[0]
        )

        border_pixels = int(self.border_width / self.horizontal_scale)
        width_pixels = int(self.size[0] / self.horizontal_scale)
        length_pixels = int(self.size[1] / self.horizontal_scale)

        noise = np.zeros((width_pixels, length_pixels), dtype=np.int16)

        if border_pixels > 0:
            inner_width_pixels = width_pixels - 2 * border_pixels
            inner_length_pixels = length_pixels - 2 * border_pixels

            amplitude_pixels = int(0.5 * amplitude / self.vertical_scale)
            wave_length = inner_length_pixels / self.num_waves
            wave_number = 2 * np.pi / wave_length

            x = np.arange(0, inner_width_pixels)
            y = np.arange(0, inner_length_pixels)
            xx, yy = np.meshgrid(x, y, sparse=True)
            xx = xx.reshape(inner_width_pixels, 1)
            yy = yy.reshape(1, inner_length_pixels)

            hf_raw = amplitude_pixels * (np.cos(yy * wave_number) + np.sin(xx * wave_number))

            noise[
                border_pixels : -border_pixels if border_pixels else width_pixels,
                border_pixels : -border_pixels if border_pixels else length_pixels,
            ] = np.rint(hf_raw).astype(np.int16)
        else:
            amplitude_pixels = int(0.5 * amplitude / self.vertical_scale)
            wave_length = length_pixels / self.num_waves
            wave_number = 2 * np.pi / wave_length

            x = np.arange(0, width_pixels)
            y = np.arange(0, length_pixels)
            xx, yy = np.meshgrid(x, y, sparse=True)
            xx = xx.reshape(width_pixels, 1)
            yy = yy.reshape(1, length_pixels)

            hf_raw = amplitude_pixels * (np.cos(yy * wave_number) + np.sin(xx * wave_number))
            noise = np.rint(hf_raw).astype(np.int16)

        spawn_height = 0.0
        origin = np.array([self.size[0] / 2, self.size[1] / 2, spawn_height])

        return _make_terrain_output(
            noise,
            size=self.size,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_thickness_ratio=self.base_thickness_ratio,
            origin=origin,
            flat_patch_sampling=self.flat_patch_sampling,
            rng=rng,
            z_offset_fn=lambda max_h: -max_h / 2,
        )


@dataclass(kw_only=True)
class HfPyramidStairsTerrainCfg(SubTerrainCfg):
    """A pyramid stairs terrain encoded as a heightfield.

    Concentric square rings from the outside in form a staircase climbing toward
    a central platform. With ``holes=True`` the four diagonal corners of each
    ring are carved out to a deep pit; agents falling into the pit reach a
    terminating depth instead of an infinite void.
    """

    step_height_range: tuple[float, float]
    """Min and max step height, in meters. Interpolated by difficulty."""
    step_width: float
    """Depth (run) of each step, in meters. Must be a multiple of horizontal_scale."""
    platform_width: float = 1.0
    """Side length of the flat square platform at the top of the staircase, in meters."""
    border_width: float = 0.0
    """Width of the flat outer border around the staircase, in meters."""
    holes: bool = False
    """If True, carve deep pits at the diagonal corners of each step ring."""
    pit_depth: float = 5.0
    """Depth of holes-mode pits below the lowest stair, in meters."""
    horizontal_scale: float = 0.05
    """Heightfield grid resolution. Overwritten by TerrainGenerator."""
    vertical_scale: float = 0.005
    """Heightfield height resolution. Overwritten by TerrainGenerator."""
    base_thickness_ratio: float = 1.0
    """Ratio of the heightfield base thickness to its surface height."""

    def function(self, difficulty: float, rng: np.random.Generator) -> TerrainOutput:
        step_height = self.step_height_range[0] + difficulty * (
            self.step_height_range[1] - self.step_height_range[0]
        )

        W = int(round(self.size[0] / self.horizontal_scale))
        L = int(round(self.size[1] / self.horizontal_scale))
        step_px = int(round(self.step_width / self.horizontal_scale))
        plat_px = int(round(self.platform_width / self.horizontal_scale))
        border_px = int(round(self.border_width / self.horizontal_scale))
        step_units = int(round(step_height / self.vertical_scale))

        noise = np.zeros((W, L), dtype=np.int16)

        inner = min(W, L) - 2 * border_px
        n_steps = max(0, (inner - plat_px) // (2 * step_px)) if step_px > 0 else 0

        for k in range(n_steps):
            lo_x = border_px + k * step_px
            hi_x = W - border_px - k * step_px
            lo_y = border_px + k * step_px
            hi_y = L - border_px - k * step_px
            noise[lo_x:hi_x, lo_y:hi_y] = (k + 1) * step_units

        plat_lo_x = border_px + n_steps * step_px
        plat_hi_x = W - border_px - n_steps * step_px
        plat_lo_y = border_px + n_steps * step_px
        plat_hi_y = L - border_px - n_steps * step_px
        noise[plat_lo_x:plat_hi_x, plat_lo_y:plat_hi_y] = (n_steps + 1) * step_units

        if self.holes:
            pit_units = -int(round(self.pit_depth / self.vertical_scale))
            if pit_units < np.iinfo(np.int16).min:
                raise ValueError(
                    f"pit_depth={self.pit_depth} m at vertical_scale="
                    f"{self.vertical_scale} m overflows int16 range."
                )
            for k in range(n_steps):
                lo_x = border_px + k * step_px
                hi_x = W - border_px - k * step_px
                lo_y = border_px + k * step_px
                hi_y = L - border_px - k * step_px
                # Four outer corner squares of this ring.
                noise[lo_x : lo_x + step_px, lo_y : lo_y + step_px] = pit_units
                noise[lo_x : lo_x + step_px, hi_y - step_px : hi_y] = pit_units
                noise[hi_x - step_px : hi_x, lo_y : lo_y + step_px] = pit_units
                noise[hi_x - step_px : hi_x, hi_y - step_px : hi_y] = pit_units

        spawn_z = (n_steps + 1) * step_units * self.vertical_scale
        origin = np.array([self.size[0] / 2, self.size[1] / 2, spawn_z])

        return _make_terrain_output(
            noise,
            size=self.size,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_thickness_ratio=self.base_thickness_ratio,
            origin=origin,
            flat_patch_sampling=self.flat_patch_sampling,
            rng=rng,
        )


@dataclass(kw_only=True)
class HfInvertedPyramidStairsTerrainCfg(HfPyramidStairsTerrainCfg):
    """A pit-style pyramid stairs terrain encoded as a heightfield.

    Inverts :class:`HfPyramidStairsTerrainCfg`: outer ring sits at world z=0,
    rings descend toward a central platform at the bottom. With ``holes=True``
    the diagonal corners are even deeper than the platform.
    """

    def function(self, difficulty: float, rng: np.random.Generator) -> TerrainOutput:
        step_height = self.step_height_range[0] + difficulty * (
            self.step_height_range[1] - self.step_height_range[0]
        )

        W = int(round(self.size[0] / self.horizontal_scale))
        L = int(round(self.size[1] / self.horizontal_scale))
        step_px = int(round(self.step_width / self.horizontal_scale))
        plat_px = int(round(self.platform_width / self.horizontal_scale))
        border_px = int(round(self.border_width / self.horizontal_scale))
        step_units = int(round(step_height / self.vertical_scale))

        noise = np.zeros((W, L), dtype=np.int16)

        inner = min(W, L) - 2 * border_px
        n_steps = max(0, (inner - plat_px) // (2 * step_px)) if step_px > 0 else 0

        # Rings descend (negative values) from outer to inner.
        for k in range(n_steps):
            lo_x = border_px + k * step_px
            hi_x = W - border_px - k * step_px
            lo_y = border_px + k * step_px
            hi_y = L - border_px - k * step_px
            noise[lo_x:hi_x, lo_y:hi_y] = -(k + 1) * step_units

        plat_lo_x = border_px + n_steps * step_px
        plat_hi_x = W - border_px - n_steps * step_px
        plat_lo_y = border_px + n_steps * step_px
        plat_hi_y = L - border_px - n_steps * step_px
        noise[plat_lo_x:plat_hi_x, plat_lo_y:plat_hi_y] = -(n_steps + 1) * step_units

        if self.holes:
            min_existing = int(noise.min())
            pit_units = min_existing - int(round(self.pit_depth / self.vertical_scale))
            if pit_units < np.iinfo(np.int16).min:
                raise ValueError(
                    f"pit_depth={self.pit_depth} m below platform at vertical_scale="
                    f"{self.vertical_scale} m overflows int16 range."
                )
            for k in range(n_steps):
                lo_x = border_px + k * step_px
                hi_x = W - border_px - k * step_px
                lo_y = border_px + k * step_px
                hi_y = L - border_px - k * step_px
                noise[lo_x : lo_x + step_px, lo_y : lo_y + step_px] = pit_units
                noise[lo_x : lo_x + step_px, hi_y - step_px : hi_y] = pit_units
                noise[hi_x - step_px : hi_x, lo_y : lo_y + step_px] = pit_units
                noise[hi_x - step_px : hi_x, hi_y - step_px : hi_y] = pit_units

        # Place data such that the original "0" layer (outer ring top) sits at
        # world z=0 — matches the existing HfPyramidSloped(inverted) convention.
        spawn_z = -(n_steps + 1) * step_units * self.vertical_scale
        origin = np.array([self.size[0] / 2, self.size[1] / 2, spawn_z])

        return _make_terrain_output(
            noise,
            size=self.size,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_thickness_ratio=self.base_thickness_ratio,
            origin=origin,
            flat_patch_sampling=self.flat_patch_sampling,
            rng=rng,
            z_offset_fn=lambda max_h: -max_h,
        )


@dataclass(kw_only=True)
class HfFlatTerrainCfg(SubTerrainCfg):
    """A flat heightfield terrain (all-zero noise array)."""

    horizontal_scale: float = 0.05
    """Heightfield grid resolution. Overwritten by TerrainGenerator."""
    vertical_scale: float = 0.005
    """Heightfield height resolution. Overwritten by TerrainGenerator."""
    base_thickness_ratio: float = 0.0
    """Ratio of the heightfield base thickness to its surface height. The
    helper enforces a minimum thickness so a literal zero is fine here."""

    def function(self, difficulty: float, rng: np.random.Generator) -> TerrainOutput:
        del difficulty  # Unused.

        width_pixels = int(round(self.size[0] / self.horizontal_scale))
        length_pixels = int(round(self.size[1] / self.horizontal_scale))
        noise = np.zeros((width_pixels, length_pixels), dtype=np.int16)

        origin = np.array([self.size[0] / 2, self.size[1] / 2, 0.0])
        return _make_terrain_output(
            noise,
            size=self.size,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_thickness_ratio=self.base_thickness_ratio,
            origin=origin,
            flat_patch_sampling=self.flat_patch_sampling,
            rng=rng,
        )
