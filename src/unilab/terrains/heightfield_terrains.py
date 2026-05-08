"""Terrains composed of heightfields.

This module provides terrain generation functionality using heightfields,
adapted from the IsaacLab terrain generation system.

References:
    IsaacLab mesh terrain implementation:
    https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/terrains/height_field/hf_terrains.py
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mujoco

from unilab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeometry,
    TerrainOutput,
)
from unilab.terrains.utils import (
    bilinear_resample_grid,
    bilinear_zoom_2d,
    find_flat_patches_from_heightfield,
)


def color_by_height(
    spec: mujoco.MjSpec,
    noise: np.ndarray,
    unique_id: str,
    normalized_elevation: np.ndarray,
    texture_size: int = 128,
) -> str:
    import mujoco

    texture_name = f"hf_texture_{unique_id}"
    texture = spec.add_texture(
        name=texture_name,
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        width=texture_size,
        height=texture_size,
    )

    texture_elevation = bilinear_zoom_2d(
        normalized_elevation,
        (texture_size / noise.shape[0], texture_size / noise.shape[1]),
    )

    hue = 0.5 - texture_elevation * 0.45
    saturation = 0.6 - texture_elevation * 0.2
    value = 0.4 + texture_elevation * 0.3

    c = value * saturation
    x = c * (1 - np.abs((hue * 6) % 2 - 1))
    m = value - c

    hue_sector = (hue * 6).astype(int) % 6

    r = np.zeros_like(hue)
    g = np.zeros_like(hue)
    b = np.zeros_like(hue)

    mask = hue_sector == 0
    r[mask] = c[mask]
    g[mask] = x[mask]

    mask = hue_sector == 1
    r[mask] = x[mask]
    g[mask] = c[mask]

    mask = hue_sector == 2
    g[mask] = c[mask]
    b[mask] = x[mask]

    mask = hue_sector == 3
    g[mask] = x[mask]
    b[mask] = c[mask]

    mask = hue_sector == 4
    r[mask] = x[mask]
    b[mask] = c[mask]

    mask = hue_sector == 5
    r[mask] = c[mask]
    b[mask] = x[mask]

    r += m
    g += m
    b += m

    rgb_data = np.stack([r, g, b], axis=-1)
    rgb_data = (rgb_data * 255).astype(np.uint8)

    rgb_data = np.flipud(rgb_data)
    texture.data = rgb_data.tobytes()

    material_name = f"hf_material_{unique_id}"
    material = spec.add_material(name=material_name)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture_name

    return material_name


def _fractal_perlin_noise_2d(
    x_size: int,
    y_size: int,
    rng: np.random.Generator,
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Generate 2D fractal Perlin noise."""

    def lerp(a, b, x):
        return a + x * (b - a)

    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def gradient(h, x, y):
        h = h % 4
        return np.where(
            h == 0,
            x + y,
            np.where(h == 1, x - y, np.where(h == 2, -x + y, -x - y)),
        )

    def perlin(x, y, p):
        xi = x.astype(int) % 256
        yi = y.astype(int) % 256
        xf = x - x.astype(int)
        yf = y - y.astype(int)
        u = fade(xf)
        v = fade(yf)

        n00 = gradient(p[p[xi] + yi], xf, yf)
        n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
        n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
        n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)

        x1 = lerp(n00, n10, u)
        x2 = lerp(n01, n11, u)
        return lerp(x1, x2, v)

    p = np.arange(256, dtype=int)
    rng.shuffle(p)
    p = np.stack([p, p]).flatten()

    noise = np.zeros((x_size, y_size))
    amplitude = 1.0
    frequency = scale
    total_amplitude = 0.0

    x = np.linspace(0, x_size, x_size, endpoint=False)
    y = np.linspace(0, y_size, y_size, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    for _ in range(octaves):
        noise += amplitude * perlin(xx * frequency / x_size, yy * frequency / y_size, p)
        total_amplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    return noise / total_amplitude


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

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        import mujoco

        body = spec.body("terrain")

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

        elevation_min = np.min(noise)
        elevation_max = np.max(noise)
        elevation_range = elevation_max - elevation_min if elevation_max != elevation_min else 1

        max_physical_height = elevation_range * self.vertical_scale
        base_thickness = max_physical_height * self.base_thickness_ratio

        if elevation_range > 0:
            normalized_elevation = (noise - elevation_min) / elevation_range
        else:
            normalized_elevation = np.zeros_like(noise)

        unique_id = uuid.uuid4().hex
        field = spec.add_hfield(
            name=f"hfield_{unique_id}",
            size=[
                self.size[0] / 2,
                self.size[1] / 2,
                max_physical_height,
                base_thickness,
            ],
            nrow=noise.shape[0],
            ncol=noise.shape[1],
            userdata=normalized_elevation.flatten().astype(np.float32).tolist(),
        )

        if self.inverted:
            hfield_z_offset = -max_physical_height
        else:
            hfield_z_offset = 0

        material_name = color_by_height(spec, noise, unique_id, normalized_elevation)

        hfield_geom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname=field.name,
            pos=[
                self.size[0] / 2,
                self.size[1] / 2,
                hfield_z_offset,
            ],
            material=material_name,
        )

        if self.inverted:
            spawn_height = hfield_z_offset
        else:
            spawn_height = max_physical_height

        origin = np.array([self.size[0] / 2, self.size[1] / 2, spawn_height])

        flat_patches = _compute_flat_patches(
            noise,
            self.vertical_scale,
            self.horizontal_scale,
            hfield_z_offset,
            self.flat_patch_sampling,
            rng,
        )

        geom = TerrainGeometry(geom=hfield_geom, hfield=field)
        return TerrainOutput(origin=origin, geometries=[geom], flat_patches=flat_patches)


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

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        import mujoco

        del difficulty  # Unused.

        body = spec.body("terrain")

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

        elevation_min = np.min(noise)
        elevation_max = np.max(noise)
        elevation_range = elevation_max - elevation_min if elevation_max != elevation_min else 1

        max_physical_height = elevation_range * self.vertical_scale
        base_thickness = max_physical_height * self.base_thickness_ratio

        if elevation_range > 0:
            normalized_elevation = (noise - elevation_min) / elevation_range
        else:
            normalized_elevation = np.zeros_like(noise)

        unique_id = uuid.uuid4().hex
        field = spec.add_hfield(
            name=f"hfield_{unique_id}",
            size=[
                self.size[0] / 2,
                self.size[1] / 2,
                max_physical_height,
                base_thickness,
            ],
            nrow=noise.shape[0],
            ncol=noise.shape[1],
            userdata=normalized_elevation.flatten().astype(np.float32).tolist(),
        )

        material_name = color_by_height(spec, noise, unique_id, normalized_elevation)

        hfield_geom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname=field.name,
            pos=[self.size[0] / 2, self.size[1] / 2, 0],
            material=material_name,
        )

        spawn_height = (self.noise_range[0] + self.noise_range[1]) / 2
        origin = np.array([self.size[0] / 2, self.size[1] / 2, spawn_height])

        flat_patches = _compute_flat_patches(
            noise,
            self.vertical_scale,
            self.horizontal_scale,
            0,
            self.flat_patch_sampling,
            rng,
        )

        geom = TerrainGeometry(geom=hfield_geom, hfield=field)
        return TerrainOutput(origin=origin, geometries=[geom], flat_patches=flat_patches)


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

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        import mujoco

        body = spec.body("terrain")

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

        elevation_min = np.min(noise)
        elevation_max = np.max(noise)
        elevation_range = elevation_max - elevation_min if elevation_max != elevation_min else 1

        max_physical_height = elevation_range * self.vertical_scale
        base_thickness = max_physical_height * self.base_thickness_ratio

        if elevation_range > 0:
            normalized_elevation = (noise - elevation_min) / elevation_range
        else:
            normalized_elevation = np.zeros_like(noise)

        unique_id = uuid.uuid4().hex
        field = spec.add_hfield(
            name=f"hfield_{unique_id}",
            size=[
                self.size[0] / 2,
                self.size[1] / 2,
                max_physical_height,
                base_thickness,
            ],
            nrow=noise.shape[0],
            ncol=noise.shape[1],
            userdata=normalized_elevation.flatten().astype(np.float32).tolist(),
        )

        material_name = color_by_height(spec, noise, unique_id, normalized_elevation)

        hfield_geom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname=field.name,
            pos=[self.size[0] / 2, self.size[1] / 2, -max_physical_height / 2],
            material=material_name,
        )

        spawn_height = 0.0
        origin = np.array([self.size[0] / 2, self.size[1] / 2, spawn_height])

        flat_patches = _compute_flat_patches(
            noise,
            self.vertical_scale,
            self.horizontal_scale,
            -max_physical_height / 2,
            self.flat_patch_sampling,
            rng,
        )

        geom = TerrainGeometry(geom=hfield_geom, hfield=field)
        return TerrainOutput(origin=origin, geometries=[geom], flat_patches=flat_patches)
