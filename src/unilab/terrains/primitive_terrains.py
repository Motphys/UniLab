"""Terrains composed of primitive geometries.

This module provides terrain generation functionality using primitive geometries,
adapted from the IsaacLab terrain generation system.

References:
    IsaacLab mesh terrain implementation:
    https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/terrains/trimesh/mesh_terrains.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import numpy as np

from unilab.terrains._color import (
    HSV,
    brand_ramp,
    clamp,
    darken_rgba,
    hsv_to_rgb,
    rgb_to_hsv,
)
from unilab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeometry,
    TerrainOutput,
)
from unilab.terrains.utils import make_border, make_plane

if TYPE_CHECKING:
    import mujoco

_MUJOCO_BLUE = (0.20, 0.45, 0.95)
_MUJOCO_RED = (0.90, 0.30, 0.30)
_MUJOCO_GREEN = (0.25, 0.80, 0.45)


def _get_platform_color(
    base_rgb: Tuple[float, float, float],
    desaturation_factor: float = 0.4,
    lightening_factor: float = 0.25,
) -> Tuple[float, float, float, float]:
    hsv = rgb_to_hsv(base_rgb)
    new_s = hsv.s * desaturation_factor
    new_v = clamp(hsv.v + lightening_factor)
    new_hsv = HSV(hsv.h, new_s, new_v)
    r, g, b = hsv_to_rgb(new_hsv)
    return (r, g, b, 1.0)


@dataclass(kw_only=True)
class BoxFlatTerrainCfg(SubTerrainCfg):
    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del difficulty, rng  # Unused.
        body = spec.body("terrain")
        origin = (self.size[0] / 2, self.size[1] / 2, 0.0)
        boxes = make_plane(body, self.size, 0.0, center_zero=False)
        box_colors = [(0.5, 0.5, 0.5, 1.0)]
        geometry = TerrainGeometry(geom=boxes[0], color=box_colors[0])
        return TerrainOutput(origin=np.array(origin), geometries=[geometry])


@dataclass(kw_only=True)
class BoxPyramidStairsTerrainCfg(SubTerrainCfg):
    """Configuration for a pyramid stairs terrain."""

    border_width: float = 0.0
    """Width of the flat border frame around the staircase, in meters. Ignored
    when holes is True."""
    step_height_range: tuple[float, float]
    """Min and max step height, in meters. Interpolated by difficulty."""
    step_width: float
    """Depth (run) of each step, in meters."""
    platform_width: float = 1.0
    """Side length of the flat square platform at the top of the staircase, in meters."""
    holes: bool = False
    """If True, steps form a cross pattern with empty gaps in the corners."""

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        import mujoco

        del rng  # Unused.
        boxes = []
        box_colors = []

        body = spec.body("terrain")

        step_height = self.step_height_range[0] + difficulty * (
            self.step_height_range[1] - self.step_height_range[0]
        )

        # Compute number of steps in x and y direction.
        num_steps_x = int(
            (self.size[0] - 2 * self.border_width - self.platform_width) / (2 * self.step_width)
        )
        num_steps_y = int(
            (self.size[1] - 2 * self.border_width - self.platform_width) / (2 * self.step_width)
        )
        num_steps = max(0, int(min(num_steps_x, num_steps_y)))

        first_step_rgba = brand_ramp(_MUJOCO_BLUE, 0.0)
        border_rgba = darken_rgba(first_step_rgba, 0.85)

        if self.border_width > 0.0 and not self.holes:
            border_center = (0.5 * self.size[0], 0.5 * self.size[1], -step_height / 2)
            border_inner_size = (
                self.size[0] - 2 * self.border_width,
                self.size[1] - 2 * self.border_width,
            )
            border_boxes = make_border(
                body, self.size, border_inner_size, step_height, border_center
            )
            boxes.extend(border_boxes)
            for _ in range(len(border_boxes)):
                box_colors.append(border_rgba)

        terrain_center = [0.5 * self.size[0], 0.5 * self.size[1], 0.0]
        terrain_size = (
            self.size[0] - 2 * self.border_width,
            self.size[1] - 2 * self.border_width,
        )
        rgba = brand_ramp(_MUJOCO_BLUE, 0.5)
        for k in range(num_steps):
            t = k / max(num_steps - 1, 1)
            rgba = brand_ramp(_MUJOCO_BLUE, t)
            for _ in range(4):
                box_colors.append(rgba)

            if self.holes:
                box_size = (self.platform_width, self.platform_width)
            else:
                box_size = (
                    terrain_size[0] - 2 * k * self.step_width,
                    terrain_size[1] - 2 * k * self.step_width,
                )
            box_z = terrain_center[2] + k * step_height / 2.0
            box_offset = (k + 0.5) * self.step_width
            box_height = (k + 2) * step_height

            box_dims = (box_size[0], self.step_width, box_height)

            safe_size = (
                np.maximum(1e-6, box_dims[0] / 2.0),
                np.maximum(1e-6, box_dims[1] / 2.0),
                np.maximum(1e-6, box_dims[2] / 2.0),
            )

            # Top.
            box_pos = (
                terrain_center[0],
                terrain_center[1] + terrain_size[1] / 2.0 - box_offset,
                box_z,
            )
            box = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=safe_size,
                pos=box_pos,
            )
            boxes.append(box)

            # Bottom.
            box_pos = (
                terrain_center[0],
                terrain_center[1] - terrain_size[1] / 2.0 + box_offset,
                box_z,
            )
            box = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=safe_size,
                pos=box_pos,
            )
            boxes.append(box)

            if self.holes:
                box_dims = (self.step_width, box_size[1], box_height)
            else:
                box_dims = (
                    self.step_width,
                    box_size[1] - 2 * self.step_width,
                    box_height,
                )
            safe_size = (
                np.maximum(1e-6, box_dims[0] / 2.0),
                np.maximum(1e-6, box_dims[1] / 2.0),
                np.maximum(1e-6, box_dims[2] / 2.0),
            )

            # Right.
            box_pos = (
                terrain_center[0] + terrain_size[0] / 2.0 - box_offset,
                terrain_center[1],
                box_z,
            )
            box = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=safe_size,
                pos=box_pos,
            )
            boxes.append(box)

            # Left.
            box_pos = (
                terrain_center[0] - terrain_size[0] / 2.0 + box_offset,
                terrain_center[1],
                box_z,
            )
            box = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=safe_size,
                pos=box_pos,
            )
            boxes.append(box)

        # Generate final box for the middle of the terrain.
        box_dims = (
            terrain_size[0] - 2 * num_steps * self.step_width,
            terrain_size[1] - 2 * num_steps * self.step_width,
            (num_steps + 2) * step_height,
        )
        box_pos = (
            terrain_center[0],
            terrain_center[1],
            terrain_center[2] + num_steps * step_height / 2,
        )
        box = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(
                np.maximum(1e-6, box_dims[0] / 2.0),
                np.maximum(1e-6, box_dims[1] / 2.0),
                np.maximum(1e-6, box_dims[2] / 2.0),
            ),
            pos=box_pos,
        )
        boxes.append(box)
        origin = np.array([terrain_center[0], terrain_center[1], (num_steps + 1) * step_height])
        box_colors.append(rgba)

        geometries = [
            TerrainGeometry(geom=box, color=color)
            for box, color in zip(boxes, box_colors, strict=True)
        ]
        return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class BoxInvertedPyramidStairsTerrainCfg(BoxPyramidStairsTerrainCfg):
    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        import mujoco

        del rng  # Unused.
        boxes = []
        box_colors = []

        body = spec.body("terrain")

        step_height = self.step_height_range[0] + difficulty * (
            self.step_height_range[1] - self.step_height_range[0]
        )

        # Compute number of steps in x and y direction.
        num_steps_x = int(
            (self.size[0] - 2 * self.border_width - self.platform_width) / (2 * self.step_width)
        )
        num_steps_y = int(
            (self.size[1] - 2 * self.border_width - self.platform_width) / (2 * self.step_width)
        )
        num_steps = max(0, int(min(num_steps_x, num_steps_y)))
        total_height = (num_steps + 1) * step_height

        first_step_rgba = brand_ramp(_MUJOCO_RED, 0.0)
        border_rgba = darken_rgba(first_step_rgba, 0.85)

        if self.border_width > 0.0 and not self.holes:
            border_center = (0.5 * self.size[0], 0.5 * self.size[1], -0.5 * step_height)
            border_inner_size = (
                self.size[0] - 2 * self.border_width,
                self.size[1] - 2 * self.border_width,
            )
            border_boxes = make_border(
                body, self.size, border_inner_size, step_height, border_center
            )
            boxes.extend(border_boxes)
            for _ in range(len(border_boxes)):
                box_colors.append(border_rgba)

        terrain_center = [0.5 * self.size[0], 0.5 * self.size[1], 0.0]
        terrain_size = (
            self.size[0] - 2 * self.border_width,
            self.size[1] - 2 * self.border_width,
        )

        rgba = brand_ramp(_MUJOCO_RED, 0.5)
        for k in range(num_steps):
            t = k / max(num_steps - 1, 1)
            rgba = brand_ramp(_MUJOCO_RED, t)
            for _ in range(4):
                box_colors.append(rgba)

            if self.holes:
                box_size = (self.platform_width, self.platform_width)
            else:
                box_size = (
                    terrain_size[0] - 2 * k * self.step_width,
                    terrain_size[1] - 2 * k * self.step_width,
                )

            box_z = terrain_center[2] - total_height / 2 - (k + 1) * step_height / 2.0
            box_offset = (k + 0.5) * self.step_width
            box_height = total_height - (k + 1) * step_height

            box_dims = (box_size[0], self.step_width, box_height)
            safe_size = (
                np.maximum(1e-6, box_dims[0] / 2.0),
                np.maximum(1e-6, box_dims[1] / 2.0),
                np.maximum(1e-6, box_dims[2] / 2.0),
            )

            # Top.
            box_pos = (
                terrain_center[0],
                terrain_center[1] + terrain_size[1] / 2.0 - box_offset,
                box_z,
            )
            box = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=safe_size,
                pos=box_pos,
            )
            boxes.append(box)

            # Bottom.
            box_pos = (
                terrain_center[0],
                terrain_center[1] - terrain_size[1] / 2.0 + box_offset,
                box_z,
            )
            box = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=safe_size,
                pos=box_pos,
            )
            boxes.append(box)

            if self.holes:
                box_dims = (self.step_width, box_size[1], box_height)
            else:
                box_dims = (
                    self.step_width,
                    box_size[1] - 2 * self.step_width,
                    box_height,
                )
            safe_size = (
                np.maximum(1e-6, box_dims[0] / 2.0),
                np.maximum(1e-6, box_dims[1] / 2.0),
                np.maximum(1e-6, box_dims[2] / 2.0),
            )

            # Right.
            box_pos = (
                terrain_center[0] + terrain_size[0] / 2.0 - box_offset,
                terrain_center[1],
                box_z,
            )
            box = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=safe_size,
                pos=box_pos,
            )
            boxes.append(box)

            # Left.
            box_pos = (
                terrain_center[0] - terrain_size[0] / 2.0 + box_offset,
                terrain_center[1],
                box_z,
            )
            box = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=safe_size,
                pos=box_pos,
            )
            boxes.append(box)

        # Generate final box for the middle of the terrain.
        box_dims = (
            terrain_size[0] - 2 * num_steps * self.step_width,
            terrain_size[1] - 2 * num_steps * self.step_width,
            step_height,
        )
        box_pos = (
            terrain_center[0],
            terrain_center[1],
            terrain_center[2] - total_height - step_height / 2,
        )
        box = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(
                np.maximum(1e-6, box_dims[0] / 2.0),
                np.maximum(1e-6, box_dims[1] / 2.0),
                np.maximum(1e-6, box_dims[2] / 2.0),
            ),
            pos=box_pos,
        )
        boxes.append(box)
        origin = np.array([terrain_center[0], terrain_center[1], -(num_steps + 1) * step_height])
        box_colors.append(rgba)

        geometries = [
            TerrainGeometry(geom=box, color=color)
            for box, color in zip(boxes, box_colors, strict=True)
        ]
        return TerrainOutput(origin=origin, geometries=geometries)
