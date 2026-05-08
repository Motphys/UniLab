"""Procedural terrain generation.

Ported from mjlab (https://github.com/mjlab/mjlab) for unilab issues #197 / #270.
The terrain generator builds a grid of difficulty-graded sub-terrains and writes
geoms / heightfields into a MuJoCo MjSpec at cold path. See
:mod:`unilab.scene.composer` for the materializer that turns a base XML plus a
TerrainGeneratorCfg into a final scene.xml + assets directory.
"""

from unilab.terrains.config import (
    ALL_TERRAIN_PRESETS,
    ROUGH_TERRAINS_CFG,
    STAIRS_TERRAINS_CFG,
    flat,
    hf_pyramid_slope,
    hf_pyramid_slope_inv,
    pyramid_stairs,
    pyramid_stairs_inv,
    random_rough,
    terrain_preset,
    wave_terrain,
)
from unilab.terrains.heightfield_terrains import (
    HfPyramidSlopedTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfWaveTerrainCfg,
)
from unilab.terrains.primitive_terrains import (
    BoxFlatTerrainCfg,
    BoxInvertedPyramidStairsTerrainCfg,
    BoxPyramidStairsTerrainCfg,
)
from unilab.terrains.terrain_generator import (
    FlatPatchSamplingCfg,
    SubTerrainCfg,
    TerrainGenerator,
    TerrainGeneratorCfg,
    TerrainGeometry,
    TerrainOutput,
)
from unilab.terrains.utils import compute_env_origins_grid

__all__ = [
    "ALL_TERRAIN_PRESETS",
    "BoxFlatTerrainCfg",
    "BoxInvertedPyramidStairsTerrainCfg",
    "BoxPyramidStairsTerrainCfg",
    "FlatPatchSamplingCfg",
    "HfPyramidSlopedTerrainCfg",
    "HfRandomUniformTerrainCfg",
    "HfWaveTerrainCfg",
    "ROUGH_TERRAINS_CFG",
    "STAIRS_TERRAINS_CFG",
    "SubTerrainCfg",
    "TerrainGenerator",
    "TerrainGeneratorCfg",
    "TerrainGeometry",
    "TerrainOutput",
    "compute_env_origins_grid",
    "flat",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "random_rough",
    "terrain_preset",
    "wave_terrain",
]
