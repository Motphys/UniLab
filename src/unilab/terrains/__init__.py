"""Procedural terrain generation.

Ported from mjlab (https://github.com/mjlab/mjlab).
The terrain generator builds a grid of difficulty-graded sub-terrains and writes
a merged heightfield PNG at cold path. The backend XML materializer replaces a
scene template's hfield asset with that generated PNG before model loading.
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
    HfFlatTerrainCfg,
    HfInvertedPyramidStairsTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfPyramidStairsTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfWaveTerrainCfg,
)
from unilab.terrains.terrain_generator import (
    FlatPatchSamplingCfg,
    GeneratedTerrain,
    SubTerrainCfg,
    TerrainGenerator,
    TerrainGeneratorCfg,
    TerrainHeightField,
    TerrainOutput,
)

__all__ = [
    "ALL_TERRAIN_PRESETS",
    "FlatPatchSamplingCfg",
    "GeneratedTerrain",
    "HfFlatTerrainCfg",
    "HfInvertedPyramidStairsTerrainCfg",
    "HfPyramidSlopedTerrainCfg",
    "HfPyramidStairsTerrainCfg",
    "HfRandomUniformTerrainCfg",
    "HfWaveTerrainCfg",
    "ROUGH_TERRAINS_CFG",
    "STAIRS_TERRAINS_CFG",
    "SubTerrainCfg",
    "TerrainGenerator",
    "TerrainGeneratorCfg",
    "TerrainHeightField",
    "TerrainOutput",
    "flat",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "random_rough",
    "terrain_preset",
    "wave_terrain",
]
