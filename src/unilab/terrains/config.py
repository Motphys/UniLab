"""Terrain configuration presets and named terrain sets.

Slim port of mjlab/terrains/config.py — keeps only the 7 sub-terrains used by
ROUGH_TERRAINS_CFG and the curriculum stairs preset.

To add a new terrain, define a function decorated with @terrain_preset::

    @terrain_preset
    def my_new_terrain(**overrides: Any) -> SomeTerrainCfg:
        defaults: dict[str, Any] = dict(...)
        defaults.update(overrides)
        return SomeTerrainCfg(**defaults)

It will be auto-included in ALL_TERRAIN_PRESETS.
"""

from collections.abc import Callable
from typing import Any, TypeVar

from unilab.terrains.heightfield_terrains import (
    HfFlatTerrainCfg,
    HfInvertedPyramidStairsTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfPyramidStairsTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfWaveTerrainCfg,
)
from unilab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg

# Preset registry.

ALL_TERRAIN_PRESETS: dict[str, Callable[..., SubTerrainCfg]] = {}

_F = TypeVar("_F", bound=Callable[..., SubTerrainCfg])


def terrain_preset(fn: _F) -> _F:
    """Register a terrain preset into ALL_TERRAIN_PRESETS."""
    ALL_TERRAIN_PRESETS[fn.__name__] = fn
    return fn


# Terrain presets.


@terrain_preset
def flat(**overrides: Any) -> HfFlatTerrainCfg:
    return HfFlatTerrainCfg(**overrides)


@terrain_preset
def pyramid_stairs(**overrides: Any) -> HfPyramidStairsTerrainCfg:
    defaults: dict[str, Any] = dict(
        step_height_range=(0.0, 0.2),
        step_width=0.3,
        platform_width=3.0,
        border_width=1.0,
    )
    defaults.update(overrides)
    return HfPyramidStairsTerrainCfg(**defaults)


@terrain_preset
def pyramid_stairs_inv(**overrides: Any) -> HfInvertedPyramidStairsTerrainCfg:
    defaults: dict[str, Any] = dict(
        step_height_range=(0.0, 0.2),
        step_width=0.3,
        platform_width=3.0,
        border_width=1.0,
    )
    defaults.update(overrides)
    return HfInvertedPyramidStairsTerrainCfg(**defaults)


@terrain_preset
def hf_pyramid_slope(**overrides: Any) -> HfPyramidSlopedTerrainCfg:
    defaults: dict[str, Any] = dict(
        slope_range=(0.0, 0.7),
        platform_width=2.0,
        border_width=0.25,
    )
    defaults.update(overrides)
    return HfPyramidSlopedTerrainCfg(**defaults)


@terrain_preset
def hf_pyramid_slope_inv(**overrides: Any) -> HfPyramidSlopedTerrainCfg:
    defaults: dict[str, Any] = dict(
        slope_range=(0.0, 0.7),
        platform_width=2.0,
        border_width=0.25,
        inverted=True,
    )
    defaults.update(overrides)
    return HfPyramidSlopedTerrainCfg(**defaults)


@terrain_preset
def random_rough(**overrides: Any) -> HfRandomUniformTerrainCfg:
    defaults: dict[str, Any] = dict(
        noise_range=(0.02, 0.10),
        noise_step=0.02,
        border_width=0.25,
    )
    defaults.update(overrides)
    return HfRandomUniformTerrainCfg(**defaults)


@terrain_preset
def wave_terrain(**overrides: Any) -> HfWaveTerrainCfg:
    defaults: dict[str, Any] = dict(
        amplitude_range=(0.0, 0.2),
        num_waves=4,
        border_width=0.25,
    )
    defaults.update(overrides)
    return HfWaveTerrainCfg(**defaults)


# Named terrain sets.

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    sub_terrains={
        "flat": flat(proportion=0.2),
        "pyramid_stairs": pyramid_stairs(proportion=0.2, step_height_range=(0.0, 0.1)),
        "pyramid_stairs_inv": pyramid_stairs_inv(proportion=0.2, step_height_range=(0.0, 0.1)),
        "hf_pyramid_slope": hf_pyramid_slope(proportion=0.1, slope_range=(0.0, 1.0)),
        "hf_pyramid_slope_inv": hf_pyramid_slope_inv(proportion=0.1, slope_range=(0.0, 1.0)),
        "random_rough": random_rough(proportion=0.1),
        "wave_terrain": wave_terrain(proportion=0.1),
    },
    add_lights=True,
)

STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=4,
    curriculum=True,
    sub_terrains={
        "flat": flat(proportion=0.25),
        "easy_stairs": pyramid_stairs(
            proportion=0.35,
            step_height_range=(0.02, 0.05),
            step_width=0.40,
        ),
        "moderate_stairs": pyramid_stairs(
            proportion=0.25,
            step_height_range=(0.05, 0.08),
            step_width=0.35,
            platform_width=2.5,
            border_width=0.8,
        ),
        "challenging_stairs": pyramid_stairs(
            proportion=0.15,
            step_height_range=(0.08, 0.10),
            step_width=0.30,
            platform_width=2.0,
            border_width=0.5,
        ),
    },
    add_lights=True,
)
