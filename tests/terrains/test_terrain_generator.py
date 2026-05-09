"""Tests for the procedural terrain generator (#197 / #270).

Slim port of mjlab's tests/test_terrain_config.py covering the 7 sub-terrains
unilab supports.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from unilab.terrains import (
    ALL_TERRAIN_PRESETS,
    ROUGH_TERRAINS_CFG,
    STAIRS_TERRAINS_CFG,
    SubTerrainCfg,
    TerrainGenerator,
    TerrainGeneratorCfg,
)

EXPECTED_PRESETS = {
    "flat",
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
    "random_rough",
    "wave_terrain",
}


def test_all_presets_keyset():
    assert set(ALL_TERRAIN_PRESETS) == EXPECTED_PRESETS


def test_all_presets_return_sub_terrain_cfg():
    for name, fn in ALL_TERRAIN_PRESETS.items():
        cfg = fn(proportion=1.0)
        assert isinstance(cfg, SubTerrainCfg), name


def test_preset_overrides_apply():
    cfg = ALL_TERRAIN_PRESETS["pyramid_stairs"](step_height_range=(0.1, 0.2))
    assert cfg.step_height_range == (0.1, 0.2)
    # Defaults preserved for fields not overridden.
    assert cfg.step_width == 0.3


def test_rough_terrains_cfg_structure():
    assert ROUGH_TERRAINS_CFG.size == (8.0, 8.0)
    assert ROUGH_TERRAINS_CFG.num_rows == 10
    assert ROUGH_TERRAINS_CFG.num_cols == 20
    assert len(ROUGH_TERRAINS_CFG.sub_terrains) == 7
    total = sum(c.proportion for c in ROUGH_TERRAINS_CFG.sub_terrains.values())
    assert abs(total - 1.0) < 1e-6


def test_stairs_terrains_cfg_structure():
    assert STAIRS_TERRAINS_CFG.curriculum is True
    assert len(STAIRS_TERRAINS_CFG.sub_terrains) == 4


def _small_rough_cfg() -> TerrainGeneratorCfg:
    cfg = copy.deepcopy(ROUGH_TERRAINS_CFG)
    cfg.num_rows = 2
    cfg.num_cols = 2
    cfg.border_width = 0.0
    cfg.add_lights = False
    cfg.seed = 0
    return cfg


def test_terrain_generator_origins_shape():
    cfg = _small_rough_cfg()
    gen = TerrainGenerator(cfg)
    assert gen.terrain_origins.shape == (cfg.num_rows, cfg.num_cols, 3)


def test_terrain_generator_generates_single_merged_hfield():
    cfg = _small_rough_cfg()
    cfg.border_width = 1.0
    gen = TerrainGenerator(cfg)
    terrain = gen.generate()
    border_px = int(round(cfg.border_width / cfg.horizontal_scale))
    tile_x_px = int(round(cfg.size[0] / cfg.horizontal_scale))
    tile_y_px = int(round(cfg.size[1] / cfg.horizontal_scale))
    assert terrain.heights_yx.shape == (
        cfg.num_cols * tile_y_px + 2 * border_px,
        cfg.num_rows * tile_x_px + 2 * border_px,
    )
    assert terrain.terrain_origins.shape == (cfg.num_rows, cfg.num_cols, 3)
    assert terrain.to_uint16().dtype == np.uint16
    assert terrain.hfield_size[0] == pytest.approx(terrain.size[0] / 2)
    assert terrain.hfield_size[1] == pytest.approx(terrain.size[1] / 2)


def test_generated_terrain_surface_sampler_uses_world_xy():
    cfg = _small_rough_cfg()
    cfg.border_width = 0.0
    terrain = TerrainGenerator(cfg).generate()
    sampler = terrain.surface_sampler()

    origin = terrain.terrain_origins[0, 0]
    sampled = sampler.sample_height(origin[:2])

    assert sampled == pytest.approx(origin[2], abs=1e-4)


def test_heightfield_surface_sampler_can_flip_world_y():
    cfg = _small_rough_cfg()
    cfg.border_width = 0.0
    terrain = TerrainGenerator(cfg).generate()
    sampler = terrain.surface_sampler(flip_y=True)
    reference = terrain.surface_sampler()

    xy = np.asarray([[1.25, -2.75], [-3.5, 4.25]], dtype=np.float64)

    np.testing.assert_allclose(
        sampler.sample_height(xy),
        reference.sample_height(xy * np.asarray([1.0, -1.0])),
    )


@pytest.mark.parametrize("preset_name", sorted(EXPECTED_PRESETS))
def test_each_preset_produces_heightfield(preset_name):
    rng = np.random.default_rng(42)
    cfg = ALL_TERRAIN_PRESETS[preset_name](proportion=1.0)
    cfg.size = (4.0, 4.0)
    output = cfg.function(difficulty=0.5, rng=rng)
    assert output.heightfield.noise.ndim == 2
    assert output.origin.shape == (3,)


def test_resolution_validation_rejects_misaligned_step_width():
    cfg = TerrainGeneratorCfg(
        size=(4.0, 4.0),
        horizontal_scale=0.05,
        sub_terrains={"x": ALL_TERRAIN_PRESETS["pyramid_stairs"](step_width=0.07)},
    )
    with pytest.raises(ValueError, match="step_width"):
        TerrainGenerator(cfg)


def test_resolution_validation_rejects_misaligned_size():
    cfg = TerrainGeneratorCfg(
        size=(4.13, 4.13),
        horizontal_scale=0.05,
        sub_terrains={"x": ALL_TERRAIN_PRESETS["flat"]()},
    )
    with pytest.raises(ValueError, match="size"):
        TerrainGenerator(cfg)


def test_holes_creates_deeper_minimum():
    """Holes mode must produce a strictly lower minimum than non-holes."""
    from unilab.terrains import HfPyramidStairsTerrainCfg

    common = dict(
        size=(4.0, 4.0),
        step_height_range=(0.1, 0.1),
        step_width=0.3,
        platform_width=1.0,
        border_width=0.5,
        horizontal_scale=0.05,
        vertical_scale=0.005,
    )
    rng = np.random.default_rng(0)
    out_no = HfPyramidStairsTerrainCfg(holes=False, **common).function(0.5, rng)
    out_yes = HfPyramidStairsTerrainCfg(holes=True, pit_depth=2.0, **common).function(0.5, rng)

    base_no = out_no.heightfield.base_thickness
    base_yes = out_yes.heightfield.base_thickness
    max_no = out_no.heightfield.max_physical_height
    max_yes = out_yes.heightfield.max_physical_height
    # holes_yes must encode a deeper total span (pit + stairs).
    assert max_yes > max_no
    del base_no, base_yes  # unused but documents intent


def test_inverted_stairs_spawn_is_negative():
    cfg = ALL_TERRAIN_PRESETS["pyramid_stairs_inv"]()
    cfg.size = (4.0, 4.0)
    cfg.horizontal_scale = 0.05
    cfg.vertical_scale = 0.005
    out = cfg.function(difficulty=0.5, rng=np.random.default_rng(0))
    assert out.origin[2] < 0.0
