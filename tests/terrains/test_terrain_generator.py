"""Tests for the procedural terrain generator (#197 / #270).

Slim port of mjlab's tests/test_terrain_config.py covering the 7 sub-terrains
unilab supports.
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np
import pytest

from unilab.terrains import (
    ALL_TERRAIN_PRESETS,
    ROUGH_TERRAINS_CFG,
    STAIRS_TERRAINS_CFG,
    SubTerrainCfg,
    TerrainGenerator,
    TerrainGeneratorCfg,
    compute_env_origins_grid,
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


def test_terrain_generator_compiles_rough():
    spec = mujoco.MjSpec()
    cfg = _small_rough_cfg()
    TerrainGenerator(cfg).compile(spec)
    body = spec.body("terrain")
    assert len(list(body.geoms)) > 0


def test_terrain_generator_origins_shape():
    cfg = _small_rough_cfg()
    gen = TerrainGenerator(cfg)
    assert gen.terrain_origins.shape == (cfg.num_rows, cfg.num_cols, 3)


@pytest.mark.parametrize("preset_name", sorted(EXPECTED_PRESETS))
def test_each_preset_produces_terrain_geom(preset_name):
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    rng = np.random.default_rng(42)
    cfg = ALL_TERRAIN_PRESETS[preset_name](proportion=1.0)
    cfg.size = (4.0, 4.0)
    output = cfg.function(difficulty=0.5, spec=spec, rng=rng)
    assert output.geometries
    assert output.origin.shape == (3,)


def test_compiled_rough_terrain_has_terrain_geoms_named():
    spec = mujoco.MjSpec()
    cfg = _small_rough_cfg()
    TerrainGenerator(cfg).compile(spec)
    geom_names = [g.name for g in spec.body("terrain").geoms]
    assert any(name.startswith("terrain_") for name in geom_names)


def test_compute_env_origins_grid_zero_spacing_returns_zeros():
    origins = compute_env_origins_grid(num_envs=16, env_spacing=0.0)
    assert origins.shape == (16, 3)
    assert np.all(origins == 0.0)


def test_compute_env_origins_grid_centered_layout():
    origins = compute_env_origins_grid(num_envs=4, env_spacing=2.0)
    assert origins.shape == (4, 3)
    expected = {(-1.0, -1.0, 0.0), (-1.0, 1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0)}
    assert {tuple(p) for p in origins} == expected
    # Z is always 0.
    assert np.all(origins[:, 2] == 0.0)


def test_compute_env_origins_grid_spacing_scales_extent():
    origins = compute_env_origins_grid(num_envs=64, env_spacing=2.0)
    span_x = origins[:, 0].max() - origins[:, 0].min()
    span_y = origins[:, 1].max() - origins[:, 1].min()
    assert span_x == pytest.approx(14.0)
    assert span_y == pytest.approx(14.0)


def test_compute_env_origins_grid_non_square_count_is_truncated():
    origins = compute_env_origins_grid(num_envs=10, env_spacing=1.0)
    assert origins.shape == (10, 3)
    unique_xy = {tuple(p[:2]) for p in origins}
    assert len(unique_xy) == 10
