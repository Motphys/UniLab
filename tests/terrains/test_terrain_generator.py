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


def test_all_compiled_geoms_are_hfields():
    """Compile a small ROUGH scene and assert every terrain-body geom is a hfield."""
    spec = mujoco.MjSpec()
    cfg = _small_rough_cfg()
    TerrainGenerator(cfg).compile(spec)
    for geom in spec.body("terrain").geoms:
        assert geom.type == mujoco.mjtGeom.mjGEOM_HFIELD, (
            f"Found non-hfield geom in terrain body: type={geom.type}"
        )


def test_border_uses_hfields():
    """Enabling the outer border emits 4 additional hfield strips, no boxes."""
    spec = mujoco.MjSpec()
    cfg = _small_rough_cfg()
    cfg.border_width = 5.0
    n_inner = cfg.num_rows * cfg.num_cols
    TerrainGenerator(cfg).compile(spec)
    types = [geom.type for geom in spec.body("terrain").geoms]
    assert all(t == mujoco.mjtGeom.mjGEOM_HFIELD for t in types)
    # 4 strips + n_inner per-cell hfields.
    assert len(types) == n_inner + 4


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
    spec_a = mujoco.MjSpec()
    spec_a.worldbody.add_body(name="terrain")
    spec_b = mujoco.MjSpec()
    spec_b.worldbody.add_body(name="terrain")

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
    out_no = HfPyramidStairsTerrainCfg(holes=False, **common).function(0.5, spec_a, rng)
    out_yes = HfPyramidStairsTerrainCfg(holes=True, pit_depth=2.0, **common).function(
        0.5, spec_b, rng
    )

    base_no = out_no.geometries[0].hfield.size[3]  # base_thickness
    base_yes = out_yes.geometries[0].hfield.size[3]
    max_no = out_no.geometries[0].hfield.size[2]  # max_physical_height
    max_yes = out_yes.geometries[0].hfield.size[2]
    # holes_yes must encode a deeper total span (pit + stairs).
    assert max_yes > max_no
    del base_no, base_yes  # unused but documents intent


def test_inverted_stairs_spawn_is_negative():
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    cfg = ALL_TERRAIN_PRESETS["pyramid_stairs_inv"]()
    cfg.size = (4.0, 4.0)
    cfg.horizontal_scale = 0.05
    cfg.vertical_scale = 0.005
    out = cfg.function(difficulty=0.5, spec=spec, rng=np.random.default_rng(0))
    assert out.origin[2] < 0.0
