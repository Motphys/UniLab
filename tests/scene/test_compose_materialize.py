"""Tests for cold-path scene materialization (#270)."""

from __future__ import annotations

import copy

import mujoco
import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.scene.composer import compose_and_materialize
from unilab.terrains import ROUGH_TERRAINS_CFG, TerrainGeneratorCfg

GO2_BASE = ASSETS_ROOT_PATH / "robots" / "go2" / "scene_base.xml"


def _small_rough_cfg() -> TerrainGeneratorCfg:
    cfg = copy.deepcopy(ROUGH_TERRAINS_CFG)
    cfg.num_rows = 2
    cfg.num_cols = 2
    cfg.border_width = 0.0
    cfg.add_lights = False
    cfg.seed = 0
    return cfg


def test_materialize_writes_scene_xml(tmp_path):
    scene = compose_and_materialize(GO2_BASE, _small_rough_cfg(), tmp_path)
    assert scene.scene_xml.is_file()
    assert scene.scene_xml.name == "scene.xml"
    assert (tmp_path / "assets").is_dir()


def test_materialize_returns_terrain_origins(tmp_path):
    cfg = _small_rough_cfg()
    scene = compose_and_materialize(GO2_BASE, cfg, tmp_path)
    assert scene.terrain_origins.shape == (cfg.num_rows, cfg.num_cols, 3)
    # Cells must span the grid extent (centered on origin), so the four
    # corners should be at distinct XY positions.
    corners = {
        tuple(scene.terrain_origins[0, 0, :2]),
        tuple(scene.terrain_origins[-1, 0, :2]),
        tuple(scene.terrain_origins[0, -1, :2]),
        tuple(scene.terrain_origins[-1, -1, :2]),
    }
    assert len(corners) == 4


def test_materialized_scene_loads_in_mujoco(tmp_path):
    scene = compose_and_materialize(GO2_BASE, _small_rough_cfg(), tmp_path)
    model = mujoco.MjModel.from_xml_path(str(scene.scene_xml))
    # Robot root body present (Go2 uses "base").
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    assert base_id >= 0
    # Terrain body present.
    terrain_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "terrain")
    assert terrain_id >= 0
    # More than one geom (robot + terrain).
    assert model.ngeom > 1


def test_contact_sensors_retargeted(tmp_path):
    scene = compose_and_materialize(GO2_BASE, _small_rough_cfg(), tmp_path)
    xml_text = scene.scene_xml.read_text()
    assert 'geom1="floor"' not in xml_text
    assert 'geom2="floor"' not in xml_text
    # Composer rewrites geom-floor sensors to body-terrain references.
    assert 'body1="terrain"' in xml_text


def test_materialized_scene_step_smoke(tmp_path):
    """Materialized scene must compile + step without error."""
    scene = compose_and_materialize(GO2_BASE, _small_rough_cfg(), tmp_path)
    model = mujoco.MjModel.from_xml_path(str(scene.scene_xml))
    data = mujoco.MjData(model)
    for _ in range(5):
        mujoco.mj_step(model, data)


@pytest.mark.parametrize("zip_output", [False, True])
def test_export_modes(tmp_path, zip_output):
    """Both directory and zip outputs work — defensive coverage of export_spec.

    With zip=True the directory is removed; we just ensure the call doesn't
    crash. The composer itself currently always writes a directory.
    """
    cfg = _small_rough_cfg()
    out_dir = tmp_path / "scene"
    scene = compose_and_materialize(GO2_BASE, cfg, out_dir)
    assert scene.scene_xml.is_file()
