"""Tests for MuJoCo-backed XML editing helpers."""

from __future__ import annotations

import os

import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend import (
    create_motrix_compatible_xml,
    inject_motrix_tracking_sensors,
    inject_mujoco_tracking_sensors,
    materialize_terrain_hfield_scene,
)


def _g1_scene() -> str:
    return str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")


def _go2_rough_scene() -> str:
    return str(ASSETS_ROOT_PATH / "robots" / "go2" / "scene_rough.xml")


def test_inject_mujoco_tracking_sensors_uses_mjspec_and_preserves_contract() -> None:
    mujoco = pytest.importorskip("mujoco")

    tmp_xml, tracked_body_ids, valid_bnames = inject_mujoco_tracking_sensors(
        _g1_scene(),
        baselink_name="pelvis",
    )
    try:
        assert tracked_body_ids == list(range(1, len(valid_bnames) + 1))
        assert valid_bnames[0] == "pelvis"

        model = mujoco.MjModel.from_xml_path(tmp_xml)
        for sensor_name in (
            "track_pos_w_pelvis",
            "track_quat_w_pelvis",
            "track_linvel_w_pelvis",
            "track_angvel_w_pelvis",
            "track_pos_b_pelvis",
            "track_quat_b_pelvis",
            "track_linvel_b_pelvis",
            "track_angvel_b_pelvis",
        ):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name) >= 0
    finally:
        os.remove(tmp_xml)


def test_inject_motrix_tracking_sensors_only_adds_baselink_relative_sensors() -> None:
    mujoco = pytest.importorskip("mujoco")

    tmp_xml, _, _ = inject_motrix_tracking_sensors(_g1_scene(), baselink_name="pelvis")
    try:
        model = mujoco.MjModel.from_xml_path(tmp_xml)
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "track_pos_b_pelvis") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "track_quat_b_pelvis") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "track_linvel_b_pelvis") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "track_angvel_b_pelvis") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "track_pos_w_pelvis") == -1
    finally:
        os.remove(tmp_xml)


def test_motrix_xml_materialization_rewrites_auto_texture_colorspace() -> None:
    tmp_xml = create_motrix_compatible_xml(_g1_scene())
    try:
        with open(tmp_xml, encoding="utf-8") as f:
            xml_text = f.read()
        assert 'colorspace="auto"' not in xml_text
        assert 'colorspace="sRGB"' in xml_text
    finally:
        os.remove(tmp_xml)


def test_materialize_terrain_hfield_scene_replaces_template_png(tmp_path) -> None:
    mujoco = pytest.importorskip("mujoco")

    import copy

    from unilab.terrains import ROUGH_TERRAINS_CFG

    cfg = copy.deepcopy(ROUGH_TERRAINS_CFG)
    cfg.num_rows = 2
    cfg.num_cols = 2
    cfg.border_width = 0.0
    cfg.add_lights = False
    cfg.seed = 0

    scene_xml, terrain_origins = materialize_terrain_hfield_scene(
        _go2_rough_scene(),
        terrain_cfg=cfg,
        output_dir=tmp_path,
    )
    assert terrain_origins.shape == (2, 2, 3)
    assert (tmp_path / "hfields" / "hfield.png").is_file()
    text = (tmp_path / "scene.xml").read_text()
    assert str(tmp_path / "hfields" / "hfield.png") in text
    assert 'hfield="terrain_hfield"' in text

    model = mujoco.MjModel.from_xml_path(scene_xml)
    hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain_hfield")
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert hfield_id >= 0
    assert geom_id >= 0
    assert int(model.geom_dataid[geom_id]) == hfield_id
