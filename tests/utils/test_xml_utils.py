"""Tests for MuJoCo-backed XML editing helpers."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend import (
    inject_mujoco_tracking_sensors,
    materialize_motrix_hfield_attached_scene,
    materialize_motrix_scene,
    materialize_mujoco_hfield_attached_scene,
    materialize_scene_fragments,
)


def _g1_scene() -> str:
    return str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")


def _go2_robot() -> str:
    return str(ASSETS_ROOT_PATH / "robots" / "go2" / "go2.xml")


def _go2_locomotion_task() -> str:
    return str(ASSETS_ROOT_PATH / "robots" / "go2" / "locomotion_task.xml")


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


def test_materialize_motrix_scene_adds_tracking_frame_sensors() -> None:
    motrixsim = pytest.importorskip("motrixsim")

    model = materialize_motrix_scene(
        model_file=_g1_scene(),
        add_body_sensors=True,
        base_name="pelvis",
    )
    data = motrixsim.SceneData(model, batch=[1])

    for sensor_name in (
        "track_pos_b_pelvis",
        "track_quat_b_pelvis",
        "track_linvel_b_pelvis",
        "track_angvel_b_pelvis",
    ):
        assert model.get_sensor_value(sensor_name, data).shape[0] == 1


def test_materialize_scene_fragments_merges_static_scene_fragment(tmp_path) -> None:
    mujoco = pytest.importorskip("mujoco")

    scene = tmp_path / "scene.xml"
    scene.write_text(
        """
        <mujoco>
          <worldbody>
            <geom name="floor" type="plane" size="1 1 0.1"/>
            <body name="body" pos="0 0 0.1">
              <geom name="foot" type="sphere" size="0.02"/>
            </body>
          </worldbody>
        </mujoco>
        """,
        encoding="utf-8",
    )
    fragment = tmp_path / "fragment.xml"
    fragment.write_text(
        """
        <mujoco>
          <sensor>
            <contact name="foot_contact" geom1="floor" geom2="foot"
              data="found" num="1" reduce="mindist"/>
          </sensor>
        </mujoco>
        """,
        encoding="utf-8",
    )

    tmp_xml = materialize_scene_fragments(str(scene), fragment_files=[str(fragment)])
    try:
        model = mujoco.MjModel.from_xml_path(tmp_xml)
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "foot_contact") >= 0
    finally:
        os.remove(tmp_xml)


def test_materialize_mujoco_hfield_attached_scene_composes_robot_and_task_fragment(
    tmp_path,
) -> None:
    mujoco = pytest.importorskip("mujoco")

    import copy

    from unilab.terrains import ROUGH_TERRAINS_CFG

    cfg = copy.deepcopy(ROUGH_TERRAINS_CFG)
    cfg.num_rows = 2
    cfg.num_cols = 2
    cfg.border_width = 0.0
    cfg.add_lights = False
    cfg.seed = 0

    model, terrain_origins = materialize_mujoco_hfield_attached_scene(
        model_file=_go2_robot(),
        terrain_cfg=cfg,
        output_dir=tmp_path,
        fragment_files=[_go2_locomotion_task()],
    )

    assert terrain_origins.shape == (2, 2, 3)
    assert (tmp_path / "hfields" / "hfield.png").is_file()
    scene_xml = tmp_path / "scene.xml"
    assert scene_xml.is_file()
    scene_root = ET.parse(scene_xml).getroot()
    compiler = scene_root.find("compiler")
    assert compiler is None or compiler.get("discardvisual") != "true"
    assert scene_root.find("./asset/texture[@name='groundplane']") is not None
    assert scene_root.find("./asset/material[@name='groundplane']") is not None
    assert scene_root.find("./visual/headlight") is not None
    assert scene_root.find(".//geom[@name='floor']").get("material") == "groundplane"
    reloaded_model = mujoco.MjModel.from_xml_path(str(scene_xml))
    assert model.ngeom < reloaded_model.ngeom
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain_hfield") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "FL_foot_contact") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home") >= 0
    assert mujoco.mj_name2id(reloaded_model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain_hfield") >= 0
    assert mujoco.mj_name2id(reloaded_model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0


def test_materialize_mujoco_hfield_attached_scene_accepts_repo_relative_fragments(
    tmp_path,
) -> None:
    mujoco = pytest.importorskip("mujoco")

    import copy

    from unilab.terrains import ROUGH_TERRAINS_CFG

    cfg = copy.deepcopy(ROUGH_TERRAINS_CFG)
    cfg.num_rows = 1
    cfg.num_cols = 1
    cfg.border_width = 0.0
    cfg.add_lights = False
    cfg.seed = 0

    model, _ = materialize_mujoco_hfield_attached_scene(
        model_file=str(ASSETS_ROOT_PATH / "robots" / "go2" / "go2.xml"),
        terrain_cfg=cfg,
        output_dir=tmp_path,
        fragment_files=["src/unilab/assets/robots/go2/locomotion_task.xml"],
    )

    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "FL_foot_contact") >= 0


def test_materialize_motrix_hfield_attached_scene_composes_robot_and_task_fragment() -> None:
    pytest.importorskip("motrixsim")

    import copy

    from unilab.terrains import ROUGH_TERRAINS_CFG

    cfg = copy.deepcopy(ROUGH_TERRAINS_CFG)
    cfg.num_rows = 1
    cfg.num_cols = 1
    cfg.border_width = 0.0
    cfg.add_lights = False
    cfg.seed = 0

    model, terrain_origins = materialize_motrix_hfield_attached_scene(
        model_file=_go2_robot(),
        terrain_cfg=cfg,
        fragment_files=[_go2_locomotion_task()],
    )

    assert terrain_origins.shape == (1, 1, 3)
    assert model.num_hfields == 1
    assert model.get_hfield_index("terrain_hfield") == 0
    assert model.num_actuators == 12
    assert model.num_keyframes == 1
    assert model.num_sensors >= 23
    assert model.get_body("base") is not None
    assert model.get_link("base") is not None
