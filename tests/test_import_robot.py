from __future__ import annotations

import importlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


def _load_script() -> Any:
    return importlib.import_module("unilab.tools.import_robot")


def test_robot_name_defaults_to_urdf_stem(tmp_path: Path) -> None:
    mod = _load_script()
    urdf = tmp_path / "OpenDoge.urdf"
    urdf.write_text("<robot/>", encoding="utf-8")

    assert mod._robot_name(urdf, None) == "OpenDoge"


def test_move_mesh_assets_renames_nested_converter_output(tmp_path: Path) -> None:
    mod = _load_script()
    robot_dir = tmp_path / "bot"
    generated = robot_dir / "meshes" / "meshes"
    generated.mkdir(parents=True)
    mesh = generated / "link.stl"
    mesh_text = "solid link\nendsolid link\n"
    mesh.write_text(mesh_text, encoding="utf-8")

    mod._move_mesh_assets(robot_dir)

    assert (robot_dir / "assets" / "link.stl").read_text(encoding="utf-8") == mesh_text
    assert not (robot_dir / "meshes").exists()


def test_postprocess_xml_matches_unilab_robot_asset_shape(tmp_path: Path) -> None:
    mod = _load_script()
    xml_path = tmp_path / "bot.xml"
    xml_path.write_text(
        """
        <mujoco>
          <compiler meshdir="meshes/meshes"/>
          <default>
            <default class="floor">
              <geom type="plane" size="0 0 0.05" material="groundplane" />
            </default>
          </default>
          <visual>
            <global offwidth="3840" offheight="2160" />
          </visual>
          <visual>
            <rgba haze="0.15 0.25 0.35 1" />
          </visual>
          <asset>
            <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2" />
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072" />
            <material name="default_material" rgba="1 1 1 1" />
            <mesh file="meshes/meshes/base_link.STL" />
          </asset>
          <worldbody>
            <light pos="0 0 2." dir="0 0 -1" directional="true" />
            <geom type="mesh" mesh="base_link" material="default_material" />
            <geom name="floor" type="plane" size="0 0 0.05" material="groundplane" />
            <geom name="floor_from_class" class="floor" size="0 0 0.05" />
          </worldbody>
          <actuator>
            <motor name="hip" joint="hip" ctrlrange="-1 1" />
          </actuator>
        </mujoco>
        """,
        encoding="utf-8",
    )

    mod._postprocess_xml(xml_path)

    root = ET.parse(xml_path).getroot()
    assert root.find("compiler").get("meshdir") == "assets"
    default_joint = root.find("./default/default[@class='robot']/joint")
    assert default_joint is not None
    assert default_joint.get("damping") == "2"
    assert default_joint.get("armature") == "0.01"
    assert default_joint.get("frictionloss") == "0.2"
    assert root.find("./asset/mesh").get("file") == "base_link.STL"
    assert root.find("./actuator/position") is not None
    assert root.find(".//material[@name='default_material']") is None
    assert not any(elem.get("material") == "default_material" for elem in root.iter())
    assert root.find(".//default[@class='floor']") is None
    assert root.find(".//texture[@name='groundplane']") is None
    assert root.find(".//material[@name='groundplane']") is None
    assert root.find(".//texture[@type='skybox']") is None
    assert root.find(".//light") is None
    assert root.find(".//geom[@name='floor']") is None
    assert root.find(".//geom[@class='floor']") is None


def test_convert_urdf_uses_temporary_converter_dependency(tmp_path: Path, monkeypatch: Any) -> None:
    mod = _load_script()
    urdf_path = tmp_path / "bot.urdf"
    output_xml = tmp_path / "bot.xml"
    calls: list[list[str]] = []

    def fake_run(command: list[str]) -> None:
        calls.append(command)

    monkeypatch.setattr(mod, "_run", fake_run)

    mod._convert_urdf(urdf_path, output_xml)

    assert calls == [
        [
            "uv",
            "run",
            "--with",
            "urdf-to-mjcf",
            "urdf-to-mjcf",
            str(urdf_path),
            "-o",
            str(output_xml),
        ]
    ]


def test_write_scene_xml_creates_only_home_keyframe(tmp_path: Path) -> None:
    mod = _load_script()
    robot_xml = tmp_path / "bot.xml"
    scene_xml = tmp_path / "scene.xml"
    robot_xml.write_text(
        """
        <mujoco>
          <worldbody>
            <body name="base" pos="0 0 0.5">
              <body name="leg">
                <joint name="hip" type="hinge" range="-1 1"/>
                <joint name="knee" type="hinge" range="-2 -1"/>
              </body>
              <freejoint name="floating_base"/>
            </body>
          </worldbody>
          <actuator>
            <position name="hip" joint="hip"/>
            <position name="knee" joint="knee"/>
          </actuator>
        </mujoco>
        """,
        encoding="utf-8",
    )

    mod._write_scene_xml(robot_xml, scene_xml, "bot")

    root = ET.parse(scene_xml).getroot()
    key = root.find("./keyframe/key")
    assert root.get("model") == "bot scene"
    assert key is not None
    assert key.get("name") == "home"
    assert key.get("qpos") == "0 0 0.5 1 0 0 0 0 0"
    assert key.get("ctrl") == "0 0"
    assert root.find("./sensor") is None


def test_tuning_scene_visuals_add_floor_and_light() -> None:
    mod = _load_script()
    root = ET.fromstring("<mujoco><worldbody /></mujoco>")

    mod._ensure_tuning_scene_visuals(root)

    assert root.find("./visual/headlight") is not None
    assert root.find("./asset/material[@name='groundplane']") is not None
    assert root.find("./worldbody/light[@name='__unilab_tuning_light']") is not None
    assert root.find("./worldbody/geom[@name='floor']").get("material") == "groundplane"


def test_changed_ctrl_updates_qpos_without_overwriting_unchanged_ctrl() -> None:
    mod = _load_script()

    class Data:
        ctrl = np.asarray([0.0, 0.0], dtype=np.float64)
        qpos = np.asarray([1.0, 2.0], dtype=np.float64)

    data = Data()
    last_ctrl = data.ctrl.copy()
    data.qpos[0] = 1.25
    assert not mod._sync_changed_ctrl_to_qpos(data, [(0, 0), (1, 1)], last_ctrl)
    np.testing.assert_allclose(data.qpos, [1.25, 2.0])

    data.ctrl[1] = -0.5
    assert mod._sync_changed_ctrl_to_qpos(data, [(0, 0), (1, 1)], last_ctrl)
    np.testing.assert_allclose(data.qpos, [1.25, -0.5])
    np.testing.assert_allclose(last_ctrl, [0.0, -0.5])


def test_tune_scene_keyframe_runs_inline_tuning_steps(tmp_path: Path, monkeypatch: Any) -> None:
    mod = _load_script()
    robot_xml = tmp_path / "bot.xml"
    scene_xml = tmp_path / "scene.xml"
    calls: list[str] = []
    model = object()
    data = object()

    def fake_compile(robot_path: Path, scene_path: Path) -> object:
        assert robot_path == robot_xml
        assert scene_path == scene_xml
        calls.append("compile")
        return model

    def fake_load(compiled_model: object) -> object:
        assert compiled_model is model
        calls.append("load")
        return data

    def fake_open(compiled_model: object, keyframe_data: object) -> None:
        assert compiled_model is model
        assert keyframe_data is data
        calls.append("open")

    def fake_write(scene_path: Path, compiled_model: object, keyframe_data: object) -> None:
        assert scene_path == scene_xml
        assert compiled_model is model
        assert keyframe_data is data
        calls.append("write")

    monkeypatch.setattr(mod, "_compile_tuning_scene", fake_compile)
    monkeypatch.setattr(mod, "_load_keyframe", fake_load)
    monkeypatch.setattr(mod, "_open_tuning_viewer", fake_open)
    monkeypatch.setattr(mod, "_write_tuned_scene_keyframe", fake_write)

    mod._tune_scene_keyframe(robot_xml, scene_xml)

    assert calls == ["compile", "load", "open", "write"]
