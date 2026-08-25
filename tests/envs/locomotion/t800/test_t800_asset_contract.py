"""Asset-level contracts for the EngineAI T800 walk-flat model."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.assets.hub import resolve_robot_asset_dir
from unilab.base.backend.mujoco.xml import create_discardvisual_xml

ROBOT_XML = ASSETS_ROOT_PATH / "robots" / "t800" / "t800.xml"
SCENE_XML = ASSETS_ROOT_PATH / "robots" / "t800" / "scene_flat.xml"
G1_ROBOT_XML = ASSETS_ROOT_PATH / "robots" / "g1" / "g1.xml"
G1_SCENE_XML = ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml"
TEXTURES_DIR = ROBOT_XML.parent / "textures"
ENGINEAI_LICENSE = ROBOT_XML.parent / "LICENSE.engineai.txt"

EXPECTED_TEXTURES = {
    "LINK_BASE": (
        "LINK_BASE.png",
        "a298400671d0868d8a404801e280d2f9bd01b33ec76660354192baba7a9b12e0",
    ),
    "LINK_HIP_PITCH": (
        "LINK_HIP_PITCH_L.png",
        "192c9753ad475eb1cd631c2de474e3f644dd2f1189a40ed25759299321265b08",
    ),
    "LINK_HIP_ROLL": (
        "LINK_HIP_ROLL_L.png",
        "f123c3a3fed080c94c0296618e8a6712f7da7234a9a8611be3e08a7972fb58ab",
    ),
    "LINK_HIP_YAW": (
        "LINK_HIP_YAW_L.png",
        "97b5f5c52a05f8315ed07c531f5a8834e92fbae0f89cd21853428ae858eacc9a",
    ),
    "LINK_KNEE_PITCH": (
        "LINK_KNEE_PITCH_L.png",
        "ce66f28fc5dac742c73e9b4dfe735e20d2b89641afb4d6e033292a7e41a2de85",
    ),
    "LINK_ANKLE_ROLL": (
        "LINK_ANKLE_ROLL_L.png",
        "c69f48f543389e72954d8ede6d93a9d4e96441576f3e35639454138d70b476b2",
    ),
    "LINK_TORSO_YAW": (
        "LINK_TORSO_YAW.png",
        "b7f88ec8e9276ac9c62e19d52037ee30cbc4eefee44e36ca46974711cd8430f4",
    ),
    "LINK_SHOULDER_PITCH": (
        "LINK_SHOULDER_PITCH_L.png",
        "6bb410836ee6431783ee0f69429f9853c14e884f069b6466b9d4b8bda0a95fea",
    ),
    "LINK_SHOULDER_ROLL": (
        "LINK_SHOULDER_ROLL_L.png",
        "fb76b91c8155478cf6d92088d89914a6eb1b907f78a1dcf18128fffee8618311",
    ),
    "LINK_SHOULDER_YAW": (
        "LINK_SHOULDER_YAW_L.png",
        "e7778698452bedb342bbaf198e5c782e17f1d2a9e10f94e2721b8a01fb815737",
    ),
    "LINK_ELBOW_PITCH_L": (
        "LINK_ELBOW_PITCH_L.png",
        "cb9c31e56644890ba5e3c356bcc1ad266a3cf459104a82505aedd62efb66f260",
    ),
    "LINK_ELBOW_PITCH_R": (
        "LINK_ELBOW_PITCH_R.png",
        "8aa0572b3ef8e819f69937d4be0536e0854dc6ef39bd821668dea9b73b0f270c",
    ),
    "LINK_ELBOW_YAW_L": (
        "LINK_ELBOW_YAW_L.png",
        "672b2be4bcdce1863424685a1f681f0d7b23d143fd910311d833268e1246c05c",
    ),
    "LINK_ELBOW_YAW_R": (
        "LINK_ELBOW_YAW_R.png",
        "6283738a1be141cdb676ad3b93cfc2145bcf1f4c7ae39ce8f886fcdcaa3beee6",
    ),
    "LINK_HEAD_YAW": (
        "LINK_HEAD_YAW.png",
        "65fc0f09e4bd0bfc5c910e2e7267ff598a1cf13f85845b56d51055150d53765f",
    ),
}

JOINT_NAMES = (
    "J00_HIP_PITCH_L",
    "J01_HIP_ROLL_L",
    "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L",
    "J04_ANKLE_PITCH_L",
    "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R",
    "J07_HIP_ROLL_R",
    "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R",
    "J10_ANKLE_PITCH_R",
    "J11_ANKLE_ROLL_R",
    "J12_TORSO_YAW",
    "J13_SHOULDER_PITCH_L",
    "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L",
    "J17_ELBOW_YAW_L",
    "J18_SHOULDER_PITCH_R",
    "J19_SHOULDER_ROLL_R",
    "J20_SHOULDER_YAW_R",
    "J21_ELBOW_PITCH_R",
    "J22_ELBOW_YAW_R",
    "J23_HEAD_PITCH",
    "J24_HEAD_YAW",
)
EXPECTED_KP = np.asarray(
    [
        180,
        100,
        100,
        180,
        40,
        40,
        180,
        100,
        100,
        180,
        40,
        40,
        100,
        60,
        50,
        50,
        60,
        50,
        60,
        50,
        50,
        60,
        50,
        100,
        100,
    ],
    dtype=np.float64,
)
EXPECTED_KD = np.asarray(
    [
        5,
        3,
        3,
        5,
        0.3,
        0.3,
        5,
        3,
        3,
        5,
        0.3,
        0.3,
        5,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        1,
        1,
    ],
    dtype=np.float64,
)
EXPECTED_FORCE_LIMITS = np.asarray(
    [
        415,
        370,
        222,
        415,
        160,
        160,
        415,
        370,
        222,
        415,
        160,
        160,
        222,
        160,
        160,
        160,
        160,
        52,
        160,
        160,
        160,
        160,
        52,
        52,
        52,
    ],
    dtype=np.float64,
)


@pytest.fixture(scope="module", autouse=True)
def _resolve_t800_binary_assets() -> None:
    resolve_robot_asset_dir("robots/t800/assets", marker="LINK_BASE.obj")
    resolve_robot_asset_dir("robots/t800/textures", marker="LINK_BASE.png")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_attrib(root: ET.Element, xpath: str) -> dict[str, str]:
    element = root.find(xpath)
    assert element is not None
    return element.attrib


def test_t800_robot_materials_bind_original_textures():
    mujoco = pytest.importorskip("mujoco")
    root = ET.parse(ROBOT_XML).getroot()
    expected_files = {
        material_name: f"textures/{file_name}"
        for material_name, (file_name, _) in EXPECTED_TEXTURES.items()
    }
    material_textures = {
        element.get("name"): element.get("texture")
        for element in root.findall("./asset/material")
        if element.get("name") in EXPECTED_TEXTURES
    }
    texture_files = {
        element.get("name"): element.get("file")
        for element in root.findall("./asset/texture")
        if element.get("name") in EXPECTED_TEXTURES
    }

    assert material_textures == {name: name for name in EXPECTED_TEXTURES}
    assert texture_files == expected_files
    assert {path.name for path in TEXTURES_DIR.glob("*.png")} == {
        file_name for file_name, _ in EXPECTED_TEXTURES.values()
    }
    for file_name, expected_digest in EXPECTED_TEXTURES.values():
        assert _sha256(TEXTURES_DIR / file_name) == expected_digest

    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    rgb_role = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
    for material_name in EXPECTED_TEXTURES:
        material_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_MATERIAL,
            material_name,
        )
        assert material_id >= 0
        assert model.mat_texid[material_id, rgb_role] >= 0


def test_t800_vendored_assets_include_engineai_license():
    assert ENGINEAI_LICENSE.is_file()
    assert _sha256(ENGINEAI_LICENSE) == (
        "a3e5f08bf7ae0983cc7a9f602e9a555f5956a8dade26cf4cf9cc95c441fe0b6c"
    )


def test_t800_flat_scene_matches_g1_ground_visual_contract():
    t800_scene = ET.parse(SCENE_XML).getroot()
    g1_scene = ET.parse(G1_SCENE_XML).getroot()

    for xpath in (
        "./visual/headlight",
        "./visual/rgba",
        "./visual/global",
        "./asset/texture[@type='skybox']",
        "./asset/texture[@name='groundplane']",
        "./asset/material[@name='groundplane']",
        "./worldbody/geom[@name='floor']",
    ):
        assert _required_attrib(t800_scene, xpath) == _required_attrib(g1_scene, xpath)

    g1_robot = ET.parse(G1_ROBOT_XML).getroot()
    expected_light = _required_attrib(g1_robot, "./worldbody/light")
    t800_lights = t800_scene.findall("./worldbody/light")
    assert [light.attrib for light in t800_lights] == [expected_light]
    assert ET.parse(ROBOT_XML).getroot().find("./worldbody/light") is None


def test_t800_robot_xml_is_robot_only_and_uses_25_position_actuators():
    mujoco = pytest.importorskip("mujoco")
    root = ET.parse(ROBOT_XML).getroot()
    assert root.find(".//keyframe") is None

    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    assert (model.nq, model.nv, model.nu) == (32, 31, 25)
    names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        for actuator_id in range(model.nu)
    )
    assert names == JOINT_NAMES

    affine = int(mujoco.mjtBias.mjBIAS_AFFINE)
    assert np.all(model.actuator_biastype == affine)
    np.testing.assert_allclose(model.actuator_gainprm[:, 0], EXPECTED_KP)
    np.testing.assert_allclose(model.actuator_biasprm[:, 2], -EXPECTED_KD)
    np.testing.assert_allclose(model.actuator_forcerange[:, 0], -EXPECTED_FORCE_LIMITS)
    np.testing.assert_allclose(model.actuator_forcerange[:, 1], EXPECTED_FORCE_LIMITS)
    np.testing.assert_allclose(model.actuator_ctrlrange, model.jnt_range[1:])


def test_t800_flat_scene_has_walk_sensors_and_stand_keyframe():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))

    sensor_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
        for sensor_id in range(model.nsensor)
    }
    required_sensors = {
        "pelvis_local_linvel",
        "torso_gyro",
        "torso_upvector",
        "left_foot_pos",
        "left_foot_quat",
        "right_foot_pos",
        "right_foot_quat",
        *(f"left_foot_contact_{index}" for index in range(4)),
        *(f"right_foot_contact_{index}" for index in range(4)),
    }
    assert required_sensors <= sensor_names

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    assert key_id >= 0
    assert model.key_qpos[key_id].shape == (32,)
    assert model.key_ctrl[key_id].shape == (25,)


def test_t800_training_compile_discards_visual_meshes():
    mujoco = pytest.importorskip("mujoco")
    training_xml = Path(create_discardvisual_xml(str(SCENE_XML)))
    try:
        model = mujoco.MjModel.from_xml_path(str(training_xml))
    finally:
        training_xml.unlink(missing_ok=True)

    assert model.nmesh == 0
    assert model.ntex == 0
    assert np.all(model.geom_type != int(mujoco.mjtGeom.mjGEOM_MESH))


def test_t800_stand_foot_sites_track_the_collision_sole_height():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    for side in ("left", "right"):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot")
        sole_heights = []
        for index in range(4):
            geom_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"{side}_foot_contact_{index}_geom",
            )
            rotation = data.geom_xmat[geom_id].reshape(3, 3)
            vertical_extent = np.abs(rotation[2]) @ model.geom_size[geom_id]
            sole_heights.append(data.geom_xpos[geom_id, 2] - vertical_extent)

        site_height = data.site_xpos[site_id, 2]
        lowest_collision = min(sole_heights)
        assert site_height >= lowest_collision - 1.0e-3
        assert site_height - lowest_collision < 5.0e-3
