from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from unilab.envs.manipulation.sharpa_inhand.base import DEFAULT_FINGERTIP_BODY_NAMES
from unilab.envs.manipulation.sharpa_inhand.rotation import SharpaInhandRotationCfg

EXPECTED_CONTACT_SENSOR_NAMES = [
    "contact_right_thumb_elastomer_force",
    "contact_right_index_elastomer_force",
    "contact_right_middle_elastomer_force",
    "contact_right_ring_elastomer_force",
    "contact_right_pinky_elastomer_force",
    "contact_right_thumb_dp_force",
    "contact_right_index_dp_force",
    "contact_right_middle_dp_force",
    "contact_right_ring_dp_force",
    "contact_right_pinky_dp_force",
]


def _split_floats(value: str) -> list[float]:
    return [float(x) for x in value.split()]


def _get_hand_xml_path() -> Path:
    cfg = SharpaInhandRotationCfg()
    return Path(cfg.model_file).with_name("right_sharpa_wave.xml")


def test_sharpa_rotation_cfg_points_to_existing_scene_xml() -> None:
    cfg = SharpaInhandRotationCfg()
    model_path = Path(cfg.model_file)

    assert model_path.suffix == ".xml"
    assert model_path.is_file()
    assert model_path.name == "scene.xml"


def test_sharpa_scene_xml_has_required_object_and_home_keyframe_layout() -> None:
    cfg = SharpaInhandRotationCfg()
    model_path = Path(cfg.model_file)

    root = ET.parse(model_path).getroot()

    include = root.find("./include")
    assert include is not None
    assert include.attrib.get("file") == "right_sharpa_wave.xml"

    object_body = root.find("./worldbody/body[@name='object']")
    assert object_body is not None

    freejoint = object_body.find("./freejoint[@name='object_joint']")
    assert freejoint is not None

    key = root.find("./keyframe/key[@name='home']")
    assert key is not None

    qpos = key.attrib.get("qpos")
    ctrl = key.attrib.get("ctrl")
    assert qpos is not None
    assert ctrl is not None

    # rotation.py + base.py assume 22 hand DoFs + 7 object freejoint DoFs.
    assert len(_split_floats(qpos)) == 29
    assert len(_split_floats(ctrl)) == 22


def test_sharpa_hand_xml_declares_expected_contact_sensors() -> None:
    hand_xml_path = _get_hand_xml_path()
    root = ET.parse(hand_xml_path).getroot()

    sensor_nodes = root.findall("./sensor/contact")
    names = [node.attrib.get("name", "") for node in sensor_nodes]

    for sensor_name in EXPECTED_CONTACT_SENSOR_NAMES:
        assert sensor_name in names

    by_name = {node.attrib["name"]: node for node in sensor_nodes if "name" in node.attrib}
    for sensor_name in EXPECTED_CONTACT_SENSOR_NAMES:
        node = by_name[sensor_name]
        assert node.attrib.get("geom2") == "object"
        assert node.attrib.get("data") == "force"
        assert node.attrib.get("reduce") == "netforce"


def test_sharpa_scene_xml_loads_with_mujoco_and_resolves_required_bodies_and_sensors() -> None:
    mujoco = pytest.importorskip("mujoco", reason="mujoco not installed")

    cfg = SharpaInhandRotationCfg()
    model = mujoco.MjModel.from_xml_path(cfg.model_file)

    assert model.nq == 29
    assert model.nu == 22

    object_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    assert object_bid >= 0

    for body_name in DEFAULT_FINGERTIP_BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        assert bid >= 0

    for sensor_name in EXPECTED_CONTACT_SENSOR_NAMES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        assert sid >= 0
        assert model.sensor_dim[sid] == 3


def main() -> int:
    checks = [
        test_sharpa_rotation_cfg_points_to_existing_scene_xml,
        test_sharpa_scene_xml_has_required_object_and_home_keyframe_layout,
        test_sharpa_hand_xml_declares_expected_contact_sensors,
        test_sharpa_scene_xml_loads_with_mujoco_and_resolves_required_bodies_and_sensors,
    ]

    skip_exc = getattr(pytest.skip, "Exception", None)
    passed = 0
    skipped = 0
    failed = 0

    for check in checks:
        try:
            check()
        except BaseException as exc:  # includes pytest skip outcomes
            if isinstance(exc, KeyboardInterrupt):
                raise
            if skip_exc is not None and isinstance(exc, skip_exc):
                skipped += 1
                print(f"[SKIP] {check.__name__}: {exc}")
            else:
                failed += 1
                print(f"[FAIL] {check.__name__}: {exc}")
        else:
            passed += 1
            print(f"[PASS] {check.__name__}")

    print(f"Summary: passed={passed}, skipped={skipped}, failed={failed}")
    if failed == 0:
        print("Sharpa XML loading checks: SUCCESS")
        return 0

    print("Sharpa XML loading checks: FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
