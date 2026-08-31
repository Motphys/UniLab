"""Asset and provenance contracts for the MicroDuck velocity task."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.assets.hub import resolve_robot_asset_dir

ROBOT_DIR = ASSETS_ROOT_PATH / "robots" / "microduck"
ROBOT_XML = ROBOT_DIR / "microduck.xml"
TASK_XML = ROBOT_DIR / "locomotion_task.xml"
MANIFEST = ROBOT_DIR / "assets.sha256"
LICENSE = ROBOT_DIR / "LICENSE.pollen-robotics.txt"


def _manifest() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        digest, filename = line.split(maxsplit=1)
        entries[filename] = digest
    return entries


def test_microduck_xml_keeps_task_keyframe_out_of_robot_description() -> None:
    robot = ET.parse(ROBOT_XML).getroot()
    task = ET.parse(TASK_XML).getroot()

    assert robot.find("keyframe") is None
    keyframe = task.find("keyframe/key[@name='home']")
    assert keyframe is not None
    assert len(keyframe.attrib["ctrl"].split()) == 14

    chosen = robot.find("default/default[@class='chosen_actuator']/position")
    assert chosen is not None
    assert float(chosen.attrib["kp"]) == pytest.approx(50.0)
    assert float(chosen.attrib["kv"]) == pytest.approx(0.5)


def test_microduck_manifest_covers_every_xml_mesh() -> None:
    manifest = _manifest()
    robot = ET.parse(ROBOT_XML).getroot()
    referenced = {mesh.attrib["file"] for mesh in robot.findall("asset/mesh")}

    assert len(manifest) == 47
    assert referenced <= set(manifest)
    assert "trunk_base.stl" in manifest
    assert "Copyright 2026 Pollen Robotics" in LICENSE.read_text(encoding="utf-8")


@pytest.mark.slow
def test_hugging_face_microduck_assets_match_sha256_manifest() -> None:
    asset_dir = resolve_robot_asset_dir(
        "robots/microduck/assets",
        marker="trunk_base.stl",
    )
    manifest = _manifest()

    assert {path.name for path in asset_dir.glob("*.stl")} == set(manifest)
    for filename, expected in manifest.items():
        actual = hashlib.sha256((Path(asset_dir) / filename).read_bytes()).hexdigest()
        assert actual == expected, filename
