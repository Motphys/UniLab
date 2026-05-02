"""Tests for scene export tool."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

pytest.importorskip("mujoco")

from unilab.tools.export_scene import export_scene, main

MINIMAL_XML = """\
<mujoco>
  <worldbody>
    <body name="box" pos="0 0 0.5">
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture()
def model_file(tmp_path: Path) -> str:
    p = tmp_path / "test_model.xml"
    p.write_text(MINIMAL_XML)
    return str(p)


def test_export_creates_xml(model_file: str, tmp_path: Path):
    out = tmp_path / "export_out"
    export_scene(model_file, str(out))
    assert (out / "scene.xml").is_file()


def test_export_directory_structure(model_file: str, tmp_path: Path):
    out = tmp_path / "export_out2"
    export_scene(model_file, str(out))
    assert out.is_dir()
    assert (out / "scene.xml").is_file()


def test_exported_scene_reloadable(model_file: str, tmp_path: Path):
    import mujoco

    out = tmp_path / "export_out3"
    export_scene(model_file, str(out))
    model = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))
    assert model.ngeom > 0


def test_export_scene_zip_contains_xml(model_file: str, tmp_path: Path):
    out = tmp_path / "export_zip"
    zip_path = Path(export_scene(model_file, str(out), as_zip=True))
    assert zip_path.is_file()
    assert zip_path.suffix == ".zip"
    with ZipFile(zip_path) as zf:
        assert "scene.xml" in zf.namelist()


def test_export_scene_cli_creates_xml(model_file: str, tmp_path: Path):
    out = tmp_path / "export_cli"
    assert main([model_file, "-o", str(out)]) == 0
    assert (out / "scene.xml").is_file()
