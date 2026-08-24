"""Tests for the robot asset prefetch CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from unilab.assets import pull as pull_assets


def _populate(directory: Path, *, suffix: str, count: int) -> Path:
    directory.mkdir(parents=True)
    for index in range(count):
        (directory / f"asset_{index}{suffix}").write_bytes(b"asset")
    return directory


def test_pull_assets_a2arm_resolves_nested_mesh_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    target = tmp_path / "meshes"
    (target / "p7_v3").mkdir(parents=True)
    (target / "adapter_plate.STL").write_bytes(b"mesh")
    for name in ("base_link", "link1", "link2", "link3", "link4", "link5", "link6", "link7"):
        (target / "p7_v3" / f"{name}.STL").write_bytes(b"mesh")

    calls: list[tuple[str, str]] = []

    def fake_resolver(directory: str, *, marker: str) -> Path:
        calls.append((directory, marker))
        return target

    monkeypatch.setattr(pull_assets, "resolve_robot_asset_dir", fake_resolver)

    assert pull_assets.main(["--robot", "a2arm"]) == 0
    assert calls == [("robots/a2arm/meshes", "adapter_plate.STL")]
    assert "9 STL files" in capsys.readouterr().out


def test_pull_assets_t800_resolves_both_asset_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    targets = {
        "robots/t800/assets": _populate(tmp_path / "assets", suffix=".obj", count=26),
        "robots/t800/textures": _populate(tmp_path / "textures", suffix=".png", count=15),
    }
    calls: list[tuple[str, str]] = []

    def fake_resolver(directory: str, *, marker: str) -> Path:
        calls.append((directory, marker))
        return targets[directory]

    monkeypatch.setattr(pull_assets, "resolve_robot_asset_dir", fake_resolver)

    assert pull_assets.main(["--robot", "t800"]) == 0
    assert calls == [
        ("robots/t800/assets", "LINK_BASE.obj"),
        ("robots/t800/textures", "LINK_BASE.png"),
    ]
    output = capsys.readouterr().out
    assert "26 OBJ files" in output
    assert "15 PNG files" in output


def test_pull_assets_x2_keeps_single_mesh_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    target = _populate(tmp_path / "meshes", suffix=".STL", count=2)
    calls: list[tuple[str, str]] = []

    def fake_resolver(directory: str, *, marker: str) -> Path:
        calls.append((directory, marker))
        return target

    monkeypatch.setattr(pull_assets, "resolve_robot_asset_dir", fake_resolver)

    assert pull_assets.main(["--robot", "x2"]) == 0
    assert calls == [("robots/x2/meshes", "pelvis.STL")]
