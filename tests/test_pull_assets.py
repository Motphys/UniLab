"""Tests for the robot asset prefetch CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from unilab.assets import pull as pull_assets


def _populate(directory: Path, *, suffix: str, count: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (directory / f"asset_{index}{suffix}").write_bytes(b"asset")
    return directory


def test_pull_assets_t800_resolves_both_asset_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    targets = {
        "robots/t800/assets": _populate(tmp_path / "assets", suffix=".obj", count=26),
        "robots/t800/textures": _populate(tmp_path / "textures", suffix=".png", count=15),
    }
    calls: list[tuple[str, str, bool]] = []

    def fake_resolver(directory: str, *, marker: str, show_progress: bool) -> Path:
        calls.append((directory, marker, show_progress))
        return targets[directory]

    monkeypatch.setattr(pull_assets, "resolve_robot_asset_dir", fake_resolver)

    assert pull_assets.main(["--robot", "t800"]) == 0
    assert calls == [
        ("robots/t800/assets", "LINK_BASE.obj", False),
        ("robots/t800/textures", "LINK_BASE.png", False),
    ]
    output = capsys.readouterr().out
    assert "26 OBJ files" in output
    assert "15 PNG files" in output
    assert len(output.strip().splitlines()) == 1


def test_pull_assets_microduck_keeps_single_stl_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = _populate(tmp_path / "assets", suffix=".stl", count=47)
    calls: list[tuple[str, str, bool]] = []

    def fake_resolver(directory: str, *, marker: str, show_progress: bool) -> Path:
        calls.append((directory, marker, show_progress))
        return target

    monkeypatch.setattr(pull_assets, "resolve_robot_asset_dir", fake_resolver)

    assert pull_assets.main(["--robot", "microduck"]) == 0
    assert calls == [("robots/microduck/assets", "trunk_base.stl", False)]
    output = capsys.readouterr().out
    assert "47 STL files" in output
    assert len(output.strip().splitlines()) == 1


def test_pull_assets_x2_keeps_single_mesh_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    target = _populate(tmp_path / "meshes", suffix=".STL", count=2)
    calls: list[tuple[str, str, bool]] = []

    def fake_resolver(directory: str, *, marker: str, show_progress: bool) -> Path:
        calls.append((directory, marker, show_progress))
        return target

    monkeypatch.setattr(pull_assets, "resolve_robot_asset_dir", fake_resolver)

    assert pull_assets.main(["--robot", "x2"]) == 0
    assert calls == [("robots/x2/meshes", "pelvis.STL", False)]


def test_pull_assets_g1_resolves_assets_and_textures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    targets = {
        "robots/g1/assets": _populate(tmp_path / "assets", suffix=".STL", count=54),
        "robots/g1/textures": _populate(tmp_path / "textures", suffix=".png", count=2),
    }
    calls: list[tuple[str, str, bool]] = []

    def fake_resolver(directory: str, *, marker: str, show_progress: bool) -> Path:
        calls.append((directory, marker, show_progress))
        return targets[directory]

    monkeypatch.setattr(pull_assets, "resolve_robot_asset_dir", fake_resolver)

    assert pull_assets.main(["--robot", "g1"]) == 0
    assert calls == [
        ("robots/g1/assets", "head_link.STL", False),
        ("robots/g1/textures", "floor.png", False),
    ]
    output = capsys.readouterr().out
    assert "54 asset files" in output
    assert "2 PNG files" in output
    assert len(output.strip().splitlines()) == 1


def test_pull_assets_all_covers_every_registered_robot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    from unilab.assets.hub import ROBOT_ASSET_SPECS

    target = _populate(tmp_path / "dir", suffix=".stl", count=1)
    calls: list[tuple[str, bool]] = []

    def fake_resolver(directory: str, *, marker: str, show_progress: bool) -> Path:
        calls.append((directory, show_progress))
        return target

    monkeypatch.setattr(pull_assets, "resolve_robot_asset_dir", fake_resolver)

    assert pull_assets.main(["--robot", "all"]) == 0
    assert calls == [
        (directory, False)
        for robot in sorted(ROBOT_ASSET_SPECS)
        for directory, _marker, _pattern, _label in ROBOT_ASSET_SPECS[robot]
    ]
    output_lines = capsys.readouterr().out.strip().splitlines()
    assert len(output_lines) == len(ROBOT_ASSET_SPECS)
    assert all(" assets ready:" in line for line in output_lines)


def test_pull_assets_all_prints_one_line_per_robot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    from unilab.assets.hub import ROBOT_ASSET_SPECS

    def fake_resolver(directory: str, *, marker: str, show_progress: bool) -> Path:
        return _populate(tmp_path / Path(directory).name, suffix=".stl", count=1)

    monkeypatch.setattr(pull_assets, "resolve_robot_asset_dir", fake_resolver)

    assert pull_assets.main(["--robot", "all"]) == 0
    output = capsys.readouterr().out.strip()
    lines = output.splitlines()
    assert len(lines) == len(ROBOT_ASSET_SPECS)
    for robot in sorted(ROBOT_ASSET_SPECS):
        assert any(line.startswith(f"{robot} assets ready:") for line in lines)


def test_pull_assets_verbose_keeps_hf_progress_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    target = _populate(tmp_path / "meshes", suffix=".STL", count=2)
    calls: list[bool] = []

    def fake_resolver(directory: str, *, marker: str, show_progress: bool) -> Path:
        calls.append(show_progress)
        return target

    monkeypatch.setattr(pull_assets, "resolve_robot_asset_dir", fake_resolver)

    assert pull_assets.main(["--robot", "x2", "--verbose"]) == 0
    assert calls == [True]
