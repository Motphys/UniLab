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


def test_pull_assets_x2_keeps_single_mesh_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    target = _populate(tmp_path / "meshes", suffix=".STL", count=2)
    calls: list[tuple[str, str, bool]] = []

    def fake_resolver(directory: str, *, marker: str, show_progress: bool) -> Path:
        calls.append((directory, marker, show_progress))
        return target

    monkeypatch.setattr(pull_assets, "resolve_robot_asset_dir", fake_resolver)

    assert pull_assets.main(["--robot", "x2"]) == 0
    assert calls == [("robots/x2/meshes", "pelvis.STL", False)]
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "Downloading x2 assets ..."
    assert lines[-1].startswith("Robot assets ready: 1 robots, 1 directories, 2 files")


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
    lines = output.strip().splitlines()
    assert lines[0] == "Downloading g1 assets ..."
    assert lines[-1].startswith("Robot assets ready: 1 robots, 2 directories, 56 files")


def test_pull_assets_fr3_v2_uses_superdex_resolver(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    target = _populate(tmp_path / "fr3_v2", suffix=".h5", count=3)
    calls: list[str] = []

    def fake_superdex_resolver(model_file: str) -> str:
        calls.append(model_file)
        return str(target / "fr3_v2.superdex_bot")

    monkeypatch.setattr(pull_assets, "resolve_superdex_robot_asset", fake_superdex_resolver)

    assert pull_assets.main(["--robot", "fr3_v2"]) == 0
    assert calls == ["bots/arms/fr3_v2/fr3_v2.superdex_bot"]
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "Downloading fr3_v2 assets ..."
    assert lines[-1].startswith("Robot assets ready: 1 robots, 1 directories, 3 files")


def test_pull_assets_all_covers_every_registered_robot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    from unilab.assets.hub import ROBOT_ASSET_SPECS

    target = _populate(tmp_path / "dir", suffix=".stl", count=1)
    superdex_target = _populate(tmp_path / "fr3_v2", suffix=".h5", count=1)
    calls: list[tuple[str, bool]] = []
    superdex_calls: list[str] = []

    def fake_resolver(directory: str, *, marker: str, show_progress: bool) -> Path:
        calls.append((directory, show_progress))
        return target

    def fake_superdex_resolver(model_file: str) -> str:
        superdex_calls.append(model_file)
        return str(superdex_target / "fr3_v2.superdex_bot")

    monkeypatch.setattr(pull_assets, "resolve_robot_asset_dir", fake_resolver)
    monkeypatch.setattr(pull_assets, "resolve_superdex_robot_asset", fake_superdex_resolver)

    robots = sorted(ROBOT_ASSET_SPECS) + sorted(pull_assets._superdex_bot_names())
    assert pull_assets.main(["--robot", "all"]) == 0
    assert calls == [
        (directory, False)
        for robot in sorted(ROBOT_ASSET_SPECS)
        for directory, _marker, _pattern, _label in ROBOT_ASSET_SPECS[robot]
    ]
    superdex_bots = pull_assets._superdex_bot_names()
    assert superdex_calls == [superdex_bots[name] for name in sorted(superdex_bots)]
    output_lines = capsys.readouterr().out.strip().splitlines()
    assert len(output_lines) == len(robots) + 1
    assert output_lines[:-1] == [f"Downloading {robot} assets ..." for robot in robots]
    assert output_lines[-1].startswith(f"Robot assets ready: {len(robots)} robots")


def test_pull_assets_verbose_keeps_output_compact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    target = _populate(tmp_path / "meshes", suffix=".STL", count=2)
    calls: list[bool] = []

    def fake_resolver(directory: str, *, marker: str, show_progress: bool) -> Path:
        calls.append(show_progress)
        return target

    monkeypatch.setattr(pull_assets, "resolve_robot_asset_dir", fake_resolver)

    assert pull_assets.main(["--robot", "x2", "--verbose"]) == 0
    assert calls == [False]
    assert len(capsys.readouterr().out.strip().splitlines()) == 2
