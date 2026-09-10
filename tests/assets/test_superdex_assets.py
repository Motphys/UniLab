"""Native SuperDex assets resolve via an audited local root or the HF hub."""

from pathlib import Path

import pytest

from unilab.assets import hub
from unilab.assets.hub import SUPERDEX_ROBOT_ASSET_SPECS, resolve_superdex_robot_asset

MODEL = "bots/arms/fr3_v2/fr3_v2.superdex_bot"


@pytest.fixture
def asset_root(tmp_path: Path) -> Path:
    model = tmp_path / MODEL
    for path in (model, *(model.parent / item for item in SUPERDEX_ROBOT_ASSET_SPECS[MODEL])):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    return tmp_path


def test_local_superdex_asset_root_precedence_and_absolute_paths(
    asset_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUPERDEX_ASSETS_PATH", "/missing/other/root")
    expected = str(asset_root / MODEL)
    assert resolve_superdex_robot_asset(MODEL, assets_root=str(asset_root)) == expected
    assert resolve_superdex_robot_asset(expected, assets_root=str(asset_root)) == expected
    monkeypatch.setenv("SUPERDEX_ASSETS_PATH", str(asset_root))
    assert resolve_superdex_robot_asset(MODEL) == expected


def _fake_snapshot_dir(asset_root: Path, calls: list[tuple[str, str, str]]):
    def fake(directory: str, *, repo_id: str, marker: str, show_progress: bool = True) -> Path:
        calls.append((directory, repo_id, marker))
        return asset_root / directory

    return fake


def test_superdex_assets_download_from_hub_without_local_root(
    asset_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SUPERDEX_ASSETS_PATH", raising=False)
    monkeypatch.setattr(hub, "ASSETS_ROOT_PATH", asset_root)
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(hub, "_resolve_snapshot_dir", _fake_snapshot_dir(asset_root, calls))

    assert resolve_superdex_robot_asset(MODEL) == str(asset_root / MODEL)
    assert calls == [("bots/arms/fr3_v2", hub._HF_ROBOTS_REPO_ID, "fr3_v2.superdex_bot")]


def test_superdex_assets_explicit_root_never_downloads(
    asset_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("audited local root must not trigger HF downloads")

    monkeypatch.setattr(hub, "_resolve_snapshot_dir", fail)
    assert resolve_superdex_robot_asset(MODEL, assets_root=str(asset_root)) == str(
        asset_root / MODEL
    )


def test_superdex_assets_reject_unregistered_and_escaping_paths(asset_root: Path) -> None:
    with pytest.raises(ValueError, match="not registered"):
        resolve_superdex_robot_asset("unknown.superdex_bot", assets_root=str(asset_root))
    with pytest.raises(ValueError, match="inside configured root"):
        resolve_superdex_robot_asset("../outside.superdex_bot", assets_root=str(asset_root))


def test_superdex_hub_mode_rejects_unregistered_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SUPERDEX_ASSETS_PATH", raising=False)
    monkeypatch.setattr(hub, "ASSETS_ROOT_PATH", tmp_path)
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(hub, "_resolve_snapshot_dir", _fake_snapshot_dir(tmp_path, calls))
    with pytest.raises(ValueError, match="not registered"):
        resolve_superdex_robot_asset("bots/arms/unknown/unknown.superdex_bot")
    with pytest.raises(ValueError, match="not registered"):
        resolve_superdex_robot_asset("/outside/fr3_v2.superdex_bot")
    assert calls == []


def test_superdex_assets_fail_before_sdk_on_incomplete_collision(asset_root: Path) -> None:
    missing = asset_root / Path(MODEL).parent / "collision/fr3_link4_collision.mochi.h5"
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="fr3_link4_collision"):
        resolve_superdex_robot_asset(MODEL, assets_root=str(asset_root))


def test_superdex_hub_mode_checks_completeness_after_download(
    asset_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SUPERDEX_ASSETS_PATH", raising=False)
    monkeypatch.setattr(hub, "ASSETS_ROOT_PATH", asset_root)
    (asset_root / Path(MODEL).parent / "NOTICE").unlink()
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(hub, "_resolve_snapshot_dir", _fake_snapshot_dir(asset_root, calls))
    with pytest.raises(FileNotFoundError, match="NOTICE"):
        resolve_superdex_robot_asset(MODEL)
