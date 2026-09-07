"""Native SuperDex assets resolve only through the audited local registry."""

from pathlib import Path

import pytest

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


def test_superdex_assets_require_explicit_local_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUPERDEX_ASSETS_PATH", raising=False)
    with pytest.raises(FileNotFoundError, match="SUPERDEX_ASSETS_PATH"):
        resolve_superdex_robot_asset(MODEL)


def test_superdex_assets_reject_unregistered_and_escaping_paths(asset_root: Path) -> None:
    with pytest.raises(ValueError, match="not registered"):
        resolve_superdex_robot_asset("unknown.superdex_bot", assets_root=str(asset_root))
    with pytest.raises(ValueError, match="inside configured root"):
        resolve_superdex_robot_asset("../outside.superdex_bot", assets_root=str(asset_root))


def test_superdex_assets_fail_before_sdk_on_incomplete_collision(asset_root: Path) -> None:
    missing = asset_root / Path(MODEL).parent / "collision/fr3_link4_collision.mochi.h5"
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="fr3_link4_collision"):
        resolve_superdex_robot_asset(MODEL, assets_root=str(asset_root))
