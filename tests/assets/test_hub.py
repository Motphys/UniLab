"""Tests for the asset resolvers (``unilab.assets.hub``)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.assets.hub import (
    resolve_grasp_cache_files,
    resolve_motion_files,
    resolve_robot_asset_dir,
    resolve_scene_dir,
)

# ---------------------------------------------------------------------------
# Local-path fast path
# ---------------------------------------------------------------------------


def test_resolve_returns_existing_absolute_path(tmp_path: Path):
    npz = tmp_path / "test.npz"
    np.savez(npz, fps=np.array([30]))
    assert resolve_motion_files(str(npz)) == str(npz)


def test_resolve_returns_list_for_list_input(tmp_path: Path):
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    np.savez(a, fps=np.array([30]))
    np.savez(b, fps=np.array([30]))
    result = resolve_motion_files([str(a), str(b)])
    assert result == [str(a), str(b)]


# ---------------------------------------------------------------------------
# Missing-file error paths
# ---------------------------------------------------------------------------


def test_resolve_raises_for_absolute_path_outside_assets_root():
    with pytest.raises(FileNotFoundError, match="not under ASSETS_ROOT_PATH"):
        resolve_motion_files("/nonexistent/outside/motion.npz")


def test_resolve_raises_import_error_when_hf_hub_missing():
    """When the file is under ASSETS_ROOT_PATH but missing, and
    huggingface_hub is not installed, a clear ImportError is raised."""
    missing = ASSETS_ROOT_PATH / "motions" / "g1" / "__test_nonexistent__.npz"
    assert not missing.exists()

    with patch.dict("sys.modules", {"huggingface_hub": None}):
        with pytest.raises(ImportError, match="huggingface_hub"):
            resolve_motion_files(str(missing))


# ---------------------------------------------------------------------------
# HF download path (mocked)
# ---------------------------------------------------------------------------


def test_resolve_calls_hf_hub_download_for_missing_file():
    """When a file under ASSETS_ROOT_PATH is missing, the resolver should
    call ``hf_hub_download`` with the correct relative path."""
    missing = ASSETS_ROOT_PATH / "motions" / "g1" / "__test_nonexistent__.npz"
    assert not missing.exists()

    expected_relative = missing.relative_to(ASSETS_ROOT_PATH).as_posix()

    fake_download = MagicMock(return_value=str(missing))
    fake_module = MagicMock()
    fake_module.hf_hub_download = fake_download

    with patch.dict("sys.modules", {"huggingface_hub": fake_module}):
        result = resolve_motion_files(str(missing))

    assert result == str(missing)
    fake_download.assert_called_once_with(
        repo_id="unilabsim/unilab-motions",
        filename=expected_relative,
        repo_type="dataset",
        local_dir=str(ASSETS_ROOT_PATH),
    )


def test_resolve_relative_path_falls_back_to_hf():
    """A relative path that doesn't exist under ASSETS_ROOT_PATH triggers
    an HF download with that relative path as the filename."""
    rel = "motions/g1/__test_nonexistent_rel__.npz"
    local = ASSETS_ROOT_PATH / rel
    assert not local.exists()

    fake_download = MagicMock(return_value=str(local))
    fake_module = MagicMock()
    fake_module.hf_hub_download = fake_download

    with patch.dict("sys.modules", {"huggingface_hub": fake_module}):
        result = resolve_motion_files(rel)

    assert result == str(local)
    fake_download.assert_called_once_with(
        repo_id="unilabsim/unilab-motions",
        filename=rel,
        repo_type="dataset",
        local_dir=str(ASSETS_ROOT_PATH),
    )


def test_resolve_absolute_windows_path_uses_posix_hf_filename():
    missing = ASSETS_ROOT_PATH / "motions" / "g1" / "__test_nonexistent_windows_abs__.npz"
    assert not missing.exists()

    expected_relative = missing.relative_to(ASSETS_ROOT_PATH).as_posix()

    fake_download = MagicMock(return_value=str(missing))
    fake_module = MagicMock()
    fake_module.hf_hub_download = fake_download

    with patch.dict("sys.modules", {"huggingface_hub": fake_module}):
        result = resolve_motion_files(str(missing))

    assert result == str(missing)
    fake_download.assert_called_once_with(
        repo_id="unilabsim/unilab-motions",
        filename=expected_relative,
        repo_type="dataset",
        local_dir=str(ASSETS_ROOT_PATH),
    )


def test_resolve_relative_windows_path_uses_posix_hf_filename():
    rel = r"motions\g1\__test_nonexistent_windows_rel__.npz"
    local = ASSETS_ROOT_PATH / rel
    assert not local.exists()

    fake_download = MagicMock(return_value=str(local))
    fake_module = MagicMock()
    fake_module.hf_hub_download = fake_download

    with patch.dict("sys.modules", {"huggingface_hub": fake_module}):
        result = resolve_motion_files(rel)

    assert result == str(local)
    fake_download.assert_called_once_with(
        repo_id="unilabsim/unilab-motions",
        filename="motions/g1/__test_nonexistent_windows_rel__.npz",
        repo_type="dataset",
        local_dir=str(ASSETS_ROOT_PATH),
    )


# ---------------------------------------------------------------------------
# Grasp cache resolve — local fast path
# ---------------------------------------------------------------------------


def test_resolve_grasp_cache_returns_existing_path(tmp_path: Path):
    npy = tmp_path / "cache.npy"
    np.save(npy, np.array([1.0]))
    assert resolve_grasp_cache_files(str(npy)) == str(npy)


def test_resolve_grasp_cache_returns_list_for_list_input(tmp_path: Path):
    a = tmp_path / "a.npy"
    b = tmp_path / "b.npy"
    np.save(a, np.array([1.0]))
    np.save(b, np.array([2.0]))
    result = resolve_grasp_cache_files([str(a), str(b)])
    assert result == [str(a), str(b)]


# ---------------------------------------------------------------------------
# Grasp cache resolve — HF download path (mocked)
# ---------------------------------------------------------------------------


def test_resolve_grasp_cache_calls_hf_download_with_caches_repo():
    """Grasp cache resolver should use the unilab-caches repo, not unilab-motions."""
    missing = ASSETS_ROOT_PATH / "caches" / "__test_nonexistent__.npy"
    assert not missing.exists()

    expected_relative = missing.relative_to(ASSETS_ROOT_PATH).as_posix()

    fake_download = MagicMock(return_value=str(missing))
    fake_module = MagicMock()
    fake_module.hf_hub_download = fake_download

    with patch.dict("sys.modules", {"huggingface_hub": fake_module}):
        result = resolve_grasp_cache_files(str(missing))

    assert result == str(missing)
    fake_download.assert_called_once()
    kwargs = fake_download.call_args.kwargs
    assert kwargs["repo_id"] == "unilabsim/unilab-caches"
    assert kwargs["filename"] == expected_relative
    assert kwargs["repo_type"] == "dataset"
    assert kwargs["local_dir"] == str(ASSETS_ROOT_PATH)
    assert kwargs["tqdm_class"] is not None


def test_resolve_grasp_cache_relative_path_uses_caches_repo():
    """A relative path triggers HF download with the caches repo."""
    rel = "caches/__test_nonexistent_rel__.npy"
    local = ASSETS_ROOT_PATH / rel
    assert not local.exists()

    fake_download = MagicMock(return_value=str(local))
    fake_module = MagicMock()
    fake_module.hf_hub_download = fake_download

    with patch.dict("sys.modules", {"huggingface_hub": fake_module}):
        result = resolve_grasp_cache_files(rel)

    assert result == str(local)
    fake_download.assert_called_once()
    kwargs = fake_download.call_args.kwargs
    assert kwargs["repo_id"] == "unilabsim/unilab-caches"
    assert kwargs["filename"] == rel
    assert kwargs["repo_type"] == "dataset"
    assert kwargs["local_dir"] == str(ASSETS_ROOT_PATH)
    assert kwargs["tqdm_class"] is not None


def test_resolve_grasp_cache_can_disable_download_progress():
    rel = "caches/__test_nonexistent_quiet__.npy"
    local = ASSETS_ROOT_PATH / rel
    assert not local.exists()

    fake_download = MagicMock(return_value=str(local))
    fake_module = MagicMock()
    fake_module.hf_hub_download = fake_download

    with patch.dict("sys.modules", {"huggingface_hub": fake_module}):
        result = resolve_grasp_cache_files(rel, show_progress=False)

    assert result == str(local)
    fake_download.assert_called_once()
    kwargs = fake_download.call_args.kwargs
    assert kwargs["repo_id"] == "unilabsim/unilab-caches"
    assert kwargs["filename"] == rel
    assert kwargs["repo_type"] == "dataset"
    assert kwargs["local_dir"] == str(ASSETS_ROOT_PATH)
    progress = kwargs["tqdm_class"](range(1))
    try:
        assert progress.disable is True
    finally:
        progress.close()


# Scene directory resolver
# ---------------------------------------------------------------------------


def test_resolve_scene_dir_returns_immediately_when_marker_exists(tmp_path: Path):
    """When the marker file exists, resolve_scene_dir returns without
    downloading anything."""
    scene_dir = tmp_path / "scenes" / "teaser"
    scene_dir.mkdir(parents=True)
    (scene_dir / "teaser.xml").write_text("<mujoco/>")

    with patch("unilab.assets.hub.ASSETS_ROOT_PATH", tmp_path):
        result = resolve_scene_dir("scenes/teaser")

    assert result == scene_dir


def test_resolve_scene_dir_calls_snapshot_download_when_missing(tmp_path: Path):
    """When the marker file is absent, resolve_scene_dir triggers a
    snapshot_download with the correct arguments."""
    fake_snapshot = MagicMock(return_value=str(tmp_path))
    fake_module = MagicMock()
    fake_module.snapshot_download = fake_snapshot

    with patch("unilab.assets.hub.ASSETS_ROOT_PATH", tmp_path):
        with patch.dict("sys.modules", {"huggingface_hub": fake_module}):
            resolve_scene_dir("scenes/teaser")

    fake_snapshot.assert_called_once_with(
        repo_id="unilabsim/unilab-scenes",
        repo_type="dataset",
        allow_patterns="scenes/teaser/**",
        local_dir=str(tmp_path),
    )


def test_resolve_scene_dir_windows_path_uses_posix_hf_pattern(tmp_path: Path):
    fake_snapshot = MagicMock(return_value=str(tmp_path))
    fake_module = MagicMock()
    fake_module.snapshot_download = fake_snapshot

    with patch("unilab.assets.hub.ASSETS_ROOT_PATH", tmp_path):
        with patch.dict("sys.modules", {"huggingface_hub": fake_module}):
            resolve_scene_dir(r"scenes\teaser")

    fake_snapshot.assert_called_once_with(
        repo_id="unilabsim/unilab-scenes",
        repo_type="dataset",
        allow_patterns="scenes/teaser/**",
        local_dir=str(tmp_path),
    )


def test_resolve_scene_dir_raises_import_error_when_hf_hub_missing(tmp_path: Path):
    """When the scene directory is missing and huggingface_hub is not
    installed, a clear ImportError is raised."""
    with patch("unilab.assets.hub.ASSETS_ROOT_PATH", tmp_path):
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with pytest.raises(ImportError, match="huggingface_hub"):
                resolve_scene_dir("scenes/teaser")


def test_resolve_robot_asset_dir_can_disable_snapshot_progress(tmp_path: Path):
    fake_snapshot = MagicMock(return_value=str(tmp_path))
    fake_module = MagicMock()
    fake_module.snapshot_download = fake_snapshot

    with patch("unilab.assets.hub.ASSETS_ROOT_PATH", tmp_path):
        with patch.dict("sys.modules", {"huggingface_hub": fake_module}):
            resolve_robot_asset_dir("robots/x2/meshes", marker="pelvis.STL", show_progress=False)

    fake_snapshot.assert_called_once()
    kwargs = fake_snapshot.call_args.kwargs
    assert kwargs["repo_id"] == "unilabsim/unilab-robots"
    assert kwargs["repo_type"] == "dataset"
    assert kwargs["allow_patterns"] == "robots/x2/meshes/**"
    assert kwargs["local_dir"] == str(tmp_path)
    tqdm_class = kwargs["tqdm_class"]
    progress = tqdm_class(range(1))
    try:
        assert progress.disable is True
    finally:
        progress.close()


# ---------------------------------------------------------------------------
# Robot asset registry + scene-path resolution
# ---------------------------------------------------------------------------


def test_robot_asset_specs_cover_hf_hosted_robots():
    """Every robot whose binary assets live on HF is registered exactly once."""
    from unilab.assets.hub import ROBOT_ASSET_SPECS

    expected = {
        "a2",
        "allegro_hand",
        "g1",
        "go2",
        "go2_arm",
        "go2w",
        "microduck",
        "sharpa_wave",
        "t800",
        "x2",
    }
    assert set(ROBOT_ASSET_SPECS) == expected
    for robot, specs in ROBOT_ASSET_SPECS.items():
        assert specs, robot
        for directory, marker, pattern, label in specs:
            # go2_arm / go2w additionally reference the shared go2 mesh dir.
            assert directory.startswith("robots/")
            assert marker and pattern and label


def test_ensure_robot_assets_resolves_registered_robot(monkeypatch: pytest.MonkeyPatch):
    from unilab.assets import hub

    calls: list[tuple[str, str]] = []

    def fake_resolve(directory: str, *, marker: str) -> Path:
        calls.append((directory, marker))
        return Path(directory)

    monkeypatch.setattr(hub, "resolve_robot_asset_dir", fake_resolve)

    hub.ensure_robot_assets_for_paths(["src/unilab/assets/robots/g1/scene_flat.xml", None, ""])

    assert calls == [
        ("robots/g1/assets", "head_link.STL"),
        ("robots/g1/textures", "floor.png"),
    ]


def test_ensure_robot_assets_handles_absolute_and_windows_paths(
    monkeypatch: pytest.MonkeyPatch,
):
    from unilab.assets import hub

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        hub,
        "resolve_robot_asset_dir",
        lambda directory, *, marker: calls.append((directory, marker)) or Path(directory),
    )

    hub.ensure_robot_assets_for_paths(
        [
            "/home/user/project/src/unilab/assets/robots/go2/go2.xml",
            r"src\unilab\assets\robots\sharpa_wave\right_sharpa_wave.xml",
        ]
    )

    assert calls == [
        ("robots/go2/assets", "base_0.obj"),
        ("robots/sharpa_wave/meshes", "DP_HB1_4F.STL"),
    ]


def test_ensure_robot_assets_ignores_unknown_robots(monkeypatch: pytest.MonkeyPatch):
    from unilab.assets import hub

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        hub,
        "resolve_robot_asset_dir",
        lambda directory, *, marker: calls.append((directory, marker)) or Path(directory),
    )

    hub.ensure_robot_assets_for_paths(
        [
            "src/unilab/assets/robots/go1/scene_flat.xml",
            "src/unilab/assets/objects/cube.xml",
            "some/other/scene.xml",
        ]
    )

    assert calls == []


def test_create_backend_resolves_robot_assets_before_dispatch(monkeypatch: pytest.MonkeyPatch):
    """create_backend must ensure HF-hosted robot assets before loading XML."""
    import unilab.base.backend_factory as backend_pkg
    from unilab.base.scene import SceneCfg

    seen: list[list[str | None]] = []

    def fake_ensure(paths):
        seen.append(list(paths))

    monkeypatch.setattr(backend_pkg, "ensure_robot_assets_for_paths", fake_ensure)

    scene = SceneCfg(
        model_file="src/unilab/assets/robots/g1/scene_flat.xml",
        fragment_files=["src/unilab/assets/robots/g1/locomotion_task.xml"],
    )
    with pytest.raises(ValueError, match="unknown UniSim backend"):
        backend_pkg.create_backend("__bogus__", scene, 1, 0.02)

    assert seen == [
        [
            "src/unilab/assets/robots/g1/scene_flat.xml",
            None,
            "src/unilab/assets/robots/g1/locomotion_task.xml",
        ]
    ]


def test_ensure_robot_assets_go2_arm_pulls_shared_go2_meshes(
    monkeypatch: pytest.MonkeyPatch,
):
    """go2_arm XMLs reference ``../go2/assets``; both dirs must resolve."""
    from unilab.assets import hub

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        hub,
        "resolve_robot_asset_dir",
        lambda directory, *, marker: calls.append((directory, marker)) or Path(directory),
    )

    hub.ensure_robot_assets_for_paths(["src/unilab/assets/robots/go2_arm/scene_flat.xml"])

    assert calls == [
        ("robots/go2_arm/assets", "arm_base_0.obj"),
        ("robots/go2/assets", "base_0.obj"),
    ]
