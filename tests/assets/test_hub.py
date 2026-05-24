"""Tests for the motion asset resolver (``unilab.assets.hub``)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.assets.hub import resolve_motion_files


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

    expected_relative = str(missing.relative_to(ASSETS_ROOT_PATH))

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
