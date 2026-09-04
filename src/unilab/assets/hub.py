"""Cold-path asset resolver with Hugging Face fallback.

Guarantees that requested asset files exist on disk before returning.
When a file is missing locally, it is downloaded from the configured
Hugging Face dataset repo and placed under ``ASSETS_ROOT_PATH`` so that
existing path references remain valid.

This module is a **cold-path** utility — import and call it once during
environment / loader initialisation, never inside step or reset.
"""

from __future__ import annotations

import logging
import ntpath
import os
import posixpath
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from unilab.assets import ASSETS_ROOT_PATH

logger = logging.getLogger(__name__)

_HF_MOTIONS_REPO_ID = "unilabsim/unilab-motions"
_HF_CACHES_REPO_ID = "unilabsim/unilab-caches"
_HF_SCENES_REPO_ID = "unilabsim/unilab-scenes"
_HF_CHECKPOINTS_REPO_ID = "unilabsim/unilab-checkpoints"
_HF_ROBOTS_REPO_ID = "unilabsim/unilab-robots"
_HF_REPO_TYPE = "dataset"
_HF_OFFICIAL_ENDPOINT = "https://huggingface.co"

# Robot binary assets (meshes / textures) hosted on ``_HF_ROBOTS_REPO_ID``
# instead of being committed to git. Each entry maps a robot directory name
# to the ``ASSETS_ROOT_PATH``-relative directories that live on HF:
# ``(directory, completeness marker, count glob, count label)``. The glob and
# label only feed the ``unilab-pull-assets`` summary line; resolution keys
# off the marker file.
ROBOT_ASSET_SPECS: dict[str, tuple[tuple[str, str, str, str], ...]] = {
    "a2": (("robots/a2/assets", "a2/base_link.STL", "**/*.STL", "STL"),),
    "allegro_hand": (("robots/allegro_hand/assets", "base_link.stl", "**/*", "asset"),),
    "g1": (
        ("robots/g1/assets", "head_link.STL", "**/*", "asset"),
        ("robots/g1/textures", "floor.png", "*.png", "PNG"),
    ),
    "go2": (("robots/go2/assets", "base_0.obj", "**/*", "asset"),),
    # go2_arm reuses the Go2 base meshes via ``../go2/assets`` references.
    "go2_arm": (
        ("robots/go2_arm/assets", "arm_base_0.obj", "**/*", "asset"),
        ("robots/go2/assets", "base_0.obj", "**/*", "asset"),
    ),
    # go2w points its meshdir at ``../go2/assets``.
    "go2w": (("robots/go2/assets", "base_0.obj", "**/*", "asset"),),
    "microduck": (("robots/microduck/assets", "trunk_base.stl", "*.stl", "STL"),),
    "sharpa_wave": (("robots/sharpa_wave/meshes", "DP_HB1_4F.STL", "*.STL", "STL"),),
    "x2": (("robots/x2/meshes", "pelvis.STL", "*.STL", "STL"),),
}


def resolve_motion_files(
    motion_file: str | Sequence[str],
) -> str | list[str]:
    """Ensure motion file(s) exist locally, downloading from HF if needed.

    Args:
        motion_file: Absolute path or ``ASSETS_ROOT_PATH``-relative path
            (single string or sequence of strings).

    Returns:
        Resolved absolute path(s) guaranteed to exist on disk.
        A single string input returns a single string; a sequence input
        returns a list of strings.
    """
    if isinstance(motion_file, str):
        return _resolve_single(motion_file, repo_id=_HF_MOTIONS_REPO_ID)
    return [_resolve_single(p, repo_id=_HF_MOTIONS_REPO_ID) for p in motion_file]


def resolve_grasp_cache_files(
    cache_file: str | Sequence[str],
    *,
    show_progress: bool = False,
) -> str | list[str]:
    """Ensure grasp cache file(s) exist locally, downloading from HF if needed.

    Args:
        cache_file: Absolute path or ``ASSETS_ROOT_PATH``-relative path
            (single string or sequence of strings).
        show_progress: Whether Hugging Face downloads may render progress bars.

    Returns:
        Resolved absolute path(s) guaranteed to exist on disk.
        A single string input returns a single string; a sequence input
        returns a list of strings.
    """
    if isinstance(cache_file, str):
        return _resolve_single(cache_file, repo_id=_HF_CACHES_REPO_ID, show_progress=show_progress)
    return [
        _resolve_single(p, repo_id=_HF_CACHES_REPO_ID, show_progress=show_progress)
        for p in cache_file
    ]


def resolve_checkpoint_file(
    checkpoint_file: str | Sequence[str],
) -> str | list[str]:
    """Ensure checkpoint file(s) exist locally, downloading from HF if needed.

    Args:
        checkpoint_file: Absolute path or ``ASSETS_ROOT_PATH``-relative path
            (single string or sequence of strings).

    Returns:
        Resolved absolute path(s) guaranteed to exist on disk.
        A single string input returns a single string; a sequence input
        returns a list of strings.
    """
    if isinstance(checkpoint_file, str):
        return _resolve_single(checkpoint_file, repo_id=_HF_CHECKPOINTS_REPO_ID)
    return [_resolve_single(p, repo_id=_HF_CHECKPOINTS_REPO_ID) for p in checkpoint_file]


def _resolve_single(
    path_str: str,
    *,
    repo_id: str = _HF_MOTIONS_REPO_ID,
    show_progress: bool = True,
) -> str:
    """Resolve one asset file path, downloading if absent."""
    path = Path(path_str)
    is_absolute_input = path.is_absolute() or ntpath.isabs(path_str) or posixpath.isabs(path_str)

    # Already exists locally — fast path.
    if path.exists():
        return str(path)

    # Try interpreting as ASSETS_ROOT_PATH-relative.
    if not is_absolute_input:
        local = ASSETS_ROOT_PATH / path
        if local.exists():
            return str(local)
        relative = _hf_relative_path(path_str)
    else:
        # Extract the portion relative to ASSETS_ROOT_PATH so we can
        # request the matching file from the HF repo.
        try:
            relative = path.relative_to(ASSETS_ROOT_PATH).as_posix()
        except ValueError:
            raise FileNotFoundError(
                f"Asset file not found and path is not under "
                f"ASSETS_ROOT_PATH ({ASSETS_ROOT_PATH}): {path_str}"
            ) from None

    return _download_from_hf(relative, repo_id=repo_id, show_progress=show_progress)


def _hf_relative_path(path_str: str) -> str:
    """Return a repo-relative HF path with POSIX separators."""
    return path_str.replace("\\", "/")


def _hf_download(
    hf_hub_download,
    relative_path: str,
    *,
    repo_id: str,
    show_progress: bool = True,
) -> str:  # type: ignore[no-untyped-def]
    """Call ``hf_hub_download`` with the standard arguments."""
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "filename": relative_path,
        "repo_type": _HF_REPO_TYPE,
        "local_dir": str(ASSETS_ROOT_PATH),
    }
    if not show_progress:
        kwargs["tqdm_class"] = _silent_tqdm_class()
    return str(hf_hub_download(**kwargs))


def _download_from_hf(
    relative_path: str,
    *,
    repo_id: str = _HF_MOTIONS_REPO_ID,
    show_progress: bool = True,
) -> str:
    """Download *relative_path* from an HF dataset repo.

    If the current ``HF_ENDPOINT`` (e.g. a mirror) fails, automatically
    retries with the official ``https://huggingface.co`` endpoint.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            f"Asset file '{relative_path}' not found locally. "
            "Install huggingface_hub to enable automatic downloading:\n"
            "  uv sync\n"
            "Or:\n"
            "  uv pip install huggingface_hub"
        ) from None

    logger.info("Downloading %s from HF repo %s ...", relative_path, repo_id)

    try:
        local_path = _hf_download(
            hf_hub_download,
            relative_path,
            repo_id=repo_id,
            show_progress=show_progress,
        )
    except Exception:
        # If a mirror endpoint is configured and it failed, retry with
        # the official endpoint before giving up.
        current_endpoint = os.environ.get("HF_ENDPOINT", "")
        if current_endpoint and current_endpoint != _HF_OFFICIAL_ENDPOINT:
            logger.warning(
                "Download failed with HF_ENDPOINT=%s, retrying with %s ...",
                current_endpoint,
                _HF_OFFICIAL_ENDPOINT,
            )
            original = os.environ["HF_ENDPOINT"]
            os.environ["HF_ENDPOINT"] = _HF_OFFICIAL_ENDPOINT
            try:
                local_path = _hf_download(
                    hf_hub_download,
                    relative_path,
                    repo_id=repo_id,
                    show_progress=show_progress,
                )
            finally:
                os.environ["HF_ENDPOINT"] = original
        else:
            raise

    logger.info("Downloaded to %s", local_path)
    return local_path


# ---------------------------------------------------------------------------
# Scene directory resolver
# ---------------------------------------------------------------------------


def _silent_tqdm_class() -> type[Any]:
    """Return a tqdm class that keeps HF snapshot downloads silent."""
    from tqdm.auto import tqdm

    class _SilentTqdm(tqdm):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

    return _SilentTqdm


def _snapshot_download(
    snapshot_download_fn,
    directory: str,
    *,
    repo_id: str,
    show_progress: bool = True,
) -> str:  # type: ignore[no-untyped-def]
    """Call ``snapshot_download`` with the standard arguments."""
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": _HF_REPO_TYPE,
        "allow_patterns": f"{directory}/**",
        "local_dir": str(ASSETS_ROOT_PATH),
    }
    if not show_progress:
        kwargs["tqdm_class"] = _silent_tqdm_class()
    return str(snapshot_download_fn(**kwargs))


def _resolve_snapshot_dir(
    directory: str, *, repo_id: str, marker: str, show_progress: bool = True
) -> Path:
    """Ensure an HF-hosted directory exists locally, downloading if needed.

    If the current ``HF_ENDPOINT`` (e.g. a mirror) fails, automatically
    retries with the official ``https://huggingface.co`` endpoint.

    Args:
        directory: ``ASSETS_ROOT_PATH``-relative directory path
            (e.g. ``"scenes/teaser"`` or ``"robots/x2/meshes"``).
        repo_id: HF dataset repo to pull from.
        marker: A file inside the directory used to check completeness.

    Returns:
        Absolute ``Path`` to the resolved directory.
    """
    hf_directory = _hf_relative_path(directory)
    target = ASSETS_ROOT_PATH / directory
    if (target / marker).is_file():
        return target

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            f"Asset directory '{directory}' not found locally. "
            "Install huggingface_hub to enable automatic downloading:\n"
            "  uv sync\n"
            "Or:\n"
            "  uv pip install huggingface_hub"
        ) from None

    logger.info("Downloading %s from HF repo %s ...", hf_directory, repo_id)

    try:
        _snapshot_download(
            snapshot_download, hf_directory, repo_id=repo_id, show_progress=show_progress
        )
    except Exception:
        current_endpoint = os.environ.get("HF_ENDPOINT", "")
        if current_endpoint and current_endpoint != _HF_OFFICIAL_ENDPOINT:
            logger.warning(
                "Download failed with HF_ENDPOINT=%s, retrying with %s ...",
                current_endpoint,
                _HF_OFFICIAL_ENDPOINT,
            )
            original = os.environ["HF_ENDPOINT"]
            os.environ["HF_ENDPOINT"] = _HF_OFFICIAL_ENDPOINT
            try:
                _snapshot_download(
                    snapshot_download, hf_directory, repo_id=repo_id, show_progress=show_progress
                )
            finally:
                os.environ["HF_ENDPOINT"] = original
        else:
            raise

    logger.info("Downloaded directory to %s", target)
    return target


def resolve_scene_dir(directory: str, *, marker: str = "teaser.xml") -> Path:
    """Ensure a scene directory exists locally, downloading from HF if needed.

    Args:
        directory: ``ASSETS_ROOT_PATH``-relative directory path
            (e.g. ``"scenes/teaser"``).
        marker: A file inside the directory used to check completeness.

    Returns:
        Absolute ``Path`` to the resolved directory.
    """
    return _resolve_snapshot_dir(directory, repo_id=_HF_SCENES_REPO_ID, marker=marker)


def resolve_robot_asset_dir(directory: str, *, marker: str, show_progress: bool = True) -> Path:
    """Ensure a robot asset directory (e.g. meshes) exists locally.

    Robot binary assets (STL meshes) are hosted on Hugging Face rather than
    committed to git. They are downloaded on first use and placed under their
    original path beneath ``ASSETS_ROOT_PATH`` so that XML ``meshdir``
    references resolve unchanged — no files need to be moved by hand.

    Args:
        directory: ``ASSETS_ROOT_PATH``-relative directory path
            (e.g. ``"robots/x2/meshes"``).
        marker: A file inside the directory used to check completeness
            (e.g. ``"pelvis.STL"``).
        show_progress: Whether Hugging Face snapshot downloads may render progress bars.

    Returns:
        Absolute ``Path`` to the resolved directory.
    """
    return _resolve_snapshot_dir(
        directory,
        repo_id=_HF_ROBOTS_REPO_ID,
        marker=marker,
        show_progress=show_progress,
    )


def ensure_robot_assets_for_paths(paths: Sequence[str | None]) -> None:
    """Resolve HF-hosted asset dirs for every robot referenced by *paths*.

    Cold-path helper called once before a backend parses scene/robot XML:
    any path under ``robots/<name>/`` whose robot is registered in
    ``ROBOT_ASSET_SPECS`` gets its HF-hosted directories downloaded when the
    completeness marker is missing. Paths outside the asset tree and unknown
    robots are ignored, so callers may pass every scene path unconditionally.
    """
    for path in paths:
        if not path:
            continue
        robot = _robot_name_from_path(path)
        if robot is None:
            continue
        for directory, marker, _pattern, _label in ROBOT_ASSET_SPECS[robot]:
            resolve_robot_asset_dir(directory, marker=marker)


def _robot_name_from_path(path_str: str) -> str | None:
    """Return the registered robot a ``robots/<name>/...`` path refers to."""
    parts = _hf_relative_path(path_str).split("/")
    for index, part in enumerate(parts[:-1]):
        if part == "robots" and parts[index + 1] in ROBOT_ASSET_SPECS:
            return parts[index + 1]
    return None
