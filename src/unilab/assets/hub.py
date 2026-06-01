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
import os
from collections.abc import Sequence
from pathlib import Path

from unilab.assets import ASSETS_ROOT_PATH

logger = logging.getLogger(__name__)

_HF_MOTIONS_REPO_ID = "unilabsim/unilab-motions"
_HF_CACHES_REPO_ID = "unilabsim/unilab-caches"
_HF_SCENES_REPO_ID = "unilabsim/unilab-scenes"
_HF_CHECKPOINTS_REPO_ID = "unilabsim/unilab-checkpoints"
_HF_REPO_TYPE = "dataset"
_HF_OFFICIAL_ENDPOINT = "https://huggingface.co"


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
) -> str | list[str]:
    """Ensure grasp cache file(s) exist locally, downloading from HF if needed.

    Args:
        cache_file: Absolute path or ``ASSETS_ROOT_PATH``-relative path
            (single string or sequence of strings).

    Returns:
        Resolved absolute path(s) guaranteed to exist on disk.
        A single string input returns a single string; a sequence input
        returns a list of strings.
    """
    if isinstance(cache_file, str):
        return _resolve_single(cache_file, repo_id=_HF_CACHES_REPO_ID)
    return [_resolve_single(p, repo_id=_HF_CACHES_REPO_ID) for p in cache_file]


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


def _resolve_single(path_str: str, *, repo_id: str = _HF_MOTIONS_REPO_ID) -> str:
    """Resolve one asset file path, downloading if absent."""
    path = Path(path_str)

    # Already exists locally — fast path.
    if path.exists():
        return str(path)

    # Try interpreting as ASSETS_ROOT_PATH-relative.
    if not path.is_absolute():
        local = ASSETS_ROOT_PATH / path
        if local.exists():
            return str(local)
        relative = path_str
    else:
        # Extract the portion relative to ASSETS_ROOT_PATH so we can
        # request the matching file from the HF repo.
        try:
            relative = str(path.relative_to(ASSETS_ROOT_PATH))
        except ValueError:
            raise FileNotFoundError(
                f"Asset file not found and path is not under "
                f"ASSETS_ROOT_PATH ({ASSETS_ROOT_PATH}): {path_str}"
            ) from None

    return _download_from_hf(relative, repo_id=repo_id)


def _hf_download(hf_hub_download, relative_path: str, *, repo_id: str) -> str:  # type: ignore[no-untyped-def]
    """Call ``hf_hub_download`` with the standard arguments."""
    return str(
        hf_hub_download(
            repo_id=repo_id,
            filename=relative_path,
            repo_type=_HF_REPO_TYPE,
            local_dir=str(ASSETS_ROOT_PATH),
        )
    )


def _download_from_hf(
    relative_path: str,
    *,
    repo_id: str = _HF_MOTIONS_REPO_ID,
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
        local_path = _hf_download(hf_hub_download, relative_path, repo_id=repo_id)
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
                local_path = _hf_download(hf_hub_download, relative_path, repo_id=repo_id)
            finally:
                os.environ["HF_ENDPOINT"] = original
        else:
            raise

    logger.info("Downloaded to %s", local_path)
    return local_path


# ---------------------------------------------------------------------------
# Scene directory resolver
# ---------------------------------------------------------------------------


def _snapshot_download(snapshot_download_fn, directory: str, *, repo_id: str) -> str:  # type: ignore[no-untyped-def]
    """Call ``snapshot_download`` with the standard arguments."""
    return str(
        snapshot_download_fn(
            repo_id=repo_id,
            repo_type=_HF_REPO_TYPE,
            allow_patterns=f"{directory}/**",
            local_dir=str(ASSETS_ROOT_PATH),
        )
    )


def resolve_scene_dir(directory: str, *, marker: str = "teaser.xml") -> Path:
    """Ensure a scene directory exists locally, downloading from HF if needed.

    Args:
        directory: ``ASSETS_ROOT_PATH``-relative directory path
            (e.g. ``"scenes/teaser"``).
        marker: A file inside the directory used to check completeness.

    Returns:
        Absolute ``Path`` to the resolved directory.
    """
    target = ASSETS_ROOT_PATH / directory
    if (target / marker).is_file():
        return target

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            f"Scene directory '{directory}' not found locally. "
            "Install huggingface_hub to enable automatic downloading:\n"
            "  uv sync\n"
            "Or:\n"
            "  uv pip install huggingface_hub"
        ) from None

    repo_id = _HF_SCENES_REPO_ID
    logger.info("Downloading %s from HF repo %s ...", directory, repo_id)

    try:
        _snapshot_download(snapshot_download, directory, repo_id=repo_id)
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
                _snapshot_download(snapshot_download, directory, repo_id=repo_id)
            finally:
                os.environ["HF_ENDPOINT"] = original
        else:
            raise

    logger.info("Downloaded scene directory to %s", target)
    return target
