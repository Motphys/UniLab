"""Cold-path motion asset resolver with Hugging Face fallback.

Guarantees that requested motion files exist on disk before returning.
When a file is missing locally, it is downloaded from the configured
Hugging Face dataset repo and placed under ``ASSETS_ROOT_PATH`` so that
existing path references remain valid.

This module is a **cold-path** utility — import and call it once during
environment / loader initialisation, never inside step or reset.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from unilab.assets import ASSETS_ROOT_PATH

logger = logging.getLogger(__name__)

_HF_REPO_ID = "LeeLeno/unilab-motions"
_HF_REPO_TYPE = "dataset"


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
        return _resolve_single(motion_file)
    return [_resolve_single(p) for p in motion_file]


def _resolve_single(path_str: str) -> str:
    """Resolve one motion file path, downloading if absent."""
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
                f"Motion file not found and path is not under "
                f"ASSETS_ROOT_PATH ({ASSETS_ROOT_PATH}): {path_str}"
            ) from None

    return _download_from_hf(relative)


def _download_from_hf(relative_path: str) -> str:
    """Download *relative_path* from the HF dataset repo."""
    try:
        from huggingface_hub import hf_hub_download  # lazy import
    except ImportError:
        raise ImportError(
            f"Motion file '{relative_path}' not found locally. "
            "Install huggingface_hub to enable automatic downloading:\n"
            "  uv sync\n"
            "Or:\n"
            "  uv pip install huggingface_hub"
        ) from None

    logger.info(
        "Downloading %s from HF repo %s ...", relative_path, _HF_REPO_ID
    )
    local_path = hf_hub_download(
        repo_id=_HF_REPO_ID,
        filename=relative_path,
        repo_type=_HF_REPO_TYPE,
        local_dir=str(ASSETS_ROOT_PATH),
    )
    logger.info("Downloaded to %s", local_path)
    return str(local_path)
