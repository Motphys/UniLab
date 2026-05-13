"""Optional native H2D submit helper for replay pipeline experiments."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.cpp_extension import CUDA_HOME, load

_SOURCE = Path(__file__).with_name("native_h2d_ext.cpp")


@lru_cache(maxsize=1)
def _load_extension():
    extra_include_paths = []
    if CUDA_HOME is not None:
        extra_include_paths.append(str(Path(CUDA_HOME) / "include"))
    return load(
        name="unilab_native_h2d",
        sources=[str(_SOURCE)],
        extra_cflags=["-O3"],
        extra_include_paths=extra_include_paths,
        verbose=False,
    )


def ensure_available() -> None:
    """Build/load the native helper before measured H2D submissions."""
    _load_extension()


def submit_h2d(
    dst: torch.Tensor,
    src: torch.Tensor,
    stream: torch.cuda.Stream,
) -> None:
    """Submit one async H2D copy on an existing CUDA stream."""
    ext = cast(Any, _load_extension())
    ext.submit_h2d(dst, src, int(stream.cuda_stream))
