"""Optional native H2D submit helper with graceful fallback.

Pre-validates build prerequisites before attempting JIT compilation.
Falls back to ``dst.copy_(src, non_blocking=True)`` on a side stream
when the native extension is unavailable — functionally identical for
pinned-memory transfers (the ROCm path already uses this).
"""

from __future__ import annotations

import logging
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import torch

logger = logging.getLogger(__name__)

_SOURCE = Path(__file__).with_name("native_h2d_ext.cpp")
_NATIVE_AVAILABLE: bool | None = None
_DIAGNOSTIC: str = ""


def _check_build_prerequisites() -> tuple[bool, str]:
    """Pre-validate JIT compilation requirements."""
    try:
        from torch.utils.cpp_extension import is_ninja_available

        if not is_ninja_available():
            return False, "ninja not found. Install: pip install ninja"
    except ImportError:
        return False, "torch.utils.cpp_extension unavailable"

    from torch.utils.cpp_extension import get_cxx_compiler

    compiler = get_cxx_compiler()
    if shutil.which(compiler) is None:
        return False, (f"C++ compiler '{compiler}' not found. Install: sudo apt install g++")

    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        return False, (
            "CUDA_HOME not set and nvcc not found. Install CUDA toolkit or set CUDA_HOME."
        )

    header = Path(CUDA_HOME) / "include" / "cuda_runtime_api.h"
    if not header.exists():
        targets_header = (
            Path(CUDA_HOME) / "targets" / "x86_64-linux" / "include" / "cuda_runtime_api.h"
        )
        if not targets_header.exists():
            return False, (
                f"cuda_runtime_api.h not found in {CUDA_HOME}/include/ "
                f"or targets/x86_64-linux/include/. "
                f"Verify CUDA toolkit installation."
            )

    if not _SOURCE.exists():
        return False, f"Extension source not found: {_SOURCE}"

    return True, ""


@lru_cache(maxsize=1)
def _try_load_extension() -> Any | None:
    global _NATIVE_AVAILABLE, _DIAGNOSTIC

    ok, reason = _check_build_prerequisites()
    if not ok:
        _DIAGNOSTIC = reason
        _NATIVE_AVAILABLE = False
        logger.info(
            "Native H2D extension unavailable: %s. "
            "Using pure-PyTorch async copy (no performance impact for pinned memory).",
            reason,
        )
        return None

    try:
        from torch.utils.cpp_extension import CUDA_HOME, load

        extra_include_paths = []
        if CUDA_HOME is not None:
            extra_include_paths.append(str(Path(CUDA_HOME) / "include"))
            targets_include = Path(CUDA_HOME) / "targets" / "x86_64-linux" / "include"
            if targets_include.exists():
                extra_include_paths.append(str(targets_include))
        ext = load(
            name="unilab_native_h2d",
            sources=[str(_SOURCE)],
            extra_cflags=["-O3"],
            extra_include_paths=extra_include_paths,
            verbose=False,
        )
        _NATIVE_AVAILABLE = True
        return ext
    except Exception as exc:
        _DIAGNOSTIC = f"Compilation failed: {exc}"
        _NATIVE_AVAILABLE = False
        logger.warning("Native H2D compilation failed: %s. Using pure-PyTorch fallback.", exc)
        return None


def is_available() -> bool:
    """True if native extension is loaded or loadable."""
    global _NATIVE_AVAILABLE
    if _NATIVE_AVAILABLE is None:
        _try_load_extension()
    return bool(_NATIVE_AVAILABLE)


def get_diagnostic() -> str:
    """Human-readable reason for unavailability."""
    if _NATIVE_AVAILABLE is None:
        _try_load_extension()
    return _DIAGNOSTIC


def ensure_available() -> bool:
    """Attempt to load native extension. Returns True if available."""
    return is_available()


def submit_h2d(
    dst: torch.Tensor,
    src: torch.Tensor,
    stream: torch.cuda.Stream,
) -> None:
    """Submit one async H2D copy on an existing CUDA stream."""
    ext = _try_load_extension()
    if ext is not None:
        ext.submit_h2d(dst, src, int(stream.cuda_stream))
    else:
        with torch.cuda.stream(stream):
            dst.copy_(src, non_blocking=True)
