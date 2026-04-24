"""Backend detection and dtype mapping utilities."""

import importlib
import importlib.util
import platform
import subprocess
import sys
from functools import lru_cache
from typing import Any, Dict

_IS_MACOS = platform.system() == "Darwin"

np: Any | None = None
try:
    import numpy as _np

    np = _np
except Exception:
    np = None

torch: Any | None = None
try:
    import torch as _torch

    torch = _torch
except Exception:
    torch = None


@lru_cache(maxsize=1)
def _mlx_runtime_usable() -> bool:
    if not _IS_MACOS:
        return False
    if importlib.util.find_spec("mlx.core") is None:
        return False
    result = subprocess.run(
        [sys.executable, "-c", "import mlx.core"], capture_output=True, text=True, timeout=10
    )
    return result.returncode == 0


@lru_cache(maxsize=1)
def _get_mlx() -> Any | None:
    if not _mlx_runtime_usable():
        return None
    try:
        return importlib.import_module("mlx.core")
    except Exception:
        return None


def available_backends() -> Dict[str, bool]:
    backends = {
        "numpy": np is not None,
        "torch_cpu": torch is not None,
    }
    if _IS_MACOS:
        backends["torch_mps"] = bool(
            torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        backends["mlx"] = _mlx_runtime_usable()
    else:
        backends["torch_cuda"] = bool(torch and torch.cuda.is_available())
    return backends


def numpy_dtype(dtype_name: str):
    if np is None:
        raise RuntimeError("numpy unavailable")
    return {"float16": np.float16, "float32": np.float32}[dtype_name]


def torch_dtype(dtype_name: str):
    if torch is None:
        raise RuntimeError("torch unavailable")
    return {"float16": torch.float16, "float32": torch.float32}[dtype_name]


def mlx_dtype(dtype_name: str):
    mx = _get_mlx()
    if mx is None:
        raise RuntimeError("mlx unavailable")
    return {"float16": mx.float16, "float32": mx.float32}[dtype_name]


def sync_backend(backend: str) -> None:
    if backend == "torch_mps" and torch and hasattr(torch.backends, "mps"):
        torch.mps.synchronize()
    elif backend == "torch_cuda" and torch and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif backend == "mlx" and _get_mlx():
        pass  # mx.eval handled per-operation
