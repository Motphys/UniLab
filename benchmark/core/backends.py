"""Backend detection and dtype mapping utilities."""
import platform
from typing import Dict

_IS_MACOS = platform.system() == "Darwin"

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None

try:
    import mlx.core as mx
except Exception:
    mx = None

def available_backends() -> Dict[str, bool]:
    backends = {
        "numpy": np is not None,
        "torch_cpu": torch is not None,
    }
    if _IS_MACOS:
        backends["torch_mps"] = bool(
            torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        backends["mlx"] = mx is not None
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
    if mx is None:
        raise RuntimeError("mlx unavailable")
    return {"float16": mx.float16, "float32": mx.float32}[dtype_name]

def sync_backend(backend: str) -> None:
    if backend == "torch_mps" and torch and hasattr(torch.backends, "mps"):
        torch.mps.synchronize()
    elif backend == "torch_cuda" and torch and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif backend == "mlx" and mx:
        pass  # mx.eval handled per-operation
