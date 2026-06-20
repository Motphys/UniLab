from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import Any, cast

import torch


def get_torch_compile_for_cuda(device: torch.device | str) -> Callable[..., Any] | None:
    """Return ``torch.compile`` when CUDA Inductor dependencies are available."""
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None or torch.device(device).type != "cuda":
        return None
    if (
        getattr(compile_fn, "__module__", "") == "torch"
        and importlib.util.find_spec("triton") is None
    ):
        return None
    return cast(Callable[..., Any], compile_fn)
