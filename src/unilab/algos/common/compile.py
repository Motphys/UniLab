from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from typing import Any, cast

import torch

_WARNED_REASONS: set[str] = set()


def _warn_once(reason: str) -> None:
    if reason in _WARNED_REASONS:
        return
    _WARNED_REASONS.add(reason)
    message = f"WARNING: torch.compile is unavailable for CUDA; using eager mode ({reason})."
    try:
        from rich.console import Console

        Console(stderr=True).print(message, style="yellow")
    except Exception:  # pragma: no cover - best-effort diagnostic only
        print(message, file=sys.stderr)


def get_torch_compile_for_cuda(
    device: torch.device | str, *, warn: bool = False
) -> Callable[..., Any] | None:
    """Return ``torch.compile`` when CUDA Inductor dependencies are available."""
    compile_fn = getattr(torch, "compile", None)
    if torch.device(device).type != "cuda":
        return None
    if compile_fn is None:
        if warn:
            _warn_once("torch.compile is not present in this PyTorch build")
        return None
    if (
        getattr(compile_fn, "__module__", "") == "torch"
        and importlib.util.find_spec("triton") is None
    ):
        if warn:
            _warn_once(
                "Triton is not installed; this environment cannot use CUDA Inductor. "
                "PyTorch's Windows torch.compile documentation currently covers "
                "CPU/XPU Inductor, not the CUDA/Triton path"
            )
        return None
    return cast(Callable[..., Any], compile_fn)
