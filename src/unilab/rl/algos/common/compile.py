from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from typing import Any, cast

import torch

_WARNED_REASONS: set[tuple[str, str]] = set()


def _warn_once(device_type: str, reason: str) -> None:
    warning_key = (device_type, reason)
    if warning_key in _WARNED_REASONS:
        return
    _WARNED_REASONS.add(warning_key)
    message = (
        f"WARNING: torch.compile is unavailable for {device_type.upper()}; "
        f"using eager mode ({reason})."
    )
    try:
        from rich.console import Console

        Console(stderr=True).print(message, style="yellow")
    except Exception:  # pragma: no cover - best-effort diagnostic only
        print(message, file=sys.stderr)


def configure_torch_compile_compat() -> None:
    """Avoid the Torch 2.10 CUDA shape-padding benchmark during graph warmup."""
    version = torch.__version__.split("+", 1)[0].split(".")[:2]
    try:
        release = tuple(int(component) for component in version)
    except ValueError:
        return
    if release != (2, 10):
        return
    try:
        from torch._inductor import config as inductor_config
    except ImportError:
        return
    inductor_config.shape_padding = False


def get_torch_compile_for_device(
    device: torch.device | str, *, warn: bool = False
) -> Callable[..., Any] | None:
    """Return a device-supported ``torch.compile`` entrypoint.

    CUDA keeps its explicit Triton dependency check.  MPS uses the portable
    Torch compiler entrypoint and deliberately does not require Triton.
    """
    compile_fn = getattr(torch, "compile", None)
    device_type = torch.device(device).type
    if device_type not in {"cuda", "mps"}:
        return None
    reason: str | None = None
    if compile_fn is None:
        reason = "torch.compile is not present in this PyTorch build"
    elif device_type == "cuda" and (
        getattr(compile_fn, "__module__", "") == "torch"
        and importlib.util.find_spec("triton") is None
    ):
        reason = (
            "Triton is not installed; this environment cannot use CUDA Inductor. "
            "PyTorch's Windows torch.compile documentation currently covers "
            "CPU/XPU Inductor, not the CUDA/Triton path"
        )
    if reason is not None:
        if warn:
            _warn_once(device_type, reason)
        return None
    if device_type == "cuda":
        configure_torch_compile_compat()
    return cast(Callable[..., Any], compile_fn)


def is_hip_runtime() -> bool:
    """Whether PyTorch presents a ROCm/HIP CUDA device runtime."""
    torch_version = cast(Any, torch.__dict__.get("version"))
    return bool(getattr(torch_version, "hip", None))
