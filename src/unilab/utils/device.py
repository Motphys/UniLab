from __future__ import annotations

from typing import Callable, cast

import torch


def _xpu_available() -> bool:
    xpu = getattr(torch, "xpu", None)
    is_available = getattr(xpu, "is_available", None)
    return bool(callable(is_available) and is_available())


def get_default_device() -> str:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if _xpu_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _device_count(device_type: str) -> int | None:
    if device_type == "cuda":
        return int(torch.cuda.device_count())
    if device_type == "xpu":
        xpu = getattr(torch, "xpu", None)
        device_count = getattr(xpu, "device_count", None)
        if callable(device_count):
            return int(cast(Callable[[], int], device_count)())
    return None


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    is_available = getattr(mps, "is_available", None)
    return bool(callable(is_available) and is_available())


def _parse_device_alias(value: str) -> tuple[str, int | None]:
    raw = value.strip().lower()
    if not raw:
        raise ValueError("Device alias must not be empty")
    if ":" not in raw:
        return raw, None
    base, index_text = raw.split(":", 1)
    if not index_text:
        raise ValueError(f"Device alias {value!r} has an empty index")
    try:
        index = int(index_text)
    except ValueError as exc:
        raise ValueError(f"Device alias {value!r} has a non-integer index") from exc
    if index < 0:
        raise ValueError(f"Device alias {value!r} has a negative index")
    return base, index


def _resolve_indexed_device(device_type: str, index: int | None, original: str) -> str:
    count = _device_count(device_type)
    if count is not None and index is not None and index >= count:
        raise ValueError(
            f"Requested device {original!r} resolves to {device_type}:{index}, "
            f"but only {count} {device_type} device(s) are available"
        )
    return device_type if index is None else f"{device_type}:{index}"


def _resolve_mps_alias(index: int | None, original: str) -> str:
    if not _mps_available():
        raise ValueError(f"Requested device {original!r} requires MPS, but MPS is unavailable")
    if index not in (None, 0):
        raise ValueError(
            f"Requested device {original!r} cannot be mapped to MPS; only index 0 is valid"
        )
    return "mps"


def resolve_torch_device_alias(device: str | None, *, default: str = "cpu") -> str:
    """Resolve a cross-platform torch device alias to a concrete device string.

    ``gpu`` is an abstract accelerator alias. ``cuda`` is also accepted on
    macOS/MPS for config portability and maps to ``mps`` when CUDA is absent.
    The function validates the resolved device and never silently falls back to
    CPU for unavailable accelerators.
    """
    original = default if device is None else str(device)
    base, index = _parse_device_alias(original)

    if base == "cpu":
        if index is not None:
            raise ValueError(f"CPU device {original!r} must not include an index")
        return "cpu"

    if base == "mps":
        return _resolve_mps_alias(index, original)

    if base == "xpu":
        if not _xpu_available():
            raise ValueError(f"Requested device {original!r} requires XPU, but XPU is unavailable")
        return _resolve_indexed_device("xpu", index, original)

    if base == "cuda":
        if torch.cuda.is_available():
            return _resolve_indexed_device("cuda", index, original)
        if _mps_available():
            return _resolve_mps_alias(index, original)
        raise ValueError(f"Requested device {original!r} requires CUDA, but CUDA is unavailable")

    if base == "gpu":
        if torch.cuda.is_available():
            return _resolve_indexed_device("cuda", index, original)
        if _xpu_available():
            return _resolve_indexed_device("xpu", index, original)
        if _mps_available():
            return _resolve_mps_alias(index, original)
        raise ValueError(
            f"Requested device {original!r} requires an accelerator, but none is available"
        )

    raise ValueError(f"Unsupported device alias {original!r}; expected cpu, gpu, cuda, mps, or xpu")
