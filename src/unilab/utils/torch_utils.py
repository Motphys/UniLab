"""Generic array <-> torch conversion utilities."""

from __future__ import annotations

import numpy as np
import torch


def to_torch(x, device: str | torch.device) -> torch.Tensor:
    """Convert numpy-like input to torch on the target device.

    Supports torch tensors, numpy arrays, and any array exposing ``__dlpack__``.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    try:
        if hasattr(x, "__dlpack__"):
            return torch.from_dlpack(x).to(device)  # pyright: ignore[reportPrivateImportUsage]
    except Exception:
        pass
    arr = np.asarray(x, dtype=np.float32)
    return torch.from_numpy(arr).to(device)


def to_numpy(x) -> np.ndarray:
    """Convert torch tensor or numpy-like input to numpy."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
