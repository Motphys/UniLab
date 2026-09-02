"""Generic array <-> torch conversion utilities."""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def to_torch(x, device: str | torch.device) -> torch.Tensor:
    """Convert numpy-like input to torch on the target device.

    Supports torch tensors, numpy arrays, and any array exposing ``__dlpack__``.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, np.ndarray):
        # Simulators such as Drake expose float64 buffers by default, while
        # policy networks are constructed in float32 (and MPS rejects float64).
        # Normalize floating observations at this boundary so every backend
        # follows the same learner contract.
        if np.issubdtype(x.dtype, np.floating) and x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        return torch.from_numpy(x).to(device)
    if hasattr(x, "__dlpack__"):
        try:
            return torch.from_dlpack(x).to(device)  # pyright: ignore[reportPrivateImportUsage]
        except (
            AttributeError,
            BufferError,
            NotImplementedError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as exc:
            logger.warning(
                "to_torch: dlpack conversion failed for %s (%s); "
                "falling back to a float32 numpy copy",
                type(x).__name__,
                exc,
            )
    arr = np.asarray(x, dtype=np.float32)
    return torch.from_numpy(arr).to(device)


def to_numpy(x) -> np.ndarray:
    """Convert torch tensor or numpy-like input to numpy."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
