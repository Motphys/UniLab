"""Convert MLX / numpy arrays to torch (e.g. MPS). MLX -> torch.mps uses torch.from_dlpack when available."""

from __future__ import annotations

import numpy as np
import torch


def mlx_to_torch(x, device: str | torch.device) -> torch.Tensor:
    """Convert MLX array or numpy array to torch on the given device. Prefer torch.from_dlpack for MLX -> MPS."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    # MLX array: try DLPack then fallback to numpy
    try:
        if hasattr(x, "__dlpack__"):
            t = torch.from_dlpack(x)
            return t.to(device)
    except Exception:
        pass
    arr = np.asarray(x, dtype=np.float32)
    return torch.from_numpy(arr).to(device)


def to_numpy(x) -> np.ndarray:
    """Convert MLX array or torch tensor to numpy for e.g. render_many."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
