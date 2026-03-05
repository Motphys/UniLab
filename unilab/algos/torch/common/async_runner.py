"""PyTorch-specific async runner."""

from __future__ import annotations
import torch
from unilab.ipc.async_runner import AsyncRunner as BaseAsyncRunner


def _get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class AsyncRunner(BaseAsyncRunner):
    """PyTorch async runner with device detection."""

    def _get_default_device(self) -> str:
        return _get_default_device()
