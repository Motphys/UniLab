"""Replay transfer backend factory."""

from __future__ import annotations

import torch

from unilab.ipc.replay_pipelines.transfer.base import ReplayTransferBackend
from unilab.ipc.replay_pipelines.transfer.cuda_like import CudaLikeReplayTransferBackend
from unilab.ipc.replay_pipelines.transfer.torch_copy import TorchCopyReplayTransferBackend


def build_replay_transfer_backend(
    *,
    device: torch.device,
    ring_depth: int,
) -> ReplayTransferBackend:
    """Build the transfer backend for a learner device."""
    if device.type == "cuda":
        return CudaLikeReplayTransferBackend(device=device, ring_depth=ring_depth)
    if device.type == "mps":
        return TorchCopyReplayTransferBackend(device=device, ring_depth=ring_depth)
    raise ValueError(f"Replay transfer requires a CUDA or MPS device, got {device.type!r}")
