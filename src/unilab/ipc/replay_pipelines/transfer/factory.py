"""Replay transfer backend factory."""

from __future__ import annotations

import torch

from unilab.ipc.replay_pipelines.transfer.base import ReplayTransferBackend
from unilab.ipc.replay_pipelines.transfer.cuda_like import CudaLikeReplayTransferBackend
from unilab.ipc.replay_pipelines.transfer.torch_copy import TorchCopyReplayTransferBackend
from unilab.ipc.replay_pipelines.transfer.xpu import XpuReplayTransferBackend


def build_replay_transfer_backend(
    *,
    device: torch.device,
    ring_depth: int,
) -> ReplayTransferBackend:
    """Build the transfer backend for a learner device."""
    if device.type == "cuda":
        return CudaLikeReplayTransferBackend(device=device, ring_depth=ring_depth)
    if device.type == "xpu":
        return XpuReplayTransferBackend(device=device, ring_depth=ring_depth)
    return TorchCopyReplayTransferBackend(device=device, ring_depth=ring_depth)
