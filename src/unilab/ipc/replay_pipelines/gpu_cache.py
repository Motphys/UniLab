"""GpuCacheReplayPipeline — thin wrapper around the existing GPU-cache path."""

from __future__ import annotations

from typing import Dict

import torch

from unilab.ipc.replay_buffer import ReplayBuffer


class GpuCacheReplayPipeline:
    """Control-path pipeline: delegates to ReplayBuffer.sample() (existing behaviour)."""

    def __init__(self, replay_buffer: ReplayBuffer, trace_recorder=None) -> None:
        self._replay_buffer = replay_buffer
        self._trace_recorder = trace_recorder

    def wait_ready(self) -> None:
        return None

    def sample_large_batch(self, tick_id: int, sample_count: int) -> Dict[str, torch.Tensor]:
        return self._replay_buffer.sample(sample_count)

    def after_tick(self) -> None:
        return None

    def close(self) -> None:
        return None
