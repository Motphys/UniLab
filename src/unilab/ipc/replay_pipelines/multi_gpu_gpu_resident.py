"""Rank-local multi-GPU GPU-resident replay pipeline.

Each learner rank owns an independent GPU-resident mirror of the shared CPU
:class:`ReplayBuffer`.  There is no collector pack step and no per-tick batch
H2D; instead a rank-local daemon thread incrementally mirrors newly written
rows into the rank's device storage, and sampling is GPU-side.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from unilab.ipc.replay_buffer import ReplayBuffer
from unilab.ipc.replay_pipelines.gpu_resident import GPUResidentReplayPipeline


class MultiGPUGPUResidentReplayPipeline:
    """Per-rank GPU-resident replay mirror for multi-GPU off-policy training."""

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        *,
        rank: int,
        world_size: int,
        device: str,
        sample_count: int,
        base_seed: int = 0,
        trace_recorder=None,
        trace_cuda_events: bool = True,
        pack_layout: str = "packed",
        use_critic_graph_packed_source: bool = False,
    ) -> None:
        self._rank = int(rank)
        self._world_size = int(world_size)
        self._device = torch.device(device)
        self._sample_count = int(sample_count)
        # GPUResidentReplayPipeline derives per-tick sample seeds from base_seed
        # + tick_id.  Shift the base per-rank so each rank samples independent
        # indices from its own mirror.
        self._base_seed = int(base_seed) + self._rank * 100_007
        self._trace_recorder = trace_recorder
        self._trace_cuda_events = bool(trace_cuda_events)
        self._pack_layout = str(pack_layout)
        self._use_critic_graph_packed_source = bool(use_critic_graph_packed_source)

        self._pipeline = GPUResidentReplayPipeline(
            replay_buffer,
            device=device,
            sample_count=sample_count,
            base_seed=self._base_seed,
            trace_recorder=trace_recorder,
            trace_cuda_events=trace_cuda_events,
            pack_layout=pack_layout,
            use_critic_graph_packed_source=use_critic_graph_packed_source,
        )

    @property
    def h2d_submitter(self) -> str:
        return self._pipeline.h2d_submitter

    @property
    def transfer_manifest(self) -> dict[str, object]:
        manifest = dict(self._pipeline.transfer_manifest)
        manifest["rank"] = self._rank
        manifest["world_size"] = self._world_size
        manifest["pipeline"] = "multi_gpu_gpu_resident"
        return manifest

    @property
    def last_incremental_h2d_time_s(self) -> float:
        return float(getattr(self._pipeline, "last_incremental_h2d_time_s", 0.0))

    def start_prepare(
        self,
        tick_id: int,
        sample_count: int,
        min_snapshot_ptr: int | None = None,
        sample_snapshot_mode: str = "service",
        exclude_write_count: int = 0,
    ) -> bool:
        del sample_snapshot_mode, exclude_write_count
        return self._pipeline.start_prepare(
            tick_id=tick_id,
            sample_count=sample_count,
            min_snapshot_ptr=min_snapshot_ptr,
        )

    def batch_ready(self, tick_id: int, sample_count: int) -> bool:
        return self._pipeline.batch_ready(tick_id, sample_count)

    def wait_ready(self) -> None:
        return self._pipeline.wait_ready()

    def wait_until_ready(self, tick_id: int, sample_count: int) -> bool:
        return self._pipeline.wait_until_ready(tick_id, sample_count)

    def sample_large_batch(self, tick_id: int, sample_count: int) -> Dict[str, torch.Tensor]:
        return self._pipeline.sample_large_batch(tick_id, sample_count)

    def after_tick(self) -> None:
        self._pipeline.after_tick()

    def close(self) -> None:
        self._pipeline.close()
