"""Replay tick metadata shared by the device transfer owner."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReplayTickMetadata:
    tick_id: int
    snapshot_ptr: int
    snapshot_size: int
    sample_seed: int
    sample_count: int
    batch_host_slot: int | None = None
    batch_gpu_slot: int | None = None
