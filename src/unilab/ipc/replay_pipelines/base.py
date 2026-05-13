"""Base types for replay pipeline abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol, runtime_checkable

import torch


@dataclass(frozen=True)
class ReplayTickMetadata:
    tick_id: int
    snapshot_ptr: int
    snapshot_size: int
    sample_seed: int
    sample_count: int
    batch_host_slot: int | None = None
    batch_gpu_slot: int | None = None


@runtime_checkable
class ReplayPipeline(Protocol):
    def start_prepare(
        self,
        tick_id: int,
        sample_count: int,
        min_snapshot_ptr: int | None = None,
    ) -> bool: ...
    def batch_ready(self, tick_id: int, sample_count: int) -> bool: ...
    def wait_ready(self) -> None: ...
    def wait_until_ready(self, tick_id: int, sample_count: int) -> bool: ...
    def sample_large_batch(self, tick_id: int, sample_count: int) -> Dict[str, torch.Tensor]: ...
    def after_tick(self) -> None: ...
    def close(self) -> None: ...
