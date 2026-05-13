"""Replay transfer backend contract."""

from __future__ import annotations

from typing import Protocol

import torch

from unilab.ipc.replay_pipelines.base import ReplayTickMetadata


class ReplayTransferBackend(Protocol):
    """Device-specific host-to-device transfer backend."""

    device: torch.device
    device_family: str
    h2d_submitter: str
    host_memory_kind: str
    host_pinned: bool
    direct_pinned_shared: bool
    supports_async_submit: bool
    supports_timing_events: bool

    def register_host_slots(self, slots: list[torch.Tensor]) -> None: ...

    def allocate_device_slots(
        self,
        *,
        count: int,
        shape: tuple[int, int],
        dtype: torch.dtype,
    ) -> list[torch.Tensor]: ...

    def submit_h2d(
        self,
        *,
        slot: int,
        dst: torch.Tensor,
        src: torch.Tensor,
        metadata: ReplayTickMetadata | None,
        trace_recorder,
        trace_cuda_events: bool,
        h2d_bytes: int,
        pack_layout: str,
        pack_executor: str,
    ) -> float: ...

    def clear_ready(self, slot: int) -> None: ...

    def ready_query(self, slot: int) -> bool: ...

    def synchronize_ready(self, slot: int) -> None: ...

    def wait_current_stream_for_ready(self, slot: int) -> None: ...

    def close(self) -> None: ...
