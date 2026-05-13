"""Portable torch-copy replay transfer backend."""

from __future__ import annotations

import threading
import time

import torch

from unilab.ipc.replay_pipelines.base import ReplayTickMetadata


class TorchCopyReplayTransferBackend:
    """Portable transfer backend for CPU, MPS, and other torch devices."""

    h2d_submitter = "torch_copy"
    host_memory_kind = "pageable_shared"
    host_pinned = False
    direct_pinned_shared = False
    supports_async_submit = False
    supports_timing_events = False

    def __init__(self, *, device: torch.device, ring_depth: int) -> None:
        self.device = device
        self.device_family = device.type
        self._ring_depth = int(ring_depth)
        self._ready_events = [threading.Event() for _ in range(self._ring_depth)]
        self._defer_copy_to_wait = device.type == "mps"
        self._pending_copies: list[tuple[torch.Tensor, torch.Tensor] | None] = [
            None for _ in range(self._ring_depth)
        ]
        self.last_wait_copy_time_s = 0.0
        if self._defer_copy_to_wait:
            self.h2d_submitter = "torch_copy_main_thread"

    def register_host_slots(self, slots: list[torch.Tensor]) -> None:
        del slots
        return None

    def allocate_device_slots(
        self,
        *,
        count: int,
        shape: tuple[int, int],
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        return [torch.empty(shape, dtype=dtype, device=self.device) for _ in range(count)]

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
    ) -> float:
        del metadata, trace_recorder, trace_cuda_events, h2d_bytes, pack_layout, pack_executor
        h2d_begin_ns = time.perf_counter_ns()
        self.clear_ready(slot)
        self.last_wait_copy_time_s = 0.0
        if self._defer_copy_to_wait:
            # PyTorch MPS command submission from a background transfer thread
            # can trip Metal command-buffer assertions.  Keep the collector CPU
            # pack overlap, but submit the actual MPS copy from the learner
            # thread when the batch is consumed.
            self._pending_copies[slot] = (dst, src)
            self._ready_events[slot].set()
            return (time.perf_counter_ns() - h2d_begin_ns) / 1e9
        dst.copy_(src, non_blocking=src.is_pinned())
        self._synchronize_device()
        self._ready_events[slot].set()
        return (time.perf_counter_ns() - h2d_begin_ns) / 1e9

    def clear_ready(self, slot: int) -> None:
        self._ready_events[slot].clear()
        self._pending_copies[slot] = None
        self.last_wait_copy_time_s = 0.0

    def ready_query(self, slot: int) -> bool:
        return self._ready_events[slot].is_set()

    def synchronize_ready(self, slot: int) -> None:
        self._ready_events[slot].wait()

    def wait_current_stream_for_ready(self, slot: int) -> None:
        self.synchronize_ready(slot)
        pending = self._pending_copies[slot]
        if pending is None:
            return
        dst, src = pending
        copy_begin_ns = time.perf_counter_ns()
        dst.copy_(src, non_blocking=False)
        self._synchronize_device()
        self.last_wait_copy_time_s = (time.perf_counter_ns() - copy_begin_ns) / 1e9
        self._pending_copies[slot] = None

    def close(self) -> None:
        for slot in range(len(self._pending_copies)):
            self._pending_copies[slot] = None
        self.last_wait_copy_time_s = 0.0
        return None

    def _synchronize_device(self) -> None:
        if self.device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()
            return
        if self.device.type == "xpu" and hasattr(torch, "xpu"):
            xpu = torch.xpu
            synchronize = getattr(xpu, "synchronize", None)
            if synchronize is not None:
                try:
                    synchronize(self.device)
                except TypeError:
                    synchronize()
