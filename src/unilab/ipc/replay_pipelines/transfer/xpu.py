"""Intel XPU replay transfer backend."""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any

import torch

from unilab.ipc.replay_pipelines.base import ReplayTickMetadata


class XpuReplayTransferBackend:
    """XPU stream/event backend for packed replay batch transfer."""

    device_family = "xpu"
    h2d_submitter = "torch_xpu_copy_stream"
    host_memory_kind = "pageable_shared"
    host_pinned = False
    direct_pinned_shared = False
    supports_async_submit = True
    supports_timing_events = False

    def __init__(self, *, device: torch.device, ring_depth: int) -> None:
        xpu = getattr(torch, "xpu", None)
        required = ("Stream", "Event", "stream", "current_stream")
        if xpu is None or any(getattr(xpu, name, None) is None for name in required):
            raise RuntimeError("XPU replay transfer requires torch.xpu Stream/Event support")
        self.device = device
        self._xpu: Any = xpu
        self._ring_depth = int(ring_depth)
        self._copy_stream = xpu.Stream(device=device)
        self._ready_events = [xpu.Event() for _ in range(self._ring_depth)]
        self._submitted = [False for _ in range(self._ring_depth)]

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
        device_context = getattr(self._xpu, "device", None)
        context: Any = device_context(self.device) if callable(device_context) else nullcontext()
        with context:
            with self._xpu.stream(self._copy_stream):
                dst.copy_(src, non_blocking=True)
                self._ready_events[slot].record(self._copy_stream)
        self._submitted[slot] = True
        return (time.perf_counter_ns() - h2d_begin_ns) / 1e9

    def clear_ready(self, slot: int) -> None:
        self._submitted[slot] = False

    def ready_query(self, slot: int) -> bool:
        return bool(self._submitted[slot]) and bool(self._ready_events[slot].query())

    def synchronize_ready(self, slot: int) -> None:
        if self._submitted[slot]:
            self._ready_events[slot].synchronize()

    def wait_current_stream_for_ready(self, slot: int) -> None:
        if not self._submitted[slot]:
            return
        current_stream = self._xpu.current_stream(self.device)
        wait_event = getattr(current_stream, "wait_event", None)
        if callable(wait_event):
            wait_event(self._ready_events[slot])
        else:
            self._ready_events[slot].synchronize()

    def close(self) -> None:
        return None
