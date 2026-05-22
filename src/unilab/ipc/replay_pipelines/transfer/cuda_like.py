"""CUDA/ROCm replay transfer backend."""

from __future__ import annotations

import time
from typing import Any, cast

import torch

from unilab.ipc.replay_pipelines.base import ReplayTickMetadata


class CudaLikeReplayTransferBackend:
    """Pinned host to CUDA-like device transfer backend.

    PyTorch ROCm exposes the same ``torch.cuda`` surface for runtime streams
    and events, so this backend intentionally keys on the PyTorch device type
    instead of NVIDIA-specific platform names.
    """

    host_memory_kind = "registered_pinned_shared"
    supports_async_submit = True
    supports_timing_events = True

    def __init__(self, *, device: torch.device, ring_depth: int) -> None:
        self.device = device
        torch_version = getattr(torch, "version", None)
        self.device_family = "rocm" if getattr(torch_version, "hip", None) else "cuda"
        self.h2d_submitter = "torch_copy_stream" if self.device_family == "rocm" else "pybind11"
        self._ring_depth = int(ring_depth)
        self._cudart: Any = torch.cuda.cudart()
        if self._cudart is None:
            raise RuntimeError("torch.cuda.cudart() is required for replay host registration")
        self._copy_stream = torch.cuda.Stream(device=device)
        self._ready_events = [torch.cuda.Event() for _ in range(self._ring_depth)]
        self._registered_shared_slots: list[torch.Tensor] = []
        self._registered_shared_ptrs: list[int] = []
        self.host_pinned = False
        self.direct_pinned_shared = False

        if self.h2d_submitter == "pybind11":
            from unilab.ipc.replay_pipelines.native_h2d import ensure_available

            ensure_available()

    def register_host_slots(self, slots: list[torch.Tensor]) -> None:
        for slot in slots:
            nbytes = int(slot.numel() * slot.element_size())
            result = self._cudart.cudaHostRegister(int(slot.data_ptr()), nbytes, 0)
            if result != self._cudart.cudaError.success:
                raise RuntimeError(f"cudaHostRegister failed for collector replay slot: {result}")
            if not slot.is_pinned():
                self._cudart.cudaHostUnregister(int(slot.data_ptr()))
                raise RuntimeError("cudaHostRegister did not make collector replay slot pinned")
            self._registered_shared_slots.append(slot)
            self._registered_shared_ptrs.append(int(slot.data_ptr()))
        self.host_pinned = True
        self.direct_pinned_shared = True

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
        h2d_begin_ns = time.perf_counter_ns()
        start_event = None
        end_event = None
        record_cuda = trace_recorder is not None and trace_cuda_events

        with torch.cuda.device(self.device):
            copy_stream = cast(torch.cuda.Stream, self._copy_stream)
            with torch.cuda.stream(copy_stream):
                if record_cuda:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    cast(Any, start_event).record()

                if self.h2d_submitter == "pybind11":
                    from unilab.ipc.replay_pipelines.native_h2d import submit_h2d

                    submit_h2d(dst, src, copy_stream)
                else:
                    dst.copy_(src, non_blocking=True)
                if end_event is not None:
                    cast(Any, end_event).record()
                self._ready_events[slot].record(copy_stream)

        h2d_end_ns = time.perf_counter_ns()
        if record_cuda and start_event is not None and end_event is not None:
            args: dict[str, object] = {
                "slot": slot,
                "h2d_bytes": h2d_bytes,
                "pinned_memory": self.host_pinned,
                "pack_layout": pack_layout,
                "pack_executor": pack_executor,
                "h2d_submitter": self.h2d_submitter,
                "direct_pinned_shared": self.direct_pinned_shared,
            }
            if metadata is not None:
                args.update(
                    {
                        "tick_id": int(metadata.tick_id),
                        "snapshot_ptr": int(metadata.snapshot_ptr),
                        "snapshot_size": int(metadata.snapshot_size),
                        "sample_seed": int(metadata.sample_seed),
                        "sample_count": int(metadata.sample_count),
                        "batch_host_slot": metadata.batch_host_slot,
                        "batch_gpu_slot": metadata.batch_gpu_slot,
                    }
                )
            trace_recorder.add_cuda_pending_span(
                "gpu/replay_pipeline_batch_h2d",
                category="gpu",
                cpu_begin_ns=h2d_begin_ns,
                start_event=cast(Any, start_event),
                end_event=cast(Any, end_event),
                args=args,
            )
        return (h2d_end_ns - h2d_begin_ns) / 1e9

    def clear_ready(self, slot: int) -> None:
        del slot
        return None

    def ready_query(self, slot: int) -> bool:
        return bool(self._ready_events[slot].query())

    def synchronize_ready(self, slot: int) -> None:
        self._ready_events[slot].synchronize()

    def wait_current_stream_for_ready(self, slot: int) -> None:
        current_stream = cast(Any, torch.cuda.current_stream(self.device))
        current_stream.wait_event(self._ready_events[slot])

    def close(self) -> None:
        while self._registered_shared_ptrs:
            ptr = self._registered_shared_ptrs.pop()
            try:
                self._cudart.cudaHostUnregister(int(ptr))
            except Exception:
                pass
        self._registered_shared_slots.clear()
        self.host_pinned = False
        self.direct_pinned_shared = False
