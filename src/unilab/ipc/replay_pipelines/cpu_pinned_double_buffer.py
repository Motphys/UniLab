"""Double-buffer replay pipeline for packed CPU replay samples.

CUDA uses the original CPU-pinned shared slot + native async H2D fast path.
Non-CUDA devices use the same collector-thread packing contract and a
portable torch copy into the device batch slots.
"""

from __future__ import annotations

import csv
import os
import queue
import threading
import time
from typing import Any, Dict, List, Tuple, cast

import torch

from unilab.ipc.replay_buffer import ReplayBuffer
from unilab.ipc.replay_pipelines.base import ReplayTickMetadata


class CPUPinnedDoubleBufferReplayPipeline:
    """Double-buffered packed replay batch pipeline.

    CUDA keeps the pinned-host → GPU fast path.  MPS/CPU keep the same
    collector-thread pack and hot/cold batch contract with a portable torch
    copy into the learner device slot.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        *,
        device: str,
        sample_count: int,
        base_seed: int = 0,
        trace_recorder=None,
        trace_cuda_events: bool = True,
        verbose: bool = False,
        verbose_output_dir: str | None = None,
        collector_pack_request_queue=None,
        collector_pack_ready_queue=None,
        collector_pack_shared_slots=None,
    ) -> None:
        self._replay_buffer = replay_buffer
        self._device = torch.device(device)
        self._device_type = self._device.type
        self._sample_count = sample_count
        self._base_seed = base_seed
        self._trace_recorder = trace_recorder
        self._trace_cuda_events = bool(trace_cuda_events) and self._device_type == "cuda"
        self._verbose = bool(verbose)
        self._verbose_output_dir = verbose_output_dir if self._verbose else None
        self._pack_layout = "packed"
        self._pack_executor = "collector_thread"
        self._h2d_submitter = "pybind11" if self._device_type == "cuda" else "torch_copy"
        self._host_pinned = False
        self._direct_pinned_shared = False
        if self._device_type == "cuda":
            from unilab.ipc.replay_pipelines.native_h2d import ensure_available

            ensure_available()
        if not getattr(replay_buffer, "_packed_cpu_storage", False):
            raise ValueError("pack_layout='packed' requires ReplayBuffer(packed_cpu_storage=True)")
        if (
            collector_pack_request_queue is None
            or collector_pack_ready_queue is None
            or collector_pack_shared_slots is None
        ):
            raise ValueError(
                "collector_thread pack executor requires collector pack IPC queues and slots"
            )
        self._verbose_pack_records: List[Tuple[int, int, str, int, int, int, int]] | None = (
            [] if self._verbose else None
        )
        self._collector_pack_request_queue = collector_pack_request_queue
        self._collector_pack_ready_queue = collector_pack_ready_queue
        self._collector_pack_shared_slots = collector_pack_shared_slots
        self._registered_shared_slots: list[torch.Tensor] = []
        self._registered_shared_ptrs: list[int] = []
        self._cudart = torch.cuda.cudart() if self._device_type == "cuda" else None
        self._fields: Dict[str, tuple[torch.Tensor, int]] = {}
        self._packed_width = int(replay_buffer._storage.shape[1])
        self._host_packed: list[torch.Tensor] = []
        self._register_collector_shared_slots()
        self._gpu_packed = [
            torch.empty(
                (self._sample_count, self._packed_width), dtype=torch.float32, device=self._device
            )
            for _ in range(2)
        ]
        self._host: list[Dict[str, torch.Tensor]] = []
        self._gpu: list[Dict[str, torch.Tensor]] = []

        if self._device_type == "cuda":
            self._copy_stream = torch.cuda.Stream(device=self._device)
            self._ready_events = [torch.cuda.Event() for _ in range(2)]
        else:
            self._copy_stream = None
            self._ready_events = [threading.Event() for _ in range(2)]

        self._hot = 0
        self._cold = 1
        self._has_hot_batch = False
        self._hot_metadata: ReplayTickMetadata | None = None
        self._prepared_metadata: ReplayTickMetadata | None = None
        self._prepare_tick_id: int | None = None
        self._prepare_state = "idle"
        self._prepare_error: BaseException | None = None
        self.last_incremental_h2d_time_s = 0.0
        self._prepare_condition = threading.Condition()
        self._closed = False
        self._collector_h2d_thread = threading.Thread(
            target=self._collector_h2d_worker,
            name="replay_collector_h2d",
            daemon=True,
        )
        self._collector_h2d_thread.start()

    @property
    def h2d_submitter(self) -> str:
        return self._h2d_submitter

    # -- allocation helpers --------------------------------------------------

    def _register_collector_shared_slots(self) -> None:
        assert self._collector_pack_shared_slots is not None
        if self._device_type != "cuda":
            return
        assert self._cudart is not None
        for slot in self._collector_pack_shared_slots:
            nbytes = int(slot.numel() * slot.element_size())
            result = self._cudart.cudaHostRegister(int(slot.data_ptr()), nbytes, 0)
            if result != self._cudart.cudaError.success:
                raise RuntimeError(f"cudaHostRegister failed for collector replay slot: {result}")
            if not slot.is_pinned():
                self._cudart.cudaHostUnregister(int(slot.data_ptr()))
                raise RuntimeError("cudaHostRegister did not make collector replay slot pinned")
            self._registered_shared_slots.append(slot)
            self._registered_shared_ptrs.append(int(slot.data_ptr()))
        self._host_pinned = True
        self._direct_pinned_shared = True

    def _unregister_collector_shared_slots(self) -> None:
        if self._cudart is None:
            return
        while self._registered_shared_ptrs:
            ptr = self._registered_shared_ptrs.pop()
            try:
                self._cudart.cudaHostUnregister(int(ptr))
            except Exception:
                pass
        self._registered_shared_slots.clear()

    def _packed_h2d_source(self, slot: int) -> torch.Tensor:
        assert self._collector_pack_shared_slots is not None
        return self._collector_pack_shared_slots[slot]

    def _packed_batch_view(self, packed: torch.Tensor) -> Dict[str, torch.Tensor]:
        rb = self._replay_buffer
        batch = {
            "obs": packed[:, rb._obs_sl],
            "next_obs": packed[:, rb._nobs_sl],
            "actions": packed[:, rb._act_sl],
            "rewards": packed[:, rb._rew_col],
            "dones": packed[:, rb._done_col],
            "truncated": packed[:, rb._trunc_col],
        }
        if rb._critic_dim > 0:
            batch["critic"] = packed[:, rb._critic_sl]
            batch["next_critic"] = packed[:, rb._ncritic_sl]
        return batch

    # -- snapshot / H2D --------------------------------------------------------

    def _snapshot(self) -> tuple[int, int]:
        return int(self._replay_buffer.ptr[0]), int(self._replay_buffer.size[0])

    def _submit_h2d(self, slot: int, metadata: ReplayTickMetadata | None = None) -> float:
        h2d_begin_ns = time.perf_counter_ns()
        start_event = None
        end_event = None
        record_cuda = self._trace_recorder is not None and self._trace_cuda_events
        self._clear_ready(slot)
        if self._device_type == "cuda":
            with torch.cuda.device(self._device):
                copy_stream = cast(torch.cuda.Stream, self._copy_stream)
                with torch.cuda.stream(copy_stream):
                    if record_cuda:
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        cast(Any, start_event).record()
                    from unilab.ipc.replay_pipelines.native_h2d import submit_h2d

                    submit_h2d(
                        self._gpu_packed[slot],
                        self._packed_h2d_source(slot),
                        copy_stream,
                    )
                    if end_event is not None:
                        cast(Any, end_event).record()
                    cast(torch.cuda.Event, self._ready_events[slot]).record(copy_stream)
        else:
            self._gpu_packed[slot].copy_(self._packed_h2d_source(slot))
            if self._device_type == "mps" and hasattr(torch, "mps"):
                torch.mps.synchronize()
            self._mark_ready(slot)
        h2d_end_ns = time.perf_counter_ns()
        if record_cuda and start_event is not None and end_event is not None:
            args: Dict[str, object] = {
                "slot": slot,
                "h2d_bytes": self._h2d_bytes(),
                "pinned_memory": self._host_pinned,
                "pack_layout": self._pack_layout,
                "pack_executor": self._pack_executor,
                "h2d_submitter": self._h2d_submitter,
                "direct_pinned_shared": self._direct_pinned_shared,
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
            trace_recorder = cast(Any, self._trace_recorder)
            trace_recorder.add_cuda_pending_span(
                "gpu/replay_pipeline_batch_h2d",
                category="gpu",
                cpu_begin_ns=h2d_begin_ns,
                start_event=cast(Any, start_event),
                end_event=cast(Any, end_event),
                args=args,
            )
        return (h2d_end_ns - h2d_begin_ns) / 1e9

    def _submit_collector_packed_h2d(self, ready: dict) -> ReplayTickMetadata:
        metadata = ReplayTickMetadata(
            tick_id=int(ready["tick_id"]),
            snapshot_ptr=int(ready["snapshot_ptr"]),
            snapshot_size=int(ready["snapshot_size"]),
            sample_seed=int(ready["sample_seed"]),
            sample_count=int(ready["sample_count"]),
            batch_host_slot=int(ready["shared_slot"]),
            batch_gpu_slot=int(ready["target_gpu_slot"]),
        )
        slot = metadata.batch_gpu_slot
        assert slot is not None
        shared_slot = int(ready["shared_slot"])
        if shared_slot != slot:
            raise RuntimeError("collector_thread shared slot must match target GPU slot")
        if self._device_type == "cuda":
            self._submit_device_transfer(metadata)
        return metadata

    def _submit_device_transfer(self, metadata: ReplayTickMetadata) -> None:
        slot = metadata.batch_gpu_slot
        shared_slot = metadata.batch_host_slot
        assert slot is not None
        assert shared_slot is not None
        h2d_submit_ns = time.perf_counter_ns()
        self.last_incremental_h2d_time_s = self._submit_h2d(slot, metadata)
        if self._trace_recorder is not None:
            self._trace_recorder.add_slice(
                "replay_pipeline/batch_h2d_submit",
                category="replay_pipeline",
                start_ns=h2d_submit_ns,
                end_ns=time.perf_counter_ns(),
                args={
                    "tick_id": metadata.tick_id,
                    "batch_gpu_slot": slot,
                    "shared_slot": shared_slot,
                    "pack_layout": self._pack_layout,
                    "pack_executor": self._pack_executor,
                    "h2d_submitter": self._h2d_submitter,
                    "h2d_bytes": self._h2d_bytes(),
                    "h2d_submitted": True,
                    "pinned_memory": self._host_pinned,
                    "direct_pinned_shared": self._direct_pinned_shared,
                    "learner_thread_submit": self._device_type != "cuda",
                },
            )

    def _ensure_device_transfer_ready(self, metadata: ReplayTickMetadata) -> None:
        slot = metadata.batch_gpu_slot
        assert slot is not None
        if self._ready_query(slot):
            return
        if self._device_type == "cuda":
            self._synchronize_ready(slot)
            return
        self._submit_device_transfer(metadata)

    def _collector_h2d_worker(self) -> None:
        while True:
            if self._closed:
                return
            try:
                ready = self._collector_pack_ready_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if ready is None:
                return
            try:
                metadata = self._submit_collector_packed_h2d(ready)
                with self._prepare_condition:
                    if self._prepare_tick_id != metadata.tick_id:
                        raise RuntimeError(
                            f"Collector packed tick {metadata.tick_id} does not match "
                            f"pending tick {self._prepare_tick_id}"
                        )
                    self._prepared_metadata = metadata
                    self._prepare_state = (
                        "h2d_submitted" if self._device_type == "cuda" else "host_ready"
                    )
                    self._prepare_error = None
                    self._prepare_condition.notify_all()
            except BaseException as exc:
                with self._prepare_condition:
                    self._prepare_error = exc
                    self._prepare_condition.notify_all()

    def _h2d_bytes(self) -> int:
        source = self._packed_h2d_source(0)
        return int(source.numel() * source.element_size())

    def _clear_ready(self, slot: int) -> None:
        if self._device_type != "cuda":
            cast(threading.Event, self._ready_events[slot]).clear()

    def _mark_ready(self, slot: int) -> None:
        if self._device_type != "cuda":
            cast(threading.Event, self._ready_events[slot]).set()

    def _ready_query(self, slot: int) -> bool:
        if self._device_type == "cuda":
            return bool(cast(torch.cuda.Event, self._ready_events[slot]).query())
        return cast(threading.Event, self._ready_events[slot]).is_set()

    def _synchronize_ready(self, slot: int) -> None:
        if self._device_type == "cuda":
            cast(torch.cuda.Event, self._ready_events[slot]).synchronize()
            return
        cast(threading.Event, self._ready_events[slot]).wait()

    def _wait_current_stream_for_ready(self, slot: int) -> None:
        if self._device_type == "cuda":
            current_stream = cast(Any, torch.cuda.current_stream(self._device))
            current_stream.wait_event(cast(torch.cuda.Event, self._ready_events[slot]))
            return
        self._synchronize_ready(slot)

    # -- public API -----------------------------------------------------------

    def _validate_sample_count(self, sample_count: int) -> None:
        if int(sample_count) != int(self._sample_count):
            raise ValueError("sample_count must match the value used to allocate the double buffer")

    def _refresh_prepare_state(self) -> None:
        if self._prepare_error is not None:
            raise self._prepare_error
        if self._prepared_metadata is not None:
            slot = self._prepared_metadata.batch_gpu_slot
            if slot is not None and self._ready_query(slot):
                self._prepare_state = "ready"

    def start_prepare(
        self,
        tick_id: int,
        sample_count: int,
        min_snapshot_ptr: int | None = None,
    ) -> bool:
        """Start CPU pack + device transfer for the current cold slot.

        Returns True when this call launches new work. If the same tick is
        already pending or prepared, returns False.
        """
        self._validate_sample_count(sample_count)
        if self._closed:
            raise RuntimeError("Cannot prepare replay batch after pipeline.close()")
        self._refresh_prepare_state()
        active_tick = self._prepare_tick_id
        if self._prepared_metadata is not None or self._prepare_state not in {"idle", "ready"}:
            prepared_tick = (
                self._prepared_metadata.tick_id
                if self._prepared_metadata is not None
                else active_tick
            )
            if prepared_tick == int(tick_id):
                return False
            raise RuntimeError(
                "Cannot prepare a new replay batch before the previous batch is consumed"
            )
        slot = self._cold
        self._clear_ready(slot)
        self._prepare_tick_id = int(tick_id)
        self._prepare_error = None
        snapshot_ptr, snapshot_size = self._snapshot()
        sample_seed = self._base_seed + int(tick_id)
        min_snapshot_ptr = snapshot_ptr if min_snapshot_ptr is None else int(min_snapshot_ptr)
        request = {
            "tick_id": int(tick_id),
            "snapshot_ptr": snapshot_ptr,
            "snapshot_size": snapshot_size,
            "min_snapshot_ptr": min_snapshot_ptr,
            "sample_seed": sample_seed,
            "sample_count": self._sample_count,
            "shared_slot": slot,
            "learner_hot_gpu_slot": self._hot,
            "target_gpu_slot": slot,
            "pack_layout": self._pack_layout,
            "pack_executor": self._pack_executor,
        }
        if self._trace_recorder is not None:
            _req_ns = time.perf_counter_ns()
            self._trace_recorder.add_slice(
                "replay_pipeline/collector_pack_request",
                category="replay_pipeline",
                start_ns=_req_ns,
                end_ns=time.perf_counter_ns(),
                args=request,
            )
        self._prepare_state = "collector_pack_requested"
        self._collector_pack_request_queue.put(request)
        return True

    def batch_ready(self, tick_id: int, sample_count: int) -> bool:
        self._validate_sample_count(sample_count)
        if self._has_hot_batch:
            if self._hot_metadata is not None and self._hot_metadata.tick_id != int(tick_id):
                return False
            return True
        self._refresh_prepare_state()
        if self._prepared_metadata is None:
            return False
        if self._prepared_metadata.tick_id != int(tick_id):
            return False
        return self._prepare_state == "ready"

    def wait_ready(self) -> None:
        return None

    def wait_until_ready(self, tick_id: int, sample_count: int) -> bool:
        self._validate_sample_count(sample_count)
        metadata = self._prepared_or_wait(tick_id)
        slot = metadata.batch_gpu_slot
        assert slot is not None
        self._ensure_device_transfer_ready(metadata)
        self._synchronize_ready(slot)
        self._prepare_state = "ready"
        return True

    def _prepared_or_wait(self, tick_id: int) -> ReplayTickMetadata:
        self._refresh_prepare_state()
        if self._prepared_metadata is None:
            if self._prepare_tick_id is None:
                self.start_prepare(tick_id, self._sample_count)
            with self._prepare_condition:
                while self._prepared_metadata is None and self._prepare_error is None:
                    self._prepare_condition.wait(timeout=0.1)
                if self._prepare_error is not None:
                    raise self._prepare_error
            assert self._prepared_metadata is not None
            return self._prepared_metadata
        if self._prepared_metadata.tick_id != int(tick_id):
            raise RuntimeError(
                f"Prepared replay batch tick {self._prepared_metadata.tick_id} "
                f"does not match requested tick {tick_id}"
            )
        return self._prepared_metadata

    def sample_large_batch(self, tick_id: int, sample_count: int) -> Dict[str, torch.Tensor]:
        self._validate_sample_count(sample_count)
        if self._has_hot_batch:
            if self._hot_metadata is not None and self._hot_metadata.tick_id != int(tick_id):
                raise RuntimeError(
                    f"Hot batch tick {self._hot_metadata.tick_id} does not match "
                    f"requested tick {tick_id}"
                )
            return self._packed_batch_view(self._gpu_packed[self._hot])
        if not self._has_hot_batch:
            if not self.batch_ready(tick_id, sample_count):
                self.wait_until_ready(tick_id, sample_count)
            metadata = self._prepared_or_wait(tick_id)
            slot = metadata.batch_gpu_slot
            assert slot is not None
            _t0 = time.perf_counter_ns()
            self._wait_current_stream_for_ready(slot)
            if self._trace_recorder is not None:
                _wait_end = time.perf_counter_ns()
                self._trace_recorder.add_slice(
                    "replay_pipeline/batch_h2d_wait",
                    category="replay_pipeline",
                    start_ns=_t0,
                    end_ns=_wait_end,
                    args={"tick_id": tick_id, "batch_gpu_slot": slot},
                )
                self._trace_recorder.add_slice(
                    "replay_pipeline/gpu_wait_for_batch",
                    category="replay_pipeline",
                    start_ns=_t0,
                    end_ns=_wait_end,
                    args={"tick_id": tick_id, "batch_gpu_slot": slot},
                )
            _swap_ns = time.perf_counter_ns()
            old_hot = self._hot
            old_cold = self._cold
            if slot != self._cold:
                raise RuntimeError("Prepared replay batch is not in the current cold slot")
            self._hot, self._cold = self._cold, self._hot
            if self._trace_recorder is not None:
                self._trace_recorder.add_slice(
                    "replay_pipeline/hot_cold_swap",
                    category="replay_pipeline",
                    start_ns=_swap_ns,
                    end_ns=time.perf_counter_ns(),
                    args={
                        "tick_id": tick_id,
                        "old_hot": old_hot,
                        "old_cold": old_cold,
                        "new_hot": self._hot,
                        "new_cold": self._cold,
                    },
                )
            self._has_hot_batch = True
            self._hot_metadata = metadata
            self._prepared_metadata = None
            self._prepare_tick_id = None
            self._prepare_state = "idle"
        return self._packed_batch_view(self._gpu_packed[self._hot])

    def after_tick(self) -> None:
        self._has_hot_batch = False
        self._hot_metadata = None

    def close(self) -> None:
        self._closed = True
        if self._collector_pack_ready_queue is not None:
            try:
                self._collector_pack_ready_queue.put_nowait(None)
            except Exception:
                pass
        if self._collector_h2d_thread is not None:
            self._collector_h2d_thread.join(timeout=2.0)
        if self._prepared_metadata is not None:
            slot = self._prepared_metadata.batch_gpu_slot
            if slot is not None:
                self._synchronize_ready(slot)
        self._unregister_collector_shared_slots()
        if self._verbose and self._verbose_output_dir and self._verbose_pack_records:
            try:
                verbose_dir = os.path.join(self._verbose_output_dir, "verbose")
                os.makedirs(verbose_dir, exist_ok=True)
                csv_path = os.path.join(verbose_dir, "pack_fields.csv")
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["tick_id", "slot", "field", "rows", "cols", "bytes", "dur_ns"])
                    for row in self._verbose_pack_records:
                        writer.writerow(row)
            except OSError:
                pass
        self._host.clear()
        self._gpu.clear()
        if hasattr(self, "_host_packed"):
            self._host_packed.clear()
        if hasattr(self, "_gpu_packed"):
            self._gpu_packed.clear()
