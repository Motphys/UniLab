"""GPU-resident replay mirror pipeline (single-GPU, CUDA only).

The packed CPU :class:`ReplayBuffer` stays authoritative: the collector's
``add()`` path is unchanged and does zero extra CPU work.  A learner-side
daemon thread incrementally mirrors newly written rows into a GPU-resident
storage on a side stream, and services learner batch prepares with GPU-side
sampling (``randint`` + ``index_select``) on the same stream.  There is no
collector pack request, no pinned batch staging, and no per-tick batch H2D.

Consistency model:

- The sample domain is ``[0, min(device_visible_ptr, capacity))`` where
  ``device_visible_ptr`` only advances past rows whose H2D completed (one
  CUDA event per submitted span batch).
- Span copies and batch gathers are totally ordered on a single side stream,
  so a gather observes a consistent snapshot: every span submitted before it
  is applied, none submitted after it.
- Ring-overwrite races against concurrent collector CPU writes are the same
  class the CPU pack path already accepts (single writer / snapshot reader).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Dict, List, Tuple, cast

import torch

from unilab.ipc.replay_buffer import ReplayBuffer
from unilab.ipc.replay_pipelines.base import ReplayTickMetadata
from unilab.ipc.replay_pipelines.transfer import build_replay_transfer_backend


def _ring_spans(start: int, end: int, capacity: int) -> List[Tuple[int, int]]:
    """Split absolute row range ``[start, end)`` into ``(offset, length)`` ring spans."""
    if end <= start or capacity <= 0:
        return []
    spans: List[Tuple[int, int]] = []
    remaining = end - start
    offset = start % capacity
    while remaining > 0:
        length = min(remaining, capacity - offset)
        spans.append((offset, length))
        remaining -= length
        offset = 0
    return spans


class GPUResidentReplayPipeline:
    """ReplayPipeline backed by a GPU-resident mirror of the packed CPU storage."""

    _POLL_INTERVAL_S = 0.0005
    _MEMORY_HEADROOM = 0.8

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        *,
        device: str,
        sample_count: int,
        base_seed: int = 0,
        trace_recorder=None,
        trace_cuda_events: bool = True,
        pack_layout: str = "packed",
        use_critic_graph_packed_source: bool = False,
    ) -> None:
        self._replay_buffer = replay_buffer
        self._device = torch.device(device)
        if pack_layout not in {"packed", "sac_graph"}:
            raise ValueError("GPUResidentReplayPipeline pack_layout must be packed or sac_graph")
        if self._device.type != "cuda":
            raise ValueError(
                "GPUResidentReplayPipeline requires a CUDA device; "
                f"got {self._device.type!r} (use cpu_pinned_double_buffer instead)"
            )
        if not getattr(replay_buffer, "_packed_cpu_storage", False):
            raise ValueError("GPUResidentReplayPipeline requires packed CPU replay storage")
        self._pack_layout = pack_layout
        self._use_critic_graph_packed_source = (
            bool(use_critic_graph_packed_source) and self._pack_layout != "sac_graph"
        )
        self._sample_count = int(sample_count)
        self._base_seed = int(base_seed)
        self._trace_recorder = trace_recorder
        self._capacity = int(replay_buffer.capacity)
        self._storage_width = int(replay_buffer._storage.shape[1])
        self._packed_width = (
            int(replay_buffer.sac_graph_packed_width())
            if self._pack_layout == "sac_graph"
            else self._storage_width
        )
        self._critic_graph_packed_width = (
            int(replay_buffer.critic_graph_packed_width())
            if self._use_critic_graph_packed_source
            else 0
        )

        # -- device memory guard (hard fail: this pipeline is opt-in) --
        storage_bytes = self._capacity * self._storage_width * 4
        slot_bytes = 2 * self._sample_count * self._packed_width * 4
        scratch_bytes = (
            self._sample_count * self._storage_width * 4 if self._pack_layout == "sac_graph" else 0
        )
        critic_slot_bytes = 2 * self._sample_count * self._critic_graph_packed_width * 4
        required_bytes = storage_bytes + slot_bytes + scratch_bytes + critic_slot_bytes
        free_bytes, total_bytes = torch.cuda.mem_get_info(self._device)
        if required_bytes > int(free_bytes * self._MEMORY_HEADROOM):
            raise RuntimeError(
                "GPU-resident replay mirror does not fit on device: "
                f"requires {required_bytes / 2**30:.2f} GiB "
                f"(storage {storage_bytes / 2**30:.2f} + batch slots "
                f"{(slot_bytes + scratch_bytes + critic_slot_bytes) / 2**30:.2f}), "
                f"device free {free_bytes / 2**30:.2f} GiB / total {total_bytes / 2**30:.2f} GiB"
            )

        self._transfer_backend = build_replay_transfer_backend(
            device=self._device,
            ring_depth=2,
        )
        self._trace_cuda_events = bool(trace_cuda_events) and (
            self._transfer_backend.supports_timing_events
        )
        self._device_family = self._transfer_backend.device_family
        self._host_pinned = False
        try:
            self._transfer_backend.register_host_slots([replay_buffer._storage])
            self._host_pinned = True
        except RuntimeError as exc:
            print(
                f"[GPUResidentReplay] Host storage registration failed ({exc}); "
                "falling back to pageable H2D on the sync thread.",
                flush=True,
            )
        self._gpu_storage: torch.Tensor = torch.empty(
            (self._capacity, self._storage_width),
            dtype=torch.float32,
            device=self._device,
        )
        self._gpu_packed = self._transfer_backend.allocate_device_slots(
            count=2,
            shape=(self._sample_count, self._packed_width),
            dtype=torch.float32,
        )
        self._gpu_critic_graph_packed: list[torch.Tensor] = (
            self._transfer_backend.allocate_device_slots(
                count=2,
                shape=(self._sample_count, self._critic_graph_packed_width),
                dtype=torch.float32,
            )
            if self._use_critic_graph_packed_source
            else []
        )
        self._gather_scratch: torch.Tensor | None = (
            torch.empty(
                (self._sample_count, self._storage_width),
                dtype=torch.float32,
                device=self._device,
            )
            if self._pack_layout == "sac_graph"
            else None
        )

        self._sync_stream = cast(torch.cuda.Stream, torch.cuda.Stream(device=self._device))
        self._slot_events: list[Any] = [torch.cuda.Event() for _ in range(2)]
        self._span_events: deque[tuple[int, Any]] = deque()
        self._submitted_ptr = 0
        self._visible_ptr = 0
        self._wrap_skip_warned = False

        self._hot = 0
        self._cold = 1
        self._has_hot_batch = False
        self._hot_metadata: ReplayTickMetadata | None = None
        self._prepared_metadata: ReplayTickMetadata | None = None
        self._prepare_tick_id: int | None = None
        self._prepare_required_ptr = 0
        self._prepare_state = "idle"
        self._prepare_error: BaseException | None = None
        self.last_incremental_h2d_time_s = 0.0
        self._prepare_condition = threading.Condition()
        self._closed = False
        self._sync_thread = threading.Thread(
            target=self._sync_worker,
            name="replay_gpu_resident_sync",
            daemon=True,
        )
        self._sync_thread.start()

    @property
    def h2d_submitter(self) -> str:
        return "gpu_resident_mirror"

    @property
    def transfer_manifest(self) -> dict[str, object]:
        return {
            "backend": type(self._transfer_backend).__name__,
            "device": str(self._device),
            "device_family": self._device_family,
            "pipeline": "gpu_resident",
            "host_memory_kind": (
                self._transfer_backend.host_memory_kind if self._host_pinned else "pageable_shared"
            ),
            "host_pinned": self._host_pinned,
            "storage_rows": self._capacity,
            "storage_width": self._storage_width,
            "storage_bytes": int(self._gpu_storage.numel() * self._gpu_storage.element_size()),
            "h2d_submitter": self.h2d_submitter,
            "ring_depth": 2,
        }

    # -- batch views ----------------------------------------------------------

    def _packed_batch_view(self, packed: torch.Tensor) -> Dict[str, torch.Tensor]:
        rb = self._replay_buffer
        if self._pack_layout == "sac_graph":
            c = 0
            obs_sl = slice(c, c + rb._obs_dim)
            c += rb._obs_dim
            critic_sl = slice(c, c + rb._critic_dim)
            c += rb._critic_dim
            act_sl = slice(c, c + rb._action_dim)
            c += rb._action_dim
            rew_col = c
            c += 1
            nobs_sl = slice(c, c + rb._obs_dim)
            c += rb._obs_dim
            ncritic_sl = slice(c, c + rb._critic_dim)
            c += rb._critic_dim
            done_col = c
            c += 1
            trunc_col = c
            return {
                "obs": packed[:, obs_sl],
                "next_obs": packed[:, nobs_sl],
                "actions": packed[:, act_sl],
                "rewards": packed[:, rew_col],
                "dones": packed[:, done_col],
                "truncated": packed[:, trunc_col],
                "critic": packed[:, critic_sl],
                "next_critic": packed[:, ncritic_sl],
                "sac_graph_packed_source": packed,
            }
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

    def _large_batch_view(self, slot: int) -> Dict[str, torch.Tensor]:
        batch = self._packed_batch_view(self._gpu_packed[slot])
        if self._use_critic_graph_packed_source:
            batch["critic_graph_packed_source"] = self._gpu_critic_graph_packed[slot]
        return batch

    # -- sync thread ------------------------------------------------------------

    def _sync_worker(self) -> None:
        while True:
            if self._closed:
                return
            try:
                did_work = self._submit_new_spans()
                did_work |= self._drain_completed_spans()
                did_work |= self._service_pending_prepare()
            except BaseException as exc:
                with self._prepare_condition:
                    self._prepare_error = exc
                    self._prepare_condition.notify_all()
                return
            if not did_work:
                time.sleep(self._POLL_INTERVAL_S)

    def _submit_new_spans(self) -> bool:
        ptr = int(self._replay_buffer.ptr[0])
        if ptr <= self._submitted_ptr:
            return False
        if ptr - self._submitted_ptr > self._capacity:
            skipped = ptr - self._capacity
            if not self._wrap_skip_warned:
                print(
                    "[GPUResidentReplay] mirror fell behind by more than one "
                    f"capacity; skipping to absolute row {skipped}",
                    flush=True,
                )
                self._wrap_skip_warned = True
            self._submitted_ptr = skipped
        start = self._submitted_ptr
        end = ptr
        h2d_begin_ns = time.perf_counter_ns()
        start_event = None
        end_event = None
        record_cuda = self._trace_recorder is not None and self._trace_cuda_events
        with torch.cuda.device(self._device):
            with torch.cuda.stream(self._sync_stream):
                if record_cuda:
                    start_event = cast(Any, torch.cuda.Event(enable_timing=True))
                    end_event = cast(Any, torch.cuda.Event(enable_timing=True))
                    start_event.record()
                for offset, length in _ring_spans(start, end, self._capacity):
                    self._gpu_storage[offset : offset + length].copy_(
                        self._replay_buffer._storage[offset : offset + length],
                        non_blocking=True,
                    )
                if end_event is not None:
                    end_event.record()
                done_event = cast(Any, torch.cuda.Event())
                done_event.record(self._sync_stream)
        self._span_events.append((end, done_event))
        self._submitted_ptr = end
        self.last_incremental_h2d_time_s = (time.perf_counter_ns() - h2d_begin_ns) / 1e9
        if self._trace_recorder is not None and start_event is not None and end_event is not None:
            self._trace_recorder.add_cuda_pending_span(
                "gpu/replay_pipeline_storage_h2d",
                category="gpu",
                cpu_begin_ns=h2d_begin_ns,
                start_event=start_event,
                end_event=end_event,
                args={
                    "h2d_bytes": (end - start) * self._storage_width * 4,
                    "rows": end - start,
                    "span_start": start,
                    "span_end": end,
                    "pinned_memory": self._host_pinned,
                    "pipeline": "gpu_resident",
                },
            )
        return True

    def _drain_completed_spans(self) -> bool:
        drained = False
        while self._span_events and self._span_events[0][1].query():
            self._visible_ptr = self._span_events[0][0]
            self._span_events.popleft()
            drained = True
        if drained:
            with self._prepare_condition:
                self._prepare_condition.notify_all()
        return drained

    def _service_pending_prepare(self) -> bool:
        with self._prepare_condition:
            if self._prepare_state != "preparing" or self._prepare_tick_id is None:
                return False
            tick_id = self._prepare_tick_id
            required_ptr = self._prepare_required_ptr
            slot = self._cold
        visible_size = min(self._visible_ptr, self._capacity)
        if self._visible_ptr < required_ptr or visible_size <= 0:
            return False
        sample_seed = self._base_seed + int(tick_id)
        gen = torch.Generator(device=self._device)
        gen.manual_seed(sample_seed)
        gather_begin_ns = time.perf_counter_ns()
        start_event = None
        end_event = None
        record_cuda = self._trace_recorder is not None and self._trace_cuda_events
        with torch.cuda.device(self._device):
            with torch.cuda.stream(self._sync_stream):
                if record_cuda:
                    start_event = cast(Any, torch.cuda.Event(enable_timing=True))
                    end_event = cast(Any, torch.cuda.Event(enable_timing=True))
                    start_event.record()
                indices = torch.randint(
                    0,
                    visible_size,
                    (self._sample_count,),
                    generator=gen,
                    device=self._device,
                )
                dst = self._gpu_packed[slot]
                if self._pack_layout == "sac_graph":
                    assert self._gather_scratch is not None
                    torch.index_select(self._gpu_storage, 0, indices, out=self._gather_scratch)
                    self._replay_buffer.pack_sac_graph_source(self._gather_scratch, out=dst)
                else:
                    torch.index_select(self._gpu_storage, 0, indices, out=dst)
                if self._use_critic_graph_packed_source:
                    self._replay_buffer.pack_critic_graph_source(
                        dst,
                        out=self._gpu_critic_graph_packed[slot],
                    )
                if end_event is not None:
                    end_event.record()
                self._slot_events[slot].record(self._sync_stream)
        metadata = ReplayTickMetadata(
            tick_id=int(tick_id),
            snapshot_ptr=int(self._visible_ptr),
            snapshot_size=visible_size,
            sample_seed=sample_seed,
            sample_count=self._sample_count,
            batch_host_slot=None,
            batch_gpu_slot=slot,
        )
        with self._prepare_condition:
            self._prepared_metadata = metadata
            self._prepare_state = "gather_submitted"
            self._prepare_condition.notify_all()
        if self._trace_recorder is not None:
            self._trace_recorder.add_slice(
                "replay_pipeline/gpu_batch_gather_submit",
                category="replay_pipeline",
                start_ns=gather_begin_ns,
                end_ns=time.perf_counter_ns(),
                args={
                    "tick_id": int(tick_id),
                    "batch_gpu_slot": slot,
                    "sample_count": self._sample_count,
                    "snapshot_size": visible_size,
                    "required_ptr": required_ptr,
                    "visible_ptr": int(self._visible_ptr),
                    "pack_layout": self._pack_layout,
                    "pipeline": "gpu_resident",
                },
            )
        if self._trace_recorder is not None and start_event is not None and end_event is not None:
            self._trace_recorder.add_cuda_pending_span(
                "gpu/replay_pipeline_batch_gather",
                category="gpu",
                cpu_begin_ns=gather_begin_ns,
                start_event=start_event,
                end_event=end_event,
                args={
                    "tick_id": int(tick_id),
                    "batch_gpu_slot": slot,
                    "sample_count": self._sample_count,
                    "gather_bytes": self._sample_count * self._packed_width * 4,
                    "pack_layout": self._pack_layout,
                    "pipeline": "gpu_resident",
                },
            )
        return True

    # -- public API -----------------------------------------------------------

    def _validate_sample_count(self, sample_count: int) -> None:
        if int(sample_count) != int(self._sample_count):
            raise ValueError("sample_count must match the value used to allocate the double buffer")

    def _refresh_prepare_state(self) -> None:
        if self._prepare_error is not None:
            raise self._prepare_error
        if self._prepared_metadata is not None:
            slot = self._prepared_metadata.batch_gpu_slot
            if slot is not None and self._slot_events[slot].query():
                self._prepare_state = "ready"

    def start_prepare(
        self,
        tick_id: int,
        sample_count: int,
        min_snapshot_ptr: int | None = None,
    ) -> bool:
        """Schedule a GPU-side gather for the current cold slot.

        Returns True when this call launches new work. If the same tick is
        already pending or prepared, returns False.
        """
        self._validate_sample_count(sample_count)
        if self._closed:
            raise RuntimeError("Cannot prepare replay batch after pipeline.close()")
        self._refresh_prepare_state()
        with self._prepare_condition:
            if self._prepared_metadata is not None or self._prepare_state not in {
                "idle",
                "ready",
            }:
                prepared_tick = (
                    self._prepared_metadata.tick_id
                    if self._prepared_metadata is not None
                    else self._prepare_tick_id
                )
                if prepared_tick == int(tick_id):
                    return False
                raise RuntimeError(
                    "Cannot prepare a new replay batch before the previous batch is consumed"
                )
            self._prepare_tick_id = int(tick_id)
            self._prepare_required_ptr = (
                int(self._replay_buffer.ptr[0])
                if min_snapshot_ptr is None
                else int(min_snapshot_ptr)
            )
            self._prepared_metadata = None
            self._prepare_error = None
            self._prepare_state = "preparing"
            self._prepare_condition.notify_all()
        if self._trace_recorder is not None:
            _req_ns = time.perf_counter_ns()
            self._trace_recorder.add_slice(
                "replay_pipeline/gpu_batch_prepare_request",
                category="replay_pipeline",
                start_ns=_req_ns,
                end_ns=time.perf_counter_ns(),
                args={
                    "tick_id": int(tick_id),
                    "required_ptr": self._prepare_required_ptr,
                    "pipeline": "gpu_resident",
                },
            )
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
        self._slot_events[slot].synchronize()
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
            return self._large_batch_view(self._hot)
        if not self.batch_ready(tick_id, sample_count):
            self.wait_until_ready(tick_id, sample_count)
        metadata = self._prepared_or_wait(tick_id)
        slot = metadata.batch_gpu_slot
        assert slot is not None
        _t0 = time.perf_counter_ns()
        torch.cuda.current_stream(self._device).wait_event(self._slot_events[slot])
        if self._trace_recorder is not None:
            _wait_end = time.perf_counter_ns()
            self._trace_recorder.add_slice(
                "replay_pipeline/batch_h2d_wait",
                category="replay_pipeline",
                start_ns=_t0,
                end_ns=_wait_end,
                args={"tick_id": tick_id, "batch_gpu_slot": slot, "pipeline": "gpu_resident"},
            )
            self._trace_recorder.add_slice(
                "replay_pipeline/gpu_wait_for_batch",
                category="replay_pipeline",
                start_ns=_t0,
                end_ns=_wait_end,
                args={"tick_id": tick_id, "batch_gpu_slot": slot, "pipeline": "gpu_resident"},
            )
        _swap_ns = time.perf_counter_ns()
        with self._prepare_condition:
            old_hot = self._hot
            old_cold = self._cold
            if slot != self._cold:
                raise RuntimeError("Prepared replay batch is not in the current cold slot")
            self._hot, self._cold = self._cold, self._hot
            self._has_hot_batch = True
            self._hot_metadata = metadata
            self._prepared_metadata = None
            self._prepare_tick_id = None
            self._prepare_state = "idle"
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
                    "pipeline": "gpu_resident",
                },
            )
        return self._large_batch_view(self._hot)

    def after_tick(self) -> None:
        self._has_hot_batch = False
        self._hot_metadata = None

    def close(self) -> None:
        self._closed = True
        with self._prepare_condition:
            self._prepare_condition.notify_all()
        if self._sync_thread is not None:
            self._sync_thread.join(timeout=2.0)
        for event in self._slot_events:
            try:
                event.synchronize()
            except Exception:
                pass
        self._transfer_backend.close()
        self._host_pinned = False
        self._gpu_packed.clear()
        self._gpu_critic_graph_packed.clear()
        self._gather_scratch = None
        if hasattr(self, "_gpu_storage"):
            del self._gpu_storage
