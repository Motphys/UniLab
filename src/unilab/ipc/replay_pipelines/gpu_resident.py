"""GPU-resident replay pipeline for one CUDA or MPS learner.

Single-device training consumes fixed-depth packed ingress slots and keeps the
full replay ring authoritative on device. CUDA uses a learner-side daemon and
side stream. MPS has no public stream API, so existing learner-thread calls
submit copies and device-side sampling. The legacy CPU-ring mirror remains only
for the multi-GPU migration path.

Consistency model:

- The sample domain is ``[0, min(device_visible_ptr, capacity))`` where
  ``device_visible_ptr`` only advances past rows whose device copy completed.
- Span copies and batch gathers are totally ordered on the CUDA side stream or
  the MPS learner-thread command queue. A gather therefore observes every
  completed span before it and none submitted after it.
- An ingress slot is not reusable and ``ptr``/``size`` do not advance until its
  device-copy completion event is observed.
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


def _device_memory_budget(device: torch.device) -> tuple[int, int]:
    """Return ``(available, total_budget)`` bytes for resident allocations."""
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        return int(free_bytes), int(total_bytes)
    if device.type == "mps":
        recommended_max_memory = getattr(torch.mps, "recommended_max_memory", None)
        driver_allocated_memory = getattr(torch.mps, "driver_allocated_memory", None)
        if not callable(recommended_max_memory) or not callable(driver_allocated_memory):
            raise RuntimeError(
                "MPS gpu_resident replay requires recommended_max_memory() and "
                "driver_allocated_memory() for its allocation guard"
            )
        total_bytes = int(cast(Any, recommended_max_memory)())
        allocated_bytes = int(cast(Any, driver_allocated_memory)())
        return max(0, total_bytes - allocated_bytes), total_bytes
    raise ValueError(f"Unsupported gpu_resident replay device type {device.type!r}")


def _validate_device_memory_budget(
    device: torch.device,
    *,
    required_bytes: int,
    storage_bytes: int,
    batch_bytes: int,
    headroom: float,
) -> None:
    available_bytes, total_bytes = _device_memory_budget(device)
    usable_bytes = int(available_bytes * headroom)
    if required_bytes <= usable_bytes:
        return
    raise RuntimeError(
        f"GPU-resident replay mirror does not fit on {device.type}: "
        f"requires {required_bytes / 2**30:.2f} GiB "
        f"(storage {storage_bytes / 2**30:.2f} + batch slots {batch_bytes / 2**30:.2f}), "
        f"device available budget {available_bytes / 2**30:.2f} GiB / "
        f"total budget {total_bytes / 2**30:.2f} GiB "
        f"({headroom:.0%} of available budget may be used)"
    )


class GPUResidentReplayPipeline:
    """ReplayPipeline backed by an authoritative packed device ring."""

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
        if self._device.type not in {"cuda", "mps"}:
            raise ValueError(
                "GPUResidentReplayPipeline requires a CUDA or MPS device; "
                f"got {self._device.type!r} (use cpu_pinned_double_buffer instead)"
            )
        if self._device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("GPUResidentReplayPipeline requires an available MPS device")
        if not getattr(replay_buffer, "_packed_cpu_storage", False):
            raise ValueError("GPUResidentReplayPipeline requires packed CPU replay storage")
        self._main_thread_submission = self._device.type == "mps"
        self._learner_thread_id = threading.get_ident()
        self._pack_layout = pack_layout
        self._use_critic_graph_packed_source = (
            bool(use_critic_graph_packed_source) and self._pack_layout != "sac_graph"
        )
        self._sample_count = int(sample_count)
        self._base_seed = int(base_seed)
        self._trace_recorder = trace_recorder
        self._capacity = int(replay_buffer.capacity)
        self._device_authoritative = bool(replay_buffer.device_authoritative)
        self._storage_width = int(replay_buffer.storage_width)
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
        _validate_device_memory_budget(
            self._device,
            required_bytes=required_bytes,
            storage_bytes=storage_bytes,
            batch_bytes=slot_bytes + scratch_bytes + critic_slot_bytes,
            headroom=self._MEMORY_HEADROOM,
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
        host_slots = (
            replay_buffer._ingress_slots if self._device_authoritative else [replay_buffer._storage]
        )
        try:
            self._transfer_backend.register_host_slots(host_slots)
            self._host_pinned = bool(self._transfer_backend.host_pinned)
        except RuntimeError as exc:
            print(
                f"[GPUResidentReplay] Host storage registration failed ({exc}); "
                "falling back to pageable device copies.",
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

        self._sync_stream: Any | None = None
        if self._device.type == "cuda":
            self._sync_stream = cast(torch.cuda.Stream, torch.cuda.Stream(device=self._device))
            self._slot_events: list[Any] = [torch.cuda.Event() for _ in range(2)]
        else:
            self._slot_events = [torch.mps.Event() for _ in range(2)]
        self._span_events: deque[tuple[int, Any, int | None, int, int]] = deque()
        self._submission_lock = threading.Lock()
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
        self._sync_thread: threading.Thread | None = None
        if not self._main_thread_submission:
            self._sync_thread = threading.Thread(
                target=self._sync_worker,
                name="replay_gpu_resident_sync",
                daemon=True,
            )
            self._sync_thread.start()

    @property
    def h2d_submitter(self) -> str:
        if self._main_thread_submission:
            return (
                "gpu_resident_ingress_main_thread"
                if self._device_authoritative
                else "gpu_resident_mirror_main_thread"
            )
        return "gpu_resident_ingress" if self._device_authoritative else "gpu_resident_mirror"

    @property
    def transfer_manifest(self) -> dict[str, object]:
        return {
            "backend": type(self._transfer_backend).__name__,
            "device": str(self._device),
            "device_family": self._device_family,
            "pipeline": "gpu_resident",
            "storage_owner": "device" if self._device_authoritative else "cpu_mirror",
            "host_memory_kind": (
                self._transfer_backend.host_memory_kind if self._host_pinned else "pageable_shared"
            ),
            "host_pinned": self._host_pinned,
            "storage_rows": self._capacity,
            "storage_width": self._storage_width,
            "storage_bytes": int(self._gpu_storage.numel() * self._gpu_storage.element_size()),
            "host_storage_bytes": self._replay_buffer.host_storage_bytes,
            "ingress_depth": (
                self._replay_buffer._ingress_depth if self._device_authoritative else None
            ),
            "h2d_submitter": self.h2d_submitter,
            "device_submission_thread": "learner" if self._main_thread_submission else "daemon",
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

    # -- device submission ---------------------------------------------------

    def _assert_mps_learner_thread(self) -> None:
        if self._main_thread_submission and threading.get_ident() != self._learner_thread_id:
            raise RuntimeError(
                "MPS gpu_resident replay device work must be submitted from the learner thread"
            )

    def _drive_mps_learner_thread(self, *, wait: bool = False) -> bool:
        """Advance MPS ingress and gather work from an existing learner call."""
        if not self._main_thread_submission:
            return False
        self._assert_mps_learner_thread()
        try:
            if wait:
                did_work = self._submit_new_spans()
                for _, event, _, _, _ in self._span_events:
                    event.synchronize()
                did_work |= self._drain_completed_spans()
                did_work |= self._service_pending_prepare()
            else:
                did_work = self._drain_completed_spans()
                did_work |= self._service_pending_prepare()
                did_work |= self._submit_new_spans()
            if wait and self._prepared_metadata is not None:
                slot = self._prepared_metadata.batch_gpu_slot
                if slot is not None:
                    self._slot_events[slot].synchronize()
                    self._prepare_state = "ready"
            return did_work
        except BaseException as exc:
            with self._prepare_condition:
                self._prepare_error = exc
                self._prepare_condition.notify_all()
            raise

    # -- CUDA sync thread -----------------------------------------------------

    def _sync_worker(self) -> None:
        while True:
            if self._closed:
                return
            try:
                if self._device_authoritative:
                    did_work = self._drain_completed_spans()
                    did_work |= self._service_pending_prepare()
                    did_work |= self._submit_new_spans()
                else:
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
        if self._main_thread_submission:
            self._assert_mps_learner_thread()
        with self._submission_lock:
            if self._device_authoritative:
                submitted = False
                while True:
                    ingress = self._replay_buffer.take_published_ingress()
                    if ingress is None:
                        return submitted
                    slot, start, count, source = ingress
                    if start != self._submitted_ptr:
                        raise RuntimeError(
                            "Bounded replay ingress publication is not contiguous: "
                            f"submitted ptr {self._submitted_ptr}, slot start {start}"
                        )
                    self._submit_span_copy(
                        start=start,
                        end=start + count,
                        source=source,
                        ingress_slot=slot,
                    )
                    submitted = True

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
            self._submit_span_copy(
                start=self._submitted_ptr,
                end=ptr,
                source=self._replay_buffer._storage,
                ingress_slot=None,
            )
            return True

    def _submit_span_copy(
        self,
        *,
        start: int,
        end: int,
        source: torch.Tensor,
        ingress_slot: int | None,
    ) -> None:
        h2d_begin_ns = time.perf_counter_ns()
        start_event = None
        end_event = None
        record_cuda = self._trace_recorder is not None and self._trace_cuda_events

        def copy_spans(*, non_blocking: bool) -> None:
            source_offset = 0
            for offset, length in _ring_spans(start, end, self._capacity):
                source_span = (
                    source[source_offset : source_offset + length]
                    if ingress_slot is not None
                    else source[offset : offset + length]
                )
                self._gpu_storage[offset : offset + length].copy_(
                    source_span,
                    non_blocking=non_blocking,
                )
                source_offset += length

        if self._device.type == "cuda":
            with torch.cuda.device(self._device), torch.cuda.stream(self._sync_stream):
                if record_cuda:
                    start_event = cast(Any, torch.cuda.Event(enable_timing=True))
                    end_event = cast(Any, torch.cuda.Event(enable_timing=True))
                    start_event.record()
                copy_spans(non_blocking=True)
                if end_event is not None:
                    end_event.record()
                done_event = cast(Any, torch.cuda.Event())
                done_event.record(self._sync_stream)
        else:
            copy_spans(non_blocking=False)
            done_event = cast(Any, torch.mps.Event())
            done_event.record()
        self._span_events.append((end, done_event, ingress_slot, start, end - start))
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
                    "ingress_slot": ingress_slot,
                    "pinned_memory": self._host_pinned,
                    "pipeline": "gpu_resident",
                    "storage_owner": "device" if self._device_authoritative else "cpu_mirror",
                },
            )

    def _drain_completed_spans(self) -> bool:
        with self._submission_lock:
            return self._drain_completed_spans_locked()

    def _drain_completed_spans_locked(self) -> bool:
        drained = False
        while self._span_events and self._span_events[0][1].query():
            end, _, ingress_slot, start, count = self._span_events.popleft()
            if ingress_slot is not None:
                commit_ns = time.perf_counter_ns()
                self._replay_buffer.commit_ingress(
                    slot=ingress_slot,
                    start=start,
                    count=count,
                )
                if self._trace_recorder is not None:
                    self._trace_recorder.add_slice(
                        "replay_pipeline/ingress_commit",
                        category="replay_pipeline",
                        start_ns=commit_ns,
                        end_ns=time.perf_counter_ns(),
                        args={
                            "ingress_slot": ingress_slot,
                            "committed_ptr": end,
                            "rows": count,
                            "pipeline": "gpu_resident",
                        },
                    )
            self._visible_ptr = end
            drained = True
        if drained:
            with self._prepare_condition:
                self._prepare_condition.notify_all()
        return drained

    def _gather_rows(self, *, visible_size: int, slot: int, gen: torch.Generator) -> None:
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

    def _service_pending_prepare(self) -> bool:
        if self._main_thread_submission:
            self._assert_mps_learner_thread()
        if self._device_authoritative:
            with self._submission_lock:
                if self._span_events:
                    return False
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
        if self._device.type == "cuda":
            with torch.cuda.device(self._device), torch.cuda.stream(self._sync_stream):
                if record_cuda:
                    start_event = cast(Any, torch.cuda.Event(enable_timing=True))
                    end_event = cast(Any, torch.cuda.Event(enable_timing=True))
                    start_event.record()
                self._gather_rows(visible_size=visible_size, slot=slot, gen=gen)
                if end_event is not None:
                    end_event.record()
                self._slot_events[slot].record(self._sync_stream)
        else:
            self._gather_rows(visible_size=visible_size, slot=slot, gen=gen)
            self._slot_events[slot].record()
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

    def progress(self, *, wait: bool = False) -> bool:
        """Advance ingress work without changing replay lifecycle state."""
        if self._main_thread_submission:
            return self._drive_mps_learner_thread(wait=wait)
        if not wait:
            return False
        did_work = self._submit_new_spans()
        with self._submission_lock:
            for _, event, _, _, _ in self._span_events:
                event.synchronize()
            did_work |= self._drain_completed_spans_locked()
        return did_work

    def read_committed_fields(
        self,
        field_names: tuple[str, ...],
        *,
        start_ptr: int,
    ) -> tuple[int, dict[str, torch.Tensor]]:
        """Return a stable ordered field snapshot through the committed pointer."""
        if not self._device_authoritative:
            end_ptr = int(self._replay_buffer.ptr[0])
            count = min(max(end_ptr - start_ptr, 0), self._capacity)
            field_start = end_ptr - count
            return end_ptr, self._replay_buffer.read_recent_fields(
                field_names,
                field_start,
                count,
            )

        published_snapshot = self._replay_buffer.published_ptr
        while self._submitted_ptr < published_snapshot:
            if not self._submit_new_spans():
                time.sleep(self._POLL_INTERVAL_S)

        with self._submission_lock:
            for _, event, _, _, _ in self._span_events:
                event.synchronize()
            self._drain_completed_spans_locked()
            end_ptr = self._visible_ptr
            count = min(max(end_ptr - start_ptr, 0), self._capacity)
            field_start = end_ptr - count
            index = field_start % self._capacity
            fields: dict[str, torch.Tensor] = {}
            for field_name in field_names:
                source = self._replay_buffer.field_view(self._gpu_storage, field_name)
                if index + count <= self._capacity:
                    fields[field_name] = source[index : index + count].clone()
                    continue
                split = self._capacity - index
                fields[field_name] = torch.cat(
                    [source[index:], source[: count - split]],
                    dim=0,
                ).clone()
            if self._device.type == "cuda":
                snapshot_event = cast(Any, torch.cuda.Event())
                snapshot_event.record(torch.cuda.current_stream(self._device))
                assert self._sync_stream is not None
                self._sync_stream.wait_event(snapshot_event)
            return end_ptr, fields

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
        if self._main_thread_submission:
            self._drive_mps_learner_thread()
        return True

    def batch_ready(self, tick_id: int, sample_count: int) -> bool:
        self._validate_sample_count(sample_count)
        if self._has_hot_batch:
            if self._hot_metadata is not None and self._hot_metadata.tick_id != int(tick_id):
                return False
            return True
        if self._main_thread_submission:
            self._drive_mps_learner_thread()
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
            if self._main_thread_submission:
                while self._prepared_metadata is None and self._prepare_error is None:
                    did_work = self._drive_mps_learner_thread(wait=True)
                    if not did_work:
                        time.sleep(self._POLL_INTERVAL_S)
            else:
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
        if self._device.type == "cuda":
            torch.cuda.current_stream(self._device).wait_event(self._slot_events[slot])
        else:
            self._slot_events[slot].synchronize()
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
        try:
            self._submit_new_spans()
            with self._submission_lock:
                for _, event, _, _, _ in self._span_events:
                    event.synchronize()
                self._drain_completed_spans_locked()
        except Exception:
            pass
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
