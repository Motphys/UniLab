"""Rank-local multi-GPU replay pipeline for packed CPU replay samples."""

from __future__ import annotations

import queue
import threading
import time
from typing import Dict

import torch

from unilab.ipc.replay_buffer import ReplayBuffer
from unilab.ipc.replay_pipelines.base import ReplayTickMetadata
from unilab.ipc.replay_pipelines.transfer import build_replay_transfer_backend


class MultiGPUCPUPinnedReplayPipeline:
    """Per-rank replay pipeline with independent host slots and H2D stream.

    The replay buffer remains authoritative CPU shared storage. Each learner
    rank owns its host staging slots and device slots, so H2D submission happens
    concurrently in the rank-local worker process instead of funnelling through
    rank 0.
    """

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
        collector_pack_request_queue=None,
        collector_pack_ready_queue=None,
        collector_pack_shared_slots=None,
    ) -> None:
        if int(world_size) <= 1:
            raise ValueError("MultiGPUCPUPinnedReplayPipeline requires world_size > 1")
        if not getattr(replay_buffer, "_packed_cpu_storage", False):
            raise ValueError("Multi-GPU replay pipeline requires packed ReplayBuffer storage")
        if (
            collector_pack_request_queue is None
            or collector_pack_ready_queue is None
            or collector_pack_shared_slots is None
        ):
            raise ValueError("Multi-GPU replay pipeline requires collector pack IPC objects")

        self._replay_buffer = replay_buffer
        self._rank = int(rank)
        self._world_size = int(world_size)
        self._device = torch.device(device)
        self._sample_count = int(sample_count)
        self._base_seed = int(base_seed)
        self._trace_recorder = trace_recorder
        self._pack_layout = "packed"
        self._pack_executor = "collector_thread"
        self._ring_depth = 2
        self._transfer_backend = build_replay_transfer_backend(
            device=self._device,
            ring_depth=self._ring_depth,
        )
        self._trace_cuda_events = bool(trace_cuda_events) and (
            self._transfer_backend.supports_timing_events
        )
        self._collector_pack_request_queue = collector_pack_request_queue
        self._collector_pack_ready_queue = collector_pack_ready_queue
        self._collector_pack_shared_slots = collector_pack_shared_slots
        if len(self._collector_pack_shared_slots) != self._ring_depth or not all(
            isinstance(slot, torch.Tensor) for slot in self._collector_pack_shared_slots
        ):
            raise ValueError(
                "Multi-GPU replay pipeline expects rank-local shared slots with "
                f"ring_depth={self._ring_depth}"
            )
        self._transfer_backend.register_host_slots(self._collector_pack_shared_slots)
        self._packed_width = int(replay_buffer._storage.shape[1])
        self._gpu_packed = self._transfer_backend.allocate_device_slots(
            count=self._ring_depth,
            shape=(self._sample_count, self._packed_width),
            dtype=torch.float32,
        )

        self._hot = 0
        self._cold = 1
        self._has_hot_batch = False
        self._hot_metadata: ReplayTickMetadata | None = None
        self._prepared_metadata: ReplayTickMetadata | None = None
        self._prepare_tick_id: int | None = None
        self._prepare_state = "idle"
        self._prepare_error: BaseException | None = None
        self._prepare_condition = threading.Condition()
        self._closed = False
        self.last_incremental_h2d_time_s = 0.0
        self._collector_h2d_thread = threading.Thread(
            target=self._collector_h2d_worker,
            name=f"replay_rank{self._rank}_h2d",
            daemon=True,
        )
        self._collector_h2d_thread.start()

    @property
    def h2d_submitter(self) -> str:
        return self._transfer_backend.h2d_submitter

    @property
    def transfer_manifest(self) -> dict[str, object]:
        return {
            "backend": type(self._transfer_backend).__name__,
            "device": str(self._device),
            "device_family": self._transfer_backend.device_family,
            "host_memory_kind": self._transfer_backend.host_memory_kind,
            "host_pinned": self._transfer_backend.host_pinned,
            "direct_pinned_shared": self._transfer_backend.direct_pinned_shared,
            "supports_async_submit": self._transfer_backend.supports_async_submit,
            "supports_timing_events": self._transfer_backend.supports_timing_events,
            "h2d_submitter": self._transfer_backend.h2d_submitter,
            "rank": self._rank,
            "world_size": self._world_size,
            "ring_depth": self._ring_depth,
        }

    def _snapshot(self) -> tuple[int, int]:
        return int(self._replay_buffer.ptr[0]), int(self._replay_buffer.size[0])

    def _packed_h2d_source(self, slot: int) -> torch.Tensor:
        return self._collector_pack_shared_slots[slot]

    def _h2d_bytes(self) -> int:
        source = self._packed_h2d_source(0)
        return int(source.numel() * source.element_size())

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

    def _submit_h2d(self, slot: int, metadata: ReplayTickMetadata) -> float:
        self._transfer_backend.clear_ready(slot)
        return self._transfer_backend.submit_h2d(
            slot=slot,
            dst=self._gpu_packed[slot],
            src=self._packed_h2d_source(slot),
            metadata=metadata,
            trace_recorder=self._trace_recorder,
            trace_cuda_events=self._trace_cuda_events,
            h2d_bytes=self._h2d_bytes(),
            pack_layout=self._pack_layout,
            pack_executor=self._pack_executor,
        )

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
                if int(ready.get("rank", self._rank)) != self._rank:
                    raise RuntimeError(
                        f"Rank {self._rank} received replay batch for rank {ready.get('rank')}"
                    )
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
                self.last_incremental_h2d_time_s = self._submit_h2d(slot, metadata)
                with self._prepare_condition:
                    if self._prepare_tick_id != metadata.tick_id:
                        raise RuntimeError(
                            f"Rank {self._rank} packed tick {metadata.tick_id} "
                            f"does not match pending tick {self._prepare_tick_id}"
                        )
                    self._prepared_metadata = metadata
                    self._prepare_state = "h2d_submitted"
                    self._prepare_error = None
                    self._prepare_condition.notify_all()
            except BaseException as exc:
                with self._prepare_condition:
                    self._prepare_error = exc
                    self._prepare_condition.notify_all()

    def _refresh_prepare_state(self) -> None:
        if self._prepare_error is not None:
            raise self._prepare_error
        if self._prepared_metadata is not None:
            slot = self._prepared_metadata.batch_gpu_slot
            if slot is not None and self._transfer_backend.ready_query(slot):
                self._prepare_state = "ready"

    def start_prepare(
        self,
        tick_id: int,
        sample_count: int,
        min_snapshot_ptr: int | None = None,
    ) -> bool:
        if int(sample_count) != self._sample_count:
            raise ValueError("sample_count must match the allocated multi-GPU replay slots")
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
                "Cannot prepare a new replay batch before consuming the previous one"
            )

        slot = self._cold
        self._transfer_backend.clear_ready(slot)
        self._prepare_tick_id = int(tick_id)
        self._prepare_error = None
        snapshot_ptr, snapshot_size = self._snapshot()
        sample_seed = self._base_seed + int(tick_id) * self._world_size + self._rank
        min_snapshot_ptr = snapshot_ptr if min_snapshot_ptr is None else int(min_snapshot_ptr)
        request = {
            "tick_id": int(tick_id),
            "rank": self._rank,
            "world_size": self._world_size,
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
        self._prepare_state = "collector_pack_requested"
        self._collector_pack_request_queue.put(request)
        return True

    def batch_ready(self, tick_id: int, sample_count: int) -> bool:
        if int(sample_count) != self._sample_count:
            raise ValueError("sample_count must match the allocated multi-GPU replay slots")
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
        if int(sample_count) != self._sample_count:
            raise ValueError("sample_count must match the allocated multi-GPU replay slots")
        self._refresh_prepare_state()
        if self._prepared_metadata is None:
            if self._prepare_tick_id is None:
                self.start_prepare(tick_id, sample_count)
            with self._prepare_condition:
                while self._prepared_metadata is None and self._prepare_error is None:
                    self._prepare_condition.wait(timeout=0.1)
                if self._prepare_error is not None:
                    raise self._prepare_error
        assert self._prepared_metadata is not None
        if self._prepared_metadata.tick_id != int(tick_id):
            raise RuntimeError(
                f"Rank {self._rank} prepared tick {self._prepared_metadata.tick_id} "
                f"does not match requested tick {tick_id}"
            )
        slot = self._prepared_metadata.batch_gpu_slot
        assert slot is not None
        self._transfer_backend.synchronize_ready(slot)
        self._prepare_state = "ready"
        return True

    def sample_large_batch(self, tick_id: int, sample_count: int) -> Dict[str, torch.Tensor]:
        if int(sample_count) != self._sample_count:
            raise ValueError("sample_count must match the allocated multi-GPU replay slots")
        if self._has_hot_batch:
            if self._hot_metadata is not None and self._hot_metadata.tick_id != int(tick_id):
                raise RuntimeError(
                    f"Rank {self._rank} hot tick {self._hot_metadata.tick_id} "
                    f"does not match requested tick {tick_id}"
                )
            return self._packed_batch_view(self._gpu_packed[self._hot])
        if not self.batch_ready(tick_id, sample_count):
            self.wait_until_ready(tick_id, sample_count)
        assert self._prepared_metadata is not None
        slot = self._prepared_metadata.batch_gpu_slot
        assert slot is not None
        wait_begin_ns = time.perf_counter_ns()
        self._transfer_backend.wait_current_stream_for_ready(slot)
        wait_copy_time_s = float(getattr(self._transfer_backend, "last_wait_copy_time_s", 0.0))
        if wait_copy_time_s > 0.0:
            self.last_incremental_h2d_time_s = wait_copy_time_s
        if self._trace_recorder is not None:
            self._trace_recorder.add_slice(
                "replay_pipeline/rank_batch_h2d_wait",
                category="replay_pipeline",
                start_ns=wait_begin_ns,
                end_ns=time.perf_counter_ns(),
                args={"tick_id": tick_id, "rank": self._rank, "batch_gpu_slot": slot},
            )
        if slot != self._cold:
            raise RuntimeError("Prepared multi-GPU replay batch is not in the current cold slot")
        self._hot, self._cold = self._cold, self._hot
        self._has_hot_batch = True
        self._hot_metadata = self._prepared_metadata
        self._prepared_metadata = None
        self._prepare_tick_id = None
        self._prepare_state = "idle"
        return self._packed_batch_view(self._gpu_packed[self._hot])

    def after_tick(self) -> None:
        self._has_hot_batch = False
        self._hot_metadata = None

    def close(self) -> None:
        self._closed = True
        try:
            self._collector_pack_ready_queue.put_nowait(None)
        except Exception:
            pass
        self._collector_h2d_thread.join(timeout=2.0)
        if self._prepared_metadata is not None:
            slot = self._prepared_metadata.batch_gpu_slot
            if slot is not None:
                self._transfer_backend.synchronize_ready(slot)
        self._transfer_backend.close()
        self._gpu_packed.clear()
