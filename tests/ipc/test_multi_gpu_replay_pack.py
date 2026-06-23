"""Tests for multi-rank replay pack IPC contract."""

from __future__ import annotations

import queue

import pytest
import torch

from unilab.algos.torch.offpolicy.worker import (
    _drain_collector_pack_requests,
    _service_collector_pack_requests,
)
from unilab.ipc.replay_buffer import ReplayBuffer
from unilab.ipc.replay_pipelines.base import ReplayTickMetadata
from unilab.ipc.replay_pipelines.multi_gpu_cpu_pinned import (
    REPLAY_PREPARE_READY_POLL_SEC,
    MultiGPUCPUPinnedReplayPipeline,
)


def _make_replay_buffer() -> ReplayBuffer:
    buf = ReplayBuffer(
        capacity=16,
        obs_dim=2,
        action_dim=1,
        device="cpu",
        critic_dim=0,
        packed_cpu_storage=True,
    )
    obs = torch.arange(32, dtype=torch.float32).reshape(16, 2)
    actions = torch.arange(16, dtype=torch.float32).reshape(16, 1)
    rewards = torch.arange(16, dtype=torch.float32)
    next_obs = obs + 100
    dones = torch.zeros(16)
    truncated = torch.zeros(16)
    buf.add(obs, actions, rewards, next_obs, dones, truncated)
    return buf


def test_collector_pack_routes_ranked_requests_to_rank_slots_and_ready_queues() -> None:
    replay_buffer = _make_replay_buffer()
    packed_width = int(replay_buffer._storage.shape[1])
    sample_count = 4
    request_queue: queue.Queue = queue.Queue()
    ready_queues = [queue.Queue(), queue.Queue()]
    shared_slots = [
        [torch.empty((sample_count, packed_width), dtype=torch.float32) for _ in range(2)]
        for _ in range(2)
    ]

    request_queue.put(
        {
            "tick_id": 7,
            "rank": 1,
            "world_size": 2,
            "sample_seed": 123,
            "sample_count": sample_count,
            "shared_slot": 0,
            "target_gpu_slot": 0,
            "learner_hot_gpu_slot": 1,
            "min_snapshot_ptr": 0,
        }
    )

    serviced, pending = _service_collector_pack_requests(
        replay_buffer,
        request_queue,
        ready_queues,
        shared_slots,
    )

    assert serviced is True
    assert pending is None
    assert ready_queues[0].empty()
    ready = ready_queues[1].get_nowait()
    assert ready["rank"] == 1
    assert ready["world_size"] == 2
    assert ready["sample_seed"] == 123
    assert ready["sample_count"] == sample_count
    assert not torch.isnan(shared_slots[1][0]).any()


def test_collector_pack_defers_until_min_snapshot_ptr_for_ranked_request() -> None:
    replay_buffer = _make_replay_buffer()
    packed_width = int(replay_buffer._storage.shape[1])
    sample_count = 2
    request = {
        "tick_id": 8,
        "rank": 0,
        "world_size": 2,
        "sample_seed": 456,
        "sample_count": sample_count,
        "shared_slot": 1,
        "target_gpu_slot": 1,
        "learner_hot_gpu_slot": 0,
        "min_snapshot_ptr": int(replay_buffer.ptr[0]) + 1,
    }
    request_queue: queue.Queue = queue.Queue()
    request_queue.put(request)
    ready_queues = [queue.Queue(), queue.Queue()]
    shared_slots = [
        [torch.empty((sample_count, packed_width), dtype=torch.float32) for _ in range(2)]
        for _ in range(2)
    ]

    serviced, pending = _service_collector_pack_requests(
        replay_buffer,
        request_queue,
        ready_queues,
        shared_slots,
    )

    assert serviced is False
    assert pending == request
    assert ready_queues[0].empty()
    assert ready_queues[1].empty()


def test_rank_local_pipeline_requests_rank_seed_and_consumes_cpu_batch() -> None:
    replay_buffer = _make_replay_buffer()
    packed_width = int(replay_buffer._storage.shape[1])
    sample_count = 3
    request_queue: queue.Queue = queue.Queue()
    ready_queue: queue.Queue = queue.Queue()
    shared_slots = [
        torch.empty((sample_count, packed_width), dtype=torch.float32) for _ in range(2)
    ]
    pipeline = MultiGPUCPUPinnedReplayPipeline(
        replay_buffer,
        rank=1,
        world_size=2,
        device="cpu",
        sample_count=sample_count,
        base_seed=100,
        collector_pack_request_queue=request_queue,
        collector_pack_ready_queue=ready_queue,
        collector_pack_shared_slots=shared_slots,
    )
    try:
        assert pipeline.start_prepare(5, sample_count)
        request = request_queue.get_nowait()
        assert request["rank"] == 1
        assert request["world_size"] == 2
        assert request["sample_seed"] == 111

        ready = {
            "tick_id": request["tick_id"],
            "rank": request["rank"],
            "world_size": request["world_size"],
            "snapshot_ptr": int(replay_buffer.ptr[0]),
            "snapshot_size": int(replay_buffer.size[0]),
            "sample_seed": request["sample_seed"],
            "sample_count": sample_count,
            "shared_slot": request["shared_slot"],
            "target_gpu_slot": request["target_gpu_slot"],
        }
        torch.manual_seed(0)
        shared_slots[request["shared_slot"]].copy_(replay_buffer._storage[:sample_count])
        ready_queue.put(ready)

        assert pipeline.wait_until_ready(5, sample_count)
        batch = pipeline.sample_large_batch(5, sample_count)
        assert batch["obs"].shape == (sample_count, 2)
        assert torch.equal(
            batch["obs"], replay_buffer._storage[:sample_count, replay_buffer._obs_sl]
        )
        pipeline.after_tick()
    finally:
        pipeline.close()


def test_rank_local_pipeline_wait_until_ready_uses_fine_grained_polling() -> None:
    replay_buffer = _make_replay_buffer()
    packed_width = int(replay_buffer._storage.shape[1])
    sample_count = 3
    request_queue: queue.Queue = queue.Queue()
    ready_queue: queue.Queue = queue.Queue()
    shared_slots = [
        torch.empty((sample_count, packed_width), dtype=torch.float32) for _ in range(2)
    ]
    pipeline = MultiGPUCPUPinnedReplayPipeline(
        replay_buffer,
        rank=1,
        world_size=2,
        device="cpu",
        sample_count=sample_count,
        base_seed=100,
        collector_pack_request_queue=request_queue,
        collector_pack_ready_queue=ready_queue,
        collector_pack_shared_slots=shared_slots,
    )

    class _RecordingCondition:
        def __init__(self) -> None:
            self.timeouts: list[float | None] = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            del exc_type, exc, tb
            return False

        def wait(self, timeout=None) -> bool:
            self.timeouts.append(timeout)
            pipeline._prepared_metadata = ReplayTickMetadata(
                tick_id=5,
                snapshot_ptr=int(replay_buffer.ptr[0]),
                snapshot_size=int(replay_buffer.size[0]),
                sample_seed=111,
                sample_count=sample_count,
                batch_host_slot=1,
                batch_gpu_slot=1,
            )
            return True

    condition = _RecordingCondition()
    pipeline._prepare_condition = condition
    pipeline._transfer_backend.synchronize_ready = lambda slot: None

    try:
        assert pipeline.wait_until_ready(5, sample_count)
    finally:
        pipeline.close()

    assert condition.timeouts == [pytest.approx(REPLAY_PREPARE_READY_POLL_SEC)]
    assert condition.timeouts[0] < 0.01


def test_rank_local_pipeline_rejects_runner_rank_matrix_slots() -> None:
    replay_buffer = _make_replay_buffer()
    packed_width = int(replay_buffer._storage.shape[1])
    sample_count = 3
    rank_matrix_slots = [
        [torch.empty((sample_count, packed_width), dtype=torch.float32) for _ in range(2)]
        for _ in range(2)
    ]

    with pytest.raises(ValueError, match="rank-local shared slots"):
        MultiGPUCPUPinnedReplayPipeline(
            replay_buffer,
            rank=1,
            world_size=2,
            device="cpu",
            sample_count=sample_count,
            collector_pack_request_queue=queue.Queue(),
            collector_pack_ready_queue=queue.Queue(),
            collector_pack_shared_slots=rank_matrix_slots,
        )


def test_multi_rank_pipeline_uses_runner_rank_local_ipc_shape() -> None:
    replay_buffer = _make_replay_buffer()
    packed_width = int(replay_buffer._storage.shape[1])
    sample_count = 3
    request_queues = [queue.Queue(), queue.Queue()]
    ready_queues = [queue.Queue(), queue.Queue()]
    shared_slots = [
        [torch.empty((sample_count, packed_width), dtype=torch.float32) for _ in range(2)]
        for _ in range(2)
    ]
    pipeline = MultiGPUCPUPinnedReplayPipeline(
        replay_buffer,
        rank=1,
        world_size=2,
        device="cpu",
        sample_count=sample_count,
        base_seed=100,
        collector_pack_request_queue=request_queues[1],
        collector_pack_ready_queue=ready_queues[1],
        collector_pack_shared_slots=shared_slots[1],
    )
    try:
        assert pipeline.start_prepare(5, sample_count)
        request = request_queues[1].get_nowait()
        pending = _drain_collector_pack_requests(
            replay_buffer,
            request_queues[1],
            ready_queues,
            shared_slots,
            pending_request=request,
        )
        assert pending is None
        assert ready_queues[0].empty()

        assert pipeline.wait_until_ready(5, sample_count)
        batch = pipeline.sample_large_batch(5, sample_count)
        assert batch["obs"].shape == (sample_count, 2)
        assert batch["obs"].device.type == "cpu"
    finally:
        pipeline.close()


def test_collector_pack_drain_services_available_multi_rank_requests() -> None:
    replay_buffer = _make_replay_buffer()
    packed_width = int(replay_buffer._storage.shape[1])
    sample_count = 2
    request_queue: queue.Queue = queue.Queue()
    ready_queues = [queue.Queue(), queue.Queue()]
    shared_slots = [
        [torch.empty((sample_count, packed_width), dtype=torch.float32) for _ in range(2)]
        for _ in range(2)
    ]
    for rank in range(2):
        request_queue.put(
            {
                "tick_id": 9,
                "rank": rank,
                "world_size": 2,
                "sample_seed": 900 + rank,
                "sample_count": sample_count,
                "shared_slot": 0,
                "target_gpu_slot": 0,
                "learner_hot_gpu_slot": 1,
                "min_snapshot_ptr": 0,
            }
        )

    pending = _drain_collector_pack_requests(
        replay_buffer,
        request_queue,
        ready_queues,
        shared_slots,
    )

    assert pending is None
    assert request_queue.empty()
    assert ready_queues[0].get_nowait()["rank"] == 0
    assert ready_queues[1].get_nowait()["rank"] == 1


def test_collector_pack_service_parallel_rank_queues() -> None:
    from unilab.algos.torch.offpolicy.worker import _CollectorPackService

    replay_buffer = _make_replay_buffer()
    packed_width = int(replay_buffer._storage.shape[1])
    sample_count = 2
    request_queues = [queue.Queue(), queue.Queue()]
    ready_queues = [queue.Queue(), queue.Queue()]
    shared_slots = [
        [torch.empty((sample_count, packed_width), dtype=torch.float32) for _ in range(2)]
        for _ in range(2)
    ]
    stop_event = type("_Stop", (), {"_set": False})()
    stop_event.is_set = lambda: bool(stop_event._set)

    service = _CollectorPackService(
        replay_buffer,
        request_queues,
        ready_queues,
        shared_slots,
        stop_event=stop_event,
    )
    try:
        service.start()
        for rank in range(2):
            request_queues[rank].put(
                {
                    "tick_id": 10,
                    "rank": rank,
                    "world_size": 2,
                    "sample_seed": 1000 + rank,
                    "sample_count": sample_count,
                    "shared_slot": 0,
                    "target_gpu_slot": 0,
                    "learner_hot_gpu_slot": 1,
                    "min_snapshot_ptr": 0,
                }
            )
        assert ready_queues[0].get(timeout=1.0)["sample_seed"] == 1000
        assert ready_queues[1].get(timeout=1.0)["sample_seed"] == 1001
    finally:
        stop_event._set = True
        service.close()
