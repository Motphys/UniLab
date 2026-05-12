"""Tests for CPUPinnedDoubleBufferReplayPipeline."""

from __future__ import annotations

import queue
import time

import pytest
import torch

from unilab.ipc.replay_buffer import ReplayBuffer
from unilab.ipc.replay_pipelines.base import ReplayPipeline, ReplayTickMetadata
from unilab.ipc.replay_pipelines.gpu_cache import GpuCacheReplayPipeline

_HAS_CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# GpuCacheReplayPipeline (control path)
# ---------------------------------------------------------------------------


def _make_cpu_replay(capacity: int = 64, obs_dim: int = 4, action_dim: int = 2) -> ReplayBuffer:
    rb = ReplayBuffer(capacity=capacity, obs_dim=obs_dim, action_dim=action_dim, device="cpu")
    n = min(32, capacity)
    rb.add(
        obs=torch.randn(n, obs_dim),
        actions=torch.randn(n, action_dim),
        rewards=torch.randn(n),
        next_obs=torch.randn(n, obs_dim),
        dones=torch.zeros(n),
        truncated=torch.zeros(n),
    )
    return rb


def test_gpu_cache_pipeline_satisfies_protocol():
    rb = _make_cpu_replay()
    pipeline = GpuCacheReplayPipeline(rb)
    assert isinstance(pipeline, ReplayPipeline)


def test_gpu_cache_pipeline_sample_returns_correct_keys():
    rb = _make_cpu_replay()
    pipeline = GpuCacheReplayPipeline(rb)
    batch = pipeline.sample_large_batch(tick_id=0, sample_count=8)
    assert "obs" in batch
    assert "actions" in batch
    assert "rewards" in batch
    assert "next_obs" in batch
    assert "dones" in batch
    assert "truncated" in batch


def test_gpu_cache_pipeline_sample_correct_shape():
    rb = _make_cpu_replay(obs_dim=5, action_dim=3)
    pipeline = GpuCacheReplayPipeline(rb)
    batch = pipeline.sample_large_batch(tick_id=0, sample_count=16)
    assert batch["obs"].shape == (16, 5)
    assert batch["actions"].shape == (16, 3)
    assert batch["rewards"].shape == (16,)


def test_collector_pack_shared_batch_writes_expected_packed_rows():
    from unilab.algos.torch.offpolicy.worker import _collector_pack_shared_batch

    rb = _make_cpu_replay(capacity=64, obs_dim=5, action_dim=3)
    sample_count = 8
    shared_slots = [
        torch.empty((sample_count, rb._storage.shape[1])).share_memory_() for _ in range(2)
    ]
    request = {
        "tick_id": 7,
        "snapshot_ptr": int(rb.ptr[0]),
        "snapshot_size": int(rb.size[0]),
        "sample_seed": 123,
        "sample_count": sample_count,
        "shared_slot": 1,
        "learner_hot_gpu_slot": 0,
        "target_gpu_slot": 1,
    }

    ready = _collector_pack_shared_batch(rb, request, shared_slots)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(123)
    expected_indices = torch.randint(0, int(rb.size[0]), (sample_count,), generator=gen)
    torch.testing.assert_close(shared_slots[1], rb._storage[expected_indices])
    assert ready["tick_id"] == 7
    assert ready["shared_slot"] == 1
    assert ready["target_gpu_slot"] == 1
    assert ready["learner_hot_gpu_slot"] == 0


def test_collector_pack_shared_batch_rejects_hot_gpu_slot_target():
    from unilab.algos.torch.offpolicy.worker import _collector_pack_shared_batch

    rb = _make_cpu_replay(capacity=64, obs_dim=5, action_dim=3)
    shared_slots = [torch.empty((4, rb._storage.shape[1])).share_memory_()]
    request = {
        "tick_id": 1,
        "snapshot_ptr": int(rb.ptr[0]),
        "snapshot_size": int(rb.size[0]),
        "sample_seed": 5,
        "sample_count": 4,
        "shared_slot": 0,
        "learner_hot_gpu_slot": 0,
        "target_gpu_slot": 0,
    }
    with pytest.raises(RuntimeError, match="target_gpu_slot must differ"):
        _collector_pack_shared_batch(rb, request, shared_slots)


def test_collector_pack_request_uses_snapshot_after_pending_replay_add():
    import queue

    from unilab.algos.torch.offpolicy.worker import _service_collector_pack_requests

    rb = _make_cpu_replay(capacity=64, obs_dim=5, action_dim=3)
    old_ptr = int(rb.ptr[0])
    sample_count = 16
    shared_slots = [
        torch.empty((sample_count, rb._storage.shape[1])).share_memory_() for _ in range(2)
    ]
    request_queue = queue.Queue()
    ready_queue = queue.Queue()
    seed = 0
    while True:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        if torch.any(torch.randint(0, old_ptr + 4, (sample_count,), generator=gen) >= old_ptr):
            break
        seed += 1
    request_queue.put(
        {
            "tick_id": 9,
            "snapshot_ptr": old_ptr,
            "snapshot_size": int(rb.size[0]),
            "min_snapshot_ptr": old_ptr + 4,
            "sample_seed": seed,
            "sample_count": sample_count,
            "shared_slot": 1,
            "learner_hot_gpu_slot": 0,
            "target_gpu_slot": 1,
        }
    )

    serviced, pending = _service_collector_pack_requests(
        rb, request_queue, ready_queue, shared_slots
    )
    assert not serviced
    assert pending is not None

    rb.add(
        obs=torch.full((4, 5), 10.0),
        actions=torch.full((4, 3), 20.0),
        rewards=torch.full((4,), 30.0),
        next_obs=torch.full((4, 5), 40.0),
        dones=torch.zeros(4),
        truncated=torch.zeros(4),
    )
    serviced, pending = _service_collector_pack_requests(
        rb, request_queue, ready_queue, shared_slots, pending_request=pending
    )

    assert serviced
    assert pending is None
    ready = ready_queue.get_nowait()
    assert ready["snapshot_ptr"] == old_ptr + 4
    assert ready["snapshot_size"] == old_ptr + 4
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    expected_indices = torch.randint(0, old_ptr + 4, (sample_count,), generator=gen)
    torch.testing.assert_close(shared_slots[1], rb._storage[expected_indices])


class _RecordingTrace:
    def __init__(self) -> None:
        self.slices = []
        self.cuda_spans = []

    def add_slice(self, name, *, category, start_ns, end_ns, args=None, tid=None):
        del start_ns, end_ns, tid
        self.slices.append({"name": name, "category": category, "args": args or {}})

    def add_cuda_pending_span(
        self, name, *, category, cpu_begin_ns, start_event, end_event, args=None, tid=None
    ):
        del cpu_begin_ns, start_event, end_event, tid
        self.cuda_spans.append({"name": name, "category": category, "args": args or {}})


def _service_collector_pack(pipeline) -> None:
    from unilab.algos.torch.offpolicy.worker import _collector_pack_shared_batch

    request = pipeline._collector_pack_request_queue.get_nowait()
    ready = _collector_pack_shared_batch(
        pipeline._replay_buffer,
        request,
        pipeline._collector_pack_shared_slots,
    )
    pipeline._collector_pack_ready_queue.put(ready)


def _sample_ready(pipeline, tick_id: int, sample_count: int):
    if pipeline.start_prepare(tick_id=tick_id, sample_count=sample_count):
        _service_collector_pack(pipeline)
    assert pipeline.wait_until_ready(tick_id=tick_id, sample_count=sample_count)
    return pipeline.sample_large_batch(tick_id=tick_id, sample_count=sample_count)


class TestPortableDoubleBuffer:
    def _make_pipeline(self, rb, sample_count=16, base_seed=42, device="cpu", trace=None):
        from unilab.ipc.replay_pipelines.cpu_pinned_double_buffer import (
            CPUPinnedDoubleBufferReplayPipeline,
        )

        shared_slots = [
            torch.empty((sample_count, rb._storage.shape[1]), dtype=torch.float32).share_memory_()
            for _ in range(2)
        ]
        return CPUPinnedDoubleBufferReplayPipeline(
            rb,
            device=device,
            sample_count=sample_count,
            base_seed=base_seed,
            trace_recorder=trace,
            collector_pack_request_queue=queue.Queue(),
            collector_pack_ready_queue=queue.Queue(),
            collector_pack_shared_slots=shared_slots,
        )

    def test_cpu_portable_path_samples_expected_batch_without_cuda_pinning(self):
        rb = _make_cpu_replay(capacity=128, obs_dim=4, action_dim=2)
        pipeline = self._make_pipeline(rb, sample_count=8, base_seed=13)

        batch = _sample_ready(pipeline, tick_id=2, sample_count=8)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(13 + 2)
        expected_indices = torch.randint(0, int(rb.size[0]), (8,), generator=gen)
        torch.testing.assert_close(batch["obs"], rb._storage[expected_indices, rb._obs_sl])
        assert all(value.device.type == "cpu" for value in batch.values())
        assert pipeline._host_pinned is False
        assert pipeline._h2d_submitter == "torch_copy"
        assert not any(slot.is_pinned() for slot in pipeline._collector_pack_shared_slots)
        pipeline.close()

    def test_cpu_portable_trace_keeps_replay_slices_without_cuda_span(self):
        rb = _make_cpu_replay(capacity=128, obs_dim=4, action_dim=2)
        trace = _RecordingTrace()
        pipeline = self._make_pipeline(rb, sample_count=8, base_seed=3, trace=trace)

        _sample_ready(pipeline, tick_id=2, sample_count=8)

        names = {event["name"] for event in trace.slices}
        cuda_names = {event["name"] for event in trace.cuda_spans}
        assert "replay_pipeline/collector_pack_request" in names
        assert "replay_pipeline/batch_h2d_submit" in names
        assert "replay_pipeline/batch_h2d_wait" in names
        assert "replay_pipeline/hot_cold_swap" in names
        assert "gpu/replay_pipeline_batch_h2d" not in cuda_names
        submit = next(
            event for event in trace.slices if event["name"] == "replay_pipeline/batch_h2d_submit"
        )
        assert submit["args"]["h2d_submitter"] == "torch_copy"
        assert submit["args"]["pinned_memory"] is False
        assert submit["args"]["direct_pinned_shared"] is False
        pipeline.close()

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
    def test_mps_portable_path_samples_on_mps(self):
        rb = _make_cpu_replay(capacity=128, obs_dim=4, action_dim=2)
        pipeline = self._make_pipeline(rb, sample_count=8, base_seed=5, device="mps")

        batch = _sample_ready(pipeline, tick_id=1, sample_count=8)

        assert all(value.device.type == "mps" for value in batch.values())
        assert pipeline._host_pinned is False
        assert pipeline._h2d_submitter == "torch_copy"
        pipeline.close()


# ---------------------------------------------------------------------------
# CPUPinnedDoubleBufferReplayPipeline (experimental path)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
class TestCPUPinnedDoubleBuffer:
    @staticmethod
    def _make_cuda_replay(
        capacity: int = 128,
        obs_dim: int = 4,
        action_dim: int = 2,
        critic_dim: int = 0,
        defer_gpu: bool = True,
        packed_cpu_storage: bool = True,
    ) -> ReplayBuffer:
        rb = ReplayBuffer(
            capacity=capacity,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device="cuda",
            critic_dim=critic_dim,
            defer_gpu=defer_gpu,
            packed_cpu_storage=packed_cpu_storage,
        )
        n = min(64, capacity)
        rb.add(
            obs=torch.randn(n, obs_dim),
            actions=torch.randn(n, action_dim),
            rewards=torch.randn(n),
            next_obs=torch.randn(n, obs_dim),
            dones=torch.zeros(n),
            truncated=torch.zeros(n),
            critic=torch.randn(n, critic_dim) if critic_dim > 0 else None,
            next_critic=torch.randn(n, critic_dim) if critic_dim > 0 else None,
        )
        return rb

    def _make_pipeline(self, rb, sample_count=16, base_seed=42):
        from unilab.ipc.replay_pipelines.cpu_pinned_double_buffer import (
            CPUPinnedDoubleBufferReplayPipeline,
        )

        shared_slots = [
            torch.empty((sample_count, rb._storage.shape[1]), dtype=torch.float32).share_memory_()
            for _ in range(2)
        ]
        return CPUPinnedDoubleBufferReplayPipeline(
            rb,
            device="cuda",
            sample_count=sample_count,
            base_seed=base_seed,
            collector_pack_request_queue=queue.Queue(),
            collector_pack_ready_queue=queue.Queue(),
            collector_pack_shared_slots=shared_slots,
        )

    def _make_packed_pipeline(self, rb, sample_count=16, base_seed=42):
        return self._make_pipeline(rb, sample_count=sample_count, base_seed=base_seed)

    @staticmethod
    def _service_collector_pack(pipeline) -> None:
        _service_collector_pack(pipeline)

    @staticmethod
    def _sample_ready(pipeline, tick_id: int, sample_count: int):
        return _sample_ready(pipeline, tick_id=tick_id, sample_count=sample_count)

    def test_allocates_two_host_and_two_gpu_slots(self):
        rb = self._make_cuda_replay()
        pipeline = self._make_pipeline(rb, sample_count=8)
        assert len(pipeline._host_packed) == 0
        assert len(pipeline._gpu_packed) == 2
        assert len(pipeline._collector_pack_shared_slots) == 2
        assert all(slot.is_pinned() for slot in pipeline._collector_pack_shared_slots)
        for slot in pipeline._gpu_packed:
            assert slot.is_cuda

    def test_deterministic_seed_produces_same_batch(self):
        rb = self._make_cuda_replay(capacity=128, obs_dim=4, action_dim=2)
        p1 = self._make_pipeline(rb, sample_count=16, base_seed=99)
        p2 = self._make_pipeline(rb, sample_count=16, base_seed=99)
        b1 = self._sample_ready(p1, tick_id=5, sample_count=16)
        b2 = self._sample_ready(p2, tick_id=5, sample_count=16)
        torch.testing.assert_close(b1["obs"], b2["obs"])
        torch.testing.assert_close(b1["actions"], b2["actions"])
        torch.testing.assert_close(b1["rewards"], b2["rewards"])

    def test_start_prepare_reuses_prepared_batch(self):
        rb = self._make_cuda_replay(capacity=128, obs_dim=4, action_dim=2)
        pipeline = self._make_pipeline(rb, sample_count=8, base_seed=11)
        assert pipeline.start_prepare(tick_id=3, sample_count=8)
        assert not pipeline.start_prepare(tick_id=3, sample_count=8)
        self._service_collector_pack(pipeline)
        deadline = time.time() + 2.0
        while time.time() < deadline and not pipeline.batch_ready(tick_id=3, sample_count=8):
            time.sleep(0.01)
        assert pipeline.batch_ready(tick_id=3, sample_count=8)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(11 + 3)
        expected_indices = torch.randint(0, int(rb.size[0]), (8,), generator=gen)
        batch = pipeline.sample_large_batch(tick_id=3, sample_count=8)
        torch.testing.assert_close(batch["obs"], rb._storage[expected_indices, rb._obs_sl].cuda())

    def test_sample_count_mismatch_is_rejected(self):
        rb = self._make_cuda_replay()
        pipeline = self._make_pipeline(rb, sample_count=8)
        with pytest.raises(ValueError, match="sample_count must match"):
            pipeline.start_prepare(tick_id=0, sample_count=4)
        with pytest.raises(ValueError, match="sample_count must match"):
            pipeline.sample_large_batch(tick_id=0, sample_count=4)

    def test_sample_waits_until_ready_batch(self):
        rb = self._make_cuda_replay()
        pipeline = self._make_pipeline(rb, sample_count=8)
        assert not pipeline.batch_ready(tick_id=0, sample_count=8)
        batch = self._sample_ready(pipeline, tick_id=0, sample_count=8)
        assert batch["obs"].is_cuda
        assert pipeline.batch_ready(tick_id=0, sample_count=8)

    def test_sampled_batch_matches_replay_rows(self):
        obs_dim, action_dim = 4, 2
        rb = self._make_cuda_replay(capacity=128, obs_dim=obs_dim, action_dim=action_dim)
        sample_count = 8
        pipeline = self._make_pipeline(rb, sample_count=sample_count, base_seed=0)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(0 + 1)
        expected_indices = torch.randint(0, int(rb.size[0]), (sample_count,), generator=gen)

        batch = self._sample_ready(pipeline, tick_id=1, sample_count=sample_count)
        torch.testing.assert_close(batch["obs"], rb._storage[expected_indices, rb._obs_sl].cuda())
        torch.testing.assert_close(
            batch["actions"], rb._storage[expected_indices, rb._act_sl].cuda()
        )
        torch.testing.assert_close(
            batch["rewards"], rb._storage[expected_indices, rb._rew_col].cuda()
        )
        torch.testing.assert_close(
            batch["next_obs"], rb._storage[expected_indices, rb._nobs_sl].cuda()
        )
        torch.testing.assert_close(
            batch["dones"], rb._storage[expected_indices, rb._done_col].cuda()
        )
        torch.testing.assert_close(
            batch["truncated"], rb._storage[expected_indices, rb._trunc_col].cuda()
        )

    def test_critic_fields_present_when_critic_dim_nonzero(self):
        rb = self._make_cuda_replay(capacity=128, obs_dim=4, action_dim=2, critic_dim=6)
        pipeline = self._make_pipeline(rb, sample_count=8, base_seed=7)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(7)
        expected_indices = torch.randint(0, int(rb.size[0]), (8,), generator=gen)

        batch = self._sample_ready(pipeline, tick_id=0, sample_count=8)
        assert "critic" in batch
        assert "next_critic" in batch
        assert batch["critic"].shape == (8, 6)
        torch.testing.assert_close(
            batch["critic"], rb._storage[expected_indices, rb._critic_sl].cuda()
        )
        torch.testing.assert_close(
            batch["next_critic"], rb._storage[expected_indices, rb._ncritic_sl].cuda()
        )

    def test_hot_cold_swap_after_tick(self):
        rb = self._make_cuda_replay()
        pipeline = self._make_pipeline(rb, sample_count=8)
        self._sample_ready(pipeline, tick_id=0, sample_count=8)
        hot_before = pipeline._hot
        pipeline.after_tick()
        assert not pipeline._has_hot_batch
        self._sample_ready(pipeline, tick_id=1, sample_count=8)
        assert pipeline._hot != hot_before

    def test_close_releases_buffers(self):
        rb = self._make_cuda_replay()
        pipeline = self._make_pipeline(rb, sample_count=8)
        pipeline.close()
        assert len(pipeline._host) == 0
        assert len(pipeline._gpu) == 0
        assert len(pipeline._gpu_packed) == 0

    def test_ring_buffer_wraparound_sampling(self):
        obs_dim, action_dim = 4, 2
        capacity = 32
        rb = self._make_cuda_replay(capacity=capacity, obs_dim=obs_dim, action_dim=action_dim)
        for _ in range(3):
            rb.add(
                obs=torch.randn(capacity, obs_dim),
                actions=torch.randn(capacity, action_dim),
                rewards=torch.randn(capacity),
                next_obs=torch.randn(capacity, obs_dim),
                dones=torch.zeros(capacity),
                truncated=torch.zeros(capacity),
            )
        assert int(rb.ptr[0]) > capacity
        pipeline = self._make_pipeline(rb, sample_count=16)
        batch = self._sample_ready(pipeline, tick_id=0, sample_count=16)
        assert batch["obs"].shape == (16, obs_dim)

    def test_defer_gpu_replay_buffer_does_not_allocate_gpu_cache(self):
        rb = self._make_cuda_replay(defer_gpu=True)
        assert not hasattr(rb, "obs_gpu")
        pipeline = self._make_pipeline(rb, sample_count=8)
        batch = self._sample_ready(pipeline, tick_id=0, sample_count=8)
        assert batch["obs"].is_cuda
        assert not hasattr(rb, "obs_gpu")

    def test_packed_layout_samples_match_replay_rows(self):
        rb = self._make_cuda_replay(
            capacity=128,
            obs_dim=4,
            action_dim=2,
            critic_dim=6,
            defer_gpu=True,
            packed_cpu_storage=True,
        )
        pipeline = self._make_packed_pipeline(rb, sample_count=8, base_seed=13)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(13 + 5)
        expected_indices = torch.randint(0, int(rb.size[0]), (8,), generator=gen)

        batch = self._sample_ready(pipeline, tick_id=5, sample_count=8)

        torch.testing.assert_close(batch["obs"], rb._storage[expected_indices, rb._obs_sl].cuda())
        torch.testing.assert_close(
            batch["next_obs"], rb._storage[expected_indices, rb._nobs_sl].cuda()
        )
        torch.testing.assert_close(
            batch["actions"], rb._storage[expected_indices, rb._act_sl].cuda()
        )
        torch.testing.assert_close(
            batch["rewards"], rb._storage[expected_indices, rb._rew_col].cuda()
        )
        torch.testing.assert_close(
            batch["dones"], rb._storage[expected_indices, rb._done_col].cuda()
        )
        torch.testing.assert_close(
            batch["truncated"], rb._storage[expected_indices, rb._trunc_col].cuda()
        )
        torch.testing.assert_close(
            batch["critic"], rb._storage[expected_indices, rb._critic_sl].cuda()
        )
        torch.testing.assert_close(
            batch["next_critic"], rb._storage[expected_indices, rb._ncritic_sl].cuda()
        )

    def test_packed_layout_requires_packed_replay_storage(self):
        rb = ReplayBuffer(
            capacity=128,
            obs_dim=4,
            action_dim=2,
            device="cuda",
            defer_gpu=True,
            packed_cpu_storage=False,
        )
        from unilab.ipc.replay_pipelines.cpu_pinned_double_buffer import (
            CPUPinnedDoubleBufferReplayPipeline,
        )

        with pytest.raises(ValueError, match="requires ReplayBuffer"):
            CPUPinnedDoubleBufferReplayPipeline(
                rb,
                device="cuda",
                sample_count=8,
                collector_pack_request_queue=queue.Queue(),
                collector_pack_ready_queue=queue.Queue(),
                collector_pack_shared_slots=[
                    torch.empty((8, 4 + 4 + 2 + 3), dtype=torch.float32).share_memory_()
                    for _ in range(2)
                ],
            )

    def test_trace_events_include_gpu_span_and_swap_metadata(self):
        rb = self._make_cuda_replay(defer_gpu=True)
        trace = _RecordingTrace()
        pipeline = self._make_pipeline(rb, sample_count=8, base_seed=3)
        pipeline._trace_recorder = trace

        self._sample_ready(pipeline, tick_id=2, sample_count=8)

        names = {event["name"] for event in trace.slices}
        cuda_names = {event["name"] for event in trace.cuda_spans}
        assert "replay_pipeline/collector_pack_request" in names
        assert "replay_pipeline/batch_h2d_submit" in names
        assert "replay_pipeline/batch_h2d_wait" in names
        assert "replay_pipeline/gpu_wait_for_batch" in names
        assert "replay_pipeline/hot_cold_swap" in names
        assert "gpu/replay_pipeline_batch_h2d" in cuda_names
        submit = next(
            event for event in trace.slices if event["name"] == "replay_pipeline/batch_h2d_submit"
        )
        assert submit["args"]["tick_id"] == 2
        assert submit["args"]["pack_layout"] == "packed"
        assert submit["args"]["pack_executor"] == "collector_thread"
        assert submit["args"]["h2d_submitter"] == "pybind11"
        assert submit["args"]["direct_pinned_shared"] is True
        assert submit["args"]["h2d_bytes"] > 0

    def test_sample_large_batch_correct_device(self):
        rb = self._make_cuda_replay()
        pipeline = self._make_pipeline(rb, sample_count=8)
        batch = self._sample_ready(pipeline, tick_id=0, sample_count=8)
        for v in batch.values():
            assert v.is_cuda

    def test_hot_batch_tick_mismatch_batch_ready_returns_false(self):
        rb = self._make_cuda_replay()
        pipeline = self._make_pipeline(rb, sample_count=8)
        self._sample_ready(pipeline, tick_id=0, sample_count=8)
        assert pipeline._has_hot_batch
        assert not pipeline.batch_ready(tick_id=1, sample_count=8)

    def test_hot_batch_tick_mismatch_sample_large_batch_raises(self):
        rb = self._make_cuda_replay()
        pipeline = self._make_pipeline(rb, sample_count=8)
        self._sample_ready(pipeline, tick_id=0, sample_count=8)
        assert pipeline._has_hot_batch
        with pytest.raises(RuntimeError, match="Hot batch tick 0 does not match requested tick 1"):
            pipeline.sample_large_batch(tick_id=1, sample_count=8)

    def test_trace_cuda_events_false_skips_h2d_cuda_pending_span(self):
        rb = self._make_cuda_replay(defer_gpu=True)
        trace = _RecordingTrace()
        pipeline = self._make_pipeline(rb, sample_count=8, base_seed=3)
        pipeline._trace_recorder = trace
        pipeline._trace_cuda_events = False
        self._sample_ready(pipeline, tick_id=2, sample_count=8)
        cuda_names = {event["name"] for event in trace.cuda_spans}
        assert "gpu/replay_pipeline_batch_h2d" not in cuda_names
        # non-CUDA slices unaffected
        slice_names = {event["name"] for event in trace.slices}
        assert "replay_pipeline/batch_h2d_submit" in slice_names

    def test_h2d_cuda_span_has_full_tick_metadata(self):
        rb = self._make_cuda_replay(defer_gpu=True)
        trace = _RecordingTrace()
        pipeline = self._make_pipeline(rb, sample_count=8, base_seed=3)
        pipeline._trace_recorder = trace
        self._sample_ready(pipeline, tick_id=4, sample_count=8)
        h2d_spans = [e for e in trace.cuda_spans if e["name"] == "gpu/replay_pipeline_batch_h2d"]
        assert h2d_spans, "expected gpu/replay_pipeline_batch_h2d cuda pending span"
        args = h2d_spans[0]["args"]
        for key in (
            "tick_id",
            "snapshot_ptr",
            "snapshot_size",
            "sample_seed",
            "sample_count",
            "batch_host_slot",
            "batch_gpu_slot",
            "pinned_memory",
            "slot",
            "h2d_bytes",
        ):
            assert key in args, f"missing {key} in h2d args"
        assert args["tick_id"] == 4
        assert args["sample_seed"] == 3 + 4
        assert args["sample_count"] == 8
        assert args["pinned_memory"] is True
        assert args["h2d_submitter"] == "pybind11"
        assert args["h2d_bytes"] > 0

    def test_fixed_pybind11_h2d_submitter_samples_expected_batch(self):
        rb = self._make_cuda_replay(defer_gpu=True, packed_cpu_storage=True)
        pipeline = self._make_pipeline(rb, sample_count=8, base_seed=13)
        batch = self._sample_ready(pipeline, tick_id=2, sample_count=8)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(13 + 2)
        expected_indices = torch.randint(0, int(rb.size[0]), (8,), generator=gen)
        torch.testing.assert_close(batch["obs"], rb._storage[expected_indices, rb._obs_sl].cuda())
        pipeline.close()

    def test_old_b_path_h2d_submitter_argument_is_removed(self):
        rb = self._make_cuda_replay(defer_gpu=True, packed_cpu_storage=True)
        with pytest.raises(TypeError, match="h2d_submitter"):
            from unilab.ipc.replay_pipelines.cpu_pinned_double_buffer import (
                CPUPinnedDoubleBufferReplayPipeline,
            )

            CPUPinnedDoubleBufferReplayPipeline(
                rb,
                device="cuda",
                sample_count=8,
                h2d_submitter="python",
                collector_pack_request_queue=queue.Queue(),
                collector_pack_ready_queue=queue.Queue(),
                collector_pack_shared_slots=[
                    torch.empty((8, rb._storage.shape[1]), dtype=torch.float32).share_memory_()
                    for _ in range(2)
                ],
            )

    def test_collector_thread_registers_shared_slots_as_pinned(self):
        import queue

        rb = self._make_cuda_replay(defer_gpu=True, packed_cpu_storage=True)
        sample_count = 8
        shared_slots = [
            torch.empty((sample_count, rb._storage.shape[1]), dtype=torch.float32).share_memory_()
            for _ in range(2)
        ]
        assert not any(slot.is_pinned() for slot in shared_slots)
        from unilab.ipc.replay_pipelines.cpu_pinned_double_buffer import (
            CPUPinnedDoubleBufferReplayPipeline,
        )

        pipeline = CPUPinnedDoubleBufferReplayPipeline(
            rb,
            device="cuda",
            sample_count=sample_count,
            collector_pack_request_queue=queue.Queue(),
            collector_pack_ready_queue=queue.Queue(),
            collector_pack_shared_slots=shared_slots,
        )
        assert len(pipeline._host_packed) == 0
        assert all(slot.is_pinned() for slot in shared_slots)
        assert pipeline._packed_h2d_source(0) is shared_slots[0]
        pipeline.close()
        assert not any(slot.is_pinned() for slot in shared_slots)

    def test_verbose_true_writes_no_learner_pack_csv_for_collector_thread(self, tmp_path):
        rb = self._make_cuda_replay()
        pipeline = self._make_pipeline(rb, sample_count=8, base_seed=0)
        pipeline._verbose = True
        pipeline._verbose_output_dir = str(tmp_path)
        pipeline._verbose_pack_records = []
        self._sample_ready(pipeline, tick_id=0, sample_count=8)
        pipeline.close()
        csv_path = tmp_path / "verbose" / "pack_fields.csv"
        assert not csv_path.exists()
        assert pipeline._verbose_pack_records == []

    def test_verbose_false_writes_no_files(self, tmp_path):
        rb = self._make_cuda_replay()
        pipeline = self._make_pipeline(rb, sample_count=8, base_seed=0)
        pipeline._verbose_output_dir = str(tmp_path)
        self._sample_ready(pipeline, tick_id=0, sample_count=8)
        pipeline.close()
        assert not (tmp_path / "verbose").exists()
        assert pipeline._verbose_pack_records is None
