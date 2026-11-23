"""Tests for DeviceReplayPipeline."""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
from collections import deque
from threading import Condition, Lock

import pytest
import torch

from unilab.rl.runtime.ipc.completion import COMPLETION_CONTRACT_VERSION, ImmediateCompletion
from unilab.rl.runtime.ipc.replay_buffer import ReplayBuffer
from unilab.rl.runtime.ipc.replay_pipelines.device import (
    DeviceReplayPipeline,
    _device_memory_budget,
    _ring_spans,
    _validate_device_memory_budget,
    require_offpolicy_replay_device,
)

_HAS_CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
_HAS_MPS = torch.backends.mps.is_available()
mps_only = pytest.mark.skipif(not _HAS_MPS, reason="MPS required")


class _FakeMPSEvent:
    def __init__(self):
        self.recorded = False
        self.complete = False

    def record(self):
        self.recorded = True

    def query(self):
        return self.complete

    def synchronize(self):
        self.complete = True


_OBS_DIM = 4
_ACTION_DIM = 2
_CRITIC_DIM = 5


def _make_replay(
    capacity: int = 128,
    obs_dim: int = _OBS_DIM,
    action_dim: int = _ACTION_DIM,
    critic_dim: int = _CRITIC_DIM,
    device: str = "cuda",
    slot_rows: int | None = None,
) -> ReplayBuffer:
    return ReplayBuffer(
        capacity=capacity,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        critic_dim=critic_dim,
        ingress_slot_rows=capacity if slot_rows is None else slot_rows,
    )


def _make_bounded_replay(
    *,
    capacity: int = 128,
    slot_rows: int = 16,
    device: str = "cuda",
) -> ReplayBuffer:
    return _make_replay(
        capacity=capacity,
        device=device,
        slot_rows=slot_rows,
    )


def _pattern_add(rb: ReplayBuffer, start_row: int, n: int) -> None:
    """Add rows whose fields are exact float32 functions of the absolute row id."""
    obs_dim, action_dim, critic_dim = rb.obs_dim, rb.action_dim, rb.critic_dim
    rows = torch.arange(start_row, start_row + n, dtype=torch.float32)
    col = rows.unsqueeze(1)
    critic = next_critic = None
    if critic_dim > 0:
        critic = col * 10000 + torch.arange(critic_dim, dtype=torch.float32)
        next_critic = col * 100000 + torch.arange(critic_dim, dtype=torch.float32)
    rb.add(
        obs=col * 10 + torch.arange(obs_dim, dtype=torch.float32),
        actions=col * 1000 + torch.arange(action_dim, dtype=torch.float32),
        rewards=rows.clone(),
        next_obs=col * 100 + torch.arange(obs_dim, dtype=torch.float32),
        dones=torch.zeros(n),
        truncated=torch.ones(n),
        critic=critic,
        next_critic=next_critic,
    )


def _pattern_add_chunks(rb: ReplayBuffer, chunk_rows: int, chunks: int) -> None:
    for chunk in range(chunks):
        _pattern_add(rb, chunk * chunk_rows, chunk_rows)


def _pattern_add_vector(rb: ReplayBuffer, rows_n: int) -> None:
    rows = torch.arange(rows_n, dtype=torch.float32)
    col = rows.unsqueeze(1)
    critic = next_critic = None
    if rb.critic_dim > 0:
        critic = col * 10000 + torch.arange(rb.critic_dim, dtype=torch.float32)
        next_critic = col * 100000 + torch.arange(rb.critic_dim, dtype=torch.float32)
    assert (
        rb.add_batch(
            obs=col * 10 + torch.arange(rb.obs_dim, dtype=torch.float32),
            actions=col * 1000 + torch.arange(rb.action_dim, dtype=torch.float32),
            rewards=rows,
            next_obs=col * 100 + torch.arange(rb.obs_dim, dtype=torch.float32),
            dones=torch.zeros(rows_n),
            truncated=torch.ones(rows_n),
            critic=critic,
            next_critic=next_critic,
        )
        is True
    )


def _expected_pattern(rb: ReplayBuffer, rewards: torch.Tensor) -> dict[str, torch.Tensor]:
    col = rewards.cpu().unsqueeze(1)
    out = {
        "obs": col * 10 + torch.arange(rb.obs_dim, dtype=torch.float32),
        "next_obs": col * 100 + torch.arange(rb.obs_dim, dtype=torch.float32),
        "actions": col * 1000 + torch.arange(rb.action_dim, dtype=torch.float32),
    }
    if rb.critic_dim > 0:
        out["critic"] = col * 10000 + torch.arange(rb.critic_dim, dtype=torch.float32)
        out["next_critic"] = col * 100000 + torch.arange(rb.critic_dim, dtype=torch.float32)
    return out


def _wait_visible(pipeline: DeviceReplayPipeline, ptr: int, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while pipeline._visible_ptr < ptr:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Device replay stalled: visible_ptr={pipeline._visible_ptr} < {ptr}"
            )
        time.sleep(0.005)


def _wait_batch_ready(
    pipeline: DeviceReplayPipeline,
    tick_id: int,
    sample_count: int,
    timeout: float = 15.0,
) -> None:
    deadline = time.monotonic() + timeout
    while not pipeline.batch_ready(tick_id, sample_count):
        if time.monotonic() > deadline:
            raise TimeoutError(f"GPU replay batch for tick {tick_id} never became ready")
        time.sleep(0.01)


class TestRingSpans:
    def test_no_wrap(self):
        assert _ring_spans(5, 12, 64) == [(5, 7)]

    def test_wrap_split(self):
        assert _ring_spans(60, 68, 64) == [(60, 4), (0, 4)]

    def test_multiple_wraps(self):
        assert _ring_spans(62, 136, 64) == [(62, 2), (0, 64), (0, 8)]

    def test_full_ring(self):
        assert _ring_spans(3, 67, 64) == [(3, 61), (0, 3)]

    def test_empty(self):
        assert _ring_spans(5, 5, 64) == []
        assert _ring_spans(5, 3, 64) == []


class TestMPSIngressEventLifecycle:
    def test_ring_wrap_span_copy_is_nonblocking_and_event_gated(self, monkeypatch):
        pipeline = object.__new__(DeviceReplayPipeline)
        pipeline._device = torch.device("mps")
        pipeline._capacity = 8
        pipeline._gpu_storage = torch.zeros(8, 2)
        pipeline._gpu_scalars = None
        pipeline._storage_dtype = torch.float32
        pipeline._submitted_ptr = 6
        pipeline._visible_ptr = 6
        pipeline._span_events = deque()
        pipeline._trace_recorder = None
        pipeline._trace_cuda_events = False
        pipeline._submission_lock = Lock()
        pipeline._prepare_condition = Condition()
        commits: list[tuple[int, int, int]] = []

        class Replay:
            def commit_ingress(self, *, slot, start, count):
                commits.append((slot, start, count))

        pipeline._replay_buffer = Replay()
        monkeypatch.setattr(torch, "mps", type("FakeMPS", (), {"Event": _FakeMPSEvent})())
        original_copy = torch.Tensor.copy_
        non_blocking_calls: list[bool] = []

        def recording_copy(self, source, *, non_blocking: bool = False):
            non_blocking_calls.append(non_blocking)
            return original_copy(self, source, non_blocking=non_blocking)

        monkeypatch.setattr(torch.Tensor, "copy_", recording_copy)
        pipeline._submit_span_copy(
            start=6,
            end=10,
            source=torch.ones(4, 2),
            ingress_slot=0,
        )
        event = pipeline._span_events[0][1]

        assert non_blocking_calls == [True, True]
        assert event.recorded is True
        assert pipeline._submitted_ptr == 10
        assert pipeline._visible_ptr == 6
        assert pipeline._drain_completed_spans() is False
        assert commits == []

        event.complete = True
        assert pipeline._drain_completed_spans() is True
        assert commits == [(0, 6, 4)]
        assert pipeline._visible_ptr == 10

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_ingress_copy_waits_public_completion_before_read(self):
        pipeline = object.__new__(DeviceReplayPipeline)
        pipeline._device = torch.device("cuda")
        pipeline._capacity = 4
        pipeline._gpu_storage = torch.zeros(4, 2, device="cuda")
        pipeline._gpu_scalars = None
        pipeline._storage_dtype = torch.float32
        pipeline._submitted_ptr = 0
        pipeline._visible_ptr = 0
        pipeline._span_events = deque()
        pipeline._trace_recorder = None
        pipeline._trace_cuda_events = False
        pipeline._submission_lock = Lock()
        pipeline._sync_stream = torch.cuda.Stream()
        waited: list[object] = []

        class Completion(ImmediateCompletion):
            def wait_before_read(self, stream=None):
                waited.append(stream)

            producer_id = "producer-a"

        class Replay:
            last_completion = Completion(operation="step", sequence=1, producer_id="producer-a")
            completion_sequence = 1
            completion_producer_id = "producer-a"

        pipeline._replay_buffer = Replay()

        pipeline._submit_span_copy(
            start=0,
            end=4,
            source=torch.ones(4, 2, device="cuda"),
            ingress_slot=0,
        )

        assert waited == [pipeline._sync_stream]
        assert pipeline._span_events

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_ingress_copy_rejects_removed_env_update_operation(self):
        pipeline = object.__new__(DeviceReplayPipeline)
        pipeline._device = torch.device("cuda")
        pipeline._capacity = 4
        pipeline._gpu_storage = torch.zeros(4, 2, device="cuda")
        pipeline._gpu_scalars = None
        pipeline._storage_dtype = torch.float32
        pipeline._submitted_ptr = 0
        pipeline._visible_ptr = 0
        pipeline._span_events = deque()
        pipeline._trace_recorder = None
        pipeline._trace_cuda_events = False
        pipeline._submission_lock = Lock()
        pipeline._sync_stream = torch.cuda.Stream()

        class InvalidCompletion:
            contract_version = COMPLETION_CONTRACT_VERSION
            operation = "env_update"
            sequence = 1
            producer_id = "producer-update"

            def wait_before_read(self, stream=None):
                del stream

        class Replay:
            last_completion = InvalidCompletion()
            completion_sequence = 1
            completion_producer_id = "producer-update"

        pipeline._replay_buffer = Replay()

        with pytest.raises(ValueError, match="unsupported completion operation"):
            pipeline._submit_span_copy(
                start=0,
                end=4,
                source=torch.ones(4, 2, device="cuda"),
                ingress_slot=0,
            )

    def test_multiple_pending_spans_commit_in_order_and_release_slots(self):
        pipeline = object.__new__(DeviceReplayPipeline)
        first_event = _FakeMPSEvent()
        second_event = _FakeMPSEvent()
        commits: list[tuple[int, int, int]] = []

        class Replay:
            def commit_ingress(self, *, slot, start, count):
                commits.append((slot, start, count))

        pipeline._span_events = deque(
            [
                (4, first_event, 0, 0, 4),
                (8, second_event, 1, 4, 4),
            ]
        )
        pipeline._trace_recorder = None
        pipeline._visible_ptr = 0
        pipeline._submission_lock = Lock()
        pipeline._prepare_condition = Condition()
        pipeline._replay_buffer = Replay()

        first_event.complete = True
        assert pipeline._drain_completed_spans() is True
        assert commits == [(0, 0, 4)]
        assert pipeline._visible_ptr == 4

        second_event.complete = True
        assert pipeline._drain_completed_spans() is True
        assert commits == [(0, 0, 4), (1, 4, 4)]
        assert pipeline._visible_ptr == 8

    def test_close_drains_pending_span_before_transfer_backend_close(self, monkeypatch):
        pipeline = object.__new__(DeviceReplayPipeline)
        event = _FakeMPSEvent()
        close_order: list[str] = []
        commits: list[tuple[int, int, int]] = []

        class Replay:
            def commit_ingress(self, *, slot, start, count):
                close_order.append("commit")
                commits.append((slot, start, count))

        class Transfer:
            def close(self):
                close_order.append("transfer_close")

        pipeline._closed = False
        pipeline._sync_thread = None
        pipeline._submission_lock = Lock()
        pipeline._prepare_condition = Condition()
        pipeline._span_events = deque([(4, event, 1, 0, 4)])
        pipeline._slot_events = [_FakeMPSEvent()]
        pipeline._replay_buffer = Replay()
        pipeline._transfer_backend = Transfer()
        pipeline._gpu_packed = []
        pipeline._gpu_storage = torch.zeros(4, 2)
        pipeline._gpu_scalars = None
        pipeline._gather_scratch = []
        pipeline._scalar_scratch = []
        pipeline._host_pinned = False
        pipeline._submit_new_spans = lambda: False

        pipeline.close()

        assert event.complete is True
        assert close_order == ["commit", "transfer_close"]
        assert commits == [(1, 0, 4)]

        assert _ring_spans(0, 4, 0) == []


class TestConstructionGuards:
    def test_non_accelerator_device_rejected(self):
        rb = _make_replay(device="cpu")
        with pytest.raises(ValueError, match="have Torch type 'cuda' or 'mps'"):
            DeviceReplayPipeline(rb, device="cpu", sample_count=8)

    def test_runner_rejects_non_accelerator_before_base_initialization(self):
        from unilab.rl.runtime.offpolicy.double_buffer_runner import (
            DoubleBufferOffPolicyRunner,
        )

        with pytest.raises(ValueError, match="have Torch type 'cuda' or 'mps'"):
            DoubleBufferOffPolicyRunner(device="cpu")

    @pytest.mark.parametrize("device", ["cuda:0", "cuda:7", "mps", "mps:0"])
    def test_supported_learner_device_types_are_returned_verbatim(self, device: str):
        assert require_offpolicy_replay_device(device) == device

    def test_direct_builders_reject_host_replay_before_env_work(self):
        import unilab.rl.algos.flash_sac.double_buffer as flashsac_builder
        import unilab.rl.algos.sac.double_buffer as sac_builder

        builders = (
            sac_builder.build_double_buffer_runner,
            flashsac_builder.build_double_buffer_runner,
        )
        for builder in builders:
            with pytest.raises(
                ValueError,
                match="replay device to equal the learner device",
            ):
                builder(
                    object(),  # rejected before the owner DictConfig is read
                    env_factory=lambda *_args, **_kwargs: None,
                    env_cfg_override=None,
                    replay_prefetch_mode="one_tick",
                    device="cpu",
                )

    def test_mps_memory_budget_uses_recommended_budget(self, monkeypatch):
        monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 1_000)
        monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 250)

        assert _device_memory_budget(torch.device("mps")) == (750, 1_000)

    def test_mps_memory_budget_failure_is_actionable(self, monkeypatch):
        monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 1_000)
        monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 250)

        with pytest.raises(RuntimeError, match=r"requires .* available budget .* total budget"):
            _validate_device_memory_budget(
                torch.device("mps"),
                required_bytes=601,
                storage_bytes=500,
                batch_bytes=101,
                headroom=0.8,
            )


@cuda_only
class TestDevicePipeline:
    @pytest.fixture
    def pipeline_factory(self):
        created = []

        def _make(rb, **kwargs):
            kwargs.setdefault("device", "cuda")
            kwargs.setdefault("sample_count", 16)
            pipeline = DeviceReplayPipeline(rb, **kwargs)
            created.append(pipeline)
            return pipeline

        yield _make
        for pipeline in created:
            pipeline.close()

    def test_allocates_gpu_storage_and_slots(self, pipeline_factory):
        rb = _make_replay(capacity=64)
        pipeline = pipeline_factory(rb, sample_count=8)
        assert pipeline._gpu_storage.shape == (64, rb.storage_width)
        assert pipeline._gpu_storage.is_cuda
        assert len(pipeline._gpu_packed) == 2
        assert all(slot.is_cuda for slot in pipeline._gpu_packed)
        assert all(slot.is_pinned() for slot in rb._ingress_slots)
        assert pipeline.h2d_submitter == "device_ingress"
        manifest = pipeline.transfer_manifest
        assert manifest["pipeline"] == "device"
        assert manifest["storage_rows"] == 64
        assert manifest["host_pinned"] is True

    def test_bounded_ingress_owns_no_full_host_ring_and_commits_after_h2d(self, pipeline_factory):
        rb = _make_bounded_replay(capacity=64, slot_rows=16)
        _pattern_add(rb, 0, 16)
        assert not hasattr(rb, "_storage")
        assert int(rb.ptr[0]) == 0
        assert rb.published_ptr == 16

        pipeline = pipeline_factory(rb, sample_count=8)
        _wait_visible(pipeline, 16)

        assert int(rb.ptr[0]) == 16
        assert int(rb.size[0]) == 16
        assert pipeline._gpu_storage.shape == (64, rb.storage_width)
        assert pipeline.transfer_manifest["storage_owner"] == "device"
        assert pipeline.transfer_manifest["host_storage_bytes"] == rb.host_storage_bytes
        assert pipeline.transfer_manifest["ingress_depth"] == 2
        assert pipeline.transfer_manifest["ingress_slot_rows"] == 16
        diagnostics = pipeline.transfer_manifest["ingress_diagnostics"]
        assert diagnostics["ingress_depth"] == 2
        assert diagnostics["ingress_slot_rows"] == 16
        assert diagnostics["published_sequence"] == 1
        assert diagnostics["release_sequence"] == 1
        assert diagnostics["occupancy"] == 0
        assert diagnostics["high_water_occupancy"] == 1

    def test_bounded_ingress_ring_wrap_and_committed_field_order(self, pipeline_factory):
        rb = _make_bounded_replay(capacity=32, slot_rows=8)
        _pattern_add(rb, 0, 8)
        _pattern_add(rb, 8, 8)
        pipeline = pipeline_factory(rb, sample_count=8)
        _wait_visible(pipeline, 16)
        for start in (16, 24, 32):
            _pattern_add(rb, start, 8)
        _wait_visible(pipeline, 40)

        end_ptr, fields = pipeline.read_committed_fields(
            ("rewards", "dones"),
            start_ptr=0,
        )

        assert end_ptr == 40
        torch.testing.assert_close(
            fields["rewards"].cpu(),
            torch.arange(8, 40, dtype=torch.float32),
        )
        assert (fields["dones"].cpu() == 0).all()

    def test_bounded_ingress_consumes_chunked_collector_vector(self, pipeline_factory):
        rb = _make_bounded_replay(capacity=64, slot_rows=6)
        pipeline = pipeline_factory(rb, sample_count=8)
        _pattern_add_vector(rb, 15)
        _wait_visible(pipeline, 15)

        assert rb.published_ptr == 15
        assert int(rb.ptr[0]) == 15
        diagnostics = rb.ingress_diagnostics()
        assert diagnostics["published_sequence"] == 3
        assert diagnostics["release_sequence"] == 3
        _, fields = pipeline.read_committed_fields(("rewards",), start_ptr=0)
        torch.testing.assert_close(
            fields["rewards"].cpu(),
            torch.arange(15, dtype=torch.float32),
        )

    @cuda_only
    def test_background_and_learner_progress_own_ingress_without_token_race(self, pipeline_factory):
        rb = _make_bounded_replay(capacity=64, slot_rows=8)
        pipeline = pipeline_factory(rb, sample_count=8)

        _pattern_add(rb, 0, 8)
        _pattern_add(rb, 8, 8)
        deadline = time.monotonic() + 10.0
        while not pipeline.progress() and time.monotonic() < deadline:
            time.sleep(0.005)
        # A learner wait must observe daemon-owned work without consuming any
        # ingress token itself.
        pipeline.progress(wait=True)
        _wait_visible(pipeline, 16)

        diagnostics = rb.ingress_diagnostics()
        assert diagnostics["published_sequence"] == 2
        assert diagnostics["release_sequence"] == 2
        assert diagnostics["occupancy"] == 0
        assert pipeline._submitted_ptr == 16

    def test_bounded_ingress_samples_only_committed_rows(self, pipeline_factory):
        rb = _make_bounded_replay(capacity=64, slot_rows=16)
        _pattern_add(rb, 0, 16)
        assert not hasattr(rb, "sample")
        pipeline = pipeline_factory(rb, sample_count=16, base_seed=41)

        batch = pipeline.sample_large_batch(3, 16)
        rewards = batch["rewards"].cpu()

        assert rewards.min() >= 0
        assert rewards.max() < 16
        expected = _expected_pattern(rb, rewards)
        for key, want in expected.items():
            torch.testing.assert_close(batch[key].cpu(), want)

    def test_bounded_ingress_pending_overwrite_cannot_prepare_batch(self, pipeline_factory):
        rb = _make_bounded_replay(capacity=32, slot_rows=8)
        _pattern_add(rb, 0, 8)
        _pattern_add(rb, 8, 8)
        pipeline = pipeline_factory(rb, sample_count=8)
        _wait_visible(pipeline, 16)
        _pattern_add(rb, 16, 8)
        _pattern_add(rb, 24, 8)
        _wait_visible(pipeline, 32)
        pipeline._closed = True
        assert pipeline._sync_thread is not None
        pipeline._sync_thread.join(timeout=2.0)
        pipeline._closed = False

        _pattern_add(rb, 32, 8)
        assert pipeline._submit_new_spans() is True
        assert int(rb.ptr[0]) == 32
        assert pipeline.start_prepare(2, 8, min_snapshot_ptr=32) is True
        assert pipeline._service_pending_prepare() is False
        assert pipeline._prepared_metadata is None

        pipeline.progress(wait=True)
        assert int(rb.ptr[0]) == 40
        assert pipeline._service_pending_prepare() is True

    def test_bounded_ingress_consumes_spawned_collector_chunks(self, pipeline_factory):
        rb = _make_bounded_replay(capacity=64, slot_rows=8)
        pipeline = pipeline_factory(rb, sample_count=8)
        process = mp.get_context("spawn").Process(
            target=_pattern_add_chunks,
            args=(rb, 8, 5),
        )

        process.start()
        process.join(timeout=15)
        assert process.exitcode == 0
        _wait_visible(pipeline, 40)

        assert int(rb.ptr[0]) == 40
        assert int(rb.size[0]) == 40

    def test_sampled_batch_matches_replay_rows(self, pipeline_factory):
        rb = _make_replay(capacity=128)
        _pattern_add(rb, 0, 64)
        pipeline = pipeline_factory(rb, sample_count=16)
        assert pipeline.start_prepare(1, 16) is True
        batch = pipeline.sample_large_batch(1, 16)
        rewards = batch["rewards"].cpu()
        assert rewards.min() >= 0
        assert rewards.max() < 64
        expected = _expected_pattern(rb, rewards)
        for key, want in expected.items():
            torch.testing.assert_close(batch[key].cpu(), want)
        assert (batch["dones"].cpu() == 0).all()
        assert (batch["truncated"].cpu() == 1).all()
        assert batch["obs"].is_cuda

    def test_deterministic_seed_produces_same_batch(self, pipeline_factory):
        rb = _make_replay(capacity=128)
        _pattern_add(rb, 0, 64)
        p1 = pipeline_factory(rb, sample_count=16, base_seed=99)
        b1 = p1.sample_large_batch(7, 16)
        r1 = b1["rewards"].cpu().clone()
        p1.close()
        replay_again = _make_replay(capacity=128)
        _pattern_add(replay_again, 0, 64)
        p2 = pipeline_factory(replay_again, sample_count=16, base_seed=99)
        b2 = p2.sample_large_batch(7, 16)
        torch.testing.assert_close(r1, b2["rewards"].cpu())

    def test_min_snapshot_ptr_gates_prepare(self, pipeline_factory):
        rb = _make_replay(capacity=128)
        _pattern_add(rb, 0, 32)
        pipeline = pipeline_factory(rb, sample_count=8)
        _wait_visible(pipeline, 32)
        assert pipeline.start_prepare(5, 8, min_snapshot_ptr=48) is True
        time.sleep(0.2)
        assert pipeline.batch_ready(5, 8) is False
        _pattern_add(rb, 32, 16)
        _wait_batch_ready(pipeline, 5, 8)
        batch = pipeline.sample_large_batch(5, 8)
        assert batch["obs"].shape == (8, rb.obs_dim)

    def test_sample_count_mismatch_is_rejected(self, pipeline_factory):
        rb = _make_replay(capacity=64)
        _pattern_add(rb, 0, 32)
        pipeline = pipeline_factory(rb, sample_count=8)
        with pytest.raises(ValueError, match="sample_count"):
            pipeline.start_prepare(1, 999)
        with pytest.raises(ValueError, match="sample_count"):
            pipeline.batch_ready(1, 999)

    def test_prepare_same_tick_idempotent_new_tick_raises(self, pipeline_factory):
        rb = _make_replay(capacity=64)
        _pattern_add(rb, 0, 32)
        pipeline = pipeline_factory(rb, sample_count=8)
        assert pipeline.start_prepare(1, 8) is True
        assert pipeline.start_prepare(1, 8) is False
        with pytest.raises(RuntimeError, match="consumed"):
            pipeline.start_prepare(2, 8)

    def test_hot_cold_swap_after_tick(self, pipeline_factory):
        rb = _make_replay(capacity=64)
        _pattern_add(rb, 0, 32)
        pipeline = pipeline_factory(rb, sample_count=8)
        pipeline.sample_large_batch(1, 8)
        first_hot = pipeline._hot
        pipeline.after_tick()
        pipeline.start_prepare(2, 8)
        pipeline.sample_large_batch(2, 8)
        assert pipeline._hot != first_hot

    def test_hot_batch_tick_mismatch_raises(self, pipeline_factory):
        rb = _make_replay(capacity=64)
        _pattern_add(rb, 0, 32)
        pipeline = pipeline_factory(rb, sample_count=8)
        pipeline.sample_large_batch(1, 8)
        with pytest.raises(RuntimeError, match="Hot batch tick"):
            pipeline.sample_large_batch(2, 8)
        assert pipeline.batch_ready(2, 8) is False

    def test_close_stops_thread_and_unregisters(self, pipeline_factory):
        rb = _make_replay(capacity=64)
        pipeline = pipeline_factory(rb, sample_count=8)
        pipeline.close()
        assert pipeline._sync_thread is not None
        assert not pipeline._sync_thread.is_alive()
        assert not any(slot.is_pinned() for slot in rb._ingress_slots)


@mps_only
class TestMPSDevicePipeline:
    @pytest.fixture
    def pipeline_factory(self):
        created = []

        def _make(rb, **kwargs):
            kwargs.setdefault("device", "mps")
            kwargs.setdefault("sample_count", 16)
            pipeline = DeviceReplayPipeline(rb, **kwargs)
            created.append(pipeline)
            return pipeline

        yield _make
        for pipeline in created:
            pipeline.close()

    def test_allocates_mps_storage_and_samples_on_learner_thread(self, pipeline_factory):
        rb = _make_replay(capacity=128, device="mps")
        _pattern_add(rb, 0, 64)
        pipeline = pipeline_factory(rb, sample_count=16)

        batch = pipeline.sample_large_batch(1, 16)

        assert pipeline._sync_thread is None
        assert pipeline._host_pinned is False
        assert pipeline.h2d_submitter == "device_ingress_main_thread"
        assert pipeline.transfer_manifest["device_submission_thread"] == "learner"
        assert all(value.device.type == "mps" for value in batch.values())
        rewards = batch["rewards"].cpu()
        expected = _expected_pattern(rb, rewards)
        for key, want in expected.items():
            torch.testing.assert_close(batch[key].cpu(), want)

    def test_bounded_ingress_commits_and_samples_on_learner_thread(self, pipeline_factory):
        rb = _make_bounded_replay(capacity=64, slot_rows=16, device="mps")
        _pattern_add(rb, 0, 16)
        assert int(rb.ptr[0]) == 0
        pipeline = pipeline_factory(rb, sample_count=8)

        batch = pipeline.sample_large_batch(1, 8)

        assert int(rb.ptr[0]) == 16
        assert pipeline.transfer_manifest["storage_owner"] == "device"
        assert pipeline.h2d_submitter == "device_ingress_main_thread"
        rewards = batch["rewards"].cpu()
        expected = _expected_pattern(rb, rewards)
        for key, want in expected.items():
            torch.testing.assert_close(batch[key].cpu(), want)

    def test_mps_ingress_copy_is_nonblocking_and_event_gated(self, pipeline_factory, monkeypatch):
        rb = _make_bounded_replay(capacity=64, slot_rows=16, device="mps")
        _pattern_add(rb, 0, 16)
        pipeline = pipeline_factory(rb, sample_count=8)
        original_copy = torch.Tensor.copy_
        non_blocking_calls: list[bool] = []

        def recording_copy(self, source, *, non_blocking: bool = False):
            non_blocking_calls.append(non_blocking)
            return original_copy(self, source, non_blocking=non_blocking)

        monkeypatch.setattr(torch.Tensor, "copy_", recording_copy)
        batch = pipeline.sample_large_batch(1, 8)
        monkeypatch.undo()

        assert non_blocking_calls == [True]
        assert int(rb.ptr[0]) == 16
        rewards = batch["rewards"].cpu()
        expected = _expected_pattern(rb, rewards)
        for key, want in expected.items():
            torch.testing.assert_close(batch[key].cpu(), want)

    def test_mps_ring_wrap_pending_spans_slot_reuse_and_snapshot(self, pipeline_factory):
        rb = _make_bounded_replay(capacity=8, slot_rows=4, device="mps")
        pipeline = pipeline_factory(rb, sample_count=4)
        _pattern_add(rb, 0, 4)
        _pattern_add(rb, 4, 4)

        assert pipeline._submit_new_spans() is True
        assert len(pipeline._span_events) == 2
        assert pipeline._visible_ptr == 0
        pipeline.progress(wait=True)
        assert pipeline._visible_ptr == 8

        _pattern_add(rb, 8, 4)
        assert pipeline._submit_new_spans() is True
        assert pipeline._submitted_ptr == 12
        pipeline.progress(wait=True)

        end_ptr, fields = pipeline.read_committed_fields(("rewards",), start_ptr=4)
        assert end_ptr == 12
        torch.testing.assert_close(fields["rewards"].cpu(), torch.arange(4.0, 12.0))

    def test_mps_ingress_submission_overlaps_learner_work_and_close_drains(self, pipeline_factory):
        rb = _make_bounded_replay(capacity=64, slot_rows=16, device="mps")
        _pattern_add(rb, 0, 16)
        pipeline = pipeline_factory(rb, sample_count=8)
        heavy = torch.ones(4096, 4096, device="mps")

        heavy_start = time.perf_counter()
        _heavy_result = heavy @ heavy
        submit_start = time.perf_counter()
        assert pipeline._submit_new_spans() is True
        submit_elapsed_ms = (time.perf_counter() - submit_start) * 1_000
        torch.mps.synchronize()
        heavy_elapsed_ms = (time.perf_counter() - heavy_start) * 1_000

        assert submit_elapsed_ms < 5.0
        assert submit_elapsed_ms < heavy_elapsed_ms / 2
        assert pipeline._span_events[0][1].query() is True
        assert pipeline._visible_ptr == 0

        pipeline.close()
        assert pipeline._visible_ptr == 16
        with pytest.raises(RuntimeError, match="after pipeline.close()"):
            pipeline.start_prepare(1, 8)

    def test_deterministic_seed_produces_same_mps_batch(self, pipeline_factory):
        rb = _make_replay(capacity=128, device="mps")
        _pattern_add(rb, 0, 64)
        first = pipeline_factory(rb, sample_count=16, base_seed=99)
        first_rewards = first.sample_large_batch(7, 16)["rewards"].cpu().clone()
        first.close()

        replay_again = _make_replay(capacity=128, device="mps")
        _pattern_add(replay_again, 0, 64)
        second = pipeline_factory(replay_again, sample_count=16, base_seed=99)
        second_rewards = second.sample_large_batch(7, 16)["rewards"].cpu()

        torch.testing.assert_close(first_rewards, second_rewards)

    def test_min_snapshot_ptr_gates_mps_prepare(self, pipeline_factory):
        rb = _make_replay(capacity=128, device="mps")
        _pattern_add(rb, 0, 32)
        pipeline = pipeline_factory(rb, sample_count=8)

        assert pipeline.start_prepare(5, 8, min_snapshot_ptr=48) is True
        deadline = time.monotonic() + 2.0
        while pipeline._visible_ptr < 32 and time.monotonic() < deadline:
            assert pipeline.batch_ready(5, 8) is False
            time.sleep(0.001)
        assert pipeline._visible_ptr == 32

        _pattern_add(rb, 32, 16)
        _wait_batch_ready(pipeline, 5, 8)
        batch = pipeline.sample_large_batch(5, 8)

        assert pipeline._visible_ptr == 48
        assert batch["obs"].shape == (8, rb.obs_dim)

    def test_mps_device_work_rejects_background_thread(self, pipeline_factory):
        rb = _make_replay(capacity=64, device="mps")
        _pattern_add(rb, 0, 32)
        pipeline = pipeline_factory(rb, sample_count=8)
        errors: list[BaseException] = []

        def drive_from_background() -> None:
            try:
                pipeline.batch_ready(1, 8)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=drive_from_background)
        thread.start()
        thread.join(timeout=2.0)

        assert not thread.is_alive()
        assert len(errors) == 1
        assert "learner thread" in str(errors[0])

    def test_mps_pipeline_uses_event_waits_not_global_sync(self, pipeline_factory, monkeypatch):
        rb = _make_replay(capacity=64, device="mps")
        _pattern_add(rb, 0, 32)
        pipeline = pipeline_factory(rb, sample_count=8)

        def reject_global_sync() -> None:
            raise AssertionError("device MPS path must not globally synchronize")

        monkeypatch.setattr(torch.mps, "synchronize", reject_global_sync)
        batch = pipeline.sample_large_batch(1, 8)

        assert batch["obs"].device.type == "mps"


def _small_value_add(rb: ReplayBuffer, start_row: int, n: int) -> None:
    """Add rows with magnitudes well inside the fp16 range.

    Bounded fields use values that are not exactly fp16-representable so the
    quantization is observable; rewards use a fine-grained fp32-only scale to
    prove the scalar block never passes through fp16.
    """
    obs_dim, action_dim, critic_dim = rb.obs_dim, rb.action_dim, rb.critic_dim
    rows = torch.arange(start_row, start_row + n, dtype=torch.float32)
    col = rows.unsqueeze(1)
    critic = next_critic = None
    if critic_dim > 0:
        critic = col * 0.031 + torch.arange(critic_dim, dtype=torch.float32) * 0.017
        next_critic = col * 0.043 + torch.arange(critic_dim, dtype=torch.float32) * 0.019
    rb.add(
        obs=col * 0.013 + torch.arange(obs_dim, dtype=torch.float32) * 0.007,
        actions=col * 0.0031 + torch.arange(action_dim, dtype=torch.float32) * 0.0011,
        rewards=rows * 0.000123456,
        next_obs=col * 0.023 + torch.arange(obs_dim, dtype=torch.float32) * 0.005,
        dones=torch.zeros(n),
        truncated=torch.ones(n),
        critic=critic,
        next_critic=next_critic,
    )


@cuda_only
class TestFp16StoragePipeline:
    @pytest.fixture
    def pipeline_factory(self):
        created = []

        def _make(rb, **kwargs):
            kwargs.setdefault("device", "cuda")
            kwargs.setdefault("sample_count", 16)
            kwargs.setdefault("storage_dtype", "float16")
            pipeline = DeviceReplayPipeline(rb, **kwargs)
            created.append(pipeline)
            return pipeline

        yield _make
        for pipeline in created:
            pipeline.close()

    def test_fp16_ring_keeps_scalars_authoritative_fp32(self, pipeline_factory):
        rb = _make_bounded_replay(capacity=64, slot_rows=16)
        _small_value_add(rb, 0, 16)
        pipeline = pipeline_factory(rb, sample_count=16)

        assert pipeline._gpu_storage.dtype == torch.float16
        assert pipeline._gpu_scalars is not None
        assert pipeline._gpu_scalars.dtype == torch.float32
        assert pipeline.transfer_manifest["storage_dtype"] == "float16"
        manifest = pipeline.transfer_manifest
        assert manifest["storage_bytes"] == 64 * rb.storage_width * 2 + 64 * 3 * 4

        batch = pipeline.sample_large_batch(1, 16)
        assert all(value.dtype == torch.float32 for value in batch.values())
        # Sampling draws with replacement; every returned reward must be one of
        # the committed fp32 values, never an fp16 rounding of it.
        rewards = batch["rewards"].cpu()
        expected_rewards = torch.arange(16, dtype=torch.float32) * 0.000123456
        assert set(rewards.tolist()) <= set(expected_rewards.tolist())
        torch.testing.assert_close(batch["dones"].cpu(), torch.zeros(16))
        torch.testing.assert_close(batch["truncated"].cpu(), torch.ones(16))
        # Bounded fields round-trip every committed row up to fp16 resolution:
        # each sampled row must match one of the 16 committed rows.
        rows = torch.arange(16, dtype=torch.float32)
        col = rows.unsqueeze(1)
        expected_rows = {
            "obs": col * 0.013 + torch.arange(rb.obs_dim, dtype=torch.float32) * 0.007,
            "next_obs": col * 0.023 + torch.arange(rb.obs_dim, dtype=torch.float32) * 0.005,
            "actions": col * 0.0031 + torch.arange(rb.action_dim, dtype=torch.float32) * 0.0011,
            "critic": col * 0.031 + torch.arange(rb.critic_dim, dtype=torch.float32) * 0.017,
            "next_critic": col * 0.043 + torch.arange(rb.critic_dim, dtype=torch.float32) * 0.019,
        }
        for key, want in expected_rows.items():
            got = batch[key].cpu()
            distances = torch.cdist(got, want)
            assert (distances.min(dim=1).values < 2e-3).all(), key

    def test_fp16_read_committed_fields_returns_exact_scalars(self, pipeline_factory):
        rb = _make_bounded_replay(capacity=64, slot_rows=16)
        _small_value_add(rb, 0, 16)
        pipeline = pipeline_factory(rb, sample_count=8)

        end_ptr, fields = pipeline.read_committed_fields(("rewards", "dones"), start_ptr=0)

        assert end_ptr == 16
        torch.testing.assert_close(fields["rewards"].cpu(), torch.arange(16) * 0.000123456)
        assert fields["rewards"].dtype == torch.float32

    def test_fp16_ring_survives_wrap(self, pipeline_factory):
        rb = _make_bounded_replay(capacity=32, slot_rows=16)
        pipeline = pipeline_factory(rb, sample_count=8)
        _small_value_add(rb, 0, 16)
        _wait_visible(pipeline, 16)
        _small_value_add(rb, 16, 16)
        _wait_visible(pipeline, 32)

        batch = pipeline.sample_large_batch(1, 8)

        assert all(torch.isfinite(value).all() for value in batch.values())
        end_ptr, fields = pipeline.read_committed_fields(("rewards",), start_ptr=16)
        assert end_ptr == 32
        torch.testing.assert_close(
            fields["rewards"].cpu(), torch.arange(16, 32, dtype=torch.float32) * 0.000123456
        )

    def test_fp16_warmup_gather_uses_same_path(self, pipeline_factory):
        rb = _make_bounded_replay(capacity=64, slot_rows=16)
        pipeline = pipeline_factory(rb, sample_count=8)

        views = pipeline.warmup()

        assert all(value.dtype == torch.float32 for value in views.values())
        assert all(torch.isfinite(value).all() for value in views.values())

    def test_layout_declarations_cover_every_column_once(self, pipeline_factory):
        rb = _make_bounded_replay(capacity=64, slot_rows=16)
        width = rb.storage_width
        coverage = [0] * width
        for sl in (*rb.fp16_safe_slices, rb.scalar_column_slice):
            for column in range(sl.start, sl.stop):
                coverage[column] += 1
        assert all(count == 1 for count in coverage)
        assert rb.obs_dim == _OBS_DIM
        assert rb.action_dim == _ACTION_DIM
        assert rb.critic_dim == _CRITIC_DIM

    def test_fp16_fails_closed_on_undeclared_columns(
        self, pipeline_factory, monkeypatch: pytest.MonkeyPatch
    ):
        rb = _make_bounded_replay(capacity=64, slot_rows=16)
        # Simulate a fork that appended a packed column without registering a
        # storage class: only the obs slice stays declared fp16-safe.
        monkeypatch.setattr(
            ReplayBuffer,
            "fp16_safe_slices",
            property(lambda self: (self._obs_sl,)),
        )
        with pytest.raises(ValueError, match="undeclared columns"):
            pipeline_factory(rb, sample_count=8, storage_dtype="float16")

    def test_fp16_fails_closed_on_overlapping_declarations(
        self, pipeline_factory, monkeypatch: pytest.MonkeyPatch
    ):
        rb = _make_bounded_replay(capacity=64, slot_rows=16)
        monkeypatch.setattr(
            ReplayBuffer,
            "fp16_safe_slices",
            property(lambda self: (self._obs_sl, slice(0, self._obs_dim + 1))),
        )
        with pytest.raises(ValueError, match="overlapping columns"):
            pipeline_factory(rb, sample_count=8, storage_dtype="float16")

    def test_fp32_layout_skips_declaration_validation(self, pipeline_factory):
        rb = _make_bounded_replay(capacity=64, slot_rows=16)
        # fp32 keeps every column at full precision, so declarations are not
        # consulted and partial coverage must not fail construction.
        pipeline = pipeline_factory(rb, sample_count=8, storage_dtype="float32")
        assert pipeline._gpu_storage.dtype == torch.float32

    def test_fp16_gather_uses_fused_kernel_and_no_scratch(self, pipeline_factory):
        import importlib.util

        rb = _make_bounded_replay(capacity=64, slot_rows=16)
        pipeline = pipeline_factory(rb, sample_count=16)
        if importlib.util.find_spec("triton") is None:
            assert pipeline.transfer_manifest["gather_impl"] == "eager_mixed"
            return
        assert pipeline.transfer_manifest["gather_impl"] == "fused_triton"
        # The fused one-pass gather needs no fp16 scratch buffers.
        assert pipeline._gather_scratch == []
        assert pipeline._scalar_scratch == []

        _small_value_add(rb, 0, 16)
        batch = pipeline.sample_large_batch(1, 16)
        expected_rewards = torch.arange(16, dtype=torch.float32) * 0.000123456
        assert set(batch["rewards"].cpu().tolist()) <= set(expected_rewards.tolist())

    def test_fp16_gather_falls_back_to_eager(self, pipeline_factory, monkeypatch):
        import unilab.rl.runtime.ipc.replay_pipelines.device as device_module

        monkeypatch.setattr(device_module, "build_fused_gather", lambda **_kwargs: None)
        rb = _make_bounded_replay(capacity=64, slot_rows=16)
        pipeline = pipeline_factory(rb, sample_count=16)

        assert pipeline.transfer_manifest["gather_impl"] == "eager_mixed"
        assert len(pipeline._gather_scratch) == 2
        _small_value_add(rb, 0, 16)
        batch = pipeline.sample_large_batch(1, 16)
        expected_rewards = torch.arange(16, dtype=torch.float32) * 0.000123456
        assert set(batch["rewards"].cpu().tolist()) <= set(expected_rewards.tolist())


class TestNormalizeReplayStorageDtype:
    def test_aliases_resolve(self):
        from unilab.rl.runtime.ipc.replay_pipelines.device import (
            normalize_replay_storage_dtype,
        )

        assert normalize_replay_storage_dtype("float16") is torch.float16
        assert normalize_replay_storage_dtype("fp16") is torch.float16
        assert normalize_replay_storage_dtype("float32") is torch.float32
        assert normalize_replay_storage_dtype("fp32") is torch.float32
        assert normalize_replay_storage_dtype(torch.float16) is torch.float16

    def test_unknown_dtype_fails_closed(self):
        from unilab.rl.runtime.ipc.replay_pipelines.device import (
            normalize_replay_storage_dtype,
        )

        with pytest.raises(ValueError, match="Unsupported replay storage dtype"):
            normalize_replay_storage_dtype("bfloat16")
