"""Tests for GPUResidentReplayPipeline."""

from __future__ import annotations

import multiprocessing as mp
import threading
import time

import pytest
import torch

from unilab.ipc.replay_buffer import ReplayBuffer
from unilab.ipc.replay_pipelines.gpu_resident import (
    GPUResidentReplayPipeline,
    _device_memory_budget,
    _ring_spans,
    _validate_device_memory_budget,
)

_HAS_CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
_HAS_MPS = torch.backends.mps.is_available()
mps_only = pytest.mark.skipif(not _HAS_MPS, reason="MPS required")

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
    obs_dim, action_dim, critic_dim = rb._obs_dim, rb._action_dim, rb._critic_dim
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


def _expected_pattern(rb: ReplayBuffer, rewards: torch.Tensor) -> dict[str, torch.Tensor]:
    col = rewards.cpu().unsqueeze(1)
    out = {
        "obs": col * 10 + torch.arange(rb._obs_dim, dtype=torch.float32),
        "next_obs": col * 100 + torch.arange(rb._obs_dim, dtype=torch.float32),
        "actions": col * 1000 + torch.arange(rb._action_dim, dtype=torch.float32),
    }
    if rb._critic_dim > 0:
        out["critic"] = col * 10000 + torch.arange(rb._critic_dim, dtype=torch.float32)
        out["next_critic"] = col * 100000 + torch.arange(rb._critic_dim, dtype=torch.float32)
    return out


def _wait_visible(pipeline: GPUResidentReplayPipeline, ptr: int, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while pipeline._visible_ptr < ptr:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Device replay stalled: visible_ptr={pipeline._visible_ptr} < {ptr}"
            )
        time.sleep(0.005)


def _wait_batch_ready(
    pipeline: GPUResidentReplayPipeline,
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
        assert _ring_spans(0, 4, 0) == []


class TestConstructionGuards:
    def test_non_accelerator_device_rejected(self):
        rb = _make_replay(device="cpu")
        with pytest.raises(ValueError, match="CUDA or MPS"):
            GPUResidentReplayPipeline(rb, device="cpu", sample_count=8)

    def test_invalid_pack_layout_rejected(self):
        rb = _make_replay(device="cpu")
        with pytest.raises(ValueError, match="pack_layout"):
            GPUResidentReplayPipeline(rb, device="cpu", sample_count=8, pack_layout="bogus")

    def test_runner_rejects_non_accelerator_before_base_initialization(self):
        from unilab.algos.torch.offpolicy.double_buffer_runner import (
            DoubleBufferOffPolicyRunner,
        )

        with pytest.raises(ValueError, match="CUDA or MPS"):
            DoubleBufferOffPolicyRunner(device="cpu")

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
class TestGPUResidentPipeline:
    @pytest.fixture
    def pipeline_factory(self):
        created = []

        def _make(rb, **kwargs):
            kwargs.setdefault("device", "cuda")
            kwargs.setdefault("sample_count", 16)
            pipeline = GPUResidentReplayPipeline(rb, **kwargs)
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
        assert pipeline.h2d_submitter == "gpu_resident_ingress"
        manifest = pipeline.transfer_manifest
        assert manifest["pipeline"] == "gpu_resident"
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
        assert batch["obs"].shape == (8, rb._obs_dim)

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

    def test_sac_graph_layout_column_order(self, pipeline_factory):
        rb = _make_replay(capacity=128)
        _pattern_add(rb, 0, 64)
        pipeline = pipeline_factory(rb, sample_count=16, pack_layout="sac_graph")
        batch = pipeline.sample_large_batch(1, 16)
        src = batch["sac_graph_packed_source"]
        assert src.shape == (16, rb.storage_width)
        rewards = batch["rewards"].cpu()
        expected = _expected_pattern(rb, rewards)
        for key, want in expected.items():
            torch.testing.assert_close(batch[key].cpu(), want)
        # graph order: obs, critic, actions, rew, next_obs, next_critic, done, trunc
        c = 0
        torch.testing.assert_close(src[:, c : c + rb._obs_dim].cpu(), expected["obs"])
        c += rb._obs_dim
        torch.testing.assert_close(src[:, c : c + rb._critic_dim].cpu(), expected["critic"])
        c += rb._critic_dim
        torch.testing.assert_close(src[:, c : c + rb._action_dim].cpu(), expected["actions"])

    def test_critic_graph_packed_source(self, pipeline_factory):
        rb = _make_replay(capacity=128)
        _pattern_add(rb, 0, 64)
        pipeline = pipeline_factory(
            rb,
            sample_count=16,
            use_critic_graph_packed_source=True,
        )
        batch = pipeline.sample_large_batch(1, 16)
        cg = batch["critic_graph_packed_source"]
        assert cg.shape == (16, rb.critic_graph_packed_width())
        expected = _expected_pattern(rb, batch["rewards"].cpu())
        # critic graph order: critic, actions, rew, next_obs, next_critic, done, trunc
        torch.testing.assert_close(cg[:, : rb._critic_dim].cpu(), expected["critic"])
        c = rb._critic_dim
        torch.testing.assert_close(cg[:, c : c + rb._action_dim].cpu(), expected["actions"])

    def test_close_stops_thread_and_unregisters(self, pipeline_factory):
        rb = _make_replay(capacity=64)
        pipeline = pipeline_factory(rb, sample_count=8)
        pipeline.close()
        assert pipeline._sync_thread is not None
        assert not pipeline._sync_thread.is_alive()
        assert not any(slot.is_pinned() for slot in rb._ingress_slots)


@mps_only
class TestMPSGPUResidentPipeline:
    @pytest.fixture
    def pipeline_factory(self):
        created = []

        def _make(rb, **kwargs):
            kwargs.setdefault("device", "mps")
            kwargs.setdefault("sample_count", 16)
            pipeline = GPUResidentReplayPipeline(rb, **kwargs)
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
        assert pipeline.h2d_submitter == "gpu_resident_ingress_main_thread"
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
        assert pipeline.h2d_submitter == "gpu_resident_ingress_main_thread"
        rewards = batch["rewards"].cpu()
        expected = _expected_pattern(rb, rewards)
        for key, want in expected.items():
            torch.testing.assert_close(batch[key].cpu(), want)

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
        assert batch["obs"].shape == (8, rb._obs_dim)

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
            raise AssertionError("gpu_resident MPS path must not globally synchronize")

        monkeypatch.setattr(torch.mps, "synchronize", reject_global_sync)
        batch = pipeline.sample_large_batch(1, 8)

        assert batch["obs"].device.type == "mps"

    def test_mps_sac_graph_layout_preserves_columns(self, pipeline_factory):
        rb = _make_replay(capacity=128, device="mps")
        _pattern_add(rb, 0, 64)
        pipeline = pipeline_factory(rb, sample_count=16, pack_layout="sac_graph")

        batch = pipeline.sample_large_batch(1, 16)
        rewards = batch["rewards"].cpu()
        expected = _expected_pattern(rb, rewards)

        assert batch["sac_graph_packed_source"].device.type == "mps"
        for key, want in expected.items():
            torch.testing.assert_close(batch[key].cpu(), want)
