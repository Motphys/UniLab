"""Tests for GPUResidentReplayPipeline."""

from __future__ import annotations

import time

import pytest
import torch

from unilab.ipc.replay_buffer import ReplayBuffer
from unilab.ipc.replay_pipelines.gpu_resident import (
    GPUResidentReplayPipeline,
    _ring_spans,
)

_HAS_CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")

_OBS_DIM = 4
_ACTION_DIM = 2
_CRITIC_DIM = 5


def _make_replay(
    capacity: int = 128,
    obs_dim: int = _OBS_DIM,
    action_dim: int = _ACTION_DIM,
    critic_dim: int = _CRITIC_DIM,
    device: str = "cuda",
) -> ReplayBuffer:
    return ReplayBuffer(
        capacity=capacity,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        critic_dim=critic_dim,
        defer_gpu=True,
        packed_cpu_storage=True,
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
                f"GPU replay mirror stalled: visible_ptr={pipeline._visible_ptr} < {ptr}"
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
    def test_non_cuda_device_rejected(self):
        rb = _make_replay(device="cpu")
        with pytest.raises(ValueError, match="CUDA"):
            GPUResidentReplayPipeline(rb, device="cpu", sample_count=8)

    def test_invalid_pack_layout_rejected(self):
        rb = _make_replay(device="cpu")
        with pytest.raises(ValueError, match="pack_layout"):
            GPUResidentReplayPipeline(rb, device="cpu", sample_count=8, pack_layout="bogus")

    def test_runner_rejects_invalid_replay_pipeline(self):
        from unilab.algos.torch.offpolicy.double_buffer_runner import (
            DoubleBufferOffPolicyRunner,
        )

        with pytest.raises(ValueError, match="replay_pipeline"):
            DoubleBufferOffPolicyRunner(replay_pipeline="bogus")


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
        assert pipeline._gpu_storage.shape == (64, rb._storage.shape[1])
        assert pipeline._gpu_storage.is_cuda
        assert len(pipeline._gpu_packed) == 2
        assert all(slot.is_cuda for slot in pipeline._gpu_packed)
        assert rb._storage.is_pinned()
        assert pipeline.h2d_submitter == "gpu_resident_mirror"
        manifest = pipeline.transfer_manifest
        assert manifest["pipeline"] == "gpu_resident"
        assert manifest["storage_rows"] == 64
        assert manifest["host_pinned"] is True

    def test_mirror_syncs_rows_incrementally(self, pipeline_factory):
        rb = _make_replay(capacity=128)
        _pattern_add(rb, 0, 32)
        pipeline = pipeline_factory(rb)
        _wait_visible(pipeline, 32)
        _pattern_add(rb, 32, 16)
        _wait_visible(pipeline, 48)
        torch.testing.assert_close(pipeline._gpu_storage[:48].cpu(), rb._storage[:48])

    def test_ring_wraparound_mirror_matches_storage(self, pipeline_factory):
        rb = _make_replay(capacity=64)
        _pattern_add(rb, 0, 64)
        pipeline = pipeline_factory(rb)
        _wait_visible(pipeline, 64)
        _pattern_add(rb, 64, 48)
        _wait_visible(pipeline, 112)
        torch.testing.assert_close(pipeline._gpu_storage.cpu(), rb._storage)

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
        p2 = pipeline_factory(rb, sample_count=16, base_seed=99)
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
        assert src.shape == (16, rb._storage.shape[1])
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
        assert not pipeline._sync_thread.is_alive()
        assert not rb._storage.is_pinned()
