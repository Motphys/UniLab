"""Unit tests for shared off-policy runner contracts."""

from __future__ import annotations

import queue
from collections import deque

import pytest
import torch

import unilab.algos.torch.offpolicy.double_buffer_runner as device_runner_module
import unilab.algos.torch.offpolicy.runner as runner_module
from unilab.algos.torch.offpolicy.runner import (
    build_offpolicy_sample_info,
    compute_train_start_threshold,
    replay_buffer_ready_for_learning,
    update_reward_stats_from_replay,
)


@pytest.mark.parametrize(
    ("batch_size", "learning_starts", "num_envs", "expected"),
    [(8, 0, 2, 8), (8, 6, 2, 12), (32, 2, 4, 32), (0, 0, 0, 0)],
)
def test_compute_train_start_threshold(batch_size, learning_starts, num_envs, expected):
    assert compute_train_start_threshold(batch_size, learning_starts, num_envs) == expected


@pytest.mark.parametrize(
    ("size", "batch_size", "learning_starts", "num_envs", "expected"),
    [(7, 8, 0, 2, False), (8, 8, 0, 2, True), (11, 8, 6, 2, False), (12, 8, 6, 2, True)],
)
def test_replay_ready_contract(size, batch_size, learning_starts, num_envs, expected):
    assert (
        replay_buffer_ready_for_learning(
            size,
            batch_size=batch_size,
            learning_starts=learning_starts,
            num_envs=num_envs,
        )
        is expected
    )


class _Symmetry:
    batch_multiplier = 2


class _SymmetryLearner:
    use_symmetry = True
    symmetry = _Symmetry()


def test_sample_info_distinguishes_replay_rows_from_effective_samples():
    assert build_offpolicy_sample_info(
        replay_batch_size_per_rank=4,
        updates_per_step=3,
        learner=_SymmetryLearner(),
    ) == {
        "batch_size_per_rank": 8,
        "effective_batch_size": 8,
        "replay_samples_per_iter": 12,
        "learner_samples_per_iter": 24,
    }


class _RewardLearner:
    reward_normalizer = object()

    def __init__(self):
        self.calls = []

    def update_reward_stats(self, rewards, dones):
        self.calls.append((rewards.clone(), dones.clone()))


class _CommittedReplaySource:
    def __init__(self):
        self.calls = []

    def read_committed_fields(self, field_names, *, start_ptr):
        self.calls.append((field_names, start_ptr))
        return 8, {
            "rewards": torch.arange(8, dtype=torch.float32),
            "dones": torch.tensor([0, 0, 1, 0, 0, 1, 0, 0], dtype=torch.float32),
        }


def test_reward_stats_read_only_pipeline_committed_rows():
    learner = _RewardLearner()
    source = _CommittedReplaySource()
    replay = type("Replay", (), {"capacity": 16})()

    end_ptr = update_reward_stats_from_replay(
        learner,
        replay,
        start_ptr=0,
        end_ptr=0,
        num_envs=2,
        replay_source=source,
    )

    assert end_ptr == 8
    assert source.calls == [(("rewards", "dones"), 0)]
    rewards, dones = learner.calls[0]
    assert rewards.shape == (4, 2)
    assert dones.shape == (4, 2)


def test_reward_stats_reject_missing_device_replay_source():
    with pytest.raises(RuntimeError, match="device-authoritative replay source"):
        update_reward_stats_from_replay(
            _RewardLearner(),
            type("Replay", (), {"capacity": 16})(),
            start_ptr=0,
            end_ptr=8,
            num_envs=2,
        )


class _Actor:
    def state_dict(self):
        return {"weight": torch.zeros(1)}


class _Learner:
    def __init__(self, *, critic_graph=False, actor_graph=False, supports_graph=True):
        self.actor = _Actor()
        self.update_count = 0
        self.use_cuda_graph_critic_packed_staging = critic_graph
        self.use_cuda_graph_actor_packed_staging = actor_graph
        self.supports_cuda_graph_packed_staging = supports_graph

    def get_state_dict(self):
        return {"update_count": self.update_count}


class _FakeReplayBuffer:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        self.capacity = kwargs["capacity"]
        self.ptr = torch.zeros(1, dtype=torch.int64)
        self.size = torch.zeros(1, dtype=torch.int64)
        self.trace_recorder = None
        self.trace_thread_time = False
        self.trace_cuda_events = False

    def close(self):
        return None


class _FakePipeline:
    last_kwargs = None
    close_calls = 0
    h2d_submitter = "gpu_resident_ingress"
    transfer_manifest = {"backend": "fake", "device_family": "cuda"}

    def __init__(self, replay_buffer, **kwargs):
        del replay_buffer
        type(self).last_kwargs = kwargs

    def close(self):
        type(self).close_calls += 1


class _FakeWeightSync:
    name = "fake-weights"
    _lock = None

    @classmethod
    def from_state_dict(cls, state_dict, create=True):
        del state_dict, create
        return cls()

    def close(self):
        return None


class _FakeLogger:
    _total_steps = 0
    _mean_ep_length = 0.0
    _collector_active_steps_per_sec = None

    def __init__(self, **kwargs):
        del kwargs
        self.statuses = []

    def set_collection_sync(self, *args):
        del args

    def set_collector_infer_device(self, *args):
        del args

    def log_status(self, value):
        self.statuses.append(value)

    def start(self):
        return None

    def log_save(self, path):
        del path

    def log_collector(self, *args):
        del args

    def finish(self):
        return None

    def close(self):
        return None


def _make_device_runner(monkeypatch: pytest.MonkeyPatch, learner=None):
    monkeypatch.setattr(
        device_runner_module, "require_offpolicy_replay_device", lambda value: value
    )
    monkeypatch.setattr(runner_module, "get_env_dims", lambda *args, **kwargs: (4, 2, 5))
    return device_runner_module.DoubleBufferOffPolicyRunner(
        learner=learner or _Learner(),
        env_name="DummyEnv",
        algo_type="sac",
        num_envs=2,
        replay_buffer_n=8,
        batch_size=4,
        learning_starts=0,
        updates_per_step=2,
        policy_frequency=1,
        sync_collection=False,
        env_steps_per_sync=1,
        device="cuda",
    )


@pytest.mark.parametrize(
    ("critic_graph", "actor_graph", "expected_layout", "expected_critic_source"),
    [
        (False, False, "packed", False),
        (True, False, "packed", True),
        (True, True, "sac_graph", False),
    ],
)
def test_runner_constructs_only_bounded_device_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    critic_graph,
    actor_graph,
    expected_layout,
    expected_critic_source,
):
    _FakePipeline.close_calls = 0
    monkeypatch.setattr(device_runner_module, "ReplayBuffer", _FakeReplayBuffer)
    monkeypatch.setattr(device_runner_module, "GPUResidentReplayPipeline", _FakePipeline)
    monkeypatch.setattr(device_runner_module, "SharedWeightSync", _FakeWeightSync)
    monkeypatch.setattr(device_runner_module, "OffPolicyLogger", _FakeLogger)
    monkeypatch.setattr(device_runner_module.torch, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(device_runner_module.time, "sleep", lambda seconds: None)

    learner = _Learner(critic_graph=critic_graph, actor_graph=actor_graph)
    runner = _make_device_runner(monkeypatch, learner)
    collector_kwargs = {}

    def capture_collector(*, target_fn, kwargs):
        del target_fn
        collector_kwargs.update(kwargs)

    monkeypatch.setattr(runner, "_start_collector", capture_collector)
    runner.learn(max_iterations=0, save_interval=0, log_dir=str(tmp_path))

    assert _FakeReplayBuffer.last_kwargs == {
        "capacity": 16,
        "obs_dim": 4,
        "action_dim": 2,
        "device": "cuda",
        "critic_dim": 5,
        "ingress_slot_rows": 2,
        "ingress_depth": 2,
    }
    assert _FakePipeline.last_kwargs["pack_layout"] == expected_layout
    assert _FakePipeline.last_kwargs["use_critic_graph_packed_source"] is expected_critic_source
    assert not any(key.startswith("collector_pack") for key in collector_kwargs)
    assert _FakePipeline.close_calls == 1


class _ReadyAfterPoll:
    def __init__(self):
        self.ready = False
        self.start_calls = 0

    def batch_ready(self, tick_id, sample_count):
        del tick_id, sample_count
        return self.ready

    def start_prepare(self, tick_id, sample_count, min_snapshot_ptr=None):
        del tick_id, sample_count, min_snapshot_ptr
        self.start_calls += 1
        return True


def test_replay_batch_wait_uses_fine_grained_polling(monkeypatch: pytest.MonkeyPatch):
    runner = _make_device_runner(monkeypatch)
    pipeline = _ReadyAfterPoll()
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        pipeline.ready = True

    monkeypatch.setattr(runner, "_check_collector_alive", lambda: True)
    monkeypatch.setattr(device_runner_module.time, "sleep", fake_sleep)
    logger = _FakeLogger()

    assert runner._wait_for_replay_batch_ready(
        pipeline,
        tick_id=1,
        sample_count=8,
        metrics_queue=queue.Queue(),
        reward_history=deque(maxlen=100),
        latest_reward_components={},
        logger=logger,
        trace_recorder=None,
        replay_buffer=type("Replay", (), {"ptr": torch.zeros(1), "size": torch.zeros(1)})(),
        ckpt_path=None,
        train_start_wall=0.0,
    )
    assert pipeline.start_calls == 1
    assert sleeps == [pytest.approx(runner.REPLAY_BATCH_READY_POLL_SEC)]


def test_drain_metrics_propagates_collector_error():
    metrics = queue.Queue()
    metrics.put({"error": "collector boom"})
    with pytest.raises(RuntimeError, match="collector boom"):
        runner_module.OffPolicyRunner._drain_metrics(
            metrics,
            deque(maxlen=10),
            {},
            _FakeLogger(),
        )
