"""Unit tests for shared off-policy runner contracts."""

from __future__ import annotations

import copy
import queue
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import unilab.algos.torch.offpolicy.double_buffer_runner as device_runner_module
import unilab.algos.torch.offpolicy.runner as runner_module
from unilab.algos.torch.offpolicy.double_buffer_runner import (
    _LearnerInferenceScheduler,
    algo_display_name,
)
from unilab.algos.torch.offpolicy.runner import (
    build_offpolicy_sample_info,
    compute_train_start_threshold,
    replay_buffer_ready_for_learning,
    update_reward_stats_from_replay,
)
from unilab.ipc.inference_slot import SharedInferenceSlot


@pytest.mark.parametrize(
    ("algo_type", "expected"),
    [("sac", "SAC"), ("td3", "TD3"), ("flashsac", "FlashSAC"), ("hora_sac", "HORA_SAC")],
)
def test_algo_display_name(algo_type, expected):
    assert algo_display_name(algo_type) == expected


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


def test_multi_step_scheduler_keeps_one_policy_version_until_update_boundary() -> None:
    scheduler = _LearnerInferenceScheduler(env_steps_per_sync=2, initial_policy_version=7)

    scheduler.record_inference(0)
    assert scheduler.policy_version == 7
    assert scheduler.update_ready is False
    assert scheduler.release_pending() == 0

    scheduler.record_inference(1)
    assert scheduler.policy_version == 7
    assert scheduler.update_ready is True
    assert scheduler.release_pending() == 1

    scheduler.finish_update()
    assert scheduler.policy_version == 8
    assert scheduler.next_tick == 2
    assert scheduler.update_ready is False


def test_scheduler_rejects_update_before_configured_tick_boundary() -> None:
    scheduler = _LearnerInferenceScheduler(env_steps_per_sync=2)
    scheduler.record_inference(0)
    scheduler.release_pending()

    with pytest.raises(RuntimeError, match="before the configured inference tick boundary"):
        scheduler.finish_update()


def test_scheduler_rejects_slot_reuse_before_response_release() -> None:
    scheduler = _LearnerInferenceScheduler(env_steps_per_sync=1)
    scheduler.record_inference(0)

    with pytest.raises(RuntimeError, match="has not been released"):
        scheduler.record_inference(1)
    with pytest.raises(RuntimeError, match="before releasing the collector tick"):
        scheduler.finish_update()


def test_scheduler_rejects_out_of_order_collector_tick() -> None:
    scheduler = _LearnerInferenceScheduler(env_steps_per_sync=1)

    with pytest.raises(RuntimeError, match="expected 0, got 1"):
        scheduler.record_inference(1)


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
        self._closed = False

    def close(self):
        if self._closed:
            return
        self._closed = True
        type(self).close_calls += 1


class _FakeLogger:
    _total_steps = 0
    _mean_ep_length = 0.0
    _collector_active_steps_per_sec = None

    def __init__(self, **kwargs):
        del kwargs
        self.statuses = []

    def set_collection_sync(self, *args):
        del args

    def update_runtime_manifest(self, manifest):
        self._runtime_manifest = dict(manifest)

    def log_status(self, value):
        self.statuses.append(value)

    def start(self):
        return None

    def start_training_timer(self):
        return 0.0

    def log_buffer_fill(self, *args):
        del args

    def update_buffer_utilization(self, value):
        del value

    def log_step(self, **kwargs):
        del kwargs

    def log_save(self, path):
        del path

    def log_collector(self, *args):
        del args

    def finish(self):
        return None

    def close(self):
        return None

    def _get_iter_steps_per_sec(self):
        return None

    def _get_effective_samples_per_sec(self):
        return None

    def _get_iter_wall_time(self):
        return 0.0


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
    assert "weight_sync_name" not in collector_kwargs
    assert "weight_param_shapes" not in collector_kwargs
    assert "collector_infer_device" not in collector_kwargs
    assert "inference_owner" not in collector_kwargs
    assert collector_kwargs["inference_slot"] is not None
    assert collector_kwargs["inference_request_queue"] is not None
    assert collector_kwargs["inference_response_queue"] is not None
    assert "collection_ready_queue" not in collector_kwargs
    assert "trainer_done_queue" not in collector_kwargs
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


def test_inference_response_freezes_next_replay_boundary_before_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _make_device_runner(monkeypatch)
    scheduler = _LearnerInferenceScheduler(env_steps_per_sync=1)
    scheduler.record_inference(0)
    replay_buffer = SimpleNamespace(published_ptr=8)
    published = []

    def publish_response(queue, *, value, timeout=5.0, label="inference_response"):
        del queue, timeout
        published.append((value, label))
        replay_buffer.published_ptr = 100

    monkeypatch.setattr(runner, "_publish_inference_response", publish_response)

    next_prepare_ptr = runner._release_inference_tick(
        object(),
        inference_scheduler=scheduler,
        replay_buffer=replay_buffer,
        trace_recorder=None,
    )

    assert next_prepare_ptr == 10
    assert published == [(0, "inference_response")]
    assert scheduler.pending_tick is None


def test_runner_releases_action_before_replay_wait_and_sample(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    events: list[str] = []

    class LoopReplayBuffer(_FakeReplayBuffer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.ptr[0] = 4
            self.size[0] = 4
            self.published_ptr = 4

    class LoopPipeline(_FakePipeline):
        last_incremental_h2d_time_s = 0.0

        def progress(self, *, wait=False):
            events.append("replay_progress")
            return wait

        def start_prepare(self, tick_id, sample_count, min_snapshot_ptr=None):
            del tick_id, sample_count, min_snapshot_ptr
            events.append("replay_prepare")
            return True

        def batch_ready(self, tick_id, sample_count):
            del tick_id, sample_count
            events.append("replay_batch_ready")
            return True

        def sample_large_batch(self, tick_id, sample_count):
            del tick_id, sample_count
            events.append("replay_sample")
            return {}

        def after_tick(self):
            events.append("replay_after_tick")

    class LoopLearner(_Learner):
        def update_critic(self, batch):
            del batch
            events.append("update_critic")
            return {}

        def update_actor(self, batch):
            del batch
            events.append("update_actor")
            return {}

        def soft_update_target(self):
            events.append("soft_update_target")

    monkeypatch.setattr(device_runner_module, "ReplayBuffer", LoopReplayBuffer)
    monkeypatch.setattr(device_runner_module, "GPUResidentReplayPipeline", LoopPipeline)
    monkeypatch.setattr(device_runner_module, "OffPolicyLogger", _FakeLogger)
    monkeypatch.setattr(device_runner_module.torch, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(device_runner_module.time, "sleep", lambda seconds: None)

    runner = _make_device_runner(monkeypatch, LoopLearner())
    runner.device = "cpu"
    monkeypatch.setattr(runner, "_start_collector", lambda **kwargs: None)
    monkeypatch.setattr(runner, "_check_collector_alive", lambda: True)
    monkeypatch.setattr(runner, "_wait_for_inference_request", lambda *args, **kwargs: 0)

    def serve_inference(*args, **kwargs):
        del args, kwargs
        events.append("action_d2h")
        return {
            "inference_h2d_time": 0.0,
            "inference_forward_time": 0.0,
            "inference_d2h_time": 0.0,
            "inference_time": 0.0,
        }

    def publish_response(*args, **kwargs):
        del args, kwargs
        events.append("inference_response")

    monkeypatch.setattr(runner, "_serve_learner_inference", serve_inference)
    monkeypatch.setattr(runner, "_publish_inference_response", publish_response)

    runner.learn(max_iterations=1, save_interval=0, log_dir=str(tmp_path))

    assert events.index("action_d2h") < events.index("inference_response")
    assert events.index("inference_response") < events.index("replay_progress")
    assert events.index("inference_response") < events.index("replay_batch_ready")
    assert events.index("inference_response") < events.index("replay_sample")
    assert events.index("inference_response") < events.index("update_critic")


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


@pytest.mark.parametrize("algo_type", ["sac", "td3", "flashsac"])
def test_learner_inference_matches_existing_actor_exploration(algo_type: str) -> None:
    if algo_type == "sac":
        from unilab.algos.torch.fast_sac.learner import SACActor

        actor = SACActor(3, 2, hidden_dim=8, use_layer_norm=False)
    elif algo_type == "flashsac":
        from unilab.algos.torch.flash_sac.network import FlashSACActor

        actor = FlashSACActor(num_blocks=1, input_dim=3, hidden_dim=8, action_dim=2)
    else:
        from unilab.algos.torch.fast_td3.learner import TD3Actor

        actor = TD3Actor(3, 2, num_envs=2, init_scale=0.01, hidden_dim=8)
    expected_actor = copy.deepcopy(actor)
    observations = np.arange(6, dtype=np.float32).reshape(2, 3) / 10.0
    dones = np.array([0.0, 1.0], dtype=np.float32)
    torch.manual_seed(17)
    expected = expected_actor.explore(
        torch.from_numpy(observations),
        dones=torch.from_numpy(dones),
        deterministic=False,
    )

    runner = object.__new__(device_runner_module.DoubleBufferOffPolicyRunner)
    runner.device = "cpu"
    runner.obs_dim = 3
    runner.obs_normalization = False
    runner.algo_type = algo_type
    runner.learner = SimpleNamespace(actor=actor)
    slot = SharedInferenceSlot(2, 3, 2)
    slot.publish_observation(tick_id=4, observations=observations, dones=dones)
    torch.manual_seed(17)
    runner._serve_learner_inference(
        slot,
        tick_id=4,
        policy_version=9,
        obs_device=torch.empty(2, 3),
        dones_device=torch.empty(2),
        trace_recorder=None,
    )
    actual, policy_version = slot.consume_action(tick_id=4)

    torch.testing.assert_close(torch.from_numpy(actual), expected)
    assert policy_version == 9
    if algo_type == "flashsac":
        torch.testing.assert_close(actor._noise, expected_actor._noise)
        torch.testing.assert_close(actor._repeat_count, expected_actor._repeat_count)
        torch.testing.assert_close(actor._repeat_target, expected_actor._repeat_target)


def test_hora_learner_inference_uses_privileged_context() -> None:
    from unilab.algos.torch.hora.sac_models import HoraSACActor

    actor = HoraSACActor(
        obs_dim=3,
        priv_info_dim=2,
        action_dim=2,
        hidden_dim=8,
        priv_mlp_hidden_dims=(4, 2),
        priv_info_embed_dim=2,
        use_layer_norm=False,
    )
    expected_actor = copy.deepcopy(actor)
    observations = np.arange(6, dtype=np.float32).reshape(2, 3) / 10.0
    priv_info = np.arange(4, dtype=np.float32).reshape(2, 2) / 10.0
    actor_input = np.concatenate((observations, priv_info), axis=1)
    dones = np.zeros(2, dtype=np.float32)
    torch.manual_seed(23)
    expected = expected_actor.explore(
        torch.from_numpy(observations),
        torch.from_numpy(priv_info),
        deterministic=False,
    )

    runner = object.__new__(device_runner_module.DoubleBufferOffPolicyRunner)
    runner.device = "cpu"
    runner.obs_dim = 3
    runner.obs_normalization = False
    runner.algo_type = "hora_sac"
    runner.learner = SimpleNamespace(actor=actor)
    slot = SharedInferenceSlot(2, 5, 2)
    slot.publish_observation(tick_id=5, observations=actor_input, dones=dones)
    torch.manual_seed(23)
    runner._serve_learner_inference(
        slot,
        tick_id=5,
        policy_version=10,
        obs_device=torch.empty(2, 5),
        dones_device=torch.empty(2),
        trace_recorder=None,
    )
    actual, policy_version = slot.consume_action(tick_id=5)

    torch.testing.assert_close(torch.from_numpy(actual), expected)
    assert policy_version == 10
