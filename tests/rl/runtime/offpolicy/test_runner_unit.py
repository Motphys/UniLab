"""Unit tests for shared off-policy runner contracts.

Off-policy inference transport matrix (UniLab #2365):

- CPU-only learner + CPU env: historical default (CPU ring, no device
  transfer); covered by the ungated tests in this module.
- CPU env + CUDA learner: CPU ring with explicit actor H2D / action D2H
  (``cpu_ring_explicit_learner_actor_h2d_action_d2h``).
- CPU env + torch.mps (Metal Performance Shaders) learner: CPU ring with
  explicit cross-device copies (``cpu_ring_explicit_learner_device_copy``).
  The torch.mps-gated tests below require an Apple Silicon runner and skip
  elsewhere; this CI gap is declared in the #2365 PR.
- Tensor-native CUDA env + CUDA learner: CUDA tensor ring with no host
  boundary (``cuda_no_host_boundary``), the Wave 2 default topology.
"""

from __future__ import annotations

import copy
import queue
import threading
from collections import deque
from types import SimpleNamespace

import pytest
import torch

import unilab.rl.runtime.offpolicy.double_buffer_runner as device_runner_module
import unilab.rl.runtime.offpolicy.runner as runner_module
from unilab.base.contracts.probe import EnvContractProbe
from unilab.rl.logging.metric_schema import METRIC_SCHEMA_VERSION, normalize_metric_map
from unilab.rl.logging.metrics_drain import RewardComponentWindow
from unilab.rl.logging.runtime_manifest_schema import (
    RUNTIME_MANIFEST_SCHEMA_VERSION,
    validate_runtime_manifest,
)
from unilab.rl.runtime.async_runner import AsyncRunner
from unilab.rl.runtime.ipc.inference_ring import SharedInferenceRing
from unilab.rl.runtime.offpolicy.coordination import LearnerPhase
from unilab.rl.runtime.offpolicy.double_buffer_runner import (
    _LearnerInferenceScheduler,
    algo_display_name,
)
from unilab.rl.runtime.offpolicy.runner import (
    OffPolicyRunner,
    build_offpolicy_sample_info,
    compute_train_start_threshold,
    replay_buffer_ready_for_learning,
    update_reward_stats_from_replay,
)
from unilab.rl.runtime.placement import (
    InferencePlacement,
    InferenceTransport,
    resolve_inference_placement,
)
from unilab.rl.runtime.settings import RuntimeSettings


@pytest.mark.parametrize(
    ("algo_type", "expected"),
    [("sac", "SAC"), ("flashsac", "FlashSAC"), ("my_algo", "MY_ALGO")],
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


def test_scheduler_rejects_out_of_order_collector_publication() -> None:
    scheduler = _LearnerInferenceScheduler(env_steps_per_sync=1)

    with pytest.raises(RuntimeError, match="expected 0, got 1"):
        scheduler.record_inference(1)


def test_runner_close_releases_ipc_when_terminal_cleanup_fails(monkeypatch) -> None:
    events: list[str] = []

    class _ConcreteOffPolicyRunner(OffPolicyRunner):
        def learn(
            self,
            max_iterations: int,
            save_interval: int = 50,
            log_dir: str = "logs",
        ) -> None:
            del max_iterations, save_interval, log_dir

    class _FailingLogger:
        def close(self) -> None:
            events.append("logger.close")
            raise RuntimeError("terminal cleanup failed")

    runner = object.__new__(_ConcreteOffPolicyRunner)
    runner._active_logger = _FailingLogger()
    monkeypatch.setattr(AsyncRunner, "close", lambda self: events.append("async.close"))

    with pytest.raises(RuntimeError, match="terminal cleanup failed"):
        runner.close()

    assert events == ["logger.close", "async.close"]
    assert runner._active_logger is None


def test_sample_info_reports_replay_rows_and_effective_samples():
    assert build_offpolicy_sample_info(
        replay_batch_size_per_rank=4,
        updates_per_step=3,
    ) == {
        "batch_size_per_rank": 4,
        "effective_batch_size": 4,
        "learner_replay_rows_per_iter": 12,
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


def test_runner_publishes_replay_ingress_metrics_and_manifest() -> None:
    diagnostics = {
        "ingress_depth": 2,
        "ingress_slot_rows": 4,
        "published_sequence": 7,
        "release_sequence": 5,
        "occupancy": 2,
        "high_water_occupancy": 2,
        "backpressure_waits": 1,
        "backpressure_wait_s": 0.25,
        "early_returns": 1,
        "dropped_batches": 1,
        "closed_returns": 0,
        "stop_returns": 1,
    }
    pipeline = SimpleNamespace(ingress_diagnostics=lambda: diagnostics)
    updates: list[dict] = []
    logger = SimpleNamespace(
        update_runtime_manifest=lambda manifest: updates.append(manifest),
        _runtime_manifest={},
    )
    runner = object.__new__(device_runner_module.DoubleBufferOffPolicyRunner)
    runner.runtime_manifest = {}
    runner.last_run_summary = None

    metrics = runner._replay_ingress_metrics(pipeline)
    assert normalize_metric_map(metrics) == {
        "Train/replay_ingress_depth": 2.0,
        "Train/replay_ingress_occupancy": 2.0,
        "Train/replay_ingress_high_water": 2.0,
        "Train/replay_ingress_backpressure_wait_ms": 250.0,
        "Train/replay_ingress_dropped_batches": 1.0,
    }
    runner._update_replay_ingress_manifest(logger, pipeline)

    assert runner.runtime_manifest["replay_ingress"] == diagnostics
    assert updates == [{"replay_ingress": diagnostics}]


def test_runner_records_final_replay_ingress_diagnostics_in_summary() -> None:
    diagnostics = {"occupancy": 0, "dropped_batches": 2}
    replay_buffer = SimpleNamespace(ingress_diagnostics=lambda: diagnostics)
    logger = SimpleNamespace(_runtime_manifest={})
    runner = object.__new__(device_runner_module.DoubleBufferOffPolicyRunner)
    runner.runtime_manifest = {"existing": True}
    runner.last_run_summary = {"runtime_manifest": {"existing": True}}

    runner._record_final_replay_ingress_diagnostics(logger, replay_buffer)

    assert runner.runtime_manifest["replay_ingress"] == diagnostics
    assert logger._runtime_manifest["replay_ingress"] == diagnostics
    assert runner.last_run_summary["runtime_manifest"]["replay_ingress"] == diagnostics


def test_runner_final_replay_ingress_diagnostics_is_safe_before_summary_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostics = {"occupancy": 0, "dropped_batches": 0}
    replay_buffer = SimpleNamespace(ingress_diagnostics=lambda: diagnostics)
    logger = SimpleNamespace(_runtime_manifest={})
    runner = _make_device_runner(monkeypatch)

    runner._record_final_replay_ingress_diagnostics(logger, replay_buffer)

    assert runner.runtime_manifest["replay_ingress"] == diagnostics
    assert logger._runtime_manifest["replay_ingress"] == diagnostics
    assert runner.last_run_summary is None


class _Actor:
    def state_dict(self):
        return {"weight": torch.zeros(1)}


class _Learner:
    def __init__(self):
        self.actor = _Actor()
        self.update_count = 0

    def get_state_dict(self):
        return {"update_count": self.update_count}


class _FakeReplayBuffer:
    last_kwargs = None
    diagnostics_calls = 0

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        self.capacity = kwargs["capacity"]
        self.ptr = torch.zeros(1, dtype=torch.int64)
        self.size = torch.zeros(1, dtype=torch.int64)
        self.trace_recorder = None
        self.trace_thread_time = False
        self.trace_cuda_events = False

    def ingress_diagnostics(self):
        type(self).diagnostics_calls += 1
        return {
            "ingress_depth": 2,
            "ingress_slot_rows": 2,
            "published_sequence": 1,
            "release_sequence": 1,
            "occupancy": 0,
            "high_water_occupancy": 1,
            "backpressure_waits": 0,
            "backpressure_wait_s": 0.0,
            "early_returns": 0,
            "dropped_batches": 0,
            "closed_returns": 0,
            "stop_returns": 0,
        }

    def close(self):
        return None


class _FakePipeline:
    last_kwargs = None
    close_calls = 0
    h2d_submitter = "device_ingress"
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


class _ReadyPipeline(_FakePipeline):
    last_incremental_h2d_time_s = 0.0

    def progress(self, *, wait=False):
        del wait
        return True

    def start_prepare(self, tick_id, sample_count, min_snapshot_ptr=None):
        del tick_id, sample_count, min_snapshot_ptr
        return True

    def batch_ready(self, tick_id, sample_count):
        del tick_id, sample_count
        return True

    def sample_large_batch(self, tick_id, sample_count):
        del tick_id, sample_count
        return {}

    def after_tick(self):
        return None


class _FakeLogger:
    last_instance: "_FakeLogger | None" = None
    _total_steps = 0
    _mean_ep_length = 0.0

    def __init__(self, **kwargs):
        del kwargs
        self.statuses = []
        self.step_calls: list[dict] = []
        type(self).last_instance = self

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
        self.step_calls.append(kwargs)

    def log_save(self, path):
        del path

    def log_collector(self, *args):
        del args

    def finish(self):
        return None

    def close(self):
        return None

    def _get_iter_env_steps_per_sec(self):
        return None

    def _get_learner_replay_rows_per_sec(self):
        return None

    def _get_tail_env_steps_per_sec(self):
        return None

    def _get_iter_wall_time(self):
        return 0.0


class _RewardSummaryLogger(_FakeLogger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mean_ep_length = 0.0
        self._runtime_manifest = {}
        self._total_steps = 0

    def update_mean_episode_length(self, length):
        self._mean_ep_length = float(length)

    def log_collector(self, total_steps, buffer_size):
        del buffer_size
        self._total_steps = int(total_steps)


def _unused_env_factory(num_envs, env_cfg_override=None):
    raise AssertionError("probe env must not be constructed when probe_env_contract is patched")


def _make_device_runner(
    monkeypatch: pytest.MonkeyPatch,
    learner=None,
    *,
    device: str = "cuda",
    sim_backend: str = "mujoco",
    env_name: str = "DummyEnv",
    algo_type: str = "sac",
    collector_tensor_native: bool | None = None,
    inference_slot_capacity: int = 1,
    inference_epoch: int = 0,
    collector_metrics_interval: int = 1,
    runtime_settings: RuntimeSettings | None = None,
    backend_device_resolver=None,
    num_envs: int = 2,
    batch_size: int = 4,
    updates_per_step: int = 2,
    produces_completion_receipts: bool = True,
    inference_placement: InferencePlacement | None = None,
):
    monkeypatch.setattr(
        device_runner_module, "require_offpolicy_replay_device", lambda value: value
    )
    monkeypatch.setattr(
        runner_module,
        "probe_env_contract",
        lambda *args, **kwargs: EnvContractProbe(
            obs_dim=4,
            action_dim=2,
            critic_obs_dim=5,
            produces_completion_receipts=produces_completion_receipts,
        ),
    )
    placement = inference_placement
    if placement is None and collector_tensor_native is not None:
        placement = resolve_inference_placement(
            learner_device=device,
            collector_tensor_device="cuda" if collector_tensor_native else "cpu",
            algo_name="TestRunner",
            # Runner unit tests deliberately exercise the host-ring topology
            # with a (fake) CUDA learner — the #2483 attribution-baseline
            # pairing, gated here by the explicit opt-in flag.
            allow_cpu_carrier_with_cuda_learner=True,
        )
    return device_runner_module.DoubleBufferOffPolicyRunner(
        learner=learner or _Learner(),
        env_name=env_name,
        algo_type=algo_type,
        env_factory=_unused_env_factory,
        num_envs=num_envs,
        replay_buffer_n=8,
        batch_size=batch_size,
        learning_starts=0,
        updates_per_step=updates_per_step,
        policy_frequency=1,
        env_steps_per_sync=1,
        device=device,
        sim_backend=sim_backend,
        inference_placement=placement,
        inference_slot_capacity=inference_slot_capacity,
        inference_epoch=inference_epoch,
        collector_metrics_interval=collector_metrics_interval,
        runtime_settings=runtime_settings,
        backend_device_resolver=backend_device_resolver,
    )


def test_mjwarp_collector_backend_device_follows_learner_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _make_device_runner(
        monkeypatch,
        device="cuda:3",
        sim_backend="mjwarp",
        backend_device_resolver=lambda _backend, learner_device: learner_device,
    )

    assert runner.device == "cuda:3"
    assert runner.collector_backend_device == "cuda:3"
    assert runner.last_run_summary is None
    assert runner.runtime_manifest["collector_accelerator_context"] is True
    assert runner.runtime_manifest["collector_backend_device"] == "cuda:3"


def test_mjwarp_collector_start_forwards_learner_device(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _FakePipeline.close_calls = 0
    _FakeReplayBuffer.diagnostics_calls = 0
    monkeypatch.setattr(device_runner_module, "ReplayBuffer", _FakeReplayBuffer)
    monkeypatch.setattr(device_runner_module, "DeviceReplayPipeline", _FakePipeline)
    monkeypatch.setattr(device_runner_module, "OffPolicyLogger", _FakeLogger)
    monkeypatch.setattr(device_runner_module.torch, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(device_runner_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        device_runner_module.torch.cuda, "mem_get_info", lambda device: (1 << 30, 2 << 30)
    )

    class _FakeInferenceRing:
        nbytes = 1

        def __init__(self, *args, **kwargs):
            del args
            self.device = torch.device(kwargs["device"])

        def close(self):
            return None

    monkeypatch.setattr(device_runner_module, "SharedInferenceRing", _FakeInferenceRing)

    real_empty = torch.empty

    def empty_without_cuda(*args, **kwargs):
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(device_runner_module.torch, "empty", empty_without_cuda)
    runner = _make_device_runner(
        monkeypatch,
        device="cuda:3",
        sim_backend="mjwarp",
        backend_device_resolver=lambda _backend, learner_device: learner_device,
    )
    monkeypatch.setattr(runner, "_prepare_inference_timing_events", lambda: None)
    collector_kwargs = {}
    lifecycle: list[str] = []

    monkeypatch.setattr(runner, "_dp_init_broadcast", lambda: lifecycle.append("dp_init"))
    monkeypatch.setattr(runner, "_prepare_learner", lambda **kwargs: lifecycle.append("prepare"))

    def capture_collector(*, target_fn, kwargs):
        del target_fn
        lifecycle.append("collector_start")
        collector_kwargs.update(kwargs)

    monkeypatch.setattr(runner, "_start_collector", capture_collector)
    runner.learn(max_iterations=0, save_interval=0, log_dir=str(tmp_path))

    assert _FakeReplayBuffer.diagnostics_calls == 4
    assert runner.last_run_summary["status"] == "completed"
    assert runner.last_run_summary["metric_schema_version"] == METRIC_SCHEMA_VERSION
    assert (
        runner.last_run_summary["runtime_manifest"]["schema_version"]
        == RUNTIME_MANIFEST_SCHEMA_VERSION
    )
    diagnostics = runner.last_run_summary["runtime_manifest"]["replay_ingress"]
    assert diagnostics["occupancy"] == 0
    assert diagnostics["published_sequence"] == diagnostics["release_sequence"]
    assert diagnostics["early_returns"] == 0
    assert diagnostics["dropped_batches"] == 0
    assert diagnostics["closed_returns"] == 0
    assert diagnostics["stop_returns"] == 0

    assert lifecycle == ["dp_init", "prepare", "collector_start"]
    assert collector_kwargs["sim_backend"] == "mjwarp"
    assert collector_kwargs["backend_device"] == "cuda:3"
    assert collector_kwargs["learner_pid"] > 0
    assert collector_kwargs["learner_coordination"] is runner._learner_coordination


def test_normal_completion_quiesces_collector_before_replay_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """The final collector transition must be published before ingress close."""

    lifecycle: list[str] = []

    class _CloseAfterCollectorQuiescePipeline(_ReadyPipeline):
        def close(self):
            assert runner._collector_quiesced, "replay pipeline closed before collector exit"
            lifecycle.append("replay_close")
            super().close()

    class _ReadyReplayBuffer(_FakeReplayBuffer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.ptr[0] = 4
            self.size[0] = 4
            self.published_ptr = 4

    class _UpdateLearner(_Learner):
        supports_deferred_update_metrics = True

        def update_critic(self, batch, *, read_metrics: bool = True):
            del batch
            return {"Loss/critic": 7.0} if read_metrics else {}

        def update_actor(self, batch, *, read_metrics: bool = True):
            del batch
            return {}

        def read_deferred_actor_metrics(self):
            return {"Loss/actor": 3.0}

        def soft_update_target(self):
            return None

    monkeypatch.setattr(device_runner_module, "ReplayBuffer", _ReadyReplayBuffer)
    monkeypatch.setattr(
        device_runner_module, "DeviceReplayPipeline", _CloseAfterCollectorQuiescePipeline
    )
    monkeypatch.setattr(device_runner_module, "OffPolicyLogger", _FakeLogger)
    monkeypatch.setattr(device_runner_module.torch, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(device_runner_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        device_runner_module.torch.cuda, "mem_get_info", lambda device: (1 << 30, 2 << 30)
    )

    class _FakeInferenceRing:
        nbytes = 1

        def __init__(self, *args, **kwargs):
            del args
            self.device = torch.device(kwargs["device"])

        def close(self):
            return None

    monkeypatch.setattr(device_runner_module, "SharedInferenceRing", _FakeInferenceRing)
    real_empty = torch.empty

    def empty_without_cuda(*args, **kwargs):
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(device_runner_module.torch, "empty", empty_without_cuda)
    runner = _make_device_runner(
        monkeypatch,
        _UpdateLearner(),
        device="cuda:3",
        sim_backend="mjwarp",
    )
    runner.device = "cpu"
    monkeypatch.setattr(runner, "_prepare_inference_timing_events", lambda: None)
    monkeypatch.setattr(runner, "_dp_init_broadcast", lambda: None)
    monkeypatch.setattr(runner, "_prepare_learner", lambda **kwargs: None)
    monkeypatch.setattr(runner, "_start_collector", lambda *, target_fn, kwargs: None)
    monkeypatch.setattr(runner, "_check_collector_alive", lambda: True)
    monkeypatch.setattr(runner, "_wait_for_inference_request", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        runner,
        "_serve_learner_inference",
        lambda *args, **kwargs: {
            "inference_h2d_time": 0.0,
            "inference_forward_time": 0.0,
            "inference_d2h_time": 0.0,
            "inference_time": 0.0,
        },
    )
    monkeypatch.setattr(runner, "_publish_inference_response", lambda *args, **kwargs: None)
    runner._collector_quiesced = False

    def quiesce_collector():
        if not runner._collector_quiesced:
            runner._collector_quiesced = True
            lifecycle.append("collector_quiesce")

    monkeypatch.setattr(runner, "_shutdown_collector", quiesce_collector)

    runner.learn(max_iterations=1, save_interval=0, log_dir=str(tmp_path))

    assert runner.last_run_summary["status"] == "completed"
    assert lifecycle == ["collector_quiesce", "replay_close"]


def test_learn_consumes_final_collector_return_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Consume a metric message queued by the collector at process shutdown."""

    _FakePipeline.close_calls = 0
    _FakeReplayBuffer.diagnostics_calls = 0
    monkeypatch.setattr(device_runner_module, "ReplayBuffer", _FakeReplayBuffer)
    monkeypatch.setattr(device_runner_module, "DeviceReplayPipeline", _FakePipeline)
    monkeypatch.setattr(device_runner_module, "OffPolicyLogger", _RewardSummaryLogger)
    monkeypatch.setattr(device_runner_module.torch, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        device_runner_module.torch.cuda, "mem_get_info", lambda device: (1 << 30, 2 << 30)
    )

    class _FakeInferenceRing:
        nbytes = 1

        def __init__(self, *args, **kwargs):
            del args
            self.device = torch.device(kwargs["device"])

        def close(self):
            return None

    monkeypatch.setattr(device_runner_module, "SharedInferenceRing", _FakeInferenceRing)
    real_empty = torch.empty

    def empty_without_cuda(*args, **kwargs):
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(device_runner_module.torch, "empty", empty_without_cuda)
    runner = _make_device_runner(monkeypatch, device="cuda:3", sim_backend="mjwarp")
    runner.device = "cpu"
    monkeypatch.setattr(runner, "_prepare_inference_timing_events", lambda: None)
    monkeypatch.setattr(runner, "_dp_init_broadcast", lambda: None)
    monkeypatch.setattr(runner, "_prepare_learner", lambda **kwargs: None)
    monkeypatch.setattr(runner, "_start_collector", lambda *, target_fn, kwargs: None)

    metrics_queue = queue.Queue()
    metrics_queue.put_nowait(
        {
            "total_steps": 8192,
            "buffer_size": 8192,
            "return_mean_ep100": 4.0,
            "mean_episode_length": 32.0,
            "metric_flush": "final",
        }
    )
    reward_history: deque[float] = deque(maxlen=10)
    reward_history.append(1.0)
    logger = _RewardSummaryLogger()

    runner._drain_collector_metrics_after_shutdown(
        metrics_queue,
        reward_history,
        RewardComponentWindow(),
        logger,
    )

    assert list(reward_history) == [1.0, 4.0]
    assert logger._mean_ep_length == pytest.approx(32.0)
    assert logger._total_steps == 8192
    assert metrics_queue.empty()


def test_learn_failure_before_summary_preserves_original_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class _FinishFailureLogger(_FakeLogger):
        def finish(self):
            raise RuntimeError("finish failed")

    _FakePipeline.close_calls = 0
    _FakeReplayBuffer.diagnostics_calls = 0
    monkeypatch.setattr(device_runner_module, "ReplayBuffer", _FakeReplayBuffer)
    monkeypatch.setattr(device_runner_module, "DeviceReplayPipeline", _FakePipeline)
    monkeypatch.setattr(device_runner_module, "OffPolicyLogger", _FinishFailureLogger)
    monkeypatch.setattr(device_runner_module.torch, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(device_runner_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        device_runner_module.torch.cuda, "mem_get_info", lambda device: (1 << 30, 2 << 30)
    )

    class _FakeInferenceRing:
        nbytes = 1

        def __init__(self, *args, **kwargs):
            del args
            self.device = torch.device(kwargs["device"])

        def close(self):
            return None

    monkeypatch.setattr(device_runner_module, "SharedInferenceRing", _FakeInferenceRing)

    real_empty = torch.empty

    def empty_without_cuda(*args, **kwargs):
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(device_runner_module.torch, "empty", empty_without_cuda)
    runner = _make_device_runner(monkeypatch, device="cuda:3", sim_backend="mjwarp")
    monkeypatch.setattr(runner, "_prepare_inference_timing_events", lambda: None)
    monkeypatch.setattr(runner, "_dp_init_broadcast", lambda: None)
    monkeypatch.setattr(runner, "_prepare_learner", lambda **kwargs: None)
    monkeypatch.setattr(runner, "_start_collector", lambda *, target_fn, kwargs: None)

    with pytest.raises(RuntimeError, match="finish failed") as excinfo:
        runner.learn(max_iterations=0, save_interval=0, log_dir=str(tmp_path))

    assert not isinstance(excinfo.value, AttributeError)
    assert _FakeReplayBuffer.diagnostics_calls == 4
    assert runner.last_run_summary["status"] == "failed"
    shutdown = runner.last_run_summary["runtime_manifest"]["shutdown"]
    assert shutdown["classification"] == "learner_failure"
    assert shutdown["owner"] == "learner"
    assert shutdown["phase"] == "finalize/logger_finish"
    assert shutdown["exception"] == {"type": "RuntimeError", "message": "finish failed"}
    assert runner.runtime_manifest["replay_ingress"]["occupancy"] == 0
    assert _FinishFailureLogger.last_instance._runtime_manifest["replay_ingress"]["occupancy"] == 0
    assert _FakePipeline.close_calls == 1


def test_learn_startup_failure_records_shutdown_and_replaces_stale_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class _StartupFailureLogger:
        def __init__(self, **kwargs):
            del kwargs
            raise RuntimeError("logger construction failed")

    _FakePipeline.close_calls = 0
    _FakeReplayBuffer.diagnostics_calls = 0
    monkeypatch.setattr(device_runner_module, "ReplayBuffer", _FakeReplayBuffer)
    monkeypatch.setattr(device_runner_module, "DeviceReplayPipeline", _FakePipeline)
    monkeypatch.setattr(device_runner_module, "OffPolicyLogger", _StartupFailureLogger)
    monkeypatch.setattr(device_runner_module.torch, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(device_runner_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        device_runner_module.torch.cuda, "mem_get_info", lambda device: (1 << 30, 2 << 30)
    )

    class _FakeInferenceRing:
        nbytes = 1

        def __init__(self, *args, **kwargs):
            del args
            self.device = torch.device(kwargs["device"])

        def close(self):
            return None

    monkeypatch.setattr(device_runner_module, "SharedInferenceRing", _FakeInferenceRing)
    real_empty = torch.empty

    def empty_without_cuda(*args, **kwargs):
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(device_runner_module.torch, "empty", empty_without_cuda)
    runner = _make_device_runner(monkeypatch, device="cuda:3", sim_backend="mjwarp")
    runner.runtime_manifest["shutdown"] = {"classification": "normal_completion"}
    runner.last_run_summary = {"status": "completed", "stale": True}
    monkeypatch.setattr(runner, "_prepare_inference_timing_events", lambda: None)

    with pytest.raises(RuntimeError, match="logger construction failed"):
        runner.learn(max_iterations=0, save_interval=0, log_dir=str(tmp_path))

    assert runner.last_run_summary["status"] == "failed"
    assert "stale" not in runner.last_run_summary
    assert runner.last_run_summary["metric_schema_version"] == METRIC_SCHEMA_VERSION
    assert (
        runner.last_run_summary["runtime_manifest"]["schema_version"]
        == device_runner_module.RUNTIME_MANIFEST_SCHEMA_VERSION
    )
    shutdown = runner.last_run_summary["runtime_manifest"]["shutdown"]
    assert shutdown["classification"] == "learner_failure"
    assert shutdown["owner"] == "learner"
    assert shutdown["phase"] == "startup/logger"
    assert shutdown["exception"] == {
        "type": "RuntimeError",
        "message": "logger construction failed",
    }
    assert "schema_version" not in shutdown
    runner.close()


def test_offpolicy_resume_restores_state_and_iteration_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    checkpoint_path = tmp_path / "model_7.pt"
    checkpoint = {"weights": torch.ones(2), "update_count": 7}
    loaded: list[dict[str, object]] = []

    class _ResumeLearner:
        def __init__(self) -> None:
            self.loaded_state: dict[str, object] | None = None

        def load_state_dict(self, state: dict[str, object]) -> None:
            self.loaded_state = state

    def fake_load(path, **kwargs):
        loaded.append({"path": path, **kwargs})
        return checkpoint

    learner = _ResumeLearner()
    runner = _make_device_runner(monkeypatch, learner=learner, device="cpu")
    monkeypatch.setattr(device_runner_module.torch, "load", fake_load)
    captured_impl: dict[str, object] = {}

    def fake_impl(**kwargs):
        captured_impl.update(kwargs)

    monkeypatch.setattr(runner, "_learn_impl", fake_impl)

    runner.learn(
        max_iterations=20,
        save_interval=4,
        log_dir=str(tmp_path),
        resume_checkpoint=str(checkpoint_path),
    )

    assert loaded == [
        {
            "path": str(checkpoint_path),
            "map_location": "cpu",
            "weights_only": False,
        }
    ]
    assert learner.loaded_state is checkpoint
    assert runner.resume_start_iteration == 8
    assert captured_impl == {
        "max_iterations": 20,
        "save_interval": 4,
        "log_dir": str(tmp_path),
        "logger_type": "tensorboard",
    }


@pytest.mark.parametrize(
    ("checkpoint", "message"),
    [
        (torch.ones(1), "must contain a learner state dictionary"),
        ({"weights": torch.ones(1)}, "missing a valid integer update_count"),
        ({"update_count": "invalid"}, "missing a valid integer update_count"),
        ({"update_count": -1}, "update_count must be non-negative"),
    ],
)
def test_offpolicy_resume_rejects_invalid_checkpoint_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    checkpoint,
    message,
) -> None:
    learner = _Learner()
    runner = _make_device_runner(monkeypatch, learner=learner, device="cpu")
    monkeypatch.setattr(
        device_runner_module.torch,
        "load",
        lambda *args, **kwargs: checkpoint,
    )

    with pytest.raises((TypeError, ValueError), match=message):
        runner.learn(
            max_iterations=10,
            save_interval=0,
            log_dir=str(tmp_path),
            resume_checkpoint=str(tmp_path / "model.pt"),
        )

    assert not hasattr(runner, "resume_start_iteration")


def test_offpolicy_resume_rejects_iteration_past_max(monkeypatch: pytest.MonkeyPatch, tmp_path):
    learner = SimpleNamespace(load_state_dict=lambda state: None)
    runner = _make_device_runner(monkeypatch, learner=learner, device="cpu")
    monkeypatch.setattr(
        device_runner_module.torch,
        "load",
        lambda *args, **kwargs: {"update_count": 10},
    )
    monkeypatch.setattr(runner, "_learn_impl", lambda **kwargs: pytest.fail("training started"))

    with pytest.raises(ValueError, match="newer than max_iterations"):
        runner.learn(
            max_iterations=9,
            save_interval=0,
            log_dir=str(tmp_path),
            resume_checkpoint=str(tmp_path / "model_10.pt"),
        )

    assert runner.resume_start_iteration == 11


def test_minimal_failed_summary_is_schema_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _make_device_runner(monkeypatch)
    summary = runner._minimal_failed_summary("failed")

    assert summary["metric_schema_version"] == METRIC_SCHEMA_VERSION
    assert summary["runtime_manifest"]["schema_version"] == (
        device_runner_module.RUNTIME_MANIFEST_SCHEMA_VERSION
    )
    validate_runtime_manifest(
        summary["runtime_manifest"],
        completed=False,
    )


def test_shutdown_diagnostics_cleanup_does_not_replace_original_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class _FinishFailureLogger(_FakeLogger):
        def finish(self):
            raise RuntimeError("finish failed")

    class _FailingDiagnosticsReplayBuffer(_FakeReplayBuffer):
        def ingress_diagnostics(self):
            raise KeyboardInterrupt("replay diagnostics unavailable")

    _FakePipeline.close_calls = 0
    _FakeReplayBuffer.diagnostics_calls = 0
    monkeypatch.setattr(device_runner_module, "ReplayBuffer", _FailingDiagnosticsReplayBuffer)
    monkeypatch.setattr(device_runner_module, "DeviceReplayPipeline", _FakePipeline)
    monkeypatch.setattr(device_runner_module, "OffPolicyLogger", _FinishFailureLogger)
    monkeypatch.setattr(device_runner_module.torch, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(device_runner_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        device_runner_module.torch.cuda, "mem_get_info", lambda device: (1 << 30, 2 << 30)
    )

    class _FakeInferenceRing:
        nbytes = 1

        def __init__(self, *args, **kwargs):
            del args
            self.device = torch.device(kwargs["device"])

        def close(self):
            return None

    monkeypatch.setattr(device_runner_module, "SharedInferenceRing", _FakeInferenceRing)
    real_empty = torch.empty

    def empty_without_cuda(*args, **kwargs):
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(device_runner_module.torch, "empty", empty_without_cuda)
    runner = _make_device_runner(monkeypatch, device="cuda:3", sim_backend="mjwarp")
    monkeypatch.setattr(runner, "_prepare_inference_timing_events", lambda: None)
    monkeypatch.setattr(runner, "_dp_init_broadcast", lambda: None)
    monkeypatch.setattr(runner, "_prepare_learner", lambda **kwargs: None)
    monkeypatch.setattr(runner, "_start_collector", lambda *, target_fn, kwargs: None)

    with pytest.raises(RuntimeError, match="finish failed"):
        runner.learn(max_iterations=0, save_interval=0, log_dir=str(tmp_path))

    shutdown = runner.last_run_summary["runtime_manifest"]["shutdown"]
    assert shutdown["classification"] == "learner_failure"
    cleanup_errors = shutdown["cleanup"]["errors"]
    assert {
        "type": "KeyboardInterrupt",
        "message": "replay diagnostics unavailable",
    } in cleanup_errors
    runner.close()


def test_collector_failure_cleanup_does_not_replace_original_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class _CleanupFailureLogger(_FakeLogger):
        def log_status(self, value):
            if "Collector died" not in value:
                return
            raise KeyboardInterrupt("logger status failed")

    class _CleanupFailurePipeline(_FakePipeline):
        _close_failed = False
        cleanup_close_calls = 0

        def close(self):
            if self._closed:
                return
            self._closed = True
            type(self).cleanup_close_calls += 1
            if not type(self)._close_failed:
                type(self)._close_failed = True
                raise SystemExit("pipeline close failed")

    _FakePipeline.close_calls = 0
    _FakeReplayBuffer.diagnostics_calls = 0
    monkeypatch.setattr(device_runner_module, "ReplayBuffer", _FakeReplayBuffer)
    monkeypatch.setattr(device_runner_module, "DeviceReplayPipeline", _CleanupFailurePipeline)
    monkeypatch.setattr(device_runner_module, "OffPolicyLogger", _CleanupFailureLogger)
    monkeypatch.setattr(device_runner_module.torch, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(device_runner_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        device_runner_module.torch.cuda, "mem_get_info", lambda device: (1 << 30, 2 << 30)
    )

    class _FakeInferenceRing:
        nbytes = 1

        def __init__(self, *args, **kwargs):
            del args
            self.device = torch.device(kwargs["device"])

        def close(self):
            return None

    monkeypatch.setattr(device_runner_module, "SharedInferenceRing", _FakeInferenceRing)
    real_empty = torch.empty

    def empty_without_cuda(*args, **kwargs):
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(device_runner_module.torch, "empty", empty_without_cuda)
    runner = _make_device_runner(monkeypatch, device="cuda:3", sim_backend="mjwarp")
    monkeypatch.setattr(runner, "_prepare_inference_timing_events", lambda: None)
    monkeypatch.setattr(runner, "_dp_init_broadcast", lambda: None)
    monkeypatch.setattr(runner, "_prepare_learner", lambda **kwargs: None)
    monkeypatch.setattr(runner, "_start_collector", lambda *, target_fn, kwargs: None)

    def collector_died(*args, **kwargs):
        del args, kwargs
        raise device_runner_module._CollectorDiedError("collector dead test")

    def shutdown_collector():
        raise KeyboardInterrupt("collector shutdown failed")

    monkeypatch.setattr(runner, "_wait_for_inference_request", collector_died)
    monkeypatch.setattr(runner, "_shutdown_collector", shutdown_collector)

    with pytest.raises(RuntimeError, match="Collector process died"):
        runner.learn(max_iterations=1, save_interval=0, log_dir=str(tmp_path))

    assert runner.last_run_summary["status"] == "collector_died"
    assert runner.last_run_summary["metric_schema_version"] == METRIC_SCHEMA_VERSION
    assert (
        runner.last_run_summary["runtime_manifest"]["schema_version"]
        == device_runner_module.RUNTIME_MANIFEST_SCHEMA_VERSION
    )
    cleanup_errors = runner.last_run_summary["runtime_manifest"]["shutdown"]["cleanup"]["errors"]
    assert {"type": "KeyboardInterrupt", "message": "logger status failed"} in cleanup_errors
    assert {"type": "SystemExit", "message": "pipeline close failed"} in cleanup_errors
    assert {"type": "KeyboardInterrupt", "message": "collector shutdown failed"} in cleanup_errors
    runner.close()


def test_collector_died_learn_refreshes_ingress_after_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _FakePipeline.close_calls = 0
    _FakeReplayBuffer.diagnostics_calls = 0

    monkeypatch.setattr(
        device_runner_module.torch.cuda, "mem_get_info", lambda device: (1 << 30, 2 << 30)
    )

    class _FakeInferenceRing:
        nbytes = 1

        def __init__(self, *args, **kwargs):
            del args
            self.device = torch.device(kwargs["device"])

        def close(self):
            return None

    monkeypatch.setattr(device_runner_module, "SharedInferenceRing", _FakeInferenceRing)

    real_empty = torch.empty

    def empty_without_cuda(*args, **kwargs):
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(device_runner_module.torch, "empty", empty_without_cuda)
    runner = _make_device_runner(monkeypatch, device="cuda:3", sim_backend="mjwarp")
    runner._shutdown_seen = False
    diagnostics_after_shutdown: list[bool] = []

    class _ShutdownAwareReplayBuffer(_FakeReplayBuffer):
        def ingress_diagnostics(self):
            assert runner._shutdown_seen
            diagnostics_after_shutdown.append(runner._shutdown_seen)
            diagnostics = super().ingress_diagnostics()
            diagnostics["stop_returns"] = 1
            return diagnostics

    monkeypatch.setattr(device_runner_module, "ReplayBuffer", _ShutdownAwareReplayBuffer)
    monkeypatch.setattr(device_runner_module, "DeviceReplayPipeline", _FakePipeline)
    monkeypatch.setattr(device_runner_module, "OffPolicyLogger", _FakeLogger)
    monkeypatch.setattr(device_runner_module.torch, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(device_runner_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(runner, "_prepare_inference_timing_events", lambda: None)
    monkeypatch.setattr(runner, "_dp_init_broadcast", lambda: None)
    monkeypatch.setattr(runner, "_prepare_learner", lambda **kwargs: None)
    monkeypatch.setattr(runner, "_start_collector", lambda *, target_fn, kwargs: None)

    def collector_died(*args, **kwargs):
        del args, kwargs
        raise device_runner_module._CollectorDiedError("collector dead test")

    monkeypatch.setattr(runner, "_wait_for_inference_request", collector_died)
    monkeypatch.setattr(
        runner,
        "_shutdown_collector",
        lambda: setattr(runner, "_shutdown_seen", True),
    )

    with pytest.raises(RuntimeError, match="Collector process died") as excinfo:
        runner.learn(max_iterations=1, save_interval=0, log_dir=str(tmp_path))

    assert not isinstance(excinfo.value, AttributeError)
    assert diagnostics_after_shutdown == [True, True]
    assert runner._shutdown_seen is True
    assert runner.last_run_summary["status"] == "collector_died"
    diagnostics = runner.last_run_summary["runtime_manifest"]["replay_ingress"]
    assert diagnostics["occupancy"] == 0
    assert diagnostics["published_sequence"] == diagnostics["release_sequence"]
    assert diagnostics["stop_returns"] == 1
    shutdown = runner.last_run_summary["runtime_manifest"]["shutdown"]
    assert shutdown["classification"] == "collector_failure"
    assert shutdown["owner"] == "collector"
    assert shutdown["phase"] == "training/wait_for_inference_request"
    assert shutdown["coordination_tick"] is None
    assert _FakePipeline.close_calls == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_runner_constructs_only_bounded_device_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    _FakePipeline.close_calls = 0
    monkeypatch.setattr(device_runner_module, "ReplayBuffer", _FakeReplayBuffer)
    monkeypatch.setattr(device_runner_module, "DeviceReplayPipeline", _FakePipeline)
    monkeypatch.setattr(device_runner_module, "OffPolicyLogger", _FakeLogger)
    monkeypatch.setattr(device_runner_module.torch, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(device_runner_module.time, "sleep", lambda seconds: None)

    learner = _Learner()
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
        "ingress_device": "cpu",
        "require_completion_receipt": True,
    }
    assert _FakePipeline.last_kwargs == {
        "device": "cuda",
        "sample_count": 8,
        "base_seed": 0,
        "trace_recorder": None,
        "trace_cuda_events": True,
    }
    runtime_manifest = runner.last_run_summary["runtime_manifest"]
    assert runtime_manifest["replay_h2d_submitter"] == runner.replay_h2d_submitter
    assert "replay_device_submission_thread" in runtime_manifest
    budget = runtime_manifest["inference_memory_budget"]
    assert budget["ipc_event_count"] == 0
    assert budget["timing_event_count"] == 2
    assert budget["cuda_event_count"] == 2
    assert runner.runtime_manifest["inference_timing"] == {
        "timer_kind": "cuda_events",
        "has_timing_events": True,
        "persistent_timing_event_count": 2,
        "classification": "capability",
    }
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_runner_relaxes_receipt_gate_for_sync_declared_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    """A sync-declared probe must relax the replay ingress receipt gate."""
    _FakePipeline.close_calls = 0
    monkeypatch.setattr(device_runner_module, "ReplayBuffer", _FakeReplayBuffer)
    monkeypatch.setattr(device_runner_module, "DeviceReplayPipeline", _FakePipeline)
    monkeypatch.setattr(device_runner_module, "OffPolicyLogger", _FakeLogger)
    monkeypatch.setattr(device_runner_module.torch, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(device_runner_module.time, "sleep", lambda seconds: None)

    runner = _make_device_runner(
        monkeypatch,
        _Learner(),
        collector_tensor_native=True,
        produces_completion_receipts=False,
    )
    assert runner.produces_completion_receipts is False
    monkeypatch.setattr(runner, "_start_collector", lambda *, target_fn, kwargs: None)
    runner.learn(max_iterations=0, save_interval=0, log_dir=str(tmp_path))

    assert _FakeReplayBuffer.last_kwargs["require_completion_receipt"] is False
    assert _FakeReplayBuffer.last_kwargs["ingress_device"] == "cuda"
    assert _FakePipeline.close_calls == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mujoco_cuda_carrier_resolves_cuda_tensor_ring_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    """Pin the Wave 2 default topology: a CUDA learner with default config.

    With ``training.collector_tensor_device`` at its default ``auto``, a CUDA
    learner resolves to the CUDA tensor ring with no host staging, replay
    ingress on the learner device, and a relaxed receipt gate for the
    sync-declared (no completion receipts) env.
    """
    _FakePipeline.close_calls = 0
    monkeypatch.setattr(device_runner_module, "ReplayBuffer", _FakeReplayBuffer)
    monkeypatch.setattr(device_runner_module, "DeviceReplayPipeline", _FakePipeline)
    monkeypatch.setattr(device_runner_module, "OffPolicyLogger", _FakeLogger)
    monkeypatch.setattr(device_runner_module.torch, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(device_runner_module.time, "sleep", lambda seconds: None)

    learner_device = f"cuda:{torch.cuda.current_device()}"
    # Default resolution: collector_tensor_device=auto follows the CUDA
    # learner, so placement selects the CUDA tensor ring with no host boundary.
    placement = resolve_inference_placement(
        learner_device=learner_device,
        collector_tensor_device="auto",
        algo_name="TestRunner",
    )
    assert placement.mode is InferenceTransport.CUDA
    assert placement.staging_policy == "cuda_no_host_boundary"

    runner = _make_device_runner(
        monkeypatch,
        _Learner(),
        device=learner_device,
        sim_backend="mujoco",
        inference_placement=placement,
        produces_completion_receipts=False,
    )
    assert runner.produces_completion_receipts is False
    collector_kwargs = {}

    def capture_collector(*, target_fn, kwargs):
        del target_fn
        collector_kwargs.update(kwargs)

    monkeypatch.setattr(runner, "_start_collector", capture_collector)
    runner.learn(max_iterations=0, save_interval=0, log_dir=str(tmp_path))

    assert _FakeReplayBuffer.last_kwargs["ingress_device"] == learner_device
    assert _FakeReplayBuffer.last_kwargs["require_completion_receipt"] is False
    assert collector_kwargs["inference_transport"] == "cuda"
    assert collector_kwargs["inference_slot"].device == torch.device(learner_device)
    manifest = runner.runtime_manifest
    assert manifest["collector_tensor_native"] is True
    assert manifest["inference_staging_policy"] == "cuda_no_host_boundary"
    assert manifest["inference_action_host_staging"] is False
    assert manifest["inference_transport"] == {
        "mode": "cuda",
        "env_device": learner_device,
        "ring_device": learner_device,
        "learner_device": learner_device,
        "staging_policy": "cuda_no_host_boundary",
        "transport_table": placement.transport_spec.manifest(),
    }
    assert _FakePipeline.close_calls == 1


_MPS_SKIP_REASON = "torch.mps (Metal Performance Shaders) runtime required"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason=_MPS_SKIP_REASON)
def test_mps_learner_cpu_env_resolves_cpu_ring_with_device_copy_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    """CPU env + torch.mps learner cell: host transport with explicit copies.

    The probe reports host (non-tensor-native) observations, so placement
    keeps the CPU ring; the mps learner owns the replay device while ingress
    stays on the host and the torch-copy transfer backend crosses to mps.
    """
    _FakePipeline.close_calls = 0
    monkeypatch.setattr(device_runner_module, "ReplayBuffer", _FakeReplayBuffer)
    monkeypatch.setattr(device_runner_module, "DeviceReplayPipeline", _FakePipeline)
    monkeypatch.setattr(device_runner_module, "OffPolicyLogger", _FakeLogger)
    monkeypatch.setattr(device_runner_module.torch, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(device_runner_module.time, "sleep", lambda seconds: None)

    placement = resolve_inference_placement(
        learner_device="mps",
        collector_tensor_device="cpu",
        algo_name="TestRunner",
    )
    assert placement.mode is InferenceTransport.HOST
    assert placement.env_device == "cpu"
    assert placement.ring_device == "cpu"
    assert placement.staging_policy == "cpu_ring_explicit_learner_device_copy"

    runner = _make_device_runner(
        monkeypatch,
        _Learner(),
        device="mps",
        sim_backend="mujoco",
        inference_placement=placement,
        produces_completion_receipts=False,
    )
    collector_kwargs = {}

    def capture_collector(*, target_fn, kwargs):
        del target_fn
        collector_kwargs.update(kwargs)

    monkeypatch.setattr(runner, "_start_collector", capture_collector)
    runner.learn(max_iterations=0, save_interval=0, log_dir=str(tmp_path))

    # The replay buffer lives on the mps learner device; the CPU-env ingress
    # stays on the host and crosses devices inside the replay pipeline.
    assert _FakeReplayBuffer.last_kwargs["device"] == "mps"
    assert _FakeReplayBuffer.last_kwargs["ingress_device"] == "cpu"
    assert _FakeReplayBuffer.last_kwargs["require_completion_receipt"] is False
    assert collector_kwargs["inference_transport"] == "host"
    assert collector_kwargs["inference_slot"].device.type == "cpu"
    manifest = runner.runtime_manifest
    assert manifest["collector_tensor_native"] is False
    assert manifest["inference_staging_policy"] == "cpu_ring_explicit_learner_device_copy"
    assert manifest["inference_action_host_staging"] is True
    assert manifest["inference_learner_scratch_device"] == "mps"
    assert manifest["inference_transport"] == {
        "mode": "host",
        "env_device": "cpu",
        "ring_device": "cpu",
        "learner_device": "mps",
        "staging_policy": "cpu_ring_explicit_learner_device_copy",
        "transport_table": placement.transport_spec.manifest(),
    }
    assert manifest["inference_timing"] == {
        "timer_kind": "synchronized_wall_clock",
        "has_timing_events": False,
        "persistent_timing_event_count": 0,
        "classification": "capability",
    }
    assert _FakePipeline.close_calls == 1


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason=_MPS_SKIP_REASON)
def test_mps_learner_serves_cpu_ring_inference_with_device_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serving path for the CPU env + torch.mps learner cell.

    Ring observations stay on the host, the actor forward runs on mps behind
    the torch.mps.synchronize() timing boundary (torch.mps has no CUDA-event
    equivalent), and actions cross back through the persistent host staging
    tensor into the CPU ring.
    """
    import unilab.rl.runtime.offpolicy.actor_adapter as actor_adapter_module

    class _DummyMpsActor:
        def explore(
            self,
            obs: torch.Tensor,
            priv_info: torch.Tensor,
            deterministic: bool = False,
        ) -> torch.Tensor:
            assert obs.device.type == "mps"
            assert priv_info.device.type == "mps"
            assert not deterministic
            return obs[:, :2] + priv_info

    monkeypatch.setitem(
        actor_adapter_module._ADAPTERS,
        "dummy_mps_sac",
        actor_adapter_module.OffPolicyActorAdapter(
            algo_type="dummy_mps_sac",
            sample_actions=lambda actor, obs, dones, priv: actor.explore(
                obs, priv, deterministic=False
            ),
            actor_context_from_obs=lambda obs_device, obs_dim: obs_device[:, obs_dim:],
        ),
    )

    actor = _DummyMpsActor()
    actor_input = torch.arange(10, dtype=torch.float32).reshape(2, 5) / 10.0
    dones = torch.zeros(2)
    expected = actor_input[:, :2] + actor_input[:, 3:]

    runner = object.__new__(device_runner_module.DoubleBufferOffPolicyRunner)
    runner.device = "mps"
    runner.obs_dim = 3
    runner.obs_normalization = False
    runner.algo_type = "dummy_mps_sac"
    runner.learner = SimpleNamespace(actor=actor)
    runner.inference_epoch = 0
    runner._inference_forward_timer = device_runner_module.build_device_timer("mps")
    runner.inference_placement = InferencePlacement(
        mode=InferenceTransport.HOST,
        env_device="cpu",
        ring_device="cpu",
        learner_device="mps",
        collector_tensor_native=False,
        staging_policy="cpu_ring_explicit_learner_device_copy",
    )

    synchronize_calls = 0
    real_synchronize = torch.mps.synchronize

    def spying_synchronize() -> None:
        nonlocal synchronize_calls
        synchronize_calls += 1
        real_synchronize()

    monkeypatch.setattr(torch.mps, "synchronize", spying_synchronize)

    slot = SharedInferenceRing(2, 5, 2)
    slot.publish_observation(tick_id=0, observations=actor_input, dones=dones, epoch=0)
    timings = runner._serve_learner_inference(
        slot,
        tick_id=0,
        policy_version=10,
        obs_device=torch.empty(2, 5),
        dones_device=torch.empty(2),
        actor_obs_device=torch.empty(2, 5, device="mps"),
        actor_dones_device=torch.empty(2, device="mps"),
        actions_host=torch.empty(2, 2),
        trace_recorder=None,
    )
    actual, policy_version = slot.consume_action(tick_id=0, epoch=0)

    assert synchronize_calls == 1
    assert actual.device.type == "cpu"
    assert policy_version == 10
    assert timings["inference_forward_time"] > 0.0
    torch.testing.assert_close(actual, expected)


def test_cuda_inference_budget_fails_before_resource_allocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    monkeypatch.setattr(device_runner_module.torch.cuda, "mem_get_info", lambda device: (1, 2))

    def fail_allocation(*args, **kwargs):
        raise AssertionError("inference resources must not be allocated after budget failure")

    monkeypatch.setattr(device_runner_module, "ReplayBuffer", fail_allocation)
    monkeypatch.setattr(device_runner_module, "SharedInferenceRing", fail_allocation)
    runner = _make_device_runner(
        monkeypatch,
        device="cuda",
        collector_tensor_native=True,
        inference_slot_capacity=2,
    )

    with pytest.raises(MemoryError) as excinfo:
        runner.learn(max_iterations=0, save_interval=0, log_dir=str(tmp_path))

    message = str(excinfo.value)
    assert "conservative CUDA tensor runtime (inference + replay) budget" in message
    assert "80% limit" in message
    assert "device free" in message
    assert "training.inference_slot_capacity" in message
    assert "training.replay_ingress_depth" in message
    assert "algo.updates_per_step" in message
    assert not runner._shared_resources


@pytest.mark.parametrize(
    ("collector_tensor_native", "expected_device"),
    [
        (True, "cuda"),
        (False, "cpu"),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_runner_collector_resources_follow_tensor_runtime_capability(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    collector_tensor_native: bool,
    expected_device: str,
):
    _FakePipeline.close_calls = 0
    monkeypatch.setattr(device_runner_module, "ReplayBuffer", _FakeReplayBuffer)
    monkeypatch.setattr(device_runner_module, "DeviceReplayPipeline", _FakePipeline)
    monkeypatch.setattr(device_runner_module, "OffPolicyLogger", _FakeLogger)
    monkeypatch.setattr(device_runner_module.torch, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(device_runner_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        device_runner_module.torch.cuda, "mem_get_info", lambda device: (1 << 30, 2 << 30)
    )

    settings = RuntimeSettings(
        inference_slot_capacity=1,
        collector_metrics_interval=1,
        replay_ingress_depth=3,
        replay_ingress_slot_rows=1,
        batch_size=4,
        updates_per_step=2,
        num_envs=2,
    )
    runner = _make_device_runner(
        monkeypatch,
        env_name="G1MotionTrackingSAC",
        algo_type="flashsac",
        device="cuda",
        sim_backend="mjwarp",
        collector_tensor_native=collector_tensor_native,
        runtime_settings=settings,
    )
    collector_kwargs = {}

    def capture_collector(*, target_fn, kwargs):
        del target_fn
        collector_kwargs.update(kwargs)

    monkeypatch.setattr(runner, "_dp_init_broadcast", lambda: None)
    monkeypatch.setattr(runner, "_prepare_learner", lambda **kwargs: None)
    monkeypatch.setattr(runner, "_start_collector", capture_collector)
    runner.learn(max_iterations=0, save_interval=0, log_dir=str(tmp_path))

    assert _FakeReplayBuffer.last_kwargs["ingress_device"] == expected_device
    assert _FakeReplayBuffer.last_kwargs["ingress_slot_rows"] == 1
    assert _FakeReplayBuffer.last_kwargs["ingress_depth"] == 3
    assert collector_kwargs["inference_slot"].device.type == expected_device
    manifest = runner.runtime_manifest
    assert manifest["collector_tensor_native"] is collector_tensor_native
    assert manifest["inference_ring_capacity"] == 1
    assert manifest["inference_publication_ordering"] == "contiguous_ticks"
    assert manifest["inference_publication_sync"] == (
        "cuda_ipc_events" if collector_tensor_native else "cpu_synchronous"
    )
    assert manifest["inference_epoch"] == 0
    assert manifest["inference_slot_bytes"] == 56
    budget = manifest["inference_memory_budget"]
    assert budget["ipc_event_count"] == (1 if collector_tensor_native else 0) * 3
    assert budget["timing_event_count"] == 2
    assert budget["cuda_event_count"] == budget["ipc_event_count"] + 2
    tensor_budget = manifest["tensor_memory_budget"]
    assert tensor_budget["total"] > 0
    assert tensor_budget["available_bytes"] == 1 << 30
    assert tensor_budget["threshold"] == 0.8
    assert tensor_budget["allowed_bytes"] == int((1 << 30) * 0.8)
    assert tensor_budget["replay_ingress_depth"] == 3
    assert tensor_budget["replay_ingress_slot_rows"] == 1
    assert manifest["collector_metrics_interval"] == 1
    assert manifest["runtime_limits"]["replay_ingress_slot_rows"]["effective"] == 1
    if torch.device(expected_device).type == "cuda":
        expected_device = "cuda:0"
    expected_learner = expected_device
    if not collector_tensor_native:
        expected_learner = "cuda:0"
    assert manifest["learner_device"] == expected_learner
    assert manifest["inference_ring_device"] == expected_device
    assert manifest["env_public_device"] == expected_device
    assert manifest["inference_staging_policy"] == (
        "cuda_no_host_boundary"
        if collector_tensor_native
        else "cpu_ring_explicit_learner_actor_h2d_action_d2h"
    )
    assert manifest["inference_transport"] == {
        "mode": "cuda" if collector_tensor_native else "host",
        "env_device": expected_device,
        "ring_device": expected_device,
        "learner_device": expected_learner,
        "staging_policy": (
            "cuda_no_host_boundary"
            if collector_tensor_native
            else "cpu_ring_explicit_learner_actor_h2d_action_d2h"
        ),
        "transport_table": {
            "ring_device_type": "cuda" if collector_tensor_native else "cpu",
            "requires_tensor_native": collector_tensor_native,
            "requires_rank_local_device_match": collector_tensor_native,
            "publication_sync": (
                "cuda_ipc_events" if collector_tensor_native else "cpu_synchronous"
            ),
            "evidence": "policy",
        },
    }
    env_override = collector_kwargs["env_cfg_override"]
    assert env_override is None or not {
        "tensor_runtime",
        "tensor_runtime_device",
        "inference_transport",
    } & set(env_override)
    assert collector_kwargs["inference_epoch"] == 0
    assert collector_kwargs["collector_metrics_interval"] == 1


def test_runner_rejects_cuda_collector_tensor_device_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        ValueError, match="collector_tensor_device=cuda requires a CUDA learner device"
    ):
        _make_device_runner(
            monkeypatch,
            device="cpu",
            collector_tensor_native=True,
        )


@pytest.mark.parametrize("field", ["inference_slot_capacity", "collector_metrics_interval"])
@pytest.mark.parametrize("value", [0, -1, True, "2", 1.0])
def test_runner_rejects_invalid_tensor_runtime_intervals(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value,
) -> None:
    expected = TypeError if type(value) is not int else ValueError
    with pytest.raises(expected, match=f"{field} must be a positive integer"):
        _make_device_runner(monkeypatch, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("inference_slot_capacity", 17),
        ("collector_metrics_interval", 10_001),
    ],
)
def test_runner_rejects_tensor_runtime_intervals_above_maxima(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: int,
) -> None:
    with pytest.raises(ValueError, match="no greater than"):
        _make_device_runner(monkeypatch, **{field: value})


def test_runner_rejects_conflicting_runtime_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = RuntimeSettings(
        inference_slot_capacity=2,
        collector_metrics_interval=3,
        replay_ingress_depth=2,
        replay_ingress_slot_rows=2,
        batch_size=4,
        updates_per_step=2,
        num_envs=2,
    )
    with pytest.raises(ValueError, match="conflict with runtime_settings"):
        _make_device_runner(
            monkeypatch,
            device="cpu",
            inference_slot_capacity=1,
            runtime_settings=settings,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_envs", True),
        ("batch_size", 4.0),
        ("updates_per_step", "2"),
    ],
)
def test_runner_rejects_non_integer_settings_overlaps(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    settings = RuntimeSettings(
        inference_slot_capacity=1,
        collector_metrics_interval=1,
        replay_ingress_depth=2,
        replay_ingress_slot_rows=2,
        batch_size=4,
        updates_per_step=2,
        num_envs=2,
    )
    with pytest.raises(TypeError, match=f"{field} must be a positive integer"):
        _make_device_runner(
            monkeypatch,
            device="cpu",
            runtime_settings=settings,
            **{field: value},
        )


@pytest.mark.parametrize("value", [-1, True, "0"])
def test_runner_rejects_invalid_inference_epoch(
    monkeypatch: pytest.MonkeyPatch,
    value,
) -> None:
    expected = TypeError if isinstance(value, bool) or not isinstance(value, int) else ValueError
    with pytest.raises(expected, match="inference_epoch must be"):
        _make_device_runner(monkeypatch, inference_epoch=value)


def test_runner_manifest_and_collector_forward_ring_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _FakePipeline.close_calls = 0
    monkeypatch.setattr(device_runner_module, "ReplayBuffer", _FakeReplayBuffer)
    monkeypatch.setattr(device_runner_module, "DeviceReplayPipeline", _FakePipeline)
    monkeypatch.setattr(device_runner_module, "OffPolicyLogger", _FakeLogger)
    monkeypatch.setattr(device_runner_module.torch, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(device_runner_module.time, "sleep", lambda seconds: None)
    runner = _make_device_runner(
        monkeypatch,
        device="cpu",
        inference_slot_capacity=3,
        inference_epoch=2,
        collector_metrics_interval=4,
    )
    collector_kwargs = {}

    monkeypatch.setattr(runner, "_dp_init_broadcast", lambda: None)
    monkeypatch.setattr(runner, "_prepare_learner", lambda **kwargs: None)

    def capture_collector(*, target_fn, kwargs):
        del target_fn
        collector_kwargs.update(kwargs)

    monkeypatch.setattr(runner, "_start_collector", capture_collector)
    runner.learn(max_iterations=0, save_interval=0, log_dir=str(tmp_path))

    assert _FakeReplayBuffer.last_kwargs["ingress_device"] == "cpu"
    assert collector_kwargs["inference_epoch"] == 2
    assert collector_kwargs["collector_metrics_interval"] == 4
    assert collector_kwargs["inference_slot"].diagnostics["capacity"] == 3
    manifest = runner.runtime_manifest
    assert manifest["inference_ring_capacity"] == 3
    assert manifest["inference_publication_ordering"] == "contiguous_ticks"
    assert manifest["inference_publication_sync"] == "cpu_synchronous"
    assert manifest["inference_epoch"] == 2
    assert manifest["inference_slot_bytes"] == 168
    assert manifest["collector_metrics_interval"] == 4


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
        supports_deferred_update_metrics = True

        def update_critic(self, batch, *, read_metrics: bool = True):
            del batch
            events.append("update_critic")
            return {"Loss/critic": 7.0} if read_metrics else {}

        def update_actor(self, batch, *, read_metrics: bool = True):
            del batch
            events.append("update_actor")
            return {}

        def read_deferred_actor_metrics(self):
            return {"Loss/actor": 3.0}

        def soft_update_target(self):
            events.append("soft_update_target")

    monkeypatch.setattr(device_runner_module, "ReplayBuffer", LoopReplayBuffer)
    monkeypatch.setattr(device_runner_module, "DeviceReplayPipeline", LoopPipeline)
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
    assert "soft_update_target" in events

    logger = _FakeLogger.last_instance
    assert logger is not None
    log_step = logger.step_calls[0]
    assert "checkpoint_return_mean_reports10" not in log_step
    deferred_metrics = log_step["metrics"]
    normalize_metric_map(deferred_metrics)
    assert deferred_metrics["Loss/critic"] == pytest.approx(7.0)
    assert deferred_metrics["Loss/actor"] == pytest.approx(3.0)


def test_runner_reads_deferred_update_metrics_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    critic_read_flags: list[bool] = []
    deferred_reads: list[str] = []

    class ConsolidatedReplayBuffer(_FakeReplayBuffer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.ptr[0] = 4
            self.size[0] = 4
            self.published_ptr = 4

    class ConsolidatedPipeline(_FakePipeline):
        last_incremental_h2d_time_s = 0.0

        def progress(self, *, wait=False):
            return wait

        def start_prepare(self, tick_id, sample_count, min_snapshot_ptr=None):
            del tick_id, sample_count, min_snapshot_ptr
            return True

        def batch_ready(self, tick_id, sample_count):
            del tick_id, sample_count
            return True

        def sample_large_batch(self, tick_id, sample_count):
            del tick_id, sample_count
            return {}

        def after_tick(self):
            return None

    class ConsolidatedLearner(_Learner):
        supports_deferred_update_metrics = True

        def update_critic(self, batch, *, read_metrics: bool = True):
            del batch
            critic_read_flags.append(read_metrics)
            return {}

        def update_actor(self, batch, *, read_metrics: bool = True):
            del batch, read_metrics
            return {}

        def read_deferred_update_metrics(self):
            deferred_reads.append("combined")
            return {"Loss/critic": 7.0, "Loss/actor": 3.0}

        def read_deferred_actor_metrics(self):
            raise AssertionError("combined metric read must replace the actor-only read")

        def soft_update_target(self):
            return None

    monkeypatch.setattr(device_runner_module, "ReplayBuffer", ConsolidatedReplayBuffer)
    monkeypatch.setattr(device_runner_module, "DeviceReplayPipeline", ConsolidatedPipeline)
    monkeypatch.setattr(device_runner_module, "OffPolicyLogger", _FakeLogger)
    monkeypatch.setattr(device_runner_module.torch, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(device_runner_module.time, "sleep", lambda seconds: None)

    runner = _make_device_runner(monkeypatch, ConsolidatedLearner(), device="cpu")
    monkeypatch.setattr(runner, "_start_collector", lambda **kwargs: None)
    monkeypatch.setattr(runner, "_check_collector_alive", lambda: True)
    monkeypatch.setattr(runner, "_wait_for_inference_request", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        runner,
        "_serve_learner_inference",
        lambda *args, **kwargs: {
            "inference_h2d_time": 0.0,
            "inference_forward_time": 0.0,
            "inference_d2h_time": 0.0,
            "inference_time": 0.0,
        },
    )
    monkeypatch.setattr(runner, "_publish_inference_response", lambda *args, **kwargs: None)

    runner.learn(max_iterations=1, save_interval=0, log_dir=str(tmp_path))

    assert critic_read_flags == [False, False]
    assert deferred_reads == ["combined"]
    logger = _FakeLogger.last_instance
    assert logger is not None
    metrics = logger.step_calls[0]["metrics"]
    normalize_metric_map(metrics)
    assert metrics["Loss/critic"] == pytest.approx(7.0)
    assert metrics["Loss/actor"] == pytest.approx(3.0)


def test_runner_deferred_finite_failure_precedes_update_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    save_calls: list[object] = []

    class FiniteReplayBuffer(_FakeReplayBuffer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.ptr[0] = 4
            self.size[0] = 4
            self.published_ptr = 4

    class FinitePipeline(_FakePipeline):
        last_incremental_h2d_time_s = 0.0

        def progress(self, *, wait=False):
            return wait

        def start_prepare(self, tick_id, sample_count, min_snapshot_ptr=None):
            del tick_id, sample_count, min_snapshot_ptr
            return True

        def batch_ready(self, tick_id, sample_count):
            del tick_id, sample_count
            return True

        def sample_large_batch(self, tick_id, sample_count):
            del tick_id, sample_count
            return {}

        def after_tick(self):
            return None

    class FiniteLearner(_Learner):
        supports_deferred_update_metrics = True

        def update_critic(self, batch, *, read_metrics: bool = True):
            del batch
            assert read_metrics is False
            return {}

        def update_actor(self, batch, *, read_metrics: bool = True):
            del batch, read_metrics
            return {}

        def begin_deferred_update_metrics(self):
            return None

        def read_deferred_update_metrics(self):
            raise FloatingPointError("non-finite critic, alpha, or actor loss")

        def soft_update_target(self):
            return None

    monkeypatch.setattr(device_runner_module, "ReplayBuffer", FiniteReplayBuffer)
    monkeypatch.setattr(device_runner_module, "DeviceReplayPipeline", FinitePipeline)
    monkeypatch.setattr(device_runner_module, "OffPolicyLogger", _FakeLogger)
    monkeypatch.setattr(
        device_runner_module.torch, "save", lambda *args, **kwargs: save_calls.append(args)
    )
    monkeypatch.setattr(device_runner_module.time, "sleep", lambda seconds: None)

    runner = _make_device_runner(monkeypatch, FiniteLearner(), device="cpu")
    monkeypatch.setattr(runner, "_start_collector", lambda **kwargs: None)
    monkeypatch.setattr(runner, "_check_collector_alive", lambda: True)
    monkeypatch.setattr(runner, "_wait_for_inference_request", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        runner,
        "_serve_learner_inference",
        lambda *args, **kwargs: {
            "inference_h2d_time": 0.0,
            "inference_forward_time": 0.0,
            "inference_d2h_time": 0.0,
            "inference_time": 0.0,
        },
    )
    monkeypatch.setattr(runner, "_publish_inference_response", lambda *args, **kwargs: None)

    with pytest.raises(FloatingPointError, match="non-finite critic"):
        runner.learn(max_iterations=1, save_interval=1, log_dir=str(tmp_path))

    assert runner.learner.update_count == 0
    assert save_calls == []


def test_runner_uses_learner_update_cycle_when_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    events: list[str] = []
    large_batches: list[dict[str, torch.Tensor]] = []

    class CycleReplayBuffer(_FakeReplayBuffer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.ptr[0] = 4
            self.size[0] = 4
            self.published_ptr = 4

    class CyclePipeline(_FakePipeline):
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
            del tick_id
            events.append("replay_sample")
            batch = {"obs": torch.empty(sample_count, 4)}
            large_batches.append(batch)
            return batch

        def after_tick(self):
            events.append("replay_after_tick")

    class CycleLearner(_Learner):
        supports_deferred_update_metrics = True
        use_update_cycle = True
        calls: list[dict[str, object]] = []

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

        def update_cycle(self, large_batch, **kwargs):
            self.calls.append({"batch": large_batch, **kwargs})
            events.append("update_cycle")

        def read_deferred_cycle_metrics(self):
            events.append("cycle_metrics")
            return {}

    monkeypatch.setattr(device_runner_module, "ReplayBuffer", CycleReplayBuffer)
    monkeypatch.setattr(device_runner_module, "DeviceReplayPipeline", CyclePipeline)
    monkeypatch.setattr(device_runner_module, "OffPolicyLogger", _FakeLogger)
    monkeypatch.setattr(device_runner_module.torch, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(device_runner_module.time, "sleep", lambda seconds: None)

    learner = CycleLearner()
    runner = _make_device_runner(monkeypatch, learner)
    runner.device = "cpu"
    monkeypatch.setattr(runner, "_start_collector", lambda **kwargs: None)
    monkeypatch.setattr(runner, "_check_collector_alive", lambda: True)
    monkeypatch.setattr(runner, "_wait_for_inference_request", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        runner,
        "_serve_learner_inference",
        lambda *args, **kwargs: {
            "inference_h2d_time": 0.0,
            "inference_forward_time": 0.0,
            "inference_d2h_time": 0.0,
            "inference_time": 0.0,
        },
    )
    monkeypatch.setattr(runner, "_publish_inference_response", lambda *args, **kwargs: None)

    runner.learn(max_iterations=1, save_interval=0, log_dir=str(tmp_path))

    assert learner.calls == [
        {
            "batch": large_batches[0],
            "updates_per_step": 2,
            "policy_frequency": 1,
            "target_frequency": 1,
            "policy_before_critic": False,
            "read_metrics": False,
        }
    ]
    assert "update_critic" not in events
    assert "update_actor" not in events
    cycle_idx = events.index("update_cycle")
    assert events[cycle_idx + 1 :].count("replay_progress") == 2
    assert events[-3:] == ["replay_progress", "cycle_metrics", "replay_after_tick"]


def test_runtime_manifest_reports_inductor_cuda_graph_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = _Learner()
    learner.use_compile = True
    learner._host_finite_checks = False

    runner = _make_device_runner(monkeypatch, learner)
    graph_manifest = runner.runtime_manifest["cuda_graph"]
    assert graph_manifest == {
        "backend": "inductor",
        "critic": True,
        "actor": True,
        "device_finite_optimizer_gating": True,
        "scope": "loss_tensors",
        "orchestration": "inductor_cuda_graph_trees",
    }

    learner.use_update_cycle = True
    assert runner._cuda_graph_runtime_manifest() == {
        "backend": "inductor",
        "critic": True,
        "actor": True,
        "device_finite_optimizer_gating": True,
        "scope": "update_cycle",
        "orchestration": "cuda_graph",
    }

    learner.use_update_cycle = False
    learner.use_compile = False
    learner._host_finite_checks = True
    assert runner._cuda_graph_runtime_manifest() == {
        "backend": "eager",
        "critic": False,
        "actor": False,
        "device_finite_optimizer_gating": False,
    }


def test_runtime_manifest_reports_device_capability_and_learner_policy_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unilab.rl.algos.common.performance import resolve_learner_performance_tier

    learner = _Learner()
    learner._performance_tier = resolve_learner_performance_tier("cpu")
    learner.amp_policy = {"enabled": False, "use_amp": False, "evidence": "policy"}
    runner = _make_device_runner(monkeypatch, learner)

    assert runner.runtime_manifest["learner_performance_tier"] == {
        "name": "cpu_compatibility",
        "device_type": "cpu",
        "vendor_runtime": "default",
        "supports_whole_cycle_graph": False,
        "requires_inductor": False,
        "supports_capturable_optimizer": False,
        "evidence": "policy",
    }
    assert runner.runtime_manifest["learner_amp_policy"]["enabled"] is False
    assert runner.runtime_manifest["learner_amp_policy"]["use_amp"] is False
    capabilities = runner.runtime_manifest["device_runtime_capabilities"]
    assert capabilities["sync_api"] == "torch.cuda.synchronize"
    assert capabilities["has_timing_events"] is True


def test_update_phase_sync_preserves_mps_phase_ordered_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _make_device_runner(monkeypatch)
    calls: list[tuple[str, bool]] = []

    def fake_sync(device, *, stream: bool = False) -> None:
        del device
        calls.append(("cuda_stream" if stream else "device", stream))

    monkeypatch.setattr(device_runner_module, "sync_device", fake_sync)

    runner._synchronize_after_update_phase(torch.device("cpu"), phase_ordered_metrics=False)
    runner._synchronize_after_update_phase(torch.device("mps"), phase_ordered_metrics=True)
    assert calls == []

    runner._synchronize_after_update_phase(torch.device("mps"), phase_ordered_metrics=False)
    runner._synchronize_after_update_phase(torch.device("cuda"), phase_ordered_metrics=True)
    assert calls == [("device", False), ("cuda_stream", True)]


def test_runtime_manifest_reports_mps_torch_compile_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = _Learner()
    learner.use_compile = True
    learner.compile_backend = "torch_compile"
    learner._host_finite_checks = True

    runner = _make_device_runner(monkeypatch, learner)

    assert runner.runtime_manifest["cuda_graph"] == {
        "backend": "torch_compile",
        "critic": True,
        "actor": True,
        "device_finite_optimizer_gating": False,
        "scope": "loss_tensors",
        "orchestration": "torch_compile",
    }


def _wait_for_inference_request(runner, inference_queue, *, expected_tick: int) -> int:
    return runner._wait_for_inference_request(
        inference_queue,
        expected_tick=expected_tick,
        replay_pipeline=SimpleNamespace(progress=lambda: None),
        metrics_queue=queue.Queue(),
        reward_history=deque(maxlen=10),
        latest_reward_components={},
        logger=_FakeLogger(),
        trace_recorder=None,
        replay_buffer=SimpleNamespace(),
        ckpt_path=None,
        train_start_wall=0.0,
    )


def test_collector_ready_wait_rejects_early_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _make_device_runner(monkeypatch)
    inference_queue: queue.Queue[int] = queue.Queue()

    def fail_collector_died(*args, **kwargs):
        raise AssertionError("ready wait must not fail while the collector is alive")

    monkeypatch.setattr(runner, "_drain_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_check_collector_alive", lambda: True)
    monkeypatch.setattr(runner, "_fail_collector_died", fail_collector_died)
    inference_queue.put(0)

    with pytest.raises(RuntimeError, match="ready signal"):
        _wait_for_inference_request(runner, inference_queue, expected_tick=0)


def test_collector_request_wait_is_liveness_based_after_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _make_device_runner(monkeypatch)
    inference_queue: queue.Queue[int] = queue.Queue()

    def publish_ready_and_tick() -> None:
        inference_queue.put(-1)
        inference_queue.put(0)

    monkeypatch.setattr(runner, "_drain_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_check_collector_alive", lambda: True)
    threading.Timer(0.03, publish_ready_and_tick).start()

    tick_id = _wait_for_inference_request(runner, inference_queue, expected_tick=0)

    assert tick_id == 0
    assert runner._learner_coordination.snapshot()[0] is LearnerPhase.BUSY
    inference_queue.put(1)
    next_tick_id = _wait_for_inference_request(runner, inference_queue, expected_tick=1)

    assert next_tick_id == 1


def test_collector_ready_wait_detects_dead_collector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _make_device_runner(monkeypatch)
    fail_calls: list[int] = []

    class _EmptyQueue:
        def get(self, timeout=None):
            del timeout
            raise queue.Empty

    def fail_collector_died(*args, **kwargs):
        fail_calls.append(args[3])
        raise RuntimeError("collector died during readiness")

    monkeypatch.setattr(runner, "_drain_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_check_collector_alive", lambda: False)
    monkeypatch.setattr(runner, "_fail_collector_died", fail_collector_died)

    with pytest.raises(RuntimeError, match="collector died during readiness"):
        _wait_for_inference_request(runner, _EmptyQueue(), expected_tick=0)

    assert fail_calls == [0]


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


@pytest.mark.parametrize("algo_type", ["sac", "flashsac"])
def test_learner_inference_matches_existing_actor_exploration(algo_type: str) -> None:
    if algo_type == "sac":
        from unilab.rl.algos.sac.learner import SACActor

        actor = SACActor(3, 2, hidden_dim=8, use_layer_norm=False)
    else:
        from unilab.rl.algos.flash_sac.network import FlashSACActor

        actor = FlashSACActor(num_blocks=1, input_dim=3, hidden_dim=8, action_dim=2)
    expected_actor = copy.deepcopy(actor)
    observations = torch.arange(6, dtype=torch.float32).reshape(2, 3) / 10.0
    dones = torch.tensor([0.0, 1.0], dtype=torch.float32)
    torch.manual_seed(17)
    expected = expected_actor.explore(
        observations,
        dones=dones,
        deterministic=False,
    )

    runner = object.__new__(device_runner_module.DoubleBufferOffPolicyRunner)
    runner.device = "cpu"
    runner.obs_dim = 3
    runner.obs_normalization = False
    runner.algo_type = algo_type
    runner.learner = SimpleNamespace(actor=actor)
    runner.inference_epoch = 0
    runner._inference_forward_timer = device_runner_module.build_device_timer("cpu")
    runner.inference_placement = InferencePlacement(
        mode=InferenceTransport.HOST,
        env_device="cpu",
        ring_device="cpu",
        learner_device="cpu",
        collector_tensor_native=False,
        staging_policy="cpu_no_device_transfer",
    )
    slot = SharedInferenceRing(2, 3, 2)
    slot.publish_observation(tick_id=0, observations=observations, dones=dones, epoch=0)
    torch.manual_seed(17)
    runner._serve_learner_inference(
        slot,
        tick_id=0,
        policy_version=9,
        obs_device=torch.empty(2, 3),
        dones_device=torch.empty(2),
        actor_obs_device=torch.empty(2, 3),
        actor_dones_device=torch.empty(2),
        actions_host=torch.empty(2, 2),
        trace_recorder=None,
    )
    actual, policy_version = slot.consume_action(tick_id=0, epoch=0)

    torch.testing.assert_close(actual, expected)
    assert policy_version == 9
    if algo_type == "flashsac":
        torch.testing.assert_close(actor._noise, expected_actor._noise)
        torch.testing.assert_close(actor._repeat_count, expected_actor._repeat_count)
        torch.testing.assert_close(actor._repeat_target, expected_actor._repeat_target)


def test_adapter_learner_inference_uses_actor_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import unilab.rl.runtime.offpolicy.actor_adapter as actor_adapter_module

    class _DummyPrivInfoActor:
        def explore(
            self,
            obs: torch.Tensor,
            priv_info: torch.Tensor,
            deterministic: bool = False,
        ) -> torch.Tensor:
            assert not deterministic
            return obs[:, :2] + priv_info

    monkeypatch.setitem(
        actor_adapter_module._ADAPTERS,
        "dummy_priv_sac",
        actor_adapter_module.OffPolicyActorAdapter(
            algo_type="dummy_priv_sac",
            sample_actions=lambda actor, obs, dones, priv: actor.explore(
                obs, priv, deterministic=False
            ),
            actor_context_from_obs=lambda obs_device, obs_dim: obs_device[:, obs_dim:],
        ),
    )

    actor = _DummyPrivInfoActor()
    observations = torch.arange(6, dtype=torch.float32).reshape(2, 3) / 10.0
    priv_info = torch.arange(4, dtype=torch.float32).reshape(2, 2) / 10.0
    actor_input = torch.cat((observations, priv_info), dim=1)
    dones = torch.zeros(2)
    expected = actor.explore(observations, priv_info)

    runner = object.__new__(device_runner_module.DoubleBufferOffPolicyRunner)
    runner.device = "cpu"
    runner.obs_dim = 3
    runner.obs_normalization = False
    runner.algo_type = "dummy_priv_sac"
    runner.learner = SimpleNamespace(actor=actor)
    runner.inference_epoch = 0
    runner._inference_forward_timer = device_runner_module.build_device_timer("cpu")
    runner.inference_placement = InferencePlacement(
        mode=InferenceTransport.HOST,
        env_device="cpu",
        ring_device="cpu",
        learner_device="cpu",
        collector_tensor_native=False,
        staging_policy="cpu_no_device_transfer",
    )
    slot = SharedInferenceRing(2, 5, 2)
    slot.publish_observation(tick_id=0, observations=actor_input, dones=dones, epoch=0)
    runner._serve_learner_inference(
        slot,
        tick_id=0,
        policy_version=10,
        obs_device=torch.empty(2, 5),
        dones_device=torch.empty(2),
        actor_obs_device=torch.empty(2, 5),
        actor_dones_device=torch.empty(2),
        actions_host=torch.empty(2, 2),
        trace_recorder=None,
    )
    actual, policy_version = slot.consume_action(tick_id=0, epoch=0)

    torch.testing.assert_close(actual, expected)
    assert policy_version == 10


def test_dp_metric_reduction_follows_the_metric_schema() -> None:
    class _FakeDpSync:
        def __init__(self) -> None:
            self.mean: dict[str, float] | None = None
            self.total: dict[str, float] | None = None

        def allreduce_statistics(self, *, mean, total):
            self.mean = dict(mean)
            self.total = dict(total)
            return {
                "metric::Loss/critic": 4.0,
                "metric::Train/rollouts_read": 6.0,
                "checkpoint::return_reports10": 5.0,
                **{key: value for key, value in mean.items() if key.startswith("timing::")},
                "logger::total_steps": 128.0,
                "logger::buffer_size": 64.0,
                "logger::buffer_target": 128.0,
                "logger::timeout_rate": 0.0,
                "logger::buffer_utilization": 0.5,
                "extra::batch_size_per_rank": 8.0,
                "extra::effective_batch_size": 8.0,
                "extra::throughput_steps": 128.0,
                "extra::learner_replay_rows_per_iter": 16.0,
            }

    runner = object.__new__(device_runner_module.DoubleBufferOffPolicyRunner)
    dp_sync = _FakeDpSync()
    runner.dp_sync = dp_sync
    logger = SimpleNamespace(
        _total_steps=64,
        _buffer_size=32,
        _buffer_target=64,
        _mean_ep_length=0.0,
        _timeout_rate=0.0,
        _buffer_utilization=0.5,
        _collector_timing={},
    )

    payload = runner._aggregate_log_statistics(
        logger,
        metrics={"Loss/critic": 4.0, "Train/rollouts_read": 3.0},
        checkpoint_return_mean_reports10=2.0,
        return_mean_ep100=None,
        reward_components={},
        train_time=0.1,
        collector_wait_time=0.0,
        replay_batch_wait_time=0.0,
        learner_replay_sample_time=0.0,
        sync_coordination_time=0.0,
        replay_ingress_h2d_submit_time=0.0,
        inference_h2d_time=0.0,
        inference_forward_time=0.0,
        inference_d2h_time=0.0,
        inference_time=0.0,
        iteration_time=1.0,
        extra_info={
            "throughput_steps": 64,
            "batch_size_per_rank": 8,
            "effective_batch_size": 8,
            "learner_replay_rows_per_iter": 8,
        },
    )

    assert dp_sync.mean is not None and dp_sync.total is not None
    assert "metric::Loss/critic" in dp_sync.mean
    assert "metric::Train/rollouts_read" not in dp_sync.mean
    assert dp_sync.total["metric::Train/rollouts_read"] == 3.0
    assert payload["metrics"]["Train/rollouts_read"] == pytest.approx(6.0)
    assert payload["checkpoint_return_mean_reports10"] == pytest.approx(5.0)


def test_collector_override_never_carries_removed_tensor_runtime_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _make_device_runner(
        monkeypatch,
        env_name="G1MotionTrackingSAC",
        algo_type="flashsac",
        device="cuda:0",
        sim_backend="mjwarp",
        collector_tensor_native=True,
    )

    override = runner._collector_env_cfg_override()

    assert override is None or not {
        "tensor_runtime",
        "tensor_runtime_device",
        "inference_transport",
    } & set(override)
