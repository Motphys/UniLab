from __future__ import annotations

import json
import queue
from typing import cast

import pytest
import torch

import unilab.algos.torch.offpolicy.double_buffer_runner as double_buffer_runner_module
import unilab.algos.torch.offpolicy.multi_gpu_runner as multi_gpu_runner_module
import unilab.algos.torch.offpolicy.runner as runner_module
from unilab.algos.torch.offpolicy.runner import (
    OffPolicyRunner,
    compute_train_start_threshold,
    replay_buffer_ready_for_learning,
)
from unilab.ipc.replay_buffer import ReplayBuffer


class _FakeActor:
    def __init__(self) -> None:
        self._param = torch.nn.Parameter(torch.zeros(1))
        self._state = {"weight": torch.zeros(1)}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value.clone() for key, value in self._state.items()}

    def parameters(self):
        return [self._param]


class _FakeLearner:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.actor = _FakeActor()
        self.qnet = _FakeActor()
        self.log_alpha = torch.zeros(1)
        self.reward_normalizer = None
        self.update_count = 0
        self.critic_updates = 0
        self.actor_updates = 0
        self.target_updates = 0

    def update_critic(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.critic_updates += 1
        return {"critic_loss": float(batch["obs"].shape[0])}

    def update_actor(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.actor_updates += 1
        return {"actor_loss": float(batch["obs"].shape[0])}

    def soft_update_target(self) -> None:
        self.target_updates += 1

    def get_state_dict(self) -> dict[str, int]:
        return {"update_count": self.update_count}


class _FakeReplayBuffer:
    last_instance: "_FakeReplayBuffer | None" = None

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: str,
        critic_dim: int = 0,
        defer_gpu: bool = False,
        packed_cpu_storage: bool = False,
    ):
        del defer_gpu, packed_cpu_storage
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.critic_dim = critic_dim
        self.size = torch.zeros(1, dtype=torch.int64)
        self.ptr = torch.zeros(1, dtype=torch.int64)
        self.collect_time_s = torch.zeros(1, dtype=torch.float32)
        self._storage = torch.zeros(capacity, 16)
        self.sample_calls = 0
        self.sample_request_sizes: list[int] = []
        self.sample_sizes_at_call: list[int] = []
        _FakeReplayBuffer.last_instance = self

    def init_local_gpu_cache(self, device: str) -> None:
        del device

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        self.sample_calls += 1
        self.sample_request_sizes.append(batch_size)
        self.sample_sizes_at_call.append(int(self.size[0]))
        return {
            "obs": torch.zeros(batch_size, 4),
            "actions": torch.zeros(batch_size, 2),
            "rewards": torch.zeros(batch_size),
            "next_obs": torch.zeros(batch_size, 4),
            "dones": torch.zeros(batch_size),
            "truncated": torch.zeros(batch_size),
        }

    def close(self) -> None:
        pass


class _FakeWeightSync:
    last_instance: "_FakeWeightSync | None" = None

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.name = "fake-weight-sync"
        self._lock = None
        self.version = 0
        self.write_calls = 0
        self.read_calls = 0

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, torch.Tensor], create: bool = True
    ) -> "_FakeWeightSync":
        del state_dict, create
        instance = cls()
        cls.last_instance = instance
        return instance

    def write_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        del state_dict
        self.write_calls += 1
        self.version += 1

    def read_weights_into(self, state_dict: dict[str, torch.Tensor]) -> int:
        del state_dict
        self.read_calls += 1
        return self.version

    def close(self) -> None:
        pass


class _FakeLogger:
    last_instance: "_FakeLogger | None" = None

    def __init__(self, **kwargs) -> None:
        del kwargs
        self.buffer_fill_calls: list[tuple[int, int]] = []
        self.step_calls: list[dict] = []
        self.finish_calls = 0
        self.close_calls = 0
        self._total_steps = 0
        self._mean_ep_length = 0.0
        _FakeLogger.last_instance = self

    def set_collection_sync(self, enabled: bool, env_steps_per_sync: int) -> None:
        del enabled, env_steps_per_sync

    def start(self) -> None:
        pass

    def log_status(self, status: str) -> None:
        del status

    def finish(self, *args, **kwargs) -> None:
        del args, kwargs
        self.finish_calls += 1

    def close(self) -> None:
        self.close_calls += 1

    def log_buffer_fill(self, current: int, target: int) -> None:
        self.buffer_fill_calls.append((current, target))

    def update_buffer_utilization(self, utilization: float) -> None:
        del utilization

    def update_ep_length(self, mean_ep_length: float) -> None:
        self._mean_ep_length = mean_ep_length

    def update_collector_timing(self, timing_ms: dict[str, float]) -> None:
        del timing_ms

    def update_done_rates(self, timeout_rate: float, terminated_rate: float) -> None:
        del timeout_rate, terminated_rate

    def log_collector(self, total_steps: int, buffer_size: int, mean_reward: float = 0.0) -> None:
        del buffer_size, mean_reward
        self._total_steps = total_steps

    def log_step(self, **kwargs) -> None:
        self.step_calls.append(kwargs)

    def log_save(self, ckpt_path: str) -> None:
        del ckpt_path


class _SyncReadyQueue:
    def __init__(
        self,
        replay_buffer: _FakeReplayBuffer,
        sizes: list[int],
        collect_times: list[float] | None = None,
    ) -> None:
        self._replay_buffer = replay_buffer
        self._sizes = list(sizes)
        self._collect_times = list(collect_times or [0.25] * len(sizes))
        self.get_calls = 0

    def get(self, timeout: float | None = None) -> int:
        del timeout
        assert self._sizes, "collection_ready_queue exhausted before learner started"
        size = self._sizes.pop(0)
        collect_time = self._collect_times.pop(0) if self._collect_times else 0.0
        self._replay_buffer.size[0] = size
        self._replay_buffer.ptr[0] = size
        self._replay_buffer.collect_time_s[0] = collect_time
        self.get_calls += 1
        return 1


class _InterruptingReadyQueue:
    def get(self, timeout: float | None = None) -> int:
        del timeout
        raise KeyboardInterrupt


class _RecordingQueue:
    def __init__(self) -> None:
        self.put_calls: list[int] = []

    def put(self, item: int) -> None:
        self.put_calls.append(item)


class _FakeProcess:
    def __init__(self) -> None:
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        del timeout
        self._alive = False

    def terminate(self) -> None:
        self._alive = False


class _FakeClock:
    def __init__(self, values: list[float]) -> None:
        self._values = list(values)
        self._index = 0

    def time(self) -> float:
        if self._index < len(self._values):
            value = self._values[self._index]
            self._index += 1
            return value
        return self._values[-1]


@pytest.fixture(autouse=True)
def _reset_fakes() -> None:
    _FakeReplayBuffer.last_instance = None
    _FakeWeightSync.last_instance = None
    _FakeLogger.last_instance = None


@pytest.mark.parametrize(
    ("batch_size", "learning_starts", "num_envs", "expected"),
    [(8, 0, 2, 8), (8, 1, 2, 8), (8, 6, 2, 12), (8, -5, 2, 8)],
)
def test_compute_train_start_threshold(
    batch_size: int, learning_starts: int, num_envs: int, expected: int
) -> None:
    assert compute_train_start_threshold(batch_size, learning_starts, num_envs) == expected


@pytest.mark.parametrize(
    ("replay_size", "batch_size", "learning_starts", "num_envs", "expected"),
    [(7, 8, 0, 2, False), (8, 8, 0, 2, True), (11, 8, 6, 2, False), (12, 8, 6, 2, True)],
)
def test_replay_buffer_ready_for_learning(
    replay_size: int, batch_size: int, learning_starts: int, num_envs: int, expected: bool
) -> None:
    assert (
        replay_buffer_ready_for_learning(
            replay_size,
            batch_size=batch_size,
            learning_starts=learning_starts,
            num_envs=num_envs,
        )
        is expected
    )


def _make_runner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sync_collection: bool,
    algo_type: str = "sac",
    updates_per_step: int = 1,
    policy_frequency: int = 1,
    trace_enabled: bool = False,
    trace_output_dir: str | None = None,
) -> OffPolicyRunner:
    monkeypatch.setattr(runner_module, "ReplayBuffer", _FakeReplayBuffer)
    monkeypatch.setattr(runner_module, "SharedWeightSync", _FakeWeightSync)
    monkeypatch.setattr(runner_module, "OffPolicyLogger", _FakeLogger)
    monkeypatch.setattr(runner_module, "get_env_dims", lambda *args, **kwargs: (4, 2, 0))
    monkeypatch.setattr(runner_module.torch, "save", lambda *args, **kwargs: None)

    learner = _FakeLearner()
    runner = OffPolicyRunner(
        learner=learner,
        env_name="DummyEnv",
        algo_type=algo_type,
        num_envs=2,
        replay_buffer_n=8,
        batch_size=8,
        learning_starts=6,
        updates_per_step=updates_per_step,
        policy_frequency=policy_frequency,
        sync_collection=sync_collection,
        env_steps_per_sync=1,
        device="cpu",
        trace_enabled=trace_enabled,
        trace_output_dir=trace_output_dir,
    )
    monkeypatch.setattr(runner, "_start_collector", lambda *args, **kwargs: None)
    runner._collector_process = _FakeProcess()
    return runner


def test_offpolicy_runner_sync_waits_for_train_start_threshold(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    runner = _make_runner(monkeypatch, sync_collection=True)
    threshold = runner.train_start_threshold
    created_queues: list[object] = []
    fake_clock = _FakeClock([100.0, 100.1, 100.2, 110.2, 110.3, 110.8, 111.0, 111.1])

    def queue_factory(maxsize: int = 0):
        del maxsize
        idx = len(created_queues)
        queue_obj: object
        if idx == 0:
            replay_buffer = _FakeReplayBuffer.last_instance
            assert replay_buffer is not None
            queue_obj = _SyncReadyQueue(replay_buffer, [4, 8, threshold], [0.1, 0.2, 0.25])
        elif idx == 1:
            queue_obj = _RecordingQueue()
        else:
            queue_obj = queue.Queue()
        created_queues.append(queue_obj)
        return queue_obj

    monkeypatch.setattr(runner_module._SPAWN_CTX, "Queue", queue_factory)
    monkeypatch.setattr(runner_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(runner_module.time, "time", fake_clock.time)

    runner.learn(max_iterations=1, save_interval=0, log_dir=str(tmp_path))

    replay_buffer = _FakeReplayBuffer.last_instance
    trainer_done_queue = created_queues[1]
    ready_queue = created_queues[0]
    logger = _FakeLogger.last_instance
    assert replay_buffer is not None
    assert isinstance(trainer_done_queue, _RecordingQueue)
    assert isinstance(ready_queue, _SyncReadyQueue)
    assert logger is not None
    assert ready_queue.get_calls == 3
    assert replay_buffer.sample_calls == 1
    assert replay_buffer.sample_sizes_at_call == [threshold]
    assert trainer_done_queue.put_calls == [1, 1, 1, 1]
    assert logger.step_calls and logger.step_calls[0]["iteration"] == 1
    assert logger.step_calls[0]["collect_time"] == pytest.approx(0.25)
    assert logger.step_calls[0]["extra_info"]["startup_wait_time"] == pytest.approx(9.75)
    assert logger.step_calls[0]["extra_info"]["throughput_steps"] == 2
    assert _FakeWeightSync.last_instance is not None
    assert _FakeWeightSync.last_instance.write_calls == 1


def test_offpolicy_runner_close_releases_active_logger_after_interrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    runner = _make_runner(monkeypatch, sync_collection=True)
    created_queues: list[object] = []

    def queue_factory(maxsize: int = 0):
        del maxsize
        idx = len(created_queues)
        if idx == 0:
            queue_obj: object = _InterruptingReadyQueue()
        elif idx == 1:
            queue_obj = _RecordingQueue()
        else:
            queue_obj = queue.Queue()
        created_queues.append(queue_obj)
        return queue_obj

    monkeypatch.setattr(runner_module._SPAWN_CTX, "Queue", queue_factory)
    monkeypatch.setattr(runner_module.time, "sleep", lambda seconds: None)

    with pytest.raises(KeyboardInterrupt):
        runner.learn(max_iterations=1, save_interval=0, log_dir=str(tmp_path))

    logger = _FakeLogger.last_instance
    assert logger is not None
    assert runner._active_logger is logger
    assert logger.finish_calls == 0
    assert logger.close_calls == 0

    runner.close()

    assert runner._active_logger is None
    assert logger.close_calls == 1


def test_offpolicy_runner_async_waits_for_train_start_threshold(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    runner = _make_runner(monkeypatch, sync_collection=False)
    threshold = runner.train_start_threshold
    sleep_sizes = iter([4, 8, threshold])
    fake_clock = _FakeClock([200.0, 200.1, 200.2, 210.2, 210.3, 210.8, 211.0, 211.1])

    def fake_sleep(seconds: float) -> None:
        if seconds < 0.5:
            next_size = next(sleep_sizes, threshold)
            replay_buffer = _FakeReplayBuffer.last_instance
            assert replay_buffer is not None
            replay_buffer.size[0] = next_size
            replay_buffer.ptr[0] = next_size
            replay_buffer.collect_time_s[0] = 0.25

    monkeypatch.setattr(runner_module._SPAWN_CTX, "Queue", lambda maxsize=0: queue.Queue())
    monkeypatch.setattr(runner_module.time, "sleep", fake_sleep)
    monkeypatch.setattr(runner_module.time, "time", fake_clock.time)

    runner.learn(max_iterations=1, save_interval=0, log_dir=str(tmp_path))

    replay_buffer = _FakeReplayBuffer.last_instance
    logger = _FakeLogger.last_instance
    assert replay_buffer is not None
    assert logger is not None
    assert replay_buffer.sample_calls == 1
    assert replay_buffer.sample_sizes_at_call == [threshold]
    assert logger.step_calls and logger.step_calls[0]["iteration"] == 1
    assert logger.step_calls[0]["collect_time"] == pytest.approx(0.25)
    assert logger.step_calls[0]["extra_info"]["startup_wait_time"] == pytest.approx(9.75)
    assert logger.step_calls[0]["extra_info"]["throughput_steps"] == 2


def test_offpolicy_runner_trace_writes_perfetto_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    runner = _make_runner(
        monkeypatch,
        sync_collection=False,
        trace_enabled=True,
        trace_output_dir=str(tmp_path),
    )
    threshold = runner.train_start_threshold
    sleep_sizes = iter([4, 8, threshold])

    def fake_sleep(seconds: float) -> None:
        if seconds < 0.5:
            next_size = next(sleep_sizes, threshold)
            replay_buffer = _FakeReplayBuffer.last_instance
            assert replay_buffer is not None
            replay_buffer.size[0] = next_size
            replay_buffer.ptr[0] = next_size

    monkeypatch.setattr(runner_module._SPAWN_CTX, "Queue", lambda maxsize=0: queue.Queue())
    monkeypatch.setattr(runner_module.time, "sleep", fake_sleep)

    runner.learn(max_iterations=1, save_interval=0, log_dir=str(tmp_path / "logs"))

    trace_path = tmp_path / "perfetto_offpolicy_timeline.json"
    assert trace_path.exists()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    names = {event.get("name") for event in payload["traceEvents"]}
    assert "learner/wait_for_data" in names
    assert "learner/replay_sample" in names
    assert "learner/update_critic" in names
    assert "learner/weight_sync_write" in names
    assert "learner/training_e2e" in names


def test_td3_offpolicy_runner_trace_writes_core_learner_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    runner = _make_runner(
        monkeypatch,
        sync_collection=False,
        algo_type="td3",
        updates_per_step=3,
        policy_frequency=2,
        trace_enabled=True,
        trace_output_dir=str(tmp_path),
    )
    threshold = runner.train_start_threshold
    sleep_sizes = iter([4, 8, threshold])

    def fake_sleep(seconds: float) -> None:
        if seconds < 0.5:
            next_size = next(sleep_sizes, threshold)
            replay_buffer = _FakeReplayBuffer.last_instance
            assert replay_buffer is not None
            replay_buffer.size[0] = next_size
            replay_buffer.ptr[0] = next_size

    monkeypatch.setattr(runner_module._SPAWN_CTX, "Queue", lambda maxsize=0: queue.Queue())
    monkeypatch.setattr(runner_module.time, "sleep", fake_sleep)

    runner.learn(max_iterations=1, save_interval=0, log_dir=str(tmp_path / "logs"))

    trace_path = tmp_path / "perfetto_offpolicy_timeline.json"
    assert trace_path.exists()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    names = {event.get("name") for event in payload["traceEvents"]}
    assert "learner/wait_for_data" in names
    assert "learner/replay_sample" in names
    assert "learner/update_critic" in names
    assert "learner/update_actor" in names
    assert "learner/soft_update_target" in names
    assert "learner/weight_sync_write" in names
    assert "learner/training_e2e" in names

    learner = cast(_FakeLearner, runner.learner)
    assert learner.critic_updates == 3
    assert learner.actor_updates == 2
    assert learner.target_updates == 2

    target_events = [
        event
        for event in payload["traceEvents"]
        if event.get("name") == "learner/soft_update_target"
    ]
    assert [event["args"]["actor_updated"] for event in target_events] == [True, False, True]
    assert [event["args"]["target_updated"] for event in target_events] == [True, False, True]
    assert {event["args"]["algo_type"] for event in target_events} == {"td3"}
    assert {event["args"]["policy_frequency"] for event in target_events} == {2}


class _FakeDoubleBufferPipeline:
    def __init__(
        self,
        replay_buffer,
        *,
        device,
        sample_count,
        base_seed,
        trace_recorder,
        trace_cuda_events,
        verbose,
        verbose_output_dir,
        collector_pack_request_queue,
        collector_pack_ready_queue,
        collector_pack_shared_slots,
    ) -> None:
        del (
            replay_buffer,
            device,
            base_seed,
            trace_cuda_events,
            verbose,
            verbose_output_dir,
            collector_pack_request_queue,
            collector_pack_ready_queue,
            collector_pack_shared_slots,
        )
        self.sample_count = sample_count
        self.trace_recorder = trace_recorder
        self.close_calls = 0

    def start_prepare(self, tick_id: int, sample_count: int, min_snapshot_ptr=None) -> bool:
        del min_snapshot_ptr
        assert sample_count == self.sample_count
        if self.trace_recorder is not None:
            now = double_buffer_runner_module.time.perf_counter_ns()
            self.trace_recorder.add_slice(
                "replay_pipeline/collector_pack_request",
                category="replay_pipeline",
                start_ns=now,
                end_ns=double_buffer_runner_module.time.perf_counter_ns(),
                args={"tick_id": tick_id, "sample_count": sample_count},
            )
            now = double_buffer_runner_module.time.perf_counter_ns()
            self.trace_recorder.add_slice(
                "replay_pipeline/batch_h2d_submit",
                category="replay_pipeline",
                start_ns=now,
                end_ns=double_buffer_runner_module.time.perf_counter_ns(),
                args={"tick_id": tick_id},
            )
            now = double_buffer_runner_module.time.perf_counter_ns()
            self.trace_recorder.add_slice(
                "gpu/replay_pipeline_batch_h2d",
                category="gpu",
                start_ns=now,
                end_ns=double_buffer_runner_module.time.perf_counter_ns(),
                args={"tick_id": tick_id},
            )
        return True

    def batch_ready(self, tick_id: int, sample_count: int) -> bool:
        del tick_id, sample_count
        return False

    def wait_until_ready(self, tick_id: int, sample_count: int) -> bool:
        self.start_prepare(tick_id, sample_count)
        return True

    def sample_large_batch(self, tick_id: int, sample_count: int) -> dict[str, torch.Tensor]:
        assert sample_count == self.sample_count
        if self.trace_recorder is not None:
            now = double_buffer_runner_module.time.perf_counter_ns()
            self.trace_recorder.add_slice(
                "replay_pipeline/batch_h2d_wait",
                category="replay_pipeline",
                start_ns=now,
                end_ns=double_buffer_runner_module.time.perf_counter_ns(),
                args={"tick_id": tick_id},
            )
            now = double_buffer_runner_module.time.perf_counter_ns()
            self.trace_recorder.add_slice(
                "replay_pipeline/gpu_wait_for_batch",
                category="replay_pipeline",
                start_ns=now,
                end_ns=double_buffer_runner_module.time.perf_counter_ns(),
                args={"tick_id": tick_id},
            )
            now = double_buffer_runner_module.time.perf_counter_ns()
            self.trace_recorder.add_slice(
                "replay_pipeline/hot_cold_swap",
                category="replay_pipeline",
                start_ns=now,
                end_ns=double_buffer_runner_module.time.perf_counter_ns(),
                args={"tick_id": tick_id},
            )
        return {
            "obs": torch.zeros(sample_count, 4),
            "actions": torch.zeros(sample_count, 2),
            "rewards": torch.zeros(sample_count),
            "next_obs": torch.zeros(sample_count, 4),
            "dones": torch.zeros(sample_count),
            "truncated": torch.zeros(sample_count),
        }

    def after_tick(self) -> None:
        pass

    def close(self) -> None:
        self.close_calls += 1


def test_flashsac_double_buffer_runner_trace_writes_b_path_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(double_buffer_runner_module, "ReplayBuffer", _FakeReplayBuffer)
    monkeypatch.setattr(double_buffer_runner_module, "SharedWeightSync", _FakeWeightSync)
    monkeypatch.setattr(double_buffer_runner_module, "OffPolicyLogger", _FakeLogger)
    monkeypatch.setattr(
        double_buffer_runner_module,
        "CPUPinnedDoubleBufferReplayPipeline",
        _FakeDoubleBufferPipeline,
    )
    monkeypatch.setattr(runner_module, "get_env_dims", lambda *args, **kwargs: (4, 2, 0))
    monkeypatch.setattr(double_buffer_runner_module.torch, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(double_buffer_runner_module.time, "sleep", lambda seconds: None)

    learner = _FakeLearner()
    runner = double_buffer_runner_module.DoubleBufferOffPolicyRunner(
        learner=learner,
        env_name="DummyEnv",
        algo_type="flashsac",
        num_envs=2,
        replay_buffer_n=8,
        batch_size=8,
        learning_starts=6,
        updates_per_step=1,
        policy_frequency=1,
        sync_collection=False,
        env_steps_per_sync=1,
        device="cpu",
        trace_enabled=True,
        trace_output_dir=str(tmp_path),
    )
    monkeypatch.setattr(runner, "_start_collector", lambda *args, **kwargs: None)
    runner._collector_process = _FakeProcess()

    threshold = runner.train_start_threshold
    sleep_sizes = iter([4, 8, threshold])

    def fake_sleep(seconds: float) -> None:
        if seconds < 0.5:
            next_size = next(sleep_sizes, threshold)
            replay_buffer = _FakeReplayBuffer.last_instance
            assert replay_buffer is not None
            replay_buffer.size[0] = next_size
            replay_buffer.ptr[0] = next_size

    monkeypatch.setattr(double_buffer_runner_module.time, "sleep", fake_sleep)
    monkeypatch.setattr(
        double_buffer_runner_module._SPAWN_CTX, "Queue", lambda maxsize=0: queue.Queue()
    )

    runner.learn(max_iterations=1, save_interval=0, log_dir=str(tmp_path / "logs"))

    trace_path = tmp_path / "perfetto_offpolicy_timeline.json"
    assert trace_path.exists()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    names = {event.get("name") for event in payload["traceEvents"]}
    assert "learner/wait_for_data" in names
    assert "learner/update_reward_stats" in names
    assert "learner/wait_for_replay_batch" in names
    assert "learner/replay_sample" in names
    assert "learner/update_critic" in names
    assert "learner/update_actor" in names
    assert "learner/soft_update_target" in names
    assert "learner/weight_sync_write" in names
    assert "learner/training_e2e" in names
    assert "replay_pipeline/collector_pack_request" in names
    assert "replay_pipeline/batch_h2d_submit" in names
    assert "gpu/replay_pipeline_batch_h2d" in names
    assert "replay_pipeline/batch_h2d_wait" in names
    assert "replay_pipeline/gpu_wait_for_batch" in names
    assert "replay_pipeline/hot_cold_swap" in names

    replay_sample_events = [
        event for event in payload["traceEvents"] if event.get("name") == "learner/replay_sample"
    ]
    assert replay_sample_events
    assert replay_sample_events[0]["args"]["pipeline"] == "cpu_pinned_double_buffer"


def test_offpolicy_runner_passes_explicit_runtime_context_to_collector(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    runner = _make_runner(monkeypatch, sync_collection=False)
    runner.sim_backend = "motrix"
    runner.env_cfg_override = {"reward_config": {"scales": {"alive": 1.0}}}
    captured: dict[str, object] = {}

    def capture_start_collector(*, target_fn, kwargs):
        del target_fn
        captured.update(kwargs)

    monkeypatch.setattr(runner, "_start_collector", capture_start_collector)
    monkeypatch.setattr(runner_module._SPAWN_CTX, "Queue", lambda maxsize=0: queue.Queue())
    monkeypatch.setattr(runner_module.time, "sleep", lambda seconds: None)

    runner.learn(max_iterations=0, save_interval=0, log_dir=str(tmp_path))

    assert captured["sim_backend"] == "motrix"
    assert captured["env_cfg_override"] == {"reward_config": {"scales": {"alive": 1.0}}}


def test_multi_gpu_runner_passes_explicit_runtime_context_to_collector(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(multi_gpu_runner_module, "ReplayBuffer", _FakeReplayBuffer)
    monkeypatch.setattr(multi_gpu_runner_module, "SharedWeightSync", _FakeWeightSync)
    monkeypatch.setattr(multi_gpu_runner_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(runner_module, "get_env_dims", lambda *args, **kwargs: (4, 2, 0))
    monkeypatch.setattr(
        multi_gpu_runner_module.tmp,
        "spawn",
        lambda *args, **kwargs: None,
    )

    learner = _FakeLearner()
    runner = multi_gpu_runner_module.MultiGPUOffPolicyRunner(
        learner=learner,
        env_name="DummyEnv",
        algo_type="sac",
        learner_kwargs={},
        num_gpus=2,
        num_envs=2,
        replay_buffer_n=8,
        batch_size=8,
        learning_starts=6,
        updates_per_step=1,
        policy_frequency=1,
        sync_collection=False,
        env_steps_per_sync=1,
        device="cpu",
        sim_backend="motrix",
        env_cfg_override={"reward_config": {"scales": {"alive": 1.0}}},
    )
    captured: dict[str, object] = {}

    def capture_start_collector(*, target_fn, kwargs):
        del target_fn
        captured.update(kwargs)

    monkeypatch.setattr(runner, "_start_collector", capture_start_collector)
    runner.learn(max_iterations=0, save_interval=0, log_dir=str(tmp_path))

    assert captured["sim_backend"] == "motrix"
    assert captured["env_cfg_override"] == {"reward_config": {"scales": {"alive": 1.0}}}


def test_multi_gpu_runner_allocates_replay_critic_storage(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(multi_gpu_runner_module, "ReplayBuffer", _FakeReplayBuffer)
    monkeypatch.setattr(multi_gpu_runner_module, "SharedWeightSync", _FakeWeightSync)
    monkeypatch.setattr(multi_gpu_runner_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(runner_module, "get_env_dims", lambda *args, **kwargs: (4, 2, 7))
    monkeypatch.setattr(
        multi_gpu_runner_module.tmp,
        "spawn",
        lambda *args, **kwargs: None,
    )

    runner = multi_gpu_runner_module.MultiGPUOffPolicyRunner(
        learner=_FakeLearner(),
        env_name="DummyEnv",
        algo_type="sac",
        learner_kwargs={},
        num_gpus=2,
        num_envs=2,
        replay_buffer_n=8,
        batch_size=8,
        learning_starts=6,
        updates_per_step=1,
        policy_frequency=1,
        sync_collection=False,
        env_steps_per_sync=1,
        device="cpu",
    )
    monkeypatch.setattr(runner, "_start_collector", lambda *args, **kwargs: None)

    runner.learn(max_iterations=0, save_interval=0, log_dir=str(tmp_path))

    replay_buffer = _FakeReplayBuffer.last_instance
    assert replay_buffer is not None
    assert replay_buffer.critic_dim == 7


def test_multi_gpu_worker_rank0_propagates_collect_time_and_extra_info(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(multi_gpu_runner_module, "ReplayBuffer", _FakeReplayBuffer)
    monkeypatch.setattr(multi_gpu_runner_module, "SharedWeightSync", _FakeWeightSync)
    monkeypatch.setattr(multi_gpu_runner_module, "OffPolicyLogger", _FakeLogger)
    monkeypatch.setattr(multi_gpu_runner_module, "FastSACLearner", _FakeLearner)
    monkeypatch.setattr(multi_gpu_runner_module.torch.cuda, "set_device", lambda rank: None)
    monkeypatch.setattr(
        multi_gpu_runner_module.dist, "init_process_group", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(multi_gpu_runner_module.dist, "broadcast", lambda *args, **kwargs: None)
    monkeypatch.setattr(multi_gpu_runner_module.dist, "barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        multi_gpu_runner_module.dist, "destroy_process_group", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(multi_gpu_runner_module.torch, "save", lambda *args, **kwargs: None)

    fake_clock = _FakeClock([300.0, 300.1, 310.1, 310.2, 310.7])
    monkeypatch.setattr(multi_gpu_runner_module.time, "time", fake_clock.time)

    sleep_sizes = iter([4, 8, 12])

    def fake_sleep(seconds: float) -> None:
        if seconds < 0.5:
            next_size = next(sleep_sizes, 12)
            replay_buffer = _FakeReplayBuffer.last_instance
            assert replay_buffer is not None
            replay_buffer.size[0] = next_size
            replay_buffer.ptr[0] = next_size
            replay_buffer.collect_time_s[0] = 0.25

    monkeypatch.setattr(multi_gpu_runner_module.time, "sleep", fake_sleep)

    replay_buffer = _FakeReplayBuffer(capacity=64, obs_dim=4, action_dim=2, device="cpu")
    weight_sync = _FakeWeightSync()
    metrics_queue = queue.Queue()
    stop_event = type("_Stop", (), {"is_set": staticmethod(lambda: False)})()

    runner_kwargs = {
        "max_iterations": 1,
        "save_interval": 0,
        "log_dir": str(tmp_path),
        "batch_size": 8,
        "learning_starts": 6,
        "updates_per_step": 1,
        "policy_frequency": 1,
        "sync_collection": False,
        "env_steps_per_sync": 1,
        "env_name": "DummyEnv",
        "num_envs": 2,
        "obs_dim": 4,
        "action_dim": 2,
        "logger_type": "wandb",
    }

    multi_gpu_runner_module._learner_worker(
        rank=0,
        world_size=2,
        learner_kwargs={},
        runner_kwargs=runner_kwargs,
        replay_buffer=cast(ReplayBuffer, replay_buffer),
        weight_sync_name=weight_sync.name,
        weight_sync_lock=weight_sync._lock,
        weight_param_shapes={"weight": torch.Size([1])},
        stop_event=stop_event,
        collection_ready_queue=None,
        trainer_done_queue=None,
        metrics_queue=metrics_queue,
        master_port=12345,
    )

    logger = _FakeLogger.last_instance
    assert logger is not None
    assert logger.step_calls
    step = logger.step_calls[0]
    assert step["collect_time"] == pytest.approx(0.25)
    assert step["extra_info"]["startup_wait_time"] == pytest.approx(9.75)
    assert step["extra_info"]["throughput_steps"] == 2
