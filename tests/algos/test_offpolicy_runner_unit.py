from __future__ import annotations

import queue

import pytest
import torch

import unilab.algos.torch.offpolicy.runner as runner_module
from unilab.algos.torch.offpolicy.runner import (
    OffPolicyRunner,
    compute_train_start_threshold,
    replay_buffer_ready_for_learning,
)


class _FakeActor:
    def __init__(self) -> None:
        self._state = {"weight": torch.zeros(1)}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value.clone() for key, value in self._state.items()}


class _FakeLearner:
    def __init__(self) -> None:
        self.actor = _FakeActor()
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

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: str, critic_dim: int):
        del capacity, obs_dim, action_dim, device, critic_dim
        self.size = torch.zeros(1, dtype=torch.int64)
        self.ptr = torch.zeros(1, dtype=torch.int64)
        self.sample_calls = 0
        self.sample_request_sizes: list[int] = []
        self.sample_sizes_at_call: list[int] = []
        _FakeReplayBuffer.last_instance = self

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

    def __init__(self) -> None:
        self.name = "fake-weight-sync"
        self._lock = None
        self.version = 0
        self.write_calls = 0

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

    def close(self) -> None:
        pass


class _FakeLogger:
    last_instance: "_FakeLogger | None" = None

    def __init__(self, **kwargs) -> None:
        del kwargs
        self.buffer_fill_calls: list[tuple[int, int]] = []
        self.step_calls: list[dict] = []
        self._total_steps = 0
        self._mean_ep_length = 0.0
        _FakeLogger.last_instance = self

    def set_collection_sync(self, enabled: bool, env_steps_per_sync: int) -> None:
        del enabled, env_steps_per_sync

    def start(self) -> None:
        pass

    def log_status(self, status: str) -> None:
        del status

    def finish(self) -> None:
        pass

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
    def __init__(self, replay_buffer: _FakeReplayBuffer, sizes: list[int]) -> None:
        self._replay_buffer = replay_buffer
        self._sizes = list(sizes)
        self.get_calls = 0

    def get(self, timeout: float | None = None) -> int:
        del timeout
        assert self._sizes, "collection_ready_queue exhausted before learner started"
        size = self._sizes.pop(0)
        self._replay_buffer.size[0] = size
        self._replay_buffer.ptr[0] = size
        self.get_calls += 1
        return 1


class _RecordingQueue:
    def __init__(self) -> None:
        self.put_calls: list[int] = []

    def put(self, item: int) -> None:
        self.put_calls.append(item)


class _FakeProcess:
    def is_alive(self) -> bool:
        return True


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


def _make_runner(monkeypatch: pytest.MonkeyPatch, *, sync_collection: bool) -> OffPolicyRunner:
    monkeypatch.setattr(runner_module, "ReplayBuffer", _FakeReplayBuffer)
    monkeypatch.setattr(runner_module, "SharedWeightSync", _FakeWeightSync)
    monkeypatch.setattr(runner_module, "OffPolicyLogger", _FakeLogger)
    monkeypatch.setattr(runner_module, "get_env_dims", lambda *args, **kwargs: (4, 2, 0))
    monkeypatch.setattr(runner_module.torch, "save", lambda *args, **kwargs: None)

    learner = _FakeLearner()
    runner = OffPolicyRunner(
        learner=learner,
        env_name="DummyEnv",
        algo_type="sac",
        num_envs=2,
        replay_buffer_n=8,
        batch_size=8,
        learning_starts=6,
        updates_per_step=1,
        policy_frequency=1,
        sync_collection=sync_collection,
        env_steps_per_sync=1,
        device="cpu",
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

    def queue_factory(maxsize: int = 0):
        del maxsize
        idx = len(created_queues)
        queue_obj: object
        if idx == 0:
            replay_buffer = _FakeReplayBuffer.last_instance
            assert replay_buffer is not None
            queue_obj = _SyncReadyQueue(replay_buffer, [4, 8, threshold])
        elif idx == 1:
            queue_obj = _RecordingQueue()
        else:
            queue_obj = queue.Queue()
        created_queues.append(queue_obj)
        return queue_obj

    monkeypatch.setattr(runner_module._SPAWN_CTX, "Queue", queue_factory)
    monkeypatch.setattr(runner_module.time, "sleep", lambda seconds: None)

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
    assert _FakeWeightSync.last_instance is not None
    assert _FakeWeightSync.last_instance.write_calls == 1


def test_offpolicy_runner_async_waits_for_train_start_threshold(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    runner = _make_runner(monkeypatch, sync_collection=False)
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

    runner.learn(max_iterations=1, save_interval=0, log_dir=str(tmp_path))

    replay_buffer = _FakeReplayBuffer.last_instance
    logger = _FakeLogger.last_instance
    assert replay_buffer is not None
    assert logger is not None
    assert replay_buffer.sample_calls == 1
    assert replay_buffer.sample_sizes_at_call == [threshold]
    assert logger.step_calls and logger.step_calls[0]["iteration"] == 1
