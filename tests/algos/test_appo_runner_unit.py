from __future__ import annotations

import queue

import pytest
import torch

import unilab.algos.torch.appo.runner as appo_runner_module
from unilab.algos.torch.appo.runner import APPORunner


@pytest.fixture(autouse=True)
def _reset_fakes() -> None:
    _FakeSharedOnPolicyStorage.last_instance = None
    _FakeSharedOnPolicyStorage.rollout_collect_times = [0.25]
    _FakeLogger.last_instance = None


class _FakeModule:
    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"weight": torch.zeros(1)}


class _FakeLearner:
    def __init__(self) -> None:
        self.actor = _FakeModule()
        self.critic = _FakeModule()
        self.num_learning_epochs = 1

    def get_state_dict(self) -> dict[str, int]:
        return {"iteration": 0}

    def process_batch(self, batch: dict[str, torch.Tensor]) -> None:
        self.last_batch = batch

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        del batch
        return {"loss": 0.5}


class _FakeSharedOnPolicyStorage:
    last_instance: "_FakeSharedOnPolicyStorage | None" = None
    rollout_collect_times: list[float] = [0.25]

    def __init__(
        self,
        *,
        num_envs: int,
        num_steps: int,
        obs_dim: int,
        action_dim: int,
        critic_dim: int,
        num_slots: int,
        create: bool,
    ) -> None:
        del num_envs, num_steps, obs_dim, action_dim, critic_dim, num_slots, create
        self.name = "fake-storage"
        self._write_ptr = object()
        self._read_ptr = object()
        self.wait_calls = 0
        self.advance_calls = 0
        _FakeSharedOnPolicyStorage.last_instance = self

    def wait_for_data(self, timeout: float = 60.0) -> bool:
        del timeout
        self.wait_calls += 1
        return True

    def available(self) -> int:
        return max(len(self.rollout_collect_times) - self.advance_calls, 0)

    def read_torch(self, device: str) -> dict[str, torch.Tensor]:
        collect_time = self.rollout_collect_times[self.advance_calls]
        return {
            "obs": torch.zeros(2, 4, 4, device=device),
            "critic": torch.zeros(2, 4, 7, device=device),
            "actions": torch.zeros(2, 4, 2, device=device),
            "log_probs": torch.zeros(2, 4, device=device),
            "rewards": torch.zeros(2, 4, device=device),
            "dones": torch.zeros(2, 4, device=device),
            "truncated": torch.zeros(2, 4, device=device),
            "last_obs": torch.zeros(2, 4, device=device),
            "last_critic": torch.zeros(2, 7, device=device),
            "rollout_collect_time_s": torch.tensor([collect_time], device=device),
        }

    def advance_read(self) -> None:
        self.advance_calls += 1

    def cleanup(self) -> None:
        pass


class _FakeWeightSync:
    def __init__(self) -> None:
        self.name = "fake-weight-sync"

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, torch.Tensor], create: bool = True
    ) -> "_FakeWeightSync":
        del state_dict, create
        return cls()

    def cleanup(self) -> None:
        pass

    def write_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        del state_dict


class _FakeLogger:
    last_instance: "_FakeLogger | None" = None

    def __init__(self, **kwargs) -> None:
        del kwargs
        self._total_steps = 0
        self._mean_ep_length = 0.0
        self.step_calls: list[dict] = []
        _FakeLogger.last_instance = self

    def set_collection_sync(self, enabled: bool, env_steps_per_sync: int) -> None:
        del enabled, env_steps_per_sync

    def start(self) -> None:
        pass

    def log_status(self, status: str) -> None:
        del status

    def log_save(self, ckpt_path: str) -> None:
        del ckpt_path

    def finish(self) -> None:
        pass

    def update_replay_queue(self, current_len: int, max_size: int) -> None:
        del current_len, max_size

    def log_collector(self, total_steps: int, buffer_size: int, mean_reward: float = 0.0) -> None:
        del buffer_size, mean_reward
        self._total_steps = total_steps

    def update_ep_length(self, mean_ep_length: float) -> None:
        self._mean_ep_length = mean_ep_length

    def update_collector_timing(self, timing_ms: dict[str, float]) -> None:
        del timing_ms

    def update_done_rates(self, timeout_rate: float, terminated_rate: float) -> None:
        del timeout_rate, terminated_rate

    def log_step(self, **kwargs) -> None:
        self.step_calls.append(kwargs)


class _FakeClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def time(self) -> float:
        return next(self._values)


def test_appo_runner_uses_explicit_runtime_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    captured_detect: dict[str, object] = {}
    captured_collector: dict[str, object] = {}

    def fake_detect_dims(self: APPORunner) -> tuple[int, int]:
        captured_detect["sim_backend"] = self.sim_backend
        self.critic_dim = 7
        self.critic_input_dim = 5
        return (4, 2)

    def capture_start_collector(*, target_fn, kwargs):
        del target_fn
        captured_collector.update(kwargs)

    monkeypatch.setattr(APPORunner, "_detect_dims", fake_detect_dims)
    monkeypatch.setattr(APPORunner, "_build_learner", lambda self: _FakeLearner())
    monkeypatch.setattr(appo_runner_module, "SharedOnPolicyStorage", _FakeSharedOnPolicyStorage)
    monkeypatch.setattr(appo_runner_module, "SharedWeightSync", _FakeWeightSync)
    monkeypatch.setattr(appo_runner_module, "OffPolicyLogger", _FakeLogger)
    monkeypatch.setattr(appo_runner_module.torch, "save", lambda *args, **kwargs: None)

    runner = APPORunner(
        env_name="DummyEnv",
        env_cfg_overrides={"reward_config": {"scales": {"alive": 1.0}}},
        rl_cfg={"actor": {}, "critic": {}, "algorithm": {}},
        device="cpu",
        collector_device="cpu",
        sim_backend="motrix",
        num_envs=2,
        steps_per_env=4,
    )
    monkeypatch.setattr(runner, "_start_collector", capture_start_collector)

    runner.learn(max_iterations=0, save_interval=0, log_dir=str(tmp_path))

    assert captured_detect["sim_backend"] == "motrix"
    assert captured_collector["sim_backend"] == "motrix"
    assert captured_collector["env_cfg_override"] == {"reward_config": {"scales": {"alive": 1.0}}}


def test_appo_runner_uses_collector_rollout_time_for_fps_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    def fake_detect_dims(self: APPORunner) -> tuple[int, int]:
        self.critic_dim = 7
        self.critic_input_dim = 5
        return (4, 2)

    monkeypatch.setattr(APPORunner, "_detect_dims", fake_detect_dims)
    monkeypatch.setattr(APPORunner, "_build_learner", lambda self: _FakeLearner())
    monkeypatch.setattr(APPORunner, "_check_collector_alive", lambda self: True)
    monkeypatch.setattr(appo_runner_module, "SharedOnPolicyStorage", _FakeSharedOnPolicyStorage)
    monkeypatch.setattr(appo_runner_module, "SharedWeightSync", _FakeWeightSync)
    monkeypatch.setattr(appo_runner_module, "OffPolicyLogger", _FakeLogger)
    monkeypatch.setattr(appo_runner_module.mp, "get_context", lambda method: queue)
    monkeypatch.setattr(appo_runner_module.torch, "save", lambda *args, **kwargs: None)

    fake_clock = _FakeClock([100.0, 100.0, 110.0, 120.0, 120.5, 121.0])
    monkeypatch.setattr(appo_runner_module.time, "time", fake_clock.time)

    runner = APPORunner(
        env_name="DummyEnv",
        env_cfg_overrides={},
        rl_cfg={"actor": {}, "critic": {}, "algorithm": {}},
        device="cpu",
        collector_device="cpu",
        sim_backend="mujoco",
        num_envs=2,
        steps_per_env=4,
    )
    monkeypatch.setattr(runner, "_start_collector", lambda *args, **kwargs: None)

    runner.learn(max_iterations=1, save_interval=0, log_dir=str(tmp_path))

    logger = _FakeLogger.last_instance
    storage = _FakeSharedOnPolicyStorage.last_instance
    assert logger is not None
    assert storage is not None
    assert storage.wait_calls == 1
    assert storage.advance_calls == 1
    assert logger.step_calls

    step = logger.step_calls[0]
    assert step["collect_time"] == pytest.approx(0.25)
    assert step["wait_time"] == pytest.approx(10.0)
    assert step["train_time"] == pytest.approx(0.5)
    assert step["extra_info"]["startup_wait_time"] == pytest.approx(9.75)
    assert step["extra_info"]["throughput_steps"] == 8
    assert step["metrics"]["rollouts_read"] == 1.0
