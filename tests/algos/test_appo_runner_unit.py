from __future__ import annotations

import pytest
import torch

import unilab.algos.torch.appo.runner as appo_runner_module
from unilab.algos.torch.appo.runner import APPORunner


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


class _FakeSharedOnPolicyStorage:
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


class _FakeLogger:
    def __init__(self, **kwargs) -> None:
        del kwargs
        self._total_steps = 0
        self._mean_ep_length = 0.0

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
