from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from tensordict import TensorDict

from unilab.algos.torch.rsl_rl_ppo import FinalObservationAwarePPO
from unilab.training.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg


class _FakeActor:
    class MLP(torch.nn.Module):
        def forward(self, x):
            return x

    mlp = MLP()
    obs_groups = ["policy"]
    is_recurrent = False

    def update_normalization(self, obs):
        return None

    def reset(self, dones):
        return None


class _FakeCritic:
    class MLP(torch.nn.Module):
        def forward(self, x):
            return x

    mlp = MLP()
    obs_groups = ["critic"]
    is_recurrent = False

    def __init__(self, values: torch.Tensor):
        self.values = values
        self.last_obs = None

    def update_normalization(self, obs):
        return None

    def reset(self, dones):
        return None

    def __call__(self, obs, **kwargs):
        del kwargs
        self.last_obs = obs
        return self.values


class _FakeTransition:
    def __init__(self):
        self.values = torch.tensor([[10.0], [20.0]])
        self.rewards = None
        self.dones = None

    def clear(self):
        return None


class _FakeStorage:
    def __init__(self):
        self.saved_rewards = None

    def add_transition(self, transition):
        self.saved_rewards = transition.rewards.clone()


def test_final_observation_aware_ppo_bootstraps_from_final_observation():
    algo: Any = object.__new__(FinalObservationAwarePPO)
    algo.actor = _FakeActor()
    algo.critic = _FakeCritic(torch.tensor([[3.0], [4.0]]))
    algo.rnd = None
    algo.gamma = 0.99
    algo.transition = _FakeTransition()
    algo.storage = _FakeStorage()
    algo.device = "cpu"

    obs = TensorDict({"policy": torch.zeros((2, 1))}, batch_size=[2])
    rewards = torch.tensor([1.0, 2.0])
    dones = torch.tensor([True, True])
    final_obs = TensorDict({"policy": torch.tensor([[30.0], [40.0]])}, batch_size=[2])

    algo.process_env_step(
        obs,
        rewards,
        dones,
        {
            "time_outs": torch.tensor([True, False]),
            "time_out_bootstrap_obs": final_obs,
        },
    )

    assert algo.storage.saved_rewards is not None
    assert algo.critic.last_obs is not None
    assert torch.allclose(algo.storage.saved_rewards, torch.tensor([1.0 + 0.99 * 3.0, 2.0]))
    assert torch.equal(algo.critic.last_obs["policy"], final_obs["policy"])


def test_final_observation_aware_ppo_compile_targets_minibatch_loss(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_compile(fn: Callable, **kwargs):
        calls.append((getattr(fn, "__qualname__", type(fn).__name__), kwargs))
        return fn

    algo: Any = object.__new__(FinalObservationAwarePPO)
    algo.device = "cuda"
    algo._minibatch_loss_fn = algo._minibatch_loss_tensors
    monkeypatch.setattr(torch, "compile", fake_compile)

    algo._compile_training_methods()

    assert calls == [
        (
            "FinalObservationAwarePPO._minibatch_loss_tensors",
            {"mode": "reduce-overhead", "fullgraph": False},
        )
    ]
    assert algo._minibatch_loss_fn == algo._minibatch_loss_tensors


def test_normalize_ppo_train_cfg_preserves_unilab_runtime_flags() -> None:
    train_cfg = normalize_ppo_train_cfg(
        {
            "algorithm": {
                "class_name": "unilab.algos.torch.rsl_rl_ppo:FinalObservationAwarePPO",
                "enable_compile": True,
                "target_kl_stop": None,
            },
            "policy": {},
        }
    )

    assert train_cfg["algorithm"]["enable_compile"] is True
    assert "target_kl_stop" not in train_cfg["algorithm"]


def test_rsl_rl_adapter_outputs_combined_dones_and_time_outs_alias():
    class FakeEnv:
        def __init__(self):
            self.num_envs = 3
            self.cfg = type("Cfg", (), {"max_episode_seconds": 10.0, "ctrl_dt": 0.02})()
            self.observation_space = type("Space", (), {"shape": (2,)})()
            self.action_space = type("Space", (), {"shape": (1,)})()
            self.obs_groups_spec = {"obs": 2}
            self.state = type("State", (), {"obs": {"obs": torch.zeros(3, 2).numpy()}})()

        def init_state(self):
            pass

        def reset(self, env_indices):
            del env_indices
            return {"obs": torch.zeros(3, 2).numpy()}, {}

        def step(self, actions):
            del actions
            return type(
                "StepState",
                (),
                {
                    "obs": {"obs": torch.zeros(3, 2).numpy()},
                    "reward": torch.zeros(3).numpy(),
                    "terminated": torch.tensor([True, False, False]).numpy(),
                    "truncated": torch.tensor([False, True, False]).numpy(),
                    "info": {},
                    "final_observation": None,
                },
            )()

    wrapper = RslRlVecEnvWrapper(FakeEnv(), device="cpu", policy_obs_mode="actor")

    _, _, dones, infos = wrapper.step(torch.zeros(3, 1))

    assert torch.equal(dones, torch.tensor([True, True, False]))
    assert torch.equal(infos["time_outs"], torch.tensor([False, True, False]))
