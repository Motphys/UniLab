from __future__ import annotations

import torch
from tensordict import TensorDict

from unilab.algos.torch.rsl_rl_ppo import FinalObservationAwarePPO


class _FakeActor:
    def update_normalization(self, obs):
        return None

    def reset(self, dones):
        return None


class _FakeCritic:
    def __init__(self, values: torch.Tensor):
        self.values = values
        self.last_obs = None

    def update_normalization(self, obs):
        return None

    def reset(self, dones):
        return None

    def __call__(self, obs):
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
    algo = object.__new__(FinalObservationAwarePPO)
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

    assert torch.allclose(algo.storage.saved_rewards, torch.tensor([1.0 + 0.99 * 3.0, 2.0]))
    assert torch.equal(algo.critic.last_obs["policy"], final_obs["policy"])
