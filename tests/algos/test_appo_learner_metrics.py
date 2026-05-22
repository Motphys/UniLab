from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from unilab.algos.torch.appo.learner import APPOLearner


class _Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(obs_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.output_mean = torch.zeros(1, action_dim)
        self.output_std = torch.ones(1, action_dim)
        self.output_entropy = torch.tensor(0.0)

    def forward(self, obs, stochastic_output: bool = False):
        del stochastic_output
        mean = self.linear(obs["policy"])
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        self.output_mean = mean
        self.output_std = std
        self.output_entropy = dist.entropy().sum(dim=-1)
        return mean

    def get_output_log_prob(self, actions):
        dist = torch.distributions.Normal(self.output_mean, self.output_std)
        return dist.log_prob(actions).sum(dim=-1)


class _NormalizingActor(_Actor):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__(obs_dim, action_dim)
        self.register_buffer("offset", torch.zeros(obs_dim))

    def update_normalization(self, obs) -> None:
        self.offset.copy_(obs["policy"].mean(dim=0))

    def forward(self, obs, stochastic_output: bool = False):
        del stochastic_output
        mean = self.linear(obs["policy"] - self.offset)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        self.output_mean = mean
        self.output_std = std
        self.output_entropy = dist.entropy().sum(dim=-1)
        return mean


class _Critic(nn.Module):
    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(obs_dim, 1)

    def forward(self, obs):
        return self.linear(obs["policy"])


def test_appo_process_batch_syncs_target_actor_normalization_buffers():
    torch.manual_seed(13)
    learner = APPOLearner(
        actor=_NormalizingActor(obs_dim=3, action_dim=2),
        critic=_Critic(obs_dim=3),
        num_learning_epochs=1,
        num_mini_batches=2,
        device="cpu",
    )
    batch = {
        "observations": torch.randn(4, 3, 3),
        "actions": torch.randn(4, 3, 2),
        "actions_log_prob": torch.randn(4, 3),
        "rewards": torch.randn(4, 3),
        "dones": torch.zeros(4, 3),
        "last_obs": torch.randn(3, 3),
    }

    learner.process_batch(batch)

    obs_flat = batch["observations"].flatten(0, 1)
    obs_td = {"policy": obs_flat}
    with torch.inference_mode():
        learner.actor(obs_td, stochastic_output=True)
        mu = learner.actor.output_mean
        sigma = learner.actor.output_std
        old_mu = batch["_old_mu"]
        old_sigma = batch["_old_sigma"]
        kl = torch.sum(
            torch.log(sigma / old_sigma + 1e-5)
            + (old_sigma.pow(2) + (old_mu - mu).pow(2)) / (2.0 * sigma.pow(2))
            - 0.5,
            dim=-1,
        )

    assert float(kl.mean().item()) == pytest.approx(0.0, abs=5e-5)
