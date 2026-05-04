from __future__ import annotations

import torch

from unilab.algos.torch.fast_sac.learner import FastSACLearner
from unilab.algos.torch.fast_td3.learner import FastTD3Learner


class _CaptureSacTargetCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_parameter("anchor", torch.nn.Parameter(torch.zeros(())))
        self.bootstrap: torch.Tensor | None = None

    def projection(self, obs, actions, rewards, bootstrap, discount):
        del obs, actions, rewards, discount
        self.bootstrap = bootstrap.detach().clone()
        batch_size = bootstrap.shape[0]
        probs = torch.zeros(2, batch_size, 3, dtype=bootstrap.dtype)
        probs[:, :, 0] = 1.0
        return probs

    def get_value(self, probs):
        return probs[:, :, 0]


class _CaptureTd3TargetCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bootstrap: torch.Tensor | None = None

    def projection(self, obs, actions, rewards, bootstrap, discount):
        del obs, actions, rewards, discount
        self.bootstrap = bootstrap.detach().clone()
        batch_size = bootstrap.shape[0]
        probs = torch.zeros(batch_size, 3, dtype=bootstrap.dtype)
        probs[:, 0] = 1.0
        return probs, probs.clone()

    def get_value(self, probs):
        return probs[:, 0]


def _offpolicy_batch(batch_size: int = 3) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.zeros(batch_size, 4),
        "critic": torch.zeros(batch_size, 5),
        "actions": torch.zeros(batch_size, 2),
        "rewards": torch.zeros(batch_size),
        "next_obs": torch.zeros(batch_size, 4),
        "next_critic": torch.zeros(batch_size, 5),
        "dones": torch.tensor([0.0, 1.0, 1.0]),
        "truncated": torch.tensor([0.0, 1.0, 0.0]),
    }


def test_fast_sac_critic_bootstrap_uses_combined_dones_and_truncated() -> None:
    learner = FastSACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=5,
        device="cpu",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        num_atoms=3,
        num_q_networks=2,
        use_layer_norm=False,
        use_autotune=False,
    )
    target_critic = _CaptureSacTargetCritic()
    learner.qnet_target = target_critic

    learner.update_critic(_offpolicy_batch())

    assert target_critic.bootstrap is not None
    torch.testing.assert_close(target_critic.bootstrap, torch.tensor([1.0, 1.0, 0.0]))


def test_fast_sac_critic_requires_truncated_field() -> None:
    learner = FastSACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=5,
        device="cpu",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        num_atoms=3,
        num_q_networks=2,
        use_layer_norm=False,
        use_autotune=False,
    )
    batch = _offpolicy_batch()
    batch.pop("truncated")

    try:
        learner.update_critic(batch)
    except KeyError as exc:
        assert exc.args == ("truncated",)
    else:  # pragma: no cover - explicit failure path
        raise AssertionError("FastSAC learner must require replay 'truncated'")


def test_fast_td3_critic_bootstrap_uses_combined_dones_and_truncated() -> None:
    learner = FastTD3Learner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=5,
        num_envs=3,
        device="cpu",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        num_atoms=3,
        obs_normalization=False,
    )
    target_critic = _CaptureTd3TargetCritic()
    learner.qnet_target = target_critic

    learner.update_critic(_offpolicy_batch())

    assert target_critic.bootstrap is not None
    torch.testing.assert_close(target_critic.bootstrap, torch.tensor([1.0, 1.0, 0.0]))
