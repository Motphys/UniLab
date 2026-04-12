"""Unit tests for FlashSAC learner and actor interfaces."""

from __future__ import annotations

import torch

from unilab.algos.torch.flash_sac.learner import FlashSACLearner


def _make_batch(batch_size: int = 32) -> dict[str, torch.Tensor]:
    obs = torch.randn(batch_size, 98)
    privileged = torch.randn(batch_size, 3)
    actions = torch.tanh(torch.randn(batch_size, 29))
    rewards = torch.randn(batch_size)
    next_obs = torch.randn(batch_size, 98)
    next_privileged = torch.randn(batch_size, 3)
    dones = torch.zeros(batch_size)
    truncated = torch.zeros(batch_size)
    return {
        "obs": obs,
        "privileged": privileged,
        "actions": actions,
        "rewards": rewards,
        "next_obs": next_obs,
        "next_privileged": next_privileged,
        "dones": dones,
        "truncated": truncated,
    }


def test_flashsac_learner_exposes_expected_dims():
    learner = FlashSACLearner(obs_dim=98, action_dim=29, privileged_dim=3, device="cpu")

    assert learner.obs_dim == 98
    assert learner.critic_obs_dim == 101
    assert learner.action_dim == 29


def test_flashsac_actor_explore_and_forward_shapes():
    learner = FlashSACLearner(obs_dim=98, action_dim=29, privileged_dim=3, device="cpu")
    obs = torch.randn(4, 98)

    actions = learner.actor.explore(obs, deterministic=False)
    deterministic_actions = learner.actor.explore(obs, deterministic=True)
    sampled_actions, info = learner.actor(obs, training=True)

    assert actions.shape == (4, 29)
    assert deterministic_actions.shape == (4, 29)
    assert sampled_actions.shape == (4, 29)
    assert info["log_prob"].shape == (4,)


def test_flashsac_update_steps_run_on_cpu():
    learner = FlashSACLearner(obs_dim=98, action_dim=29, privileged_dim=3, device="cpu")
    batch = _make_batch()

    critic_metrics = learner.update_critic(batch)
    actor_metrics = learner.update_actor(batch)
    learner.soft_update_target()

    assert "critic_loss" in critic_metrics
    assert "reward_scale_std" in critic_metrics
    assert "actor_loss" in actor_metrics
    assert "temperature" in actor_metrics


def test_flashsac_state_dict_round_trip():
    learner = FlashSACLearner(obs_dim=98, action_dim=29, privileged_dim=3, device="cpu")
    batch = _make_batch()
    learner.update_critic(batch)
    learner.update_actor(batch)
    state_dict = learner.get_state_dict()

    restored = FlashSACLearner(obs_dim=98, action_dim=29, privileged_dim=3, device="cpu")
    restored.load_state_dict(state_dict)

    assert restored.get_state_dict()["update_count"] == learner.get_state_dict()["update_count"]
