from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def test_actor_uses_newest_history_frame_and_keeps_estimator_gradient() -> None:
    from unilab.algos.cse_ppo import CSEActorCritic

    actor_critic = CSEActorCritic(
        num_actor_obs=6,
        num_critic_obs=5,
        num_one_step_obs=2,
        num_actions=1,
        actor_hidden_dims=[4],
        critic_hidden_dims=[4],
        estimator={
            "num_pred": 2,
            "enc_hidden_dims": [4],
            "latent_dim": 3,
            "dec_hidden_dims": [4],
        },
    )
    oldest_to_newest = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], requires_grad=True)

    actor_input = actor_critic._actor_input(oldest_to_newest)
    actor_input.sum().backward()

    torch.testing.assert_close(actor_input[:, :2], torch.tensor([[5.0, 6.0]]))
    assert oldest_to_newest.grad is not None
    assert torch.count_nonzero(oldest_to_newest.grad[:, :4]) > 0


def test_estimator_supervises_the_configured_current_critic_block() -> None:
    from unilab.algos.cse_ppo import CSEEstimator

    estimator = CSEEstimator(
        temporal_steps=2,
        num_one_step_obs=2,
        num_pred=2,
        enc_hidden_dims=[],
        latent_dim=2,
        dec_hidden_dims=[],
        learning_rate=0.0,
        target_start=1,
        target_group_sizes=[1, 1],
        target_weights=[1.0, 2.0],
    )
    with torch.no_grad():
        encoder = estimator.encoder[0]
        decoder = estimator.decoder[0]
        encoder.weight.zero_()
        encoder.bias.zero_()
        decoder.weight.zero_()
        decoder.bias.zero_()

    loss = estimator.update(
        torch.zeros(1, 4),
        torch.tensor([[99.0, 3.0, 4.0, 88.0]]),
    )

    # Per-group loss is mse(0, 3) + mse(0, 2 * 4).
    assert loss == pytest.approx(73.0)


def test_runner_uses_manager_based_actor_and_critic_group_dimensions() -> None:
    from unilab.algos.cse_ppo import CSEOnPolicyRunner

    env = SimpleNamespace(
        num_envs=2,
        num_obs=2336,
        num_privileged_obs=402,
        num_actions=17,
    )
    runner = CSEOnPolicyRunner(
        env,
        {
            "num_one_step_obs": 73,
            "num_actor_history": 32,
            "num_steps_per_env": 3,
            "policy": {"actor_hidden_dims": [8], "critic_hidden_dims": [8]},
            "estimator": {
                "num_pred": 12,
                "enc_hidden_dims": [8],
                "latent_dim": 4,
                "dec_hidden_dims": [8],
            },
        },
    )

    assert runner.actor_critic.num_actor_obs == 2336
    assert runner.actor_critic.num_critic_obs == 402
    assert runner.actor_critic.num_actions == 17
    assert runner.alg.storage is not None
    assert runner.alg.storage.observations.shape == (3, 2, 2336)
    assert runner.alg.storage.privileged_observations is not None
    assert runner.alg.storage.privileged_observations.shape == (3, 2, 402)
