"""CUDA compile and FP16 gradient-lifecycle regressions for FlashSAC."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from unilab.rl.algos.flash_sac.learner import FlashSACLearner
from unilab.rl.runtime.offpolicy.warmup import OffPolicyWarmupContext, build_offpolicy_warmup_batch


def _batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.randn(batch_size, 4, device="cuda:0"),
        "critic": torch.randn(batch_size, 6, device="cuda:0"),
        "actions": torch.tanh(torch.randn(batch_size, 2, device="cuda:0")),
        "rewards": torch.randn(batch_size, device="cuda:0"),
        "next_obs": torch.randn(batch_size, 4, device="cuda:0"),
        "next_critic": torch.randn(batch_size, 6, device="cuda:0"),
        "dones": torch.zeros(batch_size, device="cuda:0"),
        "truncated": torch.zeros(batch_size, device="cuda:0"),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compile and FP16 required")
def test_compile_fp16_prepare_and_repeated_updates() -> None:
    learner = FlashSACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=6,
        device="cuda:0",
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        actor_num_blocks=1,
        critic_num_blocks=1,
        num_atoms=5,
        use_compile=True,
        compile_full_objectives=True,
        use_whole_cycle_cuda_graph=False,
        use_amp=True,
        amp_dtype="fp16",
        normalize_reward=False,
        actor_config={
            "builder": "tests.rl.algos.flash_sac_actor_fixtures:build_scaled_tanh_actor",
            "kwargs": {"hidden_dim": 16},
            "optimizer_param_groups": {
                "mean": {
                    "parameters": ["mean_head.weight", "mean_head.bias"],
                    "lr_multiplier": 0.1,
                }
            },
        },
    )
    context = OffPolicyWarmupContext(
        inference_observations=torch.zeros(4, 4, device="cuda:0"),
        inference_dones=torch.zeros(4, device="cuda:0"),
        batch_size=4,
        updates_per_step=2,
        policy_frequency=1,
        target_frequency=1,
        policy_before_critic=False,
    )
    context = replace(
        context,
        replay_batch=build_offpolicy_warmup_batch(
            context,
            obs_dim=4,
            critic_obs_dim=6,
            action_dim=2,
            device="cuda:0",
        ),
    )

    learner.prepare_for_collection(context)
    optimizers = (learner.actor_optimizer, learner.critic_optimizer, learner.temperature_optimizer)
    gradient_buffers = {
        parameter: parameter.grad
        for optimizer in optimizers
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert all(buffer is not None for buffer in gradient_buffers.values())
    for _ in range(3):
        batch = _batch()
        critic_metrics = learner.update_critic(batch)
        actor_metrics = learner.update_actor(batch)
        assert all(math.isfinite(value) for value in critic_metrics.values())
        assert all(math.isfinite(value) for value in actor_metrics.values())

    assert learner.actor_optimizer.state
    assert learner.critic_optimizer.state
    assert learner.temperature_optimizer.state
    assert all(parameter.grad is buffer for parameter, buffer in gradient_buffers.items())
    groups = learner.actor_optimizer.param_groups
    scaled = next(group for group in groups if group["name"] == "mean")
    default = next(group for group in groups if group["name"] == "default")
    assert scaled["lr"] == pytest.approx(default["lr"] * 0.1)
