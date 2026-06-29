from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from unilab.algos.torch.fast_sac.learner import FastSACLearner, SACActor


def test_fast_sac_compile_targets_training_hot_paths(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_compile(fn: Callable, **kwargs):
        calls.append((fn.__qualname__, kwargs))
        return fn

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
    learner.device = "cuda"
    monkeypatch.setattr(torch, "compile", fake_compile)

    learner._compile_training_methods()

    assert calls == [
        (
            "FastSACLearner._critic_loss_tensors",
            {"options": {"triton.cudagraphs": False}},
        ),
        (
            "FastSACLearner._actor_loss_tensors",
            {"options": {"triton.cudagraphs": False}},
        ),
    ]


def test_fast_sac_amp_dtype_resolution_and_scaler_rules() -> None:
    assert FastSACLearner._resolve_amp_dtype("auto", "cuda") is torch.bfloat16
    assert FastSACLearner._resolve_amp_dtype("auto", "xpu") is torch.bfloat16
    assert FastSACLearner._resolve_amp_dtype("fp16", "cuda") is torch.float16
    assert FastSACLearner._resolve_amp_dtype("bf16", "cuda") is torch.bfloat16

    assert FastSACLearner._should_use_grad_scaler(True, "cuda", torch.float16)
    assert not FastSACLearner._should_use_grad_scaler(True, "cuda", torch.bfloat16)
    assert not FastSACLearner._should_use_grad_scaler(True, "xpu", torch.bfloat16)
    assert not FastSACLearner._should_use_grad_scaler(False, "cuda", torch.float16)

    with pytest.raises(ValueError, match="amp_dtype"):
        FastSACLearner._resolve_amp_dtype("tf32", "cuda")


def test_fast_sac_alpha_loss_helper_matches_reference_value_and_grad() -> None:
    learner = FastSACLearner(
        obs_dim=4,
        action_dim=3,
        critic_obs_dim=5,
        device="cpu",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        num_atoms=3,
        num_q_networks=2,
        use_layer_norm=False,
        use_autotune=True,
    )
    next_log_probs = torch.tensor([-1.25, -0.5, 0.25, 1.5], dtype=torch.float32)
    learner.target_entropy = -1.75
    learner.log_alpha.data.fill_(-2.0)

    reference_log_alpha = learner.log_alpha.detach().clone().requires_grad_(True)
    reference_loss = (-reference_log_alpha.exp() * (next_log_probs + learner.target_entropy)).mean()
    reference_loss.backward()

    learner.log_alpha.grad = None
    alpha_loss = learner._alpha_loss_tensor(next_log_probs)
    alpha_loss.backward()

    assert torch.allclose(alpha_loss.detach(), reference_loss.detach())
    assert learner.log_alpha.grad is not None
    assert reference_log_alpha.grad is not None
    assert torch.allclose(learner.log_alpha.grad, reference_log_alpha.grad)
    assert not next_log_probs.requires_grad


def test_sac_actor_tensor_gaussian_sampling_matches_normal_reference() -> None:
    actor = SACActor(
        obs_dim=4,
        action_dim=3,
        hidden_dim=12,
        use_layer_norm=False,
        action_scale=torch.tensor([0.5, 1.5, 2.0]),
        action_bias=torch.tensor([-0.25, 0.0, 0.75]),
    )
    obs = torch.tensor(
        [
            [-1.0, -0.25, 0.5, 1.25],
            [0.25, 0.5, -0.75, 1.0],
        ],
        dtype=torch.float32,
    )

    _, mean, log_std = actor(obs)
    std = log_std.exp()
    eps = torch.tensor(
        [
            [-0.5, 0.25, 1.0],
            [1.5, -1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    raw_action = mean + std * eps
    dist = torch.distributions.Normal(mean, std)
    tanh_action = torch.tanh(raw_action)
    expected_action = tanh_action * actor.action_scale + actor.action_bias
    expected_log_prob = dist.log_prob(raw_action)
    expected_log_prob -= torch.log(1 - tanh_action.pow(2) + 1e-6)
    expected_log_prob -= torch.log(actor.action_scale + 1e-6)
    expected_log_prob = expected_log_prob.sum(1)

    action, log_prob = actor._sample_action_and_log_prob(mean, log_std, eps=eps)

    torch.testing.assert_close(action, expected_action)
    torch.testing.assert_close(log_prob, expected_log_prob)


def test_sac_actor_tensor_gaussian_sampling_matches_normal_without_tanh() -> None:
    actor = SACActor(obs_dim=2, action_dim=3, hidden_dim=12, use_layer_norm=False, use_tanh=False)
    mean = torch.tensor(
        [[-0.5, 0.25, 1.0], [1.5, -1.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    log_std = torch.tensor(
        [[-1.0, -0.25, 0.5], [0.0, -0.75, 0.25]],
        dtype=torch.float32,
        requires_grad=True,
    )
    eps = torch.tensor(
        [[0.25, -1.5, 0.75], [-0.5, 1.0, 1.5]],
        dtype=torch.float32,
    )

    action, log_prob = actor._sample_action_and_log_prob(mean, log_std, eps=eps)

    reference_mean = mean.detach().clone().requires_grad_(True)
    reference_log_std = log_std.detach().clone().requires_grad_(True)
    reference_std = reference_log_std.exp()
    reference_raw_action = reference_mean + reference_std * eps
    reference_dist = torch.distributions.Normal(reference_mean, reference_std)
    expected_log_prob = reference_dist.log_prob(reference_raw_action).sum(1)

    torch.testing.assert_close(action, reference_raw_action)
    torch.testing.assert_close(log_prob, expected_log_prob)

    loss = (action + log_prob.unsqueeze(1)).sum()
    reference_loss = (reference_raw_action + expected_log_prob.unsqueeze(1)).sum()
    loss.backward()
    reference_loss.backward()

    assert mean.grad is not None
    assert log_std.grad is not None
    assert reference_mean.grad is not None
    assert reference_log_std.grad is not None
    torch.testing.assert_close(mean.grad, reference_mean.grad)
    torch.testing.assert_close(log_std.grad, reference_log_std.grad)
