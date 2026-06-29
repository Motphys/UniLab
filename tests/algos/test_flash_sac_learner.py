"""Unit tests for FlashSAC learner and actor interfaces."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from unilab.algos.torch.flash_sac.learner import FlashSACLearner, RewardNormalizer
from unilab.algos.torch.flash_sac.update import compute_categorical_td_target
from unilab.algos.torch.offpolicy.distributed import validate_distributed_learner_capability


def _make_batch(batch_size: int = 32) -> dict[str, torch.Tensor]:
    obs = torch.randn(batch_size, 98)
    critic = torch.randn(batch_size, 101)
    actions = torch.tanh(torch.randn(batch_size, 29))
    rewards = torch.randn(batch_size)
    next_obs = torch.randn(batch_size, 98)
    next_critic = torch.randn(batch_size, 101)
    dones = torch.zeros(batch_size)
    truncated = torch.zeros(batch_size)
    return {
        "obs": obs,
        "critic": critic,
        "actions": actions,
        "rewards": rewards,
        "next_obs": next_obs,
        "next_critic": next_critic,
        "dones": dones,
        "truncated": truncated,
    }


def _make_small_learner(**kwargs: Any) -> FlashSACLearner:
    defaults = {
        "obs_dim": 4,
        "action_dim": 2,
        "critic_obs_dim": 6,
        "actor_hidden_dim": 8,
        "critic_hidden_dim": 8,
        "actor_num_blocks": 1,
        "critic_num_blocks": 1,
        "num_atoms": 5,
        "device": "cpu",
        "use_compile": False,
    }
    defaults.update(kwargs)
    return FlashSACLearner(**defaults)


def test_flashsac_learner_exposes_expected_dims():
    learner = FlashSACLearner(obs_dim=98, action_dim=29, critic_obs_dim=101, device="cpu")

    assert learner.obs_dim == 98
    assert learner.critic_obs_dim == 101
    assert learner.action_dim == 29


def test_flashsac_learner_declares_multi_gpu_contract() -> None:
    validate_distributed_learner_capability(
        learner_cls=FlashSACLearner,
        algo_type="flashsac",
        learner_kwargs={},
        num_gpus=2,
        sync_mode="LOCAL_SGD",
    )

    learner = _make_small_learner(world_size=2, distributed_sync_mode="LOCAL_SGD")

    assert learner.supports_multi_gpu is True
    assert learner.supported_multi_gpu_sync_modes == frozenset({"sync_sgd", "local_sgd"})
    assert learner.world_size == 2
    assert learner.distributed_sync_mode == "local_sgd"


def test_flashsac_parameter_sync_tensors_include_temperature_and_persistent_buffers() -> None:
    learner = _make_small_learner()
    tensors = learner._parameter_sync_tensors()
    ptrs = {tensor.data_ptr() for tensor in tensors}

    assert learner.temperature.log_temp.data_ptr() in ptrs
    assert learner.actor.embedder.norm.running_mean.data_ptr() in ptrs
    assert learner.critic.embedder.norm.running_mean.data_ptr() in ptrs
    assert learner.target_critic.embedder.norm.running_mean.data_ptr() in ptrs
    assert learner.critic.predictor.support.data_ptr() in ptrs


def test_flashsac_reduce_gradients_averages_flat_gradient_payload(monkeypatch) -> None:
    learner = _make_small_learner(world_size=2, distributed_sync_mode="sync_sgd")
    for param in learner.actor.parameters():
        param.grad = torch.ones_like(param)

    calls = []

    def fake_all_reduce(tensor: torch.Tensor, op=None) -> None:
        del op
        calls.append(tensor.numel())
        tensor.mul_(4.0)

    monkeypatch.setattr("unilab.algos.torch.flash_sac.learner.dist.all_reduce", fake_all_reduce)

    assert learner._reduce_gradients(learner.actor) is True

    assert calls == [sum(param.numel() for param in learner.actor.parameters())]
    for param in learner.actor.parameters():
        assert param.grad is not None
        torch.testing.assert_close(param.grad, torch.full_like(param.grad, 2.0))


def test_flashsac_reduce_gradients_reports_nonfinite_payload(monkeypatch) -> None:
    learner = _make_small_learner(world_size=2, distributed_sync_mode="sync_sgd")
    first_param = next(learner.actor.parameters())
    first_param.grad = torch.ones_like(first_param)

    def fake_all_reduce(tensor: torch.Tensor, op=None) -> None:
        del op
        tensor[0] = float("inf")

    monkeypatch.setattr("unilab.algos.torch.flash_sac.learner.dist.all_reduce", fake_all_reduce)

    assert learner._reduce_gradients(learner.actor) is False
    assert first_param.grad is not None
    torch.testing.assert_close(first_param.grad, torch.ones_like(first_param))


def test_flashsac_obs_normalizer_uses_cross_rank_moments(monkeypatch) -> None:
    learner = _make_small_learner(obs_normalization=True, world_size=2)

    monkeypatch.setattr("unilab.algos.torch.flash_sac.learner.dist.is_available", lambda: True)
    monkeypatch.setattr("unilab.algos.torch.flash_sac.learner.dist.is_initialized", lambda: True)

    def fake_all_reduce(payload: torch.Tensor, op=None) -> None:
        del op
        obs_dim = 4
        payload[:obs_dim] += torch.tensor([10.0, 20.0, 30.0, 40.0])
        payload[obs_dim : 2 * obs_dim] += torch.tensor([50.0, 200.0, 450.0, 800.0])
        payload[-1] += 2.0

    monkeypatch.setattr("unilab.algos.torch.flash_sac.learner.dist.all_reduce", fake_all_reduce)

    learner._update_obs_normalizer(
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0, 6.0],
            ]
        )
    )

    normalizer = learner.obs_normalizer
    assert not isinstance(normalizer, torch.nn.Identity)
    torch.testing.assert_close(normalizer.count, torch.tensor(4))
    torch.testing.assert_close(
        normalizer.mean,
        torch.tensor([3.5, 6.5, 9.5, 12.5]),
    )


def test_flashsac_compile_targets_training_hot_paths(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_compile(fn: Callable, **kwargs):
        calls.append((fn.__qualname__, kwargs))
        return fn

    learner = FlashSACLearner(obs_dim=98, action_dim=29, critic_obs_dim=101, device="cpu")
    learner.device = torch.device("cuda")
    monkeypatch.setattr(torch, "compile", fake_compile)

    learner._compile_training_methods()

    assert calls == [
        (
            "FlashSACActor.get_mean_and_std",
            {"options": {"triton.cudagraphs": False}},
        ),
        (
            "FlashSACLearner._critic_loss_tensors",
            {"options": {"triton.cudagraphs": False}},
        ),
        (
            "FlashSACLearner._actor_loss_tensors",
            {"options": {"triton.cudagraphs": False}},
        ),
    ]


def test_flashsac_amp_dtype_resolution_and_scaler_rules() -> None:
    assert FlashSACLearner._resolve_amp_dtype("auto", "cuda") is torch.bfloat16
    assert FlashSACLearner._resolve_amp_dtype("auto", "xpu") is torch.bfloat16
    assert FlashSACLearner._resolve_amp_dtype("fp16", "cuda") is torch.float16
    assert FlashSACLearner._resolve_amp_dtype("bf16", "cuda") is torch.bfloat16

    assert FlashSACLearner._should_use_grad_scaler(True, "cuda", torch.float16)
    assert not FlashSACLearner._should_use_grad_scaler(True, "cuda", torch.bfloat16)
    assert not FlashSACLearner._should_use_grad_scaler(True, "xpu", torch.bfloat16)
    assert not FlashSACLearner._should_use_grad_scaler(False, "cuda", torch.float16)

    with pytest.raises(ValueError, match="amp_dtype"):
        FlashSACLearner._resolve_amp_dtype("tf32", "cuda")


def test_flashsac_actor_explore_and_forward_shapes():
    learner = FlashSACLearner(obs_dim=98, action_dim=29, critic_obs_dim=101, device="cpu")
    obs = torch.randn(4, 98)

    actions = learner.actor.explore(obs, deterministic=False)
    deterministic_actions = learner.actor.explore(obs, deterministic=True)
    sampled_actions, info = learner.actor(obs, training=True)

    assert actions.shape == (4, 29)
    assert deterministic_actions.shape == (4, 29)
    assert sampled_actions.shape == (4, 29)
    assert info["log_prob"].shape == (4,)


def test_flashsac_export_module_matches_deterministic_policy():
    learner = FlashSACLearner(obs_dim=98, action_dim=29, critic_obs_dim=101, device="cpu")
    obs = torch.randn(4, 98)

    export_module = learner.actor.as_export_module()

    with torch.inference_mode():
        exported_once = export_module(obs)
        exported_twice = export_module(obs)
        deterministic_actions = learner.actor.explore(obs, deterministic=True)

    torch.testing.assert_close(exported_once, exported_twice)
    torch.testing.assert_close(exported_once, deterministic_actions)


def test_flashsac_update_steps_run_on_cpu():
    learner = FlashSACLearner(obs_dim=98, action_dim=29, critic_obs_dim=101, device="cpu")
    batch = _make_batch()

    critic_metrics = learner.update_critic(batch)
    actor_metrics = learner.update_actor(batch)
    learner.soft_update_target()

    assert "critic_loss" in critic_metrics
    assert "reward_scale_std" in critic_metrics
    assert "actor_loss" in actor_metrics
    assert "temperature" in actor_metrics


def test_flashsac_state_dict_round_trip():
    learner = FlashSACLearner(obs_dim=98, action_dim=29, critic_obs_dim=101, device="cpu")
    batch = _make_batch()
    learner.update_critic(batch)
    learner.update_actor(batch)
    state_dict = learner.get_state_dict()

    restored = FlashSACLearner(obs_dim=98, action_dim=29, critic_obs_dim=101, device="cpu")
    restored.load_state_dict(state_dict)

    assert restored.get_state_dict()["update_count"] == learner.get_state_dict()["update_count"]


def test_reward_normalizer_tracks_discounted_returns() -> None:
    normalizer = RewardNormalizer(gamma=0.5, g_max=5.0, device=torch.device("cpu"))

    normalizer.update_from_transitions(
        rewards=torch.tensor([[2.0, 1.0], [4.0, 3.0]]),
        dones=torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
    )

    torch.testing.assert_close(normalizer.g_r, torch.tensor([5.0, 3.5]))
    torch.testing.assert_close(normalizer.g_r_max, torch.tensor(5.0))


def test_flashsac_critic_update_does_not_advance_reward_stats_from_sampled_batch() -> None:
    learner = FlashSACLearner(obs_dim=98, action_dim=29, critic_obs_dim=101, device="cpu")
    learner.update_reward_stats(
        rewards=torch.tensor([[1.0, 2.0]]),
        dones=torch.zeros(1, 2),
    )
    assert learner.reward_normalizer is not None
    before = learner.reward_normalizer.g_r.clone()

    learner.update_critic(_make_batch())

    torch.testing.assert_close(learner.reward_normalizer.g_r, before)


def test_flashsac_critic_requires_truncated_field() -> None:
    learner = FlashSACLearner(obs_dim=98, action_dim=29, critic_obs_dim=101, device="cpu")
    batch = _make_batch()
    batch.pop("truncated")

    try:
        learner.update_critic(batch)
    except KeyError as exc:
        assert exc.args == ("truncated",)
    else:  # pragma: no cover - explicit failure path
        raise AssertionError("FlashSAC learner must require replay 'truncated'")


def test_flashsac_td_target_treats_dones_as_combined_done_with_truncation_bootstrap() -> None:
    support = torch.tensor([0.0, 1.0, 2.0])
    target_log_probs = torch.log(
        torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        ).clamp_min(1e-8)
    )

    targets = compute_categorical_td_target(
        support=support,
        target_log_probs=target_log_probs,
        reward=torch.zeros(3),
        dones=torch.tensor([0.0, 1.0, 1.0]),
        truncated=torch.tensor([0.0, 1.0, 0.0]),
        actor_entropy=torch.zeros(3),
        gamma=1.0,
    )

    # Continuing rows and truncated rows bootstrap to support value 2.0.
    torch.testing.assert_close(targets[0], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(targets[1], torch.tensor([0.0, 0.0, 1.0]))
    # True terminal rows do not bootstrap and project to reward-only value 0.0.
    torch.testing.assert_close(targets[2], torch.tensor([1.0, 0.0, 0.0]))
