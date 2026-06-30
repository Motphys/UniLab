from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from unilab.algos.torch.fast_sac.learner import FastSACLearner, SACActor


def _small_fast_sac_learner(*, use_autotune: bool = True) -> FastSACLearner:
    return FastSACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=5,
        device="cpu",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        num_atoms=3,
        num_q_networks=2,
        use_layer_norm=False,
        use_autotune=use_autotune,
        max_grad_norm=0.0,
    )


def _small_offpolicy_batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.linspace(-0.4, 0.7, steps=batch_size * 4).view(batch_size, 4),
        "critic": torch.linspace(-0.2, 0.9, steps=batch_size * 5).view(batch_size, 5),
        "actions": torch.linspace(-0.5, 0.5, steps=batch_size * 2).view(batch_size, 2),
        "rewards": torch.linspace(-0.3, 0.6, steps=batch_size),
        "next_obs": torch.linspace(0.1, 1.2, steps=batch_size * 4).view(batch_size, 4),
        "next_critic": torch.linspace(-0.7, 0.4, steps=batch_size * 5).view(batch_size, 5),
        "dones": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "truncated": torch.tensor([0.0, 1.0, 0.0, 0.0]),
    }


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


def test_fast_sac_graph_critic_skips_compiling_critic_loss(monkeypatch) -> None:
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
        use_cuda_graph_critic=True,
    )
    learner.device = "cuda"
    learner.use_cuda_graph_critic = True
    monkeypatch.setattr(torch, "compile", fake_compile)

    learner._compile_training_methods()

    assert calls == [
        (
            "FastSACLearner._actor_loss_tensors",
            {"options": {"triton.cudagraphs": False}},
        ),
    ]


def test_fast_sac_graph_actor_skips_compiling_actor_loss(monkeypatch) -> None:
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
        use_cuda_graph_actor=True,
    )
    learner.device = "cuda"
    learner.use_cuda_graph_actor = True
    monkeypatch.setattr(torch, "compile", fake_compile)

    learner._compile_training_methods()

    assert calls == [
        (
            "FastSACLearner._critic_loss_tensors",
            {"options": {"triton.cudagraphs": False}},
        ),
    ]


def test_fast_sac_cuda_adamw_optimizers_are_capture_ready(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    class _FakeAdamW:
        def __init__(self, _params, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", _FakeAdamW)

    FastSACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=5,
        device="cuda",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        num_atoms=3,
        num_q_networks=2,
        use_layer_norm=False,
        use_autotune=False,
        use_compile=False,
    )

    assert len(calls) == 3
    assert all(call["fused"] for call in calls)
    assert all(call["capturable"] for call in calls)


def test_fast_sac_cpu_adamw_optimizers_keep_default_capturability(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    class _FakeAdamW:
        def __init__(self, _params, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", _FakeAdamW)

    FastSACLearner(
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
        use_compile=False,
    )

    assert len(calls) == 3
    assert not any(call["fused"] for call in calls)
    assert all("capturable" not in call for call in calls)


def test_fast_sac_cuda_graph_critic_is_opt_in_and_cuda_only() -> None:
    cpu_learner = _small_fast_sac_learner()
    assert not cpu_learner.use_cuda_graph_critic

    cuda_learner = FastSACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=5,
        device="cuda",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        num_atoms=3,
        num_q_networks=2,
        use_layer_norm=False,
        use_autotune=False,
        use_compile=False,
        use_amp=False,
        use_cuda_graph_critic=True,
    )
    assert cuda_learner.use_cuda_graph_critic


def test_fast_sac_cuda_graph_actor_is_opt_in_and_cuda_only() -> None:
    cpu_learner = _small_fast_sac_learner()
    assert not cpu_learner.use_cuda_graph_actor

    cuda_learner = FastSACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=5,
        device="cuda",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        num_atoms=3,
        num_q_networks=2,
        use_layer_norm=False,
        use_autotune=False,
        use_compile=False,
        use_amp=False,
        use_cuda_graph_actor=True,
    )
    assert cuda_learner.use_cuda_graph_actor


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


def test_fast_sac_capture_candidate_matches_public_critic_update_for_finite_loss() -> None:
    public_learner = _small_fast_sac_learner()
    capture_learner = _small_fast_sac_learner()
    capture_learner.actor.load_state_dict(public_learner.actor.state_dict())
    capture_learner.qnet.load_state_dict(public_learner.qnet.state_dict())
    capture_learner.qnet_target.load_state_dict(public_learner.qnet_target.state_dict())
    capture_learner.log_alpha.data.copy_(public_learner.log_alpha.data)
    batch = _small_offpolicy_batch()

    torch.manual_seed(2024)
    public_metrics = public_learner.update_critic(batch)

    torch.manual_seed(2024)
    capture_outputs = capture_learner._update_critic_capture_candidate(
        batch["critic"],
        batch["actions"],
        batch["rewards"],
        batch["next_obs"],
        batch["next_critic"],
        batch["dones"],
        batch["truncated"],
    )
    capture_metrics = {
        "qf_loss": capture_outputs[0].item(),
        "critic_grad_norm": capture_outputs[1].item(),
        "target_q_max": capture_outputs[2].item(),
        "target_q_min": capture_outputs[3].item(),
        "alpha_loss": capture_outputs[4].item(),
        "alpha": capture_outputs[5].item(),
    }

    assert public_metrics.keys() == capture_metrics.keys()
    for key, public_value in public_metrics.items():
        assert capture_metrics[key] == pytest.approx(public_value)
    for public_param, capture_param in zip(
        public_learner.qnet.parameters(),
        capture_learner.qnet.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(capture_param, public_param)
    torch.testing.assert_close(capture_learner.log_alpha, public_learner.log_alpha)


def test_fast_sac_capture_candidate_matches_public_actor_update_for_finite_loss() -> None:
    public_learner = _small_fast_sac_learner()
    capture_learner = _small_fast_sac_learner()
    capture_learner.actor.load_state_dict(public_learner.actor.state_dict())
    capture_learner.qnet.load_state_dict(public_learner.qnet.state_dict())
    capture_learner.qnet_target.load_state_dict(public_learner.qnet_target.state_dict())
    capture_learner.log_alpha.data.copy_(public_learner.log_alpha.data)
    batch = _small_offpolicy_batch()

    torch.manual_seed(2025)
    public_metrics = public_learner.update_actor(batch)

    torch.manual_seed(2025)
    capture_outputs = capture_learner._update_actor_capture_candidate(
        batch["obs"],
        batch["critic"],
    )
    capture_metrics = {
        "actor_loss": capture_outputs[0].item(),
        "actor_grad_norm": capture_outputs[1].item(),
        "policy_entropy": capture_outputs[2].item(),
        "action_std": capture_outputs[3].item(),
    }

    assert public_metrics.keys() == capture_metrics.keys()
    for key, public_value in public_metrics.items():
        assert capture_metrics[key] == pytest.approx(public_value)
    for public_param, capture_param in zip(
        public_learner.actor.parameters(),
        capture_learner.actor.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(capture_param, public_param)


def test_fast_sac_cuda_graph_state_materialization_preserves_cpu_rng_state() -> None:
    learner = _small_fast_sac_learner()
    batch = _small_offpolicy_batch()
    torch.manual_seed(12345)
    expected_rng_state = torch.random.get_rng_state()

    learner._materialize_capturable_critic_optimizer_state(batch)

    torch.testing.assert_close(torch.random.get_rng_state(), expected_rng_state)


def test_fast_sac_load_state_dict_resets_cuda_graph_caches() -> None:
    learner = _small_fast_sac_learner()
    state_dict = learner.get_state_dict()
    marker = object()
    learner._cuda_graph_critic = marker  # type: ignore[assignment]
    learner._cuda_graph_critic_static_inputs = {"obs": torch.zeros(1)}
    learner._cuda_graph_critic_action_noise = torch.zeros(1)
    learner._cuda_graph_critic_outputs = (torch.zeros(()),) * 6  # type: ignore[assignment]
    learner._cuda_graph_critic_shapes = {"obs": torch.Size([1])}
    learner._cuda_graph_actor = marker  # type: ignore[assignment]
    learner._cuda_graph_actor_static_inputs = {"obs": torch.zeros(1)}
    learner._cuda_graph_actor_action_noise = torch.zeros(1)
    learner._cuda_graph_actor_outputs = (torch.zeros(()),) * 4  # type: ignore[assignment]
    learner._cuda_graph_actor_shapes = {"obs": torch.Size([1])}

    learner.load_state_dict(state_dict)

    assert learner._cuda_graph_critic is None
    assert learner._cuda_graph_critic_static_inputs is None
    assert learner._cuda_graph_critic_action_noise is None
    assert learner._cuda_graph_critic_outputs is None
    assert learner._cuda_graph_critic_shapes is None
    assert learner._cuda_graph_actor is None
    assert learner._cuda_graph_actor_static_inputs is None
    assert learner._cuda_graph_actor_action_noise is None
    assert learner._cuda_graph_actor_outputs is None
    assert learner._cuda_graph_actor_shapes is None


def test_fast_sac_cuda_graph_replay_paths_emit_outer_nvtx_ranges(monkeypatch) -> None:
    learner = _small_fast_sac_learner()
    batch = _small_offpolicy_batch()
    pushed: list[str] = []
    popped = 0

    class _FakeGraph:
        def replay(self) -> None:
            pushed.append("graph.replay.called")

    def fake_push(name: str) -> None:
        pushed.append(name)

    def fake_pop() -> None:
        nonlocal popped
        popped += 1

    monkeypatch.setattr(torch.cuda.nvtx, "range_push", fake_push)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", fake_pop)

    learner.use_cuda_graph_critic = True
    learner.use_cuda_graph_actor = True
    learner._device_type = "cuda"
    learner.nvtx_profile_ranges = True
    learner._cuda_graph_critic_shapes = learner._critic_graph_input_shapes(batch)
    learner._cuda_graph_critic_static_inputs = {
        key: tensor.clone()
        for key, tensor in batch.items()
        if key in {"critic", "actions", "rewards", "next_obs", "next_critic", "dones", "truncated"}
    }
    learner._cuda_graph_critic_action_noise = torch.empty(batch["actions"].shape)
    learner._cuda_graph_critic_outputs = (torch.zeros(()),) * 6
    learner._cuda_graph_critic = _FakeGraph()  # type: ignore[assignment]
    learner._cuda_graph_actor_shapes = learner._actor_graph_input_shapes(batch)
    learner._cuda_graph_actor_static_inputs = {
        "obs": batch["obs"].clone(),
        "critic": batch["critic"].clone(),
    }
    learner._cuda_graph_actor_action_noise = torch.empty(batch["actions"].shape)
    learner._cuda_graph_actor_outputs = (torch.zeros(()),) * 4
    learner._cuda_graph_actor = _FakeGraph()  # type: ignore[assignment]

    learner.update_critic_cuda_graph(batch)
    learner.update_actor_cuda_graph(batch)

    assert pushed == [
        "critic_graph/copy_inputs",
        "critic_graph/replay",
        "graph.replay.called",
        "critic_graph/output_metrics_item",
        "actor_graph/copy_inputs",
        "actor_graph/replay",
        "graph.replay.called",
        "actor_graph/output_metrics_item",
    ]
    assert popped == 6


def test_fast_sac_cuda_graph_metrics_can_skip_item_reads(monkeypatch) -> None:
    learner = _small_fast_sac_learner()
    calls = 0

    class _Metric:
        def __init__(self, value: float) -> None:
            self.value = value

        def item(self) -> float:
            nonlocal calls
            calls += 1
            return self.value

    learner._cuda_graph_critic_outputs = tuple(_Metric(float(i)) for i in range(6))  # type: ignore[assignment]
    learner._cuda_graph_actor_outputs = tuple(_Metric(float(i)) for i in range(4))  # type: ignore[assignment]

    assert learner._critic_graph_output_metrics(read_items=False) == {}
    assert learner._actor_graph_output_metrics(read_items=False) == {}
    assert calls == 0

    assert learner._critic_graph_output_metrics(read_items=True)["qf_loss"] == 0.0
    assert learner._actor_graph_output_metrics(read_items=True)["actor_loss"] == 0.0
    assert calls == 10


def test_fast_sac_cuda_graph_metric_skip_preserves_replay_updates(monkeypatch) -> None:
    learner = _small_fast_sac_learner()
    batch = _small_offpolicy_batch()
    replay_calls = 0
    metric_calls = 0

    class _FakeGraph:
        def replay(self) -> None:
            nonlocal replay_calls
            replay_calls += 1

    class _Metric:
        def item(self) -> float:
            nonlocal metric_calls
            metric_calls += 1
            return 0.0

    learner.use_cuda_graph_critic = True
    learner.use_cuda_graph_actor = True
    learner._device_type = "cuda"
    learner._cuda_graph_critic_shapes = learner._critic_graph_input_shapes(batch)
    learner._cuda_graph_critic_static_inputs = {
        key: tensor.clone()
        for key, tensor in batch.items()
        if key in {"critic", "actions", "rewards", "next_obs", "next_critic", "dones", "truncated"}
    }
    learner._cuda_graph_critic_action_noise = torch.zeros_like(batch["actions"])
    learner._cuda_graph_critic_outputs = tuple(_Metric() for _ in range(6))  # type: ignore[assignment]
    learner._cuda_graph_critic = _FakeGraph()  # type: ignore[assignment]
    learner._cuda_graph_actor_shapes = learner._actor_graph_input_shapes(batch)
    learner._cuda_graph_actor_static_inputs = {
        "obs": batch["obs"].clone(),
        "critic": batch["critic"].clone(),
    }
    learner._cuda_graph_actor_action_noise = torch.zeros_like(batch["actions"])
    learner._cuda_graph_actor_outputs = tuple(_Metric() for _ in range(4))  # type: ignore[assignment]
    learner._cuda_graph_actor = _FakeGraph()  # type: ignore[assignment]

    assert learner.update_critic_cuda_graph(batch, read_metrics=False) == {}
    assert learner.update_actor_cuda_graph(batch, read_metrics=False) == {}
    assert replay_calls == 2
    assert metric_calls == 0

    learner.update_critic_cuda_graph(batch, read_metrics=True)
    learner.update_actor_cuda_graph(batch, read_metrics=True)
    assert replay_calls == 4
    assert metric_calls == 10


def test_fast_sac_cuda_graph_input_copy_fills_static_noise_in_place(monkeypatch) -> None:
    learner = _small_fast_sac_learner()
    batch = _small_offpolicy_batch()
    learner._cuda_graph_critic_static_inputs = {
        key: tensor.clone()
        for key, tensor in batch.items()
        if key in {"critic", "actions", "rewards", "next_obs", "next_critic", "dones", "truncated"}
    }
    learner._cuda_graph_actor_static_inputs = {
        "obs": batch["obs"].clone(),
        "critic": batch["critic"].clone(),
    }
    learner._cuda_graph_critic_action_noise = torch.zeros_like(batch["actions"])
    learner._cuda_graph_actor_action_noise = torch.zeros_like(batch["actions"])

    def fail_randn_like(_tensor: torch.Tensor) -> torch.Tensor:
        raise AssertionError("graph input copy should fill static noise buffers in place")

    monkeypatch.setattr(torch, "randn_like", fail_randn_like)

    learner._copy_critic_graph_inputs(batch)
    learner._copy_actor_graph_inputs(batch)

    assert not torch.equal(learner._cuda_graph_critic_action_noise, torch.zeros_like(batch["actions"]))
    assert not torch.equal(learner._cuda_graph_actor_action_noise, torch.zeros_like(batch["actions"]))
