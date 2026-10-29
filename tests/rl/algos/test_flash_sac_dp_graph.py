"""CPU regressions for opt-in DP graph ownership and rank-consistent gates.

These tests do not initialize CUDA or validate NCCL capture support.
"""

from __future__ import annotations

import pytest
import torch

from unilab.rl.algos.flash_sac.learner import FlashSACLearner


def _learner(*, opt_in: bool = False) -> FlashSACLearner:
    return FlashSACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=6,
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        actor_num_blocks=1,
        critic_num_blocks=1,
        num_atoms=5,
        device="cpu",
        use_compile=False,
        dp_whole_cycle_cuda_graph=opt_in,
    )


def test_default_dp_still_disables_whole_cycle_with_hooks() -> None:
    learner = _learner()
    learner._compile_full_update_cycle = True
    learner.set_gradient_sync(lambda _parameters: None)
    learner.set_gradient_graph_hooks(lambda: None, lambda: None, lambda: None)

    assert not learner._compile_full_update_cycle
    assert not learner.use_update_cycle
    assert learner._whole_cycle_capture_fallback


def test_opt_in_requires_sync_hooks_before_whole_cycle_can_run() -> None:
    learner = _learner(opt_in=True)
    learner._compile_full_update_cycle = True
    learner.set_gradient_sync(lambda _parameters: None)

    assert learner._compile_full_update_cycle
    assert not learner.use_update_cycle
    with pytest.raises(RuntimeError, match="whole-cycle path"):
        learner.update_cycle(
            {"obs": torch.zeros(2, 4)},
            updates_per_step=1,
            policy_frequency=1,
            target_frequency=1,
            policy_before_critic=True,
        )

    learner.set_gradient_graph_hooks(lambda: None, lambda: None, lambda: None)
    assert learner.use_update_cycle
    assert not learner._whole_cycle_capture_fallback


def test_callback_and_hook_changes_release_captured_graph() -> None:
    learner = _learner(opt_in=True)
    learner._compile_full_update_cycle = True

    def sync(_parameters):
        pass

    hooks = (lambda: None, lambda: None, lambda: None)
    learner.set_gradient_sync(sync)
    learner.set_gradient_graph_hooks(*hooks)
    graph = object()
    learner._update_cycle_graph = graph
    learner._update_cycle_graph_cache_key = ("captured",)

    # Reattaching identical owners must not discard a valid capture.
    learner.set_gradient_sync(sync)
    learner.set_gradient_graph_hooks(*hooks)
    assert learner._update_cycle_graph is graph

    learner.set_gradient_graph_hooks(hooks[0], hooks[1], lambda: None)
    assert learner._update_cycle_graph is None
    assert learner._update_cycle_graph_cache_key is None
    assert learner.use_update_cycle

    learner._update_cycle_graph = object()
    learner.set_gradient_sync(lambda _parameters: None)
    assert learner._update_cycle_graph is None
    assert learner._gradient_graph_hooks is None
    assert not learner.use_update_cycle

    learner.set_gradient_graph_hooks(*hooks)
    learner._update_cycle_graph = object()
    learner.set_gradient_sync(None)
    assert learner._update_cycle_graph is None
    assert learner._gradient_graph_hooks is None
    assert learner.use_update_cycle  # Detached owner can use the single-rank path.


@pytest.mark.parametrize("accelerator_error", [False, True])
def test_dp_capture_failure_propagates_and_ends_instrumentation(monkeypatch, accelerator_error):
    learner = _learner(opt_in=True)
    learner._compile_full_update_cycle = True
    learner.set_gradient_sync(lambda _parameters: None)
    events = []
    learner.set_gradient_graph_hooks(
        lambda: events.append("begin"),
        lambda: events.append("end"),
        lambda: events.append("replay"),
    )
    error_type = torch.AcceleratorError if accelerator_error else RuntimeError

    def fail(*_args, **_kwargs):
        raise error_type("rank-local capture failure")

    monkeypatch.setattr(learner, "_warm_update_cycle_graph", fail)
    with pytest.raises(error_type, match="rank-local capture failure"):
        learner._ensure_update_cycle_graph(
            {"obs": torch.zeros(2, 4)},
            updates_per_step=1,
            policy_frequency=1,
            target_frequency=1,
            policy_before_critic=True,
        )

    assert events == ["begin", "end"]
    assert learner._update_cycle_graph is None
    assert not learner._update_cycle_capture_failed  # No rank-local eager fallback.
    assert not learner._capture_update_cycle


def test_capture_cleanup_ends_hooks_even_if_optimizer_restore_raises(monkeypatch):
    learner = _learner(opt_in=True)
    events = []
    learner.set_gradient_graph_hooks(lambda: None, lambda: events.append("end"), lambda: None)

    def fail(*_args):
        raise RuntimeError("restore failed")

    monkeypatch.setattr(learner, "_restore_optimizer_groups", fail)
    with pytest.raises(RuntimeError, match="restore failed"):
        learner._finish_update_cycle_capture(())
    assert events == ["end"]


@pytest.mark.parametrize(
    "optimizer_name", ["critic_optimizer", "actor_optimizer", "temperature_optimizer"]
)
@pytest.mark.parametrize("nonfinite_rank", [None, 0, 1])
def test_loss_sentinel_makes_finite_gradient_gates_agree(
    monkeypatch, optimizer_name, nonfinite_rank
):
    """An infinite rank-local loss constant can leave every gradient finite."""
    for rank in range(2):
        learner = _learner(opt_in=True)
        learner._device_type = "cuda"  # Exercise device-gate algebra on CPU tensors.
        learner._host_finite_checks = False
        optimizer = getattr(learner, optimizer_name)
        parameter = optimizer.param_groups[0]["params"][0]
        parameter.grad = torch.ones_like(parameter)
        sentinel_pointer = learner._gradient_loss_flag.grad.data_ptr()

        def average(parameters):
            parameters = list(parameters)
            assert parameters[-1] is learner._gradient_loss_flag
            sentinel = parameters[-1].grad
            assert sentinel.item() == float(rank == nonfinite_rank)
            sentinel.fill_(0.0 if nonfinite_rank is None else 0.5)

        learner.set_gradient_sync(average)
        monkeypatch.setattr(
            torch, "_amp_foreach_non_finite_check_and_unscale_", lambda *_args: None
        )
        loss = torch.tensor(float("inf") if rank == nonfinite_rank else 1.0)
        learner._sync_loss_gradients([parameter], loss)
        learner._arm_optimizer_finite_gate(optimizer, loss)

        expected_skip = float(nonfinite_rank is not None)
        assert learner._optimizer_found_inf.item() == expected_skip
        assert optimizer.found_inf is learner._optimizer_found_inf
        assert learner._gradient_loss_flag.grad.data_ptr() == sentinel_pointer
        assert torch.isfinite(parameter.grad).all()
        if optimizer_name == "critic_optimizer":
            assert learner._critic_update_finite.item() == (nonfinite_rank is None)


@pytest.mark.parametrize("update", ["actor", "critic"])
def test_update_syncs_loss_validity_for_every_optimizer(monkeypatch, update):
    """Sentinel support must be wired into actor, temperature and critic calls."""
    learner = _learner()
    batch = {
        "obs": torch.randn(8, 4),
        "critic": torch.randn(8, 6),
        "actions": torch.tanh(torch.randn(8, 2)),
        "rewards": torch.randn(8),
        "next_obs": torch.randn(8, 4),
        "next_critic": torch.randn(8, 6),
        "dones": torch.zeros(8),
        "truncated": torch.zeros(8),
    }
    calls = []

    def record(parameters, loss):
        assert loss.numel() == 1
        calls.append(tuple(id(parameter) for parameter in parameters))

    monkeypatch.setattr(learner, "_sync_loss_gradients", record)
    getattr(learner, f"update_{update}")(batch, read_metrics=False)
    names = ("actor", "temperature") if update == "actor" else ("critic",)
    assert calls == [
        tuple(id(parameter) for parameter in getattr(learner, name).parameters()) for name in names
    ]
