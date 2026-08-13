"""Runner/build_runner integration tests for DP parameter sync (CPU-only)."""

from __future__ import annotations

import inspect
from collections import defaultdict

import pytest
import torch

from tests.algos.test_offpolicy_double_buffer_runner import (
    _FakeEnv,
    _FakeRunner,
    _offpolicy,
    _offpolicy_cfg,
)
from unilab.ipc.dp_launcher import UNILAB_DP_LOG_DIR, UNILAB_DP_RANK
from unilab.ipc.dp_sync import DpParameterSync


def _bare_runner():
    from unilab.algos.torch.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner

    return object.__new__(DoubleBufferOffPolicyRunner)


class _SyncLearner:
    def __init__(self) -> None:
        self.tensors = {
            "actor.w": torch.ones(2),
            "qnet.w": torch.full((2,), 2.0),
        }

    def dp_sync_tensors(self) -> dict[str, torch.Tensor]:
        return self.tensors


class _NoSyncLearner:
    pass


class _FakeDpSync:
    def __init__(self) -> None:
        self.world_size = 2
        self.backend = "gloo"
        self.calls: list[tuple[str, object]] = []

    def start(self) -> None:
        self.calls.append(("start", None))

    def broadcast_from_rank0(self, tensors) -> None:
        self.calls.append(("broadcast_from_rank0", tensors))

    def allreduce_mean(self, tensors) -> None:
        self.calls.append(("allreduce_mean", tensors))

    def close(self) -> None:
        self.calls.append(("close", None))


def _runner_with(learner, dp_sync, interval: int = 2):
    runner = _bare_runner()
    runner.learner = learner
    runner.dp_sync = dp_sync
    runner.dp_sync_interval = interval
    return runner


def test_maybe_dp_sync_allreduces_on_interval_boundary():
    learner = _SyncLearner()
    dp_sync = _FakeDpSync()
    runner = _runner_with(learner, dp_sync, interval=2)
    metrics = defaultdict(list)

    runner._maybe_dp_sync(1, metrics)
    assert dp_sync.calls == []
    assert "dp_sync_time" not in metrics

    runner._maybe_dp_sync(2, metrics)
    assert [name for name, _ in dp_sync.calls] == ["allreduce_mean"]
    # The collectives receive the learner's live tensor references.
    assert dp_sync.calls[0][1] is learner.tensors
    assert len(metrics["dp_sync_time"]) == 1
    assert metrics["dp_sync_time"][0] >= 0.0


def test_maybe_dp_sync_skips_everything_without_dp_sync():
    runner = _runner_with(_SyncLearner(), None, interval=1)
    metrics = defaultdict(list)
    runner._maybe_dp_sync(4, metrics)
    assert metrics == {}


def test_dp_init_broadcast_starts_group_then_broadcasts():
    learner = _SyncLearner()
    dp_sync = _FakeDpSync()
    runner = _runner_with(learner, dp_sync)
    runner._dp_init_broadcast()
    assert [name for name, _ in dp_sync.calls] == ["start", "broadcast_from_rank0"]
    assert dp_sync.calls[1][1] is learner.tensors


def test_dp_init_broadcast_is_a_noop_without_dp_sync():
    runner = _runner_with(_SyncLearner(), None)
    runner._dp_init_broadcast()  # must not touch the learner


def test_learner_without_dp_sync_tensors_fails_with_type_error():
    runner = _runner_with(_NoSyncLearner(), _FakeDpSync())
    with pytest.raises(TypeError, match="dp_sync_tensors"):
        runner._dp_init_broadcast()


def test_close_closes_dp_sync_idempotently():
    dp_sync = _FakeDpSync()
    runner = _runner_with(_SyncLearner(), dp_sync)
    # Avoid the full AsyncRunner.close(); only the dp_sync branch is under test.
    runner.dp_sync.close()
    runner.dp_sync.close()
    assert [name for name, _ in dp_sync.calls] == ["close", "close"]


def test_learn_source_orders_sync_around_collector_and_logging():
    """Structural contract: init broadcast precedes collector startup, and the
    periodic all-reduce sits at the update boundary before log_step."""
    from unilab.algos.torch.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner

    source = inspect.getsource(DoubleBufferOffPolicyRunner.learn)
    assert source.index("self._dp_init_broadcast()") < source.index("self._start_collector(")
    assert (
        source.index("inference_scheduler.finish_update()")
        < source.index("self._maybe_dp_sync(iteration, iter_metrics)")
        < source.index("logger.log_step(")
    )


# ---- FastSACLearner.dp_sync_tensors contract ----


def test_fast_sac_dp_sync_tensors_returns_live_references():
    from unilab.algos.torch.fast_sac.learner import FastSACLearner

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
        max_grad_norm=0.0,
    )
    tensors = learner.dp_sync_tensors()

    for prefix, module in (
        ("actor", learner.actor),
        ("qnet", learner.qnet),
        ("qnet_target", learner.qnet_target),
    ):
        state = module.state_dict()
        assert state, prefix
        for key, value in state.items():
            full_key = f"{prefix}.{key}"
            assert full_key in tensors
            # Live references, not copies: same storage as the module state.
            assert tensors[full_key].data_ptr() == value.data_ptr()
    assert tensors["log_alpha"] is learner.log_alpha

    # In-place collective semantics propagate into the module parameters.
    probe_key = next(key for key in tensors if key.startswith("actor."))
    with torch.no_grad():
        tensors[probe_key].mul_(0.0)
    assert torch.all(tensors[probe_key] == 0.0)


# ---- build_runner assembly ----


def _build_sac_runner_with_dp_fakes(monkeypatch: pytest.MonkeyPatch, overrides: list[str]):
    """build_runner("sac", ...) with learner/env/runner fakes; returns runner kwargs."""
    module = _offpolicy()
    cfg = _offpolicy_cfg(overrides)
    monkeypatch.setattr(module, "ensure_registries", lambda: None)
    monkeypatch.setattr(module, "create_env", lambda *args, **kwargs: _FakeEnv())
    monkeypatch.setattr(module.os, "cpu_count", lambda: 128)

    import unilab.algos.torch.fast_sac.learner as learner_module
    import unilab.algos.torch.offpolicy.double_buffer_runner as runner_module

    class _Learner:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    monkeypatch.setattr(learner_module, "FastSACLearner", _Learner)
    monkeypatch.setattr(runner_module, "DoubleBufferOffPolicyRunner", _FakeRunner)
    runner = module.build_runner("sac", cfg, log_dir="/tmp/dp_sync_test_run")
    return runner.kwargs


def test_offpolicy_config_dp_sync_interval_defaults_to_eight():
    cfg = _offpolicy_cfg()
    assert cfg.training.dp_sync_interval == 8


def test_build_runner_rejects_invalid_dp_sync_interval():
    cfg = _offpolicy_cfg(["algo=sac", "training.dp_sync_interval=0"])
    with pytest.raises(ValueError, match="dp_sync_interval"):
        _offpolicy().build_runner("sac", cfg)


def test_build_runner_single_rank_keeps_dp_sync_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(UNILAB_DP_RANK, raising=False)
    kwargs = _build_sac_runner_with_dp_fakes(monkeypatch, ["algo=sac", "algo.use_symmetry=false"])
    assert kwargs["dp_sync"] is None
    assert kwargs["dp_sync_interval"] == 8


def test_build_runner_multi_gpu_constructs_dp_sync_for_rank0(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(UNILAB_DP_RANK, raising=False)
    monkeypatch.delenv(UNILAB_DP_LOG_DIR, raising=False)
    kwargs = _build_sac_runner_with_dp_fakes(
        monkeypatch,
        [
            "algo=sac",
            "algo.use_symmetry=false",
            "training.devices=[0,1]",
            "training.dp_sync_interval=4",
        ],
    )
    dp_sync = kwargs["dp_sync"]
    assert isinstance(dp_sync, DpParameterSync)
    assert dp_sync.world_size == 2
    assert dp_sync.rank == 0
    assert dp_sync.backend == "nccl"
    assert dp_sync.rendezvous_path == "/tmp/dp_sync_test_run/.dp_rendezvous"
    assert kwargs["dp_sync_interval"] == 4


def test_build_runner_spawned_rank_uses_shared_run_root(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(UNILAB_DP_RANK, "1")
    monkeypatch.setenv(UNILAB_DP_LOG_DIR, "/tmp/dp_sync_shared_root")
    kwargs = _build_sac_runner_with_dp_fakes(
        monkeypatch,
        ["algo=sac", "algo.use_symmetry=false", "training.devices=[0,1]"],
    )
    dp_sync = kwargs["dp_sync"]
    assert isinstance(dp_sync, DpParameterSync)
    assert dp_sync.rank == 1
    # Spawned ranks rendezvous on rank 0's run root, not their rank sub-dir.
    assert dp_sync.rendezvous_path == "/tmp/dp_sync_shared_root/.dp_rendezvous"


def test_build_runner_multi_gpu_rank0_requires_log_dir(monkeypatch: pytest.MonkeyPatch):
    module = _offpolicy()
    cfg = _offpolicy_cfg(["algo=sac", "algo.use_symmetry=false", "training.devices=[0,1]"])
    monkeypatch.delenv(UNILAB_DP_RANK, raising=False)
    monkeypatch.delenv(UNILAB_DP_LOG_DIR, raising=False)
    monkeypatch.setattr(module, "ensure_registries", lambda: None)
    monkeypatch.setattr(module, "create_env", lambda *args, **kwargs: _FakeEnv())
    with pytest.raises(ValueError, match="log_dir"):
        module.build_runner("sac", cfg)
