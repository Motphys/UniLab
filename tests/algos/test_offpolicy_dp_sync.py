"""Runner/build_runner integration tests for synchronous off-policy DP."""

from __future__ import annotations

import inspect
from collections import defaultdict

import pytest
import torch
from uni_rl.ipc.dp_launcher import UNILAB_DP_LOG_DIR, UNILAB_DP_RANK
from uni_rl.ipc.dp_sync import DpParameterSync

from tests.algos.test_offpolicy_double_buffer_runner import (
    _fake_env_factory,
    _FakeRunner,
    _offpolicy,
    _offpolicy_cfg,
)


def _bare_runner():
    from uni_rl.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner

    return object.__new__(DoubleBufferOffPolicyRunner)


class _SyncLearner:
    def __init__(self) -> None:
        self.tensors = {
            "actor.w": torch.ones(2),
            "qnet.w": torch.full((2,), 2.0),
        }
        self.gradient_sync = None
        self.graph_replay_recorder = None
        self.dp_cuda_graph_gradient_sync = False

    def set_gradient_sync(self, sync, *, graph_replay_recorder=None) -> None:
        self.gradient_sync = sync
        self.graph_replay_recorder = graph_replay_recorder

    def dp_initial_sync_tensors(self) -> dict[str, torch.Tensor]:
        return self.tensors


class _NoSyncLearner:
    pass


class _FakeDpSync:
    def __init__(self) -> None:
        self.world_size = 2
        self.rank = 0
        self.backend = "gloo"
        self.calls: list[tuple[str, object]] = []

    def start(self) -> None:
        self.calls.append(("start", None))

    def broadcast_from_rank0(self, tensors) -> None:
        self.calls.append(("broadcast_from_rank0", tensors))

    def allreduce_gradients(self, parameters) -> None:
        self.calls.append(("allreduce_gradients", tuple(parameters)))

    def record_cuda_graph_gradient_replay(self, collective_calls: int) -> None:
        self.calls.append(("record_cuda_graph_gradient_replay", collective_calls))

    def prepare_cuda_graph_collectives(self) -> None:
        self.calls.append(("prepare_cuda_graph_collectives", None))

    def take_gradient_sync_metrics(self):
        self.calls.append(("take_gradient_sync_metrics", None))
        return 0.25, 3

    def allreduce_statistics(self, *, mean=None, total=None):
        self.calls.append(("allreduce_statistics", {"mean": mean, "total": total}))
        return {
            **(mean or {}),
            **{key: value * self.world_size for key, value in (total or {}).items()},
        }

    def close(self) -> None:
        self.calls.append(("close", None))


def _runner_with(learner, dp_sync):
    runner = _bare_runner()
    runner.learner = learner
    runner.dp_sync = dp_sync
    return runner


def test_runner_attaches_per_optimizer_gradient_collective():
    learner = _SyncLearner()
    dp_sync = _FakeDpSync()
    runner = _runner_with(learner, dp_sync)
    runner._attach_dp_gradient_sync()
    assert learner.gradient_sync is not None

    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad = torch.full_like(parameter, 2.0)
    learner.gradient_sync((parameter,))
    assert dp_sync.calls == [("allreduce_gradients", (parameter,))]
    assert learner.graph_replay_recorder is not None
    learner.graph_replay_recorder(2)
    assert dp_sync.calls[-1] == ("record_cuda_graph_gradient_replay", 2)


def test_runner_collects_per_iteration_gradient_sync_metrics():
    runner = _runner_with(_SyncLearner(), _FakeDpSync())
    metrics = defaultdict(list)
    runner._collect_dp_sync_metrics(metrics)
    assert metrics == {"dp_sync_time": [0.25], "dp_gradient_sync_calls": [3.0]}


def test_runner_requires_gradient_sync_contract():
    runner = _runner_with(_NoSyncLearner(), _FakeDpSync())
    with pytest.raises(TypeError, match="set_gradient_sync"):
        runner._attach_dp_gradient_sync()


def test_dp_init_broadcast_starts_group_then_broadcasts():
    learner = _SyncLearner()
    dp_sync = _FakeDpSync()
    runner = _runner_with(learner, dp_sync)
    runner._dp_init_broadcast()
    assert [name for name, _ in dp_sync.calls] == ["start", "broadcast_from_rank0"]
    assert dp_sync.calls[1][1] is learner.tensors


def test_dp_init_warms_nccl_collective_when_optimizer_graph_is_enabled():
    learner = _SyncLearner()
    learner.dp_cuda_graph_gradient_sync = True
    dp_sync = _FakeDpSync()
    runner = _runner_with(learner, dp_sync)
    runner._dp_init_broadcast()
    assert [name for name, _ in dp_sync.calls] == [
        "start",
        "broadcast_from_rank0",
        "prepare_cuda_graph_collectives",
    ]


def test_dp_init_broadcast_is_a_noop_without_dp_sync():
    runner = _runner_with(_SyncLearner(), None)
    runner._dp_init_broadcast()  # must not touch the learner


def test_learner_without_initial_sync_tensors_fails_with_type_error():
    runner = _runner_with(_NoSyncLearner(), _FakeDpSync())
    with pytest.raises(TypeError, match="dp_initial_sync_tensors"):
        runner._dp_init_broadcast()


def test_close_closes_dp_sync_idempotently():
    dp_sync = _FakeDpSync()
    runner = _runner_with(_SyncLearner(), dp_sync)
    # Avoid the full AsyncRunner.close(); only the dp_sync branch is under test.
    runner.dp_sync.close()
    runner.dp_sync.close()
    assert [name for name, _ in dp_sync.calls] == ["close", "close"]


def test_close_restores_terminal_and_ipc_before_destroying_process_group(monkeypatch):
    from uni_rl.offpolicy.runner import OffPolicyRunner

    events: list[str] = []

    class _GraphLearner(_SyncLearner):
        def release_cuda_graphs(self) -> None:
            events.append("release_cuda_graphs")

    class _OrderedDpSync(_FakeDpSync):
        def close(self) -> None:
            events.append("dp_sync.close")

    learner = _GraphLearner()
    learner.dp_cuda_graph_gradient_sync = True
    runner = _runner_with(learner, _OrderedDpSync())
    monkeypatch.setattr(OffPolicyRunner, "close", lambda self: events.append("runner.close"))

    runner.close()

    assert events == ["runner.close", "release_cuda_graphs", "dp_sync.close"]


def test_close_still_destroys_process_group_when_local_cleanup_fails(monkeypatch):
    from uni_rl.offpolicy.runner import OffPolicyRunner

    dp_sync = _FakeDpSync()
    runner = _runner_with(_SyncLearner(), dp_sync)

    def fail_local_cleanup(self) -> None:
        raise RuntimeError("local cleanup failed")

    monkeypatch.setattr(OffPolicyRunner, "close", fail_local_cleanup)

    with pytest.raises(RuntimeError, match="local cleanup failed"):
        runner.close()

    assert [name for name, _ in dp_sync.calls] == ["close"]


def test_log_statistics_mean_scalars_and_sum_concurrent_throughput():
    from uni_rl.logging import OffPolicyLogger

    dp_sync = _FakeDpSync()
    runner = _runner_with(_SyncLearner(), dp_sync)
    logger = OffPolicyLogger(log_backend="none", num_gpus=dp_sync.world_size)
    logger._total_steps = 100
    logger._buffer_size = 50
    logger._buffer_target = 200
    logger._collector_active_steps_per_sec = 1_000.0
    logger._mean_ep_length = 25.0
    logger._collector_timing = {"env_step_ms": 2.0}

    payload = runner._aggregate_log_statistics(
        logger,
        metrics={"critic_loss": 2.0},
        reward=3.0,
        reward_metrics={"mean_ep100": 4.0},
        reward_components={"reward/tracking": 5.0},
        train_time=0.4,
        collector_wait_time=0.1,
        replay_batch_wait_time=0.01,
        learner_replay_sample_time=0.02,
        sync_coordination_time=0.03,
        replay_ingress_h2d_submit_time=0.04,
        inference_h2d_time=0.05,
        inference_forward_time=0.06,
        inference_d2h_time=0.07,
        inference_time=0.18,
        iteration_time=0.5,
        extra_info={
            "throughput_steps": 100,
            "collector_active_steps_per_sec": 1_000.0,
            "batch_size_per_rank": 64,
            "effective_batch_size": 64,
            "replay_samples_per_iter": 128,
            "learner_samples_per_iter": 256,
        },
    )

    assert payload["metrics"] == {"critic_loss": pytest.approx(2.0)}
    assert payload["reward"] == pytest.approx(3.0)
    extra_info = payload["extra_info"]
    assert isinstance(extra_info, dict)
    assert extra_info["steps_per_sec"] == pytest.approx(400.0)
    assert extra_info["learner_samples_per_sec"] == pytest.approx(1_024.0)
    assert extra_info["collector_active_steps_per_sec"] == pytest.approx(2_000.0)
    assert extra_info["effective_batch_size"] == 128
    assert extra_info["learner_samples_per_iter"] == 512
    assert logger._total_steps == 200
    assert logger._buffer_size == 100
    assert logger._collector_timing == {"env_step_ms": pytest.approx(2.0)}

    logger.log_step(iteration=1, **payload)
    assert logger._get_iter_steps_per_sec() == pytest.approx(400.0)
    assert logger._get_effective_samples_per_sec() == pytest.approx(1_024.0)
    header = logger._build_compact_header(include_status=False).plain
    assert "Steps/s 400" in header
    assert "Samples/s 1,024" in header
    assert "Collector/s" not in header
    assert "GPUs 2" in logger._build_display().title.plain
    runner._restore_local_logger_statistics(logger)
    assert logger._total_steps == 100
    assert logger._buffer_size == 50
    assert logger._collector_active_steps_per_sec == pytest.approx(1_000.0)


def test_only_rank_zero_owns_terminal_and_tensorboard_backend():
    rank0_sync = _FakeDpSync()
    rank0 = _runner_with(_SyncLearner(), rank0_sync)
    assert rank0._logger_backend("tensorboard") == "tensorboard"

    rank1_sync = _FakeDpSync()
    rank1_sync.rank = 1
    rank1 = _runner_with(_SyncLearner(), rank1_sync)
    assert rank1._logger_backend("tensorboard") == "no_print"

    single = _runner_with(_SyncLearner(), None)
    assert single._logger_backend("tensorboard") == "tensorboard"


def test_only_rank_zero_persists_checkpoint(tmp_path):
    class _CheckpointLearner:
        def __init__(self) -> None:
            self.state_reads = 0

        def get_state_dict(self):
            self.state_reads += 1
            return {"weight": torch.ones(1)}

    class _Logger:
        def __init__(self) -> None:
            self.paths: list[str] = []

        def log_save(self, path: str) -> None:
            self.paths.append(path)

    rank0_sync = _FakeDpSync()
    rank0_learner = _CheckpointLearner()
    rank0 = _runner_with(rank0_learner, rank0_sync)
    rank0_logger = _Logger()
    rank0_dir = tmp_path / "run"
    rank0_dir.mkdir()
    path = rank0._save_checkpoint(
        log_dir=str(rank0_dir),
        iteration=10,
        logger=rank0_logger,
    )
    assert path == str(rank0_dir / "model_10.pt")
    assert (rank0_dir / "model_10.pt").is_file()
    assert rank0_learner.state_reads == 1
    assert rank0_logger.paths == [path]

    rank1_sync = _FakeDpSync()
    rank1_sync.rank = 1
    rank1_learner = _CheckpointLearner()
    rank1 = _runner_with(rank1_learner, rank1_sync)
    rank1_logger = _Logger()
    rank1_dir = rank0_dir / "rank1"
    assert (
        rank1._save_checkpoint(
            log_dir=str(rank1_dir),
            iteration=10,
            logger=rank1_logger,
        )
        is None
    )
    assert not rank1_dir.exists()
    assert rank1_learner.state_reads == 0
    assert rank1_logger.paths == []


def test_learn_source_orders_sync_around_collector_and_logging():
    """Startup broadcast precedes collection; timing is consumed after updates."""
    from uni_rl.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner

    source = inspect.getsource(DoubleBufferOffPolicyRunner.learn)
    assert source.index("self._dp_init_broadcast()") < source.index("self._start_collector(")
    assert (
        source.index("inference_scheduler.finish_update()")
        < source.index("self._collect_dp_sync_metrics(iter_metrics)")
        < source.index("logger.log_step(")
    )


# ---- FastSACLearner distributed contracts ----


def test_fast_sac_initial_sync_tensors_return_live_references():
    from uni_rl.fast_sac.learner import FastSACLearner

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
    tensors = learner.dp_initial_sync_tensors()

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


def test_fast_sac_syncs_each_optimizer_gradient_before_step():
    from uni_rl.fast_sac.learner import FastSACLearner

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
        use_autotune=True,
    )
    calls: list[tuple[int, ...]] = []

    def record_gradients(parameters) -> None:
        params = tuple(parameters)
        assert params
        assert all(parameter.grad is not None for parameter in params)
        calls.append(tuple(parameter.numel() for parameter in params))

    learner.set_gradient_sync(record_gradients)
    batch_size = 4
    batch = {
        "obs": torch.randn(batch_size, 4),
        "critic": torch.randn(batch_size, 5),
        "actions": torch.randn(batch_size, 2).tanh(),
        "rewards": torch.randn(batch_size),
        "next_obs": torch.randn(batch_size, 4),
        "next_critic": torch.randn(batch_size, 5),
        "dones": torch.zeros(batch_size),
        "truncated": torch.zeros(batch_size),
    }
    learner.update_critic(batch)
    learner.update_actor(batch)
    assert len(calls) == 3  # critic, alpha, actor
    assert calls[1] == (1,)

    calls.clear()
    learner._cuda_graph_critic_action_noise = torch.zeros_like(batch["actions"])
    learner._update_critic_capture_candidate(
        batch["critic"],
        batch["actions"],
        batch["rewards"],
        batch["next_obs"],
        batch["next_critic"],
        batch["dones"],
        batch["truncated"],
    )
    learner._cuda_graph_actor_action_noise = torch.zeros_like(batch["actions"])
    learner._update_actor_capture_candidate(batch["obs"], batch["critic"])
    assert len(calls) == 3  # captured critic, alpha, actor collectives
    assert calls[1] == (1,)


def test_fast_sac_gradient_sync_preserves_cuda_graph_capture():
    from uni_rl.fast_sac.learner import FastSACLearner

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
    )
    learner.use_cuda_graph_critic = True
    learner.use_cuda_graph_actor = True
    learner.use_cuda_graph_critic_packed_staging = True
    learner.use_cuda_graph_actor_packed_staging = True

    learner.set_gradient_sync(lambda parameters: None)

    assert learner.dp_cuda_graph_gradient_sync is True
    assert learner.use_cuda_graph_critic is True
    assert learner.use_cuda_graph_actor is True
    assert learner.use_cuda_graph_critic_packed_staging is True
    assert learner.use_cuda_graph_actor_packed_staging is True


# ---- build_runner assembly ----


def _build_sac_runner_with_dp_fakes(monkeypatch: pytest.MonkeyPatch, overrides: list[str]):
    """build_runner("sac", ...) with learner/env/runner fakes; returns runner kwargs."""
    module = _offpolicy()
    cfg = _offpolicy_cfg(overrides)
    monkeypatch.setattr(module.os, "cpu_count", lambda: 128)

    import uni_rl.fast_sac.double_buffer as owner_module

    class _Learner:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    monkeypatch.setattr(module, "registry_env_factory", lambda *args, **kwargs: _fake_env_factory)
    monkeypatch.setattr(owner_module, "FastSACLearner", _Learner)
    monkeypatch.setattr(owner_module, "DoubleBufferOffPolicyRunner", _FakeRunner)
    runner = module.build_runner("sac", cfg, log_dir="/tmp/dp_sync_test_run")
    return runner.kwargs


def test_offpolicy_config_has_no_periodic_parameter_sync_interval():
    cfg = _offpolicy_cfg()
    assert "dp_sync_interval" not in cfg.training


def test_build_runner_single_rank_keeps_dp_sync_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(UNILAB_DP_RANK, raising=False)
    kwargs = _build_sac_runner_with_dp_fakes(monkeypatch, [])
    assert kwargs["dp_sync"] is None


def test_build_runner_multi_gpu_constructs_dp_sync_for_rank0(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(UNILAB_DP_RANK, raising=False)
    monkeypatch.delenv(UNILAB_DP_LOG_DIR, raising=False)
    kwargs = _build_sac_runner_with_dp_fakes(
        monkeypatch,
        [
            "training.devices=[0,1]",
        ],
    )
    dp_sync = kwargs["dp_sync"]
    assert isinstance(dp_sync, DpParameterSync)
    assert dp_sync.world_size == 2
    assert dp_sync.rank == 0
    assert dp_sync.backend == "nccl"
    assert dp_sync.rendezvous_path == "/tmp/dp_sync_test_run/.dp_rendezvous"


def test_build_runner_spawned_rank_uses_shared_run_root(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(UNILAB_DP_RANK, "1")
    monkeypatch.setenv(UNILAB_DP_LOG_DIR, "/tmp/dp_sync_shared_root")
    kwargs = _build_sac_runner_with_dp_fakes(
        monkeypatch,
        ["training.devices=[0,1]"],
    )
    dp_sync = kwargs["dp_sync"]
    assert isinstance(dp_sync, DpParameterSync)
    assert dp_sync.rank == 1
    # Spawned ranks rendezvous on rank 0's run root, not their rank sub-dir.
    assert dp_sync.rendezvous_path == "/tmp/dp_sync_shared_root/.dp_rendezvous"


def test_build_runner_multi_gpu_rank0_requires_log_dir(monkeypatch: pytest.MonkeyPatch):
    module = _offpolicy()
    cfg = _offpolicy_cfg(["training.devices=[0,1]"])
    monkeypatch.delenv(UNILAB_DP_RANK, raising=False)
    monkeypatch.delenv(UNILAB_DP_LOG_DIR, raising=False)
    monkeypatch.setattr(module, "registry_env_factory", lambda *args, **kwargs: _fake_env_factory)
    monkeypatch.setattr(module.os, "cpu_count", lambda: 128)
    with pytest.raises(ValueError, match="log_dir"):
        module.build_runner("sac", cfg)


# ---- FlashSACLearner distributed contracts ----


def test_flash_sac_initial_sync_tensors_return_live_references():
    from uni_rl.flash_sac.learner import FlashSACLearner

    learner = FlashSACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=6,
        device="cpu",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        actor_num_blocks=1,
        critic_num_blocks=1,
        num_atoms=3,
    )
    tensors = learner.dp_initial_sync_tensors()

    for prefix, module in (
        ("actor", learner.actor),
        ("critic", learner.critic),
        ("target_critic", learner.target_critic),
        ("temperature", learner.temperature),
    ):
        state = module.state_dict()
        assert state, prefix
        for key, value in state.items():
            full_key = f"{prefix}.{key}"
            assert full_key in tensors
            # Live references, not copies: same storage as the module state.
            assert tensors[full_key].data_ptr() == value.data_ptr()

    # Every optimizer-updated tensor is covered by the sync set.
    for prefix, module in (
        ("actor", learner.actor),
        ("critic", learner.critic),
        ("target_critic", learner.target_critic),
        ("temperature", learner.temperature),
    ):
        for key, param in module.named_parameters():
            assert f"{prefix}.{key}" in tensors

    # In-place collective semantics propagate into the module parameters.
    probe_key = next(key for key in tensors if key.startswith("actor."))
    with torch.no_grad():
        tensors[probe_key].mul_(0.0)
    assert torch.all(tensors[probe_key] == 0.0)


def test_flash_sac_syncs_each_optimizer_gradient_before_step():
    from uni_rl.flash_sac.learner import FlashSACLearner

    learner = FlashSACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=6,
        device="cpu",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        actor_num_blocks=1,
        critic_num_blocks=1,
        num_atoms=3,
        normalize_reward=False,
    )
    calls: list[tuple[int, ...]] = []

    def record_gradients(parameters) -> None:
        params = tuple(parameters)
        assert params
        assert all(parameter.grad is not None for parameter in params)
        calls.append(tuple(parameter.numel() for parameter in params))

    learner.set_gradient_sync(record_gradients)
    batch_size = 4
    batch = {
        "obs": torch.randn(batch_size, 4),
        "critic": torch.randn(batch_size, 6),
        "actions": torch.randn(batch_size, 2).tanh(),
        "rewards": torch.randn(batch_size),
        "next_obs": torch.randn(batch_size, 4),
        "next_critic": torch.randn(batch_size, 6),
        "dones": torch.zeros(batch_size),
        "truncated": torch.zeros(batch_size),
    }
    learner.update_critic(batch)
    learner.update_actor(batch)
    assert len(calls) == 3  # critic, actor, temperature
    assert calls[-1] == (1,)

    calls.clear()
    learner._update_critic_capture_candidate(learner._prepare_critic_graph_inputs(batch))
    learner._update_actor_capture_candidate(learner._prepare_actor_graph_inputs(batch))
    assert len(calls) == 3  # captured critic, actor, temperature collectives
    assert calls[-1] == (1,)


def test_flash_sac_gradient_sync_preserves_cuda_graph_and_cpu_fallback_updates():
    from uni_rl.flash_sac.learner import FlashSACLearner

    learner = FlashSACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=6,
        device="cpu",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        actor_num_blocks=1,
        critic_num_blocks=1,
        num_atoms=3,
        use_cuda_graph_critic=True,
        use_cuda_graph_actor=True,
        use_cuda_graph_critic_packed_staging=True,
        use_cuda_graph_actor_packed_staging=True,
    )
    calls: list[tuple[int, ...]] = []

    def record_gradients(parameters) -> None:
        params = tuple(parameters)
        assert all(parameter.grad is not None for parameter in params)
        calls.append(tuple(parameter.numel() for parameter in params))

    learner.set_gradient_sync(record_gradients)
    assert learner.dp_cuda_graph_gradient_sync is True
    assert learner.use_cuda_graph_critic is True
    assert learner.use_cuda_graph_actor is True
    assert learner.use_cuda_graph_critic_packed_staging is True
    assert learner.use_cuda_graph_actor_packed_staging is True
    batch_size = 4
    batch = {
        "obs": torch.randn(batch_size, 4),
        "critic": torch.randn(batch_size, 6),
        "actions": torch.randn(batch_size, 2).tanh(),
        "rewards": torch.randn(batch_size),
        "next_obs": torch.randn(batch_size, 4),
        "next_critic": torch.randn(batch_size, 6),
        "dones": torch.zeros(batch_size),
        "truncated": torch.zeros(batch_size),
    }
    learner.update_critic_cuda_graph(batch)
    learner.update_actor_cuda_graph(batch)

    assert len(calls) == 3  # critic, actor, temperature


# ---- flashsac build_runner assembly ----


def _build_flashsac_runner_with_dp_fakes(monkeypatch: pytest.MonkeyPatch, overrides: list[str]):
    """build_runner("flashsac", ...) with learner/env/runner fakes; returns runner kwargs."""
    module = _offpolicy()
    cfg = _offpolicy_cfg(overrides, algo="flashsac")
    monkeypatch.setattr(module.os, "cpu_count", lambda: 128)

    import uni_rl.flash_sac.double_buffer as flash_module

    monkeypatch.setattr(module, "registry_env_factory", lambda *args, **kwargs: _fake_env_factory)

    class _Learner:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    monkeypatch.setattr(flash_module, "FlashSACLearner", _Learner)
    monkeypatch.setattr(flash_module, "DoubleBufferOffPolicyRunner", _FakeRunner)
    runner = module.build_runner("flashsac", cfg, log_dir="/tmp/dp_sync_test_run")
    return runner.kwargs


def test_build_runner_single_rank_flashsac_keeps_dp_sync_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(UNILAB_DP_RANK, raising=False)
    kwargs = _build_flashsac_runner_with_dp_fakes(monkeypatch, [])
    assert kwargs["dp_sync"] is None
    assert kwargs["collector_cpu_ids"] is None


def test_build_runner_multi_gpu_constructs_dp_sync_for_flashsac_rank0(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv(UNILAB_DP_RANK, raising=False)
    monkeypatch.delenv(UNILAB_DP_LOG_DIR, raising=False)
    kwargs = _build_flashsac_runner_with_dp_fakes(
        monkeypatch,
        [
            "training.devices=[0,1]",
        ],
    )
    dp_sync = kwargs["dp_sync"]
    assert isinstance(dp_sync, DpParameterSync)
    assert dp_sync.world_size == 2
    assert dp_sync.rank == 0
    assert dp_sync.backend == "nccl"
    assert dp_sync.rendezvous_path == "/tmp/dp_sync_test_run/.dp_rendezvous"
    # Rank 0 collector owns the first contiguous CPU block.
    assert kwargs["collector_cpu_ids"] == list(range(64))


def test_build_runner_multi_gpu_flashsac_spawned_rank(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(UNILAB_DP_RANK, "1")
    monkeypatch.setenv(UNILAB_DP_LOG_DIR, "/tmp/dp_sync_shared_root")
    kwargs = _build_flashsac_runner_with_dp_fakes(
        monkeypatch,
        ["training.devices=[0,1]"],
    )
    dp_sync = kwargs["dp_sync"]
    assert isinstance(dp_sync, DpParameterSync)
    assert dp_sync.rank == 1
    # Spawned ranks rendezvous on rank 0's run root, not their rank sub-dir.
    assert dp_sync.rendezvous_path == "/tmp/dp_sync_shared_root/.dp_rendezvous"
    assert kwargs["collector_cpu_ids"] == list(range(64, 128))
