"""Dispatch and config tests for the sole off-policy replay path."""

from __future__ import annotations

import importlib.util
import os
import platform
import queue
import socket
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace
from unittest.mock import MagicMock

import gymnasium as gym
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from uni_rl.ipc.dp_launcher import UNILAB_DP_LOG_DIR, UNILAB_DP_RANK, UNILAB_DP_WORLD_SIZE
from uni_rl.utils.tensor_runtime import (
    InferencePlacement,
    InferenceTransport,
    TensorRuntimeSettings,
)

from unilab.training import cuda_process_sharing

_ROOT = Path(__file__).parent.parent.parent
_CONF_DIR = _ROOT / "src" / "unilab" / "conf"


@pytest.fixture(autouse=True)
def _rank_local_cuda_visibility(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-rank-local")


def _offpolicy():
    path = _ROOT / "src" / "unilab" / "scripts" / "train_offpolicy.py"
    spec = importlib.util.spec_from_file_location("train_offpolicy", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _offpolicy_cfg(overrides: list[str] | None = None, *, algo: str = "sac"):
    GlobalHydra.instance().clear()
    normalized: list[str] = []
    task_selected = False
    for override in overrides or []:
        if override.startswith("task="):
            task_selected = True
        normalized.append(override)
    if not task_selected:
        normalized.append("task=g1_walk_flat/mujoco")
    with initialize_config_dir(config_dir=str(_CONF_DIR / algo), version_base="1.3"):
        return compose("config", overrides=normalized, return_hydra_config=True)


class _FakeEnv:
    obs_groups_spec = {"obs": 4, "critic": 6}
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))

    def init_state(self):
        return SimpleNamespace(obs={"obs": torch.zeros((1, 4)), "critic": torch.zeros((1, 6))})

    def close(self):
        return None


class _FakeLearner:
    class actor:
        @staticmethod
        def state_dict():
            return {"w": MagicMock(shape=(4,))}

    update_count = 0

    def __init__(self, *args, **kwargs):
        del args
        self.kwargs = kwargs


class _FakeRunner:
    def __init__(self, *args, **kwargs):
        del args
        self.kwargs = kwargs


def _fake_env_factory(num_envs, env_cfg_override):
    """EnvFactory-shaped probe stub (uni_rl runners receive envs by injection)."""
    del num_envs, env_cfg_override
    return _FakeEnv()


def _cuda_torch_module(monkeypatch: pytest.MonkeyPatch, uuid: str = "GPU-a"):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(uuid=uuid),
    )


class _FakeCudaProcessSharingEvidence:
    configured = "mps"
    effective = "mps"
    validated = True
    learner_device = "cuda:0"
    collector_device = "cuda:0"
    learner_gpu_uuid = None
    collector_gpu_uuid = None
    control_pipe = "/tmp/unilab-mps/control"
    server_pid = 2768293

    def manifest(self):
        return {
            "configured": self.configured,
            "effective": self.effective,
            "validated": self.validated,
            "learner_device": self.learner_device,
            "collector_device": self.collector_device,
            "learner_gpu_uuid": self.learner_gpu_uuid,
            "collector_gpu_uuid": self.collector_gpu_uuid,
            "control_pipe": self.control_pipe,
            "server_pid": self.server_pid,
        }


def test_offpolicy_config_has_one_replay_path():
    cfg = _offpolicy_cfg()
    assert cfg.training.replay_prefetch_mode == "one_tick"
    assert cfg.training.env_steps_per_sync == 1
    assert cfg.training.inference_slot_capacity == 1
    assert cfg.training.collector_metrics_interval == 1
    assert cfg.training.replay_ingress_depth == 2
    assert cfg.training.replay_ingress_slot_rows is None


@pytest.mark.parametrize("algo", ["sac", "flashsac"])
def test_offpolicy_owners_default_cuda_process_sharing_off(algo: str):
    cfg = _offpolicy_cfg(algo=algo)

    assert cfg.training.cuda_process_sharing is None


def test_flashsac_scoped_tensor_benchmark_reduces_metric_flush_frequency():
    cfg = _offpolicy_cfg(
        ["task=g1_motion_tracking/mjwarp"],
        algo="flashsac",
    )

    assert cfg.training.inference_slot_capacity == 1
    assert cfg.training.collector_metrics_interval == 100
    assert cfg.training.replay_ingress_depth == 2
    assert cfg.training.replay_ingress_slot_rows is None


@pytest.mark.parametrize("algo", ["sac", "flashsac"])
def test_cuda_process_sharing_request_fails_before_env_materialization(
    monkeypatch: pytest.MonkeyPatch,
    short_unix_socket_root: Path,
    algo: str,
):
    module = _offpolicy()
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    _cuda_torch_module(monkeypatch)
    cfg = _offpolicy_cfg(
        ["task=g1_walk_flat/mjwarp", "training.cuda_process_sharing=mps"],
        algo=algo,
    )

    def reject_factory(*args, **kwargs):
        del args, kwargs
        raise AssertionError("invalid MPS topology must fail before env creation")

    monkeypatch.setattr(module, "registry_env_factory", reject_factory)
    monkeypatch.setattr(
        module,
        "configure_backend_process_device",
        lambda _backend, device: device,
    )
    monkeypatch.setattr(
        module,
        "probe_cuda_process_sharing",
        lambda *_args, **kwargs: cuda_process_sharing.probe_cuda_process_sharing(
            *_args,
            **kwargs,
            run_command=lambda *_command, **_run_kwargs: CompletedProcess(
                [], 0, stdout="0,GPU-a\n"
            ),
        ),
    )
    monkeypatch.setattr(cuda_process_sharing, "_nvidia_uuid", lambda *_args, **_kwargs: "A")
    # Keep the host daemon discovery deterministic: point at a real Unix socket
    # that has no daemon behind it.
    control = short_unix_socket_root / "nvidia-mps" / "control"
    control.parent.mkdir(parents=True)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(control))
    control.chmod(0o666)
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", str(control.parent))
    monkeypatch.setenv("CUDA_MPS_LOG_DIRECTORY", str(short_unix_socket_root / "nvidia-mps-log"))
    monkeypatch.setattr(
        cuda_process_sharing.subprocess,
        "run",
        lambda *_args, **_kwargs: CompletedProcess([], 0, stdout="2768293\n"),
    )
    with pytest.raises(
        ValueError,
        match="could not reach the control daemon|found no control pipe",
    ):
        module.build_runner(algo, cfg)


def test_cuda_process_sharing_selects_sole_recorded_daemon_before_probe(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _offpolicy()
    _cuda_torch_module(monkeypatch)
    cfg = _offpolicy_cfg(["task=g1_walk_flat/mjwarp", "training.cuda_process_sharing=mps"])
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    monkeypatch.delenv("CUDA_MPS_LOG_DIRECTORY", raising=False)
    selected: dict[str, str] = {}

    def fake_selector(requested, backend, learner_device):
        selected.update(
            {
                "requested": requested,
                "backend": backend,
                "learner_device": learner_device,
                "pipe": "/tmp/unilab-recorded/pipe",
                "log": "/tmp/unilab-recorded/log",
            }
        )
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = selected["pipe"]
        os.environ["CUDA_MPS_LOG_DIRECTORY"] = selected["log"]

    def reject_selector(*args, **kwargs):
        del args, kwargs
        raise AssertionError("explicit environment must skip daemon selection")

    monkeypatch.setattr(module, "_select_cuda_mps_daemon_environment", fake_selector)
    monkeypatch.setattr(
        module,
        "probe_cuda_process_sharing",
        lambda *_args, **kwargs: _FakeCudaProcessSharingEvidence(),
    )
    monkeypatch.setattr(module, "registry_env_factory", lambda *args, **kwargs: _fake_env_factory)
    monkeypatch.setattr(
        module,
        "configure_backend_process_device",
        lambda _backend, device: device,
    )
    import uni_rl.algos.sac.double_buffer as owner_module

    monkeypatch.setattr(owner_module, "SACLearner", _FakeLearner)
    monkeypatch.setattr(owner_module, "DoubleBufferOffPolicyRunner", _FakeRunner)
    module.build_runner("sac", cfg)

    assert selected == {
        "requested": "mps",
        "backend": "mjwarp",
        "learner_device": "cuda:0",
        "pipe": "/tmp/unilab-recorded/pipe",
        "log": "/tmp/unilab-recorded/log",
    }

    # Direct helper test documents explicit-environment precedence: deployment
    # provided a pipe, so a selector that would otherwise reject is never used.
    monkeypatch.setattr(module, "select_environment_for_gpu", reject_selector)
    module._select_cuda_mps_daemon_environment("mps", "mjwarp", "cuda:0")


def test_valid_cuda_process_sharing_evidence_enters_runner_manifest(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _offpolicy()
    _cuda_torch_module(monkeypatch)
    cfg = _offpolicy_cfg(["task=g1_walk_flat/mjwarp", "training.cuda_process_sharing=mps"])
    evidence = _FakeCudaProcessSharingEvidence()
    monkeypatch.setattr(module, "registry_env_factory", lambda *args, **kwargs: _fake_env_factory)
    monkeypatch.setattr(
        module,
        "configure_backend_process_device",
        lambda _backend, device: device,
    )
    monkeypatch.setattr(
        module,
        "probe_cuda_process_sharing",
        lambda *args, **kwargs: evidence,
    )
    monkeypatch.setattr(
        module,
        "_select_cuda_mps_daemon_environment",
        lambda *_args, **_kwargs: None,
    )

    import uni_rl.algos.sac.double_buffer as owner_module

    monkeypatch.setattr(owner_module, "SACLearner", _FakeLearner)
    monkeypatch.setattr(owner_module, "DoubleBufferOffPolicyRunner", _FakeRunner)

    runner = module.build_runner("sac", cfg)

    assert runner.runtime_manifest["cuda_process_sharing"] == evidence.manifest()


@pytest.mark.parametrize("mode", ["invalid_mode", "same_tick"])
def test_non_one_tick_prefetch_is_rejected_before_dispatch(mode: str):
    cfg = _offpolicy_cfg([f"training.replay_prefetch_mode={mode}"])
    with pytest.raises(ValueError, match="Unsupported training.replay_prefetch_mode"):
        _offpolicy().build_runner("sac", cfg)


@pytest.mark.parametrize(
    ("setting", "value", "maximum"),
    [
        ("inference_slot_capacity", 17, 16),
        ("collector_metrics_interval", 10_001, 10_000),
        ("replay_ingress_depth", 17, 16),
    ],
)
@pytest.mark.parametrize("algo", ["sac", "flashsac"])
def test_tensor_runtime_bounds_fail_before_env_materialization(
    monkeypatch: pytest.MonkeyPatch,
    algo: str,
    setting: str,
    value: int,
    maximum: int,
):
    module = _offpolicy()
    cfg = _offpolicy_cfg([f"training.{setting}={value}"], algo=algo)
    env_calls = 0

    def reject_factory(num_envs, env_cfg_override):
        del num_envs, env_cfg_override
        nonlocal env_calls
        env_calls += 1
        raise AssertionError("invalid tensor-runtime bounds must fail before env creation")

    monkeypatch.setattr(module, "registry_env_factory", lambda *args, **kwargs: reject_factory)
    pattern = rf"training\.{setting}.*no greater than {maximum}"
    with pytest.raises(ValueError, match=pattern):
        module.build_runner(algo, cfg)
    assert env_calls == 0


@pytest.mark.parametrize("algo", ["sac", "flashsac"])
def test_replay_ingress_rows_fail_before_env_materialization_when_above_num_envs(
    monkeypatch: pytest.MonkeyPatch,
    algo: str,
):
    module = _offpolicy()
    cfg = _offpolicy_cfg(algo=algo)
    cfg.training.replay_ingress_slot_rows = cfg.algo.num_envs + 1
    env_calls = 0

    def reject_factory(num_envs, env_cfg_override):
        del num_envs, env_cfg_override
        nonlocal env_calls
        env_calls += 1
        raise AssertionError("invalid replay-ingress rows must fail before env creation")

    monkeypatch.setattr(module, "registry_env_factory", lambda *args, **kwargs: reject_factory)
    with pytest.raises(
        ValueError,
        match=rf"training\.replay_ingress_slot_rows.*no greater than {cfg.algo.num_envs}",
    ):
        module.build_runner(algo, cfg)
    assert env_calls == 0


def test_sac_dispatch_constructs_unique_runner(monkeypatch: pytest.MonkeyPatch):
    module = _offpolicy()
    cfg = _offpolicy_cfg([])

    import uni_rl.algos.sac.double_buffer as owner_module

    monkeypatch.setattr(module, "registry_env_factory", lambda *args, **kwargs: _fake_env_factory)
    monkeypatch.setattr(owner_module, "SACLearner", _FakeLearner)
    monkeypatch.setattr(owner_module, "DoubleBufferOffPolicyRunner", _FakeRunner)

    runner = module.build_runner("sac", cfg)
    assert isinstance(runner, _FakeRunner)
    assert runner.kwargs["algo_type"] == "sac"
    assert runner.kwargs["device"] == "cuda:0"
    assert runner.kwargs["replay_prefetch_mode"] == "one_tick"
    settings = runner.kwargs["tensor_runtime_settings"]
    assert isinstance(settings, TensorRuntimeSettings)
    assert settings.inference_slot_capacity == 1
    assert settings.collector_metrics_interval == 1
    assert settings.replay_ingress_depth == 2
    assert settings.replay_ingress_slot_rows == cfg.algo.num_envs
    assert settings.num_envs == cfg.algo.num_envs
    assert settings.batch_size == cfg.algo.batch_size
    assert settings.updates_per_step == cfg.algo.updates_per_step
    assert settings.learner_sample_count == (cfg.algo.batch_size * cfg.algo.updates_per_step)
    assert runner.kwargs["batch_size"] == settings.batch_size
    assert runner.kwargs["updates_per_step"] == settings.updates_per_step
    assert settings.configured_inference_slot_capacity == 1
    assert settings.configured_collector_metrics_interval == 1
    assert settings.configured_replay_ingress_depth == 2
    assert settings.configured_replay_ingress_slot_rows is None
    assert runner.kwargs["learner"].kwargs == {
        "device": "cuda:0",
        "obs_dim": 4,
        "action_dim": 2,
        "gamma": cfg.algo.gamma,
        "tau": cfg.algo.tau,
        "actor_lr": cfg.algo.actor_lr,
        "critic_lr": cfg.algo.critic_lr,
        "alpha_lr": cfg.algo.algo_params.alpha_lr,
        "alpha_init": cfg.algo.algo_params.alpha_init,
        "target_entropy_ratio": cfg.algo.algo_params.target_entropy_ratio,
        "actor_hidden_dim": cfg.algo.actor_hidden_dim,
        "critic_hidden_dim": cfg.algo.critic_hidden_dim,
        "num_atoms": cfg.algo.num_atoms,
        "use_layer_norm": cfg.algo.use_layer_norm,
        "max_grad_norm": cfg.algo.algo_params.max_grad_norm,
        "use_amp": cfg.training.use_amp,
        "amp_dtype": cfg.algo.algo_params.amp_dtype,
        "use_compile": cfg.algo.algo_params.use_compile,
        "obs_normalization": cfg.algo.obs_normalization,
        "nvtx_profile_ranges": cfg.training.nvtx_profile_ranges,
        "critic_obs_dim": 6,
    }


def test_sac_owner_custom_runtime_can_override_base_learner_kwargs(
    monkeypatch: pytest.MonkeyPatch,
):
    from uni_rl.algos.sac import double_buffer as owner_module
    from uni_rl.offpolicy.runtime import OffPolicyRuntime

    cfg = _offpolicy_cfg([])
    custom_runtime = OffPolicyRuntime(
        learner_cls=_FakeLearner,
        algo_type="custom_sac",
        actor_kwargs={"gamma": 0.123, "critic_obs_dim": 17},
    )
    monkeypatch.setattr(
        owner_module,
        "resolve_custom_offpolicy_runtime",
        lambda _cfg: custom_runtime,
    )
    monkeypatch.setattr(owner_module, "DoubleBufferOffPolicyRunner", _FakeRunner)

    runner = owner_module.build_sac_double_buffer_runner(
        cfg,
        env_factory=_fake_env_factory,
        env_cfg_override={},
        replay_prefetch_mode="one_tick",
        device="cuda:0",
    )

    assert runner.kwargs["algo_type"] == "custom_sac"
    assert runner.kwargs["learner"].kwargs["gamma"] == pytest.approx(0.123)
    assert runner.kwargs["learner"].kwargs["critic_obs_dim"] == 17
    assert runner.kwargs["learner"].kwargs["tau"] == cfg.algo.tau


def test_flashsac_dispatch_constructs_unique_runner(monkeypatch: pytest.MonkeyPatch):
    module = _offpolicy()
    cfg = _offpolicy_cfg(algo="flashsac")

    import uni_rl.algos.flash_sac.double_buffer as flash_module

    monkeypatch.setattr(module, "registry_env_factory", lambda *args, **kwargs: _fake_env_factory)
    monkeypatch.setattr(flash_module, "FlashSACLearner", _FakeLearner)
    monkeypatch.setattr(flash_module, "DoubleBufferOffPolicyRunner", _FakeRunner)

    runner = module.build_runner("flashsac", cfg)
    assert isinstance(runner, _FakeRunner)
    assert runner.kwargs["algo_type"] == "flashsac"
    assert runner.kwargs["device"] == "cuda:0"
    assert runner.kwargs["replay_prefetch_mode"] == "one_tick"
    settings = runner.kwargs["tensor_runtime_settings"]
    assert isinstance(settings, TensorRuntimeSettings)
    assert settings.inference_slot_capacity == 1
    assert settings.collector_metrics_interval == 1
    assert settings.replay_ingress_depth == 2
    assert settings.replay_ingress_slot_rows == cfg.algo.num_envs
    assert settings.num_envs == cfg.algo.num_envs
    assert settings.batch_size == cfg.algo.batch_size
    assert settings.updates_per_step == cfg.algo.updates_per_step
    assert settings.learner_sample_count == (cfg.algo.batch_size * cfg.algo.updates_per_step)
    assert runner.kwargs["batch_size"] == settings.batch_size
    assert runner.kwargs["updates_per_step"] == settings.updates_per_step
    assert settings.configured_inference_slot_capacity == 1
    assert settings.configured_collector_metrics_interval == 1
    assert settings.configured_replay_ingress_depth == 2
    assert settings.configured_replay_ingress_slot_rows is None


def test_flashsac_n_step_is_rejected():
    cfg = _offpolicy_cfg(["algo.algo_params.n_step=3"], algo="flashsac")
    with pytest.raises(ValueError, match="n_step=1 only"):
        _offpolicy().build_runner("flashsac", cfg)


def _bare_runner():
    from uni_rl.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner

    runner = object.__new__(DoubleBufferOffPolicyRunner)
    runner.inference_placement = InferencePlacement(
        mode=InferenceTransport.CPU,
        env_device="cpu",
        ring_device="cpu",
        learner_device="cpu",
        collector_tensor_native=False,
        staging_policy="cpu_explicit_staging",
    )
    return runner


def test_inference_response_detects_dead_collector():
    runner = _bare_runner()
    runner._check_collector_alive = lambda: False
    full_queue: queue.Queue = queue.Queue(maxsize=1)
    full_queue.put_nowait(1)

    with pytest.raises(RuntimeError, match="collector dead"):
        runner._publish_inference_response(full_queue, timeout=0.01)


def _build_sac_runner_with_fakes(
    monkeypatch: pytest.MonkeyPatch,
    overrides: list[str],
    *,
    cpu_count: int = 128,
    backend_binding_calls: list[tuple[str, str]] | None = None,
):
    """build_runner("sac", ...) with learner/env/runner fakes; returns captured state."""
    module = _offpolicy()
    cfg = _offpolicy_cfg(overrides)
    probe_env_calls: list[dict] = []

    def fake_probe_env_factory(num_envs, env_cfg_override):
        probe_env_calls.append({"num_envs": num_envs, "env_cfg_override": env_cfg_override})
        return _FakeEnv()

    monkeypatch.setattr(module.os, "cpu_count", lambda: cpu_count)
    monkeypatch.setattr(
        "uni_rl.ipc.dp_launcher.os.sched_getaffinity",
        lambda _: set(range(cpu_count)),
        raising=False,
    )
    if backend_binding_calls is not None:
        monkeypatch.setattr(
            module,
            "configure_backend_process_device",
            lambda backend, device: backend_binding_calls.append((str(backend), str(device))),
        )

    import uni_rl.algos.sac.double_buffer as owner_module

    monkeypatch.setattr(
        module, "registry_env_factory", lambda *args, **kwargs: fake_probe_env_factory
    )
    monkeypatch.setattr(owner_module, "SACLearner", _FakeLearner)
    monkeypatch.setattr(owner_module, "DoubleBufferOffPolicyRunner", _FakeRunner)

    runner = module.build_runner("sac", cfg, log_dir="/tmp/offpolicy_test_run")
    return runner, probe_env_calls


def test_build_runner_binds_mjwarp_rank_process_to_learner_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(UNILAB_DP_RANK, "1")
    monkeypatch.setenv(UNILAB_DP_WORLD_SIZE, "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-rank-local")
    monkeypatch.setenv(UNILAB_DP_LOG_DIR, "/tmp/offpolicy_test_run")
    bindings: list[tuple[str, str]] = []

    runner, _ = _build_sac_runner_with_fakes(
        monkeypatch,
        ["task=g1_walk_flat/mjwarp"],
        backend_binding_calls=bindings,
    )

    assert bindings == [("mjwarp", "cuda:0")]
    assert runner.kwargs["device"] == "cuda:0"


def test_build_runner_partitions_collector_cpus_per_rank(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "uni_rl.ipc.dp_launcher._discover_physical_cpu_groups",
        lambda _: [[core, core + 64] for core in range(64)],
    )
    # Spawned rank/world size come from the launcher environment.
    monkeypatch.setenv(UNILAB_DP_RANK, "1")
    monkeypatch.setenv(UNILAB_DP_WORLD_SIZE, "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-rank-local")
    monkeypatch.setenv(UNILAB_DP_LOG_DIR, "/tmp/offpolicy_test_run")
    runner, probe_env_calls = _build_sac_runner_with_fakes(
        monkeypatch,
        [],
        cpu_count=128,
    )
    assert runner.kwargs["collector_cpu_ids"] == [
        cpu for core in range(32, 64) for cpu in (core, core + 64)
    ]
    assert runner.kwargs["device"] == "cuda:0"
    # The thread budget is resolved against the rank's CPU share, not the host.
    assert runner.kwargs["torch_thread_runtime"]["cpu_count"] == 64
    # The num_envs=1 probe env must never see cpu_ids (it would size its
    # MuJoCo BatchEnvPool worker count from len(cpu_ids)).
    assert probe_env_calls
    for call in probe_env_calls:
        override = call.get("env_cfg_override") or {}
        assert "cpu_ids" not in override


def test_build_runner_rank_zero_partitions_without_dp_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(UNILAB_DP_RANK, raising=False)
    monkeypatch.setenv(UNILAB_DP_WORLD_SIZE, "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-rank-local")
    monkeypatch.setattr(
        "uni_rl.ipc.dp_launcher._discover_physical_cpu_groups",
        lambda _: [[core, core + 64] for core in range(64)],
    )
    runner, _ = _build_sac_runner_with_fakes(
        monkeypatch,
        [],
        cpu_count=128,
    )
    assert runner.kwargs["collector_cpu_ids"] == [
        cpu for core in range(32) for cpu in (core, core + 64)
    ]
    assert runner.kwargs["torch_thread_runtime"]["cpu_count"] == 64


def test_build_runner_single_rank_keeps_collector_cpus_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(UNILAB_DP_RANK, raising=False)
    monkeypatch.delenv(UNILAB_DP_WORLD_SIZE, raising=False)
    runner, probe_env_calls = _build_sac_runner_with_fakes(monkeypatch, [], cpu_count=128)
    assert runner.kwargs["collector_cpu_ids"] is None
    # Single-rank thread budget still resolves against the full host.
    assert runner.kwargs["torch_thread_runtime"]["cpu_count"] == 128
    override = runner.kwargs["env_cfg_override"] or {}
    assert "cpu_ids" not in override
    for call in probe_env_calls:
        assert "cpu_ids" not in (call.get("env_cfg_override") or {})


def test_build_runner_explicit_dp_collector_cpu_ids(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(UNILAB_DP_RANK, "0")
    monkeypatch.setenv(UNILAB_DP_WORLD_SIZE, "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-rank-local")
    runner, probe_env_calls = _build_sac_runner_with_fakes(
        monkeypatch,
        [
            "training.dp_collector_cpu_ids=[[0,1],[2,3]]",
        ],
        cpu_count=128,
    )
    assert runner.kwargs["collector_cpu_ids"] == [0, 1]
    for call in probe_env_calls:
        assert "cpu_ids" not in (call.get("env_cfg_override") or {})


def test_collector_env_cfg_override_merges_cpu_ids_without_mutating_base():
    runner = _bare_runner()
    base = {"reward_config": {"x": 1}}
    runner.env_cfg_override = base
    runner.collector_cpu_ids = [0, 1]
    merged = runner._collector_env_cfg_override()
    assert merged == {"reward_config": {"x": 1}, "cpu_ids": [0, 1]}
    assert "cpu_ids" not in base


def test_collector_env_cfg_override_without_cpu_ids_passes_through():
    runner = _bare_runner()
    runner.env_cfg_override = {"a": 1}
    runner.collector_cpu_ids = None
    assert runner._collector_env_cfg_override() == runner.env_cfg_override


def test_collector_env_cfg_override_from_none_base():
    runner = _bare_runner()
    runner.env_cfg_override = None
    runner.collector_cpu_ids = [4, 5]
    assert runner._collector_env_cfg_override() == {"cpu_ids": [4, 5]}


def test_train_resume_defaults_disabled():
    module = _offpolicy()
    cfg = _offpolicy_cfg()

    assert cfg.algo.resume is False
    assert module.resolve_train_resume_checkpoint(cfg) is None


def test_train_resume_resolves_selected_checkpoint(tmp_path):
    module = _offpolicy()
    run_dir = tmp_path / "G1WalkFlat" / "2026-10-05_14-28-04_mjwarp"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "model_32000.pt"
    checkpoint.write_bytes(b"stub")
    cfg = _offpolicy_cfg(
        algo="flashsac",
        overrides=[
            f"training.log_root={tmp_path}",
            "algo.resume=true",
            "algo.load_run=2026-10-05_14-28-04_mjwarp",
        ],
    )

    assert module.resolve_train_resume_checkpoint(cfg) == str(checkpoint)


def test_train_resume_fails_closed_when_checkpoint_missing(tmp_path):
    module = _offpolicy()
    cfg = _offpolicy_cfg(
        algo="flashsac",
        overrides=[
            f"training.log_root={tmp_path}",
            "algo.resume=true",
            "algo.load_run=no-such-run",
        ],
    )

    with pytest.raises(RuntimeError, match="algo.resume=true"):
        module.resolve_train_resume_checkpoint(cfg)
