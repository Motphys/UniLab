"""Dispatch and config tests for the sole off-policy replay path."""

from __future__ import annotations

import importlib.util
import queue
from pathlib import Path
from unittest.mock import MagicMock

import gymnasium as gym
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.errors import ConfigCompositionException

from unilab.ipc.dp_launcher import UNILAB_DP_LOG_DIR, UNILAB_DP_RANK, UNILAB_DP_WORLD_SIZE

_ROOT = Path(__file__).parent.parent.parent
_CONF_DIR = _ROOT / "conf"


def _offpolicy():
    path = _ROOT / "scripts" / "train_offpolicy.py"
    spec = importlib.util.spec_from_file_location("train_offpolicy", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _offpolicy_cfg(overrides: list[str] | None = None):
    GlobalHydra.instance().clear()
    normalized: list[str] = []
    algo = "sac"
    task_selected = False
    devices_selected = False
    for override in overrides or []:
        if override.startswith("algo="):
            algo = override.split("=", 1)[1]
        elif override.startswith("task="):
            task_selected = True
        elif override.startswith("training.devices="):
            devices_selected = True
        normalized.append(override)
    if not task_selected:
        normalized.append(f"task={algo}/g1_walk_flat/mujoco")
    if not devices_selected:
        normalized.append("training.devices=[0]")
    with initialize_config_dir(config_dir=str(_CONF_DIR / "offpolicy"), version_base="1.3"):
        return compose("config", overrides=normalized, return_hydra_config=True)


class _FakeEnv:
    obs_groups_spec = {"obs": 4, "critic": 6}
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))

    def build_symmetry_augmentation(self, device=None):
        del device
        return None

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


def test_offpolicy_config_has_one_replay_path():
    cfg = _offpolicy_cfg()
    assert cfg.training.replay_prefetch_mode == "one_tick"
    assert cfg.training.env_steps_per_sync == 1
    assert "env_steps_per_sync" not in cfg.algo
    assert "inference_owner" not in cfg.training
    assert "collector_infer_device" not in cfg.training
    assert "no_sync_collection" not in cfg.training
    assert "replay_pipeline" not in cfg.training
    assert "verbose_metrics" not in cfg.training
    assert "replay_pack_layout" not in cfg.training
    assert "replay_pack_executor" not in cfg.training
    assert "replay_h2d_submitter" not in cfg.training


def test_hora_uses_same_learner_inference_path():
    cfg = _offpolicy_cfg(["algo=sac", "task=sac/sharpa_inhand/mujoco_hora"])
    assert "inference_owner" not in cfg.training
    assert cfg.training.env_steps_per_sync == 2


@pytest.mark.parametrize(
    "override",
    [
        "training.replay_pipeline=cpu_pinned_double_buffer",
        "training.verbose_metrics=true",
        "training.num_gpus=2",
        "training.multi_gpu_sync_mode=sync_sgd",
        "training.multi_gpu_sync_interval=2",
        "training.device=cuda",
        "training.inference_owner=collector",
        "training.collector_infer_device=cpu",
        "training.no_sync_collection=true",
        "algo.env_steps_per_sync=2",
    ],
)
def test_removed_offpolicy_options_fail_hydra_compose(override: str):
    with pytest.raises(ConfigCompositionException, match="Could not override"):
        _offpolicy_cfg([override])


@pytest.mark.parametrize("mode", ["invalid_mode", "same_tick"])
def test_non_one_tick_prefetch_is_rejected_before_dispatch(mode: str):
    cfg = _offpolicy_cfg([f"training.replay_prefetch_mode={mode}"])
    with pytest.raises(ValueError, match="Unsupported training.replay_prefetch_mode"):
        _offpolicy().build_runner("sac", cfg)


@pytest.mark.parametrize("algo", ["sac", "td3", "flashsac"])
@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_non_cuda_training_devices_fail_before_env_materialization(
    monkeypatch: pytest.MonkeyPatch,
    algo: str,
    device: str,
):
    module = _offpolicy()
    cfg = _offpolicy_cfg([f"algo={algo}", f"training.devices=[{device}]"])
    env_calls = 0

    def reject_env(*args, **kwargs):
        nonlocal env_calls
        del args, kwargs
        env_calls += 1
        raise AssertionError("unsupported replay device must fail before env creation")

    if algo == "sac":
        import unilab.algos.torch.fast_sac.double_buffer as owner_module

        monkeypatch.setattr(owner_module, "create_env", reject_env)
    elif algo == "td3":
        import unilab.algos.torch.fast_td3.double_buffer as owner_module

        monkeypatch.setattr(owner_module, "get_env_dims", reject_env)
    else:
        import unilab.algos.torch.flash_sac.double_buffer as owner_module

        monkeypatch.setattr(owner_module, "create_env", reject_env)
    with pytest.raises(ValueError, match="training.devices entries"):
        module.build_runner(algo, cfg)
    assert env_calls == 0


def test_sac_dispatch_constructs_unique_runner(monkeypatch: pytest.MonkeyPatch):
    module = _offpolicy()
    cfg = _offpolicy_cfg(["algo=sac", "algo.use_symmetry=false"])

    import unilab.algos.torch.fast_sac.double_buffer as owner_module

    monkeypatch.setattr(owner_module, "ensure_registries", lambda: None)
    monkeypatch.setattr(owner_module, "create_env", lambda *args, **kwargs: _FakeEnv())
    monkeypatch.setattr(owner_module, "FastSACLearner", _FakeLearner)
    monkeypatch.setattr(owner_module, "DoubleBufferOffPolicyRunner", _FakeRunner)

    runner = module.build_runner("sac", cfg)
    assert isinstance(runner, _FakeRunner)
    assert runner.kwargs["algo_type"] == "sac"
    assert runner.kwargs["device"] == "cuda:0"
    assert runner.kwargs["replay_prefetch_mode"] == "one_tick"
    assert "inference_owner" not in runner.kwargs
    assert "collector_infer_device" not in runner.kwargs
    assert "sync_collection" not in runner.kwargs
    assert "replay_pipeline" not in runner.kwargs
    assert "verbose_metrics" not in runner.kwargs
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
        "use_cuda_graph_critic": cfg.algo.algo_params.use_cuda_graph_critic,
        "use_cuda_graph_actor": cfg.algo.algo_params.use_cuda_graph_actor,
        "use_cuda_graph_critic_packed_staging": (
            cfg.algo.algo_params.use_cuda_graph_critic_packed_staging
        ),
        "use_cuda_graph_actor_packed_staging": (
            cfg.algo.algo_params.use_cuda_graph_actor_packed_staging
        ),
        "nvtx_profile_ranges": cfg.training.nvtx_profile_ranges,
        "use_symmetry": False,
        "symmetry_augmentation": None,
        "critic_obs_dim": 6,
    }


def test_sac_owner_custom_runtime_can_override_base_learner_kwargs(
    monkeypatch: pytest.MonkeyPatch,
):
    from unilab.algos.torch.fast_sac import double_buffer as owner_module
    from unilab.algos.torch.offpolicy.runtime import OffPolicyRuntime

    cfg = _offpolicy_cfg(["algo=sac", "algo.use_symmetry=false"])
    custom_runtime = OffPolicyRuntime(
        learner_cls=_FakeLearner,
        algo_type="custom_sac",
        actor_kwargs={"gamma": 0.123, "critic_obs_dim": 17},
    )
    monkeypatch.setattr(owner_module, "ensure_registries", lambda: None)
    monkeypatch.setattr(owner_module, "create_env", lambda *args, **kwargs: _FakeEnv())
    monkeypatch.setattr(
        owner_module,
        "resolve_custom_offpolicy_runtime",
        lambda _cfg: custom_runtime,
    )
    monkeypatch.setattr(owner_module, "DoubleBufferOffPolicyRunner", _FakeRunner)

    runner = owner_module.build_sac_double_buffer_runner(
        cfg,
        env_cfg_override={},
        replay_prefetch_mode="one_tick",
        device="cuda:0",
    )

    assert runner.kwargs["algo_type"] == "custom_sac"
    assert runner.kwargs["learner"].kwargs["gamma"] == pytest.approx(0.123)
    assert runner.kwargs["learner"].kwargs["critic_obs_dim"] == 17
    assert runner.kwargs["learner"].kwargs["tau"] == cfg.algo.tau


def test_sac_owner_rejects_custom_runtime_without_symmetry_support(
    monkeypatch: pytest.MonkeyPatch,
):
    from unilab.algos.torch.fast_sac import double_buffer as owner_module
    from unilab.algos.torch.offpolicy.runtime import OffPolicyRuntime

    cfg = _offpolicy_cfg(["algo=sac", "algo.use_symmetry=true"])
    monkeypatch.setattr(owner_module, "ensure_registries", lambda: None)
    monkeypatch.setattr(owner_module, "create_env", lambda *args, **kwargs: _FakeEnv())
    monkeypatch.setattr(
        owner_module,
        "resolve_custom_offpolicy_runtime",
        lambda _cfg: OffPolicyRuntime(supports_symmetry=False),
    )

    with pytest.raises(ValueError, match="does not support symmetry"):
        owner_module.build_sac_double_buffer_runner(
            cfg,
            env_cfg_override={},
            replay_prefetch_mode="one_tick",
            device="cuda:0",
        )


def test_sac_owner_preserves_symmetry_batch_and_learner_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    import unilab.algos.torch.fast_sac.double_buffer as owner_module

    cfg = _offpolicy_cfg(["algo=sac", "algo.use_symmetry=true"])
    symmetry = MagicMock(batch_multiplier=4)

    class SymmetricEnv(_FakeEnv):
        def build_symmetry_augmentation(self, device=None):
            assert device == "cuda:0"
            return symmetry

    monkeypatch.setattr(owner_module, "ensure_registries", lambda: None)
    monkeypatch.setattr(owner_module, "create_env", lambda *args, **kwargs: SymmetricEnv())
    monkeypatch.setattr(owner_module, "FastSACLearner", _FakeLearner)
    monkeypatch.setattr(owner_module, "DoubleBufferOffPolicyRunner", _FakeRunner)

    runner = owner_module.build_sac_double_buffer_runner(
        cfg,
        env_cfg_override={},
        replay_prefetch_mode="one_tick",
        device="cuda:0",
    )

    assert runner.kwargs["batch_size"] == cfg.algo.batch_size // 4
    learner_kwargs = runner.kwargs["learner"].kwargs
    assert learner_kwargs["use_symmetry"] is True
    assert learner_kwargs["symmetry_augmentation"] is symmetry


@pytest.mark.parametrize(
    ("batch_size", "symmetry", "match"),
    [
        (512, None, "does not provide symmetry augmentation"),
        (10, MagicMock(batch_multiplier=4), "batch_size divisible by 4"),
    ],
)
def test_sac_owner_preserves_symmetry_validation(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
    symmetry: MagicMock | None,
    match: str,
):
    import unilab.algos.torch.fast_sac.double_buffer as owner_module

    cfg = _offpolicy_cfg(["algo=sac", "algo.use_symmetry=true", f"algo.batch_size={batch_size}"])

    class SymmetricEnv(_FakeEnv):
        def build_symmetry_augmentation(self, device=None):
            del device
            return symmetry

    monkeypatch.setattr(owner_module, "ensure_registries", lambda: None)
    monkeypatch.setattr(owner_module, "create_env", lambda *args, **kwargs: SymmetricEnv())

    with pytest.raises(ValueError, match=match):
        owner_module.build_sac_double_buffer_runner(
            cfg,
            env_cfg_override={},
            replay_prefetch_mode="one_tick",
            device="cuda:0",
        )


def test_td3_dispatch_constructs_unique_runner(monkeypatch: pytest.MonkeyPatch):
    module = _offpolicy()
    cfg = _offpolicy_cfg(["algo=td3"])

    import unilab.algos.torch.fast_td3.double_buffer as owner_module

    monkeypatch.setattr(owner_module, "get_env_dims", lambda *args, **kwargs: (4, 2, 6))
    monkeypatch.setattr(owner_module, "FastTD3Learner", _FakeLearner)
    monkeypatch.setattr(owner_module, "DoubleBufferOffPolicyRunner", _FakeRunner)

    runner = module.build_runner("td3", cfg)
    assert isinstance(runner, _FakeRunner)
    assert runner.kwargs["algo_type"] == "td3"
    assert runner.kwargs["device"] == "cuda:0"
    assert "inference_owner" not in runner.kwargs
    assert "collector_infer_device" not in runner.kwargs
    assert "sync_collection" not in runner.kwargs
    assert "replay_pipeline" not in runner.kwargs
    assert runner.kwargs["replay_prefetch_mode"] == "one_tick"
    nan_guard_cfg = runner.kwargs["nan_guard_cfg"]
    assert nan_guard_cfg.enabled is True
    assert nan_guard_cfg.buffer_size == cfg.training.nan_guard.buffer_size
    assert nan_guard_cfg.max_envs_to_dump == cfg.training.nan_guard.max_envs_to_dump
    assert nan_guard_cfg.output_dir == cfg.training.nan_guard.output_dir
    assert runner.kwargs["torch_thread_runtime"] is not None
    assert runner.kwargs["collector_cpu_ids"] is None
    assert runner.kwargs["dp_sync"] is None
    assert runner.kwargs["learner"].kwargs == {
        "obs_dim": 4,
        "action_dim": 2,
        "critic_obs_dim": 6,
        "num_envs": cfg.algo.num_envs,
        "device": "cuda:0",
        "gamma": cfg.algo.gamma,
        "tau": cfg.algo.tau,
        "actor_lr": cfg.algo.actor_lr,
        "critic_lr": cfg.algo.critic_lr,
        "actor_hidden_dim": cfg.algo.actor_hidden_dim,
        "critic_hidden_dim": cfg.algo.critic_hidden_dim,
        "num_atoms": cfg.algo.num_atoms,
        "v_min": cfg.algo.algo_params.v_min,
        "v_max": cfg.algo.algo_params.v_max,
        "init_scale": cfg.algo.algo_params.init_scale,
        "log_std_min": cfg.algo.algo_params.log_std_min,
        "log_std_max": cfg.algo.algo_params.log_std_max,
        "weight_decay": cfg.algo.algo_params.weight_decay,
        "use_cdq": cfg.algo.algo_params.use_cdq,
        "policy_noise": cfg.algo.algo_params.policy_noise,
        "noise_clip": cfg.algo.algo_params.noise_clip,
        "policy_frequency": cfg.algo.policy_frequency,
        "obs_normalization": cfg.algo.obs_normalization,
    }

    nan_guard = MagicMock()
    thread_runtime = {"marker": "threads"}
    dp_sync = MagicMock()
    forwarded = owner_module.build_td3_double_buffer_runner(
        cfg,
        env_cfg_override={},
        replay_prefetch_mode="one_tick",
        device="cuda:0",
        nan_guard_cfg=nan_guard,
        torch_thread_runtime=thread_runtime,
        collector_cpu_ids=[2, 3],
        dp_sync=dp_sync,
    )
    assert forwarded.kwargs["nan_guard_cfg"] is nan_guard
    assert forwarded.kwargs["torch_thread_runtime"] is thread_runtime
    assert forwarded.kwargs["collector_cpu_ids"] == [2, 3]
    assert forwarded.kwargs["dp_sync"] is dp_sync


def test_flashsac_dispatch_constructs_unique_runner(monkeypatch: pytest.MonkeyPatch):
    module = _offpolicy()
    cfg = _offpolicy_cfg(["algo=flashsac"])

    import unilab.algos.torch.flash_sac.double_buffer as flash_module

    monkeypatch.setattr(flash_module, "ensure_registries", lambda: None)
    monkeypatch.setattr(flash_module, "create_env", lambda *args, **kwargs: _FakeEnv())
    monkeypatch.setattr(flash_module, "FlashSACLearner", _FakeLearner)
    monkeypatch.setattr(flash_module, "DoubleBufferOffPolicyRunner", _FakeRunner)

    runner = module.build_runner("flashsac", cfg)
    assert isinstance(runner, _FakeRunner)
    assert runner.kwargs["algo_type"] == "flashsac"
    assert runner.kwargs["device"] == "cuda:0"
    assert runner.kwargs["replay_prefetch_mode"] == "one_tick"
    assert "inference_owner" not in runner.kwargs
    assert "collector_infer_device" not in runner.kwargs
    assert "sync_collection" not in runner.kwargs
    assert "replay_pipeline" not in runner.kwargs


def test_flashsac_n_step_is_rejected():
    cfg = _offpolicy_cfg(["algo=flashsac", "algo.algo_params.n_step=3"])
    with pytest.raises(ValueError, match="n_step=1 only"):
        _offpolicy().build_runner("flashsac", cfg)


def _bare_runner():
    from unilab.algos.torch.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner

    return object.__new__(DoubleBufferOffPolicyRunner)


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
):
    """build_runner("sac", ...) with learner/env/runner fakes; returns captured state."""
    module = _offpolicy()
    cfg = _offpolicy_cfg(overrides)
    probe_env_calls: list[dict] = []

    def fake_create_env(*args, **kwargs):
        del args
        probe_env_calls.append(kwargs)
        return _FakeEnv()

    monkeypatch.setattr(module.os, "cpu_count", lambda: cpu_count)

    import unilab.algos.torch.fast_sac.double_buffer as owner_module

    monkeypatch.setattr(owner_module, "ensure_registries", lambda: None)
    monkeypatch.setattr(owner_module, "create_env", fake_create_env)
    monkeypatch.setattr(owner_module, "FastSACLearner", _FakeLearner)
    monkeypatch.setattr(owner_module, "DoubleBufferOffPolicyRunner", _FakeRunner)

    runner = module.build_runner("sac", cfg, log_dir="/tmp/offpolicy_test_run")
    return runner, probe_env_calls


def test_build_runner_partitions_collector_cpus_per_rank(monkeypatch: pytest.MonkeyPatch):
    # Spawned rank: rank comes from the env, world_size from training.devices.
    monkeypatch.setenv(UNILAB_DP_RANK, "1")
    monkeypatch.setenv(UNILAB_DP_LOG_DIR, "/tmp/offpolicy_test_run")
    runner, probe_env_calls = _build_sac_runner_with_fakes(
        monkeypatch,
        ["algo=sac", "algo.use_symmetry=false", "training.devices=[0,1]"],
        cpu_count=128,
    )
    assert runner.kwargs["collector_cpu_ids"] == list(range(64, 128))
    # The thread budget is resolved against the rank's CPU share, not the host.
    assert runner.kwargs["torch_thread_runtime"]["cpu_count"] == 64
    # The num_envs=1 probe env must never see cpu_ids (it would size its
    # MuJoCo BatchEnvPool worker count from len(cpu_ids)).
    assert probe_env_calls
    for call in probe_env_calls:
        override = call.get("env_cfg_override") or {}
        assert "cpu_ids" not in override


def test_build_runner_rank_zero_partitions_without_dp_env(monkeypatch: pytest.MonkeyPatch):
    # Rank 0 carries no UNILAB_DP_* env; world_size must come from the config.
    monkeypatch.delenv(UNILAB_DP_RANK, raising=False)
    monkeypatch.delenv(UNILAB_DP_WORLD_SIZE, raising=False)
    runner, _ = _build_sac_runner_with_fakes(
        monkeypatch,
        ["algo=sac", "algo.use_symmetry=false", "training.devices=[0,1]"],
        cpu_count=128,
    )
    assert runner.kwargs["collector_cpu_ids"] == list(range(0, 64))
    assert runner.kwargs["torch_thread_runtime"]["cpu_count"] == 64


def test_build_runner_single_rank_keeps_collector_cpus_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(UNILAB_DP_RANK, raising=False)
    monkeypatch.delenv(UNILAB_DP_WORLD_SIZE, raising=False)
    runner, probe_env_calls = _build_sac_runner_with_fakes(
        monkeypatch, ["algo=sac", "algo.use_symmetry=false"], cpu_count=128
    )
    assert runner.kwargs["collector_cpu_ids"] is None
    # Single-rank thread budget still resolves against the full host.
    assert runner.kwargs["torch_thread_runtime"]["cpu_count"] == 128
    override = runner.kwargs["env_cfg_override"] or {}
    assert "cpu_ids" not in override
    for call in probe_env_calls:
        assert "cpu_ids" not in (call.get("env_cfg_override") or {})


def test_build_runner_explicit_dp_collector_cpu_ids(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(UNILAB_DP_RANK, "0")
    runner, probe_env_calls = _build_sac_runner_with_fakes(
        monkeypatch,
        [
            "algo=sac",
            "algo.use_symmetry=false",
            "training.devices=[0,1]",
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
    assert runner._collector_env_cfg_override() is runner.env_cfg_override


def test_collector_env_cfg_override_from_none_base():
    runner = _bare_runner()
    runner.env_cfg_override = None
    runner.collector_cpu_ids = [4, 5]
    assert runner._collector_env_cfg_override() == {"cpu_ids": [4, 5]}
