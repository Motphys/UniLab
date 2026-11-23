"""Forwarding tests for the double-buffer runner builder helpers.

The builders are the only public construction path for
``DoubleBufferOffPolicyRunner`` from Hydra owner configs; every runner kwarg
the builders accept must reach the runner unchanged. Tensor-runtime settings
and inference placement are resolved by the training assembly layer
(``unilab.training.tensor_runtime``) and injected; backend process-device
resolution/binding is injected as callables (issue #2372 removed the builders'
own Hydra tensor-runtime resolution and the ``backend_device_binder`` kwarg).

Merged from unilab_rl @ 3654e4b (unilab-rl 1.4.15)
``tests/algos/test_double_buffer_builders.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from unilab.rl.runtime.placement import InferencePlacement
from unilab.rl.runtime.settings import RuntimeSettings
from unilab.training.tensor_runtime import resolve_collector_tensor_placement


class _FakeEnv:
    obs_groups_spec = {"obs": 4, "critic": 6}
    action_space = SimpleNamespace(shape=(2,))

    def init_state(self):
        return SimpleNamespace(obs={"obs": torch.zeros((1, 4)), "critic": torch.zeros((1, 6))})

    def close(self):
        return None


def _fake_env_factory(num_envs, env_cfg_override):
    del num_envs, env_cfg_override
    return _FakeEnv()


class _FakeLearner:
    last_kwargs: dict[str, Any] = {}

    def __init__(self, *args, **kwargs):
        del args
        type(self).last_kwargs = kwargs


class _FakeRunner:
    def __init__(self, *args, **kwargs):
        del args
        self.kwargs = kwargs


def _resolver(backend: str, learner_device: str | None) -> str | None:
    del backend, learner_device
    return None


def _configurer(backend: str, learner_device: str | None) -> str | None:
    del backend, learner_device
    return None


def _injected_settings() -> RuntimeSettings:
    # batch_size/updates_per_step deliberately differ from _algo_cfg() so the
    # tests prove the builder honors the injected settings over ``cfg.algo``.
    return RuntimeSettings(
        inference_slot_capacity=3,
        collector_metrics_interval=7,
        replay_ingress_depth=4,
        replay_ingress_slot_rows=2,
        batch_size=8,
        updates_per_step=2,
        num_envs=4,
    )


def _training_cfg() -> dict[str, Any]:
    return {
        "task_name": "FakeTask",
        "sim_backend": "fake",
        "env_steps_per_sync": 1,
        "use_amp": False,
        "trace_enabled": False,
        "trace_output_dir": "logs",
        "trace_thread_time": False,
        "trace_cuda_events": False,
        "log_interval": 3,
    }


def _algo_cfg(extra: dict[str, Any]) -> dict[str, Any]:
    base = {
        "num_envs": 4,
        "replay_buffer_n": 8,
        "batch_size": 4,
        "learning_starts": 4,
        "updates_per_step": 1,
        "policy_frequency": 1,
        "seed": 1,
        "gamma": 0.99,
        "tau": 0.005,
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "actor_hidden_dim": 8,
        "critic_hidden_dim": 8,
        "num_atoms": 1,
        "obs_normalization": False,
    }
    base.update(extra)
    return base


def _sac_cfg() -> Any:
    return OmegaConf.create(
        {
            "training": _training_cfg(),
            "algo": _algo_cfg(
                {
                    "use_layer_norm": False,
                    "algo_params": {
                        "alpha_lr": 1e-3,
                        "alpha_init": 1.0,
                        "target_entropy_ratio": 1.0,
                        "max_grad_norm": 1.0,
                        "amp_dtype": "bf16",
                        "use_compile": False,
                    },
                }
            ),
        }
    )


def _flashsac_cfg() -> Any:
    return OmegaConf.create(
        {
            "training": _training_cfg(),
            "algo": _algo_cfg(
                {
                    "algo_params": {
                        "actor_num_blocks": 1,
                        "critic_num_blocks": 1,
                        "critic_min_v": -10.0,
                        "critic_max_v": 10.0,
                        "temp_initial_value": 1.0,
                        "temp_target_sigma": 1.0,
                        "temp_target_entropy": 1.0,
                        "actor_bc_alpha": 1.0,
                        "actor_noise_zeta_mu": 0.0,
                        "actor_noise_zeta_max": 0.0,
                        "learning_rate_init": 1e-3,
                        "learning_rate_peak": 1e-3,
                        "learning_rate_end": 1e-3,
                        "learning_rate_warmup_steps": 1,
                        "learning_rate_decay_steps": 1,
                        "normalize_reward": False,
                        "normalized_g_max": 1.0,
                        "n_step": 1,
                        "amp_dtype": "bf16",
                        "use_compile": False,
                        "compile_full_objectives": True,
                        "use_whole_cycle_cuda_graph": False,
                    },
                }
            ),
        }
    )


def _patch_flashsac_host_gates(monkeypatch: pytest.MonkeyPatch, module: Any) -> None:
    # CPU-only test host: bypass the CUDA/MPS replay-device gate and seeding.
    monkeypatch.setattr(
        module,
        "require_offpolicy_replay_device",
        lambda device: device,
        raising=False,
    )
    monkeypatch.setattr(module, "apply_worker_seed", lambda *args, **kwargs: None)


@pytest.mark.parametrize("with_injection", [False, True])
def test_sac_builder_forwards_backend_device_injection(
    monkeypatch: pytest.MonkeyPatch, with_injection: bool
) -> None:
    import unilab.rl.algos.sac.double_buffer as module

    monkeypatch.setattr(module, "SACLearner", _FakeLearner)
    monkeypatch.setattr(module, "DoubleBufferOffPolicyRunner", _FakeRunner)
    monkeypatch.setattr(
        module,
        "require_offpolicy_replay_device",
        lambda device: device,
        raising=False,
    )

    kwargs: dict[str, Any] = {}
    if with_injection:
        kwargs["backend_device_resolver"] = _resolver
        kwargs["backend_device_configurer"] = _configurer
    runner = module.build_double_buffer_runner(
        _sac_cfg(),
        env_factory=_fake_env_factory,
        env_cfg_override=None,
        replay_prefetch_mode="one_tick",
        device="cpu",
        **kwargs,
    )

    assert runner.kwargs["backend_device_resolver"] is (_resolver if with_injection else None)
    assert runner.kwargs["backend_device_configurer"] is (_configurer if with_injection else None)
    assert runner.kwargs["log_interval"] == 3
    assert runner.kwargs["replay_pipeline_factory"].keywords == {"storage_dtype": "float16"}
    # Without injection the runner-side defaults apply.
    assert runner.kwargs["inference_placement"] is None
    assert runner.kwargs["runtime_settings"] is None


@pytest.mark.parametrize("with_injection", [False, True])
def test_flashsac_builder_forwards_backend_device_injection(
    monkeypatch: pytest.MonkeyPatch, with_injection: bool
) -> None:
    import unilab.rl.algos.flash_sac.double_buffer as module

    monkeypatch.setattr(module, "FlashSACLearner", _FakeLearner)
    monkeypatch.setattr(module, "DoubleBufferOffPolicyRunner", _FakeRunner)
    _patch_flashsac_host_gates(monkeypatch, module)

    kwargs: dict[str, Any] = {}
    if with_injection:
        kwargs["backend_device_resolver"] = _resolver
        kwargs["backend_device_configurer"] = _configurer
    runner = module.build_double_buffer_runner(
        _flashsac_cfg(),
        env_factory=_fake_env_factory,
        env_cfg_override=None,
        replay_prefetch_mode="one_tick",
        device="cpu",
        **kwargs,
    )

    assert runner.kwargs["backend_device_resolver"] is (_resolver if with_injection else None)
    assert runner.kwargs["backend_device_configurer"] is (_configurer if with_injection else None)
    assert runner.kwargs["log_interval"] == 3
    assert runner.kwargs["inference_placement"] is None
    assert runner.kwargs["runtime_settings"] is None
    assert _FakeLearner.last_kwargs["compile_full_objectives"] is True
    assert _FakeLearner.last_kwargs["use_whole_cycle_cuda_graph"] is False
    assert _FakeLearner.last_kwargs["actor_normalize_parameters"] is True
    assert _FakeLearner.last_kwargs["critic_normalize_parameters"] is True
    assert runner.kwargs["target_frequency"] == 1
    assert runner.kwargs["policy_before_critic"] is False
    assert runner.kwargs["replay_pipeline_factory"].keywords == {
        "decay_step": 0,
        "min_weight": 0.1,
        "num_buckets": 2000,
        "storage_dtype": "float16",
    }


@pytest.mark.parametrize("algo", ["sac", "flashsac"])
def test_double_buffer_builders_forward_injected_tensor_runtime(
    monkeypatch: pytest.MonkeyPatch, algo: str
) -> None:
    if algo == "sac":
        import unilab.rl.algos.sac.double_buffer as module

        cfg = _sac_cfg()
        learner_name = "SACLearner"
        build = module.build_double_buffer_runner
        monkeypatch.setattr(
            module,
            "require_offpolicy_replay_device",
            lambda device: device,
            raising=False,
        )
    else:
        import unilab.rl.algos.flash_sac.double_buffer as module

        cfg = _flashsac_cfg()
        learner_name = "FlashSACLearner"
        build = module.build_double_buffer_runner
        _patch_flashsac_host_gates(monkeypatch, module)

    monkeypatch.setattr(module, learner_name, _FakeLearner)
    monkeypatch.setattr(module, "DoubleBufferOffPolicyRunner", _FakeRunner)
    settings = _injected_settings()
    placement = resolve_collector_tensor_placement(cfg, device="cpu", algo_name=algo.upper())

    runner = build(
        cfg,
        env_factory=_fake_env_factory,
        env_cfg_override=None,
        replay_prefetch_mode="one_tick",
        device="cpu",
        runtime_settings=settings,
        inference_placement=placement,
    )

    assert runner.kwargs["runtime_settings"] is settings
    assert runner.kwargs["inference_placement"] is placement
    assert isinstance(runner.kwargs["inference_placement"], InferencePlacement)
    assert runner.kwargs["inference_placement"].collector_tensor_native is False
    assert runner.kwargs["inference_placement"].mode.value == "host"
    # The injected settings win over cfg.algo for learner sampling.
    assert runner.kwargs["batch_size"] == 8
    assert runner.kwargs["updates_per_step"] == 2


def test_sac_builder_forwards_custom_runtime_preparation_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import unilab.rl.algos.sac.double_buffer as module
    from unilab.rl.runtime.offpolicy.resolution import OffPolicyRuntime

    def prepare_hook(learner, context):
        del learner, context

    runtime = OffPolicyRuntime(learner_cls=_FakeLearner, learner_prepare_hook=prepare_hook)
    monkeypatch.setattr(module, "SACLearner", _FakeLearner)
    monkeypatch.setattr(module, "DoubleBufferOffPolicyRunner", _FakeRunner)
    monkeypatch.setattr(
        module,
        "require_offpolicy_replay_device",
        lambda device: device,
        raising=False,
    )
    monkeypatch.setattr(module, "resolve_custom_offpolicy_runtime", lambda rl_cfg: runtime)

    cfg = _sac_cfg()
    runner = module.build_double_buffer_runner(
        cfg,
        env_factory=_fake_env_factory,
        env_cfg_override=None,
        replay_prefetch_mode="one_tick",
        device="cpu",
    )

    assert runner.kwargs["learner_prepare_hook"] is prepare_hook


@pytest.mark.parametrize("algo", ["sac", "flashsac"])
def test_double_buffer_builders_validate_no_inference_request_timeout(
    monkeypatch: pytest.MonkeyPatch, algo: str
) -> None:
    if algo == "sac":
        import unilab.rl.algos.sac.double_buffer as module

        build = module.build_double_buffer_runner
        cfg = _sac_cfg()
        monkeypatch.setattr(
            module,
            "require_offpolicy_replay_device",
            lambda device: device,
            raising=False,
        )
    else:
        import unilab.rl.algos.flash_sac.double_buffer as module

        build = module.build_double_buffer_runner
        cfg = _flashsac_cfg()
        _patch_flashsac_host_gates(monkeypatch, module)

    cfg.training.inference_request_timeout_sec = 17.0
    with pytest.raises(ValueError, match="inference_request_timeout_sec was removed"):
        build(
            cfg,
            env_factory=_fake_env_factory,
            env_cfg_override=None,
            replay_prefetch_mode="one_tick",
            device="cpu",
        )
