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
    device_selected = False
    for override in overrides or []:
        if override.startswith("algo="):
            algo = override.split("=", 1)[1]
        elif override.startswith("task="):
            task_selected = True
        elif override.startswith("training.device="):
            device_selected = True
        normalized.append(override)
    if not task_selected:
        normalized.append(f"task={algo}/g1_walk_flat/mujoco")
    if not device_selected:
        normalized.append("training.device=cuda")
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
    assert "replay_pipeline" not in cfg.training
    assert "verbose_metrics" not in cfg.training
    assert "replay_pack_layout" not in cfg.training
    assert "replay_pack_executor" not in cfg.training
    assert "replay_h2d_submitter" not in cfg.training


@pytest.mark.parametrize(
    "override",
    [
        "training.replay_pipeline=cpu_pinned_double_buffer",
        "training.verbose_metrics=true",
        "training.num_gpus=2",
        "training.multi_gpu_sync_mode=sync_sgd",
        "training.multi_gpu_sync_interval=2",
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
def test_unsupported_training_device_fails_before_env_materialization(
    monkeypatch: pytest.MonkeyPatch,
    algo: str,
    device: str,
):
    module = _offpolicy()
    cfg = _offpolicy_cfg([f"algo={algo}", f"training.device={device}"])
    env_calls = 0

    def reject_env(*args, **kwargs):
        nonlocal env_calls
        del args, kwargs
        env_calls += 1
        raise AssertionError("unsupported replay device must fail before env creation")

    monkeypatch.setattr(module, "create_env", reject_env)
    with pytest.raises(ValueError, match="CUDA or MPS learner device"):
        module.build_runner(algo, cfg)
    assert env_calls == 0


def test_sac_dispatch_constructs_unique_runner(monkeypatch: pytest.MonkeyPatch):
    module = _offpolicy()
    cfg = _offpolicy_cfg(["algo=sac", "algo.use_symmetry=false"])
    monkeypatch.setattr(module, "ensure_registries", lambda: None)
    monkeypatch.setattr(module, "create_env", lambda *args, **kwargs: _FakeEnv())

    import unilab.algos.torch.fast_sac.learner as learner_module
    import unilab.algos.torch.offpolicy.double_buffer_runner as runner_module

    monkeypatch.setattr(learner_module, "FastSACLearner", _FakeLearner)
    monkeypatch.setattr(runner_module, "DoubleBufferOffPolicyRunner", _FakeRunner)

    runner = module.build_runner("sac", cfg)
    assert isinstance(runner, _FakeRunner)
    assert runner.kwargs["algo_type"] == "sac"
    assert runner.kwargs["device"] == "cuda"
    assert runner.kwargs["replay_prefetch_mode"] == "one_tick"
    assert "replay_pipeline" not in runner.kwargs
    assert "verbose_metrics" not in runner.kwargs


def test_td3_dispatch_constructs_unique_runner(monkeypatch: pytest.MonkeyPatch):
    module = _offpolicy()
    cfg = _offpolicy_cfg(["algo=td3", "training.no_sync_collection=true"])

    import unilab.algos.torch.common.device as device_module
    import unilab.algos.torch.fast_td3.learner as learner_module
    import unilab.algos.torch.offpolicy.double_buffer_runner as runner_module

    monkeypatch.setattr(device_module, "get_env_dims", lambda *args, **kwargs: (4, 2, 6))
    monkeypatch.setattr(learner_module, "FastTD3Learner", _FakeLearner)
    monkeypatch.setattr(runner_module, "DoubleBufferOffPolicyRunner", _FakeRunner)

    runner = module.build_runner("td3", cfg)
    assert isinstance(runner, _FakeRunner)
    assert runner.kwargs["algo_type"] == "td3"
    assert runner.kwargs["sync_collection"] is False
    assert runner.kwargs["device"] == "cuda"
    assert "replay_pipeline" not in runner.kwargs


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
    assert runner.kwargs["device"] == "cuda"
    assert runner.kwargs["replay_prefetch_mode"] == "one_tick"
    assert "replay_pipeline" not in runner.kwargs


def test_flashsac_async_collection_is_rejected():
    cfg = _offpolicy_cfg(["algo=flashsac", "training.no_sync_collection=true"])
    with pytest.raises(ValueError, match="requires synchronized collection"):
        _offpolicy().build_runner("flashsac", cfg)


def test_flashsac_n_step_is_rejected():
    cfg = _offpolicy_cfg(["algo=flashsac", "algo.algo_params.n_step=3"])
    with pytest.raises(ValueError, match="n_step=1 only"):
        _offpolicy().build_runner("flashsac", cfg)


def _bare_runner():
    from unilab.algos.torch.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner

    return object.__new__(DoubleBufferOffPolicyRunner)


def test_safe_put_detects_dead_collector():
    runner = _bare_runner()
    runner._check_collector_alive = lambda: False
    full_queue: queue.Queue = queue.Queue(maxsize=1)
    full_queue.put_nowait(1)

    with pytest.raises(RuntimeError, match="collector dead"):
        runner._safe_put_trainer_done(full_queue, timeout=0.01)


def test_safe_put_noops_without_sync_queue():
    _bare_runner()._safe_put_trainer_done(None)
