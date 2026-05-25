"""Tests for DoubleBufferOffPolicyRunner dispatch and config integration."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
_CONF_DIR = Path(__file__).parent.parent.parent / "conf"


def _load_script(name: str):
    path = _SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _offpolicy():
    return _load_script("train_offpolicy")


def _offpolicy_cfg(overrides=None):
    GlobalHydra.instance().clear()
    normalized = []
    algo = "sac"
    task_selected = False
    for o in overrides or []:
        if o.startswith("algo="):
            algo = o.split("=", 1)[1]
            normalized.append(o)
            continue
        if o.startswith("task="):
            task_selected = True
            normalized.append(o)
            continue
        normalized.append(o)
    if not task_selected:
        normalized.append(f"task={algo}/g1_walk_flat/mujoco")
    with initialize_config_dir(config_dir=str(_CONF_DIR / "offpolicy"), version_base="1.3"):
        return compose("config", overrides=normalized, return_hydra_config=True)


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_default_replay_prefetch_mode_is_one_tick():
    cfg = _offpolicy_cfg()
    assert cfg.training.replay_prefetch_mode == "one_tick"


def test_b_path_internal_knobs_are_not_configured():
    cfg = _offpolicy_cfg()
    assert "replay_pack_layout" not in cfg.training
    assert "replay_pack_executor" not in cfg.training
    assert "replay_h2d_submitter" not in cfg.training


def test_cli_override_replay_prefetch_mode_accepted():
    cfg = _offpolicy_cfg(["training.replay_prefetch_mode=one_tick"])
    assert cfg.training.replay_prefetch_mode == "one_tick"


def test_invalid_replay_prefetch_mode_rejected():
    cfg = _offpolicy_cfg(["training.replay_prefetch_mode=invalid_mode"])
    with pytest.raises(ValueError, match="Unsupported training.replay_prefetch_mode"):
        _offpolicy().build_runner("sac", cfg)


def test_same_tick_replay_prefetch_mode_rejected():
    cfg = _offpolicy_cfg(["training.replay_prefetch_mode=same_tick"])
    with pytest.raises(ValueError, match="Unsupported training.replay_prefetch_mode"):
        _offpolicy().build_runner("sac", cfg)


# ---------------------------------------------------------------------------
# Dispatch rejections
# ---------------------------------------------------------------------------


def test_td3_dispatches_to_double_buffer_runner(monkeypatch: pytest.MonkeyPatch):
    import unilab.algos.torch.common.device as device_mod
    import unilab.algos.torch.fast_td3.learner as learner_mod
    import unilab.algos.torch.offpolicy.double_buffer_runner as db_mod

    cfg = _offpolicy_cfg(["algo=td3"])

    class _FakeLearner:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    class _FakeRunner:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(device_mod, "get_env_dims", lambda *args, **kwargs: (4, 2, 6))
    monkeypatch.setattr(learner_mod, "FastTD3Learner", _FakeLearner)
    monkeypatch.setattr(db_mod, "DoubleBufferOffPolicyRunner", _FakeRunner)

    runner = _offpolicy().build_runner("td3", cfg)

    assert isinstance(runner, _FakeRunner)
    assert runner.kwargs["algo_type"] == "td3"
    assert runner.kwargs["learner"].kwargs["critic_obs_dim"] == 6


def test_sac_multi_gpu_rejects_cpu_pinned_double_buffer():
    cfg = _offpolicy_cfg(
        [
            "algo=sac",
            "training.num_gpus=2",
        ]
    )
    with pytest.raises(ValueError, match="currently single-GPU only"):
        _offpolicy().build_runner("sac", cfg)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_sac_portable_devices_allow_cpu_pinned_double_buffer(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
):
    import gymnasium as gym

    mod = _offpolicy()
    cfg = _offpolicy_cfg(
        [
            "algo=sac",
            f"training.device={device}",
            "algo.use_symmetry=false",
        ]
    )

    class _FakeEnv:
        obs_groups_spec = {"obs": 4, "critic": 6}
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))

        def build_symmetry_augmentation(self, device=None):
            return None

        def close(self):
            pass

    class _FakeLearner:
        class actor:
            @staticmethod
            def state_dict():
                return {"w": MagicMock(shape=(4,))}

        update_count = 0

        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    class _FakeRunner:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(mod, "ensure_registries", lambda: None)
    monkeypatch.setattr(mod, "create_env", lambda *args, **kwargs: _FakeEnv())

    import unilab.algos.torch.fast_sac.learner as learner_mod

    monkeypatch.setattr(learner_mod, "FastSACLearner", _FakeLearner)

    import unilab.algos.torch.offpolicy.double_buffer_runner as db_mod

    monkeypatch.setattr(db_mod, "DoubleBufferOffPolicyRunner", _FakeRunner)

    runner = mod.build_runner("sac", cfg)

    assert isinstance(runner, _FakeRunner)
    assert runner.kwargs["device"] == device
    assert runner.kwargs["replay_prefetch_mode"] == "one_tick"
    assert runner.kwargs["learner"].kwargs["use_compile"] is True


def test_sac_compile_override_is_passed_to_learner(monkeypatch: pytest.MonkeyPatch):
    import gymnasium as gym

    mod = _offpolicy()
    cfg = _offpolicy_cfg(
        [
            "algo=sac",
            "training.device=cuda",
            "algo.use_symmetry=false",
            "algo.algo_params.use_compile=true",
        ]
    )

    class _FakeEnv:
        obs_groups_spec = {"obs": 4, "critic": 6}
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))

        def build_symmetry_augmentation(self, device=None):
            return None

        def close(self):
            pass

    class _FakeLearner:
        class actor:
            @staticmethod
            def state_dict():
                return {"w": MagicMock(shape=(4,))}

        update_count = 0

        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    class _FakeRunner:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(mod, "ensure_registries", lambda: None)
    monkeypatch.setattr(mod, "create_env", lambda *args, **kwargs: _FakeEnv())

    import unilab.algos.torch.fast_sac.learner as learner_mod

    monkeypatch.setattr(learner_mod, "FastSACLearner", _FakeLearner)

    import unilab.algos.torch.offpolicy.double_buffer_runner as db_mod

    monkeypatch.setattr(db_mod, "DoubleBufferOffPolicyRunner", _FakeRunner)

    runner = mod.build_runner("sac", cfg)

    assert runner.kwargs["learner"].kwargs["use_compile"] is True


def test_sac_async_collection_rejects_cpu_pinned_double_buffer():
    cfg = _offpolicy_cfg(
        [
            "algo=sac",
            "training.no_sync_collection=true",
        ]
    )
    with pytest.raises(ValueError, match="requires synchronized collection"):
        _offpolicy().build_runner("sac", cfg)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_flashsac_double_buffer_portable_devices_allowed(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
):
    import gymnasium as gym

    import unilab.algos.torch.flash_sac.double_buffer as flash_db_mod

    mod = _offpolicy()
    cfg = _offpolicy_cfg(
        [
            "algo=flashsac",
            f"training.device={device}",
        ]
    )

    class _FakeEnv:
        obs_groups_spec = {"obs": 4, "critic": 6}
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))

        def close(self):
            pass

    class _FakeLearner:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    class _FakeRunner:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(flash_db_mod, "ensure_registries", lambda: None)
    monkeypatch.setattr(flash_db_mod, "create_env", lambda *args, **kwargs: _FakeEnv())
    monkeypatch.setattr(flash_db_mod, "FlashSACLearner", _FakeLearner)
    monkeypatch.setattr(flash_db_mod, "DoubleBufferOffPolicyRunner", _FakeRunner)

    runner = mod.build_runner("flashsac", cfg)

    assert isinstance(runner, _FakeRunner)
    assert runner.kwargs["device"] == device


def test_flashsac_double_buffer_multi_gpu_rejected():
    cfg = _offpolicy_cfg(
        [
            "algo=flashsac",
            "training.device=cuda",
            "training.num_gpus=2",
        ]
    )
    with pytest.raises(ValueError, match="FlashSAC does not support training.num_gpus > 1"):
        _offpolicy().build_runner("flashsac", cfg)


def test_flashsac_double_buffer_async_collection_rejected():
    cfg = _offpolicy_cfg(
        [
            "algo=flashsac",
            "training.device=cuda",
            "training.no_sync_collection=true",
        ]
    )
    with pytest.raises(ValueError, match="requires synchronized collection"):
        _offpolicy().build_runner("flashsac", cfg)


def test_flashsac_double_buffer_n_step_rejected():
    cfg = _offpolicy_cfg(
        [
            "algo=flashsac",
            "training.device=cuda",
            "algo.algo_params.n_step=3",
        ]
    )
    with pytest.raises(ValueError, match="n_step=1 only"):
        _offpolicy().build_runner("flashsac", cfg)


def test_flashsac_double_buffer_compile_passed_to_learner(monkeypatch: pytest.MonkeyPatch):
    import gymnasium as gym

    import unilab.algos.torch.flash_sac.double_buffer as flash_db_mod

    cfg = _offpolicy_cfg(
        [
            "algo=flashsac",
            "training.device=cuda",
            "algo.algo_params.use_compile=true",
        ]
    )

    class _FakeEnv:
        obs_groups_spec = {"obs": 4, "critic": 6}
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))

        def close(self):
            pass

    class _FakeLearner:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    class _FakeRunner:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(flash_db_mod, "ensure_registries", lambda: None)
    monkeypatch.setattr(flash_db_mod, "create_env", lambda *args, **kwargs: _FakeEnv())
    monkeypatch.setattr(flash_db_mod, "FlashSACLearner", _FakeLearner)
    monkeypatch.setattr(flash_db_mod, "DoubleBufferOffPolicyRunner", _FakeRunner)

    runner = _offpolicy().build_runner("flashsac", cfg)

    assert runner.kwargs["learner"].kwargs["use_compile"] is True


def test_flashsac_double_buffer_amp_passed_to_learner(monkeypatch: pytest.MonkeyPatch):
    import gymnasium as gym

    import unilab.algos.torch.flash_sac.double_buffer as flash_db_mod

    cfg = _offpolicy_cfg(
        [
            "algo=flashsac",
            "training.device=cuda",
            "training.use_amp=true",
            "algo.algo_params.amp_dtype=bf16",
        ]
    )

    class _FakeEnv:
        obs_groups_spec = {"obs": 4, "critic": 6}
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))

        def close(self):
            pass

    class _FakeLearner:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    class _FakeRunner:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(flash_db_mod, "ensure_registries", lambda: None)
    monkeypatch.setattr(flash_db_mod, "create_env", lambda *args, **kwargs: _FakeEnv())
    monkeypatch.setattr(flash_db_mod, "FlashSACLearner", _FakeLearner)
    monkeypatch.setattr(flash_db_mod, "DoubleBufferOffPolicyRunner", _FakeRunner)

    runner = _offpolicy().build_runner("flashsac", cfg)

    assert runner.kwargs["learner"].kwargs["use_amp"] is True
    assert runner.kwargs["learner"].kwargs["amp_dtype"] == "bf16"


@pytest.mark.parametrize(
    "override",
    [
        "training.replay_pack_layout=fields",
        "training.replay_pack_executor=thread",
        "training.replay_h2d_submitter=python",
    ],
)
def test_old_b_path_internal_knobs_are_not_hydra_options(override):
    with pytest.raises(Exception, match=override.split("=", 1)[0]):
        _offpolicy_cfg(["algo=sac", override])


# ---------------------------------------------------------------------------
# Dispatch to DoubleBufferOffPolicyRunner
# ---------------------------------------------------------------------------


def test_sac_double_buffer_dispatches_to_correct_runner(monkeypatch: pytest.MonkeyPatch):
    import gymnasium as gym

    mod = _offpolicy()
    cfg = _offpolicy_cfg(
        [
            "algo=sac",
            "training.device=cuda",
            "algo.use_symmetry=false",
        ]
    )

    class _FakeEnv:
        obs_groups_spec = {"obs": 4, "critic": 6}
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))

        def build_symmetry_augmentation(self, device=None):
            return None

        def close(self):
            pass

    class _FakeLearner:
        class actor:
            @staticmethod
            def state_dict():
                return {"w": MagicMock(shape=(4,))}

        update_count = 0

        def __init__(self, *args, **kwargs):
            pass

    class _FakeRunner:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(mod, "ensure_registries", lambda: None)
    monkeypatch.setattr(mod, "create_env", lambda *args, **kwargs: _FakeEnv())

    import unilab.algos.torch.fast_sac.learner as learner_mod

    monkeypatch.setattr(learner_mod, "FastSACLearner", _FakeLearner)

    import unilab.algos.torch.offpolicy.double_buffer_runner as db_mod

    monkeypatch.setattr(db_mod, "DoubleBufferOffPolicyRunner", _FakeRunner)

    runner = mod.build_runner("sac", cfg)

    assert isinstance(runner, _FakeRunner)
    assert runner.kwargs["algo_type"] == "sac"
    assert runner.kwargs["replay_prefetch_mode"] == "one_tick"
    replay_kwargs = {key for key in runner.kwargs if key.startswith("replay_")}
    assert replay_kwargs == {
        "replay_buffer_n",
        "replay_prefetch_mode",
    }


def test_sac_double_buffer_one_tick_prefetch_mode_passed(monkeypatch: pytest.MonkeyPatch):
    import gymnasium as gym

    mod = _offpolicy()
    cfg = _offpolicy_cfg(
        [
            "algo=sac",
            "training.replay_prefetch_mode=one_tick",
            "training.device=cuda",
            "algo.use_symmetry=false",
        ]
    )

    class _FakeEnv:
        obs_groups_spec = {"obs": 4, "critic": 6}
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))

        def build_symmetry_augmentation(self, device=None):
            return None

        def close(self):
            pass

    class _FakeLearner:
        class actor:
            @staticmethod
            def state_dict():
                return {"w": MagicMock(shape=(4,))}

        update_count = 0

        def __init__(self, *args, **kwargs):
            pass

    class _FakeRunner:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(mod, "ensure_registries", lambda: None)
    monkeypatch.setattr(mod, "create_env", lambda *args, **kwargs: _FakeEnv())

    import unilab.algos.torch.fast_sac.learner as learner_mod

    monkeypatch.setattr(learner_mod, "FastSACLearner", _FakeLearner)

    import unilab.algos.torch.offpolicy.double_buffer_runner as db_mod

    monkeypatch.setattr(db_mod, "DoubleBufferOffPolicyRunner", _FakeRunner)

    runner = mod.build_runner("sac", cfg)

    assert isinstance(runner, _FakeRunner)
    assert runner.kwargs["replay_prefetch_mode"] == "one_tick"


def test_default_sac_dispatches_to_double_buffer_runner(monkeypatch: pytest.MonkeyPatch):
    import gymnasium as gym

    mod = _offpolicy()
    cfg = _offpolicy_cfg(
        [
            "algo=sac",
            "training.device=cuda",
            "algo.use_symmetry=false",
        ]
    )

    class _FakeEnv:
        obs_groups_spec = {"obs": 4, "critic": 6}
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))

        def build_symmetry_augmentation(self, device=None):
            return None

        def close(self):
            pass

    class _FakeLearner:
        class actor:
            @staticmethod
            def state_dict():
                return {"w": MagicMock(shape=(4,))}

        update_count = 0

        def __init__(self, *args, **kwargs):
            pass

    class _FakeRunner:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(mod, "ensure_registries", lambda: None)
    monkeypatch.setattr(mod, "create_env", lambda *args, **kwargs: _FakeEnv())

    import unilab.algos.torch.fast_sac.learner as learner_mod

    monkeypatch.setattr(learner_mod, "FastSACLearner", _FakeLearner)

    import unilab.algos.torch.offpolicy.double_buffer_runner as db_mod

    monkeypatch.setattr(db_mod, "DoubleBufferOffPolicyRunner", _FakeRunner)

    runner = mod.build_runner("sac", cfg)

    assert isinstance(runner, _FakeRunner)
    assert runner.kwargs["replay_prefetch_mode"] == "one_tick"


def test_flashsac_double_buffer_dispatches_to_correct_runner(monkeypatch: pytest.MonkeyPatch):
    import gymnasium as gym

    import unilab.algos.torch.flash_sac.double_buffer as flash_db_mod

    mod = _offpolicy()
    cfg = _offpolicy_cfg(
        [
            "algo=flashsac",
            "training.device=cuda",
            "training.trace_enabled=true",
            "training.trace_output_dir=/tmp/flashsac_trace",
        ]
    )

    class _FakeEnv:
        obs_groups_spec = {"obs": 4, "critic": 6}
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))

        def close(self):
            pass

    class _FakeLearner:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    class _FakeRunner:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(flash_db_mod, "ensure_registries", lambda: None)
    monkeypatch.setattr(flash_db_mod, "create_env", lambda *args, **kwargs: _FakeEnv())
    monkeypatch.setattr(flash_db_mod, "FlashSACLearner", _FakeLearner)
    monkeypatch.setattr(flash_db_mod, "DoubleBufferOffPolicyRunner", _FakeRunner)

    runner = mod.build_runner("flashsac", cfg)

    assert isinstance(runner, _FakeRunner)
    assert runner.kwargs["algo_type"] == "flashsac"
    assert runner.kwargs["replay_prefetch_mode"] == "one_tick"
    assert runner.kwargs["trace_enabled"] is True
    assert runner.kwargs["trace_output_dir"] == "/tmp/flashsac_trace"
    assert runner.kwargs["trace_cuda_events"] is True
    assert runner.kwargs["actor_kwargs"] == {
        "actor_num_blocks": 2,
        "actor_noise_zeta_mu": 2.0,
        "actor_noise_zeta_max": 16,
    }
    assert runner.kwargs["learner"].kwargs["critic_obs_dim"] == 6


def test_flashsac_double_buffer_actor_kwargs_passed(monkeypatch: pytest.MonkeyPatch):
    import gymnasium as gym

    import unilab.algos.torch.flash_sac.double_buffer as flash_db_mod

    mod = _offpolicy()
    cfg = _offpolicy_cfg(
        [
            "algo=flashsac",
            "training.device=cuda",
            "algo.algo_params.actor_num_blocks=3",
            "algo.algo_params.actor_noise_zeta_mu=4.0",
            "algo.algo_params.actor_noise_zeta_max=8",
        ]
    )

    class _FakeEnv:
        obs_groups_spec = {"obs": 4, "critic": 6}
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))

        def close(self):
            pass

    class _FakeLearner:
        def __init__(self, *args, **kwargs):
            pass

    class _FakeRunner:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(flash_db_mod, "ensure_registries", lambda: None)
    monkeypatch.setattr(flash_db_mod, "create_env", lambda *args, **kwargs: _FakeEnv())
    monkeypatch.setattr(flash_db_mod, "FlashSACLearner", _FakeLearner)
    monkeypatch.setattr(flash_db_mod, "DoubleBufferOffPolicyRunner", _FakeRunner)

    runner = mod.build_runner("flashsac", cfg)

    assert runner.kwargs["actor_kwargs"] == {
        "actor_num_blocks": 3,
        "actor_noise_zeta_mu": 4.0,
        "actor_noise_zeta_max": 8,
    }


def test_flashsac_double_buffer_verbose_metrics_passed(monkeypatch: pytest.MonkeyPatch):
    import gymnasium as gym

    import unilab.algos.torch.flash_sac.double_buffer as flash_db_mod

    mod = _offpolicy()
    cfg = _offpolicy_cfg(
        [
            "algo=flashsac",
            "training.verbose_metrics=true",
            "training.device=cuda",
        ]
    )

    class _FakeEnv:
        obs_groups_spec = {"obs": 4, "critic": 6}
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))

        def close(self):
            pass

    class _FakeLearner:
        def __init__(self, *args, **kwargs):
            pass

    class _FakeRunner:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(flash_db_mod, "ensure_registries", lambda: None)
    monkeypatch.setattr(flash_db_mod, "create_env", lambda *args, **kwargs: _FakeEnv())
    monkeypatch.setattr(flash_db_mod, "FlashSACLearner", _FakeLearner)
    monkeypatch.setattr(flash_db_mod, "DoubleBufferOffPolicyRunner", _FakeRunner)

    runner = mod.build_runner("flashsac", cfg)

    assert runner.kwargs["verbose_metrics"] is True


# ---------------------------------------------------------------------------
# Fake-loop tests: verify pipeline call ordering
# ---------------------------------------------------------------------------


class _CallLog:
    """Records method calls for verification."""

    def __init__(self):
        self.calls = []

    def record(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))

    def names(self):
        return [c[0] for c in self.calls]


class _FakePipeline:
    """Mock pipeline that records calls and returns dummy data."""

    def __init__(self, log: _CallLog, batch_ready_result: bool = False):
        self._log = log
        self._batch_ready_result = batch_ready_result

    def start_prepare(self, tick_id, sample_count):
        self._log.record("start_prepare", tick_id, sample_count)
        return True

    def batch_ready(self, tick_id, sample_count):
        self._log.record("batch_ready", tick_id, sample_count)
        return self._batch_ready_result

    def wait_until_ready(self, tick_id, sample_count):
        self._log.record("wait_until_ready", tick_id, sample_count)
        return True

    def sample_large_batch(self, tick_id, sample_count):
        self._log.record("sample_large_batch", tick_id, sample_count)
        import torch

        return {
            "obs": torch.zeros(sample_count, 4),
            "actions": torch.zeros(sample_count, 2),
            "rewards": torch.zeros(sample_count),
            "next_obs": torch.zeros(sample_count, 4),
            "dones": torch.zeros(sample_count),
            "truncated": torch.zeros(sample_count),
        }

    def after_tick(self):
        self._log.record("after_tick")

    def close(self):
        self._log.record("close")


def _run_fake_loop(max_iterations: int = 3, batch_ready: bool = False):
    """Simulate the runner training loop with a fake pipeline."""
    log = _CallLog()
    pipeline = _FakePipeline(log, batch_ready_result=batch_ready)
    prepared_tick: int | None = None
    sample_count = 16
    updates_per_step = 2

    critic_calls = 0
    weight_sync_calls = 0

    for iteration in range(1, max_iterations + 1):
        if prepared_tick != iteration:
            pipeline.start_prepare(iteration, sample_count)
            prepared_tick = iteration

        ready = pipeline.batch_ready(iteration, sample_count)
        if not ready:
            pipeline.wait_until_ready(iteration, sample_count)

        pipeline.sample_large_batch(iteration, sample_count)

        if iteration < max_iterations:
            pipeline.start_prepare(iteration + 1, sample_count)
            prepared_tick = iteration + 1

        for _ in range(updates_per_step):
            critic_calls += 1

        pipeline.after_tick()
        weight_sync_calls += 1

    pipeline.close()
    return log, critic_calls, weight_sync_calls


def test_one_tick_loop_prefetches_next_tick():
    log, critic_calls, ws_calls = _run_fake_loop(max_iterations=3)
    start_prepare_calls = [c for c in log.calls if c[0] == "start_prepare"]
    ticks_prepared = [c[1][0] for c in start_prepare_calls]
    assert ticks_prepared == [1, 2, 3]
    assert log.names().count("sample_large_batch") == 3
    assert log.names().count("after_tick") == 3
    assert critic_calls == 6
    assert ws_calls == 3


def test_one_tick_loop_waits_when_prefetch_not_ready():
    log, _, _ = _run_fake_loop(max_iterations=2, batch_ready=False)
    names = log.names()
    assert names.count("wait_until_ready") == 2


def test_one_tick_loop_skips_wait_when_prefetch_ready():
    log, _, _ = _run_fake_loop(max_iterations=2, batch_ready=True)
    names = log.names()
    assert "wait_until_ready" not in names


def test_one_tick_loop_does_not_prefetch_beyond_max():
    log, _, _ = _run_fake_loop(max_iterations=2)
    start_prepare_calls = [c for c in log.calls if c[0] == "start_prepare"]
    ticks_prepared = [c[1][0] for c in start_prepare_calls]
    assert ticks_prepared == [1, 2]


def test_one_tick_loop_start_prepare_after_sample():
    """In one_tick mode, start_prepare(N+1) must come after sample_large_batch(N)."""
    log, _, _ = _run_fake_loop(max_iterations=2)
    sample_idx_1 = None
    for i, c in enumerate(log.calls):
        if c[0] == "sample_large_batch" and c[1][0] == 1:
            sample_idx_1 = i
            break
    assert sample_idx_1 is not None
    next_start = None
    for i, c in enumerate(log.calls):
        if c[0] == "start_prepare" and c[1][0] == 2 and i > sample_idx_1:
            next_start = i
            break
    assert next_start is not None
    assert next_start == sample_idx_1 + 1


def test_collector_thread_explicit_compute_stream_disabled_trace_marker():
    import ast

    src = Path("src/unilab/algos/torch/offpolicy/double_buffer_runner.py").read_text()
    tree = ast.parse(src)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    constants = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)}
    assert "learner_compute_stream" not in names
    assert "explicit_compute_stream" in constants
    assert False in constants


# ---------------------------------------------------------------------------
# verbose_metrics wiring (round6)
# ---------------------------------------------------------------------------


def test_default_verbose_metrics_is_false():
    cfg = _offpolicy_cfg()
    assert cfg.training.verbose_metrics is False


def test_cli_override_verbose_metrics_accepted():
    cfg = _offpolicy_cfg(["training.verbose_metrics=true"])
    assert cfg.training.verbose_metrics is True


def test_sac_double_buffer_verbose_metrics_passed(monkeypatch: pytest.MonkeyPatch):
    import gymnasium as gym

    mod = _offpolicy()
    cfg = _offpolicy_cfg(
        [
            "algo=sac",
            "training.verbose_metrics=true",
            "training.device=cuda",
            "algo.use_symmetry=false",
        ]
    )

    class _FakeEnv:
        obs_groups_spec = {"obs": 4, "critic": 6}
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))

        def build_symmetry_augmentation(self, device=None):
            return None

        def close(self):
            pass

    class _FakeLearner:
        class actor:
            @staticmethod
            def state_dict():
                return {"w": MagicMock(shape=(4,))}

        update_count = 0

        def __init__(self, *args, **kwargs):
            pass

    class _FakeRunner:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(mod, "ensure_registries", lambda: None)
    monkeypatch.setattr(mod, "create_env", lambda *args, **kwargs: _FakeEnv())

    import unilab.algos.torch.fast_sac.learner as learner_mod

    monkeypatch.setattr(learner_mod, "FastSACLearner", _FakeLearner)

    import unilab.algos.torch.offpolicy.double_buffer_runner as db_mod

    monkeypatch.setattr(db_mod, "DoubleBufferOffPolicyRunner", _FakeRunner)

    runner = mod.build_runner("sac", cfg)
    assert isinstance(runner, _FakeRunner)
    assert runner.kwargs["verbose_metrics"] is True
