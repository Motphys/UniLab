"""Real-CUDA RSL-RL PPO smoke coverage for the managed mjwarp device profile."""

from __future__ import annotations

import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest
import torch

from tests.manager.test_g1_reference_differential import _cfg
from tests.training.device_runtime_harness import require_cuda
from unilab.algos.torch.rsl_rl_runtime import (
    resolve_rsl_rl_ppo_runtime,
    validate_rsl_rl_ppo_runtime_owner,
)
from unilab.envs.locomotion.g1.joystick import G1WalkEnv
from unilab.structured_configs import PPOConfig
from unilab.training import rsl_rl_device
from unilab.training.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg
from unilab.training.rsl_rl_device import (
    DeviceOnPolicyRunner,
    DeviceRslRlContractError,
    DeviceRslRlHostMemoryDiagnostics,
    DeviceRslRlVecEnvWrapper,
    resolve_mjwarp_device_ppo_runtime,
)

pytestmark = pytest.mark.slow


@contextmanager
def _device_env(*, num_envs: int) -> Iterator[G1WalkEnv]:
    require_cuda()
    cfg = _cfg(
        max_episode_seconds=0.1,
        observation_noise_level=0.0,
        observation_noise_seed=None,
    )
    env = G1WalkEnv(cfg, num_envs=num_envs, backend_type="mjwarp")
    try:
        yield env
    finally:
        env.close()


def _train_cfg() -> dict:
    cfg = PPOConfig()
    train_cfg = cfg.to_dict()
    train_cfg["num_steps_per_env"] = 2
    train_cfg["save_interval"] = 1
    train_cfg["check_for_nan"] = False
    train_cfg["policy"] = {
        "class_name": "ActorCritic",
        "actor_hidden_dims": [64, 64],
        "critic_hidden_dims": [64, 64],
        "activation": "elu",
        "init_noise_std": 1.0,
    }
    train_cfg["algorithm"]["num_learning_epochs"] = 1
    train_cfg["algorithm"]["num_mini_batches"] = 1
    train_cfg["algorithm"]["enable_compile"] = False
    train_cfg["runner"] = {"logger": "tensorboard"}
    train_cfg["logger"] = "tensorboard"
    return normalize_ppo_train_cfg(train_cfg)


def test_device_runtime_resolver_and_finite_guard_are_strict() -> None:
    runtime = resolve_mjwarp_device_ppo_runtime({"runtime_impl": "mjwarp_device_v1", "seed": 3})
    assert runtime.wrapper_cls is DeviceRslRlVecEnvWrapper
    assert runtime.runner_cls is DeviceOnPolicyRunner
    assert runtime.wrapper_kwargs == {"reset_seed": 3}
    assert runtime.required_backend == "mjwarp"
    assert runtime.required_execution_profile == "device_resident"

    resolved = resolve_rsl_rl_ppo_runtime(
        {
            "runtime_impl": "mjwarp_device_v1",
            "runtime_resolver": "unilab.training.rsl_rl_device:resolve_mjwarp_device_ppo_runtime",
            "seed": 3,
        },
        default_wrapper_cls=RslRlVecEnvWrapper,
    )
    validate_rsl_rl_ppo_runtime_owner(
        resolved,
        sim_backend="mjwarp",
        execution_profile="device_resident",
    )
    with pytest.raises(ValueError, match="sim_backend"):
        validate_rsl_rl_ppo_runtime_owner(
            resolved,
            sim_backend="mujoco",
            execution_profile="device_resident",
        )
    with pytest.raises(ValueError, match="execution_profile"):
        validate_rsl_rl_ppo_runtime_owner(
            resolved,
            sim_backend="mjwarp",
            execution_profile="host_numpy",
        )

    for invalid in (
        {"runtime_impl": "host_numpy_v1", "seed": 3},
        {"runtime_impl": "mjwarp_device_v1", "seed": -1},
        {"runtime_impl": "mjwarp_device_v1", "seed": True},
    ):
        with pytest.raises(DeviceRslRlContractError):
            resolve_mjwarp_device_ppo_runtime(invalid)

    with pytest.raises(DeviceRslRlContractError, match="check_for_nan=false"):
        with _device_env(num_envs=8) as env:
            wrapper = DeviceRslRlVecEnvWrapper(env, device="cuda:0", reset_seed=3)
            DeviceOnPolicyRunner(wrapper, _train_cfg() | {"check_for_nan": True}, device="cuda:0")


def test_device_runner_keeps_upstream_lifecycle_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """The specialized runner retains upstream rollout/return/update/log ordering."""

    with _device_env(num_envs=32) as env:
        wrapper = DeviceRslRlVecEnvWrapper(env, device="cuda:0", reset_seed=9)
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = DeviceOnPolicyRunner(
                wrapper,
                _train_cfg(),
                log_dir=tmpdir,
                device="cuda:0",
            )
            trace: list[str] = []
            memory_diagnostics = DeviceRslRlHostMemoryDiagnostics(
                gc_collected_objects=7,
                allocator_trim_attempted=True,
                allocator_trimmed=True,
            )

            def release_cold_path_memory() -> DeviceRslRlHostMemoryDiagnostics:
                trace.append("release_host_memory")
                return memory_diagnostics

            monkeypatch.setattr(
                rsl_rl_device,
                "_release_cold_path_host_memory",
                release_cold_path_memory,
            )

            def traced_method(owner, name: str, label: str) -> None:
                original = getattr(owner, name)

                def wrapped(*args, **kwargs):
                    trace.append(label)
                    return original(*args, **kwargs)

                monkeypatch.setattr(owner, name, wrapped)

            traced_method(runner.alg, "act", "act")
            traced_method(wrapper, "step", "step")
            traced_method(runner.alg, "process_env_step", "process")
            traced_method(runner.logger, "process_env_step", "logger_step")
            traced_method(wrapper, "finish_rollout", "finite_boundary")
            traced_method(runner.alg, "compute_returns", "returns")
            traced_method(runner.alg, "update", "update")
            traced_method(runner.logger, "log", "log")
            try:
                runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
            finally:
                writer = getattr(runner.logger, "writer", None)
                if writer is not None and hasattr(writer, "close"):
                    writer.close()

        assert trace == [
            "release_host_memory",
            "act",
            "step",
            "process",
            "logger_step",
            "act",
            "step",
            "process",
            "logger_step",
            "finite_boundary",
            "returns",
            "update",
            "log",
        ]
        assert runner.host_memory_diagnostics is memory_diagnostics
        runner._release_cold_path_host_memory_once()
        assert trace.count("release_host_memory") == 1


def test_device_runner_storage_preserves_the_action_observation_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Borrowed runtime observations cannot be overwritten before PPO stores them."""

    with _device_env(num_envs=32) as env:
        wrapper = DeviceRslRlVecEnvWrapper(env, device="cuda:0", reset_seed=10)
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = DeviceOnPolicyRunner(
                wrapper,
                _train_cfg(),
                log_dir=tmpdir,
                device="cuda:0",
            )

            def verify_storage() -> dict[str, float]:
                storage = runner.alg.storage
                observations = storage.observations.flatten(0, 1)
                actions = storage.actions.flatten(0, 1)
                old_log_prob = storage.actions_log_prob.flatten(0, 1).squeeze(-1)
                old_values = storage.values.flatten(0, 1)
                with torch.inference_mode():
                    runner.alg.actor(observations, stochastic_output=True)
                    actual_log_prob = runner.alg.actor.get_output_log_prob(actions)
                    actual_values = runner.alg.critic(observations)

                torch.testing.assert_close(actual_log_prob, old_log_prob, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(actual_values, old_values, atol=1e-6, rtol=1e-5)
                storage.clear()
                return {"value": 0.0, "surrogate": 0.0, "entropy": 0.0}

            monkeypatch.setattr(runner.alg, "update", verify_storage)
            try:
                runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
            finally:
                writer = getattr(runner.logger, "writer", None)
                if writer is not None and hasattr(writer, "close"):
                    writer.close()


def test_real_mjwarp_device_ppo_one_iteration() -> None:
    """Run the public runner lifecycle through CUDA storage, learner, logger, and save."""

    with _device_env(num_envs=32) as env:
        wrapper = DeviceRslRlVecEnvWrapper(env, device="cuda:0", reset_seed=5)
        train_cfg = _train_cfg()
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = DeviceOnPolicyRunner(
                wrapper,
                train_cfg,
                log_dir=tmpdir,
                device="cuda:0",
            )
            try:
                runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
                storage = runner.alg.storage
                assert storage.observations.device.type == "cuda"
                assert storage.actions.device.type == "cuda"
                assert storage.rewards.device.type == "cuda"
                assert storage.dones.device.type == "cuda"
                assert runner.logger.device_diagnostics.metric_materializations == 1
                assert runner.logger.device_diagnostics.metric_device_to_host_bytes > 0
                memory_diagnostics = runner.host_memory_diagnostics
                assert memory_diagnostics is not None
                assert memory_diagnostics.gc_collected_objects >= 0
                if sys.platform.startswith("linux"):
                    assert memory_diagnostics.allocator_trim_attempted
                assert (Path(tmpdir) / "model_0.pt").is_file()
                assert wrapper.runtime.traffic_diagnostics.host_to_device_transfers == 0
                assert wrapper.runtime.traffic_diagnostics.device_to_host_transfers == 0
                assert wrapper.runtime.traffic_diagnostics.global_synchronizations == 0
            finally:
                writer = getattr(runner.logger, "writer", None)
                if writer is not None and hasattr(writer, "close"):
                    writer.close()
