"""Real-CUDA contract tests for the formal RSL-RL device VecEnv adapter."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
from unittest.mock import patch

import pytest
import torch
from tests.manager.test_g1_reference_differential import _cfg
from tests.training.device_runtime_harness import forbid_host_roundtrip, require_cuda

from unilab.envs.locomotion.g1.joystick import G1WalkEnv
from unilab.envs.locomotion.g1.managed_device import G1ManagedDeviceError
from unilab.training.rsl_rl_device import (
    DeviceRslRlContractError,
    DeviceRslRlVecEnvWrapper,
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


def test_device_adapter_preserves_terminal_timeout_and_storage_contract() -> None:
    """Timeout bootstrap observes terminal CUDA buffers, never reset observations."""

    with _device_env(num_envs=32) as env:
        wrapper = DeviceRslRlVecEnvWrapper(env, device="cuda:0", reset_seed=7)
        episode_lengths = torch.zeros_like(wrapper.episode_length_buf)
        episode_lengths[::2].fill_(wrapper.max_episode_length - 1)
        wrapper.episode_length_buf = episode_lengths

        observations, rewards, dones, extras = wrapper.step(
            torch.zeros((wrapper.num_envs, wrapper.num_actions), device=wrapper.device)
        )
        transition = wrapper.last_transition
        transition.completion.event.synchronize()

        assert observations.device == wrapper.device
        assert observations["actor"].data_ptr() == transition.observation("obs").torch().data_ptr()
        assert (
            observations["critic"].data_ptr() == transition.observation("critic").torch().data_ptr()
        )
        assert rewards.device == wrapper.device
        assert dones.device == wrapper.device
        assert extras["time_outs"].device == wrapper.device
        assert extras["time_out_bootstrap_obs"]["actor"].data_ptr() == (
            transition.final_observation("obs").torch().data_ptr()
        )

        expected_timeout = torch.zeros(wrapper.num_envs, dtype=torch.bool, device=wrapper.device)
        expected_timeout[::2] = True
        assert torch.equal(extras["time_outs"], expected_timeout)
        assert torch.equal(dones, expected_timeout)
        assert torch.equal(transition.final_observation_mask.torch(), expected_timeout)
        assert torch.equal(
            extras["time_out_bootstrap_obs"]["actor"][expected_timeout],
            transition.terminal_observation("obs").torch()[expected_timeout],
        )
        assert wrapper.runtime.traffic_diagnostics.host_to_device_transfers == 0
        assert wrapper.runtime.traffic_diagnostics.device_to_host_transfers == 0
        assert wrapper.runtime.traffic_diagnostics.global_synchronizations == 0


def test_device_adapter_step_has_no_host_roundtrip_or_done_index_branch() -> None:
    """The hot adapter path consumes public CUDA views without host extraction."""

    with _device_env(num_envs=32) as env:
        wrapper = DeviceRslRlVecEnvWrapper(env, device="cuda:0", reset_seed=11)
        actions = torch.zeros((wrapper.num_envs, wrapper.num_actions), device=wrapper.device)
        with forbid_host_roundtrip(env._backend):
            observations, rewards, dones, extras = wrapper.step(actions)
            assert observations.device == wrapper.device
            assert rewards.device == wrapper.device
            assert dones.device == wrapper.device
            assert extras["time_outs"].device == wrapper.device

        wrapper.last_transition.completion.event.synchronize()
        traffic = wrapper.traffic_diagnostics
        assert traffic.action_publications == 1
        assert traffic.action_device_to_device_bytes == actions.numel() * actions.element_size()
        assert traffic.finite_metric_materializations == 0


@pytest.mark.parametrize("invalid", (float("nan"), float("inf"), -float("inf")))
def test_device_adapter_defers_nonfinite_action_failure_without_poisoning_physics(
    invalid: float,
) -> None:
    """Invalid policy actions are made physics-safe and fail at the rollout boundary."""

    with _device_env(num_envs=32) as env:
        wrapper = DeviceRslRlVecEnvWrapper(env, device="cuda:0", reset_seed=12)
        actions = torch.zeros((wrapper.num_envs, wrapper.num_actions), device=wrapper.device)
        actions[0, 0] = invalid
        with forbid_host_roundtrip(env._backend):
            observations, rewards, _, _ = wrapper.step(actions)
        wrapper.last_transition.completion.event.synchronize()
        assert torch.isfinite(observations["actor"]).all()
        assert torch.isfinite(rewards).all()
        assert torch.equal(wrapper._action_buffer[0], torch.zeros_like(wrapper._action_buffer[0]))
        with pytest.raises(DeviceRslRlContractError, match="action"):
            wrapper.finish_rollout()


def test_device_adapter_rejects_terminal_only_nonfinite_observation_before_update() -> None:
    """Timeout bootstrap buffers participate in the deferred CUDA finite guard."""

    with _device_env(num_envs=8) as env:
        wrapper = DeviceRslRlVecEnvWrapper(env, device="cuda:0", reset_seed=14)
        transition = wrapper.last_transition
        transition.completion.event.synchronize()
        transition.final_observation("obs").torch()[0, 0] = float("nan")

        wrapper._finite_accumulator.fill_(True)
        wrapper._consume_transition(transition)
        with pytest.raises(DeviceRslRlContractError, match="obs"):
            wrapper.finish_rollout()


def test_device_adapter_rejects_malformed_actions_before_runtime_step() -> None:
    """The runner boundary rejects bad action ABI before touching physics."""

    with _device_env(num_envs=8) as env:
        wrapper = DeviceRslRlVecEnvWrapper(env, device="cuda:0", reset_seed=13)
        malformed = (
            torch.zeros(
                (wrapper.num_envs, wrapper.num_actions), dtype=torch.float64, device="cuda:0"
            ),
            torch.zeros((wrapper.num_envs, wrapper.num_actions + 1), device="cuda:0"),
            torch.zeros((wrapper.num_envs, wrapper.num_actions * 2), device="cuda:0")[:, ::2],
            torch.zeros((wrapper.num_envs, wrapper.num_actions), dtype=torch.float32),
        )
        with patch.object(
            wrapper.runtime,
            "step",
            side_effect=AssertionError("malformed action reached runtime physics"),
        ):
            for actions in malformed:
                with pytest.raises(DeviceRslRlContractError, match="device policy actions"):
                    wrapper.step(actions)


def test_device_adapter_episode_schedule_handoffs_producer_stream() -> None:
    """RSL-RL's default-stream episode schedule is ordered before task CUDA work."""

    with _device_env(num_envs=32) as env:
        wrapper = DeviceRslRlVecEnvWrapper(env, device="cuda:0", reset_seed=17)
        producer = torch.cuda.Stream(device=wrapper.device)
        schedule = torch.empty_like(wrapper.episode_length_buf)
        with torch.cuda.stream(producer):
            # Ensure that the task stream would observe an uninitialized value
            # without the explicit setter event handoff.
            torch.cuda._sleep(20_000_000)
            schedule.fill_(wrapper.max_episode_length - 1)
            wrapper.episode_length_buf = schedule

        _, _, dones, extras = wrapper.step(
            torch.zeros((wrapper.num_envs, wrapper.num_actions), device=wrapper.device)
        )
        wrapper.last_transition.completion.event.synchronize()
        assert torch.equal(dones, torch.ones_like(dones))
        assert torch.equal(extras["time_outs"], torch.ones_like(dones))


@pytest.mark.parametrize(
    ("backend_type", "mutate", "match"),
    (
        (
            "mujoco",
            lambda cfg: None,
            "independent mjwarp backend",
        ),
        (
            "mjwarp",
            lambda cfg: setattr(cfg.noise_config, "level", 1.0),
            "observation noise is not implemented",
        ),
        (
            "mjwarp",
            lambda cfg: setattr(cfg.domain_rand, "randomize_kp", True),
            "typed DR/Event",
        ),
    ),
)
def test_device_adapter_rejects_unsupported_owner_profiles_before_binding(
    backend_type: str,
    mutate,
    match: str,
) -> None:
    """Backend/profile/noise/DR mismatches fail before selector binding or physics."""

    require_cuda()
    cfg = _cfg(
        max_episode_seconds=0.1,
        observation_noise_level=0.0,
        observation_noise_seed=None,
    )
    mutate(cfg)
    env = G1WalkEnv(cfg, num_envs=2, backend_type=backend_type)
    try:
        with patch.object(
            env._backend,
            "get_actuator_names",
            side_effect=AssertionError("unsupported owner reached cold binding"),
        ):
            with pytest.raises(G1ManagedDeviceError, match=match):
                DeviceRslRlVecEnvWrapper(env, device="cuda:0", reset_seed=19)
    finally:
        env.close()


def test_device_adapter_rejects_foreign_cuda_device_before_runtime_factory() -> None:
    """The active Warp/Torch CUDA device is an explicit owner contract."""

    with _device_env(num_envs=2) as env:
        with patch.object(
            env,
            "create_device_managed_runtime",
            side_effect=AssertionError("foreign device reached runtime factory"),
        ):
            foreign_index = torch.cuda.current_device() + 1
            with pytest.raises(DeviceRslRlContractError, match="active CUDA device"):
                DeviceRslRlVecEnvWrapper(
                    env,
                    device=f"cuda:{foreign_index}",
                    reset_seed=23,
                )
