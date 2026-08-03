"""CUDA logging boundaries for the formal RSL-RL device runner."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from unilab.training.rsl_rl_device import DeviceRolloutLogger, DeviceRslRlContractError

pytestmark = pytest.mark.slow


def test_device_rollout_logging_has_no_per_step_host_materialization() -> None:
    """Episode accounting is CUDA-only until one explicit iteration boundary."""

    if not torch.cuda.is_available():
        pytest.fail("device logging contract requires CUDA")
    logger = DeviceRolloutLogger(
        log_dir=None,
        cfg={"algorithm": {"rnd_cfg": None}, "logger": "tensorboard"},
        env_cfg={},
        num_envs=8,
        is_distributed=False,
        gpu_world_size=1,
        gpu_global_rank=0,
        device="cuda:0",
        rollout_steps=2,
    )
    # The upstream logger only accounts episode metrics when a writer is active.
    logger.writer = object()
    rewards = torch.ones((8,), dtype=torch.float32, device="cuda:0")
    first_done = torch.tensor(
        [True, False, False, True, False, False, False, False], device="cuda:0"
    )
    second_done = torch.tensor(
        [False, True, False, False, False, True, False, False], device="cuda:0"
    )

    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("per-step cpu")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("per-step numpy")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("per-step tolist")),
    ):
        logger.process_env_step(rewards, first_done, {})
        logger.process_env_step(rewards, second_done, {})

    assert logger.device_diagnostics.metric_materializations == 0
    logger._materialize_episode_metrics()
    assert logger.device_diagnostics.rollout_steps == 2
    assert logger.device_diagnostics.metric_materializations == 1
    assert logger.device_diagnostics.metric_device_to_host_bytes == 3 * 2 * 8 * 4
    assert len(logger.rewbuffer) == 4
    assert len(logger.lenbuffer) == 4


def test_device_rollout_logging_reduces_task_metrics_at_the_single_boundary() -> None:
    """Per-term rewards stay on CUDA until the existing rollout boundary."""

    if not torch.cuda.is_available():
        pytest.fail("device logging contract requires CUDA")
    metric_keys = ("reward/tracking", "reward/action_rate")
    logger = DeviceRolloutLogger(
        log_dir=None,
        cfg={"algorithm": {"rnd_cfg": None}, "logger": "tensorboard"},
        env_cfg={},
        num_envs=8,
        is_distributed=False,
        gpu_world_size=1,
        gpu_global_rank=0,
        device="cuda:0",
        rollout_steps=2,
        metric_keys=metric_keys,
    )
    logger.writer = object()
    rewards = torch.ones((8,), dtype=torch.float32, device="cuda:0")
    dones = torch.zeros((8,), dtype=torch.bool, device="cuda:0")
    first_metrics = {
        metric_keys[0]: torch.full((8,), 2.0, device="cuda:0"),
        metric_keys[1]: torch.arange(8, dtype=torch.float32, device="cuda:0"),
    }
    second_metrics = {
        metric_keys[0]: torch.full((8,), 4.0, device="cuda:0"),
        metric_keys[1]: torch.arange(8, dtype=torch.float32, device="cuda:0") + 2.0,
    }

    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("per-step cpu")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("per-step numpy")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("per-step tolist")),
    ):
        logger.process_env_step(rewards, dones, {"metrics": first_metrics})
        logger.process_env_step(rewards, dones, {"metrics": second_metrics})

    logger._materialize_episode_metrics()
    assert logger.device_diagnostics.metric_materializations == 1
    assert logger.device_diagnostics.metric_device_to_host_bytes == (3 * 2 * 8 + 2 * 2) * 4
    assert len(logger.ep_extras) == 1
    torch.testing.assert_close(logger.ep_extras[0][metric_keys[0]], torch.tensor((2.0, 4.0)))
    torch.testing.assert_close(logger.ep_extras[0][metric_keys[1]], torch.tensor((3.5, 5.5)))


def test_device_rollout_logger_rejects_malformed_step_tensors_before_staging() -> None:
    """Logger ABI faults before it mutates CUDA episode accounting state."""

    if not torch.cuda.is_available():
        pytest.fail("device logging contract requires CUDA")
    logger = DeviceRolloutLogger(
        log_dir=None,
        cfg={"algorithm": {"rnd_cfg": None}, "logger": "tensorboard"},
        env_cfg={},
        num_envs=8,
        is_distributed=False,
        gpu_world_size=1,
        gpu_global_rank=0,
        device="cuda:0",
        rollout_steps=2,
    )
    logger.writer = object()
    valid_rewards = torch.ones((8,), dtype=torch.float32, device="cuda:0")
    valid_dones = torch.zeros((8,), dtype=torch.bool, device="cuda:0")
    malformed = (
        (torch.ones((8,), dtype=torch.float64, device="cuda:0"), valid_dones, "rewards"),
        (torch.ones((8, 1), dtype=torch.float32, device="cuda:0"), valid_dones, "rewards"),
        (torch.ones((8,), dtype=torch.float32), valid_dones, "rewards"),
        (valid_rewards, torch.zeros((8,), dtype=torch.float32, device="cuda:0"), "dones"),
        (valid_rewards, torch.zeros((8, 1), dtype=torch.bool, device="cuda:0"), "dones"),
    )
    for rewards, dones, field in malformed:
        with pytest.raises(DeviceRslRlContractError, match=field):
            logger.process_env_step(rewards, dones, {})

    assert logger._rollout_cursor == 0
    assert torch.equal(logger.cur_reward_sum, torch.zeros_like(logger.cur_reward_sum))
    assert torch.equal(logger.cur_episode_length, torch.zeros_like(logger.cur_episode_length))


def test_device_rollout_logger_rejects_metric_layout_or_tensor_drift() -> None:
    if not torch.cuda.is_available():
        pytest.fail("device logging contract requires CUDA")
    key = "reward/tracking"
    logger = DeviceRolloutLogger(
        log_dir=None,
        cfg={"algorithm": {"rnd_cfg": None}, "logger": "tensorboard"},
        env_cfg={},
        num_envs=8,
        is_distributed=False,
        gpu_world_size=1,
        gpu_global_rank=0,
        device="cuda:0",
        rollout_steps=2,
        metric_keys=(key,),
    )
    logger.writer = object()
    rewards = torch.ones((8,), dtype=torch.float32, device="cuda:0")
    dones = torch.zeros((8,), dtype=torch.bool, device="cuda:0")

    with pytest.raises(DeviceRslRlContractError, match="metric layout"):
        logger.process_env_step(rewards, dones, {})
    with pytest.raises(DeviceRslRlContractError, match="metric 'reward/tracking'"):
        logger.process_env_step(
            rewards,
            dones,
            {"metrics": {key: torch.ones((8,), dtype=torch.float64, device="cuda:0")}},
        )

    assert logger._rollout_cursor == 0
