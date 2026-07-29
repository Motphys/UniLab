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
