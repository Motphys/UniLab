from __future__ import annotations

import numpy as np
import pytest
import torch

from unilab.algos.torch.appo.staging import RolloutStagingPool

_NUM_ENVS = 2
_NUM_STEPS = 3
_OBS_DIM = 4
_ACTION_DIM = 2
_CRITIC_DIM = 5

_SLOT_SHAPES = {
    "obs": (_NUM_ENVS, _NUM_STEPS, _OBS_DIM),
    "critic": (_NUM_ENVS, _NUM_STEPS, _CRITIC_DIM),
    "actions": (_NUM_ENVS, _NUM_STEPS, _ACTION_DIM),
    "log_probs": (_NUM_ENVS, _NUM_STEPS),
    "rewards": (_NUM_ENVS, _NUM_STEPS),
    "dones": (_NUM_ENVS, _NUM_STEPS),
    "truncated": (_NUM_ENVS, _NUM_STEPS),
    "last_obs": (_NUM_ENVS, _OBS_DIM),
    "last_critic": (_NUM_ENVS, _CRITIC_DIM),
}


def _raw_rollout(value: float) -> dict[str, np.ndarray]:
    return {field: np.full(shape, value, dtype=np.float32) for field, shape in _SLOT_SHAPES.items()}


def test_staging_pool_exposes_learner_ready_combined_batch() -> None:
    pool = RolloutStagingPool(
        capacity=2,
        num_envs=_NUM_ENVS,
        slot_shapes=_SLOT_SHAPES,
        device="cpu",
    )
    first = _raw_rollout(1.0)
    second = _raw_rollout(2.0)

    pool.stage_numpy_views(first)
    pool.stage_numpy_views(second)

    batch = pool.batch()
    expected_obs = torch.cat(
        [
            torch.from_numpy(first["obs"]).transpose(0, 1),
            torch.from_numpy(second["obs"]).transpose(0, 1),
        ],
        dim=1,
    )
    expected_last_obs = torch.cat(
        [torch.from_numpy(first["last_obs"]), torch.from_numpy(second["last_obs"])],
        dim=0,
    )

    assert batch["observations"].shape == (_NUM_STEPS, 2 * _NUM_ENVS, _OBS_DIM)
    assert batch["critic"].shape == (_NUM_STEPS, 2 * _NUM_ENVS, _CRITIC_DIM)
    assert batch["actions_log_prob"].shape == (_NUM_STEPS, 2 * _NUM_ENVS)
    assert batch["last_obs"].shape == (2 * _NUM_ENVS, _OBS_DIM)
    assert torch.equal(batch["observations"], expected_obs)
    assert torch.equal(batch["last_obs"], expected_last_obs)


def test_staging_pool_reuses_slots_and_drops_overwritten_rollouts() -> None:
    pool = RolloutStagingPool(
        capacity=2,
        num_envs=_NUM_ENVS,
        slot_shapes=_SLOT_SHAPES,
        device="cpu",
    )

    pool.stage_numpy_views(_raw_rollout(1.0))
    pool.stage_numpy_views(_raw_rollout(2.0))
    pool.stage_numpy_views(_raw_rollout(3.0))

    batch = pool.batch()
    assert pool.active_count == 2
    assert pool.slot_versions == (2, 1)
    assert torch.equal(torch.unique(batch["observations"]), torch.tensor([2.0, 3.0]))
    assert not torch.any(batch["observations"] == 1.0)


def test_staging_pool_batch_dict_does_not_retain_learner_mutations() -> None:
    pool = RolloutStagingPool(
        capacity=2,
        num_envs=_NUM_ENVS,
        slot_shapes=_SLOT_SHAPES,
        device="cpu",
    )
    pool.stage_numpy_views(_raw_rollout(1.0))

    batch = pool.batch()
    batch["values"] = torch.zeros(_NUM_STEPS, _NUM_ENVS)

    assert "values" not in pool.batch()


def test_staging_pool_rejects_empty_capacity() -> None:
    with pytest.raises(ValueError, match="capacity"):
        RolloutStagingPool(
            capacity=0,
            num_envs=_NUM_ENVS,
            slot_shapes=_SLOT_SHAPES,
            device="cpu",
        )
