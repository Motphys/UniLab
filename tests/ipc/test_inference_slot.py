"""Tests for the learner-owned inference single-slot contract."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from unilab.ipc.inference_slot import SharedInferenceSlot


def test_inference_slot_roundtrip_preserves_tick_and_policy_version() -> None:
    slot = SharedInferenceSlot(num_envs=2, obs_dim=3, action_dim=2)
    observations = np.arange(6, dtype=np.float32).reshape(2, 3)
    dones = np.array([0.0, 1.0], dtype=np.float32)

    slot.publish_observation(tick_id=7, observations=observations, dones=dones)
    obs_destination = torch.empty(2, 3)
    dones_destination = torch.empty(2)
    slot.copy_observation_to(
        tick_id=7,
        observations=obs_destination,
        dones=dones_destination,
    )
    slot.publish_action(
        tick_id=7,
        policy_version=11,
        actions=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )

    actions, policy_version = slot.consume_action(tick_id=7)

    np.testing.assert_array_equal(obs_destination.numpy(), observations)
    np.testing.assert_array_equal(dones_destination.numpy(), dones)
    np.testing.assert_array_equal(actions, [[1.0, 2.0], [3.0, 4.0]])
    assert policy_version == 11


def test_inference_slot_rejects_early_reuse_and_tick_mismatch() -> None:
    slot = SharedInferenceSlot(num_envs=1, obs_dim=2, action_dim=1)
    observations = np.zeros((1, 2), dtype=np.float32)
    dones = np.zeros(1, dtype=np.float32)
    slot.publish_observation(tick_id=3, observations=observations, dones=dones)

    with pytest.raises(RuntimeError, match="cannot be reused"):
        slot.publish_observation(tick_id=4, observations=observations, dones=dones)
    with pytest.raises(RuntimeError, match="tick mismatch"):
        slot.copy_observation_to(
            tick_id=4,
            observations=torch.empty(1, 2),
            dones=torch.empty(1),
        )

    slot.publish_action(tick_id=3, policy_version=5, actions=torch.ones(1, 1))
    with pytest.raises(RuntimeError, match="tick mismatch"):
        slot.consume_action(tick_id=4)
