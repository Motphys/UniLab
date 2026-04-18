from __future__ import annotations

import numpy as np

from unilab.utils.final_observation import (
    patch_transition_next_obs,
    resolve_terminal_observation_contract,
    resolve_transition_bootstrap_contract,
)


def test_patch_transition_next_obs_uses_final_observation_without_mutating_actor_obs():
    next_obs = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    next_critic = np.array([[10.0], [20.0]], dtype=np.float32)
    final_observation = {
        "obs": np.array([[7.0, 8.0], [9.0, 9.0]], dtype=np.float32),
        "critic": np.array([[70.0], [90.0]], dtype=np.float32),
    }

    (
        transition_next_obs,
        transition_next_critic,
        terminal_mask,
    ) = patch_transition_next_obs(
        next_obs,
        final_observation=final_observation,
        done=np.array([True, False]),
        next_critic=next_critic,
    )

    np.testing.assert_array_equal(terminal_mask, np.array([True, False]))
    np.testing.assert_array_equal(transition_next_obs, np.array([[7.0, 8.0], [2.0, 2.0]]))
    np.testing.assert_array_equal(transition_next_critic, np.array([[70.0], [20.0]]))
    np.testing.assert_array_equal(next_obs, np.array([[1.0, 1.0], [2.0, 2.0]]))
    np.testing.assert_array_equal(next_critic, np.array([[10.0], [20.0]]))


def test_resolve_transition_bootstrap_contract_separates_actor_and_storage_paths():
    contract = resolve_transition_bootstrap_contract(
        next_obs=np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
        next_critic=np.array([[10.0], [20.0]], dtype=np.float32),
        final_observation={"obs": np.array([[5.0, 5.0], [8.0, 9.0]], dtype=np.float32)},
        done=np.array([False, True]),
        truncated=np.array([False, True]),
    )

    np.testing.assert_array_equal(contract.actor_next_obs, np.array([[1.0, 1.0], [2.0, 2.0]]))
    np.testing.assert_array_equal(
        contract.transition_next_obs, np.array([[1.0, 1.0], [8.0, 9.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        contract.actor_next_critic, np.array([[10.0], [20.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        contract.transition_next_critic, np.array([[10.0], [20.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(contract.terminal_mask, np.array([False, True]))
    np.testing.assert_array_equal(contract.timeout_terminal_mask, np.array([False, True]))


def test_resolve_terminal_observation_contract_returns_terminal_rows_without_copying_next_obs():
    terminal_contract = resolve_terminal_observation_contract(
        next_obs_batch_size=2,
        final_observation={
            "obs": np.array([[5.0, 5.0], [8.0, 9.0]], dtype=np.float32),
            "critic": np.array([[1.0], [7.0]], dtype=np.float32),
        },
        done=np.array([False, True]),
        truncated=np.array([False, True]),
    )

    np.testing.assert_array_equal(terminal_contract.terminal_mask, np.array([False, True]))
    np.testing.assert_array_equal(terminal_contract.timeout_terminal_mask, np.array([False, True]))
    np.testing.assert_array_equal(
        terminal_contract.terminal_obs,
        np.array([[5.0, 5.0], [8.0, 9.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        terminal_contract.terminal_critic,
        np.array([[1.0], [7.0]], dtype=np.float32),
    )
