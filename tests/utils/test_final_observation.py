from __future__ import annotations

import numpy as np

from unilab.utils.final_observation import (
    patch_transition_next_obs,
    resolve_transition_bootstrap_contract,
)


def test_patch_transition_next_obs_uses_final_observation_without_mutating_actor_obs():
    next_obs = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    next_privileged = np.array([[10.0], [20.0]], dtype=np.float32)
    info = {
        "_final_observation": np.array([True, False]),
        "final_observation": {
            "obs": np.array([[7.0, 8.0], [9.0, 9.0]], dtype=np.float32),
            "privileged": np.array([[70.0], [90.0]], dtype=np.float32),
        },
    }

    storage_next_obs, storage_next_privileged, final_mask = patch_transition_next_obs(
        next_obs, next_privileged, info
    )

    np.testing.assert_array_equal(final_mask, np.array([True, False]))
    np.testing.assert_array_equal(storage_next_obs, np.array([[7.0, 8.0], [2.0, 2.0]]))
    np.testing.assert_array_equal(storage_next_privileged, np.array([[70.0], [20.0]]))
    np.testing.assert_array_equal(next_obs, np.array([[1.0, 1.0], [2.0, 2.0]]))
    np.testing.assert_array_equal(next_privileged, np.array([[10.0], [20.0]]))


def test_resolve_transition_bootstrap_contract_separates_actor_and_storage_paths():
    contract = resolve_transition_bootstrap_contract(
        next_obs=np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
        next_privileged=None,
        info={
            "_final_observation": np.array([False, True]),
            "final_observation": {"obs": np.array([[5.0, 5.0], [8.0, 9.0]], dtype=np.float32)},
        },
        truncated=np.array([False, True]),
    )

    np.testing.assert_array_equal(contract.actor_next_obs, np.array([[1.0, 1.0], [2.0, 2.0]]))
    np.testing.assert_array_equal(
        contract.storage_next_obs, np.array([[1.0, 1.0], [8.0, 9.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(contract.final_mask, np.array([False, True]))
    np.testing.assert_array_equal(contract.timeout_final_mask, np.array([False, True]))
