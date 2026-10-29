"""Tests for unilab.base.contracts.observations.

Ported from unilab_rl ``tests/utils/test_observations.py`` (itself adapted from
UniLab's pre-split ``tests/utils/test_obs_utils.py``).
"""

from __future__ import annotations

import numpy as np

from unilab.base.contracts.observations import get_obs_dims, split_obs_dict


class TestSplitObsDict:
    """Unit tests for split_obs_dict."""

    def test_with_critic(self):
        obs = {"obs": np.ones((4, 8)), "critic": np.full((4, 3), 2.0)}
        obs_arr, critic_arr = split_obs_dict(obs)
        assert obs_arr.shape == (4, 8)
        assert critic_arr is not None
        assert critic_arr.shape == (4, 3)
        np.testing.assert_array_equal(obs_arr, 1.0)
        np.testing.assert_array_equal(critic_arr, 2.0)

    def test_no_critic(self):
        obs = {"obs": np.ones((4, 8))}
        obs_arr, critic_arr = split_obs_dict(obs)
        assert obs_arr.shape == (4, 8)
        assert critic_arr.shape == (4, 8)
        np.testing.assert_array_equal(critic_arr, obs_arr)


class TestGetObsDims:
    """Unit tests for get_obs_dims."""

    def test_with_critic(self):
        spec = {"obs": 49, "critic": 52}
        obs_dim, critic_dim = get_obs_dims(spec)
        assert obs_dim == 49
        assert critic_dim == 52

    def test_no_critic(self):
        spec = {"obs": 49}
        obs_dim, critic_dim = get_obs_dims(spec)
        assert obs_dim == 49
        assert critic_dim == 49
