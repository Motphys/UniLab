"""Tests for unilab.utils.obs_utils."""

from __future__ import annotations

import numpy as np
import pytest

from unilab.utils.obs_utils import flatten_obs_dict, get_obs_dims, split_obs_dict

# ---------------------------------------------------------------------------
# flatten_obs_dict — basic behaviour
# ---------------------------------------------------------------------------


class TestFlattenObsDict:
    """Unit tests for flatten_obs_dict."""

    def test_single_group(self):
        obs = {"actor": np.ones((4, 8))}
        flat = flatten_obs_dict(obs)
        assert flat.shape == (4, 8)
        np.testing.assert_array_equal(flat, np.ones((4, 8)))

    def test_two_groups_concatenated(self):
        obs = {
            "actor": np.ones((2, 5)),
            "privileged": np.full((2, 3), 2.0),
        }
        flat = flatten_obs_dict(obs)
        assert flat.shape == (2, 8)
        np.testing.assert_array_equal(flat[:, :5], 1.0)
        np.testing.assert_array_equal(flat[:, 5:], 2.0)

    def test_insertion_order_preserved(self):
        """Groups are concatenated in dict insertion order."""
        a = np.array([[1.0, 2.0]])
        b = np.array([[3.0]])
        c = np.array([[4.0, 5.0, 6.0]])
        obs = {"first": a, "second": b, "third": c}
        flat = flatten_obs_dict(obs)
        expected = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        np.testing.assert_array_equal(flat, expected)

    def test_three_groups(self):
        obs = {
            "actor": np.zeros((3, 10)),
            "privileged": np.ones((3, 4)),
            "extra": np.full((3, 2), 9.0),
        }
        flat = flatten_obs_dict(obs)
        assert flat.shape == (3, 16)

    def test_dtype_preserved_float32(self):
        obs = {"actor": np.zeros((1, 5), dtype=np.float32)}
        flat = flatten_obs_dict(obs)
        assert flat.dtype == np.float32

    def test_dtype_preserved_float64(self):
        obs = {"actor": np.zeros((1, 5), dtype=np.float64)}
        flat = flatten_obs_dict(obs)
        assert flat.dtype == np.float64

    def test_mixed_dtype_upcasts(self):
        """NumPy concatenation rules apply — float64 wins over float32."""
        obs = {
            "actor": np.zeros((2, 3), dtype=np.float32),
            "privileged": np.zeros((2, 1), dtype=np.float64),
        }
        flat = flatten_obs_dict(obs)
        assert flat.dtype == np.float64

    def test_large_batch(self):
        n = 4096
        obs = {"actor": np.random.randn(n, 98), "privileged": np.random.randn(n, 3)}
        flat = flatten_obs_dict(obs)
        assert flat.shape == (n, 101)
        np.testing.assert_array_equal(flat[:, :98], obs["actor"])
        np.testing.assert_array_equal(flat[:, 98:], obs["privileged"])

    def test_single_env(self):
        obs = {"actor": np.array([[1.0, 2.0, 3.0]])}
        flat = flatten_obs_dict(obs)
        assert flat.shape == (1, 3)

    def test_values_roundtrip(self):
        """Flatten then slice recovers original groups."""
        actor = np.random.randn(8, 45)
        priv = np.random.randn(8, 3)
        obs = {"actor": actor, "privileged": priv}
        flat = flatten_obs_dict(obs)
        np.testing.assert_array_equal(flat[:, :45], actor)
        np.testing.assert_array_equal(flat[:, 45:], priv)


class TestSplitObsDict:
    """Unit tests for split_obs_dict."""

    def test_with_privileged(self):
        obs = {"obs": np.ones((4, 8)), "privileged": np.full((4, 3), 2.0)}
        obs_arr, priv_arr = split_obs_dict(obs)
        assert obs_arr.shape == (4, 8)
        assert priv_arr.shape == (4, 3)
        np.testing.assert_array_equal(obs_arr, 1.0)
        np.testing.assert_array_equal(priv_arr, 2.0)

    def test_no_privileged(self):
        obs = {"obs": np.ones((4, 8))}
        obs_arr, priv_arr = split_obs_dict(obs)
        assert obs_arr.shape == (4, 8)
        assert priv_arr is None


class TestGetObsDims:
    """Unit tests for get_obs_dims."""

    def test_with_privileged(self):
        spec = {"obs": 49, "privileged": 3}
        obs_dim, priv_dim = get_obs_dims(spec)
        assert obs_dim == 49
        assert priv_dim == 3

    def test_no_privileged(self):
        spec = {"obs": 49}
        obs_dim, priv_dim = get_obs_dims(spec)
        assert obs_dim == 49
        assert priv_dim == 0
