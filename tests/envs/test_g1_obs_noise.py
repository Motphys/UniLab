"""Tests for G1 per-step observation noise."""

from __future__ import annotations

import numpy as np

from unilab.envs.locomotion.g1.base import G1BaseCfg, G1BaseEnv, NoiseConfig


class _ConcreteG1Env(G1BaseEnv):
    """Minimal concrete subclass — only needed to satisfy the ABC."""

    def update_state(self, state):
        raise NotImplementedError


def _make_env(level: float, *, seed: int | None = None) -> G1BaseEnv:
    cfg = G1BaseCfg(noise_config=NoiseConfig(level=level, seed=seed))
    env = object.__new__(_ConcreteG1Env)
    env._cfg = cfg
    return env


class TestObsNoise:
    def test_noise_applied_when_level_positive(self):
        env = _make_env(level=1.0)
        data = np.ones((4, 10), dtype=np.float32)
        cfg = env._cfg.noise_config

        results = [env._obs_noise(data.copy(), cfg.scale_joint_angle) for _ in range(5)]
        # At least one result should differ from the original
        assert any(not np.allclose(r, data) for r in results)

    def test_no_noise_when_level_zero(self):
        env = _make_env(level=0.0)
        data = np.ones((4, 10), dtype=np.float32)
        cfg = env._cfg.noise_config

        result = env._obs_noise(data, cfg.scale_joint_angle)
        assert result is data
        np.testing.assert_array_equal(result, data)

    def test_noise_bounded_by_level_times_scale(self):
        env = _make_env(level=1.0)
        data = np.zeros((128, 29), dtype=np.float32)
        scale = 0.2
        result = env._obs_noise(data.copy(), scale)
        # uniform[-1,1] * 1.0 * 0.2 => bounded by [-0.2, 0.2]
        assert np.all(result >= -scale)
        assert np.all(result <= scale)

    def test_noise_scales_with_level(self):
        env_half = _make_env(level=0.5, seed=123)
        env_full = _make_env(level=1.0, seed=123)
        data = np.zeros((1024, 10), dtype=np.float32)
        scale = 1.0

        r_half = env_half._obs_noise(data.copy(), scale)
        r_full = env_full._obs_noise(data.copy(), scale)

        np.testing.assert_allclose(r_full, r_half * 2.0)

    def test_configured_seed_is_reproducible_across_fresh_envs(self):
        env_a = _make_env(level=1.0, seed=11)
        env_b = _make_env(level=1.0, seed=11)
        data = np.zeros((32, 10), dtype=np.float32)

        first_a = env_a._obs_noise(data.copy(), 0.25)
        first_b = env_b._obs_noise(data.copy(), 0.25)
        second_a = env_a._obs_noise(data.copy(), 0.25)

        np.testing.assert_allclose(first_a, first_b)
        assert not np.allclose(first_a, second_a)

    def test_seed_observation_noise_resets_stream(self):
        env = _make_env(level=1.0)
        data = np.zeros((32, 10), dtype=np.float32)

        env.seed_observation_noise(17)
        first = env._obs_noise(data.copy(), 0.25)
        env.seed_observation_noise(17)
        replayed = env._obs_noise(data.copy(), 0.25)

        np.testing.assert_allclose(first, replayed)

    def test_seed_observation_noise_overrides_configured_seed(self):
        env = _make_env(level=1.0, seed=11)
        replay = _make_env(level=1.0, seed=99)
        data = np.zeros((32, 10), dtype=np.float32)

        env.seed_observation_noise(99)
        result = env._obs_noise(data.copy(), 0.25)
        expected = replay._obs_noise(data.copy(), 0.25)

        np.testing.assert_allclose(result, expected)

    def test_noise_preserves_dtype(self):
        for dt in [np.float32, np.float64]:
            env = _make_env(level=1.0)
            data = np.ones((4, 5), dtype=dt)
            result = env._obs_noise(data, 0.1)
            assert result.dtype == dt

    def test_noise_preserves_shape(self):
        env = _make_env(level=1.0)
        for shape in [(1, 3), (64, 29), (1024, 10)]:
            data = np.zeros(shape, dtype=np.float32)
            result = env._obs_noise(data, 0.1)
            assert result.shape == shape
