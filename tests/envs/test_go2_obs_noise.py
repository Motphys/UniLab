"""Tests for Go2 per-step observation noise."""

from __future__ import annotations

import numpy as np

from unilab.envs.locomotion.go2.base import Go2BaseCfg, Go2BaseEnv, NoiseConfig


class _ConcreteGo2Env(Go2BaseEnv):
    """Minimal concrete subclass — only needed to satisfy the ABC."""

    def update_state(self, state):
        raise NotImplementedError


def _make_env(level: float) -> Go2BaseEnv:
    cfg = Go2BaseCfg(noise_config=NoiseConfig(level=level))
    env = object.__new__(_ConcreteGo2Env)
    env._cfg = cfg
    return env


class TestObsNoise:
    def test_noise_applied_when_level_positive(self):
        env = _make_env(level=1.0)
        data = np.ones((4, 10), dtype=np.float32)
        cfg = env._cfg.noise_config

        results = [env._obs_noise(data.copy(), cfg.scale_joint_angle) for _ in range(5)]
        assert any(not np.allclose(r, data) for r in results)

    def test_no_noise_when_level_zero(self):
        env = _make_env(level=0.0)
        data = np.ones((4, 10), dtype=np.float32)
        cfg = env._cfg.noise_config

        result = env._obs_noise(data.copy(), cfg.scale_joint_angle)
        np.testing.assert_array_equal(result, data)

    def test_noise_bounded_by_level_times_scale(self):
        env = _make_env(level=1.0)
        data = np.zeros((128, 29), dtype=np.float32)
        scale = 0.2
        result = env._obs_noise(data.copy(), scale)
        assert np.all(result >= -scale)
        assert np.all(result <= scale)

    def test_noise_scales_with_level(self):
        env_half = _make_env(level=0.5)
        env_full = _make_env(level=1.0)
        data = np.zeros((1024, 10), dtype=np.float32)
        scale = 1.0

        np.random.seed(0)
        r_half = env_half._obs_noise(data.copy(), scale)
        np.random.seed(0)
        r_full = env_full._obs_noise(data.copy(), scale)

        np.testing.assert_allclose(r_full, r_half * 2.0)

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
