"""Tests for NpEnvState and NpEnv dict-obs contract.

These tests use a minimal concrete NpEnv stub — no MuJoCo required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest

from unilab.base.base import EnvCfg
from unilab.base.np_env import NpEnv, NpEnvState

# ---------------------------------------------------------------------------
# Fixtures: minimal concrete NpEnv
# ---------------------------------------------------------------------------


@dataclass
class _StubCfg(EnvCfg):
    max_episode_seconds: float = 1.0
    ctrl_dt: float = 0.1
    sim_dt: float = 0.01


class _StubNpEnv(NpEnv):
    """Concrete NpEnv for testing — no physics, deterministic outputs."""

    OBS_SPEC = {"obs": 5, "privileged": 2}

    def __init__(self, num_envs: int = 4):
        cfg = _StubCfg()
        backend = MagicMock()
        backend.backend_type = "mujoco"
        backend.step = MagicMock()
        super().__init__(cfg, backend, num_envs)
        self._reset_count = 0

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return self.OBS_SPEC

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        return actions

    def update_state(self, state: NpEnvState) -> NpEnvState:
        obs = {
            "obs": np.ones((self._num_envs, 5), dtype=np.float32),
            "privileged": np.full((self._num_envs, 2), 0.5, dtype=np.float32),
        }
        return state.replace(
            obs=obs,
            reward=np.ones((self._num_envs,), dtype=np.float32),
            terminated=np.zeros((self._num_envs,), dtype=bool),
            truncated=np.zeros((self._num_envs,), dtype=bool),
        )

    def reset(self, env_indices: np.ndarray) -> Tuple[dict[str, np.ndarray], dict]:
        self._reset_count += 1
        n = len(env_indices)
        obs = {
            "obs": np.zeros((n, 5), dtype=np.float32),
            "privileged": np.zeros((n, 2), dtype=np.float32),
        }
        return obs, {}


class _TerminatingStubEnv(_StubNpEnv):
    """Like _StubNpEnv but terminates specified envs each step."""

    def __init__(self, num_envs: int = 4, terminate_indices: list[int] | None = None):
        super().__init__(num_envs)
        self._terminate_indices = terminate_indices or []

    def update_state(self, state: NpEnvState) -> NpEnvState:
        state = super().update_state(state)
        terminated = np.zeros((self._num_envs,), dtype=bool)
        terminated[self._terminate_indices] = True
        return state.replace(terminated=terminated)


# ---------------------------------------------------------------------------
# NpEnvState tests
# ---------------------------------------------------------------------------


class TestNpEnvState:
    def test_obs_is_dict(self):
        obs = {"obs": np.zeros((2, 4)), "privileged": np.zeros((2, 1))}
        state = NpEnvState(
            obs=obs,
            reward=np.zeros(2),
            terminated=np.zeros(2, dtype=bool),
            truncated=np.zeros(2, dtype=bool),
            info={},
        )
        assert isinstance(state.obs, dict)
        assert "obs" in state.obs
        assert "privileged" in state.obs

    def test_done_combines_terminated_and_truncated(self):
        state = NpEnvState(
            obs={"a": np.zeros((3, 1))},
            reward=np.zeros(3),
            terminated=np.array([True, False, False]),
            truncated=np.array([False, False, True]),
            info={},
        )
        done = state.done
        np.testing.assert_array_equal(done, [True, False, True])

    def test_done_both_true(self):
        state = NpEnvState(
            obs={"a": np.zeros((1, 1))},
            reward=np.zeros(1),
            terminated=np.array([True]),
            truncated=np.array([True]),
            info={},
        )
        assert state.done[0] is np.True_

    def test_replace_preserves_type(self):
        obs = {"obs": np.zeros((2, 3))}
        state = NpEnvState(
            obs=obs,
            reward=np.zeros(2),
            terminated=np.zeros(2, dtype=bool),
            truncated=np.zeros(2, dtype=bool),
            info={},
        )
        new_obs = {"obs": np.ones((2, 3))}
        state2 = state.replace(obs=new_obs)
        assert isinstance(state2, NpEnvState)
        assert isinstance(state2.obs, dict)
        np.testing.assert_array_equal(state2.obs["obs"], 1.0)
        # Original unchanged
        np.testing.assert_array_equal(state.obs["obs"], 0.0)

    def test_replace_partial_update(self):
        state = NpEnvState(
            obs={"a": np.zeros((1, 1))},
            reward=np.array([0.0]),
            terminated=np.array([False]),
            truncated=np.array([False]),
            info={"k": "v"},
        )
        state2 = state.replace(reward=np.array([5.0]))
        assert state2.reward[0] == 5.0
        assert state2.obs is state.obs  # not copied


# ---------------------------------------------------------------------------
# NpEnv — obs_groups_spec / observation_space
# ---------------------------------------------------------------------------


class TestNpEnvObsSpec:
    def test_obs_groups_spec_returns_dict(self):
        env = _StubNpEnv(num_envs=2)
        spec = env.obs_groups_spec
        assert isinstance(spec, dict)
        assert spec == {"obs": 5, "privileged": 2}

    def test_observation_space_total_dim(self):
        env = _StubNpEnv(num_envs=2)
        space = env.observation_space
        assert isinstance(space, gym.spaces.Box)
        assert space.shape == (7,)  # 5 + 2

    def test_observation_space_bounds(self):
        env = _StubNpEnv(num_envs=1)
        space = env.observation_space
        assert np.all(space.low == -np.inf)
        assert np.all(space.high == np.inf)

    def test_bare_npenv_obs_groups_spec_raises(self):
        """NpEnv.obs_groups_spec raises NotImplementedError if not overridden."""
        # Access the property via the NpEnv class descriptor directly
        with pytest.raises(NotImplementedError):
            NpEnv.obs_groups_spec.fget(MagicMock(spec=NpEnv))  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# NpEnv.init_state — dict obs allocation
# ---------------------------------------------------------------------------


class TestNpEnvInitState:
    def test_init_state_returns_np_env_state(self):
        env = _StubNpEnv(num_envs=4)
        state = env.init_state()
        assert isinstance(state, NpEnvState)

    def test_init_state_obs_is_dict(self):
        env = _StubNpEnv(num_envs=4)
        env.init_state()
        assert isinstance(env.state.obs, dict)

    def test_init_state_obs_keys_match_spec(self):
        env = _StubNpEnv(num_envs=4)
        env.init_state()
        assert set(env.state.obs.keys()) == {"obs", "privileged"}

    def test_init_state_obs_shapes(self):
        env = _StubNpEnv(num_envs=4)
        env.init_state()
        assert env.state.obs["obs"].shape == (4, 5)
        assert env.state.obs["privileged"].shape == (4, 2)

    def test_init_state_obs_zeros(self):
        """Initial obs should be zeros (before reset fills them)."""
        env = _StubNpEnv(num_envs=2)
        # init_state internally calls _reset_done_envs which calls reset(),
        # so the obs after init_state will be the reset values
        env.init_state()
        # After init_state, reset was called (terminated=True initially)
        # Our stub resets to zeros, so should be zeros
        np.testing.assert_array_equal(env.state.obs["obs"], 0.0)

    def test_init_state_triggers_reset(self):
        env = _StubNpEnv(num_envs=2)
        env.init_state()
        # init_state sets terminated=True initially, so _reset_done_envs triggers
        assert env._reset_count > 0

    def test_init_state_reward_shape(self):
        env = _StubNpEnv(num_envs=3)
        env.init_state()
        assert env.state.reward.shape == (3,)

    def test_init_state_steps_initialized(self):
        env = _StubNpEnv(num_envs=2)
        env.init_state()
        assert "steps" in env.state.info
        np.testing.assert_array_equal(env.state.info["steps"], 0)


# ---------------------------------------------------------------------------
# NpEnv.step — dict obs flow
# ---------------------------------------------------------------------------


class TestNpEnvStep:
    def test_step_returns_state_with_dict_obs(self):
        env = _StubNpEnv(num_envs=2)
        env.init_state()
        actions = np.zeros((2, 3))
        state = env.step(actions)
        assert isinstance(state.obs, dict)
        assert "obs" in state.obs
        assert "privileged" in state.obs

    def test_step_obs_values_from_update_state(self):
        env = _StubNpEnv(num_envs=2)
        env.init_state()
        state = env.step(np.zeros((2, 3)))
        # _StubNpEnv.update_state fills actor=1.0, privileged=0.5
        np.testing.assert_array_equal(state.obs["obs"], 1.0)
        np.testing.assert_array_equal(state.obs["privileged"], 0.5)

    def test_step_increments_counter(self):
        env = _StubNpEnv(num_envs=1)
        env.init_state()
        env.step(np.zeros((1, 3)))
        assert env.step_counter == 1
        env.step(np.zeros((1, 3)))
        assert env.step_counter == 2

    def test_step_increments_steps_info(self):
        env = _StubNpEnv(num_envs=2)
        env.init_state()
        env.step(np.zeros((2, 3)))
        np.testing.assert_array_equal(env.state.info["steps"], 1)

    def test_step_auto_inits_if_no_state(self):
        env = _StubNpEnv(num_envs=1)
        # Don't call init_state
        state = env.step(np.zeros((1, 3)))
        assert isinstance(state.obs, dict)


# ---------------------------------------------------------------------------
# NpEnv._reset_done_envs — dict final_observation handling
# ---------------------------------------------------------------------------


class TestResetDoneEnvs:
    def test_done_envs_get_reset_obs(self):
        env = _TerminatingStubEnv(num_envs=4, terminate_indices=[0, 2])
        env.init_state()
        state = env.step(np.zeros((4, 3)))
        # Envs 0, 2 terminated — their obs should be reset values (zeros from stub)
        np.testing.assert_array_equal(state.obs["obs"][0], 0.0)
        np.testing.assert_array_equal(state.obs["obs"][2], 0.0)
        # Envs 1, 3 not terminated — their obs should be update_state values (ones)
        np.testing.assert_array_equal(state.obs["obs"][1], 1.0)
        np.testing.assert_array_equal(state.obs["obs"][3], 1.0)

    def test_final_observation_is_dict(self):
        env = _TerminatingStubEnv(num_envs=2, terminate_indices=[0])
        env.init_state()
        env.step(np.zeros((2, 3)))
        assert isinstance(env.state.info["final_observation"], dict)
        assert "obs" in env.state.info["final_observation"]
        assert "privileged" in env.state.info["final_observation"]

    def test_final_observation_captures_pre_reset_obs(self):
        env = _TerminatingStubEnv(num_envs=2, terminate_indices=[0])
        env.init_state()
        env.step(np.zeros((2, 3)))
        # update_state fills actor=1.0 before reset replaces it with 0.0
        # final_observation should capture the pre-reset obs (1.0)
        np.testing.assert_array_equal(env.state.info["final_observation"]["obs"][0], 1.0)

    def test_final_observation_mask(self):
        env = _TerminatingStubEnv(num_envs=3, terminate_indices=[1])
        env.init_state()
        env.step(np.zeros((3, 3)))
        mask = env.state.info["_final_observation"]
        np.testing.assert_array_equal(mask, [False, True, False])

    def test_no_termination_skips_reset(self):
        env = _TerminatingStubEnv(num_envs=2, terminate_indices=[])
        env.init_state()
        reset_count_after_init = env._reset_count
        # Step without any terminations
        env.step(np.zeros((2, 3)))
        # No additional resets should have occurred
        assert env._reset_count == reset_count_after_init

    def test_steps_reset_for_done_envs(self):
        env = _TerminatingStubEnv(num_envs=3, terminate_indices=[0, 2])
        env.init_state()
        env.step(np.zeros((3, 3)))
        # After step, terminated envs had their steps reset to 0
        # then +1 from step, but _reset_done_envs is called AFTER info["steps"] += 1,
        # and it sets steps[env_indices] = 0
        assert env.state.info["steps"][0] == 0
        assert env.state.info["steps"][2] == 0
        assert env.state.info["steps"][1] == 1

    def test_truncation_triggers_reset(self):
        """max_episode_steps triggers truncation → reset."""
        env = _StubNpEnv(num_envs=1)
        env.init_state()
        # max_episode_steps = 1.0 / 0.1 = 10
        for _ in range(10):
            env.step(np.zeros((1, 3)))
        # After 10 steps, env should have been truncated and reset
        assert env.state.info["steps"][0] == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_obs_group_symmetric(self):
        """Env with only "obs" (no privileged) works correctly."""

        class _SymmetricEnv(_StubNpEnv):
            OBS_SPEC = {"obs": 10}

            def update_state(self, state):
                obs = {"obs": np.ones((self._num_envs, 10), dtype=np.float32)}
                return state.replace(
                    obs=obs,
                    reward=np.zeros(self._num_envs, dtype=np.float32),
                    terminated=np.zeros(self._num_envs, dtype=bool),
                    truncated=np.zeros(self._num_envs, dtype=bool),
                )

            def reset(self, env_indices):
                n = len(env_indices)
                obs = {"obs": np.zeros((n, 10), dtype=np.float32)}
                return obs, {}

        env = _SymmetricEnv(num_envs=2)
        env.init_state()
        state = env.step(np.zeros((2, 3)))
        assert set(state.obs.keys()) == {"obs"}
        assert state.obs["obs"].shape == (2, 10)
        assert env.observation_space.shape == (10,)

    def test_many_obs_groups(self):
        """Env with 3+ obs groups allocates correctly."""

        class _MultiGroupEnv(_StubNpEnv):
            OBS_SPEC = {"obs": 4, "privileged": 2, "history": 8}

            def update_state(self, state):
                obs = {
                    k: np.ones((self._num_envs, d), dtype=np.float32)
                    for k, d in self.OBS_SPEC.items()
                }
                return state.replace(
                    obs=obs,
                    reward=np.zeros(self._num_envs, dtype=np.float32),
                    terminated=np.zeros(self._num_envs, dtype=bool),
                    truncated=np.zeros(self._num_envs, dtype=bool),
                )

            def reset(self, env_indices):
                n = len(env_indices)
                obs = {k: np.zeros((n, d), dtype=np.float32) for k, d in self.OBS_SPEC.items()}
                return obs, {}

        env = _MultiGroupEnv(num_envs=1)
        env.init_state()
        state = env.step(np.zeros((1, 3)))
        assert set(state.obs.keys()) == {"obs", "privileged", "history"}
        assert env.observation_space.shape == (14,)  # 4+2+8
