"""Tests for NanGuard: env-layer NaN/Inf detection and state dumping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unilab.tools.viz_nan import load_dump
from unilab.utils.nan_guard import NanGuard, NanGuardCfg

NUM_ENVS = 4
OBS_DIM = 10


def _make_clean_obs() -> dict[str, np.ndarray]:
    return {"policy": np.zeros((NUM_ENVS, OBS_DIM), dtype=np.float32)}


def _make_clean_reward() -> np.ndarray:
    return np.zeros(NUM_ENVS, dtype=np.float32)


# ── 1. disabled guard ──────────────────────────────────────────────────────


def test_disabled_guard_returns_none():
    cfg = NanGuardCfg(enabled=False)
    guard = NanGuard(cfg, NUM_ENVS, supports_state_playback=True)
    obs = _make_clean_obs()
    obs["policy"][0, 0] = np.nan
    assert guard.check(obs, _make_clean_reward()) is None


# ── 2. detect NaN in obs ──────────────────────────────────────────────────


def test_detect_nan_in_obs():
    cfg = NanGuardCfg(enabled=True)
    guard = NanGuard(cfg, NUM_ENVS, supports_state_playback=False)
    obs = _make_clean_obs()
    obs["policy"][1, 3] = np.nan
    obs["policy"][3, 7] = np.nan
    result = guard.check(obs, _make_clean_reward())
    assert result is not None
    np.testing.assert_array_equal(result, [1, 3])


# ── 3. detect Inf in obs ─────────────────────────────────────────────────


def test_detect_inf_in_obs():
    cfg = NanGuardCfg(enabled=True)
    guard = NanGuard(cfg, NUM_ENVS, supports_state_playback=False)
    obs = _make_clean_obs()
    obs["policy"][2, 0] = np.inf
    result = guard.check(obs, _make_clean_reward())
    assert result is not None
    np.testing.assert_array_equal(result, [2])


def test_detect_nan_in_secondary_obs_group():
    cfg = NanGuardCfg(enabled=True)
    guard = NanGuard(cfg, NUM_ENVS, supports_state_playback=False)
    obs = _make_clean_obs()
    obs["critic"] = np.zeros((NUM_ENVS, 3), dtype=np.float32)
    obs["critic"][2, 1] = np.nan
    result = guard.check(obs, _make_clean_reward())
    assert result is not None
    np.testing.assert_array_equal(result, [2])


# ── 4. detect NaN/Inf in reward ──────────────────────────────────────────


def test_detect_nan_in_reward():
    cfg = NanGuardCfg(enabled=True)
    guard = NanGuard(cfg, NUM_ENVS, supports_state_playback=False)
    reward = _make_clean_reward()
    reward[0] = np.nan
    reward[2] = -np.inf
    result = guard.check(_make_clean_obs(), reward)
    assert result is not None
    np.testing.assert_array_equal(result, [0, 2])


# ── 5. rolling buffer capacity ───────────────────────────────────────────


def test_rolling_buffer_capacity():
    buf_size = 5
    cfg = NanGuardCfg(enabled=True, buffer_size=buf_size)
    guard = NanGuard(cfg, NUM_ENVS, supports_state_playback=True)
    for i in range(buf_size + 3):
        state = np.full((NUM_ENVS, 4), float(i), dtype=np.float32)
        guard.capture(state)
    assert len(guard._buffer) == buf_size


def test_dump_preserves_rolling_buffer_order(tmp_path):
    cfg = NanGuardCfg(enabled=True, buffer_size=3, output_dir=str(tmp_path))
    guard = NanGuard(cfg, NUM_ENVS, supports_state_playback=True)
    for i in range(5):
        guard.capture(np.full((NUM_ENVS, 2), float(i), dtype=np.float32))
    path = guard.dump(np.array([0], dtype=np.int32), model_file="", step=5)
    assert path is not None
    data = np.load(path, allow_pickle=True)
    np.testing.assert_array_equal(data["states"][:, 0, 0], [2.0, 3.0, 4.0])


# ── 6. dump output format ────────────────────────────────────────────────


def test_dump_output_format(tmp_path):
    cfg = NanGuardCfg(enabled=True, buffer_size=3, output_dir=str(tmp_path))
    guard = NanGuard(cfg, NUM_ENVS, supports_state_playback=True)
    for i in range(3):
        guard.capture(np.ones((NUM_ENVS, 4), dtype=np.float32) * i)
    nan_ids = np.array([0, 2], dtype=np.int32)
    path = guard.dump(nan_ids, model_file="", step=42)
    assert path is not None
    data = np.load(path, allow_pickle=True)
    assert "states" in data
    assert data["states"].shape[0] == 3
    assert int(data["meta_detection_step"]) == 42


def test_dump_limits_state_envs_and_records_all_nan_ids(tmp_path):
    cfg = NanGuardCfg(
        enabled=True,
        buffer_size=2,
        max_envs_to_dump=2,
        output_dir=str(tmp_path),
    )
    guard = NanGuard(cfg, NUM_ENVS, supports_state_playback=True)
    state = np.arange(NUM_ENVS * 3, dtype=np.float32).reshape(NUM_ENVS, 3)
    guard.capture(state)
    nan_ids = np.array([0, 1, 2, 3], dtype=np.int32)
    path = guard.dump(nan_ids, model_file="", step=7)
    assert path is not None
    data = np.load(path, allow_pickle=True)
    assert data["states"].shape == (1, 2, 3)
    np.testing.assert_array_equal(data["states"][0], state[[0, 1]])
    np.testing.assert_array_equal(data["meta_nan_env_ids"], nan_ids)
    np.testing.assert_array_equal(data["meta_dumped_env_ids"], [0, 1])


def test_dump_copies_model_file_and_updates_latest_link(tmp_path):
    model_file = tmp_path / "model.xml"
    model_file.write_text("<mujoco/>")
    cfg = NanGuardCfg(enabled=True, output_dir=str(tmp_path / "dumps"))
    guard = NanGuard(cfg, NUM_ENVS, supports_state_playback=True)
    guard.capture(np.zeros((NUM_ENVS, 2), dtype=np.float32))
    path = guard.dump(np.array([0], dtype=np.int32), model_file=str(model_file), step=3)
    assert path is not None
    dump_path = Path(path)
    copied_models = list(dump_path.parent.glob("*_model.xml"))
    assert len(copied_models) == 1
    assert copied_models[0].read_text() == "<mujoco/>"
    latest_link = dump_path.parent / "nan_dump_latest.npz"
    if latest_link.exists():
        assert latest_link.resolve() == dump_path


def test_load_dump_round_trips_states_and_metadata(tmp_path):
    cfg = NanGuardCfg(enabled=True, buffer_size=1, output_dir=str(tmp_path))
    guard = NanGuard(cfg, NUM_ENVS, supports_state_playback=True)
    state = np.arange(NUM_ENVS * 2, dtype=np.float32).reshape(NUM_ENVS, 2)
    guard.capture(state)
    nan_ids = np.array([1, 3], dtype=np.int32)
    path = guard.dump(nan_ids, model_file="", step=11)
    assert path is not None
    dump = load_dump(path)
    np.testing.assert_array_equal(dump["states"], state[[1, 3]][None, ...])
    assert dump["metadata"]["detection_step"] == 11
    np.testing.assert_array_equal(dump["metadata"]["nan_env_ids"], nan_ids)


# ── 7. dump only once ────────────────────────────────────────────────────


def test_dump_only_once(tmp_path):
    cfg = NanGuardCfg(enabled=True, output_dir=str(tmp_path))
    guard = NanGuard(cfg, NUM_ENVS, supports_state_playback=False)
    nan_ids = np.array([0], dtype=np.int32)
    first = guard.dump(nan_ids, model_file="", step=1)
    second = guard.dump(nan_ids, model_file="", step=2)
    assert first is not None
    assert second is None


# ── 8. no physics state still detects ────────────────────────────────────


def test_no_physics_state_still_detects(tmp_path):
    cfg = NanGuardCfg(enabled=True, output_dir=str(tmp_path))
    guard = NanGuard(cfg, NUM_ENVS, supports_state_playback=False)
    guard.capture(None)
    obs = _make_clean_obs()
    obs["policy"][0, 0] = np.nan
    result = guard.check(obs, _make_clean_reward())
    assert result is not None
    path = guard.dump(result, model_file="", step=10)
    assert path is not None
    data = np.load(path, allow_pickle=True)
    assert data["states"].size == 0


def test_check_warns_each_call_with_nan(caplog):
    cfg = NanGuardCfg(enabled=True)
    guard = NanGuard(cfg, num_envs=NUM_ENVS, supports_state_playback=False)
    obs = _make_clean_obs()
    obs["policy"][1, 0] = np.nan
    reward = _make_clean_reward()
    with caplog.at_level("WARNING", logger="unilab.utils.nan_guard"):
        guard.check(obs, reward, step=10)
        guard.check(obs, reward, step=11)
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 2
    assert "step 10" in warnings[0].getMessage()
    assert "step 11" in warnings[1].getMessage()


def test_check_ctrl_warns_each_call_with_nan(caplog):
    cfg = NanGuardCfg(enabled=True)
    guard = NanGuard(cfg, num_envs=NUM_ENVS, supports_state_playback=False)
    ctrl = np.zeros((NUM_ENVS, 3), dtype=np.float64)
    ctrl[2, 1] = np.nan
    with caplog.at_level("WARNING", logger="unilab.utils.nan_guard"):
        guard.check_ctrl(ctrl, step=5)
        guard.check_ctrl(ctrl, step=6)
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 2
    assert all("ctrl" in w.getMessage() for w in warnings)


def test_warning_includes_step_and_env_count(caplog):
    cfg = NanGuardCfg(enabled=True)
    guard = NanGuard(cfg, num_envs=NUM_ENVS, supports_state_playback=False)
    obs = _make_clean_obs()
    obs["policy"][0, 0] = np.nan
    obs["policy"][2, 0] = np.nan
    reward = _make_clean_reward()
    with caplog.at_level("WARNING", logger="unilab.utils.nan_guard"):
        guard.check(obs, reward, step=42)
    msg = caplog.records[-1].getMessage()
    assert "step 42" in msg
    assert "envs=2" in msg
    assert "sample_ids=" in msg


# ---------------------------------------------------------------------------
# Repro #1 (issue #594): NaN guard inside a real NpEnv.step loop warns
# every step, dumps exactly once, and reward is sanitized at end of step.
# ---------------------------------------------------------------------------


def test_nan_guard_warns_on_every_step_and_dumps_once_real_env_loop(tmp_path, caplog):
    """End-to-end: NaN-injected NpEnv steps 5x -> 5 warnings, 1 dump file.

    Validates the env-layer NaN guard wiring stays correct across multiple
    step calls (warns each call) while the dump (heavy I/O) only happens
    once. Also confirms reward is sanitized after step despite NaN input.
    """
    from dataclasses import dataclass
    from typing import Tuple
    from unittest.mock import MagicMock

    import gymnasium as gym

    from unilab.base.base import EnvCfg
    from unilab.base.np_env import NpEnv, NpEnvState

    @dataclass
    class _StubCfgRepro(EnvCfg):
        max_episode_seconds: float | None = 1.0
        ctrl_dt: float = 0.1
        sim_dt: float = 0.01

    class _StubNpEnvForNaNRepro(NpEnv):
        OBS_SPEC = {"obs": 5, "critic": 7}

        def __init__(self, num_envs: int = 4, bad_rewards: np.ndarray | None = None):
            cfg = _StubCfgRepro()
            backend = MagicMock()
            backend.backend_type = "mujoco"
            backend.step = MagicMock()
            super().__init__(cfg, backend, num_envs)
            self._bad_rewards = bad_rewards

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
                "critic": np.full((self._num_envs, 7), 0.5, dtype=np.float32),
            }
            reward = (
                self._bad_rewards.copy()
                if self._bad_rewards is not None
                else np.ones((self._num_envs,), dtype=np.float32)
            )
            return state.replace(
                obs=obs,
                reward=reward,
                terminated=np.zeros((self._num_envs,), dtype=bool),
                truncated=np.zeros((self._num_envs,), dtype=bool),
            )

        def reset(self, env_indices: np.ndarray) -> Tuple[dict[str, np.ndarray], dict]:
            n = len(env_indices)
            obs = {
                "obs": np.zeros((n, 5), dtype=np.float32),
                "critic": np.zeros((n, 7), dtype=np.float32),
            }
            return obs, {}

    bad_rewards = np.array([0.0, np.nan, 0.0, 0.0], dtype=np.float32)
    env = _StubNpEnvForNaNRepro(num_envs=4, bad_rewards=bad_rewards)
    env.init_state()

    dump_dir = tmp_path / "dumps"
    guard = NanGuard(
        NanGuardCfg(enabled=True, output_dir=str(dump_dir)),
        num_envs=4,
        supports_state_playback=False,
    )
    env.set_nan_guard(guard)

    with caplog.at_level("WARNING", logger="unilab.utils.nan_guard"):
        for _ in range(5):
            state = env.step(np.zeros((4, 3), dtype=np.float64))
            assert np.all(np.isfinite(state.reward))

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 5, f"expected 5 warnings, got {len(warnings)}"

    dump_files = [p for p in dump_dir.glob("nan_dump_*.npz") if not p.is_symlink()]
    assert len(dump_files) == 1, f"expected exactly 1 dump file, got {dump_files}"
