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
