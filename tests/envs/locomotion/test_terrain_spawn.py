"""Unit tests for TerrainSpawnManager (terrain-aware spawn + curriculum)."""

from __future__ import annotations

import numpy as np
import pytest

from unilab.envs.locomotion.common.terrain_spawn import (
    TerrainCurriculumCfg,
    TerrainSpawnManager,
)


def _make_terrain_origins(num_rows: int, num_cols: int, cell_size: float) -> np.ndarray:
    rs = np.arange(num_rows)
    cs = np.arange(num_cols)
    xx, yy = np.meshgrid(rs, cs, indexing="ij")
    z = (xx + yy * 0.1).astype(np.float64)
    origins = np.stack([xx * cell_size, yy * cell_size, z], axis=-1).astype(np.float64)
    return origins


def _make_manager(
    num_envs: int = 8,
    num_rows: int = 5,
    num_cols: int = 4,
    cell_size: float = 8.0,
    enabled: bool = False,
    seed: int | None = 0,
    **cfg_overrides,
) -> TerrainSpawnManager:
    origins = _make_terrain_origins(num_rows, num_cols, cell_size)
    cfg = TerrainCurriculumCfg(enabled=enabled, seed=seed, **cfg_overrides)
    return TerrainSpawnManager(num_envs, origins, cell_size, cfg)


def test_init_levels_zero_when_enabled():
    sm = _make_manager(enabled=True)
    assert np.all(sm.levels == 0)


def test_init_levels_uniform_when_disabled():
    sm = _make_manager(num_envs=400, num_rows=5, enabled=False, seed=0)
    assert sm.levels.min() == 0
    assert sm.levels.max() == 4
    assert 0 < sm.levels.mean() < 4


def test_init_type_cols_random_seeded():
    a = _make_manager(seed=42)
    b = _make_manager(seed=42)
    c = _make_manager(seed=43)
    assert np.array_equal(a.type_cols, b.type_cols)
    assert not np.array_equal(a.type_cols, c.type_cols)


def test_origins_match_terrain_origins_plus_margin():
    sm = _make_manager(num_envs=4, enabled=True, spawn_height_margin=0.07)
    sm.levels[:] = [1, 2, 3, 4]
    sm.type_cols[:] = [0, 1, 2, 3]
    out = sm.origins_for(np.arange(4))
    expected_xy = np.array(
        [
            [1 * 8.0, 0 * 8.0],
            [2 * 8.0, 1 * 8.0],
            [3 * 8.0, 2 * 8.0],
            [4 * 8.0, 3 * 8.0],
        ]
    )
    expected_z = np.array([1 + 0 * 0.1, 2 + 1 * 0.1, 3 + 2 * 0.1, 4 + 3 * 0.1]) + 0.07
    np.testing.assert_allclose(out[:, :2], expected_xy)
    np.testing.assert_allclose(out[:, 2], expected_z)


def test_promote_when_walked_far_and_enabled():
    sm = _make_manager(num_envs=2, cell_size=8.0, enabled=True)
    sm.record_episode_start(np.array([0, 1]), np.zeros((2, 3)))
    current = np.array([[10.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    stats = sm.update_on_done(np.array([0, 1]), current)
    assert sm.levels[0] == 1
    assert sm.levels[1] == 0
    assert stats["num_promoted"] == 1


def test_demote_when_walked_short_and_enabled():
    sm = _make_manager(num_envs=2, cell_size=8.0, enabled=True)
    sm.levels[:] = [3, 3]
    sm.record_episode_start(np.array([0, 1]), np.zeros((2, 3)))
    current = np.array([[0.5, 0.0, 0.0], [10.0, 0.0, 0.0]])
    stats = sm.update_on_done(np.array([0, 1]), current)
    assert sm.levels[0] == 2
    assert sm.levels[1] == 4
    assert stats["num_demoted"] == 1
    assert stats["num_promoted"] == 1


def test_levels_immutable_when_disabled():
    sm = _make_manager(num_envs=4, cell_size=8.0, enabled=False)
    initial = sm.levels.copy()
    sm.record_episode_start(np.arange(4), np.zeros((4, 3)))
    current = np.tile(np.array([[100.0, 0.0, 0.0]]), (4, 1))
    stats = sm.update_on_done(np.arange(4), current)
    np.testing.assert_array_equal(sm.levels, initial)
    assert stats["num_promoted"] == 0
    assert stats["num_demoted"] == 0


def test_clamp_at_zero():
    sm = _make_manager(num_envs=1, cell_size=8.0, enabled=True)
    sm.record_episode_start(np.array([0]), np.zeros((1, 3)))
    sm.update_on_done(np.array([0]), np.array([[0.1, 0.0, 0.0]]))
    assert sm.levels[0] == 0


def test_cycle_at_top_distribution():
    num_rows = 6
    sm = _make_manager(num_envs=1, num_rows=num_rows, cell_size=8.0, enabled=True)
    sm.levels[0] = num_rows - 1
    landed = []
    for _ in range(1000):
        sm.record_episode_start(np.array([0]), np.zeros((1, 3)))
        sm.update_on_done(np.array([0]), np.array([[100.0, 0.0, 0.0]]))
        landed.append(int(sm.levels[0]))
    landed_arr = np.array(landed)
    assert landed_arr.min() >= num_rows // 2
    assert landed_arr.max() == num_rows - 1


def test_log_stats_keys():
    sm = _make_manager(num_envs=2, enabled=True)
    sm.record_episode_start(np.array([0, 1]), np.zeros((2, 3)))
    stats = sm.update_on_done(np.array([0, 1]), np.zeros((2, 3)))
    expected = {
        "mean_level",
        "max_level",
        "mean_walked",
        "num_promoted",
        "num_demoted",
        "num_skipped",
    }
    assert set(stats.keys()) == expected


def test_first_done_skipped_via_has_started():
    sm = _make_manager(num_envs=2, enabled=True)
    stats = sm.update_on_done(np.array([0, 1]), np.zeros((2, 3)))
    assert stats["num_skipped"] == 2
    assert stats["num_promoted"] == 0


def test_num_rows_too_small_raises_when_enabled():
    origins = _make_terrain_origins(num_rows=1, num_cols=2, cell_size=8.0)
    cfg = TerrainCurriculumCfg(enabled=True)
    with pytest.raises(ValueError, match="num_rows >= 2"):
        TerrainSpawnManager(4, origins, 8.0, cfg)


def test_invalid_terrain_origins_shape_raises():
    cfg = TerrainCurriculumCfg(enabled=False)
    with pytest.raises(ValueError, match="shape"):
        TerrainSpawnManager(4, np.zeros((5, 4)), 8.0, cfg)
