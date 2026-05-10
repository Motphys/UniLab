"""Integration tests for Go2 + TerrainSpawnManager."""

from __future__ import annotations

import numpy as np
import pytest

from unilab.envs.locomotion.common.terrain_spawn import (
    BaseSpawnManager,
    TerrainCurriculumCfg,
    TerrainSpawnManager,
)


def _rough_cfg(*, curriculum_enabled: bool = False, seed: int = 0):
    from unilab.envs.locomotion.go2.joystick import RewardConfig
    from unilab.envs.locomotion.go2.rough import Go2JoystickRoughCfg

    cfg = Go2JoystickRoughCfg(
        reward_config=RewardConfig(scales={}, tracking_sigma=0.25, base_height_target=0.3)
    )
    cfg.terrain_generator.num_rows = 3
    cfg.terrain_generator.num_cols = 3
    cfg.terrain_generator.border_width = 0.0
    cfg.terrain_generator.add_lights = False
    cfg.terrain_generator.seed = seed
    cfg.terrain_curriculum = TerrainCurriculumCfg(enabled=curriculum_enabled, seed=seed)
    return cfg


def test_terrain_spawn_attached_when_rough():
    from unilab.envs.locomotion.go2.rough import Go2JoystickRoughEnv

    cfg = _rough_cfg()
    env = Go2JoystickRoughEnv(cfg, num_envs=4, backend_type="mujoco")
    try:
        assert isinstance(env._spawn, TerrainSpawnManager)
        assert env._scene_terrain_origins is not None
        assert env._scene_terrain_origins.shape == (3, 3, 3)
    finally:
        env.close()


def test_default_spawn_used_when_flat():
    from unilab.envs.locomotion.go2.joystick import (
        Go2JoystickCfg,
        Go2WalkTask,
        RewardConfig,
    )

    cfg = Go2JoystickCfg(
        reward_config=RewardConfig(scales={}, tracking_sigma=0.25, base_height_target=0.3)
    )
    env = Go2WalkTask(cfg, num_envs=4, backend_type="mujoco")
    try:
        assert type(env._spawn) is BaseSpawnManager
        assert env._scene_terrain_origins is None
        # Origins are zeros (flat scene needs no spread; per-env xy jitter still applies).
        np.testing.assert_array_equal(env._spawn.origins_for(np.arange(4)), np.zeros((4, 3)))
    finally:
        env.close()


def test_curriculum_disabled_distributes_levels_uniformly():
    from unilab.envs.locomotion.go2.rough import Go2JoystickRoughEnv

    cfg = _rough_cfg(curriculum_enabled=False, seed=0)
    env = Go2JoystickRoughEnv(cfg, num_envs=64, backend_type="mujoco")
    try:
        sm = env._spawn
        assert sm is not None
        assert sm.levels.min() == 0
        assert sm.levels.max() == 2
        assert sm.type_cols.min() >= 0
        assert sm.type_cols.max() <= 2
    finally:
        env.close()


def test_curriculum_enabled_levels_start_at_zero():
    from unilab.envs.locomotion.go2.rough import Go2JoystickRoughEnv

    cfg = _rough_cfg(curriculum_enabled=True, seed=0)
    env = Go2JoystickRoughEnv(cfg, num_envs=8, backend_type="mujoco")
    try:
        sm = env._spawn
        assert sm is not None
        assert np.all(sm.levels == 0)
    finally:
        env.close()


def test_reset_qpos_xy_matches_terrain_origins():
    from unilab.envs.locomotion.go2.rough import Go2JoystickRoughEnv

    cfg = _rough_cfg(curriculum_enabled=False, seed=0)
    env = Go2JoystickRoughEnv(cfg, num_envs=4, backend_type="mujoco")
    try:
        sm = env._spawn
        assert sm is not None
        env.init_state()
        base_pos = env._backend.get_base_pos()
        rows = sm.levels
        cols = sm.type_cols
        expected_xy = sm._terrain_origins[rows, cols, :2]
        # Reset adds a uniform [-0.5, 0.5] xy jitter on top of the spawn xy.
        diff = base_pos[:, :2] - expected_xy
        assert np.all(np.abs(diff) < 0.6)
    finally:
        env.close()


def test_rough_reset_spawns_upright_on_terrain():
    from unilab.envs.locomotion.go2.rough import Go2JoystickRoughEnv

    cfg = _rough_cfg(curriculum_enabled=False, seed=0)
    env = Go2JoystickRoughEnv(cfg, num_envs=64, backend_type="mujoco")
    try:
        env.init_state()
        upvector = env._backend.get_sensor_data("upvector")
        assert np.all(upvector[:, 2] > 0.9)
    finally:
        env.close()


def test_rough_reset_spawns_above_sampled_terrain():
    from unilab.envs.locomotion.go2.rough import Go2JoystickRoughEnv

    cfg = _rough_cfg(curriculum_enabled=False, seed=0)
    env = Go2JoystickRoughEnv(cfg, num_envs=64, backend_type="mujoco")
    try:
        env.init_state()
        raw_heights, _ = env._raw_height_scan_obs(env.num_envs)
        assert raw_heights is not None
        center_heights = raw_heights[:, raw_heights.shape[1] // 2]
        clearance = env._backend.get_base_pos()[:, 2] - center_heights
        assert np.all(clearance > 0.25)
    finally:
        env.close()


def test_curriculum_logs_appear_after_done():
    from unilab.envs.locomotion.go2.rough import Go2JoystickRoughEnv

    cfg = _rough_cfg(curriculum_enabled=True, seed=0)
    env = Go2JoystickRoughEnv(cfg, num_envs=4, backend_type="mujoco")
    try:
        state = env.init_state()
        env.apply_action(np.zeros((4, 12), dtype=np.float32), state)
        state.truncated[:] = True
        out = env.update_state(state)
        log = out.info.get("log", {})
        for key in (
            "terrain_curriculum/mean_level",
            "terrain_curriculum/max_level",
            "terrain_curriculum/mean_walked",
            "terrain_curriculum/num_promoted",
            "terrain_curriculum/num_demoted",
            "terrain_curriculum/num_skipped",
        ):
            assert key in log
    finally:
        env.close()


def test_reset_uses_spawn_manager_origins(monkeypatch):
    """When terrain present, the reset path must call spawn_manager.origins_for,
    not env._env_origins (verifies dr_provider branching)."""
    from unilab.envs.locomotion.go2.rough import Go2JoystickRoughEnv

    cfg = _rough_cfg(curriculum_enabled=False, seed=0)
    env = Go2JoystickRoughEnv(cfg, num_envs=4, backend_type="mujoco")
    try:
        sm = env._spawn
        assert sm is not None
        called: list[np.ndarray] = []
        original = sm.origins_for

        def spy(env_ids: np.ndarray) -> np.ndarray:
            called.append(env_ids.copy())
            return original(env_ids)

        monkeypatch.setattr(sm, "origins_for", spy)
        env.init_state()
        assert len(called) >= 1
        np.testing.assert_array_equal(called[0], np.arange(4))
    finally:
        env.close()


@pytest.mark.parametrize("preset", ["flat", "rough"])
def test_episode_start_recorded_after_reset(preset):
    from unilab.envs.locomotion.go2.joystick import (
        Go2JoystickCfg,
        Go2WalkTask,
        RewardConfig,
    )
    from unilab.envs.locomotion.go2.rough import Go2JoystickRoughEnv

    if preset == "flat":
        cfg = Go2JoystickCfg(
            reward_config=RewardConfig(scales={}, tracking_sigma=0.25, base_height_target=0.3)
        )
        env_cls = Go2WalkTask
    else:
        cfg = _rough_cfg(curriculum_enabled=False, seed=0)
        env_cls = Go2JoystickRoughEnv
    env = env_cls(cfg, num_envs=4, backend_type="mujoco")
    try:
        env.init_state()
        sm = env._spawn
        if isinstance(sm, TerrainSpawnManager):
            assert np.all(sm._has_started)
    finally:
        env.close()
