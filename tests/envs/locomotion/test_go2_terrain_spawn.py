"""Integration tests for Go2 + TerrainSpawnManager."""

from __future__ import annotations

import numpy as np
import pytest

from unilab.envs.locomotion.common.terrain_spawn import (
    TerrainCurriculumCfg,
    TerrainSpawnManager,
)


def _class_path(obj) -> str:
    cls = type(obj)
    return f"{cls.__module__}.{cls.__name__}"


def _rough_cfg(*, curriculum_enabled: bool = False, seed: int = 0):
    from unilab.envs.locomotion.go2.joystick import RewardConfig
    from unilab.envs.locomotion.go2.rough import Go2JoystickRoughCfg

    cfg = Go2JoystickRoughCfg(
        reward_config=RewardConfig(scales={}, tracking_sigma=0.25, base_height_target=0.3)
    )
    cfg.scene.terrain.generator.num_rows = 3
    cfg.scene.terrain.generator.num_cols = 3
    cfg.scene.terrain.generator.border_width = 0.0
    cfg.scene.terrain.generator.add_lights = False
    cfg.scene.terrain.generator.seed = seed
    cfg.terrain_curriculum = TerrainCurriculumCfg(enabled=curriculum_enabled, seed=seed)
    return cfg


def test_terrain_spawn_attached_when_rough():
    from unilab.envs.locomotion.go2.rough import Go2JoystickRoughEnv

    cfg = _rough_cfg()
    env = Go2JoystickRoughEnv(cfg, num_envs=4, backend_type="mujoco")
    try:
        assert _class_path(env._spawn) == (
            "unilab.envs.locomotion.common.terrain_spawn.TerrainSpawnManager"
        )
        assert env._scene_terrain_origins is not None
        assert env._scene_terrain_origins.shape == (3, 3, 3)
    finally:
        env.close()


def test_terrain_spawn_attached_when_rough_motrix():
    pytest.importorskip("motrixsim")

    from unilab.envs.locomotion.go2.joystick import Go2WalkTask

    cfg = _rough_cfg()
    cfg.domain_rand.randomize_kp = False
    cfg.domain_rand.randomize_kd = False
    env = Go2WalkTask(cfg, num_envs=2, backend_type="motrix")
    try:
        assert _class_path(env._spawn) == (
            "unilab.envs.locomotion.common.terrain_spawn.TerrainSpawnManager"
        )
        assert env._scene_terrain_origins is not None
        assert env._scene_terrain_origins.shape == (3, 3, 3)
        state = env.init_state()
        assert state.obs["obs"].shape == (2, 49)
    finally:
        env.close()


def test_terrain_spawn_samples_height_after_xy_jitter():
    class FakeSurface:
        def sample_height(self, xy):
            xy = np.asarray(xy, dtype=np.float64)
            return xy[:, 0] * 0.5 + xy[:, 1] * 0.25

    origins = np.zeros((1, 1, 3), dtype=np.float64)
    cfg = TerrainCurriculumCfg(spawn_height_margin=0.05)
    spawn = TerrainSpawnManager(
        1,
        origins,
        cell_size=1.0,
        cfg=cfg,
        terrain_surface_sampler=FakeSurface(),
    )

    qpos_xyz = np.asarray([[0.2, 0.4, 0.42]], dtype=np.float64)
    spawned = spawn.apply_spawn(np.asarray([0], dtype=np.int32), qpos_xyz)

    assert spawned[0, 0] == pytest.approx(0.2)
    assert spawned[0, 1] == pytest.approx(0.4)
    assert spawned[0, 2] == pytest.approx(0.2 * 0.5 + 0.4 * 0.25 + 0.42 + 0.05)


def test_go2_rough_base_height_reward_uses_terrain_relative_height():
    from unilab.envs.locomotion.go2.joystick import Go2WalkTask

    class FakeBackend:
        def get_base_pos(self):
            return np.asarray(
                [
                    [1.0, 2.0, 1.25],
                    [-1.0, 0.5, -0.2],
                ],
                dtype=np.float32,
            )

    class FakeSurface:
        def sample_height(self, xy):
            xy = np.asarray(xy, dtype=np.float64)
            return np.asarray([0.75, -0.55], dtype=np.float64)

    surface = FakeSurface()
    env = Go2WalkTask.__new__(Go2WalkTask)
    env._backend = FakeBackend()
    env._terrain_surface_sample_height = surface.sample_height

    np.testing.assert_allclose(env._reward_base_height_values(), np.asarray([0.5, 0.35]))


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
        assert _class_path(env._spawn) == (
            "unilab.envs.locomotion.common.terrain_spawn.BaseSpawnManager"
        )
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


def test_reset_uses_spawn_manager_apply_spawn(monkeypatch):
    """When terrain present, reset must go through the spawn manager after xy jitter."""
    from unilab.envs.locomotion.go2.rough import Go2JoystickRoughEnv

    cfg = _rough_cfg(curriculum_enabled=False, seed=0)
    env = Go2JoystickRoughEnv(cfg, num_envs=4, backend_type="mujoco")
    try:
        sm = env._spawn
        assert sm is not None
        called: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]] = []
        original = sm.apply_spawn

        def spy(env_ids: np.ndarray, qpos_xyz: np.ndarray, *, yaw=None) -> np.ndarray:
            called.append((env_ids.copy(), qpos_xyz.copy(), None if yaw is None else yaw.copy()))
            return original(env_ids, qpos_xyz, yaw=yaw)

        monkeypatch.setattr(sm, "apply_spawn", spy)
        env.init_state()
        assert len(called) >= 1
        env_ids, qpos_xyz, yaw = called[0]
        np.testing.assert_array_equal(env_ids, np.arange(4))
        assert qpos_xyz.shape == (4, 3)
        assert yaw is not None
        assert yaw.shape == (4,)
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
        if _class_path(sm) == "unilab.envs.locomotion.common.terrain_spawn.TerrainSpawnManager":
            assert np.all(sm._has_started)
    finally:
        env.close()
