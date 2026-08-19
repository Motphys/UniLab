"""Production contracts for the Manager-Based quadruped rough family."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from unilab.base import registry
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from unilab.tasks.locomotion.common.rough_manager_terms import (
    QuadrupedRoughTerrainCfg,
    RoughHeightScan,
    RoughJointPositionAction,
    RoughTerrainCurriculum,
    RoughTerrainOutOfBounds,
    RoughTerrainReset,
)
from unilab.tasks.locomotion.go2w.manager_terms import Go2WMixedAction
from unilab.training.backend_adapter import BackendAdapter
from unilab.training.sim2sim import extract_contract_snapshot

ROOT_DIR = Path(__file__).parents[3]
CONF_DIR = ROOT_DIR / "conf" / "ppo"

_OWNER_CASES = tuple(
    pytest.param(task_id, task_name, backend, *dims, id=f"{task_id.split('_')[0]}-{backend}")
    for task_id, task_name, dims in (
        ("go1_joystick_rough", "Go1JoystickRough", (45, 235, 12)),
        ("go2_joystick_rough", "Go2JoystickRough", (45, 235, 12)),
        ("go2w_joystick_rough", "Go2WJoystickRough", (53, 243, 16)),
    )
    for backend in ("mujoco", "motrix")
)


def _compose(task_id: str, backend: str, extra: tuple[str, ...] = ()) -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose("config", overrides=[f"task={task_id}/{backend}", *extra])


def _materialize(
    task_id: str,
    task_name: str,
    backend: str,
    extra: tuple[str, ...] = (),
) -> tuple[DictConfig, ManagerBasedRlEnvCfg, dict[str, Any]]:
    hydra_cfg = _compose(task_id, backend, extra)
    override = BackendAdapter(hydra_cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    env_cfg = registry.materialize_env_config(task_name)
    assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(env_cfg, override)
    env_cfg.validate()
    return hydra_cfg, env_cfg, override


@pytest.mark.parametrize("task_id,task_name,backend,policy_dim,critic_dim,action_dim", _OWNER_CASES)
def test_rough_registry_executes_real_backend_contract(
    task_id: str,
    task_name: str,
    backend: str,
    policy_dim: int,
    critic_dim: int,
    action_dim: int,
) -> None:
    if backend == "motrix":
        pytest.importorskip("motrixsim")
    registry.ensure_registries()
    hydra_cfg, env_cfg, override = _materialize(task_id, task_name, backend)
    assert (hydra_cfg.training.task_name, hydra_cfg.training.sim_backend) == (task_name, backend)
    assert hydra_cfg.env.commands.twist.planar_dead_zone == pytest.approx(0.08)
    override["commands"]["twist"]["ranges"]["lin_vel_x"] = [0.04, 0.04]
    override["commands"]["twist"]["ranges"]["lin_vel_y"] = [0.0, 0.0]
    assert env_cfg.scene is not None and env_cfg.scene.terrain is not None
    terrain = env_cfg.scene.terrain.generator
    assert isinstance(terrain, QuadrupedRoughTerrainCfg)
    assert (terrain.num_rows, terrain.num_cols, len(terrain.sub_terrains)) == (6, 6, 7)
    assert terrain.horizontal_scale == pytest.approx(0.1 if "go2w" in task_id else 0.2)
    assert registry.list_registered_envs()[task_name]["config_factory"] == "ManagerBasedRlEnvCfg"
    env = cast(
        ManagerBasedRlEnv,
        registry.make(
            task_name,
            sim_backend=backend,
            env_cfg_override=override,
            num_envs=2,
        ),
    )
    try:
        obs, _ = env.reset(seed=7)
        np.testing.assert_allclose(env.command_manager.get_command("twist")[:, :2], 0.0)
        assert obs["obs"].shape == (2, policy_dim)
        assert obs["critic"].shape == (2, critic_dim)
        assert env.obs_groups_spec == {"obs": policy_dim, "critic": critic_dim}
        assert env.action_space.shape == (action_dim,)
        np.testing.assert_array_equal(env.scene.env_origins, np.zeros((2, 3)))

        terrain_data = env._backend.get_terrain_spawn_data()
        assert terrain_data is not None
        assert terrain_data.sample_height is not None
        assert terrain_data.terrain_origins.shape == (6, 6, 3)
        reset_term = env.event_manager.get_term_cfg("terrain_root_state").func
        assert isinstance(reset_term, RoughTerrainReset)
        spawn = reset_term.spawn_manager
        ids = np.arange(2, dtype=np.int32)
        base_pos = env.scene["robot"].data.root_link_pos_w
        assigned_origins = spawn.origins_for(ids)
        assert np.all(np.abs(base_pos[:, :2] - assigned_origins[:, :2]) <= 0.5)
        surface_height = terrain_data.sample_height(base_pos[:, :2])
        assert np.all(base_pos[:, 2] - surface_height > 0.25)

        action = env.action_manager.get_term("motor" if "go2w" in task_id else "joint_pos")
        saturated = np.full((2, action_dim), 200.0, dtype=np.float32)
        action.process_actions(saturated)
        np.testing.assert_allclose(action.raw_action, 100.0)
        if isinstance(action, RoughJointPositionAction):
            scale = np.asarray(action.scale)[0]
            for name, value in zip(action.target_names, scale, strict=True):
                expected = 0.125 if "_hip_joint" in name else 0.25
                assert value == pytest.approx(expected)
            assert action.cfg.clip_actions == pytest.approx(100.0)
        else:
            assert isinstance(action, Go2WMixedAction)
            assert action.cfg.leg_action_scale == pytest.approx(0.25)
            assert action.cfg.hip_action_scale == pytest.approx(0.125)
            assert action.cfg.wheel_action_scale == pytest.approx(5.0)
            assert action.cfg.clip_actions == pytest.approx(100.0)

        state = env.step(np.zeros((2, action_dim), dtype=np.float32))
        assert state.obs["obs"].shape == (2, policy_dim)
        assert state.obs["critic"].shape == (2, critic_dim)
        for value in (*state.obs.values(), state.reward):
            assert np.isfinite(value).all()
    finally:
        env.close()


@pytest.mark.parametrize(
    ("task_id", "task_name"),
    (
        ("go1_joystick_rough", "Go1JoystickRough"),
        ("go2_joystick_rough", "Go2JoystickRough"),
        ("go2w_joystick_rough", "Go2WJoystickRough"),
    ),
)
def test_rough_sim2sim_snapshot_matches_across_backends(task_id: str, task_name: str) -> None:
    registry.ensure_registries()
    mujoco_cfg, _, _ = _materialize(task_id, task_name, "mujoco")
    motrix_cfg, _, _ = _materialize(task_id, task_name, "motrix")
    assert extract_contract_snapshot(mujoco_cfg) == extract_contract_snapshot(motrix_cfg)


def test_curriculum_updates_before_next_spawn_and_oob_fails_closed() -> None:
    registry.ensure_registries()
    extra = (
        "env.scene.terrain.generator.curriculum=true",
        "env.scene.terrain.generator.num_rows=3",
    )
    _, _, override = _materialize("go2_joystick_rough", "Go2JoystickRough", "mujoco", extra)
    env = cast(
        ManagerBasedRlEnv,
        registry.make(
            "Go2JoystickRough",
            sim_backend="mujoco",
            env_cfg_override=override,
            num_envs=2,
        ),
    )
    try:
        env.reset(seed=7)
        reset_term = env.event_manager.get_term_cfg("terrain_root_state").func
        assert isinstance(reset_term, RoughTerrainReset)
        spawn = reset_term.spawn_manager
        np.testing.assert_array_equal(spawn.levels, np.zeros(2, dtype=np.int32))

        current = env.scene["robot"].data.root_link_pos_w.copy()
        spawn._episode_start_xyz[0, :2] = current[0, :2] - np.asarray([5.0, 0.0])
        env.reset_buf[0] = True
        _, info = env.reset(env_ids=np.asarray([0], dtype=np.int32))

        assert spawn.levels[0] == 1
        assert info["log"]["Curriculum/terrain_levels/num_promoted"] == 1
        next_pos = env.scene["robot"].data.root_link_pos_w[0]
        next_origin = spawn.origins_for(np.asarray([0], dtype=np.int32))[0]
        assert np.all(np.abs(next_pos[:2] - next_origin[:2]) <= 0.5)
        np.testing.assert_allclose(spawn._episode_start_xyz[0], next_pos)

        oob = env.termination_manager.get_term_cfg("terrain_out_of_bounds").func
        assert isinstance(oob, RoughTerrainOutOfBounds)
        oob._asset = SimpleNamespace(
            data=SimpleNamespace(
                root_link_pos_w=np.asarray([[0.0, 0.0, 0.5], [oob._half_width + 1.0, 0.0, 0.5]])
            )
        )
        np.testing.assert_array_equal(oob(env), np.asarray([False, True]))
    finally:
        env.close()


def test_rough_hot_paths_use_only_cached_runtime_objects() -> None:
    for term in (
        RoughTerrainReset,
        RoughTerrainCurriculum,
        RoughTerrainOutOfBounds,
        RoughHeightScan,
    ):
        source = inspect.getsource(term.__call__)
        for forbidden in (
            "ASSETS_ROOT_PATH",
            "model_file",
            "getattr(",
            "hasattr(",
            "._backend",
        ):
            assert forbidden not in source
