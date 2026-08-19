"""End-to-end evidence for the Isaac Lab Manager-Based migration fixture."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from tests.fixtures.isaac_lab_cartpole import FIXTURE_ENV_NAME
from tests.fixtures.isaac_lab_cartpole import task as fixture
from unilab.base import registry
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.envs import ManagerBasedRlEnv, ManagerBasedRLEnvCfg, ManagerBasedRlEnvCfg
from unilab.training import BackendAdapter

ROOT_DIR = Path(__file__).parents[2]
FIXTURE_DIR = ROOT_DIR / "tests" / "fixtures" / "isaac_lab_cartpole"


def _compose_fixture() -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(FIXTURE_DIR / "conf"), version_base="1.3"):
        return compose("config")


def _materialize_fixture() -> tuple[DictConfig, ManagerBasedRlEnvCfg, dict[str, Any]]:
    hydra_cfg = _compose_fixture()
    override = BackendAdapter(hydra_cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    env_cfg = registry.materialize_env_config(FIXTURE_ENV_NAME)
    assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(env_cfg, override)
    env_cfg.validate()
    return hydra_cfg, env_cfg, override


def _assert_plain(value: Any) -> None:
    assert not OmegaConf.is_config(value)
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _assert_plain(getattr(value, field.name))
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_plain(key)
            _assert_plain(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_plain(item)


def test_fixture_hydra_owner_materializes_source_order_as_plain_manager_cfg() -> None:
    hydra_cfg, env_cfg, _ = _materialize_fixture()

    assert hydra_cfg.training.task_name == FIXTURE_ENV_NAME
    assert hydra_cfg.training.sim_backend == "mujoco"
    assert ManagerBasedRLEnvCfg is ManagerBasedRlEnvCfg
    assert list(env_cfg.observations) == ["policy"]
    assert list(env_cfg.observations["policy"].terms) == ["joint_pos_rel", "joint_vel_rel"]
    assert list(env_cfg.actions) == ["joint_effort"]
    assert list(env_cfg.events) == ["reset_cart_position", "reset_pole_position"]
    assert list(env_cfg.rewards) == [
        "alive",
        "terminating",
        "pole_pos",
        "cart_vel",
        "pole_vel",
    ]
    assert list(env_cfg.terminations) == ["time_out", "cart_out_of_bounds"]
    assert env_cfg.actions["joint_effort"].scale == pytest.approx(100.0)
    assert env_cfg.sim_dt == pytest.approx(1.0 / 120.0)
    assert env_cfg.ctrl_dt == pytest.approx(1.0 / 60.0)
    assert env_cfg.max_episode_seconds == pytest.approx(5.0)
    assert env_cfg.policy_observation_group == "policy"
    assert env_cfg.critic_observation_group is None
    _assert_plain(env_cfg)


def test_fixture_real_mujoco_reset_step_and_partial_reset() -> None:
    _, _, override = _materialize_fixture()
    env = registry.make(
        FIXTURE_ENV_NAME,
        sim_backend="mujoco",
        env_cfg_override=override,
        num_envs=8,
    )
    assert isinstance(env, ManagerBasedRlEnv)
    try:
        state = env.init_state()
        assert env.obs_groups_spec == {"obs": 4}
        assert env.action_space.shape == (1,)
        assert state.obs["obs"].shape == (8, 4)
        assert np.isfinite(state.obs["obs"]).all()

        before = env.scene["robot"].data.joint_pos.copy()
        reset_ids = np.asarray([1, 6], dtype=np.int32)
        reset_obs, _ = env.reset(env_ids=reset_ids)
        after = env.scene["robot"].data.joint_pos.copy()
        assert reset_obs["obs"].shape == (2, 4)
        np.testing.assert_array_equal(after[[0, 2, 3, 4, 5, 7]], before[[0, 2, 3, 4, 5, 7]])
        assert np.all(np.abs(after[reset_ids, 0]) <= 1.0)
        assert np.all(np.abs(after[reset_ids, 1]) <= 0.25 * np.pi)

        state = env.step(np.zeros((8, 1), dtype=np.float32))
        assert state.obs["obs"].shape == (8, 4)
        assert state.reward.shape == (8,)
        assert state.terminated.shape == (8,)
        assert state.truncated.shape == (8,)
        assert np.isfinite(state.obs["obs"]).all()
        assert np.isfinite(state.reward).all()

        robot = env.scene["robot"]
        pole_pos = robot.data.joint_pos[:, 1]
        expected = 1.0 - np.square(np.remainder(pole_pos + np.pi, 2.0 * np.pi) - np.pi)
        expected -= 0.01 * np.abs(robot.data.joint_vel[:, 0])
        expected -= 0.005 * np.abs(robot.data.joint_vel[:, 1])
        np.testing.assert_allclose(state.reward, expected * env.step_dt, rtol=1e-5, atol=1e-6)
    finally:
        env.close()


def test_fixture_missing_actuator_fails_during_cold_path_binding() -> None:
    _, _, override = _materialize_fixture()
    override["actions"]["joint_effort"]["actuator_names"] = ["missing_actuator"]

    with pytest.raises(
        ValueError,
        match="Not all entity selector regular expressions matched.*missing_actuator",
    ):
        registry.make(
            FIXTURE_ENV_NAME,
            sim_backend="mujoco",
            env_cfg_override=override,
            num_envs=2,
        )


def test_fixture_stays_test_only_and_has_no_external_runtime_imports() -> None:
    source = (FIXTURE_DIR / "task.py").read_text(encoding="utf-8")
    executable = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    for forbidden in (
        "import torch",
        "from torch",
        "import isaaclab",
        "from isaaclab",
        "unilab.ipc",
        "unilab.training",
    ):
        assert forbidden not in executable
    assert Path(fixture.__file__).is_relative_to(ROOT_DIR / "tests" / "fixtures")
    assert "tests.fixtures" not in tuple(registry._DEFAULT_REGISTRY_PACKAGES)
    assert set(registry._envs[FIXTURE_ENV_NAME].env_factory_dict) == {"mujoco"}


def test_fixture_local_term_math_matches_isaac_source() -> None:
    class _Data:
        joint_pos = np.asarray([[0.2, 3.5], [-0.3, -3.4]], dtype=np.float32)
        joint_vel = np.asarray([[0.4, -0.7], [-0.2, 0.9]], dtype=np.float32)

    class _Entity:
        data = _Data()

    class _Scene(dict):
        pass

    env = type("FixtureMathEnv", (), {"scene": _Scene(robot=_Entity())})()
    pole = fixture.SceneEntityCfg("robot", joint_ids=[1])
    slider = fixture.SceneEntityCfg("robot", joint_ids=[0])

    expected_wrapped = np.remainder(_Data.joint_pos[:, 1] + np.pi, 2.0 * np.pi) - np.pi
    np.testing.assert_allclose(
        fixture.joint_pos_target_l2(env, target=0.0, asset_cfg=pole),
        np.square(expected_wrapped),
    )
    np.testing.assert_allclose(
        fixture.joint_vel_l1(env, asset_cfg=pole),
        np.abs(_Data.joint_vel[:, 1]),
    )
    np.testing.assert_array_equal(
        fixture.joint_pos_out_of_manual_limit(env, bounds=(-0.25, 0.25), asset_cfg=slider),
        np.asarray([False, True]),
    )
