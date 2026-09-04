"""End-to-end evidence for the pinned mjlab task migration fixture."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from tests.fixtures.mjlab_cartpole import FIXTURE_ENV_NAME
from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

ROOT_DIR = Path(__file__).parents[2]
FIXTURE_DIR = ROOT_DIR / "tests" / "fixtures" / "mjlab_cartpole"


def _materialize() -> tuple[ManagerBasedRlEnvCfg, dict[str, Any]]:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(FIXTURE_DIR / "conf"), version_base="1.3"):
        hydra_cfg: DictConfig = compose("config")
    override = BackendAdapter(hydra_cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    env_cfg = registry.materialize_env_config(FIXTURE_ENV_NAME)
    assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(env_cfg, override)
    env_cfg.validate()
    return env_cfg, override


def _assert_plain(value: Any) -> None:
    assert not OmegaConf.is_config(value)
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _assert_plain(getattr(value, field.name))
    elif isinstance(value, dict):
        for item in value.values():
            _assert_plain(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_plain(item)


def test_mjlab_fixture_hydra_materializes_source_structure() -> None:
    cfg, _ = _materialize()

    assert list(cfg.observations) == ["actor", "critic"]
    expected_obs = ["cart_pos", "pole_angle", "cart_vel", "pole_vel"]
    assert list(cfg.observations["actor"].terms) == expected_obs
    assert list(cfg.observations["critic"].terms) == expected_obs
    assert cfg.observations["actor"].enable_corruption is True
    assert cfg.observations["critic"].enable_corruption is False
    assert list(cfg.actions) == ["effort"]
    assert list(cfg.events) == ["reset_slider", "reset_hinge"]
    assert list(cfg.rewards) == ["smooth_reward"]
    assert list(cfg.terminations) == ["time_out"]
    assert cfg.policy_observation_group == "actor"
    assert cfg.critic_observation_group == "critic"
    assert cfg.sim_dt == pytest.approx(0.01)
    assert cfg.ctrl_dt == pytest.approx(0.05)
    assert cfg.max_episode_seconds == pytest.approx(50.0)
    assert cfg.scale_rewards_by_dt is True
    _assert_plain(cfg)


def test_mjlab_fixture_real_mujoco_reset_step_and_reward() -> None:
    _, override = _materialize()
    env = registry.make(
        FIXTURE_ENV_NAME,
        sim_backend="mujoco",
        env_cfg_override=override,
        num_envs=8,
    )
    assert isinstance(env, ManagerBasedRlEnv)
    try:
        state = env.init_state()
        assert env.obs_groups_spec == {"obs": 5, "critic": 5}
        assert state.obs["obs"].shape == state.obs["critic"].shape == (8, 5)
        assert env.action_space.shape == (1,)

        before = env.scene["cartpole"].data.joint_pos.copy()
        ids = np.asarray([1, 6], dtype=np.int32)
        reset_obs, _ = env.reset(env_ids=ids)
        after = env.scene["cartpole"].data.joint_pos.copy()
        np.testing.assert_array_equal(after[[0, 2, 3, 4, 5, 7]], before[[0, 2, 3, 4, 5, 7]])
        assert reset_obs["obs"].shape == reset_obs["critic"].shape == (2, 5)
        assert np.all(np.abs(after[ids, 0]) <= 0.1)
        assert np.all(np.abs(after[ids, 1]) <= 0.034)
        assert np.all(np.abs(env.scene["cartpole"].data.joint_vel[ids]) <= 0.01)

        actions = np.full((8, 1), 0.25, dtype=np.float32)
        state = env.step(actions)
        entity = env.scene["cartpole"]
        hinge = entity.data.joint_pos[:, 1]
        cart = entity.data.joint_pos[:, 0]
        hinge_vel = entity.data.joint_vel[:, 1]
        gaussian_scale = np.sqrt(-2.0 * np.log(0.1))
        quadratic_scale = np.sqrt(0.9)
        expected = (np.cos(hinge) + 1.0) / 2.0
        expected *= (1.0 + np.exp(-0.5 * np.square(cart / 2.0 * gaussian_scale))) / 2.0
        expected *= (4.0 + np.maximum(1.0 - np.square(0.25 * quadratic_scale), 0.0)) / 5.0
        expected *= (1.0 + np.exp(-0.5 * np.square(hinge_vel / 5.0 * gaussian_scale))) / 2.0
        np.testing.assert_allclose(state.reward, expected * 0.05, rtol=1e-5, atol=1e-6)
    finally:
        env.close()


def test_mjlab_fixture_missing_actuator_fails_on_cold_path() -> None:
    _, override = _materialize()
    override["actions"]["effort"]["actuator_names"] = ["missing_actuator"]
    with pytest.raises(ValueError, match="regular expressions matched.*missing_actuator"):
        registry.make(
            FIXTURE_ENV_NAME,
            sim_backend="mujoco",
            env_cfg_override=override,
            num_envs=2,
        )


def test_mjlab_fixture_is_pinned_test_only_numpy_code() -> None:
    task_source = (FIXTURE_DIR / "task.py").read_text(encoding="utf-8")
    helper_source = (ROOT_DIR / "tests/fixtures/cartpole_manager_adapters.py").read_text(
        encoding="utf-8"
    )
    executable = "\n".join(
        line
        for line in (task_source + helper_source).splitlines()
        if not line.lstrip().startswith("#")
    )
    for forbidden in (
        "import torch",
        "from torch",
        "import mjlab",
        "from mjlab",
        "uni_rl",
        "unilab.training",
    ):
        assert forbidden not in executable
    assert "0fb8a681136be94ffc636a3dd423cabb97d91f10" in (FIXTURE_DIR / "README.md").read_text(
        encoding="utf-8"
    )
    assert "tests.fixtures" not in tuple(registry._DEFAULT_REGISTRY_PACKAGES)
    assert set(registry._envs[FIXTURE_ENV_NAME].env_factory_dict) == {"mujoco"}
