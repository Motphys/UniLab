from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from unilab.base import registry
from unilab.base.registry import ensure_registries
from unilab.training.backend_adapter import BackendAdapter

ROOT_DIR = Path(__file__).parents[4]
CONF_DIR = ROOT_DIR / "conf"


_G1_OWNER_CASES = [
    {
        "id": "ppo_mujoco",
        "config_group": "ppo",
        "overrides": ["task=g1_walk_flat/mujoco"],
        "task_name": "G1WalkFlat",
        "backend": "mujoco",
        "profile": "legacy",
        "action_scale": 0.25,
        "curriculum_enabled": False,
    },
    {
        "id": "ppo_motrix",
        "config_group": "ppo",
        "overrides": ["task=g1_walk_flat/motrix"],
        "task_name": "G1WalkFlat",
        "backend": "motrix",
        "profile": "legacy",
        "action_scale": 0.5,
        "curriculum_enabled": False,
    },
    {
        "id": "appo_mujoco",
        "config_group": "appo",
        "overrides": ["task=g1_walk_flat/mujoco"],
        "task_name": "G1WalkFlat",
        "backend": "mujoco",
        "profile": "legacy",
        "action_scale": 0.25,
        "curriculum_enabled": False,
    },
    {
        "id": "sac_mujoco",
        "config_group": "offpolicy",
        "overrides": ["algo=sac", "task=sac/g1_walk_flat/mujoco"],
        "task_name": "G1WalkFlat",
        "backend": "mujoco",
        "profile": "walk",
        "action_scale": 1.0,
        "curriculum_enabled": True,
    },
    {
        "id": "sac_motrix",
        "config_group": "offpolicy",
        "overrides": ["algo=sac", "task=sac/g1_walk_flat/motrix"],
        "task_name": "G1WalkFlat",
        "backend": "motrix",
        "profile": "walk",
        "action_scale": 1.0,
        "curriculum_enabled": True,
    },
    {
        "id": "sac_rough",
        "config_group": "offpolicy",
        "overrides": ["algo=sac", "task=sac/g1_walk_rough/mujoco"],
        "task_name": "G1WalkRough",
        "backend": "mujoco",
        "profile": "walk",
        "action_scale": 1.0,
        "curriculum_enabled": True,
        "model_suffix": "scene_rough.xml",
    },
    {
        "id": "td3_mujoco",
        "config_group": "offpolicy",
        "overrides": ["algo=td3", "task=td3/g1_walk_flat/mujoco"],
        "task_name": "G1WalkFlat",
        "backend": "mujoco",
        "profile": "walk",
        "action_scale": 1.0,
        "curriculum_enabled": True,
    },
    {
        "id": "flashsac_walk_mujoco",
        "config_group": "offpolicy",
        "overrides": ["algo=flashsac", "task=flashsac/g1_walk_flat/mujoco"],
        "task_name": "G1WalkFlat",
        "backend": "mujoco",
        "profile": "walk",
        "action_scale": 1.0,
        "curriculum_enabled": True,
    },
]


def _compose_cfg(config_group: str, overrides: list[str]):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / config_group), version_base="1.3"):
        return compose("config", overrides=overrides)


def _materialize_env_cfg(cfg: Any):
    from unilab.envs.locomotion.g1.joystick import G1WalkFlatCfg, G1WalkRoughCfg

    env_cfg_cls = G1WalkRoughCfg if cfg.training.task_name == "G1WalkRough" else G1WalkFlatCfg
    return OmegaConf.merge(OmegaConf.structured(env_cfg_cls()), cfg.env)


def _build_probe_env(cfg: Any):
    from unilab.envs.locomotion.g1.joystick import G1WalkEnv

    env = cast(Any, object.__new__(G1WalkEnv))
    env._num_envs = 1
    env._cfg = _materialize_env_cfg(cfg)
    env._reward_cfg = cfg.reward
    env.default_angles = np.zeros((1, 29), dtype=np.float32)
    env._obs_noise = lambda data, scale: np.asarray(data + 100.0, dtype=np.float32)
    return env


def _compute_probe_obs(cfg: Any) -> dict[str, np.ndarray]:
    env = _build_probe_env(cfg)
    return cast(
        dict[str, np.ndarray],
        env._compute_obs(
            {
                "commands": np.array([[0.7, 0.0, 0.2]], dtype=np.float32),
                "current_actions": np.zeros((1, 29), dtype=np.float32),
                "gait_phase": np.array([[0.3, 3.4]], dtype=np.float32),
            },
            linvel=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            gyro=np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
            gravity=np.array([[0.1, 0.2, 0.9]], dtype=np.float32),
            dof_pos=np.zeros((1, 29), dtype=np.float32),
            dof_vel=np.array([np.arange(7.0, 36.0, dtype=np.float32)], dtype=np.float32),
        ),
    )


@pytest.mark.parametrize("case", _G1_OWNER_CASES, ids=[case["id"] for case in _G1_OWNER_CASES])
def test_g1_owner_yaml_regression_contract(case: dict[str, Any]):
    from unilab.envs.locomotion.g1.joystick import G1WalkEnv

    cfg = _compose_cfg(case["config_group"], case["overrides"])
    full_env_cfg = _materialize_env_cfg(cfg)
    env = _build_probe_env(cfg)
    env_cfg_override = BackendAdapter(
        cfg, root_dir=ROOT_DIR, algo_name=cfg.algo.algo if "algo" in cfg.algo else None
    ).build_task_env_cfg_override()

    assert cfg.training.task_name == case["task_name"]
    assert cfg.training.sim_backend == case["backend"]
    assert full_env_cfg.control_config.action_scale == pytest.approx(case["action_scale"])
    assert full_env_cfg.curriculum.enabled is case["curriculum_enabled"]
    assert env._uses_walk_observation_profile() is (case["profile"] == "walk")
    assert (
        registry._envs[cfg.training.task_name].env_cls_dict[cfg.training.sim_backend] is G1WalkEnv
    )

    if "model_suffix" in case:
        assert str(full_env_cfg.model_file).endswith(case["model_suffix"])

    reward_config = OmegaConf.to_container(cfg.reward, resolve=True)
    assert env_cfg_override["reward_config"] == reward_config
    env_override = cast(dict[str, Any], OmegaConf.to_container(cfg.env, resolve=True))
    for key, value in env_override.items():
        assert env_cfg_override[key] == value

    env._reward_fns = {}
    env._init_reward_functions()
    for reward_name in cfg.reward.scales.keys():
        assert reward_name in env._reward_fns


@pytest.mark.parametrize("case", _G1_OWNER_CASES, ids=[case["id"] for case in _G1_OWNER_CASES])
def test_g1_owner_yaml_observation_profiles_match_expected_family(case: dict[str, Any]):
    cfg = _compose_cfg(case["config_group"], case["overrides"])
    obs = _compute_probe_obs(cfg)

    if case["profile"] == "legacy":
        np.testing.assert_allclose(obs["obs"][:, :3], [[104.0, 105.0, 106.0]])
        np.testing.assert_allclose(obs["obs"][:, 35:37], [[107.0, 108.0]])
        np.testing.assert_allclose(obs["critic"][:, :3], [[4.0, 5.0, 6.0]])
        np.testing.assert_allclose(obs["critic"][:, 35:37], [[7.0, 8.0]])
        np.testing.assert_allclose(obs["critic"][:, 98:101], [[1.0, 2.0, 3.0]])
    else:
        np.testing.assert_allclose(obs["obs"][:, :3], [[26.0, 26.25, 26.5]])
        np.testing.assert_allclose(obs["obs"][:, 35:37], [[5.35, 5.4]])
        np.testing.assert_allclose(obs["critic"][:, :3], [[1.0, 1.25, 1.5]])
        np.testing.assert_allclose(obs["critic"][:, 35:37], [[0.35, 0.4]])
        np.testing.assert_allclose(obs["critic"][:, 98:101], [[2.0, 4.0, 6.0]])


def test_g1_observation_profile_selection_prefers_reward_family_over_curriculum_flag():
    from unilab.envs.locomotion.g1.joystick import G1WalkEnv

    env = cast(Any, object.__new__(G1WalkEnv))

    env._cfg = cast(
        Any,
        type(
            "Cfg",
            (),
            {"curriculum": type("Curriculum", (), {"enabled": True})(), "reward_config": None},
        )(),
    )
    env._reward_cfg = cast(
        Any,
        type("RewardCfg", (), {"scales": {"orientation": -2.5, "ang_vel_xy": -0.2}})(),
    )
    assert env._uses_walk_observation_profile() is False

    env._cfg = cast(
        Any,
        type(
            "Cfg",
            (),
            {"curriculum": type("Curriculum", (), {"enabled": False})(), "reward_config": None},
        )(),
    )
    env._reward_cfg = cast(
        Any,
        type(
            "RewardCfg",
            (),
            {"scales": {"penalty_orientation": -10.0, "penalty_ang_vel_xy": -1.0, "alive": 10.0}},
        )(),
    )
    assert env._uses_walk_observation_profile() is True


def test_g1_walk_tasks_are_registered():
    ensure_registries()

    assert registry.contains("G1WalkFlat")
    assert registry.contains("G1WalkRough")
