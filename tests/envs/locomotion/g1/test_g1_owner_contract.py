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
from unilab.envs.locomotion.g1 import terms as g1_terms
from unilab.envs.locomotion.g1.joystick import G1WalkRewardConfig
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
        "id": "ppo_mjwarp",
        "config_group": "ppo",
        "overrides": ["task=g1_walk_flat/mjwarp"],
        "task_name": "G1WalkFlat",
        "backend": "mjwarp",
        "profile": "legacy",
        "action_scale": 0.25,
        "curriculum_enabled": False,
    },
    {
        "id": "sac_mjwarp",
        "config_group": "offpolicy",
        "overrides": ["algo=sac", "task=sac/g1_walk_flat/mjwarp"],
        "task_name": "G1WalkFlat",
        "backend": "mjwarp",
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
        assert full_env_cfg.scene.model_file.endswith(case["model_suffix"])

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


def test_g1_flat_term_plan_matches_legacy_numpy_and_reuses_outputs():
    cfg = _compose_cfg("ppo", ["task=g1_walk_flat/mujoco"])
    override = BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="ppo").build_task_env_cfg_override()
    env = cast(
        Any,
        registry.make("G1WalkFlat", sim_backend="mujoco", num_envs=2, env_cfg_override=override),
    )
    try:
        env._cfg.noise_config.level = 0.0
        rng = np.random.default_rng(7)
        info = {
            "steps": np.zeros(2, dtype=np.uint32),
            "commands": rng.normal(size=(2, 3)).astype(np.float32),
            "current_actions": rng.normal(size=(2, 29)).astype(np.float32),
            "last_actions": rng.normal(size=(2, 29)).astype(np.float32),
            "gait_phase": rng.uniform(0, 2 * np.pi, size=(2, 2)).astype(np.float32),
        }
        linvel = rng.normal(size=(2, 3)).astype(np.float32)
        gyro = rng.normal(size=(2, 3)).astype(np.float32)
        gravity = np.tile(np.array([0.01, -0.02, 0.99], np.float32), (2, 1))
        dof_pos = rng.normal(size=(2, 29)).astype(np.float32)
        dof_vel = rng.normal(size=(2, 29)).astype(np.float32)

        expected_reward = env._compute_reward(
            {**info, "log": {}}, linvel, gyro, gravity, dof_pos, dof_vel
        )
        expected_obs = env._compute_legacy_obs(info, linvel, gyro, gravity, dof_pos, dof_vel)
        expected_terminated = (np.arccos(gravity[:, 2]) > np.deg2rad(25.0)) | (
            env._backend.get_base_pos()[:, 2] < 0.55
        )
        obs, reward, terminated = env._compute_term_state(
            {**info, "log": {}}, linvel, gyro, gravity, dof_pos, dof_vel
        )
        output_ids = (id(obs["obs"]), id(obs["critic"]), id(reward), id(terminated))

        np.testing.assert_allclose(reward, expected_reward, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(obs["obs"], expected_obs["obs"], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(obs["critic"], expected_obs["critic"], rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(terminated, expected_terminated)
        repeated = env._compute_term_state(
            {**info, "log": {}}, linvel, gyro, gravity, dof_pos, dof_vel
        )
        assert tuple(id(value) for value in (*repeated[0].values(), *repeated[1:])) == output_ids
    finally:
        env.close()


def test_g1_flat_term_config_controls_layout_and_reward_scale():
    cfg = _compose_cfg("ppo", ["task=g1_walk_flat/mujoco"])
    cfg.env.term_plan.observations.obs = list(reversed(cfg.env.term_plan.observations.obs[:-1]))
    override = BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="ppo").build_task_env_cfg_override()
    env = cast(
        Any,
        registry.make("G1WalkFlat", sim_backend="mujoco", num_envs=1, env_cfg_override=override),
    )
    try:
        assert env.obs_groups_spec == {"obs": 96, "critic": 101}
        assert [name for name, _ in env.get_symmetry_obs_layouts()["obs"]] == [
            "command",
            "actions",
            "dof_vel",
            "dof_pos",
            "gravity",
            "gyro",
        ]
        assert "feet_phase" in dict(env._active_term_rewards)
        env._reward_cfg.scales["feet_phase"] = 0.0
        env._sync_term_reward_scales()
        assert "feet_phase" not in dict(env._active_term_rewards)
        env._reward_cfg.scales["feet_phase"] = 1.0
        env._sync_term_reward_scales()
        assert "feet_phase" in dict(env._active_term_rewards)
    finally:
        env.close()


def test_g1_flat_term_plan_rejects_unknown_nonzero_reward():
    cfg = _compose_cfg("ppo", ["task=g1_walk_flat/mujoco"])
    reward_cfg = G1WalkRewardConfig(**OmegaConf.to_container(cfg.reward, resolve=True))
    reward_cfg.scales["custom"] = 1.0
    with pytest.raises(g1_terms.TermPlanError, match="unknown nonzero reward"):
        g1_terms.resolve_g1_walk_term_plan(
            num_action=29,
            reward_cfg=reward_cfg,
            observations=g1_terms.DEFAULT_OBSERVATION_TERMS,
            terminations=g1_terms.DEFAULT_TERMINATION_TERMS,
            walk_profile=False,
        )
