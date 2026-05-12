"""Tests for structured configs and Hydra YAML loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

CONF_DIR = Path(__file__).parent.parent.parent / "conf"


# ---------------------------------------------------------------------------
# structured_configs dataclass defaults
# ---------------------------------------------------------------------------


def test_sac_config_defaults():
    from unilab.structured_configs import SACAlgoParams, SACConfig

    cfg = SACConfig()
    assert cfg.algo == "sac"
    assert cfg.num_envs == 4096
    assert cfg.batch_size == 8192
    assert cfg.use_symmetry is False
    assert isinstance(cfg.algo_params, SACAlgoParams)
    assert cfg.algo_params.alpha_init == 0.01


def test_td3_config_defaults():
    from unilab.structured_configs import TD3Config

    cfg = TD3Config()
    assert cfg.algo == "td3"
    assert cfg.num_envs == 4096
    assert cfg.use_layer_norm is False
    assert cfg.algo_params.weight_decay == 0.1


def test_flashsac_config_defaults():
    from unilab.structured_configs import FlashSACAlgoParams, FlashSACConfig

    cfg = FlashSACConfig()
    assert cfg.algo == "flashsac"
    assert cfg.num_envs == 1024
    assert cfg.batch_size == 2048
    assert cfg.learning_starts == 98
    assert cfg.gamma == pytest.approx(0.97)
    assert cfg.obs_normalization is False
    assert isinstance(cfg.algo_params, FlashSACAlgoParams)
    assert cfg.algo_params.normalize_reward is True
    assert cfg.algo_params.use_compile is False


def test_ppo_config_defaults():
    from unilab.structured_configs import PPOConfig

    cfg = PPOConfig()
    assert cfg.algo == "ppo"
    assert cfg.max_iterations == 101
    assert cfg.algorithm.clip_param == 0.2
    assert cfg.algorithm.class_name == "unilab.algos.torch.rsl_rl_ppo:FinalObservationAwarePPO"
    assert cfg.policy.class_name == "ActorCritic"


def test_appo_config_defaults():
    from unilab.structured_configs import APPOConfig

    cfg = APPOConfig()
    assert cfg.algo == "appo"
    assert cfg.num_envs == 2048
    assert cfg.actor.class_name == "rsl_rl.models.MLPModel"


def test_base_config_to_dict():
    from unilab.structured_configs import SACConfig

    cfg = SACConfig()
    d = cfg.to_dict()
    assert isinstance(d, dict)
    assert d["algo"] == "sac"
    assert "algo_params" in d
    assert isinstance(d["algo_params"], dict)


# ---------------------------------------------------------------------------
# Hydra YAML loading — offpolicy
# ---------------------------------------------------------------------------


def test_offpolicy_sac_defaults():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "offpolicy"), version_base="1.3"):
        cfg = compose("config")
    assert cfg.algo.algo == "sac"
    assert cfg.algo.num_envs == 2048


def test_offpolicy_sac_g1_task_overrides():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "offpolicy"), version_base="1.3"):
        cfg = compose("config", overrides=["algo=sac", "task=sac/g1_walk_flat/mujoco"])
    assert cfg.algo.num_envs == 2048
    assert cfg.algo.max_iterations == 5000
    assert cfg.algo.use_symmetry is True
    assert cfg.algo.algo_params.target_entropy_ratio == pytest.approx(0.0)
    assert cfg.training.task_name == "G1WalkFlat"

    assert cfg.env.control_config.action_scale == pytest.approx(1.0)
    assert cfg.env.gait_phase_init_mode == "offset_phase"
    assert cfg.env.reset_base_qvel_limit == pytest.approx(0.5)


def test_offpolicy_td3_defaults():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "offpolicy"), version_base="1.3"):
        cfg = compose("config", overrides=["algo=td3"])
    assert cfg.algo.algo == "td3"
    assert cfg.algo.use_layer_norm is False
    assert cfg.algo.algo_params.weight_decay == pytest.approx(0.001)
    assert cfg.algo.tau == pytest.approx(0.01)
    assert cfg.algo.algo_params.policy_noise == pytest.approx(0.1)
    assert cfg.algo.algo_params.noise_clip == pytest.approx(0.2)
    assert cfg.algo.algo_params.log_std_min == pytest.approx(-5.0)


def test_offpolicy_td3_g1_task_overrides():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "offpolicy"), version_base="1.3"):
        cfg = compose("config", overrides=["algo=td3", "task=td3/g1_walk_flat/mujoco"])
    assert cfg.training.task_name == "G1WalkFlat"
    assert cfg.algo.max_iterations == 100000
    assert cfg.env.control_config.action_scale == pytest.approx(1.0)


def test_offpolicy_flashsac_g1_task_overrides():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "offpolicy"), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=["algo=flashsac", "task=flashsac/g1_walk_flat/mujoco"],
        )
    assert cfg.algo.algo == "flashsac"
    assert cfg.training.task_name == "G1WalkFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.algo.algo_params.actor_num_blocks == 2
    assert cfg.algo.algo_params.normalize_reward is True


def test_offpolicy_flashsac_go2_task_overrides():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "offpolicy"), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=["algo=flashsac", "task=flashsac/go2_joystick_flat/mujoco"],
        )
    assert cfg.algo.algo == "flashsac"
    assert cfg.training.task_name == "Go2JoystickFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.algo.num_envs == 1024
    assert cfg.algo.max_iterations == 4000
    assert cfg.algo.tau == pytest.approx(0.05)
    assert cfg.algo.replay_buffer_n == 4096
    assert cfg.algo.updates_per_step == 2
    assert cfg.reward.scales.swing_feet_z == pytest.approx(4.0)
    assert cfg.env.control_config.action_scale == pytest.approx(0.4)


def test_go2_joystick_rough_uses_terrain_generator():
    from unilab.assets import ASSETS_ROOT_PATH
    from unilab.base.scene import SceneCfg, TerrainSceneCfg
    from unilab.envs.locomotion.go2.rough import Go2JoystickRoughCfg
    from unilab.terrains import TerrainGeneratorCfg

    cfg = Go2JoystickRoughCfg()
    assert isinstance(cfg.scene, SceneCfg)
    assert isinstance(cfg.scene.terrain, TerrainSceneCfg)
    assert cfg.scene.model_file.endswith("go2.xml")
    assert cfg.scene.fragment_files == [
        str(ASSETS_ROOT_PATH / "robots" / "go2" / "locomotion_task.xml")
    ]
    assert isinstance(cfg.scene.terrain.generator, TerrainGeneratorCfg)
    assert cfg.scene.terrain.hfield_name == "terrain_hfield"
    assert cfg.scene.terrain.geom_name == "floor"
    assert len(cfg.scene.terrain.generator.sub_terrains) == 7


def test_go2_joystick_rough_terrain_cfg_is_independent_per_instance():
    """Confirm rough terrain cfg defaults are not shared across instances."""
    from unilab.envs.locomotion.go2.rough import Go2JoystickRoughCfg

    a = Go2JoystickRoughCfg()
    b = Go2JoystickRoughCfg()
    b.scene.terrain.generator.num_rows = 3
    assert a.scene.terrain.generator is not b.scene.terrain.generator
    a.scene.terrain.generator.num_rows = 4
    assert b.scene.terrain.generator.num_rows == 3


def test_go2_joystick_rough_playback_model_uses_backend_scene(tmp_path):
    """Offline playback / video rendering must reuse the backend-compiled scene model."""
    import mujoco

    from unilab.envs.locomotion.go2.joystick import RewardConfig
    from unilab.envs.locomotion.go2.rough import Go2JoystickRoughCfg, Go2JoystickRoughEnv
    from unilab.visualization.playback import _resolve_render_play_model_files

    cfg = Go2JoystickRoughCfg(
        reward_config=RewardConfig(scales={}, tracking_sigma=0.25, base_height_target=0.3)
    )
    cfg.scene.terrain.generator.num_rows = 2
    cfg.scene.terrain.generator.num_cols = 2
    cfg.scene.terrain.generator.border_width = 0.0
    cfg.scene.terrain.generator.add_lights = False
    cfg.scene.terrain.generator.seed = 0

    env = Go2JoystickRoughEnv(cfg, num_envs=2, backend_type="mujoco")
    try:
        playback_model = env.get_playback_model(0)
        assert isinstance(playback_model, mujoco.MjModel)
        assert env._backend.terrain_origins is not None
        assert (Path(env._backend.scene_artifacts_dir) / "hfields" / "hfield.png").is_file()
        assert Path(env._backend.scene_visual_model_file).is_file()
        assert mujoco.mj_name2id(playback_model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain_hfield") >= 0
        assert mujoco.mj_name2id(playback_model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0
        assert mujoco.mj_name2id(playback_model, mujoco.mjtObj.mjOBJ_SENSOR, "FL_foot_contact") >= 0
        model_file = _resolve_render_play_model_files(env, num_envs=2, tmp_dir=tmp_path)
        assert isinstance(model_file, str)
        assert model_file.endswith(".mjb")
        assert Path(model_file).is_file()
        rendered_model = mujoco.MjModel.from_binary_path(model_file)
        assert mujoco.mj_name2id(rendered_model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain_hfield") >= 0
        assert mujoco.mj_name2id(rendered_model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0
        assert rendered_model.ngeom > playback_model.ngeom
    finally:
        env.close()


def test_go2_joystick_flat_no_terrain_materialized():
    """Flat task keeps the static scene source and has no terrain origins."""
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
        assert env._backend.scene_model_file == cfg.scene.model_file
        assert env._backend.terrain_origins is None
        assert env._backend.scene_artifacts_dir is None
    finally:
        env.close()


def test_ppo_go2_joystick_rough_task_compose():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=go2_joystick_rough/mujoco"])
    assert cfg.training.task_name == "Go2JoystickRough"
    assert cfg.training.sim_backend == "mujoco"


def test_ppo_go2_joystick_rough_motrix_task_compose():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=go2_joystick_rough/motrix"])
    assert cfg.training.task_name == "Go2JoystickRough"
    assert cfg.training.sim_backend == "motrix"
    assert cfg.algo.num_envs == 4096
    assert cfg.algo.max_iterations == 2000
    assert cfg.env.render_offset_mode == "zero"
    assert cfg.env.scene.model_file.endswith("go2.xml")
    assert cfg.env.scene.terrain.generator.num_rows == 6
    assert cfg.env.scene.terrain.generator.num_cols == 6
    assert cfg.env.terrain_scan.enabled is True
    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(3.0)
    assert "base_height" not in cfg.reward.scales
    assert "swing_feet_z" not in cfg.reward.scales


def test_go2_joystick_rough_motrix_registers_rough_env():
    from unilab.base import registry
    from unilab.envs.locomotion.go2.rough import Go2JoystickRoughEnv

    assert registry._envs["Go2JoystickRough"].env_cls_dict["motrix"] is Go2JoystickRoughEnv


def test_offpolicy_g1_rough_terrain_task_overrides():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from unilab.envs.locomotion.g1.joystick import G1WalkRoughCfg

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "offpolicy"), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=["algo=sac", "task=sac/g1_walk_rough/mujoco"],
        )
    assert cfg.algo.algo == "sac"
    assert cfg.training.task_name == "G1WalkRough"
    assert cfg.training.sim_backend == "mujoco"
    assert G1WalkRoughCfg().scene.model_file.endswith("scene_rough.xml")


def test_g1_task_owner_yamls_preserve_legacy_and_walk_observation_profiles():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from unilab.envs.locomotion.g1.joystick import G1WalkEnv

    def uses_walk_profile(config_group: str, overrides: list[str]) -> bool:
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(CONF_DIR / config_group), version_base="1.3"):
            cfg = compose("config", overrides=overrides)
        env = cast(Any, object.__new__(G1WalkEnv))
        env._cfg = cfg.env
        env._reward_cfg = cfg.reward
        return bool(env._uses_walk_observation_profile())

    assert uses_walk_profile("ppo", ["task=g1_walk_flat/mujoco"]) is False
    assert uses_walk_profile("appo", ["task=g1_walk_flat/mujoco"]) is False
    assert uses_walk_profile("offpolicy", ["algo=sac", "task=sac/g1_walk_flat/mujoco"]) is True
    assert uses_walk_profile("offpolicy", ["algo=sac", "task=sac/g1_walk_flat/motrix"]) is True
    assert uses_walk_profile("offpolicy", ["algo=sac", "task=sac/g1_walk_rough/mujoco"]) is True
    assert uses_walk_profile("offpolicy", ["algo=td3", "task=td3/g1_walk_flat/mujoco"]) is True
    assert (
        uses_walk_profile("offpolicy", ["algo=flashsac", "task=flashsac/g1_walk_flat/mujoco"])
        is True
    )


# ---------------------------------------------------------------------------
# Hydra YAML loading — appo
# ---------------------------------------------------------------------------


def test_appo_defaults():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "appo"), version_base="1.3"):
        cfg = compose("config")
    assert cfg.algo.algo == "appo"
    assert cfg.algo.max_iterations == 150


def test_appo_g1_task_overrides():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "appo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=g1_walk_flat/mujoco"])
    assert cfg.algo.max_iterations == 500
    assert cfg.algo.save_interval == 100
    assert cfg.training.task_name == "G1WalkFlat"
    assert "obs_profile" not in cfg.env
    assert cfg.env.curriculum.enabled is False


# ---------------------------------------------------------------------------
# Hydra YAML loading — ppo
# ---------------------------------------------------------------------------


def test_ppo_go1_max_iterations():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=go1_joystick_flat/mujoco"])
    assert cfg.algo.max_iterations == 151
    assert "actor" in cfg.algo.obs_groups


def test_ppo_g1_num_envs():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=g1_walk_flat/mujoco"])
    assert cfg.algo.num_envs == 2048
    assert cfg.algo.max_iterations == 220
    assert cfg.training.task_name == "G1WalkFlat"
    assert "obs_profile" not in cfg.env
    assert cfg.env.curriculum.enabled is False


def test_ppo_go2_num_envs():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=go2_joystick_flat/mujoco"])
    assert cfg.algo.num_envs == 1024
    assert cfg.algo.max_iterations == 151


def test_ppo_g1_motion_tracking():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=g1_motion_tracking/mujoco"])
    assert cfg.training.task_name == "G1MotionTracking"
    assert cfg.algo.max_iterations == 15000
    assert cfg.algo.algorithm.entropy_coef == pytest.approx(0.005)


def test_ppo_g1_flip_tracking():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=g1_flip_tracking/mujoco"])
    assert cfg.training.task_name == "G1FlipTracking"
    assert cfg.algo.max_iterations == 30000


def test_ppo_g1_wall_flip_tracking():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=g1_wall_flip_tracking/mujoco"])
    assert cfg.training.task_name == "G1WallFlipTracking"
    assert cfg.algo.max_iterations == 30000


# ---------------------------------------------------------------------------
# Issue #197 DoD: rough terrain profile params overridable via Hydra
# ---------------------------------------------------------------------------


def test_apply_cfg_overrides_deep_merges_dataclass_field():
    """registry.apply_cfg_overrides must deep-merge into existing dataclass
    instances rather than re-instantiating them, so partial overrides like
    `scene.terrain.generator.num_rows=4` keep `sub_terrains` and other defaults."""
    from unilab.base.registry import apply_cfg_overrides
    from unilab.envs.locomotion.go2.rough import Go2JoystickRoughCfg

    cfg = Go2JoystickRoughCfg()
    cfg.scene.terrain.generator.num_cols = 3
    cfg.scene.terrain.generator.border_width = 2.5
    cfg.scene.terrain.generator.sub_terrains = {
        "test_flat": cfg.scene.terrain.generator.sub_terrains["flat"]
    }
    cfg.scene.terrain.generator.add_lights = False
    apply_cfg_overrides(
        cfg,
        {"scene": {"terrain": {"generator": {"num_rows": 4, "seed": 42, "curriculum": True}}}},
    )

    # Overridden fields take effect.
    assert cfg.scene.terrain.generator.num_rows == 4
    assert cfg.scene.terrain.generator.seed == 42
    assert cfg.scene.terrain.generator.curriculum is True
    # Non-overridden fields preserve the pre-existing instance state.
    assert cfg.scene.terrain.generator.num_cols == 3
    assert cfg.scene.terrain.generator.border_width == pytest.approx(2.5)
    assert list(cfg.scene.terrain.generator.sub_terrains) == ["test_flat"]
    assert cfg.scene.terrain.generator.add_lights is False


def test_ppo_go2_joystick_rough_hydra_terrain_override():
    """Issue #197 DoD: rough terrain profile parameters must be overridable
    via Hydra command-line. Composes the resolved config and feeds it through
    the same BackendAdapter -> registry.apply_cfg_overrides path the trainer
    uses."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from unilab.base.registry import apply_cfg_overrides
    from unilab.envs.locomotion.go2.rough import Go2JoystickRoughCfg
    from unilab.training.backend_adapter import BackendAdapter

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=[
                "task=go2_joystick_rough/mujoco",
                "env.scene.terrain.generator.num_rows=4",
                "env.scene.terrain.generator.num_cols=6",
                "env.scene.terrain.generator.seed=42",
                "env.scene.terrain.generator.curriculum=true",
            ],
        )

    # Yaml exposes the overridable schema (struct-mode acceptance).
    assert cfg.env.scene.terrain.hfield_name == "terrain_hfield"
    assert cfg.env.scene.terrain.geom_name == "floor"
    assert cfg.env.scene.terrain.generator.num_rows == 4
    assert cfg.env.scene.terrain.generator.num_cols == 6
    assert cfg.env.scene.terrain.generator.seed == 42
    assert cfg.env.scene.terrain.generator.curriculum is True

    # End-to-end: the override dict produced by the adapter must, after the
    # registry's deep-merge, leave Go2JoystickRoughCfg in a coherent state —
    # overridden fields applied, untouched dataclass defaults preserved.
    adapter = BackendAdapter(cfg, root_dir=Path.cwd())
    env_cfg_override = adapter.build_task_env_cfg_override()
    assert "scene" in env_cfg_override
    assert env_cfg_override["scene"]["terrain"]["generator"]["num_rows"] == 4

    env_cfg = Go2JoystickRoughCfg()
    apply_cfg_overrides(env_cfg, env_cfg_override)

    assert env_cfg.scene.terrain.generator.num_rows == 4
    assert env_cfg.scene.terrain.generator.num_cols == 6
    assert env_cfg.scene.terrain.generator.seed == 42
    assert env_cfg.scene.terrain.generator.curriculum is True
    # sub_terrains is not in the yaml schema, so its Python default survives.
    assert len(env_cfg.scene.terrain.generator.sub_terrains) == 7
