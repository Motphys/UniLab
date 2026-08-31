"""Hydra, Registry, and runtime contracts for MicroduckVelocityFlat."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.envs import ManagerBasedRlEnvCfg
from unilab.envs.mdp import JointPositionActionCfg
from unilab.tasks import __unilab_registry_modules__
from unilab.tasks.locomotion.microduck.deploy_contract import (
    MICRODUCK_ACTOR_OBS_DIM,
    MICRODUCK_CRITIC_OBS_DIM,
    MICRODUCK_NUM_ACTION,
    MICRODUCK_OBS_SEGMENTS,
)
from unilab.tasks.locomotion.microduck.manager_terms import (
    MicroduckVelocityCommandCfg,
    UniformVectorCommandCfg,
)

ROOT_DIR = Path(__file__).parents[4]
CONF_DIR = ROOT_DIR / "conf" / "ppo"

JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)


def _compose_owner():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose("config", overrides=["task=microduck_velocity_flat/mujoco"])


def _materialize_owner() -> tuple[Any, ManagerBasedRlEnvCfg]:
    cfg = _compose_owner()
    registry.ensure_registries()
    override = BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name="ppo",
    ).build_task_env_cfg_override()
    env_cfg = registry.materialize_env_config("MicroduckVelocityFlat")
    assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(env_cfg, override)
    env_cfg.validate()
    return cfg, env_cfg


def test_microduck_owner_materializes_complete_manager_contract() -> None:
    cfg, env_cfg = _materialize_owner()
    env_cfg = cast(Any, env_cfg)

    assert cfg.training.task_name == "MicroduckVelocityFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.algo.obs_groups.actor == ["policy"]
    assert cfg.algo.obs_groups.critic == ["critic"]
    assert cfg.algo.empirical_normalization is True

    assert env_cfg.sim_dt == pytest.approx(0.01)
    assert env_cfg.ctrl_dt == pytest.approx(0.02)
    assert env_cfg.max_episode_seconds == pytest.approx(20.0)
    assert env_cfg.policy_observation_group == "policy"
    assert env_cfg.critic_observation_group == "critic"
    assert env_cfg.scene.model_file.endswith("robots/microduck/scene_flat.xml")
    assert env_cfg.scene.fragment_files[0].endswith("robots/microduck/locomotion_task.xml")
    assert env_cfg.scene.default_keyframe_name == "home"
    robot = env_cfg.scene.entities["robot"]
    assert robot.root_body_name == "trunk_base"
    assert tuple(robot.joint_names) == JOINT_NAMES
    assert tuple(robot.actuator_names) == JOINT_NAMES

    action = env_cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    assert action.actuator_names == [".*"]
    assert action.scale == pytest.approx(1.0)
    assert action.use_default_offset is True

    policy = env_cfg.observations["policy"].terms
    critic = env_cfg.observations["critic"].terms
    expected = (
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "twist_command",
        "head_pose_command",
        "body_pose_command",
    )
    assert tuple(policy) == expected
    assert tuple(critic) == (*expected, "base_lin_vel")
    assert sum(dim for _, dim in MICRODUCK_OBS_SEGMENTS) == MICRODUCK_ACTOR_OBS_DIM
    assert MICRODUCK_CRITIC_OBS_DIM == MICRODUCK_ACTOR_OBS_DIM + 3
    assert policy["joint_pos"].params["biased"] is True
    assert policy["base_ang_vel"].noise.n_max == pytest.approx(0.03)
    assert policy["joint_vel"].noise.n_max == pytest.approx(0.25)
    assert all(term.noise is None for term in critic.values())

    twist = env_cfg.commands["twist"]
    assert isinstance(twist, MicroduckVelocityCommandCfg)
    assert twist.ranges.lin_vel_x == [-0.2, 0.2]
    assert twist.ranges.lin_vel_y == [-0.1, 0.1]
    assert twist.ranges.ang_vel_z == [-0.5, 0.5]
    assert twist.rel_standing_envs == pytest.approx(0.02)
    assert isinstance(env_cfg.commands["head_pose"], UniformVectorCommandCfg)
    assert len(env_cfg.commands["head_pose"].ranges) == 4
    assert len(env_cfg.commands["body_pose"].ranges) == 6

    expected_rewards = (
        "tracking_lin_vel",
        "tracking_ang_vel",
        "head_pose_tracking",
        "head_pose_bias",
        "leg_pose",
        "foot_air_time_biped",
        "flight_phase",
        "lin_vel_z",
        "ang_vel_xy",
        "orientation",
        "base_height",
        "action_rate",
        "alive",
    )
    assert tuple(env_cfg.rewards) == expected_rewards
    assert env_cfg.rewards["tracking_lin_vel"].weight == pytest.approx(3.0)
    assert env_cfg.rewards["foot_air_time_biped"].weight == pytest.approx(2.0)
    assert env_cfg.rewards["flight_phase"].weight == pytest.approx(-2.0)
    assert env_cfg.rewards["head_pose_bias"].weight == pytest.approx(0.0)
    assert tuple(env_cfg.events) == (
        "reset_scene_to_default",
        "base_com",
        "encoder_bias",
        "push_robot",
    )
    assert env_cfg.events["push_robot"].mode == "interval"
    assert env_cfg.events["push_robot"].interval_range_s == [3.0, 3.0]


def test_microduck_registry_and_factory_are_manager_based(monkeypatch) -> None:
    registry.ensure_registries()
    assert "unilab.tasks.locomotion.microduck" in __unilab_registry_modules__
    assert registry.list_registered_envs()["MicroduckVelocityFlat"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mujoco"],
    }
    module = importlib.import_module("unilab.tasks.locomotion.microduck")
    assert registry._envs["MicroduckVelocityFlat"].env_factory_dict["mujoco"] is (
        module.make_microduck_velocity_env
    )
    assert registry._envs["MicroduckVelocityFlat"].env_cfg_factory is ManagerBasedRlEnvCfg
    try:
        legacy_spec = importlib.util.find_spec("unilab.envs.locomotion.microduck")
    except ModuleNotFoundError:
        legacy_spec = None
    assert legacy_spec is None

    calls: list[tuple[str, object, object]] = []
    sentinel = object()
    config = ManagerBasedRlEnvCfg()

    def fake_resolver(directory: str, *, marker: str):
        calls.append(("resolve", directory, marker))
        return sentinel

    def fake_builder(cfg, *, num_envs: int, backend_type: str):
        assert cfg is config
        calls.append(("build", num_envs, backend_type))
        return sentinel

    monkeypatch.setattr(module, "resolve_robot_asset_dir", fake_resolver)
    monkeypatch.setattr(module, "make_manager_based_rl_env", fake_builder)
    assert module.make_microduck_velocity_env(config, num_envs=4, backend_type="mujoco") is sentinel
    assert calls == [
        ("resolve", "robots/microduck/assets", "trunk_base.stl"),
        ("build", 4, "mujoco"),
    ]


@pytest.mark.slow
def test_microduck_owner_builds_and_steps_real_mujoco_env() -> None:
    pytest.importorskip("mujoco")
    cfg, _ = _materialize_owner()
    env = cast(
        Any,
        registry.make(
            "MicroduckVelocityFlat",
            sim_backend="mujoco",
            num_envs=1,
            env_cfg_override=BackendAdapter(
                cfg,
                root_dir=ROOT_DIR,
                algo_name="ppo",
            ).build_task_env_cfg_override(),
        ),
    )
    try:
        obs, info = env.reset()
        assert isinstance(info, dict)
        assert env.action_space.shape == (MICRODUCK_NUM_ACTION,)
        assert obs["obs"].shape == (1, MICRODUCK_ACTOR_OBS_DIM)
        assert obs["critic"].shape == (1, MICRODUCK_CRITIC_OBS_DIM)
        assert env.command_manager.get_command("twist").shape == (1, 3)
        assert env.command_manager.get_command("head_pose").shape == (1, 4)
        assert env.command_manager.get_command("body_pose").shape == (1, 6)

        state = env.step(np.zeros((1, MICRODUCK_NUM_ACTION), dtype=np.float32))
        assert np.isfinite(state.obs["obs"]).all()
        assert np.isfinite(state.obs["critic"]).all()
        assert np.isfinite(state.reward).all()
        assert env.scene["robot"].data.encoder_bias.shape == (1, MICRODUCK_NUM_ACTION)
    finally:
        env.close()
