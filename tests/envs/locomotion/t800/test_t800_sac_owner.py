"""Hydra and cold-path contracts for the T800 Manager-Based SAC owner."""

from __future__ import annotations

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
from unilab.tasks.locomotion.t800.manager_terms import (
    T800JointPositionActionCfg,
    penalty_close_feet_lateral,
)

ROOT_DIR = Path(__file__).parents[4]
CONF_DIR = ROOT_DIR / "conf" / "sac"

_JOINT_NAMES = (
    "J00_HIP_PITCH_L",
    "J01_HIP_ROLL_L",
    "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L",
    "J04_ANKLE_PITCH_L",
    "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R",
    "J07_HIP_ROLL_R",
    "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R",
    "J10_ANKLE_PITCH_R",
    "J11_ANKLE_ROLL_R",
    "J12_TORSO_YAW",
    "J13_SHOULDER_PITCH_L",
    "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L",
    "J17_ELBOW_YAW_L",
    "J18_SHOULDER_PITCH_R",
    "J19_SHOULDER_ROLL_R",
    "J20_SHOULDER_YAW_R",
    "J21_ELBOW_PITCH_R",
    "J22_ELBOW_YAW_R",
    "J23_HEAD_PITCH",
    "J24_HEAD_YAW",
)
_ACTIVE = _JOINT_NAMES[:12] + _JOINT_NAMES[13:23]
_HELD = (_JOINT_NAMES[12], _JOINT_NAMES[23], _JOINT_NAMES[24])
_SCALES = (1.0,) * 12 + (0.2, 0.2, 0.05, 0.2, 0.05, 0.2, 0.2, 0.05, 0.2, 0.05)
_POSE = (0.01, 2.0, 5.0, 0.01, 5.0, 5.0, 0.01, 2.0, 5.0, 0.01, 5.0, 5.0) + (50.0,) * 10


def _compose_owner():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose("config", overrides=["task=t800_walk_flat/mujoco"])


def _materialize_owner() -> tuple[Any, ManagerBasedRlEnvCfg]:
    cfg = _compose_owner()
    registry.ensure_registries()
    env_override = BackendAdapter(
        cfg, root_dir=ROOT_DIR, algo_name="sac"
    ).build_task_env_cfg_override()
    env_cfg = registry.materialize_env_config("T800WalkFlat")
    assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(env_cfg, env_override)
    env_cfg.validate()
    return cfg, env_cfg


def test_t800_sac_owner_materializes_complete_contract() -> None:
    cfg, env_cfg = _materialize_owner()
    env_cfg = cast(Any, env_cfg)

    assert CONF_DIR.name == "sac"
    assert cfg.training.task_name == "T800WalkFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.algo.gamma == pytest.approx(0.98488578)
    assert cfg.algo.learning_starts == 10
    assert cfg.algo.updates_per_step == 8
    assert cfg.algo.algo_params.alpha_init == pytest.approx(0.001)
    assert cfg.algo.algo_params.target_entropy_ratio == pytest.approx(0.0)
    assert cfg.algo.max_iterations == 5000
    assert cfg.algo.save_interval == 1000
    assert env_cfg.sim_dt == pytest.approx(0.002)
    assert env_cfg.ctrl_dt == pytest.approx(0.01)
    assert env_cfg.max_episode_seconds == pytest.approx(20.0)

    robot = env_cfg.scene.entities["robot"]
    assert env_cfg.scene.model_file.endswith("robots/t800/scene_flat.xml")
    assert env_cfg.scene.default_keyframe_name == "stand"
    assert robot.root_body_name == "LINK_BASE"
    assert robot.body_names == ["LINK_BASE"]
    assert tuple(robot.joint_names) == _JOINT_NAMES
    assert tuple(robot.actuator_names) == _JOINT_NAMES

    action = env_cfg.actions["joint_pos"]
    assert isinstance(action, T800JointPositionActionCfg)
    assert tuple(action.actuator_names) == _ACTIVE
    assert tuple(action.held_actuator_names) == _HELD
    assert list(cast(Any, action.scale)) == list(_ACTIVE)
    assert tuple(cast(Any, action.scale).values()) == pytest.approx(_SCALES)
    assert action.use_default_offset is True

    policy = env_cfg.observations["policy"].terms
    critic = env_cfg.observations["critic"].terms
    assert tuple(policy) == (
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
        "gait_phase",
    )
    assert tuple(critic) == (*tuple(policy), "base_lin_vel")
    assert env_cfg.observations["policy"].enable_corruption is True
    assert env_cfg.observations["critic"].enable_corruption is False
    assert policy["base_ang_vel"].params["sensor_name"] == "torso_gyro"
    assert policy["projected_gravity"].params["sensor_name"] == "torso_upvector"
    assert policy["base_ang_vel"].scale == pytest.approx(0.25)
    assert policy["joint_vel"].scale == pytest.approx(0.05)
    assert critic["projected_gravity"].params["sensor_name"] == "torso_upvector"
    assert critic["base_lin_vel"].params["sensor_name"] == "pelvis_local_linvel"
    assert critic["base_lin_vel"].scale == pytest.approx(2.0)
    assert policy["joint_pos"].params["asset_cfg"].joint_names == list(_ACTIVE)
    assert policy["joint_vel"].params["asset_cfg"].joint_names == list(_ACTIVE)
    assert critic["joint_pos"].params["asset_cfg"].joint_names == list(_ACTIVE)
    assert critic["joint_vel"].params["asset_cfg"].joint_names == list(_ACTIVE)
    assert policy["joint_pos"].noise.n_max == pytest.approx(0.01)
    assert policy["joint_vel"].noise.n_max == pytest.approx(0.1)
    assert all(term.noise is None for term in critic.values())

    assert sum((3, 3, 22, 22, 22, 3, 2)) == 77
    assert sum((3, 3, 22, 22, 22, 3, 2, 3)) == 80
    assert tuple(env_cfg.rewards) == (
        "tracking_lin_vel",
        "tracking_ang_vel",
        "penalty_ang_vel_xy",
        "penalty_orientation",
        "penalty_action_rate",
        "pose",
        "penalty_feet_ori",
        "penalty_close_feet_lateral",
        "feet_phase",
        "alive",
    )
    expected_weights = {
        "tracking_lin_vel": 2.0,
        "tracking_ang_vel": 1.5,
        "penalty_ang_vel_xy": -1.0,
        "penalty_orientation": -10.0,
        "penalty_action_rate": -4.0,
        "pose": -0.5,
        "penalty_feet_ori": -20.0,
        "penalty_close_feet_lateral": -5.0,
        "feet_phase": 5.0,
        "alive": 10.0,
    }
    for name, weight in expected_weights.items():
        assert env_cfg.rewards[name].weight == pytest.approx(weight)
    assert tuple(env_cfg.rewards["pose"].params["pose_weights"]) == pytest.approx(_POSE)
    lateral = env_cfg.rewards["penalty_close_feet_lateral"]
    assert lateral.func is penalty_close_feet_lateral
    assert lateral.params["min_width"] == pytest.approx(0.18)
    assert lateral.params["sigma"] == pytest.approx(0.04)
    phase = env_cfg.rewards["feet_phase"]
    assert phase.params["swing_height"] == pytest.approx(0.11)
    assert phase.params["tracking_sigma"] == pytest.approx(0.014)

    curriculum = env_cfg.curriculum["penalty_scaling"]
    assert curriculum.params == {
        "initial_scale": 0.125,
        "min_scale": 0.125,
        "max_scale": 0.25,
        "level_down_threshold": 150.0,
        "level_up_threshold": 750.0,
        "degree": 0.001,
    }


@pytest.mark.slow
def test_t800_sac_owner_builds_and_steps_real_mujoco_env() -> None:
    pytest.importorskip("mujoco")
    cfg, _ = _materialize_owner()
    env = cast(
        Any,
        registry.make(
            "T800WalkFlat",
            sim_backend="mujoco",
            num_envs=1,
            env_cfg_override=BackendAdapter(
                cfg, root_dir=ROOT_DIR, algo_name="sac"
            ).build_task_env_cfg_override(),
        ),
    )
    try:
        obs, info = env.reset()
        assert isinstance(info, dict)
        assert env.action_space.shape == (22,)
        assert obs["obs"].shape == (1, 77)
        assert obs["critic"].shape == (1, 80)
        state = env.step(np.zeros((1, 22), dtype=np.float32))
        assert np.isfinite(state.obs["obs"]).all()
        assert np.isfinite(state.obs["critic"]).all()
        assert np.isfinite(state.reward).all()
        assert env._control.shape == (1, 25)
        assert np.isfinite(env._control).all()
    finally:
        env.close()
