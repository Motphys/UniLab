"""Hydra and cold-path contracts for the T800 Manager-Based PPO owner."""

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
from unilab.tasks.locomotion.t800.manager_terms import T800JointPositionActionCfg

ROOT_DIR = Path(__file__).parents[4]
CONF_DIR = ROOT_DIR / "conf" / "ppo"

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
_ACTIVE_JOINT_NAMES = _JOINT_NAMES[:12] + _JOINT_NAMES[13:23]
_HELD_JOINT_NAMES = (_JOINT_NAMES[12], _JOINT_NAMES[23], _JOINT_NAMES[24])
_ACTION_SCALES = (
    0.5,
    0.2,
    0.2,
    0.5,
    0.5,
    0.2,
    0.5,
    0.2,
    0.2,
    0.5,
    0.5,
    0.2,
    0.2,
    0.2,
    0.05,
    0.2,
    0.05,
    0.2,
    0.2,
    0.05,
    0.2,
    0.05,
)
_POSE_WEIGHTS = (
    0.01,
    1.0,
    5.0,
    0.01,
    5.0,
    5.0,
    0.01,
    1.0,
    5.0,
    0.01,
    5.0,
    5.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
)


def _compose_owner():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose("config", overrides=["task=t800_walk_flat/mujoco"])


def _materialize_owner() -> tuple[Any, ManagerBasedRlEnvCfg]:
    cfg = _compose_owner()
    registry.ensure_registries()
    env_override = BackendAdapter(
        cfg, root_dir=ROOT_DIR, algo_name="ppo"
    ).build_task_env_cfg_override()
    env_cfg = registry.materialize_env_config("T800WalkFlat")
    assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(env_cfg, env_override)
    env_cfg.validate()
    return cfg, env_cfg


def test_t800_ppo_owner_materializes_the_complete_manager_contract() -> None:
    cfg, env_cfg = _materialize_owner()
    env_cfg = cast(Any, env_cfg)

    assert cfg.training.task_name == "T800WalkFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.training.play_steps == 2000
    assert env_cfg.sim_dt == pytest.approx(0.002)
    assert env_cfg.ctrl_dt == pytest.approx(0.01)
    assert env_cfg.max_episode_seconds == pytest.approx(20.0)
    assert env_cfg.policy_observation_group == "policy"
    assert env_cfg.critic_observation_group == "critic"

    assert env_cfg.scene is not None
    assert env_cfg.scene.model_file.endswith("robots/t800/scene_flat.xml")
    assert env_cfg.scene.default_keyframe_name == "stand"
    robot = env_cfg.scene.entities["robot"]
    assert robot.root_body_name == "LINK_BASE"
    assert robot.body_names == ["LINK_BASE"]
    assert tuple(robot.joint_names or ()) == _JOINT_NAMES
    assert tuple(robot.actuator_names or ()) == _JOINT_NAMES

    action = env_cfg.actions["joint_pos"]
    assert isinstance(action, T800JointPositionActionCfg)
    assert tuple(action.actuator_names) == _ACTIVE_JOINT_NAMES
    assert tuple(action.held_actuator_names) == _HELD_JOINT_NAMES
    action_scale = cast(Any, action.scale)
    assert list(action_scale) == list(_ACTIVE_JOINT_NAMES)
    assert tuple(action_scale.values()) == pytest.approx(_ACTION_SCALES)
    assert action.use_default_offset is True

    policy_terms = env_cfg.observations["policy"].terms
    critic_terms = env_cfg.observations["critic"].terms
    expected_terms = (
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
        "gait_phase",
    )
    assert tuple(policy_terms) == expected_terms
    assert tuple(critic_terms) == (*expected_terms, "base_lin_vel")
    assert sum((3, 3, 22, 22, 22, 3, 2)) == 77
    assert sum((3, 3, 22, 22, 22, 3, 2, 3)) == 80
    assert policy_terms["base_ang_vel"].params["sensor_name"] == "torso_gyro"
    assert policy_terms["projected_gravity"].params["sensor_name"] == "torso_upvector"
    assert critic_terms["base_lin_vel"].params["sensor_name"] == "pelvis_local_linvel"

    assert tuple(env_cfg.rewards) == (
        "tracking_lin_vel",
        "tracking_ang_vel",
        "feet_phase",
        "lin_vel_z",
        "ang_vel_xy",
        "base_height",
        "orientation",
        "penalty_feet_ori",
        "action_rate",
        "pose",
    )
    assert env_cfg.rewards["feet_phase"].weight == pytest.approx(1.5)
    assert env_cfg.rewards["feet_phase"].params["swing_height"] == pytest.approx(0.13)
    assert env_cfg.rewards["feet_phase"].params["tracking_sigma"] == pytest.approx(0.014)
    assert env_cfg.rewards["base_height"].params["target_height"] == pytest.approx(1.0165)
    assert env_cfg.terminations["tilt"].params["max_tilt_deg"] == pytest.approx(25.0)
    assert env_cfg.terminations["base_height"].params["minimum_height"] == pytest.approx(0.7165)
    assert tuple(env_cfg.rewards["pose"].params["pose_weights"]) == pytest.approx(_POSE_WEIGHTS)

    for name, bound in {
        "base_ang_vel": 0.2,
        "projected_gravity": 0.05,
        "joint_pos": 0.01,
        "joint_vel": 1.5,
    }.items():
        noise = policy_terms[name].noise
        assert noise is not None
        assert noise.n_min == pytest.approx(-bound)
        assert noise.n_max == pytest.approx(bound)
    assert all(term.noise is None for term in critic_terms.values())


@pytest.mark.slow
def test_t800_ppo_owner_builds_and_steps_real_mujoco_env() -> None:
    pytest.importorskip("mujoco")
    cfg, _ = _materialize_owner()
    env = cast(
        Any,
        registry.make(
            "T800WalkFlat",
            sim_backend=str(cfg.training.sim_backend),
            num_envs=1,
            env_cfg_override=BackendAdapter(
                cfg, root_dir=ROOT_DIR, algo_name="ppo"
            ).build_task_env_cfg_override(),
        ),
    )
    try:
        reset_obs, reset_info = env.reset()
        assert isinstance(reset_info, dict)
        assert env.action_space.shape == (22,)
        assert reset_obs["obs"].shape == (1, 77)
        assert reset_obs["critic"].shape == (1, 80)

        action = np.zeros((1, 22), dtype=np.float32)
        state = env.step(action)
        assert np.isfinite(state.obs["obs"]).all()
        assert np.isfinite(state.obs["critic"]).all()
        assert np.isfinite(state.reward).all()
        controls = env._control
        assert controls.shape == (1, 25)
        assert np.isfinite(controls).all()
    finally:
        env.close()
