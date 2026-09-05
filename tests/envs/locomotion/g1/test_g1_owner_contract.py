"""Hydra-owned production contract for the G1 walk Manager-Based tasks."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, mdp
from unilab.tasks.locomotion.g1 import manager_terms as g1_terms

# CPU-bound on the single-core CI runner; kept in the slow lane (make test-slow).
pytestmark = pytest.mark.slow

ROOT_DIR = Path(__file__).parents[4]
CONF_DIR = ROOT_DIR / "src" / "unilab" / "conf"

_RESET_EVENTS = ("reset_scene_to_default", "reset_root_state_uniform")
_PPO_REWARDS = (
    "tracking_lin_vel",
    "tracking_ang_vel",
    "feet_phase",
    "lin_vel_z",
    "ang_vel_xy",
    "base_height",
    "orientation",
    "action_rate",
    "pose",
)
# ppo g1_walk_flat additionally references the shared window-form air-time term
# (#1398) right after the private feet_phase term.
_PPO_WALK_FLAT_REWARDS = (
    "tracking_lin_vel",
    "tracking_ang_vel",
    "feet_phase",
    "feet_air_time",
    "lin_vel_z",
    "ang_vel_xy",
    "base_height",
    "orientation",
    "action_rate",
    "pose",
)
_MOTRIX_EXTRA_REWARDS = (
    "forward_progress",
    "under_speed",
    "upper_body_pose",
    "penalty_feet_ori",
    "feet_phase_contrast",
    "feet_phase_contact",
    "feet_double_stance",
)
_OFFPOLICY_REWARDS = (
    "tracking_lin_vel",
    "tracking_ang_vel",
    "penalty_ang_vel_xy",
    "penalty_orientation",
    "penalty_action_rate",
    "pose",
    "penalty_feet_ori",
    "feet_phase",
    "alive",
)

_OBSERVATION_TERMS = (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
    "gait_phase",
)

_POSE_WEIGHTS_29 = [0.01, 1.0, 5.0, 0.01, 5.0, 5.0] * 2 + [50.0] * 17
_POSE_WEIGHTS_23 = [0.01, 1.0, 5.0, 0.01, 5.0, 5.0] * 2 + [50.0] * 11

_OWNER_CASES = (
    pytest.param(
        "ppo",
        ("task=g1_walk_flat/mujoco",),
        "G1WalkFlat",
        "mujoco",
        29,
        0.25,
        "scene_flat.xml",
        _PPO_WALK_FLAT_REWARDS,
        (*_RESET_EVENTS, "pd_gains"),
        False,
        id="ppo-mujoco",
    ),
    pytest.param(
        "ppo",
        ("task=g1_walk_flat/motrix",),
        "G1WalkFlat",
        "motrix",
        29,
        0.5,
        "scene_flat.xml",
        (*_PPO_WALK_FLAT_REWARDS, *_MOTRIX_EXTRA_REWARDS),
        _RESET_EVENTS,
        False,
        id="ppo-motrix",
    ),
    pytest.param(
        "ppo",
        ("task=g1_walk_flat/mjwarp",),
        "G1WalkFlat",
        "mjwarp",
        29,
        0.25,
        "scene_flat.xml",
        _PPO_WALK_FLAT_REWARDS,
        _RESET_EVENTS,
        False,
        id="ppo-mjwarp",
    ),
    pytest.param(
        "ppo",
        ("task=g1_walk_flat/newton",),
        "G1WalkFlat",
        "newton",
        29,
        0.25,
        "scene_flat.xml",
        _PPO_WALK_FLAT_REWARDS,
        _RESET_EVENTS,
        False,
        id="ppo-newton",
    ),
    pytest.param(
        "ppo",
        ("task=g1_walk_flat/isaacgym",),
        "G1WalkFlat",
        "isaacgym",
        29,
        0.25,
        "scene_flat.xml",
        _PPO_WALK_FLAT_REWARDS,
        _RESET_EVENTS,
        False,
        id="ppo-isaacgym",
    ),
    pytest.param(
        "ppo",
        ("task=g1_walk_flat/genesis",),
        "G1WalkFlat",
        "genesis",
        29,
        0.25,
        "scene_flat.xml",
        _PPO_WALK_FLAT_REWARDS,
        # kp/kd reset randomization stays enabled: the backend declares the
        # measured RESET_TERM_KP/KD DR terms (REPORT #1372 §5.7).
        (*_RESET_EVENTS, "pd_gains"),
        False,
        id="ppo-genesis",
    ),
    pytest.param(
        "ppo",
        ("task=g1_walk_flat/isaacsim",),
        "G1WalkFlat",
        "isaacsim",
        29,
        0.25,
        "scene_flat.xml",
        _PPO_WALK_FLAT_REWARDS,
        _RESET_EVENTS,
        False,
        id="ppo-isaacsim",
    ),
    pytest.param(
        "ppo",
        ("task=g1_23dof_walk_flat/mujoco",),
        "G1Walk23DofFlat",
        "mujoco",
        23,
        0.25,
        "scene_flat_23dof.xml",
        _PPO_WALK_FLAT_REWARDS,
        (*_RESET_EVENTS, "pd_gains"),
        False,
        id="ppo-23dof-mujoco",
    ),
    pytest.param(
        "ppo",
        ("task=g1_23dof_walk_flat/motrix",),
        "G1Walk23DofFlat",
        "motrix",
        23,
        0.5,
        "scene_flat_23dof.xml",
        (*_PPO_WALK_FLAT_REWARDS, *_MOTRIX_EXTRA_REWARDS),
        _RESET_EVENTS,
        False,
        id="ppo-23dof-motrix",
    ),
    pytest.param(
        "ppo",
        ("task=g1_23dof_walk_rough/mujoco",),
        "G1Walk23DofRough",
        "mujoco",
        23,
        0.25,
        "scene_rough_23dof.xml",
        _PPO_REWARDS,
        (*_RESET_EVENTS, "pd_gains"),
        True,
        id="ppo-23dof-rough-mujoco",
    ),
    pytest.param(
        "appo",
        ("task=g1_walk_flat/mujoco",),
        "G1WalkFlat",
        "mujoco",
        29,
        0.25,
        "scene_flat.xml",
        _PPO_REWARDS,
        (*_RESET_EVENTS, "pd_gains"),
        False,
        id="appo-mujoco",
    ),
    pytest.param(
        "appo",
        ("task=g1_23dof_walk_flat/mujoco",),
        "G1Walk23DofFlat",
        "mujoco",
        23,
        0.25,
        "scene_flat_23dof.xml",
        _PPO_REWARDS,
        (*_RESET_EVENTS, "pd_gains"),
        False,
        id="appo-23dof-mujoco",
    ),
    pytest.param(
        "sac",
        ("task=g1_walk_flat/mujoco",),
        "G1WalkFlat",
        "mujoco",
        29,
        1.0,
        "scene_flat.xml",
        _OFFPOLICY_REWARDS,
        (*_RESET_EVENTS, "pd_gains"),
        True,
        id="sac-mujoco",
    ),
    pytest.param(
        "sac",
        ("task=g1_walk_flat/motrix",),
        "G1WalkFlat",
        "motrix",
        29,
        1.0,
        "scene_flat.xml",
        _OFFPOLICY_REWARDS,
        _RESET_EVENTS,
        True,
        id="sac-motrix",
    ),
    pytest.param(
        "sac",
        ("task=g1_walk_flat/mjwarp",),
        "G1WalkFlat",
        "mjwarp",
        29,
        1.0,
        "scene_flat.xml",
        _OFFPOLICY_REWARDS,
        _RESET_EVENTS,
        True,
        id="sac-mjwarp",
    ),
    pytest.param(
        "sac",
        ("task=g1_walk_flat/newton",),
        "G1WalkFlat",
        "newton",
        29,
        1.0,
        "scene_flat.xml",
        _OFFPOLICY_REWARDS,
        _RESET_EVENTS,
        True,
        id="sac-newton",
    ),
    pytest.param(
        "sac",
        ("task=g1_walk_flat/genesis",),
        "G1WalkFlat",
        "genesis",
        29,
        1.0,
        "scene_flat.xml",
        _OFFPOLICY_REWARDS,
        # kp/kd reset randomization stays enabled: the backend declares the
        # measured RESET_TERM_KP/KD DR terms (REPORT #1372 §5.7), unlike the
        # isaacgym owner which disables pd_gains.
        (*_RESET_EVENTS, "pd_gains"),
        True,
        id="sac-genesis",
    ),
    pytest.param(
        "sac",
        ("task=g1_walk_flat/isaacsim",),
        "G1WalkFlat",
        "isaacsim",
        29,
        1.0,
        "scene_flat.xml",
        _OFFPOLICY_REWARDS,
        _RESET_EVENTS,
        True,
        id="sac-isaacsim",
    ),
    pytest.param(
        "sac",
        ("task=g1_walk_rough/mujoco",),
        "G1WalkRough",
        "mujoco",
        29,
        1.0,
        "scene_rough.xml",
        _OFFPOLICY_REWARDS,
        (*_RESET_EVENTS, "pd_gains"),
        True,
        id="sac-rough-mujoco",
    ),
    pytest.param(
        "sac",
        ("task=g1_walk_rough/motrix",),
        "G1WalkRough",
        "motrix",
        29,
        1.0,
        "scene_rough.xml",
        _OFFPOLICY_REWARDS,
        _RESET_EVENTS,
        True,
        id="sac-rough-motrix",
    ),
    pytest.param(
        "sac",
        ("task=g1_23dof_walk_flat/mujoco",),
        "G1Walk23DofFlat",
        "mujoco",
        23,
        1.0,
        "scene_flat_23dof.xml",
        _OFFPOLICY_REWARDS,
        (*_RESET_EVENTS, "pd_gains"),
        True,
        id="sac-23dof-mujoco",
    ),
    pytest.param(
        "sac",
        ("task=g1_23dof_walk_rough/motrix",),
        "G1Walk23DofRough",
        "motrix",
        23,
        1.0,
        "scene_rough_23dof.xml",
        _OFFPOLICY_REWARDS,
        _RESET_EVENTS,
        True,
        id="sac-23dof-rough-motrix",
    ),
    pytest.param(
        "td3",
        ("task=g1_walk_flat/mujoco",),
        "G1WalkFlat",
        "mujoco",
        29,
        1.0,
        "scene_flat.xml",
        _OFFPOLICY_REWARDS,
        (*_RESET_EVENTS, "pd_gains"),
        True,
        id="td3-mujoco",
    ),
    pytest.param(
        "flashsac",
        ("task=g1_walk_flat/mujoco",),
        "G1WalkFlat",
        "mujoco",
        29,
        1.0,
        "scene_flat.xml",
        _OFFPOLICY_REWARDS,
        (*_RESET_EVENTS, "pd_gains"),
        True,
        id="flashsac-mujoco",
    ),
)

_WALK_PROFILE_IDS = {
    "sac-mujoco",
    "sac-motrix",
    "sac-mjwarp",
    "sac-newton",
    "sac-genesis",
    "sac-isaacsim",
    "sac-rough-mujoco",
    "sac-rough-motrix",
    "sac-23dof-mujoco",
    "sac-23dof-rough-motrix",
    "td3-mujoco",
    "flashsac-mujoco",
}


def _compose(config_group: str, overrides: Sequence[str]) -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / config_group), version_base="1.3"):
        return compose("config", overrides=list(overrides))


def _materialize(
    config_group: str, overrides: Sequence[str], task_name: str
) -> tuple[DictConfig, ManagerBasedRlEnvCfg, dict[str, Any]]:
    hydra_cfg = _compose(config_group, overrides)
    env_override = BackendAdapter(hydra_cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    env_cfg = registry.materialize_env_config(task_name)
    assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(env_cfg, env_override)
    env_cfg.validate()
    return hydra_cfg, env_cfg, env_override


def _assert_no_omegaconf(value: Any) -> None:
    assert not OmegaConf.is_config(value)
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            _assert_no_omegaconf(getattr(value, item.name))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_omegaconf(key)
            _assert_no_omegaconf(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _assert_no_omegaconf(item)


@pytest.mark.parametrize(
    "config_group,overrides,task_name,backend,num_dof,action_scale,model_suffix,"
    "expected_rewards,expected_events,has_curriculum",
    _OWNER_CASES,
    ids=[case.id for case in _OWNER_CASES],
)
def test_g1_owner_materializes_complete_plain_manager_cfg(
    config_group: str,
    overrides: tuple[str, ...],
    task_name: str,
    backend: str,
    num_dof: int,
    action_scale: float,
    model_suffix: str,
    expected_rewards: tuple[str, ...],
    expected_events: tuple[str, ...],
    has_curriculum: bool,
) -> None:
    registry.ensure_registries()
    hydra_cfg, env_cfg, _ = _materialize(config_group, overrides, task_name)
    case_id = next(
        case.id
        for case in _OWNER_CASES
        if case.values[0] == config_group and case.values[1] == overrides
    )

    assert hydra_cfg.training.task_name == task_name
    assert hydra_cfg.training.sim_backend == backend
    assert env_cfg.ctrl_dt == pytest.approx(0.02)
    assert env_cfg.max_episode_seconds == pytest.approx(20.0)
    assert env_cfg.policy_observation_group == "policy"
    assert env_cfg.critic_observation_group == "critic"
    assert env_cfg.scale_rewards_by_dt is True

    assert env_cfg.scene is not None
    assert env_cfg.scene.model_file.endswith(f"robots/g1/{model_suffix}")
    assert env_cfg.scene.default_keyframe_name == "stand"
    robot = env_cfg.scene.entities["robot"]
    assert robot.root_body_name == "pelvis"
    assert len(robot.joint_names or ()) == num_dof
    assert len(robot.actuator_names or ()) == num_dof
    assert robot.body_names == ["pelvis"]

    assert list(env_cfg.observations) == ["policy", "critic"]
    assert list(env_cfg.observations["policy"].terms) == list(_OBSERVATION_TERMS)
    assert list(env_cfg.observations["critic"].terms) == [*_OBSERVATION_TERMS, "base_lin_vel"]

    # Observation scaling profiles are explicit per-owner term scales.
    policy_terms = env_cfg.observations["policy"].terms
    critic_terms = env_cfg.observations["critic"].terms
    walk_profile = case_id in _WALK_PROFILE_IDS
    expected_gyro = 0.25 if walk_profile else None
    expected_joint_vel = 0.05 if walk_profile else None
    expected_linvel = 2.0 if walk_profile else None
    for terms in (policy_terms, critic_terms):
        assert terms["base_ang_vel"].scale == expected_gyro
        assert terms["joint_vel"].scale == expected_joint_vel
        assert terms["projected_gravity"].scale is None
        assert terms["joint_pos"].scale is None
        assert terms["actions"].scale is None
        assert terms["command"].scale is None
        assert terms["gait_phase"].scale is None
        assert terms["gait_phase"].params["frequency"] == pytest.approx(1.5)
    assert critic_terms["base_lin_vel"].scale == expected_linvel

    # Observation noise reproduces the legacy noise_config: actor-only (the
    # critic reads clean observations), applied before term scaling.
    assert env_cfg.observations["policy"].enable_corruption is True
    assert env_cfg.observations["critic"].enable_corruption is False
    expected_noise = (
        {"joint_pos": 0.01, "joint_vel": 0.1}
        if walk_profile
        else {
            "base_ang_vel": 0.2,
            "projected_gravity": 0.05,
            "joint_pos": 0.01,
            "joint_vel": 1.5,
        }
    )
    for name, term in policy_terms.items():
        if name in expected_noise:
            assert term.noise is not None
            assert term.noise.n_max == pytest.approx(expected_noise[name])
            assert term.noise.n_min == pytest.approx(-expected_noise[name])
        else:
            assert term.noise is None
    for term in critic_terms.values():
        assert term.noise is None

    assert list(env_cfg.actions) == ["joint_pos"]
    assert env_cfg.actions["joint_pos"].scale == pytest.approx(action_scale)
    assert env_cfg.actions["joint_pos"].use_default_offset is True

    assert list(env_cfg.terminations) == ["time_out", "tilt", "base_height"]
    assert env_cfg.terminations["time_out"].time_out is True
    assert env_cfg.terminations["tilt"].func is g1_terms.g1_tilt_exceeded
    assert env_cfg.terminations["base_height"].func is mdp.root_height_below_minimum

    assert tuple(name for name, term in env_cfg.events.items() if term is not None) == (
        expected_events
    )
    assert tuple(name for name, term in env_cfg.rewards.items() if term is not None) == (
        expected_rewards
    )
    if has_curriculum:
        assert list(env_cfg.curriculum) == ["penalty_scaling"]
        assert env_cfg.curriculum["penalty_scaling"].func is g1_terms.G1PenaltyCurriculum
    else:
        assert not env_cfg.curriculum

    command = env_cfg.commands["twist"]
    assert isinstance(command, g1_terms.G1VelocityCommandCfg)
    assert command.planar_dead_zone == pytest.approx(0.2)
    assert command.resampling_time_range == [20.0, 20.0]
    if backend == "motrix" and config_group == "ppo":
        assert tuple(command.ranges.lin_vel_x) == (0.4, 0.7)
        assert tuple(command.ranges.lin_vel_y) == (0.0, 0.0)
    else:
        assert tuple(command.ranges.lin_vel_x) == (-0.6, 1.0)
        assert tuple(command.ranges.lin_vel_y) == (-0.4, 0.4)
        assert tuple(command.ranges.ang_vel_z) == (-0.8, 0.8)

    if backend == "mjwarp":
        assert env_cfg.mjwarp_nconmax == 128
        assert env_cfg.mjwarp_njmax == 256
    if backend == "newton":
        assert env_cfg.newton_device is None
        assert env_cfg.newton_nconmax == 320
        assert env_cfg.newton_njmax == 512
        assert env_cfg.newton_capacity_check_steps == 1
        # Native ViewerGL rendering (interactive viewer + offscreen record)
        # is supported; playback stays on the base config's auto mode.
        assert hydra_cfg.training.play_render_mode == "auto"
    if backend == "isaacgym":
        assert env_cfg.isaacgym_device_id == 0
        # Native rendering (viewer + camera-sensor record) is supported;
        # playback stays on the base config's auto mode.
        assert hydra_cfg.training.play_render_mode == "auto"
    if backend == "genesis":
        # Re-declares the MJCF <option integrator="implicitfast"> that Genesis
        # drops at import; the other global options keep Genesis defaults.
        assert env_cfg.genesis_integrator == "implicitfast"
        assert env_cfg.genesis_constraint_solver is None
        assert env_cfg.genesis_friction_cone is None
        assert env_cfg.genesis_solver_iterations is None
        # Native rendering (interactive viewer + offscreen record) is
        # supported; playback stays on the base config's auto mode.
        assert hydra_cfg.training.play_render_mode == "auto"
    if backend == "isaacsim":
        assert env_cfg.isaacsim_device_id == 0
        assert env_cfg.isaacsim_worker_timeout_s == pytest.approx(120.0)
        assert hydra_cfg.training.play_render_mode == "auto"
        assert hydra_cfg.play_profile.enabled is False

    pose = env_cfg.rewards["pose"]
    expected_weights = _POSE_WEIGHTS_29 if num_dof == 29 else _POSE_WEIGHTS_23
    if case_id == "flashsac-mujoco":
        expected_weights = [2.0 if i in (1, 7) else w for i, w in enumerate(expected_weights)]
    assert list(pose.params["pose_weights"]) == pytest.approx(expected_weights)

    for manager_name in ("observations", "events", "rewards", "terminations", "curriculum"):
        for term in getattr(env_cfg, manager_name).values():
            if term is None:
                continue
            nested_terms = term.terms.values() if manager_name == "observations" else (term,)
            for nested in nested_terms:
                if nested is None:
                    continue
                module = nested.func.__module__
                assert ".backend." not in module
                assert not any(
                    name in module
                    for name in (
                        ".mujoco",
                        ".motrix",
                        ".mjwarp",
                        ".isaacgym",
                        ".isaacsim",
                        ".newton",
                    )
                )

    _assert_no_omegaconf(env_cfg)


def test_g1_walk_registries_are_manager_only() -> None:
    registry.ensure_registries()
    metadata = registry.list_registered_envs()

    assert metadata["G1WalkFlat"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": [
            "mujoco",
            "mjwarp",
            "motrix",
            "isaacgym",
            "genesis",
            "isaacsim",
            "newton",
        ],
    }
    assert metadata["G1WalkRough"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mujoco", "motrix"],
    }
    assert metadata["G1Walk23DofFlat"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mujoco", "motrix"],
    }
    assert metadata["G1Walk23DofRough"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mujoco", "motrix"],
    }

    for legacy_override in (
        {"reward_config": {}},
        {"domain_rand": {"randomize_kp": True}},
        {"control_config": {"action_scale": 0.25}},
        {"gait_phase_init_mode": "offset_phase"},
        {"reset_base_qvel_limit": 0.5},
        {"noise_config": {"level": 1.0}},
    ):
        with pytest.raises(ValueError, match="has no attribute"):
            apply_cfg_overrides(ManagerBasedRlEnvCfg(), legacy_override)


@pytest.mark.parametrize(
    ("config_group", "overrides", "task_name", "backend", "num_dof", "obs_dim", "critic_dim"),
    (
        pytest.param(
            "ppo",
            ("task=g1_walk_flat/mujoco",),
            "G1WalkFlat",
            "mujoco",
            29,
            98,
            101,
            id="ppo-mujoco",
        ),
        pytest.param(
            "ppo",
            ("task=g1_walk_flat/motrix",),
            "G1WalkFlat",
            "motrix",
            29,
            98,
            101,
            id="ppo-motrix",
        ),
        pytest.param(
            "sac",
            ("task=g1_walk_flat/mujoco",),
            "G1WalkFlat",
            "mujoco",
            29,
            98,
            101,
            id="sac-mujoco",
        ),
        pytest.param(
            "ppo",
            ("task=g1_23dof_walk_flat/mujoco",),
            "G1Walk23DofFlat",
            "mujoco",
            23,
            80,
            83,
            id="ppo-23dof-mujoco",
        ),
        pytest.param(
            "ppo",
            ("task=g1_23dof_walk_rough/mujoco",),
            "G1Walk23DofRough",
            "mujoco",
            23,
            80,
            83,
            id="ppo-23dof-rough-mujoco",
        ),
    ),
)
def test_g1_registry_executes_real_manager_runtime(
    config_group: str,
    overrides: tuple[str, ...],
    task_name: str,
    backend: str,
    num_dof: int,
    obs_dim: int,
    critic_dim: int,
) -> None:
    registry.ensure_registries()
    _, env_cfg, env_override = _materialize(config_group, overrides, task_name)
    try:
        env = registry.make(
            task_name,
            sim_backend=backend,
            env_cfg_override=env_override,
            num_envs=2,
        )
    except ImportError as exc:
        pytest.skip(f"{backend} runtime unavailable: {exc}")

    try:
        assert isinstance(env, ManagerBasedRlEnv)
        assert env.obs_groups_spec == {"obs": obs_dim, "critic": critic_dim}
        assert env.action_space.shape == (num_dof,)
        action = env.action_manager.get_term("joint_pos")
        assert len(action.target_names) == num_dof

        obs, info = env.reset(seed=7)
        assert {name: value.shape for name, value in obs.items()} == {
            "obs": (2, obs_dim),
            "critic": (2, critic_dim),
        }
        assert isinstance(info, dict)
        for _ in range(5):
            state = env.step(np.zeros((2, num_dof), dtype=np.float32))
        for value in (*state.obs.values(), state.reward):
            assert isinstance(value, np.ndarray)
            assert np.isfinite(value).all()

        # The command and gait-phase segments pin the legacy obs layout tail.
        command = env.command_manager.get_command("twist")
        np.testing.assert_allclose(
            state.obs["obs"][:, obs_dim - 5 : obs_dim - 2], command, rtol=0.0, atol=1.0e-6
        )
    finally:
        env.close()


def test_g1_walk_profile_runtime_obs_scaling_matches_legacy_layout() -> None:
    """Walk-profile owners scale gyro x0.25, dof_vel x0.05, critic linvel x2.0."""
    registry.ensure_registries()
    _, _, env_override = _materialize("sac", ("task=g1_walk_flat/mujoco",), "G1WalkFlat")
    # Exact comparison against raw sensor reads requires clean observations.
    env_override["observations"]["policy"]["enable_corruption"] = False
    try:
        env = registry.make(
            "G1WalkFlat", sim_backend="mujoco", env_cfg_override=env_override, num_envs=2
        )
    except ImportError as exc:
        pytest.skip(f"mujoco runtime unavailable: {exc}")

    try:
        env.reset(seed=3)
        state = env.step(np.zeros((2, 29), dtype=np.float32))
        gyro = env._backend.get_sensor_data("torso_gyro")
        upvector = env._backend.get_sensor_data("torso_upvector")
        dof_vel = env._backend.get_dof_vel()
        linvel = env._backend.get_sensor_data("pelvis_local_linvel")
        np.testing.assert_allclose(state.obs["obs"][:, :3], 0.25 * gyro, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(state.obs["obs"][:, 3:6], -upvector, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(
            state.obs["obs"][:, 35:64], 0.05 * dof_vel, rtol=0.0, atol=1.0e-6
        )
        np.testing.assert_allclose(
            state.obs["critic"][:, 98:101], 2.0 * linvel, rtol=0.0, atol=1.0e-6
        )
        np.testing.assert_allclose(state.obs["critic"][:, :3], 0.25 * gyro, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(
            state.obs["obs"][:, 96:98], state.obs["critic"][:, 96:98], rtol=0.0, atol=0.0
        )
    finally:
        env.close()


def test_g1_legacy_profile_runtime_obs_scaling_matches_legacy_layout() -> None:
    """Legacy-profile owners keep unit scaling on every observation segment."""
    registry.ensure_registries()
    _, _, env_override = _materialize("ppo", ("task=g1_walk_flat/mujoco",), "G1WalkFlat")
    # Exact comparison against raw sensor reads requires clean observations.
    env_override["observations"]["policy"]["enable_corruption"] = False
    try:
        env = registry.make(
            "G1WalkFlat", sim_backend="mujoco", env_cfg_override=env_override, num_envs=2
        )
    except ImportError as exc:
        pytest.skip(f"mujoco runtime unavailable: {exc}")

    try:
        env.reset(seed=3)
        state = env.step(np.zeros((2, 29), dtype=np.float32))
        gyro = env._backend.get_sensor_data("torso_gyro")
        dof_vel = env._backend.get_dof_vel()
        linvel = env._backend.get_sensor_data("pelvis_local_linvel")
        np.testing.assert_allclose(state.obs["obs"][:, :3], gyro, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(state.obs["obs"][:, 35:64], dof_vel, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(state.obs["critic"][:, 98:101], linvel, rtol=0.0, atol=1.0e-6)
    finally:
        env.close()


def test_g1_penalty_curriculum_scales_negative_weights_from_start() -> None:
    registry.ensure_registries()
    _, _, env_override = _materialize("sac", ("task=g1_walk_flat/mujoco",), "G1WalkFlat")
    override_snapshot = deepcopy(env_override)
    try:
        env = registry.make(
            "G1WalkFlat", sim_backend="mujoco", env_cfg_override=env_override, num_envs=2
        )
        # Repeated construction from the same override must not drift: the
        # legacy env halved the shared override dict on every construction.
        env_repeat = registry.make(
            "G1WalkFlat", sim_backend="mujoco", env_cfg_override=env_override, num_envs=2
        )
    except ImportError as exc:
        pytest.skip(f"mujoco runtime unavailable: {exc}")

    try:
        for built in (env, env_repeat):
            assert built.curriculum_manager.active_terms == ["penalty_scaling"]
            # initial_scale=0.125 scales every negative weight from construction,
            # matching the tuned legacy effective schedule (1/8 initial, 1/4 cap).
            assert built.reward_manager.get_term_cfg("penalty_orientation").weight == pytest.approx(
                -1.25
            )
            assert built.reward_manager.get_term_cfg("penalty_action_rate").weight == pytest.approx(
                -0.5
            )
            assert built.reward_manager.get_term_cfg("pose").weight == pytest.approx(-0.0625)
            # Positive weights stay untouched.
            assert built.reward_manager.get_term_cfg("alive").weight == pytest.approx(10.0)
            assert built.reward_manager.get_term_cfg("feet_phase").weight == pytest.approx(5.0)
        # The shared override dict is never mutated in place.
        assert env_override == override_snapshot

        state = env.step(np.zeros((2, 29), dtype=np.float32))
        log = state.info["log"]
        for name in _OFFPOLICY_REWARDS:
            assert f"reward/{name}" in log
        assert log["reward/penalty_action_rate"] == pytest.approx(0.0)
    finally:
        env.close()
        env_repeat.close()


# Every task owner carrying the penalty curriculum, with the schedule that
# reproduces its tuned legacy baseline. Legacy offpolicy runners built three
# envs per training run (two probe envs + the spawned collector) and the
# legacy PenaltyCurriculum halved the shared override dict in place on each
# construction, so collectors effectively trained at 1/8 initial / 1/4 cap of
# the YAML weights. The on-policy runners built a single env, so their
# effective schedule was the declared 0.5 -> 1.0. The manager runtime isolates
# each env, so these params are now the single source of truth.
_PENALTY_CURRICULUM_CASES = (
    pytest.param(
        "sac",
        ("task=g1_walk_flat/mujoco",),
        "G1WalkFlat",
        id="sac-walk-flat",
    ),
    pytest.param(
        "sac",
        ("task=g1_walk_rough/mujoco",),
        "G1WalkRough",
        id="sac-walk-rough",
    ),
    pytest.param(
        "sac",
        ("task=g1_23dof_walk_flat/mujoco",),
        "G1Walk23DofFlat",
        id="sac-23dof-walk-flat",
    ),
    pytest.param(
        "sac",
        ("task=g1_23dof_walk_rough/mujoco",),
        "G1Walk23DofRough",
        id="sac-23dof-walk-rough",
    ),
    pytest.param(
        "td3",
        ("task=g1_walk_flat/mujoco",),
        "G1WalkFlat",
        id="td3-walk-flat",
    ),
    pytest.param(
        "flashsac",
        ("task=g1_walk_flat/mujoco",),
        "G1WalkFlat",
        id="flashsac-walk-flat",
    ),
)

_OFFPOLICY_ALIGNED_SCHEDULE = {"initial_scale": 0.125, "min_scale": 0.125, "max_scale": 0.25}


@pytest.mark.parametrize(
    "config_group,overrides,task_name",
    _PENALTY_CURRICULUM_CASES,
    ids=[case.id for case in _PENALTY_CURRICULUM_CASES],
)
def test_offpolicy_penalty_curriculum_matches_legacy_effective_schedule(
    config_group: str, overrides: tuple[str, ...], task_name: str
) -> None:
    _, env_cfg, _ = _materialize(config_group, overrides, task_name)
    params = env_cfg.curriculum["penalty_scaling"].params
    for key, expected in _OFFPOLICY_ALIGNED_SCHEDULE.items():
        assert params[key] == pytest.approx(expected)
    # Thresholds and annealing rate stay as tuned.
    assert params["level_down_threshold"] == pytest.approx(150.0)
    assert params["level_up_threshold"] == pytest.approx(750.0)
    assert params["degree"] == pytest.approx(0.001)


def test_ppo_penalty_curriculum_matches_legacy_effective_schedule() -> None:
    # The on-policy runner builds a single env per training run, so the legacy
    # effective schedule equals the declared 0.5 -> 1.0 range.
    _, env_cfg, _ = _materialize("ppo", ("task=g1_23dof_walk_rough/mujoco",), "G1Walk23DofRough")
    params = env_cfg.curriculum["penalty_scaling"].params
    assert params["initial_scale"] == pytest.approx(0.5)
    assert params["min_scale"] == pytest.approx(0.5)
    assert params["max_scale"] == pytest.approx(1.0)


def _genesis_runtime_available() -> bool:
    from unisim.backend.genesis.dependencies import genesis_dependencies_available

    if not genesis_dependencies_available():
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - any torch probe failure means unavailable
        return False


# Runs in a subprocess: Genesis allows exactly one gs.init per process
# (REPORT #1372 §3.5 [9a]) and tests/base/test_genesis_runtime.py deliberately
# destroys the session to verify re-init fails closed, so an in-process smoke
# could observe a poisoned session depending on pytest collection order.
# The config group (ppo/sac) arrives as argv[1]; each parametrized case gets
# its own subprocess, hence its own gs session.
_GENESIS_ENV_SMOKE_SCRIPT = """
import sys
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

ROOT = Path.cwd()
CONFIG_GROUP = sys.argv[1]

registry.ensure_registries()
GlobalHydra.instance().clear()
with initialize_config_dir(
    config_dir=str(ROOT / "src" / "unilab" / "conf" / CONFIG_GROUP), version_base="1.3"
):
    hydra_cfg = compose("config", overrides=["task=g1_walk_flat/genesis"])
assert hydra_cfg.training.task_name == "G1WalkFlat"
assert hydra_cfg.training.sim_backend == "genesis"
assert hydra_cfg.training.play_render_mode == "auto"

env_override = BackendAdapter(
    hydra_cfg, root_dir=ROOT, algo_name=CONFIG_GROUP
).build_task_env_cfg_override()
env_cfg = registry.materialize_env_config("G1WalkFlat")
assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
apply_cfg_overrides(env_cfg, env_override)
env_cfg.validate()
assert env_cfg.genesis_integrator == "implicitfast"

env = registry.make(
    "G1WalkFlat",
    sim_backend="genesis",
    env_cfg_override=env_override,
    num_envs=2,
)
try:
    assert isinstance(env, ManagerBasedRlEnv)
    assert env.obs_groups_spec == {"obs": 98, "critic": 101}
    assert env.action_space.shape == (29,)

    # Keyframe reset: reset_scene_to_default consumes the cold-path-scanned
    # "stand" keyframe (Genesis itself drops <keyframe> at import).
    obs, info = env.reset(seed=7)
    assert isinstance(obs, dict) and isinstance(info, dict)
    assert {name: value.shape for name, value in obs.items()} == {
        "obs": (2, 98),
        "critic": (2, 101),
    }
    for _ in range(12):
        state = env.step(np.zeros((2, 29), dtype=np.float32))
        assert set(state.obs) == {"obs", "critic"}
        assert state.obs["obs"].shape == (2, 98)
        assert state.obs["critic"].shape == (2, 101)
        for value in (*state.obs.values(), state.reward):
            assert isinstance(value, np.ndarray)
            assert np.isfinite(value).all()

    # Play path: mode none enters safely as a no-op; record resolves to the
    # native offscreen plan and validates its required fields.
    plan = env.resolve_play_render_plan(
        play_render_mode="none", play_steps=100, output_video=None
    )
    assert plan.mode == "none"
    try:
        env.resolve_play_render_plan(
            play_render_mode="record", play_steps=100, output_video=None
        )
    except ValueError as exc:
        assert "output video path" in str(exc)
    else:
        raise AssertionError("genesis record playback must require an output path")
    plan = env.resolve_play_render_plan(
        play_render_mode="record", play_steps=100, output_video="out.mp4"
    )
    assert plan.mode == "record" and plan.record_video and plan.headless

    print(
        f"[genesis env smoke:{CONFIG_GROUP}] reset+12 steps OK; "
        f"obs={obs['obs'].shape} critic={obs['critic'].shape} "
        f"reward0={state.reward.mean():.4f}"
    )
finally:
    env.close()
    env._backend.close()
"""


@pytest.mark.slow
@pytest.mark.parametrize("config_group", ("ppo", "sac"))
def test_g1_walk_flat_genesis_owner_real_runtime_smoke(config_group: str) -> None:
    """Real-runtime smoke for the genesis owner (genesis-world 1.3.3 + CUDA).

    Full chain per algo tree: Hydra compose of task=g1_walk_flat/genesis ->
    registry lookup -> ManagerBasedRlEnv construction -> keyframe reset -> 12
    steps with finite/shape-stable action/state/sensor reads -> explicit
    cleanup; the play path enters safely with play_render_mode=none and the
    native record plan resolves with field validation.
    """
    if not _genesis_runtime_available():
        pytest.skip("genesis requires the genesis-world extra and a CUDA device")

    result = subprocess.run(
        [sys.executable, "-c", _GENESIS_ENV_SMOKE_SCRIPT, config_group],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert f"[genesis env smoke:{config_group}]" in result.stdout
