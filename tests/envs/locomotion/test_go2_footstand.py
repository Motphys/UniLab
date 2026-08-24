"""Hydra-owned Manager-Based contract for Go2 footstand."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping, Sequence
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
from unilab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, make_manager_based_rl_env
from unilab.managers._noise import UniformNoiseCfg
from unilab.tasks.locomotion.go2.footstand import (
    FRAME_OBS_DIM,
    NUM_ACTIONS,
    PRIVILEGED_OBS_DIM,
    FootstandIncrementalAction,
    FootstandIncrementalActionCfg,
    FootstandMassRandomization,
    FootstandReward,
    FootstandTermination,
)

ROOT_DIR = Path(__file__).parents[3]
CONF_DIR = ROOT_DIR / "conf"

_JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)
_ACTION_JOINT_NAMES = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)
_ACTUATOR_NAMES = tuple(name.removesuffix("_joint") for name in _ACTION_JOINT_NAMES)
_BODY_NAMES = (
    "base",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
)
_OWNER_CASES = (
    pytest.param(
        "ppo",
        ("task=go2_footstand/mujoco",),
        "mujoco",
        frozenset(("pd_gains", "floor_friction", "link_mass", "torso_com", "joint_armature")),
        0.05,
        2.0,
        id="ppo-mujoco",
    ),
    pytest.param(
        "ppo",
        ("task=go2_footstand/motrix",),
        "motrix",
        frozenset(("pd_gains",)),
        0.02,
        3.0,
        id="ppo-motrix",
    ),
    pytest.param(
        "ppo",
        ("task=go2_footstand/drake",),
        "drake",
        frozenset(),
        0.05,
        2.0,
        id="ppo-drake",
    ),
    pytest.param(
        "sac",
        ("task=go2_footstand/drake",),
        "drake",
        frozenset(),
        0.05,
        2.0,
        id="sac-drake",
    ),
)


def _compose(config_group: str, overrides: Sequence[str]) -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / config_group), version_base="1.3"):
        return compose("config", overrides=list(overrides))


def _materialize(
    config_group: str,
    overrides: Sequence[str],
) -> tuple[DictConfig, ManagerBasedRlEnvCfg, dict[str, Any]]:
    hydra_cfg = _compose(config_group, overrides)
    env_override = BackendAdapter(hydra_cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    env_cfg = registry.materialize_env_config("Go2FootStand")
    assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(env_cfg, env_override)
    env_cfg.validate()
    return hydra_cfg, env_cfg, env_override


def _make_env(backend: str, *, num_envs: int = 2) -> ManagerBasedRlEnv:
    hydra_cfg, _, env_override = _materialize(
        "ppo",
        (f"task=go2_footstand/{backend}", f"algo.num_envs={num_envs}"),
    )
    env = registry.make(
        str(hydra_cfg.training.task_name),
        sim_backend=backend,
        env_cfg_override=env_override,
        num_envs=num_envs,
    )
    assert isinstance(env, ManagerBasedRlEnv)
    return env


def _assert_plain(value: Any) -> None:
    assert not OmegaConf.is_config(value)
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            _assert_plain(getattr(value, item.name))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _assert_plain(key)
            _assert_plain(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _assert_plain(item)


def _action(env: ManagerBasedRlEnv) -> FootstandIncrementalAction:
    term = env.action_manager.get_term("joint_pos")
    assert isinstance(term, FootstandIncrementalAction)
    return term


def _termination(env: ManagerBasedRlEnv) -> FootstandTermination:
    term = env.termination_manager.get_term_cfg("footstand").func
    assert isinstance(term, FootstandTermination)
    return term


def _reward(env: ManagerBasedRlEnv) -> FootstandReward:
    term = env.reward_manager.get_term_cfg("footstand").func
    assert isinstance(term, FootstandReward)
    return term


@pytest.mark.parametrize(
    "config_group,overrides,backend,backend_events,joint_reset_radius,orientation_scale",
    _OWNER_CASES,
)
def test_footstand_owner_materializes_complete_plain_manager_cfg(
    config_group: str,
    overrides: tuple[str, ...],
    backend: str,
    backend_events: frozenset[str],
    joint_reset_radius: float,
    orientation_scale: float,
) -> None:
    registry.ensure_registries()
    hydra_cfg, env_cfg, _ = _materialize(config_group, overrides)

    assert hydra_cfg.training.task_name == "Go2FootStand"
    assert hydra_cfg.training.sim_backend == backend
    assert hydra_cfg.algo.num_envs == 4096
    assert env_cfg.sim_dt == pytest.approx(0.004)
    assert env_cfg.ctrl_dt == pytest.approx(0.02)
    assert env_cfg.max_episode_seconds == pytest.approx(10.0)
    assert env_cfg.adaptive_chunk_size is False
    assert env_cfg.policy_observation_group == "policy"
    assert env_cfg.critic_observation_group == "critic"

    assert env_cfg.scene is not None
    assert env_cfg.scene.default_keyframe_name == "home"
    robot = env_cfg.scene.entities["robot"]
    assert robot.root_body_name == "base"
    assert tuple(robot.joint_names or ()) == _JOINT_NAMES
    assert tuple(robot.body_names or ()) == _BODY_NAMES
    assert tuple(robot.geom_names or ()) == ("floor",)
    assert tuple(robot.actuator_names or ()) == _ACTUATOR_NAMES

    policy = env_cfg.observations["policy"]
    critic = env_cfg.observations["critic"]
    assert policy is not None and critic is not None
    assert list(policy.terms) == ["frame"]
    assert list(critic.terms) == ["frame", "privileged"]
    assert policy.enable_corruption is True
    assert policy.terms["frame"] is not None
    assert critic.terms["frame"] is not None
    assert critic.terms["privileged"] is not None
    assert policy.terms["frame"].history_length == 15
    assert critic.terms["frame"].history_length == 15
    assert critic.terms["privileged"].history_length == 0
    noise = policy.terms["frame"].noise
    assert isinstance(noise, UniformNoiseCfg)
    assert np.asarray(noise.n_min).shape == (FRAME_OBS_DIM,)
    assert np.asarray(noise.n_max).shape == (FRAME_OBS_DIM,)

    action = env_cfg.actions["joint_pos"]
    assert isinstance(action, FootstandIncrementalActionCfg)
    assert tuple(action.actuator_names) == _ACTUATOR_NAMES
    assert tuple(action.joint_names) == _ACTION_JOINT_NAMES
    assert action.action_scale == pytest.approx(0.3)
    assert action.clip_actions == pytest.approx(1.0)
    assert action.kp == pytest.approx(35.0)
    assert action.kd == pytest.approx(0.5)

    always_enabled = {
        "reset_scene_to_default",
        "reset_root_state_uniform",
        "reset_joints",
    }
    active_events = {name for name, term in env_cfg.events.items() if term is not None}
    assert active_events == always_enabled | backend_events
    reset_joints = env_cfg.events["reset_joints"]
    assert reset_joints is not None
    assert tuple(reset_joints.params["position_offset_range"]) == pytest.approx(
        (-joint_reset_radius, joint_reset_radius)
    )
    if "link_mass" in backend_events:
        link_mass = env_cfg.events["link_mass"]
        assert link_mass is not None
        assert link_mass.func is FootstandMassRandomization

    termination = env_cfg.terminations["footstand"]
    assert termination is not None
    assert termination.params["grace_steps"] == 100
    assert termination.params["energy_threshold"] == pytest.approx(200.0)
    reward = env_cfg.rewards["footstand"]
    assert reward is not None
    assert reward.func is FootstandReward
    assert reward.params["scales"]["orientation"] == pytest.approx(orientation_scale)

    assert FRAME_OBS_DIM * 15 == 675
    assert FRAME_OBS_DIM * 15 + PRIVILEGED_OBS_DIM == 724
    _assert_plain(env_cfg)


def test_footstand_registry_has_no_legacy_config_or_factory() -> None:
    registry.ensure_registries()
    bare_cfg = registry.materialize_env_config("Go2FootStand")

    assert isinstance(bare_cfg, ManagerBasedRlEnvCfg)
    assert bare_cfg.actions == {}
    assert bare_cfg.observations == {}
    assert registry.list_registered_envs()["Go2FootStand"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mujoco", "motrix", "drake"],
    }
    meta = registry._envs["Go2FootStand"]
    assert all(factory is make_manager_based_rl_env for factory in meta.env_factory_dict.values())
    for legacy_override in (
        {"reward_config": {}},
        {"domain_rand": {"randomize_floor_friction": True}},
        {"control_config": {"action_scale": 0.4}},
        {"obs_history_len": 15},
    ):
        with pytest.raises(ValueError, match="has no attribute"):
            apply_cfg_overrides(ManagerBasedRlEnvCfg(), legacy_override)


@pytest.mark.parametrize("backend", ("mujoco", "motrix"))
def test_footstand_real_runtime_preserves_history_action_and_partial_reset(
    backend: str,
) -> None:
    if backend == "motrix":
        pytest.importorskip("motrixsim", reason="motrixsim not installed")
    registry.ensure_registries()
    env = _make_env(backend)
    try:
        obs, info = env.reset(seed=7)
        assert set(info) == {"log"}
        assert env.obs_groups_spec == {"obs": 675, "critic": 724}
        assert env.action_space.shape == (NUM_ACTIONS,)
        assert obs["obs"].shape == (2, 675)
        assert obs["critic"].shape == (2, 724)
        assert np.isfinite(obs["obs"]).all()
        assert np.isfinite(obs["critic"]).all()

        reset_frames = obs["obs"].reshape(2, 15, FRAME_OBS_DIM)
        np.testing.assert_allclose(reset_frames, np.repeat(reset_frames[:, :1], 15, axis=1))
        clean_history = obs["critic"][:, :675].reshape(2, 15, FRAME_OBS_DIM)
        np.testing.assert_allclose(clean_history, np.repeat(clean_history[:, :1], 15, axis=1))

        action = _action(env)
        initial_target = action.target.copy()
        first_policy_action = np.full((2, NUM_ACTIONS), 2.0, dtype=np.float32)
        action.process_actions(first_policy_action)
        expected_target = np.clip(
            initial_target + 0.3,
            action.joint_lower[action.joint_ids],
            action.joint_upper[action.joint_ids],
        )
        np.testing.assert_allclose(action.target, expected_target, atol=1e-6)

        obs, _ = env.reset(seed=7)
        first_policy_action = np.full((2, NUM_ACTIONS), 0.05, dtype=np.float32)
        first_state = env.step(first_policy_action)
        assert not first_state.terminated.any()
        first_frames = first_state.obs["obs"].reshape(2, 15, FRAME_OBS_DIM)
        np.testing.assert_allclose(first_frames[:, -1, -NUM_ACTIONS:], 0.0)

        second_policy_action = np.full((2, NUM_ACTIONS), 0.025, dtype=np.float32)
        second_state = env.step(second_policy_action)
        assert not second_state.terminated.any()
        second_frames = second_state.obs["obs"].reshape(2, 15, FRAME_OBS_DIM)
        np.testing.assert_allclose(second_frames[:, -1, -NUM_ACTIONS:], 0.05)
        assert np.isfinite(second_state.reward).all()
        assert np.all(second_state.reward >= 0.0)

        untouched_target = action.target[1].copy()
        untouched_torque = action.state.torques[1].copy()
        assert env.state is not None
        untouched_obs = {name: value[1].copy() for name, value in env.state.obs.items()}
        reset_obs, _ = env.reset(env_ids=np.asarray([0], dtype=np.int32))
        assert reset_obs["obs"].shape == (1, 675)
        partial_frames = reset_obs["obs"].reshape(1, 15, FRAME_OBS_DIM)
        np.testing.assert_allclose(partial_frames, np.repeat(partial_frames[:, :1], 15, axis=1))
        np.testing.assert_allclose(action.target[1], untouched_target)
        np.testing.assert_allclose(action.state.torques[1], untouched_torque)
        assert env.state is not None
        for name, expected in untouched_obs.items():
            np.testing.assert_allclose(env.state.obs[name][1], expected)
    finally:
        env.close()


def test_footstand_termination_uses_grace_boundary() -> None:
    registry.ensure_registries()
    env = _make_env("mujoco")
    try:
        env.reset(seed=11)
        term = _termination(env)
        state = term.state
        state.height[:] = 0.0
        state.orientation[:] = 0.0
        state.upvector[:] = (0.0, 0.0, 1.0)
        state.termination_contact[:] = False
        state.torques[:] = 0.0
        state.joint_vel[:] = 0.0

        env.episode_length_buf[:] = 100
        assert not term(env).any()
        env.episode_length_buf[:] = 101
        assert term(env).all()
    finally:
        env.close()


def test_footstand_reward_clips_aggregate_before_dt_scaling() -> None:
    registry.ensure_registries()
    env = _make_env("mujoco")
    try:
        env.reset(seed=13)
        term = _termination(env)
        reward = _reward(env)
        state = term.state
        term.terminated[:] = False
        state.height[:] = 0.53
        state.foot_contact[:] = False
        state.foot_contact[0, 0] = True
        reward._scales = {"height": 2.0, "contact": -100.0}

        np.testing.assert_allclose(reward(env), np.asarray([0.0, 2.0], dtype=np.float32))
        np.testing.assert_allclose(
            env.reward_manager.compute(dt=env.step_dt),
            np.asarray([0.0, 0.04], dtype=np.float32),
        )
    finally:
        env.close()


@pytest.mark.skipif(
    importlib.util.find_spec("drakeuni") is None,
    reason="optional DrakeUni batch runtime is not installed",
)
def test_footstand_drake_real_runtime_when_available() -> None:
    registry.ensure_registries()
    env = _make_env("drake")
    try:
        obs, _ = env.reset(seed=17)
        assert {name: value.shape for name, value in obs.items()} == {
            "obs": (2, 675),
            "critic": (2, 724),
        }
        state = env.step(np.zeros((2, NUM_ACTIONS), dtype=np.float32))
        assert np.isfinite(state.reward).all()
    finally:
        env.close()


def test_footstand_production_terms_do_not_leak_backend_or_layout() -> None:
    source = (ROOT_DIR / "src/unilab/tasks/locomotion/go2/footstand.py").read_text(encoding="utf-8")
    for forbidden in (
        "._backend",
        "getattr(",
        "hasattr(",
        "ASSETS_ROOT_PATH",
        " qpos",
        " qvel",
    ):
        assert forbidden not in source
