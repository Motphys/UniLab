"""Hydra-owned production contract for the Stewart Manager-Based task."""

from __future__ import annotations

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
from unilab.managers import ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from unilab.tasks.manipulation.stewart.balance import (
    StewartBalanceState,
    StewartBallReset,
    StewartObservation,
    StewartTiltAction,
    StewartTiltActionCfg,
)

ROOT_DIR = Path(__file__).parents[2]
CONF_DIR = ROOT_DIR / "src" / "unilab" / "conf"

_BODY_NAMES = (
    "ball",
    "top",
    "leg00",
    "leg10",
    "leg01",
    "leg11",
    "leg02",
    "leg12",
    "top_connect00",
    "top_connect10",
    "top_connect01",
    "top_connect11",
    "top_connect02",
    "top_connect12",
)
_ACTUATOR_NAMES = ("a0", "a1", "a2", "a3", "a4", "a5")

_OWNER_CASES = (
    pytest.param("ppo", ("task=stewart_balance/motrix",), "motrix", id="ppo-motrix"),
    pytest.param("ppo", ("task=stewart_balance/mujoco",), "mujoco", id="ppo-mujoco"),
    pytest.param("ppo", ("task=stewart_balance/drake",), "drake", id="ppo-drake"),
    pytest.param(
        "sac",
        ("task=stewart_balance/drake",),
        "drake",
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
    env_cfg = registry.materialize_env_config("StewartBalance")
    assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(env_cfg, env_override)
    env_cfg.validate()
    return hydra_cfg, env_cfg, env_override


def _make_env(backend: str, *, num_envs: int = 2) -> ManagerBasedRlEnv:
    hydra_cfg, _, env_override = _materialize(
        "ppo",
        (f"task=stewart_balance/{backend}", f"algo.num_envs={num_envs}"),
    )
    env = registry.make(
        str(hydra_cfg.training.task_name),
        sim_backend=backend,
        env_cfg_override=env_override,
        num_envs=num_envs,
    )
    assert isinstance(env, ManagerBasedRlEnv)
    return env


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


@pytest.mark.parametrize("config_group,overrides,backend", _OWNER_CASES)
def test_stewart_owner_materializes_complete_plain_manager_cfg(
    config_group: str,
    overrides: tuple[str, ...],
    backend: str,
) -> None:
    registry.ensure_registries()
    hydra_cfg, env_cfg, _ = _materialize(config_group, overrides)

    assert hydra_cfg.training.task_name == "StewartBalance"
    assert hydra_cfg.training.sim_backend == backend
    assert env_cfg.sim_dt == pytest.approx(0.004)
    assert env_cfg.ctrl_dt == pytest.approx(0.02)
    assert env_cfg.max_episode_seconds == pytest.approx(24.0)
    assert env_cfg.max_episode_steps == 1200
    assert env_cfg.scale_rewards_by_dt is False
    assert env_cfg.policy_observation_group == "policy"
    assert env_cfg.critic_observation_group is None

    assert env_cfg.scene is not None
    assert env_cfg.scene.model_file.endswith("robots/stewart/scene.xml")
    assert list(env_cfg.scene.entities) == ["stewart"]
    entity = env_cfg.scene.entities["stewart"]
    assert entity.root_body_name == "ball"
    assert entity.joint_names is None
    assert tuple(entity.actuator_names or ()) == _ACTUATOR_NAMES
    assert tuple(entity.body_names or ()) == _BODY_NAMES

    assert list(env_cfg.actions) == ["tilt"]
    action_cfg = env_cfg.actions["tilt"]
    assert isinstance(action_cfg, StewartTiltActionCfg)
    assert action_cfg.entity_name == "stewart"
    assert tuple(action_cfg.actuator_names) == _ACTUATOR_NAMES
    assert action_cfg.target_rotation_limit_deg == pytest.approx(6.0)
    assert action_cfg.action_smooth == pytest.approx(0.60)
    assert action_cfg.center_control_radius == pytest.approx(0.25)
    assert action_cfg.center_control_min_gain == pytest.approx(0.15)

    assert list(env_cfg.observations) == ["policy"]
    policy_group = env_cfg.observations["policy"]
    assert policy_group is not None
    assert list(policy_group.terms) == ["balance"]
    observation_cfg = policy_group.terms["balance"]
    assert isinstance(observation_cfg, ObservationTermCfg)
    assert observation_cfg.func is StewartObservation
    assert observation_cfg.params["vel_smooth"] == pytest.approx(0.25)

    assert list(env_cfg.events) == ["reset_scene_to_default", "reset_ball"]
    reset_ball = env_cfg.events["reset_ball"]
    assert reset_ball is not None
    assert reset_ball.func is StewartBallReset
    assert reset_ball.params == {
        "entity_name": "stewart",
        "platform_radius": 0.8,
        "init_ball_radius_ratio": 0.18,
        "ball_home_z": 1.2,
    }
    assert list(env_cfg.terminations) == ["balance_state", "time_out"]
    state_cfg = env_cfg.terminations["balance_state"]
    assert isinstance(state_cfg, TerminationTermCfg)
    assert state_cfg.func is StewartBalanceState
    assert state_cfg.params["still_steps_needed"] == 5
    time_out = env_cfg.terminations["time_out"]
    assert time_out is not None
    assert time_out.time_out is True

    assert list(env_cfg.rewards) == ["center", "progress", "still", "fall"]
    weights = {
        name: term.weight
        for name, term in env_cfg.rewards.items()
        if isinstance(term, RewardTermCfg)
    }
    assert weights == {"center": 0.7, "progress": 0.6, "still": 3.0, "fall": -6.0}

    for manager_name in ("observations", "events", "rewards", "terminations"):
        for manager_entry in getattr(env_cfg, manager_name).values():
            if manager_entry is None:
                continue
            terms = (
                manager_entry.terms.values() if manager_name == "observations" else (manager_entry,)
            )
            for term in terms:
                if term is None:
                    continue
                module = term.func.__module__
                assert ".backend." not in module
                assert not any(name in module for name in (".mujoco", ".motrix", ".drake"))

    _assert_no_omegaconf(env_cfg)


def test_stewart_registry_is_manager_only_and_legacy_overrides_fail_closed() -> None:
    registry.ensure_registries()
    assert registry.list_registered_envs()["StewartBalance"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mujoco", "motrix", "drake"],
    }

    for legacy_override in (
        {"reward_config": {}},
        {"platform_radius": 0.8},
        {"action_smooth": 0.6},
    ):
        with pytest.raises(ValueError, match="has no attribute"):
            apply_cfg_overrides(ManagerBasedRlEnvCfg(), legacy_override)


def test_stewart_terms_do_not_access_physics_implementations() -> None:
    source = (ROOT_DIR / "src/unilab/tasks/manipulation/stewart/balance.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "import mujoco",
        "import motrixsim",
        "create_backend",
        "env_backend_kwargs",
        "._backend",
        "get_body_pos_w",
        "get_body_quat_w",
        "get_default_qpos",
        "get_init_qvel",
        "set_state(",
    ):
        assert forbidden not in source


@pytest.mark.parametrize("backend", ("motrix", "mujoco"))
def test_stewart_real_manager_runtime_preserves_io_reset_and_level_ik(backend: str) -> None:
    try:
        env = _make_env(backend, num_envs=2)
    except ImportError as exc:
        pytest.skip(f"{backend} runtime unavailable: {exc}")

    try:
        assert env.obs_groups_spec == {"obs": 15}
        assert env.action_space.shape == (2,)
        assert env.event_manager.active_terms == {"reset": ["reset_scene_to_default", "reset_ball"]}
        obs, info = env.reset(seed=7)
        assert {name: value.shape for name, value in obs.items()} == {"obs": (2, 15)}
        assert isinstance(info, dict)
        assert np.isfinite(obs["obs"]).all()

        entity = env.scene["stewart"]
        ball_id = entity.find_bodies("ball")[0][0]
        top_id = entity.find_bodies("top")[0][0]
        ball_pos = entity.data.body_link_pos_w[:, ball_id]
        assert np.all(np.linalg.norm(ball_pos[:, :2], axis=-1) <= 0.8 * 0.18 + 1e-6)
        np.testing.assert_allclose(ball_pos[:, 2], 1.2, atol=1e-6)

        action = env.action_manager.get_term("tilt")
        assert isinstance(action, StewartTiltAction)
        np.testing.assert_allclose(action.neutral_leg_lengths, 1.1, atol=1e-4)
        level_control = action.leg_control_for_tilt(np.zeros((2, 2), dtype=np.float32))
        np.testing.assert_allclose(level_control, 0.0, atol=1e-4)

        state = env.step(np.zeros((2, 2), dtype=np.float32))
        for _ in range(19):
            state = env.step(np.zeros((2, 2), dtype=np.float32))
        assert state.obs["obs"].shape == (2, 15)
        assert np.isfinite(state.obs["obs"]).all()
        assert np.isfinite(state.reward).all()
        assert state.terminated.dtype == np.bool_
        top_z = entity.data.body_link_pos_w[:, top_id, 2]
        assert np.all(np.abs(top_z - 1.0) < 0.1)
    finally:
        env.close()


def test_stewart_action_smoothing_and_center_authority_match_legacy_equations() -> None:
    try:
        env = _make_env("motrix", num_envs=2)
    except ImportError as exc:
        pytest.skip(f"motrix runtime unavailable: {exc}")

    try:
        env.reset(seed=11)
        action = env.action_manager.get_term("tilt")
        observation = env.observation_manager.get_term_cfg("policy", "balance").func
        assert isinstance(action, StewartTiltAction)
        assert isinstance(observation, StewartObservation)

        action.process_actions(np.full((2, 2), 2.0, dtype=np.float32))
        np.testing.assert_allclose(action.executed_action, 0.6, atol=1e-6)
        ratio = np.clip(observation.relative_xy / 0.25, 0.0, 1.0)
        expected_gain = 0.15 + 0.85 * ratio
        np.testing.assert_allclose(
            action.target_tilt_deg,
            np.broadcast_to(0.6 * expected_gain[:, None] * 6.0, (2, 2)),
            atol=1e-6,
        )

        action.process_actions(np.ones((2, 2), dtype=np.float32))
        np.testing.assert_allclose(action.executed_action, 0.84, atol=1e-6)
        action.reset(np.array([1], dtype=np.int32))
        np.testing.assert_allclose(action.executed_action[1], 0.0)
        np.testing.assert_allclose(action.executed_action[0], 0.84)
    finally:
        env.close()


def test_stewart_state_machine_and_fall_reward_are_exact() -> None:
    try:
        env = _make_env("motrix", num_envs=2)
    except ImportError as exc:
        pytest.skip(f"motrix runtime unavailable: {exc}")

    try:
        env.reset(seed=3)
        state_term = env.termination_manager.get_term_cfg("balance_state").func
        assert isinstance(state_term, StewartBalanceState)
        state_term._previous_zero_velocity_xy[:] = 0.4
        state_term._update(
            np.array([0.2, 0.6], dtype=np.float32),
            np.array([0.0, 0.2], dtype=np.float32),
            np.array([[0.0, 0.0, 1.2], [0.0, 0.0, 1.2]], dtype=np.float32),
        )
        np.testing.assert_array_equal(state_term.fallen, [False, True])
        np.testing.assert_allclose(state_term.center_score, [0.6, 0.0], atol=1e-6)
        np.testing.assert_allclose(state_term.progress, [0.25, 0.0], atol=1e-6)

        # Prove fall masking independently of the geometric scores themselves.
        state_term.center_score[1] = 0.9
        state_term.progress[1] = 0.5
        state_term.success[1] = True
        reward = env.reward_manager.compute(dt=env.step_dt)
        assert reward[0] == pytest.approx(0.7 * 0.6 + 0.6 * 0.25)
        # Positive terms are explicitly masked for fallen environments.
        assert reward[1] == pytest.approx(-6.0)

        state_term.fallen[:] = False
        state_term.success[:] = False
        state_term.still_steps[:] = 0
        state_term.still_window_active[:] = False
        state_term._previous_zero_velocity_xy[:] = 0.1
        centered = np.full(2, 0.1, dtype=np.float32)
        slow = np.full(2, 0.05, dtype=np.float32)
        ball_pos = np.full((2, 3), (0.0, 0.0, 1.2), dtype=np.float32)
        for _ in range(5):
            state_term._update(centered, slow, ball_pos)
        np.testing.assert_array_equal(state_term.still_steps, [5, 5])
        np.testing.assert_array_equal(state_term.success, [True, True])
    finally:
        env.close()


def test_stewart_drake_materializes_or_fails_at_optional_runtime_boundary() -> None:
    _, env_cfg, _ = _materialize("ppo", ("task=stewart_balance/drake",))
    try:
        env = make_manager_based_rl_env(env_cfg, num_envs=1, backend_type="drake")
    except ImportError as exc:
        # The optional runtime can be absent entirely or installed without its
        # native extension.  Both are actionable optional-boundary failures.
        message = str(exc)
        assert (
            "DrakeUni batch runtime is not installed" in message
            or "DrakeEnvPool batch extension has not been built" in message
        )
        return
    try:
        obs, _ = env.reset(seed=5)
        assert obs["obs"].shape == (1, 15)
    finally:
        env.close()


@pytest.mark.slow
def test_stewart_solver_stable_under_random_actions() -> None:
    try:
        env = _make_env("motrix", num_envs=8)
    except ImportError as exc:
        pytest.skip(f"motrix runtime unavailable: {exc}")
    try:
        rng = np.random.default_rng(0)
        for _ in range(400):
            state = env.step(rng.uniform(-1.0, 1.0, (8, 2)).astype(np.float32))
            assert np.isfinite(state.obs["obs"]).all()
    finally:
        env.close()
