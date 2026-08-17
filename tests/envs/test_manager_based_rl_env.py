"""Focused contract tests for the NumPy Manager-Based environment lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest

import unilab.envs.manager_based_rl_env as manager_env_module
from unilab.base.backend.base import SimBackend
from unilab.base.entity import EntityCfg
from unilab.base.scene import SceneCfg
from unilab.envs import (
    ManagerBasedRLEnv,
    ManagerBasedRlEnv,
    ManagerBasedRLEnvCfg,
    ManagerBasedRlEnvCfg,
    mdp,
)
from unilab.managers import (
    ActionTerm,
    ActionTermCfg,
    CommandTerm,
    CommandTermCfg,
    CurriculumTermCfg,
    EventTermCfg,
    MetricsTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RecorderTerm,
    RecorderTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)


class _FakeBackend:
    backend_type = "fake"

    def __init__(self, num_envs: int, *, reject_pre_step: bool = False) -> None:
        self.num_envs = num_envs
        self.num_actuators = 1
        self.reject_pre_step = reject_pre_step
        self.pre_step_control = None
        self.applied_controls: list[np.ndarray] = []
        self.cleanup_calls = 0

    def get_actuator_names(self) -> tuple[str, ...]:
        return ("motor",)

    def get_actuator_ctrl_range(self) -> np.ndarray:
        return np.array([[-2.0, 2.0]], dtype=np.float32)

    def get_dof_pos(self) -> np.ndarray:
        return np.empty((self.num_envs, 0), dtype=np.float32)

    def get_dof_vel(self) -> np.ndarray:
        return np.empty((self.num_envs, 0), dtype=np.float32)

    def set_pre_step_control(self, fn) -> None:
        if fn is not None and self.reject_pre_step:
            raise NotImplementedError("host callback disabled")
        self.pre_step_control = fn

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> None:
        native = ctrl
        for _ in range(nsteps):
            if self.pre_step_control is not None:
                native = self.pre_step_control(self, ctrl)
            self.applied_controls.append(native.copy())

    def cleanup_scene_assets(self) -> None:
        self.cleanup_calls += 1


class _ResetBackend(_FakeBackend):
    def __init__(self, num_envs: int) -> None:
        super().__init__(num_envs)
        self.default_qpos_calls = 0
        self.init_qvel_calls = 0
        self.joint_layout_calls = 0
        self.set_state_calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def get_default_qpos(self) -> np.ndarray:
        self.default_qpos_calls += 1
        return np.array([0.0, 0.0, 0.5, 0.0], dtype=np.float64)

    def get_init_qvel(self) -> np.ndarray:
        self.init_qvel_calls += 1
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def get_actuator_joint_names(self) -> tuple[str, ...]:
        return ("joint",)

    def get_joint_dof_pos_indices(self, names) -> np.ndarray:
        assert tuple(names) == ("joint",)
        return np.array([0], dtype=np.int32)

    def get_joint_dof_vel_indices(self, names) -> np.ndarray:
        assert tuple(names) == ("joint",)
        return np.array([0], dtype=np.int32)

    def get_joint_state_qpos_indices(self, names) -> np.ndarray:
        assert tuple(names) == ("joint",)
        self.joint_layout_calls += 1
        return np.array([3], dtype=np.int32)

    def get_joint_state_qvel_indices(self, names) -> np.ndarray:
        assert tuple(names) == ("joint",)
        self.joint_layout_calls += 1
        return np.array([2], dtype=np.int32)

    def get_dof_pos(self) -> np.ndarray:
        return np.zeros((self.num_envs, 1), dtype=np.float32)

    def get_default_dof_pos(self) -> np.ndarray:
        return np.zeros((1,), dtype=np.float32)

    def get_dof_vel(self) -> np.ndarray:
        return np.zeros((self.num_envs, 1), dtype=np.float32)

    def set_state(
        self,
        env_ids: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization=None,
    ) -> None:
        assert randomization is None
        self.set_state_calls.append((env_ids.copy(), qpos.copy(), qvel.copy()))


@dataclass(kw_only=True)
class _DriveCfg(ActionTermCfg):
    gain: float = 1.0

    def build(self, env) -> ActionTerm:
        return _DriveAction(self, env)


class _DriveAction(ActionTerm):
    def __init__(self, cfg: _DriveCfg, env) -> None:
        super().__init__(cfg, env)
        self._processed = np.zeros((self.num_envs, 1), dtype=np.float32)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_action(self) -> np.ndarray:
        return self._processed

    def process_actions(self, actions: np.ndarray) -> None:
        self._processed[:] = actions * cast(_DriveCfg, self.cfg).gain

    def apply_actions(self) -> None:
        self._env.trace.append("action_apply")
        self._env.action_sim_steps.append(self._env._sim_step_counter)
        self._entity.data.write_ctrl(self._processed)


@dataclass(kw_only=True)
class _CommandCfg(CommandTermCfg):
    def build(self, env) -> CommandTerm:
        return _Command(self, env)


class _Command(CommandTerm):
    def __init__(self, cfg: _CommandCfg, env) -> None:
        super().__init__(cfg, env)
        self._command = np.zeros((self.num_envs, 1), dtype=np.float32)

    @property
    def command(self) -> np.ndarray:
        return self._command

    def _update_metrics(self) -> None:
        return None

    def _resample_command(self, env_ids: np.ndarray) -> None:
        self._command[env_ids, 0] = self._env.rng.uniform(size=len(env_ids))

    def _update_command(self, env_ids: np.ndarray | None) -> None:
        ids = None if env_ids is None else env_ids.copy()
        self._env.command_update_ids.append(ids)


class _Recorder(RecorderTerm):
    def record_pre_reset(self, env_ids: np.ndarray) -> None:
        self._env.trace.append(("pre_reset", env_ids.tolist()))

    def record_post_reset(self, env_ids: np.ndarray) -> None:
        self._env.trace.append(("post_reset", env_ids.tolist()))

    def record_post_step(self) -> None:
        self._env.trace.append("post_step")

    def close(self) -> None:
        self._env.trace.append("recorder_close")


class _TestEnv(ManagerBasedRlEnv):
    def __init__(self, cfg: ManagerBasedRlEnvCfg, backend: SimBackend, num_envs: int) -> None:
        self.trace: list[Any] = []
        self.command_update_ids: list[np.ndarray | None] = []
        self.action_sim_steps: list[int] = []
        super().__init__(cfg, backend, num_envs)


def _policy_obs(env: _TestEnv) -> np.ndarray:
    return np.column_stack(
        (env.episode_length_buf.astype(np.float32), env.action_manager.action[:, 0])
    )


def _critic_obs(env: _TestEnv) -> np.ndarray:
    return env.episode_length_buf[:, None].astype(np.float32)


def _reward(env: _TestEnv) -> np.ndarray:
    return env.action_manager.action[:, 0].copy()


def _failure(env: _TestEnv) -> np.ndarray:
    return env.action_manager.action[:, 0] > 0.8


def _time_out(env: _TestEnv) -> np.ndarray:
    return env.episode_length_buf >= 2


def _metric(env: _TestEnv) -> np.ndarray:
    return env.episode_length_buf.astype(np.float32)


def _curriculum(env: _TestEnv, env_ids: np.ndarray | slice) -> float:
    del env
    return float(len(env_ids)) if isinstance(env_ids, np.ndarray) else 0.0


def _event(env: _TestEnv, env_ids: np.ndarray | None) -> None:
    rendered_ids = None if env_ids is None else env_ids.tolist()
    env.trace.append(("event", rendered_ids))


def _observe_uncommitted_reset(env: _TestEnv, env_ids: np.ndarray | None) -> None:
    del env_ids
    backend = cast(_ResetBackend, env._backend)
    env.trace.append(("commit_count_during_event", len(backend.set_state_calls)))


def _write_reset_joint_state(env: _TestEnv, env_ids: np.ndarray | None) -> None:
    assert env_ids is not None
    count = len(env_ids)
    env.scene["robot"].write_joint_state_to_sim(
        np.full((count, 1), 0.25, dtype=np.float32),
        np.full((count, 1), -0.5, dtype=np.float32),
        env_ids=env_ids,
    )


def _make_cfg(
    *,
    sim_substeps: int = 2,
    finite_horizon: bool = False,
    auto_reset: bool = True,
    metrics: dict[str, MetricsTermCfg | None] | None = None,
    observations: dict[str, ObservationGroupCfg | None] | None = None,
    include_optional_managers: bool = True,
) -> ManagerBasedRlEnvCfg:
    if observations is None:
        observations = {
            "actor": ObservationGroupCfg(terms={"policy": ObservationTermCfg(func=_policy_obs)}),
            "value": ObservationGroupCfg(terms={"critic": ObservationTermCfg(func=_critic_obs)}),
        }
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            model_file="fake.xml",
            entities={"robot": EntityCfg(actuator_names=("motor",))},
        ),
        sim_dt=0.01,
        ctrl_dt=0.01 * sim_substeps,
        max_episode_seconds=1.0,
        seed=7,
        observations=observations,
        actions={"drive": _DriveCfg(entity_name="robot")},
        rewards={"track": RewardTermCfg(func=_reward, weight=1.0)},
        terminations={
            "failure": TerminationTermCfg(func=_failure),
            "time_out": TerminationTermCfg(func=_time_out, time_out=True),
        },
        events={"reset": EventTermCfg(func=_event, mode="reset")},
        metrics={} if metrics is None else metrics,
        policy_observation_group="actor",
        critic_observation_group="value",
        is_finite_horizon=finite_horizon,
        auto_reset=auto_reset,
    )
    if include_optional_managers:
        cfg.commands = {"target": _CommandCfg(resampling_time_range=(1.0, 1.0))}
        cfg.curriculum = {"difficulty": CurriculumTermCfg(func=_curriculum)}
        cfg.recorders = {"trace": RecorderTermCfg(func=_Recorder)}
    return cfg


def _make_env(
    cfg: ManagerBasedRlEnvCfg | None = None,
    *,
    num_envs: int = 2,
    reject_pre_step: bool = False,
) -> tuple[_TestEnv, _FakeBackend]:
    backend = _FakeBackend(num_envs, reject_pre_step=reject_pre_step)
    env = _TestEnv(
        cfg or _make_cfg(),
        cast(SimBackend, backend),
        num_envs,
    )
    return env, backend


def test_public_names_are_spelling_only_aliases() -> None:
    assert ManagerBasedRLEnv is ManagerBasedRlEnv
    assert ManagerBasedRLEnvCfg is ManagerBasedRlEnvCfg


@pytest.mark.parametrize(
    ("mutate", "error", "match"),
    [
        (lambda cfg: setattr(cfg, "sim_dt", True), TypeError, "sim_dt must be a real"),
        (
            lambda cfg: setattr(cfg, "ctrl_dt", 0.015),
            ValueError,
            "integer multiple",
        ),
        (
            lambda cfg: setattr(cfg, "max_episode_seconds", None),
            ValueError,
            "max_episode_seconds",
        ),
        (lambda cfg: setattr(cfg, "scene", None), TypeError, "scene must be a SceneCfg"),
    ],
)
def test_manager_based_config_rejects_invalid_contracts(mutate, error, match: str) -> None:
    cfg = _make_cfg()
    mutate(cfg)
    with pytest.raises(error, match=match):
        cfg.validate()


def test_manager_construction_uses_pinned_order(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    names = (
        "EventManager",
        "CommandManager",
        "ActionManager",
        "ObservationManager",
        "TerminationManager",
        "RewardManager",
        "CurriculumManager",
        "MetricsManager",
        "RecorderManager",
    )
    cfg = _make_cfg(metrics={"progress": MetricsTermCfg(func=_metric)})
    for name in names:
        original = getattr(manager_env_module, name)

        def wrapped(*args, _name=name, _original=original, **kwargs):
            order.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(manager_env_module, name, wrapped)

    _make_env(cfg)
    assert order == list(names)


def test_np_env_owns_substeps_autoreset_and_final_observation() -> None:
    env, backend = _make_env()
    initial_obs, initial_info = env.reset()
    assert env.state is not None
    initial = env.state

    assert initial_obs["obs"].shape == (2, 2)
    assert initial_obs["critic"].shape == (2, 1)
    assert "log" in initial_info
    np.testing.assert_array_equal(initial.info["steps"], [0, 0])
    assert env._dr_manager is None

    state = env.step(np.array([[0.25], [0.5]], dtype=np.float32))
    assert len(backend.applied_controls) == 2
    assert env.action_sim_steps == [1, 2]
    np.testing.assert_allclose(backend.applied_controls[0][:, 0], [0.25, 0.5])
    np.testing.assert_allclose(backend.applied_controls[1][:, 0], [0.25, 0.5])
    np.testing.assert_allclose(state.reward, [0.005, 0.01])
    np.testing.assert_array_equal(state.terminated, [False, False])
    np.testing.assert_array_equal(state.truncated, [False, False])
    np.testing.assert_array_equal(state.info["steps"], [1, 1])

    state = env.step(np.array([[0.25], [0.5]], dtype=np.float32))
    np.testing.assert_array_equal(state.truncated, [True, True])
    np.testing.assert_array_equal(state.terminated, [False, False])
    np.testing.assert_array_equal(state.info["steps"], [0, 0])
    np.testing.assert_array_equal(state.obs["obs"][:, 0], [0.0, 0.0])
    assert state.final_observation is not None
    np.testing.assert_array_equal(state.final_observation["obs"][:, 0], [2.0, 2.0])
    assert state.info["_final_observation"].tolist() == [True, True]
    assert state.info["log"]["Episode_Termination/time_out"] == 2

    pre_index = env.trace.index(("pre_reset", [0, 1]))
    post_reset_index = env.trace.index(("post_reset", [0, 1]), pre_index)
    post_step_index = len(env.trace) - 1
    assert env.trace[post_step_index] == "post_step"
    assert pre_index < post_reset_index < post_step_index


def test_reset_events_compose_then_commit_default_state_once() -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.events = {
        "default_first": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset"),
        "joint_state": EventTermCfg(func=_write_reset_joint_state, mode="reset"),
        "observe_uncommitted": EventTermCfg(func=_observe_uncommitted_reset, mode="reset"),
    }
    cfg.scene.entities["robot"] = EntityCfg(
        joint_names=("joint",),
        actuator_names=("motor",),
    )
    backend = _ResetBackend(2)
    env = _TestEnv(cfg, cast(SimBackend, backend), 2)

    assert backend.default_qpos_calls == 0
    assert backend.init_qvel_calls == 0
    assert backend.joint_layout_calls == 0
    env.reset()

    assert env.trace[-1] == ("commit_count_during_event", 0)
    assert backend.default_qpos_calls == 1
    assert backend.init_qvel_calls == 1
    assert len(backend.set_state_calls) == 1
    ids, qpos, qvel = backend.set_state_calls[0]
    np.testing.assert_array_equal(ids, [0, 1])
    np.testing.assert_array_equal(
        qpos,
        [[0.0, 0.0, 0.5, 0.25], [0.0, 0.0, 0.5, 0.25]],
    )
    np.testing.assert_array_equal(qvel, [[0.0, 0.0, -0.5], [0.0, 0.0, -0.5]])
    assert backend.joint_layout_calls == 2

    env.reset(env_ids=np.array([1], dtype=np.int32))
    assert ("commit_count_during_event", 1) in env.trace
    assert len(backend.set_state_calls) == 2
    np.testing.assert_array_equal(backend.set_state_calls[1][0], [1])
    assert backend.default_qpos_calls == 1
    assert backend.init_qvel_calls == 1
    assert backend.joint_layout_calls == 2


def test_pure_reset_event_does_not_request_backend_state_capability() -> None:
    env, backend = _make_env()

    env.reset()
    env.reset(env_ids=np.array([0], dtype=np.int32))

    assert isinstance(backend, _FakeBackend)
    assert not hasattr(backend, "get_default_qpos")


def test_partial_reset_preserves_other_env_counter_and_terminal_obs() -> None:
    env, _ = _make_env()
    env.init_state()

    state = env.step(np.array([[1.0], [0.0]], dtype=np.float32))

    np.testing.assert_array_equal(state.terminated, [True, False])
    np.testing.assert_array_equal(state.truncated, [False, False])
    np.testing.assert_array_equal(state.info["steps"], [0, 1])
    np.testing.assert_array_equal(state.obs["obs"][:, 0], [0.0, 1.0])
    assert state.final_observation is not None
    np.testing.assert_array_equal(state.final_observation["obs"][0], [1.0, 1.0])
    assert state.info["_final_observation"].tolist() == [True, False]
    assert ("event", [0]) in env.trace


def test_finite_horizon_maps_time_out_to_terminated() -> None:
    env, _ = _make_env(_make_cfg(finite_horizon=True))
    env.init_state()
    env.step(np.zeros((2, 1), dtype=np.float32))
    state = env.step(np.zeros((2, 1), dtype=np.float32))
    np.testing.assert_array_equal(state.terminated, [True, True])
    np.testing.assert_array_equal(state.truncated, [False, False])


def test_manual_reset_is_required_when_autoreset_is_disabled() -> None:
    env, _ = _make_env(_make_cfg(auto_reset=False))
    env.init_state()
    state = env.step(np.array([[1.0], [0.0]], dtype=np.float32))
    np.testing.assert_array_equal(state.info["steps"], [1, 1])
    with pytest.raises(RuntimeError, match="must be reset"):
        env.step(np.zeros((2, 1), dtype=np.float32))

    reset_obs, info = env.reset(env_ids=np.array([0], dtype=np.int32))
    assert reset_obs["obs"].shape == (1, 2)
    assert "log" in info
    assert not state.terminated[0]
    env.step(np.zeros((2, 1), dtype=np.float32))


def test_reset_seed_updates_the_shared_generator_in_place() -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.events = {}
    env, _ = _make_env(cfg)
    env.init_state()
    generator = env.rng

    env.reset(seed=123, env_ids=np.array([0], dtype=np.int32))

    assert env.rng is generator
    expected = np.random.default_rng(123).random()
    assert env.rng.random() == pytest.approx(expected)


def test_per_substep_metrics_fail_without_post_substep_backend_hook() -> None:
    cfg = _make_cfg(
        sim_substeps=2,
        metrics={"energy": MetricsTermCfg(func=_metric, per_substep=True)},
    )
    with pytest.raises(NotImplementedError, match="post-physics per-substep metrics.*fake"):
        _make_env(cfg)


def test_multisubstep_action_fails_when_backend_rejects_callback() -> None:
    with pytest.raises(NotImplementedError, match="ActionManager.*every physics substep.*fake"):
        _make_env(reject_pre_step=True)


@pytest.mark.parametrize(
    ("mutate", "error", "match"),
    [
        (
            lambda cfg: setattr(cfg, "policy_observation_group", "missing"),
            KeyError,
            "requests group 'missing'",
        ),
        (
            lambda cfg: setattr(
                cfg.observations["actor"],
                "concatenate_terms",
                False,  # type: ignore[union-attr]
            ),
            ValueError,
            "must concatenate terms",
        ),
        (
            lambda cfg: setattr(cfg, "critic_observation_group", "actor"),
            ValueError,
            "must be different",
        ),
    ],
)
def test_observation_mapping_fails_closed(mutate, error, match: str) -> None:
    cfg = _make_cfg()
    mutate(cfg)
    with pytest.raises(error, match=match):
        _make_env(cfg)


def test_close_unhooks_callback_and_closes_owned_resources() -> None:
    env, backend = _make_env()
    env.close()
    assert backend.pre_step_control is None
    assert backend.cleanup_calls == 1
    assert env.trace[-1] == "recorder_close"
