"""Focused contract tests for the NumPy Manager-Based environment lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest
from unisim.backend.base import SimBackend

import unilab.envs.manager_based_rl_env as manager_env_module
from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.base.entity import EntityCfg
from unilab.base.scene import SceneCfg
from unilab.envs import (
    ManagerBasedRLEnv,
    ManagerBasedRlEnv,
    ManagerBasedRLEnvCfg,
    ManagerBasedRlEnvCfg,
    make_manager_based_rl_env,
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

    def __init__(
        self,
        num_envs: int,
        *,
        reject_pre_step: bool = False,
        reject_materialize: bool = False,
    ) -> None:
        self.num_envs = num_envs
        self.num_actuators = 1
        self.reject_pre_step = reject_pre_step
        self.reject_materialize = reject_materialize
        self.pre_step_control = None
        self.applied_controls: list[np.ndarray] = []
        self.step_nsteps: list[int] = []
        self.cleanup_calls = 0
        self.materialize_calls = 0
        self.lifecycle: list[str] = []

    def materialize(self) -> None:
        if self.reject_materialize:
            raise RuntimeError("pool construction failed")
        self.materialize_calls += 1
        self.lifecycle.append("materialize")

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
        assert self.materialize_calls == 1
        self.step_nsteps.append(nsteps)
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

    def get_joint_range(self) -> np.ndarray:
        return np.array([[-1.0, 1.0]], dtype=np.float32)

    def set_state(
        self,
        env_ids: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization=None,
    ) -> None:
        assert randomization is None
        assert self.materialize_calls == 1
        self.set_state_calls.append((env_ids.copy(), qpos.copy(), qvel.copy()))


class _StateBackend(_ResetBackend):
    def __init__(self, num_envs: int) -> None:
        super().__init__(num_envs)
        self.dof_pos = np.zeros((num_envs, 1), dtype=np.float32)
        self.dof_pos_calls = 0

    def get_dof_pos(self) -> np.ndarray:
        self.dof_pos_calls += 1
        return self.dof_pos.copy()

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> None:
        super().step(ctrl, nsteps)
        self.dof_pos += 1.0

    def set_state(
        self,
        env_ids: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization=None,
    ) -> None:
        super().set_state(env_ids, qpos, qvel, randomization=randomization)
        self.dof_pos[env_ids, 0] = qpos[:, 3]


class _KeyframeBackend(_ResetBackend):
    def __init__(
        self,
        num_envs: int,
        *,
        keyframe_qpos: Any = None,
        keyframe_error: Exception | None = None,
    ) -> None:
        super().__init__(num_envs)
        self.keyframe_qpos = (
            np.array([0.0, 0.0, 0.3, 0.75], dtype=np.float64)
            if keyframe_qpos is None
            else keyframe_qpos
        )
        self.keyframe_error = keyframe_error
        self.keyframe_qpos_calls = 0

    def get_keyframe_qpos(self, name: str) -> Any:
        self.keyframe_qpos_calls += 1
        if self.keyframe_error is not None:
            raise self.keyframe_error
        if name != "home":
            raise ValueError(f"unknown keyframe {name!r}")
        return self.keyframe_qpos


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


class _FeedbackDriveAction(_DriveAction):
    requires_substep_state_feedback = True


@dataclass(kw_only=True)
class _FeedbackDriveCfg(_DriveCfg):
    def build(self, env) -> ActionTerm:
        return _FeedbackDriveAction(self, env)


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

    def _update_metrics(self, env_ids: np.ndarray | None = None) -> None:
        return None

    def _resample_command(self, env_ids: np.ndarray) -> None:
        self._command[env_ids, 0] = self._env.rng.uniform(size=len(env_ids))

    def _update_command(self, env_ids: np.ndarray | None) -> None:
        ids = None if env_ids is None else env_ids.copy()
        self._env.command_update_ids.append(ids)


@dataclass(kw_only=True)
class _StateWritingCommandCfg(CommandTermCfg):
    def build(self, env) -> CommandTerm:
        return _StateWritingCommand(self, env)


class _StateWritingCommand(_Command):
    def _resample_command(self, env_ids: np.ndarray) -> None:
        self._command[env_ids, 0] = 0.75
        self._env.scene["robot"].write_joint_state_to_sim(
            np.full((len(env_ids), 1), 0.75, dtype=np.float32),
            np.full((len(env_ids), 1), -0.75, dtype=np.float32),
            env_ids=env_ids,
        )


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


def _episode_step_observation(env: ManagerBasedRlEnv) -> np.ndarray:
    return env.episode_length_buf[:, None].astype(np.float32)


def _reward(env: _TestEnv) -> np.ndarray:
    return env.action_manager.action[:, 0].copy()


def _joint_state_obs(env: _TestEnv) -> np.ndarray:
    return env.scene["robot"].data.joint_pos


def _joint_state_reward(env: _TestEnv) -> np.ndarray:
    return env.scene["robot"].data.joint_pos[:, 0]


def _joint_state_termination(env: _TestEnv) -> np.ndarray:
    return env.scene["robot"].data.joint_pos[:, 0] > 100.0


def _mutate_joint_state_step_event(env: _TestEnv, env_ids: np.ndarray | None) -> None:
    assert env_ids is None
    cast(_StateBackend, env._backend).dof_pos += 10.0


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


def _startup_event(env: _TestEnv, env_ids: np.ndarray | None) -> None:
    assert env_ids is None
    backend = cast(_FakeBackend, env._backend)
    backend.lifecycle.append("startup")


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
    actions: dict[str, ActionTermCfg | None] | None = None,
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
        actions={"drive": _DriveCfg(entity_name="robot")} if actions is None else actions,
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
    reject_materialize: bool = False,
) -> tuple[_TestEnv, _FakeBackend]:
    backend = _FakeBackend(
        num_envs,
        reject_pre_step=reject_pre_step,
        reject_materialize=reject_materialize,
    )
    env = _TestEnv(
        cfg or _make_cfg(),
        cast(SimBackend, backend),
        num_envs,
    )
    return env, backend


def _make_state_env(
    *,
    commands: dict[str, CommandTermCfg | None] | None = None,
    events: dict[str, EventTermCfg | None] | None = None,
    terminations: dict[str, TerminationTermCfg | None] | None = None,
) -> tuple[_TestEnv, _StateBackend]:
    cfg = _make_cfg(
        sim_substeps=1,
        observations={
            "actor": ObservationGroupCfg(
                terms={"joint": ObservationTermCfg(func=_joint_state_obs)}
            ),
            "value": ObservationGroupCfg(
                terms={"joint": ObservationTermCfg(func=_joint_state_obs)}
            ),
        },
        include_optional_managers=False,
    )
    cfg.scene.entities["robot"] = EntityCfg(
        joint_names=("joint",),
        actuator_names=("motor",),
    )
    cfg.scale_rewards_by_dt = False
    cfg.rewards = {"joint": RewardTermCfg(func=_joint_state_reward, weight=1.0)}
    cfg.terminations = (
        {"joint": TerminationTermCfg(func=_joint_state_termination)}
        if terminations is None
        else terminations
    )
    cfg.commands = {} if commands is None else commands
    cfg.events = {} if events is None else events
    backend = _StateBackend(2)
    return _TestEnv(cfg, cast(SimBackend, backend), 2), backend


def test_public_names_are_spelling_only_aliases() -> None:
    assert ManagerBasedRLEnv is ManagerBasedRlEnv
    assert ManagerBasedRLEnvCfg is ManagerBasedRlEnvCfg
    assert make_manager_based_rl_env is manager_env_module.make_manager_based_rl_env


@pytest.mark.parametrize("backend_type", ["mujoco", "motrix", "mjwarp", "drake"])
def test_generic_factory_routes_only_public_backend_contract(
    monkeypatch: pytest.MonkeyPatch,
    backend_type: str,
) -> None:
    cfg = _make_cfg(include_optional_managers=False)
    assert cfg.scene is not None
    cfg.scene.entities["robot"] = EntityCfg(
        root_body_name="base",
        actuator_names=("motor",),
    )
    cfg.scene.entities["payload"] = EntityCfg(root_body_name="box")
    backend = _FakeBackend(3)
    constructed: dict[str, Any] = {}

    def fake_create_backend(
        requested_backend: str,
        scene: SceneCfg,
        num_envs: int,
        sim_dt: float,
        **kwargs: Any,
    ) -> SimBackend:
        constructed.update(
            backend_type=requested_backend,
            scene=scene,
            num_envs=num_envs,
            sim_dt=sim_dt,
            kwargs=kwargs,
        )
        return cast(SimBackend, backend)

    sentinel = object()

    def fake_make_env(
        received_cfg: ManagerBasedRlEnvCfg,
        received_backend: SimBackend,
        received_num_envs: int,
    ) -> Any:
        assert received_cfg is cfg
        assert received_backend is backend
        assert received_num_envs == 3
        return sentinel

    monkeypatch.setattr(manager_env_module, "create_backend", fake_create_backend)
    monkeypatch.setattr(manager_env_module, "ManagerBasedRlEnv", fake_make_env)

    result = make_manager_based_rl_env(cfg, num_envs=3, backend_type=backend_type)

    assert result is sentinel
    assert constructed["backend_type"] == backend_type
    assert constructed["scene"] is cfg.scene
    assert constructed["num_envs"] == 3
    assert constructed["sim_dt"] == cfg.sim_dt
    kwargs = constructed["kwargs"]
    assert kwargs["base_name"] == "base"
    assert kwargs["body_state_required"] is True
    assert "add_body_sensors" not in kwargs
    for key, value in env_backend_kwargs(cfg).items():
        assert kwargs[key] == value


@pytest.mark.parametrize(
    ("entities", "match"),
    [
        (
            {"robot": EntityCfg(actuator_names=("motor",))},
            "at least one scene entity with an explicit root_body_name",
        ),
        (
            {
                "payload": EntityCfg(root_body_name="box"),
                "tool": EntityCfg(root_body_name="tool"),
            },
            "conventional 'robot' root entity.*payload.*tool",
        ),
    ],
)
def test_generic_factory_rejects_missing_or_ambiguous_root_entity(
    monkeypatch: pytest.MonkeyPatch,
    entities: dict[str, EntityCfg],
    match: str,
) -> None:
    cfg = _make_cfg(include_optional_managers=False)
    assert cfg.scene is not None
    cfg.scene.entities = entities
    backend_constructed = False

    def reject_backend_construction(*args: Any, **kwargs: Any) -> SimBackend:
        nonlocal backend_constructed
        backend_constructed = True
        raise AssertionError("backend construction must not run")

    monkeypatch.setattr(manager_env_module, "create_backend", reject_backend_construction)

    with pytest.raises(ValueError, match=match):
        make_manager_based_rl_env(cfg, num_envs=2, backend_type="mujoco")
    assert not backend_constructed


@pytest.mark.parametrize(
    ("cfg_value", "num_envs", "backend_type", "error", "match"),
    [
        (object(), 2, "mujoco", TypeError, "expected ManagerBasedRlEnvCfg"),
        (None, 0, "mujoco", ValueError, "num_envs must be a positive integer"),
        (None, True, "mujoco", ValueError, "num_envs must be a positive integer"),
        (None, 2, "", ValueError, "backend_type must be a non-empty string"),
    ],
)
def test_generic_factory_rejects_invalid_public_arguments_before_backend(
    monkeypatch: pytest.MonkeyPatch,
    cfg_value: object | None,
    num_envs: int,
    backend_type: str,
    error: type[Exception],
    match: str,
) -> None:
    cfg = _make_cfg(include_optional_managers=False) if cfg_value is None else cfg_value
    backend_constructed = False

    def reject_backend_construction(*args: Any, **kwargs: Any) -> SimBackend:
        nonlocal backend_constructed
        backend_constructed = True
        raise AssertionError("backend construction must not run")

    monkeypatch.setattr(manager_env_module, "create_backend", reject_backend_construction)

    with pytest.raises(error, match=match):
        make_manager_based_rl_env(
            cast(ManagerBasedRlEnvCfg, cfg),
            num_envs=num_envs,
            backend_type=backend_type,
        )
    assert not backend_constructed


@pytest.mark.parametrize(
    ("entities", "error", "match"),
    [
        (
            cast(dict[str, EntityCfg], {1: EntityCfg(root_body_name="base")}),
            TypeError,
            "entity names must be non-empty strings",
        ),
        (
            cast(dict[str, EntityCfg], {"robot": object()}),
            TypeError,
            "scene entity 'robot' must be EntityCfg",
        ),
        (
            {"robot": EntityCfg(root_body_name="")},
            TypeError,
            "root_body_name must be a non-empty string",
        ),
    ],
)
def test_generic_factory_rejects_invalid_entity_contract_before_backend(
    monkeypatch: pytest.MonkeyPatch,
    entities: dict[str, EntityCfg],
    error: type[Exception],
    match: str,
) -> None:
    cfg = _make_cfg(include_optional_managers=False)
    assert cfg.scene is not None
    cfg.scene.entities = entities
    backend_constructed = False

    def reject_backend_construction(*args: Any, **kwargs: Any) -> SimBackend:
        nonlocal backend_constructed
        backend_constructed = True
        raise AssertionError("backend construction must not run")

    monkeypatch.setattr(manager_env_module, "create_backend", reject_backend_construction)

    with pytest.raises(error, match=match):
        make_manager_based_rl_env(cfg, num_envs=2, backend_type="mujoco")
    assert not backend_constructed


def test_generic_factory_preserves_backend_construction_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_cfg(include_optional_managers=False)
    assert cfg.scene is not None
    cfg.scene.entities["robot"] = EntityCfg(
        root_body_name="base",
        actuator_names=("motor",),
    )

    def fail_backend_construction(*args: Any, **kwargs: Any) -> SimBackend:
        raise NotImplementedError("body-state capability is unavailable")

    monkeypatch.setattr(manager_env_module, "create_backend", fail_backend_construction)

    with pytest.raises(NotImplementedError, match="body-state capability is unavailable"):
        make_manager_based_rl_env(cfg, num_envs=2, backend_type="mjwarp")


def test_generic_factory_cleans_backend_when_env_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_cfg(include_optional_managers=False)
    assert cfg.scene is not None
    cfg.scene.entities["robot"] = EntityCfg(
        root_body_name="base",
        actuator_names=("motor",),
    )
    backend = _FakeBackend(2)

    monkeypatch.setattr(
        manager_env_module,
        "create_backend",
        lambda *args, **kwargs: cast(SimBackend, backend),
    )

    def fail_env_construction(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("manager initialization failed")

    monkeypatch.setattr(manager_env_module, "ManagerBasedRlEnv", fail_env_construction)

    with pytest.raises(RuntimeError, match="manager initialization failed"):
        make_manager_based_rl_env(cfg, num_envs=2, backend_type="mujoco")
    assert backend.cleanup_calls == 1


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


def test_backend_materializes_once_after_startup_and_before_runtime() -> None:
    cfg = _make_cfg()
    cfg.events = {
        "startup": EventTermCfg(func=_startup_event, mode="startup"),
        "reset": EventTermCfg(func=_event, mode="reset"),
    }
    env, backend = _make_env(cfg)

    assert backend.lifecycle == ["startup", "materialize"]
    assert backend.materialize_calls == 1

    env.reset()
    env.step(np.zeros((2, 1), dtype=np.float32))
    env.reset()

    assert backend.materialize_calls == 1


def test_backend_materialization_failure_has_lifecycle_context() -> None:
    with pytest.raises(
        RuntimeError,
        match="ManagerBasedRlEnv failed to materialize backend 'fake'.*pool construction failed",
    ):
        _make_env(reject_materialize=True)


@pytest.mark.parametrize(
    ("selector", "error"),
    [
        (1, TypeError),
        ("", ValueError),
    ],
)
def test_default_keyframe_selector_fails_at_env_initialization(
    selector: Any,
    error: type[Exception],
) -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.scene.default_keyframe_name = selector
    backend = _KeyframeBackend(2)

    with pytest.raises(error, match="default_keyframe_name.*non-empty string or None"):
        _TestEnv(cfg, cast(SimBackend, backend), 2)

    assert backend.keyframe_qpos_calls == 0


@pytest.mark.parametrize(
    ("value", "error", "match"),
    [
        ([0.0], TypeError, "must return np.ndarray"),
        (np.zeros((1, 1), dtype=np.float32), ValueError, "expected 1-D"),
        (np.zeros(1, dtype=np.int32), TypeError, "must be floating"),
        (np.array([np.nan], dtype=np.float32), ValueError, "NaN or Inf"),
    ],
)
def test_default_keyframe_qpos_contract_fails_at_env_initialization(
    value: Any,
    error: type[Exception],
    match: str,
) -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.scene.default_keyframe_name = "home"
    backend = _KeyframeBackend(2, keyframe_qpos=value)

    with pytest.raises(error, match=f"default keyframe 'home'.*backend 'fake'.*{match}"):
        _TestEnv(cfg, cast(SimBackend, backend), 2)

    assert backend.keyframe_qpos_calls == 1


@pytest.mark.parametrize(
    ("failure", "error", "match"),
    [
        (
            NotImplementedError("named keyframes disabled"),
            NotImplementedError,
            "default keyframe 'home'.*backend 'fake'.*named keyframes disabled",
        ),
        (
            ValueError("keyframe missing"),
            ValueError,
            "resolve default keyframe 'home'.*backend 'fake'.*keyframe missing",
        ),
    ],
)
def test_default_keyframe_resolution_names_backend_and_keyframe(
    failure: Exception,
    error: type[Exception],
    match: str,
) -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.scene.default_keyframe_name = "home"
    backend = _KeyframeBackend(2, keyframe_error=failure)

    with pytest.raises(error, match=match):
        _TestEnv(cfg, cast(SimBackend, backend), 2)

    assert backend.keyframe_qpos_calls == 1


def test_named_keyframe_snapshot_is_shared_by_entity_and_reset_cold_path() -> None:
    source_qpos = np.array([0.0, 0.0, 0.3, 0.75], dtype=np.float64)
    backend = _KeyframeBackend(2, keyframe_qpos=source_qpos)
    cfg = _make_cfg(include_optional_managers=False)
    cfg.scene.default_keyframe_name = "home"
    cfg.scene.entities["robot"] = EntityCfg(
        joint_names=("joint",),
        actuator_names=("motor",),
    )
    cfg.events = {
        "reset_default": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset"),
    }

    env = _TestEnv(cfg, cast(SimBackend, backend), 2)
    selected_qpos = env._reset_state._selected_default_qpos

    assert backend.keyframe_qpos_calls == 1
    assert backend.default_qpos_calls == 0
    assert backend.joint_layout_calls == 1
    assert selected_qpos is not source_qpos
    assert selected_qpos is not None
    assert not selected_qpos.flags.writeable
    np.testing.assert_array_equal(env.scene["robot"].data.default_joint_pos, 0.75)
    assert not env.scene["robot"].data.default_joint_pos.flags.writeable

    source_qpos[:] = 9.0
    env.reset()
    env.reset(env_ids=np.array([1], dtype=np.int32))

    assert backend.keyframe_qpos_calls == 1
    assert backend.default_qpos_calls == 0
    assert backend.joint_layout_calls == 1
    assert len(backend.set_state_calls) == 2
    np.testing.assert_array_equal(
        backend.set_state_calls[0][1],
        [[0.0, 0.0, 0.3, 0.75], [0.0, 0.0, 0.3, 0.75]],
    )
    np.testing.assert_array_equal(backend.set_state_calls[1][1], [[0.0, 0.0, 0.3, 0.75]])


def test_real_mujoco_backend_is_materialized_before_first_reset() -> None:
    scene = SceneCfg(
        model_file=str(ASSETS_ROOT_PATH / "robots" / "go2" / "scene_flat.xml"),
        entities={"robot": EntityCfg(root_body_name="base")},
    )
    cfg = ManagerBasedRlEnvCfg(
        scene=scene,
        sim_dt=0.01,
        ctrl_dt=0.02,
        max_episode_seconds=1.0,
        observations={
            "actor": ObservationGroupCfg(
                terms={"state": ObservationTermCfg(func=_episode_step_observation)}
            )
        },
        actions={},
        events={"reset_default": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")},
        rewards={"alive": RewardTermCfg(func=mdp.is_alive, weight=1.0)},
        terminations={"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)},
        policy_observation_group="actor",
    )
    env = make_manager_based_rl_env(cfg, num_envs=2, backend_type="mujoco")
    try:
        state = env.init_state()
        assert state.obs["obs"].shape == (2, 1)
        assert np.isfinite(state.obs["obs"]).all()

        state = env.step(np.empty((2, 0), dtype=np.float32))
        assert np.isfinite(state.obs["obs"]).all()
        assert np.isfinite(state.reward).all()
    finally:
        env.close()


@pytest.mark.parametrize(
    ("default_keyframe_name", "expected_root_z", "expected_joint_pos"),
    [
        (None, 0.445, np.zeros(12, dtype=np.float32)),
        (
            "home",
            0.3,
            np.array(
                [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5],
                dtype=np.float32,
            ),
        ),
    ],
)
def test_real_mujoco_default_state_matches_qpos0_or_named_home(
    default_keyframe_name: str | None,
    expected_root_z: float,
    expected_joint_pos: np.ndarray,
) -> None:
    joint_names = (
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
    scene = SceneCfg(
        model_file=str(ASSETS_ROOT_PATH / "robots" / "go2" / "scene_flat.xml"),
        entities={
            "robot": EntityCfg(
                root_body_name="base",
                joint_names=joint_names,
            )
        },
        default_keyframe_name=default_keyframe_name,
    )
    cfg = ManagerBasedRlEnvCfg(
        scene=scene,
        sim_dt=0.01,
        ctrl_dt=0.02,
        max_episode_seconds=1.0,
        observations={
            "actor": ObservationGroupCfg(
                terms={"state": ObservationTermCfg(func=_episode_step_observation)}
            )
        },
        actions={},
        events={"reset_default": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")},
        rewards={"alive": RewardTermCfg(func=mdp.is_alive, weight=1.0)},
        terminations={"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)},
        policy_observation_group="actor",
    )
    backend = create_backend(
        "mujoco",
        scene,
        2,
        cfg.sim_dt,
        base_name="base",
        add_body_sensors=True,
        **env_backend_kwargs(cfg),
    )
    env = ManagerBasedRlEnv(cfg, backend, 2)
    try:
        robot = env.scene["robot"]
        np.testing.assert_allclose(robot.data.default_root_state[:, 2], expected_root_z)
        expected_batch = np.broadcast_to(expected_joint_pos, (2, expected_joint_pos.size))
        np.testing.assert_allclose(robot.data.default_joint_pos, expected_batch)

        env.init_state()

        np.testing.assert_allclose(robot.data.root_link_pos_w[:, 2], expected_root_z)
        np.testing.assert_allclose(robot.data.joint_pos, expected_batch)
    finally:
        env.close()


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
    assert backend.pre_step_control is None
    assert backend.step_nsteps == [2]
    assert len(backend.applied_controls) == 2
    assert env.action_sim_steps == [2]
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
    assert not any(key.startswith("mba_") for key in state.info["timing"])

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


def test_reset_event_and_command_state_writes_share_one_commit() -> None:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.events = {
        "default_first": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset"),
        "joint_state": EventTermCfg(func=_write_reset_joint_state, mode="reset"),
    }
    cfg.commands = {"state_writer": _StateWritingCommandCfg(resampling_time_range=(1.0, 1.0))}
    cfg.scene.entities["robot"] = EntityCfg(
        joint_names=("joint",),
        actuator_names=("motor",),
    )
    backend = _ResetBackend(2)
    env = _TestEnv(cfg, cast(SimBackend, backend), 2)

    env.reset()

    assert len(backend.set_state_calls) == 1
    ids, qpos, qvel = backend.set_state_calls[0]
    np.testing.assert_array_equal(ids, [0, 1])
    np.testing.assert_array_equal(
        qpos,
        [[0.0, 0.0, 0.5, 0.75], [0.0, 0.0, 0.5, 0.75]],
    )
    np.testing.assert_array_equal(qvel, [[0.0, 0.0, -0.75], [0.0, 0.0, -0.75]])


def test_pure_reset_event_does_not_request_backend_state_capability() -> None:
    env, backend = _make_env()

    env.reset()
    env.reset(env_ids=np.array([0], dtype=np.int32))

    assert isinstance(backend, _FakeBackend)
    assert not hasattr(backend, "get_default_qpos")


def test_update_state_reuses_backend_state_once_and_refreshes_after_physics() -> None:
    env, backend = _make_state_env()
    env.init_state()

    before = backend.dof_pos_calls
    state = env.step(np.zeros((2, 1), dtype=np.float32))
    assert backend.dof_pos_calls == before + 1
    np.testing.assert_array_equal(state.reward, [1.0, 1.0])
    np.testing.assert_array_equal(state.obs["obs"], [[1.0], [1.0]])
    np.testing.assert_array_equal(state.obs["critic"], [[1.0], [1.0]])

    before = backend.dof_pos_calls
    state = env.step(np.zeros((2, 1), dtype=np.float32))
    assert backend.dof_pos_calls == before + 1
    np.testing.assert_array_equal(state.reward, [2.0, 2.0])
    np.testing.assert_array_equal(state.obs["obs"], [[2.0], [2.0]])


def test_update_state_invalidates_cache_after_command_set_state() -> None:
    env, backend = _make_state_env(
        commands={
            "writer": _StateWritingCommandCfg(resampling_time_range=(0.01, 0.01)),
        }
    )
    env.init_state()

    before = backend.dof_pos_calls
    state = env.step(np.zeros((2, 1), dtype=np.float32))

    # Physics advances the pre-command state to 1.75. The command then commits
    # 0.75 through set_state, so post-command observations must not reuse the
    # termination/reward snapshot.
    assert backend.dof_pos_calls == before + 2
    np.testing.assert_array_equal(state.reward, [1.75, 1.75])
    np.testing.assert_array_equal(state.obs["obs"], [[0.75], [0.75]])
    np.testing.assert_array_equal(state.obs["critic"], [[0.75], [0.75]])


def test_update_state_invalidates_cache_after_runtime_event() -> None:
    env, backend = _make_state_env(
        events={
            "mutate": EventTermCfg(func=_mutate_joint_state_step_event, mode="step"),
        }
    )
    env.init_state()

    before = backend.dof_pos_calls
    state = env.step(np.zeros((2, 1), dtype=np.float32))

    assert backend.dof_pos_calls == before + 2
    np.testing.assert_array_equal(state.reward, [1.0, 1.0])
    np.testing.assert_array_equal(state.obs["obs"], [[11.0], [11.0]])
    np.testing.assert_array_equal(state.obs["critic"], [[11.0], [11.0]])


def test_partial_reset_observation_reads_post_set_state_rows() -> None:
    env, backend = _make_state_env(
        events={
            "joint_state": EventTermCfg(func=_write_reset_joint_state, mode="reset"),
        },
        terminations={"failure": TerminationTermCfg(func=_failure)},
    )
    env.init_state()

    state = env.step(np.array([[1.0], [0.0]], dtype=np.float32))

    np.testing.assert_array_equal(state.reward, [1.25, 1.25])
    np.testing.assert_array_equal(state.terminated, [True, False])
    np.testing.assert_array_equal(state.obs["obs"], [[0.25], [1.25]])
    np.testing.assert_array_equal(state.obs["critic"], [[0.25], [1.25]])
    assert state.final_observation is not None
    np.testing.assert_array_equal(state.final_observation["obs"][0], [1.25])
    np.testing.assert_array_equal(backend.dof_pos[:, 0], [0.25, 1.25])


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


def test_multisubstep_state_independent_action_skips_callback() -> None:
    env, backend = _make_env(reject_pre_step=True)
    try:
        assert backend.pre_step_control is None
        assert not env.action_manager.requires_substep_state_feedback
    finally:
        env.close()


def test_multisubstep_state_feedback_action_uses_callback() -> None:
    cfg = _make_cfg(
        actions={"drive": _FeedbackDriveCfg(entity_name="robot")},
    )
    env, backend = _make_env(cfg)
    try:
        env.init_state()
        env.step(np.array([[0.25], [0.5]], dtype=np.float32))

        assert env.action_manager.requires_substep_state_feedback
        assert backend.pre_step_control is not None
        assert backend.step_nsteps == [2]
        assert env.action_sim_steps == [1, 2]
        assert len(backend.applied_controls) == 2
    finally:
        env.close()


def test_multisubstep_state_feedback_action_fails_when_backend_rejects_callback() -> None:
    cfg = _make_cfg(
        actions={"drive": _FeedbackDriveCfg(entity_name="robot")},
    )
    with pytest.raises(
        NotImplementedError,
        match="ActionManager.*state-feedback actions.*every physics substep.*fake",
    ):
        _make_env(cfg, reject_pre_step=True)


def test_single_substep_feedback_action_does_not_require_callback() -> None:
    cfg = _make_cfg(
        sim_substeps=1,
        actions={"drive": _FeedbackDriveCfg(entity_name="robot")},
    )
    env, backend = _make_env(cfg, reject_pre_step=True)
    try:
        env.init_state()
        env.step(np.array([[0.25], [0.5]], dtype=np.float32))

        assert backend.pre_step_control is None
        assert backend.step_nsteps == [1]
        assert env.action_sim_steps == [1]
    finally:
        env.close()


def test_empty_action_manager_does_not_require_callback() -> None:
    observations = {
        "actor": ObservationGroupCfg(
            terms={"episode_step": ObservationTermCfg(func=_episode_step_observation)}
        ),
        "value": ObservationGroupCfg(
            terms={"episode_step": ObservationTermCfg(func=_episode_step_observation)}
        ),
    }
    env, backend = _make_env(
        _make_cfg(actions={}, observations=observations),
        reject_pre_step=True,
    )
    try:
        assert env.action_manager.active_terms == []
        assert not env.action_manager.requires_substep_state_feedback
        assert backend.pre_step_control is None
    finally:
        env.close()


def test_state_feedback_action_error_propagates_from_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_cfg(
        actions={"drive": _FeedbackDriveCfg(entity_name="robot")},
    )
    env, _ = _make_env(cfg)
    env.init_state()
    term = env.action_manager.get_term("drive")

    def fail() -> None:
        raise ValueError("feedback failed")

    monkeypatch.setattr(term, "apply_actions", fail)
    try:
        with pytest.raises(ValueError, match="ActionManager term 'drive'.*feedback failed"):
            env.step(np.zeros((2, 1), dtype=np.float32))
    finally:
        env.close()


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
    cfg = _make_cfg(
        actions={"drive": _FeedbackDriveCfg(entity_name="robot")},
    )
    env, backend = _make_env(cfg)
    assert backend.pre_step_control is not None
    env.close()
    assert backend.pre_step_control is None
    assert backend.cleanup_calls == 1
    assert env.trace[-1] == "recorder_close"
