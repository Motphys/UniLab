# Derived from mujocolab/mjlab v1.6.0 (0fb8a681),
# src/mjlab/envs/manager_based_rl_env.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for the NumPy NpEnv/SimBackend contracts; Apache-2.0.
"""Community-compatible manager lifecycle on UniLab's NumPy runtime."""

from __future__ import annotations

import math
import secrets
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

from unilab.base.backend import SimBackend, create_backend, env_backend_kwargs
from unilab.base.base import EnvCfg
from unilab.base.config_overrides import (
    CONFIG_MAPPING_POLICY_KEY,
    MANAGER_TERM_MAPPING_POLICY,
)
from unilab.base.entity import EntityCfg, EntityScene
from unilab.base.np_env import NpEnv, NpEnvState
from unilab.base.reset_state import ResetStateTransaction
from unilab.base.scene import SceneCfg, resolve_scene_default_qpos
from unilab.dtype_config import get_global_dtype
from unilab.managers import (
    ActionManager,
    ActionTermCfg,
    CommandManager,
    CommandTermCfg,
    CurriculumManager,
    CurriculumTermCfg,
    EventManager,
    EventTermCfg,
    MetricsManager,
    MetricsTermCfg,
    NullCommandManager,
    NullCurriculumManager,
    NullMetricsManager,
    NullRecorderManager,
    ObservationGroupCfg,
    ObservationManager,
    RecorderManager,
    RecorderTermCfg,
    RewardManager,
    RewardTermCfg,
    TerminationManager,
    TerminationTermCfg,
)


def _manager_terms_field() -> Any:
    return field(
        default_factory=dict,
        metadata={CONFIG_MAPPING_POLICY_KEY: MANAGER_TERM_MAPPING_POLICY},
    )


@dataclass
class ManagerBasedRlEnvCfg(EnvCfg):
    """Configuration for the manager-based NumPy environment.

    Production task owners declare these fields in Hydra. The Registry materializes
    them into this plain typed config on the cold path; Python factories do not mirror
    task-specific manager or term declarations.
    """

    observations: dict[str, ObservationGroupCfg | None] = _manager_terms_field()
    actions: dict[str, ActionTermCfg | None] = _manager_terms_field()
    events: dict[str, EventTermCfg | None] = _manager_terms_field()
    rewards: dict[str, RewardTermCfg | None] = _manager_terms_field()
    terminations: dict[str, TerminationTermCfg | None] = _manager_terms_field()
    commands: dict[str, CommandTermCfg | None] = _manager_terms_field()
    curriculum: dict[str, CurriculumTermCfg | None] = _manager_terms_field()
    metrics: dict[str, MetricsTermCfg | None] = _manager_terms_field()
    recorders: dict[str, RecorderTermCfg | None] = _manager_terms_field()

    seed: int | None = None
    is_finite_horizon: bool = False
    auto_reset: bool = True
    scale_rewards_by_dt: bool = True
    policy_observation_group: str = "policy"
    critic_observation_group: str | None = None

    def validate(self) -> None:
        for name, value in (("sim_dt", self.sim_dt), ("ctrl_dt", self.ctrl_dt)):
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                raise TypeError(f"ManagerBasedRlEnvCfg {name} must be a real number")
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"ManagerBasedRlEnvCfg {name} must be finite and positive")
        super().validate()
        ratio = self.ctrl_dt / self.sim_dt
        if not np.isclose(ratio, round(ratio), rtol=0.0, atol=1e-9):
            raise ValueError(
                "ManagerBasedRlEnvCfg ctrl_dt must be an integer multiple of sim_dt; "
                f"received ctrl_dt={self.ctrl_dt}, sim_dt={self.sim_dt}"
            )
        if self.max_episode_seconds is None:
            raise ValueError("ManagerBasedRlEnvCfg max_episode_seconds must be finite and positive")
        if isinstance(self.max_episode_seconds, bool) or not isinstance(
            self.max_episode_seconds, (int, float, np.number)
        ):
            raise TypeError("ManagerBasedRlEnvCfg max_episode_seconds must be a real number")
        if not np.isfinite(self.max_episode_seconds) or self.max_episode_seconds <= 0.0:
            raise ValueError("ManagerBasedRlEnvCfg max_episode_seconds must be finite and positive")
        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, (int, np.integer))
            or self.seed < 0
        ):
            raise ValueError("ManagerBasedRlEnvCfg seed must be a non-negative integer or None")
        if self.seed is not None:
            self.seed = int(self.seed)
        for name in (
            "observations",
            "actions",
            "events",
            "rewards",
            "terminations",
            "commands",
            "curriculum",
            "metrics",
            "recorders",
        ):
            if not isinstance(getattr(self, name), dict):
                raise TypeError(f"ManagerBasedRlEnvCfg {name} must be a dict")
        for name in ("is_finite_horizon", "auto_reset", "scale_rewards_by_dt"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"ManagerBasedRlEnvCfg {name} must be bool")
        if not isinstance(self.policy_observation_group, str) or not self.policy_observation_group:
            raise ValueError("policy_observation_group must be a non-empty string")
        if self.critic_observation_group is not None:
            if (
                not isinstance(self.critic_observation_group, str)
                or not self.critic_observation_group
            ):
                raise ValueError("critic_observation_group must be a non-empty string or None")
            if self.critic_observation_group == self.policy_observation_group:
                raise ValueError("policy and critic observation groups must be different")
        if not isinstance(self.scene, SceneCfg):
            raise TypeError(
                "ManagerBasedRlEnvCfg scene must be a SceneCfg instance, "
                f"got {type(self.scene).__name__}"
            )


def _resolve_backend_entity_contract(cfg: ManagerBasedRlEnvCfg) -> tuple[str, bool]:
    """Resolve task-independent backend inputs from declared scene entities."""
    assert cfg.scene is not None
    root_entities: list[tuple[str, str]] = []
    body_state_requested = False
    for entity_name, entity_cfg in cfg.scene.entities.items():
        if not isinstance(entity_name, str) or not entity_name:
            raise TypeError(
                f"ManagerBasedRlEnv scene entity names must be non-empty strings; "
                f"got {entity_name!r}"
            )
        if not isinstance(entity_cfg, EntityCfg):
            raise TypeError(
                f"ManagerBasedRlEnv scene entity '{entity_name}' must be EntityCfg, "
                f"got {type(entity_cfg).__name__}"
            )
        root_body_name = entity_cfg.root_body_name
        if root_body_name is not None:
            if not isinstance(root_body_name, str) or not root_body_name:
                raise TypeError(
                    f"ManagerBasedRlEnv root entity '{entity_name}' root_body_name must be "
                    "a non-empty string"
                )
            root_entities.append((entity_name, root_body_name))
            body_state_requested = True
        if entity_cfg.body_names is not None:
            body_state_requested = True

    if not root_entities:
        raise ValueError(
            "ManagerBasedRlEnv factory requires at least one scene entity with an explicit "
            "root_body_name"
        )
    primary = next((item for item in root_entities if item[0] == "robot"), None)
    if primary is None and len(root_entities) != 1:
        declared = [name for name, _ in root_entities]
        raise ValueError(
            "ManagerBasedRlEnv factory requires a conventional 'robot' root entity when "
            f"multiple floating entities are declared; found {declared}"
        )
    return (primary or root_entities[0])[1], body_state_requested


class ManagerBasedRlEnv(NpEnv):
    """Manager-Based API adapter that reuses the single :class:`NpEnv` lifecycle."""

    is_vector_env = True
    _cfg: ManagerBasedRlEnvCfg
    event_manager: EventManager
    command_manager: CommandManager | NullCommandManager
    action_manager: ActionManager
    observation_manager: ObservationManager
    termination_manager: TerminationManager
    reward_manager: RewardManager
    curriculum_manager: CurriculumManager | NullCurriculumManager
    metrics_manager: MetricsManager | NullMetricsManager
    recorder_manager: RecorderManager | NullRecorderManager

    def __init__(self, cfg: ManagerBasedRlEnvCfg, backend: SimBackend, num_envs: int):
        if not isinstance(cfg, ManagerBasedRlEnvCfg):
            raise TypeError(
                f"ManagerBasedRlEnv expected ManagerBasedRlEnvCfg, received {type(cfg).__name__}"
            )
        cfg.validate()
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError(
                f"ManagerBasedRlEnv num_envs must be a positive integer, got {num_envs!r}"
            )
        if backend.num_envs != num_envs:
            raise ValueError(
                f"ManagerBasedRlEnv num_envs={num_envs} does not match backend "
                f"'{backend.backend_type}' num_envs={backend.num_envs}"
            )

        super().__init__(cfg, backend, num_envs)
        actual_seed = cfg.seed if cfg.seed is not None else secrets.randbits(63)
        cfg.seed = actual_seed
        self.rng = np.random.default_rng(actual_seed)

        assert cfg.scene is not None
        default_qpos = resolve_scene_default_qpos(cfg.scene, backend)
        self._control = np.zeros((num_envs, backend.num_actuators), dtype=get_global_dtype())
        self._reset_state = ResetStateTransaction(backend, default_qpos=default_qpos)
        self.scene = EntityScene.from_scene_cfg(
            cfg.scene,
            backend,
            self._control,
            reset_state=self._reset_state,
            default_qpos=default_qpos,
        )

        self.common_step_counter = 0
        self._sim_step_counter = 0
        self.episode_length_buf = np.zeros(num_envs, dtype=np.int64)
        self.reset_buf = np.zeros(num_envs, dtype=np.bool_)
        self.reset_terminated = np.zeros(num_envs, dtype=np.bool_)
        self.reset_time_outs = np.zeros(num_envs, dtype=np.bool_)
        self.reward_buf = np.zeros(num_envs, dtype=get_global_dtype())
        self.obs_buf: dict[str, np.ndarray] = {}
        self.extras: dict[str, Any] = {"log": {}}
        self._command_dt = np.zeros(num_envs, dtype=get_global_dtype())
        self._no_truncation = np.zeros(num_envs, dtype=np.bool_)
        self._manual_reset_pending = np.zeros(num_envs, dtype=np.bool_)
        self._all_env_ids = np.arange(num_envs, dtype=np.int32)
        self._all_env_ids.setflags(write=False)
        self._has_transition = False
        self._uses_pre_step_control = False

        self._load_managers()
        self._mapped_obs_dims = self._validate_observation_mapping()
        self._validate_substep_capabilities()
        self._configure_action_control()
        self.set_autoreset(cfg.auto_reset)

        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self._materialize_backend()

    def _materialize_backend(self) -> None:
        """Finalize backend runtime resources before the first reset or step."""
        try:
            self._backend.materialize()
        except NotImplementedError as exc:
            raise NotImplementedError(
                "ManagerBasedRlEnv lifecycle capability 'SimBackend.materialize' is "
                f"unavailable on backend '{self._backend.backend_type}': {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "ManagerBasedRlEnv failed to materialize backend "
                f"'{self._backend.backend_type}' after startup events: {exc}"
            ) from exc

    @property
    def physics_dt(self) -> float:
        return self._cfg.sim_dt

    @property
    def step_dt(self) -> float:
        return self._cfg.ctrl_dt

    @property
    def max_episode_length_s(self) -> float:
        assert self._cfg.max_episode_seconds is not None
        return self._cfg.max_episode_seconds

    @property
    def max_episode_length(self) -> int:
        return math.ceil(self.max_episode_length_s / self.step_dt)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return dict(self._mapped_obs_dims)

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.action_manager.total_action_dim,),
            dtype=get_global_dtype(),
        )

    @property
    def unwrapped(self) -> ManagerBasedRlEnv:
        return self

    def _load_managers(self) -> None:
        """Construct managers in the pinned community dependency order."""
        self.event_manager = EventManager(self._cfg.events, self)
        self.command_manager = (
            CommandManager(self._cfg.commands, self) if self._cfg.commands else NullCommandManager()
        )
        self.action_manager = ActionManager(self._cfg.actions, self)
        self.observation_manager = ObservationManager(self._cfg.observations, self)
        self.termination_manager = TerminationManager(self._cfg.terminations, self)
        self.reward_manager = RewardManager(
            self._cfg.rewards,
            self,
            scale_by_dt=self._cfg.scale_rewards_by_dt,
        )
        self.curriculum_manager = (
            CurriculumManager(self._cfg.curriculum, self)
            if self._cfg.curriculum
            else NullCurriculumManager()
        )
        self.metrics_manager = (
            MetricsManager(self._cfg.metrics, self) if self._cfg.metrics else NullMetricsManager()
        )
        self.recorder_manager = (
            RecorderManager(self._cfg.recorders, self)
            if self._cfg.recorders
            else NullRecorderManager()
        )

    def _validate_observation_mapping(self) -> dict[str, int]:
        mapping = {"obs": self._cfg.policy_observation_group}
        if self._cfg.critic_observation_group is not None:
            mapping["critic"] = self._cfg.critic_observation_group
        dims: dict[str, int] = {}
        for output_name, group_name in mapping.items():
            if group_name not in self.observation_manager.active_terms:
                raise KeyError(
                    f"ManagerBasedRlEnv observation mapping '{output_name}' requests group "
                    f"'{group_name}', available={list(self.observation_manager.active_terms)}"
                )
            if not self.observation_manager.group_obs_concatenate[group_name]:
                raise ValueError(
                    f"ManagerBasedRlEnv observation group '{group_name}' mapped to "
                    f"NpEnvState.obs['{output_name}'] must concatenate terms"
                )
            group_dim = self.observation_manager.group_obs_dim[group_name]
            if not isinstance(group_dim, tuple) or len(group_dim) != 1:
                raise ValueError(
                    f"ManagerBasedRlEnv observation group '{group_name}' mapped to "
                    f"NpEnvState.obs['{output_name}'] must be one-dimensional; got {group_dim}"
                )
            dims[output_name] = int(group_dim[0])
        return dims

    def _validate_substep_capabilities(self) -> None:
        per_substep_terms = [
            name
            for name, term_cfg in self._cfg.metrics.items()
            if term_cfg is not None and term_cfg.per_substep
        ]
        if self._cfg.sim_substeps > 1 and per_substep_terms:
            raise NotImplementedError(
                "MetricsManager capability 'post-physics per-substep metrics' is unavailable "
                f"on backend '{self._backend.backend_type}' with sim_substeps="
                f"{self._cfg.sim_substeps}; terms={per_substep_terms}. SimBackend does not "
                "declare a post-substep hook."
            )

    def _configure_action_control(self) -> None:
        if (
            self._cfg.sim_substeps <= 1
            or not self.action_manager.active_terms
            or not self.action_manager.requires_substep_state_feedback
        ):
            return
        try:
            self._backend.set_pre_step_control(self._apply_manager_control)
        except NotImplementedError as exc:
            raise NotImplementedError(
                "ActionManager capability 'state-feedback actions on every physics substep' is "
                f"unavailable on backend '{self._backend.backend_type}': {exc}"
            ) from exc
        self._uses_pre_step_control = True

    def _apply_manager_control(
        self,
        backend: SimBackend,
        control: np.ndarray,
    ) -> np.ndarray:
        del backend, control
        self._sim_step_counter += 1
        self.action_manager.apply_action()
        return self._control

    def _initial_episode_steps(self) -> np.ndarray:
        return np.zeros((self.num_envs,), dtype=np.uint32)

    def init_state(self) -> NpEnvState:
        state = super().init_state()
        self.obs_buf = state.obs
        self.reward_buf = state.reward
        self.extras = state.info
        return state

    def step(self, actions: np.ndarray) -> NpEnvState:
        if not self._autoreset and np.any(self._manual_reset_pending):
            pending = np.flatnonzero(self._manual_reset_pending).tolist()
            raise RuntimeError(
                f"ManagerBasedRlEnv environments {pending} must be reset before step() "
                "when auto_reset=False"
            )
        state = super().step(actions)
        if not self._autoreset:
            self._manual_reset_pending |= state.terminated | state.truncated
        self.recorder_manager.record_post_step()
        return state

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        del state
        self.action_manager.process_action(actions)
        if not self._uses_pre_step_control:
            self._sim_step_counter += self._cfg.sim_substeps
            self.action_manager.apply_action()
        return self._control

    def update_state(self, state: NpEnvState) -> NpEnvState:
        # Physics stepping and reset/set_state lifecycles sit outside this private
        # scope. In-phase mutations explicitly invalidate it below.
        with self.scene._scoped_state_reads():
            return self._update_state_in_read_phase(state)

    def _update_state_in_read_phase(self, state: NpEnvState) -> NpEnvState:
        log: dict[str, Any] = {}
        state.info["log"] = log
        self.extras = state.info

        np.add(state.info["steps"], 1, out=self.episode_length_buf)
        self.common_step_counter = self.step_counter + 1
        self._sim_step_counter = self.common_step_counter * self._cfg.sim_substeps

        self.termination_manager.compute()
        if self._cfg.is_finite_horizon:
            np.logical_or(
                self.termination_manager.terminated,
                self.termination_manager.time_outs,
                out=self.reset_terminated,
            )
            self.reset_time_outs.fill(False)
        else:
            np.copyto(self.reset_terminated, self.termination_manager.terminated)
            np.copyto(self.reset_time_outs, self.termination_manager.time_outs)
        np.logical_or(self.reset_terminated, self.reset_time_outs, out=self.reset_buf)

        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        log.update(self.reward_manager.step_reward_extras())

        if self._cfg.sim_substeps == 1:
            self.metrics_manager.compute_substep()
        self.metrics_manager.compute()

        applied_runtime_event = False
        if "step" in self.event_manager.available_modes:
            self.event_manager.apply(mode="step", dt=self.step_dt)
            applied_runtime_event = True
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
            applied_runtime_event = True
        if applied_runtime_event:
            # Event terms may mutate simulation state through their formal
            # interval/step capabilities; EventManager does not expose whether a
            # particular interval fired, so this boundary stays fail-closed.
            self.scene._invalidate_state_reads()

        self._command_dt.fill(self.step_dt)
        self._command_dt[self.reset_buf] = 0.0
        with self._reset_state.scoped(self._all_env_ids):
            self.command_manager.compute(dt=self._command_dt)
        if self._reset_state.last_commit_had_writes:
            self.scene._invalidate_state_reads()
        self.command_manager.post_compute()

        manager_obs = self.observation_manager.compute(update_history=True)

        self.obs_buf = self._map_observations(manager_obs)
        self._has_transition = True

        return state.replace(
            obs=self.obs_buf,
            reward=self.reward_buf,
            terminated=self.reset_terminated,
            truncated=self.reset_time_outs,
            info=state.info,
        )

    def _compute_truncated(self, state: NpEnvState) -> np.ndarray:
        del state
        self._no_truncation.fill(False)
        return self._no_truncation

    def reset(
        self,
        env_indices: np.ndarray | None = None,
        *,
        seed: int | None = None,
        env_ids: np.ndarray | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del options
        ids = self._normalize_reset_ids(env_indices, env_ids)
        if seed is not None:
            self.seed(seed)
        if self._state is None:
            all_ids = np.arange(self.num_envs, dtype=np.int32)
            if not np.array_equal(ids, all_ids):
                raise RuntimeError(
                    "ManagerBasedRlEnv requires a full reset before the first partial reset"
                )
            state = self.init_state()
            return state.obs, {"log": state.info.get("log", {})}

        done_ids = ids[self.reset_buf[ids]]
        if self._has_transition and len(done_ids) > 0:
            self.recorder_manager.record_pre_reset(done_ids)

        log: dict[str, Any] = {}
        self.curriculum_manager.compute(env_ids=ids)
        with self._reset_state.scoped(ids):
            if "reset" in self.event_manager.available_modes:
                self.event_manager.apply(
                    mode="reset",
                    env_ids=ids,
                    global_env_step_count=self.step_counter,
                )
            log.update(self.command_manager.reset(ids))

        for manager in (
            self.observation_manager,
            self.action_manager,
            self.reward_manager,
            self.metrics_manager,
            self.curriculum_manager,
            self.event_manager,
            self.termination_manager,
        ):
            log.update(manager.reset(ids))

        self.episode_length_buf[ids] = 0
        self._control[ids] = 0.0
        self._manual_reset_pending[ids] = False
        if self._state is not None:
            self._state.info["steps"][ids] = 0

        # The read phase starts only after the reset-state transaction above
        # committed, so cached getter values are post-set_state reads shared
        # across terms (issue #1295).
        with self.scene._scoped_state_reads():
            self.command_manager.compute(dt=0.0, env_ids=ids)
            self.command_manager.post_compute()
            # Row-scoped reset rebuild (issue #1259 R2): the observation manager
            # returns only the reset rows, so no full-batch slice is needed here.
            manager_obs = self.observation_manager.compute(update_history=True, env_ids=ids)
        mapped_obs = self._map_observations(manager_obs, num_rows=len(ids))
        reset_obs = {name: values.copy() for name, values in mapped_obs.items()}

        if self._state is not None:
            for name, values in reset_obs.items():
                self._state.obs[name][ids] = values
            if self._autoreset_reset_active:
                # Autoreset runs at the tail of step(): keep this step's
                # per-step log entries (reward/* etc., computed pre-reset) and
                # layer the reset extras (Episode_Reward/* etc.) on top, so
                # consumers still see the transition's reward breakdown.
                step_log = self._state.info.get("log")
                if step_log:
                    log = {**step_log, **log}
            self._state.info["log"] = log
            if not self._autoreset_reset_active:
                self._state.terminated[ids] = False
                self._state.truncated[ids] = False
                self.reset_buf[ids] = False
                self.reset_terminated[ids] = False
                self.reset_time_outs[ids] = False
        self.obs_buf = self._state.obs if self._state is not None else mapped_obs
        self.extras = self._state.info if self._state is not None else {"log": log}
        self.recorder_manager.record_post_reset(ids)
        return reset_obs, {"log": log}

    def _collect_reset_backend_timing_ms(self) -> dict[str, float]:
        timing = dict(super()._collect_reset_backend_timing_ms())
        timing.update(self._reset_state.last_set_state_timing_ms)
        return timing

    def _normalize_reset_ids(
        self,
        env_indices: np.ndarray | None,
        env_ids: np.ndarray | None,
    ) -> np.ndarray:
        if env_indices is not None and env_ids is not None:
            raise ValueError("Pass either env_indices or env_ids, not both")
        values = env_ids if env_ids is not None else env_indices
        if values is None:
            return np.arange(self.num_envs, dtype=np.int32)
        raw = np.asarray(values)
        if (
            raw.ndim != 1
            or not np.issubdtype(raw.dtype, np.integer)
            or np.issubdtype(raw.dtype, np.bool_)
        ):
            raise TypeError(
                "ManagerBasedRlEnv reset env IDs must be a 1-D integer np.ndarray; "
                f"got shape={raw.shape}, dtype={raw.dtype}"
            )
        ids = np.asarray(raw, dtype=np.int32)
        if np.any(ids < 0) or np.any(ids >= self.num_envs):
            raise IndexError(
                f"ManagerBasedRlEnv reset env IDs out of range for {self.num_envs} envs: "
                f"{ids.tolist()}"
            )
        if np.unique(ids).size != ids.size:
            raise ValueError(f"ManagerBasedRlEnv reset env IDs contain duplicates: {ids.tolist()}")
        return ids

    def _map_observations(
        self,
        manager_obs: dict[str, np.ndarray | dict[str, np.ndarray]],
        num_rows: int | None = None,
    ) -> dict[str, np.ndarray]:
        mapping = {"obs": self._cfg.policy_observation_group}
        if self._cfg.critic_observation_group is not None:
            mapping["critic"] = self._cfg.critic_observation_group
        mapped: dict[str, np.ndarray] = {}
        for output_name, group_name in mapping.items():
            value = manager_obs[group_name]
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"ManagerBasedRlEnv observation group '{group_name}' returned "
                    f"{type(value).__name__}, expected np.ndarray"
                )
            expected = (
                self.num_envs if num_rows is None else num_rows,
                self._mapped_obs_dims[output_name],
            )
            if value.shape != expected:
                raise ValueError(
                    f"ManagerBasedRlEnv observation group '{group_name}' returned shape "
                    f"{value.shape}, expected {expected} for NpEnvState.obs['{output_name}']"
                )
            mapped[output_name] = value
        return mapped

    def get_observations(self) -> dict[str, np.ndarray]:
        if self._state is None:
            return self.init_state().obs
        self.obs_buf = self._state.obs
        return self.obs_buf

    def seed(self, seed: int = -1) -> int:
        if seed == -1:
            seed = secrets.randbits(63)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"ManagerBasedRlEnv seed must be a non-negative integer, got {seed!r}")
        replacement = np.random.default_rng(seed)
        self.rng.bit_generator.state = replacement.bit_generator.state
        self._cfg.seed = seed
        return seed

    def close(self) -> None:
        self.recorder_manager.close()
        if self._uses_pre_step_control:
            self._backend.set_pre_step_control(None)
            self._uses_pre_step_control = False
        super().close()


def make_manager_based_rl_env(
    cfg: ManagerBasedRlEnvCfg,
    num_envs: int = 1,
    backend_type: str = "mujoco",
) -> ManagerBasedRlEnv:
    """Construct the generic Registry-owned Manager-Based production runtime."""
    if not isinstance(cfg, ManagerBasedRlEnvCfg):
        raise TypeError(
            "make_manager_based_rl_env expected ManagerBasedRlEnvCfg, "
            f"received {type(cfg).__name__}"
        )
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError(
            f"make_manager_based_rl_env num_envs must be a positive integer, got {num_envs!r}"
        )
    if not isinstance(backend_type, str) or not backend_type:
        raise ValueError(
            "make_manager_based_rl_env backend_type must be a non-empty string, "
            f"got {backend_type!r}"
        )

    cfg.validate()
    assert cfg.scene is not None
    base_name, body_state_requested = _resolve_backend_entity_contract(cfg)
    backend_kwargs = env_backend_kwargs(cfg)
    backend_kwargs["base_name"] = base_name

    backend = create_backend(
        backend_type,
        cfg.scene,
        num_envs,
        cfg.sim_dt,
        body_state_required=body_state_requested,
        **backend_kwargs,
    )
    try:
        return ManagerBasedRlEnv(cfg, backend, num_envs)
    except Exception:
        backend.cleanup_scene_assets()
        raise


# Isaac Lab capitalization is a spelling-only alias.  There is one implementation.
ManagerBasedRLEnv = ManagerBasedRlEnv
ManagerBasedRLEnvCfg = ManagerBasedRlEnvCfg

__all__ = [
    "ManagerBasedRLEnv",
    "ManagerBasedRLEnvCfg",
    "ManagerBasedRlEnv",
    "ManagerBasedRlEnvCfg",
    "make_manager_based_rl_env",
]
