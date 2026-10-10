# Derived from mujocolab/mjlab v1.6.0 (0fb8a681),
# src/mjlab/envs/manager_based_rl_env.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for the Manager-Based/Torch runtime contracts; Apache-2.0.
"""Community-compatible Manager-Based lifecycle with a Torch public contract."""

from __future__ import annotations

import math
import re
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
from unisim.backend.base import (
    DebugOverlayGetter,
    DebugPrimitive,
    SimBackend,
    TensorDataPlane,
    TensorExecution,
    TensorLifecycleCapabilities,
    TensorProcessTopology,
)

from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.base.base import EnvCfg
from unilab.base.config_overrides import (
    CONFIG_MAPPING_POLICY_KEY,
    MANAGER_TERM_MAPPING_POLICY,
)
from unilab.base.cpu_runtime import apply_env_cpu_runtime
from unilab.base.entity import EntityCfg, EntityScene, SceneTensorReadPlan, SceneTensorReadSpec
from unilab.base.reset_state import ResetStateTransaction
from unilab.base.scene import SceneCfg, resolve_scene_default_qpos
from unilab.base.torch_env import TorchEnv, TorchEnvState
from unilab.base.variants import (
    _build_fixed_variant_plan,
    _require_fixed_variant_support,
)
from unilab.dtype_config import get_global_dtype
from unilab.envs import mdp
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
    NullResetOwnerManager,
    ObservationGroupCfg,
    ObservationManager,
    ObservationTermCfg,
    RecorderManager,
    RecorderTermCfg,
    ResetOwnerCfg,
    ResetOwnerManager,
    RewardManager,
    RewardTermCfg,
    TerminationManager,
    TerminationTermCfg,
    TorchManagerRng,
)
from unilab.managers.scene_entity_config import SceneEntityCfg


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
    reset_owners: dict[str, ResetOwnerCfg | None] = _manager_terms_field()

    # HOST_BRIDGE physics remains CPU-authoritative. CPU is therefore the default;
    # a training process may request accelerator Torch carriers shared with its
    # learner. This selects public-manager carrier placement only; it is neither a
    # physics-device claim nor a tensor/NumPy runtime compatibility switch.
    manager_torch_device: str | None = None
    seed: int | None = None
    is_finite_horizon: bool = False
    auto_reset: bool = True
    scale_rewards_by_dt: bool = True
    policy_observation_group: str = "policy"
    critic_observation_group: str | None = None

    def validate(self) -> None:
        if self.manager_torch_device is not None:
            if (
                not isinstance(self.manager_torch_device, str)
                or not self.manager_torch_device.strip()
            ):
                raise TypeError(
                    "ManagerBasedRlEnvCfg manager_torch_device must be 'cpu', "
                    f"'cuda', or 'cuda:<index>', or None; got {self.manager_torch_device!r}"
                )
            try:
                requested_device = torch.device(self.manager_torch_device)
            except (RuntimeError, TypeError, ValueError) as exc:
                raise ValueError(
                    "ManagerBasedRlEnvCfg manager_torch_device is not a valid Torch "
                    f"device: {self.manager_torch_device!r}"
                ) from exc
            if requested_device.type not in {"cpu", "cuda"}:
                raise ValueError(
                    "ManagerBasedRlEnvCfg manager_torch_device must be cpu, cuda, or "
                    f"None; got {self.manager_torch_device!r}"
                )
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
            "reset_owners",
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
        self.scene.__post_init__()


def _resolve_backend_entity_contract(
    cfg: ManagerBasedRlEnvCfg,
) -> tuple[str, bool, tuple[str, ...] | None]:
    """Resolve task-independent backend inputs from declared scene entities."""
    assert cfg.scene is not None
    if cfg.scene.entity_assets:
        names = {entity.name: entity for entity in cfg.scene.entity_assets}
        primary = cfg.scene.primary_entity
        if primary is None:
            controlled = [
                name
                for name, selector in cfg.scene.entities.items()
                if isinstance(selector, EntityCfg) and selector.actuator_names
            ]
            if len(controlled) != 1:
                raise ValueError("composed manager scene requires explicit primary_entity")
            selector = cfg.scene.entities[controlled[0]]
            assert isinstance(selector, EntityCfg)
            primary = selector.physical_entity
            if primary is None and selector.root_body_name:
                primary = selector.root_body_name.split("/", 1)[0]
        if primary is None or primary not in names:
            raise ValueError("primary entity must name a physical scene entity")
        root_names = [
            selector.root_body_name
            for selector in cfg.scene.entities.values()
            if isinstance(selector, EntityCfg)
            and selector.root_body_name
            and selector.root_body_name.startswith(primary + "/")
        ]
        if not root_names:
            raise ValueError("primary_entity needs an entity-qualified logical root selector")
        return root_names[0], True, None
    root_entities: list[tuple[str, str]] = []
    body_state_requested = False
    tracked_body_names: list[str] | None = []
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
            if tracked_body_names is not None:
                tracked_body_names.append(root_body_name)
        if entity_cfg.body_names is not None:
            body_state_requested = True
            selector_names = (
                (entity_cfg.body_names,)
                if isinstance(entity_cfg.body_names, str)
                else entity_cfg.body_names
            )
            if tracked_body_names is not None and any(
                not isinstance(name, str) or not name or re.search(r"[\\^$.|?*+()\[\]{}]", name)
                for name in selector_names
            ):
                tracked_body_names = None
            elif tracked_body_names is not None:
                tracked_body_names.extend(selector_names)

    if not root_entities:
        raise ValueError(
            "ManagerBasedRlEnv factory requires at least one scene entity with an explicit "
            "root_body_name"
        )
    primary_root = next((item for item in root_entities if item[0] == "robot"), None)
    if primary_root is None and len(root_entities) != 1:
        declared = [name for name, _ in root_entities]
        raise ValueError(
            "ManagerBasedRlEnv factory requires a conventional 'robot' root entity when "
            f"multiple floating entities are declared; found {declared}"
        )
    selected_names = (
        None if tracked_body_names is None else tuple(dict.fromkeys(tracked_body_names))
    )
    return (primary_root or root_entities[0])[1], body_state_requested, selected_names


class ManagerBasedRlEnv(TorchEnv):
    """Manager-Based runtime with a Torch public lifecycle.

    P1 keeps Manager terms, Entity state, reset transactions, and recorder
    scratch on an explicit NumPy host boundary. Public action/reset inputs,
    backend control, state, observations, reward, termination, truncation, and
    final observations are Torch tensors. This is a migration boundary, not a
    durable NumPy fallback: unsupported tensor capabilities fail closed.
    """

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
    _tensor_read_plan: SceneTensorReadPlan | None
    _tensor_reset_default_root_state: torch.Tensor | None
    _last_reset_manager_timing_ms: dict[str, float]
    _tensor_reset_env_origins: torch.Tensor | None
    _tensor_reset_env_origins_nonzero: bool | None
    _tensor_reset_pose_bounds: torch.Tensor | None
    _tensor_reset_velocity_bounds: torch.Tensor | None

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

        initial_capabilities = backend.get_tensor_capabilities()
        runtime_device = self._resolve_runtime_device(
            initial_capabilities, cfg.manager_torch_device
        )
        super().__init__(cfg, backend, num_envs, device=runtime_device)
        actual_seed = cfg.seed if cfg.seed is not None else secrets.randbits(63)
        cfg.seed = actual_seed
        self.rng = np.random.default_rng(actual_seed)
        self._torch_rng_owner = TorchManagerRng(device=self.device)
        self.torch_rng = self._torch_rng_owner
        self._torch_generator = self._torch_rng_owner.generator if self._torch_rng_owner else None
        self._reward_log_names: tuple[str, ...] = ()
        self._reward_log_means = torch.empty(0, dtype=torch.float32, device=self.device)
        self._last_reset_manager_timing_ms = {}
        self._tensor_reset_default_root_state = None
        self._tensor_reset_env_origins = None
        self._tensor_reset_env_origins_nonzero = None
        self._tensor_reset_pose_bounds = None
        self._tensor_reset_velocity_bounds = None

        assert cfg.scene is not None
        default_qpos = resolve_scene_default_qpos(cfg.scene, backend)
        self._control = torch.zeros(
            (num_envs, backend.num_actuators), dtype=torch.float32, device=self.device
        )
        if cfg.scene.entity_assets:
            self._control.copy_(self._initial_backend_control())
        self._reset_state = ResetStateTransaction(
            backend,
            default_qpos=default_qpos,
            scene_layout=backend.get_scene_layout() if cfg.scene.entity_assets else None,
        )
        self.scene = EntityScene.from_scene_cfg(
            cfg.scene,
            backend,
            self._control,
            reset_state=self._reset_state,
            default_qpos=default_qpos,
        )

        self.common_step_counter = 0
        self._sim_step_counter = 0
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self.reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.reset_terminated = torch.zeros_like(self.reset_buf)
        self.reset_time_outs = torch.zeros_like(self.reset_buf)
        self.reward_buf = torch.zeros(num_envs, dtype=self._dtype, device=self.device)
        self.obs_buf: dict[str, torch.Tensor] = {}
        self.extras: dict[str, Any] = {"log": {}}
        self._command_dt = torch.full(
            (num_envs,), self.step_dt, dtype=self._dtype, device=self.device
        )
        self._manual_reset_pending = torch.zeros_like(self.reset_buf)
        self._all_env_rows = torch.arange(num_envs, dtype=torch.int64, device=self.device)
        self._has_transition = False

        self._load_managers()
        self._mapped_obs_dims = self._validate_observation_mapping()
        self._validate_substep_capabilities()
        self._configure_action_control()
        self.set_autoreset(cfg.auto_reset)

        self._materialize_backend()
        if "startup" in self.event_manager.available_modes:
            # Startup model-field DR uses the same public set_state transaction
            # as reset DR. Materialize the backend first: adapters may make the
            # selected-row reset boundary and model pools available only there.
            # This remains a cold-path commit before read-plan binding/reset.
            with self._reset_state.scoped(self._all_env_rows):
                self.event_manager.apply(
                    mode="startup",
                    env_ids=self._all_env_rows if self.event_manager.uses_startup_rows else None,
                )
        self._compile_tensor_read_plan()
        self.command_manager.bind_read_phase()
        self._validate_manager_tensor_runtime()

    def _compile_tensor_read_plan(self) -> None:
        """Compile the scene's only packed tensor read phase."""
        specs: list[SceneTensorReadSpec] = []
        specs.extend(self._action_tensor_read_specs())
        specs.extend(self._observation_tensor_read_specs())
        specs.extend(self._task_tensor_state_read_specs())
        specs.extend(self._manager_term_tensor_read_specs())
        specs.extend(self._command_tensor_read_specs())
        self.scene._tensor_read_plan = (
            self.scene.compile_tensor_reads(self.device, specs) if specs else None
        )

    def _action_tensor_read_specs(self) -> list[SceneTensorReadSpec]:
        specs: list[SceneTensorReadSpec] = []
        for name in self.action_manager.active_terms:
            term = self.action_manager.get_term(name)
            body_names = getattr(term, "tensor_body_names", None)
            if body_names is None:
                continue
            if (
                not isinstance(body_names, (tuple, list))
                or any(not isinstance(value, str) or not value for value in body_names)
                or len(set(body_names)) != len(body_names)
            ):
                raise TypeError(
                    "ManagerBasedRlEnv tensor read declaration for action term "
                    f"'{name}' must be a unique sequence of body names; got {body_names!r}"
                )
            specs.append(
                SceneTensorReadSpec(entity=term.cfg.entity_name, body_names=tuple(body_names))
            )
        return specs

    def _observation_tensor_read_specs(self) -> list[SceneTensorReadSpec]:
        specs: list[SceneTensorReadSpec] = []
        for group_name, terms in self.observation_manager.active_terms.items():
            for name in terms:
                term_cfg = self.observation_manager.get_term_cfg(group_name, name)
                sensor_names = getattr(term_cfg.func, "tensor_sensor_names", None)
                if sensor_names is not None:
                    if (
                        not isinstance(sensor_names, (tuple, list))
                        or any(
                            not isinstance(sensor_name, str) or not sensor_name
                            for sensor_name in sensor_names
                        )
                        or len(set(sensor_names)) != len(sensor_names)
                    ):
                        raise TypeError(
                            "ManagerBasedRlEnv tensor read declaration for observation "
                            f"term '{group_name}/{name}' must be a unique sequence of "
                            f"sensor names; got {sensor_names!r}"
                        )
                    if (
                        self._backend.get_tensor_capabilities().execution
                        is TensorExecution.DEVICE_RESIDENT
                    ):
                        self._require_device_resident_sensors(
                            tuple(sensor_names),
                            f"observation term '{group_name}/{name}'",
                        )
                    specs.append(
                        SceneTensorReadSpec(
                            entity=self._observation_tensor_entity(term_cfg),
                            sensor_names=tuple(sensor_names),
                        )
                    )
                if self.device.type != "cpu" and term_cfg.func in (
                    mdp.joint_pos_rel,
                    mdp.joint_vel_rel,
                ):
                    specs.append(
                        SceneTensorReadSpec(
                            entity=self._observation_tensor_entity(term_cfg), body_names=()
                        )
                    )
                term = term_cfg.func
                body_names = term_cfg.params.get("tensor_body_names")
                if body_names is None:
                    body_names = getattr(term, "tensor_body_names", None)
                if body_names is None:
                    continue
                if (
                    not isinstance(body_names, (tuple, list))
                    or any(not isinstance(value, str) or not value for value in body_names)
                    or len(set(body_names)) != len(body_names)
                ):
                    raise TypeError(
                        "ManagerBasedRlEnv tensor read declaration for observation term "
                        f"'{name}' must be a unique sequence of body names; got {body_names!r}"
                    )
                entity_name = term_cfg.params.get("entity_name")
                if not isinstance(entity_name, str) or not entity_name:
                    entity_name = getattr(term, "entity_name", None)
                if not isinstance(entity_name, str) or not entity_name:
                    raise TypeError(
                        "ManagerBasedRlEnv observation tensor read declaration "
                        f"'{name}' requires an entity_name parameter"
                    )
                specs.append(SceneTensorReadSpec(entity=entity_name, body_names=tuple(body_names)))
        return specs

    def _manager_term_tensor_read_specs(self) -> list[SceneTensorReadSpec]:
        specs: list[SceneTensorReadSpec] = []
        narrow = (
            self._backend.get_tensor_capabilities().execution is TensorExecution.DEVICE_RESIDENT
        )
        for manager_name, manager in (
            ("reward", self.reward_manager),
            ("termination", self.termination_manager),
        ):
            for name in manager.active_terms:
                specs.extend(
                    self._term_tensor_read_specs(
                        manager_name,
                        name,
                        manager.get_term_cfg(name),
                        narrow=narrow,
                    )
                )
                if manager_name == "termination":
                    term = manager.get_term_cfg(name).func
                    specs.extend(self._manager_term_body_read_specs(manager_name, name, term))
        return specs

    def _task_tensor_state_read_specs(self) -> list[SceneTensorReadSpec]:
        """Collect explicit task-owned state-only packed reads."""
        specs: list[SceneTensorReadSpec] = []
        seen: set[str] = set()
        for group_name, terms in self.observation_manager.active_terms.items():
            for name in terms:
                term_cfg = self.observation_manager.get_term_cfg(group_name, name)
                entity_name = self._declared_tensor_state_entity(
                    term_cfg.func, f"observation/{group_name}/{name}"
                )
                if entity_name is None:
                    continue
                self._append_tensor_state_spec(specs, seen, entity_name)
        for manager_name, manager in (
            ("reward", self.reward_manager),
            ("termination", self.termination_manager),
        ):
            for name in manager.active_terms:
                term = manager.get_term_cfg(name).func
                entity_name = self._declared_tensor_state_entity(term, f"{manager_name}/{name}")
                if entity_name is None:
                    continue
                self._append_tensor_state_spec(specs, seen, entity_name)
        return specs

    @staticmethod
    def _declared_tensor_state_entity(term: Any, label: str) -> str | None:
        entity_name = getattr(term, "tensor_state_entity", None)
        if entity_name is None:
            return None
        if not isinstance(entity_name, str) or not entity_name:
            raise TypeError(
                f"Manager tensor state declaration for term '{label}' requires a "
                "non-empty entity_name"
            )
        return entity_name

    @staticmethod
    def _append_tensor_state_spec(
        specs: list[SceneTensorReadSpec], seen: set[str], entity_name: str
    ) -> None:
        if entity_name in seen:
            return
        seen.add(entity_name)
        specs.append(SceneTensorReadSpec(entity=entity_name))

    @staticmethod
    def _manager_term_body_read_specs(
        manager_name: str, term_name: str, term: Any
    ) -> list[SceneTensorReadSpec]:
        body_names = getattr(term, "tensor_body_names", None)
        if body_names is None:
            return []
        if (
            not isinstance(body_names, (tuple, list))
            or any(not isinstance(body_name, str) or not body_name for body_name in body_names)
            or len(set(body_names)) != len(body_names)
        ):
            raise TypeError(
                f"ManagerBasedRlEnv tensor body declaration for {manager_name} term "
                f"'{term_name}' must be a unique sequence of body names; got {body_names!r}"
            )
        params = getattr(getattr(term, "cfg", None), "params", {})
        entity_name = params.get("entity_name") if isinstance(params, dict) else None
        if not isinstance(entity_name, str) or not entity_name:
            entity_name = "robot"
        return [SceneTensorReadSpec(entity=entity_name, body_names=tuple(body_names))]

    def _command_tensor_read_specs(self) -> list[SceneTensorReadSpec]:
        """Collect optional command-term reads for the packed phase."""
        specs: list[SceneTensorReadSpec] = []
        capabilities = self._backend.get_tensor_capabilities()
        narrow = capabilities.execution is TensorExecution.DEVICE_RESIDENT
        command_cfgs = self._cfg.commands or {}
        for name in self.command_manager.active_terms:
            command_cfg = command_cfgs.get(name)
            if command_cfg is None:
                raise KeyError(
                    f"Manager command term '{name}' has no ManagerBasedRlEnvCfg declaration"
                )
            command_term = self.command_manager.get_term(name)
            entity_name = getattr(command_cfg, "entity_name", "robot") if command_cfg else "robot"
            entity_name = entity_name if isinstance(entity_name, str) and entity_name else "robot"
            body_names = getattr(command_term, "tensor_body_names", None)
            if body_names is not None:
                if (
                    not isinstance(body_names, (tuple, list))
                    or any(
                        not isinstance(body_name, str) or not body_name for body_name in body_names
                    )
                    or len(set(body_names)) != len(body_names)
                ):
                    raise TypeError(
                        "ManagerBasedRlEnv tensor body declaration for command term "
                        f"'{name}' must be a unique sequence of body names; got {body_names!r}"
                    )
                specs.append(
                    SceneTensorReadSpec(
                        entity=entity_name,
                        body_names=tuple(body_names),
                    )
                )
                continue
            sensor_names = getattr(command_term, "tensor_sensor_names", None)
            if sensor_names is None:
                continue
            if not isinstance(sensor_names, (tuple, list)) or any(
                not isinstance(sensor_name, str) or not sensor_name for sensor_name in sensor_names
            ):
                raise TypeError(
                    "ManagerBasedRlEnv tensor read declaration for command term "
                    f"'{name}' must be a sequence of sensor names; got {sensor_names!r}"
                )
            names = tuple(sensor_names)
            if narrow:
                self._require_device_resident_sensors(names, f"command term '{name}'")
            elif capabilities.packed_host_bridge:
                # Only HOST_BRIDGE packs need backend-local aliases. Device-
                # resident plans consume the term's canonical semantic names.
                names = self._resolve_host_bridge_sensor_names(
                    self.command_manager.get_term(name),
                    tuple(sensor_names),
                    "command",
                    name,
                )
            if not names:
                continue
            specs.append(
                SceneTensorReadSpec(
                    entity=entity_name,
                    sensor_names=names,
                )
            )
        return specs

    def _published_device_resident_sensors(self) -> frozenset[str] | None:
        """Return the backend-declared named sensor namespace when it exists."""
        try:
            return frozenset(self._backend.get_sensor_names())
        except (AttributeError, NotImplementedError, KeyError, ValueError, RuntimeError):
            return None

    def _require_device_resident_sensors(
        self, sensor_names: tuple[str, ...], term_label: str
    ) -> None:
        """Fail closed on tensor sensors a DEVICE_RESIDENT backend does not publish.

        Validation consumes the backend-declared named sensor namespace. A
        backend that cannot declare one is probed through the public cold-path
        sensor binding instead, so unsupported declared names never silently
        select a host reader.
        """
        backend = self._backend
        declared = self._published_device_resident_sensors()
        if declared is not None:
            missing = tuple(name for name in sensor_names if name not in declared)
        else:
            unpublished: list[str] = []
            for name in sensor_names:
                try:
                    backend.get_sensor_data(name)
                except (KeyError, ValueError):
                    unpublished.append(name)
                except (TypeError, NotImplementedError, AttributeError) as exc:
                    raise type(exc)(
                        "Manager DEVICE_RESIDENT sensor negotiation on backend "
                        f"'{backend.backend_type}': {exc}"
                    ) from exc
            missing = tuple(unpublished)
        if missing:
            raise NotImplementedError(
                "DEVICE_RESIDENT backend "
                f"'{backend.backend_type}' does not publish tensor "
                f"sensors {missing} required by {term_label}"
            )

    def _resolve_host_bridge_sensor_names(
        self,
        term: Any,
        requested: tuple[str, ...],
        manager_name: str,
        term_name: str,
    ) -> tuple[str, ...]:
        """Negotiate optional packed reads against available named sensors.

        Optional ``tensor_sensor_names`` describe a device-plane optimization,
        not the semantic term contract. A HOST_BRIDGE packet must contain only
        backend-native sensor names (Go2 exposes ``local_linvel`` while G1
        exposes ``pelvis_local_linvel``), so unavailable names retain their
        validated public host-carrier implementation.
        """
        aliases = getattr(term, "packed_sensor_aliases", None)
        if aliases is None:
            return requested
        if not isinstance(aliases, Mapping) or any(
            not isinstance(canonical, str)
            or not canonical
            or not isinstance(local_names, (tuple, list))
            or any(not isinstance(local_name, str) or not local_name for local_name in local_names)
            for canonical, local_names in aliases.items()
        ):
            raise TypeError(
                f"ManagerBasedRlEnv packed sensor declaration for {manager_name} term "
                f"'{term_name}' must map canonical names to sequences of names; "
                f"got {aliases!r}"
            )
        candidates = tuple(dict.fromkeys(name for alias in aliases.values() for name in alias))
        available = self._available_entity_sensors(candidates)
        resolved: list[str] = []
        for name in requested:
            if name in resolved:
                continue
            for local_name in aliases.get(name, (name,)):
                if local_name in resolved:
                    continue
                if local_name in available:
                    resolved.append(local_name)
                    break
        return tuple(resolved)

    def _available_entity_sensors(self, names: tuple[str, ...]) -> set[str]:
        """Resolve requested names through the public cold-path sensor binding."""
        available: set[str] = set()
        for name in names:
            try:
                self._backend.get_sensor_data(name)
            except KeyError:
                continue
            except (TypeError, ValueError, NotImplementedError, AttributeError) as exc:
                raise type(exc)(
                    "Manager packed sensor negotiation on backend "
                    f"'{self._backend.backend_type}': {exc}"
                ) from exc
            available.add(name)
        return available

    def _term_tensor_read_specs(
        self,
        manager_name: str,
        term_name: str,
        term_cfg: RewardTermCfg | TerminationTermCfg,
        *,
        narrow: bool = False,
    ) -> list[SceneTensorReadSpec]:
        sensor_names = getattr(term_cfg.func, "tensor_sensor_names", None)
        if sensor_names is None:
            return []
        if (
            not isinstance(sensor_names, (tuple, list))
            or any(not isinstance(name, str) or not name for name in sensor_names)
            or len(set(sensor_names)) != len(sensor_names)
        ):
            raise TypeError(
                f"ManagerBasedRlEnv tensor read declaration for {manager_name} term "
                f"'{term_name}' must be a unique sequence of sensor names; got "
                f"{sensor_names!r}"
            )
        if narrow:
            self._require_device_resident_sensors(
                tuple(sensor_names), f"{manager_name} term '{term_name}'"
            )
        return [SceneTensorReadSpec(entity="robot", sensor_names=tuple(sensor_names))]

    def _warm_external_cuda_ipc_views(self) -> None:
        """Materialize IsaacGym's post-step view aliases before Manager reads."""

        # IsaacGym's worker CUDA IPC projection does not publish rigid-body
        # state during its reset stale window. The public selected-reset path
        # therefore performs one untimed zero-control tensor step after commit,
        # matching the documented phase-local benchmark warm-up, before Manager
        # observation terms read the stable views.
        if self._backend.backend_type != "isaacgym" or self.device.type != "cuda":
            return
        try:
            self._backend.step_tensor(
                torch.zeros_like(self._control), nsteps=self._cfg.sim_substeps
            )
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "ManagerBasedRlEnv could not materialize IsaacGym CUDA IPC sensor "
                f"views after reset: {exc}"
            ) from exc

    def _initial_backend_control(self) -> torch.Tensor:
        """Return mapped-scene initial control without assuming a ctrl state view."""

        try:
            return self._backend.get_state_views(("ctrl",))["ctrl"].to(device=self.device)
        except KeyError:
            # IsaacSim's mapped CUDA IPC arena owns a control tensor, but state
            # views remain deliberately limited to qpos/qvel. Zero is the
            # backend bootstrap value and remains the cold-path seed.
            return torch.zeros_like(self._control)

    @staticmethod
    def _uses_device_resident_reset(
        read_plan: SceneTensorReadPlan | None, capabilities: TensorLifecycleCapabilities
    ) -> bool:
        """Dispatch by negotiated capability, not worker topology."""
        return (
            read_plan is not None
            and capabilities.execution is TensorExecution.DEVICE_RESIDENT
            and capabilities.selected_reset
        )

    @staticmethod
    def _observation_tensor_entity(term_cfg: ObservationTermCfg) -> str:
        asset_cfg = term_cfg.params.get("asset_cfg")
        if isinstance(asset_cfg, SceneEntityCfg):
            return asset_cfg.name
        entity_name = term_cfg.params.get("entity_name")
        if entity_name is None:
            return "robot"
        if not isinstance(entity_name, str) or not entity_name:
            raise TypeError(
                "Manager tensor observation entity_name must be a non-empty string; "
                f"got {entity_name!r}"
            )
        return entity_name

    def _validate_manager_tensor_runtime(self) -> None:
        """Bind the tensor lifecycle after backend materialization."""
        self._bind_tensor_runtime()

    @staticmethod
    def _resolve_runtime_device(
        capabilities: TensorLifecycleCapabilities, requested_device: str | None
    ) -> torch.device:
        """Resolve authoritative Manager carrier placement.

        ``DEVICE_RESIDENT`` owns accelerator storage and has no CPU public
        carrier. For ``HOST_BRIDGE``, capability inventory says which device
        families are possible, while the caller's request chooses whether this
        process pays the packed H2D/D2H boundary. CPU is the default; explicit
        ``cuda`` resolves to the current Torch ordinal (including ROCm/HIP's
        CUDA namespace). Final capability validation remains fail-closed in
        ``TorchEnv._bind_tensor_runtime``.
        """

        if capabilities.execution is TensorExecution.DEVICE_RESIDENT:
            return torch.device("cuda", index=torch.cuda.current_device())
        if capabilities.execution is not TensorExecution.HOST_BRIDGE:
            raise ValueError(
                "TorchEnv got a non-tensor backend capability "
                f"{capabilities.execution!r}; expected HOST_BRIDGE or DEVICE_RESIDENT"
            )
        if requested_device is None:
            return torch.device("cpu")
        try:
            device = torch.device(requested_device)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(
                "Manager HOST_BRIDGE runtime device is not a valid Torch device: "
                f"{requested_device!r}"
            ) from exc
        if device.type == "cuda" and device.index is None:
            return torch.device("cuda", index=torch.cuda.current_device())
        return device

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

    def get_playback_debug_overlays(self) -> DebugOverlayGetter | None:
        """Aggregate playback overlays from command terms that provide one.

        Terms opt in by implementing ``playback_debug_overlay_getter()``,
        which returns a per-frame getter following the
        :data:`unisim.backend.base.DebugOverlayGetter` contract. The returned
        getter merges primitives per env across all providing terms. Returns
        ``None`` when no term provides an overlay.
        """
        getters: list[DebugOverlayGetter] = []
        for name in self.command_manager.active_terms:
            term = self.command_manager.get_term(name)
            provider = getattr(term, "playback_debug_overlay_getter", None)
            if provider is None:
                continue
            getter = provider()
            if getter is not None:
                getters.append(getter)
        if not getters:
            return None
        num_envs = self.num_envs

        def _get_overlay() -> list[list[DebugPrimitive]] | None:
            per_term = [getter() for getter in getters]
            if all(overlays is None for overlays in per_term):
                return None
            merged: list[list[DebugPrimitive]] = []
            for env_idx in range(num_envs):
                env_primitives: list[DebugPrimitive] = []
                for overlays in per_term:
                    if overlays is None:
                        continue
                    entry = overlays[env_idx]
                    if entry:
                        env_primitives.extend(entry)
                merged.append(env_primitives)
            return merged

        return _get_overlay

    def _enter_suppressed_hot_state_refresh(self) -> None:
        """Allow a cold proxy to temporarily replace NumPy body-state reads."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support suppressed hot state refresh"
        )

    def _exit_suppressed_hot_state_refresh(self) -> None:
        """Undo :meth:`_enter_suppressed_hot_state_refresh`."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support suppressed hot state refresh"
        )

    def _load_managers(self) -> None:
        """Construct managers in the pinned community dependency order."""
        suppress_hot_state_refresh = bool(getattr(self, "_suppress_hot_state_refresh", False))
        if suppress_hot_state_refresh:
            self._enter_suppressed_hot_state_refresh()
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
        if suppress_hot_state_refresh:
            self._exit_suppressed_hot_state_refresh()
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
        self.reset_owner_manager = (
            ResetOwnerManager(self._cfg.reset_owners, self)
            if self._cfg.reset_owners
            else NullResetOwnerManager()
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
                    f"TorchEnvState.obs['{output_name}'] must concatenate terms"
                )
            group_dim = self.observation_manager.group_obs_dim[group_name]
            if not isinstance(group_dim, tuple) or len(group_dim) != 1:
                raise ValueError(
                    f"ManagerBasedRlEnv observation group '{group_name}' mapped to "
                    f"TorchEnvState.obs['{output_name}'] must be one-dimensional; got {group_dim}"
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

        if self._cfg.sim_substeps > 1 and self.action_manager.requires_substep_state_feedback:
            raise NotImplementedError(
                "ManagerBasedRlEnv Torch lifecycle P1 does not support state-feedback "
                "actions on every physics substep"
            )

    def _configure_action_control(self) -> None:
        # Tensor Manager stepping consumes one control tensor per control step in
        # P1; multi-substep state feedback fails closed during validation.
        return

    def _control_to_backend_boundary(self) -> torch.Tensor:
        """Return the contiguous authoritative Torch control tensor."""
        return self._control

    def _manager_tensor(
        self, values: np.ndarray | torch.Tensor, *, dtype: torch.dtype
    ) -> torch.Tensor:
        """Publish one completed Manager result on the public Torch carrier."""
        if isinstance(values, torch.Tensor):
            if values.device == self.device and values.dtype == dtype and values.is_contiguous():
                return values
            return values.to(device=self.device, dtype=dtype, copy=True)
        host = np.array(values, order="C", copy=True)
        return torch.from_numpy(host).to(device=self.device, dtype=dtype, copy=True)

    def _initial_episode_steps(self) -> torch.Tensor:
        max_steps = self._cfg.max_episode_steps
        if max_steps is None:
            return super()._initial_episode_steps()
        if max_steps <= 1:
            raise ValueError(
                "ManagerBasedRlEnv randomized initial episode counters require "
                f"max_episode_steps > 1; got {max_steps}"
            )
        return torch.from_numpy(
            self.rng.integers(0, max_steps, size=self.num_envs, dtype=np.int64)
        ).to(device=self.device, dtype=torch.int64)

    def init_state(self) -> TorchEnvState:
        state = super().init_state()
        # TorchEnv's initial state allocates counters, marks every row
        # terminated, and performs one full bootstrap reset (which clears the
        # public counters).  The old NpEnv production behavior kept randomized
        # initial offsets from the seed stream, preventing all vector rows in a
        # large rollout from synchronizing on the same timeout boundary.  Keep
        # the original draw made by ``super().init_state()`` rather than drawing
        # a second time.
        initial_steps = self._last_initial_episode_steps
        self._last_initial_episode_steps = None
        state.info["steps"].copy_(initial_steps)
        self.episode_length_buf.copy_(state.info["steps"])
        self.extras = state.info
        return state

    def step(self, actions: torch.Tensor) -> TorchEnvState:
        if not self._autoreset and bool(self._manual_reset_pending.any()):
            pending = (
                self._manual_reset_pending.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
            )
            raise RuntimeError(
                f"ManagerBasedRlEnv environments {pending} must be reset before step() "
                "when auto_reset=False"
            )
        state = super().step(actions)
        state.info["log"].update(self.publish_reward_log())
        if not self._autoreset:
            self._manual_reset_pending.logical_or_(state.terminated | state.truncated)
        self.recorder_manager.record_post_step()
        return state

    def apply_action(self, actions: torch.Tensor, state: TorchEnvState) -> torch.Tensor:
        del state
        read_plan = self.scene._tensor_read_plan
        if read_plan is not None and self.action_manager.uses_tensor_state_packet:
            # State-feedback action terms consume the current packet before
            # physics. Ordinary feed-forward terms match mjlab's ordering:
            # process control from the previous state boundary, then refresh
            # state once after physics.
            read_plan.refresh()
        self.action_manager.process_action(actions)
        self._sim_step_counter += self._cfg.sim_substeps
        self.action_manager.apply_action()
        return self._control_to_backend_boundary()

    def update_state(self, state: TorchEnvState) -> TorchEnvState:
        # Physics stepping and reset/set_state lifecycles sit outside this private
        # scope, so values packed before ``step_tensor`` are stale here. Drop
        # the action-phase packet first; in-phase mutations explicitly invalidate
        # it below.
        timing = state.info.setdefault("timing", {})
        boundary_started = time.perf_counter()
        self.scene._invalidate_state_reads()
        with self.scene._scoped_state_reads():
            read_plan = self.scene._tensor_read_plan
            if read_plan is not None and not read_plan.ready:
                read_plan.refresh()
            boundary_ms = (time.perf_counter() - boundary_started) * 1000.0
            timing["update_state_read_boundary_ms"] = boundary_ms
            return self._update_state_in_read_phase(state)

    def _update_state_in_read_phase(self, state: TorchEnvState) -> TorchEnvState:
        timing = state.info.setdefault("timing", {})
        update_started_ns = time.perf_counter_ns()
        prelude_started_ns = time.perf_counter_ns()
        queue_started = time.perf_counter()
        phase_started = time.perf_counter()
        log: dict[str, Any] = {}
        state.info["log"] = log
        self.extras = state.info

        self.episode_length_buf.copy_(state.info["steps"]).add_(1)
        self.common_step_counter = self.step_counter + 1
        self._sim_step_counter = self.common_step_counter * self._cfg.sim_substeps
        timing["update_state_prelude_ms"] = (time.perf_counter_ns() - prelude_started_ns) / 1.0e6

        termination_compute_started = time.perf_counter()
        self.termination_manager.compute()
        termination_compute_ms = (time.perf_counter() - termination_compute_started) * 1000.0
        termination_timing = getattr(self.termination_manager, "last_step_timing_ms", {})
        timing.update(termination_timing)
        state.info["update_state_timing"] = dict(termination_timing)
        # Termination is normally the first update-state reduction after the
        # backend step. Its wall time can therefore drain pre-update GPU work;
        # keep that synchronization attribution separate from Manager work.
        timing["update_state_queue_drain_ms"] = (time.perf_counter() - queue_started) * 1000.0
        timing["update_state_termination_ms"] = (time.perf_counter() - phase_started) * 1000.0
        timing["update_state_termination_host_ms"] = termination_compute_ms
        phase_started = time.perf_counter()
        terminated = self.termination_manager.terminated
        time_outs = self.termination_manager.time_outs
        if self._cfg.is_finite_horizon:
            torch.logical_or(terminated, time_outs, out=self.reset_terminated)
            self.reset_time_outs.fill_(False)
        else:
            self.reset_terminated.copy_(terminated)
            self.reset_time_outs.copy_(time_outs)
        torch.logical_or(self.reset_terminated, self.reset_time_outs, out=self.reset_buf)
        timing["update_state_reset_flags_ms"] = (time.perf_counter() - phase_started) * 1000.0
        phase_started = time.perf_counter()

        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        reward_timing = getattr(self.reward_manager, "last_step_timing_ms", {})
        state.info["update_state_timing"].update(reward_timing)
        timing.update(reward_timing)
        timing["update_state_reward_ms"] = (time.perf_counter() - phase_started) * 1000.0
        phase_started = time.perf_counter()
        self._reward_log_names = self.reward_manager.step_reward_names
        self._reward_log_means = self.reward_manager.step_reward_means

        if self._cfg.sim_substeps == 1:
            self.metrics_manager.compute_substep()
        self.metrics_manager.compute()
        timing["update_state_metrics_ms"] = (time.perf_counter() - phase_started) * 1000.0
        phase_started = time.perf_counter()

        events_started = time.perf_counter()
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
            self._refresh_tensor_reads_after_mutation()
        timing["update_state_events_ms"] = (time.perf_counter() - events_started) * 1000.0

        command_preflight_started = time.perf_counter()
        self._command_dt.fill_(self.step_dt)
        self._command_dt.masked_fill_(self.reset_buf, 0.0)
        step_read_plan = self.scene._tensor_read_plan
        step_reset_capabilities = self._backend.get_tensor_capabilities()
        if self._uses_device_resident_reset(step_read_plan, step_reset_capabilities):
            assert step_read_plan is not None
            self._reset_state.declare_packed_reset_device(step_read_plan.device)
            step_reset_context = self._reset_state.scoped_device_tensor(self._all_env_rows)
        else:
            step_reset_context = self._reset_state.scoped(self._all_env_rows)
        timing["update_state_command_preflight_ms"] = (
            time.perf_counter() - command_preflight_started
        ) * 1000.0
        command_started = time.perf_counter()
        with step_reset_context:
            self.command_manager.compute(dt=self._command_dt)
            step_timing = getattr(self.command_manager, "last_step_timing_ms", {})
            state.info["update_state_timing"].update(step_timing)
            timing.update(step_timing)
        timing["update_state_command_compute_ms"] = (time.perf_counter() - command_started) * 1000.0
        command_epilogue_started = time.perf_counter()
        if self._reset_state.last_commit_had_writes:
            self.scene._invalidate_state_reads()
            self._refresh_tensor_reads_after_mutation()
        read_refresh_ms = (time.perf_counter() - command_epilogue_started) * 1000.0
        post_compute_started = time.perf_counter()
        self.command_manager.post_compute()
        post_compute_ms = (time.perf_counter() - post_compute_started) * 1000.0
        timing["update_state_command_read_refresh_ms"] = read_refresh_ms
        timing["update_state_command_post_compute_ms"] = post_compute_ms
        timing["update_state_command_epilogue_ms"] = (
            time.perf_counter() - command_epilogue_started
        ) * 1000.0
        timing["update_state_command_ms"] = (time.perf_counter() - phase_started) * 1000.0
        phase_started = time.perf_counter()

        manager_obs = self.observation_manager.compute(update_history=True)
        observation_timing = getattr(self.observation_manager, "last_step_timing_ms", {})
        state.info["update_state_timing"].update(observation_timing)
        timing.update(observation_timing)
        timing["update_state_observation_ms"] = (time.perf_counter() - phase_started) * 1000.0
        phase_started = time.perf_counter()

        mapped_obs = self._map_observations(manager_obs)
        self.obs_buf = mapped_obs
        self._has_transition = True
        timing["update_state_map_ms"] = (time.perf_counter() - phase_started) * 1000.0
        phase_started = time.perf_counter()

        publish_started = time.perf_counter()
        public_obs = {
            name: self._manager_tensor(values, dtype=self._dtype)
            for name, values in mapped_obs.items()
        }
        public_reward = self._manager_tensor(self.reward_buf, dtype=self._dtype)
        public_terminated = self._manager_tensor(self.reset_terminated, dtype=torch.bool)
        public_truncated = self._manager_tensor(self.reset_time_outs, dtype=torch.bool)
        timing["update_state_publication_ms"] = (time.perf_counter() - publish_started) * 1000.0

        replace_started = time.perf_counter()
        replacement = state.replace(
            obs=public_obs,
            reward=public_reward,
            terminated=public_terminated,
            truncated=public_truncated,
        )
        timing["update_state_state_replace_ms"] = (time.perf_counter() - replace_started) * 1000.0
        timing["update_state_publish_ms"] = (time.perf_counter() - phase_started) * 1000.0
        update_children = (
            timing.get("update_state_termination_ms"),
            timing.get("update_state_reward_ms"),
            timing.get("update_state_metrics_ms"),
            timing.get("update_state_command_ms"),
            timing.get("update_state_observation_ms"),
            timing.get("update_state_map_ms"),
            timing.get("update_state_publish_ms"),
        )
        if all(value is not None for value in update_children):
            epilogue_started_ns = time.perf_counter_ns()
            # Queue drain and the same boundary inside termination are excluded:
            # they drain pre-update GPU work already timed by step_core_ms.
            drain_ns = int(float(timing["update_state_queue_drain_ms"]) * 1.0e6)
            child_sum_ms = sum(float(value) for value in update_children)
            timing["update_state_child_sum_ms"] = child_sum_ms
            timing["update_state_nonattributed_ms"] = (
                time.perf_counter_ns() - update_started_ns - child_sum_ms * 1.0e6 + drain_ns
            ) / 1.0e6
            timing["update_state_timing_epilogue_ms"] = (
                time.perf_counter_ns() - epilogue_started_ns
            ) / 1.0e6
        return replacement

    def publish_reward_log(self) -> dict[str, float]:
        """Publish latest device reward means through the legacy log contract."""
        if not self._reward_log_names:
            return {}
        means = self._reward_log_means.detach().cpu().tolist()
        return {
            f"reward/{name}": float(mean)
            for name, mean in zip(self._reward_log_names, means, strict=True)
        }

    def _refresh_tensor_reads_after_mutation(self) -> None:
        """Repack scene tensor reads after an in-phase simulation mutation."""
        read_plan = self.scene._tensor_read_plan
        if read_plan is not None:
            read_plan.refresh()

    def _reset_manager_state(self, rows: torch.Tensor) -> None:
        """Re-run row-scoped manager reset after initial state allocation."""
        for manager in (
            self.observation_manager,
            self.action_manager,
            self.reward_manager,
            self.metrics_manager,
            self.curriculum_manager,
            self.event_manager,
            self.termination_manager,
        ):
            manager.reset(rows)

    def _compute_truncated(self, state: TorchEnvState) -> torch.Tensor:
        del state
        return torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

    def reset(
        self,
        env_indices: torch.Tensor | None = None,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        del options
        reset_timing: dict[str, float] = {}
        rows = self._normalize_reset_indices(env_indices)
        if seed is not None:
            self.seed(seed)
        if self._state is None:
            all_rows = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
            if not torch.equal(rows, all_rows):
                raise RuntimeError(
                    "ManagerBasedRlEnv requires a full reset before the first partial reset"
                )
            state = self.init_state()
            self._reset_manager_state(rows)
            return state.obs, {"log": state.info.get("log", {})}

        done_rows = rows[self.reset_buf[rows]]
        if self._has_transition and done_rows.numel() > 0:
            self.recorder_manager.record_pre_reset(done_rows)

        log: dict[str, Any] = {}
        self.curriculum_manager.compute(env_ids=rows)
        read_plan = self.scene._tensor_read_plan
        reset_capabilities = self._backend.get_tensor_capabilities()
        device_resident_reset = self._uses_device_resident_reset(read_plan, reset_capabilities)
        tensor_reset_events = False
        use_packed_reset = (
            read_plan is not None
            and read_plan.host_plan is not None
            # Force eager default materialization: a backend with different
            # native and public layouts must fall back before terms run.
            and self._reset_state.can_commit_packed(term_name="reset")
        )
        if use_packed_reset:
            assert read_plan is not None
            assert read_plan.host_plan is not None
            self._reset_state.declare_packed_reset_device(read_plan.device)
            tensor_reset_events = (
                self.event_manager.uses_tensor_reset_rows
                or self.command_manager.uses_tensor_reset_rows()
            )
            if tensor_reset_events:
                reset_context = self._reset_state.scoped_device_event_tensor_with_host_commit(
                    rows, read_plan.host_plan
                )
            else:
                reset_context = self._reset_state.scoped_tensor(rows, read_plan.host_plan)
        elif device_resident_reset:
            assert read_plan is not None
            self._reset_state.declare_packed_reset_device(read_plan.device)
            tensor_reset_events = self.event_manager.uses_tensor_reset_rows
            tensor_reset_events = tensor_reset_events or (
                self.command_manager.uses_tensor_reset_rows()
            )
            if tensor_reset_events:
                reset_context = self._reset_state.scoped_device_event_tensor(rows)
            else:
                reset_context = self._reset_state.scoped_device_tensor(rows)
        else:
            reset_context = self._reset_state.scoped(rows)
        command_event_started = time.perf_counter()
        event_term_count = len(self.event_manager.active_terms.get("reset", ()))
        command_term_count = len(self.command_manager.active_terms)
        reset_owner = self.reset_owner_manager.owner
        owns_command_reset = bool(reset_owner is not None and reset_owner.cfg.owns_command_reset)
        with reset_context:
            if "reset" in self.event_manager.available_modes:
                self.event_manager.apply(
                    mode="reset",
                    env_ids=rows,
                    global_env_step_count=self.step_counter,
                )
            if owns_command_reset:
                assert reset_owner is not None
                log.update(reset_owner.reset_transaction(rows))
            else:
                log.update(self.command_manager.reset(rows))
            reset_timing.update(getattr(self.command_manager, "last_reset_timing_ms", {}))
            reset_timing.update(self.command_manager.reset_diagnostics())
            reset_commit_started = time.perf_counter()
        reset_timing["reset_done_reset_commit_ms"] = (
            time.perf_counter() - reset_commit_started
        ) * 1000.0
        reset_timing["reset_done_command_event_ms"] = (
            time.perf_counter() - command_event_started
        ) * 1000.0

        manager_state_started = time.perf_counter()
        reset_managers: tuple[Any, ...] = (
            *(
                ()
                if reset_owner is not None and reset_owner.cfg.owns_observation_reset
                else (self.observation_manager,)
            ),
            *(
                ()
                if reset_owner is not None and reset_owner.cfg.owns_action_reset
                else (self.action_manager,)
            ),
            self.reward_manager,
            *(
                ()
                if reset_owner is not None and reset_owner.cfg.owns_metric_reset
                else (self.metrics_manager,)
            ),
            self.curriculum_manager,
            self.event_manager,
            self.termination_manager,
        )
        for manager in reset_managers:
            log.update(manager.reset(rows))
        if reset_owner is not None:
            log.update(reset_owner.reset_committed(rows))
        observation_term_count = sum(
            len(terms) for terms in self.observation_manager.active_terms.values()
        )
        reset_timing.update(
            {
                "reset_done_event_term_count": float(event_term_count),
                "reset_done_command_term_count": float(command_term_count),
                "reset_done_manager_reset_count": float(len(reset_managers)),
                "reset_done_observation_term_count": float(observation_term_count),
            }
        )
        reset_timing["reset_done_manager_state_ms"] = (
            time.perf_counter() - manager_state_started
        ) * 1000.0

        self.episode_length_buf[rows] = 0
        if self._reset_state.scene_layout is not None:
            self._control[rows] = self._initial_backend_control()[rows]
        else:
            self._control[rows] = 0.0
        self._manual_reset_pending[rows] = False
        if self._state is not None:
            self._state.info["steps"][rows] = 0

        # The read phase starts only after the reset-state transaction above
        # committed, so cached getter values are post-set_state reads shared
        # across terms (issue #1295).
        state_publish_started = time.perf_counter()
        with self.scene._scoped_state_reads():
            read_plan = self.scene._tensor_read_plan
            if read_plan is not None:
                # A host-bridge plan owns the paired selected-row boundary from
                # ``apply_reset``. IsaacGym's external CUDA IPC worker marks
                # body/sensor views stale after a selected reset, so one public
                # zero-control tensor step materializes the post-reset packet
                # before Manager terms read those stable views.
                if use_packed_reset:
                    selected_read_started = time.perf_counter()
                    read_plan.refresh_selected()
                    reset_timing["reset_done_selected_read_ms"] = (
                        time.perf_counter() - selected_read_started
                    ) * 1000.0
                else:
                    read_drain_started = time.perf_counter()
                    self._warm_external_cuda_ipc_views()
                    reset_timing["reset_done_read_drain_ms"] = (
                        time.perf_counter() - read_drain_started
                    ) * 1000.0
                    selected_read_started = time.perf_counter()
                    read_plan.refresh()
                    reset_timing["reset_done_selected_read_ms"] = (
                        time.perf_counter() - selected_read_started
                    ) * 1000.0
            command_refresh_started = time.perf_counter()
            self.command_manager.compute(dt=0.0, env_ids=rows)
            reset_timing["reset_done_command_refresh_ms"] = (
                time.perf_counter() - command_refresh_started
            ) * 1000.0
            command_post_compute_started = time.perf_counter()
            self.command_manager.post_compute()
            reset_timing["reset_done_command_post_compute_ms"] = (
                time.perf_counter() - command_post_compute_started
            ) * 1000.0
            reset_timing.update(getattr(self.command_manager, "last_post_compute_timing_ms", {}))
            observation_started = time.perf_counter()
            # Row-scoped reset rebuild (issue #1259 R2): the observation manager
            # returns only the reset rows, so no full-batch slice is needed here.
            manager_obs = self.observation_manager.compute(update_history=True, env_ids=rows)
            reset_timing["reset_done_observation_ms"] = (
                time.perf_counter() - observation_started
            ) * 1000.0
        reset_timing["reset_done_state_publish_ms"] = (
            time.perf_counter() - state_publish_started
        ) * 1000.0
        mapped_obs = self._map_observations(manager_obs, num_rows=rows.numel())
        observation_scatter_started = time.perf_counter()
        reset_obs = {
            name: self._manager_tensor(values, dtype=self._dtype)
            for name, values in mapped_obs.items()
        }
        reset_timing["reset_done_observation_scatter_ms"] = (
            time.perf_counter() - observation_scatter_started
        ) * 1000.0
        self._last_reset_manager_timing_ms = reset_timing

        if self._state is not None:
            for name in mapped_obs:
                self._state.obs[name].index_copy_(0, rows, reset_obs[name])
            if self._autoreset_reset_active:
                # The Manager already published into the full public state.
                # ``TorchEnv._reset_done_envs`` must not validate and scatter
                # the selected rows a second time.
                self._autoreset_scattered_reset_obs = True
                # Autoreset runs at the tail of step(): keep this step's
                # per-step log entries (reward/* etc., computed pre-reset) and
                # layer manager reset extras on top, so consumers still see the
                # transition's metrics.
                step_log = self._state.info.get("log")
                if step_log:
                    log = {**step_log, **log}
            self._state.info["log"] = log
            if not self._autoreset_reset_active:
                self._state.terminated[rows] = False
                self._state.truncated[rows] = False
                self.reset_buf[rows] = False
                self.reset_terminated[rows] = False
                self.reset_time_outs[rows] = False
        if not self.obs_buf or set(self.obs_buf) != set(mapped_obs):
            self.obs_buf = mapped_obs
        else:
            for name, values in mapped_obs.items():
                self.obs_buf[name][rows] = values
        self.extras = self._state.info if self._state is not None else {"log": log}
        self.recorder_manager.record_post_reset(rows)
        return reset_obs, {"log": log}

    def _collect_reset_backend_timing_ms(self) -> dict[str, float]:
        timing = dict(super()._collect_reset_backend_timing_ms())
        timing.update(self._reset_state.last_set_state_timing_ms)
        timing.update(self._last_reset_manager_timing_ms)
        return timing

    def _map_observations(
        self,
        manager_obs: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        num_rows: int | None = None,
    ) -> dict[str, torch.Tensor]:
        mapping = {"obs": self._cfg.policy_observation_group}
        if self._cfg.critic_observation_group is not None:
            mapping["critic"] = self._cfg.critic_observation_group
        mapped: dict[str, torch.Tensor] = {}
        for output_name, group_name in mapping.items():
            value = manager_obs[group_name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"ManagerBasedRlEnv observation group '{group_name}' returned "
                    f"{type(value).__name__}, expected torch.Tensor"
                )
            expected = (
                self.num_envs if num_rows is None else num_rows,
                self._mapped_obs_dims[output_name],
            )
            if value.shape != expected:
                raise ValueError(
                    f"ManagerBasedRlEnv observation group '{group_name}' returned shape "
                    f"{value.shape}, expected {expected} for TorchEnvState.obs['{output_name}']"
                )
            mapped[output_name] = value
        return mapped

    def get_observations(self) -> dict[str, torch.Tensor]:
        if self._state is None:
            return self.init_state().obs
        return self._state.obs

    def set_episode_length_buf(self, values: torch.Tensor) -> None:
        """Overwrite per-env episode counters (cold path, runner-init only).

        ``episode_length_buf`` mirrors ``state.info["steps"]``: each step writes
        ``info["steps"] + 1`` into the buffer and ``reset()`` zeroes both, so a
        direct assignment must update the two together. RL runners (e.g. RSL-RL
        ``init_at_random_ep_len``) call this once before learning starts to
        stagger initial episode lengths across envs.
        """
        if not isinstance(values, torch.Tensor):
            raise TypeError(
                "ManagerBasedRlEnv episode counters must be a torch.Tensor, "
                f"got {type(values).__name__}"
            )
        if values.shape != (self.num_envs,):
            raise ValueError(
                f"ManagerBasedRlEnv.set_episode_length_buf expects shape "
                f"({self.num_envs},), got {values.shape}"
            )
        if values.dtype != torch.int64 or values.device != self.device:
            raise TypeError(
                f"ManagerBasedRlEnv episode counters must be int64 tensors on {self.device}"
            )
        if bool((values < 0).any()):
            raise ValueError("episode length counters must be non-negative")
        self.episode_length_buf.copy_(values)
        if self._state is not None:
            self._state.info["steps"].copy_(values)

    def seed(self, seed: int = -1) -> int:
        if seed == -1:
            seed = secrets.randbits(63)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"ManagerBasedRlEnv seed must be a non-negative integer, got {seed!r}")
        replacement = np.random.default_rng(seed)
        self.rng.bit_generator.state = replacement.bit_generator.state
        if self._torch_rng_owner is not None:
            self._torch_rng_owner.manual_seed(seed)
        self._cfg.seed = seed
        return seed

    def import_training_state(self, state: Mapping[str, Any]) -> None:
        """Restore the authoritative counter and its manager-derived counters."""
        super().import_training_state(state)
        self.common_step_counter = self.step_counter
        self._sim_step_counter = self.step_counter * self._cfg.sim_substeps

    def close(self) -> None:
        read_plan = self.scene._tensor_read_plan
        if read_plan is not None:
            read_plan.close()
            self.scene._tensor_read_plan = None
        self.recorder_manager.close()
        super().close()


def make_manager_based_rl_env(
    cfg: ManagerBasedRlEnvCfg,
    num_envs: int = 1,
    backend_type: str = "mujoco",
) -> ManagerBasedRlEnv:
    """Construct the generic Registry-owned Manager-Based production runtime."""
    from unilab.base.registry import (
        _SHELVED_SIM_BACKENDS,
        _TENSOR_MANAGER_BACKEND_SCOPE_ERROR,
        _TENSOR_MANAGER_SUPPORTED_SIM_BACKENDS,
    )

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
    if backend_type in _SHELVED_SIM_BACKENDS:
        raise ValueError(_TENSOR_MANAGER_BACKEND_SCOPE_ERROR.format(backend=backend_type))
    if backend_type not in _TENSOR_MANAGER_SUPPORTED_SIM_BACKENDS:
        raise ValueError(
            "make_manager_based_rl_env supports only the tensor Manager backends "
            f"{', '.join(_TENSOR_MANAGER_SUPPORTED_SIM_BACKENDS)}; got {backend_type!r}."
        )

    cfg.validate()
    if cfg.fixed_model_variants is not None and cfg.scene is not None:
        # Validate the complete task identity before allocating backend resources.
        cfg.scene.fixed_variant_plan = _build_fixed_variant_plan(cfg.fixed_model_variants, num_envs)
    # Constrain the process before backend materialization so native pools size
    # themselves from the rank-owned CPU block.
    apply_env_cpu_runtime(cfg.cpu_ids)
    assert cfg.scene is not None
    base_name, body_state_requested, tracked_body_names = _resolve_backend_entity_contract(cfg)
    backend_kwargs = env_backend_kwargs(cfg)
    backend_kwargs["base_name"] = base_name
    if backend_type == "mujoco" and tracked_body_names is not None:
        backend_kwargs["tracked_body_names"] = tracked_body_names

    backend = create_backend(
        backend_type,
        cfg.scene,
        num_envs,
        cfg.sim_dt,
        body_state_required=body_state_requested,
        **backend_kwargs,
    )
    try:
        if cfg.scene.fixed_variant_plan is not None:
            _require_fixed_variant_support(
                backend.get_dr_capabilities(), cfg.scene.fixed_variant_plan
            )
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
