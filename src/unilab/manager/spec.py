"""Declarative, backend-independent specifications for managed tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import TypeAlias

import numpy as np

from unilab.base.backend.batch import (
    BackendBatchCounterBudget,
    BufferContract,
    ControlSpec,
    ExecutionProfile,
    PhysicalUnit,
    ReferenceFrame,
    StateEntityKind,
    StateFieldKind,
)
from unilab.base.backend.mutation import (
    MutationBaseline,
    MutationCommitPhase,
    MutationEntityKind,
    MutationFieldKind,
    MutationOperation,
    MutationPersistence,
    MutationRecomputeLevel,
    MutationTargetKind,
    MutationTrigger,
)
from unilab.dr.keyed_rng import (
    KEYED_RNG_ALGORITHM,
    KeyedRandomSpec,
    RandomCorrelation,
    RandomDistribution,
)

from .entities import EntityKind, EntitySelector, ManagerContractError

ParameterValue: TypeAlias = bool | int | float | str | tuple[int, ...] | tuple[float, ...]
FrozenParameters: TypeAlias = tuple[tuple[str, ParameterValue], ...]


class TermPhase(str, Enum):
    STARTUP = "startup"
    RESET = "reset"
    ACTION = "action"
    PRE_PHYSICS = "pre_physics"
    POST_PHYSICS = "post_physics"
    TERMINATION = "termination"
    REWARD = "reward"
    METRIC = "metric"
    TERMINAL_OBSERVATION = "terminal_observation"
    AUTORESET = "autoreset"
    OBSERVATION = "observation"


TERM_PHASE_ORDER: tuple[TermPhase, ...] = (
    TermPhase.STARTUP,
    TermPhase.RESET,
    TermPhase.ACTION,
    TermPhase.PRE_PHYSICS,
    TermPhase.POST_PHYSICS,
    TermPhase.TERMINATION,
    TermPhase.REWARD,
    TermPhase.METRIC,
    TermPhase.TERMINAL_OBSERVATION,
    TermPhase.AUTORESET,
    TermPhase.OBSERVATION,
)


class TermRole(str, Enum):
    ACTION = "action"
    EVENT = "event"
    OBSERVATION = "observation"
    REWARD = "reward"
    TERMINATION = "termination"
    METRIC = "metric"


_TERM_ROLE_PHASES = {
    TermRole.ACTION: frozenset({TermPhase.ACTION}),
    TermRole.EVENT: frozenset(
        {
            TermPhase.STARTUP,
            TermPhase.RESET,
            TermPhase.PRE_PHYSICS,
            TermPhase.POST_PHYSICS,
            TermPhase.AUTORESET,
        }
    ),
    TermRole.OBSERVATION: frozenset({TermPhase.TERMINAL_OBSERVATION, TermPhase.OBSERVATION}),
    TermRole.REWARD: frozenset({TermPhase.REWARD}),
    TermRole.TERMINATION: frozenset({TermPhase.TERMINATION}),
    TermRole.METRIC: frozenset({TermPhase.METRIC}),
}


class ParameterKind(str, Enum):
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    INT_TUPLE = "int_tuple"
    FLOAT_TUPLE = "float_tuple"


class QuaternionOrder(str, Enum):
    NONE = "none"
    WXYZ = "wxyz"
    XYZW = "xyzw"


class NormalizationMode(str, Enum):
    NONE = "none"
    EMPIRICAL = "empirical"


def _non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManagerContractError(f"{name} must be a non-empty string")
    return value.strip()


def _shape(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise ManagerContractError(f"{name} must be a non-empty tuple")
    if any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in value):
        raise ManagerContractError(f"{name} dimensions must be positive integers")
    return value


def _freeze_parameter_value(value: object) -> ParameterValue:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ManagerContractError("parameter floats must be finite")
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = tuple(value)
        if not items:
            raise ManagerContractError("tuple parameters must not be empty")
        if all(isinstance(item, int) and not isinstance(item, bool) for item in items):
            return tuple(int(item) for item in items)
        if all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and np.isfinite(float(item))
            for item in items
        ):
            return tuple(float(item) for item in items)
    raise ManagerContractError(f"unsupported parameter value {value!r}")


def freeze_parameters(values: Mapping[str, object] | None = None) -> FrozenParameters:
    if values is None:
        return ()
    if not isinstance(values, Mapping):
        raise ManagerContractError("parameters must be a mapping")
    normalized = []
    for key, value in values.items():
        normalized.append((_non_empty(key, "parameter key"), _freeze_parameter_value(value)))
    keys = [key for key, _ in normalized]
    if len(set(keys)) != len(keys):
        raise ManagerContractError("parameter keys must be unique")
    return tuple(sorted(normalized, key=lambda item: item[0]))


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    kind: ParameterKind
    required: bool = True
    default: ParameterValue | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _non_empty(self.key, "parameter spec key"))
        if not isinstance(self.kind, ParameterKind):
            raise ManagerContractError("parameter kind must be a ParameterKind")
        if not isinstance(self.required, bool):
            raise ManagerContractError("parameter required must be a bool")
        if self.required and self.default is not None:
            raise ManagerContractError("required parameters cannot declare a default")
        if not self.required and self.default is None:
            raise ManagerContractError("optional parameters require a default")
        if self.default is not None:
            object.__setattr__(self, "default", _freeze_parameter_value(self.default))


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str
    frame: ReferenceFrame = ReferenceFrame.NONE
    unit: PhysicalUnit = PhysicalUnit.UNITLESS
    quaternion_order: QuaternionOrder = QuaternionOrder.NONE

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", _shape(self.shape, "tensor shape"))
        try:
            dtype = np.dtype(_non_empty(self.dtype, "tensor dtype")).name
        except TypeError as exc:
            raise ManagerContractError(f"invalid tensor dtype {self.dtype!r}") from exc
        object.__setattr__(self, "dtype", dtype)
        if not isinstance(self.frame, ReferenceFrame):
            raise ManagerContractError("tensor frame must be a ReferenceFrame")
        if not isinstance(self.unit, PhysicalUnit):
            raise ManagerContractError("tensor unit must be a PhysicalUnit")
        if not isinstance(self.quaternion_order, QuaternionOrder):
            raise ManagerContractError("quaternion_order must be a QuaternionOrder")
        is_quaternion = self.unit is PhysicalUnit.QUATERNION
        if is_quaternion != (self.quaternion_order is not QuaternionOrder.NONE):
            raise ManagerContractError(
                "quaternion tensors require wxyz/xyzw order and non-quaternion tensors require none"
            )
        if is_quaternion and self.shape[-1] != 4:
            raise ManagerContractError("quaternion tensor trailing dimension must be 4")

    @property
    def width(self) -> int:
        return int(np.prod(self.shape))


_STATE_ENTITY_KINDS = {
    EntityKind.ROOT: StateEntityKind.ROOT,
    EntityKind.BODY: StateEntityKind.BODY,
    EntityKind.JOINT: StateEntityKind.JOINT,
    EntityKind.DOF: StateEntityKind.DOF,
    EntityKind.SENSOR: StateEntityKind.SENSOR,
    EntityKind.SITE: StateEntityKind.SITE,
}


@dataclass(frozen=True)
class StateRequirement:
    semantic_key: str
    selector: EntitySelector
    field_kind: StateFieldKind
    tensor: TensorSpec
    entity_axis: int | None = 0
    capability_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "semantic_key", _non_empty(self.semantic_key, "state key"))
        if not isinstance(self.selector, EntitySelector):
            raise ManagerContractError("state selector must be an EntitySelector")
        if self.selector.kind not in _STATE_ENTITY_KINDS:
            raise ManagerContractError(
                f"entity kind {self.selector.kind.value!r} cannot be used as typed state"
            )
        if not isinstance(self.field_kind, StateFieldKind):
            raise ManagerContractError("state field_kind must be a StateFieldKind")
        if not isinstance(self.tensor, TensorSpec):
            raise ManagerContractError("state tensor must be a TensorSpec")
        if self.entity_axis is not None:
            if (
                isinstance(self.entity_axis, bool)
                or not isinstance(self.entity_axis, int)
                or not 0 <= self.entity_axis < len(self.tensor.shape)
            ):
                raise ManagerContractError("entity_axis must index the state tensor shape")
        capability = (
            self.capability_key or f"state.{self.selector.kind.value}.{self.field_kind.value}"
        )
        object.__setattr__(self, "capability_key", _non_empty(capability, "state capability key"))

    @property
    def state_entity_kind(self) -> StateEntityKind:
        return _STATE_ENTITY_KINDS[self.selector.kind]


_MUTATION_ENTITY_KINDS = {
    EntityKind.GLOBAL: MutationEntityKind.GLOBAL,
    # A semantic ROOT is a task-level alias for the single floating-base
    # body.  The shared backend mutation contract intentionally has no ROOT
    # entity kind: concrete backends bind the exact base body on their cold
    # path and own the qpos/qvel translation.
    EntityKind.ROOT: MutationEntityKind.BODY,
    EntityKind.BODY: MutationEntityKind.BODY,
    EntityKind.JOINT: MutationEntityKind.JOINT,
    EntityKind.DOF: MutationEntityKind.DOF,
    EntityKind.ACTUATOR: MutationEntityKind.ACTUATOR,
    EntityKind.GEOM: MutationEntityKind.GEOM,
    EntityKind.SITE: MutationEntityKind.SITE,
    EntityKind.TASK: MutationEntityKind.TASK,
}


@dataclass(frozen=True)
class MutationRandomization:
    """Backend-neutral random semantics for one managed mutation Event.

    The backend mutation ABI deliberately does not carry this descriptor.
    A compiler freezes the semantic identity here, then an executor adds
    the actual row shape from the backend-bound mutation value contract.
    """

    distribution: RandomDistribution
    parameters: tuple[float, float]
    correlation: RandomCorrelation
    algorithm: str = KEYED_RNG_ALGORITHM

    def __post_init__(self) -> None:
        try:
            probe = KeyedRandomSpec(
                term_key="manager.randomization.validation",
                term_version="1",
                row_shape=(1,),
                distribution=self.distribution,
                correlation=self.correlation,
                parameters=self.parameters,
                algorithm=self.algorithm,
            )
        except ValueError as exc:
            raise ManagerContractError(f"invalid mutation randomization: {exc}") from exc
        object.__setattr__(self, "parameters", probe.parameters)
        object.__setattr__(self, "algorithm", probe.algorithm)


@dataclass(frozen=True)
class MutationTemplate:
    key_suffix: str
    target_key: str
    target_kind: MutationTargetKind
    selector: EntitySelector | None
    field_kind: MutationFieldKind
    trigger: MutationTrigger
    commit_phase: MutationCommitPhase
    operation: MutationOperation
    baseline: MutationBaseline
    persistence: MutationPersistence
    recompute: MutationRecomputeLevel
    value_template: BufferContract
    capability_key: str | None = None
    randomization: MutationRandomization | None = None

    def __post_init__(self) -> None:
        suffix = self.key_suffix.strip() if isinstance(self.key_suffix, str) else None
        if suffix is None:
            raise ManagerContractError("mutation key_suffix must be a string")
        object.__setattr__(self, "key_suffix", suffix)
        object.__setattr__(self, "target_key", _non_empty(self.target_key, "mutation target key"))
        if not isinstance(self.target_kind, MutationTargetKind):
            raise ManagerContractError("mutation target_kind must be a MutationTargetKind")
        if self.selector is None:
            if self.target_kind is not MutationTargetKind.MODEL_PARAMETER:
                raise ManagerContractError("only global model mutations may omit a selector")
        else:
            if not isinstance(self.selector, EntitySelector):
                raise ManagerContractError("mutation selector must be an EntitySelector or None")
            try:
                _MUTATION_ENTITY_KINDS[self.selector.kind]
            except KeyError as exc:
                raise ManagerContractError(
                    f"entity kind {self.selector.kind.value!r} cannot be mutated"
                ) from exc
        if not isinstance(self.field_kind, MutationFieldKind):
            raise ManagerContractError("mutation field_kind must be a MutationFieldKind")
        for value, expected, name in (
            (self.trigger, MutationTrigger, "trigger"),
            (self.commit_phase, MutationCommitPhase, "commit_phase"),
            (self.operation, MutationOperation, "operation"),
            (self.baseline, MutationBaseline, "baseline"),
            (self.persistence, MutationPersistence, "persistence"),
            (self.recompute, MutationRecomputeLevel, "recompute"),
        ):
            if not isinstance(value, expected):
                raise ManagerContractError(f"mutation {name} must be a {expected.__name__}")
        if not isinstance(self.value_template, BufferContract):
            raise ManagerContractError("mutation value_template must be a BufferContract")
        if self.randomization is not None and not isinstance(
            self.randomization, MutationRandomization
        ):
            raise ManagerContractError(
                "mutation randomization must be a MutationRandomization or None"
            )
        capability = self.capability_key or self.target_key
        object.__setattr__(
            self, "capability_key", _non_empty(capability, "mutation capability key")
        )

    @property
    def entity_kind(self) -> MutationEntityKind:
        if self.selector is None:
            return MutationEntityKind.GLOBAL
        return _MUTATION_ENTITY_KINDS[self.selector.kind]


@dataclass(frozen=True)
class TermDefinition:
    key: str
    version: str
    phase: TermPhase
    role: TermRole
    parameters: tuple[ParameterSpec, ...] = ()
    state_requirements: tuple[StateRequirement, ...] = ()
    output: TensorSpec | None = None
    mutation_templates: tuple[MutationTemplate, ...] = ()
    required_capabilities: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _non_empty(self.key, "term definition key"))
        object.__setattr__(self, "version", _non_empty(self.version, "term definition version"))
        if not isinstance(self.phase, TermPhase):
            raise ManagerContractError("term phase must be a TermPhase")
        if not isinstance(self.role, TermRole):
            raise ManagerContractError("term role must be a TermRole")
        if self.phase not in _TERM_ROLE_PHASES[self.role]:
            raise ManagerContractError(
                f"term role {self.role.value!r} cannot run in phase {self.phase.value!r}"
            )
        for name, values, expected in (
            ("parameters", self.parameters, ParameterSpec),
            ("state_requirements", self.state_requirements, StateRequirement),
            ("mutation_templates", self.mutation_templates, MutationTemplate),
        ):
            if not isinstance(values, tuple) or any(
                not isinstance(item, expected) for item in values
            ):
                raise ManagerContractError(f"term {name} must be a tuple of {expected.__name__}")
        parameter_keys = [item.key for item in self.parameters]
        if len(set(parameter_keys)) != len(parameter_keys):
            raise ManagerContractError("term parameter schema keys must be unique")
        state_keys = [item.semantic_key for item in self.state_requirements]
        if len(set(state_keys)) != len(state_keys):
            raise ManagerContractError("term state requirement keys must be unique")
        suffixes = [item.key_suffix for item in self.mutation_templates]
        if len(set(suffixes)) != len(suffixes):
            raise ManagerContractError("term mutation key suffixes must be unique")
        if len(self.mutation_templates) > 1 and "" in suffixes:
            raise ManagerContractError("multi-mutation terms require a non-empty key suffix")
        if self.output is not None and not isinstance(self.output, TensorSpec):
            raise ManagerContractError("term output must be a TensorSpec or None")
        if self.role is TermRole.OBSERVATION and self.output is None:
            raise ManagerContractError("observation terms require an output")
        if not isinstance(self.required_capabilities, frozenset):
            raise ManagerContractError("required_capabilities must be a frozenset")
        normalized_caps = frozenset(
            _non_empty(item, "required capability") for item in self.required_capabilities
        )
        object.__setattr__(self, "required_capabilities", normalized_caps)


@dataclass(frozen=True)
class TermInvocation:
    key: str
    definition_key: str
    dependencies: tuple[str, ...] = ()
    parameters: FrozenParameters = ()
    observation_group: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _non_empty(self.key, "term key"))
        object.__setattr__(
            self, "definition_key", _non_empty(self.definition_key, "term definition key")
        )
        if not isinstance(self.dependencies, tuple):
            raise ManagerContractError("term dependencies must be a tuple")
        dependencies = tuple(_non_empty(item, "term dependency") for item in self.dependencies)
        if self.key in dependencies:
            raise ManagerContractError("term cannot depend on itself")
        if len(set(dependencies)) != len(dependencies):
            raise ManagerContractError("term dependencies must be unique")
        object.__setattr__(self, "dependencies", dependencies)
        if not isinstance(self.parameters, tuple):
            raise ManagerContractError("term parameters must be a frozen tuple")
        parameter_keys = [key for key, _ in self.parameters]
        if parameter_keys != sorted(parameter_keys) or len(set(parameter_keys)) != len(
            parameter_keys
        ):
            raise ManagerContractError("term parameters must use unique canonical key order")
        if self.observation_group is not None:
            object.__setattr__(
                self,
                "observation_group",
                _non_empty(self.observation_group, "observation group"),
            )

    @classmethod
    def create(
        cls,
        *,
        key: str,
        definition_key: str,
        dependencies: Sequence[str] = (),
        parameters: Mapping[str, object] | None = None,
        observation_group: str | None = None,
    ) -> TermInvocation:
        return cls(
            key=key,
            definition_key=definition_key,
            dependencies=tuple(dependencies),
            parameters=freeze_parameters(parameters),
            observation_group=observation_group,
        )


@dataclass(frozen=True)
class PolicySpec:
    observation_groups: tuple[str, ...]
    action_scale: tuple[float, ...]
    normalization: NormalizationMode = NormalizationMode.NONE

    def __post_init__(self) -> None:
        if not isinstance(self.observation_groups, tuple) or not self.observation_groups:
            raise ManagerContractError("policy observation_groups must be a non-empty tuple")
        groups = tuple(
            _non_empty(item, "policy observation group") for item in self.observation_groups
        )
        if len(set(groups)) != len(groups):
            raise ManagerContractError("policy observation groups must be unique")
        object.__setattr__(self, "observation_groups", groups)
        if not isinstance(self.action_scale, tuple) or not self.action_scale:
            raise ManagerContractError("policy action_scale must be a non-empty tuple")
        if any(isinstance(item, bool) or not isinstance(item, Real) for item in self.action_scale):
            raise ManagerContractError("policy action_scale values must be real numbers")
        scales = tuple(float(item) for item in self.action_scale)
        if any(not np.isfinite(item) or item <= 0 for item in scales):
            raise ManagerContractError("policy action_scale values must be finite and positive")
        object.__setattr__(self, "action_scale", scales)
        if not isinstance(self.normalization, NormalizationMode):
            raise ManagerContractError("policy normalization must be a NormalizationMode")


@dataclass(frozen=True)
class TaskSpec:
    key: str
    terms: tuple[TermInvocation, ...]
    control: ControlSpec
    execution_profile: ExecutionProfile
    executor_key: str
    policy: PolicySpec
    hot_path_budget: BackendBatchCounterBudget | None = None
    reset_hot_path_budget: BackendBatchCounterBudget | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _non_empty(self.key, "task key"))
        if not isinstance(self.terms, tuple) or not self.terms:
            raise ManagerContractError("task terms must be a non-empty tuple")
        if any(not isinstance(item, TermInvocation) for item in self.terms):
            raise ManagerContractError("task terms must contain only TermInvocation values")
        keys = [item.key for item in self.terms]
        if len(set(keys)) != len(keys):
            raise ManagerContractError("task term keys must be unique")
        if not isinstance(self.control, ControlSpec):
            raise ManagerContractError("task control must be a ControlSpec")
        if not isinstance(self.execution_profile, ExecutionProfile):
            raise ManagerContractError("task execution_profile must be an ExecutionProfile")
        if self.execution_profile is not ExecutionProfile.HOST_NUMPY:
            raise ManagerContractError("task execution_profile must be host_numpy")
        object.__setattr__(self, "executor_key", _non_empty(self.executor_key, "executor key"))
        if not isinstance(self.policy, PolicySpec):
            raise ManagerContractError("task policy must be a PolicySpec")
        for name, budget in (
            ("hot_path_budget", self.hot_path_budget),
            ("reset_hot_path_budget", self.reset_hot_path_budget),
        ):
            if budget is not None and not isinstance(budget, BackendBatchCounterBudget):
                raise ManagerContractError(
                    f"task {name} must be a BackendBatchCounterBudget or None"
                )

    @classmethod
    def create(
        cls,
        *,
        key: str,
        terms: Sequence[TermInvocation],
        control: ControlSpec,
        execution_profile: ExecutionProfile,
        executor_key: str,
        policy: PolicySpec,
        hot_path_budget: BackendBatchCounterBudget | None = None,
        reset_hot_path_budget: BackendBatchCounterBudget | None = None,
    ) -> TaskSpec:
        return cls(
            key=key,
            terms=tuple(terms),
            control=control,
            execution_profile=execution_profile,
            executor_key=executor_key,
            policy=policy,
            hot_path_budget=hot_path_budget,
            reset_hot_path_budget=reset_hot_path_budget,
        )
