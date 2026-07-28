"""Typed mutation contracts shared by managed tasks and simulation backends."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum, IntEnum
from typing import Any

from .batch import (
    BackendBatchContractError,
    BufferContract,
    BufferLifetime,
    BufferMutability,
    BufferOwner,
)

MUTATION_CONTRACT_VERSION = "backend-mutation-contract-v1"


class MutationContractError(BackendBatchContractError):
    """Raised when a typed mutation specification or batch violates its contract."""


class MutationTargetKind(str, Enum):
    MODEL_PARAMETER = "model_parameter"
    SIMULATION_STATE = "simulation_state"
    EXTERNAL_WRENCH = "external_wrench"
    TASK_STATE = "task_state"


class MutationEntityKind(str, Enum):
    GLOBAL = "global"
    BODY = "body"
    JOINT = "joint"
    DOF = "dof"
    ACTUATOR = "actuator"
    GEOM = "geom"
    SITE = "site"
    TASK = "task"


class MutationSelectorMode(str, Enum):
    """Cold-path matching mode for a semantic mutation selector.

    This intentionally mirrors no manager type.  The backend mutation
    contract must be usable by direct backend callers without importing the
    manager package, while a compiled manager task can still lower its
    selector metadata into this backend-neutral representation.
    """

    EXACT = "exact"
    REGEX = "regex"


class MutationFieldKind(str, Enum):
    GRAVITY = "gravity"
    MASS = "mass"
    CENTER_OF_MASS = "center_of_mass"
    INERTIA = "inertia"
    ARMATURE = "armature"
    FRICTION = "friction"
    STIFFNESS = "stiffness"
    DAMPING = "damping"
    POSITION = "position"
    ORIENTATION = "orientation"
    LINEAR_VELOCITY = "linear_velocity"
    ANGULAR_VELOCITY = "angular_velocity"
    FORCE = "force"
    TORQUE = "torque"
    VALUE = "value"


class MutationTrigger(str, Enum):
    STARTUP = "startup"
    RESET = "reset"
    INTERVAL = "interval"
    STEP = "step"


class MutationCommitPhase(str, Enum):
    STARTUP = "startup"
    RESET = "reset"
    PRE_PHYSICS = "pre_physics"
    POST_PHYSICS = "post_physics"


class MutationOperation(str, Enum):
    SET = "set"
    ADD = "add"
    SCALE = "scale"


class MutationBaseline(str, Enum):
    DEFAULT = "default"
    CURRENT = "current"


class MutationPersistence(str, Enum):
    PERSISTENT = "persistent"
    EPISODE = "episode"
    ONE_STEP = "one_step"


class MutationRecomputeLevel(IntEnum):
    NONE = 0
    KINEMATICS = 1
    DYNAMICS = 2
    FULL = 3


_TARGET_ENTITY_KINDS = {
    MutationTargetKind.MODEL_PARAMETER: frozenset(
        {
            MutationEntityKind.GLOBAL,
            MutationEntityKind.BODY,
            MutationEntityKind.JOINT,
            MutationEntityKind.DOF,
            MutationEntityKind.ACTUATOR,
            MutationEntityKind.GEOM,
            MutationEntityKind.SITE,
        }
    ),
    MutationTargetKind.SIMULATION_STATE: frozenset(
        {
            MutationEntityKind.BODY,
            MutationEntityKind.JOINT,
            MutationEntityKind.DOF,
        }
    ),
    MutationTargetKind.EXTERNAL_WRENCH: frozenset({MutationEntityKind.BODY}),
    MutationTargetKind.TASK_STATE: frozenset({MutationEntityKind.TASK}),
}

_ENTITY_FIELD_KINDS = {
    (MutationTargetKind.MODEL_PARAMETER, MutationEntityKind.GLOBAL): frozenset(
        {MutationFieldKind.GRAVITY}
    ),
    (MutationTargetKind.MODEL_PARAMETER, MutationEntityKind.BODY): frozenset(
        {
            MutationFieldKind.MASS,
            MutationFieldKind.CENTER_OF_MASS,
            MutationFieldKind.INERTIA,
            MutationFieldKind.POSITION,
            MutationFieldKind.ORIENTATION,
        }
    ),
    (MutationTargetKind.MODEL_PARAMETER, MutationEntityKind.JOINT): frozenset(
        {
            MutationFieldKind.POSITION,
            MutationFieldKind.STIFFNESS,
            MutationFieldKind.DAMPING,
        }
    ),
    (MutationTargetKind.MODEL_PARAMETER, MutationEntityKind.DOF): frozenset(
        {
            MutationFieldKind.ARMATURE,
            MutationFieldKind.DAMPING,
            MutationFieldKind.FRICTION,
        }
    ),
    (MutationTargetKind.MODEL_PARAMETER, MutationEntityKind.ACTUATOR): frozenset(
        {MutationFieldKind.STIFFNESS, MutationFieldKind.DAMPING}
    ),
    (MutationTargetKind.MODEL_PARAMETER, MutationEntityKind.GEOM): frozenset(
        {
            MutationFieldKind.FRICTION,
            MutationFieldKind.POSITION,
            MutationFieldKind.ORIENTATION,
        }
    ),
    (MutationTargetKind.MODEL_PARAMETER, MutationEntityKind.SITE): frozenset(
        {MutationFieldKind.POSITION, MutationFieldKind.ORIENTATION}
    ),
    (MutationTargetKind.SIMULATION_STATE, MutationEntityKind.BODY): frozenset(
        {
            MutationFieldKind.POSITION,
            MutationFieldKind.ORIENTATION,
            MutationFieldKind.LINEAR_VELOCITY,
            MutationFieldKind.ANGULAR_VELOCITY,
        }
    ),
    (MutationTargetKind.SIMULATION_STATE, MutationEntityKind.JOINT): frozenset(
        {
            MutationFieldKind.POSITION,
            MutationFieldKind.LINEAR_VELOCITY,
            MutationFieldKind.ANGULAR_VELOCITY,
        }
    ),
    (MutationTargetKind.SIMULATION_STATE, MutationEntityKind.DOF): frozenset(
        {
            MutationFieldKind.POSITION,
            MutationFieldKind.LINEAR_VELOCITY,
            MutationFieldKind.ANGULAR_VELOCITY,
        }
    ),
    (MutationTargetKind.EXTERNAL_WRENCH, MutationEntityKind.BODY): frozenset(
        {MutationFieldKind.FORCE, MutationFieldKind.TORQUE}
    ),
    (MutationTargetKind.TASK_STATE, MutationEntityKind.TASK): frozenset({MutationFieldKind.VALUE}),
}

_TARGET_FIELD_KINDS = {
    MutationTargetKind.MODEL_PARAMETER: frozenset(
        {
            MutationFieldKind.GRAVITY,
            MutationFieldKind.MASS,
            MutationFieldKind.CENTER_OF_MASS,
            MutationFieldKind.INERTIA,
            MutationFieldKind.ARMATURE,
            MutationFieldKind.FRICTION,
            MutationFieldKind.STIFFNESS,
            MutationFieldKind.DAMPING,
            MutationFieldKind.POSITION,
            MutationFieldKind.ORIENTATION,
        }
    ),
    MutationTargetKind.SIMULATION_STATE: frozenset(
        {
            MutationFieldKind.POSITION,
            MutationFieldKind.ORIENTATION,
            MutationFieldKind.LINEAR_VELOCITY,
            MutationFieldKind.ANGULAR_VELOCITY,
        }
    ),
    MutationTargetKind.EXTERNAL_WRENCH: frozenset(
        {MutationFieldKind.FORCE, MutationFieldKind.TORQUE}
    ),
    MutationTargetKind.TASK_STATE: frozenset({MutationFieldKind.VALUE}),
}


def _non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MutationContractError(f"{name} must be a non-empty string")
    return value.strip()


def _enum(value: Any, enum_type: type[Enum], name: str) -> None:
    if not isinstance(value, enum_type):
        choices = ", ".join(str(item.value) for item in enum_type)
        raise MutationContractError(f"{name} must be one of: {choices}")


def _count(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MutationContractError(f"{name} must be an integer >= {minimum}")
    return value


def _enum_set(value: Any, enum_type: type[Enum], name: str) -> None:
    if not isinstance(value, frozenset) or not value:
        raise MutationContractError(f"{name} must be a non-empty frozenset")
    if any(not isinstance(item, enum_type) for item in value):
        raise MutationContractError(f"{name} contains an invalid {enum_type.__name__}")


def _validate_value_template(value: BufferContract) -> None:
    if not isinstance(value, BufferContract):
        raise MutationContractError("mutation value_template must be a BufferContract")
    if value.owner is not BufferOwner.MANAGER:
        raise MutationContractError("mutation values must be manager-owned")
    if value.mutability is not BufferMutability.READ_ONLY:
        raise MutationContractError("mutation values must be read-only to the backend")
    if value.lifetime is not BufferLifetime.UNTIL_COMMIT:
        raise MutationContractError("mutation values must use until_commit lifetime")
    if not value.address_stable:
        raise MutationContractError("mutation value addresses must be plan-stable")


@dataclass(frozen=True)
class MutationSelectorSpec:
    """Immutable selector metadata supplied to a backend on the cold path.

    ``semantic_key`` identifies the task-level selector.  ``expressions`` are
    the raw model-facing expressions that a concrete backend may resolve
    without consulting a manager registry.  ``entity_ids`` records the
    compiler's already validated selector binding; it is diagnostic and part
    of the manager binding fingerprint, not a physics-backend ID handoff.

    A direct backend caller may use :meth:`exact` (or a legacy string accepted
    by :class:`MutationTargetSpec`) and therefore has no compiler IDs.  A
    manager-compiled selector always provides its concrete IDs.
    """

    semantic_key: str
    mode: MutationSelectorMode
    expressions: tuple[str, ...]
    entity_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "semantic_key", _non_empty(self.semantic_key, "selector key"))
        _enum(self.mode, MutationSelectorMode, "selector mode")
        if not isinstance(self.expressions, tuple) or not self.expressions:
            raise MutationContractError("selector expressions must be a non-empty tuple")
        expressions = tuple(_non_empty(value, "selector expression") for value in self.expressions)
        if len(set(expressions)) != len(expressions):
            raise MutationContractError("selector expressions must be unique")
        object.__setattr__(self, "expressions", expressions)
        if not isinstance(self.entity_ids, tuple):
            raise MutationContractError("selector entity_ids must be a tuple")
        if any(
            isinstance(entity_id, bool) or not isinstance(entity_id, int) or entity_id < 0
            for entity_id in self.entity_ids
        ):
            raise MutationContractError("selector entity_ids must be non-negative integers")
        if len(set(self.entity_ids)) != len(self.entity_ids):
            raise MutationContractError("selector entity_ids must be unique")
        if (
            self.mode is MutationSelectorMode.EXACT
            and self.entity_ids
            and len(self.expressions) != len(self.entity_ids)
        ):
            raise MutationContractError(
                "exact selector expressions must have one compiler entity_id each"
            )

    @classmethod
    def exact(cls, expression: str) -> MutationSelectorSpec:
        """Normalize a direct legacy exact selector without compiler binding."""

        expression = _non_empty(expression, "selector")
        return cls(
            semantic_key=expression,
            mode=MutationSelectorMode.EXACT,
            expressions=(expression,),
        )

    def require_exact_singleton(self, *, context: str) -> str:
        """Return one raw expression or reject an unsupported backend shape.

        Narrow backends such as the first typed MuJoCo and mjwarp reset paths
        support exactly one raw body/joint name.  They must reject regex and
        multi-selector descriptors during cold binding rather than guessing a
        mapping or doing a hot-path fallback lookup.
        """

        context = _non_empty(context, "selector context")
        if self.mode is not MutationSelectorMode.EXACT:
            raise MutationContractError(
                f"{context} only supports exact mutation selectors, got {self.mode.value!r}"
            )
        if len(self.expressions) != 1:
            raise MutationContractError(
                f"{context} requires exactly one mutation selector expression"
            )
        if self.entity_ids and len(self.entity_ids) != 1:
            raise MutationContractError(
                f"{context} requires exactly one compiler-bound mutation entity"
            )
        return self.expressions[0]


@dataclass(frozen=True)
class MutationTargetSpec:
    target_key: str
    target_kind: MutationTargetKind
    entity_kind: MutationEntityKind
    field_kind: MutationFieldKind
    selector: MutationSelectorSpec | str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_key", _non_empty(self.target_key, "target_key"))
        _enum(self.target_kind, MutationTargetKind, "target_kind")
        _enum(self.entity_kind, MutationEntityKind, "entity_kind")
        _enum(self.field_kind, MutationFieldKind, "field_kind")
        if self.entity_kind not in _TARGET_ENTITY_KINDS[self.target_kind]:
            raise MutationContractError(
                f"{self.target_kind.value} does not support entity kind {self.entity_kind.value}"
            )
        if self.field_kind not in _TARGET_FIELD_KINDS[self.target_kind]:
            raise MutationContractError(
                f"{self.target_kind.value} does not support field kind {self.field_kind.value}"
            )
        if self.field_kind not in _ENTITY_FIELD_KINDS[(self.target_kind, self.entity_kind)]:
            raise MutationContractError(
                f"{self.target_kind.value}/{self.entity_kind.value} does not support "
                f"field kind {self.field_kind.value}"
            )
        if self.entity_kind is MutationEntityKind.GLOBAL:
            if self.selector is not None:
                raise MutationContractError("global mutation targets cannot declare a selector")
        else:
            selector = self.selector
            if isinstance(selector, str):
                selector = MutationSelectorSpec.exact(selector)
            if not isinstance(selector, MutationSelectorSpec):
                raise MutationContractError(
                    "mutation selector must be a MutationSelectorSpec, exact string, or None"
                )
            object.__setattr__(self, "selector", selector)

    @property
    def selector_spec(self) -> MutationSelectorSpec | None:
        """Return normalized selector metadata without exposing legacy input."""

        selector = self.selector
        if selector is None:
            return None
        if not isinstance(selector, MutationSelectorSpec):  # pragma: no cover - invariant
            raise MutationContractError("mutation selector normalization failed")
        return selector


@dataclass(frozen=True)
class MutationSpec:
    term_key: str
    target: MutationTargetSpec
    trigger: MutationTrigger
    commit_phase: MutationCommitPhase
    operation: MutationOperation
    baseline: MutationBaseline
    persistence: MutationPersistence
    recompute: MutationRecomputeLevel
    value_template: BufferContract

    def __post_init__(self) -> None:
        object.__setattr__(self, "term_key", _non_empty(self.term_key, "term_key"))
        if not isinstance(self.target, MutationTargetSpec):
            raise MutationContractError("target must be a MutationTargetSpec")
        _enum(self.trigger, MutationTrigger, "trigger")
        _enum(self.commit_phase, MutationCommitPhase, "commit_phase")
        _enum(self.operation, MutationOperation, "operation")
        _enum(self.baseline, MutationBaseline, "baseline")
        _enum(self.persistence, MutationPersistence, "persistence")
        _enum(self.recompute, MutationRecomputeLevel, "recompute")
        _validate_value_template(self.value_template)


@dataclass(frozen=True)
class MutationCapability:
    target_key: str
    target_kind: MutationTargetKind
    entity_kind: MutationEntityKind
    field_kind: MutationFieldKind
    entity_count: int | None
    value_template: BufferContract
    triggers: frozenset[MutationTrigger]
    commit_phases: frozenset[MutationCommitPhase]
    operations: frozenset[MutationOperation]
    baselines: frozenset[MutationBaseline]
    persistences: frozenset[MutationPersistence]
    recompute_levels: frozenset[MutationRecomputeLevel]

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_key", _non_empty(self.target_key, "target_key"))
        _enum(self.target_kind, MutationTargetKind, "target_kind")
        _enum(self.entity_kind, MutationEntityKind, "entity_kind")
        _enum(self.field_kind, MutationFieldKind, "field_kind")
        if self.entity_kind not in _TARGET_ENTITY_KINDS[self.target_kind]:
            raise MutationContractError("capability target/entity kinds are incompatible")
        if self.field_kind not in _TARGET_FIELD_KINDS[self.target_kind]:
            raise MutationContractError("capability target/field kinds are incompatible")
        if self.field_kind not in _ENTITY_FIELD_KINDS[(self.target_kind, self.entity_kind)]:
            raise MutationContractError("capability entity/field kinds are incompatible")
        if self.entity_kind is MutationEntityKind.GLOBAL:
            if self.entity_count is not None:
                raise MutationContractError("global capability entity_count must be None")
        else:
            if self.entity_count is None:
                raise MutationContractError("non-global capability requires entity_count")
            _count(self.entity_count, "entity_count", minimum=1)
        _validate_value_template(self.value_template)
        _enum_set(self.triggers, MutationTrigger, "triggers")
        _enum_set(self.commit_phases, MutationCommitPhase, "commit_phases")
        _enum_set(self.operations, MutationOperation, "operations")
        _enum_set(self.baselines, MutationBaseline, "baselines")
        _enum_set(self.persistences, MutationPersistence, "persistences")
        _enum_set(self.recompute_levels, MutationRecomputeLevel, "recompute_levels")


@dataclass(frozen=True)
class BoundMutationTarget:
    target_key: str
    target_kind: MutationTargetKind
    entity_kind: MutationEntityKind
    field_kind: MutationFieldKind
    entity_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_key", _non_empty(self.target_key, "target_key"))
        _enum(self.target_kind, MutationTargetKind, "target_kind")
        _enum(self.entity_kind, MutationEntityKind, "entity_kind")
        _enum(self.field_kind, MutationFieldKind, "field_kind")
        if not isinstance(self.entity_ids, tuple):
            raise MutationContractError("bound mutation entity_ids must be a tuple")
        if self.entity_kind is MutationEntityKind.GLOBAL:
            if self.entity_ids:
                raise MutationContractError(
                    "global bound mutation target cannot contain entity_ids"
                )
            return
        if not self.entity_ids:
            raise MutationContractError("bound mutation target must contain entity_ids")
        if any(
            isinstance(entity_id, bool) or not isinstance(entity_id, int) or entity_id < 0
            for entity_id in self.entity_ids
        ):
            raise MutationContractError("bound mutation entity_ids must be non-negative integers")
        if len(set(self.entity_ids)) != len(self.entity_ids):
            raise MutationContractError("bound mutation entity_ids must be unique")


@dataclass(frozen=True)
class BoundMutationSpec:
    term_key: str
    target: BoundMutationTarget
    trigger: MutationTrigger
    commit_phase: MutationCommitPhase
    operation: MutationOperation
    baseline: MutationBaseline
    persistence: MutationPersistence
    recompute: MutationRecomputeLevel
    value_buffer: BufferContract
    capability_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "term_key", _non_empty(self.term_key, "term_key"))
        if not isinstance(self.target, BoundMutationTarget):
            raise MutationContractError("target must be a BoundMutationTarget")
        _enum(self.trigger, MutationTrigger, "trigger")
        _enum(self.commit_phase, MutationCommitPhase, "commit_phase")
        _enum(self.operation, MutationOperation, "operation")
        _enum(self.baseline, MutationBaseline, "baseline")
        _enum(self.persistence, MutationPersistence, "persistence")
        _enum(self.recompute, MutationRecomputeLevel, "recompute")
        _validate_value_template(self.value_buffer)
        object.__setattr__(
            self,
            "capability_fingerprint",
            _non_empty(self.capability_fingerprint, "capability_fingerprint"),
        )


@dataclass(frozen=True)
class BoundMutationPlan:
    backend_type: str
    backend_instance_id: str
    num_envs: int
    specs: tuple[BoundMutationSpec, ...]
    fingerprint: str
    contract_version: str = MUTATION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend_type", _non_empty(self.backend_type, "backend_type"))
        object.__setattr__(
            self,
            "backend_instance_id",
            _non_empty(self.backend_instance_id, "backend_instance_id"),
        )
        _count(self.num_envs, "num_envs", minimum=1)
        if not isinstance(self.specs, tuple) or not self.specs:
            raise MutationContractError("bound mutation plan specs must be a non-empty tuple")
        if any(not isinstance(spec, BoundMutationSpec) for spec in self.specs):
            raise MutationContractError("bound mutation plan contains an invalid spec")
        term_keys = [spec.term_key for spec in self.specs]
        if len(set(term_keys)) != len(term_keys):
            raise MutationContractError("bound mutation term keys must be unique")
        if tuple(term_keys) != tuple(sorted(term_keys)):
            raise MutationContractError("bound mutation specs must use canonical term-key order")
        object.__setattr__(self, "fingerprint", _non_empty(self.fingerprint, "fingerprint"))
        if self.contract_version != MUTATION_CONTRACT_VERSION:
            raise MutationContractError(
                f"unsupported mutation contract version {self.contract_version!r}"
            )

    def spec_index(self, term_key: str) -> int:
        for index, spec in enumerate(self.specs):
            if spec.term_key == term_key:
                return index
        raise MutationContractError(f"mutation term {term_key!r} is not bound")

    def require_owner(self, *, backend_type: str, backend_instance_id: str) -> None:
        if self.backend_type != backend_type or self.backend_instance_id != backend_instance_id:
            raise MutationContractError(
                "bound mutation plan belongs to a different backend type or instance"
            )

    def require_compatible(self, other: BoundMutationPlan) -> None:
        if not isinstance(other, BoundMutationPlan) or self != other:
            raise MutationContractError(
                "mutation batch was built from a different plan or fingerprint"
            )


MutationSelectorResolver = Callable[[MutationTargetSpec], tuple[int, ...]]


def _capability_payload(capability: MutationCapability) -> dict[str, Any]:
    return {
        "target_key": capability.target_key,
        "target_kind": capability.target_kind.value,
        "entity_kind": capability.entity_kind.value,
        "field_kind": capability.field_kind.value,
        "entity_count": capability.entity_count,
        "value_template": _buffer_payload(capability.value_template),
        "triggers": sorted(item.value for item in capability.triggers),
        "commit_phases": sorted(item.value for item in capability.commit_phases),
        "operations": sorted(item.value for item in capability.operations),
        "baselines": sorted(item.value for item in capability.baselines),
        "persistences": sorted(item.value for item in capability.persistences),
        "recompute_levels": sorted(int(item) for item in capability.recompute_levels),
    }


def _buffer_payload(buffer: BufferContract) -> dict[str, Any]:
    return {
        "row_shape": buffer.row_shape,
        "dtype": buffer.dtype,
        "layout": buffer.layout.value,
        "placement": {
            "memory_space": buffer.placement.memory_space.value,
            "device_type": buffer.placement.device_type,
            "device_index": buffer.placement.device_index,
        },
        "owner": buffer.owner.value,
        "mutability": buffer.mutability.value,
        "lifetime": buffer.lifetime.value,
        "dlpack_exportable": buffer.dlpack_exportable,
        "address_stable": buffer.address_stable,
    }


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _require_supported(spec: MutationSpec, capability: MutationCapability) -> None:
    target = spec.target
    expected = (
        capability.target_kind,
        capability.entity_kind,
        capability.field_kind,
    )
    actual = (target.target_kind, target.entity_kind, target.field_kind)
    if actual != expected:
        raise MutationContractError(
            f"mutation target {target.target_key!r} does not match its registered capability"
        )
    if spec.value_template != capability.value_template:
        raise MutationContractError(
            f"mutation term {spec.term_key!r} value metadata does not match capability"
        )
    checks = (
        (spec.trigger, capability.triggers, "trigger"),
        (spec.commit_phase, capability.commit_phases, "commit phase"),
        (spec.operation, capability.operations, "operation"),
        (spec.baseline, capability.baselines, "baseline"),
        (spec.persistence, capability.persistences, "persistence"),
        (spec.recompute, capability.recompute_levels, "recompute level"),
    )
    for value, allowed, label in checks:
        if value not in allowed:
            raise MutationContractError(
                f"mutation term {spec.term_key!r} uses unsupported {label} {value.value!r}"
            )


def _resolve_entity_ids(
    target: MutationTargetSpec,
    capability: MutationCapability,
    resolver: MutationSelectorResolver,
    cache: dict[tuple[str, MutationEntityKind, MutationSelectorSpec], tuple[int, ...]],
) -> tuple[int, ...]:
    if target.entity_kind is MutationEntityKind.GLOBAL:
        return ()
    selector = target.selector_spec
    assert selector is not None
    cache_key = (target.target_key, target.entity_kind, selector)
    if cache_key not in cache:
        try:
            resolved = resolver(target)
        except MutationContractError:
            raise
        except Exception as exc:
            raise MutationContractError(
                f"failed to resolve mutation selector {selector.semantic_key!r}"
            ) from exc
        if not isinstance(resolved, tuple):
            raise MutationContractError("mutation selector resolver must return a tuple")
        cache[cache_key] = resolved
    entity_ids = cache[cache_key]
    if not entity_ids:
        raise MutationContractError(
            f"mutation selector {selector.semantic_key!r} resolved no entities"
        )
    if any(
        isinstance(entity_id, bool) or not isinstance(entity_id, int) or entity_id < 0
        for entity_id in entity_ids
    ):
        raise MutationContractError("resolved mutation entity IDs must be non-negative integers")
    if len(set(entity_ids)) != len(entity_ids):
        raise MutationContractError("resolved mutation entity IDs must be unique")
    if selector.entity_ids and len(entity_ids) != len(selector.entity_ids):
        raise MutationContractError(
            "backend mutation selector cardinality differs from the compiled selector binding"
        )
    assert capability.entity_count is not None
    if any(entity_id >= capability.entity_count for entity_id in entity_ids):
        raise MutationContractError(
            f"mutation selector {selector.semantic_key!r} resolved an out-of-range entity ID"
        )
    return entity_ids


def _bound_spec(
    spec: MutationSpec,
    capability: MutationCapability,
    entity_ids: tuple[int, ...],
) -> BoundMutationSpec:
    row_shape = (
        spec.value_template.row_shape
        if spec.target.entity_kind is MutationEntityKind.GLOBAL
        else (len(entity_ids), *spec.value_template.row_shape)
    )
    return BoundMutationSpec(
        term_key=spec.term_key,
        target=BoundMutationTarget(
            target_key=spec.target.target_key,
            target_kind=spec.target.target_kind,
            entity_kind=spec.target.entity_kind,
            field_kind=spec.target.field_kind,
            entity_ids=entity_ids,
        ),
        trigger=spec.trigger,
        commit_phase=spec.commit_phase,
        operation=spec.operation,
        baseline=spec.baseline,
        persistence=spec.persistence,
        recompute=spec.recompute,
        value_buffer=replace(spec.value_template, row_shape=row_shape),
        capability_fingerprint=_digest(_capability_payload(capability)),
    )


def _conflict_errors(specs: tuple[BoundMutationSpec, ...]) -> list[str]:
    errors: list[str] = []
    for index, left in enumerate(specs):
        left_ids = {-1} if not left.target.entity_ids else set(left.target.entity_ids)
        for right in specs[index + 1 :]:
            if (
                left.target.target_key != right.target.target_key
                or left.commit_phase is not right.commit_phase
            ):
                continue
            right_ids = {-1} if not right.target.entity_ids else set(right.target.entity_ids)
            overlap = sorted(left_ids & right_ids)
            if overlap:
                rendered = "global" if overlap == [-1] else ",".join(map(str, overlap))
                errors.append(
                    f"mutation terms {left.term_key!r} and {right.term_key!r} conflict on "
                    f"target {left.target.target_key!r}, entities {rendered}, "
                    f"phase {left.commit_phase.value!r}"
                )
    return errors


def _bound_spec_payload(spec: BoundMutationSpec) -> dict[str, Any]:
    return {
        "term_key": spec.term_key,
        "target": {
            "target_key": spec.target.target_key,
            "target_kind": spec.target.target_kind.value,
            "entity_kind": spec.target.entity_kind.value,
            "field_kind": spec.target.field_kind.value,
            "entity_ids": spec.target.entity_ids,
        },
        "trigger": spec.trigger.value,
        "commit_phase": spec.commit_phase.value,
        "operation": spec.operation.value,
        "baseline": spec.baseline.value,
        "persistence": spec.persistence.value,
        "recompute": int(spec.recompute),
        "value_buffer": _buffer_payload(spec.value_buffer),
        "capability_fingerprint": spec.capability_fingerprint,
    }


def bind_mutation_plan(
    *,
    backend_type: str,
    backend_instance_id: str,
    num_envs: int,
    specs: tuple[MutationSpec, ...],
    capabilities: tuple[MutationCapability, ...],
    resolve_selector: MutationSelectorResolver,
) -> BoundMutationPlan:
    """Bind registry-owned mutation specs to one backend on the cold path."""

    backend_type = _non_empty(backend_type, "backend_type")
    backend_instance_id = _non_empty(backend_instance_id, "backend_instance_id")
    _count(num_envs, "num_envs", minimum=1)
    if not isinstance(specs, tuple) or not specs:
        raise MutationContractError("mutation specs must be a non-empty tuple")
    if any(not isinstance(spec, MutationSpec) for spec in specs):
        raise MutationContractError("mutation specs contain an invalid value")
    term_keys = [spec.term_key for spec in specs]
    if len(set(term_keys)) != len(term_keys):
        raise MutationContractError("mutation term keys must be unique")
    if not isinstance(capabilities, tuple) or not capabilities:
        raise MutationContractError("mutation capabilities must be a non-empty tuple")
    if any(not isinstance(capability, MutationCapability) for capability in capabilities):
        raise MutationContractError("mutation capabilities contain an invalid value")
    capability_map = {capability.target_key: capability for capability in capabilities}
    if len(capability_map) != len(capabilities):
        raise MutationContractError("mutation capability target keys must be unique")
    if not callable(resolve_selector):
        raise MutationContractError("resolve_selector must be callable")

    # A selector may map into a different coordinate domain for each semantic
    # target (for example MuJoCo qpos versus qvel coordinates).  Cache aliases
    # of one target, but never reuse an entity ID across distinct targets.
    selector_cache: dict[tuple[str, MutationEntityKind, MutationSelectorSpec], tuple[int, ...]] = {}
    bound_specs: list[BoundMutationSpec] = []
    for spec in sorted(specs, key=lambda item: item.term_key):
        try:
            capability = capability_map[spec.target.target_key]
        except KeyError as exc:
            raise MutationContractError(
                f"mutation target {spec.target.target_key!r} is not supported by the backend"
            ) from exc
        _require_supported(spec, capability)
        entity_ids = _resolve_entity_ids(
            spec.target,
            capability,
            resolve_selector,
            selector_cache,
        )
        bound_specs.append(_bound_spec(spec, capability, entity_ids))
    canonical_specs = tuple(bound_specs)
    conflicts = _conflict_errors(canonical_specs)
    if conflicts:
        raise MutationContractError("; ".join(conflicts))
    payload = {
        "contract_version": MUTATION_CONTRACT_VERSION,
        "backend_type": backend_type,
        "num_envs": num_envs,
        "specs": [_bound_spec_payload(spec) for spec in canonical_specs],
    }
    return BoundMutationPlan(
        backend_type=backend_type,
        backend_instance_id=backend_instance_id,
        num_envs=num_envs,
        specs=canonical_specs,
        fingerprint=f"{MUTATION_CONTRACT_VERSION}:{_digest(payload)}",
    )


__all__ = [
    "MUTATION_CONTRACT_VERSION",
    "BoundMutationPlan",
    "BoundMutationSpec",
    "BoundMutationTarget",
    "MutationBaseline",
    "MutationCapability",
    "MutationCommitPhase",
    "MutationContractError",
    "MutationEntityKind",
    "MutationFieldKind",
    "MutationOperation",
    "MutationPersistence",
    "MutationRecomputeLevel",
    "MutationSelectorMode",
    "MutationSelectorResolver",
    "MutationSelectorSpec",
    "MutationSpec",
    "MutationTargetKind",
    "MutationTargetSpec",
    "MutationTrigger",
    "bind_mutation_plan",
]
