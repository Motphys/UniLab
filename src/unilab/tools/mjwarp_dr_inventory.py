"""Strict field-level mjwarp DR capability inventory contracts for managed MuJoCo/MJWarp rollout."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from omegaconf import OmegaConf

from unilab.tools.claim_gap_audit import (
    ClaimGapInventory,
    EvidenceKind,
    EvidenceRole,
)

SCHEMA_VERSION = 1
ISSUE = 705
BACKEND = "mjwarp"
MJLAB_VERSION = "1.5.3"
MJLAB_COMMIT = "f643d245303ff439a90f37151056ff987bdb95f7"
EXPECTED_DEPENDENCIES = frozenset({"mujoco-warp>=3.10.0.3,~=3.10.0", "warp-lang>=1.14.0"})
EXPECTED_INSPECTED_FILES = frozenset(
    {
        "src/mjlab/managers/event_manager.py",
        "src/mjlab/sim/randomization.py",
        "src/mjlab/sim/sim.py",
        "src/mjlab/envs/mdp/dr/actuator.py",
        "src/mjlab/envs/mdp/dr/body.py",
        "src/mjlab/envs/mdp/dr/geom.py",
        "src/mjlab/envs/mdp/dr/joint.py",
        "src/mjlab/envs/mdp/dr/tendon.py",
        "src/mjlab/envs/mdp/events.py",
        "tests/test_domain_randomization.py",
        "tests/test_events.py",
    }
)

DERIVED_FIELD_SETS: dict[str, tuple[str, ...]] = {
    "none": (),
    "set_const_fixed": ("body_subtreemass",),
    "set_const_0": (
        "dof_invweight0",
        "body_invweight0",
        "tendon_length0",
        "tendon_invweight0",
        "actuator_acc0",
    ),
    "set_const": (
        "body_subtreemass",
        "dof_invweight0",
        "body_invweight0",
        "tendon_length0",
        "tendon_invweight0",
        "actuator_acc0",
    ),
}

EXPECTED_CAPABILITY_IDS = frozenset(
    {
        "state.qpos_qvel_reset",
        "state.velocity_impulse",
        "actuator.pd_gains",
        "geom.friction",
        "joint.damping_friction",
        "actuator.force_limits",
        "joint.armature",
        "joint.qpos0",
        "body.pose",
        "tendon.armature",
        "body.coupled_inertia",
        "model.body_gravcomp",
        "geom.primitive_size",
        "structural.scene_variant",
        "wrench.body_external",
        "global.gravity",
    }
)

EXPECTED_TIERS = {
    "state.qpos_qvel_reset": "A",
    "state.velocity_impulse": "A",
    "actuator.pd_gains": "B",
    "geom.friction": "B",
    "joint.damping_friction": "B",
    "actuator.force_limits": "B",
    "joint.armature": "C",
    "joint.qpos0": "C",
    "body.pose": "C",
    "tendon.armature": "C",
    "body.coupled_inertia": "D",
    "model.body_gravcomp": "C",
    "geom.primitive_size": "E",
    "structural.scene_variant": "E",
    "wrench.body_external": "F",
    "global.gravity": "D",
}

EXPECTED_SUPPORT_STATES = {
    "state.qpos_qvel_reset": "phase2_required",
    "state.velocity_impulse": "phase6_candidate",
    "actuator.pd_gains": "phase2_decision",
    "geom.friction": "phase6_candidate",
    "joint.damping_friction": "phase6_candidate",
    "actuator.force_limits": "phase6_candidate",
    "joint.armature": "phase6_candidate",
    "joint.qpos0": "phase6_candidate",
    "body.pose": "phase6_candidate",
    "tendon.armature": "phase6_candidate",
    "body.coupled_inertia": "phase6_candidate",
    "model.body_gravcomp": "phase6_candidate",
    "geom.primitive_size": "phase6_candidate",
    "structural.scene_variant": "cold_path_only",
    "wrench.body_external": "phase6_candidate",
    "global.gravity": "blocked_pending_evidence",
}

EXPECTED_LEGACY_TERMS = frozenset(
    {
        "base_com_offset",
        "base_mass_delta",
        "gravity",
        "body_iquat",
        "body_inertia",
        "body_ipos",
        "body_mass",
        "dof_armature",
        "geom_friction",
        "kp",
        "kd",
        "interval_push",
        "interval_body_velocity_delta",
        "interval_body_force",
    }
)

MODEL_EXPANSION_INVALIDATIONS = frozenset(
    {
        "model_bridge_cache",
        "sensor_context",
        "step_graph",
        "forward_graph",
        "reset_graph",
        "sense_graph",
    }
)

EXPECTED_EXCLUSION_SCOPES = frozenset(
    {
        "camera light material color and other visual-only model fields",
        "mesh hfield SDF deformable and topology-changing runtime mutation",
        "benchmark mjwarp capability flags and silently discarded reset payload",
        "render playback Jacobian and host substep controller",
    }
)

_CAPABILITY_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_FIELD_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_TEST_ID_RE = re.compile(
    r"^(?:tests|benchmark)/[^:\s]+\.py"
    r"(?:::[A-Za-z_][A-Za-z0-9_]*(?:\[[^\]\r\n]+\])?)+$"
)
_FORMULA_RE = re.compile(r"^[A-Za-z0-9_+*(). /-]+$")
_MISSING = object()


class Tier(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


class TargetKind(str, Enum):
    SIMULATION_STATE = "simulation_state"
    MODEL_PARAMETER = "model_parameter"
    EXTERNAL_WRENCH = "external_wrench"
    STRUCTURAL_VARIANT = "structural_variant"


class SupportState(str, Enum):
    PHASE2_REQUIRED = "phase2_required"
    PHASE2_DECISION = "phase2_decision"
    PHASE6_CANDIDATE = "phase6_candidate"
    COLD_PATH_ONLY = "cold_path_only"
    BLOCKED_PENDING_EVIDENCE = "blocked_pending_evidence"


class FieldResolution(str, Enum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


class PerWorldStorage(str, Enum):
    NATIVE = "native"
    EXPAND_REQUIRED = "expand_required"
    COLD_VARIANTS = "cold_variants"
    UNRESOLVED = "unresolved"


class RowScope(str, Enum):
    SELECTED = "selected"
    COLD_PATH = "cold_path"
    UNRESOLVED = "unresolved"


class ExpansionKind(str, Enum):
    DATA_NATIVE = "data_native"
    MODEL_FIELD_EXPAND = "model_field_expand"
    COLD_MATERIALIZATION = "cold_materialization"
    UNRESOLVED = "unresolved"


class BackendRecompute(str, Enum):
    NONE = "none"
    FORWARD = "forward"
    SET_CONST_FIXED = "set_const_fixed"
    SET_CONST_0 = "set_const_0"
    SET_CONST = "set_const"
    SPECIALIZED_BOUNDS = "specialized_bounds"
    RECOMPILE = "recompile"
    UNRESOLVED = "unresolved"


class RecomputeScope(str, Enum):
    SELECTED_ROWS = "selected_rows"
    ALL_WORLDS = "all_worlds"
    COLD_PATH = "cold_path"
    UNRESOLVED = "unresolved"


class GraphImpact(str, Enum):
    STABLE_DATA_ADDRESS = "stable_data_address"
    RECAPTURE_REQUIRED = "recapture_required"
    REBUILD_REQUIRED = "rebuild_required"
    UNKNOWN_FAIL_CLOSED = "unknown_fail_closed"


class ExclusionHandling(str, Enum):
    DEFERRED = "deferred"
    COLD_PATH = "cold_path"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class SourceBaseline:
    project: str
    version: str
    commit: str
    dependencies: tuple[str, ...]
    inspected_files: tuple[str, ...]


@dataclass(frozen=True)
class StorageSpec:
    per_world: PerWorldStorage
    row_scope: RowScope
    expansion: ExpansionKind
    dtype_bytes: int | None
    direct_elements_per_world: str
    derived_elements_per_world: str
    bytes_formula: str
    measurement_test_id: str


@dataclass(frozen=True)
class MutationSemantics:
    operations: tuple[str, ...]
    baseline_sources: tuple[str, ...]
    triggers: tuple[str, ...]
    commit_phases: tuple[str, ...]
    persistence: tuple[str, ...]


@dataclass(frozen=True)
class RecomputeSpec:
    semantic: str
    backend: BackendRecompute
    scope: RecomputeScope


@dataclass(frozen=True)
class GraphSpec:
    impact: GraphImpact
    invalidations: tuple[str, ...]


@dataclass(frozen=True)
class RequiredTests:
    contract: str
    row_isolation: str
    physics_effect: str
    graph: str
    memory: str


@dataclass(frozen=True)
class DrCapability:
    capability_id: str
    tier: Tier
    semantic_target: str
    target_kind: TargetKind
    support_state: SupportState
    owner: str
    selector_kind: str
    legacy_terms: tuple[str, ...]
    field_resolution: FieldResolution
    direct_fields: tuple[str, ...]
    derived_fields: tuple[str, ...]
    storage: StorageSpec
    mutation: MutationSemantics
    recompute: RecomputeSpec
    graph: GraphSpec
    validity_constraints: tuple[str, ...]
    required_tests: RequiredTests
    evidence_refs: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class ScopeExclusion:
    scope: str
    handling: ExclusionHandling
    reason: str
    test_id: str


@dataclass(frozen=True)
class MjwarpDrInventory:
    schema_version: int
    issue: int
    backend: str
    source: SourceBaseline
    derived_field_sets: tuple[tuple[str, tuple[str, ...]], ...]
    required_capability_ids: tuple[str, ...]
    capabilities: tuple[DrCapability, ...]
    exclusions: tuple[ScopeExclusion, ...]
    source_path: Path


class DrInventoryValidationError(ValueError):
    def __init__(self, source: Path, errors: Iterable[str]) -> None:
        self.source = source
        self.errors = tuple(errors)
        detail = "\n".join(f"- {error}" for error in self.errors)
        super().__init__(f"invalid mjwarp DR inventory {source}:\n{detail}")


class _Parser:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def mapping(self, value: Any, path: str, keys: tuple[str, ...]) -> dict[str, Any]:
        if not isinstance(value, dict):
            self.errors.append(f"{path}: expected mapping")
            return {}
        expected = set(keys)
        actual = set(value)
        for key in sorted(expected - actual):
            self.errors.append(f"{path}: missing key `{key}`")
        for key in sorted(actual - expected, key=str):
            self.errors.append(f"{path}: unknown key `{key}`")
        return value

    def string(self, value: Any, path: str, *, allow_empty: bool = False) -> str:
        if value is _MISSING:
            return ""
        if not isinstance(value, str):
            self.errors.append(f"{path}: expected string")
            return ""
        if not allow_empty and not value.strip():
            self.errors.append(f"{path}: must not be empty")
        if "${" in value:
            self.errors.append(f"{path}: interpolation is not allowed")
        return value

    def nullable_integer(self, value: Any, path: str) -> int | None:
        if value is _MISSING or value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            self.errors.append(f"{path}: expected positive integer or null")
            return None
        if value <= 0:
            self.errors.append(f"{path}: must be > 0")
        return int(value)

    def integer(self, value: Any, path: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            self.errors.append(f"{path}: expected integer")
            return 0
        return int(value)

    def string_list(self, value: Any, path: str, *, allow_empty: bool = False) -> tuple[str, ...]:
        if not isinstance(value, list):
            self.errors.append(f"{path}: expected list of strings")
            return ()
        values = tuple(self.string(item, f"{path}[{index}]") for index, item in enumerate(value))
        if not allow_empty and not values:
            self.errors.append(f"{path}: must not be empty")
        duplicates = sorted(item for item, count in Counter(values).items() if count > 1)
        if duplicates:
            self.errors.append(f"{path}: duplicate values {duplicates!r}")
        return values

    def enum(self, value: Any, path: str, enum_type: type[Enum], default: Enum) -> Any:
        if value is _MISSING:
            return default
        try:
            return enum_type(value)
        except (TypeError, ValueError):
            allowed = [item.value for item in enum_type]
            self.errors.append(f"{path}: expected one of {allowed!r}")
            return default


_ROOT_KEYS = (
    "schema_version",
    "issue",
    "backend",
    "source",
    "derived_field_sets",
    "required_capability_ids",
    "capabilities",
    "exclusions",
)
_SOURCE_KEYS = ("project", "version", "commit", "dependencies", "inspected_files")
_CAPABILITY_KEYS = (
    "capability_id",
    "tier",
    "semantic_target",
    "target_kind",
    "support_state",
    "owner",
    "selector_kind",
    "legacy_terms",
    "field_resolution",
    "direct_fields",
    "derived_fields",
    "storage",
    "mutation",
    "recompute",
    "graph",
    "validity_constraints",
    "required_tests",
    "evidence_refs",
    "notes",
)
_STORAGE_KEYS = (
    "per_world",
    "row_scope",
    "expansion",
    "dtype_bytes",
    "direct_elements_per_world",
    "derived_elements_per_world",
    "bytes_formula",
    "measurement_test_id",
)
_MUTATION_KEYS = (
    "operations",
    "baseline_sources",
    "triggers",
    "commit_phases",
    "persistence",
)
_RECOMPUTE_KEYS = ("semantic", "backend", "scope")
_GRAPH_KEYS = ("impact", "invalidations")
_TEST_KEYS = ("contract", "row_isolation", "physics_effect", "graph", "memory")
_EXCLUSION_KEYS = ("scope", "handling", "reason", "test_id")


def _parse_test_id(parser: _Parser, value: Any, path: str) -> str:
    test_id = parser.string(value, path)
    if test_id and _TEST_ID_RE.fullmatch(test_id) is None:
        parser.errors.append(f"{path}: expected an explicit pytest node under tests/ or benchmark/")
    elif test_id and not test_id.rsplit("::", maxsplit=1)[-1].split("[", maxsplit=1)[0].startswith(
        "test_"
    ):
        parser.errors.append(f"{path}: explicit pytest ID must end in a test node")
    candidate = Path(test_id.split("::", maxsplit=1)[0])
    if candidate.is_absolute() or ".." in candidate.parts:
        parser.errors.append(f"{path}: test path must stay within the repository")
    return test_id


def _parse_source(parser: _Parser, raw: Any) -> SourceBaseline:
    values = parser.mapping(raw, "source", _SOURCE_KEYS)
    commit = parser.string(values.get("commit", _MISSING), "source.commit")
    if commit and _COMMIT_RE.fullmatch(commit) is None:
        parser.errors.append("source.commit: expected full 40-character SHA")
    return SourceBaseline(
        project=parser.string(values.get("project", _MISSING), "source.project"),
        version=parser.string(values.get("version", _MISSING), "source.version"),
        commit=commit,
        dependencies=parser.string_list(
            values.get("dependencies", _MISSING), "source.dependencies"
        ),
        inspected_files=parser.string_list(
            values.get("inspected_files", _MISSING), "source.inspected_files"
        ),
    )


def _parse_storage(parser: _Parser, raw: Any, path: str) -> StorageSpec:
    values = parser.mapping(raw, path, _STORAGE_KEYS)
    formulas: dict[str, str] = {}
    for key in ("direct_elements_per_world", "derived_elements_per_world", "bytes_formula"):
        formula = parser.string(values.get(key, _MISSING), f"{path}.{key}")
        if formula and _FORMULA_RE.fullmatch(formula) is None:
            parser.errors.append(f"{path}.{key}: invalid symbolic formula")
        formulas[key] = formula
    return StorageSpec(
        per_world=parser.enum(
            values.get("per_world", _MISSING),
            f"{path}.per_world",
            PerWorldStorage,
            PerWorldStorage.UNRESOLVED,
        ),
        row_scope=parser.enum(
            values.get("row_scope", _MISSING),
            f"{path}.row_scope",
            RowScope,
            RowScope.UNRESOLVED,
        ),
        expansion=parser.enum(
            values.get("expansion", _MISSING),
            f"{path}.expansion",
            ExpansionKind,
            ExpansionKind.UNRESOLVED,
        ),
        dtype_bytes=parser.nullable_integer(
            values.get("dtype_bytes", _MISSING), f"{path}.dtype_bytes"
        ),
        direct_elements_per_world=formulas["direct_elements_per_world"],
        derived_elements_per_world=formulas["derived_elements_per_world"],
        bytes_formula=formulas["bytes_formula"],
        measurement_test_id=_parse_test_id(
            parser, values.get("measurement_test_id", _MISSING), f"{path}.measurement_test_id"
        ),
    )


def _parse_mutation(parser: _Parser, raw: Any, path: str) -> MutationSemantics:
    values = parser.mapping(raw, path, _MUTATION_KEYS)
    return MutationSemantics(
        operations=parser.string_list(values.get("operations", _MISSING), f"{path}.operations"),
        baseline_sources=parser.string_list(
            values.get("baseline_sources", _MISSING), f"{path}.baseline_sources"
        ),
        triggers=parser.string_list(values.get("triggers", _MISSING), f"{path}.triggers"),
        commit_phases=parser.string_list(
            values.get("commit_phases", _MISSING), f"{path}.commit_phases"
        ),
        persistence=parser.string_list(values.get("persistence", _MISSING), f"{path}.persistence"),
    )


def _parse_recompute(parser: _Parser, raw: Any, path: str) -> RecomputeSpec:
    values = parser.mapping(raw, path, _RECOMPUTE_KEYS)
    return RecomputeSpec(
        semantic=parser.string(values.get("semantic", _MISSING), f"{path}.semantic"),
        backend=parser.enum(
            values.get("backend", _MISSING),
            f"{path}.backend",
            BackendRecompute,
            BackendRecompute.UNRESOLVED,
        ),
        scope=parser.enum(
            values.get("scope", _MISSING),
            f"{path}.scope",
            RecomputeScope,
            RecomputeScope.UNRESOLVED,
        ),
    )


def _parse_graph(parser: _Parser, raw: Any, path: str) -> GraphSpec:
    values = parser.mapping(raw, path, _GRAPH_KEYS)
    return GraphSpec(
        impact=parser.enum(
            values.get("impact", _MISSING),
            f"{path}.impact",
            GraphImpact,
            GraphImpact.UNKNOWN_FAIL_CLOSED,
        ),
        invalidations=parser.string_list(
            values.get("invalidations", _MISSING), f"{path}.invalidations", allow_empty=True
        ),
    )


def _parse_tests(parser: _Parser, raw: Any, path: str) -> RequiredTests:
    values = parser.mapping(raw, path, _TEST_KEYS)
    return RequiredTests(
        contract=_parse_test_id(parser, values.get("contract", _MISSING), f"{path}.contract"),
        row_isolation=_parse_test_id(
            parser, values.get("row_isolation", _MISSING), f"{path}.row_isolation"
        ),
        physics_effect=_parse_test_id(
            parser, values.get("physics_effect", _MISSING), f"{path}.physics_effect"
        ),
        graph=_parse_test_id(parser, values.get("graph", _MISSING), f"{path}.graph"),
        memory=_parse_test_id(parser, values.get("memory", _MISSING), f"{path}.memory"),
    )


def _parse_capability(parser: _Parser, raw: Any, index: int) -> DrCapability:
    path = f"capabilities[{index}]"
    values = parser.mapping(raw, path, _CAPABILITY_KEYS)
    capability_id = parser.string(values.get("capability_id", _MISSING), f"{path}.capability_id")
    if capability_id and _CAPABILITY_ID_RE.fullmatch(capability_id) is None:
        parser.errors.append(f"{path}.capability_id: expected stable lowercase dotted ID")
    direct_fields = parser.string_list(
        values.get("direct_fields", _MISSING), f"{path}.direct_fields", allow_empty=True
    )
    derived_fields = parser.string_list(
        values.get("derived_fields", _MISSING), f"{path}.derived_fields", allow_empty=True
    )
    for field_path, fields in (
        ("direct_fields", direct_fields),
        ("derived_fields", derived_fields),
    ):
        for field in fields:
            if _FIELD_RE.fullmatch(field) is None:
                parser.errors.append(f"{path}.{field_path}: invalid field name {field!r}")
    return DrCapability(
        capability_id=capability_id,
        tier=parser.enum(values.get("tier", _MISSING), f"{path}.tier", Tier, Tier.A),
        semantic_target=parser.string(
            values.get("semantic_target", _MISSING), f"{path}.semantic_target"
        ),
        target_kind=parser.enum(
            values.get("target_kind", _MISSING),
            f"{path}.target_kind",
            TargetKind,
            TargetKind.MODEL_PARAMETER,
        ),
        support_state=parser.enum(
            values.get("support_state", _MISSING),
            f"{path}.support_state",
            SupportState,
            SupportState.BLOCKED_PENDING_EVIDENCE,
        ),
        owner=parser.string(values.get("owner", _MISSING), f"{path}.owner"),
        selector_kind=parser.string(values.get("selector_kind", _MISSING), f"{path}.selector_kind"),
        legacy_terms=parser.string_list(
            values.get("legacy_terms", _MISSING), f"{path}.legacy_terms", allow_empty=True
        ),
        field_resolution=parser.enum(
            values.get("field_resolution", _MISSING),
            f"{path}.field_resolution",
            FieldResolution,
            FieldResolution.UNRESOLVED,
        ),
        direct_fields=direct_fields,
        derived_fields=derived_fields,
        storage=_parse_storage(parser, values.get("storage", _MISSING), f"{path}.storage"),
        mutation=_parse_mutation(parser, values.get("mutation", _MISSING), f"{path}.mutation"),
        recompute=_parse_recompute(parser, values.get("recompute", _MISSING), f"{path}.recompute"),
        graph=_parse_graph(parser, values.get("graph", _MISSING), f"{path}.graph"),
        validity_constraints=parser.string_list(
            values.get("validity_constraints", _MISSING), f"{path}.validity_constraints"
        ),
        required_tests=_parse_tests(
            parser, values.get("required_tests", _MISSING), f"{path}.required_tests"
        ),
        evidence_refs=parser.string_list(
            values.get("evidence_refs", _MISSING), f"{path}.evidence_refs"
        ),
        notes=parser.string(values.get("notes", _MISSING), f"{path}.notes"),
    )


def _semantic_errors(inventory: MjwarpDrInventory) -> list[str]:
    errors: list[str] = []
    if inventory.issue != ISSUE:
        errors.append(f"issue: expected {ISSUE}, got {inventory.issue}")
    if inventory.backend != BACKEND:
        errors.append(f"backend: expected {BACKEND!r}, got {inventory.backend!r}")
    if inventory.source.project != "mjlab":
        errors.append(f"source.project: expected 'mjlab', got {inventory.source.project!r}")
    if inventory.source.version != MJLAB_VERSION:
        errors.append(
            f"source.version: expected {MJLAB_VERSION!r}, got {inventory.source.version!r}"
        )
    if inventory.source.commit != MJLAB_COMMIT:
        errors.append(f"source.commit: expected {MJLAB_COMMIT!r}, got {inventory.source.commit!r}")

    derived_sets = dict(inventory.derived_field_sets)
    if derived_sets != DERIVED_FIELD_SETS:
        errors.append("derived_field_sets: does not match the frozen mjlab recompute contract")

    declared_ids = set(inventory.required_capability_ids)
    actual_ids = {capability.capability_id for capability in inventory.capabilities}
    if declared_ids != EXPECTED_CAPABILITY_IDS:
        errors.append(
            "required_capability_ids: does not match the frozen managed MuJoCo/MJWarp rollout Tier A-F scope"
        )
    if set(inventory.source.dependencies) != EXPECTED_DEPENDENCIES:
        errors.append("source.dependencies: does not match the frozen mjlab dependency baseline")
    if set(inventory.source.inspected_files) != EXPECTED_INSPECTED_FILES:
        errors.append("source.inspected_files: does not match the frozen review surface")
    for inspected_file in inventory.source.inspected_files:
        candidate = Path(inspected_file)
        if candidate.is_absolute() or ".." in candidate.parts:
            errors.append(
                f"source.inspected_files: invalid repository-relative path {inspected_file!r}"
            )
    if actual_ids != declared_ids:
        errors.append(
            "capabilities: IDs must exactly match required_capability_ids; "
            f"missing={sorted(declared_ids - actual_ids)!r}, "
            f"extra={sorted(actual_ids - declared_ids)!r}"
        )

    actual_tiers = {
        capability.capability_id: capability.tier.value for capability in inventory.capabilities
    }
    if actual_tiers != EXPECTED_TIERS:
        errors.append(
            "capabilities: Tier assignments do not match the frozen managed MuJoCo/MJWarp rollout matrix"
        )
    actual_states = {
        capability.capability_id: capability.support_state.value
        for capability in inventory.capabilities
    }
    if actual_states != EXPECTED_SUPPORT_STATES:
        errors.append("capabilities: support states do not match the frozen rollout decisions")

    duplicate_ids = sorted(
        item
        for item, count in Counter(cap.capability_id for cap in inventory.capabilities).items()
        if count > 1
    )
    if duplicate_ids:
        errors.append(f"capabilities: duplicate IDs {duplicate_ids!r}")

    legacy_terms = [
        term for capability in inventory.capabilities for term in capability.legacy_terms
    ]
    duplicate_terms = sorted(term for term, count in Counter(legacy_terms).items() if count > 1)
    if duplicate_terms:
        errors.append(f"legacy_terms: duplicate ownership {duplicate_terms!r}")
    if set(legacy_terms) != EXPECTED_LEGACY_TERMS:
        errors.append(
            "legacy_terms: current UniLab reset/interval terms must be covered exactly; "
            f"missing={sorted(EXPECTED_LEGACY_TERMS - set(legacy_terms))!r}, "
            f"extra={sorted(set(legacy_terms) - EXPECTED_LEGACY_TERMS)!r}"
        )

    exclusion_scopes = [exclusion.scope for exclusion in inventory.exclusions]
    duplicate_exclusions = sorted(
        scope for scope, count in Counter(exclusion_scopes).items() if count > 1
    )
    if duplicate_exclusions:
        errors.append(f"exclusions: duplicate scopes {duplicate_exclusions!r}")
    if set(exclusion_scopes) != EXPECTED_EXCLUSION_SCOPES:
        errors.append("exclusions: does not match the frozen deferred and unsupported scope")

    for capability in inventory.capabilities:
        prefix = capability.capability_id
        direct = set(capability.direct_fields)
        derived = set(capability.derived_fields)
        if direct & derived:
            errors.append(
                f"{prefix}: direct and derived fields overlap {sorted(direct & derived)!r}"
            )
        if capability.field_resolution == FieldResolution.RESOLVED and not direct:
            errors.append(f"{prefix}: resolved capability requires direct fields")
        if capability.field_resolution == FieldResolution.UNRESOLVED and direct:
            errors.append(f"{prefix}: unresolved capability cannot assert direct fields")
        if capability.owner != "backend/DR":
            errors.append(f"{prefix}: owner must remain 'backend/DR' during Phase 0")

        backend_level = capability.recompute.backend.value
        if backend_level in DERIVED_FIELD_SETS:
            expected_derived = set(DERIVED_FIELD_SETS[backend_level])
            if derived != expected_derived:
                errors.append(
                    f"{prefix}: derived fields do not match {backend_level}; "
                    f"expected={sorted(expected_derived)!r}, got={sorted(derived)!r}"
                )

        if capability.storage.expansion == ExpansionKind.MODEL_FIELD_EXPAND:
            if capability.storage.per_world != PerWorldStorage.EXPAND_REQUIRED:
                errors.append(f"{prefix}: model field expansion requires per-world expansion")
            if capability.graph.impact != GraphImpact.RECAPTURE_REQUIRED:
                errors.append(f"{prefix}: model field expansion requires graph recapture")
            actual_invalidations = set(capability.graph.invalidations)
            if actual_invalidations != MODEL_EXPANSION_INVALIDATIONS:
                errors.append(
                    f"{prefix}: model field expansion invalidations must be exact; "
                    f"missing={sorted(MODEL_EXPANSION_INVALIDATIONS - actual_invalidations)!r}, "
                    f"extra={sorted(actual_invalidations - MODEL_EXPANSION_INVALIDATIONS)!r}"
                )
            if capability.storage.row_scope != RowScope.SELECTED:
                errors.append(f"{prefix}: runtime model mutation requires selected-row writes")
            if "nworld" not in capability.storage.bytes_formula:
                errors.append(f"{prefix}: model expansion byte formula must scale with nworld")
        elif capability.storage.expansion == ExpansionKind.DATA_NATIVE:
            if capability.storage.per_world != PerWorldStorage.NATIVE:
                errors.append(f"{prefix}: native Data capability requires native per-world storage")
            if capability.graph.impact != GraphImpact.STABLE_DATA_ADDRESS:
                errors.append(f"{prefix}: native Data capability must preserve graph addresses")
            if capability.graph.invalidations:
                errors.append(f"{prefix}: native Data writes cannot declare graph invalidations")
            if capability.storage.row_scope != RowScope.SELECTED:
                errors.append(f"{prefix}: native Data mutation requires selected-row writes")
            if "nworld" not in capability.storage.bytes_formula:
                errors.append(f"{prefix}: native Data byte formula must scale with nworld")
        elif capability.storage.expansion == ExpansionKind.COLD_MATERIALIZATION:
            if capability.support_state != SupportState.COLD_PATH_ONLY:
                errors.append(f"{prefix}: cold materialization must be cold_path_only")
            if capability.storage.row_scope != RowScope.COLD_PATH:
                errors.append(f"{prefix}: cold materialization requires cold_path row scope")
            if capability.graph.impact != GraphImpact.REBUILD_REQUIRED:
                errors.append(f"{prefix}: cold materialization requires graph rebuild")
            if set(capability.graph.invalidations) != MODEL_EXPANSION_INVALIDATIONS:
                errors.append(f"{prefix}: cold materialization invalidations must be exact")
        elif capability.storage.expansion == ExpansionKind.UNRESOLVED:
            if capability.support_state != SupportState.BLOCKED_PENDING_EVIDENCE:
                errors.append(f"{prefix}: unresolved expansion must remain blocked")
            if capability.field_resolution != FieldResolution.UNRESOLVED:
                errors.append(f"{prefix}: unresolved expansion cannot claim resolved fields")
            if capability.graph.impact != GraphImpact.UNKNOWN_FAIL_CLOSED:
                errors.append(f"{prefix}: unresolved expansion must fail closed on graph impact")
            if capability.storage.per_world != PerWorldStorage.UNRESOLVED:
                errors.append(f"{prefix}: unresolved expansion requires unresolved storage")
            if capability.storage.row_scope != RowScope.UNRESOLVED:
                errors.append(f"{prefix}: unresolved expansion requires unresolved row scope")
            if capability.graph.invalidations:
                errors.append(f"{prefix}: unresolved graph impact cannot assert invalidations")

        if derived and capability.storage.derived_elements_per_world == "0":
            errors.append(f"{prefix}: derived storage formula cannot be zero")
        if not derived and capability.storage.derived_elements_per_world != "0":
            errors.append(f"{prefix}: empty derived fields require a zero storage formula")
        if capability.field_resolution == FieldResolution.UNRESOLVED:
            if capability.storage.dtype_bytes is not None:
                errors.append(f"{prefix}: unresolved fields cannot claim dtype bytes")
            if capability.storage.direct_elements_per_world != "unknown":
                errors.append(f"{prefix}: unresolved direct storage formula must be unknown")
            if capability.storage.bytes_formula != "unknown":
                errors.append(f"{prefix}: unresolved byte formula must be unknown")
        elif capability.storage.dtype_bytes is None:
            errors.append(f"{prefix}: resolved fields require dtype bytes")
        elif capability.storage.bytes_formula == "unknown":
            errors.append(f"{prefix}: resolved fields require a concrete symbolic byte formula")

        expected_scope = {
            BackendRecompute.NONE: RecomputeScope.SELECTED_ROWS,
            BackendRecompute.FORWARD: RecomputeScope.SELECTED_ROWS,
            BackendRecompute.SET_CONST_FIXED: RecomputeScope.ALL_WORLDS,
            BackendRecompute.SET_CONST_0: RecomputeScope.ALL_WORLDS,
            BackendRecompute.SET_CONST: RecomputeScope.ALL_WORLDS,
            BackendRecompute.SPECIALIZED_BOUNDS: RecomputeScope.SELECTED_ROWS,
            BackendRecompute.RECOMPILE: RecomputeScope.COLD_PATH,
            BackendRecompute.UNRESOLVED: RecomputeScope.UNRESOLVED,
        }[capability.recompute.backend]
        if capability.recompute.scope != expected_scope:
            errors.append(
                f"{prefix}: recompute scope must be {expected_scope.value!r} for "
                f"{capability.recompute.backend.value!r}"
            )

    return errors


def parse_mjwarp_dr_inventory(raw: Any, *, source: Path = Path("<memory>")) -> MjwarpDrInventory:
    parser = _Parser()
    values = parser.mapping(raw, "inventory", _ROOT_KEYS)
    schema_version = parser.integer(values.get("schema_version", _MISSING), "schema_version")
    if schema_version != SCHEMA_VERSION:
        parser.errors.append(f"schema_version: expected {SCHEMA_VERSION}, got {schema_version!r}")

    raw_derived = values.get("derived_field_sets", _MISSING)
    derived_rows: list[tuple[str, tuple[str, ...]]] = []
    if not isinstance(raw_derived, dict):
        parser.errors.append("derived_field_sets: expected mapping")
    else:
        expected_keys = set(DERIVED_FIELD_SETS)
        actual_keys = set(raw_derived)
        for key in sorted(expected_keys - actual_keys):
            parser.errors.append(f"derived_field_sets: missing key `{key}`")
        for key in sorted(actual_keys - expected_keys, key=str):
            parser.errors.append(f"derived_field_sets: unknown key `{key}`")
        for key, fields in raw_derived.items():
            parsed_key = parser.string(key, "derived_field_sets.<key>")
            derived_rows.append(
                (
                    parsed_key,
                    parser.string_list(
                        fields, f"derived_field_sets.{parsed_key}", allow_empty=True
                    ),
                )
            )

    raw_capabilities = values.get("capabilities", _MISSING)
    if not isinstance(raw_capabilities, list):
        parser.errors.append("capabilities: expected non-empty list")
        raw_capabilities = []
    elif not raw_capabilities:
        parser.errors.append("capabilities: must not be empty")
    capabilities = tuple(
        _parse_capability(parser, raw_capability, index)
        for index, raw_capability in enumerate(raw_capabilities)
    )

    raw_exclusions = values.get("exclusions", _MISSING)
    if not isinstance(raw_exclusions, list):
        parser.errors.append("exclusions: expected non-empty list")
        raw_exclusions = []
    elif not raw_exclusions:
        parser.errors.append("exclusions: must not be empty")
    exclusions: list[ScopeExclusion] = []
    for index, raw_exclusion in enumerate(raw_exclusions):
        path = f"exclusions[{index}]"
        exclusion_values = parser.mapping(raw_exclusion, path, _EXCLUSION_KEYS)
        exclusions.append(
            ScopeExclusion(
                scope=parser.string(exclusion_values.get("scope", _MISSING), f"{path}.scope"),
                handling=parser.enum(
                    exclusion_values.get("handling", _MISSING),
                    f"{path}.handling",
                    ExclusionHandling,
                    ExclusionHandling.UNSUPPORTED,
                ),
                reason=parser.string(exclusion_values.get("reason", _MISSING), f"{path}.reason"),
                test_id=_parse_test_id(
                    parser, exclusion_values.get("test_id", _MISSING), f"{path}.test_id"
                ),
            )
        )

    inventory = MjwarpDrInventory(
        schema_version=schema_version,
        issue=parser.integer(values.get("issue", _MISSING), "issue"),
        backend=parser.string(values.get("backend", _MISSING), "backend"),
        source=_parse_source(parser, values.get("source", _MISSING)),
        derived_field_sets=tuple(derived_rows),
        required_capability_ids=parser.string_list(
            values.get("required_capability_ids", _MISSING), "required_capability_ids"
        ),
        capabilities=capabilities,
        exclusions=tuple(exclusions),
        source_path=source,
    )
    parser.errors.extend(_semantic_errors(inventory))
    if parser.errors:
        raise DrInventoryValidationError(source, parser.errors)
    return inventory


def load_mjwarp_dr_inventory(path: Path) -> MjwarpDrInventory:
    try:
        config = OmegaConf.load(path)
        raw = OmegaConf.to_container(config, resolve=False)
    except Exception as exc:  # noqa: BLE001 - normalize malformed YAML for the CLI
        raise DrInventoryValidationError(
            path, [f"cannot load YAML: {type(exc).__name__}: {exc}"]
        ) from exc
    return parse_mjwarp_dr_inventory(raw, source=path)


def inventory_claim_gap_errors(
    inventory: MjwarpDrInventory, claim_inventory: ClaimGapInventory
) -> tuple[str, ...]:
    errors: list[str] = []
    entries_by_test: dict[str, list[Any]] = {}
    for entry in claim_inventory.entries:
        if entry.role == EvidenceRole.ACCEPTANCE:
            entries_by_test.setdefault(entry.test_id, []).append(entry)

    expected_kinds = {
        "contract": EvidenceKind.CONTRACT,
        "row_isolation": EvidenceKind.DIFFERENTIAL,
        "physics_effect": EvidenceKind.EFFECT,
        "graph": EvidenceKind.FAULT,
        "memory": EvidenceKind.PERFORMANCE,
    }
    for capability in inventory.capabilities:
        for role, expected_kind in expected_kinds.items():
            test_id = getattr(capability.required_tests, role)
            entries = entries_by_test.get(test_id, [])
            if not entries:
                errors.append(
                    f"{capability.capability_id}: {role} test `{test_id}` is absent from "
                    "the claim-gap acceptance inventory"
                )
            elif not any(entry.evidence_kind == expected_kind for entry in entries):
                actual = sorted({entry.evidence_kind.value for entry in entries})
                errors.append(
                    f"{capability.capability_id}: {role} test `{test_id}` requires "
                    f"{expected_kind.value} evidence, got {actual!r}"
                )
        if capability.storage.measurement_test_id != capability.required_tests.memory:
            errors.append(
                f"{capability.capability_id}: storage measurement test must equal the "
                "required memory test"
            )
    for exclusion in inventory.exclusions:
        entries = entries_by_test.get(exclusion.test_id, [])
        if not entries:
            errors.append(
                f"exclusion {exclusion.scope!r}: test `{exclusion.test_id}` is absent from "
                "the claim-gap acceptance inventory"
            )
        elif not any(entry.evidence_kind == EvidenceKind.FAULT for entry in entries):
            errors.append(
                f"exclusion {exclusion.scope!r}: unsupported boundary requires fault evidence"
            )
    return tuple(errors)
