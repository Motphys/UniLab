from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from unilab.base.backend import (
    BoundMutationPlan,
    BufferContract,
    BufferLayout,
    BufferLifetime,
    BufferMutability,
    BufferOwner,
    BufferPlacement,
    BufferView,
    ExecutionProfile,
    ExternalWrenchMutationBatch,
    ModelParameterMutationBatch,
    MutationBaseline,
    MutationCapability,
    MutationCapabilityCase,
    MutationCapabilityDescriptor,
    MutationCapabilityManifest,
    MutationCapabilityRowScope,
    MutationCommitPhase,
    MutationContractError,
    MutationEntityKind,
    MutationFieldKind,
    MutationFieldStorageKind,
    MutationGraphImpact,
    MutationGraphInvalidation,
    MutationOperation,
    MutationPersistence,
    MutationRecomputeLevel,
    MutationSelectorMode,
    MutationSelectorSpec,
    MutationSpec,
    MutationTargetKind,
    MutationTargetSpec,
    MutationTrigger,
    MutationValueBatch,
    RowSelection,
    SimBackend,
    SimulationStateMutationBatch,
    TaskStateMutationBatch,
    TypedBackendMutationBatch,
    bind_mutation_plan,
)
from unilab.dr import (
    DomainRandomizationCapabilities,
    DomainRandomizationExecutionMode,
    DomainRandomizationManager,
    DomainRandomizationProvider,
    ResetPlan,
    ResetRandomizationPayload,
    UnsupportedDomainRandomizationError,
)
from unilab.dr.types import RESET_TERM_BASE_MASS


def _value_template(
    row_shape: tuple[int, ...],
    *,
    dtype: str = "float32",
    placement: BufferPlacement | None = None,
    owner: BufferOwner = BufferOwner.MANAGER,
    mutability: BufferMutability = BufferMutability.READ_ONLY,
    lifetime: BufferLifetime = BufferLifetime.UNTIL_COMMIT,
    address_stable: bool = True,
) -> BufferContract:
    return BufferContract(
        row_shape=row_shape,
        dtype=dtype,
        layout=BufferLayout.C_CONTIGUOUS,
        placement=placement or BufferPlacement.host(),
        owner=owner,
        mutability=mutability,
        lifetime=lifetime,
        dlpack_exportable=False,
        address_stable=address_stable,
    )


def _target(
    target_key: str,
    target_kind: MutationTargetKind,
    entity_kind: MutationEntityKind,
    field_kind: MutationFieldKind,
    selector: str | None,
) -> MutationTargetSpec:
    return MutationTargetSpec(
        target_key=target_key,
        target_kind=target_kind,
        entity_kind=entity_kind,
        field_kind=field_kind,
        selector=selector,
    )


def _spec(
    term_key: str,
    target: MutationTargetSpec,
    *,
    trigger: MutationTrigger,
    phase: MutationCommitPhase,
    operation: MutationOperation,
    baseline: MutationBaseline,
    persistence: MutationPersistence,
    recompute: MutationRecomputeLevel,
    value_template: BufferContract,
) -> MutationSpec:
    return MutationSpec(
        term_key=term_key,
        target=target,
        trigger=trigger,
        commit_phase=phase,
        operation=operation,
        baseline=baseline,
        persistence=persistence,
        recompute=recompute,
        value_template=value_template,
    )


def _capability(
    target: MutationTargetSpec,
    *,
    entity_count: int,
    trigger: MutationTrigger,
    phase: MutationCommitPhase,
    operation: MutationOperation,
    baseline: MutationBaseline,
    persistence: MutationPersistence,
    recompute: MutationRecomputeLevel,
    value_template: BufferContract,
) -> MutationCapability:
    return MutationCapability(
        target_key=target.target_key,
        target_kind=target.target_kind,
        entity_kind=target.entity_kind,
        field_kind=target.field_kind,
        entity_count=entity_count,
        value_template=value_template,
        triggers=frozenset({trigger}),
        commit_phases=frozenset({phase}),
        operations=frozenset({operation}),
        baselines=frozenset({baseline}),
        persistences=frozenset({persistence}),
        recompute_levels=frozenset({recompute}),
    )


def _verification_case(
    spec: MutationSpec,
    *,
    case_id: str = "fake.host.state-reset",
    test_id: str = "tests/dr/test_mutation_contract.py::test_manifest_case",
    execution_profile: ExecutionProfile = ExecutionProfile.HOST_NUMPY,
) -> MutationCapabilityCase:
    return MutationCapabilityCase(
        case_id=case_id,
        mandatory_test_id=test_id,
        execution_profile=execution_profile,
        trigger=spec.trigger,
        commit_phase=spec.commit_phase,
        operation=spec.operation,
        baseline=spec.baseline,
        persistence=spec.persistence,
        recompute=spec.recompute,
        row_scope=MutationCapabilityRowScope.SELECTED_ROWS,
    )


def _descriptor(
    spec: MutationSpec,
    *,
    cases: tuple[MutationCapabilityCase, ...] | None = None,
) -> MutationCapabilityDescriptor:
    return MutationCapabilityDescriptor(
        capability_id="fake.state-reset",
        direct_fields=("data.qpos",),
        derived_fields=(),
        storage_kind=MutationFieldStorageKind.DATA_NATIVE,
        graph_impact=MutationGraphImpact.STABLE_ADDRESS,
        graph_invalidations=frozenset(),
        cases=cases or (_verification_case(spec),),
    )


def _fixtures() -> tuple[
    tuple[MutationSpec, ...],
    tuple[MutationCapability, ...],
    dict[tuple[MutationEntityKind, str], tuple[int, ...]],
]:
    model_target = _target(
        "physics.body.mass",
        MutationTargetKind.MODEL_PARAMETER,
        MutationEntityKind.BODY,
        MutationFieldKind.MASS,
        "robot.links",
    )
    state_target = _target(
        "state.dof.position",
        MutationTargetKind.SIMULATION_STATE,
        MutationEntityKind.DOF,
        MutationFieldKind.POSITION,
        "robot.joints",
    )
    wrench_target = _target(
        "wrench.body.force",
        MutationTargetKind.EXTERNAL_WRENCH,
        MutationEntityKind.BODY,
        MutationFieldKind.FORCE,
        "robot.feet",
    )
    task_target = _target(
        "task.command.value",
        MutationTargetKind.TASK_STATE,
        MutationEntityKind.TASK,
        MutationFieldKind.VALUE,
        "commands.velocity",
    )
    model_value = _value_template((1,))
    state_value = _value_template((1,))
    wrench_value = _value_template((3,))
    task_value = _value_template((2,))
    specs = (
        _spec(
            "randomize.mass",
            model_target,
            trigger=MutationTrigger.RESET,
            phase=MutationCommitPhase.RESET,
            operation=MutationOperation.SCALE,
            baseline=MutationBaseline.DEFAULT,
            persistence=MutationPersistence.EPISODE,
            recompute=MutationRecomputeLevel.FULL,
            value_template=model_value,
        ),
        _spec(
            "reset.joint_position",
            state_target,
            trigger=MutationTrigger.RESET,
            phase=MutationCommitPhase.RESET,
            operation=MutationOperation.SET,
            baseline=MutationBaseline.DEFAULT,
            persistence=MutationPersistence.EPISODE,
            recompute=MutationRecomputeLevel.KINEMATICS,
            value_template=state_value,
        ),
        _spec(
            "push.feet",
            wrench_target,
            trigger=MutationTrigger.INTERVAL,
            phase=MutationCommitPhase.PRE_PHYSICS,
            operation=MutationOperation.SET,
            baseline=MutationBaseline.CURRENT,
            persistence=MutationPersistence.ONE_STEP,
            recompute=MutationRecomputeLevel.NONE,
            value_template=wrench_value,
        ),
        _spec(
            "command.velocity",
            task_target,
            trigger=MutationTrigger.STEP,
            phase=MutationCommitPhase.POST_PHYSICS,
            operation=MutationOperation.SET,
            baseline=MutationBaseline.CURRENT,
            persistence=MutationPersistence.ONE_STEP,
            recompute=MutationRecomputeLevel.NONE,
            value_template=task_value,
        ),
    )
    capabilities = (
        _capability(
            model_target,
            entity_count=8,
            trigger=MutationTrigger.RESET,
            phase=MutationCommitPhase.RESET,
            operation=MutationOperation.SCALE,
            baseline=MutationBaseline.DEFAULT,
            persistence=MutationPersistence.EPISODE,
            recompute=MutationRecomputeLevel.FULL,
            value_template=model_value,
        ),
        _capability(
            state_target,
            entity_count=12,
            trigger=MutationTrigger.RESET,
            phase=MutationCommitPhase.RESET,
            operation=MutationOperation.SET,
            baseline=MutationBaseline.DEFAULT,
            persistence=MutationPersistence.EPISODE,
            recompute=MutationRecomputeLevel.KINEMATICS,
            value_template=state_value,
        ),
        _capability(
            wrench_target,
            entity_count=8,
            trigger=MutationTrigger.INTERVAL,
            phase=MutationCommitPhase.PRE_PHYSICS,
            operation=MutationOperation.SET,
            baseline=MutationBaseline.CURRENT,
            persistence=MutationPersistence.ONE_STEP,
            recompute=MutationRecomputeLevel.NONE,
            value_template=wrench_value,
        ),
        _capability(
            task_target,
            entity_count=4,
            trigger=MutationTrigger.STEP,
            phase=MutationCommitPhase.POST_PHYSICS,
            operation=MutationOperation.SET,
            baseline=MutationBaseline.CURRENT,
            persistence=MutationPersistence.ONE_STEP,
            recompute=MutationRecomputeLevel.NONE,
            value_template=task_value,
        ),
    )
    selectors = {
        (MutationEntityKind.BODY, "robot.links"): (1, 3),
        (MutationEntityKind.DOF, "robot.joints"): (0, 2, 4),
        (MutationEntityKind.BODY, "robot.feet"): (5, 6),
        (MutationEntityKind.TASK, "commands.velocity"): (0,),
    }
    return specs, capabilities, selectors


class _Resolver:
    def __init__(self, values: dict[tuple[MutationEntityKind, str], tuple[int, ...]]) -> None:
        self.values = values
        self.calls: list[tuple[MutationEntityKind, str]] = []

    def __call__(self, target: MutationTargetSpec) -> tuple[int, ...]:
        selector = target.selector_spec
        assert selector is not None
        key = (target.entity_kind, selector.semantic_key)
        self.calls.append(key)
        return self.values[key]


def _bind(
    *,
    num_envs: int = 19,
    specs: tuple[MutationSpec, ...] | None = None,
    capabilities: tuple[MutationCapability, ...] | None = None,
    selectors: dict[tuple[MutationEntityKind, str], tuple[int, ...]] | None = None,
    backend_type: str = "fake",
    backend_instance_id: str = "fake:0",
) -> tuple[BoundMutationPlan, _Resolver]:
    default_specs, default_capabilities, default_selectors = _fixtures()
    resolver = _Resolver(selectors or default_selectors)
    plan = bind_mutation_plan(
        backend_type=backend_type,
        backend_instance_id=backend_instance_id,
        num_envs=num_envs,
        specs=specs or default_specs,
        capabilities=capabilities or default_capabilities,
        resolve_selector=resolver,
    )
    return plan, resolver


def _value(
    plan: BoundMutationPlan,
    term_key: str,
    rows: RowSelection,
    rng: np.random.Generator,
) -> MutationValueBatch:
    field_index = plan.spec_index(term_key)
    contract = plan.specs[field_index].value_buffer
    shape = (rows.count, *contract.row_shape)
    handle = np.ascontiguousarray(rng.normal(size=shape), dtype=contract.dtype)
    return MutationValueBatch(
        plan=plan,
        field_index=field_index,
        rows=rows,
        buffer=BufferView(handle=handle, shape=shape, contract=contract),
    )


@pytest.mark.parametrize("seed", [0, 7])
@pytest.mark.parametrize("num_envs", [1, 19])
def test_typed_mutation_plan_and_batch_cover_all_target_categories(
    seed: int, num_envs: int
) -> None:
    specs, capabilities, _ = _fixtures()
    plan, resolver = _bind(
        num_envs=num_envs,
        specs=tuple(reversed(specs)),
        capabilities=tuple(reversed(capabilities)),
    )
    assert plan.fingerprint.startswith("backend-mutation-contract-v1:")
    assert [spec.term_key for spec in plan.specs] == sorted(spec.term_key for spec in specs)
    assert len(resolver.calls) == 4
    assert all(not hasattr(spec.target, "selector") for spec in plan.specs)

    rows = (
        RowSelection.all(num_envs) if num_envs == 1 else RowSelection.selected(num_envs, (18, 0, 7))
    )
    rng = np.random.default_rng(seed)
    model = _value(plan, "randomize.mass", rows, rng)
    state = _value(plan, "reset.joint_position", rows, rng)
    wrench = _value(plan, "push.feet", rows, rng)
    task_state = _value(plan, "command.velocity", rows, rng)
    batch = TypedBackendMutationBatch(
        plan=plan,
        rows=rows,
        model=ModelParameterMutationBatch((model,)),
        state=SimulationStateMutationBatch((state,)),
        wrench=ExternalWrenchMutationBatch((wrench,)),
        task_state=TaskStateMutationBatch((task_state,)),
    )
    assert batch.plan_fingerprint == plan.fingerprint
    assert model.buffer.shape == (rows.count, 2, 1)
    assert state.buffer.shape == (rows.count, 3, 1)
    assert wrench.buffer.shape == (rows.count, 2, 3)
    assert task_state.buffer.shape == (rows.count, 1, 2)

    empty = TypedBackendMutationBatch(plan=plan, rows=rows)
    assert not empty.model.values
    assert not empty.state.values
    assert not empty.wrench.values
    assert not empty.task_state.values


def test_selector_is_resolved_once_per_shared_cold_path_binding() -> None:
    specs, capabilities, selectors = _fixtures()
    alias = replace(specs[0], term_key="randomize.mass.alias")
    model_capability = capabilities[0]
    expanded_capability = replace(
        model_capability,
        commit_phases=frozenset({MutationCommitPhase.RESET, MutationCommitPhase.PRE_PHYSICS}),
    )
    alias = replace(alias, commit_phase=MutationCommitPhase.PRE_PHYSICS)
    plan, resolver = _bind(
        specs=(specs[0], alias),
        capabilities=(expanded_capability,),
        selectors=selectors,
    )
    assert len(plan.specs) == 2
    assert resolver.calls == [(MutationEntityKind.BODY, "robot.links")]


def test_mutation_selector_normalizes_legacy_exact_input_and_preserves_structured_metadata() -> (
    None
):
    legacy = _target(
        "state.dof.position",
        MutationTargetKind.SIMULATION_STATE,
        MutationEntityKind.DOF,
        MutationFieldKind.POSITION,
        "left_hip_pitch_joint",
    )
    selector = legacy.selector_spec
    assert selector == MutationSelectorSpec(
        semantic_key="left_hip_pitch_joint",
        mode=MutationSelectorMode.EXACT,
        expressions=("left_hip_pitch_joint",),
    )
    assert selector.require_exact_singleton(context="test") == "left_hip_pitch_joint"

    with pytest.raises(MutationContractError, match="only supports exact"):
        MutationSelectorSpec(
            semantic_key="robot.joints",
            mode=MutationSelectorMode.REGEX,
            expressions=(".*_joint",),
            entity_ids=(0, 1),
        ).require_exact_singleton(context="test")
    with pytest.raises(MutationContractError, match="exactly one mutation selector expression"):
        MutationSelectorSpec(
            semantic_key="robot.joints",
            mode=MutationSelectorMode.EXACT,
            expressions=("left_joint", "right_joint"),
            entity_ids=(0, 1),
        ).require_exact_singleton(context="test")

    specs, capabilities, selectors = _fixtures()
    compiled_regex = replace(
        specs[0],
        target=replace(
            specs[0].target,
            selector=MutationSelectorSpec(
                semantic_key="robot.links",
                mode=MutationSelectorMode.REGEX,
                expressions=("link_.*",),
                entity_ids=(1, 3),
            ),
        ),
    )
    mismatched = dict(selectors)
    mismatched[(MutationEntityKind.BODY, "robot.links")] = (1,)
    with pytest.raises(MutationContractError, match="cardinality differs"):
        _bind(
            specs=(compiled_regex,),
            capabilities=(capabilities[0],),
            selectors=mismatched,
        )


def test_typed_mutation_conflicts_fail_closed() -> None:
    specs, capabilities, selectors = _fixtures()
    model = specs[0]
    overlap_target = replace(model.target, selector="robot.overlap")
    overlap = replace(model, term_key="randomize.mass.overlap", target=overlap_target)
    selectors = dict(selectors)
    selectors[(MutationEntityKind.BODY, "robot.overlap")] = (3, 4)

    with pytest.raises(MutationContractError, match="conflict.*entities 3"):
        _bind(specs=(model, overlap), capabilities=(capabilities[0],), selectors=selectors)

    duplicate_key = replace(model, target=replace(model.target, selector="robot.other"))
    selectors[(MutationEntityKind.BODY, "robot.other")] = (6,)
    with pytest.raises(MutationContractError, match="term keys must be unique"):
        _bind(
            specs=(model, duplicate_key),
            capabilities=(capabilities[0],),
            selectors=selectors,
        )


def test_mutation_plan_fingerprint_is_canonical_and_semantically_sensitive() -> None:
    specs, capabilities, selectors = _fixtures()
    baseline, _ = _bind(specs=specs, capabilities=capabilities, selectors=selectors)
    reordered, _ = _bind(
        specs=tuple(reversed(specs)),
        capabilities=tuple(reversed(capabilities)),
        selectors=selectors,
    )
    assert reordered.fingerprint == baseline.fingerprint

    changed_selectors = dict(selectors)
    changed_selectors[(MutationEntityKind.BODY, "robot.links")] = (1, 4)
    changed_entities, _ = _bind(
        specs=specs,
        capabilities=capabilities,
        selectors=changed_selectors,
    )
    assert changed_entities.fingerprint != baseline.fingerprint

    model_capability = capabilities[0]
    expanded_capability = replace(
        model_capability,
        operations=frozenset({MutationOperation.SCALE, MutationOperation.SET}),
    )
    changed_capability, _ = _bind(
        specs=specs,
        capabilities=(expanded_capability, *capabilities[1:]),
        selectors=selectors,
    )
    assert changed_capability.fingerprint != baseline.fingerprint

    float64_value = replace(specs[0].value_template, dtype="float64")
    changed_value_spec = replace(specs[0], value_template=float64_value)
    changed_value_capability = replace(model_capability, value_template=float64_value)
    changed_value, _ = _bind(
        specs=(changed_value_spec, *specs[1:]),
        capabilities=(changed_value_capability, *capabilities[1:]),
        selectors=selectors,
    )
    assert changed_value.fingerprint != baseline.fingerprint

    broad_capability = replace(
        model_capability,
        triggers=frozenset({MutationTrigger.RESET, MutationTrigger.INTERVAL}),
        commit_phases=frozenset({MutationCommitPhase.RESET, MutationCommitPhase.PRE_PHYSICS}),
        operations=frozenset({MutationOperation.SCALE, MutationOperation.SET}),
        baselines=frozenset({MutationBaseline.DEFAULT, MutationBaseline.CURRENT}),
        persistences=frozenset({MutationPersistence.EPISODE, MutationPersistence.ONE_STEP}),
        recompute_levels=frozenset({MutationRecomputeLevel.FULL, MutationRecomputeLevel.DYNAMICS}),
    )
    variants = (
        replace(specs[0], trigger=MutationTrigger.INTERVAL),
        replace(specs[0], commit_phase=MutationCommitPhase.PRE_PHYSICS),
        replace(specs[0], operation=MutationOperation.SET),
        replace(specs[0], baseline=MutationBaseline.CURRENT),
        replace(specs[0], persistence=MutationPersistence.ONE_STEP),
        replace(specs[0], recompute=MutationRecomputeLevel.DYNAMICS),
    )
    for variant in variants:
        changed, _ = _bind(
            specs=(variant,),
            capabilities=(broad_capability,),
            selectors=selectors,
        )
        original, _ = _bind(
            specs=(specs[0],),
            capabilities=(broad_capability,),
            selectors=selectors,
        )
        assert changed.fingerprint != original.fingerprint


def test_capability_manifest_is_bound_as_the_complete_plan_identity() -> None:
    specs, capabilities, selectors = _fixtures()
    spec = specs[1]
    capability = replace(capabilities[1], descriptor=_descriptor(spec))
    manifest = MutationCapabilityManifest(
        backend_type="fake",
        execution_profile=ExecutionProfile.HOST_NUMPY,
        capabilities=(capability,),
    )
    resolver = _Resolver(selectors)
    plan = bind_mutation_plan(
        backend_type="fake",
        backend_instance_id="fake:0",
        num_envs=19,
        specs=(spec,),
        capabilities=(capability,),
        resolve_selector=resolver,
        capability_manifest=manifest,
    )
    assert plan.capability_manifest_fingerprint == manifest.fingerprint

    descriptor = capability.descriptor
    assert descriptor is not None
    changed_capability = replace(
        capability,
        descriptor=replace(descriptor, direct_fields=("data.qpos_expanded",)),
    )
    changed_manifest = MutationCapabilityManifest(
        backend_type="fake",
        execution_profile=ExecutionProfile.HOST_NUMPY,
        capabilities=(changed_capability,),
    )
    changed_plan = bind_mutation_plan(
        backend_type="fake",
        backend_instance_id="fake:0",
        num_envs=19,
        specs=(spec,),
        capabilities=(changed_capability,),
        resolve_selector=_Resolver(selectors),
        capability_manifest=changed_manifest,
    )
    assert changed_manifest.fingerprint != manifest.fingerprint
    assert changed_plan.fingerprint != plan.fingerprint

    tampered_plan = replace(
        plan,
        capability_manifest_fingerprint="mutation-capability-manifest-v1:tampered",
    )
    with pytest.raises(MutationContractError, match="different plan or fingerprint"):
        plan.require_compatible(tampered_plan)

    object.__setattr__(manifest, "fingerprint", "mutation-capability-manifest-v1:tampered")
    with pytest.raises(MutationContractError, match="fingerprint does not match"):
        bind_mutation_plan(
            backend_type="fake",
            backend_instance_id="fake:0",
            num_envs=19,
            specs=(spec,),
            capabilities=(capability,),
            resolve_selector=_Resolver(selectors),
            capability_manifest=manifest,
        )


def test_capability_cases_reject_unverified_cartesian_combinations() -> None:
    specs, capabilities, selectors = _fixtures()
    first = specs[1]
    second = replace(
        first,
        term_key="interval.state",
        trigger=MutationTrigger.INTERVAL,
        commit_phase=MutationCommitPhase.PRE_PHYSICS,
        baseline=MutationBaseline.CURRENT,
        persistence=MutationPersistence.ONE_STEP,
        recompute=MutationRecomputeLevel.NONE,
    )
    cases = (
        _verification_case(
            second,
            case_id="fake.host.interval",
            test_id="tests/dr/test_mutation_contract.py::test_interval_manifest_case",
        ),
        _verification_case(first, case_id="fake.host.reset"),
    )
    capability = replace(
        capabilities[1],
        triggers=frozenset({first.trigger, second.trigger}),
        commit_phases=frozenset({first.commit_phase, second.commit_phase}),
        baselines=frozenset({first.baseline, second.baseline}),
        persistences=frozenset({first.persistence, second.persistence}),
        recompute_levels=frozenset({first.recompute, second.recompute}),
        descriptor=_descriptor(first, cases=cases),
    )
    manifest = MutationCapabilityManifest(
        backend_type="fake",
        execution_profile=ExecutionProfile.HOST_NUMPY,
        capabilities=(capability,),
    )
    mixed = replace(first, commit_phase=second.commit_phase)
    with pytest.raises(MutationContractError, match="unverified capability combination"):
        bind_mutation_plan(
            backend_type="fake",
            backend_instance_id="fake:0",
            num_envs=19,
            specs=(mixed,),
            capabilities=(capability,),
            resolve_selector=_Resolver(selectors),
            capability_manifest=manifest,
        )


def test_capability_manifest_validation_fails_near_malformed_metadata() -> None:
    specs, capabilities, _ = _fixtures()
    spec = specs[1]
    case = _verification_case(spec)

    with pytest.raises(MutationContractError, match="duplicate semantic"):
        _descriptor(
            spec,
            cases=(
                case,
                replace(
                    case,
                    case_id="fake.host.state-reset.alias",
                    mandatory_test_id="tests/dr/test_mutation_contract.py::test_manifest_alias",
                ),
            ),
        )
    with pytest.raises(MutationContractError, match="native mutation storage"):
        replace(
            _descriptor(spec),
            graph_impact=MutationGraphImpact.RECAPTURE_REQUIRED,
            graph_invalidations=frozenset({MutationGraphInvalidation.RESET_GRAPH}),
        )
    with pytest.raises(MutationContractError, match="no field-level descriptor"):
        MutationCapabilityManifest(
            backend_type="fake",
            execution_profile=ExecutionProfile.HOST_NUMPY,
            capabilities=(capabilities[1],),
        )

    capability = replace(capabilities[1], descriptor=_descriptor(spec))
    with pytest.raises(MutationContractError, match="placement does not match profile"):
        MutationCapabilityManifest(
            backend_type="fake",
            execution_profile=ExecutionProfile.DEVICE_RESIDENT,
            capabilities=(capability,),
        )
    manifest = MutationCapabilityManifest(
        backend_type="fake",
        execution_profile=ExecutionProfile.HOST_NUMPY,
        capabilities=(capability,),
    )
    with pytest.raises(MutationContractError, match="does not match the supplied capability"):
        bind_mutation_plan(
            backend_type="fake",
            backend_instance_id="fake:0",
            num_envs=19,
            specs=(spec,),
            capabilities=(replace(capability, entity_count=13),),
            resolve_selector=_Resolver({}),
            capability_manifest=manifest,
        )


def _unsupported_target() -> None:
    specs, capabilities, selectors = _fixtures()
    invalid = replace(
        specs[0],
        target=replace(specs[0].target, target_key="physics.body.unknown"),
    )
    with pytest.raises(MutationContractError, match="not supported"):
        _bind(specs=(invalid,), capabilities=capabilities, selectors=selectors)


def _unsupported_target_metadata() -> None:
    specs, capabilities, selectors = _fixtures()
    invalid = replace(
        specs[0],
        target=replace(specs[0].target, field_kind=MutationFieldKind.INERTIA),
    )
    with pytest.raises(MutationContractError, match="registered capability"):
        _bind(specs=(invalid,), capabilities=(capabilities[0],), selectors=selectors)


def _unsupported_trigger() -> None:
    specs, capabilities, selectors = _fixtures()
    invalid = replace(specs[0], trigger=MutationTrigger.INTERVAL)
    with pytest.raises(MutationContractError, match="unsupported trigger"):
        _bind(specs=(invalid,), capabilities=(capabilities[0],), selectors=selectors)


def _unsupported_phase() -> None:
    specs, capabilities, selectors = _fixtures()
    invalid = replace(specs[0], commit_phase=MutationCommitPhase.PRE_PHYSICS)
    with pytest.raises(MutationContractError, match="unsupported commit phase"):
        _bind(specs=(invalid,), capabilities=(capabilities[0],), selectors=selectors)


def _unsupported_operation() -> None:
    specs, capabilities, selectors = _fixtures()
    invalid = replace(specs[0], operation=MutationOperation.ADD)
    with pytest.raises(MutationContractError, match="unsupported operation"):
        _bind(specs=(invalid,), capabilities=(capabilities[0],), selectors=selectors)


def _unsupported_baseline() -> None:
    specs, capabilities, selectors = _fixtures()
    invalid = replace(specs[0], baseline=MutationBaseline.CURRENT)
    with pytest.raises(MutationContractError, match="unsupported baseline"):
        _bind(specs=(invalid,), capabilities=(capabilities[0],), selectors=selectors)


def _unsupported_persistence() -> None:
    specs, capabilities, selectors = _fixtures()
    invalid = replace(specs[0], persistence=MutationPersistence.ONE_STEP)
    with pytest.raises(MutationContractError, match="unsupported persistence"):
        _bind(specs=(invalid,), capabilities=(capabilities[0],), selectors=selectors)


def _unsupported_recompute() -> None:
    specs, capabilities, selectors = _fixtures()
    invalid = replace(specs[0], recompute=MutationRecomputeLevel.NONE)
    with pytest.raises(MutationContractError, match="unsupported recompute level"):
        _bind(specs=(invalid,), capabilities=(capabilities[0],), selectors=selectors)


def _unsupported_value_metadata() -> None:
    specs, capabilities, selectors = _fixtures()
    invalid = replace(
        specs[0],
        value_template=replace(specs[0].value_template, dtype="float64"),
    )
    with pytest.raises(MutationContractError, match="value metadata"):
        _bind(specs=(invalid,), capabilities=(capabilities[0],), selectors=selectors)


def _missing_selector_result() -> None:
    specs, capabilities, selectors = _fixtures()
    selectors = dict(selectors)
    selectors[(MutationEntityKind.BODY, "robot.links")] = ()
    with pytest.raises(MutationContractError, match="resolved no entities"):
        _bind(specs=(specs[0],), capabilities=(capabilities[0],), selectors=selectors)


def _unknown_selector() -> None:
    specs, capabilities, selectors = _fixtures()
    selectors = dict(selectors)
    del selectors[(MutationEntityKind.BODY, "robot.links")]
    with pytest.raises(MutationContractError, match="failed to resolve.*robot.links"):
        _bind(specs=(specs[0],), capabilities=(capabilities[0],), selectors=selectors)


def _invalid_selector_result_type() -> None:
    specs, capabilities, _ = _fixtures()

    def invalid_resolver(_: MutationTargetSpec) -> tuple[int, ...]:
        return cast(Any, [1, 3])

    with pytest.raises(MutationContractError, match="resolver must return a tuple"):
        bind_mutation_plan(
            backend_type="fake",
            backend_instance_id="fake:0",
            num_envs=19,
            specs=(specs[0],),
            capabilities=(capabilities[0],),
            resolve_selector=invalid_resolver,
        )


def _duplicate_selector_result() -> None:
    specs, capabilities, selectors = _fixtures()
    selectors = dict(selectors)
    selectors[(MutationEntityKind.BODY, "robot.links")] = (1, 1)
    with pytest.raises(MutationContractError, match="IDs must be unique"):
        _bind(specs=(specs[0],), capabilities=(capabilities[0],), selectors=selectors)


def _out_of_range_selector_result() -> None:
    specs, capabilities, selectors = _fixtures()
    selectors = dict(selectors)
    selectors[(MutationEntityKind.BODY, "robot.links")] = (8,)
    with pytest.raises(MutationContractError, match="out-of-range"):
        _bind(specs=(specs[0],), capabilities=(capabilities[0],), selectors=selectors)


def _duplicate_capability() -> None:
    specs, capabilities, selectors = _fixtures()
    with pytest.raises(MutationContractError, match="capability target keys must be unique"):
        _bind(
            specs=(specs[0],),
            capabilities=(capabilities[0], capabilities[0]),
            selectors=selectors,
        )


_BIND_FAULTS: tuple[Callable[[], None], ...] = (
    _unsupported_target,
    _unsupported_target_metadata,
    _unsupported_trigger,
    _unsupported_phase,
    _unsupported_operation,
    _unsupported_baseline,
    _unsupported_persistence,
    _unsupported_recompute,
    _unsupported_value_metadata,
    _missing_selector_result,
    _unknown_selector,
    _invalid_selector_result_type,
    _duplicate_selector_result,
    _out_of_range_selector_result,
    _duplicate_capability,
)


@pytest.mark.parametrize("fault", _BIND_FAULTS, ids=lambda fault: fault.__name__)
def test_typed_mutation_binding_faults_fail_closed(fault: Callable[[], None]) -> None:
    fault()


def test_mutation_value_contract_rejects_wrong_owner_lifetime_and_stability() -> None:
    specs, _, _ = _fixtures()
    target = specs[0].target

    def build(value_template: BufferContract) -> MutationSpec:
        return MutationSpec(
            term_key="invalid.value",
            target=target,
            trigger=MutationTrigger.RESET,
            commit_phase=MutationCommitPhase.RESET,
            operation=MutationOperation.SET,
            baseline=MutationBaseline.DEFAULT,
            persistence=MutationPersistence.EPISODE,
            recompute=MutationRecomputeLevel.NONE,
            value_template=value_template,
        )

    with pytest.raises(MutationContractError, match="manager-owned"):
        build(_value_template((1,), owner=BufferOwner.RUNTIME))
    with pytest.raises(MutationContractError, match="read-only"):
        build(_value_template((1,), mutability=BufferMutability.READ_WRITE))
    with pytest.raises(MutationContractError, match="until_commit"):
        build(_value_template((1,), lifetime=BufferLifetime.PLAN))
    with pytest.raises(MutationContractError, match="plan-stable"):
        build(_value_template((1,), address_stable=False))


def test_mutation_target_semantics_reject_invalid_entity_field_pairs() -> None:
    with pytest.raises(MutationContractError, match="actuator.*does not support field kind mass"):
        _target(
            "physics.actuator.mass",
            MutationTargetKind.MODEL_PARAMETER,
            MutationEntityKind.ACTUATOR,
            MutationFieldKind.MASS,
            "robot.actuators",
        )


def test_mutation_batch_plan_rows_metadata_and_category_faults_fail_closed() -> None:
    plan, _ = _bind()
    rows = RowSelection.selected(19, (18, 0, 7))
    rng = np.random.default_rng(0)
    model = _value(plan, "randomize.mass", rows, rng)

    wrong_plan, _ = _bind(backend_type="other", backend_instance_id="other:0")
    with pytest.raises(MutationContractError, match="different plan"):
        TypedBackendMutationBatch(
            plan=wrong_plan,
            rows=rows,
            model=ModelParameterMutationBatch((model,)),
        )

    wrong_rows = RowSelection.selected(19, (0, 7, 18))
    with pytest.raises(MutationContractError, match="rows do not match"):
        TypedBackendMutationBatch(
            plan=plan,
            rows=wrong_rows,
            model=ModelParameterMutationBatch((model,)),
        )

    with pytest.raises(MutationContractError, match="wrong typed sub-batch"):
        TypedBackendMutationBatch(
            plan=plan,
            rows=rows,
            state=SimulationStateMutationBatch((model,)),
        )

    with pytest.raises(MutationContractError, match="duplicate mutation fields"):
        TypedBackendMutationBatch(
            plan=plan,
            rows=rows,
            model=ModelParameterMutationBatch((model, model)),
        )

    spec = model.spec
    wrong_contracts = (
        replace(spec.value_buffer, owner=BufferOwner.RUNNER),
        replace(spec.value_buffer, lifetime=BufferLifetime.PLAN),
        replace(spec.value_buffer, dtype="float64"),
        replace(spec.value_buffer, placement=BufferPlacement.device("cuda", 0)),
    )
    for wrong_contract in wrong_contracts:
        wrong_handle = np.zeros(model.buffer.shape, dtype=wrong_contract.dtype)
        with pytest.raises(MutationContractError, match="metadata does not match"):
            MutationValueBatch(
                plan=plan,
                field_index=model.field_index,
                rows=rows,
                buffer=BufferView(wrong_handle, model.buffer.shape, wrong_contract),
            )

    tampered_plan = replace(plan, fingerprint="backend-mutation-contract-v1:tampered")
    tampered_value = replace(model, plan=tampered_plan)
    with pytest.raises(MutationContractError, match="different plan or fingerprint"):
        TypedBackendMutationBatch(
            plan=plan,
            rows=rows,
            model=ModelParameterMutationBatch((tampered_value,)),
        )

    with pytest.raises(MutationContractError, match="different backend"):
        plan.require_owner(backend_type="other", backend_instance_id="other:0")

    with pytest.raises(MutationContractError, match="requires shape"):
        MutationValueBatch(
            plan=plan,
            field_index=model.field_index,
            rows=rows,
            buffer=BufferView(
                np.zeros((rows.count, 1, 1), dtype=spec.value_buffer.dtype),
                (rows.count, 1, 1),
                spec.value_buffer,
            ),
        )

    with pytest.raises(MutationContractError, match="field_index is not bound"):
        MutationValueBatch(
            plan=plan,
            field_index=len(plan.specs),
            rows=rows,
            buffer=model.buffer,
        )

    with pytest.raises(MutationContractError, match="row universe"):
        MutationValueBatch(
            plan=plan,
            field_index=model.field_index,
            rows=RowSelection.all(18),
            buffer=model.buffer,
        )


def test_sim_backend_mutation_extension_is_additive_and_fail_closed() -> None:
    specs, _, _ = _fixtures()
    backend = cast(Any, object())
    assert "bind_mutation_plan" not in SimBackend.__abstractmethods__
    with pytest.raises(NotImplementedError, match="typed backend mutations"):
        SimBackend.bind_mutation_plan(backend, specs)
    assert "get_mutation_capability_manifest" not in SimBackend.__abstractmethods__
    with pytest.raises(NotImplementedError, match="mutation capability manifest"):
        SimBackend.get_mutation_capability_manifest(backend, ExecutionProfile.HOST_NUMPY)


def test_global_mutation_target_binds_without_runtime_selector() -> None:
    target = _target(
        "physics.gravity",
        MutationTargetKind.MODEL_PARAMETER,
        MutationEntityKind.GLOBAL,
        MutationFieldKind.GRAVITY,
        None,
    )
    value_template = _value_template((3,))
    spec = _spec(
        "randomize.gravity",
        target,
        trigger=MutationTrigger.RESET,
        phase=MutationCommitPhase.RESET,
        operation=MutationOperation.SET,
        baseline=MutationBaseline.DEFAULT,
        persistence=MutationPersistence.EPISODE,
        recompute=MutationRecomputeLevel.DYNAMICS,
        value_template=value_template,
    )
    capability = MutationCapability(
        target_key=target.target_key,
        target_kind=target.target_kind,
        entity_kind=target.entity_kind,
        field_kind=target.field_kind,
        entity_count=None,
        value_template=value_template,
        triggers=frozenset({spec.trigger}),
        commit_phases=frozenset({spec.commit_phase}),
        operations=frozenset({spec.operation}),
        baselines=frozenset({spec.baseline}),
        persistences=frozenset({spec.persistence}),
        recompute_levels=frozenset({spec.recompute}),
    )

    def unexpected_selector(_: MutationTargetSpec) -> tuple[int, ...]:
        raise AssertionError("global target must not resolve a selector")

    plan = bind_mutation_plan(
        backend_type="fake",
        backend_instance_id="fake:0",
        num_envs=3,
        specs=(spec,),
        capabilities=(capability,),
        resolve_selector=unexpected_selector,
    )
    assert plan.specs[0].target.entity_ids == ()
    assert plan.specs[0].value_buffer.row_shape == (3,)

    with pytest.raises(MutationContractError, match="cannot declare a selector"):
        replace(target, selector="world")


def test_legacy_warning_and_compiled_strict_boundary(caplog, monkeypatch) -> None:
    class Backend:
        backend_type = "boundary-fixture"

        def __init__(self, capabilities: DomainRandomizationCapabilities) -> None:
            self.capabilities = capabilities
            self.call_count = 0
            self.committed_payloads: list[ResetRandomizationPayload | None] = []

        def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
            return self.capabilities

        def set_state(
            self,
            env_indices: np.ndarray,
            qpos: np.ndarray,
            qvel: np.ndarray,
            randomization: ResetRandomizationPayload | None = None,
        ) -> None:
            self.call_count += 1
            self.committed_payloads.append(randomization)

    class Provider(DomainRandomizationProvider):
        def __init__(self) -> None:
            self.observation_call_count = 0

        def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
            return None

        def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
            return ResetPlan(
                env_ids=env_ids,
                qpos=np.zeros((len(env_ids), 1), dtype=np.float32),
                qvel=np.zeros((len(env_ids), 1), dtype=np.float32),
                info_updates={},
                randomization=ResetRandomizationPayload(
                    base_mass_delta=np.ones((len(env_ids),), dtype=np.float32),
                    kp=np.ones((len(env_ids), 1), dtype=np.float32),
                ),
            )

        def build_reset_observation(
            self, env: Any, env_ids: np.ndarray, info_updates: dict[str, Any]
        ) -> dict[str, np.ndarray]:
            self.observation_call_count += 1
            return {"obs": np.zeros((len(env_ids), 1), dtype=np.float32)}

    filter_calls: dict[int, int] = {}
    filter_reset_payload = DomainRandomizationCapabilities.filter_reset_payload

    def track_filter(
        capabilities: DomainRandomizationCapabilities,
        payload: ResetRandomizationPayload,
    ) -> tuple[ResetRandomizationPayload | None, frozenset[str]]:
        key = id(capabilities)
        filter_calls[key] = filter_calls.get(key, 0) + 1
        return filter_reset_payload(capabilities, payload)

    monkeypatch.setattr(
        DomainRandomizationCapabilities,
        "filter_reset_payload",
        track_filter,
    )

    legacy_capabilities = DomainRandomizationCapabilities(
        supported_reset_terms=frozenset({RESET_TERM_BASE_MASS})
    )
    legacy_backend = Backend(legacy_capabilities)
    legacy_provider = Provider()
    legacy_manager = DomainRandomizationManager(
        SimpleNamespace(_backend=legacy_backend),
        legacy_provider,
        execution_mode=DomainRandomizationExecutionMode.LEGACY_WARN_AND_FILTER,
    )
    env_ids = np.array([0, 1], dtype=np.int32)

    with caplog.at_level(logging.WARNING):
        legacy_manager.reset(env_ids)
        legacy_manager.reset(env_ids)

    assert filter_calls[id(legacy_capabilities)] == 2
    assert legacy_backend.call_count == 2
    assert legacy_provider.observation_call_count == 2
    assert len(legacy_backend.committed_payloads) == 2
    for payload in legacy_backend.committed_payloads:
        assert payload is not None
        assert payload.base_mass_delta is not None
        assert payload.kp is None
    assert (
        caplog.messages.count(
            "boundary-fixture backend does not support reset randomization terms: kp; skipping them."
        )
        == 1
    )

    caplog.clear()
    strict_capabilities = DomainRandomizationCapabilities(
        supported_reset_terms=frozenset({RESET_TERM_BASE_MASS})
    )
    strict_backend = Backend(strict_capabilities)
    strict_provider = Provider()
    strict_manager = DomainRandomizationManager.for_compiled_task(
        SimpleNamespace(_backend=strict_backend), strict_provider
    )

    with caplog.at_level(logging.WARNING):
        with pytest.raises(
            UnsupportedDomainRandomizationError,
            match="does not support compiled reset randomization terms: kp",
        ) as exc_info:
            strict_manager.reset(env_ids)

    assert exc_info.value.backend_type == "boundary-fixture"
    assert exc_info.value.unsupported_terms == frozenset({"kp"})
    assert strict_manager.execution_mode is DomainRandomizationExecutionMode.COMPILED_STRICT
    assert filter_calls.get(id(strict_capabilities), 0) == 0
    assert strict_backend.call_count == 0
    assert strict_backend.committed_payloads == []
    assert strict_provider.observation_call_count == 0
    assert caplog.messages == []

    specs, capabilities, selectors = _fixtures()
    unknown = replace(
        specs[0],
        target=replace(specs[0].target, target_key="physics.body.unknown"),
    )
    with pytest.raises(MutationContractError, match="not supported"):
        _bind(specs=(unknown,), capabilities=capabilities, selectors=selectors)
