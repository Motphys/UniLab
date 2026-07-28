from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
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
    ExternalWrenchMutationBatch,
    ModelParameterMutationBatch,
    MutationBaseline,
    MutationCapability,
    MutationCommitPhase,
    MutationContractError,
    MutationEntityKind,
    MutationFieldKind,
    MutationOperation,
    MutationPersistence,
    MutationRecomputeLevel,
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
        assert target.selector is not None
        key = (target.entity_kind, target.selector)
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
