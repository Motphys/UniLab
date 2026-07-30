from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from unilab.base.backend import (
    BufferContract,
    BufferLayout,
    BufferLifetime,
    BufferMutability,
    BufferOwner,
    BufferPlacement,
    ControlSpec,
    ExecutionProfile,
    MutationBaseline,
    MutationCommitPhase,
    MutationFieldKind,
    MutationOperation,
    MutationPersistence,
    MutationRecomputeLevel,
    MutationTargetKind,
    MutationTrigger,
    StateFieldKind,
)
from unilab.dr.keyed_rng import (
    KEYED_RNG_ALGORITHM,
    RandomCorrelation,
    RandomDistribution,
)
from unilab.manager import (
    EntityKind,
    EntitySelector,
    ManagerContractError,
    MutationRandomization,
    MutationTemplate,
    PolicySpec,
    StateRequirement,
    TaskCompiler,
    TaskSpec,
    TensorSpec,
    TermDefinition,
    TermInvocation,
    TermPhase,
    TermRegistry,
    TermRole,
    validate_compiled_plan_fingerprints,
)
from unilab.manager.fingerprint import canonical_digest, compiled_plan_payload


class _Resolver:
    def __init__(self, actuator_ids: tuple[int, ...] = (2, 4)) -> None:
        self._actuator_ids = actuator_ids

    def resolve(self, selector: EntitySelector) -> tuple[int, ...]:
        if selector.kind is EntityKind.ACTUATOR:
            return self._actuator_ids
        return (0,)


def _buffer(
    row_shape: tuple[int, ...],
    *,
    placement: BufferPlacement | None = None,
    owner: BufferOwner = BufferOwner.MANAGER,
    lifetime: BufferLifetime = BufferLifetime.UNTIL_COMMIT,
) -> BufferContract:
    resolved = placement or BufferPlacement.device("cuda", 0)
    return BufferContract(
        row_shape=row_shape,
        dtype="float32",
        layout=BufferLayout.C_CONTIGUOUS,
        placement=resolved,
        owner=owner,
        mutability=BufferMutability.READ_ONLY,
        lifetime=lifetime,
        dlpack_exportable=resolved.device_type == "cuda",
        address_stable=resolved.device_type == "cuda",
    )


def _compile(
    *,
    parameters: tuple[float, float] = (0.9, 1.1),
    correlation: RandomCorrelation = RandomCorrelation.PER_ENV,
    version: str = "3",
    actuator_ids: tuple[int, ...] = (2, 4),
    random: bool = True,
    role: TermRole = TermRole.EVENT,
    phase: TermPhase = TermPhase.RESET,
    trigger: MutationTrigger = MutationTrigger.RESET,
    commit_phase: MutationCommitPhase = MutationCommitPhase.RESET,
    event_placement: BufferPlacement | None = None,
):
    actuators = EntitySelector(
        key="robot.actuators",
        entity="robot",
        kind=EntityKind.ACTUATOR,
        expressions=("left", "right"),
    )
    root = EntitySelector(
        key="robot.root",
        entity="robot",
        kind=EntityKind.ROOT,
        expressions=("base",),
    )
    registry = TermRegistry()
    registry.register(
        TermDefinition(
            key="event.randomize_kp",
            version=version,
            phase=phase,
            role=role,
            mutation_templates=(
                MutationTemplate(
                    key_suffix="",
                    target_key="actuator.pd_stiffness",
                    target_kind=MutationTargetKind.MODEL_PARAMETER,
                    selector=actuators,
                    field_kind=MutationFieldKind.STIFFNESS,
                    trigger=trigger,
                    commit_phase=commit_phase,
                    operation=MutationOperation.SCALE,
                    baseline=MutationBaseline.DEFAULT,
                    persistence=MutationPersistence.EPISODE,
                    recompute=MutationRecomputeLevel.NONE,
                    value_template=_buffer((1,), placement=event_placement),
                    randomization=(
                        MutationRandomization(
                            distribution=RandomDistribution.UNIFORM,
                            parameters=parameters,
                            correlation=correlation,
                        )
                        if random
                        else None
                    ),
                ),
            ),
        )
    )
    registry.register(
        TermDefinition(
            key="obs.root",
            version="1",
            phase=TermPhase.OBSERVATION,
            role=TermRole.OBSERVATION,
            state_requirements=(
                StateRequirement(
                    semantic_key="robot.root.position",
                    selector=root,
                    field_kind=StateFieldKind.POSITION,
                    tensor=TensorSpec((3,), "float32"),
                    entity_axis=None,
                ),
            ),
            output=TensorSpec((3,), "float32"),
        )
    )
    task = TaskSpec.create(
        key="compiled_event_fixture",
        terms=(
            TermInvocation.create(key="randomize_kp", definition_key="event.randomize_kp"),
            TermInvocation.create(
                key="observation",
                definition_key="obs.root",
                dependencies=("randomize_kp",),
                observation_group="policy",
            ),
        ),
        control=ControlSpec(
            semantic_key="robot.control",
            buffer=_buffer(
                (2,),
                owner=BufferOwner.RUNNER,
                lifetime=BufferLifetime.UNTIL_STEP_COMPLETE,
            ),
            physics_substeps_per_control=1,
        ),
        execution_profile=ExecutionProfile.DEVICE_RESIDENT,
        executor_key="device.fixture.v1",
        policy=PolicySpec(("policy",), (1.0, 1.0)),
    )
    return TaskCompiler(registry).compile(
        task,
        resolver=_Resolver(actuator_ids),
        capabilities=frozenset({"actuator.pd_stiffness", "state.root.position"}),
    )


def test_compiler_freezes_canonical_random_event_metadata() -> None:
    plan = _compile()

    assert len(plan.mutation_events) == 1
    event = plan.mutation_events[0]
    assert event.mutation_index == 0
    assert event.term_index == plan.term_index("randomize_kp")
    assert event.term_key == "randomize_kp"
    assert event.term_version == "3"
    assert event.trigger is MutationTrigger.RESET
    assert event.commit_phase is MutationCommitPhase.RESET
    assert event.distribution is RandomDistribution.UNIFORM
    assert event.parameters == (0.9, 1.1)
    assert event.correlation is RandomCorrelation.PER_ENV
    assert event.algorithm == KEYED_RNG_ALGORITHM
    validate_compiled_plan_fingerprints(plan)

    with pytest.raises(FrozenInstanceError):
        event.term_version = "4"  # type: ignore[misc]


def test_event_semantics_affect_only_the_semantic_fingerprint() -> None:
    original = _compile()
    changed_range = _compile(parameters=(0.8, 1.2))
    changed_correlation = _compile(correlation=RandomCorrelation.PER_ELEMENT)
    changed_version = _compile(version="4")
    changed_binding = _compile(actuator_ids=(7, 9))

    assert changed_range.fingerprint != original.fingerprint
    assert changed_correlation.fingerprint != original.fingerprint
    assert changed_version.fingerprint != original.fingerprint
    assert changed_binding.fingerprint == original.fingerprint
    assert changed_binding.selector_binding_fingerprint != original.selector_binding_fingerprint

    event = replace(original.mutation_events[0], algorithm="future-keyed-rng-v2")
    forged = replace(original, mutation_events=(event,))
    original_digest = canonical_digest(compiled_plan_payload(original, include_bindings=False))
    forged_digest = canonical_digest(compiled_plan_payload(forged, include_bindings=False))
    assert forged_digest != original_digest
    with pytest.raises(ManagerContractError, match="fingerprints do not match"):
        validate_compiled_plan_fingerprints(forged)


def test_non_random_plan_keeps_the_legacy_payload_shape() -> None:
    plan = _compile(random=False)

    assert plan.mutation_events == ()
    assert "mutation_events" not in compiled_plan_payload(plan, include_bindings=False)
    validate_compiled_plan_fingerprints(plan)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"role": TermRole.REWARD, "phase": TermPhase.REWARD}, "Event term"),
        ({"phase": TermPhase.STARTUP}, "reset trigger and phase"),
        ({"trigger": MutationTrigger.STARTUP}, "reset trigger and phase"),
        (
            {"commit_phase": MutationCommitPhase.PRE_PHYSICS},
            "reset trigger and phase",
        ),
        ({"event_placement": BufferPlacement.host()}, "stable manager-owned CUDA"),
    ),
)
def test_compiler_rejects_unsupported_random_event_contracts(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ManagerContractError, match=message):
        _compile(**overrides)  # type: ignore[arg-type]


def test_plan_rejects_duplicate_dangling_and_removed_event_descriptors() -> None:
    plan = _compile()
    event = plan.mutation_events[0]

    with pytest.raises(ManagerContractError, match="unique canonical mutation order"):
        replace(plan, mutation_events=(event, event))
    with pytest.raises(ManagerContractError, match="unknown mutation"):
        replace(plan, mutation_events=(replace(event, mutation_index=4),))

    removed = replace(plan, mutation_events=())
    with pytest.raises(ManagerContractError, match="fingerprints do not match"):
        validate_compiled_plan_fingerprints(removed)


@pytest.mark.parametrize(
    "parameters",
    ((1.1, 0.9), (float("nan"), 1.0), (0.9,)),
)
def test_randomization_descriptor_rejects_invalid_ranges(
    parameters: tuple[float, float],
) -> None:
    with pytest.raises(ManagerContractError, match="invalid mutation randomization"):
        MutationRandomization(
            distribution=RandomDistribution.UNIFORM,
            parameters=parameters,
            correlation=RandomCorrelation.PER_ENV,
        )
