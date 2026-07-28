from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass, replace

import pytest

from unilab.base.backend import (
    BackendBatchCounterBudget,
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
    MutationSelectorMode,
    MutationSelectorSpec,
    MutationTargetKind,
    MutationTrigger,
    PhysicalUnit,
    ReferenceFrame,
    StateFieldKind,
)
from unilab.manager import (
    MANAGER_TASK_CONTRACT_VERSION,
    CompiledTaskPlan,
    EntityKind,
    EntitySelector,
    ManagerContractError,
    MutationTemplate,
    NormalizationMode,
    OutputSlice,
    ParameterKind,
    ParameterSpec,
    PolicySpec,
    QuaternionOrder,
    StateRequirement,
    TaskCompiler,
    TaskSpec,
    TensorSpec,
    TermDefinition,
    TermInvocation,
    TermPhase,
    TermRegistry,
    TermRole,
)


class RecordingResolver:
    def __init__(self, ids: dict[str, tuple[int, ...]]) -> None:
        self.ids = ids
        self.calls: list[str] = []

    def resolve(self, selector: EntitySelector) -> tuple[int, ...]:
        self.calls.append(selector.key)
        return self.ids[selector.key]


def _state_tensor(
    shape: tuple[int, ...],
    *,
    frame: ReferenceFrame,
    unit: PhysicalUnit,
) -> TensorSpec:
    return TensorSpec(shape=shape, dtype="float32", frame=frame, unit=unit)


def _output_tensor(
    shape: tuple[int, ...],
    *,
    frame: ReferenceFrame = ReferenceFrame.NONE,
    unit: PhysicalUnit = PhysicalUnit.UNITLESS,
    quaternion_order: QuaternionOrder = QuaternionOrder.NONE,
    dtype: str = "float32",
) -> TensorSpec:
    return TensorSpec(
        shape=shape,
        dtype=dtype,
        frame=frame,
        unit=unit,
        quaternion_order=quaternion_order,
    )


def _control_buffer(action_dim: int = 2) -> BufferContract:
    return BufferContract(
        row_shape=(action_dim,),
        dtype="float32",
        layout=BufferLayout.C_CONTIGUOUS,
        placement=BufferPlacement.host(),
        owner=BufferOwner.MANAGER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.UNTIL_STEP_COMPLETE,
        dlpack_exportable=False,
    )


def _mutation_buffer(shape: tuple[int, ...]) -> BufferContract:
    return BufferContract(
        row_shape=shape,
        dtype="float32",
        layout=BufferLayout.C_CONTIGUOUS,
        placement=BufferPlacement.host(),
        owner=BufferOwner.MANAGER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.UNTIL_COMMIT,
        dlpack_exportable=False,
    )


def _selectors() -> tuple[EntitySelector, EntitySelector]:
    joints = EntitySelector(
        key="robot.policy_joints",
        entity="robot",
        kind=EntityKind.DOF,
        expressions=("left_joint", "right_joint"),
    )
    base = EntitySelector(
        key="robot.base",
        entity="robot",
        kind=EntityKind.BODY,
        expressions=("base",),
    )
    return joints, base


def _definitions() -> tuple[TermDefinition, ...]:
    joints, base = _selectors()
    return (
        TermDefinition(
            key="reset.base_position",
            version="1",
            phase=TermPhase.RESET,
            role=TermRole.EVENT,
            mutation_templates=(
                MutationTemplate(
                    key_suffix="",
                    target_key="state.body.position",
                    target_kind=MutationTargetKind.SIMULATION_STATE,
                    selector=base,
                    field_kind=MutationFieldKind.POSITION,
                    trigger=MutationTrigger.RESET,
                    commit_phase=MutationCommitPhase.RESET,
                    operation=MutationOperation.SET,
                    baseline=MutationBaseline.DEFAULT,
                    persistence=MutationPersistence.EPISODE,
                    recompute=MutationRecomputeLevel.KINEMATICS,
                    value_template=_mutation_buffer((1, 3)),
                ),
            ),
        ),
        TermDefinition(
            key="reward.alive",
            version="2",
            phase=TermPhase.REWARD,
            role=TermRole.REWARD,
            parameters=(ParameterSpec("weight", ParameterKind.FLOAT, required=False, default=1.0),),
            state_requirements=(
                StateRequirement(
                    semantic_key="robot.base.position",
                    selector=base,
                    field_kind=StateFieldKind.POSITION,
                    tensor=_state_tensor(
                        (1, 3), frame=ReferenceFrame.WORLD, unit=PhysicalUnit.METER
                    ),
                ),
            ),
            output=_output_tensor((1,)),
        ),
        TermDefinition(
            key="obs.joint_position",
            version="3",
            phase=TermPhase.OBSERVATION,
            role=TermRole.OBSERVATION,
            parameters=(ParameterSpec("scale", ParameterKind.FLOAT),),
            state_requirements=(
                StateRequirement(
                    semantic_key="robot.joint.position",
                    selector=joints,
                    field_kind=StateFieldKind.POSITION,
                    tensor=_state_tensor(
                        (2,), frame=ReferenceFrame.JOINT, unit=PhysicalUnit.RADIAN
                    ),
                ),
            ),
            output=_output_tensor((2,), frame=ReferenceFrame.JOINT, unit=PhysicalUnit.RADIAN),
        ),
        TermDefinition(
            key="obs.base_quaternion",
            version="1",
            phase=TermPhase.OBSERVATION,
            role=TermRole.OBSERVATION,
            state_requirements=(
                StateRequirement(
                    semantic_key="robot.base.orientation",
                    selector=base,
                    field_kind=StateFieldKind.ORIENTATION,
                    tensor=TensorSpec(
                        shape=(1, 4),
                        dtype="float32",
                        frame=ReferenceFrame.WORLD,
                        unit=PhysicalUnit.QUATERNION,
                        quaternion_order=QuaternionOrder.WXYZ,
                    ),
                ),
            ),
            output=_output_tensor(
                (4,),
                frame=ReferenceFrame.WORLD,
                unit=PhysicalUnit.QUATERNION,
                quaternion_order=QuaternionOrder.WXYZ,
            ),
        ),
    )


def _registry(*, reverse: bool = False) -> TermRegistry:
    registry = TermRegistry()
    definitions = _definitions()
    for definition in reversed(definitions) if reverse else definitions:
        registry.register(definition)
    return registry


def _task(*, reverse: bool = False, joint_scale: float = 0.5) -> TaskSpec:
    terms = [
        TermInvocation.create(key="reset", definition_key="reset.base_position"),
        TermInvocation.create(
            key="alive",
            definition_key="reward.alive",
            dependencies=("reset",),
            parameters={"weight": 0.25},
        ),
        TermInvocation.create(
            key="joint_pos",
            definition_key="obs.joint_position",
            dependencies=("alive",),
            parameters={"scale": joint_scale},
            observation_group="policy",
        ),
        TermInvocation.create(
            key="base_quat",
            definition_key="obs.base_quaternion",
            dependencies=("alive",),
            observation_group="critic",
        ),
    ]
    if reverse:
        terms.reverse()
    return TaskSpec.create(
        key="compiler_fixture",
        terms=terms,
        control=ControlSpec(
            semantic_key="robot.joint.command",
            buffer=_control_buffer(),
            physics_substeps_per_control=4,
        ),
        execution_profile=ExecutionProfile.HOST_NUMPY,
        executor_key="reference.numpy.v1",
        policy=PolicySpec(
            observation_groups=("policy", "critic"),
            action_scale=(0.25,),
            normalization=NormalizationMode.EMPIRICAL,
        ),
        hot_path_budget=BackendBatchCounterBudget(state_materializations=1),
    )


def _capabilities() -> frozenset[str]:
    return frozenset(
        {
            "state.body.position",
            "state.body.orientation",
            "state.dof.position",
        }
    )


def _compile(
    *,
    reverse_registry: bool = False,
    reverse_terms: bool = False,
    joint_scale: float = 0.5,
    ids: dict[str, tuple[int, ...]] | None = None,
) -> tuple[CompiledTaskPlan, RecordingResolver, TermRegistry]:
    registry = _registry(reverse=reverse_registry)
    resolver = RecordingResolver(ids or {"robot.base": (0,), "robot.policy_joints": (3, 7)})
    plan = TaskCompiler(registry).compile(
        _task(reverse=reverse_terms, joint_scale=joint_scale),
        resolver=resolver,
        capabilities=_capabilities(),
    )
    return plan, resolver, registry


def test_compiler_binds_and_freezes_complete_plan() -> None:
    plan, resolver, registry = _compile()

    assert plan.contract_version == MANAGER_TASK_CONTRACT_VERSION
    assert plan.task_key == "compiler_fixture"
    assert plan.executor_key == "reference.numpy.v1"
    assert plan.fingerprint.startswith(f"{MANAGER_TASK_CONTRACT_VERSION}:")
    assert len(plan.fingerprint.rsplit(":", 1)[1]) == 64
    assert [item.key for item in plan.selectors] == ["robot.base", "robot.policy_joints"]
    assert plan.selectors[0].entity_ids == (0,)
    assert plan.selectors[1].entity_ids == (3, 7)
    assert sorted(resolver.calls) == ["robot.base", "robot.policy_joints"]
    assert len(resolver.calls) == 2
    assert registry.lookup_count == 4

    assert [item.key for item in plan.terms] == ["reset", "alive", "base_quat", "joint_pos"]
    assert plan.terms[1].dependency_indices == (0,)
    assert plan.terms[2].dependency_indices == (1,)
    assert plan.terms[3].dependency_indices == (1,)
    assert plan.terms[3].parameters == (("scale", 0.5),)
    assert plan.terms[0].mutation_indices == (0,)
    mutation_selector = plan.mutation_specs[0].target.selector_spec
    assert mutation_selector == MutationSelectorSpec(
        semantic_key="robot.base",
        mode=MutationSelectorMode.EXACT,
        expressions=("base",),
        entity_ids=(0,),
    )

    assert [item.semantic_key for item in plan.backend_io.state_fields] == [
        "robot.base.orientation",
        "robot.base.position",
        "robot.joint.position",
    ]
    assert plan.backend_io.control.semantic_key == "robot.joint.command"
    assert plan.backend_io.hot_path_budget == BackendBatchCounterBudget(state_materializations=1)
    assert plan.backend_io.reset_hot_path_budget is None
    assert plan.required_capabilities == (
        "state.body.orientation",
        "state.body.position",
        "state.dof.position",
    )

    assert [item.key for item in plan.policy_abi.observation_groups] == ["policy", "critic"]
    assert [item.width for item in plan.policy_abi.observation_groups] == [2, 4]
    assert plan.policy_abi.action_dim == 2
    assert plan.policy_abi.action_scale == (0.25, 0.25)
    assert plan.policy_abi.normalization is NormalizationMode.EMPIRICAL
    assert plan.policy_abi.quaternion_outputs == (("base_quat", QuaternionOrder.WXYZ),)
    assert f"policy_abi={plan.policy_abi.fingerprint}" in plan.diagnostic_signature
    assert plan.diagnostic_signature[-1] == (
        f"selector_binding={plan.selector_binding_fingerprint}"
    )
    assert [item.key for item in plan.output_channels] == [
        "obs:critic",
        "obs:policy",
        "reward",
    ]
    assert all(item.buffer.owner is BufferOwner.RUNTIME for item in plan.output_channels)
    assert all(item.buffer.lifetime is BufferLifetime.PLAN for item in plan.output_channels)

    with pytest.raises(FrozenInstanceError):
        plan.task_key = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        plan.terms[0] = plan.terms[0]  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        plan.selectors[0].entity_ids = (99,)  # type: ignore[misc]

    def assert_deeply_immutable(value: object) -> None:
        assert not isinstance(value, (dict, list, set))
        if is_dataclass(value) and not isinstance(value, type):
            for item in fields(value):
                assert_deeply_immutable(getattr(value, item.name))
        elif isinstance(value, (tuple, frozenset)):
            for item in value:
                assert_deeply_immutable(item)

    assert_deeply_immutable(plan)


def test_compiler_fingerprint_is_canonical_and_sensitive() -> None:
    plan, _, _ = _compile()
    reordered, _, _ = _compile(reverse_registry=True, reverse_terms=True)
    changed_parameter, _, _ = _compile(joint_scale=0.75)
    changed_binding, _, _ = _compile(ids={"robot.base": (2,), "robot.policy_joints": (4, 8)})

    assert reordered == plan
    assert reordered.fingerprint == plan.fingerprint
    assert changed_parameter.fingerprint != plan.fingerprint
    assert changed_parameter.policy_abi.fingerprint == plan.policy_abi.fingerprint
    assert changed_binding.fingerprint == plan.fingerprint
    assert changed_binding.selector_binding_fingerprint != plan.selector_binding_fingerprint


def test_compiled_mutation_selector_must_match_its_immutable_selector_binding() -> None:
    plan, _, _ = _compile()
    mutation = plan.mutation_specs[0]
    selector = mutation.target.selector_spec
    assert selector is not None

    wrong_expression = replace(
        mutation,
        target=replace(
            mutation.target,
            selector=replace(selector, expressions=("wrong_base",)),
        ),
    )
    with pytest.raises(ManagerContractError, match="does not match its selector binding"):
        replace(plan, mutation_specs=(wrong_expression,))

    wrong_semantic_key = replace(
        mutation,
        target=replace(
            mutation.target,
            selector=replace(selector, semantic_key="not.a.compiled.selector"),
        ),
    )
    with pytest.raises(ManagerContractError, match="does not reference a compiled selector"):
        replace(plan, mutation_specs=(wrong_semantic_key,))


def test_registry_rejects_duplicate_and_post_freeze_registration() -> None:
    registry = TermRegistry()
    definition = _definitions()[0]
    registry.register(definition)
    with pytest.raises(ManagerContractError, match="already registered"):
        registry.register(definition)
    registry.freeze()
    with pytest.raises(ManagerContractError, match="frozen"):
        registry.register(_definitions()[1])


@pytest.mark.parametrize(
    ("terms", "message"),
    [
        (
            (
                TermInvocation.create(
                    key="a", definition_key="reward.alive", dependencies=("missing",)
                ),
            ),
            "unknown term",
        ),
        (
            (
                TermInvocation.create(key="a", definition_key="reward.alive", dependencies=("b",)),
                TermInvocation.create(key="b", definition_key="reward.alive", dependencies=("a",)),
            ),
            "cycle",
        ),
        (
            (
                TermInvocation.create(
                    key="early", definition_key="reset.base_position", dependencies=("late",)
                ),
                TermInvocation.create(
                    key="late",
                    definition_key="obs.joint_position",
                    parameters={"scale": 1.0},
                    observation_group="policy",
                ),
            ),
            "later phase",
        ),
    ],
)
def test_compiler_rejects_invalid_dependency_graph(
    terms: tuple[TermInvocation, ...], message: str
) -> None:
    task = replace(_task(), terms=terms, policy=PolicySpec(("policy",), (1.0,)))
    with pytest.raises(ManagerContractError, match=message):
        TaskCompiler(_registry()).compile(
            task,
            resolver=RecordingResolver({"robot.base": (0,), "robot.policy_joints": (3, 7)}),
            capabilities=_capabilities(),
        )


def test_compiler_rejects_unknown_definition_and_parameter_contract_mismatch() -> None:
    unknown = replace(
        _task(),
        terms=(TermInvocation.create(key="bad", definition_key="missing"),),
    )
    with pytest.raises(ManagerContractError, match="not registered"):
        TaskCompiler(_registry()).compile(
            unknown,
            resolver=RecordingResolver({}),
            capabilities=frozenset(),
        )

    missing_parameter = replace(
        _task(),
        terms=(
            TermInvocation.create(
                key="joint_pos",
                definition_key="obs.joint_position",
                observation_group="policy",
            ),
        ),
        policy=PolicySpec(("policy",), (1.0,)),
    )
    with pytest.raises(ManagerContractError, match="missing required parameter"):
        TaskCompiler(_registry()).compile(
            missing_parameter,
            resolver=RecordingResolver({"robot.policy_joints": (3, 7)}),
            capabilities=_capabilities(),
        )


@pytest.mark.parametrize(
    ("resolved", "message"),
    [
        ({"robot.base": (), "robot.policy_joints": (3, 7)}, "at least one entity id"),
        ({"robot.base": (0,), "robot.policy_joints": (3, 3)}, "must be unique"),
        ({"robot.base": (0,), "robot.policy_joints": (3,)}, "entity axis"),
    ],
)
def test_compiler_rejects_invalid_or_shape_mismatched_selector_binding(
    resolved: dict[str, tuple[int, ...]], message: str
) -> None:
    with pytest.raises(ManagerContractError, match=message):
        TaskCompiler(_registry()).compile(
            _task(),
            resolver=RecordingResolver(resolved),
            capabilities=_capabilities(),
        )


def test_compiler_rejects_non_tuple_resolver_output() -> None:
    class BadResolver:
        def resolve(self, selector: EntitySelector) -> tuple[int, ...]:
            return [0]  # type: ignore[return-value]

    with pytest.raises(ManagerContractError, match="must return a tuple"):
        TaskCompiler(_registry()).compile(
            _task(), resolver=BadResolver(), capabilities=_capabilities()
        )


def test_compiler_reports_missing_entity_at_manager_boundary() -> None:
    class MissingEntityResolver:
        def resolve(self, selector: EntitySelector) -> tuple[int, ...]:
            raise KeyError(selector.expressions[0])

    with pytest.raises(ManagerContractError, match="failed to resolve selector 'robot.base'"):
        TaskCompiler(_registry()).compile(
            _task(), resolver=MissingEntityResolver(), capabilities=_capabilities()
        )


def test_compiler_rejects_missing_capability_before_backend_binding() -> None:
    resolver = RecordingResolver({"robot.base": (0,), "robot.policy_joints": (3, 7)})
    with pytest.raises(ManagerContractError, match="unsupported capabilities") as exc_info:
        TaskCompiler(_registry()).compile(
            _task(),
            resolver=resolver,
            capabilities=frozenset({"state.body.position"}),
        )
    message = str(exc_info.value)
    assert "state.body.orientation" in message
    assert "state.dof.position" in message
    assert resolver.calls == []


def test_compiler_propagates_explicit_device_placement_without_cuda_zero_assumption() -> None:
    placement = BufferPlacement.device("cuda", 3)
    control = BufferContract(
        row_shape=(2,),
        dtype="float32",
        layout=BufferLayout.C_CONTIGUOUS,
        placement=placement,
        owner=BufferOwner.RUNNER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.UNTIL_STEP_COMPLETE,
        dlpack_exportable=True,
    )
    task = TaskSpec.create(
        key="device_placement_fixture",
        terms=(
            TermInvocation.create(
                key="joint_pos",
                definition_key="obs.joint_position",
                parameters={"scale": 1.0},
                observation_group="policy",
            ),
            TermInvocation.create(
                key="base_quat",
                definition_key="obs.base_quaternion",
                observation_group="critic",
            ),
        ),
        control=ControlSpec("robot.joint.command", control),
        execution_profile=ExecutionProfile.DEVICE_RESIDENT,
        executor_key="device.fused.v1",
        policy=PolicySpec(("policy", "critic"), (1.0,)),
    )
    plan = TaskCompiler(_registry()).compile(
        task,
        resolver=RecordingResolver({"robot.base": (0,), "robot.policy_joints": (3, 7)}),
        capabilities=frozenset({"state.body.orientation", "state.dof.position"}),
    )

    assert all(item.buffer.placement == placement for item in plan.backend_io.state_fields)
    assert all(item.buffer.placement == placement for item in plan.output_channels)
    assert all(item.buffer.dlpack_exportable for item in plan.backend_io.state_fields)
    assert all(item.buffer.dlpack_exportable for item in plan.output_channels)


def test_compiler_rejects_mutation_value_on_different_placement() -> None:
    placement = BufferPlacement.device("cuda", 2)
    control = replace(
        _control_buffer(),
        placement=placement,
        dlpack_exportable=True,
    )
    task = replace(
        _task(),
        control=ControlSpec("robot.joint.command", control, physics_substeps_per_control=4),
        execution_profile=ExecutionProfile.DEVICE_RESIDENT,
    )
    with pytest.raises(ManagerContractError, match="placement does not match"):
        TaskCompiler(_registry()).compile(
            task,
            resolver=RecordingResolver({"robot.base": (0,), "robot.policy_joints": (3, 7)}),
            capabilities=_capabilities(),
        )


def test_compiler_rejects_conflicting_mutation_writes() -> None:
    reset_definition = _definitions()[0]
    duplicate = replace(reset_definition, key="reset.base_position.duplicate")
    registry = _registry()
    registry.register(duplicate)
    task = replace(
        _task(),
        terms=(
            TermInvocation.create(key="reset", definition_key=reset_definition.key),
            TermInvocation.create(key="reset_b", definition_key=duplicate.key),
            *_task().terms[1:],
        ),
    )
    with pytest.raises(ManagerContractError, match="conflict at one commit barrier"):
        TaskCompiler(registry).compile(
            task,
            resolver=RecordingResolver({"robot.base": (0,), "robot.policy_joints": (3, 7)}),
            capabilities=_capabilities(),
        )


def test_compiled_plan_rejects_overlapping_output_slices() -> None:
    plan, _, _ = _compile()
    joint_index = plan.term_index("joint_pos")
    joint = plan.terms[joint_index]
    assert joint.output is not None
    overlapping = replace(
        joint,
        output=OutputSlice(
            channel="obs:critic",
            start=0,
            stop=2,
            tensor=joint.output.tensor,
        ),
    )
    terms = list(plan.terms)
    terms[joint_index] = overlapping
    with pytest.raises(ManagerContractError, match="overlap"):
        replace(plan, terms=tuple(terms))


def test_compiled_plan_rejects_policy_abi_drift_from_terms_and_control() -> None:
    plan, _, _ = _compile()
    with pytest.raises(ManagerContractError, match="action does not match"):
        replace(plan, policy_abi=replace(plan.policy_abi, action_key="other.command"))

    policy_group = plan.policy_abi.observation_groups[0]
    output = policy_group.outputs[0]
    wrong_term_index = plan.term_index("base_quat")
    changed_group = replace(
        policy_group,
        outputs=(replace(output, term_index=wrong_term_index),),
    )
    changed_abi = replace(
        plan.policy_abi,
        observation_groups=(changed_group, *plan.policy_abi.observation_groups[1:]),
    )
    with pytest.raises(ManagerContractError, match="does not match its compiled term"):
        replace(plan, policy_abi=changed_abi)


def test_compiler_rejects_inconsistent_state_and_selector_semantics() -> None:
    definitions = _definitions()
    base_requirement = definitions[1].state_requirements[0]

    incompatible_state = replace(
        definitions[1],
        key="reward.incompatible_state",
        state_requirements=(
            replace(
                base_requirement,
                tensor=_state_tensor((1, 4), frame=ReferenceFrame.WORLD, unit=PhysicalUnit.METER),
            ),
        ),
    )
    registry = _registry()
    registry.register(incompatible_state)
    task = replace(
        _task(),
        terms=(
            *_task().terms,
            TermInvocation.create(
                key="bad_state",
                definition_key=incompatible_state.key,
                dependencies=("reset",),
            ),
        ),
    )
    with pytest.raises(ManagerContractError, match="incompatible declarations"):
        TaskCompiler(registry).compile(
            task,
            resolver=RecordingResolver({"robot.base": (0,), "robot.policy_joints": (3, 7)}),
            capabilities=_capabilities(),
        )

    inconsistent_selector = replace(
        definitions[1],
        key="reward.inconsistent_selector",
        state_requirements=(
            replace(
                base_requirement,
                selector=replace(
                    base_requirement.selector,
                    expressions=("different_base",),
                ),
            ),
        ),
    )
    registry = _registry()
    registry.register(inconsistent_selector)
    task = replace(
        _task(),
        terms=(
            *_task().terms,
            TermInvocation.create(
                key="bad_selector",
                definition_key=inconsistent_selector.key,
                dependencies=("reset",),
            ),
        ),
    )
    with pytest.raises(ManagerContractError, match="inconsistent declarations"):
        TaskCompiler(registry).compile(
            task,
            resolver=RecordingResolver({"robot.base": (0,), "robot.policy_joints": (3, 7)}),
            capabilities=_capabilities(),
        )


def test_term_definition_rejects_role_phase_mismatch() -> None:
    with pytest.raises(ManagerContractError, match="cannot run in phase"):
        TermDefinition(
            key="invalid.reward",
            version="1",
            phase=TermPhase.OBSERVATION,
            role=TermRole.REWARD,
            output=_output_tensor((1,)),
        )


def test_compiler_rejects_policy_abi_group_action_and_dtype_mismatch() -> None:
    missing_group = replace(_task(), policy=PolicySpec(("policy",), (0.25,)))
    with pytest.raises(ManagerContractError, match="groups do not match"):
        TaskCompiler(_registry()).compile(
            missing_group,
            resolver=RecordingResolver({"robot.base": (0,), "robot.policy_joints": (3, 7)}),
            capabilities=_capabilities(),
        )

    bad_scale = replace(_task(), policy=PolicySpec(("policy", "critic"), (1.0, 2.0, 3.0)))
    with pytest.raises(ManagerContractError, match="action_scale"):
        TaskCompiler(_registry()).compile(
            bad_scale,
            resolver=RecordingResolver({"robot.base": (0,), "robot.policy_joints": (3, 7)}),
            capabilities=_capabilities(),
        )

    registry = _registry()
    joint_definition = registry.resolve("obs.joint_position")
    registry = TermRegistry()
    for definition in _definitions():
        registry.register(
            replace(definition, output=_output_tensor((2,), dtype="float64"))
            if definition == joint_definition
            else definition
        )
    mixed = replace(
        _task(),
        terms=tuple(
            replace(term, observation_group="critic") if term.key == "joint_pos" else term
            for term in _task().terms
        ),
        policy=PolicySpec(("critic",), (0.25,)),
    )
    with pytest.raises(ManagerContractError, match="mixes output dtypes"):
        TaskCompiler(registry).compile(
            mixed,
            resolver=RecordingResolver({"robot.base": (0,), "robot.policy_joints": (3, 7)}),
            capabilities=_capabilities(),
        )


def test_builder_copies_mutable_config_inputs() -> None:
    dependencies = ["alive"]
    parameters: dict[str, object] = {"scale": [0.5, 1.0]}
    invocation = TermInvocation.create(
        key="obs",
        definition_key="obs.joint_position",
        dependencies=dependencies,
        parameters=parameters,
        observation_group="policy",
    )
    terms = [invocation]
    task = TaskSpec.create(
        key="copy_fixture",
        terms=terms,
        control=ControlSpec("control", _control_buffer()),
        execution_profile=ExecutionProfile.HOST_NUMPY,
        executor_key="reference",
        policy=PolicySpec(("policy",), (1.0,)),
    )

    dependencies.append("later")
    parameters["scale"] = [9.0]
    terms.clear()

    assert invocation.dependencies == ("alive",)
    assert invocation.parameters == (("scale", (0.5, 1.0)),)
    assert task.terms == (invocation,)
