from __future__ import annotations

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
    PhysicalUnit,
    ReferenceFrame,
    StateEntityKind,
    StateFieldKind,
)
from unilab.manager import (
    EntityKind,
    EntitySelector,
    MutationTemplate,
    PolicySpec,
    SelectorMode,
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


class FixtureResolver:
    _ids = {
        "robot.arm_joints": (4, 8, 12),
        "robot.gripper_site": (9,),
        "object.body": (20,),
        "object.collision_geoms": (30, 31),
    }

    def resolve(self, selector: EntitySelector) -> tuple[int, ...]:
        return self._ids[selector.key]


def _buffer(
    shape: tuple[int, ...], *, lifetime: BufferLifetime = BufferLifetime.UNTIL_COMMIT
) -> BufferContract:
    return BufferContract(
        row_shape=shape,
        dtype="float32",
        layout=BufferLayout.C_CONTIGUOUS,
        placement=BufferPlacement.host(),
        owner=BufferOwner.MANAGER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=lifetime,
        dlpack_exportable=False,
    )


def test_multi_entity_manipulation_fixture_compiles() -> None:
    arm_joints = EntitySelector(
        key="robot.arm_joints",
        entity="robot",
        kind=EntityKind.JOINT,
        expressions=("shoulder", "elbow", "wrist"),
    )
    gripper_site = EntitySelector(
        key="robot.gripper_site",
        entity="robot",
        kind=EntityKind.SITE,
        expressions=("grasp_site",),
    )
    object_body = EntitySelector(
        key="object.body",
        entity="object",
        kind=EntityKind.BODY,
        expressions=("cube",),
    )
    object_geoms = EntitySelector(
        key="object.collision_geoms",
        entity="object",
        kind=EntityKind.GEOM,
        expressions=("cube_collision.*",),
        mode=SelectorMode.REGEX,
    )

    registry = TermRegistry()
    registry.register(
        TermDefinition(
            key="manipulation.joint_position",
            version="1",
            phase=TermPhase.OBSERVATION,
            role=TermRole.OBSERVATION,
            state_requirements=(
                StateRequirement(
                    semantic_key="robot.arm.position",
                    selector=arm_joints,
                    field_kind=StateFieldKind.POSITION,
                    tensor=TensorSpec((3,), "float32", ReferenceFrame.JOINT, PhysicalUnit.RADIAN),
                ),
            ),
            output=TensorSpec((3,), "float32", ReferenceFrame.JOINT, PhysicalUnit.RADIAN),
        )
    )
    registry.register(
        TermDefinition(
            key="manipulation.site_position",
            version="1",
            phase=TermPhase.OBSERVATION,
            role=TermRole.OBSERVATION,
            state_requirements=(
                StateRequirement(
                    semantic_key="robot.gripper.position",
                    selector=gripper_site,
                    field_kind=StateFieldKind.POSITION,
                    tensor=TensorSpec((1, 3), "float32", ReferenceFrame.WORLD, PhysicalUnit.METER),
                ),
            ),
            output=TensorSpec((3,), "float32", ReferenceFrame.WORLD, PhysicalUnit.METER),
        )
    )
    registry.register(
        TermDefinition(
            key="manipulation.object_position",
            version="1",
            phase=TermPhase.OBSERVATION,
            role=TermRole.OBSERVATION,
            state_requirements=(
                StateRequirement(
                    semantic_key="object.position",
                    selector=object_body,
                    field_kind=StateFieldKind.POSITION,
                    tensor=TensorSpec((1, 3), "float32", ReferenceFrame.WORLD, PhysicalUnit.METER),
                ),
            ),
            output=TensorSpec((3,), "float32", ReferenceFrame.WORLD, PhysicalUnit.METER),
        )
    )
    registry.register(
        TermDefinition(
            key="manipulation.reset_object",
            version="1",
            phase=TermPhase.RESET,
            role=TermRole.EVENT,
            mutation_templates=(
                MutationTemplate(
                    key_suffix="pose",
                    target_key="state.body.position",
                    target_kind=MutationTargetKind.SIMULATION_STATE,
                    selector=object_body,
                    field_kind=MutationFieldKind.POSITION,
                    trigger=MutationTrigger.RESET,
                    commit_phase=MutationCommitPhase.RESET,
                    operation=MutationOperation.SET,
                    baseline=MutationBaseline.DEFAULT,
                    persistence=MutationPersistence.EPISODE,
                    recompute=MutationRecomputeLevel.KINEMATICS,
                    value_template=_buffer((1, 3)),
                ),
                MutationTemplate(
                    key_suffix="friction",
                    target_key="physics.geom.friction",
                    target_kind=MutationTargetKind.MODEL_PARAMETER,
                    selector=object_geoms,
                    field_kind=MutationFieldKind.FRICTION,
                    trigger=MutationTrigger.RESET,
                    commit_phase=MutationCommitPhase.RESET,
                    operation=MutationOperation.SCALE,
                    baseline=MutationBaseline.DEFAULT,
                    persistence=MutationPersistence.EPISODE,
                    recompute=MutationRecomputeLevel.NONE,
                    value_template=_buffer((2, 3)),
                ),
            ),
        )
    )

    terms = (
        TermInvocation.create(key="reset_object", definition_key="manipulation.reset_object"),
        TermInvocation.create(
            key="arm_joint_pos",
            definition_key="manipulation.joint_position",
            dependencies=("reset_object",),
            observation_group="policy",
        ),
        TermInvocation.create(
            key="gripper_site_pos",
            definition_key="manipulation.site_position",
            dependencies=("reset_object",),
            observation_group="policy",
        ),
        TermInvocation.create(
            key="object_pos",
            definition_key="manipulation.object_position",
            dependencies=("reset_object",),
            observation_group="policy",
        ),
    )
    task = TaskSpec.create(
        key="multi_entity_manipulation_fixture",
        terms=terms,
        control=ControlSpec(
            "robot.arm.command",
            _buffer((7,), lifetime=BufferLifetime.UNTIL_STEP_COMPLETE),
        ),
        execution_profile=ExecutionProfile.HOST_NUMPY,
        executor_key="reference.numpy.v1",
        policy=PolicySpec(("policy",), (0.1,)),
    )
    capabilities = frozenset(
        {
            "state.joint.position",
            "state.site.position",
            "state.body.position",
            "physics.geom.friction",
        }
    )

    plan = TaskCompiler(registry).compile(
        task,
        resolver=FixtureResolver(),
        capabilities=capabilities,
    )

    assert plan.task_key == "multi_entity_manipulation_fixture"
    assert {(item.entity, item.kind) for item in plan.selectors} == {
        ("robot", EntityKind.JOINT),
        ("robot", EntityKind.SITE),
        ("object", EntityKind.BODY),
        ("object", EntityKind.GEOM),
    }
    assert plan.selectors[1].key == "object.collision_geoms"
    assert plan.selectors[1].mode is SelectorMode.REGEX
    assert {item.identity.entity_kind for item in plan.backend_io.state_fields} == {
        StateEntityKind.JOINT,
        StateEntityKind.SITE,
        StateEntityKind.BODY,
    }
    assert {item.target.selector for item in plan.mutation_specs} == {
        "object.body",
        "object.collision_geoms",
    }
    assert plan.policy_abi.observation_groups[0].width == 9
    assert plan.policy_abi.action_dim == 7
    assert plan.policy_abi.action_scale == (0.1,) * 7
    assert plan.required_capabilities == tuple(sorted(capabilities))
    assert "g1" not in repr(plan).lower()
