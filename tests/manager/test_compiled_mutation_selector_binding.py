"""Real backend coverage for manager-lowered mutation selector metadata.

The task plan is compiled once from static G1 scene semantics.  Both physics
backends then consume the same immutable plan; neither receives a manager
registry or uses the semantic selector key as a raw model name.
"""

from __future__ import annotations

import numpy as np
import pytest

from unilab.base.backend import (
    BufferContract,
    BufferLayout,
    BufferLifetime,
    BufferMutability,
    BufferOwner,
    BufferPlacement,
    BufferView,
    ControlSpec,
    ExecutionProfile,
    MutationBaseline,
    MutationCommitPhase,
    MutationEntityKind,
    MutationFieldKind,
    MutationOperation,
    MutationPersistence,
    MutationRecomputeLevel,
    MutationTargetKind,
    MutationTrigger,
    MutationValueBatch,
    PhysicalUnit,
    ReferenceFrame,
    RowSelection,
    SimulationStateMutationBatch,
    StateFieldKind,
    TypedBackendMutationBatch,
    create_backend,
)
from unilab.base.backend.mjwarp.dependencies import load_mjwarp_dependencies
from unilab.base.backend.mujoco.backend import MuJoCoBackend
from unilab.base.scene import SceneCfg
from unilab.manager import (
    EntityKind,
    EntitySelector,
    MutationTemplate,
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

pytestmark = pytest.mark.slow

_BASE_NAME = "pelvis"
_HINGE_NAME = "left_hip_pitch_joint"
_ACTION_DIM = 29


class _StaticG1Resolver:
    """Cold scene metadata fixture; it deliberately owns no backend instance."""

    _ids = {
        "robot.root_state": (1,),
        "robot.left_hip_pitch": (0,),
    }

    def __init__(self) -> None:
        self.calls: list[str] = []

    def resolve(self, selector: EntitySelector) -> tuple[int, ...]:
        self.calls.append(selector.key)
        return self._ids[selector.key]


def _manager_buffer(*, row_shape: tuple[int, ...], lifetime: BufferLifetime) -> BufferContract:
    return BufferContract(
        row_shape=row_shape,
        dtype="float32",
        layout=BufferLayout.C_CONTIGUOUS,
        placement=BufferPlacement.host(),
        owner=BufferOwner.MANAGER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=lifetime,
        dlpack_exportable=False,
    )


def _compiled_g1_full_reset_plan():
    root = EntitySelector(
        key="robot.root_state",
        entity="robot",
        kind=EntityKind.ROOT,
        expressions=(_BASE_NAME,),
    )
    hinge = EntitySelector(
        key="robot.left_hip_pitch",
        entity="robot",
        kind=EntityKind.DOF,
        expressions=(_HINGE_NAME,),
    )
    registry = TermRegistry()
    registry.register(
        TermDefinition(
            key="reset.full_state",
            version="1",
            phase=TermPhase.RESET,
            role=TermRole.EVENT,
            mutation_templates=(
                MutationTemplate(
                    key_suffix="root_position",
                    target_key="state.root.position",
                    target_kind=MutationTargetKind.SIMULATION_STATE,
                    selector=root,
                    field_kind=MutationFieldKind.POSITION,
                    trigger=MutationTrigger.RESET,
                    commit_phase=MutationCommitPhase.RESET,
                    operation=MutationOperation.SET,
                    baseline=MutationBaseline.DEFAULT,
                    persistence=MutationPersistence.EPISODE,
                    recompute=MutationRecomputeLevel.KINEMATICS,
                    value_template=_manager_buffer(
                        row_shape=(3,), lifetime=BufferLifetime.UNTIL_COMMIT
                    ),
                ),
                MutationTemplate(
                    key_suffix="root_orientation",
                    target_key="state.root.orientation",
                    target_kind=MutationTargetKind.SIMULATION_STATE,
                    selector=root,
                    field_kind=MutationFieldKind.ORIENTATION,
                    trigger=MutationTrigger.RESET,
                    commit_phase=MutationCommitPhase.RESET,
                    operation=MutationOperation.SET,
                    baseline=MutationBaseline.DEFAULT,
                    persistence=MutationPersistence.EPISODE,
                    recompute=MutationRecomputeLevel.KINEMATICS,
                    value_template=_manager_buffer(
                        row_shape=(4,), lifetime=BufferLifetime.UNTIL_COMMIT
                    ),
                ),
                MutationTemplate(
                    key_suffix="root_linear_velocity",
                    target_key="state.root.linear_velocity",
                    target_kind=MutationTargetKind.SIMULATION_STATE,
                    selector=root,
                    field_kind=MutationFieldKind.LINEAR_VELOCITY,
                    trigger=MutationTrigger.RESET,
                    commit_phase=MutationCommitPhase.RESET,
                    operation=MutationOperation.SET,
                    baseline=MutationBaseline.DEFAULT,
                    persistence=MutationPersistence.EPISODE,
                    recompute=MutationRecomputeLevel.KINEMATICS,
                    value_template=_manager_buffer(
                        row_shape=(3,), lifetime=BufferLifetime.UNTIL_COMMIT
                    ),
                ),
                MutationTemplate(
                    key_suffix="root_angular_velocity",
                    target_key="state.root.angular_velocity",
                    target_kind=MutationTargetKind.SIMULATION_STATE,
                    selector=root,
                    field_kind=MutationFieldKind.ANGULAR_VELOCITY,
                    trigger=MutationTrigger.RESET,
                    commit_phase=MutationCommitPhase.RESET,
                    operation=MutationOperation.SET,
                    baseline=MutationBaseline.DEFAULT,
                    persistence=MutationPersistence.EPISODE,
                    recompute=MutationRecomputeLevel.KINEMATICS,
                    value_template=_manager_buffer(
                        row_shape=(3,), lifetime=BufferLifetime.UNTIL_COMMIT
                    ),
                ),
                MutationTemplate(
                    key_suffix="hinge_position",
                    target_key="state.dof.position",
                    target_kind=MutationTargetKind.SIMULATION_STATE,
                    selector=hinge,
                    field_kind=MutationFieldKind.POSITION,
                    trigger=MutationTrigger.RESET,
                    commit_phase=MutationCommitPhase.RESET,
                    operation=MutationOperation.SET,
                    baseline=MutationBaseline.DEFAULT,
                    persistence=MutationPersistence.EPISODE,
                    recompute=MutationRecomputeLevel.KINEMATICS,
                    value_template=_manager_buffer(
                        row_shape=(1,), lifetime=BufferLifetime.UNTIL_COMMIT
                    ),
                ),
                MutationTemplate(
                    key_suffix="hinge_angular_velocity",
                    target_key="state.dof.angular_velocity",
                    target_kind=MutationTargetKind.SIMULATION_STATE,
                    selector=hinge,
                    field_kind=MutationFieldKind.ANGULAR_VELOCITY,
                    trigger=MutationTrigger.RESET,
                    commit_phase=MutationCommitPhase.RESET,
                    operation=MutationOperation.SET,
                    baseline=MutationBaseline.DEFAULT,
                    persistence=MutationPersistence.EPISODE,
                    recompute=MutationRecomputeLevel.KINEMATICS,
                    value_template=_manager_buffer(
                        row_shape=(1,), lifetime=BufferLifetime.UNTIL_COMMIT
                    ),
                ),
            ),
        )
    )
    registry.register(
        TermDefinition(
            key="obs.root_position",
            version="1",
            phase=TermPhase.OBSERVATION,
            role=TermRole.OBSERVATION,
            state_requirements=(
                StateRequirement(
                    semantic_key="robot.root.position",
                    selector=root,
                    field_kind=StateFieldKind.POSITION,
                    tensor=TensorSpec(
                        (3,),
                        "float32",
                        frame=ReferenceFrame.WORLD,
                        unit=PhysicalUnit.METER,
                    ),
                    entity_axis=None,
                ),
                StateRequirement(
                    semantic_key="robot.root.orientation",
                    selector=root,
                    field_kind=StateFieldKind.ORIENTATION,
                    tensor=TensorSpec(
                        (4,),
                        "float32",
                        frame=ReferenceFrame.WORLD,
                        unit=PhysicalUnit.QUATERNION,
                        quaternion_order=QuaternionOrder.WXYZ,
                    ),
                    entity_axis=None,
                ),
                StateRequirement(
                    semantic_key="robot.root.linear_velocity",
                    selector=root,
                    field_kind=StateFieldKind.LINEAR_VELOCITY,
                    tensor=TensorSpec(
                        (3,),
                        "float32",
                        frame=ReferenceFrame.WORLD,
                        unit=PhysicalUnit.METER_PER_SECOND,
                    ),
                    entity_axis=None,
                ),
                StateRequirement(
                    semantic_key="robot.root.angular_velocity",
                    selector=root,
                    field_kind=StateFieldKind.ANGULAR_VELOCITY,
                    tensor=TensorSpec(
                        (3,),
                        "float32",
                        frame=ReferenceFrame.WORLD,
                        unit=PhysicalUnit.RADIAN_PER_SECOND,
                    ),
                    entity_axis=None,
                ),
                StateRequirement(
                    semantic_key="robot.left_hip_pitch.position",
                    selector=hinge,
                    field_kind=StateFieldKind.POSITION,
                    tensor=TensorSpec(
                        (1,),
                        "float32",
                        frame=ReferenceFrame.JOINT,
                        unit=PhysicalUnit.RADIAN,
                    ),
                    entity_axis=None,
                ),
                StateRequirement(
                    semantic_key="robot.left_hip_pitch.angular_velocity",
                    selector=hinge,
                    field_kind=StateFieldKind.ANGULAR_VELOCITY,
                    tensor=TensorSpec(
                        (1,),
                        "float32",
                        frame=ReferenceFrame.JOINT,
                        unit=PhysicalUnit.RADIAN_PER_SECOND,
                    ),
                    entity_axis=None,
                ),
            ),
            output=TensorSpec(
                (3,),
                "float32",
                frame=ReferenceFrame.WORLD,
                unit=PhysicalUnit.METER,
            ),
        )
    )
    task = TaskSpec.create(
        key="g1_selector_reset_slice",
        terms=(
            TermInvocation.create(
                key="reset_full_state",
                definition_key="reset.full_state",
            ),
            TermInvocation.create(
                key="root_position",
                definition_key="obs.root_position",
                dependencies=("reset_full_state",),
                observation_group="policy",
            ),
        ),
        control=ControlSpec(
            "robot.joint.position_target",
            _manager_buffer(row_shape=(_ACTION_DIM,), lifetime=BufferLifetime.UNTIL_STEP_COMPLETE),
        ),
        execution_profile=ExecutionProfile.HOST_NUMPY,
        executor_key="reference.numpy.g1-selector.v1",
        policy=PolicySpec(("policy",), (0.25,) * _ACTION_DIM),
    )
    resolver = _StaticG1Resolver()
    plan = TaskCompiler(registry).compile(
        task,
        resolver=resolver,
        capabilities=frozenset(
            {
                "state.root.position",
                "state.root.orientation",
                "state.root.linear_velocity",
                "state.root.angular_velocity",
                "state.dof.position",
                "state.dof.angular_velocity",
            }
        ),
    )
    return plan, resolver


def _scene() -> SceneCfg:
    from unilab.assets import ASSETS_ROOT_PATH

    return SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml"))


def _compiled_reset_batch(mutation_plan, rows: RowSelection) -> TypedBackendMutationBatch:
    """Materialize one manager-owned full-state envelope without backend access."""

    target_values = {
        "state.root.position": (0.25, -0.5, 1.0),
        "state.root.orientation": (1.0, 0.0, 0.0, 0.0),
        "state.root.linear_velocity": (0.1, -0.2, 0.3),
        "state.root.angular_velocity": (-0.4, 0.5, -0.6),
        "state.dof.position": (0.125,),
        "state.dof.angular_velocity": (-0.25,),
    }
    values = []
    for field_index, spec in enumerate(mutation_plan.specs):
        payload = np.asarray(target_values[spec.target.target_key], dtype=np.float32)
        handle = np.empty((rows.count, *spec.value_buffer.row_shape), dtype=np.float32)
        handle[...] = payload
        values.append(
            MutationValueBatch(
                plan=mutation_plan,
                field_index=field_index,
                rows=rows,
                buffer=BufferView(handle, handle.shape, spec.value_buffer),
            )
        )
    return TypedBackendMutationBatch(
        plan=mutation_plan,
        rows=rows,
        state=SimulationStateMutationBatch(tuple(values)),
    )


def test_g1_full_reset_slice_is_shared_by_mujoco_and_mjwarp() -> None:
    """Both independent backends cold-bind the same manager plan unchanged."""

    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.fail("compiled mjwarp selector binding requires an active CUDA Warp device")

    plan, resolver = _compiled_g1_full_reset_plan()
    assert resolver.calls == ["robot.left_hip_pitch", "robot.root_state"]
    selectors = {spec.target.target_key: spec.target.selector_spec for spec in plan.mutation_specs}
    root_selector = selectors["state.root.position"]
    hinge_selector = selectors["state.dof.position"]
    assert root_selector is not None
    assert root_selector.semantic_key == "robot.root_state"
    assert root_selector.expressions == (_BASE_NAME,)
    assert root_selector.entity_ids == (1,)
    assert hinge_selector is not None
    assert hinge_selector.semantic_key == "robot.left_hip_pitch"
    assert hinge_selector.expressions == (_HINGE_NAME,)
    assert hinge_selector.entity_ids == (0,)

    mujoco = MuJoCoBackend(
        _scene(),
        2,
        0.02 / 3.0,
        base_name=_BASE_NAME,
        np_dtype=np.float32,
    )
    mujoco.materialize()
    mjwarp = create_backend(
        "mjwarp",
        _scene(),
        2,
        0.02 / 3.0,
        base_name=_BASE_NAME,
    )
    try:
        mujoco_io = mujoco.bind_task_io(plan.backend_io)
        mjwarp_io = mjwarp.bind_task_io(plan.backend_io)
        mujoco_mutation = mujoco.bind_mutation_plan(plan.mutation_specs)
        mjwarp_mutation = mjwarp.bind_mutation_plan(plan.mutation_specs)

        assert {spec.target.target_key for spec in plan.mutation_specs} == {
            "state.root.position",
            "state.root.orientation",
            "state.root.linear_velocity",
            "state.root.angular_velocity",
            "state.dof.position",
            "state.dof.angular_velocity",
        }
        assert mujoco_io.state.fields == mjwarp_io.state.fields == plan.backend_io.state_fields
        mujoco_targets = {
            spec.target.target_key: spec.target.entity_ids for spec in mujoco_mutation.specs
        }
        mjwarp_targets = {
            spec.target.target_key: spec.target.entity_ids for spec in mjwarp_mutation.specs
        }
        assert (
            mujoco_targets
            == mjwarp_targets
            == {
                "state.root.position": (1,),
                "state.root.orientation": (1,),
                "state.root.linear_velocity": (1,),
                "state.root.angular_velocity": (1,),
                "state.dof.position": (0,),
                "state.dof.angular_velocity": (0,),
            }
        )
        assert mujoco_mutation.backend_type == "mujoco"
        assert mjwarp_mutation.backend_type == "mjwarp"
        assert mujoco_mutation.fingerprint != mjwarp_mutation.fingerprint

        rows = RowSelection.selected(2, (1,))
        mujoco_reset = mujoco.reset_batch(
            mujoco_io,
            rows,
            mutation_batch=_compiled_reset_batch(mujoco_mutation, rows),
        )
        mjwarp_reset = mjwarp.reset_batch(
            mjwarp_io,
            rows,
            mutation_batch=_compiled_reset_batch(mjwarp_mutation, rows),
        )
        expected_fields = {
            "robot.root.position": (0.25, -0.5, 1.0),
            "robot.root.orientation": (1.0, 0.0, 0.0, 0.0),
            "robot.root.linear_velocity": (0.1, -0.2, 0.3),
            "robot.root.angular_velocity": (-0.4, 0.5, -0.6),
            "robot.left_hip_pitch.position": (0.125,),
            "robot.left_hip_pitch.angular_velocity": (-0.25,),
        }
        for key, expected in expected_fields.items():
            expected_array = np.asarray((expected,), dtype=np.float32)
            np.testing.assert_allclose(
                mujoco_reset.reset_state.buffer(key).handle,
                expected_array,
                atol=2.0e-5,
                rtol=2.0e-5,
            )
            np.testing.assert_allclose(
                mjwarp_reset.reset_state.buffer(key).handle,
                expected_array,
                atol=2.0e-5,
                rtol=2.0e-5,
            )
    finally:
        assert mujoco._pool is not None
        mujoco._pool.close()
