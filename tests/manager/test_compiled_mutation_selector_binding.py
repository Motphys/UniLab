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
    PhysicalUnit,
    ReferenceFrame,
    StateFieldKind,
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


def _compiled_g1_hinge_reset_plan():
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
            key="reset.left_hip_pitch",
            version="1",
            phase=TermPhase.RESET,
            role=TermRole.EVENT,
            mutation_templates=(
                MutationTemplate(
                    key_suffix="",
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
                key="reset_left_hip_pitch",
                definition_key="reset.left_hip_pitch",
            ),
            TermInvocation.create(
                key="root_position",
                definition_key="obs.root_position",
                dependencies=("reset_left_hip_pitch",),
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
        capabilities=frozenset({"state.root.position", "state.dof.position"}),
    )
    return plan, resolver


def _scene() -> SceneCfg:
    from unilab.assets import ASSETS_ROOT_PATH

    return SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml"))


def test_g1_selector_reset_slice_is_shared_by_mujoco_and_mjwarp() -> None:
    """Both independent backends cold-bind the same manager plan unchanged."""

    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.fail("compiled mjwarp selector binding requires an active CUDA Warp device")

    plan, resolver = _compiled_g1_hinge_reset_plan()
    assert resolver.calls == ["robot.left_hip_pitch", "robot.root_state"]
    selector = plan.mutation_specs[0].target.selector_spec
    assert selector is not None
    assert selector.semantic_key == "robot.left_hip_pitch"
    assert selector.expressions == (_HINGE_NAME,)
    assert selector.entity_ids == (0,)

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

        assert plan.mutation_specs[0].target.selector_spec == selector
        assert mujoco_io.state.fields == mjwarp_io.state.fields == plan.backend_io.state_fields
        assert mujoco_mutation.specs[0].target.entity_ids == (0,)
        assert mjwarp_mutation.specs[0].target.entity_ids == (0,)
        assert mujoco_mutation.backend_type == "mujoco"
        assert mjwarp_mutation.backend_type == "mjwarp"
        assert mujoco_mutation.fingerprint != mjwarp_mutation.fingerprint
    finally:
        assert mujoco._pool is not None
        mujoco._pool.close()
