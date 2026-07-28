"""Real G1 selector binding through the public backend contract only."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

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
    PhysicalUnit,
    ReferenceFrame,
    RowSelection,
    StateFieldKind,
    create_backend,
)
from unilab.base.backend.base import SimBackend
from unilab.base.backend.mjwarp.dependencies import load_mjwarp_dependencies
from unilab.base.backend.mujoco.backend import MuJoCoBackend
from unilab.base.scene import SceneCfg
from unilab.manager import (
    BackendEntityResolver,
    CompiledTaskPlan,
    EntityKind,
    EntitySelector,
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
_SENSOR_CONTRACTS = (
    ("pelvis_local_linvel", ReferenceFrame.SENSOR, PhysicalUnit.METER_PER_SECOND),
    ("torso_gyro", ReferenceFrame.SENSOR, PhysicalUnit.RADIAN_PER_SECOND),
    ("torso_upvector", ReferenceFrame.WORLD, PhysicalUnit.UNITLESS),
    ("left_foot_pos", ReferenceFrame.WORLD, PhysicalUnit.METER),
    ("right_foot_pos", ReferenceFrame.WORLD, PhysicalUnit.METER),
)


def _scene() -> SceneCfg:
    from unilab.assets import ASSETS_ROOT_PATH

    return SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml"))


def _control_buffer(action_dim: int) -> BufferContract:
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


def _full_g1_task(actuator_names: tuple[str, ...]) -> tuple[TaskSpec, TermRegistry, frozenset[str]]:
    root = EntitySelector(
        key="g1.root",
        entity="g1",
        kind=EntityKind.ROOT,
        expressions=(_BASE_NAME,),
    )
    dofs = EntitySelector(
        key="g1.actuated_dofs",
        entity="g1",
        kind=EntityKind.DOF,
        expressions=actuator_names,
    )
    state_requirements = [
        StateRequirement(
            semantic_key="g1.root.position",
            selector=root,
            field_kind=StateFieldKind.POSITION,
            tensor=TensorSpec((3,), "float32", frame=ReferenceFrame.WORLD, unit=PhysicalUnit.METER),
            entity_axis=None,
        ),
        StateRequirement(
            semantic_key="g1.root.orientation",
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
            semantic_key="g1.root.linear_velocity",
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
            semantic_key="g1.root.angular_velocity",
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
            semantic_key="g1.dof.position",
            selector=dofs,
            field_kind=StateFieldKind.POSITION,
            tensor=TensorSpec(
                (len(actuator_names),),
                "float32",
                frame=ReferenceFrame.JOINT,
                unit=PhysicalUnit.RADIAN,
            ),
        ),
        StateRequirement(
            semantic_key="g1.dof.angular_velocity",
            selector=dofs,
            field_kind=StateFieldKind.ANGULAR_VELOCITY,
            tensor=TensorSpec(
                (len(actuator_names),),
                "float32",
                frame=ReferenceFrame.JOINT,
                unit=PhysicalUnit.RADIAN_PER_SECOND,
            ),
        ),
    ]
    for sensor_name, frame, unit in _SENSOR_CONTRACTS:
        state_requirements.append(
            StateRequirement(
                semantic_key=f"g1.sensor.{sensor_name}",
                selector=EntitySelector(
                    key=f"g1.sensor.{sensor_name}",
                    entity="g1",
                    kind=EntityKind.SENSOR,
                    expressions=(sensor_name,),
                ),
                field_kind=StateFieldKind.VALUE,
                tensor=TensorSpec((3,), "float32", frame=frame, unit=unit),
                entity_axis=None,
            )
        )

    registry = TermRegistry()
    registry.register(
        TermDefinition(
            key="obs.g1_full_state",
            version="1",
            phase=TermPhase.OBSERVATION,
            role=TermRole.OBSERVATION,
            state_requirements=tuple(state_requirements),
            # This slice binds data only. A later task-kernel issue owns the
            # real observation layout and numerical differential oracle.
            output=TensorSpec((1,), "float32"),
        )
    )
    task = TaskSpec.create(
        key="g1_public_selector_plan",
        terms=(
            TermInvocation.create(
                key="full_state",
                definition_key="obs.g1_full_state",
                observation_group="policy",
            ),
        ),
        control=ControlSpec(
            semantic_key="g1.joint.position_target",
            buffer=_control_buffer(len(actuator_names)),
        ),
        execution_profile=ExecutionProfile.HOST_NUMPY,
        executor_key="reference.numpy.g1-public-selector.v1",
        policy=PolicySpec(("policy",), (0.25,) * len(actuator_names)),
    )
    capabilities = frozenset(
        {
            "state.root.position",
            "state.root.orientation",
            "state.root.linear_velocity",
            "state.root.angular_velocity",
            "state.dof.position",
            "state.dof.angular_velocity",
            "state.sensor.value",
        }
    )
    return task, registry, capabilities


def _compile_g1_plan(
    task: TaskSpec,
    registry: TermRegistry,
    capabilities: frozenset[str],
    backend: SimBackend,
) -> tuple[CompiledTaskPlan, BackendEntityResolver]:
    resolver = BackendEntityResolver(backend)
    return (
        TaskCompiler(registry).compile(
            task,
            resolver=resolver,
            capabilities=capabilities,
        ),
        resolver,
    )


def _assert_runtime_uses_only_bound_plan(
    backend: SimBackend,
    plan: CompiledTaskPlan,
    resolver: BackendEntityResolver,
) -> None:
    with (
        patch.object(resolver, "resolve", side_effect=AssertionError("runtime resolver lookup")),
        patch.object(backend, "get_body_ids", side_effect=AssertionError("body selector fallback")),
        patch.object(
            backend,
            "get_joint_dof_pos_indices",
            side_effect=AssertionError("joint selector fallback"),
        ),
        patch.object(
            backend, "get_sensor_ids", side_effect=AssertionError("sensor selector fallback")
        ),
        patch.object(backend, "get_base_pos", side_effect=AssertionError("legacy getter fallback")),
        patch.object(
            backend, "get_base_quat", side_effect=AssertionError("legacy getter fallback")
        ),
        patch.object(
            backend, "get_base_lin_vel", side_effect=AssertionError("legacy getter fallback")
        ),
        patch.object(
            backend, "get_base_ang_vel", side_effect=AssertionError("legacy getter fallback")
        ),
        patch.object(backend, "get_dof_pos", side_effect=AssertionError("legacy getter fallback")),
        patch.object(backend, "get_dof_vel", side_effect=AssertionError("legacy getter fallback")),
        patch.object(
            backend, "get_sensor_data", side_effect=AssertionError("legacy getter fallback")
        ),
        patch.object(Path, "read_bytes", side_effect=AssertionError("asset fallback")),
        patch.object(Path, "read_text", side_effect=AssertionError("asset fallback")),
    ):
        bound = backend.bind_task_io(plan.backend_io)

    raw_model_attribute = "_cpu_model" if backend.backend_type == "mjwarp" else "_model"
    with (
        patch.object(resolver, "resolve", side_effect=AssertionError("runtime resolver lookup")),
        patch.object(backend, "get_body_ids", side_effect=AssertionError("body selector fallback")),
        patch.object(
            backend,
            "get_joint_dof_pos_indices",
            side_effect=AssertionError("joint selector fallback"),
        ),
        patch.object(
            backend, "get_sensor_ids", side_effect=AssertionError("sensor selector fallback")
        ),
        patch.object(backend, "get_base_pos", side_effect=AssertionError("legacy getter fallback")),
        patch.object(
            backend, "get_base_quat", side_effect=AssertionError("legacy getter fallback")
        ),
        patch.object(
            backend, "get_base_lin_vel", side_effect=AssertionError("legacy getter fallback")
        ),
        patch.object(
            backend, "get_base_ang_vel", side_effect=AssertionError("legacy getter fallback")
        ),
        patch.object(backend, "get_dof_pos", side_effect=AssertionError("legacy getter fallback")),
        patch.object(backend, "get_dof_vel", side_effect=AssertionError("legacy getter fallback")),
        patch.object(
            backend, "get_sensor_data", side_effect=AssertionError("legacy getter fallback")
        ),
        patch.object(backend, raw_model_attribute, new=object()),
        patch.object(Path, "read_bytes", side_effect=AssertionError("asset fallback")),
        patch.object(Path, "read_text", side_effect=AssertionError("asset fallback")),
    ):
        state = backend.read_state_batch(
            bound,
            RowSelection.selected(2, (1,)),
        ).state
        assert tuple(
            state.buffer_at(index).shape for index in range(len(plan.backend_io.state_fields))
        ) == tuple((1, *field.buffer.row_shape) for field in plan.backend_io.state_fields)


def test_g1_full_state_plan_is_publicly_resolved_and_bound_by_both_backends() -> None:
    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.fail("public G1 selector plan requires an active CUDA Warp device")

    mujoco = MuJoCoBackend(
        _scene(),
        2,
        0.02 / 3.0,
        base_name=_BASE_NAME,
        np_dtype=np.float32,
    )
    mujoco.materialize()
    mjwarp = create_backend("mjwarp", _scene(), 2, 0.02 / 3.0, base_name=_BASE_NAME)
    try:
        actuator_names = mujoco.get_actuator_names()
        assert len(actuator_names) == 29
        assert mjwarp.get_actuator_names() == actuator_names
        sensor_names = tuple(item[0] for item in _SENSOR_CONTRACTS)
        assert np.array_equal(
            mujoco.get_sensor_ids(sensor_names), mjwarp.get_sensor_ids(sensor_names)
        )
        assert np.array_equal(
            mujoco.get_joint_dof_pos_indices(actuator_names),
            mjwarp.get_joint_dof_pos_indices(actuator_names),
        )
        with pytest.raises(ValueError, match="Sensor .* not found"):
            mjwarp.get_sensor_ids(("missing_sensor",))
        with pytest.raises(ValueError, match="Joint .* not found"):
            mjwarp.get_joint_dof_pos_indices(("missing_joint",))
        task, mujoco_registry, capabilities = _full_g1_task(actuator_names)
        mjwarp_registry = TermRegistry()
        for definition in mujoco_registry.definitions():
            mjwarp_registry.register(definition)

        mujoco_plan, mujoco_resolver = _compile_g1_plan(
            task,
            mujoco_registry,
            capabilities,
            mujoco,
        )
        mjwarp_plan, mjwarp_resolver = _compile_g1_plan(
            task,
            mjwarp_registry,
            capabilities,
            mjwarp,
        )

        assert mujoco_plan.fingerprint == mjwarp_plan.fingerprint
        assert mujoco_plan.policy_abi == mjwarp_plan.policy_abi
        assert mujoco_plan.selector_binding_fingerprint == mjwarp_plan.selector_binding_fingerprint
        assert mujoco_plan.backend_io.state_fields == mjwarp_plan.backend_io.state_fields
        selector_ids = {selector.key: selector.entity_ids for selector in mujoco_plan.selectors}
        assert selector_ids["g1.root"] == (1,)
        assert selector_ids["g1.actuated_dofs"] == tuple(range(29))
        assert selector_ids["g1.sensor.pelvis_local_linvel"] == (0,)
        assert selector_ids["g1.sensor.torso_gyro"] == (4,)
        assert selector_ids["g1.sensor.torso_upvector"] == (6,)
        assert selector_ids["g1.sensor.left_foot_pos"] == (7,)
        assert selector_ids["g1.sensor.right_foot_pos"] == (10,)

        _assert_runtime_uses_only_bound_plan(mujoco, mujoco_plan, mujoco_resolver)
        _assert_runtime_uses_only_bound_plan(mjwarp, mjwarp_plan, mjwarp_resolver)
    finally:
        assert mujoco._pool is not None
        mujoco._pool.close()
