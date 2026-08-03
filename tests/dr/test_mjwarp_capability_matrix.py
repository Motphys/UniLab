"""Bidirectional capability evidence for the ``mjwarp`` typed reset surface.

The mandatory registry in this module is deliberately independent from the
production manifest.  A production capability may therefore be advertised
only when its complete metadata and executable pytest node both match this
registry.  The CUDA cases use public batch contracts as their oracle boundary;
backend-private arrays are not inspected.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np
import pytest
import torch
from tests.dr.mjwarp_model_mutation_support import reset_device_state

from unilab.base.backend import (
    BackendIORequirements,
    BoundBackendPlan,
    BoundFieldIdentity,
    BufferContract,
    BufferLayout,
    BufferLifetime,
    BufferMutability,
    BufferOwner,
    BufferPlacement,
    BufferView,
    ControlBatch,
    ControlSpec,
    DeviceBufferLease,
    DeviceCompletion,
    DeviceResetMutationBatch,
    DeviceTensorView,
    ExecutionProfile,
    MemorySpace,
    MutationBaseline,
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
    MutationSpec,
    MutationTargetKind,
    MutationTargetSpec,
    MutationTrigger,
    MutationValueBatch,
    PhysicalUnit,
    ReferenceFrame,
    RowSelection,
    SimulationStateMutationBatch,
    StateBatch,
    StateBatchPhase,
    StateEntityKind,
    StateFieldKind,
    StateFieldSpec,
    TypedBackendMutationBatch,
)
from unilab.base.backend.mjwarp.backend import MjwarpBackend as ProductionMjwarpBackend
from unilab.base.backend.mjwarp.dependencies import load_mjwarp_dependencies
from unilab.base.scene import SceneCfg

_NUM_ENVS = 32
_BASE = "pelvis"
_HINGE = "left_hip_pitch_joint"
_CAPABILITY_ID = "state.qpos_qvel_reset"
_MODEL_CAPABILITY_ID = "actuator.position_servo_pd_gain"
_ARMATURE_CAPABILITY_ID = "joint.armature"
_GRAVCOMP_CAPABILITY_ID = "body.gravity_compensation"
_MODEL_GRAPH_INVALIDATIONS = (
    MutationGraphInvalidation.FORWARD_GRAPH,
    MutationGraphInvalidation.MODEL_BRIDGE_CACHE,
    MutationGraphInvalidation.RESET_GRAPH,
    MutationGraphInvalidation.SENSE_GRAPH,
    MutationGraphInvalidation.SENSOR_CONTEXT,
    MutationGraphInvalidation.STEP_GRAPH,
)
_MANDATORY_NODE = (
    "tests/dr/test_mjwarp_capability_matrix.py::test_mjwarp_advertised_capability_case"
)
_MANDATORY_MODEL_NODE = (
    "tests/dr/test_mjwarp_model_mutation.py::test_mjwarp_advertised_model_capability_case"
)
_MANDATORY_RECOMPUTE_NODE = (
    "tests/dr/test_mjwarp_recompute.py::test_mjwarp_advertised_recompute_capability_case"
)
_SELECTED = (29, 3, 17, 8, 25, 1, 14, 21)
_PAIRED_CONTROLS = (28, 2, 16, 9, 24, 0, 15, 20)
_STATE_KEYS = (
    "root.position",
    "root.orientation",
    "root.linear_velocity",
    "root.angular_velocity",
    "dof.position",
    "dof.angular_velocity",
)


@dataclass(frozen=True)
class _CapabilityRecord:
    mandatory_test_id: str
    case_id: str
    capability_id: str
    execution_profile: ExecutionProfile
    target_key: str
    target_kind: MutationTargetKind
    entity_kind: MutationEntityKind
    field_kind: MutationFieldKind
    value_row_shape: tuple[int, ...]
    value_dtype: str
    value_layout: BufferLayout
    value_memory_space: MemorySpace
    value_device_type: str | None
    value_owner: BufferOwner
    value_mutability: BufferMutability
    value_lifetime: BufferLifetime
    value_dlpack_exportable: bool
    value_address_stable: bool
    direct_fields: tuple[str, ...]
    derived_fields: tuple[str, ...]
    storage_kind: MutationFieldStorageKind
    graph_impact: MutationGraphImpact
    graph_invalidations: tuple[MutationGraphInvalidation, ...]
    trigger: MutationTrigger
    commit_phase: MutationCommitPhase
    operation: MutationOperation
    baseline: MutationBaseline
    persistence: MutationPersistence
    recompute: MutationRecomputeLevel
    row_scope: MutationCapabilityRowScope

    @property
    def parameter_id(self) -> str:
        return f"{self.execution_profile.value}-{self.target_key}"

    @property
    def semantic_identity(self) -> tuple[ExecutionProfile, str, MutationOperation]:
        return self.execution_profile, self.target_key, self.operation


@dataclass(frozen=True)
class _MandatoryCase:
    record: _CapabilityRecord
    selector: str
    state_key: str

    @property
    def parameter_id(self) -> str:
        return self.record.parameter_id


@dataclass(frozen=True)
class _FieldExpectation:
    target_key: str
    entity_kind: MutationEntityKind
    mutation_field_kind: MutationFieldKind
    value_row_shape: tuple[int, ...]
    direct_field: str
    selector: str
    state_key: str


_FIELD_EXPECTATIONS = (
    _FieldExpectation(
        target_key="state.root.position",
        entity_kind=MutationEntityKind.BODY,
        mutation_field_kind=MutationFieldKind.POSITION,
        value_row_shape=(3,),
        direct_field="data.qpos",
        selector=_BASE,
        state_key="root.position",
    ),
    _FieldExpectation(
        target_key="state.root.orientation",
        entity_kind=MutationEntityKind.BODY,
        mutation_field_kind=MutationFieldKind.ORIENTATION,
        value_row_shape=(4,),
        direct_field="data.qpos",
        selector=_BASE,
        state_key="root.orientation",
    ),
    _FieldExpectation(
        target_key="state.root.linear_velocity",
        entity_kind=MutationEntityKind.BODY,
        mutation_field_kind=MutationFieldKind.LINEAR_VELOCITY,
        value_row_shape=(3,),
        direct_field="data.qvel",
        selector=_BASE,
        state_key="root.linear_velocity",
    ),
    _FieldExpectation(
        target_key="state.root.angular_velocity",
        entity_kind=MutationEntityKind.BODY,
        mutation_field_kind=MutationFieldKind.ANGULAR_VELOCITY,
        value_row_shape=(3,),
        direct_field="data.qvel",
        selector=_BASE,
        state_key="root.angular_velocity",
    ),
    _FieldExpectation(
        target_key="state.dof.position",
        entity_kind=MutationEntityKind.DOF,
        mutation_field_kind=MutationFieldKind.POSITION,
        value_row_shape=(1,),
        direct_field="data.qpos",
        selector=_HINGE,
        state_key="dof.position",
    ),
    _FieldExpectation(
        target_key="state.dof.angular_velocity",
        entity_kind=MutationEntityKind.DOF,
        mutation_field_kind=MutationFieldKind.ANGULAR_VELOCITY,
        value_row_shape=(1,),
        direct_field="data.qvel",
        selector=_HINGE,
        state_key="dof.angular_velocity",
    ),
)


def _mandatory_case(
    profile: ExecutionProfile,
    field: _FieldExpectation,
) -> _MandatoryCase:
    device = profile is ExecutionProfile.DEVICE_RESIDENT
    parameter_id = f"{profile.value}-{field.target_key}"
    return _MandatoryCase(
        record=_CapabilityRecord(
            mandatory_test_id=f"{_MANDATORY_NODE}[{parameter_id}]",
            case_id=f"mjwarp.{profile.value}.{field.target_key}.reset",
            capability_id=_CAPABILITY_ID,
            execution_profile=profile,
            target_key=field.target_key,
            target_kind=MutationTargetKind.SIMULATION_STATE,
            entity_kind=field.entity_kind,
            field_kind=field.mutation_field_kind,
            value_row_shape=field.value_row_shape,
            value_dtype="float32",
            value_layout=BufferLayout.C_CONTIGUOUS,
            value_memory_space=MemorySpace.DEVICE if device else MemorySpace.HOST,
            value_device_type="cuda" if device else "cpu",
            value_owner=BufferOwner.MANAGER,
            value_mutability=BufferMutability.READ_ONLY,
            value_lifetime=BufferLifetime.UNTIL_COMMIT,
            value_dlpack_exportable=device,
            value_address_stable=True,
            direct_fields=(field.direct_field,),
            derived_fields=(),
            storage_kind=MutationFieldStorageKind.DATA_NATIVE,
            graph_impact=MutationGraphImpact.STABLE_ADDRESS,
            graph_invalidations=(),
            trigger=MutationTrigger.RESET,
            commit_phase=MutationCommitPhase.RESET,
            operation=MutationOperation.SET,
            baseline=MutationBaseline.DEFAULT,
            persistence=MutationPersistence.EPISODE,
            recompute=MutationRecomputeLevel.KINEMATICS,
            row_scope=MutationCapabilityRowScope.SELECTED_ROWS,
        ),
        selector=field.selector,
        state_key=field.state_key,
    )


_MANDATORY_CASES = tuple(
    _mandatory_case(profile, field)
    for profile in (ExecutionProfile.HOST_NUMPY, ExecutionProfile.DEVICE_RESIDENT)
    for field in _FIELD_EXPECTATIONS
)


def _mandatory_model_record(
    *,
    target_key: str,
    field_kind: MutationFieldKind,
    direct_fields: tuple[str, ...],
    operation: MutationOperation,
) -> _CapabilityRecord:
    parameter_id = f"device_resident-{target_key}-{operation.value}"
    return _CapabilityRecord(
        mandatory_test_id=f"{_MANDATORY_MODEL_NODE}[{parameter_id}]",
        case_id=f"mjwarp.device_resident.{target_key}.reset.{operation.value}",
        capability_id=_MODEL_CAPABILITY_ID,
        execution_profile=ExecutionProfile.DEVICE_RESIDENT,
        target_key=target_key,
        target_kind=MutationTargetKind.MODEL_PARAMETER,
        entity_kind=MutationEntityKind.ACTUATOR,
        field_kind=field_kind,
        value_row_shape=(1,),
        value_dtype="float32",
        value_layout=BufferLayout.C_CONTIGUOUS,
        value_memory_space=MemorySpace.DEVICE,
        value_device_type="cuda",
        value_owner=BufferOwner.MANAGER,
        value_mutability=BufferMutability.READ_ONLY,
        value_lifetime=BufferLifetime.UNTIL_COMMIT,
        value_dlpack_exportable=True,
        value_address_stable=True,
        direct_fields=direct_fields,
        derived_fields=(),
        storage_kind=MutationFieldStorageKind.MODEL_FIELD_EXPANSION,
        graph_impact=MutationGraphImpact.RECAPTURE_REQUIRED,
        graph_invalidations=_MODEL_GRAPH_INVALIDATIONS,
        trigger=MutationTrigger.RESET,
        commit_phase=MutationCommitPhase.RESET,
        operation=operation,
        baseline=MutationBaseline.DEFAULT,
        persistence=MutationPersistence.EPISODE,
        recompute=MutationRecomputeLevel.NONE,
        row_scope=MutationCapabilityRowScope.SELECTED_ROWS,
    )


_MANDATORY_MODEL_RECORDS = tuple(
    _mandatory_model_record(
        target_key=target_key,
        field_kind=field_kind,
        direct_fields=direct_fields,
        operation=operation,
    )
    for target_key, field_kind, direct_fields in (
        (
            "actuator.pd_stiffness",
            MutationFieldKind.STIFFNESS,
            ("actuator_biasprm", "actuator_gainprm"),
        ),
        (
            "actuator.pd_damping",
            MutationFieldKind.DAMPING,
            ("actuator_biasprm",),
        ),
    )
    for operation in (MutationOperation.SET, MutationOperation.SCALE)
)
_MANDATORY_RECOMPUTE_RECORDS = tuple(
    _CapabilityRecord(
        mandatory_test_id=(
            f"{_MANDATORY_RECOMPUTE_NODE}[device_resident-{target_key}-{operation.value}]"
        ),
        case_id=f"mjwarp.device_resident.{target_key}.reset.{operation.value}",
        capability_id=capability_id,
        execution_profile=ExecutionProfile.DEVICE_RESIDENT,
        target_key=target_key,
        target_kind=MutationTargetKind.MODEL_PARAMETER,
        entity_kind=entity_kind,
        field_kind=field_kind,
        value_row_shape=(1,),
        value_dtype="float32",
        value_layout=BufferLayout.C_CONTIGUOUS,
        value_memory_space=MemorySpace.DEVICE,
        value_device_type="cuda",
        value_owner=BufferOwner.MANAGER,
        value_mutability=BufferMutability.READ_ONLY,
        value_lifetime=BufferLifetime.UNTIL_COMMIT,
        value_dlpack_exportable=True,
        value_address_stable=True,
        direct_fields=direct_fields,
        derived_fields=derived_fields,
        storage_kind=MutationFieldStorageKind.MODEL_FIELD_EXPANSION,
        graph_impact=MutationGraphImpact.RECAPTURE_REQUIRED,
        graph_invalidations=_MODEL_GRAPH_INVALIDATIONS,
        trigger=MutationTrigger.RESET,
        commit_phase=MutationCommitPhase.RESET,
        operation=operation,
        baseline=MutationBaseline.DEFAULT,
        persistence=MutationPersistence.EPISODE,
        recompute=recompute,
        row_scope=MutationCapabilityRowScope.SELECTED_ROWS,
    )
    for (
        target_key,
        capability_id,
        entity_kind,
        field_kind,
        direct_fields,
        derived_fields,
        recompute,
    ) in (
        (
            "joint.armature",
            _ARMATURE_CAPABILITY_ID,
            MutationEntityKind.DOF,
            MutationFieldKind.ARMATURE,
            ("dof_armature",),
            (
                "actuator_acc0",
                "body_invweight0",
                "dof_invweight0",
                "tendon_invweight0",
                "tendon_length0",
            ),
            MutationRecomputeLevel.DYNAMICS,
        ),
        (
            "body.gravity_compensation",
            _GRAVCOMP_CAPABILITY_ID,
            MutationEntityKind.BODY,
            MutationFieldKind.GRAVITY_COMPENSATION,
            ("body_gravcomp",),
            ("body_subtreemass",),
            MutationRecomputeLevel.KINEMATICS,
        ),
    )
    for operation in (MutationOperation.SET, MutationOperation.SCALE)
)
_MANDATORY_RECORDS = (
    *(case.record for case in _MANDATORY_CASES),
    *_MANDATORY_MODEL_RECORDS,
    *_MANDATORY_RECOMPUTE_RECORDS,
)


def _duplicate_values(values: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(sorted(value for value, count in Counter(values).items() if count > 1))


def _require_exact_case_bijection(
    advertised: Sequence[_CapabilityRecord],
    mandatory: Sequence[_CapabilityRecord],
) -> None:
    """Reject any non-bijective or metadata-inexact support claim."""

    for label, records in (("advertised", advertised), ("mandatory", mandatory)):
        duplicate_test_ids = _duplicate_values(
            tuple(record.mandatory_test_id for record in records)
        )
        if duplicate_test_ids:
            raise AssertionError(f"duplicate {label} mandatory test IDs: {duplicate_test_ids}")
        duplicate_case_ids = _duplicate_values(tuple(record.case_id for record in records))
        if duplicate_case_ids:
            raise AssertionError(f"duplicate {label} case IDs: {duplicate_case_ids}")
        duplicate_semantics = _duplicate_values(
            tuple(record.semantic_identity for record in records)
        )
        if duplicate_semantics:
            raise AssertionError(
                f"duplicate {label} semantic capability cases: {duplicate_semantics}"
            )

    advertised_by_test = {record.mandatory_test_id: record for record in advertised}
    mandatory_by_test = {record.mandatory_test_id: record for record in mandatory}
    unadvertised = tuple(sorted(mandatory_by_test.keys() - advertised_by_test.keys()))
    untested = tuple(sorted(advertised_by_test.keys() - mandatory_by_test.keys()))
    if unadvertised:
        raise AssertionError(f"unadvertised mandatory parameter cases: {unadvertised}")
    if untested:
        raise AssertionError(f"advertised cases without mandatory parameters: {untested}")
    mismatched = tuple(
        test_id
        for test_id in sorted(mandatory_by_test)
        if advertised_by_test[test_id] != mandatory_by_test[test_id]
    )
    if mismatched:
        raise AssertionError(f"advertised capability metadata differs: {mismatched}")


def _manifest_records(manifest: MutationCapabilityManifest) -> tuple[_CapabilityRecord, ...]:
    manifest.require_valid()
    records: list[_CapabilityRecord] = []
    for capability in manifest.capabilities:
        descriptor = capability.descriptor
        assert descriptor is not None
        placement = capability.value_template.placement
        if (
            manifest.execution_profile is ExecutionProfile.DEVICE_RESIDENT
            and placement.device_index is None
        ):
            raise AssertionError("device capability placement has no CUDA index")
        for case in descriptor.cases:
            records.append(
                _CapabilityRecord(
                    mandatory_test_id=case.mandatory_test_id,
                    case_id=case.case_id,
                    capability_id=descriptor.capability_id,
                    execution_profile=case.execution_profile,
                    target_key=capability.target_key,
                    target_kind=capability.target_kind,
                    entity_kind=capability.entity_kind,
                    field_kind=capability.field_kind,
                    value_row_shape=capability.value_template.row_shape,
                    value_dtype=capability.value_template.dtype,
                    value_layout=capability.value_template.layout,
                    value_memory_space=placement.memory_space,
                    value_device_type=placement.device_type,
                    value_owner=capability.value_template.owner,
                    value_mutability=capability.value_template.mutability,
                    value_lifetime=capability.value_template.lifetime,
                    value_dlpack_exportable=capability.value_template.dlpack_exportable,
                    value_address_stable=capability.value_template.address_stable,
                    direct_fields=descriptor.direct_fields,
                    derived_fields=descriptor.derived_fields,
                    storage_kind=descriptor.storage_kind,
                    graph_impact=descriptor.graph_impact,
                    graph_invalidations=tuple(sorted(descriptor.graph_invalidations, key=str)),
                    trigger=case.trigger,
                    commit_phase=case.commit_phase,
                    operation=case.operation,
                    baseline=case.baseline,
                    persistence=case.persistence,
                    recompute=case.recompute,
                    row_scope=case.row_scope,
                )
            )
    return tuple(records)


@dataclass(frozen=True)
class _ProfileRuntime:
    backend: Any
    manifest: MutationCapabilityManifest
    plan: BoundBackendPlan
    placement: BufferPlacement
    hinge_position_index: int
    hinge_velocity_index: int


def _require_cuda_mjwarp() -> None:
    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.fail("mjwarp mandatory capability cases require an active CUDA Warp device")


def _state_contract(
    profile: ExecutionProfile,
    placement: BufferPlacement,
    row_shape: tuple[int, ...],
) -> BufferContract:
    device = profile is ExecutionProfile.DEVICE_RESIDENT
    return BufferContract(
        row_shape=row_shape,
        dtype="float32",
        layout=BufferLayout.C_CONTIGUOUS,
        placement=placement,
        owner=BufferOwner.BACKEND,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.BORROWED_UNTIL_MUTATION,
        dlpack_exportable=device,
    )


def _state_field(
    *,
    profile: ExecutionProfile,
    placement: BufferPlacement,
    key: str,
    entity_kind: StateEntityKind,
    field_kind: StateFieldKind,
    entity_ids: tuple[int, ...],
    row_shape: tuple[int, ...],
    frame: ReferenceFrame,
    unit: PhysicalUnit,
) -> StateFieldSpec:
    return StateFieldSpec(
        semantic_key=key,
        identity=BoundFieldIdentity(entity_kind, field_kind, entity_ids),
        frame=frame,
        unit=unit,
        buffer=_state_contract(profile, placement, row_shape),
    )


def _control_contract(
    profile: ExecutionProfile,
    placement: BufferPlacement,
    num_actuators: int,
) -> BufferContract:
    device = profile is ExecutionProfile.DEVICE_RESIDENT
    return BufferContract(
        row_shape=(num_actuators,),
        dtype="float32",
        layout=BufferLayout.C_CONTIGUOUS,
        placement=placement,
        owner=BufferOwner.RUNNER if device else BufferOwner.MANAGER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.UNTIL_STEP_COMPLETE,
        dlpack_exportable=device,
    )


def _build_runtime(profile: ExecutionProfile) -> _ProfileRuntime:
    from unilab.assets import ASSETS_ROOT_PATH

    placement = (
        BufferPlacement.host()
        if profile is ExecutionProfile.HOST_NUMPY
        else BufferPlacement.device("cuda", int(torch.cuda.current_device()))
    )
    backend = ProductionMjwarpBackend(
        SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")),
        _NUM_ENVS,
        0.02 / 3.0,
        base_name=_BASE,
    )
    assert type(backend) is ProductionMjwarpBackend
    assert type(backend).__module__ == "unilab.base.backend.mjwarp.backend"
    manifest_before_bind = backend.get_mutation_capability_manifest(profile)
    base_id = int(backend.get_body_ids((_BASE,))[0])
    all_dofs = tuple(range(backend.num_dof_vel))
    fields = (
        _state_field(
            profile=profile,
            placement=placement,
            key="root.position",
            entity_kind=StateEntityKind.ROOT,
            field_kind=StateFieldKind.POSITION,
            entity_ids=(base_id,),
            row_shape=(3,),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.METER,
        ),
        _state_field(
            profile=profile,
            placement=placement,
            key="root.orientation",
            entity_kind=StateEntityKind.ROOT,
            field_kind=StateFieldKind.ORIENTATION,
            entity_ids=(base_id,),
            row_shape=(4,),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.QUATERNION,
        ),
        _state_field(
            profile=profile,
            placement=placement,
            key="root.linear_velocity",
            entity_kind=StateEntityKind.ROOT,
            field_kind=StateFieldKind.LINEAR_VELOCITY,
            entity_ids=(base_id,),
            row_shape=(3,),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.METER_PER_SECOND,
        ),
        _state_field(
            profile=profile,
            placement=placement,
            key="root.angular_velocity",
            entity_kind=StateEntityKind.ROOT,
            field_kind=StateFieldKind.ANGULAR_VELOCITY,
            entity_ids=(base_id,),
            row_shape=(3,),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.RADIAN_PER_SECOND,
        ),
        _state_field(
            profile=profile,
            placement=placement,
            key="dof.position",
            entity_kind=StateEntityKind.DOF,
            field_kind=StateFieldKind.POSITION,
            entity_ids=all_dofs,
            row_shape=(len(all_dofs),),
            frame=ReferenceFrame.JOINT,
            unit=PhysicalUnit.RADIAN,
        ),
        _state_field(
            profile=profile,
            placement=placement,
            key="dof.angular_velocity",
            entity_kind=StateEntityKind.DOF,
            field_kind=StateFieldKind.ANGULAR_VELOCITY,
            entity_ids=all_dofs,
            row_shape=(len(all_dofs),),
            frame=ReferenceFrame.JOINT,
            unit=PhysicalUnit.RADIAN_PER_SECOND,
        ),
    )
    plan = backend.bind_task_io(
        BackendIORequirements(
            state_fields=fields,
            control=ControlSpec(
                "joint.position_target",
                _control_contract(profile, placement, backend.num_actuators),
                physics_substeps_per_control=1,
            ),
            execution_profile=profile,
        )
    )
    manifest_after_bind = backend.get_mutation_capability_manifest(profile)
    assert manifest_after_bind.fingerprint == manifest_before_bind.fingerprint
    assert manifest_after_bind.capabilities == manifest_before_bind.capabilities
    return _ProfileRuntime(
        backend=backend,
        manifest=manifest_after_bind,
        plan=plan,
        placement=placement,
        hinge_position_index=int(backend.get_joint_dof_pos_indices((_HINGE,))[0]),
        hinge_velocity_index=int(backend.get_joint_dof_vel_indices((_HINGE,))[0]),
    )


@pytest.fixture(scope="module")
def mjwarp_profile_runtimes() -> dict[ExecutionProfile, _ProfileRuntime]:
    _require_cuda_mjwarp()
    return {
        ExecutionProfile.HOST_NUMPY: _build_runtime(ExecutionProfile.HOST_NUMPY),
        ExecutionProfile.DEVICE_RESIDENT: _build_runtime(ExecutionProfile.DEVICE_RESIDENT),
    }


def _mutation_spec(case: _MandatoryCase, runtime: _ProfileRuntime) -> MutationSpec:
    record = case.record
    return MutationSpec(
        term_key=f"verify.{record.execution_profile.value}.{record.target_key}",
        target=MutationTargetSpec(
            target_key=record.target_key,
            target_kind=record.target_kind,
            entity_kind=record.entity_kind,
            field_kind=record.field_kind,
            selector=case.selector,
        ),
        trigger=record.trigger,
        commit_phase=record.commit_phase,
        operation=record.operation,
        baseline=record.baseline,
        persistence=record.persistence,
        recompute=record.recompute,
        value_template=BufferContract(
            row_shape=record.value_row_shape,
            dtype=record.value_dtype,
            layout=record.value_layout,
            placement=runtime.placement,
            owner=record.value_owner,
            mutability=record.value_mutability,
            lifetime=record.value_lifetime,
            dlpack_exportable=record.value_dlpack_exportable,
            address_stable=record.value_address_stable,
        ),
    )


def _selected_values(case: _MandatoryCase) -> np.ndarray:
    count = len(_SELECTED)
    order = np.arange(count, dtype=np.float32)
    target_key = case.record.target_key
    if target_key == "state.root.position":
        values = np.tile(np.asarray((0.24, -0.13, 1.08), dtype=np.float32), (count, 1))
        values[:, 0] += 0.015 * order
    elif target_key == "state.root.orientation":
        angle = 0.50 + 0.025 * order
        values = np.stack(
            (np.cos(angle / 2.0), np.zeros(count), np.sin(angle / 2.0), np.zeros(count)),
            axis=1,
        ).astype(np.float32)
    elif target_key == "state.root.linear_velocity":
        values = np.tile(np.asarray((0.62, -0.31, 0.24), dtype=np.float32), (count, 1))
        values[:, 0] += 0.02 * order
    elif target_key == "state.root.angular_velocity":
        values = np.tile(np.asarray((0.43, -0.51, 0.34), dtype=np.float32), (count, 1))
        values[:, 2] += 0.02 * order
    elif target_key == "state.dof.position":
        values = (0.44 + 0.02 * order)[:, None]
    elif target_key == "state.dof.angular_velocity":
        values = (-1.20 + 0.03 * order)[:, None]
    else:  # pragma: no cover - the independent registry is exhaustive.
        raise AssertionError(f"missing mandatory value fixture for {target_key!r}")
    return values[:, None, :]


def _host_mutation_batch(
    mutation_plan: Any,
    values: np.ndarray,
) -> tuple[RowSelection, TypedBackendMutationBatch]:
    rows = RowSelection.selected(_NUM_ENVS, _SELECTED)
    contract = mutation_plan.specs[0].value_buffer
    value = MutationValueBatch(
        plan=mutation_plan,
        field_index=0,
        rows=rows,
        buffer=BufferView(values, values.shape, contract),
    )
    return rows, TypedBackendMutationBatch(
        plan=mutation_plan,
        rows=rows,
        state=SimulationStateMutationBatch((value,)),
    )


def _device_mutation_batch(
    case: _MandatoryCase,
    runtime: _ProfileRuntime,
    mutation_plan: Any,
    selected_values: np.ndarray,
) -> tuple[RowSelection, DeviceResetMutationBatch]:
    rows = RowSelection.all(_NUM_ENVS)
    device = torch.device(f"cuda:{runtime.placement.device_index}")
    value_tensor = torch.zeros(
        (_NUM_ENVS, *mutation_plan.specs[0].value_buffer.row_shape),
        dtype=torch.float32,
        device=device,
    )
    value_tensor[list(_SELECTED)] = torch.as_tensor(selected_values, device=device)
    active_mask = torch.zeros((_NUM_ENVS,), dtype=torch.bool, device=device)
    active_mask[list(_SELECTED)] = True
    lease = DeviceBufferLease(f"mandatory-reset-{case.parameter_id}")
    completion = DeviceCompletion.record(
        placement=runtime.placement,
        owner_id=lease.owner_id,
        epoch=lease.epoch,
    )
    value_contract = mutation_plan.specs[0].value_buffer
    value_view = DeviceTensorView(
        tensor_handle=value_tensor,
        contract=value_contract,
        lease=lease,
        completion=completion,
    )
    value = MutationValueBatch(
        plan=mutation_plan,
        field_index=0,
        rows=rows,
        buffer=BufferView(value_view, tuple(value_tensor.shape), value_contract),
    )
    mutation = TypedBackendMutationBatch(
        plan=mutation_plan,
        rows=rows,
        state=SimulationStateMutationBatch((value,)),
    )
    mask_contract = BufferContract(
        row_shape=(),
        dtype="bool",
        layout=BufferLayout.C_CONTIGUOUS,
        placement=runtime.placement,
        owner=BufferOwner.MANAGER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.UNTIL_COMMIT,
        dlpack_exportable=True,
    )
    mask_view = DeviceTensorView(
        tensor_handle=active_mask,
        contract=mask_contract,
        lease=lease,
        completion=completion,
    )
    return rows, DeviceResetMutationBatch(
        plan=mutation_plan,
        rows=rows,
        mutation=mutation,
        active_mask=BufferView(mask_view, tuple(active_mask.shape), mask_contract),
    )


def _snapshot(runtime: _ProfileRuntime, state: StateBatch | None = None) -> dict[str, np.ndarray]:
    if state is None:
        state = runtime.backend.read_state_batch(
            runtime.plan,
            RowSelection.all(_NUM_ENVS),
            phase=StateBatchPhase.CURRENT,
        ).state
    snapshot: dict[str, np.ndarray] = {}
    for key in _STATE_KEYS:
        handle = state.buffer(key).handle
        if isinstance(handle, DeviceTensorView):
            handle.wait()
            snapshot[key] = handle.torch().detach().cpu().numpy().copy()
        else:
            snapshot[key] = np.asarray(handle).copy()
    return snapshot


def _reset_to_stand(runtime: _ProfileRuntime) -> dict[str, np.ndarray]:
    qpos = np.tile(runtime.backend.get_keyframe_qpos("stand"), (_NUM_ENVS, 1)).astype(np.float32)
    qvel = np.zeros((_NUM_ENVS, runtime.backend.get_init_qvel().size), dtype=np.float32)
    if runtime.plan.execution_profile is ExecutionProfile.DEVICE_RESIDENT:
        reset_device_state(
            backend=runtime.backend,
            plan=runtime.plan,
            placement=runtime.placement,
            base_name=_BASE,
            qpos=qpos,
            qvel=qvel,
        )
    else:
        runtime.backend.set_state(np.arange(_NUM_ENVS, dtype=np.int32), qpos, qvel)
    snapshot = _snapshot(runtime)
    for key in _STATE_KEYS:
        np.testing.assert_allclose(
            snapshot[key][list(_SELECTED)],
            snapshot[key][list(_PAIRED_CONTROLS)],
            atol=2.0e-5,
            rtol=2.0e-5,
        )
    return snapshot


def _target_projection(
    case: _MandatoryCase,
    runtime: _ProfileRuntime,
    snapshot: dict[str, np.ndarray],
) -> np.ndarray:
    values = snapshot[case.state_key]
    if case.record.target_key == "state.dof.position":
        return values[:, runtime.hinge_position_index : runtime.hinge_position_index + 1]
    if case.record.target_key == "state.dof.angular_velocity":
        return values[:, runtime.hinge_velocity_index : runtime.hinge_velocity_index + 1]
    return values


def _zero_control(runtime: _ProfileRuntime, *, owner_suffix: str) -> ControlBatch:
    contract = runtime.plan.control.buffer
    rows = RowSelection.all(_NUM_ENVS)
    if runtime.plan.execution_profile is ExecutionProfile.HOST_NUMPY:
        values = np.zeros((_NUM_ENVS, *contract.row_shape), dtype=np.float32)
        return ControlBatch(
            plan=runtime.plan,
            rows=rows,
            buffer=BufferView(values, values.shape, contract),
        )
    device = torch.device(f"cuda:{runtime.placement.device_index}")
    values = torch.zeros((_NUM_ENVS, *contract.row_shape), dtype=torch.float32, device=device)
    lease = DeviceBufferLease(f"mandatory-control-{owner_suffix}")
    completion = DeviceCompletion.record(
        placement=runtime.placement,
        owner_id=lease.owner_id,
        epoch=lease.epoch,
    )
    view = DeviceTensorView(
        tensor_handle=values,
        contract=contract,
        lease=lease,
        completion=completion,
    )
    return ControlBatch(
        plan=runtime.plan,
        rows=rows,
        buffer=BufferView(view, tuple(values.shape), contract),
    )


@pytest.mark.slow
def test_advertised_capabilities_equal_mandatory_parameter_cases(
    mjwarp_profile_runtimes: dict[ExecutionProfile, _ProfileRuntime],
) -> None:
    """Production support and executable mandatory cases form an exact bijection."""

    manifests = tuple(
        mjwarp_profile_runtimes[profile].manifest
        for profile in (ExecutionProfile.HOST_NUMPY, ExecutionProfile.DEVICE_RESIDENT)
    )
    advertised = tuple(record for manifest in manifests for record in _manifest_records(manifest))
    _require_exact_case_bijection(advertised, _MANDATORY_RECORDS)
    assert len(_MANDATORY_CASES) == 12
    assert len(advertised) == len(_MANDATORY_RECORDS) == 20


@pytest.mark.slow
@pytest.mark.parametrize("case", _MANDATORY_CASES, ids=lambda case: case.parameter_id)
def test_mjwarp_advertised_capability_case(
    case: _MandatoryCase,
    mjwarp_profile_runtimes: dict[ExecutionProfile, _ProfileRuntime],
) -> None:
    """Each advertised node commits selected rows and affects the next step."""

    runtime = mjwarp_profile_runtimes[case.record.execution_profile]
    before = _reset_to_stand(runtime)
    mutation_plan = runtime.backend.bind_mutation_plan((_mutation_spec(case, runtime),))
    assert mutation_plan.capability_manifest_fingerprint == runtime.manifest.fingerprint
    assert mutation_plan.specs[0].capability_fingerprint.startswith("mutation-capability-v1:")
    selected_values = _selected_values(case)
    if case.record.execution_profile is ExecutionProfile.HOST_NUMPY:
        rows, mutation_batch = _host_mutation_batch(mutation_plan, selected_values)
    else:
        rows, mutation_batch = _device_mutation_batch(
            case,
            runtime,
            mutation_plan,
            selected_values,
        )

    result = runtime.backend.reset_batch(runtime.plan, rows, mutation_batch=mutation_batch)
    assert result.reset_state.phase is StateBatchPhase.RESET
    assert result.reset_state.rows == rows
    assert result.diagnostics.counters.instrumentation_complete
    if case.record.execution_profile is ExecutionProfile.DEVICE_RESIDENT:
        counters = result.diagnostics.counters
        assert counters.host_to_device_transfers == 0
        assert counters.device_to_host_transfers == 0
        assert counters.global_synchronizations == 0

    immediate = _snapshot(runtime)
    expected = selected_values[:, 0, :]
    projection = _target_projection(case, runtime, immediate)
    np.testing.assert_allclose(
        projection[list(_SELECTED)],
        expected,
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    complement = sorted(set(range(_NUM_ENVS)) - set(_SELECTED))
    for key in _STATE_KEYS:
        np.testing.assert_allclose(
            immediate[key][complement],
            before[key][complement],
            atol=2.0e-5,
            rtol=2.0e-5,
        )

    terminal = runtime.backend.step_batch(
        runtime.plan,
        _zero_control(runtime, owner_suffix=case.parameter_id),
        nsteps=1,
    ).terminal_state
    after_step = _snapshot(runtime, terminal)
    stepped_target = _target_projection(case, runtime, after_step)
    selected_after = stepped_target[list(_SELECTED)]
    controls_after = stepped_target[list(_PAIRED_CONTROLS)]
    assert np.isfinite(selected_after).all()
    assert np.max(np.abs(selected_after - controls_after)) > 1.0e-3


def test_case_bijection_audit_rejects_missing_extra_and_duplicate_cases() -> None:
    missing = _MANDATORY_RECORDS[:-1]
    with pytest.raises(AssertionError, match="unadvertised mandatory parameter"):
        _require_exact_case_bijection(missing, _MANDATORY_RECORDS)

    extra = replace(
        _MANDATORY_RECORDS[0],
        mandatory_test_id=f"{_MANDATORY_NODE}[host_numpy-state.unregistered]",
        case_id="mjwarp.host_numpy.state.unregistered.reset",
        target_key="state.unregistered",
    )
    with pytest.raises(AssertionError, match="without mandatory parameters"):
        _require_exact_case_bijection((*_MANDATORY_RECORDS, extra), _MANDATORY_RECORDS)

    with pytest.raises(AssertionError, match="duplicate advertised mandatory test IDs"):
        _require_exact_case_bijection(
            (*_MANDATORY_RECORDS, _MANDATORY_RECORDS[0]),
            _MANDATORY_RECORDS,
        )
    with pytest.raises(AssertionError, match="duplicate mandatory mandatory test IDs"):
        _require_exact_case_bijection(
            _MANDATORY_RECORDS,
            (*_MANDATORY_RECORDS, _MANDATORY_RECORDS[0]),
        )


def test_case_bijection_audit_rejects_capability_metadata_tamper() -> None:
    tampered = replace(_MANDATORY_RECORDS[0], direct_fields=("data.qvel",))
    with pytest.raises(AssertionError, match="metadata differs"):
        _require_exact_case_bijection(
            (tampered, *_MANDATORY_RECORDS[1:]),
            _MANDATORY_RECORDS,
        )


@pytest.mark.slow
def test_manifest_rejects_profile_mix_duplicate_evidence_and_fingerprint_tamper(
    mjwarp_profile_runtimes: dict[ExecutionProfile, _ProfileRuntime],
) -> None:
    source = mjwarp_profile_runtimes[ExecutionProfile.HOST_NUMPY].manifest
    capabilities = list(source.capabilities)
    profile_index = next(
        index
        for index, capability in enumerate(capabilities)
        if capability.target_key == "state.root.position"
    )
    profile_capability = capabilities[profile_index]
    assert profile_capability.descriptor is not None
    profile_case = replace(
        profile_capability.descriptor.cases[0],
        execution_profile=ExecutionProfile.DEVICE_RESIDENT,
    )
    profile_descriptor = replace(profile_capability.descriptor, cases=(profile_case,))
    mixed = capabilities.copy()
    mixed[profile_index] = replace(profile_capability, descriptor=profile_descriptor)
    with pytest.raises(MutationContractError, match="mixes execution profiles"):
        MutationCapabilityManifest(
            backend_type="mjwarp",
            execution_profile=ExecutionProfile.HOST_NUMPY,
            capabilities=tuple(mixed),
        )

    first = capabilities[0]
    second = capabilities[1]
    assert first.descriptor is not None and second.descriptor is not None
    duplicate_case = replace(
        second.descriptor.cases[0],
        mandatory_test_id=first.descriptor.cases[0].mandatory_test_id,
    )
    duplicate_descriptor = replace(second.descriptor, cases=(duplicate_case,))
    duplicate = capabilities.copy()
    duplicate[1] = replace(second, descriptor=duplicate_descriptor)
    with pytest.raises(MutationContractError, match="mandatory test IDs must be globally unique"):
        MutationCapabilityManifest(
            backend_type="mjwarp",
            execution_profile=ExecutionProfile.HOST_NUMPY,
            capabilities=tuple(duplicate),
        )

    fingerprint_tamper = MutationCapabilityManifest(
        backend_type=source.backend_type,
        execution_profile=source.execution_profile,
        capabilities=source.capabilities,
    )
    object.__setattr__(fingerprint_tamper, "fingerprint", "mutation-capability-manifest-v1:bad")
    with pytest.raises(MutationContractError, match="fingerprint does not match"):
        fingerprint_tamper.require_valid()


@pytest.mark.slow
def test_manifest_rejects_profile_placement_mismatch(
    mjwarp_profile_runtimes: dict[ExecutionProfile, _ProfileRuntime],
) -> None:
    source = mjwarp_profile_runtimes[ExecutionProfile.HOST_NUMPY].manifest
    first = source.capabilities[0]
    device_value = replace(
        first.value_template,
        placement=BufferPlacement.device("cuda", int(torch.cuda.current_device())),
        dlpack_exportable=True,
    )
    mismatched = (replace(first, value_template=device_value), *source.capabilities[1:])
    with pytest.raises(MutationContractError, match="placement does not match profile"):
        MutationCapabilityManifest(
            backend_type="mjwarp",
            execution_profile=ExecutionProfile.HOST_NUMPY,
            capabilities=mismatched,
        )
