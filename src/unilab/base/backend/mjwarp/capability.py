"""Canonical field-level capability descriptors owned by the ``mjwarp`` backend."""

from __future__ import annotations

from ..batch import ExecutionProfile
from ..mutation import (
    MutationBaseline,
    MutationCapabilityCase,
    MutationCapabilityDescriptor,
    MutationCapabilityRowScope,
    MutationCommitPhase,
    MutationFieldStorageKind,
    MutationGraphImpact,
    MutationOperation,
    MutationPersistence,
    MutationRecomputeLevel,
    MutationTrigger,
)
from .materialization import MJWARP_MODEL_INVALIDATIONS

MJWARP_STATE_RESET_CAPABILITY_ID = "state.qpos_qvel_reset"
MJWARP_ACTUATOR_PD_CAPABILITY_ID = "actuator.position_servo_pd_gain"
_MANDATORY_TEST = (
    "tests/dr/test_mjwarp_capability_matrix.py::test_mjwarp_advertised_capability_case"
)
_MANDATORY_MODEL_TEST = (
    "tests/dr/test_mjwarp_model_mutation.py::test_mjwarp_advertised_model_capability_case"
)
_QPOS_TARGETS = frozenset(
    {
        "state.root.position",
        "state.root.orientation",
        "state.dof.position",
    }
)
_QVEL_TARGETS = frozenset(
    {
        "state.root.linear_velocity",
        "state.root.angular_velocity",
        "state.dof.angular_velocity",
    }
)


def mjwarp_state_reset_descriptor(
    *,
    target_key: str,
    execution_profile: ExecutionProfile,
) -> MutationCapabilityDescriptor:
    """Describe one existing qpos/qvel reset target and its mandatory oracle."""

    if target_key in _QPOS_TARGETS:
        direct_fields = ("data.qpos",)
    elif target_key in _QVEL_TARGETS:
        direct_fields = ("data.qvel",)
    else:
        raise ValueError(f"unsupported mjwarp state reset target {target_key!r}")
    case_id = f"mjwarp.{execution_profile.value}.{target_key}.reset"
    parameter_id = f"{execution_profile.value}-{target_key}"
    case = MutationCapabilityCase(
        case_id=case_id,
        mandatory_test_id=f"{_MANDATORY_TEST}[{parameter_id}]",
        execution_profile=execution_profile,
        trigger=MutationTrigger.RESET,
        commit_phase=MutationCommitPhase.RESET,
        operation=MutationOperation.SET,
        baseline=MutationBaseline.DEFAULT,
        persistence=MutationPersistence.EPISODE,
        recompute=MutationRecomputeLevel.KINEMATICS,
        row_scope=MutationCapabilityRowScope.SELECTED_ROWS,
    )
    return MutationCapabilityDescriptor(
        capability_id=MJWARP_STATE_RESET_CAPABILITY_ID,
        direct_fields=direct_fields,
        derived_fields=(),
        storage_kind=MutationFieldStorageKind.DATA_NATIVE,
        graph_impact=MutationGraphImpact.STABLE_ADDRESS,
        graph_invalidations=frozenset(),
        cases=(case,),
    )


def mjwarp_actuator_pd_descriptor(*, target_key: str) -> MutationCapabilityDescriptor:
    """Describe verified device-resident position-servo gain mutations."""

    direct_fields: tuple[str, ...]
    if target_key == "actuator.pd_stiffness":
        direct_fields = ("actuator_biasprm", "actuator_gainprm")
    elif target_key == "actuator.pd_damping":
        direct_fields = ("actuator_biasprm",)
    else:
        raise ValueError(f"unsupported mjwarp actuator PD target {target_key!r}")
    cases = tuple(
        sorted(
            (
                MutationCapabilityCase(
                    case_id=(f"mjwarp.device_resident.{target_key}.reset.{operation.value}"),
                    mandatory_test_id=(
                        f"{_MANDATORY_MODEL_TEST}[device_resident-{target_key}-{operation.value}]"
                    ),
                    execution_profile=ExecutionProfile.DEVICE_RESIDENT,
                    trigger=MutationTrigger.RESET,
                    commit_phase=MutationCommitPhase.RESET,
                    operation=operation,
                    baseline=MutationBaseline.DEFAULT,
                    persistence=MutationPersistence.EPISODE,
                    recompute=MutationRecomputeLevel.NONE,
                    row_scope=MutationCapabilityRowScope.SELECTED_ROWS,
                )
                for operation in (MutationOperation.SET, MutationOperation.SCALE)
            ),
            key=lambda case: case.case_id,
        )
    )
    return MutationCapabilityDescriptor(
        capability_id=MJWARP_ACTUATOR_PD_CAPABILITY_ID,
        direct_fields=direct_fields,
        derived_fields=(),
        storage_kind=MutationFieldStorageKind.MODEL_FIELD_EXPANSION,
        graph_impact=MutationGraphImpact.RECAPTURE_REQUIRED,
        graph_invalidations=frozenset(MJWARP_MODEL_INVALIDATIONS),
        cases=cases,
    )


__all__ = [
    "MJWARP_ACTUATOR_PD_CAPABILITY_ID",
    "MJWARP_STATE_RESET_CAPABILITY_ID",
    "mjwarp_actuator_pd_descriptor",
    "mjwarp_state_reset_descriptor",
]
