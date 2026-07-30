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

MJWARP_STATE_RESET_CAPABILITY_ID = "state.qpos_qvel_reset"
_MANDATORY_TEST = (
    "tests/dr/test_mjwarp_capability_matrix.py::test_mjwarp_advertised_capability_case"
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


__all__ = ["MJWARP_STATE_RESET_CAPABILITY_ID", "mjwarp_state_reset_descriptor"]
