"""Verified typed-reset capability descriptors for the ``mjwarp`` adapter."""

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
    "tests/manager/test_cross_backend_plan.py::test_mjwarp_advertised_reset_capability"
)
_QPOS_TARGETS = frozenset({"state.root.position", "state.root.orientation", "state.dof.position"})
_QVEL_TARGETS = frozenset(
    {
        "state.root.linear_velocity",
        "state.root.angular_velocity",
        "state.dof.angular_velocity",
    }
)


def mjwarp_host_reset_descriptor(*, target_key: str) -> MutationCapabilityDescriptor:
    """Describe one selected-row host reset translated to adapter-owned storage."""

    if target_key in _QPOS_TARGETS:
        direct_fields = ("data.qpos",)
    elif target_key in _QVEL_TARGETS:
        direct_fields = ("data.qvel",)
    else:
        raise ValueError(f"unsupported mjwarp state reset target {target_key!r}")
    return MutationCapabilityDescriptor(
        capability_id=MJWARP_STATE_RESET_CAPABILITY_ID,
        direct_fields=direct_fields,
        derived_fields=(),
        storage_kind=MutationFieldStorageKind.DATA_NATIVE,
        graph_impact=MutationGraphImpact.STABLE_ADDRESS,
        graph_invalidations=frozenset(),
        cases=(
            MutationCapabilityCase(
                case_id=f"mjwarp.host_numpy.{target_key}.reset",
                mandatory_test_id=f"{_MANDATORY_TEST}[{target_key}]",
                execution_profile=ExecutionProfile.HOST_NUMPY,
                trigger=MutationTrigger.RESET,
                commit_phase=MutationCommitPhase.RESET,
                operation=MutationOperation.SET,
                baseline=MutationBaseline.DEFAULT,
                persistence=MutationPersistence.EPISODE,
                recompute=MutationRecomputeLevel.KINEMATICS,
                row_scope=MutationCapabilityRowScope.SELECTED_ROWS,
            ),
        ),
    )


__all__ = ["MJWARP_STATE_RESET_CAPABILITY_ID", "mjwarp_host_reset_descriptor"]
