"""Verified field-level mutation descriptors for the MuJoCo host backend."""

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

_WRENCH_TEST = (
    "tests/base/test_mujoco_typed_mutation.py::"
    "test_mujoco_typed_wrench_commits_to_next_step_with_selected_row_isolation"
)
_SELECTED_RESET_TEST = (
    "tests/base/test_mujoco_typed_reset.py::"
    "test_mujoco_typed_reset_commits_selected_hinge_state_and_exposes_reset_oracle"
)
_FULL_RESET_TEST = (
    "tests/base/test_mujoco_typed_reset.py::"
    "test_mujoco_typed_reset_commits_full_floating_root_and_hinge_slice"
)
_COLD_RESET_TEST = (
    "tests/base/test_mujoco_typed_reset.py::"
    "test_mujoco_cold_bound_reset_buffers_commit_complete_state_without_value_wrappers"
)

_RESET_EVIDENCE = {
    "state.dof.position": f"{_SELECTED_RESET_TEST}[float32]",
    "state.dof.angular_velocity": f"{_SELECTED_RESET_TEST}[float64]",
    "state.root.position": f"{_FULL_RESET_TEST}[float32]",
    "state.root.orientation": f"{_FULL_RESET_TEST}[float64]",
    "state.root.linear_velocity": f"{_COLD_RESET_TEST}[float32]",
    "state.root.angular_velocity": f"{_COLD_RESET_TEST}[float64]",
}


def _case(
    *,
    target_key: str,
    mandatory_test_id: str,
    trigger: MutationTrigger,
    phase: MutationCommitPhase,
    baseline: MutationBaseline,
    persistence: MutationPersistence,
    recompute: MutationRecomputeLevel,
) -> MutationCapabilityCase:
    return MutationCapabilityCase(
        case_id=f"mujoco.host_numpy.{target_key}.{trigger.value}",
        mandatory_test_id=mandatory_test_id,
        execution_profile=ExecutionProfile.HOST_NUMPY,
        trigger=trigger,
        commit_phase=phase,
        operation=MutationOperation.SET,
        baseline=baseline,
        persistence=persistence,
        recompute=recompute,
        row_scope=MutationCapabilityRowScope.SELECTED_ROWS,
    )


def mujoco_host_mutation_descriptor(*, target_key: str) -> MutationCapabilityDescriptor:
    """Return the exact host case and effect-test identity for one target."""

    if target_key == "wrench.body.force":
        cases = tuple(
            _case(
                target_key=target_key,
                mandatory_test_id=f"{_WRENCH_TEST}[float64-{trigger.value}]",
                trigger=trigger,
                phase=MutationCommitPhase.PRE_PHYSICS,
                baseline=MutationBaseline.CURRENT,
                persistence=MutationPersistence.ONE_STEP,
                recompute=MutationRecomputeLevel.NONE,
            )
            for trigger in (MutationTrigger.INTERVAL, MutationTrigger.STEP)
        )
        direct_fields = ("data.xfrc_applied",)
        capability_id = "external_wrench.body.force"
    else:
        try:
            mandatory_test_id = _RESET_EVIDENCE[target_key]
        except KeyError as exc:
            raise ValueError(f"unsupported MuJoCo mutation target {target_key!r}") from exc
        cases = (
            _case(
                target_key=target_key,
                mandatory_test_id=mandatory_test_id,
                trigger=MutationTrigger.RESET,
                phase=MutationCommitPhase.RESET,
                baseline=MutationBaseline.DEFAULT,
                persistence=MutationPersistence.EPISODE,
                recompute=MutationRecomputeLevel.KINEMATICS,
            ),
        )
        direct_fields = (
            ("data.qpos",)
            if target_key
            in {
                "state.root.position",
                "state.root.orientation",
                "state.dof.position",
            }
            else ("data.qvel",)
        )
        capability_id = "state.qpos_qvel_reset"

    return MutationCapabilityDescriptor(
        capability_id=capability_id,
        direct_fields=direct_fields,
        derived_fields=(),
        storage_kind=MutationFieldStorageKind.DATA_NATIVE,
        graph_impact=MutationGraphImpact.STABLE_ADDRESS,
        graph_invalidations=frozenset(),
        cases=cases,
    )


__all__ = ["mujoco_host_mutation_descriptor"]
