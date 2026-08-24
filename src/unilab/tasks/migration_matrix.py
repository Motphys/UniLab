"""Production task migration status and closeout ownership.

The matrix is deliberately small and explicit.  It is an audit boundary for
the grouped #1042 migration work; it does not provide a second task runtime or
translate task configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MigrationStatus = Literal["Compatible", "Adapted"]
MigrationTarget = Literal["complete", "mba", "compatibility"]


@dataclass(frozen=True)
class TaskMigrationRecord:
    task_name: str
    family: str
    status: MigrationStatus
    target: MigrationTarget
    rationale: str
    next_step: str


_MBA_TASKS = frozenset(
    {
        "A2JoystickFlat",
        "AllegroInhandRotation",
        "AllegroInhandRotationGrasp",
        "Go1JoystickFlat",
        "Go2FootStand",
        "Go2JoystickFlat",
        "Go2WJoystickFlat",
        "StewartBalance",
    }
)

_ROUGH_TASKS = frozenset(
    {
        "Go1JoystickRough",
        "Go2JoystickRough",
        "Go2WJoystickRough",
    }
)

_G1_LOCOMOTION_TASKS = frozenset(
    {
        "G1WalkFlat",
        "G1WalkRough",
        "G1Walk23DofFlat",
        "G1Walk23DofRough",
    }
)

_T800_LOCOMOTION_TASKS = frozenset({"T800WalkFlat"})

_CUSTOM_COMPAT_TASKS = frozenset(
    {
        "Go2ArmManipLoco",
        "SharpaInhandRotation",
        "SharpaInhandRotationGrasp",
    }
)

_MOTION_CORE_TASKS = frozenset(
    {
        "G1MotionTracking",
        "G1MotionTracking23Dof",
        "G1MotionTracking23DofDeploy",
        "G1MotionTrackingDeploy",
        "G1MotionTrackingSAC",
        "G1MotionTrackingSAC23Dof",
    }
)

_MOTION_TASKS = frozenset(
    {
        "G1BoxTracking",
        "G1BoxTracking23Dof",
        "G1ClimbTracking",
        "G1ClimbTracking23Dof",
        "G1FlipTracking",
        "G1FlipTracking23Dof",
        "G1FlipTrackingSAC",
        "G1FlipTrackingSAC23Dof",
        "G1WallFlipTracking",
        "G1WallFlipTracking23Dof",
        "G1WallFlipTrackingSAC",
        "G1WallFlipTrackingSAC23Dof",
        "G1WBTObs",
        "G1WBTObs23Dof",
        "X2WallFlipTracking",
    }
)

PRODUCTION_TASK_NAMES = frozenset(
    _MBA_TASKS
    | _ROUGH_TASKS
    | _G1_LOCOMOTION_TASKS
    | _T800_LOCOMOTION_TASKS
    | _CUSTOM_COMPAT_TASKS
    | _MOTION_CORE_TASKS
    | _MOTION_TASKS
)


def migration_record(task_name: str) -> TaskMigrationRecord:
    """Return the closeout status for one registered production task.

    Unknown names fail closed so adding a production registration requires an
    explicit migration decision and cannot silently escape the audit.
    """

    if task_name in _MBA_TASKS:
        return TaskMigrationRecord(
            task_name,
            "manager_based",
            "Compatible",
            "complete",
            "Hydra owner YAML materializes the canonical NumPy Manager-Based runtime.",
            "Keep the manager contract and regression evidence current.",
        )
    if task_name in _ROUGH_TASKS:
        return TaskMigrationRecord(
            task_name,
            "quadruped_rough",
            "Compatible",
            "complete",
            "Hydra owners materialize shared terrain, height-scan, reset, and curriculum manager terms on the canonical runtime.",
            "Keep the shared rough-family contract and both backend owners in sync. "
            "Known intentional divergence from the legacy rough env (recorded 2026-08): "
            "the legacy reward terms feet_gait, feet_air_time(+variance), "
            "feet_contact_without_cmd, feet_height_body, feet_slide, contact_forces, "
            "undesired_contacts, joint_mirror, joint_power, joint_torques_l2, "
            "joint_acc_l2(+wheel) have no manager port and are not part of the "
            "manager-based rough reward set.",
        )
    if task_name in _G1_LOCOMOTION_TASKS:
        return TaskMigrationRecord(
            task_name,
            "g1_locomotion",
            "Compatible",
            "complete",
            "Hydra owners materialize biped gait, sensor, command, and penalty-curriculum manager terms on the canonical runtime.",
            "Keep the manager contract and regression evidence current.",
        )
    if task_name in _T800_LOCOMOTION_TASKS:
        return TaskMigrationRecord(
            task_name,
            "t800_locomotion",
            "Compatible",
            "complete",
            "Hydra owner YAML materializes the T800 walk-flat task on the canonical NumPy Manager-Based runtime.",
            "Keep the Manager-Based owner and cold-path asset resolver contract current.",
        )
    if task_name in _CUSTOM_COMPAT_TASKS:
        family = "go2_arm" if task_name == "Go2ArmManipLoco" else "sharpa"
        return TaskMigrationRecord(
            task_name,
            family,
            "Adapted",
            "compatibility",
            "Custom IK/history or tactile/contact/cache behavior is retained behind one frozen adapter.",
            "Keep Hydra/Registry ownership single; migrate only when the formal capability exists.",
        )
    if task_name in _MOTION_CORE_TASKS:
        return TaskMigrationRecord(
            task_name,
            "motion_tracking",
            "Compatible",
            "complete",
            "Hydra owner YAML materializes task-owned NumPy motion manager terms on the canonical runtime.",
            "Keep PPO, APPO, and SAC owners aligned with the shared motion manager contract.",
        )
    if task_name in _MOTION_TASKS:
        return TaskMigrationRecord(
            task_name,
            "motion_tracking",
            "Compatible",
            "complete",
            "Hydra profile owners specialize the shared NumPy motion managers without a legacy runtime.",
            "Keep profile scene, motion, observation, reward, and termination declarations aligned.",
        )
    raise KeyError(f"Task '{task_name}' has no #1042 migration-matrix entry")


def migration_records(
    task_names: list[str] | tuple[str, ...] | set[str],
) -> tuple[TaskMigrationRecord, ...]:
    """Return records in deterministic task-name order."""

    return tuple(migration_record(name) for name in sorted(task_names))


__all__ = [
    "MigrationStatus",
    "MigrationTarget",
    "PRODUCTION_TASK_NAMES",
    "TaskMigrationRecord",
    "migration_record",
    "migration_records",
]
