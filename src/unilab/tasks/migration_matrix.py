"""Production task migration status and closeout ownership.

The matrix is deliberately small and explicit.  It is an audit boundary for
the grouped #1042 migration work; it does not provide a second task runtime or
translate task configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MigrationStatus = Literal["Compatible", "Adapted"]
MigrationTarget = Literal["complete", "mba"]


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
        "AllegroInhandRotation",
        "AllegroInhandRotationGrasp",
        # #1534 starts directly on the canonical manager runtime; no legacy seam.
        "FR3JointTarget",
        "Go2JoystickFlat",
        "StewartBalance",
    }
)

_G1_LOCOMOTION_TASKS = frozenset(
    {
        "G1WalkFlat",
    }
)

_MOTION_CORE_TASKS = frozenset(
    {
        "G1MotionTracking",
        "G1MotionTrackingSAC",
    }
)

_MOTION_TASKS = frozenset(
    {
        "G1BoxTracking",
        "G1FlipTracking",
        "G1FlipTrackingSAC",
        "G1WBTObs",
        "X2WallFlipTracking",
    }
)

PRODUCTION_TASK_NAMES = frozenset(
    _MBA_TASKS | _G1_LOCOMOTION_TASKS | _MOTION_CORE_TASKS | _MOTION_TASKS
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
    if task_name in _G1_LOCOMOTION_TASKS:
        return TaskMigrationRecord(
            task_name,
            "g1_locomotion",
            "Compatible",
            "complete",
            "Hydra owners materialize biped gait, sensor, command, and penalty-curriculum manager terms on the canonical runtime.",
            "Keep the manager contract and regression evidence current.",
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
