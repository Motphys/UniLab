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

_CUSTOM_COMPAT_TASKS = frozenset(
    {
        "Go2ArmManipLoco",
        "SharpaInhandRotation",
        "SharpaInhandRotationGrasp",
    }
)


def _motion_task(task_name: str) -> bool:
    return task_name.startswith(("G1", "X2")) and (
        "Tracking" in task_name or task_name in {"G1WBTObs", "G1WBTObs23Dof"}
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
            "Adapted",
            "mba",
            "Terrain and height-scan terms depend on the pending raycaster capability boundary.",
            "Migrate as one rough-family PR; use the compatibility seam only if a new public capability is required.",
        )
    if task_name in _G1_LOCOMOTION_TASKS:
        return TaskMigrationRecord(
            task_name,
            "g1_locomotion",
            "Adapted",
            "mba",
            "The locomotion equations are reusable, but the 29/23-DoF sensor and gait surface is not yet manager-owned.",
            "Migrate flat and rough variants together and delete the legacy owner.",
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
    if _motion_task(task_name):
        return TaskMigrationRecord(
            task_name,
            "motion_tracking",
            "Adapted",
            "mba",
            "Stateful motion loading and profile-specific tracking terms need a grouped manager port.",
            "Migrate the shared engine and all profiles together; stop on new backend contracts.",
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
    "TaskMigrationRecord",
    "migration_record",
    "migration_records",
]
