from __future__ import annotations

import pytest

from unilab.base import registry
from unilab.tasks.migration_matrix import (
    PRODUCTION_TASK_NAMES,
    migration_record,
    migration_records,
)


def test_registered_tasks_have_explicit_migration_records() -> None:
    registry.ensure_registries()
    registered = registry.list_registered_envs()
    records = migration_records(set(PRODUCTION_TASK_NAMES))

    assert PRODUCTION_TASK_NAMES <= registered.keys()
    assert {record.task_name for record in records} == set(PRODUCTION_TASK_NAMES)
    assert len(records) == 40
    assert sum(record.status == "Compatible" for record in records) == 37
    assert sum(record.target == "compatibility" for record in records) == 3


@pytest.mark.parametrize(
    ("task_name", "family", "target", "status"),
    [
        ("Go2ArmManipLoco", "go2_arm", "compatibility", "Adapted"),
        ("SharpaInhandRotation", "sharpa", "compatibility", "Adapted"),
        ("G1MotionTracking", "motion_tracking", "complete", "Compatible"),
        ("G1WBTObs", "motion_tracking", "complete", "Compatible"),
        ("X2WallFlipTracking", "motion_tracking", "complete", "Compatible"),
        ("G1WalkRough", "g1_locomotion", "complete", "Compatible"),
        ("T800WalkFlat", "t800_locomotion", "complete", "Compatible"),
        ("Go2JoystickRough", "quadruped_rough", "complete", "Compatible"),
    ],
)
def test_matrix_records_high_risk_families(
    task_name: str, family: str, target: str, status: str
) -> None:
    record = migration_record(task_name)
    assert record.family == family
    assert record.target == target
    assert record.status == status


def test_unknown_task_fails_closed() -> None:
    with pytest.raises(KeyError, match="no #1042 migration-matrix entry"):
        migration_record("NewTaskWithoutDecision")
