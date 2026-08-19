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
    assert len(records) == 39
    assert sum(record.status == "Compatible" for record in records) == 8
    assert sum(record.target == "compatibility" for record in records) == 3


@pytest.mark.parametrize(
    ("task_name", "family", "target"),
    [
        ("Go2ArmManipLoco", "go2_arm", "compatibility"),
        ("SharpaInhandRotation", "sharpa", "compatibility"),
        ("G1MotionTracking", "motion_tracking", "mba"),
        ("G1WalkRough", "g1_locomotion", "mba"),
        ("Go2JoystickRough", "quadruped_rough", "mba"),
    ],
)
def test_matrix_records_high_risk_families(task_name: str, family: str, target: str) -> None:
    record = migration_record(task_name)
    assert record.family == family
    assert record.target == target
    assert record.status == "Adapted"


def test_unknown_task_fails_closed() -> None:
    with pytest.raises(KeyError, match="no #1042 migration-matrix entry"):
        migration_record("NewTaskWithoutDecision")
