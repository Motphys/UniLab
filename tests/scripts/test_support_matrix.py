from __future__ import annotations

from pathlib import Path

from unilab.docs.support_matrix import EvidenceLevel, build_support_rows


def _row(entrypoint_label: str, task_slug: str):
    root = Path(__file__).resolve().parents[2]
    for row in build_support_rows(root):
        if row.entrypoint_label == entrypoint_label and row.task_slug == task_slug:
            return row
    raise AssertionError(f"Missing support row: {entrypoint_label} / {task_slug}")


def test_support_matrix_marks_go2_ppo_backends_as_tested():
    row = _row("PPO (torch)", "go2_joystick")

    assert row.cells["mujoco"].level == EvidenceLevel.TESTED
    assert row.cells["motrix"].level == EvidenceLevel.TESTED


def test_support_matrix_marks_appo_go1_motrix_as_registered_only():
    row = _row("APPO (torch)", "go1_joystick")

    assert row.cells["mujoco"].level == EvidenceLevel.TESTED
    assert row.cells["motrix"].level == EvidenceLevel.REGISTERED


def test_support_matrix_keeps_uncovered_mlx_tasks_at_configured():
    row = _row("PPO (mlx)", "g1_motion_tracking")

    assert row.cells["mujoco"].level == EvidenceLevel.CONFIGURED
    assert row.cells["motrix"].level == EvidenceLevel.CONFIGURED
