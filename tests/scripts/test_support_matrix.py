from __future__ import annotations

from pathlib import Path

from unilab.utils.support_matrix import EvidenceLevel, build_support_rows


def _row(entrypoint_label: str, task_slug: str):
    root = Path(__file__).resolve().parents[2]
    for row in build_support_rows(root):
        if row.entrypoint_label == entrypoint_label and row.task_slug == task_slug:
            return row
    raise AssertionError(f"Missing support row: {entrypoint_label} / {task_slug}")


def test_support_matrix_marks_go2_ppo_backends_as_tested():
    row = _row("PPO (torch)", "go2_joystick_flat")

    assert row.cells["mujoco"].level == EvidenceLevel.TESTED
    assert row.cells["motrix"].level == EvidenceLevel.TESTED


def test_support_matrix_marks_appo_go1_motrix_as_registered_only():
    row = _row("APPO (torch)", "go1_joystick_flat")

    assert row.cells["mujoco"].level == EvidenceLevel.TESTED
    assert row.cells["motrix"].level == EvidenceLevel.REGISTERED


def test_support_matrix_keeps_uncovered_mlx_tasks_at_configured():
    row = _row("PPO (mlx)", "g1_motion_tracking")

    assert row.cells["mujoco"].level == EvidenceLevel.CONFIGURED
    assert row.cells["motrix"].level == EvidenceLevel.CONFIGURED


def test_support_matrix_marks_sharpa_motrix_phase1_support():
    row = _row("PPO (torch)", "sharpa_inhand")

    assert row.cells["mujoco"].level == EvidenceLevel.TESTED
    assert row.cells["motrix"].level == EvidenceLevel.TESTED

    appo_row = _row("APPO (torch)", "sharpa_inhand")

    assert appo_row.cells["mujoco"].level == EvidenceLevel.TESTED
    assert appo_row.cells["motrix"].level == EvidenceLevel.REGISTERED
    allegro_appo_row = _row("APPO (torch)", "allegro_inhand")

    assert allegro_appo_row.cells["mujoco"].level == EvidenceLevel.TESTED
    assert allegro_appo_row.cells["motrix"].level == EvidenceLevel.TESTED
