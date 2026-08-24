from __future__ import annotations

from pathlib import Path

from scripts.tools.support_matrix import BACKENDS, EvidenceLevel, build_support_rows


def _row(entrypoint_label: str, task_slug: str):
    root = Path(__file__).resolve().parents[2]
    for row in build_support_rows(root):
        if row.entrypoint_label == entrypoint_label and row.task_slug == task_slug:
            return row
    raise AssertionError(f"Missing support row: {entrypoint_label} / {task_slug}")


def test_support_matrix_marks_go2_ppo_backends_as_tested():
    row = _row("PPO (torch)", "go2_joystick_flat")

    assert row.cells["mujoco"].level == EvidenceLevel.TESTED
    assert row.cells["mjwarp"].level == EvidenceLevel.MISSING
    assert row.cells["motrix"].level == EvidenceLevel.TESTED


def test_support_matrix_marks_t800_ppo_and_sac_mujoco_tested_only():
    """T800 owner evidence is limited to the configured MuJoCo paths."""
    ppo_row = _row("PPO (torch)", "t800_walk_flat")
    sac_row = _row("SAC (torch)", "t800_walk_flat")

    for row in (ppo_row, sac_row):
        assert row.cells["mujoco"].level == EvidenceLevel.TESTED
        assert row.cells["mjwarp"].level == EvidenceLevel.MISSING
        assert row.cells["motrix"].level == EvidenceLevel.MISSING


def test_support_matrix_marks_validated_g1_mjwarp_entrypoints_as_tested():
    torch_row = _row("PPO (torch)", "g1_walk_flat")
    sac_row = _row("SAC (torch)", "g1_walk_flat")

    assert BACKENDS == ("mujoco", "mjwarp", "motrix")
    assert torch_row.cells["mjwarp"].level == EvidenceLevel.TESTED
    assert sac_row.cells["mjwarp"].level == EvidenceLevel.TESTED


def test_support_matrix_does_not_promote_unvalidated_mjwarp_entries():
    rows = build_support_rows(Path(__file__).resolve().parents[2])

    tested = {
        (row.entrypoint_label, row.task_slug)
        for row in rows
        if row.cells["mjwarp"].level >= EvidenceLevel.TESTED
    }
    assert tested == {
        ("PPO (torch)", "g1_walk_flat"),
        ("SAC (torch)", "g1_walk_flat"),
    }
    appo_row = _row("APPO (torch)", "g1_walk_flat")
    assert appo_row.cells["mjwarp"].level == EvidenceLevel.REGISTERED


def test_support_matrix_marks_appo_go1_backends_as_tested():
    row = _row("APPO (torch)", "go1_joystick_flat")

    assert row.cells["mujoco"].level == EvidenceLevel.TESTED
    assert row.cells["motrix"].level == EvidenceLevel.TESTED


def test_support_matrix_marks_sharpa_motrix_phase1_support():
    row = _row("PPO (torch)", "sharpa_inhand")

    assert row.cells["mujoco"].level == EvidenceLevel.TESTED
    assert row.cells["motrix"].level == EvidenceLevel.TESTED

    appo_row = _row("APPO (torch)", "sharpa_inhand")

    assert appo_row.cells["mujoco"].level == EvidenceLevel.TESTED
    assert appo_row.cells["motrix"].level == EvidenceLevel.TESTED
    allegro_appo_row = _row("APPO (torch)", "allegro_inhand")

    assert allegro_appo_row.cells["mujoco"].level == EvidenceLevel.TESTED
    assert allegro_appo_row.cells["motrix"].level == EvidenceLevel.TESTED
