from __future__ import annotations

from pathlib import Path

from tests.scripts import doc_checks


def test_documentation_files_match_current_repo_contracts():
    root = Path(__file__).resolve().parents[2]
    errors = doc_checks.collect_doc_errors(root)
    assert errors == []


def test_check_training_entrypoint_semantics_flags_issue_204_patterns():
    root = Path(__file__).resolve().parents[2]
    doc_path = root / "README.md"
    content = """
uv run python scripts/train_rsl_rl.py task=go1_joystick_flat
uv run python scripts/train_rsl_rl.py task=go1_joystick_flat/mujoco training.load_run=2026-01-01
Training logs are saved to logs/rsl_rl_train/MyTask/.
"""

    errors = doc_checks.check_training_entrypoint_semantics(content, doc_path, root)

    assert any("training.load_run" in error for error in errors)
    assert any("logs/rsl_rl_train/" in error for error in errors)
    assert any("task=go1_joystick_flat" in error for error in errors)


def test_check_training_entrypoint_semantics_accepts_current_patterns():
    root = Path(__file__).resolve().parents[2]
    doc_path = root / "README.md"
    content = """
uv run python scripts/train_rsl_rl.py task=go1_joystick_flat/mujoco algo.load_run=2026-01-01
uv run python scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco
Logs live under logs/<algo.algo_log_name>/<task>/.
"""

    errors = doc_checks.check_training_entrypoint_semantics(content, doc_path, root)

    assert errors == []


def test_check_script_references_ignores_tests_paths():
    root = Path(__file__).resolve().parents[2]
    doc_path = root / "CONTRIBUTING.md"
    content = "Run uv run pytest tests/scripts/test_check_docs.py -q"

    errors = doc_checks.check_script_references(content, doc_path, root)

    assert errors == []


def test_collect_doc_errors_scans_issue_templates_for_hydra_semantics(tmp_path):
    issue_template = tmp_path / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml"
    issue_template.parent.mkdir(parents=True)
    issue_template.write_text(
        "placeholder: |\n  uv run python scripts/train_offpolicy.py algo=sac task=g1_walk_flat ...\n",
        encoding="utf-8",
    )
    script_path = tmp_path / "scripts" / "train_offpolicy.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("", encoding="utf-8")

    errors = doc_checks.collect_doc_errors(tmp_path)

    assert any("task=g1_walk_flat" in error for error in errors)


def test_collect_doc_errors_scans_issue_templates_for_script_paths(tmp_path):
    issue_template = tmp_path / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml"
    issue_template.parent.mkdir(parents=True)
    issue_template.write_text(
        "placeholder: |\n  uv run python scripts/missing_entrypoint.py task=go1_joystick_flat/mujoco\n",
        encoding="utf-8",
    )

    errors = doc_checks.collect_doc_errors(tmp_path)

    assert any("Script not found: scripts/missing_entrypoint.py" in error for error in errors)
