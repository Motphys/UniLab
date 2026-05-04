from __future__ import annotations

from pathlib import Path

from tests.scripts import repo_hygiene_checks


def test_repo_does_not_track_backup_or_temporary_artifacts():
    root = Path(__file__).resolve().parents[2]

    tracked_artifacts = repo_hygiene_checks.tracked_hygiene_artifacts(root)

    assert tracked_artifacts == []


def test_gitignore_covers_banned_backup_and_temporary_patterns():
    root = Path(__file__).resolve().parents[2]

    missing_patterns = repo_hygiene_checks.missing_gitignore_patterns(root)

    assert missing_patterns == []


def test_script_usage_examples_do_not_invoke_python_for_repo_scripts():
    root = Path(__file__).resolve().parents[2]

    errors = repo_hygiene_checks.source_command_anti_patterns(root)

    assert errors == []


def test_source_command_anti_patterns_flags_python_script_invocation(tmp_path):
    script_path = tmp_path / "scripts" / "tool.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text(
        '"""Usage:\n    uv run python scripts/tool.py\n"""\n',
        encoding="utf-8",
    )

    errors = repo_hygiene_checks.source_command_anti_patterns(tmp_path)

    assert any("Use `uv run scripts/...`" in error for error in errors)
