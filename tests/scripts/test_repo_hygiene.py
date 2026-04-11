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
