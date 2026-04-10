from __future__ import annotations

import subprocess
from pathlib import Path

BANNED_TRACKED_GLOBS = (
    "*.bak",
    "*.backup",
    "*.old",
    "*.orig",
    "*.rej",
    "*.temp",
    "*.tmp",
    "*~",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def tracked_hygiene_artifacts(root: Path | None = None) -> list[str]:
    resolved_root = root or repo_root()
    result = subprocess.run(
        ["git", "ls-files", *BANNED_TRACKED_GLOBS],
        cwd=resolved_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line and (resolved_root / line).exists()]


def missing_gitignore_patterns(root: Path | None = None) -> list[str]:
    resolved_root = root or repo_root()
    gitignore_lines = (resolved_root / ".gitignore").read_text(encoding="utf-8").splitlines()
    declared_patterns = {line.strip() for line in gitignore_lines if line.strip()}
    return [pattern for pattern in BANNED_TRACKED_GLOBS if pattern not in declared_patterns]
