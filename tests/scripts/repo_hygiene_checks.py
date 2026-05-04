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

BANNED_SOURCE_SNIPPETS = {
    "uv run python scripts/": "Use `uv run scripts/...` for repository scripts.",
}


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


def source_command_anti_patterns(root: Path | None = None) -> list[str]:
    resolved_root = root or repo_root()
    errors: list[str] = []
    for script_path in sorted((resolved_root / "scripts").rglob("*.py")):
        rel_path = script_path.relative_to(resolved_root)
        content = script_path.read_text(encoding="utf-8")
        for line_no, line in enumerate(content.splitlines(), start=1):
            for snippet, message in BANNED_SOURCE_SNIPPETS.items():
                if snippet in line:
                    errors.append(f"{rel_path}:{line_no}: {message}")
    return errors
