#!/usr/bin/env python3
"""Docs validation script for UniLab.

Validates documentation for:
- Script references (scripts/*.py must exist)
- Hydra config keys (common keys must match expected patterns)
- Relative links (markdown links must point to existing files)
- File paths (paths mentioned in docs must exist)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

# Known valid scripts
VALID_SCRIPTS = {
    "scripts/train_rsl_rl.py",
    "scripts/train_mlx_ppo.py",
    "scripts/train_appo.py",
    "scripts/train_offpolicy.py",
    "scripts/play_interactive.py",
    "scripts/visualization_env.py",
}

# Known valid Hydra config keys
VALID_HYDRA_KEYS = {
    "task",
    "algo",
    "training",
    "reward",
    # algo subkeys
    "num_envs",
    "max_iterations",
    "num_steps_per_env",
    "learning_rate",
    "entropy_coef",
    "desired_kl",
    "load_run",
    # training subkeys
    "play_only",
    "no_play",
    "load_run",
    "logger",
    "sim_backend",
    "play_env_num",
    "play_steps",
    "cam_distance",
    "cam_elevation",
    "cam_azimuth",
    "num_timesteps",
    "num_gpus",
    "device",
    "wandb_project",
    "wandb_entity",
    "wandb_group",
    "wandb_name",
    "wandb_tags",
    "wandb_notes",
    "wandb_mode",
}

# File patterns to check
DOC_PATTERNS = [
    "*.md",
    "docs/**/*.md",
    "src/**/*.md",
]

# Patterns to skip (generated, vendor, etc.)
SKIP_PATTERNS = [
    r"\.git",
    r"\.venv",
    r"__pycache__",
    r"\.pytest_cache",
]


def should_skip(path: Path) -> bool:
    """Check if path should be skipped."""
    path_str = str(path)
    for pattern in SKIP_PATTERNS:
        if re.search(pattern, path_str):
            return True
    return False


def find_docs(root: Path) -> list[Path]:
    """Find all documentation files."""
    docs: list[Path] = []
    for pattern in DOC_PATTERNS:
        if pattern.startswith("*/"):
            # Handle recursive patterns
            for path in root.rglob(pattern[3:]):
                if not should_skip(path):
                    docs.append(path)
        else:
            for path in root.glob(pattern):
                if not should_skip(path):
                    docs.append(path)
    return docs


def check_script_references(content: str, doc_path: Path, root: Path) -> list[str]:
    """Check that script references point to existing files."""
    errors: list[str] = []

    # Pattern: scripts/xxx.py or uv run python scripts/xxx.py
    script_pattern = r"(?:uv run python |python )?(scripts/\w+\.py)"
    for match in re.finditer(script_pattern, content):
        script_path = match.group(1)
        full_path = root / script_path
        if not full_path.exists():
            errors.append(f"{doc_path}: Script not found: {script_path}")

    return errors


def check_file_paths(content: str, doc_path: Path, root: Path) -> list[str]:
    """Check that file paths in code blocks exist."""
    errors: list[str] = []

    # Pattern: `path/to/file` or 'path/to/file' in code-like contexts
    # Look for paths that look like source files
    path_patterns = [
        r"`(src/[^`]+)`",
        r"`(conf/[^`]+)`",
        r"`(tests/[^`]+)`",
        r"\(src/[^\)]+\)",
        r"\(conf/[^\)]+\)",
        r"\(tests/[^\)]+\)",
    ]

    for pattern in path_patterns:
        for match in re.finditer(pattern, content):
            rel_path = match.group(1)
            full_path = root / rel_path
            # Skip if it looks like a pattern or has wildcards
            if "*" in rel_path or "<" in rel_path:
                continue
            if not full_path.exists():
                errors.append(f"{doc_path}: Path not found: {rel_path}")

    return errors


def check_markdown_links(content: str, doc_path: Path, root: Path) -> list[str]:
    """Check that markdown relative links point to existing files."""
    errors: list[str] = []

    # Pattern: [text](path/to/file.md) or [text](../path/to/file.md)
    link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"

    for match in re.finditer(link_pattern, content):
        link_text = match.group(1)
        link_target = match.group(2)

        # Skip external links
        if link_target.startswith(("http://", "https://", "#", "mailto:")):
            continue

        # Resolve relative to document
        if link_target.startswith("/"):
            full_path = root / link_target.lstrip("/")
        else:
            full_path = doc_path.parent / link_target

        # Remove anchor
        full_path_str = str(full_path).split("#")[0]
        full_path = Path(full_path_str)

        if not full_path.exists():
            errors.append(f"{doc_path}: Link not found: {link_target} (text: '{link_text}')")

    return errors


def check_hydra_keys(content: str, doc_path: Path, root: Path) -> list[str]:
    """Check Hydra config keys for common patterns."""
    errors: list[str] = []

    # Pattern: key=value or key.nested=value
    # Look for patterns like algo.num_envs=2048
    hydra_pattern = r"(?:^|\s)([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)="

    for match in re.finditer(hydra_pattern, content):
        full_key = match.group(1)
        # Split by dots and check each component
        parts = full_key.split(".")

        # Skip if the first part is a known top-level key
        if parts[0] not in VALID_HYDRA_KEYS and parts[0] not in {"python", "uv"}:
            # Only warn for what look like config keys (lowercase with underscores)
            if re.match(r"^[a-z_]+$", parts[0]):
                errors.append(f"{doc_path}: Unknown config key: {full_key}")

    return errors


def check_argparse_vs_hydra(content: str, doc_path: Path, root: Path) -> list[str]:
    """Check for outdated argparse style commands in Hydra scripts."""
    errors: list[str] = []

    # Pattern for old argparse style flags that should be Hydra style
    old_flags = [
        (r"--task\s+", "Use Hydra style: task=<value>"),
        (r"--env_num\s+", "Use Hydra style: algo.num_envs=<value>"),
        (r"--play_only\b", "Use Hydra style: training.play_only=true"),
        (r"--no_play\b", "Use Hydra style: training.no_play=true"),
        (r"--load_run\s+", "Use Hydra style: training.load_run=<value>"),
        (r"--cam_distance\s+", "Use Hydra style: training.cam_distance=<value>"),
        (r"--cam_elevation\s+", "Use Hydra style: training.cam_elevation=<value>"),
        (r"--cam_azimuth\s+", "Use Hydra style: training.cam_azimuth=<value>"),
    ]

    # Scripts that use Hydra (not argparse)
    hydra_scripts = [
        "train_rsl_rl.py",
        "train_mlx_ppo.py",
        "train_appo.py",
        "train_offpolicy.py",
    ]

    for pattern, message in old_flags:
        for match in re.finditer(pattern, content):
            start = max(0, match.start() - 100)
            end = min(len(content), match.end() + 50)
            context = content[start:end]
            # Only flag if context contains a Hydra script
            is_hydra_context = any(script in context for script in hydra_scripts)
            if is_hydra_context:
                errors.append(
                    f"{doc_path}: Outdated argparse flag '{match.group(0).strip()}': {message}"
                )

    return errors


def check_document(doc_path: Path, root: Path) -> list[str]:
    """Run all checks on a single document."""
    errors: list[str] = []

    try:
        content = doc_path.read_text(encoding="utf-8")
    except Exception as e:
        return [f"{doc_path}: Error reading file: {e}"]

    errors.extend(check_script_references(content, doc_path, root))
    errors.extend(check_file_paths(content, doc_path, root))
    errors.extend(check_markdown_links(content, doc_path, root))
    errors.extend(check_hydra_keys(content, doc_path, root))
    errors.extend(check_argparse_vs_hydra(content, doc_path, root))

    return errors


def main() -> int:
    """Main entry point."""
    root = Path(__file__).parent.parent
    docs = find_docs(root)

    all_errors: list[str] = []

    print(f"Checking {len(docs)} documentation files...")

    for doc_path in docs:
        errors = check_document(doc_path, root)
        all_errors.extend(errors)

    if all_errors:
        print(f"\nFound {len(all_errors)} issue(s):")
        for error in all_errors:
            print(f"  - {error}")
        return 1
    else:
        print("\nAll documentation checks passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
