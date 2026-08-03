#!/usr/bin/env python3
"""Reject oversized changed files and any attempt to introduce Git LFS."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

MAX_COMMITTED_FILE_BYTES = 1024 * 1024
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def changed_paths(root: Path, base: str) -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=AM", "-z", f"{base}...HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return tuple(Path(raw.decode("utf-8")) for raw in result.stdout.split(b"\0") if raw)


def repository_file_errors(
    root: Path,
    paths: Iterable[Path],
    *,
    max_bytes: int = MAX_COMMITTED_FILE_BYTES,
) -> tuple[str, ...]:
    errors: list[str] = []
    for relative in sorted(set(paths)):
        path = root / relative
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > max_bytes:
            errors.append(
                f"{relative}: {size} bytes exceeds the {max_bytes}-byte committed-file limit"
            )
        prefix = path.read_bytes()[:256]
        if prefix.startswith(LFS_POINTER_PREFIX):
            errors.append(f"{relative}: Git LFS pointer files are forbidden")
        if relative.as_posix() == ".gitattributes":
            text = path.read_text(encoding="utf-8", errors="replace")
            if any(token in text for token in ("filter=lfs", "diff=lfs", "merge=lfs")):
                errors.append(".gitattributes: Git LFS rules are forbidden")
    return tuple(errors)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--max-bytes", type=int, default=MAX_COMMITTED_FILE_BYTES)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        paths = changed_paths(root, args.base)
    except subprocess.CalledProcessError as exc:
        print(f"FAIL: cannot compare changed files with {args.base}: {exc}")
        return 1
    errors = repository_file_errors(root, paths, max_bytes=args.max_bytes)
    if errors:
        print("FAIL: repository file policy")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"PASS: {len(paths)} changed files satisfy size and no-LFS policy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
