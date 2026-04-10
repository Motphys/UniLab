#!/usr/bin/env python3

"""Refresh the generated support matrix section in docs."""

from __future__ import annotations

import argparse
from pathlib import Path

from unilab.docs.support_matrix import (
    render_generated_block,
    render_support_matrix,
    replace_generated_block,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Update docs/zh_CN/02-simulation-backends.md in place.",
    )
    args = parser.parse_args()

    root = _repo_root()
    if not args.write:
        print(render_support_matrix(root))
        return 0

    doc_path = root / "docs" / "zh_CN" / "02-simulation-backends.md"
    content = doc_path.read_text(encoding="utf-8")
    updated = replace_generated_block(content, render_generated_block(root))
    doc_path.write_text(updated, encoding="utf-8")
    print(f"Updated {doc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
