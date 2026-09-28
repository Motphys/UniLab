#!/usr/bin/env python3

"""Refresh the generated bilingual support matrix sections in docs."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.tools.support_matrix import (
    render_generated_block as render_tool_generated_block,
)
from scripts.tools.support_matrix import (
    render_support_matrix as render_tool_support_matrix,
)
from scripts.tools.support_matrix import (
    replace_generated_block,
)

_SUPPORT_LEVEL_ENUM = re.compile(r"\bSupportLevel\.([A-Z][A-Z_]+)\b")


def _with_support_level_values(content: str) -> str:
    """Render hardened UniSim lifecycle enum values as their public strings."""
    return _SUPPORT_LEVEL_ENUM.sub(lambda match: match.group(1).lower(), content)


def render_support_matrix(root: Path, language: str) -> str:
    return _with_support_level_values(render_tool_support_matrix(root, language))


def render_generated_block(root: Path, language: str) -> str:
    return _with_support_level_values(render_tool_generated_block(root, language))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Update both bilingual support-matrix pages in place.",
    )
    args = parser.parse_args()

    root = _repo_root()
    if not args.write:
        print(render_support_matrix(root, "zh"))
        return 0

    # Keep refresh output stable regardless of local dict/hash behavior or future
    # call-site changes. English sorts before Chinese, making failures easy to
    # map to the newly generated page order.
    doc_paths = (
        ("en", root / "docs" / "sphinx" / "source" / "en" / "5-reference" / "5-support_matrix.md"),
        (
            "zh",
            root / "docs" / "sphinx" / "source" / "zh_CN" / "5-reference" / "5-support_matrix.md",
        ),
    )
    for language, doc_path in doc_paths:
        content = doc_path.read_text(encoding="utf-8")
        updated = replace_generated_block(content, render_generated_block(root, language))
        doc_path.write_text(updated, encoding="utf-8")
        print(f"Updated {doc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
