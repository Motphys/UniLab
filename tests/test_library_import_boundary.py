"""Library-layer import boundary tests (issue #1240)."""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LIBRARY_PACKAGE = _REPO_ROOT / "src" / "unilab"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def test_library_does_not_import_scripts() -> None:
    violations = [
        (path.relative_to(_REPO_ROOT).as_posix(), module)
        for path in sorted(_LIBRARY_PACKAGE.rglob("*.py"))
        for module in sorted(_imports(path))
        if module == "scripts" or module.startswith("scripts.")
    ]

    assert violations == [], "src/unilab must not import scripts/ modules"
