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


def test_algos_does_not_import_training() -> None:
    violations = [
        (path.relative_to(_REPO_ROOT).as_posix(), module)
        for path in sorted((_LIBRARY_PACKAGE / "algos").rglob("*.py"))
        for module in sorted(_imports(path))
        if module == "unilab.training" or module.startswith("unilab.training.")
    ]

    assert violations == [], "src/unilab/algos must not import unilab.training"


_UTILS_FORBIDDEN_LAYERS = (
    "unilab.base",
    "unilab.envs",
    "unilab.tasks",
    "unilab.managers",
    "unilab.algos",
    "unilab.training",
    "unilab.visualization",
    "unilab.ipc",
    "unilab.logging",
)


def test_utils_does_not_import_higher_layers() -> None:
    violations = [
        (path.relative_to(_REPO_ROOT).as_posix(), module)
        for path in sorted((_LIBRARY_PACKAGE / "utils").rglob("*.py"))
        for module in sorted(_imports(path))
        if any(
            module == layer or module.startswith(f"{layer}.") for layer in _UTILS_FORBIDDEN_LAYERS
        )
    ]

    assert violations == [], "src/unilab/utils must not import higher unilab layers"
