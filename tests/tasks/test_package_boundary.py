"""Task bootstrap and package dependency boundary tests."""

from __future__ import annotations

import ast
from pathlib import Path

from unilab.base import registry
from unilab.tasks import __unilab_registry_modules__

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_PACKAGE = _REPO_ROOT / "src" / "unilab" / "envs"
_CONCRETE_TASK_PACKAGES = ("locomotion", "manipulation", "motion_tracking")

_TASK_REGISTRY_MODULES = (
    "unilab.tasks.locomotion.go1",
    "unilab.tasks.locomotion.go2",
    "unilab.tasks.locomotion.go2w",
    "unilab.tasks.locomotion.g1",
    "unilab.tasks.locomotion.go2_arm",
    "unilab.tasks.locomotion.a2",
    "unilab.tasks.locomotion.t800",
    "unilab.tasks.manipulation.allegro_inhand",
    "unilab.tasks.manipulation.sharpa_inhand",
    "unilab.tasks.manipulation.stewart",
    "unilab.tasks.motion_tracking.g1",
    "unilab.tasks.motion_tracking.x2",
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def test_tasks_is_the_only_default_registry_bootstrap() -> None:
    assert registry._DEFAULT_REGISTRY_PACKAGES == ("unilab.tasks",)
    assert __unilab_registry_modules__ == _TASK_REGISTRY_MODULES


def test_env_runtime_does_not_depend_on_tasks() -> None:
    violations = [
        (path.relative_to(_REPO_ROOT).as_posix(), module)
        for path in sorted(_ENV_PACKAGE.rglob("*.py"))
        for module in sorted(_imports(path))
        if module == "unilab.tasks" or module.startswith("unilab.tasks.")
    ]

    assert violations == [], "unilab.envs must not import concrete unilab.tasks modules"


def test_env_runtime_does_not_own_concrete_task_packages() -> None:
    violations = [
        path.relative_to(_REPO_ROOT).as_posix()
        for package in _CONCRETE_TASK_PACKAGES
        for path in sorted((_ENV_PACKAGE / package).rglob("*.py"))
    ]

    assert violations == [], "concrete task source must be owned by unilab.tasks"
