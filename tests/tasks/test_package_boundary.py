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
    "unilab.tasks.locomotion.go2",
    "unilab.tasks.locomotion.g1",
    "unilab.tasks.motion_tracking.g1",
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


def test_tensor_runtime_switch_is_absent_from_source_and_owner_configs() -> None:
    offenders: list[str] = []
    patterns = (
        "cfg.tensor_runtime",
        "env.tensor_runtime",
        "tensor_runtime_device",
        "tensor_runtime:",
    )
    for path in (_REPO_ROOT / "src").rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".yaml"}:
            continue
        if ".venv" in path.parts or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern in text for pattern in patterns):
            offenders.append(path.relative_to(_REPO_ROOT).as_posix())
    assert offenders == [], f"tensor-runtime switch references remain: {offenders}"


def test_locomotion_runtime_terms_do_not_use_environment_numpy_rng() -> None:
    paths = (
        _REPO_ROOT / "src" / "unilab" / "tasks" / "locomotion" / "g1" / "manager_terms.py",
        _REPO_ROOT
        / "src"
        / "unilab"
        / "tasks"
        / "locomotion"
        / "common"
        / "rough_manager_terms.py",
    )
    offenders = [
        (path.relative_to(_REPO_ROOT).as_posix(), line_number)
        for path in paths
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "env.rng" in line or "self._env.rng" in line
    ]

    assert offenders == [], f"locomotion NumPy RNG references remain: {offenders}"


def test_flashsac_motion_owner_uses_generic_manager_runtime() -> None:
    registry.ensure_registries()
    assert "motrix" in registry._envs["G1MotionTracking"].env_factory_dict


def test_motion_direct_runtime_is_removed() -> None:
    direct_runtime = _REPO_ROOT / "src/unilab/tasks/motion_tracking/g1/torch_flashsac_env.py"
    assert not direct_runtime.exists()
