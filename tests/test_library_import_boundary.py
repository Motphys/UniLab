"""Library-layer import boundary tests (issue #1240, updated for #1480).

The RL algorithm layer (``unilab.algos``), the async IPC layer
(``unilab.ipc``), and the training-logging layer (``unilab.logging``) moved to
the independently released uni_rl package. uni_rl's own purity (no
``unilab``/``unisim`` imports) is enforced by that repo's smoke gate; here we
assert the UniLab side of the boundary.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LIBRARY_PACKAGE = _REPO_ROOT / "src" / "unilab"

# Symbols that moved to uni_rl.utils.* must not be re-defined anywhere in
# src/unilab (importing them from uni_rl is the supported path).
_MIGRATED_SYMBOLS = {
    "split_obs_dict",
    "get_obs_dims",
    "get_critic_base_dim",
    "TerminalObservationContract",
    "resolve_terminal_observation_contract",
}


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


def test_migrated_layer_directories_are_gone() -> None:
    leftovers = [name for name in ("algos", "ipc", "logging") if (_LIBRARY_PACKAGE / name).exists()]

    assert leftovers == [], (
        "RL algorithm/IPC/logging layers live in the uni_rl package now; "
        f"src/unilab must not re-grow: {leftovers}"
    )


def test_no_module_redefines_migrated_symbols() -> None:
    violations = []
    for path in sorted(_LIBRARY_PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in _MIGRATED_SYMBOLS:
                    violations.append((path.relative_to(_REPO_ROOT).as_posix(), node.name))

    assert violations == [], (
        "these symbols live in uni_rl.utils.*; import them from there instead "
        f"of re-defining: {violations}"
    )


_UTILS_FORBIDDEN_LAYERS = (
    "unilab.base",
    "unilab.envs",
    "unilab.tasks",
    "unilab.managers",
    "unilab.training",
    "unilab.visualization",
    "uni_rl",
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

    assert violations == [], "src/unilab/utils must not import higher layers"


_NUMPY_MDP_MODULES = (
    "envs/mdp/events.py",
    "envs/mdp/observations.py",
    "envs/mdp/rewards.py",
    "envs/mdp/terminations.py",
    "envs/mdp/commands/velocity_command.py",
)
_NUMPY_MDP_FORBIDDEN_LAYERS = (
    "torch",
    "uni_rl",
    "unilab.training",
    "unilab.base.backend",
)


@pytest.mark.parametrize("relative_path", _NUMPY_MDP_MODULES)
def test_numpy_mdp_modules_do_not_import_runtime_layers(relative_path: str) -> None:
    path = _LIBRARY_PACKAGE / relative_path
    imports = _imports(path)
    violations = sorted(
        module
        for module in imports
        if any(
            module == forbidden or module.startswith(f"{forbidden}.")
            for forbidden in _NUMPY_MDP_FORBIDDEN_LAYERS
        )
    )

    assert violations == [], f"{relative_path} imports forbidden runtime layers: {violations}"
