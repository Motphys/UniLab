"""Library-layer import boundary tests (issue #1240, updated for #2372).

The RL subsystem (contracts, adapters, algorithms, runtime, logging) merged
back into ``src/unilab/rl/``; it was the standalone unilab-rl package before
issue #2372. This module guards the merged internal DAG:

- ``unilab.rl.contracts`` is a leaf: stdlib/Torch types only, never unilab
  runtime/training/env layers or unisim.
- ``unilab.rl.logging`` only depends on itself.
- ``unilab.rl.runtime`` may use contracts/logging and the leaf shims
  ``unilab.utils``/``unilab.process_context``, but never algos, training,
  base, or env construction.
- ``unilab.rl.algos`` may use contracts/logging/runtime, but never training,
  base, or env construction.
- ``unilab.rl.adapters`` adapt UniLab envs to upstream rsl_rl; they must not
  depend on the training/base/algos assembly layers.
- ``unilab.utils`` stays a leaf: the allowed direction is
  ``unilab.rl.runtime`` -> ``unilab.utils`` (e.g. ``runtime.device`` reuses
  ``utils.device``), never the reverse.
- ``src/unilab`` never imports ``scripts``.

The AST-level guards are complemented by subprocess import-closure probes
that run in an isolated interpreter (``-I``).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LIBRARY_PACKAGE = _REPO_ROOT / "src" / "unilab"
_RL_PACKAGE = _LIBRARY_PACKAGE / "rl"

# Observation-contract symbols have exactly one owner module; everywhere else
# they must be imported, not re-defined.
_SINGLE_OWNER_SYMBOLS = {
    "split_obs_dict": _RL_PACKAGE / "contracts" / "observations.py",
    "get_obs_dims": _RL_PACKAGE / "contracts" / "observations.py",
    "get_default_device": _LIBRARY_PACKAGE / "utils" / "device.py",
    "TrainingSeedInfo": _LIBRARY_PACKAGE / "training" / "seed.py",
    "resolve_training_seed": _LIBRARY_PACKAGE / "training" / "seed.py",
    "apply_training_seed": _LIBRARY_PACKAGE / "training" / "seed.py",
    "apply_worker_seed": _RL_PACKAGE / "runtime" / "seed.py",
    "resolve_backend_process_device": _LIBRARY_PACKAGE / "base" / "process_device.py",
    "configure_backend_process_device": _LIBRARY_PACKAGE / "base" / "process_device.py",
    "NanGuard": _LIBRARY_PACKAGE / "training" / "tensor_diagnostics.py",
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


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _layer_violations(root: Path, forbidden: tuple[str, ...]) -> list[tuple[str, str]]:
    return [
        (_display_path(path), module)
        for path in sorted(root.rglob("*.py"))
        for module in sorted(_imports(path))
        if any(module == layer or module.startswith(f"{layer}.") for layer in forbidden)
    ]


def test_library_does_not_import_scripts() -> None:
    violations = [
        (path.relative_to(_REPO_ROOT).as_posix(), module)
        for path in sorted(_LIBRARY_PACKAGE.rglob("*.py"))
        for module in sorted(_imports(path))
        if module == "scripts" or module.startswith("scripts.")
    ]

    assert violations == [], "src/unilab must not import scripts/ modules"


def test_rl_layer_does_not_regrow_top_level_packages() -> None:
    leftovers = [name for name in ("algos", "ipc", "logging") if (_LIBRARY_PACKAGE / name).exists()]

    assert leftovers == [], (
        "RL algorithm/IPC/logging layers live under src/unilab/rl/; "
        f"src/unilab must not re-grow top-level copies: {leftovers}"
    )


def test_observation_contract_symbols_have_a_single_owner() -> None:
    violations = []
    for path in sorted(_LIBRARY_PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                owner = _SINGLE_OWNER_SYMBOLS.get(node.name)
                if owner is not None and path != owner:
                    violations.append((path.relative_to(_REPO_ROOT).as_posix(), node.name))

    assert violations == [], (
        "these symbols have exactly one owner module; import them instead of "
        f"re-defining: {violations}"
    )


def test_library_does_not_import_former_uni_rl_package() -> None:
    violations = [
        (path.relative_to(_REPO_ROOT).as_posix(), module)
        for path in sorted(_LIBRARY_PACKAGE.rglob("*.py"))
        for module in sorted(_imports(path))
        if module == "uni_rl" or module.startswith("uni_rl.")
    ]

    assert violations == [], "the merged RL stack must not import the former uni_rl package"


_UTILS_FORBIDDEN_LAYERS = (
    "unilab.base",
    "unilab.envs",
    "unilab.tasks",
    "unilab.managers",
    "unilab.training",
    "unilab.visualization",
    "unilab.rl",
)


def test_utils_does_not_import_higher_layers() -> None:
    violations = _layer_violations(_LIBRARY_PACKAGE / "utils", _UTILS_FORBIDDEN_LAYERS)

    assert violations == [], (
        "src/unilab/utils must not import higher layers "
        "(the allowed direction is unilab.rl.runtime -> unilab.utils)"
    )


# Internal DAG for the merged RL subsystem. Each entry lists the import roots
# forbidden for ``src/unilab/rl/<layer>``.
_RL_DAG_FORBIDDEN: dict[str, tuple[str, ...]] = {
    "contracts": (
        "unilab.base",
        "unilab.envs",
        "unilab.tasks",
        "unilab.managers",
        "unilab.training",
        "unilab.scripts",
        "unilab.rl.algos",
        "unilab.rl.runtime",
        "unisim",
    ),
    "runtime": (
        "unilab.rl.algos",
        "unilab.training",
        "unilab.base",
        "unilab.envs",
        "unilab.tasks",
        "unilab.managers",
        "unilab.scripts",
        "unisim",
    ),
    "algos": (
        "unilab.training",
        "unilab.base",
        "unilab.envs",
        "unilab.tasks",
        "unilab.managers",
        "unilab.scripts",
        "unisim",
    ),
    "adapters": (
        "unilab.training",
        "unilab.base",
        "unilab.rl.algos",
        "unilab.envs",
        "unilab.tasks",
        "unilab.managers",
        "unilab.scripts",
        "unisim",
    ),
}


@pytest.mark.parametrize("layer", sorted(_RL_DAG_FORBIDDEN))
def test_rl_layer_respects_internal_dag(layer: str) -> None:
    violations = _layer_violations(_RL_PACKAGE / layer, _RL_DAG_FORBIDDEN[layer])

    assert violations == [], f"src/unilab/rl/{layer} violates the RL-internal DAG: {violations}"


@pytest.mark.parametrize("forbidden", _RL_DAG_FORBIDDEN["adapters"])
def test_layer_violations_catches_forbidden_adapter_imports(tmp_path: Path, forbidden: str) -> None:
    path = tmp_path / "forbidden.py"
    path.write_text(f"from {forbidden} import marker\n", encoding="utf-8")

    assert _layer_violations(tmp_path, (forbidden,)) == [(path.as_posix(), forbidden)]


def test_layer_violations_allows_adapter_contracts_typing_import(tmp_path: Path) -> None:
    path = tmp_path / "allowed.py"
    path.write_text(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from unilab.rl.contracts import observations\n",
        encoding="utf-8",
    )

    assert _layer_violations(tmp_path, _RL_DAG_FORBIDDEN["adapters"]) == []


def test_rl_logging_does_not_import_other_unilab_layers() -> None:
    violations = [
        (path.relative_to(_REPO_ROOT).as_posix(), module)
        for path in sorted((_RL_PACKAGE / "logging").rglob("*.py"))
        for module in sorted(_imports(path))
        if (module.startswith("unilab") and not module.startswith("unilab.rl.logging"))
        or module == "unisim"
        or module.startswith("unisim.")
    ]

    assert violations == [], (
        f"src/unilab/rl/logging must not import other unilab subpackages: {violations}"
    )


_NUMPY_MDP_MODULES = (
    "envs/mdp/events.py",
    "envs/mdp/observations.py",
    "envs/mdp/rewards.py",
    "envs/mdp/terminations.py",
    "envs/mdp/commands/velocity_command.py",
)
_MDP_FORBIDDEN_LAYERS = (
    "unilab.rl",
    "unilab.training",
    "unilab.base.backend",
)


@pytest.mark.parametrize("relative_path", _NUMPY_MDP_MODULES)
def test_mdp_modules_do_not_import_training_or_ipc_layers(relative_path: str) -> None:
    path = _LIBRARY_PACKAGE / relative_path
    imports = _imports(path)
    violations = sorted(
        module
        for module in imports
        if any(
            module == forbidden or module.startswith(f"{forbidden}.")
            for forbidden in _MDP_FORBIDDEN_LAYERS
        )
    )

    assert violations == [], f"{relative_path} imports forbidden runtime layers: {violations}"


# ---------------------------------------------------------------------------
# Import-closure probes (real imports in an isolated interpreter)
# ---------------------------------------------------------------------------


def _run_isolated(code: str) -> None:
    """Run ``code`` in a fresh ``python -I`` interpreter (no PYTHONPATH/cwd)."""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"isolated import probe failed:\n{result.stderr}"


def test_rl_package_import_stays_lightweight() -> None:
    """``import unilab.rl`` must not pull in Torch or TensorDict."""
    _run_isolated(
        "import sys\n"
        "import unilab.rl\n"
        "heavy = [m for m in sys.modules if m.split('.')[0] in ('torch', 'tensordict')]\n"
        "assert not heavy, f'unilab.rl pulled in heavy modules: {heavy}'\n"
    )


def test_distributed_context_import_closure_stays_leaf() -> None:
    """The DP topology context must not need Torch or env/training layers."""
    _run_isolated(
        "import sys\n"
        "import unilab.rl.runtime.distributed.context\n"
        "forbidden = ('torch', 'tensordict', 'unisim',\n"
        "             'unilab.base', 'unilab.envs', 'unilab.tasks',\n"
        "             'unilab.managers', 'unilab.training')\n"
        "bad = [m for m in sys.modules\n"
        "       if any(m == f or m.startswith(f + '.') for f in forbidden)]\n"
        "assert not bad, f'distributed.context import closure too heavy: {bad}'\n"
    )


def test_contracts_import_without_simulator_packages() -> None:
    """Contracts modules must not require unisim or any simulator backend."""
    _run_isolated(
        "import sys\n"
        "import unilab.rl.contracts.env\n"
        "import unilab.rl.contracts.observations\n"
        "import unilab.rl.contracts.probe\n"
        "bad = [m for m in sys.modules if m == 'unisim' or m.startswith('unisim.')]\n"
        "assert not bad, f'contracts pulled in simulator packages: {bad}'\n"
    )
