"""Lazy dependency boundary for the independent ``genesis`` backend.

The Genesis runtime (PyPI distribution ``genesis-world``, import name
``genesis``) is a heavy optional dependency (torch, quadrants) that must never
be imported at package import time.  It is pinned exactly to the release probed
in ``scripts/tools/genesis_feasibility/REPORT.md`` (#1372) because the adapter
relies on measured API behavior of that version.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from importlib import metadata
from importlib.util import find_spec
from typing import Any


class GenesisDependencyError(ImportError):
    """Raised with an actionable install command when the optional extra is absent."""


@dataclass(frozen=True)
class GenesisDependencies:
    """Modules required by the production ``genesis`` implementation."""

    genesis: Any
    torch: Any
    mujoco: Any


_REQUIRED_MODULES = ("genesis", "torch", "mujoco")
_REQUIRED_GENESIS_VERSION = "1.3.3"
_INSTALL_HINT = "Install it with `uv sync --extra genesis`."


def genesis_dependencies_available() -> bool:
    """Return availability without importing CUDA/torch runtime modules."""
    return all(find_spec(module_name) is not None for module_name in _REQUIRED_MODULES)


def load_genesis_dependencies() -> GenesisDependencies:
    """Import optional runtime modules only when a backend instance is built."""
    try:
        installed_version = metadata.version("genesis-world")
    except metadata.PackageNotFoundError as exc:
        raise GenesisDependencyError(
            f"genesis backend requires optional dependency 'genesis-world'. {_INSTALL_HINT}"
        ) from exc
    if installed_version != _REQUIRED_GENESIS_VERSION:
        raise GenesisDependencyError(
            "genesis backend requires exact genesis-world version "
            f"{_REQUIRED_GENESIS_VERSION}, found {installed_version}. {_INSTALL_HINT}"
        )
    try:
        genesis = importlib.import_module("genesis")
        torch = importlib.import_module("torch")
        mujoco = importlib.import_module("mujoco")
    except ModuleNotFoundError as exc:
        missing = exc.name or "a genesis dependency"
        raise GenesisDependencyError(
            f"genesis backend requires optional dependency {missing!r}. {_INSTALL_HINT}"
        ) from exc
    except ImportError as exc:
        raise GenesisDependencyError(
            f"genesis backend could not import its optional runtime: {exc}. {_INSTALL_HINT}"
        ) from exc
    return GenesisDependencies(genesis=genesis, torch=torch, mujoco=mujoco)
