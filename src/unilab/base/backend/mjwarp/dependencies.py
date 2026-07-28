"""Lazy dependency boundary for the independent ``mjwarp`` backend."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any


class MjwarpDependencyError(ImportError):
    """Raised with an actionable install command when the optional extra is absent."""


@dataclass(frozen=True)
class MjwarpDependencies:
    """Modules required by the production ``mjwarp`` implementation."""

    mujoco: Any
    mujoco_warp: Any
    warp: Any


_REQUIRED_MODULES = ("mujoco", "mujoco_warp", "warp")
_INSTALL_HINT = "Install it with `uv sync --extra mjwarp`."


def mjwarp_dependencies_available() -> bool:
    """Return availability without importing CUDA/Warp runtime modules."""
    return all(find_spec(module_name) is not None for module_name in _REQUIRED_MODULES)


def load_mjwarp_dependencies() -> MjwarpDependencies:
    """Import optional runtime modules only when a backend instance is built."""
    try:
        mujoco = importlib.import_module("mujoco")
        mujoco_warp = importlib.import_module("mujoco_warp")
        warp = importlib.import_module("warp")
    except ModuleNotFoundError as exc:
        missing = exc.name or "an mjwarp dependency"
        raise MjwarpDependencyError(
            f"mjwarp backend requires optional dependency {missing!r}. {_INSTALL_HINT}"
        ) from exc
    except ImportError as exc:
        raise MjwarpDependencyError(
            f"mjwarp backend could not import its optional runtime: {exc}. {_INSTALL_HINT}"
        ) from exc
    return MjwarpDependencies(mujoco=mujoco, mujoco_warp=mujoco_warp, warp=warp)
