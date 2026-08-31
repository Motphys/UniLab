"""IsaacSim/IsaacLab subprocess backend.

The package is intentionally lazy: importing UniLab does not import Kit or
probe the external Python 3.11 runtime.  Runtime discovery happens when the
backend is materialized (or explicitly through ``dependencies``).
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "IsaacSimBackend":
        from .backend import IsaacSimBackend

        return IsaacSimBackend
    if name in ("IsaacSimModelInfo", "IsaacSimWorkerError"):
        from . import backend

        return getattr(backend, name)
    if name in ("IsaacSimDependencyError", "isaacsim_runtime_available"):
        from . import dependencies

        return getattr(dependencies, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "IsaacSimBackend",
    "IsaacSimDependencyError",
    "IsaacSimModelInfo",
    "IsaacSimWorkerError",
    "isaacsim_runtime_available",
]
