"""IsaacGym subprocess backend package.

Imports are lazy: importing this package must not spawn a worker or probe the
filesystem for the Python 3.8 runtime.  Use ``create_backend("isaacgym", ...)``
which defers runtime resolution to ``materialize()``.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "IsaacGymBackend":
        from .backend import IsaacGymBackend

        return IsaacGymBackend
    if name in ("IsaacGymModelInfo", "IsaacGymWorkerError"):
        from . import backend

        return getattr(backend, name)
    if name in ("IsaacGymDependencyError", "isaacgym_runtime_available"):
        from . import dependencies

        return getattr(dependencies, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "IsaacGymBackend",
    "IsaacGymDependencyError",
    "IsaacGymModelInfo",
    "IsaacGymWorkerError",
    "isaacgym_runtime_available",
]
