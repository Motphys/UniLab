"""Independent optional backend built on the Genesis runtime (``genesis-world``).

This package deliberately owns its own runtime implementation.  It reuses
shared *cold-path* scene materialization helpers (fragment merge) and the
``mujoco`` package for one-time MJCF metadata scans, but it never subclasses
or reads the runtime-private state of :mod:`unilab.base.backend.mujoco`.
"""

from __future__ import annotations


def __getattr__(name: str):
    if name == "GenesisBackend":
        from .backend import GenesisBackend

        return GenesisBackend
    if name == "GENESIS_AVAILABLE":
        from .dependencies import genesis_dependencies_available

        return genesis_dependencies_available()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GENESIS_AVAILABLE",
    "GenesisBackend",
]
