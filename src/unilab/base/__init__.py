"""Environment registry and base classes.

``ensure_registries`` is re-exported lazily so importing ``unilab.base`` (and
therefore ``unilab.base.contracts``) does not pull in the registry, the torch
env, or any backend package. Most callers import the concrete submodule
directly (``from unilab.base import registry``).
"""

from __future__ import annotations

from typing import Any

__all__ = ["ensure_registries"]


def __getattr__(name: str) -> Any:
    if name == "ensure_registries":
        from unilab.base.registry import ensure_registries

        return ensure_registries
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
