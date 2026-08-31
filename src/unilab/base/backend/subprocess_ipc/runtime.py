"""Small, optional-runtime helpers shared by subprocess backends.

This module is deliberately standard-library only.  Runtime-specific modules
validate their package layout and then use :class:`WorkerRuntime` to construct
the child process environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkerRuntime:
    """Resolved interpreter and optional library/search paths for one worker."""

    python: Path
    package_path: Path | None = None
    lib_path: Path | None = None
    bin_path: Path | None = None


def build_worker_environment(runtime: WorkerRuntime) -> dict[str, str]:
    """Return a child environment with runtime paths prepended.

    Empty paths are ignored.  Existing user values remain after the resolved
    runtime paths so a worker always loads its pinned native libraries first.
    """

    env = dict(os.environ)
    if runtime.lib_path is not None:
        old = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{runtime.lib_path}:{old}" if old else str(runtime.lib_path)
    if runtime.bin_path is not None:
        old = env.get("PATH", "")
        env["PATH"] = f"{runtime.bin_path}:{old}" if old else str(runtime.bin_path)
    return env


__all__ = ["WorkerRuntime", "build_worker_environment"]
