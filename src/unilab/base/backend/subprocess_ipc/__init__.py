"""Backend-agnostic subprocess IPC primitives.

The host-side simulation adapters and their external workers share one wire
protocol and one shared-memory slot layout.  Backend packages may add their
own runtime and physics adapters, but must import the canonical protocol from
this package instead of maintaining a fork.
"""

from . import protocol
from .backend import MjcfSubprocessBackend, SubprocessModelInfo, SubprocessWorkerError
from .playback import run_subprocess_playback
from .runtime import WorkerRuntime, build_worker_environment

__all__ = [
    "MjcfSubprocessBackend",
    "SubprocessModelInfo",
    "SubprocessWorkerError",
    "WorkerRuntime",
    "build_worker_environment",
    "protocol",
    "run_subprocess_playback",
]
