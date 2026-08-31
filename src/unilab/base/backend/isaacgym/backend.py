"""IsaacGym specialization of the shared MJCF subprocess host adapter.

IsaacGym Preview 4 runs in an external Python 3.8 process. The shared owner
layer supplies pipe/shm lifecycle, MJCF metadata, selected reset, NumPy state
views, and native-render command plumbing; this module only selects the
IsaacGym runtime and worker entrypoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unilab.base.backend.subprocess_ipc.backend import (
    MjcfSubprocessBackend,
    SubprocessModelInfo,
    SubprocessWorkerError,
    _normalize_camera_kwargs,
)

from .dependencies import build_worker_env, resolve_isaacgym_runtime

_WORKER_PATH = Path(__file__).resolve().parent / "worker.py"


class IsaacGymWorkerError(SubprocessWorkerError):
    """Raised when the external IsaacGym worker fails."""


@dataclass(frozen=True)
class IsaacGymModelInfo(SubprocessModelInfo):
    """Opaque metadata returned by the IsaacGym worker handshake."""


class IsaacGymBackend(MjcfSubprocessBackend):
    """NumPy-facing ``SimBackend`` client for the IsaacGym worker."""

    _BACKEND_TYPE = "isaacgym"
    _BACKEND_LABEL = "isaacgym"
    _WORKER_ERROR_CLS = IsaacGymWorkerError
    _MODEL_INFO_CLS = IsaacGymModelInfo

    def _worker_entrypoint(self) -> Path:
        return _WORKER_PATH

    def _resolve_worker_runtime(self) -> Any:
        return resolve_isaacgym_runtime()

    def _build_worker_environment(self, runtime: Any) -> dict[str, str]:
        return build_worker_env(runtime)

    def _runtime_payload(self, runtime: Any) -> dict[str, str]:
        return {"isaacgym_python": str(runtime.isaacgym_python)}


__all__ = [
    "IsaacGymBackend",
    "IsaacGymModelInfo",
    "IsaacGymWorkerError",
    "_normalize_camera_kwargs",
]
