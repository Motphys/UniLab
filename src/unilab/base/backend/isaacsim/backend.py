"""Host-side IsaacSim/IsaacLab subprocess backend.

IsaacSim is deliberately kept out of the UniLab interpreter: the supported
IsaacSim 5.1 wheels require Python 3.11, while the main project supports a
different Python range.  The NumPy-facing contract and lifecycle come from the
backend-neutral ``subprocess_ipc`` owner; this module supplies IsaacSim runtime
discovery, the worker entrypoint, clone-origin validation, and the headless-only
capability boundary.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from unilab.base.backend.isaacgym.backend import IsaacGymWorkerError
from unilab.base.backend.subprocess_ipc.backend import MjcfSubprocessBackend, SubprocessModelInfo
from unilab.base.backend.subprocess_ipc.sensors import (
    KIND_CONTACT_FOUND,
    UnsupportedSensorSpec,
)

from .dependencies import build_worker_env, resolve_isaacsim_runtime

_MODULE_DIR = Path(__file__).resolve().parent
_WORKER_PATH = _MODULE_DIR / "worker.py"


class IsaacSimWorkerError(IsaacGymWorkerError):
    """Raised when the external IsaacSim/IsaacLab worker fails.

    ``IsaacGymWorkerError`` is retained as a compatibility ancestor because
    early subprocess adapters exposed that exception as the public worker
    failure type.  The concrete class remains distinct, so callers can still
    distinguish IsaacSim failures with an exact type check while existing
    ``except IsaacGymWorkerError`` handlers continue to work.
    """


@dataclass(frozen=True)
class IsaacSimModelInfo(SubprocessModelInfo):
    """Opaque metadata returned by the IsaacSim worker handshake."""


class IsaacSimBackend(MjcfSubprocessBackend):
    """Thin host client for the IsaacLab/PhysX worker.

    The shared client owns pipe framing, shared-memory slot allocation,
    timeout/crash diagnostics, XML cold-path metadata, and all NumPy state
    views.  IsaacSim currently promises headless physics only; native GUI,
    camera capture, and offline rendering are intentionally rejected until a
    separately validated driver/runtime slice exists.
    """

    _BACKEND_TYPE = "isaacsim"
    _BACKEND_LABEL = "isaacsim"
    _WORKER_ERROR_CLS = IsaacSimWorkerError
    _MODEL_INFO_CLS = IsaacSimModelInfo

    def _worker_entrypoint(self) -> Path:
        return _WORKER_PATH

    def _resolve_worker_runtime(self) -> Any:
        return resolve_isaacsim_runtime()

    def _build_worker_environment(self, runtime: Any) -> dict[str, str]:
        return build_worker_env(runtime)

    def _runtime_payload(self, runtime: Any) -> dict[str, str]:
        # The worker receives the resolved interpreter path as a diagnostic
        # and a stable contract field.  It does not import the host package.
        return {
            "isaacsim_python": str(runtime.python),
            "isaaclab_source": (
                "" if runtime.isaaclab_source is None else str(runtime.isaaclab_source)
            ),
        }

    def _resolve_sensor_map(self) -> dict[str, tuple[Any, int]]:
        """Resolve only sensors backed by a real IsaacSim state quantity.

        The current worker reserves a contact-force slot for protocol
        compatibility but does not populate it from a PhysX contact reporter.
        Contact declarations must therefore remain unsupported rather than
        appearing to work while always returning zero.
        """
        resolved = super()._resolve_sensor_map()
        metadata = self._get_scene_metadata()
        for name, (spec, _body_id) in tuple(resolved.items()):
            if spec.kind != KIND_CONTACT_FOUND:
                continue
            metadata.unsupported_sensors[name] = UnsupportedSensorSpec(
                name=name,
                reason=(
                    "IsaacSim contact-force reporting is not implemented in the headless "
                    "worker; a reserved shared-memory slot is not a contact sensor"
                ),
            )
            del resolved[name]
        return resolved

    def _bind_model_metadata(self, meta: dict[str, Any]) -> None:
        """Validate the worker's private clone-origin/collision contract."""
        raw_origins = meta.get("env_origins")
        if raw_origins is None:
            raise self._worker_error(
                "isaacsim worker did not report environment origins; refusing to expose "
                "world-space clone state through the local-frame SimBackend contract"
            )
        origins = np.asarray(raw_origins, dtype=np.float32)
        expected = (self._num_envs, 3)
        if origins.shape != expected or not np.isfinite(origins).all():
            raise self._worker_error(
                f"isaacsim worker environment origins have invalid shape or values: "
                f"got shape {origins.shape}, expected {expected}"
            )
        if self._num_envs > 1:
            unique = np.unique(origins, axis=0)
            if unique.shape[0] != self._num_envs:
                raise self._worker_error(
                    "isaacsim worker returned duplicate environment origins; cloned actors "
                    "would overlap in world space"
                )
            if not bool(meta.get("collision_filtering_applied", False)):
                raise self._worker_error(
                    "isaacsim worker did not apply PhysX collision filtering between environments"
                )
        self._worker_env_origins = origins.copy()
        self._collision_filtering_applied = bool(meta.get("collision_filtering_applied", False))
        super()._bind_model_metadata(meta)

    def get_play_capabilities(self):
        """IsaacSim's supported slice is headless physics only."""
        # Import lazily to keep this module cheap and mirror the parent API.
        from unilab.base.backend.base import BackendPlayCapabilities

        return BackendPlayCapabilities(
            supports_native_interactive_renderer=False,
            supports_physics_state_playback=False,
            supports_native_video_capture=False,
        )

    def resolve_play_render_plan(
        self,
        *,
        play_render_mode: str | None,
        play_steps: int | None,
        output_video: str | os.PathLike[str] | None,
    ):
        # ``none`` remains a valid explicit plan so callers can disable
        # playback cleanly. Every rendering mode fails closed at this boundary.
        from unilab.base.backend.base import BackendPlayRenderPlan, normalize_play_render_mode

        del play_steps, output_video
        mode = normalize_play_render_mode(play_render_mode)
        if mode == "none":
            return BackendPlayRenderPlan(
                mode="none",
                headless=True,
                record_video=False,
                num_steps=None,
                output_video=None,
            )
        raise NotImplementedError(
            "isaacsim currently supports headless physics only; native GUI/camera "
            "playback is not available for this runtime"
        )

    def init_renderer(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise NotImplementedError(
            "isaacsim currently supports headless physics only; native rendering is unavailable"
        )

    def render(self) -> None:
        raise NotImplementedError(
            "isaacsim currently supports headless physics only; interactive rendering is unavailable"
        )

    def capture_video_frame(self):
        raise NotImplementedError(
            "isaacsim currently supports headless physics only; video capture is unavailable"
        )

    def run_playback(self, **kwargs: Any) -> str | None:
        del kwargs
        raise NotImplementedError(
            "isaacsim currently supports headless physics only; playback rendering is unavailable"
        )


# The model metadata shape is backend-neutral; retaining the alias avoids a
# second, structurally identical dataclass while the error class above remains
# intentionally distinct.
__all__ = [
    "IsaacSimBackend",
    "IsaacSimModelInfo",
    "IsaacSimWorkerError",
]
