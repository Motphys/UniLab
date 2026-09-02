"""Host-side IsaacSim/IsaacLab subprocess backend.

IsaacSim is deliberately kept out of the UniLab interpreter: the supported
IsaacSim 5.1 wheels require Python 3.11, while the main project supports a
different Python range.  The NumPy-facing contract and lifecycle come from the
backend-neutral ``subprocess_ipc`` owner; this module supplies IsaacSim runtime
discovery, the worker entrypoint, clone-origin validation, and the eval-owned
Kit/viewer/camera capability boundary.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from unilab.base.backend.base import (
    BackendPlayCapabilities,
    BackendPlayRenderPlan,
    normalize_play_render_mode,
)
from unilab.base.backend.isaacgym.backend import IsaacGymWorkerError
from unilab.base.backend.subprocess_ipc.backend import (
    MjcfSubprocessBackend,
    SubprocessModelInfo,
)
from unilab.base.backend.subprocess_ipc.sensors import (
    KIND_CONTACT_FOUND,
    UnsupportedSensorSpec,
)

from .dependencies import build_worker_env, resolve_isaacsim_runtime

_MODULE_DIR = Path(__file__).resolve().parent
_WORKER_PATH = _MODULE_DIR / "worker.py"


def _display_available() -> bool:
    """Return whether a local Kit window can be opened."""

    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class IsaacSimRenderError(RuntimeError):
    """Raised when the requested IsaacSim render mode cannot be initialized."""


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
    views.  Physics remains the default (``render_mode=None``). Eval/play
    profiles pass a concrete render intent through the cold ``INIT`` payload
    so Kit selects the correct experience before simulation materialization.
    The worker owns all camera/viewport operations; this class validates the
    NumPy-facing frame contract.
    """

    _BACKEND_TYPE = "isaacsim"
    _BACKEND_LABEL = "isaacsim"
    _WORKER_ERROR_CLS = IsaacSimWorkerError
    _MODEL_INFO_CLS = IsaacSimModelInfo

    def __init__(
        self,
        scene: Any,
        num_envs: int,
        sim_dt: float,
        *,
        render_mode: str | None = None,
        render_width: int = 1280,
        render_height: int = 720,
        **kwargs: Any,
    ) -> None:
        mode = None if render_mode is None else normalize_play_render_mode(render_mode)
        for name, value in (("render_width", render_width), ("render_height", render_height)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        self._requested_render_mode = mode
        self._resolved_render_mode: str | None = None
        super().__init__(scene, num_envs, sim_dt, **kwargs)
        self._render_width = int(render_width)
        self._render_height = int(render_height)

    def _resolve_render_mode(self) -> str:
        """Resolve eval intent before Kit is launched."""
        requested = self._requested_render_mode
        if requested is None or requested == "none":
            return "none"
        if requested == "auto":
            return "interactive" if _display_available() else "record"
        if requested == "interactive" and not _display_available():
            raise IsaacSimRenderError(
                "IsaacSim interactive rendering was requested but no local display was found; "
                "set DISPLAY/WAYLAND_DISPLAY or use training.play_render_mode=record for "
                "headless RGB capture."
            )
        return requested

    def _worker_init_payload(self) -> dict[str, Any]:
        mode = self._resolve_render_mode()
        self._resolved_render_mode = mode
        return {
            "render_mode": mode,
            "render_width": self._render_width,
            "render_height": self._render_height,
        }

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
        """Validate the worker's private clone, collision, and render contract."""
        expected_render_mode = self._resolved_render_mode
        if expected_render_mode is None:
            raise self._worker_error(
                "isaacsim worker metadata arrived before the host resolved its render mode"
            )
        required_render_fields = {
            "graphics_enabled",
            "render_mode",
            "render_width",
            "render_height",
        }
        missing_render_fields = sorted(required_render_fields.difference(meta))
        if missing_render_fields:
            raise self._worker_error(
                "isaacsim worker metadata is missing render startup fields: "
                + ", ".join(missing_render_fields)
            )

        reported_render_mode = meta["render_mode"]
        if reported_render_mode != expected_render_mode:
            raise self._worker_error(
                "isaacsim worker render_mode does not match the host INIT request: "
                f"worker={reported_render_mode!r}, host={expected_render_mode!r}"
            )
        raw_width = meta["render_width"]
        raw_height = meta["render_height"]
        if (
            isinstance(raw_width, bool)
            or not isinstance(raw_width, (int, np.integer))
            or isinstance(raw_height, bool)
            or not isinstance(raw_height, (int, np.integer))
            or (int(raw_width), int(raw_height)) != (self._render_width, self._render_height)
        ):
            raise self._worker_error(
                "isaacsim worker render dimensions do not match the host INIT request: "
                f"worker={raw_width!r}x{raw_height!r}, "
                f"host={self._render_width}x{self._render_height}"
            )
        reported_graphics = meta["graphics_enabled"]
        expected_graphics = expected_render_mode != "none"
        if (
            not isinstance(reported_graphics, (bool, np.bool_))
            or bool(reported_graphics) != expected_graphics
        ):
            raise self._worker_error(
                "isaacsim worker graphics_enabled does not match its startup render mode: "
                f"render_mode={expected_render_mode!r}, "
                f"graphics_enabled={reported_graphics!r}, expected={expected_graphics!r}"
            )

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

    def get_play_capabilities(self) -> BackendPlayCapabilities:
        """Return the native Kit viewer and RGB camera capabilities."""
        return BackendPlayCapabilities(
            supports_native_interactive_renderer=True,
            supports_physics_state_playback=False,
            supports_native_video_capture=True,
        )

    def resolve_play_render_plan(
        self,
        *,
        play_render_mode: str | None,
        play_steps: int | None,
        output_video: str | os.PathLike[str] | None,
    ) -> BackendPlayRenderPlan:
        requested_mode = normalize_play_render_mode(play_render_mode)
        if requested_mode == "none":
            return BackendPlayRenderPlan(
                mode="none",
                headless=True,
                record_video=False,
                num_steps=None,
                output_video=None,
            )

        # Kit's experience is immutable after AppLauncher starts. Resolve
        # ``auto`` from the same cold-start decision and fail before playback
        # if a direct caller created a no-rendering or differently configured
        # worker. The eval adapters inject matching intent before env creation.
        startup_mode = self._resolved_render_mode
        if startup_mode is None:
            startup_mode = self._resolve_render_mode()
        mode = startup_mode if requested_mode == "auto" else requested_mode
        if startup_mode == "none" or startup_mode != mode:
            raise IsaacSimRenderError(
                f"IsaacSim worker started in render_mode={startup_mode!r}, but playback requested "
                f"{requested_mode!r}; select the matching training.play_render_mode before env creation."
            )
        if mode == "interactive":
            return BackendPlayRenderPlan(
                mode=mode,
                headless=False,
                record_video=False,
                num_steps=None,
                output_video=None,
            )
        if isinstance(play_steps, bool) or play_steps is None or int(play_steps) <= 0:
            raise ValueError(
                "isaacsim record playback requires a positive finite training.play_steps value."
            )
        if output_video is None:
            raise ValueError("isaacsim record playback requires an output video path.")
        return BackendPlayRenderPlan(
            mode="record",
            headless=True,
            record_video=True,
            num_steps=int(play_steps),
            output_video=output_video,
        )

    def init_renderer(
        self,
        spacing: float = 1.0,
        *,
        offset_mode: str = "grid",
        headless: bool = False,
        capture: bool = False,
        width: int = 1280,
        height: int = 720,
        camera_kwargs: dict[str, Any] | None = None,
    ) -> None:
        mode = self._resolved_render_mode
        if mode is None:
            mode = self._resolve_render_mode()
        requested = "record" if (headless or capture) else "interactive"
        if mode != requested:
            raise IsaacSimRenderError(
                f"IsaacSim worker started in render_mode={mode!r}, but renderer requested "
                f"{requested!r}; select the matching training.play_render_mode before env creation."
            )
        if (
            isinstance(width, bool)
            or not isinstance(width, (int, np.integer))
            or isinstance(height, bool)
            or not isinstance(height, (int, np.integer))
            or int(width) <= 0
            or int(height) <= 0
        ):
            raise IsaacSimRenderError(
                f"IsaacSim render dimensions must be positive integers; got {width!r}x{height!r}."
            )
        if int(width) != self._render_width or int(height) != self._render_height:
            raise IsaacSimRenderError(
                "IsaacSim render dimensions are fixed at worker startup: "
                f"configured {self._render_width}x{self._render_height}, requested {width}x{height}."
            )
        # ``super`` owns the protocol/lifecycle and is intentionally called
        # only after the mode/dimension checks above.
        super().init_renderer(
            spacing=spacing,
            offset_mode=offset_mode,
            headless=headless,
            capture=capture,
            width=width,
            height=height,
            camera_kwargs=camera_kwargs,
        )

    def capture_video_frame(self) -> np.ndarray:
        frame = np.asarray(super().capture_video_frame())
        expected = (self._render_height, self._render_width, 3)
        if frame.dtype != np.uint8 or frame.shape != expected:
            raise IsaacSimRenderError(
                "IsaacSim camera returned an invalid RGB frame: "
                f"shape={frame.shape}, dtype={frame.dtype}, expected shape={expected}, dtype=uint8"
            )
        if frame.size == 0 or int(np.ptp(frame)) == 0:
            raise IsaacSimRenderError(
                "IsaacSim camera returned an empty/uniform RGB frame; refusing to write a "
                "placeholder video. Check camera pose, lighting, and RTX camera support."
            )
        return np.ascontiguousarray(frame)


# The model metadata shape is backend-neutral; retaining the alias avoids a
# second, structurally identical dataclass while the error class above remains
# intentionally distinct.
__all__ = [
    "IsaacSimBackend",
    "IsaacSimModelInfo",
    "IsaacSimRenderError",
    "IsaacSimWorkerError",
]
