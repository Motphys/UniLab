"""Embeddable snapshot-cache + deferred-render playback session.

Custom eval loops with their own trial protocol cannot use the monolithic
``env.run_playback()`` entrypoint.  :class:`SnapshotPlaybackSession` exposes
the record pipeline's two halves as session-level operations: call
:meth:`SnapshotPlaybackSession.snapshot` once per trial step to cache physics
states (and, optionally, per-frame debug overlays), then call
:meth:`SnapshotPlaybackSession.render_snapshots` at the end of a trial to
render the cached states into one mp4 through the shared MuJoCo offline
snapshot pipeline.

The session holds plain NumPy arrays and typed primitives only, so cached
state stays picklable; the owning env reference is used for capability gating
and model resolution at render time.  Call :meth:`SnapshotPlaybackSession.clear`
between trials to bound memory.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Mapping, Sequence
from os import PathLike
from typing import Any

import numpy as np
from unisim.backend.base import (
    CameraCfg,
    DebugOverlayGetter,
    DebugPrimitive,
    unsupported_debug_overlay_error,
    validate_debug_overlays,
)

OnFrameFn = Callable[[int, np.ndarray], "np.ndarray | None"]


class SnapshotPlaybackSession:
    """Cache physics snapshots during a trial; render them to video afterwards."""

    def __init__(
        self,
        env: Any,
        *,
        frame_state_getter: Callable[[], np.ndarray] | None = None,
        overlay_getter: DebugOverlayGetter | None = None,
        render_spacing: float | None = None,
        width: int = 1280,
        height: int = 720,
        num_processes: int = 8,
    ) -> None:
        capabilities = getattr(env, "play_capabilities", None)
        if capabilities is None or not capabilities.supports_physics_state_playback:
            raise NotImplementedError(
                f"{type(env).__name__} does not support physics-state playback, so "
                "SnapshotPlaybackSession is unavailable"
            )
        if overlay_getter is not None and not capabilities.supports_debug_overlay:
            raise unsupported_debug_overlay_error(type(env).__name__)
        self._env = env
        self._frame_state_getter = frame_state_getter
        self._overlay_getter = overlay_getter
        self._render_spacing = render_spacing
        self._width = int(width)
        self._height = int(height)
        self._num_processes = int(num_processes)
        self._snapshots: list[np.ndarray] = []
        self._overlays: list[Sequence[Sequence[DebugPrimitive] | None] | None] = []

    def __len__(self) -> int:
        return len(self._snapshots)

    @property
    def snapshots(self) -> tuple[np.ndarray, ...]:
        """The cached physics states, in capture order."""
        return tuple(self._snapshots)

    def snapshot(self) -> np.ndarray:
        """Cache one physics snapshot (and the current overlays, if configured)."""
        getter = (
            self._frame_state_getter
            if self._frame_state_getter is not None
            else self._env.get_physics_state_snapshot
        )
        state = np.asarray(getter(), dtype=np.float32).copy()
        self._snapshots.append(state)
        self._overlays.append(self._overlay_getter() if self._overlay_getter is not None else None)
        return state.copy()

    def clear(self) -> None:
        """Drop all cached snapshots and overlays (call between trials)."""
        self._snapshots.clear()
        self._overlays.clear()

    def render_snapshots(
        self,
        *,
        output_video: str | PathLike[str],
        overlay_getter: DebugOverlayGetter | None = None,
        camera: CameraCfg | Mapping[str, Any] | None = None,
        fps: int | None = None,
        on_frame: OnFrameFn | None = None,
    ) -> str | None:
        """Render cached snapshots to ``output_video`` and return its path.

        ``overlay_getter`` here is evaluated once per cached frame at render
        time and overrides the snapshot-time overlays captured through the
        constructor getter; prefer the constructor ``overlay_getter`` for
        state-coupled overlays.  ``camera`` is normalized through
        :meth:`CameraCfg.from_kwargs` (unknown keys fail closed).  ``on_frame``
        receives ``(frame_index, frame)`` after rendering and may return a
        modified frame.  Returns ``None`` when the host cannot render
        off-screen (a warning is printed by the renderer).
        """
        if not self._snapshots:
            raise ValueError("SnapshotPlaybackSession has no cached snapshots to render")
        camera_cfg = CameraCfg.from_kwargs(camera)

        num_envs = int(self._snapshots[0].shape[0])
        overlays: list[Sequence[Sequence[DebugPrimitive] | None] | None]
        if overlay_getter is not None:
            overlays = [overlay_getter() for _ in self._snapshots]
        else:
            overlays = list(self._overlays)
        validated = [validate_debug_overlays(entry, num_envs) for entry in overlays]
        debug_overlays_list = validated if any(entry is not None for entry in validated) else None
        if debug_overlays_list is not None:
            capabilities = getattr(self._env, "play_capabilities", None)
            if capabilities is None or not capabilities.supports_debug_overlay:
                raise unsupported_debug_overlay_error(type(self._env).__name__)

        from unisim.backend.mujoco.playback import resolve_render_play_model_files
        from unisim.backend.playback_common import write_playback_video
        from unisim.visualization import render_many

        spacing = self._render_spacing
        if spacing is None:
            spacing = float(getattr(getattr(self._env, "cfg", None), "render_spacing", 1.0))

        with tempfile.TemporaryDirectory(prefix="unilab-snapshot-session-") as tmp_dir:
            model_files = resolve_render_play_model_files(
                self._env,
                num_envs=num_envs,
                tmp_dir=tmp_dir,
            )
            if camera_cfg.cam_tracking:
                frames = render_many.render_states_get_frames_tracking(
                    list(self._snapshots),
                    model_files,
                    width=self._width,
                    height=self._height,
                    tracking_env_idx=camera_cfg.cam_tracking_env_idx,
                    max_extra_envs=camera_cfg.cam_tracking_extra_envs,
                    cam_distance=camera_cfg.cam_distance,
                    cam_elevation=camera_cfg.cam_elevation,
                    cam_azimuth=camera_cfg.cam_azimuth,
                    cam_fov=camera_cfg.cam_fov,
                    render_spacing=spacing,
                    debug_overlays_list=debug_overlays_list,
                )
            else:
                frames = render_many.render_states_get_frames(
                    list(self._snapshots),
                    model_files,
                    width=self._width,
                    height=self._height,
                    num_processes=self._num_processes,
                    camera_id=-1,
                    cam_distance=camera_cfg.cam_distance,
                    cam_elevation=camera_cfg.cam_elevation,
                    cam_azimuth=camera_cfg.cam_azimuth,
                    cam_lookat=camera_cfg.cam_lookat,
                    cam_fov=camera_cfg.cam_fov,
                    render_spacing=spacing,
                    debug_overlays_list=debug_overlays_list,
                )

        if not frames:
            print(f"[playback] No frames rendered; skipping video export to {output_video}.")
            return None

        if on_frame is not None:
            frames = [
                modified if (modified := on_frame(index, frame)) is not None else frame
                for index, frame in enumerate(frames)
            ]

        if fps is None:
            ctrl_dt = float(getattr(getattr(self._env, "cfg", None), "ctrl_dt", 1.0 / 60.0))
            fps = max(1, int(round(1.0 / ctrl_dt)))
        output = str(output_video)
        write_playback_video(output, frames, fps=fps)
        return output


__all__ = ["OnFrameFn", "SnapshotPlaybackSession"]
