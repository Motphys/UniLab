"""Genesis-owned playback execution helpers (in-process native rendering).

Mirrors ``isaacgym/playback.py``: the interactive path drives a post-build
Genesis viewer at ~60 Hz; the record path captures offscreen camera frames
headlessly and writes them through the shared ``write_playback_video``.
Genesis attaches viewers/cameras lazily after ``scene.build`` (verified on
genesis-world 1.3.3), so rendering never touches the training hot path.
"""

from __future__ import annotations

import os
import time
from os import PathLike
from typing import Any, Callable, TypeVar

import numpy as np

from unilab.base.backend.playback_common import env_cfg_value, write_playback_video

ObsT = TypeVar("ObsT")


def display_available() -> bool:
    """Return whether a display is reachable for the interactive viewer."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def camera_pose_from_kwargs(
    camera_kwargs: dict[str, Any] | None, lookat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map repo-wide MuJoCo-style spherical camera kwargs onto (pos, lookat).

    MuJoCo's negative-elevation convention is kept: ``cam_elevation=-20``
    places the camera above the horizon looking down. ``cam_lookat`` pins the
    lookat point when set.
    """
    kwargs = dict(camera_kwargs or {})
    distance = float(kwargs.get("cam_distance", kwargs.get("distance", 2.0)))
    elevation_deg = float(kwargs.get("cam_elevation", -20.0))
    azimuth_deg = float(kwargs.get("cam_azimuth", 90.0))
    lookat_override = kwargs.get("cam_lookat")
    target = (
        np.asarray(lookat_override, dtype=np.float64)
        if lookat_override is not None
        else np.asarray(lookat, dtype=np.float64)
    )
    elevation = np.deg2rad(elevation_deg)
    azimuth = np.deg2rad(azimuth_deg)
    offset = distance * np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            -np.sin(elevation),
        ]
    )
    return target + offset, target


def camera_pose_matrix_z_up(pos: np.ndarray, lookat: np.ndarray) -> np.ndarray:
    """Build the 4x4 camera-to-world matrix for a Z-up world.

    The interactive viewer's ``set_camera_pose(pos, lookat)`` reuses the
    viewer's current ``_camera_up``, which genesis initializes from its own
    default camera pose — a tilted vector, not world Z-up (#1396). Passing the
    full ``pose=`` matrix bypasses that polluted state entirely.
    """
    pos = np.asarray(pos, dtype=np.float64)
    lookat = np.asarray(lookat, dtype=np.float64)
    z_axis = pos - lookat
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(np.array([0.0, 0.0, 1.0]), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 0] = x_axis
    transform[:3, 1] = y_axis
    transform[:3, 2] = z_axis
    transform[:3, 3] = pos
    return transform


def run_genesis_playback(
    *,
    backend: Any,
    env: Any,
    initialize: Callable[[], ObsT],
    step: Callable[[ObsT], ObsT],
    num_steps: int | None,
    output_video: str | PathLike[str] | None,
    headless: bool,
    record_video: bool,
    camera_kwargs: dict[str, Any] | None,
) -> str | None:
    if record_video and not headless:
        raise ValueError("genesis video recording requires headless=true.")

    if headless or record_video:
        if num_steps is None:
            raise ValueError("genesis captured playback requires a finite num_steps value.")
        if record_video and output_video is None:
            raise ValueError("genesis video recording requires an output_video path.")

        backend.init_renderer(
            headless=headless,
            capture=True,
            width=1280,
            height=720,
            camera_kwargs=dict(camera_kwargs or {}),
        )

        obs = initialize()
        frames: list[np.ndarray] | None = [] if record_video else None
        for _ in range(num_steps):
            obs = step(obs)
            frame = np.asarray(backend.capture_video_frame(), dtype=np.uint8)
            if frames is not None:
                frames.append(frame.copy())

        if not record_video:
            return None

        assert output_video is not None
        assert frames is not None
        ctrl_dt = float(env_cfg_value(env, "ctrl_dt", 1.0 / 60.0))
        write_playback_video(str(output_video), frames, fps=int(1.0 / ctrl_dt))
        return str(output_video)

    backend.init_renderer(headless=False, camera_kwargs=dict(camera_kwargs or {}))
    obs = initialize()
    last_render_time = time.perf_counter()
    render_dt = 1.0 / 60.0
    steps_run = 0

    while num_steps is None or steps_run < num_steps:
        obs = step(obs)
        current_time = time.perf_counter()
        elapsed = current_time - last_render_time
        if elapsed < render_dt:
            time.sleep(render_dt - elapsed)
        last_render_time = time.perf_counter()
        backend.render()
        steps_run += 1
    return None
