from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MotrixTrackingCamera:
    env_idx: int
    distance: float
    elevation: float
    azimuth: float


@dataclass(frozen=True)
class MotrixCameraView:
    lookat: list[float]
    distance: float
    elevation: float
    azimuth: float
    tracking: MotrixTrackingCamera | None = None


def render_offsets(num_envs: int, spacing: float, offset_mode: str = "grid") -> list[list[float]]:
    if offset_mode == "zero":
        return [[0.0, 0.0, 0.0] for _ in range(num_envs)]
    if offset_mode != "grid":
        raise ValueError(f"Unsupported Motrix render_offset_mode: {offset_mode!r}")
    cols = int(np.ceil(np.sqrt(num_envs)))
    offsets = []
    for i in range(num_envs):
        row = i // cols
        col = i % cols
        offsets.append([col * spacing, row * spacing, 0.0])
    return offsets


def tracking_camera_lookat(
    base_positions: np.ndarray,
    tracking_camera: MotrixTrackingCamera,
    offsets: np.ndarray,
) -> list[float]:
    base_pos = np.asarray(base_positions[tracking_camera.env_idx], dtype=np.float64)
    render_offset = np.asarray(offsets[tracking_camera.env_idx], dtype=np.float64)
    lookat = base_pos + render_offset
    return [float(lookat[0]), float(lookat[1]), float(lookat[2])]


def resolve_system_camera_view(
    num_envs: int,
    base_positions: np.ndarray | None,
    offsets: Sequence[Sequence[float]],
    camera_kwargs: dict[str, Any] | None,
) -> MotrixCameraView:
    cam_kw = dict(camera_kwargs or {})
    if bool(cam_kw.get("cam_tracking", False)):
        if base_positions is None:
            raise ValueError("base_positions is required when cam_tracking=true")
        env_idx = int(cam_kw.get("cam_tracking_env_idx", 0))
        env_idx = max(0, min(env_idx, num_envs - 1))
        tracking_camera = MotrixTrackingCamera(
            env_idx=env_idx,
            distance=float(cam_kw.get("cam_distance", 2.0)),
            elevation=float(cam_kw.get("cam_elevation", -20.0)),
            azimuth=float(cam_kw.get("cam_azimuth", 90.0)),
        )
        lookat = tracking_camera_lookat(
            base_positions,
            tracking_camera,
            np.asarray(offsets, dtype=np.float64),
        )
        return MotrixCameraView(
            lookat=lookat,
            distance=tracking_camera.distance,
            elevation=tracking_camera.elevation,
            azimuth=tracking_camera.azimuth,
            tracking=tracking_camera,
        )

    lookat_raw = cam_kw.get("cam_lookat")
    if lookat_raw is None:
        offsets_np = np.asarray(offsets, dtype=np.float64)
        lookat = [
            float(np.mean(offsets_np[:, 0])),
            float(np.mean(offsets_np[:, 1])),
            0.75,
        ]
    else:
        lookat_arr = np.asarray(lookat_raw, dtype=np.float64).reshape(-1)
        if lookat_arr.shape != (3,):
            raise ValueError(f"cam_lookat must contain 3 values, got {lookat_raw!r}")
        lookat = [float(lookat_arr[0]), float(lookat_arr[1]), float(lookat_arr[2])]

    return MotrixCameraView(
        lookat=lookat,
        distance=float(cam_kw.get("cam_distance", 2.0)),
        elevation=float(cam_kw.get("cam_elevation", -20.0)),
        azimuth=float(cam_kw.get("cam_azimuth", 90.0)),
    )
