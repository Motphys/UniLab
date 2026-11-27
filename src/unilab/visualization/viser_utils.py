"""Small viser playback/rendering helpers shared by offline viewers.

Ported from downstream leanretarget's Viser visualization boundary.  These
helpers are viser-agnostic numpy / handle utilities: they operate on
timestamps, world-space vectors, and duck-typed scene handles, so tests can
exercise them without a running viser server.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def set_handle_property(handle: Any, name: str, value: Any) -> None:
    """Assign one viser handle property only when its represented value changes.

    Args:
        handle: Duck-typed viser scene/GUI handle exposing ``name``.
        name: Property to assign.
        value: New value; numpy scalars are unwrapped and arrays compared
            elementwise against the current value.

    Returns:
        None.
    """
    if isinstance(value, np.generic):
        value = value.item()
    current = getattr(handle, name)
    if isinstance(value, np.ndarray):
        unchanged = np.array_equal(np.asarray(current), value)
    else:
        unchanged = current == value
    if not unchanged:
        setattr(handle, name, value)


def nominal_fps(timestamps: np.ndarray | Sequence[float]) -> float | None:
    """Estimate the nominal sampling frequency from stored timestamps.

    A timeline may retain an exact source endpoint, leaving one final interval
    shorter than the configured sampling period.  The median interval therefore
    represents the nominal preprocessing FPS more faithfully than total frame
    count divided by duration.

    Args:
        timestamps: Strictly increasing trajectory timestamps.

    Returns:
        Positive nominal frames per second, or ``None`` for a single frame.
    """
    values = np.asarray(timestamps, dtype=np.float64)
    if values.size < 2:
        return None
    period = float(np.median(np.diff(values)))
    return 1.0 / period


def cylinder_transform(
    start: np.ndarray, stop: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Compute a unit-z cylinder transform spanning two world-space points.

    Args:
        start: First world-space endpoint.
        stop: Second world-space endpoint.

    Returns:
        Midpoint, wxyz orientation, and xyz scale for a unit-height cylinder.
    """
    start_value = np.asarray(start, dtype=np.float32)
    stop_value = np.asarray(stop, dtype=np.float32)
    direction = stop_value - start_value
    length = float(np.linalg.norm(direction))
    midpoint = (start_value + stop_value) / 2.0
    if length <= 1.0e-9:
        return midpoint, np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (1.0, 1.0, 1.0e-9)
    unit = direction / length
    dot = float(unit[2])
    if dot < -1.0 + 1.0e-7:
        quaternion = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    else:
        cross = np.asarray([-unit[1], unit[0], 0.0], dtype=np.float64)
        quaternion = np.asarray([1.0 + dot, *cross], dtype=np.float64)
        quaternion /= np.linalg.norm(quaternion)
        quaternion = quaternion.astype(np.float32)
    return midpoint, quaternion, (1.0, 1.0, length)


def scaled_vector_endpoint(
    origin: np.ndarray,
    vector: np.ndarray,
    scale: float,
    maximum_length: float,
) -> np.ndarray:
    """Scale a diagnostic vector to a bounded world-space display endpoint.

    Args:
        origin: World-space vector tail.
        vector: Diagnostic vector, scaled by ``scale`` for display.
        scale: Display multiplier applied to ``vector``.
        maximum_length: Display-length cap applied after scaling.

    Returns:
        World-space endpoint of the bounded display vector.
    """
    origin_value = np.asarray(origin, dtype=np.float64)
    display = np.asarray(vector, dtype=np.float64) * scale
    length = float(np.linalg.norm(display))
    if length > maximum_length:
        display *= maximum_length / length
    return np.asarray(origin_value + display, dtype=np.float64)
