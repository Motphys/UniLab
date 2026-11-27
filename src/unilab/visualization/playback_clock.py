"""Offline playback clock mapping wall time onto arbitrary timestamps.

Ported from downstream leanretarget's Viser playback viewers, where the clock
drives non-uniform HumanDemo and robot-reference timelines.  The clock is
rendering-backend agnostic: any loop that can supply ``time.monotonic()``
values can advance it, and deterministic tests can inject explicit ``now``
values.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np


class PlaybackClock:
    """Map monotonic wall time onto a non-uniform timestamp grid.

    The clock is anchored at one discrete frame; every mutator re-anchors so
    that pausing, seeking, or changing speed never introduces a scene-time
    jump.  When playback runs past the final timestamp it either wraps to the
    first timestamp (``loop=True``) or clamps and pauses (``loop=False``).
    """

    def __init__(
        self,
        timestamps: np.ndarray | Sequence[float],
        initial_frame: int = 0,
        speed: float = 1.0,
        loop: bool = True,
        playing: bool = False,
    ) -> None:
        """Create a playback clock anchored at one discrete frame.

        Args:
            timestamps: Strictly increasing trajectory timestamps.
            initial_frame: Initial discrete frame index.
            speed: Positive scene-time multiplier.
            loop: Whether playback wraps at the final timestamp.
            playing: Whether the clock initially advances.

        Returns:
            None.
        """
        self.timestamps = np.asarray(timestamps, dtype=np.float64)
        if self.timestamps.ndim != 1 or self.timestamps.size < 1:
            raise ValueError("playback timestamps must be a non-empty vector")
        if initial_frame >= self.timestamps.size:
            raise ValueError("initial_frame exceeds the final trajectory frame")
        if speed <= 0.0:
            raise ValueError("playback speed must be positive")
        self.current_frame = int(initial_frame)
        self.speed = float(speed)
        self.loop = bool(loop)
        self.playing = bool(playing)
        now = time.monotonic()
        self._anchor_wall_time = now
        self._anchor_scene_time = float(self.timestamps[self.current_frame])

    def seek(self, frame: int, now: float | None = None) -> int:
        """Move to a discrete frame and reset the wall-time anchor.

        Args:
            frame: Requested frame index.
            now: Optional monotonic time supplied by deterministic tests.

        Returns:
            Selected frame index.
        """
        if frame < 0 or frame >= self.timestamps.size:
            raise ValueError("playback frame is outside the trajectory")
        self.current_frame = int(frame)
        self._anchor_wall_time = time.monotonic() if now is None else float(now)
        self._anchor_scene_time = float(self.timestamps[self.current_frame])
        return self.current_frame

    def set_playing(self, playing: bool, now: float | None = None) -> None:
        """Pause or resume without introducing a scene-time jump.

        Args:
            playing: Requested playback state.
            now: Optional monotonic time supplied by deterministic tests.

        Returns:
            None.
        """
        current_now = time.monotonic() if now is None else float(now)
        if self.playing:
            self.advance(current_now)
        self.playing = bool(playing)
        self._anchor_wall_time = current_now
        self._anchor_scene_time = float(self.timestamps[self.current_frame])

    def set_speed(self, speed: float, now: float | None = None) -> None:
        """Change speed while preserving the currently displayed frame.

        Args:
            speed: Positive scene-time multiplier.
            now: Optional monotonic time supplied by deterministic tests.

        Returns:
            None.
        """
        if speed <= 0.0:
            raise ValueError("playback speed must be positive")
        current_now = time.monotonic() if now is None else float(now)
        if self.playing:
            self.advance(current_now)
        self.speed = float(speed)
        self._anchor_wall_time = current_now
        self._anchor_scene_time = float(self.timestamps[self.current_frame])

    def advance(self, now: float | None = None) -> int:
        """Return the frame corresponding to the current monotonic time.

        Args:
            now: Optional monotonic time supplied by deterministic tests.

        Returns:
            Current discrete frame after applying loop or clamp behavior.
        """
        if not self.playing or self.timestamps.size == 1:
            return self.current_frame
        current_now = time.monotonic() if now is None else float(now)
        target = self._anchor_scene_time + (current_now - self._anchor_wall_time) * self.speed
        start = float(self.timestamps[0])
        stop = float(self.timestamps[-1])
        if target > stop:
            if self.loop and stop > start:
                target = start + ((target - start) % (stop - start))
                self._anchor_wall_time = current_now
                self._anchor_scene_time = target
            else:
                target = stop
                self.playing = False
        frame = int(np.searchsorted(self.timestamps, target, side="right") - 1)
        self.current_frame = max(0, min(frame, self.timestamps.size - 1))
        return self.current_frame
