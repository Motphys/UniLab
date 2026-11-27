"""Reusable viser timeline controls wired to a :class:`PlaybackClock`.

Ported from the duplicated GUI skeleton in downstream leanretarget's Viser
playback viewers.  The component creates the ``Frame`` slider plus
``Play``/``Speed``/``Loop`` controls on a duck-typed viser ``gui`` object and
connects them to one :class:`~unilab.visualization.playback_clock.PlaybackClock`.
The serve loop advances the clock and calls :meth:`sync` each iteration; seek
events flow back through the ``on_seek`` callback so the caller can re-render.
Tests can substitute a stub ``gui`` object since only ``add_slider`` /
``add_checkbox`` / ``add_number`` and the returned handle attributes are used.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from unilab.visualization.playback_clock import PlaybackClock


class PlaybackTimelineGui:
    """Frame slider and Play/Speed/Loop controls bound to one playback clock.

    The slider's update callback seeks the clock and invokes ``on_seek`` with
    the selected frame; programmatic slider writebacks from :meth:`sync` are
    guarded by an internal re-entrancy flag so they never trigger seek
    callbacks.  Mutations from viser GUI callbacks and the serve loop are
    serialized by an internal reentrant lock, replacing the per-viewer lock
    around the same pattern.
    """

    def __init__(
        self,
        gui: Any,
        clock: PlaybackClock,
        *,
        on_seek: Callable[[int], None] | None = None,
        speed_min: float = 0.05,
        speed_max: float = 8.0,
        speed_step: float = 0.05,
    ) -> None:
        """Create the timeline controls and wire them to ``clock``.

        Args:
            gui: Duck-typed viser GUI registry (``server.gui``).
            clock: Playback clock driving the timeline.
            on_seek: Optional callback invoked with the selected frame after
                every user slider seek.
            speed_min: Lower bound of the Speed number input.
            speed_max: Upper bound of the Speed number input.
            speed_step: Step of the Speed number input.

        Returns:
            None.
        """
        self.clock = clock
        self._on_seek = on_seek
        self._lock = threading.RLock()
        self._updating_slider = False
        self.slider = gui.add_slider(
            "Frame",
            min=0,
            max=clock.timestamps.size - 1,
            step=1,
            initial_value=clock.current_frame,
        )
        self.play = gui.add_checkbox("Play", initial_value=clock.playing)
        self.speed = gui.add_number(
            "Speed", initial_value=clock.speed, min=speed_min, max=speed_max, step=speed_step
        )
        self.loop = gui.add_checkbox("Loop", initial_value=clock.loop)

        @self.slider.on_update
        def _seek(_: Any) -> None:
            with self._lock:
                if self._updating_slider:
                    return
                frame = int(self.slider.value)
                self.clock.seek(frame)
                if self._on_seek is not None:
                    self._on_seek(frame)

        @self.play.on_update
        def _toggle_play(_: Any) -> None:
            with self._lock:
                self.clock.set_playing(bool(self.play.value))

        @self.speed.on_update
        def _set_speed(_: Any) -> None:
            with self._lock:
                self.clock.set_speed(float(self.speed.value))

        @self.loop.on_update
        def _set_loop(_: Any) -> None:
            self.clock.loop = bool(self.loop.value)

    def sync(self, frame: int) -> bool:
        """Write back the slider and play checkbox after a clock advance.

        Args:
            frame: Current clock frame to display.

        Returns:
            ``True`` when the displayed frame changed and the caller should
            re-render; ``False`` when the slider already showed ``frame``
            (e.g. the user just seeked and the callback already rendered).
        """
        changed = False
        if int(self.slider.value) != frame:
            with self._lock:
                self._updating_slider = True
                try:
                    self.slider.value = frame
                finally:
                    self._updating_slider = False
            changed = True
        if bool(self.play.value) != self.clock.playing:
            self.play.value = self.clock.playing
        return changed
