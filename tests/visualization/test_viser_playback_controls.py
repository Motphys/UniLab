"""Tests for the reusable viser timeline controls bound to a playback clock.

The stub ``gui`` object below mirrors the viser handle surface consumed by
:class:`PlaybackTimelineGui` (``add_slider`` / ``add_checkbox`` /
``add_number`` plus ``on_update`` registration and a mutable ``value``), so no
real viser server is required.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from unilab.visualization.playback_clock import PlaybackClock
from unilab.visualization.viser_playback_controls import PlaybackTimelineGui


class _FakeHandle:
    """Record value assignments and dispatch ``on_update`` callbacks."""

    def __init__(self, initial_value: Any) -> None:
        self.value = initial_value
        self._callbacks: list[Callable[[Any], None]] = []

    def on_update(self, callback: Callable[[Any], None]) -> Callable[[Any], None]:
        """Register one viser-style update callback."""
        self._callbacks.append(callback)
        return callback

    def set(self, value: Any) -> None:
        """Emulate a frontend edit by assigning and firing the callbacks."""
        self.value = value
        for callback in self._callbacks:
            callback(self)


class _FakeGui:
    """Capture the controls created by :class:`PlaybackTimelineGui`."""

    def __init__(self) -> None:
        self.handles: dict[str, _FakeHandle] = {}

    def add_slider(self, name: str, **kwargs: Any) -> _FakeHandle:
        """Create one fake slider handle."""
        handle = _FakeHandle(kwargs["initial_value"])
        self.handles[name] = handle
        return handle

    def add_checkbox(self, name: str, **kwargs: Any) -> _FakeHandle:
        """Create one fake checkbox handle."""
        handle = _FakeHandle(kwargs["initial_value"])
        self.handles[name] = handle
        return handle

    def add_number(self, name: str, **kwargs: Any) -> _FakeHandle:
        """Create one fake number handle."""
        handle = _FakeHandle(kwargs["initial_value"])
        self.handles[name] = handle
        return handle


def _make_controls() -> tuple[PlaybackTimelineGui, PlaybackClock, _FakeGui, list[int]]:
    clock = PlaybackClock([0.0, 0.1, 0.35], speed=1.0, loop=True, playing=False)
    gui = _FakeGui()
    rendered: list[int] = []
    controls = PlaybackTimelineGui(gui, clock, on_seek=rendered.append)
    return controls, clock, gui, rendered


def test_creates_frame_play_speed_loop_controls() -> None:
    """Verify the component creates one control per playback option."""
    clock = PlaybackClock([0.0, 0.1, 0.35], initial_frame=1, speed=2.0, loop=False, playing=True)
    gui = _FakeGui()
    controls = PlaybackTimelineGui(gui, clock)

    assert set(gui.handles) == {"Frame", "Play", "Speed", "Loop"}
    assert controls.slider.value == 1
    assert controls.play.value is True
    assert controls.speed.value == 2.0
    assert controls.loop.value is False


def test_slider_seek_updates_clock_and_invokes_callback() -> None:
    """Verify a user slider edit seeks the clock and triggers a re-render."""
    controls, clock, gui, rendered = _make_controls()

    gui.handles["Frame"].set(2)

    assert clock.current_frame == 2
    assert rendered == [2]


def test_play_speed_loop_callbacks_drive_clock() -> None:
    """Verify the remaining controls mutate the clock without scene jumps."""
    controls, clock, gui, rendered = _make_controls()

    gui.handles["Play"].set(True)
    assert clock.playing is True

    gui.handles["Speed"].set(2.0)
    assert clock.speed == 2.0

    gui.handles["Loop"].set(False)
    assert clock.loop is False


def test_sync_writes_back_slider_and_reports_changed_frame() -> None:
    """Verify the serve-loop writeback mirrors the advanced clock frame."""
    controls, clock, gui, rendered = _make_controls()
    clock.set_playing(True, now=10.0)
    frame = clock.advance(now=10.2)

    assert controls.sync(frame) is True
    assert gui.handles["Frame"].value == frame
    assert rendered == []
    assert controls.sync(frame) is False


def test_sync_writeback_does_not_retrigger_seek() -> None:
    """Verify programmatic slider writebacks are ignored by the seek callback."""
    controls, clock, gui, rendered = _make_controls()

    assert controls.sync(1) is True
    assert clock.current_frame == 0
    assert rendered == []


def test_sync_mirrors_play_state_when_clock_clamps() -> None:
    """Verify the play checkbox resets when the clock clamps at the end."""
    clock = PlaybackClock([0.0, 0.1, 0.35], loop=False, playing=True)
    gui = _FakeGui()
    controls = PlaybackTimelineGui(gui, clock)
    clock.set_playing(True, now=10.0)

    controls.sync(clock.advance(now=100.0))

    assert clock.playing is False
    assert gui.handles["Play"].value is False
    assert gui.handles["Frame"].value == clock.current_frame
