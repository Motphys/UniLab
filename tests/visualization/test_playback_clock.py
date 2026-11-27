"""Tests for the offline playback clock over non-uniform timestamp grids."""

from __future__ import annotations

import pytest

from unilab.visualization.playback_clock import PlaybackClock


def test_playback_clock_uses_irregular_timestamps_and_clamps() -> None:
    """Verify playback does not assume a fixed frame frequency."""
    clock = PlaybackClock(
        timestamps=[0.0, 0.1, 0.35],
        initial_frame=0,
        speed=1.0,
        loop=False,
        playing=False,
    )
    clock.set_playing(True, now=10.0)

    assert clock.advance(now=10.2) == 1
    assert clock.advance(now=10.5) == 2
    assert clock.playing is False


def test_playback_clock_loops_without_accumulating_fixed_fps_steps() -> None:
    """Verify loop behavior maps elapsed time directly onto scene timestamps."""
    clock = PlaybackClock(
        timestamps=[0.0, 0.1, 0.35],
        speed=1.0,
        loop=True,
        playing=False,
    )
    clock.set_playing(True, now=5.0)

    assert clock.advance(now=5.46) == 1
    assert clock.playing is True

    assert clock.advance(now=5.70) == 0
    assert clock.advance(now=6.00) == 1
    assert clock.advance(now=6.10) == 0
    assert clock.playing is True


def test_playback_clock_rejects_invalid_arguments() -> None:
    """Verify construction validates the timestamp grid and playback options."""
    with pytest.raises(ValueError, match="non-empty vector"):
        PlaybackClock([])
    with pytest.raises(ValueError, match="exceeds the final trajectory frame"):
        PlaybackClock([0.0, 1.0], initial_frame=2)
    with pytest.raises(ValueError, match="speed must be positive"):
        PlaybackClock([0.0, 1.0], speed=0.0)
    clock = PlaybackClock([0.0, 1.0])
    with pytest.raises(ValueError, match="outside the trajectory"):
        clock.seek(2, now=0.0)
    with pytest.raises(ValueError, match="speed must be positive"):
        clock.set_speed(-1.0, now=0.0)


def test_playback_clock_set_speed_preserves_displayed_frame() -> None:
    """Verify a mid-playback speed change keeps the current frame."""
    clock = PlaybackClock([0.0, 0.1, 0.35], speed=1.0, loop=False, playing=False)
    clock.set_playing(True, now=10.0)
    assert clock.advance(now=10.2) == 1

    clock.set_speed(2.0, now=10.2)
    assert clock.current_frame == 1
    assert clock.advance(now=10.35) == 2
