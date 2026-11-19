"""Unit tests for the structured phase timer used by the env step lifecycle."""

from __future__ import annotations

import pytest

from unilab.base.phase_timer import PhaseTimer


def test_phase_accumulates_repeated_entries_into_one_sample() -> None:
    timer = PhaseTimer(cpu=False)
    with timer.phase("step_core"):
        pass
    first = timer.wall_ms("step_core")
    with timer.phase("step_core"):
        pass

    assert timer.wall_ms("step_core") >= first


def test_publish_writes_paired_keys_and_skips_unmeasured_cpu() -> None:
    timer = PhaseTimer()
    with timer.phase("action_validate", cpu=False):
        pass
    with timer.phase("apply_action"):
        pass

    timing: dict[str, float] = {}
    timer.publish(timing)

    assert set(timing) == {
        "action_validate_ms",
        "apply_action_ms",
        "apply_action_cpu_ms",
    }
    assert all(value >= 0.0 for value in timing.values())


def test_move_wall_reattributes_and_floors_source_at_zero() -> None:
    timer = PhaseTimer(cpu=False)
    with timer.phase("update_state"):
        pass
    with timer.phase("step_core"):
        pass
    update_before = timer.wall_ms("update_state")
    step_before = timer.wall_ms("step_core")

    timer.move_wall("update_state", "step_core", 0.5)
    assert timer.wall_ms("step_core") == pytest.approx(step_before + 0.5)
    assert timer.wall_ms("update_state") == pytest.approx(max(update_before - 0.5, 0.0))

    timer.move_wall("update_state", "step_core", 10.0)
    assert timer.wall_ms("update_state") == 0.0


def test_residual_cpu_is_total_minus_attributed_and_floored() -> None:
    timer = PhaseTimer()
    with timer.phase("apply_action"):
        pass
    with timer.phase("action_validate", cpu=False):
        pass

    assert timer.residual_cpu_ms() >= 0.0
    assert timer.total_cpu_ms() >= 0.0


def test_wall_only_timer_rejects_cpu_phases_and_cpu_totals() -> None:
    timer = PhaseTimer(cpu=False)
    with timer.phase("terminal_obs"):
        pass
    with pytest.raises(ValueError, match="wall-only timer"):
        timer.total_cpu_ms()
    with pytest.raises(ValueError, match="wall-only timer"):
        with timer.phase("other", cpu=True):
            pass


def test_span_ms_measures_between_mark_and_now() -> None:
    timer = PhaseTimer(cpu=False)
    timer.mark("queue")
    with timer.phase("work"):
        pass
    span = timer.span_ms("queue")

    assert span >= timer.wall_ms("work")
