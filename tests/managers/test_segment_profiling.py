"""Tests for the opt-in manager segment profiler (issue #1404)."""

from __future__ import annotations

import unilab.managers._profiling as profiling
from unilab.managers._profiling import SEGMENT_PROFILER, profile_segment


def test_disabled_by_default_returns_shared_null_ctx() -> None:
    assert not SEGMENT_PROFILER.enabled
    ctx = profile_segment("term/group/name|step")
    assert ctx is profiling._NULL_CTX
    with ctx:
        pass
    assert SEGMENT_PROFILER.stats() == {}


def test_enabled_accumulates_and_resets(monkeypatch) -> None:
    monkeypatch.setattr(SEGMENT_PROFILER, "enabled", True)
    try:
        with profile_segment("noise/actor/base_lin_vel|step"):
            pass
        with profile_segment("noise/actor/base_lin_vel|step"):
            pass
        with profile_segment("term/actor/base_lin_vel|step"):
            pass
        stats = SEGMENT_PROFILER.stats()
        assert stats["noise/actor/base_lin_vel|step"][1] == 2
        assert stats["term/actor/base_lin_vel|step"][1] == 1
        assert all(total_s >= 0.0 for total_s, _ in stats.values())
        SEGMENT_PROFILER.reset()
        assert SEGMENT_PROFILER.stats() == {}
    finally:
        SEGMENT_PROFILER.reset()
