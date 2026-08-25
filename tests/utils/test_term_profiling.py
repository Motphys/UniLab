"""PROFILING_TEMP (#1293, TODO: remove after #1292): tests for the temporary
per-term profiling instrumentation."""

import time

from unilab.utils.term_profiling import TERM_PROFILER, TermProfiler, profile_term


def test_disabled_by_default_is_noop():
    # Test session never sets UNILAB_TERM_PROFILING, so profiling is disabled.
    assert not TERM_PROFILER.enabled
    with profile_term("x/y|step"):
        pass
    assert not TERM_PROFILER._calls


def test_enabled_profiler_records_and_dumps(capsys):
    profiler = TermProfiler()
    profiler.enabled = True
    with profiler.time("reward/test"):
        time.sleep(0.001)
    with profiler.time("reward/test"):
        pass
    assert profiler._calls["reward/test"] == 2
    assert profiler._total_s["reward/test"] >= 0.001
    profiler.dump()
    out = capsys.readouterr().out
    assert "reward/test" in out
    assert "PROFILING_TEMP" in out


def test_reset_dumps_then_clears(capsys):
    profiler = TermProfiler()
    profiler.enabled = True
    with profiler.time("obs/g/t|step"):
        pass
    profiler.reset()
    assert not profiler._calls
    assert "obs/g/t|step" in capsys.readouterr().out
    # Reset on empty stats prints nothing.
    profiler.reset()
    assert capsys.readouterr().out == ""
