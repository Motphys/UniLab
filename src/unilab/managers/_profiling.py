"""Opt-in per-segment timing for manager hot paths (issue #1404).

Enable with ``UNILAB_TERM_PROFILING=1`` (same switch as the #1293 round).
When disabled, ``profile_segment()`` returns a shared no-op context manager,
so each call site costs one attribute check and no allocation. When enabled,
timing overhead (two ``perf_counter`` calls per segment) is accepted — this is
profiling code, not a hot-path feature.

Stats are pulled programmatically via ``SEGMENT_PROFILER.stats()`` by
``scripts/benchmark/env/benchmark_obs_term_profile.py``; nothing is printed
implicitly.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Literal

_ENABLED = os.environ.get("UNILAB_TERM_PROFILING", "") == "1"


class _NullCtx:
    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> Literal[False]:
        return False


class SegmentProfiler:
    """Accumulates wall-clock time per segment key.

    Keys are ``<category>/<group>[/<term>]|<phase>`` where phase is ``step``
    (full-batch per-step path) or ``reset`` (row-scoped reset path).
    """

    def __init__(self) -> None:
        self.enabled = _ENABLED
        self._total_s: dict[str, float] = defaultdict(float)
        self._calls: dict[str, int] = defaultdict(int)

    def reset(self) -> None:
        """Drop accumulated stats (e.g. after benchmark warmup)."""
        self._total_s.clear()
        self._calls.clear()

    @contextmanager
    def time(self, key: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._total_s[key] += time.perf_counter() - t0
            self._calls[key] += 1

    def stats(self) -> dict[str, tuple[float, int]]:
        """Return ``{key: (total_seconds, calls)}`` for accumulated segments."""
        return {key: (self._total_s[key], self._calls[key]) for key in self._total_s}


SEGMENT_PROFILER = SegmentProfiler()

_NULL_CTX = _NullCtx()


def profile_segment(key: str) -> AbstractContextManager[None]:
    """Time one pipeline segment; no-op when profiling is disabled."""
    if SEGMENT_PROFILER.enabled:
        return SEGMENT_PROFILER.time(key)
    return _NULL_CTX
