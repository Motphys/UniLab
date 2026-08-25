"""PROFILING_TEMP (issue #1293): temporary per-term timing for manager hot paths.

TODO(#1292 cleanup): this module and every ``PROFILING_TEMP`` call site are
temporary instrumentation. Delete them once the #1292 optimization sub-issues
(#1294/#1295/#1296) are complete.

Enable with ``UNILAB_TERM_PROFILING=1``. When disabled, ``profile_term()``
returns a shared no-op context manager, so each call site costs one attribute
check and no allocation. When enabled, timing overhead (two ``perf_counter``
calls per term) is accepted — this is profiling code, not a hot-path feature.

Stats accumulate per benchmark case: ``ManagerBasedRlEnv._load_managers``
resets the profiler (dumping the previous case first), and an atexit hook
dumps the final case.
"""

from __future__ import annotations

import atexit
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


class TermProfiler:
    """Accumulates wall-clock time per term key; mean_ms ≈ ms/vector-step for
    step-phase terms (one call per term per vector step)."""

    def __init__(self) -> None:
        self.enabled = _ENABLED
        self._total_s: dict[str, float] = defaultdict(float)
        self._calls: dict[str, int] = defaultdict(int)

    def reset(self) -> None:
        """Drop accumulated stats, dumping them first (new env = new case)."""
        if self.enabled and self._calls:
            self.dump()
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

    def dump(self) -> None:
        lines = [
            "[TERM_PROFILING] per-term timing (PROFILING_TEMP, issue #1293; "
            "mean_ms ≈ ms/vector-step for step-phase terms)",
            f"{'term':<64} {'calls':>8} {'total_ms':>12} {'mean_ms':>10}",
        ]
        for key, total in sorted(self._total_s.items(), key=lambda kv: -kv[1]):
            calls = self._calls[key]
            lines.append(f"{key:<64} {calls:>8} {total * 1e3:>12.3f} {total / calls * 1e3:>10.4f}")
        print("\n".join(lines), flush=True)


TERM_PROFILER = TermProfiler()

_NULL_CTX = _NullCtx()


def profile_term(key: str) -> AbstractContextManager[None]:
    """PROFILING_TEMP (#1293): time one term call; no-op when disabled."""
    if TERM_PROFILER.enabled:
        return TERM_PROFILER.time(key)
    return _NULL_CTX


if _ENABLED:
    atexit.register(TERM_PROFILER.dump)
