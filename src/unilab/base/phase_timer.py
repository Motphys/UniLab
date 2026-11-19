"""Structured per-phase wall/CPU timing collector for one host lifecycle.

`PhaseTimer` replaces hand-written ``perf_counter`` / ``os.times``
bookkeeping in hot environment paths (one ``TorchEnv.step``, one
``_reset_done_envs``, one ``update_state``). Each phase is declared with a
single ``with timer.phase(name):`` statement; the paired wall and process
CPU samples are taken by the timer itself, and derived quantities (elapsed
total, unattributed CPU residual, wall re-attribution between phases) are
computed from the recorded samples at publication time.

The collector is host-only bookkeeping: it performs no device work and never
reads environment state, so it stays outside CUDA-graph-captured segments.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


def _cpu_time() -> float:
    """Process CPU time (user + system) in seconds, for paired CPU samples."""
    process = os.times()
    return process.user + process.system


@dataclass
class _PhaseSample:
    """Accumulated milliseconds for one named phase."""

    wall_ms: float = 0.0
    cpu_ms: float | None = None


class PhaseTimer:
    """Collect paired wall/CPU timings for the phases of one host lifecycle.

    A timer instance covers exactly one lifecycle invocation (for example
    one ``TorchEnv.step``). Construction captures the start clocks; each
    ``phase`` context then accumulates its wall window, and — when CPU
    collection is enabled — its process CPU window. Re-entering the same
    phase name accumulates into the same sample, so a phase split across
    non-contiguous segments (for example a backend step plus its later
    done-gate host read) keeps one declared name.

    Derived values are computed on demand from the recorded samples:
    :meth:`total_wall_ms` for the whole lifecycle, :meth:`residual_cpu_ms`
    for the CPU time not attributed to any CPU-measured phase, and
    :meth:`move_wall` for re-attributing a measured wall window between two
    phases (for example draining an asynchronous backend step out of the
    update-state phase). :meth:`publish` writes the recorded phases into a
    timing mapping under ``"<name>_ms"`` / ``"<name>_cpu_ms"`` keys.
    """

    def __init__(self, *, cpu: bool = True) -> None:
        """Start one lifecycle timing collection.

        Args:
            cpu: Whether phases pair wall windows with process CPU windows
                by default, and whether the CPU residual is computed against
                the construction-time CPU clock. Wall-only collectors (for
                example reset bookkeeping) pass ``cpu=False`` to skip every
                ``os.times`` sample.
        """
        self._cpu_enabled = cpu
        self._wall_start = time.perf_counter()
        self._cpu_start = _cpu_time() if cpu else None
        self._phases: dict[str, _PhaseSample] = {}
        self._marks: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str, *, cpu: bool | None = None) -> Iterator[None]:
        """Accumulate one execution of the named phase into its sample.

        Args:
            name: Phase name. Re-entering an existing name accumulates into
                its sample, preserving one name for logically contiguous
                phases that the code splits into non-adjacent segments.
            cpu: Pair the wall window with a process CPU window. ``None``
                (default) follows the timer-wide ``cpu`` construction flag.

        Yields:
            Nothing; the wrapped block is the measured phase body.
        """
        if cpu is None:
            cpu = self._cpu_enabled
        if cpu and not self._cpu_enabled:
            raise ValueError(f"phase {name!r} requested CPU time on a wall-only timer")
        wall_started = time.perf_counter()
        cpu_started = _cpu_time() if cpu else None
        try:
            yield
        finally:
            self._add(
                name,
                wall_ms=(time.perf_counter() - wall_started) * 1000.0,
                cpu_ms=None if cpu_started is None else (_cpu_time() - cpu_started) * 1000.0,
            )

    def _add(self, name: str, *, wall_ms: float, cpu_ms: float | None) -> None:
        sample = self._phases.get(name)
        if sample is None:
            sample = _PhaseSample()
            self._phases[name] = sample
        sample.wall_ms += wall_ms
        if cpu_ms is not None:
            sample.cpu_ms = cpu_ms if sample.cpu_ms is None else sample.cpu_ms + cpu_ms

    def mark(self, name: str) -> None:
        """Record a wall-clock checkpoint for later :meth:`span_ms` reads.

        Marks cover windows that are not phase bodies (for example a span
        that must include context-manager entry overhead).
        """
        self._marks[name] = time.perf_counter()

    def span_ms(self, start: str, *, end: str | None = None) -> float:
        """Return the wall milliseconds between two checkpoints.

        Args:
            start: The mark recorded with :meth:`mark`.
            end: Optional end mark; ``None`` (default) reads the clock now.
        """
        wall_start = self._marks[start]
        wall_end = time.perf_counter() if end is None else self._marks[end]
        return (wall_end - wall_start) * 1000.0

    def wall_ms(self, name: str) -> float:
        """Return the accumulated wall milliseconds of one phase."""
        return self._phases[name].wall_ms

    def move_wall(self, source: str, target: str, wall_ms: float) -> None:
        """Re-attribute a measured wall window from one phase to another.

        The source phase is debited (floored at zero); the target phase is
        credited with the full window. Used for windows measured inside one
        phase that semantically belong to another, such as an update-state
        host read that drains an asynchronous backend step.
        """
        source_sample = self._phases[source]
        source_sample.wall_ms = max(source_sample.wall_ms - wall_ms, 0.0)
        self._add(target, wall_ms=wall_ms, cpu_ms=None)

    def total_wall_ms(self) -> float:
        """Return the wall milliseconds since timer construction."""
        return (time.perf_counter() - self._wall_start) * 1000.0

    def total_cpu_ms(self) -> float:
        """Return the process CPU milliseconds since timer construction."""
        if self._cpu_start is None:
            raise ValueError("wall-only timer has no CPU clock")
        return (_cpu_time() - self._cpu_start) * 1000.0

    def residual_cpu_ms(self) -> float:
        """Return the CPU milliseconds not attributed to any CPU phase.

        Computed as total CPU time minus the sum of every CPU-measured
        phase, floored at zero.
        """
        attributed = sum(sample.cpu_ms for sample in self._phases.values() if sample.cpu_ms)
        return max(self.total_cpu_ms() - attributed, 0.0)

    def attributed_wall_ms(self) -> float:
        """Return the sum of every recorded phase wall window."""
        return sum(sample.wall_ms for sample in self._phases.values())

    def publish(self, timing: MutableMapping[str, Any]) -> None:
        """Write every recorded phase into a timing mapping.

        Each phase ``name`` publishes ``"<name>_ms"`` for its wall window
        and, when CPU was measured, ``"<name>_cpu_ms"`` for its CPU window.
        Phases publish in declaration order.
        """
        for name, sample in self._phases.items():
            timing[f"{name}_ms"] = sample.wall_ms
            if sample.cpu_ms is not None:
                timing[f"{name}_cpu_ms"] = sample.cpu_ms


__all__ = [
    "PhaseTimer",
]
