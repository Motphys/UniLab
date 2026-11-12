"""Explicit host-state transitions for CUDA-graph-captured env segments.

A CUDA graph replays device work only. Python state that a captured segment
updates (a "pending metrics" flag, an invalidated cache, a rebinding of a
published carrier) would otherwise keep the value it had after the capture
pass. Code that runs inside a capturable segment therefore declares each such
update as a named transition through :func:`record_host_transition`:

- outside a capture the transition simply runs;
- during a capture it runs once and is recorded on the active
  ``unisim.backend.graph_capture.ExternalGraphCapture``, and the graph owner
  re-applies it after every replay.

Only state that other code changes between two executions of the segment needs
a transition. Assignments whose value is the same on every execution (a flag
set and cleared inside the segment, a binding to a persistent buffer) are
idempotent and replay-correct without one.
"""

from __future__ import annotations

from collections.abc import Callable


def record_host_transition(owner: object, name: str, transition: Callable[[], None]) -> None:
    """Run ``transition`` now and replay it after graph replays of the segment.

    ``owner`` is the environment (or anything exposing
    ``record_host_transition(name, transition)``); owners without it run the
    transition eagerly, which is the correct behavior outside any capture.
    """
    recorder = getattr(owner, "record_host_transition", None)
    if recorder is None:
        transition()
        return
    recorder(name, transition)


__all__ = ["record_host_transition"]
