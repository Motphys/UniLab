"""Shared collector-metrics drain for the async runners.

``APPORunner`` and ``OffPolicyRunner`` consume the same collector metrics
message protocol; this module owns the shared dispatch so each runner only
keeps a thin wrapper encoding its own semantics (collector-error propagation,
``buffer_size`` handling, trace events).
"""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RewardComponentWindow:
    """Aggregate reward-term reports between learner iterations.

    Tensor-native interval reports already carry per-step component means, so
    the window aggregates them as a mean of per-step means.
    """

    _sums: dict[str, float] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)

    def update(self, components: dict[str, float]) -> None:
        for term, value in components.items():
            self._sums[term] = self._sums.get(term, 0.0) + float(value)
            self._counts[term] = self._counts.get(term, 0) + 1

    def take(self) -> dict[str, float]:
        if not self._sums:
            return {}
        values = {
            term: total / max(self._counts.get(term, 1), 1) for term, total in self._sums.items()
        }
        self._sums.clear()
        self._counts.clear()
        return values


def drain_collector_metrics(
    queue: Any,
    reward_history: deque,
    reward_components: RewardComponentWindow,
    logger: Any,
    trace_recorder: Any | None = None,
    *,
    runner_label: str,
    raise_on_collector_error: bool,
    require_buffer_size: bool,
) -> None:
    """Drain all pending collector metrics messages and dispatch them to the logger.

    ``runner_label`` prefixes the stderr drain-error line.
    ``raise_on_collector_error`` controls whether a collector ``error`` message
    propagates as ``RuntimeError`` (off-policy) or is reported on stderr and
    swallowed (APPO). ``require_buffer_size`` controls whether ``log_collector``
    requires a ``buffer_size`` key (off-policy replay) or defaults it to 0
    (APPO uses shared memory, not a separate buffer).
    """
    while True:
        try:
            metrics = queue.get_nowait()
        except Exception:
            break
        if "error" in metrics:
            logger.log_status(f"[red]Collector ERROR: {metrics['error']}[/]")
            error = RuntimeError(f"Collector process failed: {metrics['error']}")
            if raise_on_collector_error:
                raise error
            print(f"[{runner_label}] metrics drain error: {error}", file=sys.stderr)
            break

        try:
            if "runtime_manifest" in metrics:
                logger.update_runtime_manifest(metrics["runtime_manifest"])
            if "return_mean_ep100" in metrics:
                reward_history.append(metrics["return_mean_ep100"])
            if "reward_components" in metrics:
                reward_components.update(metrics["reward_components"])
            if "curriculum_metrics" in metrics:
                logger.update_curriculum_metrics(metrics["curriculum_metrics"])
            if "mean_episode_length" in metrics:
                logger.update_mean_episode_length(metrics["mean_episode_length"])
            if "collector_timing_ms" in metrics:
                logger.update_collector_timing(metrics["collector_timing_ms"])
            if "collector_inference" in metrics:
                logger.update_collector_inference(metrics["collector_inference"])
            if "timeout_rate" in metrics:
                logger.update_timeout_rate(float(metrics["timeout_rate"]))
            if "total_steps" in metrics and (not require_buffer_size or "buffer_size" in metrics):
                logger.log_collector(
                    metrics["total_steps"],
                    metrics.get("buffer_size", 0),
                )
            if trace_recorder and "trace_events" in metrics:
                trace_recorder.extend(metrics["trace_events"])
        except Exception as exc:
            print(f"[{runner_label}] metrics drain error: {exc}", file=sys.stderr)
            break
