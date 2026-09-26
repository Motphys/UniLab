#!/usr/bin/env python3
"""Extract end-to-end off-policy benchmark metrics from TensorBoard event files.

unilab-rl 1.4.1 replaced the ``timing/*`` and lowercase ``perf/*`` tags with the
canonical schema in ``uni_rl.logging.metric_schema`` (see ``docs/metrics.md`` in
the unilab_rl repo). Historical event files are immutable, so every row below
lists ``(tag, scale)`` candidates newest-first: the first tag present in the
file wins and its values are multiplied by ``scale`` to keep each row's
long-standing unit (milliseconds / per-second rates). Two canonical fields are
seconds where the retired tags were milliseconds: ``Perf/iteration_time``
(was ``perf/iter_ms``) and ``Perf/learning_time`` (was
``timing/learner_train_ms``). Rows whose source chart was retired without a
replacement (collector active throughput, collector cycle total) only carry
their legacy tag and report NaN on post-1.4.1 runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator

TAGS: dict[str, tuple[tuple[str, float], ...]] = {
    "iter_ms": (("Perf/iteration_time", 1000.0), ("perf/iter_ms", 1.0)),
    "steps_per_sec": (("Perf/total_fps", 1.0), ("perf/steps_per_sec", 1.0)),
    "collector_active_steps_per_sec": (("perf/collector_active_steps_per_sec", 1.0),),
    "collector_cycle_ms": (("perf/collector_cycle_ms", 1.0),),
    "collector_learner_action_wait_ms": (
        ("Perf/collector_learner_action_wait_ms", 1.0),
        ("timing/collector_learner_action_wait_ms", 1.0),
    ),
    "collector_replay_write_ms": (
        ("Perf/collector_replay_write_ms", 1.0),
        ("timing/collector_replay_write_ms", 1.0),
    ),
    "learner_collector_wait_ms": (
        ("Perf/learner_collector_wait_ms", 1.0),
        ("timing/learner_collector_wait_ms", 1.0),
    ),
    "learner_inference_ms": (
        ("Perf/learner_inference_ms", 1.0),
        ("timing/learner_inference_ms", 1.0),
    ),
    "learner_replay_batch_wait_ms": (
        ("Perf/learner_replay_batch_wait_ms", 1.0),
        ("timing/learner_replay_batch_wait_ms", 1.0),
    ),
    "learner_replay_sample_ms": (
        ("Perf/learner_replay_sample_ms", 1.0),
        ("timing/learner_replay_sample_ms", 1.0),
    ),
    "replay_ingress_h2d_submit_ms": (
        ("Perf/replay_ingress_h2d_submit_ms", 1.0),
        ("timing/replay_ingress_h2d_submit_ms", 1.0),
    ),
    "learner_collector_release_ms": (
        ("Perf/learner_collector_release_ms", 1.0),
        ("timing/learner_collector_release_ms", 1.0),
    ),
    "learner_train_ms": (("Perf/learning_time", 1000.0), ("timing/learner_train_ms", 1.0)),
}


def _find_event_file(log_dir: Path) -> Path | None:
    candidates = list(log_dir.rglob("events.out.tfevents.*"))
    if not candidates:
        return None
    # Prefer the deepest (most specific) event file.
    return sorted(candidates, key=lambda p: len(str(p)))[-1]


def _average_last(scalars: list[event_accumulator.ScalarEvent], n: int) -> float:
    values = [s.value for s in scalars]
    if not values:
        return float("nan")
    return sum(values[-n:]) / len(values[-n:])


def _extract_row(
    ea: event_accumulator.EventAccumulator,
    candidates: tuple[tuple[str, float], ...],
    last: int,
) -> float:
    available = set(ea.Tags()["scalars"])
    for tag, scale in candidates:
        if tag in available:
            return _average_last(ea.Scalars(tag), last) * scale
    return float("nan")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--last", type=int, default=20, help="Average over the last N iterations")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args(argv)

    event_file = _find_event_file(args.log_dir)
    if event_file is None:
        print(f"No event file found under {args.log_dir}", file=sys.stderr)
        return 1

    ea = event_accumulator.EventAccumulator(str(event_file))
    ea.Reload()

    rows = [(label, _extract_row(ea, candidates, args.last)) for label, candidates in TAGS.items()]

    if args.json:
        import json

        print(json.dumps({label: value for label, value in rows}, indent=2))
    else:
        for label, value in rows:
            print(f"{label:40} {value:10.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
