#!/usr/bin/env python3
"""Extract end-to-end off-policy benchmark metrics from TensorBoard event files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator

TAGS = {
    "iter_ms": "perf/iter_ms",
    "steps_per_sec": "perf/steps_per_sec",
    "collector_active_steps_per_sec": "perf/collector_active_steps_per_sec",
    "collector_replay_write_ms": "timing/collector_replay_write_ms",
    "learner_collector_wait_ms": "timing/learner_collector_wait_ms",
    "learner_replay_batch_wait_ms": "timing/learner_replay_batch_wait_ms",
    "learner_replay_sample_ms": "timing/learner_replay_sample_ms",
    "learner_incremental_h2d_ms": "timing/learner_incremental_h2d_ms",
    "learner_collector_release_ms": "timing/learner_collector_release_ms",
    "learner_train_ms": "timing/learner_train_ms",
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

    available_tags = {tag: False for tag in TAGS.values()}
    for tag in ea.Tags()["scalars"]:
        available_tags[tag] = True

    rows = []
    for label, tag in TAGS.items():
        if not available_tags.get(tag, False):
            rows.append((label, float("nan")))
            continue
        scalars = ea.Scalars(tag)
        rows.append((label, _average_last(scalars, args.last)))

    if args.json:
        import json

        print(json.dumps({label: value for label, value in rows}, indent=2))
    else:
        for label, value in rows:
            print(f"{label:40} {value:10.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
