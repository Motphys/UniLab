"""Analyze UniLab off-policy Perfetto traces."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def _percentile(values: list[float], pct: float) -> float:
    xs = sorted(values)
    k = (len(xs) - 1) * pct / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[f]
    return xs[f] * (c - k) + xs[c] * (k - f)


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
    }


def _format_stats(
    label: str, stats: dict[str, float], *, steps_per_cycle: int | None = None
) -> None:
    suffix = ""
    if steps_per_cycle is not None:
        suffix = f" steps/s={steps_per_cycle / (stats['mean'] / 1000.0):.3f}"
    print(
        f"{label}: n={int(stats['n'])} mean={stats['mean']:.3f}ms "
        f"median={stats['median']:.3f}ms p90={stats['p90']:.3f}ms "
        f"p95={stats['p95']:.3f}ms p99={stats['p99']:.3f}ms{suffix}"
    )


def _event_durations_ms(trace_events: list[dict], name: str) -> list[float]:
    return [
        float(event["dur"]) / 1000.0
        for event in trace_events
        if event.get("name") == name and event.get("ph") == "X" and "dur" in event
    ]


def _event_end_to_next_start_ms(trace_events: list[dict], name: str) -> list[float]:
    slices = sorted(
        (
            (float(event["ts"]), float(event["dur"]))
            for event in trace_events
            if event.get("name") == name and event.get("ph") == "X" and "dur" in event
        ),
        key=lambda item: item[0],
    )
    return [
        max(slices[i + 1][0] - (slices[i][0] + slices[i][1]), 0.0) / 1000.0
        for i in range(len(slices) - 1)
    ]


def _iteration_resume_gap_ms(trace_events: list[dict]) -> list[float]:
    weight_writes = sorted(
        (
            (float(event["ts"]), float(event["dur"]))
            for event in trace_events
            if event.get("name") == "learner/weight_sync_write"
            and event.get("ph") == "X"
            and "dur" in event
        ),
        key=lambda item: item[0],
    )
    first_updates = sorted(
        float(event["ts"])
        for event in trace_events
        if event.get("name") == "learner/update_critic"
        and event.get("ph") == "X"
        and event.get("args", {}).get("update_idx") == 0
    )
    gaps: list[float] = []
    update_pos = 0
    for start, dur in weight_writes:
        end = start + dur
        while update_pos < len(first_updates) and first_updates[update_pos] <= end:
            update_pos += 1
        if update_pos < len(first_updates):
            gaps.append((first_updates[update_pos] - end) / 1000.0)
    return gaps


def analyze(path: Path, *, steps_per_cycle: int, drop_first: int, events: list[str]) -> None:
    with path.open() as f:
        trace_events = json.load(f)["traceEvents"]

    waits = sorted(
        float(event["ts"])
        for event in trace_events
        if event.get("name") == "learner/wait_for_data" and event.get("ph") == "X"
    )
    gaps = [(waits[i + 1] - waits[i]) / 1000.0 for i in range(len(waits) - 1)]
    cycle = gaps[drop_first:]
    print(path)
    if cycle:
        _format_stats("cycle", _stats(cycle), steps_per_cycle=steps_per_cycle)
    else:
        print("cycle: n=0")

    for name in events:
        durations = _event_durations_ms(trace_events, name)
        if durations:
            _format_stats(name, _stats(durations))


def analyze_gaps(path: Path, *, gap_events: list[str]) -> None:
    with path.open() as f:
        trace_events = json.load(f)["traceEvents"]
    print(path)
    for name in gap_events:
        gaps = _event_end_to_next_start_ms(trace_events, name)
        if gaps:
            _format_stats(f"{name} end_to_next_start_gap", _stats(gaps))
        else:
            print(f"{name} end_to_next_start_gap: n=0")


def analyze_iteration_resume_gap(path: Path) -> None:
    with path.open() as f:
        trace_events = json.load(f)["traceEvents"]
    gaps = _iteration_resume_gap_ms(trace_events)
    print(path)
    if gaps:
        _format_stats("weight_sync_end_to_next_update0_start_gap", _stats(gaps))
    else:
        print("weight_sync_end_to_next_update0_start_gap: n=0")


def analyze_training_e2e(path: Path) -> None:
    with path.open() as f:
        trace_events = json.load(f)["traceEvents"]
    durations = _event_durations_ms(trace_events, "learner/training_e2e")
    print(path)
    if durations:
        _format_stats("training_e2e", _stats(durations))
    else:
        print("training_e2e: n=0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--steps-per-cycle", type=int, default=2048)
    parser.add_argument("--drop-first", type=int, default=5)
    parser.add_argument(
        "--event",
        action="append",
        default=[],
        help="Also report duration stats for this event name. May be repeated.",
    )
    parser.add_argument(
        "--gap-event",
        action="append",
        default=[],
        help="Report end-to-next-start gap stats for this event name. May be repeated.",
    )
    parser.add_argument(
        "--iteration-resume-gap",
        action="store_true",
        help="Report gap from learner/weight_sync_write end to next update_critic(update_idx=0) start.",
    )
    parser.add_argument(
        "--training-e2e",
        action="store_true",
        help="Report learner/training_e2e duration stats.",
    )
    args = parser.parse_args()

    for i, path in enumerate(args.paths):
        if i:
            print()
        analyze(
            path,
            steps_per_cycle=args.steps_per_cycle,
            drop_first=args.drop_first,
            events=args.event,
        )
        if args.gap_event:
            analyze_gaps(path, gap_events=args.gap_event)
        if args.iteration_resume_gap:
            analyze_iteration_resume_gap(path)
        if args.training_e2e:
            analyze_training_e2e(path)


if __name__ == "__main__":
    main()
