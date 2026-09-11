#!/usr/bin/env python3
"""Characterize executor-swap drift between two mujoco drift-baseline captures.

Roadmap issue #1552 replaces the mujoco backend's native executor
(mujoco-uni-runtime) with mjbatch. Numerical drift between the two is expected
and accepted; this tool quantifies it as a regression reference (issue #1554),
not as a pass/fail gate — it always exits 0 when the comparison completes.

For every task present in both ``--before`` and ``--after`` it loads the .npz
artifacts written by ``capture_mujoco_drift_baseline.py`` and reports, per
array and per observation group:

- max / mean absolute difference,
- the magnitude of the recorded values (so relative scale is visible),
- the first diverging step (leading-dim index of the first step whose slice
  contains any difference; -1 when identical).

Per-step arrays have a leading ``steps`` dimension; ``*_init`` snapshots and
``actions`` are compared as single snapshots (the action generator is seeded
and must be bit-identical — a mismatch there invalidates the comparison).

Usage:
    uv run scripts/tools/compare_mujoco_drift_baseline.py \
        --before scripts/tools/drift_baseline/before \
        --after scripts/tools/drift_baseline/after \
        --report scripts/tools/drift_baseline/drift_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_BEFORE_DIR = ROOT_DIR / "scripts" / "tools" / "drift_baseline" / "before"
DEFAULT_AFTER_DIR = ROOT_DIR / "scripts" / "tools" / "drift_baseline" / "after"
DEFAULT_REPORT = ROOT_DIR / "scripts" / "tools" / "drift_baseline" / "drift_report.md"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before",
        type=Path,
        default=DEFAULT_BEFORE_DIR,
        help="BEFORE artifact directory (default: %(default)s).",
    )
    parser.add_argument(
        "--after",
        type=Path,
        default=DEFAULT_AFTER_DIR,
        help="AFTER artifact directory (default: %(default)s).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Markdown report output path (default: %(default)s).",
    )
    return parser.parse_args(argv)


def _load(dir_path: Path) -> dict[str, dict[str, np.ndarray]]:
    tasks: dict[str, dict[str, np.ndarray]] = {}
    for npz_path in sorted(dir_path.glob("*.npz")):
        with np.load(npz_path) as data:
            tasks[npz_path.stem] = {name: data[name].copy() for name in data.files}
    return tasks


def _compare_array(before: np.ndarray, after: np.ndarray) -> dict[str, object]:
    if before.shape != after.shape:
        return {"shape_before": before.shape, "shape_after": after.shape, "mismatch": True}
    diff = np.abs(after.astype(np.float64) - before.astype(np.float64))
    result: dict[str, object] = {
        "shape": before.shape,
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "ref_max_abs": float(np.abs(before).max()) if before.size else 0.0,
        "first_divergence": -1,
    }
    if diff.ndim >= 1 and diff.shape[0] > 1 and diff.size:
        per_step = diff.reshape(diff.shape[0], -1).max(axis=1)
        nonzero = np.flatnonzero(per_step > 0.0)
        if nonzero.size:
            result["first_divergence"] = int(nonzero[0])
            result["first_divergence_max_abs_diff"] = float(per_step[nonzero[0]])
    elif diff.size and diff.max() > 0.0:
        result["first_divergence"] = 0
        result["first_divergence_max_abs_diff"] = float(diff.max())
    return result


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6e}"
    return str(value)


def compare(before_dir: Path, after_dir: Path) -> dict[str, dict[str, dict[str, object]]]:
    before_tasks = _load(before_dir)
    after_tasks = _load(after_dir)
    common = sorted(set(before_tasks) & set(after_tasks))
    if not common:
        raise SystemExit(f"no task artifacts common to {before_dir} and {after_dir}")

    report: dict[str, dict[str, dict[str, object]]] = {}
    for task in common:
        before_arrays = before_tasks[task]
        after_arrays = after_tasks[task]
        arrays: dict[str, dict[str, object]] = {}
        for name in sorted(set(before_arrays) | set(after_arrays)):
            if name not in before_arrays or name not in after_arrays:
                arrays[name] = {"mismatch": True, "note": "present in only one artifact"}
                continue
            arrays[name] = _compare_array(before_arrays[name], after_arrays[name])
        report[task] = arrays
    return report


def render_markdown(
    report: dict[str, dict[str, dict[str, object]]],
    before_dir: Path,
    after_dir: Path,
) -> str:
    lines = [
        "# MuJoCo executor-swap drift characterization (#1554)",
        "",
        "BEFORE: mujoco-uni-runtime 0.5.0 executor, captured from"
        " `scripts/tools/drift_baseline/before/`.",
        "AFTER: mjbatch executor (unilabsim fork), captured from"
        " `scripts/tools/drift_baseline/after/`.",
        "",
        "Both captures use the identical configuration: 8 envs, 300 steps,"
        " env seed 42, action seed 1234, `cpu_ids=[0, 1, 2, 3]`, fixed"
        " pseudo-random action sequence (no policy network). Drift between the"
        " two executors is expected and accepted; this report is a regression"
        " reference, not a pass/fail gate.",
        "",
        f"- BEFORE dir: `{before_dir}`",
        f"- AFTER dir: `{after_dir}`",
        "",
    ]
    for task, arrays in report.items():
        lines.append(f"## {task}")
        lines.append("")
        lines.append(
            "| array | shape | max abs diff | mean abs diff | ref max abs "
            "| first divergence step | max abs diff at first divergence |"
        )
        lines.append("|---|---|---|---|---|---|")
        step_arrays = {
            name: res
            for name, res in arrays.items()
            if not res.get("mismatch") and int(res.get("first_divergence", -1)) >= 0
        }
        first_overall = (
            min(int(res["first_divergence"]) for res in step_arrays.values()) if step_arrays else -1
        )
        for name, res in arrays.items():
            if res.get("mismatch"):
                lines.append(f"| `{name}` | — | — | — | — | present in only one artifact |")
                continue
            lines.append(
                f"| `{name}` | {res['shape']} | {_fmt(res['max_abs_diff'])} "
                f"| {_fmt(res['mean_abs_diff'])} | {_fmt(res['ref_max_abs'])} "
                f"| {res['first_divergence']} "
                f"| {_fmt(res.get('first_divergence_max_abs_diff', 0.0))} |"
            )
        lines.append("")
        lines.append(f"First divergence step (any array): **{first_overall}**")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = compare(args.before, args.after)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_markdown(report, args.before, args.after) + "\n")

    for task, arrays in report.items():
        print(f"[compare] {task}")
        for name, res in arrays.items():
            if res.get("mismatch"):
                print(f"  {name}: present in only one artifact")
                continue
            print(
                f"  {name}: max={_fmt(res['max_abs_diff'])} "
                f"mean={_fmt(res['mean_abs_diff'])} first_divergence={res['first_divergence']}"
            )
    print(f"[compare] wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
