"""Audit MicroDuck alignment against the upstream microduck_rl recipe.

Evaluates ``unilab.tasks.locomotion.microduck.alignment_contract`` for the three
MicroDuck locomotion tasks (ppo tree, mjwarp owner) and prints a human-readable
report grouped by declared status (match / gap / note) with current vs target
values. Read-only.

    uv run scripts/audit_microduck_alignment.py
    uv run scripts/audit_microduck_alignment.py --json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from typing import Any

from unilab.tasks.locomotion.microduck.alignment_contract import (
    MICRODUCK_TASKS,
    UPSTREAM_COMMIT,
    UPSTREAM_DATE,
    UPSTREAM_REPO,
    EntryResult,
    evaluate_all,
)


def _fmt(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return repr(value)


def _print_human(results: list[EntryResult]) -> None:
    print(f"Upstream anchor: {UPSTREAM_REPO} @ {UPSTREAM_COMMIT} ({UPSTREAM_DATE})")
    for task in MICRODUCK_TASKS:
        task_results = [r for r in results if r.task == task]
        print(f"\n### {task}")
        for status in ("gap", "match", "note"):
            group = [r for r in task_results if r.status == status]
            if not group:
                continue
            print(f"  [{status}] ({len(group)})")
            for r in group:
                drift = r.matches is not None and r.matches != (r.status == "match")
                line = f"    {r.name}: current={_fmt(r.current)}  target={_fmt(r.target)}"
                if drift:
                    line += "  [DRIFT: declared status is stale]"
                if r.note:
                    line += f"\n        {r.note}"
                print(line)

    gaps = [r for r in results if r.status == "gap"]
    matches = [r for r in results if r.status == "match"]
    notes = [r for r in results if r.status == "note"]
    drifted = [r for r in results if r.matches is not None and r.matches != (r.status == "match")]
    print("\n" + "=" * 80)
    print(
        f"MATCH: {len(matches)}   GAP: {len(gaps)}   NOTE: {len(notes)}   "
        f"DECLARED-STATUS MISMATCH: {len(drifted)}"
    )
    if drifted:
        print("Entries whose evaluated value no longer matches the declared status:")
        for r in drifted:
            print(f"  - [{r.task}] {r.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    args = parser.parse_args()

    results = evaluate_all()
    if args.json:
        payload = {
            "upstream": {
                "repo": UPSTREAM_REPO,
                "commit": UPSTREAM_COMMIT,
                "date": UPSTREAM_DATE,
            },
            "tasks": list(MICRODUCK_TASKS),
            "results": [dataclasses.asdict(r) for r in results],
            "summary": {
                status: sum(1 for r in results if r.status == status)
                for status in ("match", "gap", "note")
            },
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_human(results)


if __name__ == "__main__":
    main()
