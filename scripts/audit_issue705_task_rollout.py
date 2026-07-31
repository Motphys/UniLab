#!/usr/bin/env python3

"""Audit the Issue #705 capability-derived mjwarp task rollout plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from unilab.tools.issue705_task_rollout import (
    ROLLOUT_PLAN_PATH,
    TaskRolloutPlanError,
    audit_task_rollout_plan,
    load_task_rollout_plan,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="Rollout plan YAML (defaults to the Issue #705 repository plan).",
    )
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable report.")
    args = parser.parse_args(argv)
    root = _repo_root()
    path = args.plan or root / ROLLOUT_PLAN_PATH
    if not path.is_absolute():
        path = root / path
    try:
        plan = load_task_rollout_plan(path)
        report = audit_task_rollout_plan(plan, root=root)
    except (OSError, RuntimeError, TaskRolloutPlanError, ValueError) as exc:
        if args.json:
            print(json.dumps({"ok": False, "errors": [str(exc)]}, indent=2, sort_keys=True))
        else:
            print(f"FAIL Issue #705 task rollout audit\n- {exc}")
        return 1

    payload = {
        "ok": report.ok,
        "entries": report.entries,
        "prerequisites": report.prerequisites,
        "errors": list(report.errors),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif report.ok:
        print(
            "PASS Issue #705 task rollout audit "
            f"entries={report.entries} prerequisites={report.prerequisites}"
        )
    else:
        print("FAIL Issue #705 task rollout audit")
        for error in report.errors:
            print(f"- {error}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
