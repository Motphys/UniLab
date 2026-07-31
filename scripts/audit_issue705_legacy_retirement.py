#!/usr/bin/env python3

"""Audit Issue #705 mjwarp legacy-route retirement and rollback evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from unilab.tools.issue705_legacy_retirement import (
    EVIDENCE_PATH,
    PLAN_PATH,
    ROLLBACK_PATH,
    LegacyRetirementError,
    audit_legacy_retirement,
    load_legacy_retirement_evidence,
    load_legacy_retirement_plan,
    load_rollback_receipt,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--rollback", type=Path, default=ROLLBACK_PATH)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    try:
        plan = load_legacy_retirement_plan(resolve(args.plan))
        rollback = load_rollback_receipt(resolve(args.rollback))
        evidence = load_legacy_retirement_evidence(resolve(args.evidence))
        report = audit_legacy_retirement(plan, rollback, evidence, root=REPO_ROOT)
    except (LegacyRetirementError, OSError, RuntimeError, ValueError) as exc:
        report_payload = {"ok": False, "errors": [str(exc)]}
        if args.json:
            print(json.dumps(report_payload, indent=2, sort_keys=True))
        else:
            print(f"FAIL Issue #705 legacy retirement audit\n- {exc}")
        return 1

    payload = {
        "ok": report.ok,
        "changed_paths": report.changed_paths,
        "entrypoint_repetitions": report.entrypoint_repetitions,
        "retained_routes": report.retained_routes,
        "errors": list(report.errors),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif report.ok:
        print(
            "PASS Issue #705 legacy retirement audit "
            f"changed_paths={report.changed_paths} "
            f"entrypoint_repetitions={report.entrypoint_repetitions} "
            f"retained_routes={report.retained_routes}"
        )
    else:
        print("FAIL Issue #705 legacy retirement audit")
        for error in report.errors:
            print(f"- {error}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
