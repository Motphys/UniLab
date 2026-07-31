#!/usr/bin/env python3

"""Audit Issue #705 support claims against independent repository evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from unilab.tools.issue705_support import (
    SUPPORT_EVIDENCE_PATH,
    SupportEvidenceError,
    audit_support_evidence,
    load_support_evidence,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Support evidence YAML (defaults to the Issue #705 repository manifest).",
    )
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable report.")
    args = parser.parse_args(argv)
    root = _repo_root()
    path = args.manifest or root / SUPPORT_EVIDENCE_PATH
    if not path.is_absolute():
        path = root / path
    try:
        support = load_support_evidence(path)
        report = audit_support_evidence(support, root=root)
    except (OSError, RuntimeError, SupportEvidenceError, ValueError) as exc:
        if args.json:
            print(json.dumps({"ok": False, "errors": [str(exc)]}, indent=2, sort_keys=True))
        else:
            print(f"FAIL Issue #705 support audit\n- {exc}")
        return 1

    payload = {
        "ok": report.ok,
        "combinations": report.combinations,
        "benchmarked": report.benchmarked,
        "recommended": report.recommended,
        "phase_gates": report.phase_gates,
        "errors": list(report.errors),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif report.ok:
        print(
            "PASS Issue #705 support audit "
            f"combinations={report.combinations} benchmarked={report.benchmarked} "
            f"recommended={report.recommended} phase_gates={report.phase_gates}"
        )
    else:
        print("FAIL Issue #705 support audit")
        for error in report.errors:
            print(f"- {error}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
