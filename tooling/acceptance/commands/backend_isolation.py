"""Audit the managed MuJoCo/MJWarp rollout runtime backend isolation boundary.

uv run scripts/audit_acceptance.py backend-isolation
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from unilab.tools.backend_isolation import (
    audit_backend_isolation,
    format_backend_isolation_report,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    report = audit_backend_isolation(REPO_ROOT)
    if args.json:
        payload = asdict(report)
        payload["ok"] = report.ok
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for line in format_backend_isolation_report(report):
            print(line)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
