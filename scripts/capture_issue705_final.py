"""Capture the Issue #705 Phase 7 A/B/C/D evidence from a clean commit.

The output should be written outside the repository first:

    uv run --extra mjwarp scripts/capture_issue705_final.py \
      --output /tmp/issue705-phase7-final.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from unilab.tools.issue705_final_gate import (
    PLAN_PATH,
    FinalGateError,
    FinalGatePlanError,
    capture_final_gate_evidence,
    load_final_gate_plan,
    write_final_gate_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plan_path = args.plan if args.plan.is_absolute() else REPO_ROOT / args.plan
    output = args.output.resolve()
    if output.is_relative_to(REPO_ROOT):
        print("FAIL: capture output must be outside the repository")
        return 1
    try:
        plan = load_final_gate_plan(plan_path)
        artifact = capture_final_gate_evidence(REPO_ROOT, plan)
        write_final_gate_evidence(artifact, output)
    except (FinalGateError, FinalGatePlanError, OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(
        f"PASS: captured Issue #705 final evidence at {output} "
        f"for {artifact['source']['commit_sha']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
