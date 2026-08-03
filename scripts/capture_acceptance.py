#!/usr/bin/env python3
"""Capture one managed MuJoCo/MJWarp acceptance phase from a clean commit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _phase_handlers() -> Mapping[
    str,
    tuple[Callable[[Path], dict[str, Any]], Callable[[Mapping[str, Any], Path], None]],
]:
    from tooling.acceptance.phase1 import capture_phase1_evidence, write_phase1_evidence
    from tooling.acceptance.phase2 import capture_phase2_evidence, write_phase2_evidence
    from tooling.acceptance.phase3 import capture_phase3_evidence, write_phase3_evidence
    from tooling.acceptance.phase4 import capture_phase4_evidence, write_phase4_evidence
    from tooling.acceptance.phase5 import capture_phase5_evidence, write_phase5_evidence
    from tooling.acceptance.phase6 import capture_phase6_evidence, write_phase6_evidence

    return {
        "1": (capture_phase1_evidence, write_phase1_evidence),
        "2": (capture_phase2_evidence, write_phase2_evidence),
        "3": (capture_phase3_evidence, write_phase3_evidence),
        "4": (capture_phase4_evidence, write_phase4_evidence),
        "5": (capture_phase5_evidence, write_phase5_evidence),
        "6": (capture_phase6_evidence, write_phase6_evidence),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", required=True, choices=["1", "2", "3", "4", "5", "6", "final", "legacy"]
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.phase == "final":
            from tooling.acceptance.final_gate import (
                PLAN_PATH,
                capture_final_gate_evidence,
                load_final_gate_plan,
                write_final_gate_evidence,
            )

            if args.output is None:
                parser.error("--output is required for --phase final")
            output = args.output.resolve()
            if output.is_relative_to(REPO_ROOT):
                parser.error("final capture output must be outside the repository")
            plan_path = args.plan or PLAN_PATH
            if not plan_path.is_absolute():
                plan_path = REPO_ROOT / plan_path
            payload = capture_final_gate_evidence(REPO_ROOT, load_final_gate_plan(plan_path))
            write_final_gate_evidence(payload, output)
        elif args.phase == "legacy":
            from tooling.acceptance.legacy_retirement import (
                EVIDENCE_PATH,
                capture_legacy_retirement_evidence,
                write_legacy_retirement_evidence,
            )

            output = args.output or EVIDENCE_PATH
            if not output.is_absolute():
                output = REPO_ROOT / output
            payload = capture_legacy_retirement_evidence(REPO_ROOT)
            write_legacy_retirement_evidence(payload, output)
        else:
            if args.plan is not None:
                parser.error("--plan is supported only for --phase final")
            if args.output is None:
                parser.error(f"--output is required for --phase {args.phase}")
            capture, write = _phase_handlers()[args.phase]
            output = args.output
            payload = capture(REPO_ROOT)
            write(payload, output)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1

    commit = payload.get("source", {}).get("commit_sha", "unknown")
    print(f"PASS: captured phase {args.phase} evidence at {output} for {commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
