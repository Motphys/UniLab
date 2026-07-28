"""Validate the frozen Issue #705 G1 MuJoCo baseline artifact.

uv run scripts/validate_issue705_g1_baseline.py
uv run scripts/validate_issue705_g1_baseline.py --json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from unilab.tools.g1_baseline_provenance import (
    BaselineValidationError,
    load_g1_baseline_artifact,
    load_g1_baseline_plan,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = Path("tests/acceptance/issue_705/g1_mujoco_baseline_plan.yaml")
ARTIFACT_PATH = Path("tests/acceptance/issue_705/artifacts/g1_mujoco_phase0_baseline.json")


def _payload(plan_path: Path, artifact_path: Path) -> dict[str, Any]:
    absolute_plan = plan_path if plan_path.is_absolute() else REPO_ROOT / plan_path
    absolute_artifact = artifact_path if artifact_path.is_absolute() else REPO_ROOT / artifact_path
    try:
        plan = load_g1_baseline_plan(absolute_plan)
        plan = replace(plan, source_path=absolute_plan.relative_to(REPO_ROOT))
        artifact = load_g1_baseline_artifact(
            absolute_artifact,
            plan,
            repo_root=REPO_ROOT,
        )
    except (BaselineValidationError, OSError, ValueError) as exc:
        errors = list(exc.errors) if isinstance(exc, BaselineValidationError) else [str(exc)]
        return {
            "ok": False,
            "plan": str(plan_path),
            "artifact": str(artifact_path),
            "errors": errors,
        }
    lane_counts: dict[str, int] = {}
    for case in artifact["cases"]:
        lane = str(case["lane"])
        lane_counts[lane] = lane_counts.get(lane, 0) + 1
    return {
        "ok": True,
        "plan": str(plan_path),
        "artifact": str(artifact_path),
        "source_commit": artifact["source"]["commit"],
        "plan_sha256": artifact["plan"]["sha256"],
        "lane_counts": dict(sorted(lane_counts.items())),
        "errors": [],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = _payload(args.plan, args.artifact)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif payload["ok"]:
        print(
            f"PASS source={payload['source_commit']} lanes={payload['lane_counts']} "
            f"plan={payload['plan_sha256']}"
        )
    else:
        print(f"FAIL artifact={payload['artifact']}")
        for error in payload["errors"]:
            print(f"  - {error}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
