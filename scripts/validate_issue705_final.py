"""Validate the Issue #705 final integration evidence.

Examples:
    uv run scripts/validate_issue705_final.py --head-only --allow-unpromoted
    uv run scripts/validate_issue705_final.py --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from unilab.tools.issue705_final_gate import (
    ARTIFACT_PATH,
    PLAN_PATH,
    FinalGateError,
    FinalGatePlanError,
    load_final_gate_evidence,
    load_final_gate_plan,
    validate_final_gate_evidence,
    validate_final_head,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _payload(
    *,
    plan_path: Path,
    artifact_path: Path,
    head_only: bool,
    require_promotion: bool,
) -> dict[str, object]:
    try:
        plan = load_final_gate_plan(_resolve(plan_path))
        if head_only:
            report = validate_final_head(
                REPO_ROOT,
                plan,
                require_promotion=require_promotion,
            )
            return {
                "ok": report.ok,
                "mode": "head-only",
                "promotion_required": require_promotion,
                "components": [component.to_dict() for component in report.components],
                "errors": list(report.errors),
            }
        artifact = load_final_gate_evidence(_resolve(artifact_path))
        errors = validate_final_gate_evidence(
            artifact,
            root=REPO_ROOT,
            plan=plan,
            require_promotion=require_promotion,
        )
    except (FinalGateError, FinalGatePlanError, OSError, RuntimeError, ValueError) as exc:
        return {
            "ok": False,
            "mode": "artifact",
            "promotion_required": require_promotion,
            "errors": [str(exc)],
        }
    return {
        "ok": not errors,
        "mode": "artifact",
        "promotion_required": require_promotion,
        "artifact": str(artifact_path),
        "source_commit": artifact.get("source", {}).get("commit_sha"),
        "required_lanes": artifact.get("summary", {}).get("required_lanes"),
        "command_runs": artifact.get("summary", {}).get("command_runs"),
        "errors": list(errors),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--head-only", action="store_true")
    parser.add_argument(
        "--allow-unpromoted",
        action="store_true",
        help="Validate implementation-source evidence before Phase 7 manifest promotion.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = _payload(
        plan_path=args.plan,
        artifact_path=args.artifact,
        head_only=args.head_only,
        require_promotion=not args.allow_unpromoted,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif payload["ok"]:
        print(
            "PASS Issue #705 final gate "
            f"mode={payload['mode']} promotion_required={payload['promotion_required']}"
        )
    else:
        print(
            "FAIL Issue #705 final gate "
            f"mode={payload['mode']} promotion_required={payload['promotion_required']}"
        )
        errors = payload.get("errors")
        if isinstance(errors, list):
            for error in errors:
                print(f"- {error}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
