"""Validate an managed MuJoCo/MJWarp rollout phase acceptance manifest without executing its commands.

uv run scripts/audit_acceptance.py phase --phase 0 --mode schema
uv run scripts/audit_acceptance.py phase --phase 0 --mode gate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from unilab.tools.phase_acceptance import (
    ManifestValidationError,
    load_phase_acceptance,
    manifest_status_counts,
    phase_gate_errors,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_DIR = REPO_ROOT / "tests" / "acceptance" / "manager_mjwarp" / "manifests"
INTEGRATION_BRANCH = "feat/manager-mjwarp-manager-mjwarp"


def _build_payload(path: Path, mode: str) -> dict[str, object]:
    try:
        manifest = load_phase_acceptance(path)
    except ManifestValidationError as exc:
        return {
            "ok": False,
            "mode": mode,
            "manifest": str(path),
            "errors": list(exc.errors),
        }

    errors: list[str] = []
    if manifest.issue != 705:
        errors.append(f"issue: expected 705, got {manifest.issue}")
    if manifest.integration_branch != INTEGRATION_BRANCH:
        errors.append(
            f"integration_branch: expected {INTEGRATION_BRANCH!r}, "
            f"got {manifest.integration_branch!r}"
        )
    if mode == "gate":
        errors.extend(phase_gate_errors(manifest))

    return {
        "ok": not errors,
        "mode": mode,
        "manifest": str(path),
        "issue": manifest.issue,
        "phase": manifest.phase,
        "required_lanes": [lane.value for lane in manifest.required_lanes],
        "status_counts": manifest_status_counts(manifest),
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--mode", choices=("schema", "gate"), default="schema")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    path = args.manifest or MANIFEST_DIR / f"phase_{args.phase}.yaml"
    payload = _build_payload(path, args.mode)
    if payload.get("phase") is not None and payload["phase"] != args.phase:
        payload["ok"] = False
        payload["errors"] = [
            *payload["errors"],
            f"phase: requested {args.phase}, manifest declares {payload['phase']}",
        ]

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif payload["ok"]:
        print(
            f"PASS issue={payload['issue']} phase={payload['phase']} mode={payload['mode']} "
            f"status_counts={payload['status_counts']}"
        )
    else:
        print(f"FAIL manifest={payload['manifest']} mode={payload['mode']}")
        for error in payload["errors"]:
            print(f"  - {error}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
