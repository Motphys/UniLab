"""Audit managed MuJoCo/MJWarp rollout claim-to-test ownership and explicit implementation gaps.

uv run scripts/audit_acceptance.py claims --all
uv run scripts/audit_acceptance.py claims --phase 0 --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from unilab.tools.claim_gap_audit import (
    PHASES,
    ClaimGapInventoryError,
    audit_claim_gaps,
    load_claim_gap_inventory,
    load_phase_manifests,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
ACCEPTANCE_DIR = REPO_ROOT / "tests" / "acceptance" / "manager_mjwarp"
INVENTORY_PATH = ACCEPTANCE_DIR / "claim_test_inventory.yaml"
MANIFEST_DIR = ACCEPTANCE_DIR / "manifests"


def _payload(phases: tuple[int, ...]) -> dict[str, Any]:
    manifests, manifest_errors = load_phase_manifests(MANIFEST_DIR, phases)
    try:
        inventory = load_claim_gap_inventory(INVENTORY_PATH)
    except ClaimGapInventoryError as exc:
        return {
            "ok": False,
            "phases": list(phases),
            "inventory": str(INVENTORY_PATH),
            "errors": list(exc.errors),
        }

    report = audit_claim_gaps(inventory, manifests, repo_root=REPO_ROOT, phases=phases)
    errors = [*manifest_errors, *report.errors]
    return {
        "ok": not errors,
        "phases": list(phases),
        "inventory": str(INVENTORY_PATH),
        "claims": report.claims,
        "entries": report.entries,
        "existing": report.existing,
        "targets": report.targets,
        "supporting": report.supporting,
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--phase", type=int, choices=PHASES)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.all:
        phases = PHASES
    else:
        assert args.phase is not None
        phases = (args.phase,)
    payload = _payload(phases)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif payload["ok"]:
        print(
            f"PASS phases={payload['phases']} claims={payload['claims']} "
            f"entries={payload['entries']} existing={payload['existing']} "
            f"targets={payload['targets']} supporting={payload['supporting']}"
        )
    else:
        print(f"FAIL inventory={payload['inventory']} phases={payload['phases']}")
        for error in payload["errors"]:
            print(f"  - {error}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
