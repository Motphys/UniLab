"""Validate the frozen Issue #705 mjwarp field-level DR inventory.

uv run scripts/validate_issue705_dr_inventory.py
uv run scripts/validate_issue705_dr_inventory.py --json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from unilab.tools.claim_gap_audit import ClaimGapInventoryError, load_claim_gap_inventory
from unilab.tools.mjwarp_dr_inventory import (
    DrInventoryValidationError,
    inventory_claim_gap_errors,
    load_mjwarp_dr_inventory,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_DIR = REPO_ROOT / "tests" / "acceptance" / "issue_705"
INVENTORY_PATH = ACCEPTANCE_DIR / "mjwarp_dr_inventory.yaml"
CLAIM_INVENTORY_PATH = ACCEPTANCE_DIR / "claim_test_inventory.yaml"


def _payload(inventory_path: Path, claim_inventory_path: Path) -> dict[str, Any]:
    try:
        inventory = load_mjwarp_dr_inventory(inventory_path)
    except DrInventoryValidationError as exc:
        return {"ok": False, "inventory": str(inventory_path), "errors": list(exc.errors)}
    try:
        claim_inventory = load_claim_gap_inventory(claim_inventory_path)
    except ClaimGapInventoryError as exc:
        return {
            "ok": False,
            "inventory": str(inventory_path),
            "errors": [f"claim inventory: {error}" for error in exc.errors],
        }

    errors = list(inventory_claim_gap_errors(inventory, claim_inventory))
    states = Counter(capability.support_state.value for capability in inventory.capabilities)
    return {
        "ok": not errors,
        "inventory": str(inventory_path),
        "backend": inventory.backend,
        "source_commit": inventory.source.commit,
        "capabilities": len(inventory.capabilities),
        "support_states": dict(sorted(states.items())),
        "exclusions": len(inventory.exclusions),
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=INVENTORY_PATH)
    parser.add_argument("--claim-inventory", type=Path, default=CLAIM_INVENTORY_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = _payload(args.inventory, args.claim_inventory)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif payload["ok"]:
        print(
            f"PASS backend={payload['backend']} capabilities={payload['capabilities']} "
            f"states={payload['support_states']} exclusions={payload['exclusions']}"
        )
    else:
        print(f"FAIL inventory={payload['inventory']}")
        for error in payload["errors"]:
            print(f"  - {error}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
