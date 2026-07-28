"""Validate the frozen Issue #705 threshold manifest and receipt.

Usage:
uv run scripts/validate_issue705_thresholds.py
uv run scripts/validate_issue705_thresholds.py --manifest-only --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from unilab.tools.g1_baseline_provenance import sha256_file
from unilab.tools.issue705_thresholds import (
    FREEZE_RECEIPT_PATH,
    THRESHOLD_MANIFEST_PATH,
    ThresholdValidationError,
    load_freeze_receipt,
    load_threshold_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=THRESHOLD_MANIFEST_PATH)
    parser.add_argument("--receipt", type=Path, default=FREEZE_RECEIPT_PATH)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Validate the manifest before its independent freeze receipt is created.",
    )
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest_path = _resolve(args.manifest)
    receipt_path = _resolve(args.receipt)
    try:
        manifest = load_threshold_manifest(manifest_path, repo_root=REPO_ROOT)
        receipt = None
        if not args.manifest_only:
            receipt = load_freeze_receipt(
                receipt_path,
                manifest=manifest,
                repo_root=REPO_ROOT,
            )
    except (OSError, ThresholdValidationError) as exc:
        if args.json_output:
            print(json.dumps({"result": "FAIL", "error": str(exc)}, sort_keys=True))
        else:
            print(f"FAIL {exc}")
        return 1

    payload = {
        "result": "PASS",
        "threshold_set_id": manifest.data["threshold_set_id"],
        "manifest_sha256": sha256_file(manifest_path),
        "baseline_artifact_sha256": manifest.data["baseline"]["artifact_sha256"],
        "freeze_commit": None if receipt is None else receipt.freeze_commit,
        "env_batches": sorted(int(key) for key in manifest.baseline_reference["env"]),
        "dr_densities": sorted(float(key) for key in manifest.baseline_reference["dr"]),
        "ppo_seeds": manifest.baseline_reference["ppo"]["seeds"],
    }
    if args.json_output:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(
            "PASS "
            f"threshold_set={payload['threshold_set_id']} "
            f"manifest={payload['manifest_sha256']} "
            f"freeze={payload['freeze_commit']} "
            f"env={payload['env_batches']} dr={payload['dr_densities']} "
            f"seeds={payload['ppo_seeds']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
