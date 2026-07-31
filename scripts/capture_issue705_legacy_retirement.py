#!/usr/bin/env python3

"""Capture repeated full-entrypoint evidence for Issue #705 legacy retirement."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from unilab.tools.issue705_legacy_retirement import (
    EVIDENCE_PATH,
    LegacyRetirementError,
    capture_legacy_retirement_evidence,
    write_legacy_retirement_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EVIDENCE_PATH)
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    try:
        payload = capture_legacy_retirement_evidence(REPO_ROOT)
        write_legacy_retirement_evidence(payload, output)
    except (LegacyRetirementError, OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(
        "PASS: captured Issue #705 legacy retirement entrypoint evidence at "
        f"{output} for {payload['source']['commit_sha']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
