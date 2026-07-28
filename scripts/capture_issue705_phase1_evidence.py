"""Capture real A/B Phase 1 evidence for Issue #705.

Run only from a clean commit and write outside the repository first:

    uv run scripts/capture_issue705_phase1_evidence.py \
      --output /tmp/issue705-phase1-gate.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from unilab.tools.issue705_phase1_evidence import (
    PhaseEvidenceError,
    capture_phase1_evidence,
    write_phase1_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = capture_phase1_evidence(REPO_ROOT)
        write_phase1_evidence(report, args.output)
    except PhaseEvidenceError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(
        f"PASS: captured Issue #705 Phase 1 A/B evidence at {args.output} "
        f"for {report['source']['commit_sha']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
