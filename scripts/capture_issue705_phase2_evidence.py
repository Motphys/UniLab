"""Capture real A/C Phase 2 evidence for Issue #705.

Run only from a clean commit and write outside the repository first:

    uv run --extra mjwarp scripts/capture_issue705_phase2_evidence.py \
      --output /tmp/issue705-phase2-gate.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from unilab.tools.issue705_phase2_evidence import (
    Phase2EvidenceError,
    capture_phase2_evidence,
    write_phase2_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = capture_phase2_evidence(REPO_ROOT)
        write_phase2_evidence(report, args.output)
    except Phase2EvidenceError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(
        f"PASS: captured Issue #705 Phase 2 A/C evidence at {args.output} "
        f"for {report['source']['commit_sha']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
