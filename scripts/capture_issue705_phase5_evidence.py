"""Capture real C/D Phase 5 evidence for Issue #705.

Run only after a clean commit has frozen the complete PPO artifact, profiler
trace, validator, and required tests:

    uv run scripts/capture_issue705_phase5_evidence.py \
      --output /tmp/issue705-phase5-gate.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from unilab.tools.issue705_phase5_evidence import (
    PhaseEvidenceError,
    capture_phase5_evidence,
    write_phase5_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = capture_phase5_evidence(REPO_ROOT)
        write_phase5_evidence(report, args.output)
    except PhaseEvidenceError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(
        f"PASS: captured Issue #705 Phase 5 C/D evidence at {args.output} "
        f"for {report['source']['commit_sha']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
