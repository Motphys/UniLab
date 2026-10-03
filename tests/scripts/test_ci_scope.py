from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
SHELVED_BACKENDS = ("motrix", "isaacgym", "isaacsim", "superdex", "drake")
# Some optional packages and identifiers contain these words without routing a
# UniLab backend (for example CUDA nccl/cusparselt). The production gate is the
# explicit ``--extra <backend>`` install pattern.
_EXTRA_PATTERN = re.compile(r"--extra\s+([A-Za-z0-9_-]+)")


def test_ci_installs_only_scoped_backend_extras() -> None:
    offenders: list[str] = []
    for workflow in sorted(WORKFLOWS.glob("*.yml")):
        text = workflow.read_text(encoding="utf-8")
        for extra in _EXTRA_PATTERN.findall(text):
            if extra.lower() in SHELVED_BACKENDS:
                offenders.append(f"{workflow.name}:--extra={extra}")
    assert offenders == [], f"CI installs shelved backend extras: {offenders}"
