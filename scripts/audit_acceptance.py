#!/usr/bin/env python3
"""Run managed MuJoCo/MJWarp repository acceptance audits."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

_COMMANDS = {
    "backend-isolation": "backend_isolation",
    "claims": "claims",
    "dr-inventory": "dr_inventory",
    "final": "final_gate",
    "g1-baseline": "g1_baseline",
    "legacy-retirement": "legacy_retirement",
    "phase": "phase",
    "support": "support",
    "task-rollout": "task_rollout",
    "thresholds": "thresholds",
    "workflow-triggers": "workflow_triggers",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(_COMMANDS))
    args, remainder = parser.parse_known_args(argv)
    module = importlib.import_module(f"tooling.acceptance.commands.{_COMMANDS[args.command]}")
    return int(module.main(remainder))


if __name__ == "__main__":
    raise SystemExit(main())
