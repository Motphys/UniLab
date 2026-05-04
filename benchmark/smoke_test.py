#!/usr/bin/env python3
"""Lightweight smoke test for benchmark entrypoints.

This checks that benchmark modules import, without requiring every optional
dependency or full benchmark run.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BENCHMARK_DIR = Path(__file__).resolve().parent
MODULES = sorted(
    name
    for _, name, is_pkg in pkgutil.iter_modules([str(BENCHMARK_DIR)])
    if not is_pkg and name.startswith("benchmark_")
)

passed: list[str] = []
failed: list[tuple[str, str]] = []

print("Testing benchmark modules...\n")
for name in MODULES:
    print(f"Testing {name}...")
    try:
        importlib.import_module(f"benchmark.{name}")
    except Exception as exc:
        print(f"  ✗ Import failed: {exc}")
        failed.append((name, str(exc)))
    else:
        print("  ✓ Import OK")
        passed.append(name)

print(f"\n{'=' * 50}")
print(f"Passed: {len(passed)}/{len(MODULES)}")
print(f"Failed: {len(failed)}/{len(MODULES)}")

if failed:
    print("\nFailed tests:")
    for name, err in failed:
        print(f"  - {name}: {err[:120]}")
    sys.exit(1)
