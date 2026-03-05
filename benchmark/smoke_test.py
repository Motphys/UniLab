#!/usr/bin/env python3
"""Smoke test all benchmark files."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

tests = [
    "benchmark_backends",
    "benchmark_conversions",
    "benchmark_ane_inference",
    "benchmark_mlp_inference",
    "benchmark_forward_reset_methods",
    "benchmark_mj_step",
    "benchmark_mlx_compile",
    "benchmark_postprocess",
    "benchmark_reset_batch_vs_loop",
    "benchmark_sim",
]

print("Testing benchmark files...\n")
passed = []
failed = []

for name in tests:
    print(f"Testing {name}...")
    try:
        mod = __import__(f"benchmark.{name}", fromlist=[name])
        print(f"  ✓ Import OK")
        passed.append(name)
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        failed.append((name, str(e)))

print(f"\n{'='*50}")
print(f"Passed: {len(passed)}/{len(tests)}")
print(f"Failed: {len(failed)}/{len(tests)}")

if failed:
    print("\nFailed tests:")
    for name, err in failed:
        print(f"  - {name}: {err[:80]}")
    sys.exit(1)
