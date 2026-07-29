#!/usr/bin/env python3
"""Lightweight smoke test for benchmark entrypoints.

Two phases, each catching a distinct failure mode:

1. Module-mode import (``import benchmark.<category>.<module>``) — verifies the
   package surface used by ``tests/benchmark`` and cross-module imports.
2. Script-mode import — verifies ``uv run benchmark/<category>/<script>.py``
   works, i.e. module-level imports resolve when only the script's own
   directory is on ``sys.path`` (repo root deliberately removed). This is the
   mode users actually run, and the mode that broke after the benchmark
   subpackage refactor: module-mode checks pass while direct script runs fail
   with ``ModuleNotFoundError``.

Optional dependencies (motrixsim, genesis, mujoco_warp, ...) are not required:
entrypoints guard those imports and only fail at run time with a clear error.
"""

from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BENCHMARK_DIR = Path(__file__).resolve().parent
CATEGORIES = ("compute", "env", "ipc", "physics", "rl")
MODULES = sorted(
    f"{category}.{name}"
    for category in CATEGORIES
    for _, name, is_pkg in pkgutil.iter_modules([str(BENCHMARK_DIR / category)])
    if not is_pkg and name.startswith("benchmark_")
)
ENTRYPOINTS = sorted(BENCHMARK_DIR.glob("benchmark_*.py")) + sorted(
    path for category in CATEGORIES for path in (BENCHMARK_DIR / category).glob("benchmark_*.py")
)

# Executed in a clean subprocess per entrypoint. Mimics `python <script>`:
# sys.path[0] is the script's directory and neither the cwd nor the repo root
# is importable, then the file's module-level code runs (without triggering
# the `__main__` guard). argv: <script> <repo_root>
_SCRIPT_MODE_SNIPPET = r"""
import runpy
import sys
from pathlib import Path

script = Path(sys.argv[1]).resolve()
repo_root = Path(sys.argv[2]).resolve()

def _keep(entry: str) -> bool:
    if entry in ("", str(script.parent)):
        return False
    try:
        resolved = Path(entry).resolve()
    except OSError:
        return True
    return resolved != repo_root and resolved != Path.cwd().resolve()

sys.path[:] = [str(script.parent)] + [p for p in sys.path if _keep(p)]
runpy.run_path(str(script), run_name="__benchmark_smoke__")
"""

SCRIPT_MODE_TIMEOUT_SEC = 300
SCRIPT_MODE_WORKERS = 4


def check_module_mode(name: str) -> str | None:
    try:
        importlib.import_module(f"benchmark.{name}")
    except Exception as exc:
        return str(exc)
    return None


def check_script_mode(path: Path) -> str | None:
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT_MODE_SNIPPET, str(path), str(ROOT)],
        capture_output=True,
        text=True,
        timeout=SCRIPT_MODE_TIMEOUT_SEC,
    )
    if proc.returncode == 0:
        return None
    tail = (proc.stderr.strip().splitlines() or ["<no stderr>"])[-1]
    return f"exit={proc.returncode}: {tail}"


def report(phase: str, failures: list[tuple[str, str]], total: int) -> bool:
    passed = total - len(failures)
    print(f"\n[{phase}] Passed: {passed}/{total}")
    for name, err in failures:
        print(f"  ✗ {name}: {err[:200]}")
    return not failures


def main() -> int:
    print(f"Phase 1: module-mode import ({len(MODULES)} modules)...")
    module_failures = []
    for name in MODULES:
        err = check_module_mode(name)
        status = "✗" if err else "✓"
        print(f"  {status} benchmark.{name}")
        if err:
            module_failures.append((f"benchmark.{name}", err))

    print(f"\nPhase 2: script-mode import ({len(ENTRYPOINTS)} entrypoints)...")
    script_failures = []
    with ThreadPoolExecutor(max_workers=SCRIPT_MODE_WORKERS) as pool:
        results = list(zip(ENTRYPOINTS, pool.map(check_script_mode, ENTRYPOINTS)))
    for path, err in results:
        rel = path.relative_to(ROOT)
        status = "✗" if err else "✓"
        print(f"  {status} {rel}")
        if err:
            script_failures.append((str(rel), err))

    ok = report("module-mode", module_failures, len(MODULES))
    ok = report("script-mode", script_failures, len(ENTRYPOINTS)) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
