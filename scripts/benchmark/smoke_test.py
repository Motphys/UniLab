#!/usr/bin/env python3
"""Lightweight smoke test for benchmark entrypoints.

Two phases, each catching a distinct failure mode:

1. Module-mode import (``import scripts.benchmark.<category>.<module>``) — verifies the
   package surface used by ``tests/benchmark`` and cross-module imports.
2. Script-mode import — verifies ``uv run scripts/benchmark/<category>/<script>.py``
   works, i.e. module-level imports resolve when only the script's own
   directory is on ``sys.path`` (repo root deliberately removed). This is the
   mode users actually run, and the mode that broke after the benchmark
   subpackage refactor: module-mode checks pass while direct script runs fail
   with ``ModuleNotFoundError``.

Optional dependencies (motrixsim, genesis, mujoco_warp, ...) are not required:
entrypoints guard those imports and only fail at run time with a clear error.
Platform-exclusive packages (``mlx``, ``coremltools`` — Apple Silicon only)
cannot be guarded this way when an entrypoint is single-platform by design;
entries that fail to import solely because of them are reported as SKIP,
not FAIL. Genuine import regressions (``scripts.benchmark``, ``core``, ``unilab``,
or any cross-platform package) still fail the check.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
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

# Top-level packages that exist only on specific platforms (mlx/coremltools:
# Apple Silicon). An entrypoint requiring one at import time is single-platform
# by design, so a miss is reported as SKIP instead of FAIL. Keep this list
# minimal: every other ModuleNotFoundError (benchmark, core, unilab, numpy,
# torch, ...) is a real regression and must fail.
PLATFORM_OPTIONAL_MODULES = frozenset({"mlx", "coremltools"})
_MISSING_MODULE_RE = re.compile(r"No module named '([\w.]+)'")


def platform_optional_miss(err: str | None) -> str | None:
    """Return the missing platform-optional package, or None if `err` is a
    genuine failure (or no error at all)."""
    if err is None:
        return None
    match = _MISSING_MODULE_RE.search(err)
    if match is None:
        return None
    package = match.group(1).split(".")[0]
    return package if package in PLATFORM_OPTIONAL_MODULES else None


def check_module_mode(name: str) -> str | None:
    try:
        importlib.import_module(f"scripts.benchmark.{name}")
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


def report(
    phase: str,
    failures: list[tuple[str, str]],
    skips: list[tuple[str, str]],
    total: int,
) -> bool:
    passed = total - len(failures) - len(skips)
    summary = f"\n[{phase}] Passed: {passed}/{total}"
    if skips:
        summary += f" ({len(skips)} skipped: platform-optional)"
    print(summary)
    for name, package in skips:
        print(f"  - {name}: skipped (needs platform-optional '{package}')")
    for name, err in failures:
        print(f"  ✗ {name}: {err[:200]}")
    return not failures


def main() -> int:
    print(f"Phase 1: module-mode import ({len(MODULES)} modules)...")
    module_failures = []
    module_skips = []
    for name in MODULES:
        err = check_module_mode(name)
        package = platform_optional_miss(err)
        if err is None:
            print(f"  ✓ scripts.benchmark.{name}")
        elif package is not None:
            print(f"  - scripts.benchmark.{name} (skipped: needs '{package}')")
            module_skips.append((f"scripts.benchmark.{name}", package))
        else:
            print(f"  ✗ scripts.benchmark.{name}")
            module_failures.append((f"scripts.benchmark.{name}", err))

    print(f"\nPhase 2: script-mode import ({len(ENTRYPOINTS)} entrypoints)...")
    script_failures = []
    script_skips = []
    with ThreadPoolExecutor(max_workers=SCRIPT_MODE_WORKERS) as pool:
        results = list(zip(ENTRYPOINTS, pool.map(check_script_mode, ENTRYPOINTS)))
    for path, err in results:
        rel = str(path.relative_to(ROOT))
        package = platform_optional_miss(err)
        if err is None:
            print(f"  ✓ {rel}")
        elif package is not None:
            print(f"  - {rel} (skipped: needs '{package}')")
            script_skips.append((rel, package))
        else:
            print(f"  ✗ {rel}")
            script_failures.append((rel, err))

    ok = report("module-mode", module_failures, module_skips, len(MODULES))
    ok = report("script-mode", script_failures, script_skips, len(ENTRYPOINTS)) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
