"""Pytest collection control for tests/nan_injection/.

These scripts are MANUAL validation tools, not pytest tests. They build real
runners, spin up MuJoCo, train for a single iteration, and assert NaN guard
behavior end-to-end. They are intentionally kept OUT of CI/CD because each
case takes ~10-30s and requires a working GPU/CPU MuJoCo install.

This conftest tells pytest to skip the entire directory during automatic
collection. Run the scripts manually instead, e.g.:

    .venv/bin/python tests/nan_injection/stage2_nan_inject.py
    .venv/bin/python tests/nan_injection/stage3_nan_inject.py
"""

collect_ignore_glob = ["*.py"]
