"""CPU host-fixture ablation of M2 reset staging; no physics or IPC timing.

Run from a source checkout with its development dependencies installed. The
baseline is loaded from local Git history, not from a temporary source file.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATH = "src/unilab/base/reset_state.py"
DEFAULT_BASELINE = "044a11ff"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def run(baseline_revision: str, *, warmup: int, samples: int) -> dict:
    """Compare old, row-map-only, and current sparse staging with equal outputs."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import numpy as np
    from tests.base.test_entity_reset_staging import fixture

    from unilab.base.reset_state import ResetStateTransaction

    baseline_sha = _git("rev-parse", "--verify", baseline_revision + "^{commit}")
    source = _git("show", baseline_sha + ":" + SOURCE_PATH)
    expression = "order = [request.env_ids.index(i) for i in rows]"
    if source.count(expression) != 1:
        raise ValueError("baseline must contain exactly one known quadratic row-order expression")
    index_source = source.replace(
        expression,
        "incoming_rows = {value: index for index, value in enumerate(request.env_ids)}\n"
        "        order = [incoming_rows[i] for i in rows]",
        1,
    )
    paths = []
    for name, code in (("A_git_baseline", source), ("B_row_index_only", index_source)):
        module = types.ModuleType("_m2_ablation_" + name)
        exec(compile(code, f"git:{baseline_sha}:{SOURCE_PATH}:{name}", "exec"), module.__dict__)
        paths.append((name, module.ResetStateTransaction))
    paths.append(("C_row_index_and_sparse_fields", ResetStateTransaction))
    results = []
    for count, selected in ((64, 1), (4096, 1), (4096, 1024), (4096, 4096)):
        ids = np.arange(selected - 1, -1, -1)
        pose = np.tile([0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0], (selected, 1))
        for kind in ("pose", "defaults"):
            reference = None
            for name, transaction_type in paths:
                seed, reads, commits = fixture(count)
                # The test fixture owns this simple stand-in; it is not a native backend.
                transaction = transaction_type(seed._backend, scene_layout=seed.scene_layout)
                times = []
                for iteration in range(warmup + samples):
                    reads.clear()
                    commits.clear()
                    start = time.perf_counter_ns()
                    with transaction.scoped(ids):
                        if kind == "pose":
                            transaction.write_entity_state(
                                "object", ids, term_name="ablation", root_pose=pose
                            )
                        else:
                            transaction.reset_to_default(ids, term_name="ablation")
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    if iteration >= warmup:
                        times.append(elapsed)
                    if len(commits) != 1:
                        raise AssertionError("every path must submit exactly one request")
                    request = commits[0]
                    patch = request.patches[0]
                    signature = {
                        key: None if getattr(patch, key) is None else getattr(patch, key).tolist()
                        for key in (
                            "root_pose",
                            "root_velocity",
                            "joint_positions",
                            "joint_velocities",
                        )
                    }
                    signature.update(
                        joint_names=patch.joint_names,
                        env_ids=request.env_ids,
                        restore_default_controls=request.restore_default_controls,
                    )
                    if reference is None:
                        reference = signature
                    if signature != reference:
                        raise AssertionError(
                            f"semantic mismatch: {count}, {selected}, {kind}, {name}"
                        )
                results.append(
                    {
                        "N": count,
                        "selected": selected,
                        "kind": kind,
                        "path": name,
                        "median_ms": statistics.median(times),
                        "min_ms": min(times),
                        "max_ms": max(times),
                        "current_snapshot_calls": len(reads),
                        # Fixture's four float64 state arrays contain 7+6+2+2 values per env.
                        # This counts getter-return bytes, not all Python/NumPy allocation.
                        "current_snapshot_bytes": len(reads) * count * (7 + 6 + 2 + 2) * 8,
                        "commit_calls": len(commits),
                        "signature_matches_baseline": True,
                    }
                )
    return {
        "scope": "CPU host fixture only; no engine, IPC, worker, or training timing",
        "baseline_sha": baseline_sha,
        "current_head": _git("rev-parse", "HEAD"),
        "working_tree_status": _git("status", "--short"),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "warmup": warmup,
        "samples": samples,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", default=DEFAULT_BASELINE, help="local Git revision for path A"
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON evidence destination")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=9)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.samples <= 0:
        parser.error("warmup must be nonnegative and samples must be positive")
    report = run(args.baseline, warmup=args.warmup, samples=args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(report['results'])} host-fixture measurements to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
