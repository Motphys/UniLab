"""Per-segment wall-clock attribution for ObservationManager.compute (issue #1404).

Builds a real task env through the same Hydra compose + ``BackendAdapter``
override path as ``benchmark_env_step_phase_cpu.py``, then reports where time
inside ``ObservationManager.compute`` / ``compute_group`` goes, split by phase:

- ``step``: the full-batch per-step path (``compute(update_history=True)``)
- ``reset``: the row-scoped partial-reset path
  (``compute(update_history=True, env_ids=ids)``)

Segments (per term unless noted): ``term`` (raw term func), ``noise``,
``copy`` (defensive copy / reset row slice), ``clip``, ``scale``, ``nan``
(per-term NaN scan), ``delay``, ``history``, plus group-level ``concat``,
``group_nan`` and ``rows`` (temporal-group reset row slice). ``total`` wraps
each ``compute_group`` call; ``residual = total - sum(segments)`` is the
unattributed pipeline assembly (dict/list building, zip loop, isinstance and
shape checks, share-cache bookkeeping) and must stay below 30% of the total
for the attribution to count as explained.

Profiling is opt-in via ``UNILAB_TERM_PROFILING=1`` (this script sets it
before importing unilab); when disabled the instrumentation is a no-op.

Caveat (issue #1404 Phase 2): per-segment timings include the instrumentation
itself (two ``perf_counter`` calls plus a context manager per segment, and the
cache pollution that comes with them), which inflates and skews the
attribution — at 4096 envs the instrumented ``compute`` total reads ~1.7 ms
while the uninstrumented wall time is ~1.0 ms. Use the table for relative
per-term ranking inside a category, not for absolute category shares; verify
category-level claims with uninstrumented ablation (strip term noise /
``nan_policy`` at runtime) before acting on them.

Run:
    uv run scripts/benchmark/env/benchmark_obs_term_profile.py

    # tuning:
    uv run scripts/benchmark/env/benchmark_obs_term_profile.py \
        --config-group sac --task g1_motion_tracking/mujoco \
        --num-envs 4096 --warmup 20 --iters 150
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("UNILAB_TERM_PROFILING", "1")

import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from collections.abc import Sequence  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Segment categories in pipeline order; everything else attributed per term.
_CATEGORIES = (
    "term",
    "noise",
    "copy",
    "clip",
    "scale",
    "nan",
    "delay",
    "history",
    "concat",
    "group_nan",
    "rows",
)


def _parse_key(key: str) -> tuple[str, str, str]:
    """Split '<category>/<group>[/<term>]|<phase>' into (category, rest, phase)."""
    head, _, phase = key.partition("|")
    category, _, rest = head.partition("/")
    return category, rest, phase


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-group", default="sac", help="conf/<group> used for compose")
    parser.add_argument("--task", default="g1_motion_tracking/mujoco")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=150)
    parser.add_argument(
        "--reset-interval",
        type=int,
        default=5,
        help="Force a partial reset of --reset-rows envs every N steps so the "
        "reset path is covered even when no episode terminates naturally; "
        "0 disables forced resets",
    )
    parser.add_argument("--reset-rows", type=int, default=64)
    parser.add_argument("--top", type=int, default=15, help="Top-N per-term rows to print")
    args = parser.parse_args(argv)

    import hydra
    from omegaconf import OmegaConf

    from unilab.base.config_adapter import BackendAdapter, create_env
    from unilab.managers._profiling import SEGMENT_PROFILER
    from unilab.training import ensure_registries

    assert SEGMENT_PROFILER.enabled, "UNILAB_TERM_PROFILING=1 must reach the profiler import"

    ensure_registries()
    with hydra.initialize_config_dir(
        version_base="1.3", config_dir=os.path.join(REPO_ROOT, "conf", args.config_group)
    ):
        cfg = hydra.compose(
            config_name="config",
            overrides=[f"task={args.task}", f"algo.num_envs={args.num_envs}"],
        )
    OmegaConf.resolve(cfg)
    env_cfg_override = BackendAdapter(
        cfg, root_dir=REPO_ROOT, algo_name=str(cfg.algo.algo)
    ).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=args.num_envs, env_cfg_override=env_cfg_override)
    if env.state is None:
        env.init_state()

    action_dim = env.action_space.shape[-1]
    rng = np.random.default_rng(0)

    def actions() -> np.ndarray:
        return rng.uniform(-0.2, 0.2, size=(args.num_envs, action_dim)).astype(np.float32)

    reset_rows = np.arange(min(args.reset_rows, args.num_envs), dtype=np.int32)

    for _ in range(args.warmup):
        env.step(actions())
    SEGMENT_PROFILER.reset()

    n_step_iters = 0
    n_reset_calls = 0
    wall0 = time.perf_counter()
    for it in range(args.iters):
        env.step(actions())
        n_step_iters += 1
        if args.reset_interval > 0 and (it + 1) % args.reset_interval == 0:
            env.reset(env_ids=reset_rows)
            n_reset_calls += 1
    total_wall = time.perf_counter() - wall0

    stats = SEGMENT_PROFILER.stats()
    # category -> phase -> seconds
    cat_s: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    # (category, rest) -> phase -> seconds, for the per-term table
    row_s: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for key, (total_s, _calls) in stats.items():
        category, rest, phase = _parse_key(key)
        cat_s[category][phase] += total_s
        if category != "total":
            row_s[(category, rest)][phase] += total_s

    step_calls = max(n_step_iters, 1)
    reset_calls = max(n_reset_calls, 1)

    print(f"task={args.task} num_envs={args.num_envs} iters={args.iters}")
    print(f"step_calls={n_step_iters} reset_calls={n_reset_calls}")
    print(f"env step wall (profiling on): {total_wall / n_step_iters * 1e3:.2f} ms/step")
    phases = ("step", "reset")
    divisors = {"step": step_calls, "reset": reset_calls}

    totals = {p: cat_s["total"].get(p, 0.0) for p in phases}
    print("\n== per-category attribution (ms per call of that phase) ==")
    print(f"{'category':>10s} {'step_ms':>9s} {'reset_ms':>9s}")
    for cat in _CATEGORIES:
        row = cat_s.get(cat, {})
        print(
            f"{cat:>10s} {row.get('step', 0.0) / step_calls * 1e3:9.3f} "
            f"{row.get('reset', 0.0) / reset_calls * 1e3:9.3f}"
        )
    print(
        f"{'total':>10s} {totals['step'] / step_calls * 1e3:9.3f} "
        f"{totals['reset'] / reset_calls * 1e3:9.3f}"
    )

    for p in phases:
        attributed = sum(cat_s[c].get(p, 0.0) for c in _CATEGORIES)
        residual = totals[p] - attributed
        share = residual / totals[p] * 100.0 if totals[p] > 0 else 0.0
        print(
            f"[{p}] total={totals[p] / divisors[p] * 1e3:.3f} ms/call "
            f"residual={residual / divisors[p] * 1e3:.3f} ms/call ({share:.1f}% unexplained)"
        )

    print(f"\n== top {args.top} per-term segments (step ms/call) ==")
    print(f"{'segment':<56} {'step_ms':>9s} {'reset_ms':>9s}")
    rows = sorted(row_s.items(), key=lambda kv: -kv[1].get("step", 0.0))[: args.top]
    for (cat, rest), per_phase in rows:
        print(
            f"{cat + '/' + rest:<56} {per_phase.get('step', 0.0) / step_calls * 1e3:9.3f} "
            f"{per_phase.get('reset', 0.0) / reset_calls * 1e3:9.3f}"
        )

    step_total = totals["step"]
    if step_total > 0:
        attributed = sum(cat_s[c].get("step", 0.0) for c in _CATEGORIES)
        unexplained = (step_total - attributed) / step_total
        print(f"\nunexplained_gap(step)={unexplained * 100:.1f}% (must be < 30%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
