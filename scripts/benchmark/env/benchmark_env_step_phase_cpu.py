"""Per-phase wall/CPU attribution for a full task env step (issue #1328).

Builds a real task env through the same Hydra compose + ``BackendAdapter``
override path the off-policy collector uses, wraps ``backend.step`` /
``update_state`` / ``_reset_done_envs`` with process-wide CPU-time measurement
(``os.times``), and reports each phase's wall share and the average number of
cores it kept busy. This makes low-parallelism host phases visible next to the
thread-pool physics phase.

``--cpu-ids 0-31`` additionally injects ``EnvCfg.cpu_ids`` into the env
override (the same key the multi-GPU DP collector path uses), which both pins
the MuJoCo pool workers and confines the process's host-side compute via
``apply_env_cpu_runtime`` — the A/B used in the issue.

Run:
    uv run scripts/benchmark/env/benchmark_env_step_phase_cpu.py

    # pinned A/B:
    uv run scripts/benchmark/env/benchmark_env_step_phase_cpu.py --cpu-ids 0-31

    # tuning:
    uv run scripts/benchmark/env/benchmark_env_step_phase_cpu.py \
        --config-group sac --task g1_motion_tracking/mujoco \
        --num-envs 4096 --warmup 20 --iters 150
"""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def _cpu_time() -> float:
    t = os.times()
    return t.user + t.system


def _parse_cpu_ids(spec: str) -> list[int]:
    ids: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        elif part:
            ids.append(int(part))
    return ids


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-group", default="sac", help="conf/<group> used for compose")
    parser.add_argument("--task", default="g1_motion_tracking/mujoco")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=150)
    parser.add_argument(
        "--cpu-ids",
        default=None,
        help="Optional env cpu_ids override, e.g. '0-31'; pins the MuJoCo pool "
        "and confines host-side compute (sizes the pool to len(cpu_ids))",
    )
    args = parser.parse_args(argv)

    import hydra
    from omegaconf import OmegaConf

    from unilab.base.config_adapter import BackendAdapter, create_env
    from unilab.training import ensure_registries

    ensure_registries()
    with hydra.initialize_config_dir(
        version_base="1.3",
        config_dir=os.path.join(REPO_ROOT, "src", "unilab", "conf", args.config_group),
    ):
        cfg = hydra.compose(
            config_name="config",
            overrides=[f"task={args.task}", f"algo.num_envs={args.num_envs}"],
        )
    OmegaConf.resolve(cfg)
    env_cfg_override = BackendAdapter(
        cfg, root_dir=REPO_ROOT, algo_name=str(cfg.algo.algo)
    ).build_task_env_cfg_override()
    if args.cpu_ids is not None:
        env_cfg_override = {
            **(env_cfg_override or {}),
            "cpu_ids": _parse_cpu_ids(args.cpu_ids),
        }
    env = create_env(cfg, num_envs=args.num_envs, env_cfg_override=env_cfg_override)
    if env.state is None:
        env.init_state()

    wall_ms: defaultdict[str, float] = defaultdict(float)
    cpu_ms: defaultdict[str, float] = defaultdict(float)
    counts: defaultdict[str, int] = defaultdict(int)

    def wrap(name, fn):
        def wrapped(*a, **kw):
            w0 = time.perf_counter()
            c0 = _cpu_time()
            out = fn(*a, **kw)
            wall_ms[name] += (time.perf_counter() - w0) * 1000.0
            cpu_ms[name] += (_cpu_time() - c0) * 1000.0
            counts[name] += 1
            return out

        return wrapped

    env._backend.step = wrap("backend_step", env._backend.step)
    env.update_state = wrap("update_state", env.update_state)
    env._reset_done_envs = wrap("reset_done", env._reset_done_envs)

    action_dim = env.action_space.shape[-1]
    rng = np.random.default_rng(0)

    def actions():
        return rng.uniform(-0.2, 0.2, size=(args.num_envs, action_dim)).astype(np.float32)

    for _ in range(args.warmup):
        env.step(actions())
    wall_ms.clear()
    cpu_ms.clear()
    counts.clear()

    n_reset = 0
    wall0 = time.perf_counter()
    cpu0 = _cpu_time()
    for _ in range(args.iters):
        state = env.step(actions())
        n_reset += int(np.count_nonzero(state.terminated | state.truncated))
    total_wall = (time.perf_counter() - wall0) * 1000.0
    total_cpu = (_cpu_time() - cpu0) * 1000.0

    print(
        f"pool nthread={env._backend._n_threads} num_envs={args.num_envs} "
        f"cpu_ids={'None' if args.cpu_ids is None else args.cpu_ids}"
    )
    print(f"iters={args.iters} total_resets={n_reset}")
    print(f"{'phase':>16s} {'wall_ms':>9s} {'cpu_ms':>9s} {'cores':>6s} {'wall%':>6s}")
    step_wall = total_wall / args.iters
    for name in ("backend_step", "update_state", "reset_done"):
        w = wall_ms[name] / args.iters
        c = cpu_ms[name] / args.iters
        print(f"{name:>16s} {w:9.2f} {c:9.2f} {c / w if w else 0:6.2f} {100 * w / step_wall:6.1f}")
    other_w = total_wall - sum(wall_ms.values())
    other_c = total_cpu - sum(cpu_ms.values())
    print(
        f"{'other(step glue)':>16s} {other_w / args.iters:9.2f} {other_c / args.iters:9.2f} "
        f"{(other_c / other_w) if other_w > 0 else 0:6.2f} {100 * other_w / total_wall:6.1f}"
    )
    print(
        f"{'TOTAL step':>16s} {step_wall:9.2f} {total_cpu / args.iters:9.2f} "
        f"{total_cpu / total_wall:6.2f} {100.0:6.1f}"
    )
    print(f"steps/s={args.num_envs * args.iters / (total_wall / 1000.0):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
