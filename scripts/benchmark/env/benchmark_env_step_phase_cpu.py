"""Per-phase wall/CPU attribution for a full task env step (issue #1328).

Builds a real task env through the same Hydra compose and ``EnvConfigAdapter``
override path the off-policy collector uses. The environment owns its phase
timings; this script only consumes ``state.info["timing"]`` after public
``env.step()`` calls, then reports each phase's wall share and the average
number of cores it kept busy. This keeps the benchmark independent of runtime
and backend-private methods.

``--cpu-ids 0-31`` additionally injects ``EnvCfg.cpu_ids`` into the env
override (the same key the multi-GPU DP collector path uses), which both pins
the MuJoCo pool workers and confines the process's host-side compute via
``apply_cpu_confinement`` — the A/B used in the issue.

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

import torch

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

PHASE_KEYS = (
    "apply_action_ms",
    "step_core_ms",
    "update_state_ms",
    "reset_done_ms",
    "env_step_other_ms",
)
PHASE_CPU_KEYS = (
    "apply_action_cpu_ms",
    "step_core_cpu_ms",
    "update_state_cpu_ms",
    "reset_done_cpu_ms",
    "env_step_other_cpu_ms",
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

    from unilab.base.config_adapter import EnvConfigAdapter, create_env
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
    env_cfg_override = EnvConfigAdapter(cfg, root_dir=REPO_ROOT).build_task_env_cfg_override()
    if args.cpu_ids is not None:
        env_cfg_override = {
            **(env_cfg_override or {}),
            "cpu_ids": _parse_cpu_ids(args.cpu_ids),
        }
    env = create_env(cfg, num_envs=args.num_envs, env_cfg_override=env_cfg_override)
    if env.state is None:
        env.init_state()

    action_dim = env.action_space.shape[-1]

    def actions() -> torch.Tensor:
        return (
            torch.rand((args.num_envs, action_dim), dtype=torch.float32, device=env.device).mul_(
                0.4
            )
            - 0.2
        )

    for _ in range(args.warmup):
        env.step(actions())

    wall_ms: dict[str, float] = defaultdict(float)
    cpu_ms: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    n_reset = 0
    wall0 = time.perf_counter()
    cpu0 = _cpu_time()
    for _ in range(args.iters):
        state = env.step(actions())
        timing = state.info.get("timing", {})
        total_ms = float(timing["env_step_total_ms"])
        measured_ms = sum(float(timing[key]) for key in PHASE_KEYS[:-1])
        timing["env_step_other_ms"] = max(total_ms - measured_ms, 0.0)
        for wall_key, cpu_key in zip(PHASE_KEYS, PHASE_CPU_KEYS, strict=True):
            wall_ms[wall_key] += float(timing[wall_key])
            cpu_ms[cpu_key] += float(timing[cpu_key])
            counts[wall_key] += 1
        n_reset += int((state.terminated | state.truncated).sum().item())
    total_wall = (time.perf_counter() - wall0) * 1000.0
    total_cpu = (_cpu_time() - cpu0) * 1000.0

    print(f"num_envs={args.num_envs} cpu_ids={'None' if args.cpu_ids is None else args.cpu_ids}")
    print(f"iters={args.iters} total_resets={n_reset}")
    print(f"{'phase':>16s} {'wall_ms':>9s} {'cpu_ms':>9s} {'cores':>6s} {'wall%':>6s}")
    step_wall = total_wall / args.iters
    for name in PHASE_KEYS:
        w = wall_ms[name] / max(counts[name], 1)
        c = cpu_ms[name] / max(counts[name], 1)
        print(f"{name:>16s} {w:9.2f} {c:9.2f} {c / w if w else 0:6.2f} {100 * w / step_wall:6.1f}")
    print(
        f"{'TOTAL step':>16s} {step_wall:9.2f} {total_cpu / args.iters:9.2f} "
        f"{total_cpu / total_wall:6.2f} {100.0:6.1f}"
    )
    print(f"steps/s={args.num_envs * args.iters / (total_wall / 1000.0):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
