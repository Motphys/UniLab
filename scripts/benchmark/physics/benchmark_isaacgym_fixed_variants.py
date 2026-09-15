#!/usr/bin/env python3
"""Benchmark IsaacGym fixed-variant construction, memory, and stepping.

Each variant count runs in a fresh child process so ``ru_maxrss`` for the
IsaacGym worker is not contaminated by an earlier K point. The child prints one
JSON line; the parent collects the lines into the final artifact.

Example:
    uv run --no-sync python \\
        scripts/benchmark/physics/benchmark_isaacgym_fixed_variants.py \\
        --variant-counts 1 4 64 600 --num-envs 4096 \\
        --output /tmp/isaacgym-fixed-variants.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from unisim.backend.isaacgym.dependencies import isaacgym_runtime_available
from unisim.dr.types import FixedVariantPlan, ModelSourceDescriptor
from unisim.factory import create_backend
from unisim.scene import SceneCfg

_RESULT_MARKER = "__ISAACGYM_FIXED_VARIANT_RESULT__"


def _descendant_rss_kib() -> int:
    """Return the live RSS sum of this process's descendants (the worker)."""
    processes: dict[int, tuple[int, int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            ppid: int | None = None
            rss_kib = 0
            for line in (entry / "status").read_text(encoding="utf-8").splitlines():
                if line.startswith("PPid:"):
                    ppid = int(line.split()[1])
                elif line.startswith("VmRSS:"):
                    rss_kib = int(line.split()[1])
                    break
            if ppid is not None:
                processes[int(entry.name)] = (ppid, rss_kib)
        except (FileNotFoundError, ProcessLookupError, ValueError, PermissionError):
            continue

    children: dict[int, list[int]] = defaultdict(list)
    for pid, (ppid, _rss) in processes.items():
        children[ppid].append(pid)
    reachable: list[int] = list(children.get(os.getpid(), ()))
    total = 0
    while reachable:
        pid = reachable.pop()
        total += processes[pid][1]
        reachable.extend(children.get(pid, ()))
    return total


def _variant_xml(index: int) -> str:
    mass = 1.0 + 0.0025 * index
    size = 0.08 + 0.0001 * (index % 100)
    key = (index % 20) * 0.01
    kp = 20.0 + (index % 30)
    return f"""<mujoco model="IsaacGymFixedVariantBenchmark">
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="base">
      <freejoint/>
      <geom name="base_geom" type="box" size="{size} {size} {size}" mass="{mass}"/>
      <body name="link0">
        <joint name="j0" type="hinge" range="-1.5 1.5"/>
        <geom name="g0" type="box" size="0.1 0.1 0.1"/>
        <body name="link1">
          <joint name="j1" type="hinge" range="-1.5 1.5"/>
          <geom name="g1" type="box" size="0.1 0.1 0.1"/>
          <body name="link2">
            <joint name="j2" type="hinge" range="-1.5 1.5"/>
            <geom name="g2" type="box" size="0.1 0.1 0.1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="j0" joint="j0" kp="{kp}" kv="0.5" forcerange="-100 100"/>
    <position name="j1" joint="j1" kp="30" kv="0.5" forcerange="-100 100"/>
    <position name="j2" joint="j2" kp="40" kv="0.5" forcerange="-100 100"/>
  </actuator>
  <keyframe>
    <key name="home" qpos="0 0 0.8 1 0 0 0 {key} 0.2 -0.1"/>
  </keyframe>
</mujoco>
"""


def _write_sources(root: Path, count: int) -> tuple[Path, ...]:
    files: list[Path] = []
    for index in range(count):
        path = root / f"variant_{index:04d}.xml"
        path.write_text(_variant_xml(index), encoding="utf-8")
        files.append(path)
    return tuple(files)


def _gpu_info() -> dict[str, str]:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
        name, driver = (part.strip() for part in raw.splitlines()[0].split(",", 1))
        return {"gpu": name, "driver": driver}
    except Exception:
        return {"gpu": "unknown", "driver": "unknown"}


def _run_one(args: argparse.Namespace) -> None:
    if not isaacgym_runtime_available():
        raise RuntimeError("IsaacGym runtime is unavailable; set UNISIM_ISAACGYM_HOME")

    with tempfile.TemporaryDirectory(prefix="isaacgym-fixed-variants-") as tmp:
        source_root = Path(tmp)
        source_start = time.perf_counter()
        sources = _write_sources(source_root, args.variant_count)
        source_seconds = time.perf_counter() - source_start

        assignment = np.arange(args.num_envs, dtype=np.int32) % np.int32(args.variant_count)
        plan = FixedVariantPlan(
            assignment=assignment,
            variants=tuple(ModelSourceDescriptor(str(path)) for path in sources),
        )
        construction_start = time.perf_counter()
        backend = create_backend(
            "isaacgym",
            SceneCfg(model_file=str(sources[0]), fixed_variant_plan=plan),
            args.num_envs,
            args.sim_dt,
            base_name="base",
            device_id=args.device_id,
            worker_timeout_s=args.worker_timeout_s,
        )
        try:
            if not backend.get_dr_capabilities().supports_fixed_variants:
                raise RuntimeError("installed unisim-core lacks IsaacGym fixed variants")
            backend.materialize()
            construction_seconds = time.perf_counter() - construction_start
            worker_rss_kib = _descendant_rss_kib()

            ctrl = np.zeros((args.num_envs, 3), dtype=np.float32)
            for _ in range(args.warmup_steps):
                backend.step(ctrl, nsteps=args.nsteps)
            step_start = time.perf_counter()
            for _ in range(args.measure_steps):
                backend.step(ctrl, nsteps=args.nsteps)
            step_seconds = time.perf_counter() - step_start
            if not np.isfinite(backend.get_dof_pos()).all():
                raise RuntimeError("benchmark rollout produced non-finite dof state")
        finally:
            backend.close()

    physics_steps = args.measure_steps * args.nsteps
    result = {
        "variant_count": args.variant_count,
        "num_envs": args.num_envs,
        "device_id": args.device_id,
        "nsteps": args.nsteps,
        "source_generation_s": source_seconds,
        "construction_s": construction_seconds,
        "worker_rss_kib": worker_rss_kib,
        "measure_control_steps": args.measure_steps,
        "step_s": step_seconds,
        "control_steps_per_s": args.measure_steps / step_seconds,
        "physics_steps_per_s": physics_steps / step_seconds,
        "env_steps_per_s": (physics_steps * args.num_envs) / step_seconds,
    }
    print(_RESULT_MARKER + json.dumps(result), flush=True)


def _spawn_measurement(args: argparse.Namespace, variant_count: int) -> dict[str, float | int]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-one",
        "--variant-count",
        str(variant_count),
        "--num-envs",
        str(args.num_envs),
        "--device-id",
        str(args.device_id),
        "--nsteps",
        str(args.nsteps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--measure-steps",
        str(args.measure_steps),
        "--sim-dt",
        str(args.sim_dt),
        "--worker-timeout-s",
        str(args.worker_timeout_s),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    lines = [line for line in completed.stdout.splitlines() if line.startswith(_RESULT_MARKER)]
    if len(lines) != 1:
        raise RuntimeError(
            f"measurement child for K={variant_count} did not emit one result; "
            f"stdout={completed.stdout!r}, stderr={completed.stderr!r}"
        )
    result: dict[str, float | int] = json.loads(lines[0][len(_RESULT_MARKER) :])
    print(
        f"K={variant_count:>3}: construct={result['construction_s']:.3f}s, "
        f"worker_rss={result['worker_rss_kib'] / 1024:.1f}MiB, "
        f"env_steps/s={result['env_steps_per_s']:.0f}",
        flush=True,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-counts", type=int, nargs="+", default=[1, 4, 64, 600])
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--nsteps", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument("--sim-dt", type=float, default=0.005)
    parser.add_argument("--worker-timeout-s", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--run-one", action="store_true")
    parser.add_argument("--variant-count", type=int, default=1)
    args = parser.parse_args(argv)

    if args.run_one:
        _run_one(args)
        return 0

    artifact = {
        "schema": "unilab.isaacgym_fixed_variants.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        **_gpu_info(),
        "parameters": {
            "num_envs": args.num_envs,
            "device_id": args.device_id,
            "nsteps": args.nsteps,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "sim_dt": args.sim_dt,
        },
        "results": [
            _spawn_measurement(args, variant_count) for variant_count in args.variant_counts
        ],
    }
    rendered = json.dumps(artifact, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {args.output}", flush=True)
    else:
        print(rendered, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
