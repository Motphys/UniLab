#!/usr/bin/env python3
"""
Benchmark Isaac Gym physics execution using URDF assets.

Benchmarks Isaac Gym across go1/go2/g1 locomotion robots and outputs JSON + plots
aligned with benchmark/benchmark_physics_step_mj_step.py.

Run without creating any new environment:
    export UNILAB_BENCHMARK_HOLOSOMA_DEPS=/home/admin1/.holosoma_deps
    export UNILAB_BENCHMARK_HSGYM_PYTHON=$UNILAB_BENCHMARK_HOLOSOMA_DEPS/miniconda3/envs/hsgym/bin/python3.8
    export UNILAB_BENCHMARK_HSGYM_LIB=$UNILAB_BENCHMARK_HOLOSOMA_DEPS/miniconda3/envs/hsgym/lib
    export UNILAB_BENCHMARK_MODELS_ROOT=/home/admin1/ws/models
    PYTHONPATH=$UNILAB_BENCHMARK_HOLOSOMA_DEPS/isaacgym/python \
    LD_LIBRARY_PATH=$UNILAB_BENCHMARK_HSGYM_LIB \
    uv run --no-project $UNILAB_BENCHMARK_HSGYM_PYTHON \
        benchmark/benchmark_physics_step_isaacgym.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Machine-specific runtime paths. Each fallback below is the path on the original
# benchmarking machine; override via environment variables when running elsewhere,
# or edit the fallback strings directly if you prefer hard-coding for your machine.
#   UNILAB_BENCHMARK_HOLOSOMA_DEPS   -> root of the holosoma deps tree (contains isaacgym/python)
#   UNILAB_BENCHMARK_HSGYM_PYTHON    -> the python3.8 binary that ships with the hsgym conda env
#   UNILAB_BENCHMARK_HSGYM_LIB       -> the hsgym conda env lib/ (LD_LIBRARY_PATH at launch)
#   UNILAB_BENCHMARK_MODELS_ROOT     -> directory containing URDF model trees referenced by TASK_SPECS
# Legacy UNILAB_ISAACGYM_* variables remain accepted as aliases for existing
# local benchmark setups.
DEFAULT_DEPS_ROOT = Path(
    os.environ.get(
        "UNILAB_BENCHMARK_HOLOSOMA_DEPS",
        os.environ.get("UNILAB_ISAACGYM_DEPS_ROOT", "/home/admin1/.holosoma_deps"),
    )
).expanduser()
DEFAULT_ISAACGYM_PYTHON = DEFAULT_DEPS_ROOT / "isaacgym" / "python"
DEFAULT_HSGYM_PYTHON = Path(
    os.environ.get(
        "UNILAB_BENCHMARK_HSGYM_PYTHON",
        DEFAULT_DEPS_ROOT / "miniconda3" / "envs" / "hsgym" / "bin" / "python3.8",
    )
).expanduser()
DEFAULT_HSGYM_LIB = Path(
    os.environ.get(
        "UNILAB_BENCHMARK_HSGYM_LIB",
        DEFAULT_DEPS_ROOT / "miniconda3" / "envs" / "hsgym" / "lib",
    )
).expanduser()
DEFAULT_MODELS_ROOT = Path(
    os.environ.get(
        "UNILAB_BENCHMARK_MODELS_ROOT",
        os.environ.get("UNILAB_ISAACGYM_MODELS_ROOT", "/home/admin1/ws/models"),
    )
).expanduser()

if str(DEFAULT_ISAACGYM_PYTHON) not in sys.path:
    sys.path.insert(0, str(DEFAULT_ISAACGYM_PYTHON))

_ISAACGYM_IMPORT_ERROR: Exception | None = None

try:
    from isaacgym import gymapi
except Exception as _isaacgym_error:
    gymapi = None
    _ISAACGYM_IMPORT_ERROR = _isaacgym_error


@dataclass(frozen=True)
class TaskSpec:
    owner_task_id: str
    display_name: str
    asset_root: Path
    asset_file: str
    initial_height: float


@dataclass
class BenchRecord:
    task: str
    backend: str
    batch_size: int
    nstep: int
    nthread: int
    avg_time_sec: float
    sps: float


TASK_SPECS = {
    "go1_joystick_flat": TaskSpec(
        owner_task_id="go1_joystick_flat",
        display_name="go1_joystick_flat",
        asset_root=DEFAULT_MODELS_ROOT,
        asset_file="go1_description/urdf/go1.urdf",
        initial_height=0.40,
    ),
    "go2_joystick_flat": TaskSpec(
        owner_task_id="go2_joystick_flat",
        display_name="go2_joystick_flat",
        asset_root=DEFAULT_MODELS_ROOT,
        asset_file="go2_description/urdf/go2_description.urdf",
        initial_height=0.40,
    ),
    "g1_walk_flat": TaskSpec(
        owner_task_id="g1_walk_flat",
        display_name="g1_walk_flat",
        asset_root=DEFAULT_MODELS_ROOT,
        asset_file="g1_description/g1_29dof_rev_1_0.urdf",
        initial_height=0.78,
    ),
    "sharpa_inhand": TaskSpec(
        owner_task_id="sharpa_inhand",
        display_name="sharpa_inhand",
        asset_root=DEFAULT_MODELS_ROOT,
        asset_file="right_sharpa_wave/right_sharpa_wave.urdf",
        initial_height=0.30,
    ),
}
TASK_ALIASES = {
    "Go1JoystickFlat": "go1_joystick_flat",
    "Go2JoystickFlat": "go2_joystick_flat",
    "G1WalkFlat": "g1_walk_flat",
    "SharpaInhandRotation": "sharpa_inhand",
    "task=go1_joystick_flat/isaacgym": "go1_joystick_flat",
    "task=go2_joystick_flat/isaacgym": "go2_joystick_flat",
    "task=g1_walk_flat/isaacgym": "g1_walk_flat",
    "task=sharpa_inhand/isaacgym": "sharpa_inhand",
    "go1_joystick_flat/isaacgym": "go1_joystick_flat",
    "go2_joystick_flat/isaacgym": "go2_joystick_flat",
    "g1_walk_flat/isaacgym": "g1_walk_flat",
    "sharpa_inhand/isaacgym": "sharpa_inhand",
    "go1": "go1_joystick_flat",
    "go2": "go2_joystick_flat",
    "g1": "g1_walk_flat",
    "sharpa": "sharpa_inhand",
    "sharpahand": "sharpa_inhand",
}
DEFAULT_TASK_IDS = list(TASK_SPECS.keys())
DEFAULT_BATCH_SIZES = [2**k for k in range(8, 15)]  # 256 .. 16384


def _device_info_dict() -> Dict[str, str]:
    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip()
        if gpu:
            info["gpu"] = gpu.splitlines()[0]
    except Exception:
        info["gpu"] = "unknown"
    return info


def _device_info_line() -> str:
    info = _device_info_dict()
    return f"{info.get('gpu', 'unknown')} | Python {info.get('python', 'unknown')}"


def _display_backend(backend: str) -> str:
    return backend


def _runtime_hint() -> str:
    return (
        f"PYTHONPATH={DEFAULT_ISAACGYM_PYTHON} "
        f"LD_LIBRARY_PATH={DEFAULT_HSGYM_LIB} "
        f"uv run --no-project {DEFAULT_HSGYM_PYTHON} "
        "benchmark/benchmark_physics_step_isaacgym.py"
    )


def _require_isaacgym() -> None:
    if gymapi is not None:
        return
    detail = repr(_ISAACGYM_IMPORT_ERROR) if _ISAACGYM_IMPORT_ERROR is not None else "unknown error"
    raise RuntimeError(
        "Isaac Gym is unavailable in the current runtime.\n"
        f"Import detail: {detail}\n"
        "Set UNILAB_BENCHMARK_HOLOSOMA_DEPS / UNILAB_BENCHMARK_HSGYM_* "
        "for your Isaac Gym runtime, then run:\n"
        f"  {_runtime_hint()}"
    )


def _normalize_task_name(task_name: str) -> str:
    normalized = task_name.strip()
    if normalized.startswith("task="):
        normalized = normalized[len("task=") :]
    if (
        normalized.endswith("/mujoco")
        or normalized.endswith("/motrix")
        or normalized.endswith("/genesis")
    ):
        normalized = normalized.rsplit("/", 1)[0]
    if normalized in TASK_SPECS:
        return normalized
    alias_target = TASK_ALIASES.get(normalized)
    if alias_target is not None:
        return alias_target
    raise ValueError(
        f"Unknown task '{task_name}'. Available task ids: {list(TASK_SPECS.keys())}. "
        "Accepted aliases include legacy env names and task=<name>/isaacgym forms."
    )


def _task_spec(task_name: str) -> TaskSpec:
    return TASK_SPECS[_normalize_task_name(task_name)]


def _default_dof_targets(dof_props) -> np.ndarray:
    lower = np.asarray(dof_props["lower"], dtype=np.float32)
    upper = np.asarray(dof_props["upper"], dtype=np.float32)
    targets = np.zeros_like(lower, dtype=np.float32)
    finite = np.isfinite(lower) & np.isfinite(upper) & (lower <= upper)
    targets[finite] = np.clip(targets[finite], lower[finite], upper[finite])
    return targets


def _create_sim(compute_device_id: int, graphics_device_id: int, nthread: int):
    gymapi_mod = cast(Any, gymapi)
    gym = gymapi_mod.acquire_gym()
    sim_params = gymapi_mod.SimParams()
    sim_params.dt = 0.01
    sim_params.substeps = 1
    sim_params.up_axis = gymapi_mod.UpAxis.UP_AXIS_Z
    sim_params.gravity = gymapi_mod.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = nthread
    sim_params.physx.use_gpu = True
    sim_params.use_gpu_pipeline = True
    sim = gym.create_sim(compute_device_id, graphics_device_id, gymapi_mod.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym simulation.")
    return gym, sim


def _build_task_sim(
    task_name: str,
    batch_size: int,
    nthread: int,
    compute_device_id: int,
    graphics_device_id: int,
    models_root: Path,
):
    spec = _task_spec(task_name)
    gym, sim = _create_sim(compute_device_id, graphics_device_id, nthread)
    gymapi_mod = cast(Any, gymapi)

    plane_params = gymapi_mod.PlaneParams()
    plane_params.normal = gymapi_mod.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    asset_options = gymapi_mod.AssetOptions()
    asset_options.flip_visual_attachments = True
    asset_options.armature = 0.01
    asset_options.default_dof_drive_mode = int(gymapi_mod.DOF_MODE_POS)

    asset = gym.load_asset(sim, str(models_root), spec.asset_file, asset_options)
    dof_props = gym.get_asset_dof_properties(asset)
    dof_props["driveMode"][:].fill(gymapi_mod.DOF_MODE_POS)
    dof_props["stiffness"][:].fill(1000.0)
    dof_props["damping"][:].fill(10.0)
    dof_targets = _default_dof_targets(dof_props)

    spacing = 1.0
    env_lower = gymapi_mod.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi_mod.Vec3(spacing, spacing, spacing)
    num_per_row = max(1, int(math.sqrt(batch_size)))

    pose = gymapi_mod.Transform()
    pose.r = gymapi_mod.Quat(0.0, 0.0, 0.0, 1.0)
    pose.p = gymapi_mod.Vec3(0.0, 0.0, spec.initial_height)

    for env_idx in range(batch_size):
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        actor = gym.create_actor(env, asset, pose, spec.display_name, env_idx, 2)
        gym.set_actor_dof_properties(env, actor, dof_props)
        if dof_targets.size > 0:
            gym.set_actor_dof_position_targets(env, actor, dof_targets)

    gym.prepare_sim(sim)
    return gym, sim


def _run_sim(gym, sim, nstep: int, niter: int) -> float:
    if niter <= 0:
        return 0.0
    t0 = time.perf_counter()
    for _ in range(niter):
        for _ in range(nstep):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
    return (time.perf_counter() - t0) / niter


def _bench_one_task(
    task_name: str,
    batch_sizes: List[int],
    nstep: int,
    nthread: int,
    warmup: int,
    iters: int,
    compute_device_id: int,
    graphics_device_id: int,
    models_root: Path,
) -> List[BenchRecord]:
    task_key = _normalize_task_name(task_name)
    records: List[BenchRecord] = []

    for batch_size in batch_sizes:
        gym, sim = _build_task_sim(
            task_name=task_key,
            batch_size=batch_size,
            nthread=nthread,
            compute_device_id=compute_device_id,
            graphics_device_id=graphics_device_id,
            models_root=models_root,
        )
        try:
            _run_sim(gym, sim, nstep=nstep, niter=warmup)
            avg_t = _run_sim(gym, sim, nstep=nstep, niter=iters)
        finally:
            gym.destroy_sim(sim)

        records.append(
            BenchRecord(
                task=task_key,
                backend="isaacgym",
                batch_size=batch_size,
                nstep=nstep,
                nthread=nthread,
                avg_time_sec=avg_t,
                sps=batch_size * nstep / avg_t,
            )
        )
        print(
            f"[{task_key}] batch={batch_size:5d} "
            f"isaacgym={avg_t * 1000:.3f}ms ({batch_size * nstep / avg_t / 1e4:.2f}万fps)"
        )

    return records


def _plot_fps(
    records: List[BenchRecord], out_png: Path, batch_sizes: List[int], task_names: List[str]
):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(f"Parallel Physics — Total FPS\n{_device_info_line()}", fontsize=9)

    for task_name in task_names:
        for backend in sorted({r.backend for r in records if r.task == task_name}):
            recs = sorted(
                [r for r in records if r.task == task_name and r.backend == backend],
                key=lambda r: r.batch_size,
            )
            ax.plot(
                [r.batch_size for r in recs],
                [r.sps / 1e4 for r in recs],
                marker="o",
                linestyle="-",
                label=f"{_task_spec(task_name).display_name} ({_display_backend(backend)})",
            )

    ax.set_xscale("log", base=2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(b) for b in batch_sizes], rotation=30, ha="right")
    ax.set_xlabel("Batch Size (Num Envs)")
    ax.set_ylabel("Total FPS (x1e4)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved FPS plot to {out_png}")


def _plot_per_env_sps(
    records: List[BenchRecord], out_png: Path, batch_sizes: List[int], task_names: List[str]
):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(f"Parallel Physics — Per-Env SPS\n{_device_info_line()}", fontsize=9)

    for task_name in task_names:
        for backend in sorted({r.backend for r in records if r.task == task_name}):
            recs = sorted(
                [r for r in records if r.task == task_name and r.backend == backend],
                key=lambda r: r.batch_size,
            )
            ax.plot(
                [r.batch_size for r in recs],
                [r.sps / r.batch_size for r in recs],
                marker="o",
                linestyle="-",
                label=f"{_task_spec(task_name).display_name} ({_display_backend(backend)})",
            )

    ax.set_xscale("log", base=2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(b) for b in batch_sizes], rotation=30, ha="right")
    ax.set_xlabel("Batch Size (Num Envs)")
    ax.set_ylabel("Per-Env Steps/s")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved per-env SPS plot to {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Isaac Gym physics execution")
    parser.add_argument("--nstep", type=int, default=20)
    parser.add_argument("--nthread", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--compute-device-id", type=int, default=0)
    parser.add_argument("--graphics-device-id", type=int, default=-1)
    parser.add_argument(
        "--models-root",
        type=str,
        default=str(DEFAULT_MODELS_ROOT),
        help="Root containing go1_description/go2_description/g1_description/sharpa_wave URDFs",
    )
    parser.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASK_IDS))
    parser.add_argument(
        "--batch-sizes", type=str, default=",".join(str(x) for x in DEFAULT_BATCH_SIZES)
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="benchmark/outputs/physics_step/isaacgym/results.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="benchmark/outputs/physics_step/isaacgym",
        help="Directory for output plots",
    )
    args = parser.parse_args()

    _require_isaacgym()

    task_names = [_normalize_task_name(x) for x in args.tasks.split(",") if x.strip()]
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    models_root = Path(args.models_root).expanduser().resolve()
    if not models_root.is_dir():
        raise NotADirectoryError(f"--models-root is not a directory: {models_root}")

    print("Isaac Gym backend available")
    print(f"Tasks: {task_names}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Model root: {models_root}")

    records: List[BenchRecord] = []
    for task_name in task_names:
        records.extend(
            _bench_one_task(
                task_name=task_name,
                batch_sizes=batch_sizes,
                nstep=args.nstep,
                nthread=args.nthread,
                warmup=args.warmup,
                iters=args.iters,
                compute_device_id=args.compute_device_id,
                graphics_device_id=args.graphics_device_id,
                models_root=models_root,
            )
        )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "device_info": _device_info_dict(),
            "tasks": task_names,
            "batch_sizes": batch_sizes,
            "nstep": args.nstep,
            "nthread": args.nthread,
            "warmup": args.warmup,
            "iters": args.iters,
            "compute_device_id": args.compute_device_id,
            "graphics_device_id": args.graphics_device_id,
            "backends": sorted({r.backend for r in records}),
            "models_root": str(models_root),
            "asset_files": {task: _task_spec(task).asset_file for task in task_names},
            "isaacgym_python_path": str(DEFAULT_ISAACGYM_PYTHON),
            "hsgym_python": str(DEFAULT_HSGYM_PYTHON),
        },
        "results": [asdict(r) for r in records],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved results to {out_json}")

    out_dir = Path(args.out_dir)
    _plot_fps(records, out_dir / "fps.png", batch_sizes=batch_sizes, task_names=task_names)
    _plot_per_env_sps(
        records, out_dir / "per_env_sps.png", batch_sizes=batch_sizes, task_names=task_names
    )


if __name__ == "__main__":
    main()
