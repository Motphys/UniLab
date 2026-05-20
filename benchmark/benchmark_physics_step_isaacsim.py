#!/usr/bin/env python3
"""
Benchmark Isaac Sim physics execution using USD assets.

The layout intentionally mirrors benchmark/benchmark_physics_step_isaacgym.py:
runtime setup, task specs, build/run helpers, one-task benchmark loop, plotting,
and a thin main().

Run without creating any new environment:
    /home/admin/isaacsim/python.sh benchmark/benchmark_physics_step_isaacsim.py \\
        --batch-sizes 256,512,1024,2048,4096,8192

What is timed:
    world.step(render=False)

For Isaac Sim 4.5 this reaches SimulationContext.step(render=False), which
steps PhysX through _physics_context._step() without rendering. Stage creation,
USD reference loading, GridCloner replication, world.reset(), and teardown are
outside the timed section.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ISAACSIM_ROOT = Path(os.environ.get("ISAACSIM_ROOT", "/home/admin/isaacsim"))
DEFAULT_USD_ROOT = Path(
    os.environ.get(
        "ISAACSIM_USD_ROOT",
        "/home/admin/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/UniLabBenchmark",
    )
)
DEFAULT_EXPERIENCE = "apps/isaacsim.exp.base.python.kit"
DEFAULT_OUT_DIR = REPO_ROOT / "benchmark" / "outputs" / "physics_step" / "isaacsim"
DEFAULT_BATCH_SIZES = [2**k for k in range(8, 14)]  # 256 .. 8192

_simulation_app: Any = None
_isaac_modules: Dict[str, Any] = {}


@dataclass(frozen=True)
class TaskSpec:
    owner_task_id: str
    display_name: str
    usd_file: str
    articulation_root_prim: str
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


TASK_SPECS: Dict[str, TaskSpec] = {
    "go1_joystick_flat": TaskSpec(
        owner_task_id="go1_joystick_flat",
        display_name="go1_joystick_flat",
        usd_file="go1_description/go1.usd",
        articulation_root_prim="base",
        initial_height=0.40,
    ),
    "go2_joystick_flat": TaskSpec(
        owner_task_id="go2_joystick_flat",
        display_name="go2_joystick_flat",
        usd_file="go2_description/go2.usd",
        articulation_root_prim="base",
        initial_height=0.40,
    ),
    "g1_walk_flat": TaskSpec(
        owner_task_id="g1_walk_flat",
        display_name="g1_walk_flat",
        usd_file="g1_description/g1_29dof_rev_1_0.usd",
        articulation_root_prim="pelvis",
        initial_height=0.78,
    ),
    "sharpa_inhand": TaskSpec(
        owner_task_id="sharpa_inhand",
        display_name="sharpa_inhand",
        usd_file="sharpa_wave/right_sharpa_wave.usda",
        articulation_root_prim="root_joint",
        initial_height=0.30,
    ),
}
TASK_ALIASES = {
    "Go1JoystickFlat": "go1_joystick_flat",
    "Go2JoystickFlat": "go2_joystick_flat",
    "G1WalkFlat": "g1_walk_flat",
    "SharpaInhandRotation": "sharpa_inhand",
    "go1": "go1_joystick_flat",
    "go2": "go2_joystick_flat",
    "g1": "g1_walk_flat",
    "sharpa": "sharpa_inhand",
    "sharpahand": "sharpa_inhand",
    **{f"task={task}/isaacsim": task for task in TASK_SPECS},
    **{f"{task}/isaacsim": task for task in TASK_SPECS},
}
DEFAULT_TASK_IDS = list(TASK_SPECS.keys())


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
        f"{DEFAULT_ISAACSIM_ROOT}/python.sh "
        "benchmark/benchmark_physics_step_isaacsim.py "
        "--batch-sizes 256,512,1024,2048,4096,8192"
    )


def _normalize_task_name(task_name: str) -> str:
    normalized = task_name.strip()
    if normalized.startswith("task="):
        normalized = normalized[len("task=") :]
    if (
        normalized.endswith("/mujoco")
        or normalized.endswith("/motrix")
        or normalized.endswith("/genesis")
        or normalized.endswith("/isaacgym")
    ):
        normalized = normalized.rsplit("/", 1)[0]
    if normalized in TASK_SPECS:
        return normalized
    alias_target = TASK_ALIASES.get(normalized)
    if alias_target is not None:
        return alias_target
    raise ValueError(
        f"Unknown task '{task_name}'. Available task ids: {list(TASK_SPECS.keys())}. "
        "Accepted aliases include legacy env names and task=<name>/isaacsim forms."
    )


def _task_spec(task_name: str) -> TaskSpec:
    return TASK_SPECS[_normalize_task_name(task_name)]


def _resolve_experience(isaacsim_root: Path, experience_arg: str) -> Path:
    candidate = Path(experience_arg).expanduser()
    if not candidate.is_absolute():
        candidate = isaacsim_root / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Isaac Sim experience file not found at {candidate}. "
            "Override --isaacsim-root or --experience."
        )
    return candidate


def _resolve_usd_path(usd_root: Path, spec: TaskSpec) -> Path:
    path = (usd_root / spec.usd_file).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"USD asset for task '{spec.owner_task_id}' not found at {path}. "
            f"Expected relative path under --usd-root: {spec.usd_file}"
        )
    return path


def _bootstrap_isaacsim(experience: Path, headless: bool = True) -> None:
    """Start SimulationApp and import Isaac Sim modules lazily."""
    global _simulation_app
    if _simulation_app is not None:
        return
    try:
        try:
            from isaacsim import SimulationApp
        except ImportError:
            from omni.isaac.kit import SimulationApp  # legacy Isaac Sim path

        _simulation_app = SimulationApp({"headless": headless, "experience": str(experience)})

        import carb
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.cloner import GridCloner
        from isaacsim.core.prims import Articulation as ArticulationView
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import Usd

        settings = carb.settings.get_settings()
        settings.set("/physics/updateToUsd", False)
        settings.set("/physics/updateVelocitiesToUsd", False)
        settings.set("/physics/updateForceSensorsToUsd", False)
        settings.set("/persistent/omnihydra/useSceneGraphInstancing", True)

        _isaac_modules.update(
            {
                "World": World,
                "ArticulationView": ArticulationView,
                "GridCloner": GridCloner,
                "add_reference_to_stage": add_reference_to_stage,
                "omni_usd": omni.usd,
                "Usd": Usd,
            }
        )
    except Exception as exc:
        raise RuntimeError(
            "Isaac Sim is unavailable in the current runtime.\n"
            f"Import detail: {exc!r}\n"
            f"Use the existing Isaac Sim runtime directly:\n  {_runtime_hint()}"
        ) from exc


def _new_stage() -> None:
    _isaac_modules["omni_usd"].get_context().new_stage()
    cast(Any, _simulation_app).update()


def _strip_visuals(source_prim_path: str) -> int:
    Usd = _isaac_modules["Usd"]
    stage = _isaac_modules["omni_usd"].get_context().get_stage()
    source = stage.GetPrimAtPath(source_prim_path)
    if not source.IsValid():
        return 0
    count = 0
    for prim in Usd.PrimRange(source):
        if prim.GetName() == "visuals":
            prim.SetActive(False)
            count += 1
    return count


def _scale_physx_caps(world: Any, batch_size: int) -> None:
    # These caps avoid large-batch PhysX warnings and dropped contacts.
    ctx = world.get_physics_context()
    pairs = max(1 << 17, batch_size * 256)
    contacts = max(1 << 20, batch_size * 1024)
    ctx.set_gpu_found_lost_pairs_capacity(pairs)
    ctx.set_gpu_found_lost_aggregate_pairs_capacity(pairs)
    ctx.set_gpu_total_aggregate_pairs_capacity(pairs)
    ctx.set_gpu_max_rigid_contact_count(contacts)
    ctx.set_gpu_max_rigid_patch_count(contacts)


def _set_isaacgym_solver_iterations(articulations: Any) -> None:
    counts = int(articulations.count)
    articulations.set_solver_position_iteration_counts(np.full((counts,), 4, dtype=np.int32))
    articulations.set_solver_velocity_iteration_counts(np.full((counts,), 1, dtype=np.int32))


def _build_task_sim(
    task_name: str,
    batch_size: int,
    usd_root: Path,
    physics_dt: float,
    device: str,
    strip_visuals: bool,
    match_isaacgym_solver: bool,
) -> Any:
    spec = _task_spec(task_name)
    usd_path = _resolve_usd_path(usd_root, spec)

    World = _isaac_modules["World"]
    ArticulationView = _isaac_modules["ArticulationView"]
    GridCloner = _isaac_modules["GridCloner"]
    add_reference_to_stage = _isaac_modules["add_reference_to_stage"]

    world = World(
        physics_dt=physics_dt,
        rendering_dt=physics_dt,
        backend="torch",
        device=device,
    )
    _scale_physx_caps(world, batch_size)
    world.scene.add_default_ground_plane()

    source_prim = "/World/envs/env_0"
    add_reference_to_stage(usd_path=str(usd_path), prim_path=source_prim)
    stripped_visual_count = _strip_visuals(source_prim) if strip_visuals else 0

    cloner = GridCloner(spacing=2.0)
    cloner.define_base_env("/World/envs")
    target_paths = cloner.generate_paths("/World/envs/env", batch_size)
    position_offsets = np.zeros((batch_size, 3), dtype=np.float32)
    position_offsets[:, 2] = spec.initial_height
    cloner.clone(
        source_prim_path=source_prim,
        prim_paths=target_paths,
        position_offsets=position_offsets,
        replicate_physics=True,
    )

    articulations = ArticulationView(
        prim_paths_expr=f"/World/envs/.*/{spec.articulation_root_prim}",
        name=f"{spec.owner_task_id}_view",
    )
    if articulations.count != batch_size:
        raise RuntimeError(
            f"Articulation view matched {articulations.count} prims for "
            f"{spec.owner_task_id}, expected batch_size={batch_size}. "
            f"Pattern root prim: {spec.articulation_root_prim}"
        )
    if match_isaacgym_solver:
        _set_isaacgym_solver_iterations(articulations)

    world.scene.add(articulations)
    world.reset()

    prim_count = len(articulations.prim_paths)
    if prim_count != batch_size:
        raise RuntimeError(
            f"Initialized articulation view has {prim_count} prim paths for "
            f"{spec.owner_task_id}, expected batch_size={batch_size}."
        )
    print(
        f"[{spec.owner_task_id}] batch={batch_size:5d} "
        f"articulations={prim_count} dof={articulations.num_dof} "
        f"bodies={articulations.num_bodies} stripped_visuals={stripped_visual_count}"
    )
    return world


def _run_sim(world: Any, nstep: int, niter: int) -> float:
    if niter <= 0:
        return 0.0
    t0 = time.perf_counter()
    for _ in range(niter):
        for _ in range(nstep):
            world.step(render=False)
    return (time.perf_counter() - t0) / niter


def _destroy_task_sim(world: Any) -> None:
    world.stop()
    _isaac_modules["World"].clear_instance()
    _new_stage()


def _bench_one_task(
    task_name: str,
    batch_sizes: List[int],
    nstep: int,
    warmup: int,
    iters: int,
    usd_root: Path,
    physics_dt: float,
    device: str,
    strip_visuals: bool,
    match_isaacgym_solver: bool,
    on_record: Optional[Callable[[BenchRecord], None]] = None,
) -> List[BenchRecord]:
    task_key = _normalize_task_name(task_name)
    records: List[BenchRecord] = []

    for batch_size in batch_sizes:
        world = _build_task_sim(
            task_name=task_key,
            batch_size=batch_size,
            usd_root=usd_root,
            physics_dt=physics_dt,
            device=device,
            strip_visuals=strip_visuals,
            match_isaacgym_solver=match_isaacgym_solver,
        )
        try:
            _run_sim(world, nstep=nstep, niter=warmup)
            avg_t = _run_sim(world, nstep=nstep, niter=iters)
        finally:
            _destroy_task_sim(world)

        record = BenchRecord(
            task=task_key,
            backend="isaacsim",
            batch_size=batch_size,
            nstep=nstep,
            nthread=0,
            avg_time_sec=avg_t,
            sps=batch_size * nstep / avg_t,
        )
        records.append(record)
        print(
            f"[{task_key}] batch={batch_size:5d} "
            f"isaacsim={avg_t * 1000:.3f}ms ({record.sps / 1e4:.2f}万fps)"
        )
        if on_record is not None:
            on_record(record)

    return records


def _payload(
    records: List[BenchRecord],
    args: argparse.Namespace,
    task_names: List[str],
    batch_sizes: List[int],
    isaacsim_root: Path,
    experience: Path,
    usd_root: Path,
    complete: bool,
) -> Dict[str, object]:
    return {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "complete": complete,
            "device_info": _device_info_dict(),
            "tasks": task_names,
            "batch_sizes": batch_sizes,
            "nstep": args.nstep,
            "nthread": 0,
            "warmup": args.warmup,
            "iters": args.iters,
            "physics_dt": args.physics_dt,
            "device": args.device,
            "headless": not args.headed,
            "strip_visuals": args.strip_visuals,
            "update_to_usd": False,
            "scene_instancing": True,
            "match_isaacgym_solver": args.match_isaacgym_solver,
            "backends": ["isaacsim"],
            "isaacsim_root": str(isaacsim_root),
            "experience": str(experience),
            "usd_root": str(usd_root),
            "usd_files": {
                task: str(_resolve_usd_path(usd_root, _task_spec(task))) for task in task_names
            },
        },
        "results": [asdict(r) for r in records],
    }


def _write_results(
    out_json: Path,
    records: List[BenchRecord],
    args: argparse.Namespace,
    task_names: List[str],
    batch_sizes: List[int],
    isaacsim_root: Path,
    experience: Path,
    usd_root: Path,
    complete: bool,
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            _payload(
                records=records,
                args=args,
                task_names=task_names,
                batch_sizes=batch_sizes,
                isaacsim_root=isaacsim_root,
                experience=experience,
                usd_root=usd_root,
                complete=complete,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved results to {out_json}")


def _plot_fps(
    records: List[BenchRecord], out_png: Path, batch_sizes: List[int], task_names: List[str]
) -> None:
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
) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Isaac Sim physics execution")
    parser.add_argument("--isaacsim-root", type=str, default=str(DEFAULT_ISAACSIM_ROOT))
    parser.add_argument("--experience", type=str, default=DEFAULT_EXPERIENCE)
    parser.add_argument("--nstep", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--physics-dt", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--usd-root",
        type=str,
        default=str(DEFAULT_USD_ROOT),
        help="Directory containing go1/go2/g1/sharpa USD assets",
    )
    parser.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASK_IDS))
    parser.add_argument(
        "--batch-sizes", type=str, default=",".join(str(x) for x in DEFAULT_BATCH_SIZES)
    )
    parser.add_argument(
        "--no-strip-visuals",
        dest="strip_visuals",
        action="store_false",
        help="Keep USD visuals active while benchmarking",
    )
    parser.set_defaults(strip_visuals=True)
    parser.add_argument(
        "--match-isaacgym-solver",
        action="store_true",
        help="Set articulation solver iterations to position=4, velocity=1 before reset",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Launch with rendering window (default: headless)",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(DEFAULT_OUT_DIR / "results.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Directory for output plots",
    )
    args = parser.parse_args()

    if args.nstep <= 0:
        raise ValueError("--nstep must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")

    isaacsim_root = Path(args.isaacsim_root).expanduser().resolve()
    usd_root = Path(args.usd_root).expanduser().resolve()
    if not usd_root.is_dir():
        raise NotADirectoryError(f"--usd-root is not a directory: {usd_root}")
    experience = _resolve_experience(isaacsim_root, args.experience)

    task_names = [_normalize_task_name(x) for x in args.tasks.split(",") if x.strip()]
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    if not task_names:
        raise ValueError("--tasks must contain at least one task")
    if not batch_sizes:
        raise ValueError("--batch-sizes must contain at least one size")
    if any(size <= 0 for size in batch_sizes):
        raise ValueError("--batch-sizes must contain positive integers")

    # Validate USD paths before booting Kit, which is slow and noisy.
    for task_name in task_names:
        _resolve_usd_path(usd_root, _task_spec(task_name))

    print("Isaac Sim backend available")
    print(f"Tasks: {task_names}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"USD root: {usd_root}")
    print(f"Isaac Sim root: {isaacsim_root}")
    print(f"Experience: {experience}")
    print(f"Device: {args.device}")
    print(f"Strip visuals: {args.strip_visuals}")
    print(f"Match IsaacGym solver: {args.match_isaacgym_solver}")

    _bootstrap_isaacsim(experience=experience, headless=not args.headed)

    records: List[BenchRecord] = []
    out_json = Path(args.out_json).expanduser().resolve()

    def _on_record(record: BenchRecord) -> None:
        records.append(record)
        _write_results(
            out_json=out_json,
            records=records,
            args=args,
            task_names=task_names,
            batch_sizes=batch_sizes,
            isaacsim_root=isaacsim_root,
            experience=experience,
            usd_root=usd_root,
            complete=False,
        )

    try:
        for task_name in task_names:
            _bench_one_task(
                task_name=task_name,
                batch_sizes=batch_sizes,
                nstep=args.nstep,
                warmup=args.warmup,
                iters=args.iters,
                usd_root=usd_root,
                physics_dt=args.physics_dt,
                device=args.device,
                strip_visuals=args.strip_visuals,
                match_isaacgym_solver=args.match_isaacgym_solver,
                on_record=_on_record,
            )

        _write_results(
            out_json=out_json,
            records=records,
            args=args,
            task_names=task_names,
            batch_sizes=batch_sizes,
            isaacsim_root=isaacsim_root,
            experience=experience,
            usd_root=usd_root,
            complete=True,
        )

        out_dir = Path(args.out_dir).expanduser().resolve()
        _plot_fps(records, out_dir / "fps.png", batch_sizes=batch_sizes, task_names=task_names)
        _plot_per_env_sps(
            records, out_dir / "per_env_sps.png", batch_sizes=batch_sizes, task_names=task_names
        )
    finally:
        if _simulation_app is not None:
            try:
                cast(Any, _simulation_app).close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
