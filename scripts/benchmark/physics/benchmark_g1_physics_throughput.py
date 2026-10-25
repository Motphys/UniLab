#!/usr/bin/env python3
# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false
"""Compare G1 physics throughput on Genesis, MJWarp, Newton, mjbatch, and MotrixSim.

This calls each engine directly and excludes controls, state/sensor copies,
observations, rewards, and resets from the measured region. All engines load the
same owner MJCF and start from its stand keyframe with the keyframe controls held.

Run on Apple Silicon (Metal) or Linux with NVIDIA CUDA / AMD ROCm:
    uv run --extra genesis --extra mujoco --extra motrix \
        scripts/benchmark/physics/benchmark_g1_physics_throughput.py

Add `--backends mjwarp` and the `mjwarp` extra to include MuJoCo Warp on CUDA.
Add `--backends newton` and the `newton` extra to include Newton on CUDA.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.benchmark.core.output import print_table, save_json
from scripts.benchmark.core.plotting import save_line_plot
from scripts.benchmark.core.runner import bench_callable, summarize
from scripts.benchmark.core.task_names import locomotion_task_model_file
from scripts.benchmark.core.utils import parse_sizes
from unilab.utils.device import get_device_info_dict, get_device_info_line

try:
    import genesis as gs
except Exception as exc:
    gs = None
    _GENESIS_IMPORT_ERROR: Exception | None = exc
else:
    _GENESIS_IMPORT_ERROR = None

try:
    import mjbatch
except Exception as exc:
    mjbatch = None
    _MJBATCH_IMPORT_ERROR: Exception | None = exc
else:
    _MJBATCH_IMPORT_ERROR = None

try:
    import motrixsim as mtx
except Exception as exc:
    mtx = None
    _MOTRIX_IMPORT_ERROR: Exception | None = exc
else:
    _MOTRIX_IMPORT_ERROR = None

try:
    import mujoco_warp as mj_warp
    import warp
except Exception as exc:
    mj_warp = None
    warp = None
    _MJWARP_IMPORT_ERROR: Exception | None = exc
else:
    _MJWARP_IMPORT_ERROR = None

try:
    import newton
except Exception as exc:
    newton = None
    _NEWTON_IMPORT_ERROR: Exception | None = exc
else:
    _NEWTON_IMPORT_ERROR = None


TASK = "g1_walk_flat"
BACKENDS = ("genesis", "mjbatch", "motrixsim")
SUPPORTED_BACKENDS = (*BACKENDS, "mjwarp", "newton")
DEFAULT_BATCH_SIZES = (1024, 2048, 4096, 8192)
SIM_DT = 1.0 / 150.0
DEFAULT_NSTEP = 20
DEFAULT_WARMUP = 2
DEFAULT_ITERS = 3
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "scripts/benchmark/outputs/physics_step/g1_throughput"


@dataclass
class PhysicsRun:
    step: Callable[[], None]
    synchronize: Callable[[], None]
    cleanup: Callable[[], None]
    device: str
    nthread: int


def _noop() -> None:
    pass


def _effective_cpu_count() -> int:
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        return len(affinity(0))
    return os.cpu_count() or 1


def _validate_positive(values: list[int], name: str) -> list[int]:
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def _parse_backends(text: str) -> list[str]:
    backends = [part.strip() for part in text.split(",") if part.strip()]
    if not backends or any(backend not in SUPPORTED_BACKENDS for backend in backends):
        raise ValueError(f"backends must be selected from {','.join(SUPPORTED_BACKENDS)}")
    if len(set(backends)) != len(backends):
        raise ValueError("backends must not contain duplicates")
    return backends


def _require_backend(backend: str) -> None:
    module, error, package = {
        "genesis": (gs, _GENESIS_IMPORT_ERROR, "genesis"),
        "mjbatch": (mjbatch, _MJBATCH_IMPORT_ERROR, "mujoco"),
        "motrixsim": (mtx, _MOTRIX_IMPORT_ERROR, "motrix"),
        "mjwarp": (
            mj_warp if mj_warp is not None and warp is not None else None,
            _MJWARP_IMPORT_ERROR,
            "mjwarp",
        ),
        "newton": (
            newton if newton is not None and warp is not None else None,
            _NEWTON_IMPORT_ERROR,
            "newton",
        ),
    }[backend]
    if module is None:
        raise RuntimeError(
            f"{backend} is unavailable ({error!r}). Install the project's {package} extra."
        )


def _genesis_device(
    platform_name: str = sys.platform,
    machine: str = platform.machine(),
    hip: str | None = torch.version.hip,
) -> str:
    platform = platform_name
    if platform == "darwin":
        if machine not in ("arm64", "aarch64"):
            raise RuntimeError("Genesis Metal physics requires an Apple Silicon Mac")
        return "metal"
    if platform == "linux":
        return "amdgpu" if hip is not None else "cuda"
    raise RuntimeError("Genesis GPU benchmark supports Apple Silicon Metal or Linux CUDA/ROCm")


def _genesis_synchronize(device: str) -> Callable[[], None]:
    if device == "metal":
        return torch.mps.synchronize
    if device in ("cuda", "amdgpu"):
        return torch.cuda.synchronize
    raise ValueError(f"Unsupported Genesis GPU device: {device}")


def _keyframe0_state_and_ctrl(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model)
    ctrl = model.key_ctrl[0] if model.nkey and model.nu else np.zeros(model.nu)
    return (
        np.array(data.qpos, dtype=np.float64),
        np.array(data.qvel, dtype=np.float64),
        np.array(ctrl, dtype=np.float64),
    )


def _load_keyframe(xml_path: str) -> tuple[mujoco.MjModel, np.ndarray, np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(xml_path)
    qpos, qvel, ctrl = _keyframe0_state_and_ctrl(model)
    if not np.isclose(model.opt.timestep, SIM_DT):
        raise RuntimeError(f"G1 timestep {model.opt.timestep} does not match expected {SIM_DT}")
    return model, qpos, qvel, ctrl


def _build_genesis(xml_path: str, batch_size: int, nstep: int) -> PhysicsRun:
    if gs is None:
        _require_backend("genesis")
    device = _genesis_device()
    gs.destroy()
    gs.init(backend=getattr(gs, device), precision="32")
    if gs.backend != getattr(gs, device):
        gs.destroy()
        raise RuntimeError(f"Genesis requested {device} but initialized {gs.backend}")

    try:
        scene = gs.Scene(
            show_viewer=False,
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            rigid_options=gs.options.RigidOptions(
                dt=SIM_DT,
                integrator=gs.integrator.implicitfast,
            ),
        )
        robot = scene.add_entity(gs.morphs.MJCF(file=xml_path))
        scene.build(n_envs=batch_size)
        model, qpos, _, ctrl = _load_keyframe(xml_path)
        robot.set_qpos(np.broadcast_to(qpos, (batch_size, qpos.size)).astype(np.float32))
        dofs_idx = model.jnt_dofadr[model.actuator_trnid[:, 0]] - model.jnt_dofadr[0]
        robot.control_dofs_position(
            np.broadcast_to(ctrl, (batch_size, ctrl.size)).astype(np.float32),
            dofs_idx_local=dofs_idx,
        )
    except Exception:
        gs.destroy()
        raise

    def step() -> None:
        for _ in range(nstep):
            scene.step()

    def cleanup() -> None:
        scene.destroy()
        gs.destroy()

    return PhysicsRun(step, _genesis_synchronize(device), cleanup, f"genesis_{device}", 0)


def _build_mjbatch(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    ctrl: np.ndarray,
    batch_size: int,
    nstep: int,
    nthread: int,
) -> PhysicsRun:
    if mjbatch is None:
        _require_backend("mjbatch")
    threads = nthread or min(batch_size, _effective_cpu_count())
    batch = mjbatch.Batch(model, batch_size, num_threads=threads)
    batch.bind("qpos", np.float64)[:] = qpos
    batch.bind("qvel", np.float64)[:] = qvel
    batch.bind("ctrl", np.float64)[:] = ctrl
    return PhysicsRun(lambda: batch.step(nstep=nstep), _noop, _noop, "cpu", threads)


def _build_motrixsim(
    xml_path: str,
    batch_size: int,
    nstep: int,
    nthread: int,
) -> PhysicsRun:
    if mtx is None:
        _require_backend("motrixsim")
    model = mtx.load_model(xml_path)
    from unisim.backend.motrix.backend import DEFAULT_MOTRIX_MAX_ITERATIONS

    model.options.max_iterations = DEFAULT_MOTRIX_MAX_ITERATIONS
    data = mtx.SceneData(model, batch=[batch_size])
    if model.num_keyframes:
        model.keyframes[0].apply(data)
        model.forward_kinematic(data)
        ctrl = np.asarray(model.keyframes[0].ctrl, dtype=np.float32)
    else:
        ctrl = np.zeros(model.num_actuators, dtype=np.float32)
    data.actuator_ctrls = np.ascontiguousarray(
        np.broadcast_to(ctrl, (batch_size, model.num_actuators))
    )
    threads = nthread or _effective_cpu_count()
    return PhysicsRun(lambda: model.step_n(data, nstep), _noop, _noop, "cpu", threads)


def _build_mjwarp(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    ctrl: np.ndarray,
    batch_size: int,
    nstep: int,
) -> PhysicsRun:
    if mj_warp is None or warp is None:
        _require_backend("mjwarp")
    if not warp.get_device().is_cuda:
        raise RuntimeError("MuJoCo Warp throughput benchmark requires a CUDA device")
    warp_model = mj_warp.put_model(model)
    data = mj_warp.make_data(model, nworld=batch_size, njmax=256, nconmax=256)
    data.qpos.assign(np.broadcast_to(qpos, (batch_size, qpos.size)).astype(np.float32))
    data.qvel.assign(np.broadcast_to(qvel, (batch_size, qvel.size)).astype(np.float32))
    data.ctrl.assign(np.broadcast_to(ctrl, (batch_size, ctrl.size)).astype(np.float32))

    def step() -> None:
        for _ in range(nstep):
            mj_warp.step(warp_model, data)

    return PhysicsRun(step, warp.synchronize, _noop, "cuda", 0)


def _newton_state_and_targets(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    ctrl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not np.allclose(qvel, 0.0):
        raise ValueError("Newton benchmark requires the zero-velocity stand keyframe")
    newton_qpos = np.array(qpos, dtype=np.float32, copy=True)
    newton_qpos[3:7] = np.roll(newton_qpos[3:7], -1)
    target_q = np.zeros(model.nq, dtype=np.float32)
    target_qd = np.zeros(model.nv, dtype=np.float32)
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        bias = model.actuator_biasprm[actuator_id]
        if abs(float(bias[1])) > 1e-8:
            target_q[model.jnt_qposadr[joint_id]] = ctrl[actuator_id]
        elif abs(float(bias[2])) > 1e-8:
            target_qd[model.jnt_dofadr[joint_id]] = ctrl[actuator_id]
        else:
            raise ValueError("Newton benchmark requires position or velocity actuators")
    return newton_qpos, np.asarray(qvel, dtype=np.float32), target_q, target_qd


def _build_newton(
    xml_path: str,
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    ctrl: np.ndarray,
    batch_size: int,
    nstep: int,
) -> PhysicsRun:
    if newton is None or warp is None:
        _require_backend("newton")
    if not warp.get_device().is_cuda:
        raise RuntimeError("Newton throughput benchmark requires a CUDA device")
    device = warp.get_device().alias
    template = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(template)
    template.add_mjcf(xml_path, ctrl_direct=False)
    builder = newton.ModelBuilder()
    builder.replicate(template, batch_size)
    previous_layout = newton.use_coord_layout_targets
    newton.use_coord_layout_targets = True
    try:
        newton_model = builder.finalize(device=device)
    finally:
        newton.use_coord_layout_targets = previous_layout
    solver = newton.solvers.SolverMuJoCo(
        newton_model,
        separate_worlds=True,
        nconmax=320,
        njmax=512,
        solver="newton",
        integrator="implicitfast",
        use_mujoco_cpu=False,
        use_mujoco_contacts=True,
        update_data_interval=1,
    )
    state_in = newton_model.state()
    state_out = newton_model.state()
    control = newton_model.control()
    contacts = newton.Contacts(solver.get_max_contact_count(), 0, device=device)
    newton_qpos, newton_qvel, target_q, target_qd = _newton_state_and_targets(
        model, qpos, qvel, ctrl
    )
    state_in.joint_q.assign(
        np.ascontiguousarray(np.broadcast_to(newton_qpos, (batch_size, model.nq))).reshape(-1)
    )
    state_in.joint_qd.assign(
        np.ascontiguousarray(np.broadcast_to(newton_qvel, (batch_size, model.nv))).reshape(-1)
    )
    control.joint_target_q.assign(
        np.ascontiguousarray(np.broadcast_to(target_q, (batch_size, model.nq))).reshape(-1)
    )
    control.joint_target_qd.assign(
        np.ascontiguousarray(np.broadcast_to(target_qd, (batch_size, model.nv))).reshape(-1)
    )
    newton.eval_fk(newton_model, state_in.joint_q, state_in.joint_qd, state_in)
    solver.reset(state_in, flags=0)

    def step() -> None:
        nonlocal state_in, state_out
        for _ in range(nstep):
            state_in.clear_forces()
            solver.step(state_in, state_out, control, contacts, SIM_DT)
            state_in, state_out = state_out, state_in

    return PhysicsRun(step, warp.synchronize, _noop, "cuda", 0)


def _benchmark(
    backend: str,
    xml_path: str,
    batch_size: int,
    nstep: int,
    warmup: int,
    iters: int,
    nthread: int,
) -> dict[str, Any]:
    model, qpos, qvel, ctrl = _load_keyframe(xml_path)
    if backend == "genesis":
        run = _build_genesis(xml_path, batch_size, nstep)
    elif backend == "mjbatch":
        run = _build_mjbatch(model, qpos, qvel, ctrl, batch_size, nstep, nthread)
    elif backend == "mjwarp":
        run = _build_mjwarp(model, qpos, qvel, ctrl, batch_size, nstep)
    elif backend == "newton":
        run = _build_newton(xml_path, model, qpos, qvel, ctrl, batch_size, nstep)
    else:
        run = _build_motrixsim(xml_path, batch_size, nstep, nthread)

    try:
        elapsed = bench_callable(run.step, run.synchronize, warmup=warmup, repeat=iters)
        mean_sec = sum(elapsed) / len(elapsed)
        record = summarize(
            elapsed,
            metric=batch_size * nstep / mean_sec,
            metric_name="physics_steps_per_sec",
            task=TASK,
            backend=backend,
            device=run.device,
            batch_size=batch_size,
            nstep=nstep,
            nthread=run.nthread,
        )
        record["avg_time_sec"] = record["mean_sec"]
        record["sps"] = record["metric"]
        record["per_env_sps"] = record["sps"] / batch_size
        record["label"] = run.device if backend == "genesis" else f"{backend}_{run.device}"
        return record
    finally:
        run.cleanup()


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in (
        "genesis-world",
        "mujoco",
        "mjbatch-uni",
        "motrixsim-core",
        "mujoco-warp",
        "warp-lang",
        "newton",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--batch-sizes", default=",".join(map(str, DEFAULT_BATCH_SIZES)))
    parser.add_argument("--nstep", type=int, default=DEFAULT_NSTEP)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument(
        "--nthread",
        type=int,
        default=0,
        help="CPU workers per backend; 0 uses the effective CPU count",
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT_DIR / "results.json")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-plots", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        backends = _parse_backends(args.backends)
        batch_sizes = _validate_positive(parse_sizes(args.batch_sizes), "batch-sizes")
        _validate_positive([args.nstep, args.iters], "nstep and iters")
        if args.warmup < 0 or args.nthread < 0:
            raise ValueError("warmup and nthread must be non-negative")
    except ValueError as exc:
        build_parser().error(str(exc))

    for backend in backends:
        _require_backend(backend)
    if args.nthread and "motrixsim" in backends:
        mtx.init_thread_pool(num_threads=args.nthread)

    xml_path = locomotion_task_model_file(TASK)
    records = []
    for backend in backends:
        for batch_size in batch_sizes:
            record = _benchmark(
                backend, xml_path, batch_size, args.nstep, args.warmup, args.iters, args.nthread
            )
            records.append(record)
            print(
                f"{backend:9s} {record['device']:14s} num_env={batch_size:5d} "
                f"{record['avg_time_sec'] * 1000:9.3f} ms  {record['sps']:14.1f} steps/s"
            )

    print_table(
        [
            {
                "backend": record["label"],
                "num_env": record["batch_size"],
                "steps/s": f"{record['sps']:.1f}",
                "per-env steps/s": f"{record['per_env_sps']:.1f}",
                "std ms": f"{record['std_sec'] * 1000:.3f}",
            }
            for record in records
        ],
        ["backend", "num_env", "steps/s", "per-env steps/s", "std ms"],
    )
    save_json(
        args.out_json,
        records,
        {
            "task": TASK,
            "model_file": xml_path,
            "sim_dt": SIM_DT,
            "integrator": {
                "genesis": "implicitfast",
                "mjbatch": "implicitfast (MJCF)",
                "mjwarp": "implicitfast (MJCF)",
                "newton": "implicitfast",
                "motrixsim": "engine default (MJCF integrator unsupported)",
            },
            "solver_settings": {
                "genesis": "RigidOptions defaults",
                "mjbatch": "MJCF/MuJoCo defaults",
                "mjwarp": "eager kernels; njmax=256; nconmax=256; sync per nstep block",
                "newton": "SolverMuJoCo Newton; njmax=512; nconmax=320; sync per nstep block",
                "motrixsim": "UniSim production max_iterations=3",
            },
            "initial_state": "stand keyframe; keyframe ctrl held",
            "scope": "pure physics step; no control/state/sensor copies in timed region",
            "batch_sizes": batch_sizes,
            "backends": backends,
            "nstep": args.nstep,
            "warmup": args.warmup,
            "iters": args.iters,
            "package_versions": _package_versions(),
            "device_info": get_device_info_dict(),
        },
    )
    if not args.skip_plots:
        save_line_plot(
            records,
            "batch_size",
            "sps",
            "label",
            "G1 pure physics throughput",
            "Number of environments",
            "Physics steps/s",
            args.out_dir / "physics_steps_per_sec.png",
            yscale="linear",
            device_info=get_device_info_line(),
        )
        save_line_plot(
            records,
            "batch_size",
            "per_env_sps",
            "label",
            "G1 per-environment physics throughput",
            "Number of environments",
            "Per-env physics steps/s",
            args.out_dir / "per_env_steps_per_sec.png",
            yscale="linear",
            device_info=get_device_info_line(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
