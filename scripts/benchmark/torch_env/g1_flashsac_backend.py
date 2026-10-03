# HISTORICAL SHELVED-BACKEND BENCHMARK (#1811)
# This adapter-specific probe is retained for migration context only.
# It is not part of the scoped tensor Manager benchmark surface
# (mujoco/mjwarp/genesis/newton), is not discovered by default benchmark
# selection, and must not be used as a production support claim.

#!/usr/bin/env python3
"""Phase-local G1 FlashSAC backend tensor benchmark.

This minimal probe combines the real G1 owner scene/backend with the existing
Torch motion-tracking manager kernel. It is not an end-to-end learner/run
benchmark: inference IPC, replay insertion, and the production Manager-Based
dispatch are intentionally excluded. The measured phases are physics/backend
exchange, Update State, reset-row selection, Reset Done, backend reset, and
reset-state publication. Every iteration is Torch-synchronized before timing;
some backend phase boundaries are stream-ordered, so compare iteration totals
rather than individual phase durations across backend families.
Run it inside the sibling checkout workspace: this branch resolves UniSim,
UniLab-RL, and mjbatch through the relative paths in ``pyproject.toml``.
Each requested backend is measured in its own Python process to avoid sharing
 warmed vendor context and allocator state.
The MuJoCo result also records the packed transfer counters and semantic H2D/D2H
boundary inventory so host-bridge claims remain independently checkable.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
from unisim.backend.base import TensorIOSpec

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.benchmark.torch_env.motion_tracking import (  # noqa: E402
    CLIP_END_FRAME,
    MotionTrackingWorkload,
)

_PROFILER_ENV_KEYS = (
    "UNISIM_ISAAC_WORKER_PROFILE_TRACE",
    "UNISIM_ISAAC_WORKER_PROFILE_START_COMMAND",
    "UNISIM_ISAAC_WORKER_PROFILE_STOP_COMMAND",
)
_RUNTIME_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "UNILAB_LOCAL_UNISIM",
    "UNISIM_ISAACGYM_HOME",
    "UNISIM_ISAACGYM_PYTHON",
    "UNISIM_ISAACSIM_HOME",
    "UNISIM_ISAACSIM_PYTHON",
    *_PROFILER_ENV_KEYS,
)
_AFTER_RUN_GPU_QUIESCE_TIMEOUT_S = 10.0
_AFTER_RUN_GPU_QUIESCE_POLL_S = 0.25
_SCOPED_BACKENDS = ("mujoco", "mjwarp", "newton")
_EXTERNAL_WORKER_PACKAGES = {
    "isaacgym": ("isaacgym", "isaacgym-preview.4", "torch"),
    "isaacsim": ("isaacsim", "isaacsim-core", "isaaclab", "omniverse-kit", "torch"),
}
from scripts.benchmark.torch_env.xp import TorchBackend, TorchRng  # noqa: E402


def _host_bridge_sensor_names(body_names: tuple[str, ...]) -> tuple[str, ...]:
    return (
        "pelvis_local_linvel",
        "torso_gyro",
        *(
            f"{prefix}_{name}"
            for prefix in (
                "track_pos_w",
                "track_quat_w",
                "track_linvel_w",
                "track_angvel_w",
            )
            for name in body_names
        ),
    )


def _transfer_stats_delta(
    start: dict[str, int] | None, end: dict[str, int] | None
) -> dict[str, int] | None:
    if start is None or end is None:
        return None
    return {key: end.get(key, 0) - start.get(key, 0) for key in sorted(set(start) | set(end))}


_TRANSFER_BOUNDARY_INVENTORY: tuple[dict[str, object], ...] = (
    {
        "phase": "control",
        "direction": "d2h",
        "semantic_limit_per_control_step": 1,
        "implementation": "packed control tensor to pinned host staging",
    },
    {
        "phase": "cpu_physics",
        "direction": "none",
        "semantic_limit_per_control_step": 0,
        "implementation": "the selected CPU physics backend remains authoritative",
    },
    {
        "phase": "update_state_full_read",
        "direction": "h2d",
        "semantic_limit_per_control_step": 1,
        "implementation": "qpos/qvel/requested sensors packed into one host packet",
    },
    {
        "phase": "update_state_manager_compute",
        "direction": "none",
        "semantic_limit_per_control_step": 0,
        "implementation": "Torch reward, termination, observation, and bookkeeping on device",
    },
    {
        "phase": "reset_done_selection",
        "direction": "none",
        "semantic_limit_per_control_step": 0,
        "implementation": "done/row selection and reset-state construction on device",
    },
    {
        "phase": "reset_commit",
        "direction": "d2h",
        "semantic_limit_per_control_step": 1,
        "implementation": "selected row IDs and qpos/qvel packed into one reset packet",
    },
    {
        "phase": "reset_publication",
        "direction": "h2d",
        "semantic_limit_per_control_step": 1,
        "implementation": "selected state/sensors copied as one contiguous prefix, then scattered on device",
    },
    {
        "phase": "replay_publication",
        "direction": "excluded",
        "semantic_limit_per_control_step": None,
        "implementation": "outside this phase-local backend probe; no transfer claim is made",
    },
)


def _build_cfg(backend: str, num_envs: int, *, isaacsim_test_fixture: bool = False) -> Any:
    from hydra import compose, initialize_config_dir

    from unilab.base.config_adapter import BackendAdapter
    from unilab.base.registry import apply_cfg_overrides
    from unilab.envs import ManagerBasedRlEnvCfg

    if isaacsim_test_fixture:
        raise ValueError(
            "the IsaacSim test-fixture owner was productionized in #1771; "
            "use the flashsac g1_motion_tracking/isaacsim owner directly"
        )
    task_backend = backend

    with initialize_config_dir(
        config_dir=str(ROOT_DIR / "src" / "unilab" / "conf" / "flashsac"),
        version_base="1.3",
    ):
        owner_cfg = compose(
            config_name="config",
            overrides=[
                f"task=g1_motion_tracking/{task_backend}",
                f"algo.num_envs={num_envs}",
                "training.no_play=true",
                "hydra.run.dir=.",
                "hydra.output_subdir=null",
                "hydra/job_logging=disabled",
                "hydra/hydra_logging=disabled",
            ],
        )
    override = BackendAdapter(
        owner_cfg, root_dir=ROOT_DIR, algo_name="flashsac"
    ).build_task_env_cfg_override()
    cfg = ManagerBasedRlEnvCfg()
    apply_cfg_overrides(cfg, override)
    return cfg


def _backend_base_name(backend_name: str, robot: Any, scene: Any) -> str:
    """Map the logical G1 root to the physical root required by each owner."""
    root_name: str = robot.root_body_name
    if backend_name != "isaacsim" or not getattr(scene, "entity_assets", ()):
        return root_name
    physical_entity = getattr(robot, "physical_entity", None)
    if not physical_entity:
        raise ValueError("mapped IsaacSim robot selector must declare physical_entity")
    prefix = f"{physical_entity}/"
    local_name = root_name[len(prefix) :] if root_name.startswith(prefix) else root_name
    return f"{physical_entity}/{local_name}"


def _reset_stride(num_envs: int) -> int:
    """Keep smoke runs non-empty while retaining the 1/16 production cadence."""
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}")
    return 16 if num_envs >= 16 else num_envs


def _release_cuda_view_aliases(views: dict[str, Any]) -> None:
    """Drop the benchmark's only strong aliases before fail-closed IPC close."""
    views.clear()
    gc.collect()


def _clear_workload_state_aliases(workload: Any) -> None:
    """Drop state/sensor views retained by the benchmark workload."""
    for attr in (
        "dof_pos",
        "dof_vel",
        "linvel",
        "gyro",
        "body_pos",
        "body_quat",
        "body_lin_vel",
        "body_ang_vel",
    ):
        setattr(workload, attr, None)


def _bind_newton_benchmark_device() -> None:
    """Bind the standalone Newton benchmark to its benchmark CUDA device."""
    from unilab.base.process_device import configure_backend_process_device

    configure_backend_process_device("newton", "cuda:0")


def _build_backend(backend: str, num_envs: int, *, isaacsim_test_fixture: bool = False) -> Any:
    from unilab.base.backend_factory import create_backend, env_backend_kwargs

    if backend == "newton":
        _bind_newton_benchmark_device()
    cfg = _build_cfg(backend, num_envs, isaacsim_test_fixture=isaacsim_test_fixture)
    cfg.validate()
    assert cfg.scene is not None
    robot = cfg.scene.entities["robot"]
    kwargs = env_backend_kwargs(cfg)
    kwargs["base_name"] = _backend_base_name(backend, robot, cfg.scene)
    if backend == "mujoco":
        kwargs["tracked_body_names"] = tuple(robot.body_names)
    return create_backend(
        backend,
        cfg.scene,
        num_envs,
        cfg.sim_dt,
        body_state_required=True,
        **kwargs,
    )


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _worker_package_versions(python: Path, names: tuple[str, ...]) -> dict[str, str]:
    code = (
        "import importlib.metadata as metadata, json; "
        "names = json.loads(input());\n"
        "def package_version(name):\n"
        "    try:\n"
        "        return metadata.version(name)\n"
        "    except metadata.PackageNotFoundError:\n"
        "        return 'not-installed'\n"
        "print(json.dumps({name: package_version(name) for name in names}))"
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    try:
        result = subprocess.run(
            [str(python), "-c", code],
            input=json.dumps(names),
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if result.returncode != 0:
        return {"error": result.stderr.strip()[-1000:] or f"exit {result.returncode}"}
    try:
        values = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": "worker returned invalid version JSON"}
    return {
        name: str(values.get(name, "not-installed"))
        if isinstance(values.get(name), str)
        else "invalid"
        for name in names
    }


def _external_worker_runtime_provenance(backends: list[str]) -> dict[str, Any]:
    provenance: dict[str, Any] = {}
    for backend in sorted(set(backends) & set(_EXTERNAL_WORKER_PACKAGES)):
        try:
            module = importlib.import_module(f"unisim.backend.{backend}.dependencies")
            resolver = (
                module.resolve_isaacgym_runtime
                if backend == "isaacgym"
                else module.resolve_isaacsim_runtime
            )
            runtime = resolver()
        except Exception as exc:  # noqa: BLE001 - provenance must explain unavailable SDKs
            provenance[backend] = {"resolver_error": f"{type(exc).__name__}: {exc}"}
            continue
        provenance[backend] = {
            "python": str(runtime.python),
            "package_path": (
                None
                if getattr(runtime, "package_path", None) is None
                else str(runtime.package_path)
            ),
            "lib_path": None if runtime.lib_path is None else str(runtime.lib_path),
            "versions": _worker_package_versions(
                runtime.python, _EXTERNAL_WORKER_PACKAGES[backend]
            ),
        }
    return provenance


def _git_source_info(path: Path) -> dict[str, str | bool | None]:
    root = next(
        (candidate for candidate in (path, *path.parents) if (candidate / ".git").exists()),
        None,
    )
    if root is None:
        return {"git_root": None, "branch": None, "commit": None, "dirty": None}

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    def _worktree_digest() -> str | None:
        patch = subprocess.run(
            ["git", "-C", str(root), "diff", "--binary", "HEAD"],
            check=False,
            capture_output=True,
        ).stdout
        untracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"],
            check=False,
            capture_output=True,
        ).stdout
        digest = hashlib.sha256()
        digest.update(b"git-diff-binary\0")
        digest.update(patch)
        digest.update(b"untracked-files\0")
        for relative in untracked.split(b"\0"):
            if not relative:
                continue
            source = root / relative.decode(errors="surrogateescape")
            if not source.is_file():
                continue
            digest.update(relative + b"\0")
            digest.update(source.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    return {
        "git_root": str(root),
        "branch": git("branch", "--show-current") or None,
        "commit": git("rev-parse", "HEAD") or None,
        "dirty": bool(git("status", "--porcelain")),
        "worktree_patch_sha256": _worktree_digest() if root is not None else None,
    }


def _package_source_info(name: str) -> dict[str, str | bool | None]:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        return {"path": None, **_git_source_info(Path.cwd())}
    path = Path(spec.origin).resolve()
    return {"path": str(path), **_git_source_info(path)}


def _cpu_model() -> str:
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _nvidia_smi_fields() -> dict[str, str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,clocks.sm,clocks.mem,power.draw,power.limit",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if result.returncode != 0:
        return {}
    names = ("driver_version", "sm_clock_mhz", "memory_clock_mhz", "power_draw_w", "power_limit_w")
    values = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
    if not values:
        return {}
    return {
        name: value.strip().removesuffix(" W").removesuffix(" MHz")
        for name, value in zip(names, values.split(","), strict=False)
    }


def _nvidia_smi_snapshot() -> dict[str, Any]:
    """Record all GPUs and compute processes, including external contention."""
    queries = {
        "gpus": (
            "--query-gpu",
            "index,uuid,driver_version,clocks.sm,clocks.mem,power.draw,power.limit",
            (
                "index",
                "uuid",
                "driver_version",
                "sm_clock_mhz",
                "memory_clock_mhz",
                "power_draw_w",
                "power_limit_w",
            ),
        ),
        "compute_apps": (
            "--query-compute-apps",
            "pid,process_name,used_gpu_memory,gpu_uuid",
            ("pid", "process_name", "used_gpu_memory", "gpu_uuid"),
        ),
    }
    snapshot: dict[str, Any] = {}
    errors: list[str] = []
    for label, (command, query, names) in queries.items():
        try:
            result = subprocess.run(
                ["nvidia-smi", command, query, "--format=csv,noheader"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except OSError as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
            snapshot[label] = []
            continue
        except subprocess.TimeoutExpired as exc:
            errors.append(f"{label}: nvidia-smi timed out after {exc.timeout} seconds")
            snapshot[label] = []
            continue
        rows: list[dict[str, str]] = []
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            suffix = f": {detail}" if detail else ""
            errors.append(f"{label}: nvidia-smi exited {result.returncode}{suffix}")
        else:
            for line in result.stdout.splitlines():
                values = [value.strip() for value in line.split(",")]
                if values and values != ["No running processes found"]:
                    rows.append(dict(zip(names, values, strict=False)))
        snapshot[label] = rows
    if errors:
        snapshot["errors"] = errors
    return snapshot


def _benchmark_environment() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in _RUNTIME_ENV_KEYS}


def _validate_acceptance_gpu_snapshot(snapshot: dict[str, Any], phase: str) -> None:
    """Fail closed on either unavailable or contended nvidia-smi evidence."""
    if not isinstance(snapshot, dict):
        raise RuntimeError(
            f"acceptance benchmark cannot verify {phase} nvidia-smi GPU state: "
            "snapshot must be an object"
        )
    errors = snapshot.get("errors", [])
    if errors:
        details = "; ".join(str(error) for error in errors)
        raise RuntimeError(
            f"acceptance benchmark cannot verify {phase} nvidia-smi GPU state: {details}"
        )
    missing = [label for label in ("gpus", "compute_apps") if label not in snapshot]
    if missing:
        raise RuntimeError(
            f"acceptance benchmark cannot verify {phase} nvidia-smi GPU state: "
            f"missing {', '.join(missing)}"
        )
    gpus = snapshot["gpus"]
    compute_apps = snapshot["compute_apps"]
    if not isinstance(gpus, list) or not gpus:
        raise RuntimeError(
            f"acceptance benchmark cannot verify {phase} nvidia-smi GPU state: no GPUs"
        )
    if not isinstance(compute_apps, list):
        raise RuntimeError(
            f"acceptance benchmark cannot verify {phase} nvidia-smi GPU state: "
            "compute process query is unavailable"
        )
    if compute_apps:
        pids = ", ".join(str(row.get("pid", "?")) for row in compute_apps)
        raise RuntimeError(
            f"acceptance benchmark requires an idle GPU {phase}; active PIDs: {pids}"
        )


def _record_gpu_snapshots(result: dict[str, Any], before: dict[str, Any]) -> None:
    """Keep both snapshots and allow worker CUDA teardown to quiesce.

    Isaac external workers can deregister from CUDA slightly after the isolated
    benchmark process returns.  A bounded retry still fails closed on a process
    that remains resident; it does not turn contention into accepted evidence.
    """
    after = _nvidia_smi_snapshot()
    started = time.perf_counter()
    deadline = started + _AFTER_RUN_GPU_QUIESCE_TIMEOUT_S
    while (
        isinstance(after, dict)
        and isinstance(after.get("compute_apps"), list)
        and after["compute_apps"]
        and time.perf_counter() < deadline
    ):
        time.sleep(_AFTER_RUN_GPU_QUIESCE_POLL_S)
        after = _nvidia_smi_snapshot()
    result["after_run_gpu_quiesce_wait_s"] = time.perf_counter() - started
    result["gpu_snapshots"] = {"before": before, "after": after}
    _validate_acceptance_gpu_snapshot(after, "after-run")


def _validate_acceptance_environment() -> None:
    # The workers opt in based on variable presence, even when the value is empty.
    enabled = [name for name in _PROFILER_ENV_KEYS if name in os.environ]
    if enabled:
        raise RuntimeError(
            "acceptance benchmark requires Isaac worker profiling to be disabled; "
            f"unset {', '.join(enabled)}"
        )
    snapshot = _nvidia_smi_snapshot()
    _validate_acceptance_gpu_snapshot(snapshot, "before-run")


def _validate_benchmark_shape(num_envs: int, warmup: int, iters: int) -> None:
    _reset_stride(num_envs)
    if warmup < 0:
        raise ValueError(f"warmup must be nonnegative, got {warmup!r}")
    if iters < 1:
        raise ValueError(f"iters must be positive, got {iters!r}")


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(values),
        "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p50_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "iters": len(values),
    }


def _reset_row_stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p50": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "iters": len(values),
    }


def _backend_timing_stats(results: list[dict[str, Any] | None]) -> dict[str, dict[str, float]]:
    samples: dict[str, list[float]] = {}
    for result in results:
        for key, value in (result or {}).get("timing", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                samples.setdefault(key, []).append(float(value))
    return {key: _stats(values) for key, values in samples.items()}


def _materialize_and_negotiate(backend_name: str, sim_backend: Any) -> tuple[str, Any]:
    """Materialize before lifecycle negotiation for subprocess model metadata."""
    sim_backend.materialize()
    mode = sim_backend.tensor_execution().value
    if mode not in {"device_resident", "host_bridge"}:
        raise RuntimeError(f"backend {backend_name} did not negotiate a tensor lifecycle: {mode}")
    return mode, sim_backend.get_tensor_capabilities()


def _tensor_runtime_diagnostics(sim_backend: Any) -> dict[str, dict[str, bool | str | None]]:
    return {
        name: {
            "requested": diagnostic.requested,
            "enabled": diagnostic.enabled,
            "disable_reason": diagnostic.disable_reason,
        }
        for name, diagnostic in sim_backend.get_tensor_runtime_diagnostics().items()
    }


def _run(
    backend: str,
    num_envs: int,
    warmup: int,
    iters: int,
    *,
    isaacsim_test_fixture: bool = False,
) -> dict[str, Any]:
    _validate_benchmark_shape(num_envs, warmup, iters)
    if not torch.cuda.is_available():
        raise RuntimeError("G1 FlashSAC tensor benchmark requires CUDA")
    robot_body_names = tuple(
        _build_cfg(backend, num_envs, isaacsim_test_fixture=isaacsim_test_fixture)
        .scene.entities["robot"]
        .body_names
    )
    sim_backend = _build_backend(backend, num_envs, isaacsim_test_fixture=isaacsim_test_fixture)
    try:
        return _run_with_backend(
            backend,
            num_envs,
            warmup,
            iters,
            sim_backend,
            robot_body_names,
            isaacsim_test_fixture,
        )
    finally:
        sim_backend.close()


def _run_with_backend(
    backend: str,
    num_envs: int,
    warmup: int,
    iters: int,
    sim_backend: Any,
    robot_body_names: tuple[str, ...],
    isaacsim_test_fixture: bool,
) -> dict[str, Any]:
    device = torch.device("cuda", index=torch.cuda.current_device())
    xp = TorchBackend("cuda")
    workload = MotionTrackingWorkload(
        xp, TorchRng("cuda"), seed=7, vectorized_reset_rng=True, num_envs=num_envs
    )
    mode, capabilities = _materialize_and_negotiate(backend, sim_backend)
    runtime_diagnostics = _tensor_runtime_diagnostics(sim_backend)
    sim_backend.reset()
    sim_backend.get_body_ids(tuple(robot_body_names))
    host_bridge_plan = None
    cuda_view_aliases: dict[str, Any] = {}
    host_bridge_start_stats: dict[str, int] | None = None

    action = torch.zeros((num_envs, sim_backend.num_actuators), device=device)
    reset_rows = max(1, num_envs // 16)
    reset_stride = _reset_stride(num_envs)
    row_pattern = torch.arange(num_envs, device=device) % reset_stride
    cuda_view_aliases["device_state"] = sim_backend.get_state_views(("qpos", "qvel"), device=device)
    cuda_view_aliases["device_sensors"] = {}
    if mode == "device_resident":
        if backend == "isaacgym":
            # IsaacGym intentionally fails closed for reset-time rigid-body views.
            # One untimed refresh step establishes the persistent view aliases;
            # all measured iterations begin from the documented post-step phase.
            sim_backend.step_tensor(action, nsteps=3)
        cuda_view_aliases["device_sensors"] = {
            "linvel": sim_backend.get_sensor_view("pelvis_local_linvel", device=device),
            "gyro": sim_backend.get_sensor_view("torso_gyro", device=device),
            "body_pos": torch.stack(
                tuple(
                    sim_backend.get_sensor_view(f"track_pos_w_{name}", device=device)
                    for name in robot_body_names
                ),
                dim=1,
            ),
            "body_quat": torch.stack(
                tuple(
                    sim_backend.get_sensor_view(f"track_quat_w_{name}", device=device)
                    for name in robot_body_names
                ),
                dim=1,
            ),
            "body_lin_vel": torch.stack(
                tuple(
                    sim_backend.get_sensor_view(f"track_linvel_w_{name}", device=device)
                    for name in robot_body_names
                ),
                dim=1,
            ),
            "body_ang_vel": torch.stack(
                tuple(
                    sim_backend.get_sensor_view(f"track_angvel_w_{name}", device=device)
                    for name in robot_body_names
                ),
                dim=1,
            ),
        }
    if mode == "host_bridge":
        if not capabilities.packed_host_bridge:
            raise RuntimeError("host-bridge benchmark backend does not support packed I/O")
        host_bridge_plan = sim_backend.compile_host_bridge_io(
            TensorIOSpec(
                state_fields=("qpos", "qvel"),
                sensor_names=_host_bridge_sensor_names(tuple(robot_body_names)),
                device=device,
            )
        )
        cuda_view_aliases["host_bridge_initial"] = host_bridge_plan.read_state_sensors()
        assert cuda_view_aliases["host_bridge_initial"] is not None
        workload.dof_pos = cuda_view_aliases["host_bridge_initial"]["qpos"][:, 7:]
        workload.dof_vel = cuda_view_aliases["host_bridge_initial"]["qvel"][:, 6:]
        workload.linvel = cuda_view_aliases["host_bridge_initial"]["pelvis_local_linvel"]
        workload.gyro = cuda_view_aliases["host_bridge_initial"]["torso_gyro"]
        workload.body_pos = torch.stack(
            tuple(
                cuda_view_aliases["host_bridge_initial"][f"track_pos_w_{name}"]
                for name in robot_body_names
            ),
            dim=1,
        )
        workload.body_quat = torch.stack(
            tuple(
                cuda_view_aliases["host_bridge_initial"][f"track_quat_w_{name}"]
                for name in robot_body_names
            ),
            dim=1,
        )
        workload.body_lin_vel = torch.stack(
            tuple(
                cuda_view_aliases["host_bridge_initial"][f"track_linvel_w_{name}"]
                for name in robot_body_names
            ),
            dim=1,
        )
        workload.body_ang_vel = torch.stack(
            tuple(
                cuda_view_aliases["host_bridge_initial"][f"track_angvel_w_{name}"]
                for name in robot_body_names
            ),
            dim=1,
        )
    phase_samples: dict[str, list[float]] = {
        "backend_step_ms": [],
        "state_exchange_ms": [],
        "update_state_ms": [],
        "reset_selection_ms": [],
        "reset_done_ms": [],
        "backend_reset_ms": [],
        "reset_publish_ms": [],
        "iteration_ms": [],
    }
    reset_row_samples: list[float] = []
    backend_step_results: list[dict[str, Any] | None] = []
    backend_reset_results: list[dict[str, Any] | None] = []
    post_reset_results: list[dict[str, Any] | None] = []

    # Keep the synthetic action/reset schedule identical across backends even
    # when multiple runs share one CUDA context.
    torch.cuda.manual_seed(7)

    try:
        for iteration in range(warmup + iters):
            if iteration == warmup:
                torch.cuda.synchronize()
                phase_samples = {name: [] for name in phase_samples}
                reset_row_samples = []
                backend_step_results = []
                backend_reset_results = []
                post_reset_results = []
                host_bridge_start_stats = (
                    host_bridge_plan.transfer_stats if host_bridge_plan is not None else None
                )

            action.uniform_(-1.0, 1.0)
            workload.current_actions.copy_(action)
            started = time.perf_counter()

            phase = time.perf_counter()
            if host_bridge_plan is None:
                backend_step_results.append(sim_backend.step_tensor(action, nsteps=3))
            else:
                host_bridge_plan.write_control(action)
                backend_step_results.append(host_bridge_plan.step(nsteps=3))
            phase_samples["backend_step_ms"].append((time.perf_counter() - phase) * 1000.0)

            phase = time.perf_counter()
            if mode == "device_resident":
                workload.dof_pos = cuda_view_aliases["device_state"]["qpos"][:, 7:]
                workload.dof_vel = cuda_view_aliases["device_state"]["qvel"][:, 6:]
                torch.stack(
                    tuple(
                        sim_backend.get_sensor_view(f"track_pos_w_{name}", device=device)
                        for name in robot_body_names
                    ),
                    dim=1,
                    out=cuda_view_aliases["device_sensors"]["body_pos"],
                )
                torch.stack(
                    tuple(
                        sim_backend.get_sensor_view(f"track_quat_w_{name}", device=device)
                        for name in robot_body_names
                    ),
                    dim=1,
                    out=cuda_view_aliases["device_sensors"]["body_quat"],
                )
                torch.stack(
                    tuple(
                        sim_backend.get_sensor_view(f"track_linvel_w_{name}", device=device)
                        for name in robot_body_names
                    ),
                    dim=1,
                    out=cuda_view_aliases["device_sensors"]["body_lin_vel"],
                )
                torch.stack(
                    tuple(
                        sim_backend.get_sensor_view(f"track_angvel_w_{name}", device=device)
                        for name in robot_body_names
                    ),
                    dim=1,
                    out=cuda_view_aliases["device_sensors"]["body_ang_vel"],
                )
                workload.body_pos = cuda_view_aliases["device_sensors"]["body_pos"]
                workload.body_quat = cuda_view_aliases["device_sensors"]["body_quat"]
                workload.body_lin_vel = cuda_view_aliases["device_sensors"]["body_lin_vel"]
                workload.body_ang_vel = cuda_view_aliases["device_sensors"]["body_ang_vel"]
                workload.linvel = cuda_view_aliases["device_sensors"]["linvel"]
                workload.gyro = cuda_view_aliases["device_sensors"]["gyro"]
            else:
                assert host_bridge_plan is not None
                cuda_view_aliases["host_bridge_state"] = host_bridge_plan.read_state_sensors()
                workload.dof_pos = cuda_view_aliases["host_bridge_state"]["qpos"][:, 7:]
                workload.dof_vel = cuda_view_aliases["host_bridge_state"]["qvel"][:, 6:]
                workload.linvel = cuda_view_aliases["host_bridge_state"]["pelvis_local_linvel"]
                workload.gyro = cuda_view_aliases["host_bridge_state"]["torso_gyro"]
                torch.stack(
                    tuple(
                        cuda_view_aliases["host_bridge_state"][f"track_pos_w_{name}"]
                        for name in robot_body_names
                    ),
                    dim=1,
                    out=workload.body_pos,
                )
                torch.stack(
                    tuple(
                        cuda_view_aliases["host_bridge_state"][f"track_quat_w_{name}"]
                        for name in robot_body_names
                    ),
                    dim=1,
                    out=workload.body_quat,
                )
                torch.stack(
                    tuple(
                        cuda_view_aliases["host_bridge_state"][f"track_linvel_w_{name}"]
                        for name in robot_body_names
                    ),
                    dim=1,
                    out=workload.body_lin_vel,
                )
                torch.stack(
                    tuple(
                        cuda_view_aliases["host_bridge_state"][f"track_angvel_w_{name}"]
                        for name in robot_body_names
                    ),
                    dim=1,
                    out=workload.body_ang_vel,
                )
            xp.sync()
            phase_samples["state_exchange_ms"].append((time.perf_counter() - phase) * 1000.0)

            phase = time.perf_counter()
            obs, reward, terminated = workload.update_state(should_log=False)
            xp.sync()
            phase_samples["update_state_ms"].append((time.perf_counter() - phase) * 1000.0)
            del obs, reward

            # Keep a comparable scheduled-reset floor while resetting clip-end
            # rows immediately so subsequent frame gathers remain in range.
            phase = time.perf_counter()
            mask = (row_pattern == (iteration % reset_stride)) | (
                workload.current_frames > CLIP_END_FRAME
            )
            env_ids = mask.nonzero(as_tuple=False).reshape(-1)
            xp.sync()
            phase_samples["reset_selection_ms"].append((time.perf_counter() - phase) * 1000.0)
            reset_row_samples.append(float(env_ids.shape[0]))
            phase = time.perf_counter()
            qpos, qvel, reset_obs, _ = workload.reset_done(env_ids)
            xp.sync()
            phase_samples["reset_done_ms"].append((time.perf_counter() - phase) * 1000.0)
            del reset_obs

            phase = time.perf_counter()
            if host_bridge_plan is None:
                backend_reset_results.append(sim_backend.set_state_tensor(env_ids, qpos, qvel))
            else:
                backend_reset_results.append(host_bridge_plan.apply_reset(env_ids, qpos, qvel))
            phase_samples["backend_reset_ms"].append((time.perf_counter() - phase) * 1000.0)
            phase = time.perf_counter()
            if host_bridge_plan is None:
                workload.dof_pos[env_ids] = qpos[:, 7:]
                workload.dof_vel[env_ids] = qvel[:, 6:]
            else:
                cuda_view_aliases["host_bridge_reset_state"] = (
                    host_bridge_plan.read_selected_state_sensors()
                )
                post_reset_results.append({"timing": dict(host_bridge_plan.last_timing)})
                workload.body_pos.index_copy_(
                    0,
                    env_ids,
                    torch.stack(
                        tuple(
                            cuda_view_aliases["host_bridge_reset_state"][f"track_pos_w_{name}"][
                                env_ids
                            ]
                            for name in robot_body_names
                        ),
                        dim=1,
                    ),
                )
                workload.body_quat.index_copy_(
                    0,
                    env_ids,
                    torch.stack(
                        tuple(
                            cuda_view_aliases["host_bridge_reset_state"][f"track_quat_w_{name}"][
                                env_ids
                            ]
                            for name in robot_body_names
                        ),
                        dim=1,
                    ),
                )
                workload.body_lin_vel.index_copy_(
                    0,
                    env_ids,
                    torch.stack(
                        tuple(
                            cuda_view_aliases["host_bridge_reset_state"][f"track_linvel_w_{name}"][
                                env_ids
                            ]
                            for name in robot_body_names
                        ),
                        dim=1,
                    ),
                )
                workload.body_ang_vel.index_copy_(
                    0,
                    env_ids,
                    torch.stack(
                        tuple(
                            cuda_view_aliases["host_bridge_reset_state"][f"track_angvel_w_{name}"][
                                env_ids
                            ]
                            for name in robot_body_names
                        ),
                        dim=1,
                    ),
                )
            xp.sync()
            phase_samples["reset_publish_ms"].append((time.perf_counter() - phase) * 1000.0)
            phase_samples["iteration_ms"].append((time.perf_counter() - started) * 1000.0)
    finally:
        # CUDA IPC arenas intentionally fail closed while public views remain.
        # Clear workload-backed state aliases first, then drop the canonical
        # alias holder before releasing the worker.
        _clear_workload_state_aliases(workload)
        qpos = qvel = None
        _release_cuda_view_aliases(cuda_view_aliases)

    stats = {name: _stats(values) for name, values in phase_samples.items()}
    mean_total_s = stats["iteration_ms"]["mean_ms"] / 1000.0
    transfer_stats = (
        _transfer_stats_delta(host_bridge_start_stats, host_bridge_plan.transfer_stats)
        if host_bridge_plan is not None
        else None
    )
    return {
        "backend": backend,
        "isaacsim_owner": {
            "test_fixture_requested": isaacsim_test_fixture,
            "production_owner": "src/unilab/conf/flashsac/task/g1_motion_tracking/isaacsim.yaml"
            if backend == "isaacsim"
            else None,
        },
        "tensor_execution": mode,
        "tensor_process_topology": capabilities.process_topology.value,
        "tensor_data_plane": capabilities.data_plane.value,
        "tensor_stream_event_ownership": capabilities.stream_event_ownership,
        "tensor_torch_devices": list(capabilities.torch_devices),
        "tensor_runtime_diagnostics": runtime_diagnostics,
        "num_envs": num_envs,
        "physics_substeps_per_control_step": 3,
        "warmup": warmup,
        "iters": iters,
        "scheduled_reset_rows": reset_rows,
        "reset_rows": _reset_row_stats(reset_row_samples),
        "reset_policy": {
            "scheduled_cadence_control_steps": reset_stride,
            "scheduled_rows_target": reset_rows,
            "minimal_env_dense_reset": reset_stride != 16,
            "clip_end_overflow_reset": True,
            "terminated_reset": False,
        },
        "seeds": {
            "numpy_synthetic_workload": 7,
            "torch_cuda": 7,
        },
        "throughput_env_control_steps_per_s": num_envs / mean_total_s,
        "throughput_physics_substeps_per_s": 3 * num_envs / mean_total_s,
        "phases": stats,
        "transfer_boundary_inventory": (
            _TRANSFER_BOUNDARY_INVENTORY if mode == "host_bridge" else None
        ),
        "gpu_cpu_overlap": {
            "policy": "none_by_design" if mode == "host_bridge" else "not_applicable",
            "reason": (
                "each measured iteration ends at a Torch synchronization point"
                if mode == "host_bridge"
                else "device-resident physics has no host-bridge boundary in this probe"
            ),
        },
        "host_bridge_transfer_stats": transfer_stats,
        "backend_timings": {
            "step_tensor": _backend_timing_stats(backend_step_results),
            "set_state_tensor": _backend_timing_stats(backend_reset_results),
            "post_reset_read": _backend_timing_stats(post_reset_results),
        },
    }


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default="mjwarp,mujoco")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--acceptance",
        action="store_true",
        help="fail closed if Isaac worker profiling or other benchmark instrumentation is active",
    )
    parser.add_argument(
        "--isaacsim-test-fixture",
        action="store_true",
        help=(
            "explicitly load the #1675 test-fixture-only IsaacSim owner; "
            "requires --backends isaacsim"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--_single-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_quiet", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    backends = [backend.strip() for backend in args.backends.split(",") if backend.strip()]
    if not backends:
        parser.error("at least one backend is required")
    unsupported = sorted(set(backends) - set(_SCOPED_BACKENDS))
    if unsupported:
        parser.error(
            "shelved benchmark backend(s) are outside the tensor-only Manager runtime "
            f"({_SCOPED_BACKENDS}): {', '.join(unsupported)}; see issue #1811"
        )
    if args.isaacsim_test_fixture:
        parser.error(
            "--isaacsim-test-fixture is obsolete; the #1771 production owner is used by default"
        )
    return args, backends


def main() -> None:
    args, backends = _parse_args()
    _validate_benchmark_shape(args.num_envs, args.warmup, args.iters)
    if args.acceptance:
        _validate_acceptance_environment()
    if args._single_result is not None:
        result = _run(
            backends[0],
            args.num_envs,
            args.warmup,
            args.iters,
            isaacsim_test_fixture=args.isaacsim_test_fixture,
        )
        args._single_result.parent.mkdir(parents=True, exist_ok=True)
        args._single_result.write_text(json.dumps({"result": result}) + "\n")
        return

    results = []
    with tempfile.TemporaryDirectory(prefix="g1-flashsac-tensor-") as temporary_dir:
        for index, backend in enumerate(backends):
            print(f"benchmarking {backend} in an isolated process...", flush=True)
            result_path = Path(temporary_dir) / f"result-{index}.json"
            gpu_snapshot_before = _nvidia_smi_snapshot()
            if args.acceptance:
                _validate_acceptance_gpu_snapshot(gpu_snapshot_before, "before-run")
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--backends",
                    backend,
                    "--num-envs",
                    str(args.num_envs),
                    "--warmup",
                    str(args.warmup),
                    "--iters",
                    str(args.iters),
                    *(["--acceptance"] if args.acceptance else []),
                    *(["--isaacsim-test-fixture"] if args.isaacsim_test_fixture else []),
                    "--_single-result",
                    str(result_path),
                    "--_quiet",
                ],
                check=True,
                cwd=ROOT_DIR,
            )
            result = json.loads(result_path.read_text())["result"]
            if args.acceptance:
                _record_gpu_snapshots(result, gpu_snapshot_before)
            else:
                result["gpu_snapshots"] = {
                    "before": gpu_snapshot_before,
                    "after": _nvidia_smi_snapshot(),
                }
            results.append(result)
            print(
                f"{backend}: {result['throughput_env_control_steps_per_s']:,.0f} "
                "env control steps/s; "
                f"iteration={result['phases']['iteration_ms']['mean_ms']:.3f} ms",
                flush=True,
            )
    cuda_index = torch.cuda.current_device()
    payload = {
        "schema_version": "0.3.0",
        "scope": "phase-local",
        "excluded_components": [
            "inference_ipc",
            "replay_ingestion",
            "learner_update",
            "production_manager_dispatch",
        ],
        "timing_semantics": "total_iteration_synchronized; phase_boundaries_may_be_stream_ordered",
        "process_isolation": "one_process_per_backend",
        "isaacsim_owner": {
            "test_fixture_requested": args.isaacsim_test_fixture,
            "production_owner": "src/unilab/conf/flashsac/task/g1_motion_tracking/isaacsim.yaml",
        },
        "acceptance_mode": args.acceptance,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "versions": {
            "installed_unisim_core_metadata": _package_version("unisim-core"),
            "mujoco_warp": _package_version("mujoco_warp"),
            "mujoco": _package_version("mujoco"),
            "warp_lang": _package_version("warp-lang"),
            "newton": _package_version("newton"),
            "mjbatch_uni": _package_version("mjbatch-uni"),
            "motrixsim_core": _package_version("motrixsim-core"),
            "superdex_physics_uni": _package_version("superdex-physics-uni"),
            "superdex_robotics_uni": _package_version("superdex-robotics-uni"),
            "drake": _package_version("drake"),
            "genesis_world": _package_version("genesis-world"),
        },
        "unisim_source": _package_source_info("unisim"),
        "unilab_source": _git_source_info(ROOT_DIR),
        "external_worker_runtimes": _external_worker_runtime_provenance(backends),
        "local_dependencies": {
            "unilab_rl": _package_source_info("uni_rl"),
            "mjbatch_uni": _git_source_info(ROOT_DIR.parent / "mjbatch_uni"),
        },
        "runtime": {
            "torch_cuda": torch.version.cuda,
            "torch_cudnn": torch.backends.cudnn.version(),
            "gpu_capability": ".".join(map(str, torch.cuda.get_device_capability(cuda_index))),
            "gpu_total_memory_gib": (
                torch.cuda.get_device_properties(cuda_index).total_memory / 1024**3
            ),
            **_nvidia_smi_fields(),
            "nvidia_smi_snapshot": _nvidia_smi_snapshot(),
        },
        "invocation": {
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
        },
        "environment": _benchmark_environment(),
        "cuda_device": torch.cuda.get_device_name(cuda_index),
        "cuda_device_index": cuda_index,
        "results": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if not args._quiet:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
