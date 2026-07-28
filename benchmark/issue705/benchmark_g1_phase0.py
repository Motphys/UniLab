"""Collect the frozen Issue #705 G1 MuJoCo baseline.

The public execution mode always launches each case in a fresh ``uv run``
process. Internal worker flags exist only so the orchestrator can collect raw
env and reset samples without serializing live environment objects.

Usage:
    uv run benchmark/issue705/benchmark_g1_phase0.py --list-cases
    uv run benchmark/issue705/benchmark_g1_phase0.py --execute
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import psutil
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark.core.mem_profile import build_memory_summary, memory_snapshot
from benchmark.env.benchmark_env_step import _g1_flat_cfg, _g1_walk_env_cls
from unilab.tools.g1_baseline_provenance import (
    G1BaselinePlan,
    assert_clean_affinity,
    build_aggregates,
    canonical_sha256,
    expected_case_ids,
    load_g1_baseline_plan,
    sha256_file,
    source_tree_sha256,
    summarize_dr_raw,
    summarize_env_raw,
    summarize_ppo_raw,
    validate_g1_baseline_artifact,
)

DEFAULT_PLAN = Path("tests/acceptance/issue_705/g1_mujoco_baseline_plan.yaml")
DEFAULT_OUTPUT = Path("tests/acceptance/issue_705/artifacts/g1_mujoco_phase0_baseline.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _plan_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT_DIR / path


def _load_plan(path: Path) -> G1BaselinePlan:
    absolute = _plan_path(path)
    plan = load_g1_baseline_plan(absolute)
    return dataclasses.replace(plan, source_path=absolute.relative_to(ROOT_DIR))


def _configure_process(plan: G1BaselinePlan) -> None:
    for key, value in plan.environment.env_vars:
        os.environ[key] = value
    assert_clean_affinity(plan)
    os.sched_setaffinity(0, set(plan.hardware.affinity_cpus))


def _build_env_config(plan: G1BaselinePlan, dr_mode: str) -> Any:
    cfg = _g1_flat_cfg("mujoco")
    cfg.adaptive_chunk_size = False
    cfg.chunk_size = None
    if dr_mode == "disabled":
        cfg.domain_rand.randomize_kp = False
        cfg.domain_rand.randomize_kd = False
    elif dr_mode != "default_kp_kd":
        raise ValueError(f"unsupported DR mode {dr_mode!r}")
    cfg.validate()
    return cfg


def _run_env_worker(plan: G1BaselinePlan, *, batch_size: int, repeat: int) -> dict[str, Any]:
    _configure_process(plan)
    seed = plan.env_lane.action_seed_base + batch_size * 10 + repeat
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    cfg = _build_env_config(plan, "default_kp_kd")
    config = cast(dict[str, Any], _json_safe(cfg))

    env = None
    timing_records: dict[str, list[float]] = {}
    memory_samples: dict[str, dict[str, Any]] = {}
    try:
        memory_samples["before_env"] = memory_snapshot("before_env")
        env = _g1_walk_env_cls()(cfg, num_envs=batch_size, backend_type="mujoco")
        state = env.init_state()
        action_dim = int(env.action_space.shape[-1])
        for _ in range(plan.env_lane.warmup_steps):
            action = rng.uniform(-1.0, 1.0, size=(batch_size, action_dim)).astype(np.float32)
            state = env.step(action)
        for _ in range(plan.env_lane.measure_steps):
            action = rng.uniform(-1.0, 1.0, size=(batch_size, action_dim)).astype(np.float32)
            state = env.step(action)
            timing = state.info.get("timing", {})
            if not isinstance(timing, dict):
                raise RuntimeError("env step produced no timing mapping")
            for key, value in timing.items():
                timing_records.setdefault(str(key), []).append(float(value))
        memory_samples["after_benchmark"] = memory_snapshot("after_benchmark")
    finally:
        if env is not None:
            env.close()
            memory_samples["after_close"] = memory_snapshot("after_close")
    raw = {
        "timing_records": timing_records,
        "memory": build_memory_summary(memory_samples, batch_size),
        "resolved_env_config": config,
        "resolved_config_sha256": canonical_sha256(config),
    }
    return cast(dict[str, Any], _json_safe(raw))


def _run_dr_worker(
    plan: G1BaselinePlan,
    *,
    mode: str,
    density: float,
    repeat: int,
) -> dict[str, Any]:
    _configure_process(plan)
    density_index = plan.dr_lane.reset_densities.index(density)
    mode_index = plan.dr_lane.modes.index(mode)
    seed = plan.dr_lane.reset_seed_base + mode_index * 1000 + density_index * 100 + repeat
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    cfg = _build_env_config(plan, mode)
    config = cast(dict[str, Any], _json_safe(cfg))
    num_envs = plan.dr_lane.num_envs
    row_count = max(1, round(num_envs * density))

    env = None
    reset_samples: list[dict[str, Any]] = []
    memory_samples: dict[str, dict[str, Any]] = {}
    try:
        memory_samples["before_env"] = memory_snapshot("before_env")
        env = _g1_walk_env_cls()(cfg, num_envs=num_envs, backend_type="mujoco")
        env.init_state()

        def reset_once() -> dict[str, Any]:
            rows = np.sort(rng.choice(num_envs, size=row_count, replace=False)).astype(np.int32)
            started = time.perf_counter()
            obs, _ = env.reset(rows)
            external_ms = (time.perf_counter() - started) * 1000.0
            actual_rows = len(next(iter(obs.values()))) if obs else 0
            timing = env.last_reset_timing_ms
            timing["external_reset_ms"] = external_ms
            return {
                "requested_rows": int(row_count),
                "actual_rows": int(actual_rows),
                "timing": {str(key): float(value) for key, value in sorted(timing.items())},
            }

        for _ in range(plan.dr_lane.warmup_resets):
            reset_once()
        for _ in range(plan.dr_lane.measure_resets):
            reset_samples.append(reset_once())
        memory_samples["after_benchmark"] = memory_snapshot("after_benchmark")
    finally:
        if env is not None:
            env.close()
            memory_samples["after_close"] = memory_snapshot("after_close")
    raw = {
        "reset_samples": reset_samples,
        "memory": build_memory_summary(memory_samples, num_envs),
        "resolved_env_config": config,
        "resolved_config_sha256": canonical_sha256(config),
    }
    return cast(dict[str, Any], _json_safe(raw))


def _process_tree_rss(pid: int) -> int:
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except (psutil.Error, ProcessLookupError):
        return 0
    total = 0
    for process in processes:
        try:
            total += int(process.memory_info().rss)
        except (psutil.Error, ProcessLookupError):
            continue
    return total


def _run_subprocess(
    command: list[str],
    plan: G1BaselinePlan,
    *,
    memory_poll_interval: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], str, str]:
    child_env = os.environ.copy()
    child_env.update(dict(plan.environment.env_vars))
    started_at = _utc_now()
    started = time.perf_counter()
    memory_samples: list[dict[str, Any]] = []

    def set_child_affinity() -> None:
        os.sched_setaffinity(0, set(plan.hardware.affinity_cpus))

    with tempfile.TemporaryDirectory(prefix="unilab_issue705_process_") as temp_dir:
        stdout_path = Path(temp_dir) / "stdout.log"
        stderr_path = Path(temp_dir) / "stderr.log"
        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            process = subprocess.Popen(
                command,
                cwd=ROOT_DIR,
                env=child_env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                preexec_fn=set_child_affinity,
            )
            if memory_poll_interval is None:
                return_code = process.wait()
            else:
                while True:
                    memory_samples.append(
                        {
                            "elapsed_sec": time.perf_counter() - started,
                            "rss_bytes": _process_tree_rss(process.pid),
                        }
                    )
                    try:
                        return_code = process.wait(timeout=memory_poll_interval)
                        break
                    except subprocess.TimeoutExpired:
                        continue
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    record = {
        "run_id": str(uuid.uuid4()),
        "pid": int(process.pid),
        "started_at": started_at,
        "duration_sec": time.perf_counter() - started,
        "return_code": int(return_code),
        "command": command,
        "affinity_cpus": list(plan.hardware.affinity_cpus),
        "env_vars": dict(plan.environment.env_vars),
        "stdout_sha256": _sha256_bytes(stdout.encode("utf-8")),
        "stderr_sha256": _sha256_bytes(stderr.encode("utf-8")),
    }
    return record, memory_samples, stdout, stderr


def _worker_case(
    plan: G1BaselinePlan,
    *,
    lane: str,
    batch_size: int | None = None,
    repeat: int | None = None,
    mode: str | None = None,
    density: float | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="unilab_issue705_worker_") as temp_dir:
        output = Path(temp_dir) / "raw.json"
        command = [
            "uv",
            "run",
            "benchmark/issue705/benchmark_g1_phase0.py",
            "--plan",
            plan.source_path.as_posix(),
            "--worker",
            lane,
            "--worker-output",
            str(output),
        ]
        if lane == "env":
            command.extend(["--batch-size", str(batch_size), "--repeat", str(repeat)])
        else:
            command.extend(
                [
                    "--dr-mode",
                    str(mode),
                    "--reset-density",
                    str(density),
                    "--repeat",
                    str(repeat),
                ]
            )
        process, _, stdout, stderr = _run_subprocess(command, plan)
        if process["return_code"] != 0 or not output.is_file():
            raise RuntimeError(
                f"worker failed: command={command!r}\nstdout:\n{stdout[-4000:]}\nstderr:\n{stderr[-4000:]}"
            )
        raw = json.loads(output.read_text(encoding="utf-8"))

    if lane == "env":
        assert batch_size is not None and repeat is not None
        case_id = f"env-b{batch_size}-r{repeat}"
        summary = summarize_env_raw(raw, batch_size)
        return {
            "case_id": case_id,
            "lane": lane,
            "repeat_index": repeat,
            "seed": None,
            "batch_size": batch_size,
            "dr_mode": None,
            "reset_density": None,
            "process": process,
            "raw": raw,
            "summary": summary,
        }
    assert mode is not None and density is not None and repeat is not None
    density_id = f"{density:.4f}".replace(".", "p")
    return {
        "case_id": f"dr-{mode}-d{density_id}-r{repeat}",
        "lane": lane,
        "repeat_index": repeat,
        "seed": None,
        "batch_size": plan.dr_lane.num_envs,
        "dr_mode": mode,
        "reset_density": density,
        "process": process,
        "raw": raw,
        "summary": summarize_dr_raw(raw),
    }


def _event_scalars(run_dir: Path, tags: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    missing = set(tags) - available
    if missing:
        raise RuntimeError(f"training event file is missing scalar tags: {sorted(missing)!r}")
    return {
        tag: [
            {
                "step": int(event.step),
                "wall_time": float(event.wall_time),
                "value": float(event.value),
            }
            for event in accumulator.Scalars(tag)
        ]
        for tag in tags
    }


def _run_ppo_case(plan: G1BaselinePlan, seed: int) -> dict[str, Any]:
    foreign_processes = _gpu_compute_processes()
    if foreign_processes:
        raise RuntimeError(
            f"PPO seed {seed} cannot start with foreign GPU compute processes: "
            f"{foreign_processes!r}"
        )
    with tempfile.TemporaryDirectory(prefix=f"unilab_issue705_ppo_seed{seed}_") as temp_dir:
        log_root = Path(temp_dir) / "logs"
        command = [
            "uv",
            "run",
            "scripts/train_rsl_rl.py",
            "task=g1_walk_flat/mujoco",
            f"algo.seed={seed}",
            f"algo.num_envs={plan.ppo_lane.num_envs}",
            f"algo.num_steps_per_env={plan.ppo_lane.num_steps_per_env}",
            f"algo.max_iterations={plan.ppo_lane.max_iterations}",
            f"algo.save_interval={plan.ppo_lane.save_interval}",
            "training.no_play=true",
            "training.logger=tensorboard",
            f"training.log_root={log_root}",
            *plan.environment.hydra_overrides,
        ]
        process, memory_samples, stdout, stderr = _run_subprocess(
            command,
            plan,
            memory_poll_interval=plan.ppo_lane.memory_poll_interval_sec,
        )
        if process["return_code"] != 0:
            raise RuntimeError(
                f"PPO seed {seed} failed\nstdout:\n{stdout[-6000:]}\nstderr:\n{stderr[-6000:]}"
            )
        run_dirs = sorted((log_root / "G1WalkFlat").glob("*_mujoco"))
        if len(run_dirs) != 1:
            raise RuntimeError(f"PPO seed {seed} produced {len(run_dirs)} run directories")
        run_dir = run_dirs[0]
        run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        run_summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        scalars = _event_scalars(run_dir, plan.ppo_lane.required_scalar_tags)
    raw = {
        "scalars": scalars,
        "memory_samples": memory_samples,
        "run_config": run_config,
        "run_config_sha256": canonical_sha256(run_config),
        "run_summary": run_summary,
    }
    return {
        "case_id": f"ppo-seed-{seed}",
        "lane": "ppo",
        "repeat_index": None,
        "seed": seed,
        "batch_size": plan.ppo_lane.num_envs,
        "dr_mode": None,
        "reset_density": None,
        "process": process,
        "raw": raw,
        "summary": summarize_ppo_raw(raw, plan.ppo_lane),
    }


def _nvidia_hardware() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    first = completed.stdout.strip().splitlines()[0]
    name, uuid_value, memory, driver = [item.strip() for item in first.split(",")]
    return {
        "gpu_name": name,
        "gpu_uuid": uuid_value,
        "gpu_memory_mib": int(memory),
        "driver_version": driver,
    }


def _gpu_compute_processes() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    processes: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        pid, name, used_memory = [item.strip() for item in line.split(",", maxsplit=2)]
        processes.append(
            {
                "pid": int(pid),
                "process_name": name,
                "used_memory_mib": int(used_memory),
            }
        )
    return processes


def _gpu_idle_sample() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,temperature.gpu,pstate",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    utilization, memory, temperature, pstate = [
        item.strip() for item in completed.stdout.strip().splitlines()[0].split(",")
    ]
    return {
        "utilization_percent": int(utilization),
        "memory_used_mib": int(memory),
        "temperature_c": int(temperature),
        "pstate": pstate,
    }


def _preflight_payload(plan: G1BaselinePlan) -> dict[str, Any]:
    load_average = float(os.getloadavg()[0])
    load_per_core = load_average / plan.hardware.cpu_physical_cores
    processes = _gpu_compute_processes()
    samples: list[dict[str, Any]] = []
    for index in range(plan.preflight.gpu_samples):
        samples.append(_gpu_idle_sample())
        if index + 1 < plan.preflight.gpu_samples:
            time.sleep(plan.preflight.sample_interval_sec)
    payload = {
        "timestamp": _utc_now(),
        "load_average_1m": load_average,
        "load_per_physical_core": load_per_core,
        "gpu_compute_processes": processes,
        "gpu_samples": samples,
    }
    if load_per_core > plan.preflight.max_load_per_physical_core:
        raise RuntimeError(
            f"preflight CPU load {load_per_core:.3f} exceeds "
            f"{plan.preflight.max_load_per_physical_core:.3f} per physical core"
        )
    if len(processes) > plan.preflight.max_gpu_compute_processes:
        raise RuntimeError(f"preflight found foreign GPU compute processes: {processes!r}")
    peak_utilization = max(sample["utilization_percent"] for sample in samples)
    if peak_utilization > plan.preflight.max_gpu_utilization_percent:
        raise RuntimeError(
            f"preflight GPU utilization {peak_utilization}% exceeds "
            f"{plan.preflight.max_gpu_utilization_percent}%"
        )
    return payload


def _cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            return line.split(":", maxsplit=1)[1].strip()
    return platform.processor() or "unknown"


def _hardware_payload(plan: G1BaselinePlan) -> dict[str, Any]:
    payload = {
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "cpu_model": _cpu_model(),
        "cpu_physical_cores": int(psutil.cpu_count(logical=False) or 0),
        "cpu_logical_cores": int(psutil.cpu_count(logical=True) or 0),
        "affinity_cpus": list(plan.hardware.affinity_cpus),
        **_nvidia_hardware(),
        "cuda_runtime": str(
            getattr(getattr(torch, "version", None), "cuda", None) or "unavailable"
        ),
        "torch_version": str(torch.__version__),
        "hostname": socket.gethostname(),
    }
    expected = dataclasses.asdict(plan.hardware)
    for key, value in expected.items():
        expected_value = list(value) if isinstance(value, tuple) else value
        if payload.get(key) != expected_value:
            raise RuntimeError(
                f"hardware mismatch for {key}: expected {expected_value!r}, "
                f"got {payload.get(key)!r}"
            )
    return payload


def _source_payload(plan: G1BaselinePlan) -> dict[str, Any]:
    dirty = bool(_git("status", "--short"))
    if dirty:
        raise RuntimeError("baseline execution requires a clean git worktree")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": dirty,
        "tree_sha256": source_tree_sha256(ROOT_DIR, plan.source_inputs),
        "uv_lock_sha256": sha256_file(ROOT_DIR / "uv.lock"),
        "owner_yaml_sha256": sha256_file(ROOT_DIR / plan.owner_yaml),
    }


def _collect(plan: G1BaselinePlan) -> dict[str, Any]:
    hardware = _hardware_payload(plan)
    source = _source_payload(plan)
    preflight = _preflight_payload(plan)
    cases: list[dict[str, Any]] = []
    total = len(expected_case_ids(plan))

    def append(case: dict[str, Any]) -> None:
        cases.append(case)
        print(f"[{len(cases):02d}/{total:02d}] {case['case_id']} PASS", flush=True)

    for batch_size in plan.env_lane.batch_sizes:
        for repeat in range(plan.env_lane.process_repeats):
            append(
                _worker_case(
                    plan,
                    lane="env",
                    batch_size=batch_size,
                    repeat=repeat,
                )
            )
    for mode in plan.dr_lane.modes:
        for density in plan.dr_lane.reset_densities:
            for repeat in range(plan.dr_lane.process_repeats):
                append(
                    _worker_case(
                        plan,
                        lane="dr",
                        mode=mode,
                        density=density,
                        repeat=repeat,
                    )
                )
    for seed in plan.ppo_lane.seeds:
        append(_run_ppo_case(plan, seed))

    artifact = {
        "schema_version": 1,
        "issue": 705,
        "baseline_id": plan.baseline_id,
        "generated_at": _utc_now(),
        "plan": {
            "path": plan.source_path.as_posix(),
            "sha256": sha256_file(ROOT_DIR / plan.source_path),
        },
        "source": source,
        "hardware": hardware,
        "execution": {
            "process_isolation": True,
            "affinity_cpus": list(plan.hardware.affinity_cpus),
            "env_vars": dict(plan.environment.env_vars),
            "hydra_overrides": list(plan.environment.hydra_overrides),
            "preflight": preflight,
        },
        "cases": cases,
        "aggregates": build_aggregates(cases),
    }
    errors = validate_g1_baseline_artifact(artifact, plan, repo_root=ROOT_DIR)
    if errors:
        detail = "\n".join(f"- {error}" for error in errors)
        raise RuntimeError(f"generated artifact failed validation:\n{detail}")
    return artifact


def _worker_main(args: argparse.Namespace, plan: G1BaselinePlan) -> int:
    if args.worker_output is None:
        raise ValueError("--worker-output is required in worker mode")
    if args.worker == "env":
        if args.batch_size is None or args.repeat is None:
            raise ValueError("env worker requires --batch-size and --repeat")
        raw = _run_env_worker(plan, batch_size=args.batch_size, repeat=args.repeat)
    else:
        if args.dr_mode is None or args.reset_density is None or args.repeat is None:
            raise ValueError("DR worker requires --dr-mode, --reset-density, and --repeat")
        raw = _run_dr_worker(
            plan,
            mode=args.dr_mode,
            density=args.reset_density,
            repeat=args.repeat,
        )
    args.worker_output.parent.mkdir(parents=True, exist_ok=True)
    args.worker_output.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--worker", choices=("env", "dr"))
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--dr-mode")
    parser.add_argument("--reset-density", type=float)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = _load_plan(args.plan)
    if args.worker is not None:
        return _worker_main(args, plan)
    if args.list_cases:
        for case_id in sorted(expected_case_ids(plan)):
            print(case_id)
        return 0
    if not args.execute:
        raise SystemExit("Refusing to run implicitly; pass --execute or --list-cases")
    artifact = _collect(plan)
    output = args.out if args.out.is_absolute() else ROOT_DIR / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    print(f"PASS wrote {output.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
