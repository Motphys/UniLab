"""Side-effect-free process and host evidence helpers for Issue #705 benchmarks.

This module must remain independent of task benchmark modules.  In particular,
it must never import ``benchmark.env.benchmark_env_step`` because that legacy
benchmark installs a benchmark-only ``mjwarp`` factory patch at import time.
Production training and profiler evidence must resolve the registered backend.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import platform
import socket
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import psutil
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from unilab.tools.g1_baseline_provenance import (
    G1BaselinePlan,
    load_g1_baseline_plan,
    resolve_benchmark_affinity,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def load_plan(path: Path, *, repo_root: Path) -> G1BaselinePlan:
    absolute = path if path.is_absolute() else repo_root / path
    plan = load_g1_baseline_plan(absolute)
    return dataclasses.replace(plan, source_path=absolute.relative_to(repo_root))


def event_scalars(run_dir: Path, tags: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
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


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


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


def run_subprocess(
    command: list[str],
    plan: G1BaselinePlan,
    *,
    repo_root: Path,
    memory_poll_interval: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], str, str]:
    child_env = os.environ.copy()
    child_env.update(dict(plan.environment.env_vars))
    started_at = utc_now()
    started = time.perf_counter()
    memory_samples: list[dict[str, Any]] = []

    affinity = list(resolve_benchmark_affinity(plan))

    def set_child_affinity() -> None:
        os.sched_setaffinity(0, set(affinity))

    with tempfile.TemporaryDirectory(prefix="unilab_issue705_process_") as temp_dir:
        stdout_path = Path(temp_dir) / "stdout.log"
        stderr_path = Path(temp_dir) / "stderr.log"
        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            process = subprocess.Popen(
                command,
                cwd=repo_root,
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
        "affinity_cpus": affinity,
        "env_vars": dict(plan.environment.env_vars),
        "stdout_sha256": _sha256_bytes(stdout.encode("utf-8")),
        "stderr_sha256": _sha256_bytes(stderr.encode("utf-8")),
    }
    return record, memory_samples, stdout, stderr


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


def preflight_payload(
    plan: G1BaselinePlan,
    *,
    enforce_cpu_load: bool = True,
) -> dict[str, Any]:
    load_average = float(os.getloadavg()[0])
    physical_cores = int(psutil.cpu_count(logical=False) or 0)
    if physical_cores <= 0:
        raise RuntimeError("preflight could not determine the host physical CPU count")
    load_per_core = load_average / physical_cores
    processes = _gpu_compute_processes()
    samples: list[dict[str, Any]] = []
    for index in range(plan.preflight.gpu_samples):
        samples.append(_gpu_idle_sample())
        if index + 1 < plan.preflight.gpu_samples:
            time.sleep(plan.preflight.sample_interval_sec)
    payload = {
        "timestamp": utc_now(),
        "load_average_1m": load_average,
        "load_per_physical_core": load_per_core,
        "gpu_compute_processes": processes,
        "gpu_samples": samples,
    }
    if enforce_cpu_load and load_per_core > plan.preflight.max_load_per_physical_core:
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


def hardware_payload(plan: G1BaselinePlan) -> dict[str, Any]:
    return {
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "cpu_model": _cpu_model(),
        "cpu_physical_cores": int(psutil.cpu_count(logical=False) or 0),
        "cpu_logical_cores": int(psutil.cpu_count(logical=True) or 0),
        "affinity_cpus": list(resolve_benchmark_affinity(plan)),
        **_nvidia_hardware(),
        "cuda_runtime": str(
            getattr(getattr(torch, "version", None), "cuda", None) or "unavailable"
        ),
        "torch_version": str(torch.__version__),
        "hostname": socket.gethostname(),
    }
