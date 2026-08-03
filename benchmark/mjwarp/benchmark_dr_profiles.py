"""Capture the frozen Issue #829 mjwarp domain-randomization matrix.

The controller always expands the complete 300-case plan and launches every
case in a fresh process. Direct reset/env workers use the public Hydra owner,
backend adapter, environment registry, and device RSL-RL wrapper. Train cases
invoke the production trainer without a benchmark-only runner path.

Usage:
    uv run benchmark/mjwarp/benchmark_dr_profiles.py --list-cases
    uv run benchmark/mjwarp/benchmark_dr_profiles.py --execute
    uv run benchmark/mjwarp/benchmark_dr_profiles.py \
        --worker --case-id reset-b128-d0p0000-disabled-r0 \
        --worker-output /tmp/mjwarp-dr-worker.json
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark.mjwarp.process_dr_evidence import event_scalars, json_safe, utc_now
from unilab.tools.g1_baseline_provenance import (
    canonical_sha256,
    sha256_file,
    source_tree_sha256_at_commit,
)
from unilab.tools.mjwarp_dr_performance import (
    ARTIFACT_KIND,
    BENCHMARK_ID,
    DEFAULT_ARTIFACT_PATH,
    FREEZE_RECEIPT_PATH,
    ISSUE,
    PARENT_ISSUE,
    PLAN_PATH,
    RESET_PHASES,
    SCHEMA_VERSION,
    SOURCE_INPUTS,
    TRAIN_SCALAR_TAGS,
    DrPerformanceCaseSpec,
    DrPerformanceProfile,
    MjwarpDrPerformancePlan,
    compact_mjwarp_dr_performance_artifact,
    dependency_version_satisfies,
    expected_mjwarp_dr_performance_cases,
    load_mjwarp_dr_performance_freeze_receipt,
    load_mjwarp_dr_performance_plan,
    recompute_mjwarp_dr_performance_evidence,
    summarize_mjwarp_dr_performance_case,
    validate_mjwarp_dr_performance_artifact,
)


class MjwarpDrBenchmarkError(RuntimeError):
    """Raised when capture cannot satisfy the frozen evidence protocol."""


_PROFILE_FLAGS = (
    "randomize_kp",
    "randomize_kd",
    "randomize_dof_armature",
    "randomize_body_gravity_compensation",
)


def profile_domain_rand_flags(profile_id: str) -> dict[str, bool]:
    """Return the complete, explicit G1 DR identity for one frozen profile."""

    flags: dict[str, bool] = {name: False for name in _PROFILE_FLAGS}
    if profile_id == "tier_b_pd":
        flags["randomize_kp"] = True
        flags["randomize_kd"] = True
    elif profile_id == "tier_c_armature":
        flags["randomize_dof_armature"] = True
    elif profile_id == "tier_c_mixed":
        flags["randomize_dof_armature"] = True
        flags["randomize_body_gravity_compensation"] = True
    elif profile_id != "disabled":
        raise MjwarpDrBenchmarkError(f"unknown mjwarp DR profile {profile_id!r}")
    return flags


def _profile_overrides(profile_id: str) -> list[str]:
    return [
        f"env.domain_rand.{name}={str(enabled).lower()}"
        for name, enabled in profile_domain_rand_flags(profile_id).items()
    ]


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT_DIR,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _git_file_sha256(commit: str, path: str) -> str:
    return _sha256_bytes(_git("show", f"{commit}:{path}").stdout)


def _current_rss_bytes() -> int:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        fields = Path("/proc/self/statm").read_text(encoding="utf-8").split()
        return int(fields[1]) * page_size
    except (IndexError, OSError, TypeError, ValueError) as exc:
        raise MjwarpDrBenchmarkError("could not sample Linux process RSS") from exc


def _memory_sample(torch_module: Any, device: Any) -> tuple[int, int, int]:
    return (
        _current_rss_bytes(),
        int(torch_module.cuda.memory_allocated(device)),
        int(torch_module.cuda.memory_reserved(device)),
    )


def build_memory_windows(
    samples: Sequence[tuple[int, int, int]], *, windows: int
) -> list[dict[str, Any]]:
    """Partition ordered post-warmup samples without dropping any observation."""

    if not samples or windows <= 0 or len(samples) % windows:
        raise MjwarpDrBenchmarkError("memory samples do not divide into frozen windows")
    width = len(samples) // windows
    return [
        {
            "window_index": index,
            "rss_samples_bytes": [
                sample[0] for sample in samples[index * width : (index + 1) * width]
            ],
            "cuda_allocated_samples_bytes": [
                sample[1] for sample in samples[index * width : (index + 1) * width]
            ],
            "cuda_reserved_samples_bytes": [
                sample[2] for sample in samples[index * width : (index + 1) * width]
            ],
        }
        for index in range(windows)
    ]


def _event_traffic_payload(runtime: Any) -> dict[str, Any]:
    return {
        term_key: dataclasses.asdict(diagnostics)
        for term_key, diagnostics in runtime.event_traffic_diagnostics
    }


def _direct_diagnostic_snapshot(wrapper: Any) -> dict[str, Any]:
    stability = wrapper.runtime.stability_diagnostics
    if stability is None:
        raise MjwarpDrBenchmarkError("direct worker lacks stability diagnostics")
    return {
        "performance": dataclasses.asdict(wrapper.runtime.capture_performance_diagnostics()),
        "stability": dataclasses.asdict(stability),
        "runtime_traffic": dataclasses.asdict(wrapper.runtime.traffic_diagnostics),
        "event_traffic": _event_traffic_payload(wrapper.runtime),
        "wrapper_traffic": dataclasses.asdict(wrapper.traffic_diagnostics),
    }


def _direct_diagnostics(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "performance_before": before["performance"],
        "performance_after": after["performance"],
        "stability_before": before["stability"],
        "stability_after": after["stability"],
        "runtime_traffic_before": before["runtime_traffic"],
        "runtime_traffic_after": after["runtime_traffic"],
        "event_traffic_before": before["event_traffic"],
        "event_traffic_after": after["event_traffic"],
        "wrapper_traffic_before": before["wrapper_traffic"],
        "wrapper_traffic_after": after["wrapper_traffic"],
    }


def _counter_delta(after: Mapping[str, Any], before: Mapping[str, Any], key: str) -> int:
    result = int(after[key]) - int(before[key])
    if result < 0:
        raise MjwarpDrBenchmarkError(f"diagnostic counter {key!r} regressed")
    return result


def _event_traffic_delta(
    after: Mapping[str, Any], before: Mapping[str, Any]
) -> dict[str, dict[str, int]]:
    if tuple(after) != tuple(before):
        raise MjwarpDrBenchmarkError("Event traffic terms changed during profiler scope")
    return {
        term: {
            key: _counter_delta(
                cast(Mapping[str, Any], after[term]),
                cast(Mapping[str, Any], before[term]),
                key,
            )
            for key in (
                "host_to_device_transfers",
                "device_to_host_transfers",
                "global_synchronizations",
                "sample_allocations",
            )
        }
        for term in after
    }


def _events_in_scope(trace: Mapping[str, Any], scope_name: str) -> list[dict[str, Any]]:
    raw_events = trace.get("traceEvents")
    if not isinstance(raw_events, list):
        raise MjwarpDrBenchmarkError("PyTorch profiler trace has no traceEvents")
    scopes = [
        event
        for event in raw_events
        if isinstance(event, Mapping)
        and event.get("name") == scope_name
        and event.get("cat") == "user_annotation"
        and isinstance(event.get("ts"), (int, float))
        and isinstance(event.get("dur"), (int, float))
    ]
    if len(scopes) != 1:
        raise MjwarpDrBenchmarkError(
            f"profiler expected one {scope_name!r} annotation, got {len(scopes)}"
        )
    start = float(scopes[0]["ts"])
    end = start + float(scopes[0]["dur"])
    events: list[dict[str, Any]] = []
    for event in raw_events:
        if not isinstance(event, Mapping):
            continue
        timestamp = event.get("ts")
        duration = event.get("dur")
        name = event.get("name")
        if (
            not isinstance(timestamp, (int, float))
            or not isinstance(duration, (int, float))
            or not isinstance(name, str)
            or not name
            or not start <= float(timestamp) <= end
        ):
            continue
        events.append(
            {
                "name": name,
                "category": str(event.get("cat") or "uncategorized"),
                "timestamp_us": float(timestamp),
                "duration_us": max(0.0, float(duration)),
                "args": json_safe(event.get("args", {})),
            }
        )
    if not events:
        raise MjwarpDrBenchmarkError("profiler scope contains no duration events")
    return events


def _capture_representative_profiler(
    wrapper: Any,
    *,
    spec: DrPerformanceCaseSpec,
    actions: Any,
    reset_schedule: Any | None,
) -> dict[str, Any]:
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    scope_name = f"issue829.mjwarp_dr.{spec.lane}"
    runtime_before = dataclasses.asdict(wrapper.runtime.traffic_diagnostics)
    events_before = _event_traffic_payload(wrapper.runtime)
    with tempfile.TemporaryDirectory(prefix="unilab_issue829_profiler_") as temp_dir:
        trace_path = Path(temp_dir) / "trace.json"
        with profile(activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA)) as profiler:
            with record_function(scope_name):
                if reset_schedule is not None:
                    wrapper.episode_length_buf = reset_schedule
                wrapper.step(actions)
            wrapper.last_transition.completion.event.synchronize()
        profiler.export_chrome_trace(str(trace_path))
        trace = cast(
            Mapping[str, Any],
            json.loads(trace_path.read_text(encoding="utf-8")),
        )
        events = _events_in_scope(trace, scope_name)
    runtime_after = dataclasses.asdict(wrapper.runtime.traffic_diagnostics)
    events_after = _event_traffic_payload(wrapper.runtime)
    runtime_delta = {
        key: _counter_delta(runtime_after, runtime_before, key)
        for key in (
            "host_to_device_transfers",
            "device_to_host_transfers",
            "global_synchronizations",
            "backend_allocations",
        )
    }
    return {
        "scope_name": scope_name,
        "coverage_lanes": ["reset"] if spec.lane == "reset" else ["env", "train"],
        "events": events,
        "runtime_delta": runtime_delta,
        "event_traffic_delta": _event_traffic_delta(events_after, events_before),
    }


def _configure_worker(plan: MjwarpDrPerformancePlan) -> None:
    hardware = cast(Mapping[str, Any], plan.data["hardware"])
    env_vars = cast(Mapping[str, str], hardware["environment_variables"])
    for key, value in env_vars.items():
        os.environ[key] = value
    affinity = set(_runtime_affinity(plan))
    os.sched_setaffinity(0, affinity)
    if os.sched_getaffinity(0) != affinity:
        raise MjwarpDrBenchmarkError("worker CPU affinity differs from the selected runtime set")


def _compose_direct_owner(
    spec: DrPerformanceCaseSpec,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from unilab.training import BackendAdapter, ensure_registries

    ensure_registries()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf/ppo"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=g1_walk_flat/mjwarp",
                f"algo.num_envs={spec.batch_size}",
                f"algo.seed={spec.seed}",
                *_profile_overrides(spec.profile_id),
                "training.no_play=true",
                "hydra.run.dir=.",
                "hydra.output_subdir=null",
                "hydra/job_logging=disabled",
                "hydra/hydra_logging=disabled",
            ],
        )
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise MjwarpDrBenchmarkError("Hydra owner did not resolve to a mapping")
    env_override = BackendAdapter(
        cfg, root_dir=ROOT_DIR, algo_name="ppo"
    ).build_task_env_cfg_override()
    return cfg, cast(dict[str, Any], resolved), env_override


def _reset_schedule(
    spec: DrPerformanceCaseSpec,
    *,
    sample_count: int,
    max_episode_length: int,
    torch_module: Any,
    device: Any,
) -> tuple[Any, Any]:
    if spec.reset_density is None:
        raise MjwarpDrBenchmarkError("reset schedule requires a density")
    row_count = int(round(spec.batch_size * spec.reset_density))
    rng = np.random.default_rng(
        spec.seed * 1_000_003 + spec.batch_size * 101 + int(round(spec.reset_density * 10_000))
    )
    masks = np.zeros((sample_count, spec.batch_size), dtype=np.bool_)
    if row_count:
        for index in range(sample_count):
            rows = rng.choice(spec.batch_size, size=row_count, replace=False)
            masks[index, rows] = True
    values = np.zeros((sample_count, spec.batch_size), dtype=np.int64)
    values[masks] = max_episode_length - 1
    return (
        torch_module.as_tensor(values, dtype=torch_module.int64, device=device),
        torch_module.as_tensor(masks, dtype=torch_module.bool, device=device),
    )


def _encode_reset_masks(value: np.ndarray) -> dict[str, Any]:
    packed = np.packbits(value, axis=1, bitorder="little")
    return {
        "encoding": "numpy-packbits-base64-little-v1",
        "shape": list(value.shape),
        "data": base64.b64encode(packed.tobytes(order="C")).decode("ascii"),
    }


def _run_reset_worker(
    plan: MjwarpDrPerformancePlan,
    spec: DrPerformanceCaseSpec,
    cfg: Any,
    resolved_config: dict[str, Any],
    env_override: dict[str, Any],
) -> dict[str, Any]:
    import torch

    from unilab.training import apply_configured_training_seed, create_env
    from unilab.training.rsl_rl_device import DeviceRslRlVecEnvWrapper

    lane = cast(Mapping[str, Any], cast(Mapping[str, Any], plan.data["measurement"])["reset"])
    warmup = int(lane["warmup_barriers"])
    measured = int(lane["measured_barriers"])
    apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    env = create_env(cfg, num_envs=spec.batch_size, env_cfg_override=env_override)
    wrapper = DeviceRslRlVecEnvWrapper(
        env,
        device="cuda:0",
        reset_seed=spec.seed,
        enable_stability_diagnostics=True,
    )
    try:
        actions = torch.zeros(
            (wrapper.num_envs, wrapper.num_actions),
            dtype=torch.float32,
            device=wrapper.device,
        )
        schedule, expected_masks = _reset_schedule(
            spec,
            sample_count=warmup + measured + 1,
            max_episode_length=wrapper.max_episode_length,
            torch_module=torch,
            device=wrapper.device,
        )
        mask_history = torch.empty(
            (measured, spec.batch_size), dtype=torch.bool, device=wrapper.device
        )
        for index in range(warmup):
            wrapper.episode_length_buf = schedule[index]
            wrapper.step(actions)

        wrapper.runtime.begin_reset_phase_timing(capacity=measured)
        before = _direct_diagnostic_snapshot(wrapper)
        memory_samples: list[tuple[int, int, int]] = []
        for index in range(measured):
            wrapper.episode_length_buf = schedule[warmup + index]
            _, _, _, extras = wrapper.step(actions)
            mask_history[index].copy_(extras["time_outs"], non_blocking=True)
            memory_samples.append(_memory_sample(torch, wrapper.device))
        after = _direct_diagnostic_snapshot(wrapper)
        timing_trace = wrapper.runtime.materialize_reset_phase_timings()
        actual_masks = mask_history.to(device="cpu", non_blocking=False).numpy()
        expected = (
            expected_masks[warmup : warmup + measured].to(device="cpu", non_blocking=False).numpy()
        )
        if not np.array_equal(actual_masks, expected):
            raise MjwarpDrBenchmarkError("actual reset rows differ from pre-generated schedule")

        profiler = None
        if spec.repeat_index == 0 and spec.batch_size == 1024 and spec.reset_density == 0.1:
            profiler = _capture_representative_profiler(
                wrapper,
                spec=spec,
                actions=actions,
                reset_schedule=schedule[-1],
            )
        trace_payload = cast(dict[str, Any], json_safe(timing_trace))
        phase_samples = {
            phase: [
                float(
                    next(
                        interval["end_ms"] - interval["start_ms"]
                        for interval in sample["intervals"]
                        if interval["phase"] == phase
                    )
                )
                for sample in trace_payload["samples"]
            ]
            for phase in RESET_PHASES
        }
        return {
            "resolved_config": resolved_config,
            "resolved_config_sha256": canonical_sha256(resolved_config),
            "phase_samples_ms": phase_samples,
            "reset_masks": _encode_reset_masks(actual_masks),
            "memory_windows": build_memory_windows(
                memory_samples, windows=int(lane["memory_windows"])
            ),
            "diagnostics": _direct_diagnostics(before, after),
            "timing_lifecycle": trace_payload,
            "profiler": profiler,
        }
    finally:
        wrapper.close()


def _run_env_worker(
    plan: MjwarpDrPerformancePlan,
    spec: DrPerformanceCaseSpec,
    cfg: Any,
    resolved_config: dict[str, Any],
    env_override: dict[str, Any],
) -> dict[str, Any]:
    import torch

    from unilab.training import apply_configured_training_seed, create_env
    from unilab.training.rsl_rl_device import DeviceRslRlVecEnvWrapper

    lane = cast(Mapping[str, Any], cast(Mapping[str, Any], plan.data["measurement"])["env"])
    warmup = int(lane["warmup_steps"])
    measured = int(lane["measured_steps"])
    apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    env = create_env(cfg, num_envs=spec.batch_size, env_cfg_override=env_override)
    wrapper = DeviceRslRlVecEnvWrapper(
        env,
        device="cuda:0",
        reset_seed=spec.seed,
        enable_stability_diagnostics=True,
    )
    try:
        actions = torch.zeros(
            (wrapper.num_envs, wrapper.num_actions),
            dtype=torch.float32,
            device=wrapper.device,
        )
        for _ in range(warmup):
            wrapper.step(actions)
        event_pairs = tuple(
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(measured)
        )
        stream = torch.cuda.current_stream(wrapper.device)
        for start, end in event_pairs:
            start.record(stream)
            end.record(stream)
        event_pairs[-1][1].synchronize()

        before = _direct_diagnostic_snapshot(wrapper)
        memory_samples: list[tuple[int, int, int]] = []
        for start, end in event_pairs:
            start.record(stream)
            wrapper.step(actions)
            end.record(stream)
            memory_samples.append(_memory_sample(torch, wrapper.device))
        after = _direct_diagnostic_snapshot(wrapper)
        event_pairs[-1][1].synchronize()
        samples = [float(start.elapsed_time(end)) for start, end in event_pairs]

        profiler = None
        if spec.repeat_index == 0 and spec.batch_size == 1024:
            profiler = _capture_representative_profiler(
                wrapper,
                spec=spec,
                actions=actions,
                reset_schedule=None,
            )
        return {
            "resolved_config": resolved_config,
            "resolved_config_sha256": canonical_sha256(resolved_config),
            "env_step_samples_ms": samples,
            "memory_windows": build_memory_windows(
                memory_samples, windows=int(lane["memory_windows"])
            ),
            "diagnostics": _direct_diagnostics(before, after),
            "timing_lifecycle": {
                "capacity": measured,
                "events_preallocated": measured * 2,
                "priming_synchronizations": 1,
                "materialization_synchronizations": 1,
            },
            "profiler": profiler,
        }
    finally:
        wrapper.close()


def run_direct_worker(plan: MjwarpDrPerformancePlan, spec: DrPerformanceCaseSpec) -> dict[str, Any]:
    """Run exactly one reset/env case through public production contracts."""

    if spec.lane not in {"reset", "env"}:
        raise MjwarpDrBenchmarkError("direct worker only owns reset and env lanes")
    _configure_worker(plan)
    cfg, resolved_config, env_override = _compose_direct_owner(spec)
    if spec.lane == "reset":
        raw = _run_reset_worker(plan, spec, cfg, resolved_config, env_override)
    else:
        raw = _run_env_worker(plan, spec, cfg, resolved_config, env_override)
    summary = summarize_mjwarp_dr_performance_case(raw, spec=spec, plan=plan)
    return {"case_id": spec.case_id, "raw": raw, "summary": summary}


def _process_env(plan: MjwarpDrPerformancePlan) -> dict[str, str]:
    hardware = cast(Mapping[str, Any], plan.data["hardware"])
    return {
        str(key): str(value)
        for key, value in cast(Mapping[str, Any], hardware["environment_variables"]).items()
    }


def _run_process(
    command: list[str], plan: MjwarpDrPerformancePlan
) -> tuple[dict[str, Any], str, str]:
    affinity = _runtime_affinity(plan)
    env_vars = _process_env(plan)
    child_env = os.environ.copy()
    child_env.update(env_vars)
    started_at = utc_now()
    started = time.perf_counter()

    def set_affinity() -> None:
        os.sched_setaffinity(0, set(affinity))

    with tempfile.TemporaryDirectory(prefix="unilab_issue829_process_") as temp_dir:
        stdout_path = Path(temp_dir) / "stdout.log"
        stderr_path = Path(temp_dir) / "stderr.log"
        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            process = subprocess.Popen(
                command,
                cwd=ROOT_DIR,
                env=child_env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                preexec_fn=set_affinity,
            )
            return_code = process.wait()
        stdout_bytes = stdout_path.read_bytes()
        stderr_bytes = stderr_path.read_bytes()
    receipt = {
        "run_id": str(uuid.uuid4()),
        "pid": int(process.pid),
        "started_at": started_at,
        "duration_sec": time.perf_counter() - started,
        "return_code": int(return_code),
        "command": command,
        "affinity_cpus": affinity,
        "env_vars": env_vars,
        "stdout_sha256": _sha256_bytes(stdout_bytes),
        "stderr_sha256": _sha256_bytes(stderr_bytes),
    }
    return (
        receipt,
        stdout_bytes.decode("utf-8", errors="replace"),
        stderr_bytes.decode("utf-8", errors="replace"),
    )


def _case_identity(spec: DrPerformanceCaseSpec) -> dict[str, Any]:
    return {
        "case_id": spec.case_id,
        "ordinal": spec.ordinal,
        "lane": spec.lane,
        "profile_id": spec.profile_id,
        "tier": spec.tier,
        "batch_size": spec.batch_size,
        "reset_density": spec.reset_density,
        "repeat_index": spec.repeat_index,
        "seed": spec.seed,
    }


def _worker_failure(
    spec: DrPerformanceCaseSpec, command: Sequence[str], stdout: str, stderr: str
) -> MjwarpDrBenchmarkError:
    return MjwarpDrBenchmarkError(
        f"worker {spec.case_id} failed: {list(command)!r}\n"
        f"stdout tail:\n{stdout[-4000:]}\nstderr tail:\n{stderr[-4000:]}"
    )


def _run_direct_case(plan: MjwarpDrPerformancePlan, spec: DrPerformanceCaseSpec) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="unilab_issue829_worker_") as temp_dir:
        worker_output = Path(temp_dir) / "worker.json"
        command = [
            "uv",
            "run",
            "benchmark/mjwarp/benchmark_dr_profiles.py",
            "--worker",
            "--case-id",
            spec.case_id,
            "--worker-output",
            str(worker_output),
        ]
        process, stdout, stderr = _run_process(command, plan)
        if process["return_code"] != 0 or not worker_output.is_file():
            raise _worker_failure(spec, command, stdout, stderr)
        payload = json.loads(worker_output.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("case_id") != spec.case_id:
        raise MjwarpDrBenchmarkError(f"worker {spec.case_id} returned a foreign payload")
    raw = payload.get("raw")
    if not isinstance(raw, Mapping):
        raise MjwarpDrBenchmarkError(f"worker {spec.case_id} returned no raw evidence")
    computed = summarize_mjwarp_dr_performance_case(raw, spec=spec, plan=plan)
    if payload.get("summary") != computed:
        raise MjwarpDrBenchmarkError(f"worker {spec.case_id} summary is not reproducible")
    return {**_case_identity(spec), "process": process, "raw": raw, "summary": computed}


def _find_train_run(log_root: Path) -> Path:
    task_root = log_root / "G1WalkFlat"
    runs = sorted(path for path in task_root.glob("*_mjwarp") if path.is_dir())
    if len(runs) != 1:
        raise MjwarpDrBenchmarkError(
            f"production trainer created {len(runs)} run directories under {task_root}"
        )
    return runs[0]


def _train_memory_windows(run_summary: Mapping[str, Any], *, warmup: int) -> list[dict[str, Any]]:
    samples = run_summary.get("iteration_memory_diagnostics")
    if not isinstance(samples, list):
        raise MjwarpDrBenchmarkError("trainer run summary lacks iteration memory diagnostics")
    post_warmup = samples[warmup:]
    tuples = [
        (
            int(cast(Mapping[str, Any], sample)["rss_bytes"]),
            int(cast(Mapping[str, Any], sample)["cuda_allocated_bytes"]),
            int(cast(Mapping[str, Any], sample)["cuda_reserved_bytes"]),
        )
        for sample in post_warmup
    ]
    return build_memory_windows(tuples, windows=4)


def _train_command(
    plan: MjwarpDrPerformancePlan,
    spec: DrPerformanceCaseSpec,
    *,
    log_root: Path,
    hydra_root: Path,
) -> list[str]:
    lane = cast(Mapping[str, Any], cast(Mapping[str, Any], plan.data["measurement"])["train"])
    return [
        "uv",
        "run",
        "scripts/train_rsl_rl.py",
        "task=g1_walk_flat/mjwarp",
        f"algo.seed={spec.seed}",
        f"algo.num_envs={spec.batch_size}",
        f"algo.num_steps_per_env={int(lane['num_steps_per_env'])}",
        f"algo.max_iterations={int(lane['iterations'])}",
        "algo.capture_performance_diagnostics=true",
        "training.no_play=true",
        "training.logger=tensorboard",
        f"training.log_root={log_root}",
        *_profile_overrides(spec.profile_id),
        f"hydra.run.dir={hydra_root}",
        "hydra.output_subdir=null",
    ]


def _run_train_case(plan: MjwarpDrPerformancePlan, spec: DrPerformanceCaseSpec) -> dict[str, Any]:
    lane = cast(Mapping[str, Any], cast(Mapping[str, Any], plan.data["measurement"])["train"])
    with tempfile.TemporaryDirectory(prefix="unilab_issue829_train_") as temp_dir:
        root = Path(temp_dir)
        log_root = root / "logs"
        hydra_root = root / "hydra"
        command = _train_command(
            plan,
            spec,
            log_root=log_root,
            hydra_root=hydra_root,
        )
        process, stdout, stderr = _run_process(command, plan)
        if process["return_code"] != 0:
            raise _worker_failure(spec, command, stdout, stderr)
        run_dir = _find_train_run(log_root)
        run_config_path = run_dir / "run_config.json"
        run_summary_path = run_dir / "run_summary.json"
        if not run_config_path.is_file() or not run_summary_path.is_file():
            raise MjwarpDrBenchmarkError(
                f"trainer {spec.case_id} did not emit run_config.json and run_summary.json"
            )
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
        scalars = event_scalars(run_dir, TRAIN_SCALAR_TAGS)
        raw = {
            "scalars": scalars,
            "memory_windows": _train_memory_windows(
                cast(Mapping[str, Any], run_summary),
                warmup=int(lane["warmup_iterations"]),
            ),
            "run_config": run_config,
            "run_config_sha256": canonical_sha256(run_config),
            "run_summary": run_summary,
        }
        summary = summarize_mjwarp_dr_performance_case(raw, spec=spec, plan=plan)
    return {**_case_identity(spec), "process": process, "raw": raw, "summary": summary}


def _gpu_compute_processes() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    processes: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        pid, process_name, memory = [item.strip() for item in line.split(",", maxsplit=2)]
        processes.append(
            {
                "pid": int(pid),
                "process_name": process_name,
                "used_memory_mib": int(memory),
            }
        )
    return processes


def _preflight() -> dict[str, Any]:
    processes = _gpu_compute_processes()
    if processes:
        raise MjwarpDrBenchmarkError(f"foreign GPU compute processes are present: {processes!r}")
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,temperature.gpu,pstate",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    utilization, memory, temperature, pstate = [
        value.strip() for value in completed.stdout.strip().splitlines()[0].split(",")
    ]
    return {
        "timestamp": utc_now(),
        "gpu_compute_processes": processes,
        "gpu_sample": {
            "utilization_percent": int(utilization),
            "memory_used_mib": int(memory),
            "temperature_c": int(temperature),
            "pstate": pstate,
        },
    }


def _cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            return line.split(":", maxsplit=1)[1].strip()
    return platform.processor() or "unknown"


def _runtime_affinity(plan: MjwarpDrPerformancePlan) -> list[int]:
    available = set(os.sched_getaffinity(0))
    if not available:
        raise MjwarpDrBenchmarkError("benchmark requires at least one available CPU")
    frozen = cast(Mapping[str, Any], plan.data["hardware"])
    preferred = {int(value) for value in cast(list[int], frozen["affinity_cpus"])}
    return sorted(preferred if preferred.issubset(available) else available)


def _hardware_payload(plan: MjwarpDrPerformancePlan) -> dict[str, Any]:
    import torch

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    name, uuid_value, memory, driver = [
        value.strip() for value in completed.stdout.strip().splitlines()[0].split(",")
    ]
    frozen = cast(Mapping[str, Any], plan.data["hardware"])
    actual = {
        "hostname": socket.gethostname(),
        "cpu_model": _cpu_model(),
        "affinity_cpus": _runtime_affinity(plan),
        "gpu_name": name,
        "gpu_uuid": uuid_value,
        "gpu_memory_mib": int(memory),
        "driver_version": driver,
        "cuda_runtime": str(
            getattr(getattr(torch, "version", None), "cuda", None) or "unavailable"
        ),
        "environment_variables": frozen["environment_variables"],
    }
    return dict(actual)


def _dependencies_payload(plan: MjwarpDrPerformancePlan) -> dict[str, Any]:
    frozen = cast(Mapping[str, Any], plan.data["dependencies"])
    packages = cast(Mapping[str, str], frozen["packages"])
    resolved: dict[str, dict[str, str]] = {}
    for name, constraint in packages.items():
        version = importlib.metadata.version(name)
        try:
            satisfies = dependency_version_satisfies(version, constraint)
        except ValueError as exc:
            raise MjwarpDrBenchmarkError(
                f"cannot validate frozen dependency {name!r}: {exc}"
            ) from exc
        if not satisfies:
            raise MjwarpDrBenchmarkError(
                f"dependency {name!r} version {version!r} does not satisfy {constraint!r}"
            )
        resolved[name] = {"constraint": constraint, "version": version}
    return {
        "lockfile": frozen["lockfile"],
        "packages": resolved,
    }


def _source_payload(commit: str) -> dict[str, Any]:
    status = _git("status", "--porcelain", "--untracked-files=all").stdout.decode().strip()
    if status:
        raise MjwarpDrBenchmarkError("candidate capture requires a clean Git worktree")
    return {
        "commit": commit,
        "git_status": status,
        "source_inputs": list(SOURCE_INPUTS),
        "source_tree_sha256": source_tree_sha256_at_commit(ROOT_DIR, SOURCE_INPUTS, commit),
        "owner_yaml_sha256": _git_file_sha256(commit, "conf/ppo/task/g1_walk_flat/mjwarp.yaml"),
        "lockfile_sha256": _git_file_sha256(commit, "uv.lock"),
    }


def execute_matrix(*, output: Path, allow_gate_failure: bool = False) -> dict[str, Any]:
    """Execute all 300 isolated processes and write one self-validating artifact."""

    plan = load_mjwarp_dr_performance_plan(ROOT_DIR / PLAN_PATH)
    receipt = load_mjwarp_dr_performance_freeze_receipt(
        ROOT_DIR / FREEZE_RECEIPT_PATH,
        plan=plan,
        repo_root=ROOT_DIR,
    )
    commit = _git("rev-parse", "HEAD").stdout.decode().strip()
    source = _source_payload(commit)
    hardware = _hardware_payload(plan)
    dependencies = _dependencies_payload(plan)
    specs = expected_mjwarp_dr_performance_cases(plan)
    started_at = utc_now()
    preflight_before = _preflight()
    cases: list[dict[str, Any]] = []
    for spec in specs:
        print(f"[{spec.ordinal + 1:03d}/{len(specs):03d}] {spec.case_id}", flush=True)
        case = _run_train_case(plan, spec) if spec.lane == "train" else _run_direct_case(plan, spec)
        cases.append(compact_mjwarp_dr_performance_artifact(case))
    preflight_after = _preflight()
    aggregates, threshold_gate = recompute_mjwarp_dr_performance_evidence(cases, plan=plan)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "benchmark_id": BENCHMARK_ID,
        "issue": ISSUE,
        "parent_issue": PARENT_ISSUE,
        "contract": {
            "plan_path": PLAN_PATH.as_posix(),
            "plan_sha256": plan.plan_sha256,
            "freeze_receipt_path": FREEZE_RECEIPT_PATH.as_posix(),
            "freeze_receipt_sha256": sha256_file(receipt.source_path),
            "freeze_commit": receipt.freeze_commit,
        },
        "source": source,
        "hardware": hardware,
        "dependencies": dependencies,
        "execution": {
            "started_at": started_at,
            "finished_at": utc_now(),
            "preflight_before": preflight_before,
            "preflight_after": preflight_after,
            "case_order": [spec.case_id for spec in specs],
            "outcomes": {
                "completed": len(cases),
                "failed": 0,
                "skipped": 0,
                "xfailed": 0,
                "xpassed": 0,
                "filtered": 0,
            },
        },
        "tier_d_eligibility": plan.data["tier_d_eligibility"],
        "cases": cases,
        "aggregates": aggregates,
        "gate": threshold_gate,
    }
    output = output if output.is_absolute() else ROOT_DIR / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            json_safe(artifact),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )
    errors = validate_mjwarp_dr_performance_artifact(
        cast(Mapping[str, Any], artifact),
        plan=plan,
        receipt=receipt,
        repo_root=ROOT_DIR,
        require_passing_gate=not allow_gate_failure,
    )
    if errors:
        raise MjwarpDrBenchmarkError(
            "captured artifact failed independent validation:\n"
            + "\n".join(f"- {error}" for error in errors)
        )
    return artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list-cases", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--allow-gate-failure", action="store_true")
    parser.add_argument("--case-id")
    parser.add_argument("--worker-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = load_mjwarp_dr_performance_plan(ROOT_DIR / PLAN_PATH)
    specs = expected_mjwarp_dr_performance_cases(plan)
    if args.list_cases:
        for spec in specs:
            print(spec.case_id)
        return 0
    if args.worker:
        if not args.case_id or args.worker_output is None:
            raise MjwarpDrBenchmarkError("worker mode requires --case-id and --worker-output")
        matches = [spec for spec in specs if spec.case_id == args.case_id]
        if len(matches) != 1:
            raise MjwarpDrBenchmarkError(f"unknown case ID {args.case_id!r}")
        payload = run_direct_worker(plan, matches[0])
        output = args.worker_output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")
        return 0
    execute_matrix(output=args.output, allow_gate_failure=args.allow_gate_failure)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MjwarpDrBenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
