"""Fail-closed, process-isolated PPO benchmark for Issue #705's ``mjwarp`` profile.

The benchmark intentionally keeps performance collection outside the training
entrypoint.  Every measured PPO run is a fresh invocation of the public
``scripts/train_rsl_rl.py`` owner route, so the resulting evidence covers the
same Hydra composition, runner lifecycle, tracker receipt and checkpoint path
as production training.

There are three deliberately separate measurements:

* an interleaved, isolated ``mujoco``/``mjwarp`` throughput matrix at the
  frozen small/medium/large batch sizes;
* the frozen five-seed, 100-iteration ``mjwarp`` behavior matrix;
* a typed-device rollout profiler trace and a separately reported co-located
  contention matrix.

``--execute`` only accepts a clean committed tree and writes the raw artifact
outside the repository.  A later evidence PR copies the JSON and its sibling
trace unchanged into the acceptance artifact directory.  This prevents an
uncommitted benchmark result from being attributed to a different candidate.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark.issue705.process_evidence import (  # noqa: E402
    event_scalars as _event_scalars,
)
from benchmark.issue705.process_evidence import (
    hardware_payload as _hardware_payload,
)
from benchmark.issue705.process_evidence import (
    json_safe as _json_safe,
)
from benchmark.issue705.process_evidence import (
    load_plan as _load_evidence_plan,
)
from benchmark.issue705.process_evidence import (
    preflight_payload as _preflight_payload,
)
from benchmark.issue705.process_evidence import (
    run_subprocess as _run_evidence_subprocess,
)
from benchmark.issue705.process_evidence import (
    utc_now as _utc_now,
)
from unilab.tools.g1_baseline_provenance import (  # noqa: E402
    canonical_sha256,
    numeric_stats,
    sha256_file,
    source_tree_sha256,
    source_tree_sha256_at_commit,
)
from unilab.tools.issue705_thresholds import (  # noqa: E402
    ThresholdManifest,
    load_amendment_freeze_receipt,
    load_freeze_receipt,
    load_threshold_amendment,
    load_threshold_manifest,
)

ISSUE = 705
SCHEMA_VERSION = 1
ARTIFACT_KIND = "issue705-mjwarp-device-ppo-benchmark-v1"
PROFILE = "device_resident"
DEFAULT_BASELINE_PLAN = Path("tests/acceptance/issue_705/g1_mujoco_baseline_plan.yaml")
DEFAULT_THRESHOLD_MANIFEST = Path("tests/acceptance/issue_705/g1_threshold_manifest.yaml")
DEFAULT_THRESHOLD_RECEIPT = Path("tests/acceptance/issue_705/g1_threshold_freeze_receipt.yaml")
DEFAULT_THRESHOLD_AMENDMENT = Path(
    "tests/acceptance/issue_705/g1_phase5_ppo_threshold_amendment.yaml"
)
DEFAULT_THRESHOLD_AMENDMENT_RECEIPT = Path(
    "tests/acceptance/issue_705/g1_phase5_ppo_threshold_amendment_freeze_receipt.yaml"
)
DEFAULT_OUTPUT = Path("/tmp/unilab_issue705_mjwarp_device_ppo.json")
DEFAULT_TRACE_OUTPUT = Path("/tmp/unilab_issue705_mjwarp_device_ppo_trace.json")
THROUGHPUT_MODES = ("mujoco_host", "mjwarp_device")
THROUGHPUT_ITERATIONS = 20
CONTENTION_ITERATIONS = 20
PROFILE_STEPS = 16
COMMON_PERFORMANCE_OVERRIDES = (
    "env.noise_config.level=0.0",
    # ``mujoco`` intentionally leaves these dataclass defaults out of its
    # owner YAML, whereas the independent ``mjwarp`` owner declares them to
    # fail closed.  Hydra's force-add/override form is the only config-first
    # spelling that produces the same explicit benchmark profile for both
    # owners without changing either production default.
    "++env.domain_rand.randomize_kp=false",
    "++env.domain_rand.randomize_kd=false",
    "env.curriculum.enabled=false",
)
SOURCE_INPUTS = (
    "benchmark/rl/benchmark_mjwarp_ppo.py",
    "benchmark/issue705/process_evidence.py",
    "scripts/train_rsl_rl.py",
    "src/unilab/algos/torch/rsl_rl_ppo.py",
    "src/unilab/algos/torch/rsl_rl_runtime.py",
    "src/unilab/base/backend",
    "src/unilab/base/np_env.py",
    "src/unilab/envs/locomotion/g1",
    "src/unilab/manager",
    "src/unilab/training",
    "src/unilab/tools/g1_baseline_provenance.py",
    "src/unilab/tools/issue705_thresholds.py",
    "conf/ppo/config.yaml",
    "conf/ppo/task/g1_walk_flat/mujoco.yaml",
    "conf/ppo/task/g1_walk_flat/mjwarp.yaml",
    "tests/acceptance/issue_705/g1_mujoco_baseline_plan.yaml",
    "tests/acceptance/issue_705/g1_threshold_manifest.yaml",
    "tests/acceptance/issue_705/g1_threshold_freeze_receipt.yaml",
    "tests/acceptance/issue_705/g1_phase5_ppo_threshold_amendment.yaml",
    "tests/acceptance/issue_705/g1_phase5_ppo_threshold_amendment_freeze_receipt.yaml",
    "docs/sphinx/source/adr/ADR-0006-phase5-ppo-rss-threshold-amendment.md",
    "tests/benchmark/test_mjwarp_ppo_benchmark.py",
    "uv.lock",
)
REQUIRED_SCALAR_TAGS = (
    "Perf/total_fps",
    "Perf/collection_time",
    "Perf/learning_time",
    "Train/mean_reward",
    "Train/mean_episode_length",
)


class MjwarpPpoBenchmarkError(RuntimeError):
    """Raised when a benchmark input, raw case, or artifact is invalid."""


def _load_plan(path: Path) -> Any:
    return _load_evidence_plan(path, repo_root=ROOT_DIR)


def _run_subprocess(
    command: list[str],
    plan: Any,
    *,
    memory_poll_interval: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], str, str]:
    return _run_evidence_subprocess(
        command,
        plan,
        repo_root=ROOT_DIR,
        memory_poll_interval=memory_poll_interval,
    )


@dataclasses.dataclass(frozen=True)
class BenchmarkBinding:
    """Immutable Phase-5 subset of the pre-registered Phase-0 thresholds."""

    threshold_set_id: str
    threshold_manifest_path: Path
    threshold_manifest_sha256: str
    threshold_freeze_commit: str
    amendment_id: str
    amendment_manifest_path: Path
    amendment_manifest_sha256: str
    amendment_freeze_commit: str
    batch_sizes: tuple[int, ...]
    process_repeats: int
    behavior_seeds: tuple[int, ...]
    behavior_num_envs: int
    behavior_steps_per_env: int
    behavior_iterations: int
    warmup_iterations: int
    memory_poll_interval_sec: float
    max_population_cv_by_batch: Mapping[int, float]
    p50_latency_ratio_max: float
    p95_latency_ratio_max: float
    throughput_ratio_min: float
    fps_p50_median_ratio_min: float
    reward_auc_median_drop_max: float
    final_reward_p50_median_drop_max: float
    episode_length_median_ratio_min: float
    host_memory_ratio_max: float
    device_peak_reserved_capacity_ratio_max: float
    device_peak_reserved_growth_bytes_max: int
    h2d_per_policy_step_max: float
    d2h_per_policy_step_max: float
    host_global_sync_per_policy_step_max: float
    baseline_ppo: Mapping[str, Any]
    affinity_cpus: tuple[int, ...]
    environment_vars: Mapping[str, str]
    hydra_overrides: tuple[str, ...]
    gpu_name: str
    gpu_uuid: str
    gpu_driver_version: str
    gpu_capacity_bytes: int


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MjwarpPpoBenchmarkError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MjwarpPpoBenchmarkError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _number(value: object, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MjwarpPpoBenchmarkError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise MjwarpPpoBenchmarkError(f"{label} must be a finite number >= {minimum}")
    return result


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MjwarpPpoBenchmarkError(f"{label} must be a non-empty string")
    return value


def _sha256(value: object, label: str) -> str:
    result = _string(value, label)
    if not (result.startswith("sha256:") and len(result) == len("sha256:") + 64):
        raise MjwarpPpoBenchmarkError(f"{label} must be a sha256:<64 hex> value")
    if any(character not in "0123456789abcdef" for character in result.removeprefix("sha256:")):
        raise MjwarpPpoBenchmarkError(f"{label} must use lowercase hexadecimal")
    return result


def _is_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT_DIR, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def load_binding(
    *,
    threshold_manifest_path: Path = DEFAULT_THRESHOLD_MANIFEST,
    threshold_receipt_path: Path = DEFAULT_THRESHOLD_RECEIPT,
    threshold_amendment_path: Path = DEFAULT_THRESHOLD_AMENDMENT,
    threshold_amendment_receipt_path: Path = DEFAULT_THRESHOLD_AMENDMENT_RECEIPT,
    baseline_plan_path: Path = DEFAULT_BASELINE_PLAN,
) -> BenchmarkBinding:
    """Bind the exact frozen matrix before a benchmark command can be constructed."""

    manifest_path = (
        threshold_manifest_path
        if threshold_manifest_path.is_absolute()
        else ROOT_DIR / threshold_manifest_path
    )
    receipt_path = (
        threshold_receipt_path
        if threshold_receipt_path.is_absolute()
        else ROOT_DIR / threshold_receipt_path
    )
    amendment_path = (
        threshold_amendment_path
        if threshold_amendment_path.is_absolute()
        else ROOT_DIR / threshold_amendment_path
    )
    amendment_receipt_path = (
        threshold_amendment_receipt_path
        if threshold_amendment_receipt_path.is_absolute()
        else ROOT_DIR / threshold_amendment_receipt_path
    )
    plan_path = (
        baseline_plan_path if baseline_plan_path.is_absolute() else ROOT_DIR / baseline_plan_path
    )
    manifest: ThresholdManifest = load_threshold_manifest(manifest_path, repo_root=ROOT_DIR)
    receipt = load_freeze_receipt(
        receipt_path,
        manifest=manifest,
        repo_root=ROOT_DIR,
    )
    amendment = load_threshold_amendment(
        amendment_path,
        base_manifest=manifest,
        base_receipt=receipt,
        repo_root=ROOT_DIR,
    )
    amendment_receipt = load_amendment_freeze_receipt(
        amendment_receipt_path,
        amendment=amendment,
        base_receipt=receipt,
        repo_root=ROOT_DIR,
    )
    data = manifest.data
    measurement = _mapping(data.get("measurement"), "measurement")
    gates = _mapping(data.get("gates"), "gates")
    performance = _mapping(gates.get("performance"), "gates.performance")
    training = _mapping(gates.get("training"), "gates.training")
    memory = _mapping(gates.get("memory"), "gates.memory")
    transfer = _mapping(gates.get("transfer"), "gates.transfer")
    baseline = _mapping(data.get("baseline"), "baseline")
    hardware = _mapping(baseline.get("hardware"), "baseline.hardware")
    reference = _mapping(data.get("baseline_reference"), "baseline_reference")
    baseline_ppo = _mapping(reference.get("ppo"), "baseline_reference.ppo")
    baseline_plan = _load_plan(plan_path)

    manifest_sha = sha256_file(manifest_path)
    freeze_commit = receipt.freeze_commit

    batches_raw = measurement.get("batch_sizes")
    if not isinstance(batches_raw, list):
        raise MjwarpPpoBenchmarkError("measurement.batch_sizes must be a list")
    batch_sizes = tuple(
        _integer(item, f"measurement.batch_sizes[{index}]", minimum=1)
        for index, item in enumerate(batches_raw)
    )
    if batch_sizes != (128, 1024, 4096):
        raise MjwarpPpoBenchmarkError(
            "Phase 5 benchmark batch matrix must remain [128, 1024, 4096]"
        )
    cv_raw = _mapping(performance.get("max_population_cv_by_batch"), "performance CV")
    cv = {batch: _number(cv_raw.get(str(batch)), f"CV[{batch}]") for batch in batch_sizes}
    if baseline_plan.ppo_lane.seeds != (0, 1, 2, 3, 4):
        raise MjwarpPpoBenchmarkError("frozen PPO seed matrix must remain [0, 1, 2, 3, 4]")
    if baseline_plan.ppo_lane.max_iterations != _integer(
        measurement.get("ppo_iterations"), "measurement.ppo_iterations", minimum=2
    ):
        raise MjwarpPpoBenchmarkError("baseline plan and frozen PPO iteration count differ")

    return BenchmarkBinding(
        threshold_set_id=_string(data.get("threshold_set_id"), "threshold_set_id"),
        threshold_manifest_path=threshold_manifest_path,
        threshold_manifest_sha256=manifest_sha,
        threshold_freeze_commit=freeze_commit,
        amendment_id=amendment.amendment_id,
        amendment_manifest_path=threshold_amendment_path,
        amendment_manifest_sha256=sha256_file(amendment_path),
        amendment_freeze_commit=amendment_receipt.freeze_commit,
        batch_sizes=batch_sizes,
        process_repeats=_integer(
            measurement.get("env_process_repeats"), "env_process_repeats", minimum=5
        ),
        behavior_seeds=baseline_plan.ppo_lane.seeds,
        behavior_num_envs=baseline_plan.ppo_lane.num_envs,
        behavior_steps_per_env=baseline_plan.ppo_lane.num_steps_per_env,
        behavior_iterations=baseline_plan.ppo_lane.max_iterations,
        warmup_iterations=baseline_plan.ppo_lane.warmup_iterations,
        memory_poll_interval_sec=baseline_plan.ppo_lane.memory_poll_interval_sec,
        max_population_cv_by_batch=cv,
        p50_latency_ratio_max=_number(performance.get("p50_latency_ratio_max"), "p50 latency gate"),
        p95_latency_ratio_max=_number(performance.get("p95_latency_ratio_max"), "p95 latency gate"),
        throughput_ratio_min=_number(performance.get("throughput_ratio_min"), "throughput gate"),
        fps_p50_median_ratio_min=_number(training.get("fps_p50_median_ratio_min"), "fps gate"),
        reward_auc_median_drop_max=_number(training.get("reward_auc_median_drop_max"), "AUC gate"),
        final_reward_p50_median_drop_max=_number(
            training.get("final_reward_p50_median_drop_max"), "final reward gate"
        ),
        episode_length_median_ratio_min=_number(
            training.get("episode_length_median_ratio_min"), "episode length gate"
        ),
        host_memory_ratio_max=amendment.host_memory_ratio_max,
        device_peak_reserved_capacity_ratio_max=_number(
            memory.get("device_peak_reserved_capacity_ratio_max"), "device capacity gate"
        ),
        device_peak_reserved_growth_bytes_max=_integer(
            memory.get("device_peak_reserved_growth_bytes_max"), "device growth gate"
        ),
        h2d_per_policy_step_max=_number(transfer.get("h2d_per_policy_step_max"), "H2D gate"),
        d2h_per_policy_step_max=_number(transfer.get("d2h_per_policy_step_max"), "D2H gate"),
        host_global_sync_per_policy_step_max=_number(
            transfer.get("host_global_sync_per_policy_step_max"), "sync gate"
        ),
        baseline_ppo=baseline_ppo,
        affinity_cpus=baseline_plan.hardware.affinity_cpus,
        environment_vars=dict(baseline_plan.environment.env_vars),
        hydra_overrides=baseline_plan.environment.hydra_overrides,
        gpu_name=_string(hardware.get("gpu_name"), "baseline.hardware.gpu_name"),
        gpu_uuid=_string(hardware.get("gpu_uuid"), "baseline.hardware.gpu_uuid"),
        gpu_driver_version=_string(
            hardware.get("driver_version"), "baseline.hardware.driver_version"
        ),
        gpu_capacity_bytes=_integer(hardware.get("gpu_memory_mib"), "GPU memory MiB", minimum=1)
        * 1024**2,
    )


def _threshold_payload(binding: BenchmarkBinding) -> dict[str, Any]:
    return {
        "base": {
            "threshold_set_id": binding.threshold_set_id,
            "manifest_path": binding.threshold_manifest_path.as_posix(),
            "manifest_sha256": binding.threshold_manifest_sha256,
            "freeze_commit": binding.threshold_freeze_commit,
        },
        "amendment": {
            "amendment_id": binding.amendment_id,
            "manifest_path": binding.amendment_manifest_path.as_posix(),
            "manifest_sha256": binding.amendment_manifest_sha256,
            "freeze_commit": binding.amendment_freeze_commit,
        },
    }


def expected_case_ids(binding: BenchmarkBinding) -> tuple[str, ...]:
    """Return the non-filterable complete raw case matrix."""

    result: list[str] = []
    for batch_size in binding.batch_sizes:
        for repeat in range(binding.process_repeats):
            for mode in THROUGHPUT_MODES:
                result.append(f"throughput-{mode}-b{batch_size}-r{repeat}")
    for seed in binding.behavior_seeds:
        result.append(f"behavior-mjwarp_device-seed{seed}")
    for repeat in range(binding.process_repeats):
        result.append(f"contention-mjwarp_device-b1024-r{repeat}")
    return tuple(result)


def _throughput_mode_order(repeat_index: int) -> tuple[str, ...]:
    if repeat_index % 2 == 0:
        return THROUGHPUT_MODES
    return tuple(reversed(THROUGHPUT_MODES))


def expected_case_specs(binding: BenchmarkBinding) -> dict[str, dict[str, Any]]:
    """Return the frozen metadata for every process-isolated training worker."""

    specs: dict[str, dict[str, Any]] = {}
    for batch_size in binding.batch_sizes:
        for repeat_index in range(binding.process_repeats):
            for sequence_index, mode in enumerate(_throughput_mode_order(repeat_index)):
                case_id = f"throughput-{mode}-b{batch_size}-r{repeat_index}"
                specs[case_id] = {
                    "lane": "throughput",
                    "mode": mode,
                    "batch_size": batch_size,
                    "seed": repeat_index,
                    "repeat_index": repeat_index,
                    "sequence_index": sequence_index,
                    "iterations": THROUGHPUT_ITERATIONS,
                }
    for seed in binding.behavior_seeds:
        case_id = f"behavior-mjwarp_device-seed{seed}"
        specs[case_id] = {
            "lane": "behavior",
            "mode": "mjwarp_device",
            "batch_size": binding.behavior_num_envs,
            "seed": seed,
            "repeat_index": None,
            "sequence_index": None,
            "iterations": binding.behavior_iterations,
        }
    for repeat_index in range(binding.process_repeats):
        case_id = f"contention-mjwarp_device-b1024-r{repeat_index}"
        specs[case_id] = {
            "lane": "contention",
            "mode": "mjwarp_device",
            "batch_size": 1024,
            "seed": repeat_index,
            "repeat_index": repeat_index,
            "sequence_index": None,
            "iterations": CONTENTION_ITERATIONS,
        }
    if set(specs) != set(expected_case_ids(binding)):
        raise MjwarpPpoBenchmarkError("internal PPO benchmark case registration differs")
    return specs


def _source_payload() -> dict[str, Any]:
    if _git("status", "--short"):
        raise MjwarpPpoBenchmarkError("Phase 5 benchmark execution requires a clean git worktree")
    commit = _git("rev-parse", "HEAD")
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": False,
        "tree_sha256": source_tree_sha256(ROOT_DIR, SOURCE_INPUTS),
        "uv_lock_sha256": sha256_file(ROOT_DIR / "uv.lock"),
        "owner_yaml_sha256": sha256_file(ROOT_DIR / "conf/ppo/task/g1_walk_flat/mjwarp.yaml"),
    }


def _assert_capture_source_unchanged(source: Mapping[str, Any]) -> None:
    """Reject a long capture whose benchmark inputs changed underneath it."""

    if _git("status", "--short"):
        raise MjwarpPpoBenchmarkError(
            "Phase 5 benchmark changed the candidate source during capture"
        )
    if _git("rev-parse", "HEAD") != source.get("commit"):
        raise MjwarpPpoBenchmarkError("candidate HEAD changed during Phase 5 benchmark capture")
    if source_tree_sha256(ROOT_DIR, SOURCE_INPUTS) != source.get("tree_sha256"):
        raise MjwarpPpoBenchmarkError("benchmark source inputs changed during capture")
    if sha256_file(ROOT_DIR / "uv.lock") != source.get("uv_lock_sha256"):
        raise MjwarpPpoBenchmarkError("uv.lock changed during Phase 5 benchmark capture")
    if sha256_file(ROOT_DIR / "conf/ppo/task/g1_walk_flat/mjwarp.yaml") != source.get(
        "owner_yaml_sha256"
    ):
        raise MjwarpPpoBenchmarkError("mjwarp owner YAML changed during Phase 5 benchmark capture")


def _mode_owner(mode: str) -> tuple[str, str]:
    if mode == "mujoco_host":
        return "mujoco", "host_numpy"
    if mode == "mjwarp_device":
        return "mjwarp", "device_resident"
    raise MjwarpPpoBenchmarkError(f"unsupported PPO benchmark mode {mode!r}")


def _run_dirs(log_root: Path, backend: str) -> list[Path]:
    return sorted((log_root / "G1WalkFlat").glob(f"*_{backend}"))


def _run_training_case(
    binding: BenchmarkBinding,
    baseline_plan: Any,
    *,
    mode: str,
    batch_size: int,
    seed: int,
    iterations: int,
    lane: str,
    repeat_index: int | None,
    sequence_index: int | None,
    common_overrides: Iterable[str] = (),
) -> dict[str, Any]:
    """Run one public PPO process and retain every raw output needed by the gate."""

    backend, expected_profile = _mode_owner(mode)
    with tempfile.TemporaryDirectory(prefix=f"unilab_issue705_p5_{lane}_{mode}_") as temp_dir:
        log_root = Path(temp_dir) / "logs"
        command = [
            "uv",
            "run",
            "scripts/train_rsl_rl.py",
            f"task=g1_walk_flat/{backend}",
            f"algo.seed={seed}",
            f"algo.num_envs={batch_size}",
            f"algo.num_steps_per_env={binding.behavior_steps_per_env}",
            f"algo.max_iterations={iterations}",
            "algo.save_interval=1000",
            "training.no_play=true",
            "training.logger=tensorboard",
            f"training.log_root={log_root}",
            *baseline_plan.environment.hydra_overrides,
            *COMMON_PERFORMANCE_OVERRIDES,
            *tuple(common_overrides),
        ]
        process, memory_samples, stdout, stderr = _run_subprocess(
            command,
            baseline_plan,
            memory_poll_interval=binding.memory_poll_interval_sec,
        )
        if process["return_code"] != 0:
            raise MjwarpPpoBenchmarkError(
                f"{lane} {mode} case failed: command={command!r}\n"
                f"stdout:\n{stdout[-6000:]}\nstderr:\n{stderr[-6000:]}"
            )
        run_dirs = _run_dirs(log_root, backend)
        if len(run_dirs) != 1:
            raise MjwarpPpoBenchmarkError(
                f"{lane} {mode} produced {len(run_dirs)} expected run directories"
            )
        run_dir = run_dirs[0]
        run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        run_summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        scalars = _event_scalars(run_dir, REQUIRED_SCALAR_TAGS)

    raw = {
        "scalars": scalars,
        "memory_samples": memory_samples,
        "run_config": run_config,
        "run_config_sha256": canonical_sha256(run_config),
        "run_summary": run_summary,
    }
    summary = summarize_training_raw(raw, warmup_iterations=binding.warmup_iterations)
    case_id = (
        f"behavior-{mode}-seed{seed}"
        if lane == "behavior"
        else f"{lane}-{mode}-b{batch_size}-r{cast(int, repeat_index)}"
    )
    return {
        "case_id": case_id,
        "lane": lane,
        "mode": mode,
        "batch_size": batch_size,
        "seed": seed,
        "repeat_index": repeat_index,
        "sequence_index": sequence_index,
        "iterations": iterations,
        "expected_execution_profile": expected_profile,
        "process": process,
        "raw": raw,
        "summary": summary,
    }


def _finite_scalar_values(
    scalars: Mapping[str, Any], tag: str, *, warmup_iterations: int
) -> list[float]:
    points = scalars.get(tag)
    if not isinstance(points, list):
        raise MjwarpPpoBenchmarkError(f"missing scalar tag {tag}")
    values: list[float] = []
    for index, point in enumerate(points):
        if not isinstance(point, Mapping):
            raise MjwarpPpoBenchmarkError(f"scalar {tag}[{index}] must be a mapping")
        step = _integer(point.get("step"), f"scalar {tag}[{index}].step", minimum=0)
        if step < warmup_iterations:
            continue
        values.append(
            _number(point.get("value"), f"scalar {tag}[{index}].value", minimum=-float("inf"))
        )
    if not values:
        raise MjwarpPpoBenchmarkError(f"scalar {tag} has no post-warmup values")
    return values


def _reward_auc(points: object, *, warmup_iterations: int) -> float:
    if not isinstance(points, list):
        raise MjwarpPpoBenchmarkError("reward scalar points must be a list")
    selected: list[tuple[int, float]] = []
    for index, point in enumerate(points):
        if not isinstance(point, Mapping):
            raise MjwarpPpoBenchmarkError(f"reward point {index} must be a mapping")
        step = _integer(point.get("step"), f"reward point {index}.step", minimum=0)
        if step >= warmup_iterations:
            selected.append(
                (
                    step,
                    _number(
                        point.get("value"), f"reward point {index}.value", minimum=-float("inf")
                    ),
                )
            )
    if len(selected) < 2:
        raise MjwarpPpoBenchmarkError("reward AUC requires two post-warmup points")
    selected.sort()
    return float(np.trapezoid([value for _, value in selected], [step for step, _ in selected]))


def summarize_training_raw(raw: Mapping[str, Any], *, warmup_iterations: int) -> dict[str, Any]:
    """Recompute one worker's training summaries directly from raw scalar events."""

    scalars = _mapping(raw.get("scalars"), "raw.scalars")
    scalar_stats = {
        tag: numeric_stats(_finite_scalar_values(scalars, tag, warmup_iterations=warmup_iterations))
        for tag in REQUIRED_SCALAR_TAGS
    }
    collection = _finite_scalar_values(
        scalars, "Perf/collection_time", warmup_iterations=warmup_iterations
    )
    learner = _finite_scalar_values(
        scalars, "Perf/learning_time", warmup_iterations=warmup_iterations
    )
    if len(collection) != len(learner):
        raise MjwarpPpoBenchmarkError("collection and learner scalar lengths differ")
    iteration_ms = [
        (collect + learn) * 1000.0 for collect, learn in zip(collection, learner, strict=True)
    ]
    memory_samples = raw.get("memory_samples")
    if not isinstance(memory_samples, list) or not memory_samples:
        raise MjwarpPpoBenchmarkError("raw.memory_samples must be a non-empty list")
    rss_values = [
        _integer(_mapping(sample, f"memory sample {index}").get("rss_bytes"), f"rss[{index}]")
        for index, sample in enumerate(memory_samples)
    ]
    run_summary = _mapping(raw.get("run_summary"), "raw.run_summary")
    return {
        "scalar_stats": scalar_stats,
        "iteration_wall_ms": numeric_stats(iteration_ms),
        "collection_ms": numeric_stats([value * 1000.0 for value in collection]),
        "learner_ms": numeric_stats([value * 1000.0 for value in learner]),
        "reward_auc": _reward_auc(
            scalars.get("Train/mean_reward"), warmup_iterations=warmup_iterations
        ),
        "peak_rss_bytes": max(rss_values),
        "peak_gpu_memory_allocated_bytes": _integer(
            run_summary.get("peak_gpu_memory_allocated_bytes"), "run_summary allocated"
        ),
        "peak_gpu_memory_reserved_bytes": _integer(
            run_summary.get("peak_gpu_memory_reserved_bytes"), "run_summary reserved"
        ),
    }


def _population_cv(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise MjwarpPpoBenchmarkError("population CV requires at least two finite values")
    mean = float(array.mean())
    if mean <= 0.0:
        raise MjwarpPpoBenchmarkError("population CV requires positive mean")
    return float(array.std(ddof=0) / mean)


def _aggregate_cases(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not cases:
        raise MjwarpPpoBenchmarkError("cannot aggregate an empty case group")
    summaries = [_mapping(case.get("summary"), "case.summary") for case in cases]

    def stat(name: str, field: str) -> float:
        return float(
            statistics.median(float(_mapping(summary[name], name)[field]) for summary in summaries)
        )

    fps_p50 = [
        float(_mapping(summary["scalar_stats"], "scalar_stats")["Perf/total_fps"]["p50"])
        for summary in summaries
    ]
    return {
        "process_count": len(cases),
        "iteration_p50_median_ms": stat("iteration_wall_ms", "p50"),
        "iteration_p95_median_ms": stat("iteration_wall_ms", "p95"),
        "collection_p50_median_ms": stat("collection_ms", "p50"),
        "learner_p50_median_ms": stat("learner_ms", "p50"),
        "fps_p50_median": float(statistics.median(fps_p50)),
        "fps_p50_population_cv": _population_cv(fps_p50),
        "peak_rss_median_bytes": int(
            statistics.median(int(summary["peak_rss_bytes"]) for summary in summaries)
        ),
        "peak_gpu_reserved_median_bytes": int(
            statistics.median(
                int(summary["peak_gpu_memory_reserved_bytes"]) for summary in summaries
            )
        ),
        "peak_gpu_reserved_max_bytes": max(
            int(summary["peak_gpu_memory_reserved_bytes"]) for summary in summaries
        ),
        "peak_gpu_allocated_max_bytes": max(
            int(summary["peak_gpu_memory_allocated_bytes"]) for summary in summaries
        ),
    }


def build_aggregates(
    cases: Sequence[Mapping[str, Any]], binding: BenchmarkBinding
) -> dict[str, Any]:
    """Aggregate every planned raw case without dropping an outlier or a failed seed."""

    throughput: dict[str, dict[str, Any]] = {}
    for batch_size in binding.batch_sizes:
        per_mode: dict[str, Any] = {}
        for mode in THROUGHPUT_MODES:
            selected = [
                case
                for case in cases
                if case.get("lane") == "throughput"
                and case.get("batch_size") == batch_size
                and case.get("mode") == mode
            ]
            per_mode[mode] = _aggregate_cases(selected)
        throughput[str(batch_size)] = per_mode

    behavior_cases = [case for case in cases if case.get("lane") == "behavior"]
    behavior = _aggregate_cases(behavior_cases)
    summaries = [_mapping(case.get("summary"), "behavior summary") for case in behavior_cases]
    behavior.update(
        {
            "seeds": [
                int(case["seed"])
                for case in sorted(behavior_cases, key=lambda item: int(item["seed"]))
            ],
            "failed_seeds": [],
            "nan_seeds": [],
            "reward_auc_median": float(
                statistics.median(float(summary["reward_auc"]) for summary in summaries)
            ),
            "final_reward_p50_median": float(
                statistics.median(
                    float(summary["scalar_stats"]["Train/mean_reward"]["p50"])
                    for summary in summaries
                )
            ),
            "episode_length_p50_median": float(
                statistics.median(
                    float(summary["scalar_stats"]["Train/mean_episode_length"]["p50"])
                    for summary in summaries
                )
            ),
        }
    )

    contention_cases = [case for case in cases if case.get("lane") == "contention"]
    contention = _aggregate_cases(contention_cases)
    idle = throughput["1024"]["mjwarp_device"]
    contention.update(
        {
            "idle_reference": idle,
            "co_located_to_idle_iteration_p50_ratio": _ratio(
                float(contention["iteration_p50_median_ms"]),
                float(idle["iteration_p50_median_ms"]),
            ),
            "co_located_to_idle_iteration_p95_ratio": _ratio(
                float(contention["iteration_p95_median_ms"]),
                float(idle["iteration_p95_median_ms"]),
            ),
            "co_located_to_idle_fps_p50_ratio": _ratio(
                float(contention["fps_p50_median"]), float(idle["fps_p50_median"])
            ),
        }
    )
    return {
        "throughput": throughput,
        "behavior": behavior,
        "contention": contention,
    }


def _trace_transfer_counts(trace: Mapping[str, Any], *, scope_name: str) -> tuple[int, int, int]:
    events = trace.get("traceEvents")
    if not isinstance(events, list):
        raise MjwarpPpoBenchmarkError("profiler trace must contain traceEvents")
    scopes = [
        event
        for event in events
        if isinstance(event, Mapping)
        and event.get("name") == scope_name
        and event.get("cat") == "user_annotation"
        and isinstance(event.get("ts"), (int, float))
        and isinstance(event.get("dur"), (int, float))
    ]
    if not scopes:
        raise MjwarpPpoBenchmarkError(f"profiler trace has no {scope_name!r} scope")
    intervals = tuple(
        (float(scope["ts"]), float(scope["ts"]) + float(scope["dur"])) for scope in scopes
    )
    h2d = d2h = sync = 0
    for event in events:
        if not isinstance(event, Mapping) or not isinstance(event.get("ts"), (int, float)):
            continue
        timestamp = float(event["ts"])
        if not any(start <= timestamp <= end for start, end in intervals):
            continue
        payload = (
            str(event.get("name", "")) + " " + json.dumps(event.get("args", {}), sort_keys=True)
        ).lower()
        if any(token in payload for token in ("htod", "host to device", "host -> device")):
            h2d += 1
        if any(token in payload for token in ("dtoh", "device to host", "device -> host")):
            d2h += 1
        if "cudadevicesynchronize" in payload:
            sync += 1
    return h2d, d2h, sync


def _sha256_path(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _validate_scalar_coverage(scalars: Mapping[str, Any], *, iterations: int, case_id: str) -> None:
    expected_steps = tuple(range(iterations))
    for tag in REQUIRED_SCALAR_TAGS:
        points = scalars.get(tag)
        if not isinstance(points, list):
            raise MjwarpPpoBenchmarkError(f"{case_id}: missing scalar tag {tag}")
        steps = tuple(
            _integer(
                _mapping(point, f"{case_id}.{tag}[{index}]").get("step"),
                f"{case_id}.{tag}[{index}].step",
            )
            for index, point in enumerate(points)
        )
        if steps != expected_steps:
            raise MjwarpPpoBenchmarkError(
                f"{case_id}: scalar {tag} must contain every iteration exactly once"
            )
        for index, point in enumerate(points):
            _number(
                _mapping(point, f"{case_id}.{tag}[{index}]").get("value"),
                f"{case_id}.{tag}[{index}].value",
                minimum=-float("inf"),
            )


def _validate_process_receipt(
    case: Mapping[str, Any], binding: BenchmarkBinding, errors: list[str]
) -> None:
    case_id = _string(case.get("case_id"), "case_id")
    process = _mapping(case.get("process"), f"{case_id}.process")
    if process.get("return_code") != 0:
        errors.append(f"{case_id}: worker did not succeed")
    command = process.get("command")
    if not isinstance(command, list) or command[:3] != ["uv", "run", "scripts/train_rsl_rl.py"]:
        errors.append(f"{case_id}: worker did not use the public train_rsl_rl owner route")
    elif all(isinstance(item, str) for item in command):
        backend, _ = _mode_owner(_string(case.get("mode"), f"{case_id}.mode"))
        required_arguments = {
            f"task=g1_walk_flat/{backend}",
            f"algo.seed={case.get('seed')}",
            f"algo.num_envs={case.get('batch_size')}",
            f"algo.num_steps_per_env={binding.behavior_steps_per_env}",
            f"algo.max_iterations={case.get('iterations')}",
            "training.no_play=true",
            "training.logger=tensorboard",
            *binding.hydra_overrides,
            *COMMON_PERFORMANCE_OVERRIDES,
        }
        if not required_arguments.issubset(set(command)):
            errors.append(f"{case_id}: worker command differs from the frozen owner protocol")
    if process.get("affinity_cpus") != list(binding.affinity_cpus):
        errors.append(f"{case_id}: worker affinity differs from frozen benchmark plan")
    if process.get("env_vars") != dict(binding.environment_vars):
        errors.append(f"{case_id}: worker thread environment differs from frozen benchmark plan")
    if not isinstance(process.get("run_id"), str) or not process["run_id"].strip():
        errors.append(f"{case_id}: worker receipt has no run_id")
    _number(process.get("duration_sec"), f"{case_id}.process.duration_sec", minimum=0.0)
    _sha256(process.get("stdout_sha256"), f"{case_id}.process.stdout_sha256")
    _sha256(process.get("stderr_sha256"), f"{case_id}.process.stderr_sha256")


def _validate_orchestrator_receipt(
    case: Mapping[str, Any], binding: BenchmarkBinding, errors: list[str]
) -> None:
    case_id = _string(case.get("case_id"), "case_id")
    process = _mapping(case.get("orchestrator_process"), f"{case_id}.orchestrator_process")
    command = process.get("command")
    if (
        not isinstance(command, list)
        or command[:3] != ["uv", "run", "benchmark/rl/benchmark_mjwarp_ppo.py"]
        or "--worker" not in command
        or not {"--case-id", case_id}.issubset(set(command))
    ):
        errors.append(f"{case_id}: case was not isolated through the registered worker CLI")
    if process.get("return_code") != 0:
        errors.append(f"{case_id}: benchmark worker process did not succeed")
    if process.get("affinity_cpus") != list(binding.affinity_cpus):
        errors.append(f"{case_id}: benchmark worker affinity differs from frozen plan")
    if process.get("env_vars") != dict(binding.environment_vars):
        errors.append(f"{case_id}: benchmark worker environment differs from frozen plan")
    _number(process.get("duration_sec"), f"{case_id}.orchestrator.duration_sec")
    _sha256(process.get("stdout_sha256"), f"{case_id}.orchestrator.stdout_sha256")
    _sha256(process.get("stderr_sha256"), f"{case_id}.orchestrator.stderr_sha256")


def _validate_case_shape(
    case: Mapping[str, Any], binding: BenchmarkBinding, errors: list[str]
) -> None:
    case_id = _string(case.get("case_id"), "case_id")
    expected = expected_case_specs(binding).get(case_id)
    if expected is None:
        errors.append(f"{case_id}: is not in the frozen case matrix")
        return
    for key, expected_value in expected.items():
        if case.get(key) != expected_value:
            errors.append(f"{case_id}: {key} differs from frozen matrix")
    if case.get("expected_execution_profile") != _mode_owner(expected["mode"])[1]:
        errors.append(f"{case_id}: expected_execution_profile differs from mode")
    _validate_orchestrator_receipt(case, binding, errors)
    _validate_process_receipt(case, binding, errors)
    if expected["lane"] == "contention":
        contention = _mapping(case.get("contention"), f"{case_id}.contention")
        if contention.get("load_worker_return_code") != 0:
            errors.append(f"{case_id}: owned GPU contention worker did not succeed")
        if contention.get("load_worker_matrix") != [2048, 2048]:
            errors.append(f"{case_id}: contention worker matrix differs from protocol")
        _sha256(contention.get("load_worker_stdout_sha256"), f"{case_id}.contention.stdout_sha256")
        _sha256(contention.get("load_worker_stderr_sha256"), f"{case_id}.contention.stderr_sha256")


def _validate_run_config(
    case: Mapping[str, Any], binding: BenchmarkBinding, errors: list[str]
) -> None:
    case_id = _string(case.get("case_id"), "case_id")
    raw = _mapping(case.get("raw"), f"{case_id}.raw")
    run_config = _mapping(raw.get("run_config"), "run_config")
    if raw.get("run_config_sha256") != canonical_sha256(run_config):
        errors.append(f"{case.get('case_id')}: run_config SHA does not match raw config")
    config = _mapping(run_config.get("config"), "run_config.config")
    training = _mapping(config.get("training"), "run_config.config.training")
    algo = _mapping(config.get("algo"), "run_config.config.algo")
    env_config = _mapping(config.get("env"), "run_config.config.env")
    backend, profile = _mode_owner(_string(case.get("mode"), "case.mode"))
    if training.get("sim_backend") != backend:
        errors.append(f"{case.get('case_id')}: owner backend is not {backend}")
    if backend == "mjwarp" and training.get("execution_profile") != profile:
        errors.append(f"{case.get('case_id')}: mjwarp owner did not select device_resident")
    if algo.get("num_envs") != case.get("batch_size"):
        errors.append(f"{case.get('case_id')}: num_envs differs from case batch")
    if algo.get("num_steps_per_env") != binding.behavior_steps_per_env:
        errors.append(f"{case.get('case_id')}: num_steps_per_env differs from frozen plan")
    if algo.get("max_iterations") != case.get("iterations"):
        errors.append(f"{case.get('case_id')}: max_iterations differs from case")
    if algo.get("seed") != case.get("seed"):
        errors.append(f"{case.get('case_id')}: configured seed differs from case")
    noise = _mapping(env_config.get("noise_config"), "run_config.config.env.noise_config")
    domain_rand = _mapping(env_config.get("domain_rand"), "run_config.config.env.domain_rand")
    curriculum = _mapping(env_config.get("curriculum"), "run_config.config.env.curriculum")
    if noise.get("level") != 0.0:
        errors.append(f"{case_id}: observation noise differs from performance protocol")
    if domain_rand.get("randomize_kp") is not False or domain_rand.get("randomize_kd") is not False:
        errors.append(f"{case_id}: actuator gain DR differs from performance protocol")
    if curriculum.get("enabled") is not False:
        errors.append(f"{case_id}: curriculum differs from performance protocol")
    if backend == "mjwarp":
        snapshot = _mapping(run_config.get("contract_snapshot"), "contract_snapshot")
        policy_abi = snapshot.get("manager.policy_abi")
        if not isinstance(policy_abi, Mapping):
            errors.append(f"{case.get('case_id')}: mjwarp run receipt lacks manager.policy_abi")
        else:
            for key in (
                "plan_fingerprint",
                "policy_abi_fingerprint",
                "executor_key",
                "execution_profile",
            ):
                if not isinstance(policy_abi.get(key), str) or not policy_abi[key].strip():
                    errors.append(f"{case_id}: manager.policy_abi.{key} is missing")
            if policy_abi.get("execution_profile") != PROFILE:
                errors.append(
                    f"{case_id}: manager.policy_abi execution profile is not device_resident"
                )
    summary = _mapping(raw.get("run_summary"), "run_summary")
    if summary.get("status") != "completed" or summary.get("completed_iterations") != case.get(
        "iterations"
    ):
        errors.append(f"{case.get('case_id')}: PPO run did not complete the exact iteration budget")
    _number(summary.get("training_wall_time_sec"), f"{case_id}.run_summary.training_wall_time_sec")
    _integer(summary.get("peak_process_rss_bytes"), f"{case_id}.run_summary.peak_process_rss_bytes")
    _integer(
        summary.get("peak_gpu_memory_allocated_bytes"),
        f"{case_id}.run_summary.peak_gpu_memory_allocated_bytes",
    )
    _integer(
        summary.get("peak_gpu_memory_reserved_bytes"),
        f"{case_id}.run_summary.peak_gpu_memory_reserved_bytes",
    )
    scalars = _mapping(raw.get("scalars"), f"{case_id}.raw.scalars")
    _validate_scalar_coverage(
        scalars,
        iterations=_integer(case.get("iterations"), f"{case_id}.iterations", minimum=2),
        case_id=case_id,
    )


def _ratio(value: float | int, reference: float | int) -> float:
    if reference <= 0:
        raise MjwarpPpoBenchmarkError("ratio reference must be positive")
    return float(value) / float(reference)


def _validate_device_evidence_contract(
    artifact: Mapping[str, Any], binding: BenchmarkBinding, errors: list[str]
) -> None:
    """Validate device/profiler facts that must never become diagnostic-only.

    A failed performance threshold is useful optimization evidence.  A trace
    that cannot be reconciled, a different GPU, or a substituted backend is
    not.  Keep those provenance and measurement-contract checks separate from
    the threshold comparison so ``--allow-gate-failure`` cannot preserve an
    artifact whose raw evidence is untrustworthy.
    """

    device = _mapping(artifact.get("device"), "device")
    if _integer(device.get("gpu_capacity_bytes"), "device.capacity") != binding.gpu_capacity_bytes:
        errors.append("device GPU capacity differs from frozen benchmark host")
    peak_reserved = _integer(device.get("peak_gpu_reserved_bytes"), "device.reserved")
    all_device_reserved = [
        _integer(
            _mapping(case.get("summary"), f"{case.get('case_id')}.summary").get(
                "peak_gpu_memory_reserved_bytes"
            ),
            f"{case.get('case_id')}.summary.reserved",
        )
        for case in _mapping(artifact, "artifact").get("cases", [])
        if isinstance(case, Mapping) and case.get("mode") == "mjwarp_device"
    ]
    if not all_device_reserved or peak_reserved != max(all_device_reserved):
        errors.append("device reserved memory does not reconcile with all mjwarp cases")
    if device.get("profiler_reconciled") is not True:
        errors.append("device profiler reconciliation is required")
    profile_summary = _mapping(device.get("profiler_summary"), "device.profiler_summary")
    profile_steps = _integer(
        profile_summary.get("steps"), "device.profiler_summary.steps", minimum=1
    )
    if profile_steps != PROFILE_STEPS:
        errors.append("device profiler summary steps differ from frozen profile protocol")
    runtime_delta = _mapping(
        profile_summary.get("runtime_delta"), "device.profiler_summary.runtime_delta"
    )
    trace_counts = _mapping(
        profile_summary.get("trace_counts"), "device.profiler_summary.trace_counts"
    )
    counter_keys = (
        ("host_to_device_transfers", "h2d_per_policy_step"),
        ("device_to_host_transfers", "d2h_per_policy_step"),
        ("global_synchronizations", "host_global_sync_per_policy_step"),
    )
    for counter_key, metric_key in counter_keys:
        count = _integer(runtime_delta.get(counter_key), f"device.runtime_delta.{counter_key}")
        if _integer(trace_counts.get(counter_key), f"device.trace_counts.{counter_key}") != count:
            errors.append(f"device profiler trace does not reconcile {counter_key}")
        if _number(device.get(metric_key), f"device.{metric_key}") != count / profile_steps:
            errors.append(f"device {metric_key} does not reconcile profiler summary")
    wrapper_delta = _mapping(
        profile_summary.get("wrapper_delta"), "device.profiler_summary.wrapper_delta"
    )
    if _integer(
        wrapper_delta.get("finite_metric_materializations"),
        "device.profiler_summary.wrapper_delta.finite_metric_materializations",
    ) != _integer(device.get("metrics_materializations"), "device.metrics_materializations"):
        errors.append(
            "device finite metric materialization count does not reconcile profiler summary"
        )
    if _integer(
        wrapper_delta.get("finite_metric_device_to_host_bytes"),
        "device.profiler_summary.wrapper_delta.finite_metric_device_to_host_bytes",
    ) != _integer(
        device.get("metrics_device_to_host_bytes"), "device.metrics_device_to_host_bytes"
    ):
        errors.append("device finite metric bytes do not reconcile profiler summary")
    backend_receipt = _mapping(
        profile_summary.get("backend_receipt"), "device.profiler_summary.backend_receipt"
    )
    if backend_receipt.get("backend_type") != "mjwarp":
        errors.append("device profiler backend receipt is not mjwarp")
    if backend_receipt.get("execution_profile") != PROFILE:
        errors.append("device profiler execution profile is not device_resident")
    for key in ("task_plan_fingerprint", "policy_abi_fingerprint", "backend_plan_fingerprint"):
        if not isinstance(backend_receipt.get(key), str) or not backend_receipt[key].strip():
            errors.append(f"device profiler backend receipt lacks {key}")
    trace = _mapping(device.get("profiler_trace"), "device.profiler_trace")
    _sha256(trace.get("sha256"), "device profiler trace hash")
    if not _string(trace.get("filename"), "device profiler trace filename"):
        errors.append("device profiler trace filename is required")


def _recompute_gate(artifact: Mapping[str, Any], binding: BenchmarkBinding) -> list[str]:
    """Recompute only frozen threshold outcomes from a valid raw artifact."""

    errors: list[str] = []
    aggregates = _mapping(artifact.get("aggregates"), "aggregates")
    behavior = _mapping(aggregates.get("behavior"), "aggregates.behavior")
    baseline = binding.baseline_ppo
    throughput = _mapping(aggregates.get("throughput"), "aggregates.throughput")
    for batch_size in binding.batch_sizes:
        per_mode = _mapping(throughput.get(str(batch_size)), f"throughput[{batch_size}]")
        for mode in THROUGHPUT_MODES:
            result = _mapping(per_mode.get(mode), f"throughput[{batch_size}].{mode}")
            if result.get("process_count") != binding.process_repeats:
                errors.append(f"throughput[{batch_size}].{mode}: wrong process count")
            cv = _number(result.get("fps_p50_population_cv"), f"throughput[{batch_size}].{mode}.cv")
            if cv > binding.max_population_cv_by_batch[batch_size]:
                errors.append(
                    f"throughput[{batch_size}].{mode}: population CV exceeds frozen limit"
                )
        host = _mapping(per_mode.get("mujoco_host"), f"throughput[{batch_size}].mujoco_host")
        device = _mapping(per_mode.get("mjwarp_device"), f"throughput[{batch_size}].mjwarp_device")
        if (
            _ratio(
                _number(
                    device.get("iteration_p50_median_ms"), f"throughput[{batch_size}].device.p50"
                ),
                _number(host.get("iteration_p50_median_ms"), f"throughput[{batch_size}].host.p50"),
            )
            > binding.p50_latency_ratio_max
        ):
            errors.append(
                f"throughput[{batch_size}]: iteration p50 violates frozen performance gate"
            )
        if (
            _ratio(
                _number(
                    device.get("iteration_p95_median_ms"), f"throughput[{batch_size}].device.p95"
                ),
                _number(host.get("iteration_p95_median_ms"), f"throughput[{batch_size}].host.p95"),
            )
            > binding.p95_latency_ratio_max
        ):
            errors.append(
                f"throughput[{batch_size}]: iteration p95 violates frozen performance gate"
            )
        if (
            _ratio(
                _number(device.get("fps_p50_median"), f"throughput[{batch_size}].device.fps"),
                _number(host.get("fps_p50_median"), f"throughput[{batch_size}].host.fps"),
            )
            < binding.throughput_ratio_min
        ):
            errors.append(f"throughput[{batch_size}]: FPS violates frozen throughput gate")
        if (
            _ratio(
                _integer(
                    device.get("peak_rss_median_bytes"), f"throughput[{batch_size}].device.rss"
                ),
                _integer(host.get("peak_rss_median_bytes"), f"throughput[{batch_size}].host.rss"),
            )
            > binding.host_memory_ratio_max
        ):
            errors.append(f"throughput[{batch_size}]: RSS violates frozen memory gate")

    if tuple(behavior.get("seeds", ())) != binding.behavior_seeds:
        errors.append("behavior seeds do not exactly match frozen [0, 1, 2, 3, 4]")
    if behavior.get("failed_seeds") or behavior.get("nan_seeds"):
        errors.append("behavior contains failed or non-finite seeds")
    if (
        _ratio(
            _number(behavior.get("fps_p50_median"), "behavior.fps"),
            float(baseline["fps_p50_median"]),
        )
        < binding.fps_p50_median_ratio_min
    ):
        errors.append("behavior FPS p50 median violates frozen training gate")
    if (
        _number(behavior.get("reward_auc_median"), "behavior.auc", minimum=-float("inf"))
        < float(baseline["reward_auc_median"]) - binding.reward_auc_median_drop_max
    ):
        errors.append("behavior reward AUC violates frozen training gate")
    if (
        _number(behavior.get("final_reward_p50_median"), "behavior.reward", minimum=-float("inf"))
        < float(baseline["final_reward_p50_median"]) - binding.final_reward_p50_median_drop_max
    ):
        errors.append("behavior final reward violates frozen training gate")
    if (
        _ratio(
            _number(behavior.get("episode_length_p50_median"), "behavior.length"),
            float(baseline["episode_length_p50_median"]),
        )
        < binding.episode_length_median_ratio_min
    ):
        errors.append("behavior episode length violates frozen training gate")
    if (
        _ratio(
            _integer(behavior.get("peak_rss_median_bytes"), "behavior.rss"),
            int(baseline["peak_rss_median_bytes"]),
        )
        > binding.host_memory_ratio_max
    ):
        errors.append("behavior RSS violates frozen memory gate")

    device = _mapping(artifact.get("device"), "device")
    peak_reserved = _integer(device.get("peak_gpu_reserved_bytes"), "device.reserved")
    if (
        _ratio(peak_reserved, binding.gpu_capacity_bytes)
        > binding.device_peak_reserved_capacity_ratio_max
    ):
        errors.append("device reserved capacity ratio violates frozen memory gate")
    if (
        peak_reserved - int(baseline["peak_gpu_reserved_median_bytes"])
        > binding.device_peak_reserved_growth_bytes_max
    ):
        errors.append("device reserved growth violates frozen memory gate")
    limits = {
        "h2d_per_policy_step": binding.h2d_per_policy_step_max,
        "d2h_per_policy_step": binding.d2h_per_policy_step_max,
        "host_global_sync_per_policy_step": binding.host_global_sync_per_policy_step_max,
    }
    for name, limit in limits.items():
        if _number(device.get(name), f"device.{name}") > limit:
            errors.append(f"device {name} violates frozen transfer gate")
    return errors


def _git_file_sha256(repository: Path, commit: str, path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    return f"sha256:{hashlib.sha256(completed.stdout).hexdigest()}"


def _validate_source_at_commit(
    source: Mapping[str, Any], binding: BenchmarkBinding, repository: Path, errors: list[str]
) -> None:
    commit = source.get("commit")
    if not isinstance(commit, str) or not _is_commit(commit):
        return
    if commit == binding.amendment_freeze_commit:
        errors.append("candidate commit cannot equal amendment freeze commit")
    for command, error in (
        (
            ["git", "merge-base", "--is-ancestor", binding.amendment_freeze_commit, commit],
            "candidate commit does not descend from amendment freeze",
        ),
        (
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            "candidate commit is not available in current history",
        ),
    ):
        try:
            subprocess.run(command, cwd=repository, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            errors.append(error)
    try:
        expected = {
            "tree_sha256": source_tree_sha256_at_commit(repository, SOURCE_INPUTS, commit),
            "uv_lock_sha256": _git_file_sha256(repository, commit, "uv.lock"),
            "owner_yaml_sha256": _git_file_sha256(
                repository, commit, "conf/ppo/task/g1_walk_flat/mjwarp.yaml"
            ),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(f"candidate source verification failed: {type(exc).__name__}: {exc}")
        return
    for key, expected_value in expected.items():
        if source.get(key) != expected_value:
            errors.append(f"source.{key} does not match candidate commit")


def _validate_trace_sibling(
    root: Mapping[str, Any], artifact_path: Path, errors: list[str]
) -> None:
    device = _mapping(root.get("device"), "device")
    trace = _mapping(device.get("profiler_trace"), "device.profiler_trace")
    trace_path = artifact_path.parent / _string(trace.get("filename"), "trace filename")
    if not trace_path.is_file():
        errors.append("profiler trace sibling is missing")
        return
    if trace.get("sha256") != _sha256_path(trace_path):
        errors.append("profiler trace hash does not match sibling")
        return
    try:
        trace_data = _mapping(json.loads(trace_path.read_text(encoding="utf-8")), "trace")
        actual = _trace_transfer_counts(trace_data, scope_name="issue705.mjwarp_ppo_rollout")
        profile = _mapping(device.get("profiler_summary"), "device.profiler_summary")
        trace_counts = _mapping(profile.get("trace_counts"), "device.profiler_summary.trace_counts")
        expected = (
            _integer(trace_counts.get("host_to_device_transfers"), "trace H2D"),
            _integer(trace_counts.get("device_to_host_transfers"), "trace D2H"),
            _integer(trace_counts.get("global_synchronizations"), "trace sync"),
        )
        if actual != expected:
            errors.append("profiler trace counts do not match the recorded profiler summary")
    except (OSError, UnicodeError, json.JSONDecodeError, MjwarpPpoBenchmarkError) as exc:
        errors.append(f"profiler trace cannot be validated: {type(exc).__name__}: {exc}")


def _integrity_validation_errors(
    artifact: object,
    *,
    binding: BenchmarkBinding | None = None,
    repo_root: Path | None = None,
    artifact_path: Path | None = None,
) -> tuple[str, ...]:
    """Recompute raw/provenance facts without inspecting threshold gate outcome.

    This deliberately excludes frozen performance/training/memory/transfer
    thresholds.  A complete result that misses one of those thresholds is
    valuable diagnostic evidence; a broken source receipt, matrix, profiler,
    or raw-to-summary reconciliation is not.
    """

    errors: list[str] = []
    try:
        root = _mapping(artifact, "artifact")
        active_binding = binding or load_binding()
        if root.get("schema_version") != SCHEMA_VERSION:
            errors.append("schema_version differs")
        if root.get("issue") != ISSUE or root.get("kind") != ARTIFACT_KIND:
            errors.append("artifact issue or kind differs")
        if root.get("profile") != PROFILE:
            errors.append("artifact profile must be device_resident")
        threshold = _mapping(root.get("threshold"), "threshold")
        expected_threshold = _threshold_payload(active_binding)
        if threshold != expected_threshold:
            errors.append("threshold differs from frozen binding")
        source = _mapping(root.get("source"), "source")
        commit = source.get("commit")
        if not _is_commit(commit):
            errors.append("source.commit must be a full SHA")
        if source.get("dirty") is not False:
            errors.append("benchmark source must be clean")
        _sha256(source.get("tree_sha256"), "source.tree_sha256")
        _sha256(source.get("uv_lock_sha256"), "source.uv_lock_sha256")
        _sha256(source.get("owner_yaml_sha256"), "source.owner_yaml_sha256")
        if commit == active_binding.amendment_freeze_commit:
            errors.append("candidate commit cannot equal amendment freeze commit")

        hardware = _mapping(root.get("hardware"), "hardware")
        expected_hardware = {
            "gpu_name": active_binding.gpu_name,
            "gpu_uuid": active_binding.gpu_uuid,
            "gpu_memory_mib": active_binding.gpu_capacity_bytes // 1024**2,
            "driver_version": active_binding.gpu_driver_version,
            "affinity_cpus": list(active_binding.affinity_cpus),
        }
        for key, expected_value in expected_hardware.items():
            if hardware.get(key) != expected_value:
                errors.append(f"hardware.{key} differs from frozen benchmark host")

        execution = _mapping(root.get("execution"), "execution")
        if execution.get("process_isolation") is not True:
            errors.append("execution.process_isolation must be true")
        if execution.get("throughput_iterations") != THROUGHPUT_ITERATIONS:
            errors.append("execution.throughput_iterations differs from protocol")
        if execution.get("contention_iterations") != CONTENTION_ITERATIONS:
            errors.append("execution.contention_iterations differs from protocol")
        if execution.get("common_performance_overrides") != list(COMMON_PERFORMANCE_OVERRIDES):
            errors.append("execution.common_performance_overrides differs from protocol")
        if execution.get("frozen_hydra_overrides") != list(active_binding.hydra_overrides):
            errors.append("execution.frozen_hydra_overrides differs from baseline plan")
        for preflight_key in ("preflight_before", "preflight_after"):
            preflight = _mapping(execution.get(preflight_key), f"execution.{preflight_key}")
            if preflight.get("gpu_compute_processes") != []:
                errors.append(f"execution.{preflight_key} records foreign GPU compute processes")

        device_receipt = _mapping(root.get("device"), "device")
        try:
            _validate_device_evidence_contract(root, active_binding, errors)
        except MjwarpPpoBenchmarkError as exc:
            errors.append(f"device evidence contract is invalid: {exc}")
        profiler_process = _mapping(
            device_receipt.get("profiler_process"), "device.profiler_process"
        )
        profiler_command = profiler_process.get("command")
        if (
            not isinstance(profiler_command, list)
            or profiler_command[:3] != ["uv", "run", "benchmark/rl/benchmark_mjwarp_ppo.py"]
            or "--profile-worker" not in profiler_command
        ):
            errors.append("device profiler did not run in an independent benchmark worker process")
        if profiler_process.get("return_code") != 0:
            errors.append("device profiler worker did not succeed")
        if profiler_process.get("affinity_cpus") != list(active_binding.affinity_cpus):
            errors.append("device profiler affinity differs from frozen benchmark plan")
        if profiler_process.get("env_vars") != dict(active_binding.environment_vars):
            errors.append("device profiler environment differs from frozen benchmark plan")
        _sha256(profiler_process.get("stdout_sha256"), "device.profiler_process.stdout_sha256")
        _sha256(profiler_process.get("stderr_sha256"), "device.profiler_process.stderr_sha256")

        cases_value = root.get("cases")
        if not isinstance(cases_value, list):
            raise MjwarpPpoBenchmarkError("cases must be a list")
        cases = cast(list[Mapping[str, Any]], cases_value)
        expected_ids = expected_case_ids(active_binding)
        observed: list[str] = []
        for raw_case in cases_value:
            try:
                observed.append(_string(_mapping(raw_case, "case").get("case_id"), "case_id"))
            except MjwarpPpoBenchmarkError as exc:
                errors.append(str(exc))
        if len(observed) != len(expected_ids) or set(observed) != set(expected_ids):
            errors.append(
                "raw case matrix is incomplete, duplicated, or contains an unplanned case"
            )
        run_ids: list[str] = []
        mjwarp_policy_abis: list[Mapping[str, Any]] = []
        for raw_case in cases_value:
            try:
                case = _mapping(raw_case, "case")
                _validate_case_shape(case, active_binding, errors)
                process = _mapping(case.get("process"), "case.process")
                run_id = process.get("run_id")
                if isinstance(run_id, str):
                    run_ids.append(run_id)
                _validate_run_config(case, active_binding, errors)
                raw = _mapping(case.get("raw"), "case.raw")
                if case.get("mode") == "mjwarp_device":
                    run_config = _mapping(raw.get("run_config"), "case.raw.run_config")
                    contract_snapshot = _mapping(
                        run_config.get("contract_snapshot"), "case.raw.contract_snapshot"
                    )
                    mjwarp_policy_abis.append(
                        _mapping(
                            contract_snapshot.get("manager.policy_abi"),
                            "case.raw.contract_snapshot.manager.policy_abi",
                        )
                    )
                expected_summary = summarize_training_raw(
                    raw, warmup_iterations=active_binding.warmup_iterations
                )
                if case.get("summary") != expected_summary:
                    errors.append(
                        f"{case.get('case_id')}: summary is not recomputed from raw scalars"
                    )
            except MjwarpPpoBenchmarkError as exc:
                errors.append(str(exc))
        if len(run_ids) != len(set(run_ids)):
            errors.append("process-isolated worker receipts reuse a run_id")
        if len({canonical_sha256(snapshot) for snapshot in mjwarp_policy_abis}) != 1:
            errors.append("mjwarp process matrix does not share one managed policy ABI")
        if mjwarp_policy_abis:
            profile_summary = _mapping(
                device_receipt.get("profiler_summary"), "device.profiler_summary"
            )
            profile_backend = _mapping(
                profile_summary.get("backend_receipt"),
                "device.profiler_summary.backend_receipt",
            )
            policy_abi = mjwarp_policy_abis[0]
            if profile_backend.get("task_plan_fingerprint") != policy_abi.get("plan_fingerprint"):
                errors.append("profiler task plan fingerprint differs from training receipts")
            if profile_backend.get("policy_abi_fingerprint") != policy_abi.get(
                "policy_abi_fingerprint"
            ):
                errors.append("profiler policy ABI fingerprint differs from training receipts")
        try:
            expected_aggregates = build_aggregates(cases, active_binding)
            if root.get("aggregates") != expected_aggregates:
                errors.append("aggregates are not an exact recomputation of every raw case")
        except (KeyError, TypeError, ValueError, MjwarpPpoBenchmarkError) as exc:
            errors.append(f"aggregate recomputation failed: {type(exc).__name__}: {exc}")

        if repo_root is not None:
            repository = repo_root.resolve()
            _validate_source_at_commit(source, active_binding, repository, errors)
            try:
                plan = _load_plan(repository / DEFAULT_BASELINE_PLAN)
                if root.get("hardware") != _hardware_payload(plan):
                    errors.append("hardware differs from live frozen benchmark host")
            except Exception as exc:  # noqa: BLE001 - evidence must fail on an unavailable host probe.
                errors.append(f"hardware validation failed: {type(exc).__name__}: {exc}")
        if artifact_path is not None:
            _validate_trace_sibling(root, artifact_path, errors)
    except MjwarpPpoBenchmarkError as exc:
        errors.append(str(exc))
    return tuple(errors)


def _validation_error_parts(
    artifact: object,
    *,
    binding: BenchmarkBinding | None = None,
    repo_root: Path | None = None,
    artifact_path: Path | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return non-negotiable evidence errors separately from threshold misses."""

    integrity_errors = list(
        _integrity_validation_errors(
            artifact,
            binding=binding,
            repo_root=repo_root,
            artifact_path=artifact_path,
        )
    )
    gate_errors: tuple[str, ...] = ()
    try:
        root = _mapping(artifact, "artifact")
        active_binding = binding or load_binding()
        gate_errors = tuple(_recompute_gate(root, active_binding))
    except MjwarpPpoBenchmarkError as exc:
        integrity_errors.append(f"gate recomputation failed: {type(exc).__name__}: {exc}")
    return tuple(integrity_errors), gate_errors


def _core_validation_errors(
    artifact: object,
    *,
    binding: BenchmarkBinding | None = None,
    repo_root: Path | None = None,
    artifact_path: Path | None = None,
) -> tuple[str, ...]:
    """Recompute all raw/provenance/gate facts without inspecting ``artifact.gate``."""

    integrity_errors, gate_errors = _validation_error_parts(
        artifact,
        binding=binding,
        repo_root=repo_root,
        artifact_path=artifact_path,
    )
    return (*integrity_errors, *gate_errors)


def validate_artifact(
    artifact: object,
    *,
    binding: BenchmarkBinding | None = None,
    repo_root: Path | None = None,
    artifact_path: Path | None = None,
    require_passing_gate: bool = True,
) -> tuple[str, ...]:
    """Validate raw evidence and the recorded gate.

    ``require_passing_gate=False`` is intentionally narrow: it accepts only a
    fully reconciled artifact whose *sole* failures are recomputed frozen
    thresholds.  It is useful for retaining diagnostic measurements, but the
    default remains evidence-grade validation and therefore rejects every
    failed gate.
    """

    integrity_errors, threshold_errors = _validation_error_parts(
        artifact,
        binding=binding,
        repo_root=repo_root,
        artifact_path=artifact_path,
    )
    core_errors = (*integrity_errors, *threshold_errors)
    gate_errors: list[str] = []
    try:
        recorded_gate = _mapping(_mapping(artifact, "artifact").get("gate"), "gate")
        expected_gate = {"passed": not core_errors, "errors": list(core_errors)}
        if recorded_gate != expected_gate:
            gate_errors.append("recorded gate does not match independent core validation")
    except MjwarpPpoBenchmarkError as exc:
        gate_errors.append(str(exc))
    if require_passing_gate:
        return (*core_errors, *gate_errors)
    return (*integrity_errors, *gate_errors)


def _device_profile_worker(*, output: Path, steps: int) -> int:
    """Capture the formal public wrapper rollout path, never a benchmark monkey patch."""

    import torch
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from torch.profiler import ProfilerActivity, profile, record_function

    from unilab.training import BackendAdapter, create_env, ensure_registries
    from unilab.training.rsl_rl_device import DeviceRslRlVecEnvWrapper

    output.parent.mkdir(parents=True, exist_ok=True)
    ensure_registries()
    GlobalHydra.instance().clear()
    baseline_plan = _load_plan(ROOT_DIR / DEFAULT_BASELINE_PLAN)
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf/ppo"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=g1_walk_flat/mjwarp",
                "algo.num_envs=128",
                *baseline_plan.environment.hydra_overrides,
                *COMMON_PERFORMANCE_OVERRIDES,
                "hydra.run.dir=.",
                "hydra.output_subdir=null",
                "hydra/job_logging=disabled",
                "hydra/hydra_logging=disabled",
            ],
        )
    env_override = BackendAdapter(
        cfg, root_dir=ROOT_DIR, algo_name="ppo"
    ).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=128, env_cfg_override=env_override)
    wrapper = DeviceRslRlVecEnvWrapper(env, device="cuda:0", reset_seed=0)
    try:
        actions = torch.zeros((wrapper.num_envs, wrapper.num_actions), device=wrapper.device)
        for _ in range(3):
            wrapper.step(actions)
        wrapper.last_transition.completion.event.synchronize()
        runtime_before = wrapper.runtime.traffic_diagnostics
        wrapper_before = wrapper.traffic_diagnostics
        with profile(activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA)) as profiler:
            with record_function("issue705.mjwarp_ppo_rollout"):
                for _ in range(steps):
                    wrapper.step(actions)
            wrapper.last_transition.completion.event.synchronize()
        profiler.export_chrome_trace(str(output))
        trace = _mapping(json.loads(output.read_text(encoding="utf-8")), "profiler trace")
        h2d, d2h, sync = _trace_transfer_counts(trace, scope_name="issue705.mjwarp_ppo_rollout")
        runtime_after = wrapper.runtime.traffic_diagnostics
        wrapper.finish_rollout()
        wrapper_after = wrapper.traffic_diagnostics
        policy_abi = wrapper.runtime.policy_abi_snapshot
        payload = {
            "steps": steps,
            "runtime_delta": {
                "host_to_device_transfers": runtime_after.host_to_device_transfers
                - runtime_before.host_to_device_transfers,
                "device_to_host_transfers": runtime_after.device_to_host_transfers
                - runtime_before.device_to_host_transfers,
                "global_synchronizations": runtime_after.global_synchronizations
                - runtime_before.global_synchronizations,
            },
            "wrapper_delta": {
                "action_publications": wrapper_after.action_publications
                - wrapper_before.action_publications,
                "finite_metric_materializations": wrapper_after.finite_metric_materializations
                - wrapper_before.finite_metric_materializations,
                "finite_metric_device_to_host_bytes": wrapper_after.finite_metric_device_to_host_bytes
                - wrapper_before.finite_metric_device_to_host_bytes,
            },
            "trace_counts": {
                "host_to_device_transfers": h2d,
                "device_to_host_transfers": d2h,
                "global_synchronizations": sync,
            },
            "backend_receipt": {
                "backend_type": wrapper.runtime.bound_plan.backend_type,
                "execution_profile": wrapper.runtime.bound_plan.execution_profile.value,
                "task_plan_fingerprint": wrapper.runtime.plan.fingerprint,
                "policy_abi_fingerprint": policy_abi["policy_abi_fingerprint"],
                "backend_plan_fingerprint": wrapper.runtime.bound_plan.fingerprint,
            },
        }
        output.with_suffix(".summary.json").write_text(
            json.dumps(payload, sort_keys=True), encoding="utf-8"
        )
    finally:
        wrapper.close()
    return 0


def _run_profile_case(
    baseline_plan: Any, *, trace_output: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Capture profiler evidence in a fresh owner-route process."""

    trace_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output = trace_output.with_suffix(".summary.json")
    command = [
        "uv",
        "run",
        "benchmark/rl/benchmark_mjwarp_ppo.py",
        "--profile-worker",
        "--profile-steps",
        str(PROFILE_STEPS),
        "--trace-out",
        str(trace_output),
    ]
    process, _, stdout, stderr = _run_subprocess(command, baseline_plan)
    if process["return_code"] != 0 or not trace_output.is_file() or not summary_output.is_file():
        raise MjwarpPpoBenchmarkError(
            f"device profiler worker failed\nstdout:\n{stdout[-6000:]}\nstderr:\n{stderr[-6000:]}"
        )
    try:
        summary = _mapping(
            json.loads(summary_output.read_text(encoding="utf-8")), "profile summary"
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MjwarpPpoBenchmarkError(f"device profile summary cannot be loaded: {exc}") from exc
    return process, dict(summary)


def _contention_worker(*, ready_file: Path, stop_file: Path) -> int:
    """Run a declared, reproducible GPU compute load until the parent stops it."""

    import torch

    ready_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    left = torch.randn((2048, 2048), device=device)
    right = torch.randn((2048, 2048), device=device)
    result = torch.empty((2048, 2048), device=device)
    torch.mm(left, right, out=result)
    torch.cuda.synchronize(device)
    ready_file.write_text("ready", encoding="utf-8")
    while not stop_file.exists():
        torch.mm(left, right, out=result)
        torch.cuda.synchronize(device)
    return 0


def _run_contention_case(
    binding: BenchmarkBinding,
    baseline_plan: Any,
    *,
    repeat: int,
) -> dict[str, Any]:
    """Measure a device case while an explicitly-owned same-GPU load is active."""

    with tempfile.TemporaryDirectory(prefix="unilab_issue705_p5_contention_") as temp_dir:
        root = Path(temp_dir)
        ready = root / "ready"
        stop = root / "stop"
        stdout_path = root / "load.stdout"
        stderr_path = root / "load.stderr"
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            load = subprocess.Popen(
                [
                    "uv",
                    "run",
                    "benchmark/rl/benchmark_mjwarp_ppo.py",
                    "--contention-worker",
                    "--ready-file",
                    str(ready),
                    "--stop-file",
                    str(stop),
                ],
                cwd=ROOT_DIR,
                env={**os.environ, **dict(baseline_plan.environment.env_vars)},
                stdout=stdout,
                stderr=stderr,
                preexec_fn=lambda: os.sched_setaffinity(
                    0, set(baseline_plan.hardware.affinity_cpus)
                ),
            )
            deadline = time.monotonic() + 60.0
            while not ready.exists() and time.monotonic() < deadline and load.poll() is None:
                time.sleep(0.05)
            if not ready.exists():
                load.kill()
                raise MjwarpPpoBenchmarkError("contention worker did not become ready")
            try:
                case = _run_training_case(
                    binding,
                    baseline_plan,
                    mode="mjwarp_device",
                    batch_size=1024,
                    seed=repeat,
                    iterations=CONTENTION_ITERATIONS,
                    lane="contention",
                    repeat_index=repeat,
                    sequence_index=None,
                )
            finally:
                stop.write_text("stop", encoding="utf-8")
                with suppress(subprocess.TimeoutExpired):
                    load.wait(timeout=30.0)
                if load.poll() is None:
                    load.kill()
        case["contention"] = {
            "load_worker_return_code": load.returncode,
            "load_worker_stdout_sha256": _sha256_path(stdout_path),
            "load_worker_stderr_sha256": _sha256_path(stderr_path),
            "load_worker_matrix": [2048, 2048],
        }
    return case


def _execute_worker_case(*, case_id: str, output: Path) -> int:
    """Execute one registered case and serialize its inner owner-process receipt."""

    binding = load_binding()
    expected = expected_case_specs(binding).get(case_id)
    if expected is None:
        raise MjwarpPpoBenchmarkError(f"worker case {case_id!r} is not registered")
    baseline_plan = _load_plan(ROOT_DIR / DEFAULT_BASELINE_PLAN)
    if expected["lane"] == "contention":
        case = _run_contention_case(
            binding,
            baseline_plan,
            repeat=cast(int, expected["repeat_index"]),
        )
    else:
        case = _run_training_case(
            binding,
            baseline_plan,
            mode=cast(str, expected["mode"]),
            batch_size=cast(int, expected["batch_size"]),
            seed=cast(int, expected["seed"]),
            iterations=cast(int, expected["iterations"]),
            lane=cast(str, expected["lane"]),
            repeat_index=cast(int | None, expected["repeat_index"]),
            sequence_index=cast(int | None, expected["sequence_index"]),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(case), sort_keys=True), encoding="utf-8")
    return 0


def _run_registered_case_process(
    baseline_plan: Any,
    *,
    case_id: str,
) -> dict[str, Any]:
    """Run one benchmark worker in a fresh process and retain both process layers."""

    with tempfile.TemporaryDirectory(prefix=f"unilab_issue705_p5_worker_{case_id}_") as temp_dir:
        output = Path(temp_dir) / "case.json"
        command = [
            "uv",
            "run",
            "benchmark/rl/benchmark_mjwarp_ppo.py",
            "--worker",
            "--case-id",
            case_id,
            "--worker-out",
            str(output),
        ]
        process, _, stdout, stderr = _run_subprocess(command, baseline_plan)
        if process["return_code"] != 0 or not output.is_file():
            raise MjwarpPpoBenchmarkError(
                f"registered worker {case_id} failed\n"
                f"stdout:\n{stdout[-6000:]}\nstderr:\n{stderr[-6000:]}"
            )
        try:
            case = _mapping(json.loads(output.read_text(encoding="utf-8")), case_id)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise MjwarpPpoBenchmarkError(f"worker {case_id} output is invalid: {exc}") from exc
    result = dict(case)
    result["orchestrator_process"] = process
    return result


def collect_artifact(
    *, output: Path, trace_output: Path, allow_gate_failure: bool = False
) -> dict[str, Any]:
    """Execute the pre-registered matrix and return a validated artifact.

    A diagnostic capture is permitted only when all raw/provenance contracts
    pass and the recomputed failures are frozen thresholds.  This gives an
    optimization child issue the complete evidence it needs without allowing a
    malformed or substituted result to escape as a benchmark artifact.
    """

    output = output.resolve()
    trace_output = trace_output.resolve()
    if output.is_relative_to(ROOT_DIR) or trace_output.is_relative_to(ROOT_DIR):
        raise MjwarpPpoBenchmarkError(
            "benchmark raw output and trace must be outside the repository"
        )
    if output.parent != trace_output.parent or output == trace_output:
        raise MjwarpPpoBenchmarkError("benchmark artifact and trace must be distinct sibling files")
    output.parent.mkdir(parents=True, exist_ok=True)
    binding = load_binding()
    baseline_plan = _load_plan(ROOT_DIR / DEFAULT_BASELINE_PLAN)
    source = _source_payload()
    if source["commit"] == binding.amendment_freeze_commit:
        raise MjwarpPpoBenchmarkError(
            "candidate benchmark cannot run at the amendment freeze commit"
        )
    hardware = _hardware_payload(baseline_plan)
    preflight_before = _preflight_payload(baseline_plan)
    cases: list[dict[str, Any]] = []
    registered_cases = expected_case_specs(binding)
    total = len(registered_cases)
    for case_id in registered_cases:
        case = _run_registered_case_process(baseline_plan, case_id=case_id)
        cases.append(case)
        print(f"[{len(cases):02d}/{total:02d}] {case['case_id']} PASS", flush=True)
    profile_process, profile_raw = _run_profile_case(baseline_plan, trace_output=trace_output)
    profile = _mapping(profile_raw, "profile summary")
    runtime_delta = _mapping(profile.get("runtime_delta"), "profile runtime delta")
    wrapper_delta = _mapping(profile.get("wrapper_delta"), "profile wrapper delta")
    trace_counts = _mapping(profile.get("trace_counts"), "profile trace counts")
    _assert_capture_source_unchanged(source)
    aggregates = build_aggregates(cases, binding)
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "kind": ARTIFACT_KIND,
        "profile": PROFILE,
        "generated_at": _utc_now(),
        "source": source,
        "threshold": _threshold_payload(binding),
        "hardware": hardware,
        "execution": {
            "process_isolation": True,
            "throughput_iterations": THROUGHPUT_ITERATIONS,
            "contention_iterations": CONTENTION_ITERATIONS,
            "frozen_hydra_overrides": list(binding.hydra_overrides),
            "common_performance_overrides": list(COMMON_PERFORMANCE_OVERRIDES),
            "preflight_before": preflight_before,
            "preflight_after": _preflight_payload(baseline_plan, enforce_cpu_load=False),
        },
        "cases": cases,
        "aggregates": aggregates,
        "device": {
            "gpu_capacity_bytes": binding.gpu_capacity_bytes,
            "peak_gpu_reserved_bytes": max(
                int(case["summary"]["peak_gpu_memory_reserved_bytes"])
                for case in cases
                if case["mode"] == "mjwarp_device"
            ),
            "h2d_per_policy_step": runtime_delta["host_to_device_transfers"] / PROFILE_STEPS,
            "d2h_per_policy_step": runtime_delta["device_to_host_transfers"] / PROFILE_STEPS,
            "host_global_sync_per_policy_step": runtime_delta["global_synchronizations"]
            / PROFILE_STEPS,
            "metrics_materializations": wrapper_delta["finite_metric_materializations"],
            "metrics_device_to_host_bytes": wrapper_delta["finite_metric_device_to_host_bytes"],
            "profiler_reconciled": (
                trace_counts["host_to_device_transfers"],
                trace_counts["device_to_host_transfers"],
                trace_counts["global_synchronizations"],
            )
            == (
                runtime_delta["host_to_device_transfers"],
                runtime_delta["device_to_host_transfers"],
                runtime_delta["global_synchronizations"],
            ),
            "profiler_process": profile_process,
            "profiler_summary": dict(profile),
            "profiler_trace": {"filename": trace_output.name, "sha256": _sha256_path(trace_output)},
        },
    }
    core_errors = _core_validation_errors(
        artifact,
        binding=binding,
        repo_root=ROOT_DIR,
        artifact_path=output,
    )
    artifact["gate"] = {"passed": not core_errors, "errors": list(core_errors)}
    errors = validate_artifact(
        artifact,
        binding=binding,
        repo_root=ROOT_DIR,
        artifact_path=output,
        require_passing_gate=not allow_gate_failure,
    )
    if errors:
        raise MjwarpPpoBenchmarkError(
            "generated benchmark artifact failed validation:\n- " + "\n- ".join(errors)
        )
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--allow-gate-failure",
        action="store_true",
        help=(
            "write a structurally validated diagnostic artifact when only frozen "
            "thresholds fail; it remains invalid as Phase 5 evidence"
        ),
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case-id")
    parser.add_argument("--worker-out", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trace-out", type=Path, default=DEFAULT_TRACE_OUTPUT)
    parser.add_argument("--validate-artifact", type=Path)
    parser.add_argument("--profile-worker", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=PROFILE_STEPS)
    parser.add_argument("--contention-worker", action="store_true")
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--stop-file", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.allow_gate_failure and not args.execute:
        raise MjwarpPpoBenchmarkError("--allow-gate-failure is valid only with --execute")
    if args.list_cases:
        for case_id in expected_case_specs(load_binding()):
            print(case_id)
        return 0
    if args.worker:
        if args.case_id is None or args.worker_out is None:
            raise SystemExit("worker requires --case-id and --worker-out")
        return _execute_worker_case(case_id=args.case_id, output=args.worker_out)
    if args.profile_worker:
        if args.profile_steps <= 0:
            raise SystemExit("--profile-steps must be positive")
        return _device_profile_worker(output=args.trace_out, steps=args.profile_steps)
    if args.contention_worker:
        if args.ready_file is None or args.stop_file is None:
            raise SystemExit("contention worker requires --ready-file and --stop-file")
        return _contention_worker(ready_file=args.ready_file, stop_file=args.stop_file)
    if args.validate_artifact is not None:
        path = args.validate_artifact.resolve()
        artifact = json.loads(path.read_text(encoding="utf-8"))
        errors = validate_artifact(artifact, repo_root=ROOT_DIR, artifact_path=path)
        if errors:
            print("FAIL")
            for error in errors:
                print(f"- {error}")
            return 1
        print(f"PASS {path}")
        return 0
    if not args.execute:
        raise SystemExit("Refusing to run implicitly; pass --execute or --validate-artifact")
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = collect_artifact(
        output=output,
        trace_output=args.trace_out,
        allow_gate_failure=args.allow_gate_failure,
    )
    output.write_text(json.dumps(_json_safe(artifact), indent=2, sort_keys=True), encoding="utf-8")
    if artifact["gate"]["passed"] is not True:
        print(f"DIAGNOSTIC gate failed; wrote non-evidence artifact to {output}")
        return 2
    print(f"PASS wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
