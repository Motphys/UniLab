"""End-to-end contract tests for the frozen Issue #829 DR benchmark."""

from __future__ import annotations

import base64
import json
import math
import zlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from benchmark.mjwarp import benchmark_dr_profiles as benchmark

from unilab.tools import mjwarp_dr_performance as contract
from unilab.tools.g1_baseline_provenance import canonical_sha256, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
HIGH_STABLE_RSS_BYTES = 96 * 1024**3

_FIELD_LAYOUT = {
    "actuator_acc0": ((29,), (29,), 4),
    "actuator_biasprm": ((29,), (29, 10), 40),
    "actuator_gainprm": ((29,), (29, 10), 40),
    "body_gravcomp": ((31,), (31,), 4),
    "body_invweight0": ((31,), (31, 2), 8),
    "body_subtreemass": ((31,), (31,), 4),
    "dof_armature": ((35,), (35,), 4),
    "dof_invweight0": ((35,), (35,), 4),
    "tendon_invweight0": ((0,), (0,), 4),
    "tendon_length0": ((0,), (0,), 4),
}
_EVENT_TERMS = {
    "disabled": (),
    "tier_b_pd": ("g1_randomize_kd", "g1_randomize_kp"),
    "tier_c_armature": ("g1_randomize_dof_armature",),
    "tier_c_mixed": (
        "g1_randomize_body_gravity_compensation",
        "g1_randomize_dof_armature",
    ),
}
_DEPENDENCY_VERSIONS = {
    "mujoco-warp": "3.10.0.3",
    "warp-lang": "1.14.0",
    "torch": "2.7.0+cu128",
}
_TRAFFIC_ZERO_KEYS = (
    "host_to_device_transfers",
    "device_to_host_transfers",
    "host_to_device_bytes",
    "device_to_host_bytes",
    "global_synchronizations",
    "backend_allocations",
    "dynamic_getter_calls",
    "selector_resolutions",
    "asset_metadata_reads",
    "registry_lookups",
)


@pytest.fixture(scope="module")
def plan() -> contract.MjwarpDrPerformancePlan:
    return contract.load_mjwarp_dr_performance_plan(REPO_ROOT / contract.PLAN_PATH)


@pytest.fixture(scope="module")
def receipt(
    plan: contract.MjwarpDrPerformancePlan,
) -> contract.MjwarpDrPerformanceFreezeReceipt:
    return contract.load_mjwarp_dr_performance_freeze_receipt(
        REPO_ROOT / contract.FREEZE_RECEIPT_PATH,
        plan=plan,
        repo_root=REPO_ROOT,
        verify_git=False,
    )


def _profile_flags(profile_id: str) -> dict[str, bool]:
    return benchmark.profile_domain_rand_flags(profile_id)


def _config(spec: contract.DrPerformanceCaseSpec) -> dict[str, Any]:
    return {
        "training": {
            "sim_backend": "mjwarp",
            "execution_profile": "device_resident",
        },
        "algo": {"num_envs": spec.batch_size, "seed": spec.seed},
        "env": {"domain_rand": _profile_flags(spec.profile_id)},
    }


def _graph_storage(batch_size: int) -> list[dict[str, Any]]:
    return [
        {
            "name": "state",
            "address": batch_size,
            "shape": [batch_size],
            "dtype": "float32",
            "device": "cuda:0",
        }
    ]


def _graph_storage_fingerprint(batch_size: int) -> str:
    return canonical_sha256(_graph_storage(batch_size)).removeprefix("sha256:")


def _materialization(
    profile: contract.DrPerformanceProfile, batch_size: int
) -> dict[str, Any] | None:
    if not profile.model_targets:
        return None
    fields: list[dict[str, Any]] = []
    for index, name in enumerate(sorted((*profile.direct_fields, *profile.derived_fields))):
        tail_shape, compiled_shape, itemsize = _FIELD_LAYOUT[name]
        shape = (batch_size, *tail_shape)
        model_bytes = math.prod(shape) * itemsize
        fields.append(
            {
                "field_name": name,
                "role": "direct" if name in profile.direct_fields else "derived",
                "materialized_shape": list(shape),
                "materialized_address": 0 if model_bytes == 0 else 10_000 + index,
                "model_bytes": model_bytes,
                "replaced": True,
                "compiled_default_shape": list(compiled_shape),
            }
        )
    return {
        "num_worlds": batch_size,
        "fields": fields,
        "expanded_model_bytes": sum(field["model_bytes"] for field in fields),
        "baseline_bytes": sum(
            math.prod(_FIELD_LAYOUT[name][1]) * 4 for name in profile.direct_fields
        ),
        "storage_generation": 1,
        "storage_fingerprint": _graph_storage_fingerprint(batch_size),
    }


def _graph(
    profile: contract.DrPerformanceProfile, batch_size: int, *, launches: int
) -> dict[str, Any]:
    storage_fingerprint = _graph_storage_fingerprint(batch_size)
    return {
        "backend_type": "mjwarp",
        "execution_mode": "cuda_graph",
        "active_keys": [
            {
                "backend_type": "mjwarp",
                "plan_fingerprint": f"synthetic-plan-{profile.profile_id}",
                "num_envs": batch_size,
                "state_dtype": "float32",
                "control_dtype": "float32",
                "physics_substeps": 3,
                "storage_generation": 1,
                "storage_fingerprint": storage_fingerprint,
                "contract_version": 1,
            }
        ],
        "storage_buffers": contract.encode_mjwarp_dr_graph_storage_buffers(
            _graph_storage(batch_size)
        ),
        "storage_generation": 1,
        "storage_fingerprint": storage_fingerprint,
        "capture_count": 1,
        "launch_count": launches,
        "recapture_count": 0,
        "stale_rejection_count": 0,
        "eager_fallback_count": 0,
        "storage_verification_count": 1,
        "instrumentation_complete": True,
        "contract_version": 1,
    }


def _lifecycle(*, steps: int) -> dict[str, Any]:
    return {
        "runtime_barriers": 2 * steps,
        "step_graph_launches": steps,
        "reset_graph_launches": steps,
        "forward_graph_launches": steps,
        "state_refreshes": 2 * steps,
        "instrumentation_complete": True,
    }


def _performance(
    profile: contract.DrPerformanceProfile,
    batch_size: int,
    *,
    steps: int,
    recompute_offset: int = 0,
) -> dict[str, Any]:
    return {
        "backend_type": "mjwarp",
        "backend_instance_id": f"synthetic-{profile.profile_id}-{batch_size}",
        "mutation_plan_fingerprint": f"synthetic-plan-{profile.profile_id}",
        "model_targets": list(profile.model_targets),
        "recompute_kind": profile.strongest_recompute,
        "direct_fields": list(profile.direct_fields),
        "derived_fields": list(profile.derived_fields),
        "recompute_capture_count": 1 if profile.tier == "C" else 0,
        "recompute_launch_count": recompute_offset + (steps if profile.tier == "C" else 0),
        "materialization": _materialization(profile, batch_size),
        "lifecycle": _lifecycle(steps=steps),
        "graph": _graph(profile, batch_size, launches=3 * steps),
        "instrumentation_complete": True,
    }


def _traffic(
    *, policy_steps: int, step_barriers: int, reset_barriers: int, materializations: int
) -> dict[str, Any]:
    return {
        "policy_steps": policy_steps,
        "step_barriers": step_barriers,
        "reset_barriers": reset_barriers,
        "state_materializations": materializations,
        **{key: 0 for key in _TRAFFIC_ZERO_KEYS},
        "instrumentation_complete": True,
    }


def _event_traffic(profile_id: str) -> dict[str, Any]:
    return {
        term: {
            "host_to_device_transfers": 0,
            "device_to_host_transfers": 0,
            "global_synchronizations": 0,
            "sample_allocations": 0,
        }
        for term in _EVENT_TERMS[profile_id]
    }


def _wrapper(*, actions: int, observations: int, metrics: int) -> dict[str, int]:
    return {
        "action_publications": actions,
        "action_device_to_device_bytes": actions * 4,
        "observation_snapshots": observations,
        "observation_device_to_device_bytes": observations * 4,
        "finite_metric_materializations": metrics,
        "finite_metric_device_to_host_bytes": metrics * 4,
    }


def _stability(
    profile: contract.DrPerformanceProfile, batch_size: int, *, steps: int
) -> dict[str, Any]:
    return {
        "warm_numeric_allocations": 0,
        "address_churn": 0,
        "observations": steps,
        "output_epoch": steps,
        "control_epoch": steps,
        "reset_epoch": steps,
        "buffers": [{"name": "policy", "address": 1}],
        "state_buffers": [{"name": "qpos", "address": 2}],
        "state_epochs": [],
        "traffic": _traffic(
            policy_steps=steps,
            step_barriers=steps,
            reset_barriers=steps,
            materializations=2 * steps,
        ),
        "graph": _graph(profile, batch_size, launches=3 * steps),
        "instrumentation_complete": True,
    }


def _memory_windows(*, samples_per_window: int, rss_step: int = 0) -> list[dict[str, Any]]:
    return [
        {
            "window_index": index,
            "rss_samples_bytes": [HIGH_STABLE_RSS_BYTES + index * rss_step] * samples_per_window,
            "cuda_allocated_samples_bytes": [16 * 1024**2] * samples_per_window,
            "cuda_reserved_samples_bytes": [32 * 1024**2] * samples_per_window,
        }
        for index in range(4)
    ]


def _profiler(spec: contract.DrPerformanceCaseSpec) -> dict[str, Any] | None:
    expected = (
        spec.repeat_index == 0
        and spec.batch_size == 1024
        and ((spec.lane == "reset" and spec.reset_density == 0.1) or spec.lane == "env")
    )
    if not expected:
        return None
    return {
        "scope_name": f"synthetic.{spec.lane}",
        "coverage_lanes": ["reset"] if spec.lane == "reset" else ["env", "train"],
        "events": [],
        "runtime_delta": {
            "host_to_device_transfers": 0,
            "device_to_host_transfers": 0,
            "global_synchronizations": 0,
            "backend_allocations": 0,
        },
        "event_traffic_delta": _event_traffic(spec.profile_id),
    }


def _direct_diagnostics(
    spec: contract.DrPerformanceCaseSpec,
    profile: contract.DrPerformanceProfile,
    *,
    measured: int,
) -> dict[str, Any]:
    performance_before = _performance(profile, spec.batch_size, steps=1, recompute_offset=1)
    performance_after = _performance(
        profile,
        spec.batch_size,
        steps=1 + measured,
        recompute_offset=1,
    )
    runtime_before = _traffic(policy_steps=1, step_barriers=1, reset_barriers=1, materializations=2)
    runtime_after = _traffic(
        policy_steps=1 + measured,
        step_barriers=1 + measured,
        reset_barriers=1 + measured,
        materializations=2 + 2 * measured,
    )
    return {
        "performance_before": performance_before,
        "performance_after": performance_after,
        "stability_before": _stability(profile, spec.batch_size, steps=1),
        "stability_after": _stability(profile, spec.batch_size, steps=1 + measured),
        "runtime_traffic_before": runtime_before,
        "runtime_traffic_after": runtime_after,
        "event_traffic_before": _event_traffic(spec.profile_id),
        "event_traffic_after": _event_traffic(spec.profile_id),
        "wrapper_traffic_before": _wrapper(actions=1, observations=1, metrics=0),
        "wrapper_traffic_after": _wrapper(
            actions=1 + measured,
            observations=1 + measured,
            metrics=0,
        ),
    }


def _reset_masks(spec: contract.DrPerformanceCaseSpec, *, samples: int) -> dict[str, Any]:
    assert spec.reset_density is not None
    rows = int(round(spec.batch_size * spec.reset_density))
    masks = np.zeros((samples, spec.batch_size), dtype=np.bool_)
    masks[:, :rows] = True
    packed = np.packbits(masks, axis=1, bitorder="little")
    return {
        "encoding": "numpy-packbits-base64-little-v1",
        "shape": [samples, spec.batch_size],
        "data": base64.b64encode(packed.tobytes()).decode("ascii"),
    }


def _reset_raw(
    plan: contract.MjwarpDrPerformancePlan,
    spec: contract.DrPerformanceCaseSpec,
) -> dict[str, Any]:
    lane = plan.data["measurement"]["reset"]
    measured = int(lane["measured_barriers"])
    bounds = {
        "mutation_sample": (0.0, 0.1),
        "mutation_commit": (0.1, 0.2),
        "recompute_constants": (0.2, 0.3),
        "reset_forward": (0.3, 1.3),
        "reset_barrier": (0.0, 5.0),
    }
    samples = [
        {
            "sample_index": index,
            "intervals": [
                {"phase": phase, "start_ms": bounds[phase][0], "end_ms": bounds[phase][1]}
                for phase in contract.RESET_PHASES
            ],
        }
        for index in range(measured)
    ]
    phase_samples = {
        phase: [bounds[phase][1] - bounds[phase][0]] * measured for phase in contract.RESET_PHASES
    }
    config = _config(spec)
    profile = plan.profile(spec.profile_id)
    return {
        "resolved_config": config,
        "resolved_config_sha256": canonical_sha256(config),
        "phase_samples_ms": phase_samples,
        "reset_masks": _reset_masks(spec, samples=measured),
        "memory_windows": _memory_windows(
            samples_per_window=int(lane["samples_per_memory_window"])
        ),
        "diagnostics": _direct_diagnostics(spec, profile, measured=measured),
        "timing_lifecycle": {
            "backend_type": "mjwarp",
            "backend_instance_id": "synthetic",
            "placement": "cuda:0",
            "capacity": measured,
            "samples": samples,
            "events_preallocated": measured * len(contract.RESET_PHASES) * 2,
            "priming_synchronizations": 1,
            "materialization_synchronizations": 1,
        },
        "profiler": _profiler(spec),
    }


def _env_raw(
    plan: contract.MjwarpDrPerformancePlan,
    spec: contract.DrPerformanceCaseSpec,
) -> dict[str, Any]:
    lane = plan.data["measurement"]["env"]
    measured = int(lane["measured_steps"])
    config = _config(spec)
    return {
        "resolved_config": config,
        "resolved_config_sha256": canonical_sha256(config),
        "env_step_samples_ms": [5.0] * measured,
        "memory_windows": _memory_windows(
            samples_per_window=int(lane["samples_per_memory_window"])
        ),
        "diagnostics": _direct_diagnostics(
            spec,
            plan.profile(spec.profile_id),
            measured=measured,
        ),
        "timing_lifecycle": {
            "capacity": measured,
            "events_preallocated": measured * 2,
            "priming_synchronizations": 1,
            "materialization_synchronizations": 1,
        },
        "profiler": _profiler(spec),
    }


def _train_raw(
    plan: contract.MjwarpDrPerformancePlan,
    spec: contract.DrPerformanceCaseSpec,
) -> dict[str, Any]:
    lane = plan.data["measurement"]["train"]
    iterations = int(lane["iterations"])
    warmup = int(lane["warmup_iterations"])
    rollout_steps = iterations * int(lane["num_steps_per_env"])
    profile = plan.profile(spec.profile_id)
    before = _performance(profile, spec.batch_size, steps=1, recompute_offset=1)
    after = _performance(
        profile,
        spec.batch_size,
        steps=1 + rollout_steps,
        recompute_offset=1,
    )
    iteration_memory = [
        {
            "iteration": index,
            "rss_bytes": HIGH_STABLE_RSS_BYTES,
            "cuda_allocated_bytes": 16 * 1024**2,
            "cuda_reserved_bytes": 32 * 1024**2,
        }
        for index in range(iterations)
    ]
    post_warmup = iteration_memory[warmup:]
    memory_windows = [
        {
            "window_index": index,
            "rss_samples_bytes": [item["rss_bytes"] for item in post_warmup[index * 2 :][:2]],
            "cuda_allocated_samples_bytes": [
                item["cuda_allocated_bytes"] for item in post_warmup[index * 2 :][:2]
            ],
            "cuda_reserved_samples_bytes": [
                item["cuda_reserved_bytes"] for item in post_warmup[index * 2 :][:2]
            ],
        }
        for index in range(4)
    ]
    scalar_values = {
        "Perf/total_fps": 40_000.0,
        "Perf/collection_time": 0.12,
        "Perf/learning_time": 0.08,
    }
    scalars = {
        tag: [
            {"step": index, "wall_time": float(index), "value": value}
            for index in range(iterations)
        ]
        for tag, value in scalar_values.items()
    }
    config = _config(spec)
    run_config = {"config": config}
    run_summary = {
        "status": "completed",
        "completed_iterations": iterations,
        "training_wall_time_sec": 3.0,
        "peak_process_rss_bytes": HIGH_STABLE_RSS_BYTES,
        "peak_gpu_memory_allocated_bytes": 16 * 1024**2,
        "peak_gpu_memory_reserved_bytes": 32 * 1024**2,
        "runtime_performance_diagnostics_before_training": before,
        "runtime_performance_diagnostics": after,
        "runtime_traffic_diagnostics": _traffic(
            policy_steps=rollout_steps,
            step_barriers=rollout_steps,
            reset_barriers=1 + rollout_steps,
            materializations=1 + 2 * rollout_steps,
        ),
        "runtime_event_traffic_diagnostics": _event_traffic(spec.profile_id),
        "wrapper_traffic_diagnostics": _wrapper(
            actions=rollout_steps,
            observations=1 + rollout_steps,
            metrics=iterations,
        ),
        "runtime_stability_diagnostics": _stability(
            profile,
            spec.batch_size,
            steps=rollout_steps,
        ),
        "logging_traffic_diagnostics": {
            "rollout_steps": rollout_steps,
            "metric_materializations": iterations,
            "metric_device_to_host_bytes": iterations * 4,
        },
        "runner_host_memory_diagnostics": {
            "gc_collected_objects": 0,
            "allocator_trim_attempted": True,
            "allocator_trimmed": True,
        },
        "iteration_memory_diagnostics": iteration_memory,
    }
    return {
        "scalars": scalars,
        "memory_windows": memory_windows,
        "run_config": run_config,
        "run_config_sha256": canonical_sha256(run_config),
        "run_summary": run_summary,
    }


def _process(
    plan: contract.MjwarpDrPerformancePlan,
    spec: contract.DrPerformanceCaseSpec,
) -> dict[str, Any]:
    if spec.lane == "train":
        command = benchmark._train_command(
            plan,
            spec,
            log_root=Path("/tmp/synthetic-logs"),
            hydra_root=Path("/tmp/synthetic-hydra"),
        )
    else:
        command = [
            "uv",
            "run",
            "benchmark/mjwarp/benchmark_dr_profiles.py",
            "--worker",
            "--case-id",
            spec.case_id,
            "--worker-output",
            f"/tmp/{spec.case_id}.json",
        ]
    hardware = plan.data["hardware"]
    return {
        "run_id": f"synthetic-{spec.ordinal}",
        "pid": spec.ordinal + 1,
        "started_at": "2026-07-30T00:00:00+00:00",
        "duration_sec": 1.0,
        "return_code": 0,
        "command": command,
        "affinity_cpus": list(hardware["affinity_cpus"]),
        "env_vars": dict(hardware["environment_variables"]),
        "stdout_sha256": "sha256:" + "1" * 64,
        "stderr_sha256": "sha256:" + "2" * 64,
    }


def _case(
    plan: contract.MjwarpDrPerformancePlan,
    spec: contract.DrPerformanceCaseSpec,
) -> dict[str, Any]:
    raw = (
        _reset_raw(plan, spec)
        if spec.lane == "reset"
        else _env_raw(plan, spec)
        if spec.lane == "env"
        else _train_raw(plan, spec)
    )
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
        "process": _process(plan, spec),
        "raw": raw,
        "summary": contract.summarize_mjwarp_dr_performance_case(raw, spec=spec, plan=plan),
    }


def _artifact(
    plan: contract.MjwarpDrPerformancePlan,
    receipt: contract.MjwarpDrPerformanceFreezeReceipt,
) -> dict[str, Any]:
    specs = contract.expected_mjwarp_dr_performance_cases(plan)
    cases = [_case(plan, spec) for spec in specs]
    aggregates, gate = contract.recompute_mjwarp_dr_performance_evidence(cases, plan=plan)
    dependencies = plan.data["dependencies"]
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "artifact_kind": contract.ARTIFACT_KIND,
        "benchmark_id": contract.BENCHMARK_ID,
        "issue": contract.ISSUE,
        "parent_issue": contract.PARENT_ISSUE,
        "contract": {
            "plan_path": contract.PLAN_PATH.as_posix(),
            "plan_sha256": plan.plan_sha256,
            "freeze_receipt_path": contract.FREEZE_RECEIPT_PATH.as_posix(),
            "freeze_receipt_sha256": sha256_file(receipt.source_path),
            "freeze_commit": receipt.freeze_commit,
        },
        "source": {
            "commit": "a" * 40,
            "git_status": "",
            "source_inputs": list(contract.SOURCE_INPUTS),
            "source_tree_sha256": "sha256:" + "3" * 64,
            "owner_yaml_sha256": "sha256:" + "4" * 64,
            "lockfile_sha256": "sha256:" + "5" * 64,
        },
        "hardware": deepcopy(plan.data["hardware"]),
        "dependencies": {
            "lockfile": dependencies["lockfile"],
            "packages": {
                name: {"constraint": constraint, "version": _DEPENDENCY_VERSIONS[name]}
                for name, constraint in dependencies["packages"].items()
            },
        },
        "execution": {
            "started_at": "2026-07-30T00:00:00+00:00",
            "finished_at": "2026-07-30T00:01:00+00:00",
            "preflight_before": {
                "timestamp": "2026-07-30T00:00:00+00:00",
                "gpu_compute_processes": [],
                "gpu_sample": {
                    "utilization_percent": 0,
                    "memory_used_mib": 0,
                    "temperature_c": 35,
                    "pstate": "P8",
                },
            },
            "preflight_after": {
                "timestamp": "2026-07-30T00:01:00+00:00",
                "gpu_compute_processes": [],
                "gpu_sample": {
                    "utilization_percent": 0,
                    "memory_used_mib": 0,
                    "temperature_c": 35,
                    "pstate": "P8",
                },
            },
            "case_order": [spec.case_id for spec in specs],
            "outcomes": {
                "completed": len(specs),
                "failed": 0,
                "skipped": 0,
                "xfailed": 0,
                "xpassed": 0,
                "filtered": 0,
            },
        },
        "tier_d_eligibility": deepcopy(plan.data["tier_d_eligibility"]),
        "cases": cases,
        "aggregates": aggregates,
        "gate": gate,
    }


@pytest.fixture(scope="module")
def passing_artifact(
    plan: contract.MjwarpDrPerformancePlan,
    receipt: contract.MjwarpDrPerformanceFreezeReceipt,
) -> dict[str, Any]:
    return _artifact(plan, receipt)


def _refresh_evidence(artifact: dict[str, Any], plan: contract.MjwarpDrPerformancePlan) -> None:
    aggregates, gate = contract.recompute_mjwarp_dr_performance_evidence(
        artifact["cases"], plan=plan
    )
    artifact["aggregates"] = aggregates
    artifact["gate"] = gate


def test_canonical_matrix_is_exactly_300_isolated_processes(
    plan: contract.MjwarpDrPerformancePlan,
) -> None:
    specs = contract.expected_mjwarp_dr_performance_cases(plan)

    assert len(specs) == 300
    assert sum(spec.lane == "reset" for spec in specs) == 240
    assert sum(spec.lane == "env" for spec in specs) == 45
    assert sum(spec.lane == "train" for spec in specs) == 15
    assert len({spec.case_id for spec in specs}) == 300
    assert specs[0].case_id == "reset-b128-d0p0000-disabled-r0"
    assert specs[-1].case_id == "train-b1024-tier_c_armature-r4"


def test_complete_synthetic_artifact_accepts_high_but_stable_rss(
    passing_artifact: dict[str, Any],
    plan: contract.MjwarpDrPerformancePlan,
    receipt: contract.MjwarpDrPerformanceFreezeReceipt,
) -> None:
    assert passing_artifact["gate"] == {"passed": True, "errors": []}
    assert all(
        case["summary"]["memory"]["rss_window_medians_bytes"][0] == HIGH_STABLE_RSS_BYTES
        for case in passing_artifact["cases"]
    )
    assert (
        contract.validate_mjwarp_dr_performance_artifact(
            passing_artifact,
            plan=plan,
            receipt=receipt,
            repo_root=None,
        )
        == ()
    )


def test_materialization_accepts_backend_canonical_model_target_order(
    plan: contract.MjwarpDrPerformancePlan,
) -> None:
    profile = plan.profile("tier_b_pd")
    performance = _performance(profile, 128, steps=1)
    performance["model_targets"] = list(reversed(profile.model_targets))

    contract._validate_materialization(
        performance,
        profile=profile,
        batch_size=128,
        path="test.performance",
    )


def test_graph_storage_codec_is_lossless_and_idempotent() -> None:
    storage = _graph_storage(1024)
    encoded = contract.encode_mjwarp_dr_graph_storage_buffers(storage)

    assert contract._decode_mjwarp_dr_graph_storage_buffers(encoded, "test.storage") == storage
    graph = {
        "backend_type": "mjwarp",
        "execution_mode": "cuda_graph",
        "storage_buffers": storage,
    }
    contract.compact_mjwarp_dr_performance_artifact(graph)
    first = deepcopy(graph)
    contract.compact_mjwarp_dr_performance_artifact(graph)

    assert graph == first
    assert graph["storage_buffers"] == encoded


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("digest", "differs from decoded graph storage"),
        ("stream", "invalid zlib payload"),
        ("buffer", "graph buffer keys differ from v1"),
    ),
)
def test_graph_storage_codec_tampering_fails_closed(mutation: str, message: str) -> None:
    encoded = contract.encode_mjwarp_dr_graph_storage_buffers(_graph_storage(128))
    if mutation == "digest":
        encoded["sha256"] = "sha256:" + "0" * 64
    elif mutation == "stream":
        encoded["data"] = base64.b64encode(b"not-zlib").decode("ascii")
    else:
        malformed = [{"name": "state"}]
        payload = json.dumps(malformed, sort_keys=True, separators=(",", ":")).encode("utf-8")
        encoded = {
            "encoding": encoded["encoding"],
            "count": 1,
            "uncompressed_bytes": len(payload),
            "sha256": canonical_sha256(malformed),
            "data": base64.b64encode(zlib.compress(payload, level=9)).decode("ascii"),
        }

    with pytest.raises(ValueError, match=message):
        contract._decode_mjwarp_dr_graph_storage_buffers(encoded, "test.storage")


@pytest.mark.parametrize(
    ("model_targets", "message"),
    (
        (["actuator.pd_damping"], "differs from frozen profile"),
        (
            ["actuator.pd_damping", "actuator.pd_damping"],
            "duplicate targets are not allowed",
        ),
    ),
)
def test_materialization_rejects_incomplete_or_duplicate_model_targets(
    plan: contract.MjwarpDrPerformancePlan,
    model_targets: list[str],
    message: str,
) -> None:
    profile = plan.profile("tier_b_pd")
    performance = _performance(profile, 128, steps=1)
    performance["model_targets"] = model_targets

    with pytest.raises(ValueError, match=message):
        contract._validate_materialization(
            performance,
            profile=profile,
            batch_size=128,
            path="test.performance",
        )


def test_artifact_tampering_fails_closed(
    passing_artifact: dict[str, Any],
    plan: contract.MjwarpDrPerformancePlan,
    receipt: contract.MjwarpDrPerformanceFreezeReceipt,
) -> None:
    artifact = deepcopy(passing_artifact)
    artifact["cases"].pop()
    artifact["execution"]["outcomes"]["completed"] -= 1
    artifact["source"]["source_inputs"] = ["benchmark-only.py"]

    errors = contract.validate_mjwarp_dr_performance_artifact(
        artifact,
        plan=plan,
        receipt=receipt,
        repo_root=None,
    )

    assert any("source_inputs" in error for error in errors)
    assert any("incomplete or filtered" in error for error in errors)
    assert any("expected 300 cases" in error for error in errors)


def test_dependency_versions_must_satisfy_frozen_constraints(
    passing_artifact: dict[str, Any],
    plan: contract.MjwarpDrPerformancePlan,
    receipt: contract.MjwarpDrPerformanceFreezeReceipt,
) -> None:
    artifact = deepcopy(passing_artifact)
    artifact["dependencies"]["packages"]["mujoco-warp"]["version"] = "3.9.9"

    errors = contract.validate_mjwarp_dr_performance_artifact(
        artifact,
        plan=plan,
        receipt=receipt,
        repo_root=None,
    )

    assert any("does not satisfy frozen constraint" in error for error in errors)


def test_preflight_payload_is_exact_and_typed(
    passing_artifact: dict[str, Any],
    plan: contract.MjwarpDrPerformancePlan,
    receipt: contract.MjwarpDrPerformanceFreezeReceipt,
) -> None:
    artifact = deepcopy(passing_artifact)
    del artifact["execution"]["preflight_before"]["gpu_sample"]
    artifact["execution"]["preflight_after"]["gpu_sample"]["utilization_percent"] = "0"

    errors = contract.validate_mjwarp_dr_performance_artifact(
        artifact,
        plan=plan,
        receipt=receipt,
        repo_root=None,
    )

    assert any("preflight_before: missing keys ['gpu_sample']" in error for error in errors)
    assert any("utilization_percent: expected an integer" in error for error in errors)


def test_train_rss_windows_must_match_iteration_diagnostics(
    passing_artifact: dict[str, Any],
    plan: contract.MjwarpDrPerformancePlan,
    receipt: contract.MjwarpDrPerformanceFreezeReceipt,
) -> None:
    artifact = deepcopy(passing_artifact)
    index = next(index for index, case in enumerate(artifact["cases"]) if case["lane"] == "train")
    case = artifact["cases"][index]
    case["raw"]["memory_windows"][0]["rss_samples_bytes"][0] += 1
    spec = contract.expected_mjwarp_dr_performance_cases(plan)[index]
    case["summary"] = contract.summarize_mjwarp_dr_performance_case(
        case["raw"], spec=spec, plan=plan
    )

    errors = contract.validate_mjwarp_dr_performance_artifact(
        artifact,
        plan=plan,
        receipt=receipt,
        repo_root=None,
    )

    assert any("RSS windows differ from trainer raw data" in error for error in errors)


def test_recomputed_memory_gate_failure_is_not_accepted_as_passing(
    passing_artifact: dict[str, Any],
    plan: contract.MjwarpDrPerformancePlan,
    receipt: contract.MjwarpDrPerformanceFreezeReceipt,
) -> None:
    artifact = deepcopy(passing_artifact)
    case = artifact["cases"][0]
    for window in case["raw"]["memory_windows"]:
        window["rss_samples_bytes"] = [
            HIGH_STABLE_RSS_BYTES + window["window_index"] * 8 * 1024**2
        ] * len(window["rss_samples_bytes"])
    spec = contract.expected_mjwarp_dr_performance_cases(plan)[0]
    case["summary"] = contract.summarize_mjwarp_dr_performance_case(
        case["raw"], spec=spec, plan=plan
    )
    _refresh_evidence(artifact, plan)

    assert artifact["gate"]["passed"] is False
    assert any("host_rss_positive_slope" in error for error in artifact["gate"]["errors"])
    assert (
        contract.validate_mjwarp_dr_performance_artifact(
            artifact,
            plan=plan,
            receipt=receipt,
            repo_root=None,
            require_passing_gate=False,
        )
        == ()
    )
    assert contract.validate_mjwarp_dr_performance_artifact(
        artifact,
        plan=plan,
        receipt=receipt,
        repo_root=None,
        require_passing_gate=True,
    )


def test_train_launcher_uses_production_cli_without_broken_hydra_logging_override(
    plan: contract.MjwarpDrPerformancePlan,
) -> None:
    spec = next(
        spec
        for spec in contract.expected_mjwarp_dr_performance_cases(plan)
        if spec.case_id == "train-b1024-tier_c_armature-r0"
    )
    command = benchmark._train_command(
        plan,
        spec,
        log_root=Path("/tmp/logs"),
        hydra_root=Path("/tmp/hydra"),
    )

    assert command[:3] == ["uv", "run", "scripts/train_rsl_rl.py"]
    assert "algo.max_iterations=12" in command
    assert "algo.capture_performance_diagnostics=true" in command
    assert not any(argument.startswith("hydra/job_logging=") for argument in command)
    assert not any(argument.startswith("hydra/hydra_logging=") for argument in command)


@pytest.mark.local_evidence
def test_dr_profiles_meet_preregistered_density_gates(
    plan: contract.MjwarpDrPerformancePlan,
    receipt: contract.MjwarpDrPerformanceFreezeReceipt,
) -> None:
    artifact_path = REPO_ROOT / contract.DEFAULT_ARTIFACT_PATH
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert len(artifact["cases"]) == 300
    assert artifact["gate"] == {"passed": True, "errors": []}
    assert (
        contract.validate_mjwarp_dr_performance_artifact(
            artifact,
            plan=plan,
            receipt=receipt,
            repo_root=REPO_ROOT,
        )
        == ()
    )
