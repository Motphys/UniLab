"""Contract tests for the Issue #705 paired managed-G1 host benchmark."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from benchmark.env import benchmark_managed_g1 as managed_benchmark

from unilab.tools.issue705_phase4_evidence import validate_host_benchmark_artifact

REPO_ROOT = Path(__file__).resolve().parents[2]


def _raw(
    *,
    mode: str,
    batch_size: int,
    repeat_index: int,
    factor: float,
    memory_factor: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    total = [float((1.0 + batch_size / 1024.0) * factor)] * 50
    timing_records: dict[str, list[float]] = {"env_step_total_ms": total}
    if mode == "hand_written":
        timing_records["legacy_env_step_total_ms"] = list(total)
        plan_fingerprint = "legacy.g1-walk-flat.hand-written.v1"
    else:
        materialize = [0.1] * 50
        timing_records["backend_state_materialize_ms"] = materialize
        timing_records["runtime_non_materialize_ms"] = [value - 0.1 for value in total]
        plan_fingerprint = f"{mode}-plan-v1"
    action_seed = 705000 + batch_size * 100 + repeat_index
    return {
        "timing_records": timing_records,
        "memory": {
            "preferred_metric": "uss",
            "total_rss_delta_bytes": int(memory_factor),
            "total_uss_delta_bytes": int(memory_factor),
            "after_benchmark_rss_bytes": int(memory_factor),
            "after_benchmark_uss_bytes": int(memory_factor),
        },
        "resolved_env_config": config,
        "resolved_config_sha256": managed_benchmark.canonical_sha256(config),
        "action_seed": action_seed,
        "reset_seed": action_seed,
        "action_sha256": managed_benchmark._expected_action_sha256(
            batch_size=batch_size,
            sample_count=60,
            seed=action_seed,
        ),
        "executor_key": managed_benchmark.EXPECTED_EXECUTOR_KEYS[mode],
        "plan_fingerprint": plan_fingerprint,
        "backend_identity": "mujoco",
    }


def _artifact(
    *, fused_factor: float = 1.0, fused_memory_factor: float = 1.0
) -> tuple[dict[str, Any], Any, Any]:
    plan = managed_benchmark._load_plan(managed_benchmark.DEFAULT_BASELINE_PLAN)
    binding = managed_benchmark.load_threshold_binding()
    config = managed_benchmark._json_safe(managed_benchmark._host_cfg())
    assert isinstance(config, dict)
    cases: list[dict[str, Any]] = []
    factors = {
        "hand_written": 1.0,
        "managed_reference": 1.02,
        "managed_fused": fused_factor,
    }
    memory_factors = {
        "hand_written": 1.0,
        "managed_reference": 1.02,
        "managed_fused": fused_memory_factor,
    }
    for batch_size in binding.batch_sizes:
        for repeat_index in range(binding.process_repeats):
            for sequence_index, mode in enumerate(managed_benchmark._mode_order(repeat_index)):
                raw = _raw(
                    mode=mode,
                    batch_size=batch_size,
                    repeat_index=repeat_index,
                    factor=factors[mode],
                    memory_factor=memory_factors[mode],
                    config=config,
                )
                cases.append(
                    {
                        "case_id": managed_benchmark._case_id(
                            batch_size=batch_size,
                            repeat_index=repeat_index,
                            mode=mode,
                        ),
                        "mode": mode,
                        "batch_size": batch_size,
                        "repeat_index": repeat_index,
                        "sequence_index": sequence_index,
                        "process": {
                            "run_id": (
                                "00000000-0000-0000-0000-"
                                f"{batch_size * 100 + repeat_index * 10 + managed_benchmark.MODES.index(mode):012d}"
                            ),
                            "pid": 1,
                            "started_at": "2026-07-28T00:00:00+00:00",
                            "duration_sec": 1.0,
                            "return_code": 0,
                            "command": [
                                "uv",
                                "run",
                                "benchmark/env/benchmark_managed_g1.py",
                                "--worker",
                                "--mode",
                                mode,
                                "--batch-size",
                                str(batch_size),
                                "--repeat-index",
                                str(repeat_index),
                            ],
                            "affinity_cpus": list(plan.hardware.affinity_cpus),
                            "env_vars": dict(plan.environment.env_vars),
                            "stdout_sha256": "sha256:" + "b" * 64,
                            "stderr_sha256": "sha256:" + "c" * 64,
                        },
                        "raw": raw,
                        "summary": managed_benchmark.summarize_worker_raw(
                            raw, batch_size=batch_size
                        ),
                    }
                )
    aggregates = managed_benchmark.build_aggregates(cases, binding)
    artifact = {
        "schema_version": managed_benchmark.SCHEMA_VERSION,
        "issue": managed_benchmark.ISSUE,
        "profile": managed_benchmark.PROFILE,
        "threshold": {
            "threshold_set_id": binding.threshold_set_id,
            "manifest_path": binding.manifest_path.as_posix(),
            "manifest_sha256": binding.manifest_sha256,
            "freeze_commit": binding.freeze_commit,
        },
        "candidate": {
            "candidate_commit": "b" * 40,
            "branch": "feat/issue-705-manager-mjwarp",
            "source_dirty": False,
            "source_tree_sha256": "sha256:" + "d" * 64,
            "uv_lock_sha256": "sha256:" + "e" * 64,
            "owner_yaml_sha256": "sha256:" + "f" * 64,
        },
        "hardware": {
            "platform_system": plan.hardware.platform_system,
            "platform_release": "test-release",
            "cpu_model": plan.hardware.cpu_model,
            "cpu_physical_cores": plan.hardware.cpu_physical_cores,
            "cpu_logical_cores": plan.hardware.cpu_logical_cores,
            "affinity_cpus": list(plan.hardware.affinity_cpus),
            "gpu_name": plan.hardware.gpu_name,
            "gpu_uuid": plan.hardware.gpu_uuid,
            "gpu_memory_mib": plan.hardware.gpu_memory_mib,
            "driver_version": plan.hardware.driver_version,
            "cuda_runtime": "test-cuda",
            "torch_version": "test-torch",
            "hostname": "test-host",
        },
        "execution": {
            "process_isolation": True,
            "affinity_cpus": list(plan.hardware.affinity_cpus),
            "env_vars": dict(plan.environment.env_vars),
            "warmup_steps": binding.warmup_steps,
            "measure_steps": binding.measure_steps,
            "mode_order_policy": "repeat-index cyclic rotation of hand_written, managed_reference, managed_fused",
            "preflight_before": {
                "timestamp": "2026-07-28T00:00:00+00:00",
                "load_average_1m": 0.0,
                "load_per_physical_core": 0.0,
                "gpu_compute_processes": [],
                "gpu_samples": [
                    {
                        "utilization_percent": 0,
                        "memory_used_mib": 0,
                        "temperature_c": 0,
                        "pstate": "P0",
                    }
                    for _ in range(plan.preflight.gpu_samples)
                ],
            },
            "preflight_after": {
                "timestamp": "2026-07-28T00:00:00+00:00",
                "load_average_1m": 0.0,
                "load_per_physical_core": 0.0,
                "gpu_compute_processes": [],
                "gpu_samples": [
                    {
                        "utilization_percent": 0,
                        "memory_used_mib": 0,
                        "temperature_c": 0,
                        "pstate": "P0",
                    }
                    for _ in range(plan.preflight.gpu_samples)
                ],
            },
        },
        "cases": cases,
        "aggregates": aggregates,
        "gate": managed_benchmark.build_gate(aggregates, binding),
    }
    return artifact, binding, plan


def test_list_cases_exposes_frozen_three_way_process_matrix(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert managed_benchmark.main(["--list-cases"]) == 0
    case_ids = capsys.readouterr().out.splitlines()
    assert len(case_ids) == 45
    assert len(set(case_ids)) == 45
    assert "host-b128-r0-hand_written" in case_ids
    assert "host-b1024-r4-managed_reference" in case_ids
    assert "host-b4096-r4-managed_fused" in case_ids


def test_runner_refuses_implicit_expensive_execution() -> None:
    with pytest.raises(SystemExit, match="Refusing to run implicitly"):
        managed_benchmark.main([])
    with pytest.raises(managed_benchmark.HostBenchmarkError, match="only with --execute"):
        managed_benchmark.main(["--allow-gate-failure"])


def test_capture_rejects_dirty_candidate_source(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = managed_benchmark._load_plan(managed_benchmark.DEFAULT_BASELINE_PLAN)
    binding = managed_benchmark.load_threshold_binding()
    monkeypatch.setattr(managed_benchmark, "_git", lambda *_: " M src/unilab/manager/runtime.py")

    with pytest.raises(managed_benchmark.HostBenchmarkError, match="clean git worktree"):
        managed_benchmark._source_payload(plan, binding)


def test_host_profile_explicitly_disables_unimplemented_manager_features() -> None:
    config = managed_benchmark._json_safe(managed_benchmark._host_cfg())
    assert isinstance(config, dict)
    assert config["adaptive_chunk_size"] is False
    assert config["chunk_size"] is None
    assert config["domain_rand"]["randomize_kp"] is False
    assert config["domain_rand"]["randomize_kd"] is False
    assert config["noise_config"]["level"] == 0.0
    assert config["max_episode_seconds"] is None


def test_artifact_validator_accepts_complete_paired_frozen_matrix() -> None:
    artifact, binding, plan = _artifact()
    assert artifact["gate"]["passed"] is True
    assert (
        managed_benchmark.validate_artifact(
            artifact,
            binding=binding,
            plan=plan,
            repo_root=None,
        )
        == ()
    )


def test_summary_recompute_tolerance_is_limited_to_throughput_roundoff() -> None:
    expected = {
        "timing_stats_ms": {
            "env_step_total_ms": {
                "count": 2,
                "mean": 1.0,
                "p50": 1.0,
                "p95": 1.0,
                "min": 1.0,
                "max": 1.0,
            }
        },
        "throughput_env_steps_per_sec": 30_000.0,
        "memory": {
            "preferred_metric": "uss",
            "total_preferred_delta_bytes": 1024,
            "after_benchmark_preferred_bytes": 4096,
        },
    }

    portable = deepcopy(expected)
    portable["throughput_env_steps_per_sec"] = math.nextafter(30_000.0, math.inf)
    assert managed_benchmark._summary_matches_recomputed(portable, expected)

    throughput_tamper = deepcopy(expected)
    throughput_tamper["throughput_env_steps_per_sec"] *= 1.0 + 1e-12
    assert not managed_benchmark._summary_matches_recomputed(throughput_tamper, expected)

    timing_tamper = deepcopy(expected)
    timing_tamper["timing_stats_ms"]["env_step_total_ms"]["mean"] = math.nextafter(1.0, math.inf)
    assert not managed_benchmark._summary_matches_recomputed(timing_tamper, expected)

    memory_tamper = deepcopy(expected)
    memory_tamper["memory"]["total_preferred_delta_bytes"] += 1
    assert not managed_benchmark._summary_matches_recomputed(memory_tamper, expected)


@pytest.mark.local_evidence
def test_fused_host_meets_preregistered_gate() -> None:
    """The committed raw matrix must independently re-pass every frozen gate."""

    assert validate_host_benchmark_artifact(root=REPO_ROOT) == ()


def test_artifact_validator_rejects_missing_or_reordered_pair() -> None:
    artifact, binding, plan = _artifact()
    missing = deepcopy(artifact)
    missing["cases"].pop()
    errors = managed_benchmark.validate_artifact(
        missing,
        binding=binding,
        plan=plan,
        repo_root=None,
    )
    assert any("incomplete" in error for error in errors)

    reordered = deepcopy(artifact)
    reordered["cases"][0]["sequence_index"] = 2
    errors = managed_benchmark.validate_artifact(
        reordered,
        binding=binding,
        plan=plan,
        repo_root=None,
    )
    assert any("cyclically interleaved" in error for error in errors)


def test_artifact_validator_rejects_preregistered_performance_gate_failure() -> None:
    artifact, binding, plan = _artifact(fused_factor=1.2)
    assert artifact["gate"]["passed"] is False
    errors = managed_benchmark.validate_artifact(
        artifact,
        binding=binding,
        plan=plan,
        repo_root=None,
    )
    assert any("performance threshold failed" in error for error in errors)
    assert (
        managed_benchmark.validate_artifact(
            artifact,
            binding=binding,
            plan=plan,
            repo_root=None,
            require_passing_gate=False,
        )
        == ()
    )


def test_artifact_validator_rejects_memory_gate_failure() -> None:
    artifact, binding, plan = _artifact(fused_memory_factor=2.0)
    assert artifact["gate"]["passed"] is False
    errors = managed_benchmark.validate_artifact(
        artifact,
        binding=binding,
        plan=plan,
        repo_root=None,
    )
    assert any("performance threshold failed" in error for error in errors)


def test_artifact_validator_rejects_action_or_executor_substitution() -> None:
    artifact, binding, plan = _artifact()
    tampered = deepcopy(artifact)
    tampered["cases"][0]["raw"]["action_seed"] = 1
    tampered["cases"][1]["raw"]["executor_key"] = "invalid.executor.v1"
    errors = managed_benchmark.validate_artifact(
        tampered,
        binding=binding,
        plan=plan,
        repo_root=None,
    )
    assert any("action_seed" in error for error in errors)
    assert any("executor_key" in error for error in errors)


def test_artifact_validator_allows_zero_legacy_component_timings() -> None:
    artifact, binding, plan = _artifact()
    hand_written_case = artifact["cases"][0]
    timing_records = hand_written_case["raw"]["timing_records"]
    timing_records["legacy_env_step_total_ms"] = [0.0] * binding.measure_steps
    hand_written_case["summary"] = managed_benchmark.summarize_worker_raw(
        hand_written_case["raw"], batch_size=128
    )
    assert (
        managed_benchmark.validate_artifact(
            artifact,
            binding=binding,
            plan=plan,
            repo_root=None,
        )
        == ()
    )


def test_artifact_validator_warns_for_hardware_and_rejects_preflight_tampering() -> None:
    artifact, binding, plan = _artifact()
    tampered = deepcopy(artifact)
    tampered["hardware"]["cpu_model"] = "not-the-frozen-host"
    tampered["execution"]["preflight_before"]["gpu_compute_processes"] = [
        {"pid": 1, "process_name": "foreign", "used_memory_mib": 1}
    ]
    with pytest.warns(UserWarning, match="hardware provenance"):
        errors = managed_benchmark.validate_artifact(
            tampered,
            binding=binding,
            plan=plan,
            repo_root=None,
        )
    assert not any("hardware.cpu_model" in error for error in errors)
    assert any("foreign GPU" in error for error in errors)


def test_artifact_validator_rejects_affinity_receipts_that_disagree() -> None:
    artifact, binding, plan = _artifact()
    artifact["execution"]["affinity_cpus"] = [99]

    errors = managed_benchmark.validate_artifact(
        artifact,
        binding=binding,
        plan=plan,
        repo_root=None,
    )

    assert any("recorded hardware provenance" in error for error in errors)


def test_artifact_validator_gates_cpu_load_before_not_after_cpu_workers() -> None:
    artifact, binding, plan = _artifact()
    post_capture_load = deepcopy(artifact)
    post_capture_load["execution"]["preflight_after"]["load_per_physical_core"] = 10.0
    assert (
        managed_benchmark.validate_artifact(
            post_capture_load,
            binding=binding,
            plan=plan,
            repo_root=None,
        )
        == ()
    )

    pre_capture_load = deepcopy(artifact)
    pre_capture_load["execution"]["preflight_before"]["load_per_physical_core"] = 10.0
    errors = managed_benchmark.validate_artifact(
        pre_capture_load,
        binding=binding,
        plan=plan,
        repo_root=None,
    )
    assert any("CPU load exceeds" in error for error in errors)
