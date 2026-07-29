"""Contract and tamper tests for the Issue #705 mjwarp PPO benchmark."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from benchmark.rl import benchmark_mjwarp_ppo as ppo_benchmark
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


def _scalars(*, iterations: int, fps: float = 30_000.0, reward: float = -0.4) -> dict[str, Any]:
    values = {
        "Perf/total_fps": fps,
        "Perf/collection_time": 0.8,
        "Perf/learning_time": 0.06,
        "Train/mean_reward": reward,
        "Train/mean_episode_length": 20.0,
    }
    return {
        tag: [
            {"step": step, "wall_time": float(step), "value": value} for step in range(iterations)
        ]
        for tag, value in values.items()
    }


def _case(
    *,
    binding: ppo_benchmark.BenchmarkBinding,
    case_id: str,
    expected: dict[str, Any],
    ordinal: int,
) -> dict[str, Any]:
    backend, profile = ppo_benchmark._mode_owner(expected["mode"])
    policy_abi = {
        "plan_fingerprint": "managed-g1-plan-v1",
        "policy_abi_fingerprint": "managed-g1-policy-v1",
        "executor_key": "mjwarp.device.g1.v1",
        "execution_profile": ppo_benchmark.PROFILE,
    }
    run_config = {
        "run": {"sim_backend": backend, "device": "cuda:0"},
        "config": {
            "training": {
                "sim_backend": backend,
                **({"execution_profile": profile} if backend == "mjwarp" else {}),
            },
            "algo": {
                "num_envs": expected["batch_size"],
                "num_steps_per_env": binding.behavior_steps_per_env,
                "max_iterations": expected["iterations"],
                "seed": expected["seed"],
            },
            "env": {
                "noise_config": {"level": 0.0},
                "domain_rand": {"randomize_kp": False, "randomize_kd": False},
                "curriculum": {"enabled": False},
            },
        },
        "contract_snapshot": ({"manager.policy_abi": policy_abi} if backend == "mjwarp" else {}),
    }
    peak_rss = 2_000_000_000 if expected["lane"] == "behavior" else 1_000_000_000
    raw = {
        "scalars": _scalars(iterations=expected["iterations"]),
        "memory_samples": [{"elapsed_sec": 0.0, "rss_bytes": peak_rss}],
        "run_config": run_config,
        "run_config_sha256": ppo_benchmark.canonical_sha256(run_config),
        "run_summary": {
            "status": "completed",
            "completed_iterations": expected["iterations"],
            "training_wall_time_sec": 1.0,
            "peak_process_rss_bytes": peak_rss,
            "peak_gpu_memory_allocated_bytes": 200_000_000,
            "peak_gpu_memory_reserved_bytes": 300_000_000,
        },
    }
    case: dict[str, Any] = {
        "case_id": case_id,
        **expected,
        "expected_execution_profile": profile,
        "orchestrator_process": {
            "run_id": f"synthetic-orchestrator-{ordinal}",
            "duration_sec": 2.0,
            "return_code": 0,
            "command": [
                "uv",
                "run",
                "benchmark/rl/benchmark_mjwarp_ppo.py",
                "--worker",
                "--case-id",
                case_id,
            ],
            "affinity_cpus": list(binding.affinity_cpus),
            "env_vars": dict(binding.environment_vars),
            "stdout_sha256": "sha256:" + "a" * 64,
            "stderr_sha256": "sha256:" + "b" * 64,
        },
        "process": {
            "run_id": f"synthetic-worker-{ordinal}",
            "pid": ordinal + 1,
            "started_at": "2026-07-29T00:00:00+00:00",
            "duration_sec": 1.0,
            "return_code": 0,
            "command": [
                "uv",
                "run",
                "scripts/train_rsl_rl.py",
                f"task=g1_walk_flat/{backend}",
                f"algo.seed={expected['seed']}",
                f"algo.num_envs={expected['batch_size']}",
                f"algo.num_steps_per_env={binding.behavior_steps_per_env}",
                f"algo.max_iterations={expected['iterations']}",
                "training.no_play=true",
                "training.logger=tensorboard",
                *binding.hydra_overrides,
                *ppo_benchmark.COMMON_PERFORMANCE_OVERRIDES,
            ],
            "affinity_cpus": list(binding.affinity_cpus),
            "env_vars": dict(binding.environment_vars),
            "stdout_sha256": "sha256:" + "1" * 64,
            "stderr_sha256": "sha256:" + "2" * 64,
        },
        "raw": raw,
        "summary": ppo_benchmark.summarize_training_raw(
            raw, warmup_iterations=binding.warmup_iterations
        ),
    }
    if expected["lane"] == "contention":
        case["contention"] = {
            "load_worker_return_code": 0,
            "load_worker_stdout_sha256": "sha256:" + "3" * 64,
            "load_worker_stderr_sha256": "sha256:" + "4" * 64,
            "load_worker_matrix": [2048, 2048],
        }
    return case


def _artifact() -> tuple[dict[str, Any], ppo_benchmark.BenchmarkBinding]:
    binding = ppo_benchmark.load_binding()
    specs = ppo_benchmark.expected_case_specs(binding)
    cases = [
        _case(binding=binding, case_id=case_id, expected=expected, ordinal=ordinal)
        for ordinal, (case_id, expected) in enumerate(specs.items())
    ]
    artifact: dict[str, Any] = {
        "schema_version": ppo_benchmark.SCHEMA_VERSION,
        "issue": ppo_benchmark.ISSUE,
        "kind": ppo_benchmark.ARTIFACT_KIND,
        "profile": ppo_benchmark.PROFILE,
        "generated_at": "2026-07-29T00:00:00+00:00",
        "source": {
            "commit": "b" * 40,
            "branch": "feat/issue-789-mjwarp-ppo-benchmark",
            "dirty": False,
            "tree_sha256": "sha256:" + "5" * 64,
            "uv_lock_sha256": "sha256:" + "6" * 64,
            "owner_yaml_sha256": "sha256:" + "7" * 64,
        },
        "threshold": {
            "threshold_set_id": binding.threshold_set_id,
            "manifest_path": binding.threshold_manifest_path.as_posix(),
            "manifest_sha256": binding.threshold_manifest_sha256,
            "freeze_commit": binding.threshold_freeze_commit,
        },
        "hardware": {
            "gpu_name": binding.gpu_name,
            "gpu_uuid": binding.gpu_uuid,
            "gpu_memory_mib": binding.gpu_capacity_bytes // 1024**2,
            "driver_version": binding.gpu_driver_version,
            "affinity_cpus": list(binding.affinity_cpus),
        },
        "execution": {
            "process_isolation": True,
            "throughput_iterations": ppo_benchmark.THROUGHPUT_ITERATIONS,
            "contention_iterations": ppo_benchmark.CONTENTION_ITERATIONS,
            "frozen_hydra_overrides": list(binding.hydra_overrides),
            "common_performance_overrides": list(ppo_benchmark.COMMON_PERFORMANCE_OVERRIDES),
            "preflight_before": {"gpu_compute_processes": []},
            "preflight_after": {"gpu_compute_processes": []},
        },
        "cases": cases,
        "aggregates": ppo_benchmark.build_aggregates(cases, binding),
        "device": {
            "gpu_capacity_bytes": binding.gpu_capacity_bytes,
            "peak_gpu_reserved_bytes": 300_000_000,
            "h2d_per_policy_step": 0.0,
            "d2h_per_policy_step": 0.0,
            "host_global_sync_per_policy_step": 0.0,
            "metrics_materializations": 1,
            "metrics_device_to_host_bytes": 1,
            "profiler_reconciled": True,
            "profiler_process": {
                "run_id": "synthetic-profiler",
                "duration_sec": 1.0,
                "return_code": 0,
                "command": [
                    "uv",
                    "run",
                    "benchmark/rl/benchmark_mjwarp_ppo.py",
                    "--profile-worker",
                ],
                "affinity_cpus": list(binding.affinity_cpus),
                "env_vars": dict(binding.environment_vars),
                "stdout_sha256": "sha256:" + "8" * 64,
                "stderr_sha256": "sha256:" + "9" * 64,
            },
            "profiler_summary": {
                "steps": ppo_benchmark.PROFILE_STEPS,
                "runtime_delta": {
                    "host_to_device_transfers": 0,
                    "device_to_host_transfers": 0,
                    "global_synchronizations": 0,
                },
                "wrapper_delta": {
                    "action_publications": ppo_benchmark.PROFILE_STEPS,
                    "finite_metric_materializations": 1,
                    "finite_metric_device_to_host_bytes": 1,
                },
                "trace_counts": {
                    "host_to_device_transfers": 0,
                    "device_to_host_transfers": 0,
                    "global_synchronizations": 0,
                },
                "backend_receipt": {
                    "backend_type": "mjwarp",
                    "execution_profile": ppo_benchmark.PROFILE,
                    "task_plan_fingerprint": "managed-g1-plan-v1",
                    "policy_abi_fingerprint": "managed-g1-policy-v1",
                    "backend_plan_fingerprint": "mjwarp-bound-plan-v1",
                },
            },
            "profiler_trace": {
                "filename": "phase_5_mjwarp_ppo_trace.json",
                "sha256": "sha256:" + "a" * 64,
            },
        },
    }
    _refresh(artifact, binding)
    return artifact, binding


def _refresh(artifact: dict[str, Any], binding: ppo_benchmark.BenchmarkBinding) -> None:
    for case in artifact["cases"]:
        case["summary"] = ppo_benchmark.summarize_training_raw(
            case["raw"], warmup_iterations=binding.warmup_iterations
        )
    artifact["aggregates"] = ppo_benchmark.build_aggregates(artifact["cases"], binding)
    artifact["device"]["peak_gpu_reserved_bytes"] = max(
        case["summary"]["peak_gpu_memory_reserved_bytes"]
        for case in artifact["cases"]
        if case["mode"] == "mjwarp_device"
    )
    core_errors = ppo_benchmark._core_validation_errors(artifact, binding=binding)
    artifact["gate"] = {"passed": not core_errors, "errors": list(core_errors)}


def test_frozen_matrix_is_complete_and_pair_order_is_interleaved(
    capsys: pytest.CaptureFixture[str],
) -> None:
    binding = ppo_benchmark.load_binding()
    specs = ppo_benchmark.expected_case_specs(binding)
    assert len(specs) == 40
    assert set(specs) == set(ppo_benchmark.expected_case_ids(binding))
    assert specs["throughput-mujoco_host-b128-r0"]["sequence_index"] == 0
    assert specs["throughput-mjwarp_device-b128-r1"]["sequence_index"] == 0
    assert tuple(specs)[-1] == "contention-mjwarp_device-b1024-r4"
    assert ppo_benchmark.main(["--list-cases"]) == 0
    assert capsys.readouterr().out.splitlines() == list(specs)


def test_worker_cli_requires_and_forwards_a_registered_case(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit, match="requires --case-id"):
        ppo_benchmark.main(["--worker"])
    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        ppo_benchmark,
        "_execute_worker_case",
        lambda *, case_id, output: calls.append((case_id, output)) or 0,
    )
    output = tmp_path / "case.json"
    assert (
        ppo_benchmark.main(
            [
                "--worker",
                "--case-id",
                "throughput-mujoco_host-b128-r0",
                "--worker-out",
                str(output),
            ]
        )
        == 0
    )
    assert calls == [("throughput-mujoco_host-b128-r0", output)]


def test_complete_artifact_passes_independent_gate_recomputation() -> None:
    artifact, binding = _artifact()
    assert artifact["gate"] == {"passed": True, "errors": []}
    assert ppo_benchmark.validate_artifact(artifact, binding=binding) == ()


def test_gate_is_not_self_referential_and_recorded_result_must_match() -> None:
    artifact, binding = _artifact()
    gate = artifact.pop("gate")
    assert ppo_benchmark._core_validation_errors(artifact, binding=binding) == ()
    assert any(
        "gate must be a mapping" in error
        for error in ppo_benchmark.validate_artifact(artifact, binding=binding)
    )

    artifact["gate"] = gate
    artifact["gate"]["passed"] = False
    assert any(
        "recorded gate" in error
        for error in ppo_benchmark.validate_artifact(artifact, binding=binding)
    )


def test_diagnostic_validation_accepts_only_recomputed_threshold_failures() -> None:
    artifact, binding = _artifact()
    for case in artifact["cases"]:
        if case["lane"] == "throughput" and case["mode"] == "mjwarp_device":
            for point in case["raw"]["scalars"]["Perf/collection_time"]:
                point["value"] = 2.0
    _refresh(artifact, binding)

    assert artifact["gate"]["passed"] is False
    assert any("iteration p50 violates" in error for error in artifact["gate"]["errors"])
    assert ppo_benchmark.validate_artifact(artifact, binding=binding) != ()
    assert (
        ppo_benchmark.validate_artifact(artifact, binding=binding, require_passing_gate=False) == ()
    )

    artifact["hardware"]["gpu_uuid"] = "wrong-gpu"
    _refresh(artifact, binding)
    errors = ppo_benchmark.validate_artifact(artifact, binding=binding, require_passing_gate=False)
    assert any("hardware.gpu_uuid" in error for error in errors)


def test_allow_gate_failure_is_execute_only_and_writes_diagnostic_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(ppo_benchmark.MjwarpPpoBenchmarkError, match="only with --execute"):
        ppo_benchmark.main(["--allow-gate-failure"])

    artifact, binding = _artifact()
    for case in artifact["cases"]:
        if case["lane"] == "behavior":
            for point in case["raw"]["scalars"]["Train/mean_reward"]:
                point["value"] = -10.0
    _refresh(artifact, binding)
    calls: list[dict[str, Any]] = []

    def fake_collect(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return artifact

    monkeypatch.setattr(ppo_benchmark, "collect_artifact", fake_collect)
    output = tmp_path / "diagnostic.json"
    trace = tmp_path / "diagnostic_trace.json"
    assert (
        ppo_benchmark.main(
            [
                "--execute",
                "--allow-gate-failure",
                "--out",
                str(output),
                "--trace-out",
                str(trace),
            ]
        )
        == 2
    )
    assert json.loads(output.read_text(encoding="utf-8"))["gate"] == artifact["gate"]
    assert calls == [
        {"output": output.resolve(), "trace_output": trace, "allow_gate_failure": True}
    ]


def test_matrix_process_and_contention_tampering_fail_closed() -> None:
    artifact, binding = _artifact()
    artifact["cases"].pop()
    artifact["cases"][0]["process"]["return_code"] = 1
    artifact["cases"][0]["orchestrator_process"]["command"] = ["uv", "run", "wrong.py"]
    artifact["cases"][1]["sequence_index"] = 99
    contention = next(case for case in artifact["cases"] if case["lane"] == "contention")
    contention["contention"]["load_worker_matrix"] = [1024, 1024]
    errors = ppo_benchmark.validate_artifact(artifact, binding=binding)
    assert any("matrix is incomplete" in error for error in errors)
    assert any("worker did not succeed" in error for error in errors)
    assert any("registered worker CLI" in error for error in errors)
    assert any("sequence_index differs" in error for error in errors)
    assert any("contention worker matrix" in error for error in errors)


def test_production_benchmark_uses_side_effect_free_evidence_helpers() -> None:
    assert ppo_benchmark._event_scalars.__module__ == "benchmark.issue705.process_evidence"
    assert "benchmark/issue705/benchmark_g1_phase0.py" not in ppo_benchmark.SOURCE_INPUTS
    assert "benchmark/issue705/process_evidence.py" in ppo_benchmark.SOURCE_INPUTS


@pytest.mark.parametrize("backend", ["mujoco", "mjwarp"])
def test_common_performance_overrides_compose_for_both_owner_configs(backend: str) -> None:
    """The paired lane must not rely on an owner-specific DR key layout."""

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(
            config_dir=str(ppo_benchmark.ROOT_DIR / "conf/ppo"), version_base="1.3"
        ):
            cfg = compose(
                config_name="config",
                overrides=[
                    f"task=g1_walk_flat/{backend}",
                    *ppo_benchmark.COMMON_PERFORMANCE_OVERRIDES,
                    "hydra.run.dir=.",
                    "hydra.output_subdir=null",
                    "hydra/job_logging=disabled",
                    "hydra/hydra_logging=disabled",
                ],
            )
        assert cfg.env.noise_config.level == 0.0
        assert cfg.env.curriculum.enabled is False
        assert cfg.env.domain_rand.randomize_kp is False
        assert cfg.env.domain_rand.randomize_kd is False
        if backend == "mjwarp":
            assert cfg.env.mjwarp_nconmax == 128
            assert cfg.env.mjwarp_njmax == 256
    finally:
        GlobalHydra.instance().clear()


def test_scalar_failure_nonfinite_value_and_filtered_iteration_fail_closed() -> None:
    artifact, binding = _artifact()
    first = artifact["cases"][0]
    first["raw"]["scalars"]["Perf/total_fps"][10]["value"] = float("nan")
    behavior = next(case for case in artifact["cases"] if case["lane"] == "behavior")
    behavior["raw"]["scalars"]["Train/mean_reward"].pop()
    errors = ppo_benchmark.validate_artifact(artifact, binding=binding)
    assert any("finite number" in error for error in errors)
    assert any("every iteration exactly once" in error for error in errors)


def test_provenance_owner_policy_and_aggregate_tampering_fail_closed() -> None:
    artifact, binding = _artifact()
    artifact["threshold"]["manifest_sha256"] = "sha256:" + "0" * 64
    artifact["hardware"]["gpu_uuid"] = "wrong-gpu"
    artifact["source"]["dirty"] = True
    device_case = next(case for case in artifact["cases"] if case["mode"] == "mjwarp_device")
    device_case["raw"]["run_config"]["contract_snapshot"]["manager.policy_abi"][
        "execution_profile"
    ] = "host_numpy"
    artifact["aggregates"]["behavior"]["fps_p50_median"] = 999_999.0
    errors = ppo_benchmark.validate_artifact(artifact, binding=binding)
    assert any("threshold differs" in error for error in errors)
    assert any("hardware.gpu_uuid" in error for error in errors)
    assert any("source must be clean" in error for error in errors)
    assert any("execution profile" in error for error in errors)
    assert any("aggregates are not" in error for error in errors)


@pytest.mark.parametrize(
    ("fault", "expected_error"),
    [
        ("throughput", "iteration p50 violates"),
        ("training", "reward AUC violates"),
        ("memory", "reserved capacity ratio violates"),
        ("transfer", "h2d_per_policy_step violates"),
    ],
)
def test_each_frozen_gate_failure_is_recomputed_from_raw(fault: str, expected_error: str) -> None:
    artifact, binding = _artifact()
    if fault == "throughput":
        for case in artifact["cases"]:
            if case["lane"] == "throughput" and case["mode"] == "mjwarp_device":
                for point in case["raw"]["scalars"]["Perf/collection_time"]:
                    point["value"] = 2.0
    elif fault == "training":
        for case in artifact["cases"]:
            if case["lane"] == "behavior":
                for point in case["raw"]["scalars"]["Train/mean_reward"]:
                    point["value"] = -10.0
    elif fault == "memory":
        for case in artifact["cases"]:
            if case["mode"] == "mjwarp_device":
                case["raw"]["run_summary"]["peak_gpu_memory_reserved_bytes"] = int(
                    binding.gpu_capacity_bytes * 0.9
                )
    else:
        profile = artifact["device"]["profiler_summary"]
        profile["runtime_delta"]["host_to_device_transfers"] = ppo_benchmark.PROFILE_STEPS
        profile["trace_counts"]["host_to_device_transfers"] = ppo_benchmark.PROFILE_STEPS
        artifact["device"]["h2d_per_policy_step"] = 1.0
    _refresh(artifact, binding)
    errors = ppo_benchmark.validate_artifact(artifact, binding=binding)
    assert any(expected_error in error for error in errors)
    assert artifact["gate"]["passed"] is False
    assert (
        ppo_benchmark.validate_artifact(artifact, binding=binding, require_passing_gate=False) == ()
    )


def test_profiler_trace_sidecar_hash_and_counts_are_independently_checked(
    tmp_path: Path,
) -> None:
    artifact, binding = _artifact()
    artifact_path = tmp_path / "phase_5_mjwarp_ppo.json"
    trace_path = tmp_path / artifact["device"]["profiler_trace"]["filename"]
    trace = {
        "traceEvents": [
            {
                "name": "issue705.mjwarp_ppo_rollout",
                "cat": "user_annotation",
                "ts": 0.0,
                "dur": 100.0,
            }
        ]
    }
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    artifact["device"]["profiler_trace"]["sha256"] = ppo_benchmark._sha256_path(trace_path)
    core_errors = ppo_benchmark._core_validation_errors(
        artifact, binding=binding, artifact_path=artifact_path
    )
    artifact["gate"] = {"passed": not core_errors, "errors": list(core_errors)}
    assert (
        ppo_benchmark.validate_artifact(artifact, binding=binding, artifact_path=artifact_path)
        == ()
    )

    trace["traceEvents"].append(
        {"name": "Memcpy DtoH", "cat": "cuda_runtime", "ts": 50.0, "dur": 1.0}
    )
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    errors = ppo_benchmark.validate_artifact(artifact, binding=binding, artifact_path=artifact_path)
    assert any("trace hash" in error for error in errors)


def test_expensive_capture_is_explicit_and_requires_clean_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit, match="Refusing to run implicitly"):
        ppo_benchmark.main([])
    monkeypatch.setattr(ppo_benchmark, "_git", lambda *_: " M benchmark/rl/benchmark_mjwarp_ppo.py")
    with pytest.raises(ppo_benchmark.MjwarpPpoBenchmarkError, match="clean git worktree"):
        ppo_benchmark._source_payload()


def test_capture_outputs_must_be_external_siblings(tmp_path: Path) -> None:
    with pytest.raises(ppo_benchmark.MjwarpPpoBenchmarkError, match="sibling"):
        ppo_benchmark.collect_artifact(
            output=tmp_path / "one" / "artifact.json",
            trace_output=tmp_path / "two" / "trace.json",
        )
