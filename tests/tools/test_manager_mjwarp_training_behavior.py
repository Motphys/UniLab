"""Contract tests for the frozen Issue #837 paired-seed behavior plan."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, cast

import pytest
import yaml
from tooling.acceptance.training_behavior import (
    ARTIFACT_KIND,
    BENCHMARK_ID,
    CLAIM_ID,
    FREEZE_RECEIPT_PATH,
    ISSUE,
    PARENT_ISSUE,
    PLAN_PATH,
    SCHEMA_VERSION,
    TrainingBehaviorContractError,
    evaluate_training_behavior_cases,
    load_frozen_training_inputs,
    load_training_behavior_freeze_receipt,
    load_training_behavior_plan,
    summarize_training_behavior_raw,
    validate_training_behavior_artifact,
)

pytestmark = [pytest.mark.slow, pytest.mark.local_evidence]

from unilab.tools.g1_baseline_provenance import canonical_sha256, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE5_ARTIFACT = REPO_ROOT / "tests/acceptance/manager_mjwarp/artifacts/phase_5_mjwarp_ppo.json"
ZERO_TRAFFIC_COUNTERS = (
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


def _phase5_candidate_cases() -> list[dict[str, Any]]:
    plan = load_training_behavior_plan(REPO_ROOT / PLAN_PATH, repo_root=REPO_ROOT)
    raw = json.loads(PHASE5_ARTIFACT.read_text(encoding="utf-8"))
    selected = [case for case in raw["cases"] if case.get("lane") == "behavior"]
    result: list[dict[str, Any]] = []
    for index, case in enumerate(selected):
        candidate = {
            "case_id": case["case_id"],
            "seed": case["seed"],
            "sequence_index": index,
            "process_retries": 0,
            "batch_size": case["batch_size"],
            "num_steps_per_env": plan.measurement["num_steps_per_env"],
            "iterations": case["iterations"],
            "mode": case["mode"],
            "raw": copy.deepcopy(case["raw"]),
        }
        candidate["summary"] = summarize_training_behavior_raw(
            cast(Mapping[str, Any], candidate["raw"]),
            plan,
            label=f"phase5/seed={case['seed']}",
        )
        result.append(candidate)
    return result


def _process_receipt(*, plan: Any, command: list[str], run_id: str) -> dict[str, Any]:
    return {
        "return_code": 0,
        "command": command,
        "affinity_cpus": list(plan.hardware["affinity_cpus"]),
        "env_vars": dict(plan.hardware["environment_variables"]),
        "run_id": run_id,
        "duration_sec": 1.0,
        "stdout_sha256": "sha256:" + "1" * 64,
        "stderr_sha256": "sha256:" + "2" * 64,
    }


def _synthetic_candidate_cases(plan: Any) -> list[dict[str, Any]]:
    cases = _phase5_candidate_cases()
    for case in cases:
        seed = int(case["seed"])
        worker_command = [
            "uv",
            "run",
            "benchmark/rl/evaluate_training_behavior.py",
            "--worker",
            "--seed",
            str(seed),
            "--worker-out",
            f"/tmp/manager_mjwarp-seed-{seed}.json",
        ]
        training_command = [
            "uv",
            "run",
            "scripts/train_rsl_rl.py",
            "task=g1_walk_flat/mjwarp",
            f"algo.seed={seed}",
            f"algo.num_envs={plan.measurement['num_envs']}",
            f"algo.num_steps_per_env={plan.measurement['num_steps_per_env']}",
            f"algo.max_iterations={plan.measurement['max_iterations']}",
            f"algo.save_interval={plan.measurement['save_interval']}",
            "algo.capture_performance_diagnostics=true",
            "training.no_play=true",
            "training.logger=tensorboard",
            f"training.log_root=/tmp/manager_mjwarp-seed-{seed}",
            *plan.measurement["hydra_overrides"],
        ]
        case["worker_process"] = _process_receipt(
            plan=plan,
            command=worker_command,
            run_id=f"worker-{seed}",
        )
        case["process"] = _process_receipt(
            plan=plan,
            command=training_command,
            run_id=f"training-{seed}",
        )
        raw = case["raw"]
        config = raw["run_config"]["config"]
        config["algo"]["capture_performance_diagnostics"] = True
        policy = raw["run_config"]["contract_snapshot"]["manager.policy_abi"]
        policy.update(
            {
                "task_key": plan.signature["task_key"],
                "executor_key": plan.signature["executor_key"],
                "plan_fingerprint": plan.signature["task_plan_fingerprint"],
                "policy_abi_fingerprint": plan.signature["policy_abi_fingerprint"],
                "execution_profile": plan.measurement["execution_profile"],
            }
        )
        raw["run_config_sha256"] = canonical_sha256(raw["run_config"])
        run_summary = raw["run_summary"]
        run_summary.update(
            {
                "status": "completed",
                "algo": "ppo",
                "task": plan.measurement["env_name"],
                "sim_backend": plan.measurement["candidate_backend"],
                "configured_seed": seed,
                "effective_seed": seed,
                "completed_iterations": plan.measurement["max_iterations"],
                "total_env_steps": (
                    plan.measurement["num_envs"]
                    * plan.measurement["num_steps_per_env"]
                    * plan.measurement["max_iterations"]
                ),
                "runtime_performance_diagnostics": {
                    "backend_type": plan.measurement["candidate_backend"],
                    "instrumentation_complete": True,
                    "graph": {
                        "active_keys": [
                            {
                                "plan_fingerprint": plan.signature["backend_plan_fingerprint"],
                                "num_envs": plan.measurement["num_envs"],
                            }
                        ]
                    },
                },
                "runtime_traffic_diagnostics": {
                    "policy_steps": (
                        plan.measurement["num_steps_per_env"] * plan.measurement["max_iterations"]
                    ),
                    "instrumentation_complete": True,
                    **{key: 0 for key in ZERO_TRAFFIC_COUNTERS},
                },
                "runtime_stability_diagnostics": {
                    "warm_numeric_allocations": 0,
                    "address_churn": 0,
                },
            }
        )
        case["summary"] = summarize_training_behavior_raw(raw, plan, label=f"synthetic/seed={seed}")
    return cases


def _synthetic_artifact() -> tuple[Any, Any, Mapping[str, Any], Mapping[str, Any], dict[str, Any]]:
    plan = load_training_behavior_plan(REPO_ROOT / PLAN_PATH, repo_root=REPO_ROOT)
    receipt = load_training_behavior_freeze_receipt(
        REPO_ROOT / FREEZE_RECEIPT_PATH,
        plan=plan,
        repo_root=REPO_ROOT,
    )
    threshold, baseline = load_frozen_training_inputs(plan, repo_root=REPO_ROOT)
    cases = _synthetic_candidate_cases(plan)
    pairs, aggregates, errors = evaluate_training_behavior_cases(
        plan=plan,
        threshold=threshold,
        baseline=baseline,
        candidate_cases=cases,
    )
    assert errors == []
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "parent_issue": PARENT_ISSUE,
        "claim_id": CLAIM_ID,
        "kind": ARTIFACT_KIND,
        "generated_at": "2026-07-30T00:00:00+00:00",
        "contract": {
            "plan_path": PLAN_PATH.as_posix(),
            "plan_sha256": sha256_file(REPO_ROOT / PLAN_PATH),
            "freeze_receipt_path": FREEZE_RECEIPT_PATH.as_posix(),
            "freeze_receipt_sha256": sha256_file(REPO_ROOT / FREEZE_RECEIPT_PATH),
            "freeze_commit": receipt.freeze_commit,
            "threshold_manifest_path": plan.source_contract["threshold_manifest"],
            "threshold_manifest_sha256": sha256_file(
                REPO_ROOT / plan.source_contract["threshold_manifest"]
            ),
            "baseline_artifact_path": plan.source_contract["baseline_artifact"],
            "baseline_artifact_sha256": threshold["baseline"]["artifact_sha256"],
        },
        "source": {
            "tree_sha256": "sha256:" + "3" * 64,
            "uv_lock_sha256": "sha256:" + "4" * 64,
            "owner_yaml_sha256": "sha256:" + "5" * 64,
        },
        "hardware": dict(plan.hardware),
        "execution": {
            "case_order": [case["case_id"] for case in cases],
            "process_isolation": True,
            "process_retries": 0,
            "environment_variables": dict(plan.hardware["environment_variables"]),
            "preflight_before": {"gpu_compute_processes": []},
            "preflight_after": {"gpu_compute_processes": []},
        },
        "success_metric": dict(plan.measurement["success_metric"]),
        "cases": cases,
        "pairs": pairs,
        "aggregates": aggregates,
        "gate": {"passed": True, "errors": []},
    }
    return plan, receipt, threshold, baseline, artifact


def test_frozen_plan_matches_phase0_phase7_and_support_signature() -> None:
    plan = load_training_behavior_plan(REPO_ROOT / PLAN_PATH, repo_root=REPO_ROOT)

    assert plan.seeds == (0, 1, 2, 3, 4)
    assert plan.measurement["num_envs"] == 1024
    assert plan.measurement["num_steps_per_env"] == 24
    assert plan.measurement["max_iterations"] == 100
    assert plan.measurement["final_window_iterations"] == 10
    assert plan.measurement["success_metric"]["disposition"] == "not_applicable"


def test_existing_phase5_curves_satisfy_stricter_per_seed_oracle() -> None:
    plan = load_training_behavior_plan(REPO_ROOT / PLAN_PATH, repo_root=REPO_ROOT)
    threshold, baseline = load_frozen_training_inputs(plan, repo_root=REPO_ROOT)
    pairs, aggregates, errors = evaluate_training_behavior_cases(
        plan=plan,
        threshold=threshold,
        baseline=baseline,
        candidate_cases=_phase5_candidate_cases(),
    )

    assert errors == []
    assert [pair["seed"] for pair in pairs] == [0, 1, 2, 3, 4]
    assert all(pair["gate"]["passed"] for pair in pairs)
    assert aggregates["gate"]["passed"] is True


def test_pair_gate_rejects_one_degraded_seed_without_median_masking() -> None:
    plan = load_training_behavior_plan(REPO_ROOT / PLAN_PATH, repo_root=REPO_ROOT)
    threshold, baseline = load_frozen_training_inputs(plan, repo_root=REPO_ROOT)
    cases = _phase5_candidate_cases()
    degraded = cases[3]
    reward = degraded["raw"]["scalars"]["Train/mean_reward"]
    for point in reward:
        point["value"] -= 1.0
    degraded["summary"] = summarize_training_behavior_raw(
        degraded["raw"], plan, label="degraded/seed=3"
    )

    pairs, _, errors = evaluate_training_behavior_cases(
        plan=plan,
        threshold=threshold,
        baseline=baseline,
        candidate_cases=cases,
    )

    seed_three = next(pair for pair in pairs if pair["seed"] == 3)
    assert seed_three["gate"]["passed"] is False
    assert any("seed 3: reward_auc_drop" in error for error in errors)
    assert any("seed 3: final_window_reward_drop" in error for error in errors)


def test_pair_gate_rejects_omitted_reordered_and_duplicate_seed() -> None:
    plan = load_training_behavior_plan(REPO_ROOT / PLAN_PATH, repo_root=REPO_ROOT)
    threshold, baseline = load_frozen_training_inputs(plan, repo_root=REPO_ROOT)
    cases = _phase5_candidate_cases()

    _, _, omitted = evaluate_training_behavior_cases(
        plan=plan,
        threshold=threshold,
        baseline=baseline,
        candidate_cases=cases[:-1],
    )
    assert "candidate omitted or added a training run" in omitted

    reordered_cases = [cases[1], cases[0], *cases[2:]]
    _, _, reordered = evaluate_training_behavior_cases(
        plan=plan,
        threshold=threshold,
        baseline=baseline,
        candidate_cases=reordered_cases,
    )
    assert "candidate case order must exactly match frozen seeds" in reordered

    duplicate_cases = [*cases[:-1], copy.deepcopy(cases[-2])]
    _, _, duplicate = evaluate_training_behavior_cases(
        plan=plan,
        threshold=threshold,
        baseline=baseline,
        candidate_cases=duplicate_cases,
    )
    assert "candidate contains duplicate seeds" in duplicate


def test_raw_curve_rejects_missing_iteration_nan_and_fabricated_success() -> None:
    plan = load_training_behavior_plan(REPO_ROOT / PLAN_PATH, repo_root=REPO_ROOT)
    source = _phase5_candidate_cases()[0]["raw"]

    missing = copy.deepcopy(source)
    missing["scalars"]["Train/mean_reward"].pop()
    with pytest.raises(TrainingBehaviorContractError, match="every iteration"):
        summarize_training_behavior_raw(missing, plan, label="missing")

    non_finite = copy.deepcopy(source)
    non_finite["scalars"]["Train/mean_reward"][50]["value"] = float("nan")
    with pytest.raises(TrainingBehaviorContractError, match="finite value"):
        summarize_training_behavior_raw(non_finite, plan, label="non-finite")

    fabricated = copy.deepcopy(source)
    fabricated["scalars"]["Train/success_rate"] = copy.deepcopy(
        fabricated["scalars"]["Train/mean_reward"]
    )
    with pytest.raises(TrainingBehaviorContractError, match="success metric"):
        summarize_training_behavior_raw(fabricated, plan, label="fabricated")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw.update({"extra": True}), "unknown key `extra`"),
        (
            lambda raw: raw["measurement"].update({"num_envs": 4096}),
            "measurement.num_envs",
        ),
        (
            lambda raw: raw["measurement"]["seeds"].pop(),
            "measurement.seeds",
        ),
        (
            lambda raw: raw["measurement"]["success_metric"].update(
                {"disposition": "available", "scalar_tag": "Train/success_rate"}
            ),
            "success_metric.disposition",
        ),
        (
            lambda raw: raw["gates"].update({"maximum_candidate_fps_population_cv": 0.2}),
            "maximum_candidate_fps_population_cv",
        ),
    ],
)
def test_plan_rejects_schema_budget_success_and_threshold_tamper(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    raw = yaml.safe_load((REPO_ROOT / PLAN_PATH).read_text(encoding="utf-8"))
    mutation(raw)
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(TrainingBehaviorContractError, match=message):
        load_training_behavior_plan(path)


def test_linked_plan_hardware_is_advisory_but_thread_environment_is_strict(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load((REPO_ROOT / PLAN_PATH).read_text(encoding="utf-8"))
    raw["hardware"]["gpu_uuid"] = "GPU-another-host"
    advisory_path = tmp_path / "advisory.yaml"
    advisory_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.warns(UserWarning, match="hardware plan differs"):
        load_training_behavior_plan(advisory_path, repo_root=REPO_ROOT)

    raw = yaml.safe_load((REPO_ROOT / PLAN_PATH).read_text(encoding="utf-8"))
    raw["hardware"]["environment_variables"]["OMP_NUM_THREADS"] = "2"
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(TrainingBehaviorContractError, match="thread environment"):
        load_training_behavior_plan(invalid_path, repo_root=REPO_ROOT)


def test_freeze_receipt_binds_plan_bytes_and_git_history() -> None:
    plan = load_training_behavior_plan(REPO_ROOT / PLAN_PATH, repo_root=REPO_ROOT)

    receipt = load_training_behavior_freeze_receipt(
        REPO_ROOT / FREEZE_RECEIPT_PATH,
        plan=plan,
        repo_root=REPO_ROOT,
    )

    assert receipt.freeze_commit == "6e3008d5ab7461a3320f0dea0a9e466e7c45323e"
    assert receipt.git_history_verified is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("plan_sha256", "sha256:" + "0" * 64, "plan_sha256"),
        ("plan_git_blob", "0" * 40, "plan_git_blob"),
    ],
)
def test_freeze_receipt_rejects_hash_and_git_blob_tamper(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    plan = load_training_behavior_plan(REPO_ROOT / PLAN_PATH, repo_root=REPO_ROOT)
    raw = yaml.safe_load((REPO_ROOT / FREEZE_RECEIPT_PATH).read_text(encoding="utf-8"))
    raw[field] = value
    path = tmp_path / "receipt.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(TrainingBehaviorContractError, match=message):
        load_training_behavior_freeze_receipt(
            path,
            plan=plan,
            repo_root=REPO_ROOT,
        )


@pytest.fixture(scope="module")
def synthetic_artifact_bundle() -> tuple[
    Any, Any, Mapping[str, Any], Mapping[str, Any], dict[str, Any]
]:
    return _synthetic_artifact()


def test_synthetic_artifact_satisfies_full_independent_validator(
    synthetic_artifact_bundle: tuple[
        Any, Any, Mapping[str, Any], Mapping[str, Any], dict[str, Any]
    ],
) -> None:
    plan, receipt, threshold, baseline, artifact = synthetic_artifact_bundle

    report = validate_training_behavior_artifact(
        artifact,
        plan=plan,
        receipt=receipt,
        threshold=threshold,
        baseline=baseline,
        repo_root=None,
    )

    assert report.ok, report.errors


def test_artifact_hardware_difference_is_provenance_advisory(
    synthetic_artifact_bundle: tuple[
        Any, Any, Mapping[str, Any], Mapping[str, Any], dict[str, Any]
    ],
) -> None:
    plan, receipt, threshold, baseline, valid_artifact = synthetic_artifact_bundle
    artifact = copy.deepcopy(valid_artifact)
    artifact["hardware"]["driver_version"] = "580.173.02"

    with pytest.warns(UserWarning, match="hardware provenance"):
        report = validate_training_behavior_artifact(
            artifact,
            plan=plan,
            receipt=receipt,
            threshold=threshold,
            baseline=baseline,
            repo_root=None,
        )

    assert report.ok, report.errors


def test_artifact_affinity_receipts_must_match_recorded_hardware(
    synthetic_artifact_bundle: tuple[
        Any, Any, Mapping[str, Any], Mapping[str, Any], dict[str, Any]
    ],
) -> None:
    plan, receipt, threshold, baseline, valid_artifact = synthetic_artifact_bundle
    artifact = copy.deepcopy(valid_artifact)
    artifact["cases"][0]["process"]["affinity_cpus"] = [99]

    report = validate_training_behavior_artifact(
        artifact,
        plan=plan,
        receipt=receipt,
        threshold=threshold,
        baseline=baseline,
        repo_root=None,
    )

    assert any("recorded hardware provenance" in error for error in report.errors)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda artifact: artifact["cases"][0]["raw"]["scalars"]["Train/mean_reward"].pop(),
            "every iteration exactly once",
        ),
        (
            lambda artifact: artifact["pairs"][0]["metrics"].update({"fps_ratio": 999.0}),
            "artifact.pairs",
        ),
        (
            lambda artifact: artifact["cases"][0]["worker_process"]["command"].__setitem__(
                0, "python"
            ),
            "unexpected process route",
        ),
        (
            lambda artifact: artifact["cases"][0]["process"]["command"].append([]),
            "expected string tokens",
        ),
        (
            lambda artifact: artifact["cases"][0]["raw"]["run_summary"][
                "runtime_traffic_diagnostics"
            ].update({"host_to_device_transfers": 1}),
            "host_to_device_transfers",
        ),
        (
            lambda artifact: artifact["cases"][0]["raw"]["run_summary"][
                "runtime_performance_diagnostics"
            ]["graph"]["active_keys"][0].update(
                {"plan_fingerprint": "mjwarp-device-batch-v1:tampered"}
            ),
            "compiled signature mismatch",
        ),
        (
            lambda artifact: artifact["cases"][0].update({"seed": [0]}),
            "candidate seed set differs",
        ),
        (
            lambda artifact: artifact["cases"][0]["raw"]["run_summary"].update(
                {"peak_gpu_memory_reserved_bytes": None}
            ),
            "peak_gpu_memory_reserved_bytes",
        ),
        (
            lambda artifact: artifact["success_metric"].update(
                {"disposition": "available", "scalar_tag": "Train/success_rate"}
            ),
            "explicitly not applicable",
        ),
    ],
)
def test_artifact_tamper_fails_closed_without_uncaught_type_errors(
    synthetic_artifact_bundle: tuple[
        Any, Any, Mapping[str, Any], Mapping[str, Any], dict[str, Any]
    ],
    mutation: Any,
    message: str,
) -> None:
    plan, receipt, threshold, baseline, valid_artifact = synthetic_artifact_bundle
    artifact = copy.deepcopy(valid_artifact)
    mutation(artifact)

    report = validate_training_behavior_artifact(
        artifact,
        plan=plan,
        receipt=receipt,
        threshold=threshold,
        baseline=baseline,
        repo_root=None,
    )

    assert not report.ok
    assert any(message in error for error in report.errors), report.errors
