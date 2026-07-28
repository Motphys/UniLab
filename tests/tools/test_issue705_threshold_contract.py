from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import pytest
from omegaconf import OmegaConf

from unilab.tools.g1_baseline_provenance import sha256_file
from unilab.tools.issue705_thresholds import (
    CandidateCompatibilityMetrics,
    CandidateDeviceMetrics,
    CandidateDrMetrics,
    CandidateEnvMetrics,
    CandidateGateInput,
    CandidateProvenance,
    CandidateRawEvidence,
    CandidateTrainingMetrics,
    FreezeReceipt,
    ThresholdManifest,
    ThresholdValidationError,
    candidate_gate_errors,
    candidate_provenance_errors,
    load_threshold_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "tests/acceptance/issue_705/g1_threshold_manifest.yaml"
FREEZE_COMMIT = "a" * 40
CANDIDATE_COMMIT = "b" * 40


@pytest.fixture(scope="module")
def manifest() -> ThresholdManifest:
    return load_threshold_manifest(MANIFEST_PATH, repo_root=REPO_ROOT)


@pytest.fixture(scope="module")
def receipt(manifest: ThresholdManifest) -> FreezeReceipt:
    return FreezeReceipt(
        source_path=Path("<memory>"),
        data={
            "threshold_set_id": manifest.data["threshold_set_id"],
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "freeze_commit": FREEZE_COMMIT,
        },
    )


def _write_mutated_manifest(tmp_path: Path, mutate: Callable[[dict[str, Any]], None]) -> Path:
    raw = OmegaConf.to_container(OmegaConf.load(MANIFEST_PATH), resolve=False)
    assert isinstance(raw, dict)
    mutate(raw)
    destination = tmp_path / "thresholds.yaml"
    OmegaConf.save(OmegaConf.create(raw), destination)
    return destination


def _compatibility(**overrides: Any) -> CandidateCompatibilityMetrics:
    values: dict[str, Any] = {
        "identity_exact": True,
        "policy_abi_exact": True,
        "lifecycle_exact": True,
        "unsupported_fail_closed": True,
        "fallback_used": False,
        "mandatory_cases_skipped": 0,
        "manager_obs_max_abs": 0.0,
        "manager_obs_max_rel": 0.0,
        "manager_reward_max_abs": 0.0,
        "manager_reward_max_rel": 0.0,
        "physics_one_step_max_abs": 0.0,
        "physics_one_step_max_rel": 0.0,
        "physics_trajectory_qpos_max_abs": 0.0,
        "physics_trajectory_qpos_max_rel": 0.0,
        "physics_trajectory_qvel_max_abs": 0.0,
        "physics_trajectory_qvel_max_rel": 0.0,
    }
    values.update(overrides)
    return CandidateCompatibilityMetrics(**values)


def _candidate(
    manifest: ThresholdManifest,
    *,
    profile: str = "host_fused",
) -> CandidateGateInput:
    env = {
        int(batch): CandidateEnvMetrics(
            process_repeats=int(reference["process_count"]),
            step_p50_median_ms=float(reference["step_p50_median_ms"]),
            step_p95_median_ms=float(reference["step_p95_median_ms"]),
            throughput_median_env_steps_per_sec=float(
                reference["throughput_median_env_steps_per_sec"]
            ),
            throughput_population_cv=float(reference["throughput_population_cv"]),
            host_memory_median_bytes=int(reference["host_uss_delta_median_bytes"]),
            host_memory_metric="uss",
        )
        for batch, reference in manifest.baseline_reference["env"].items()
    }
    ppo = manifest.baseline_reference["ppo"]
    training = CandidateTrainingMetrics(
        seeds=tuple(ppo["seeds"]),
        failed_seeds=(),
        nan_seeds=(),
        fps_p50_median=float(ppo["fps_p50_median"]),
        reward_auc_median=float(ppo["reward_auc_median"]),
        final_reward_p50_median=float(ppo["final_reward_p50_median"]),
        episode_length_p50_median=float(ppo["episode_length_p50_median"]),
        peak_rss_median_bytes=int(ppo["peak_rss_median_bytes"]),
        peak_gpu_reserved_median_bytes=int(ppo["peak_gpu_reserved_median_bytes"]),
    )
    dr = {}
    for density, modes in manifest.baseline_reference["dr"].items():
        disabled = modes["disabled"]
        enabled = modes["default_kp_kd"]
        dr[float(density)] = CandidateDrMetrics(
            process_repeats=int(disabled["process_count"]),
            actual_rows=int(disabled["actual_rows"]),
            disabled_total_p50_median_ms=float(disabled["total_p50_median_ms"]),
            disabled_total_p95_median_ms=float(disabled["total_p95_median_ms"]),
            enabled_total_p50_median_ms=float(enabled["total_p50_median_ms"]),
            enabled_total_p95_median_ms=float(enabled["total_p95_median_ms"]),
            enabled_extra_resident_bytes=int(enabled["host_uss_delta_median_bytes"])
            - int(disabled["host_uss_delta_median_bytes"]),
            resident_memory_metric="uss",
        )
    device = None
    if profile == "device_resident":
        device = CandidateDeviceMetrics(
            gpu_capacity_bytes=49140 * 1024**2,
            peak_gpu_reserved_bytes=int(ppo["peak_gpu_reserved_median_bytes"]),
            h2d_per_policy_step=0.0,
            d2h_per_policy_step=0.0,
            host_global_sync_per_policy_step=0.0,
            metrics_materializations=1,
            profiler_reconciled=True,
            profiler_trace_refs=("artifacts/profile.sqlite",),
            profiler_trace_sha256s=("sha256:" + "2" * 64,),
        )
    case_ids = tuple(f"case-{index}" for index in range(50))
    return CandidateGateInput(
        profile=profile,
        provenance=CandidateProvenance(
            threshold_set_id=str(manifest.data["threshold_set_id"]),
            threshold_manifest_sha256=sha256_file(MANIFEST_PATH),
            threshold_freeze_commit=FREEZE_COMMIT,
            candidate_commit=CANDIDATE_COMMIT,
            source_dirty=False,
        ),
        raw_evidence=CandidateRawEvidence(
            planned_case_ids=case_ids,
            observed_case_ids=case_ids,
            included_case_ids=case_ids,
            failed_case_ids=(),
            filtered_case_ids=(),
            raw_artifact_sha256="sha256:" + "1" * 64,
            aggregate_recomputed_from_raw=True,
        ),
        environment=env,
        training=training,
        dr=dr,
        compatibility=_compatibility(),
        device=device,
    )


def _errors(
    candidate: CandidateGateInput,
    manifest: ThresholdManifest,
    receipt: FreezeReceipt,
    *,
    ancestor: bool = True,
) -> list[str]:
    return candidate_gate_errors(
        candidate,
        manifest=manifest,
        receipt=receipt,
        is_ancestor=lambda _ancestor, _descendant: ancestor,
    )


def test_frozen_manifest_binds_all_raw_baseline_dimensions(
    manifest: ThresholdManifest,
) -> None:
    assert set(manifest.baseline_reference["env"]) == {"128", "1024", "4096"}
    assert set(manifest.baseline_reference["dr"]) == {"0.01", "0.1", "1.0"}
    assert manifest.baseline_reference["ppo"]["seeds"] == [0, 1, 2, 3, 4]
    assert manifest.baseline_reference["env"]["128"]["throughput_population_cv"] == pytest.approx(
        0.11464559902566135
    )
    assert manifest.gates["performance"]["max_population_cv_by_batch"]["128"] == 0.15


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw.update({"unexpected": True}), "unknown key `unexpected`"),
        (lambda raw: raw["measurement"].pop("ppo_seeds"), "missing key `ppo_seeds`"),
        (
            lambda raw: raw["gates"]["performance"].update({"p50_latency_ratio_max": 1.051}),
            "frozen value is 1.05",
        ),
        (
            lambda raw: raw["measurement"].update({"prohibit_filtering": False}),
            "measurement.prohibit_filtering",
        ),
        (
            lambda raw: raw["baseline_reference"]["env"].pop("128"),
            "baseline_reference.env: keys differ",
        ),
        (
            lambda raw: raw["baseline"].update({"artifact_sha256": "sha256:" + "0" * 64}),
            "expected current",
        ),
    ],
)
def test_manifest_rejects_missing_loosened_or_tampered_data(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    path = _write_mutated_manifest(tmp_path, mutate)
    with pytest.raises(ThresholdValidationError, match=message):
        load_threshold_manifest(path, repo_root=REPO_ROOT)


def test_baseline_candidate_passes_host_and_device_profiles(
    manifest: ThresholdManifest, receipt: FreezeReceipt
) -> None:
    assert not _errors(_candidate(manifest), manifest, receipt)
    assert not _errors(_candidate(manifest, profile="device_resident"), manifest, receipt)


def test_exact_gate_boundaries_pass(manifest: ThresholdManifest, receipt: FreezeReceipt) -> None:
    candidate = _candidate(manifest, profile="device_resident")
    performance = manifest.gates["performance"]
    memory = manifest.gates["memory"]
    env = {}
    for batch, metrics in candidate.environment.items():
        env[batch] = replace(
            metrics,
            step_p50_median_ms=metrics.step_p50_median_ms
            * float(performance["p50_latency_ratio_max"]),
            step_p95_median_ms=metrics.step_p95_median_ms
            * float(performance["p95_latency_ratio_max"]),
            throughput_median_env_steps_per_sec=metrics.throughput_median_env_steps_per_sec
            * float(performance["throughput_ratio_min"]),
            throughput_population_cv=float(performance["max_population_cv_by_batch"][str(batch)]),
            host_memory_median_bytes=int(
                metrics.host_memory_median_bytes * float(memory["host_preferred_metric_ratio_max"])
            ),
        )
    ppo = manifest.baseline_reference["ppo"]
    training_gate = manifest.gates["training"]
    device_peak_boundary = int(ppo["peak_gpu_reserved_median_bytes"]) + int(
        memory["device_peak_reserved_growth_bytes_max"]
    )
    training = replace(
        candidate.training,
        fps_p50_median=float(ppo["fps_p50_median"])
        * float(training_gate["fps_p50_median_ratio_min"]),
        reward_auc_median=float(ppo["reward_auc_median"])
        - float(training_gate["reward_auc_median_drop_max"]),
        final_reward_p50_median=float(ppo["final_reward_p50_median"])
        - float(training_gate["final_reward_p50_median_drop_max"]),
        episode_length_p50_median=float(ppo["episode_length_p50_median"])
        * float(training_gate["episode_length_median_ratio_min"]),
        peak_rss_median_bytes=int(
            float(ppo["peak_rss_median_bytes"]) * float(memory["host_preferred_metric_ratio_max"])
        ),
        peak_gpu_reserved_median_bytes=device_peak_boundary,
    )
    dr_gate = manifest.gates["dr"]
    dr = {
        density: replace(
            metrics,
            enabled_total_p50_median_ms=metrics.disabled_total_p50_median_ms
            * float(dr_gate["enabled_to_disabled_p50_ratio_max"]),
            enabled_total_p95_median_ms=metrics.disabled_total_p95_median_ms
            * float(dr_gate["enabled_to_disabled_p95_ratio_max"]),
            enabled_extra_resident_bytes=int(dr_gate["enabled_extra_resident_bytes_max"]),
        )
        for density, metrics in candidate.dr.items()
    }
    compatibility_gate = manifest.gates["compatibility"]
    compatibility = _compatibility(
        manager_obs_max_abs=compatibility_gate["manager_obs_atol"],
        manager_obs_max_rel=compatibility_gate["manager_obs_rtol"],
        manager_reward_max_abs=compatibility_gate["manager_reward_atol"],
        manager_reward_max_rel=compatibility_gate["manager_reward_rtol"],
        physics_one_step_max_abs=compatibility_gate["physics_one_step_atol"],
        physics_one_step_max_rel=compatibility_gate["physics_one_step_rtol"],
        physics_trajectory_qpos_max_abs=compatibility_gate["physics_trajectory_qpos_atol"],
        physics_trajectory_qpos_max_rel=compatibility_gate["physics_trajectory_qpos_rtol"],
        physics_trajectory_qvel_max_abs=compatibility_gate["physics_trajectory_qvel_atol"],
        physics_trajectory_qvel_max_rel=compatibility_gate["physics_trajectory_qvel_rtol"],
    )
    device = replace(
        candidate.device,
        peak_gpu_reserved_bytes=device_peak_boundary,
    )
    boundary = replace(
        candidate,
        environment=env,
        training=training,
        dr=dr,
        compatibility=compatibility,
        device=device,
    )
    assert not _errors(boundary, manifest, receipt)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_dirty", True, "clean tree"),
        ("threshold_manifest_sha256", "sha256:" + "0" * 64, "manifest_sha256"),
        ("threshold_freeze_commit", "c" * 40, "freeze_commit"),
        ("candidate_commit", FREEZE_COMMIT, "cannot share a commit"),
    ],
)
def test_candidate_provenance_rejects_stale_or_post_hoc_measurement(
    manifest: ThresholdManifest,
    receipt: FreezeReceipt,
    field: str,
    value: Any,
    message: str,
) -> None:
    candidate = _candidate(manifest)
    provenance = replace(candidate.provenance, **{field: value})
    errors = candidate_provenance_errors(
        provenance,
        manifest=manifest,
        receipt=receipt,
        is_ancestor=lambda _ancestor, _descendant: True,
    )
    assert any(message in error for error in errors)


def test_candidate_provenance_requires_freeze_ancestor(
    manifest: ThresholdManifest, receipt: FreezeReceipt
) -> None:
    candidate = _candidate(manifest)
    errors = candidate_provenance_errors(
        candidate.provenance,
        manifest=manifest,
        receipt=receipt,
        is_ancestor=lambda _ancestor, _descendant: False,
    )
    assert errors == ["provenance.candidate_commit: must descend from threshold freeze commit"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda value: replace(value, observed_case_ids=value.observed_case_ids[:-1]),
            "expected 50 cases",
        ),
        (
            lambda value: replace(
                value,
                observed_case_ids=value.observed_case_ids[:-1] + (value.observed_case_ids[0],),
            ),
            "duplicate case IDs",
        ),
        (
            lambda value: replace(value, included_case_ids=value.included_case_ids[:-1]),
            "every observed case",
        ),
        (lambda value: replace(value, failed_case_ids=("case-2",)), "failed cases are FAIL"),
        (
            lambda value: replace(value, filtered_case_ids=("case-0",)),
            "post-hoc filtering is forbidden",
        ),
        (
            lambda value: replace(value, raw_artifact_sha256="not-a-hash"),
            "sha256:<64 lowercase hex>",
        ),
        (
            lambda value: replace(value, aggregate_recomputed_from_raw=False),
            "must be true",
        ),
    ],
)
def test_raw_evidence_fails_missing_duplicate_filtered_or_unreconciled_cases(
    manifest: ThresholdManifest,
    receipt: FreezeReceipt,
    change: Callable[[CandidateRawEvidence], CandidateRawEvidence],
    message: str,
) -> None:
    candidate = _candidate(manifest)
    errors = _errors(
        replace(candidate, raw_evidence=change(candidate.raw_evidence)),
        manifest,
        receipt,
    )
    assert any(message in error for error in errors)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda candidate: replace(candidate, environment={128: candidate.environment[128]}),
            "expected batches",
        ),
        (
            lambda candidate: replace(
                candidate,
                environment={
                    **candidate.environment,
                    128: replace(candidate.environment[128], process_repeats=4),
                },
            ),
            "process_repeats",
        ),
        (
            lambda candidate: replace(
                candidate,
                environment={
                    **candidate.environment,
                    1024: replace(
                        candidate.environment[1024],
                        step_p50_median_ms=candidate.environment[1024].step_p50_median_ms * 1.051,
                    ),
                },
            ),
            "p50_latency_ratio",
        ),
        (
            lambda candidate: replace(
                candidate,
                environment={
                    **candidate.environment,
                    4096: replace(candidate.environment[4096], throughput_population_cv=0.031),
                },
            ),
            "throughput_population_cv",
        ),
        (
            lambda candidate: replace(
                candidate,
                environment={
                    **candidate.environment,
                    128: replace(
                        candidate.environment[128],
                        host_memory_median_bytes=candidate.environment[128].host_memory_median_bytes
                        * 2,
                    ),
                },
            ),
            "host_memory_ratio",
        ),
        (
            lambda candidate: replace(
                candidate,
                environment={
                    **candidate.environment,
                    128: replace(candidate.environment[128], host_memory_metric="rss"),
                },
            ),
            "host_memory_metric",
        ),
    ],
)
def test_environment_gate_fails_on_any_missing_or_regressed_batch(
    manifest: ThresholdManifest,
    receipt: FreezeReceipt,
    change: Callable[[CandidateGateInput], CandidateGateInput],
    message: str,
) -> None:
    assert any(
        message in error for error in _errors(change(_candidate(manifest)), manifest, receipt)
    )


@pytest.mark.parametrize(
    ("training", "message"),
    [
        (lambda value: replace(value, seeds=(0, 1, 2, 3)), "training.seeds"),
        (lambda value: replace(value, failed_seeds=(4,)), "failed_seeds"),
        (lambda value: replace(value, nan_seeds=(2,)), "nan_seeds"),
        (lambda value: replace(value, fps_p50_median=1.0), "fps_p50_median_ratio"),
        (lambda value: replace(value, reward_auc_median=-100.0), "reward_auc_median"),
        (
            lambda value: replace(value, final_reward_p50_median=-1.0),
            "final_reward_p50_median",
        ),
        (
            lambda value: replace(value, episode_length_p50_median=1.0),
            "episode_length_p50_median_ratio",
        ),
        (lambda value: replace(value, peak_rss_median_bytes=10**12), "peak_rss_ratio"),
    ],
)
def test_training_gate_fails_each_seed_behavior_performance_and_memory_dimension(
    manifest: ThresholdManifest,
    receipt: FreezeReceipt,
    training: Callable[[CandidateTrainingMetrics], CandidateTrainingMetrics],
    message: str,
) -> None:
    candidate = _candidate(manifest)
    errors = _errors(replace(candidate, training=training(candidate.training)), manifest, receipt)
    assert any(message in error for error in errors)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda value: replace(value, process_repeats=4), "process_repeats"),
        (lambda value: replace(value, actual_rows=11), "actual_rows"),
        (
            lambda value: replace(value, enabled_total_p50_median_ms=100.0),
            "enabled_to_disabled_p50_ratio",
        ),
        (
            lambda value: replace(value, enabled_total_p95_median_ms=100.0),
            "enabled_to_disabled_p95_ratio",
        ),
        (
            lambda value: replace(value, enabled_extra_resident_bytes=10**12),
            "enabled_extra_resident_bytes",
        ),
        (lambda value: replace(value, resident_memory_metric="rss"), "resident_memory_metric"),
    ],
)
def test_dr_gate_fails_each_density_row_timing_and_memory_dimension(
    manifest: ThresholdManifest,
    receipt: FreezeReceipt,
    change: Callable[[CandidateDrMetrics], CandidateDrMetrics],
    message: str,
) -> None:
    candidate = _candidate(manifest)
    dr = {**candidate.dr, 0.01: change(candidate.dr[0.01])}
    assert any(message in error for error in _errors(replace(candidate, dr=dr), manifest, receipt))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"identity_exact": False}, "identity_exact"),
        ({"policy_abi_exact": False}, "policy_abi_exact"),
        ({"lifecycle_exact": False}, "lifecycle_exact"),
        ({"unsupported_fail_closed": False}, "unsupported_fail_closed"),
        ({"fallback_used": True}, "fallback_used"),
        ({"mandatory_cases_skipped": 1}, "mandatory_cases_skipped"),
        ({"manager_obs_max_abs": 2e-6}, "manager_obs_max_abs"),
        ({"manager_reward_max_rel": 2e-6}, "manager_reward_max_rel"),
        ({"physics_one_step_max_abs": 2e-5}, "physics_one_step_max_abs"),
        ({"physics_trajectory_qpos_max_rel": 1e-2}, "qpos_max_rel"),
        ({"physics_trajectory_qvel_max_abs": 2e-2}, "qvel_max_abs"),
    ],
)
def test_compatibility_gate_fails_each_contract_dimension(
    manifest: ThresholdManifest,
    receipt: FreezeReceipt,
    overrides: dict[str, Any],
    message: str,
) -> None:
    candidate = replace(_candidate(manifest), compatibility=_compatibility(**overrides))
    assert any(message in error for error in _errors(candidate, manifest, receipt))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda value: replace(value, gpu_capacity_bytes=1), "capacity_ratio"),
        (lambda value: replace(value, peak_gpu_reserved_bytes=20 * 1024**3), "growth_bytes"),
        (lambda value: replace(value, h2d_per_policy_step=1.0), "h2d_per_policy_step"),
        (lambda value: replace(value, d2h_per_policy_step=1.0), "d2h_per_policy_step"),
        (
            lambda value: replace(value, host_global_sync_per_policy_step=1.0),
            "host_global_sync_per_policy_step",
        ),
        (lambda value: replace(value, metrics_materializations=-1), "non-negative"),
        (lambda value: replace(value, profiler_reconciled=False), "reconciliation"),
        (lambda value: replace(value, profiler_trace_refs=()), "raw profiler trace"),
        (
            lambda value: replace(
                value,
                profiler_trace_refs=("artifacts/profile.sqlite", "artifacts/profile.sqlite"),
                profiler_trace_sha256s=("sha256:" + "2" * 64,) * 2,
            ),
            "duplicate trace",
        ),
        (
            lambda value: replace(value, profiler_trace_sha256s=("not-a-hash",)),
            "SHA-256 hash",
        ),
    ],
)
def test_device_gate_fails_memory_transfer_sync_and_profiler_dimensions(
    manifest: ThresholdManifest,
    receipt: FreezeReceipt,
    change: Callable[[CandidateDeviceMetrics], CandidateDeviceMetrics],
    message: str,
) -> None:
    candidate = _candidate(manifest, profile="device_resident")
    assert candidate.device is not None
    errors = _errors(replace(candidate, device=change(candidate.device)), manifest, receipt)
    assert any(message in error for error in errors)


def test_profile_payloads_fail_closed(manifest: ThresholdManifest, receipt: FreezeReceipt) -> None:
    host = _candidate(manifest)
    device = _candidate(manifest, profile="device_resident")
    assert any(
        "required" in error for error in _errors(replace(device, device=None), manifest, receipt)
    )
    assert any(
        "must not report" in error
        for error in _errors(replace(host, device=device.device), manifest, receipt)
    )
    assert any(
        "unsupported execution profile" in error
        for error in _errors(replace(host, profile="implicit_auto"), manifest, receipt)
    )
