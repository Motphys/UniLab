from __future__ import annotations

import copy
import json
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import pytest
from omegaconf import OmegaConf

from unilab.tools.g1_baseline_provenance import (
    BaselineValidationError,
    G1BaselinePlan,
    build_aggregates,
    canonical_sha256,
    expected_case_ids,
    load_g1_baseline_artifact,
    load_g1_baseline_plan,
    numeric_stats,
    parse_g1_baseline_plan,
    source_tree_sha256,
    summarize_dr_raw,
    summarize_env_raw,
    summarize_ppo_raw,
    validate_g1_baseline_artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = REPO_ROOT / "tests/acceptance/issue_705/g1_mujoco_baseline_plan.yaml"


def _plan() -> G1BaselinePlan:
    plan = load_g1_baseline_plan(PLAN_PATH)
    return replace(plan, source_path=PLAN_PATH.relative_to(REPO_ROOT))


def _plan_raw() -> dict:
    return cast(
        dict,
        copy.deepcopy(OmegaConf.to_container(OmegaConf.load(PLAN_PATH), resolve=False)),
    )


def _memory() -> dict:
    return {
        "preferred_metric": "uss",
        "total_rss_delta_bytes": 100,
        "total_uss_delta_bytes": 90,
        "after_benchmark_rss_bytes": 1000,
        "after_benchmark_uss_bytes": 900,
    }


def _config(**updates) -> dict:
    config = {"backend": "mujoco", "task": "g1_walk_flat"}
    config.update(updates)
    return config


def _process(plan: G1BaselinePlan, run_id: str, pid: int) -> dict:
    return {
        "run_id": str(uuid.uuid5(uuid.NAMESPACE_URL, run_id)),
        "pid": pid,
        "started_at": "2026-07-28T00:00:00+00:00",
        "duration_sec": 1.0,
        "return_code": 0,
        "command": ["uv", "run", "worker"],
        "affinity_cpus": list(plan.hardware.affinity_cpus),
        "env_vars": dict(plan.environment.env_vars),
        "stdout_sha256": f"sha256:{'1' * 64}",
        "stderr_sha256": f"sha256:{'2' * 64}",
    }


def _env_raw(plan: G1BaselinePlan, batch_size: int) -> dict:
    config = _config(batch_size=batch_size)
    values = [float(index + 1) for index in range(plan.env_lane.measure_steps)]
    return {
        "timing_records": {
            "env_step_total_ms": values,
            "backend_physics_ms": [value * 0.5 for value in values],
        },
        "memory": _memory(),
        "resolved_env_config": config,
        "resolved_config_sha256": canonical_sha256(config),
    }


def _dr_raw(plan: G1BaselinePlan, mode: str, density: float) -> dict:
    config = _config(mode=mode, density=density)
    rows = max(1, round(plan.dr_lane.num_envs * density))
    return {
        "reset_samples": [
            {
                "requested_rows": rows,
                "actual_rows": rows,
                "timing": {
                    "dr_reset_total_ms": float(index + 1),
                    "dr_reset_set_state_ms": float(index + 1) * 0.5,
                },
            }
            for index in range(plan.dr_lane.measure_resets)
        ],
        "memory": _memory(),
        "resolved_env_config": config,
        "resolved_config_sha256": canonical_sha256(config),
    }


def _ppo_raw(plan: G1BaselinePlan, seed: int) -> dict:
    run_config = {
        "config": {
            "training": {"sim_backend": "mujoco"},
            "algo": {
                "seed": seed,
                "num_envs": plan.ppo_lane.num_envs,
                "num_steps_per_env": plan.ppo_lane.num_steps_per_env,
                "max_iterations": plan.ppo_lane.max_iterations,
            },
        }
    }
    scalars = {
        tag: [
            {
                "step": index,
                "wall_time": 1000.0 + index,
                "value": float(index + 1),
            }
            for index in range(plan.ppo_lane.max_iterations)
        ]
        for tag in plan.ppo_lane.required_scalar_tags
    }
    return {
        "scalars": scalars,
        "memory_samples": [
            {"elapsed_sec": 0.0, "rss_bytes": 1000},
            {"elapsed_sec": 1.0, "rss_bytes": 2000},
        ],
        "run_config": run_config,
        "run_config_sha256": canonical_sha256(run_config),
        "run_summary": {
            "status": "completed",
            "completed_iterations": plan.ppo_lane.max_iterations,
            "peak_gpu_memory_allocated_bytes": 3000,
            "peak_gpu_memory_reserved_bytes": 4000,
        },
    }


def _artifact() -> dict:
    plan = _plan()
    cases: list[dict] = []
    pid = 1000
    for batch_size in plan.env_lane.batch_sizes:
        for repeat in range(plan.env_lane.process_repeats):
            raw = _env_raw(plan, batch_size)
            case_id = f"env-b{batch_size}-r{repeat}"
            cases.append(
                {
                    "case_id": case_id,
                    "lane": "env",
                    "repeat_index": repeat,
                    "seed": None,
                    "batch_size": batch_size,
                    "dr_mode": None,
                    "reset_density": None,
                    "process": _process(plan, f"run-{case_id}", pid),
                    "raw": raw,
                    "summary": summarize_env_raw(raw, batch_size),
                }
            )
            pid += 1
    for mode in plan.dr_lane.modes:
        for density in plan.dr_lane.reset_densities:
            density_id = f"{density:.4f}".replace(".", "p")
            for repeat in range(plan.dr_lane.process_repeats):
                raw = _dr_raw(plan, mode, density)
                case_id = f"dr-{mode}-d{density_id}-r{repeat}"
                cases.append(
                    {
                        "case_id": case_id,
                        "lane": "dr",
                        "repeat_index": repeat,
                        "seed": None,
                        "batch_size": plan.dr_lane.num_envs,
                        "dr_mode": mode,
                        "reset_density": density,
                        "process": _process(plan, f"run-{case_id}", pid),
                        "raw": raw,
                        "summary": summarize_dr_raw(raw),
                    }
                )
                pid += 1
    for seed in plan.ppo_lane.seeds:
        raw = _ppo_raw(plan, seed)
        case_id = f"ppo-seed-{seed}"
        cases.append(
            {
                "case_id": case_id,
                "lane": "ppo",
                "repeat_index": None,
                "seed": seed,
                "batch_size": plan.ppo_lane.num_envs,
                "dr_mode": None,
                "reset_density": None,
                "process": _process(plan, f"run-{case_id}", pid),
                "raw": raw,
                "summary": summarize_ppo_raw(raw, plan.ppo_lane),
            }
        )
        pid += 1
    return {
        "schema_version": 1,
        "issue": 705,
        "baseline_id": plan.baseline_id,
        "generated_at": "2026-07-28T00:00:00+00:00",
        "plan": {"path": plan.source_path.as_posix(), "sha256": f"sha256:{'3' * 64}"},
        "source": {
            "commit": "a" * 40,
            "branch": "test",
            "dirty": False,
            "tree_sha256": f"sha256:{'4' * 64}",
            "uv_lock_sha256": f"sha256:{'5' * 64}",
            "owner_yaml_sha256": f"sha256:{'6' * 64}",
        },
        "hardware": {
            **asdict(plan.hardware),
            "affinity_cpus": list(plan.hardware.affinity_cpus),
            "platform_release": "test-kernel",
            "cuda_runtime": "12.8",
            "torch_version": "2.8.0",
            "hostname": "test-host",
        },
        "execution": {
            "process_isolation": True,
            "affinity_cpus": list(plan.hardware.affinity_cpus),
            "env_vars": dict(plan.environment.env_vars),
            "hydra_overrides": list(plan.environment.hydra_overrides),
        },
        "cases": cases,
        "aggregates": build_aggregates(cases),
    }


def _errors(raw: dict) -> tuple[str, ...]:
    return validate_g1_baseline_artifact(raw, _plan())


def test_frozen_plan_declares_complete_matrix() -> None:
    plan = _plan()

    assert len(expected_case_ids(plan)) == 50
    assert plan.env_lane.batch_sizes == (128, 1024, 4096)
    assert plan.ppo_lane.seeds == (0, 1, 2, 3, 4)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw.update({"unknown": True}), "plan: unknown key `unknown`"),
        (lambda raw: raw["env_lane"].update({"batch_sizes": [128, 1024]}), "must remain"),
        (lambda raw: raw["ppo_lane"].update({"seeds": [0, 1, 2]}), "must remain"),
        (
            lambda raw: raw["environment"]["hydra_overrides"].pop(),
            "does not match the frozen benchmark profile",
        ),
        (
            lambda raw: raw["source_inputs"].append("../outside"),
            "invalid repository-relative path",
        ),
        (
            lambda raw: raw["dr_lane"].update({"reset_densities": [0.0, 1.0]}),
            "must be in (0, 1]",
        ),
    ],
)
def test_plan_rejects_mutable_or_ambiguous_matrix(mutate, message: str) -> None:
    raw = _plan_raw()
    mutate(raw)

    with pytest.raises(BaselineValidationError) as exc_info:
        parse_g1_baseline_plan(raw)

    assert any(message in error for error in exc_info.value.errors)


def test_valid_artifact_recomputes_all_fifty_cases() -> None:
    artifact = _artifact()

    assert _errors(artifact) == ()
    assert artifact["aggregates"]["env"]["case_count"] == 15
    assert artifact["aggregates"]["dr"]["case_count"] == 30
    assert artifact["aggregates"]["ppo"]["case_count"] == 5


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw.update({"unexpected": 1}), "artifact: unknown key `unexpected`"),
        (lambda raw: raw["cases"].pop(), "matrix mismatch"),
        (
            lambda raw: raw["cases"][1]["process"].update(
                {"run_id": raw["cases"][0]["process"]["run_id"]}
            ),
            "duplicate process run IDs",
        ),
        (
            lambda raw: raw["cases"][0]["raw"]["timing_records"]["env_step_total_ms"].pop(),
            "expected 50 samples",
        ),
        (
            lambda raw: raw["cases"][0]["summary"].update(
                {"throughput_env_steps_per_sec": 999999.0}
            ),
            "does not recompute from raw samples",
        ),
        (
            lambda raw: raw["cases"][15]["raw"]["reset_samples"][0].update({"actual_rows": 0}),
            "row isolation failed",
        ),
        (
            lambda raw: raw["cases"][-1]["raw"]["scalars"].pop("Perf/total_fps"),
            "missing scalar tag",
        ),
        (
            lambda raw: raw["cases"][-1]["raw"]["run_config"]["config"]["algo"].update(
                {"num_envs": 1}
            ),
            "run_config_sha256",
        ),
        (
            lambda raw: raw["aggregates"]["env"].update({"case_count": 1}),
            "does not recompute",
        ),
        (
            lambda raw: raw["hardware"].update({"gpu_uuid": "another-gpu"}),
            "expected frozen value",
        ),
    ],
)
def test_artifact_rejects_missing_tampered_or_filtered_evidence(mutate, message: str) -> None:
    artifact = _artifact()
    mutate(artifact)

    assert any(message in error for error in _errors(artifact))


def test_numeric_stats_records_p50_p95_and_raw_count() -> None:
    assert numeric_stats([1.0, 2.0, 3.0, 4.0]) == {
        "count": 4,
        "mean": 2.5,
        "p50": 2.5,
        "p95": pytest.approx(3.85),
        "min": 1.0,
        "max": 4.0,
    }


def test_source_tree_hash_changes_with_any_registered_input(tmp_path: Path) -> None:
    (tmp_path / "owner.yaml").write_text("value: 1\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/code.py").write_text("VALUE = 1\n", encoding="utf-8")
    before = source_tree_sha256(tmp_path, ["owner.yaml", "src"])

    (tmp_path / "src/code.py").write_text("VALUE = 2\n", encoding="utf-8")

    assert source_tree_sha256(tmp_path, ["owner.yaml", "src"]) != before


def test_load_artifact_normalizes_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(BaselineValidationError, match="cannot load JSON"):
        load_g1_baseline_artifact(path, _plan())


def test_artifact_strings_are_data_not_commands(tmp_path: Path) -> None:
    artifact = _artifact()
    marker = tmp_path / "must-not-exist"
    artifact["cases"][0]["process"]["command"] = ["uv", "run", "touch", str(marker)]

    assert _errors(artifact) == ()
    assert not marker.exists()


def test_artifact_fixture_is_json_serializable() -> None:
    json.dumps(_artifact(), sort_keys=True)
