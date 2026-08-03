"""Synthetic receipt and diagnostic-tamper tests for the task rollout gate."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import pytest
from tooling.acceptance.task_rollout import (
    ROLLOUT_PLAN_PATH,
    TaskRolloutEntry,
    load_task_rollout_plan,
)
from tooling.acceptance.task_rollout_run import (
    TaskRolloutRunValidationReport,
    validate_task_rollout_run,
)

ROOT_DIR = Path(__file__).resolve().parents[2]


def _entry() -> TaskRolloutEntry:
    plan = load_task_rollout_plan(ROOT_DIR / ROLLOUT_PLAN_PATH)
    assert len(plan.entries) == 1
    return plan.entries[0]


def _receipts(tmp_path: Path) -> tuple[TaskRolloutEntry, Path, dict[str, Any], dict[str, Any]]:
    entry = _entry()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "model_0.pt").write_bytes(b"checkpoint")
    signature = entry.rollout_compiled_signature
    graph_key = {
        "backend_type": "mjwarp",
        "plan_fingerprint": signature.backend_plan_fingerprint,
        "num_envs": entry.num_envs,
        "state_dtype": "float32",
        "control_dtype": "float32",
        "physics_substeps": 3,
        "storage_generation": 0,
        "storage_fingerprint": "storage-fingerprint",
        "contract_version": 1,
    }

    def graph(launch_count: int) -> dict[str, Any]:
        return {
            "backend_type": "mjwarp",
            "execution_mode": "cuda_graph",
            "active_keys": [deepcopy(graph_key)],
            "capture_count": 1,
            "launch_count": launch_count,
            "recapture_count": 0,
            "stale_rejection_count": 0,
            "eager_fallback_count": 0,
            "storage_verification_count": 1,
            "instrumentation_complete": True,
        }

    def lifecycle(step: int, reset: int, forward: int) -> dict[str, Any]:
        return {
            "runtime_barriers": step + reset,
            "step_graph_launches": step,
            "reset_graph_launches": reset,
            "forward_graph_launches": forward,
            "state_refreshes": step + forward,
            "instrumentation_complete": True,
        }

    traffic = {
        "policy_steps": 2,
        "step_barriers": 2,
        "reset_barriers": 1,
        "host_to_device_transfers": 0,
        "device_to_host_transfers": 0,
        "host_to_device_bytes": 0,
        "device_to_host_bytes": 0,
        "global_synchronizations": 0,
        "backend_allocations": 0,
        "state_materializations": 3,
        "dynamic_getter_calls": 0,
        "selector_resolutions": 0,
        "asset_metadata_reads": 0,
        "registry_lookups": 0,
        "instrumentation_complete": True,
    }
    performance = {
        "backend_type": "mjwarp",
        "model_targets": [],
        "recompute_kind": "none",
        "direct_fields": [],
        "derived_fields": [],
        "recompute_capture_count": 0,
        "recompute_launch_count": 0,
        "materialization": None,
        "lifecycle": lifecycle(2, 1, 1),
        "graph": graph(4),
        "instrumentation_complete": True,
    }
    before = {
        "backend_type": "mjwarp",
        "model_targets": [],
        "recompute_kind": "none",
        "direct_fields": [],
        "derived_fields": [],
        "recompute_capture_count": 0,
        "recompute_launch_count": 0,
        "materialization": None,
        "lifecycle": lifecycle(0, 1, 1),
        "graph": graph(2),
        "instrumentation_complete": True,
    }
    run_config = {
        "run": {
            "algo": "ppo",
            "task": entry.env_name,
            "sim_backend": entry.backend,
            "device": "cuda",
            "configured_seed": 0,
            "effective_seed": 0,
        },
        "config": {
            "training": {
                "task_name": entry.env_name,
                "sim_backend": entry.backend,
                "execution_profile": entry.execution_profile,
                "no_play": True,
                "play_render_mode": "none",
            },
            "algo": {
                "seed": 0,
                "num_envs": entry.num_envs,
                "num_steps_per_env": entry.num_steps_per_env,
                "max_iterations": entry.max_iterations,
                "runtime_impl": entry.runtime_impl,
                "runtime_resolver": entry.runtime_resolver,
                "capture_performance_diagnostics": True,
            },
            "env": {
                "domain_rand": {name: False for name in entry.disabled_domain_rand},
                "noise_config": {"level": 0},
            },
        },
        "contract_snapshot": {
            "manager.policy_abi": {
                "task_key": signature.task_key,
                "executor_key": signature.executor_key,
                "plan_fingerprint": signature.task_plan_fingerprint,
                "policy_abi_fingerprint": signature.policy_abi_fingerprint,
                "execution_profile": entry.execution_profile,
            }
        },
    }
    run_summary = {
        "status": "completed",
        "algo": "ppo",
        "task": entry.env_name,
        "sim_backend": entry.backend,
        "configured_seed": 0,
        "effective_seed": 0,
        "completed_iterations": 1,
        "total_env_steps": 256,
        "training_wall_time_sec": 1.0,
        "wall_time_sec": 1.5,
        "peak_process_rss_bytes": 100,
        "peak_gpu_memory_allocated_bytes": 200,
        "peak_gpu_memory_reserved_bytes": 300,
        "last_checkpoint": str(run_dir / "model_0.pt"),
        "runtime_performance_diagnostics": performance,
        "runtime_performance_diagnostics_before_training": before,
        "runtime_traffic_diagnostics": traffic,
        "runtime_event_traffic_diagnostics": {},
        "wrapper_traffic_diagnostics": {
            "action_publications": 2,
            "observation_snapshots": 3,
            "finite_metric_materializations": 1,
            "finite_metric_device_to_host_bytes": 10,
        },
        "logging_traffic_diagnostics": {
            "rollout_steps": 2,
            "metric_materializations": 1,
            "metric_device_to_host_bytes": 10,
        },
        "runtime_stability_diagnostics": {
            "buffers": [{"name": "runtime.control"}],
            "state_buffers": [{"name": "backend.state.qpos"}],
            "warm_numeric_allocations": 0,
            "address_churn": 0,
            "observations": 2,
            "instrumentation_complete": True,
            "traffic": traffic,
            "graph": graph(4),
        },
        "iteration_memory_diagnostics": [{"iteration": 0}],
    }
    return entry, run_dir, run_config, run_summary


def _validate(
    entry: TaskRolloutEntry,
    run_dir: Path,
    run_config: dict[str, Any],
    run_summary: dict[str, Any],
) -> TaskRolloutRunValidationReport:
    return validate_task_rollout_run(
        entry,
        seed=0,
        run_dir=run_dir,
        run_config=run_config,
        run_summary=run_summary,
        stdout="Using device: cuda\nLearning iteration 0/1\nCollection time: 1\nLearning time: 1\n",
    )


def test_valid_task_rollout_receipt_passes(tmp_path: Path) -> None:
    entry, run_dir, run_config, run_summary = _receipts(tmp_path)

    report = _validate(entry, run_dir, run_config, run_summary)

    assert report.ok, report.errors


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (
            lambda summary: summary["runtime_traffic_diagnostics"].update(
                {"host_to_device_transfers": 1}
            ),
            "host_to_device_transfers",
        ),
        (
            lambda summary: summary["runtime_performance_diagnostics"]["graph"].update(
                {"eager_fallback_count": 1}
            ),
            "eager_fallback_count",
        ),
        (
            lambda summary: summary["runtime_stability_diagnostics"].update({"address_churn": 1}),
            "address_churn",
        ),
        (
            lambda summary: summary["runtime_performance_diagnostics"].update(
                {"model_targets": ["joint.kp"]}
            ),
            "model_targets",
        ),
        (
            lambda config: config["contract_snapshot"]["manager.policy_abi"].update(
                {"plan_fingerprint": "manager-task-contract-v1:tampered"}
            ),
            "plan_fingerprint",
        ),
        (
            lambda config: config["contract_snapshot"]["manager.policy_abi"].update(
                {"executor_key": None}
            ),
            "executor_key",
        ),
    ],
)
def test_task_rollout_receipt_diagnostics_fail_closed(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    fragment: str,
) -> None:
    entry, run_dir, run_config, run_summary = _receipts(tmp_path)
    target = run_config if fragment in {"plan_fingerprint", "executor_key"} else run_summary
    mutate(target)

    report = _validate(entry, run_dir, run_config, run_summary)

    assert not report.ok
    assert any(fragment in error for error in report.errors), report.errors
