from __future__ import annotations

import ast
from pathlib import Path

import pytest
from benchmark.issue705 import benchmark_g1_phase0, process_evidence
from benchmark.issue705.benchmark_g1_phase0 import (
    DEFAULT_PLAN,
    _build_env_config,
    _load_plan,
    _run_subprocess,
    _source_payload,
    main,
)


def test_list_cases_exposes_the_frozen_fifty_process_matrix(capsys) -> None:
    assert main(["--list-cases"]) == 0

    case_ids = capsys.readouterr().out.splitlines()
    assert len(case_ids) == 50
    assert len(set(case_ids)) == 50
    assert "env-b4096-r4" in case_ids
    assert "dr-default_kp_kd-d1p0000-r4" in case_ids
    assert "ppo-seed-4" in case_ids


def test_runner_refuses_implicit_expensive_execution() -> None:
    with pytest.raises(SystemExit, match="Refusing to run implicitly"):
        main([])


def test_baseline_generation_rejects_a_dirty_source_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _load_plan(DEFAULT_PLAN)
    monkeypatch.setattr(
        benchmark_g1_phase0,
        "_git",
        lambda *args: " M src/unilab/base/backend/mujoco/backend.py",
    )

    with pytest.raises(RuntimeError, match="clean git worktree"):
        _source_payload(plan)


@pytest.mark.parametrize(
    ("module", "preflight"),
    [
        (benchmark_g1_phase0, benchmark_g1_phase0._preflight_payload),
        (process_evidence, process_evidence.preflight_payload),
    ],
)
def test_preflight_normalizes_load_by_actual_host_core_count(
    monkeypatch: pytest.MonkeyPatch,
    module,
    preflight,
) -> None:
    plan = _load_plan(DEFAULT_PLAN)
    monkeypatch.setattr(module.os, "getloadavg", lambda: (4.0, 0.0, 0.0))
    monkeypatch.setattr(module.psutil, "cpu_count", lambda *, logical: 8 if not logical else 16)
    monkeypatch.setattr(module, "_gpu_compute_processes", lambda: [])
    monkeypatch.setattr(
        module,
        "_gpu_idle_sample",
        lambda: {
            "utilization_percent": 0,
            "memory_used_mib": 0,
            "temperature_c": 30,
            "pstate": "P8",
        },
    )
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    payload = preflight(plan, enforce_cpu_load=False)

    assert payload["load_per_physical_core"] == 0.5


def test_env_config_uses_owner_then_applies_frozen_benchmark_profile() -> None:
    plan = _load_plan(DEFAULT_PLAN)

    enabled = _build_env_config(plan, "default_kp_kd")
    disabled = _build_env_config(plan, "disabled")

    assert enabled.domain_rand.randomize_kp is True
    assert enabled.domain_rand.randomize_kd is True
    assert disabled.domain_rand.randomize_kp is False
    assert disabled.domain_rand.randomize_kd is False
    assert enabled.adaptive_chunk_size is False
    assert enabled.chunk_size is None


@pytest.mark.local_evidence
def test_subprocess_is_uv_run_and_enforces_cpu_affinity() -> None:
    plan = _load_plan(DEFAULT_PLAN)
    command = [
        "uv",
        "run",
        "python",
        "-c",
        "import os; print(sorted(os.sched_getaffinity(0)))",
    ]

    process, memory, stdout, stderr = _run_subprocess(
        command,
        plan,
        memory_poll_interval=0.05,
    )

    assert process["return_code"] == 0
    assert process["command"][:2] == ["uv", "run"]
    assert ast.literal_eval(stdout.strip()) == list(plan.hardware.affinity_cpus)
    assert stderr == ""
    assert memory
    assert all(sample["rss_bytes"] >= 0 for sample in memory)


def test_worker_output_path_is_not_created_by_list_mode(tmp_path: Path) -> None:
    output = tmp_path / "must-not-exist.json"

    assert main(["--list-cases", "--out", str(output)]) == 0
    assert not output.exists()
