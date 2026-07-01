from __future__ import annotations

import builtins
import sys
from collections import namedtuple
from types import SimpleNamespace

import pytest
from benchmark import benchmark_offpolicy_collector_active as bench


def _make_result(
    *,
    algo: str = "sac",
    task: str = "g1_walk_flat",
    sim: str = "motrix",
    runtime_sim_backend: str = "motrix",
    num_envs: int = 8192,
    throughput: float = 123456.7,
    cpu_util_pct: float = 42.5,
    include_env_step_breakdown: bool = False,
) -> bench.CollectorResult:
    case = bench.CollectorCase(
        algo=algo,
        task=task,
        sim=sim,
        runtime_sim_backend=runtime_sim_backend,
        command=f"uv run train --algo {algo} --task {task} --sim {runtime_sim_backend}",
        training_task_name="G1WalkFlat",
        collector_algo_type=algo,
        num_envs=num_envs,
        replay_capacity_rows=num_envs * 2,
        replay_capacity_steps=2,
        obs_dim=3,
        critic_dim=4,
        action_dim=1,
        actor_hidden_dim=8,
        use_layer_norm=False,
        env_steps_per_sync=1,
    )
    physics_stats = (
        bench.TimingStats([0.75], 0.75, 0.75, 0.0, 0.75, 0.75)
        if include_env_step_breakdown
        else None
    )
    env_step_overhead_stats = (
        bench.TimingStats([0.25], 0.25, 0.25, 0.0, 0.25, 0.25)
        if include_env_step_breakdown
        else None
    )
    return bench.CollectorResult(
        case=case,
        warmup_steps=0,
        measure_steps=1,
        total_active_ms=1.0,
        collector_active_steps_per_sec=throughput,
        phase_ms_per_vector_step={
            key: bench.TimingStats([1.0], 1.0, 1.0, 0.0, 1.0, 1.0) for key in bench.COLLECTOR_PHASES
        },
        phase_pct={key: 20.0 for key in bench.COLLECTOR_PHASES},
        notes=[],
        physics_ms_per_vector_step=physics_stats,
        env_step_overhead_ms_per_vector_step=env_step_overhead_stats,
        cpu_util_pct=cpu_util_pct,
    )


def test_parse_case_requires_algo_task_sim() -> None:
    assert bench._parse_case("sac/g1_walk_flat/mujoco") == ("sac", "g1_walk_flat", "mujoco")

    with pytest.raises(ValueError, match="<algo>/<task>/<sim>"):
        bench._parse_case("g1_walk_flat/mujoco")


def test_default_cases_cover_motrix_only_without_sharpa() -> None:
    specs = bench._resolve_case_specs(
        "default",
        algos_arg="sac,flashsac,td3",
        backends=("motrix",),
    )

    assert "sac/g1_motion_tracking/motrix" in specs
    assert "flashsac/g1_walk_flat/motrix" in specs
    assert "sac/g1_motion_tracking/mujoco" not in specs
    assert "sac/sharpa_inhand/mujoco_hora" not in specs


def test_all_backend_selection_expands_default_cases() -> None:
    backends = bench._resolve_backend_selection(backend="mujoco", all_backends=True)
    specs = bench._resolve_case_specs(
        "default",
        algos_arg="sac,flashsac,td3",
        backends=backends,
    )

    assert backends == ("mujoco", "motrix")
    assert "sac/g1_motion_tracking/mujoco" in specs
    assert "sac/g1_motion_tracking/motrix" in specs
    assert "flashsac/g1_walk_flat/mujoco" in specs
    assert "flashsac/g1_walk_flat/motrix" in specs


def test_resolve_case_specs_deduplicates_explicit_specs() -> None:
    specs = bench._resolve_case_specs(
        "sac/g1_walk_flat/mujoco,sac/g1_walk_flat/mujoco,td3/g1_walk_flat/mujoco",
        algos_arg="sac,td3",
        backends=("mujoco",),
    )

    assert specs == ["sac/g1_walk_flat/mujoco", "td3/g1_walk_flat/mujoco"]


def test_motrixsim_case_alias_uses_motrix_owner_config() -> None:
    assert bench._owner_config_path("sac", "g1_walk_flat", "motrixsim").name == "motrix.yaml"

    cfg = bench._compose_offpolicy_cfg(
        "sac",
        "g1_walk_flat",
        "motrixsim",
        num_envs=2,
    )

    assert cfg.training.sim_backend == "motrix"
    assert cfg.algo.num_envs == 2


def test_auto_discovery_supports_motrixsim_alias() -> None:
    specs = bench._resolve_case_specs(
        "auto",
        algos_arg="sac,flashsac,td3",
        backends=("motrix",),
    )

    assert "sac/g1_walk_flat/motrix" in specs
    assert "flashsac/g1_walk_flat/motrix" in specs
    assert "td3/go2_joystick_flat/motrix" in specs


def test_stats_reports_distribution() -> None:
    stats = bench._stats([1.0, 2.0, 3.0])

    assert stats.mean_ms == 2.0
    assert stats.median_ms == 2.0
    assert stats.min_ms == 1.0
    assert stats.max_ms == 3.0
    assert stats.std_ms == pytest.approx(0.81649658)


def test_parse_args_defaults_to_large_env_count_and_longer_measure_window() -> None:
    args = bench.parse_args([])

    assert args.num_envs == 8192
    assert args.measure_steps == 100
    assert args.backend == "motrix"
    assert not args.all_backends


def test_parse_args_accepts_backend_and_all_backend_modes() -> None:
    motrix_args = bench.parse_args(["--backend", "motrix"])
    all_args = bench.parse_args(["--all"])
    legacy_sim_args = bench.parse_args(["--sim", "motrixsim"])

    assert motrix_args.backend == "motrix"
    assert not motrix_args.all_backends
    assert all_args.backend == "motrix"
    assert all_args.all_backends
    assert legacy_sim_args.backend == "motrix"


def test_hardware_table_includes_cpu_and_memory_details() -> None:
    table = bench._format_hardware_table(
        {
            "chip": "AMD Ryzen 9 9950X3D",
            "cpu_total_cores": "16",
            "cpu_frequency": "5.75 GHz",
            "memory": "128.0 GB",
            "memory_frequency": "5600 MT/s",
        },
    )

    assert "AMD Ryzen 9 9950X3D" in table
    assert "16" in table
    assert "5.75 GHz" in table
    assert "128.0 GB" in table
    assert "5600 MT/s" in table


def test_throughput_table_includes_case_throughput_and_num_env() -> None:
    table = bench._format_throughput_table([_make_result(include_env_step_breakdown=True)])

    assert "AMD Ryzen" not in table
    assert "g1_walk_flat" in table
    assert "motrix" in table
    assert "8,192" in table
    assert "123,457" in table
    assert "42.5" in table
    assert "Total active ms" in table
    assert "Weight sync ms (% active)" in table
    assert "1.000 (20.0%)" in table
    assert "Physics ms" not in table


def test_env_step_breakdown_table_keeps_subparts_separate() -> None:
    table = bench._format_env_step_breakdown_table([_make_result(include_env_step_breakdown=True)])

    assert "Env step ms (% env, % active)" in table
    assert "Physics ms (% env, % active)" in table
    assert "Env overhead ms (% env, % active)" in table
    assert "Physics % active" not in table
    assert "Overhead % active" not in table
    assert "1.000 (100.0%, 20.0%)" in table
    assert "0.750 (75.0%, 15.0%)" in table
    assert "0.250 (25.0%, 5.0%)" in table
    assert "0.000000" in table


def test_cpu_util_pct_uses_system_time_delta() -> None:
    assert bench._cpu_util_pct((20.0, 100.0), (50.0, 200.0)) == pytest.approx(30.0)


def test_read_cpu_times_falls_back_to_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/stat":
            raise OSError("no procfs")
        return real_open(path, *args, **kwargs)

    CpuTimes = namedtuple("CpuTimes", ["user", "nice", "system", "idle", "iowait"])
    fake_psutil = SimpleNamespace(cpu_times=lambda: CpuTimes(10.0, 1.0, 4.0, 85.0, 5.0))

    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert bench._read_cpu_times() == pytest.approx((15.0, 105.0))


def test_write_csv_includes_all_phase_columns(tmp_path) -> None:
    result = _make_result(num_envs=2, throughput=2000.0, include_env_step_breakdown=True)
    out_csv = tmp_path / "collector.csv"

    bench._write_csv(out_csv, [result])

    header = out_csv.read_text(encoding="utf-8").splitlines()[0]
    assert "bookkeeping_ms" in header
    assert "env_step_overhead_ms" in header
    assert "physics_pct" in header
    assert "env_step_overhead_pct" in header
    for key in bench.COLLECTOR_PHASES:
        assert key in header
