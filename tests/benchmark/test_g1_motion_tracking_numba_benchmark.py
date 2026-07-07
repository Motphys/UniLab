from __future__ import annotations

from benchmark import benchmark_g1_motion_tracking_numba as bench


def test_g1_motion_tracking_numba_benchmark_builds_records_and_matches_numpy() -> None:
    spec = bench.make_profile_specs()["sac_default"]

    records, component, parity = bench.bench_one(
        profile=spec,
        num_envs=64,
        thread_counts=[1],
        iters=1,
        warmup=0,
        seed=0,
    )

    assert parity["termination_mismatch"] == 0.0
    assert parity["max_abs_reward_diff"] < 1.0e-5
    assert {record.path for record in records} == {"numpy_dispatch", "numba_accelerator"}
    assert any(record.path == "numba_accelerator" and record.threads == 1 for record in records)
    assert component.profile == "sac_default"
    assert component.speedup_vs_numpy > 0.0
    assert component.numpy_total_ms >= component.numpy_reward_termination_ms
    assert all(
        record.parallel_speedup_vs_numba_1t is not None
        for record in records
        if record.path == "numba_accelerator"
    )


def test_g1_motion_tracking_numba_benchmark_formats_end_to_end_records() -> None:
    record = bench.EndToEndCase(
        case=bench.DEFAULT_E2E_CASE,
        path="training_collector_numba",
        num_envs=64,
        warmup_steps=1,
        measure_steps=2,
        numba_acceleration=True,
        numba_threads=4,
        collector_active_steps_per_sec=30_000.0,
        total_active_ms=4.2,
        collector_step_ms=2.1,
        env_step_ms=1.5,
        physics_step_ms=0.9,
        update_state_ms=0.2,
        other_ms=1.0,
        speedup_vs_numpy=1.25,
        env_step_speedup_vs_numpy=1.5,
        update_state_speedup_vs_numpy=2.0,
    )

    payload = bench._e2e_case_to_dict(record)
    table = bench._format_e2e_table([record])

    assert payload["numba_acceleration"] is True
    assert payload["numba_threads"] == 4
    assert "training_collector_numba" in table
    assert "1.25x" in table
    assert "motrixsim" in table
    assert "physics ms" in table


def test_g1_motion_tracking_numba_benchmark_formats_component_records() -> None:
    record = bench.ComponentCase(
        profile="sac_default",
        num_envs=1024,
        numba_threads=8,
        relative_transform_ms=0.5,
        numpy_reward_termination_ms=1.0,
        numba_reward_termination_ms=0.25,
        numpy_update_state_ms=3.0,
        numba_update_state_ms=1.5,
        numpy_total_ms=1.5,
        numba_total_ms=0.75,
        speedup_vs_numpy=2.0,
    )

    payload = bench._component_case_to_dict(record)
    table = bench._format_component_table([record])

    assert payload["relative_transform_ms"] == 0.5
    assert "rel ms" in table
    assert "numpy update_state ms" in table
    assert "2.00x" in table


def test_g1_motion_tracking_numba_benchmark_formats_e2e_reconciliation() -> None:
    hot_records = [
        bench.BenchCase(
            "sac_default",
            1024,
            "numpy_dispatch",
            None,
            1.0,
            1.0,
            0.0,
            1024.0,
            1.0,
        ),
        bench.BenchCase(
            "sac_default",
            1024,
            "numba_accelerator",
            4,
            0.25,
            0.25,
            0.0,
            4096.0,
            4.0,
        ),
    ]
    e2e_records = [
        bench.EndToEndCase(
            case=bench.DEFAULT_E2E_CASE,
            path="training_collector_numpy",
            num_envs=1024,
            warmup_steps=1,
            measure_steps=2,
            numba_acceleration=False,
            numba_threads=None,
            collector_active_steps_per_sec=20_000.0,
            total_active_ms=8.0,
            collector_step_ms=4.0,
            env_step_ms=3.0,
            physics_step_ms=1.0,
            update_state_ms=2.0,
            other_ms=1.0,
        ),
        bench.EndToEndCase(
            case=bench.DEFAULT_E2E_CASE,
            path="training_collector_numba",
            num_envs=1024,
            warmup_steps=1,
            measure_steps=2,
            numba_acceleration=True,
            numba_threads=4,
            collector_active_steps_per_sec=25_000.0,
            total_active_ms=6.4,
            collector_step_ms=3.2,
            env_step_ms=2.2,
            physics_step_ms=1.0,
            update_state_ms=1.4,
            other_ms=0.8,
        ),
    ]

    table = bench._format_e2e_reconciliation_table(
        hot_records=hot_records,
        e2e_records=e2e_records,
    )

    assert "hot saved ms" in table
    assert "0.750" in table
    assert "30.0%" in table


def test_g1_motion_tracking_numba_benchmark_selects_best_hot_slice_threads() -> None:
    records = [
        bench.BenchCase("sac_default", 1024, "numpy_dispatch", None, 1.0, 1.0, 0.0, 1000.0, 1.0),
        bench.BenchCase(
            "sac_default", 1024, "numba_accelerator", 2, 0.8, 0.8, 0.0, 1250.0, 1.25
        ),
        bench.BenchCase(
            "sac_default", 1024, "numba_accelerator", 4, 0.6, 0.6, 0.0, 1666.0, 1.67
        ),
        bench.BenchCase("ppo_default", 1024, "numba_accelerator", 8, 0.5, 0.5, 0.0, 2000.0, 2.0),
    ]

    assert bench._best_threads_for_profile(records, profile="sac_default", num_envs=[1024]) == {
        1024: 4
    }
