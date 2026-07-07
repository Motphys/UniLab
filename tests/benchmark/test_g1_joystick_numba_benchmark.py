from __future__ import annotations

from benchmark import benchmark_g1_joystick_numba as bench


def test_g1_joystick_numba_benchmark_builds_records_and_matches_numpy() -> None:
    spec = bench.make_profile_specs()["sac_default"]

    records, parity = bench.bench_one(
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


def test_g1_joystick_numba_benchmark_formats_end_to_end_records() -> None:
    record = bench.EndToEndCase(
        case="sac/g1_walk_flat/mujoco",
        path="training_collector_numba",
        num_envs=64,
        warmup_steps=1,
        measure_steps=2,
        numba_acceleration=True,
        numba_threads=4,
        collector_active_steps_per_sec=30_000.0,
        total_active_ms=4.2,
        env_step_ms=1.5,
        update_state_ms=0.2,
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
