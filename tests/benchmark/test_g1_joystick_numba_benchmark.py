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
