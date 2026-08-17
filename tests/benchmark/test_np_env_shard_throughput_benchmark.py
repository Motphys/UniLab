from __future__ import annotations

import numpy as np
import pytest
from scripts.benchmark.env import benchmark_np_env_shard_throughput as bench


def test_shard_env_rows_covers_each_row_exactly_once() -> None:
    for total_envs, num_shards in [(8192, 1), (8192, 2), (32768, 4), (10, 3), (7, 7)]:
        rows = bench.shard_env_rows(total_envs, num_shards)
        assert len(rows) == num_shards
        assert sum(rows) == total_envs
        assert max(rows) - min(rows) <= 1
        # Contiguous partitions cover [0, total_envs) with no overlap or gap.
        covered: list[int] = []
        start = 0
        for size in rows:
            covered.extend(range(start, start + size))
            start += size
        assert covered == list(range(total_envs))


def test_shard_env_rows_rejects_invalid_splits() -> None:
    with pytest.raises(ValueError, match="total_envs"):
        bench.shard_env_rows(0, 1)
    with pytest.raises(ValueError, match="num_shards"):
        bench.shard_env_rows(8192, 0)
    with pytest.raises(ValueError, match="num_shards"):
        bench.shard_env_rows(4, 5)


def test_shard_cpu_sets_are_disjoint_and_complete() -> None:
    cpus = list(range(96))
    for num_shards in (1, 2, 4, 7):
        sets = bench.shard_cpu_sets(cpus, num_shards)
        assert len(sets) == num_shards
        union: set[int] = set()
        for cpu_set in sets:
            assert cpu_set.isdisjoint(union)
            union |= set(cpu_set)
        assert union == set(cpus)
        sizes = [len(s) for s in sets]
        assert max(sizes) - min(sizes) <= 1


def test_shard_cpu_sets_rejects_over_subscription() -> None:
    with pytest.raises(ValueError, match="exceeds available CPUs"):
        bench.shard_cpu_sets([0, 1], 3)
    with pytest.raises(ValueError, match="num_shards"):
        bench.shard_cpu_sets([0, 1], 0)


def test_prepare_actions_is_seeded_and_legal() -> None:
    a1 = bench.prepare_actions(seed=42, num_steps=5, num_envs=8, action_dim=29)
    a2 = bench.prepare_actions(seed=42, num_steps=5, num_envs=8, action_dim=29)
    a3 = bench.prepare_actions(seed=43, num_steps=5, num_envs=8, action_dim=29)
    assert a1.shape == (5, 8, 29)
    assert a1.dtype == np.float32
    assert np.array_equal(a1, a2)
    assert not np.array_equal(a1, a3)
    assert float(a1.min()) >= -1.0
    assert float(a1.max()) <= 1.0


def test_run_measured_steps_times_only_the_step_loop() -> None:
    calls: list[np.ndarray] = []

    def step_fn(action: np.ndarray) -> None:
        calls.append(action)

    actions = bench.prepare_actions(seed=0, num_steps=4, num_envs=3, action_dim=2)
    ticks = iter([10.0, 17.5])

    real_perf_counter = bench.time.perf_counter
    bench.time.perf_counter = lambda: next(ticks)  # type: ignore[assignment]
    try:
        elapsed = bench.run_measured_steps(step_fn, actions)
    finally:
        bench.time.perf_counter = real_perf_counter  # type: ignore[assignment]

    assert elapsed == pytest.approx(7.5)
    # Exactly one step call per pre-prepared action row, in order — action
    # preparation happened before the timed window and is not re-done inside.
    assert len(calls) == actions.shape[0]
    for i, call in enumerate(calls):
        assert np.array_equal(call, actions[i])


def test_total_throughput_formula_uses_paired_wall_time() -> None:
    assert bench.total_throughput(8192, 50, 2.0) == pytest.approx(8192 * 50 / 2.0)
    assert bench.total_throughput(8192, 50, 0.0) == 0.0


def _case_record(
    task_id: str, total_envs: int, num_shards: int, throughputs: list[float] | None
) -> dict:
    record = {
        "task_id": task_id,
        "total_envs": total_envs,
        "num_shards": num_shards,
        "status": "ok" if throughputs is not None else "failed",
        "error": None if throughputs is not None else "boom",
        "runs": [],
        "summary": None,
    }
    if throughputs is not None:
        record["runs"] = [
            {"repeat": i, "status": "ok", "total_env_steps_per_s": v}
            for i, v in enumerate(throughputs)
        ]
        record["summary"] = bench.summarize_runs(record["runs"])
    return record


def test_summarize_runs_aggregates_repeats() -> None:
    summary = bench.summarize_runs(
        [
            {"total_env_steps_per_s": 100.0},
            {"total_env_steps_per_s": 120.0},
            {"total_env_steps_per_s": 80.0},
        ]
    )
    assert summary["num_runs"] == 3
    assert summary["mean_env_steps_per_s"] == pytest.approx(100.0)
    assert summary["min_env_steps_per_s"] == pytest.approx(80.0)
    assert summary["max_env_steps_per_s"] == pytest.approx(120.0)
    assert summary["scaling_vs_s1"] is None


def test_attach_scaling_uses_same_case_s1_baseline() -> None:
    records = [
        _case_record("g1_walk_flat", 8192, 1, [100_000.0]),
        _case_record("g1_walk_flat", 8192, 2, [180_000.0]),
        _case_record("g1_walk_flat", 32768, 1, [90_000.0]),
        _case_record("g1_walk_flat", 32768, 2, None),
    ]
    bench.attach_scaling(records)

    assert records[0]["summary"]["scaling_vs_s1"] == pytest.approx(1.0)
    assert records[1]["summary"]["scaling_vs_s1"] == pytest.approx(1.8)
    assert records[2]["summary"]["scaling_vs_s1"] == pytest.approx(1.0)
    # Failed cases keep their failure record and get no scaling number.
    assert records[3]["status"] == "failed"
    assert records[3]["error"] == "boom"
    assert records[3]["summary"] is None


def test_failed_run_records_error_and_no_throughput() -> None:
    run = bench._failed_run(repeat=2, error="timeout after 10s waiting for readiness")
    assert run["repeat"] == 2
    assert run["status"] == "failed"
    assert "timeout" in run["error"]
    assert run["wall_time_s"] is None
    assert run["total_env_steps_per_s"] is None
    assert run["shards"] == []
