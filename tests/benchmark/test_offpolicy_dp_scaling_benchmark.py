from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from benchmark.rl import benchmark_offpolicy_dp_scaling as bench

torch = pytest.importorskip("torch")
from torch.utils.tensorboard import SummaryWriter  # noqa: E402


def _write_tfevents(log_dir: Path, series: dict[str, list[float]]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        for tag, values in series.items():
            for step, value in enumerate(values):
                writer.add_scalar(tag, value, step)
    finally:
        writer.close()


def _write_run_summary(run_dir: Path, **overrides: object) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "completed",
        "completed_iterations": 300,
        "total_env_steps": 1_200_000,
        "training_wall_time_sec": 600.0,
        **overrides,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _make_run_dir(
    run_dir: Path,
    *,
    steps_per_sec: list[float] | None = None,
    samples_per_sec: list[float] | None = None,
    reward: list[float] | None = None,
    dp_sync_time: list[float] | None = None,
) -> None:
    _write_run_summary(run_dir)
    rank0_series = {
        bench.STEPS_PER_SEC_TAG: steps_per_sec or [],
        bench.SAMPLES_PER_SEC_TAG: samples_per_sec or [],
    }
    if reward is not None:
        rank0_series[bench.REWARD_TAG] = reward
    if dp_sync_time is not None:
        rank0_series[bench.DP_SYNC_TIME_TAG] = dp_sync_time
    _write_tfevents(run_dir, rank0_series)


def test_steady_state_mean_uses_tail_half() -> None:
    # ceil(3/2)=2 tail points: (30+40)/2; warm-up prefix 10 is excluded.
    assert bench.steady_state_mean([10.0, 30.0, 40.0]) == pytest.approx(35.0)
    assert bench.steady_state_mean([1.0, 2.0, 3.0, 4.0]) == pytest.approx(3.5)
    assert bench.steady_state_mean([7.0]) == pytest.approx(7.0)
    assert bench.steady_state_mean([0.0, 100.0], tail_fraction=1.0) == pytest.approx(50.0)
    with pytest.raises(ValueError, match="at least one sample"):
        bench.steady_state_mean([])


def test_verdict_for_ratio() -> None:
    assert bench.verdict_for_ratio(1.7) == "pass"
    assert bench.verdict_for_ratio(2.0) == "pass"
    assert bench.verdict_for_ratio(1.69) == "below threshold"
    assert bench.verdict_for_ratio(None) is None


def test_build_train_command_matches_production_overrides(tmp_path: Path) -> None:
    command = bench.build_train_command(tmp_path / "runs" / "n1", iterations=300, devices=None)
    assert command[0] == sys.executable
    assert command[1].endswith("scripts/train_offpolicy.py")
    assert "algo=sac" in command
    assert "task=sac/g1_walk_flat/mujoco" in command
    assert "training.no_play=true" in command
    assert "algo.max_iterations=300" in command
    assert any(arg.startswith("training.log_dir=") for arg in command)
    # N=1 baseline must not carry training.devices at all.
    assert not any(arg.startswith("training.devices") for arg in command)
    assert not any(arg.startswith("training.dp_sync_interval") for arg in command)

    command_dp = bench.build_train_command(
        tmp_path / "runs" / "n2",
        iterations=300,
        devices=[0, 1],
        extra_overrides=("algo.num_envs=2048",),
    )
    assert "training.devices=[0,1]" in command_dp
    assert not any(arg.startswith("training.dp_sync_interval") for arg in command_dp)
    assert "algo.num_envs=2048" in command_dp


def test_parse_run_single_rank(tmp_path: Path) -> None:
    run_dir = tmp_path / "n1"
    _make_run_dir(
        run_dir,
        steps_per_sec=[100.0, 200.0, 300.0, 400.0],
        samples_per_sec=[1_000.0, 2_000.0, 3_000.0, 4_000.0],
        reward=[0.5, 1.5],
    )
    metrics = bench.parse_run(run_dir, world_size=1)
    assert metrics["completed_iterations"] == 300
    assert metrics["total_env_steps"] == 1_200_000
    assert metrics["training_wall_time_sec"] == pytest.approx(600.0)
    assert metrics["steady_state_collector_steps_per_s"] == pytest.approx(350.0)
    assert metrics["steady_state_learner_samples_per_s"] == pytest.approx(3_500.0)
    assert metrics["final_mean_reward"] == pytest.approx(1.5)
    assert metrics["mean_dp_sync_time_sec"] is None
    assert metrics["num_collector_throughput_samples"] == 4
    assert metrics["num_learner_throughput_samples"] == 4


def test_parse_run_two_ranks_reads_canonical_aggregate_tags(tmp_path: Path) -> None:
    run_dir = tmp_path / "n2"
    _make_run_dir(
        run_dir,
        steps_per_sec=[300.0, 450.0],
        samples_per_sec=[1_200.0, 1_800.0],
        dp_sync_time=[0.02, 0.04],
    )
    metrics = bench.parse_run(run_dir, world_size=2)
    assert metrics["world_size"] == 2
    assert metrics["steady_state_collector_steps_per_s"] == pytest.approx(450.0)
    assert metrics["steady_state_learner_samples_per_s"] == pytest.approx(1_800.0)
    assert metrics["mean_dp_sync_time_sec"] == pytest.approx(0.03)


def test_parse_run_missing_learner_throughput_is_a_hard_error(tmp_path: Path) -> None:
    run_dir = tmp_path / "n2"
    _make_run_dir(run_dir, steps_per_sec=[100.0, 200.0])
    with pytest.raises(bench.RunParseError, match="effective_samples_per_sec"):
        bench.parse_run(run_dir, world_size=2)


def test_parse_run_rejects_non_completed_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "n1"
    _make_run_dir(run_dir, steps_per_sec=[100.0], samples_per_sec=[200.0])
    _write_run_summary(run_dir, status="failed", error="boom")
    with pytest.raises(bench.RunParseError, match="did not complete"):
        bench.parse_run(run_dir, world_size=1)

    (run_dir / "run_summary.json").unlink()
    with pytest.raises(bench.RunParseError, match="missing run summary"):
        bench.parse_run(run_dir, world_size=1)


def _record(
    config: str,
    collector: float | None,
    learner: float | None,
    status: str = "ok",
) -> dict[str, object]:
    return {
        "config": config,
        "status": status,
        "error": None if status == "ok" else "boom",
        "metrics": (
            {
                "steady_state_collector_steps_per_s": collector,
                "steady_state_learner_samples_per_s": learner,
            }
            if status == "ok" and collector is not None and learner is not None
            else None
        ),
    }


def test_attach_scaling_computes_ratio_and_verdict() -> None:
    records = [
        _record("n1", 100_000.0, 200_000.0),
        _record("n2", 180_000.0, 300_000.0),
        _record("n2", None, None, status="failed"),
    ]
    bench.attach_scaling(records)
    assert records[0]["collector_scaling_vs_n1"] is None
    assert records[0]["learner_scaling_vs_n1"] is None
    assert records[0]["verdict"] is None
    assert records[1]["collector_scaling_vs_n1"] == pytest.approx(1.8)
    assert records[1]["learner_scaling_vs_n1"] == pytest.approx(1.5)
    assert records[1]["verdict"] == "pass"
    assert records[2]["collector_scaling_vs_n1"] is None
    assert records[2]["learner_scaling_vs_n1"] is None
    assert records[2]["verdict"] is None


def test_attach_scaling_below_threshold_verdict() -> None:
    records = [
        _record("n1", 100_000.0, 200_000.0),
        _record("n2", 120_000.0, 240_000.0),
    ]
    bench.attach_scaling(records)
    assert records[1]["collector_scaling_vs_n1"] == pytest.approx(1.2)
    assert records[1]["learner_scaling_vs_n1"] == pytest.approx(1.2)
    assert records[1]["verdict"] == "below threshold"


def test_parse_args_skips_dp_config_with_single_device() -> None:
    args = bench.parse_args(["--devices", "0"])
    assert args.devices == "0"
    assert args.keep_runs is True
    args = bench.parse_args(["--no-keep-runs"])
    assert args.keep_runs is False
