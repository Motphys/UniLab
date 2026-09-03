from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "microduck_alignment_compare.py"


def _load_compare_module():
    spec = importlib.util.spec_from_file_location("microduck_alignment_compare", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolves cls.__module__ via sys.modules
    spec.loader.exec_module(module)
    return module


def _write_unilab_run(run_dir: Path, rewards: list[float], seed: int) -> None:
    run_dir.mkdir(parents=True)
    writer = SummaryWriter(str(run_dir))
    for step, reward in enumerate(rewards):
        writer.add_scalar("Train/mean_reward", reward, step)
        writer.add_scalar("Train/mean_episode_length", 100.0 + step, step)
        writer.add_scalar("Episode_Reward/tracking_lin_vel", reward / 10.0, step)
        writer.add_scalar("Episode_Reward/action_rate", -0.1, step)
        writer.add_scalar("Episode_Termination/tilt", 1.0, step)
        writer.add_scalar("Episode_Termination/time_out", 3.0, step)
        writer.add_scalar("Loss/value", 0.5, step)  # out of scope, must be ignored
    writer.close()
    (run_dir / "run_config.json").write_text(
        json.dumps({"run": {"effective_seed": seed}, "config": {"algo": {"max_iterations": 2000}}})
    )


def _write_upstream_run(run_dir: Path, rewards: list[float], seed: int) -> None:
    run_dir.mkdir(parents=True)
    writer = SummaryWriter(str(run_dir))
    for step, reward in enumerate(rewards):
        writer.add_scalar("Train/mean_reward", reward, step)
        writer.add_scalar("Train/mean_episode_length", 50.0 + step, step)
        writer.add_scalar("Episode_Reward/track_linear_velocity", reward / 20.0, step)
        writer.add_scalar("Episode_Reward/action_rate_l2", -0.2, step)
        writer.add_scalar("Episode_Reward/head_pose_bias", 0.01, step)  # unpaired on unilab side
        writer.add_scalar("Episode_Termination/fell_over", 3.0, step)
        writer.add_scalar("Episode_Termination/time_out", 1.0, step)
    writer.close()
    params = run_dir / "params"
    params.mkdir()
    (params / "agent.yaml").write_text(
        f"seed: {seed}\nnum_steps_per_env: 24\nmax_iterations: 2000\n"
        "obs_groups: !!python/tuple\n  - actor\n"
    )


def test_load_runs_scoped_tags_and_meta(tmp_path: Path) -> None:
    compare = _load_compare_module()
    run_dir = tmp_path / "unilab_run"
    _write_unilab_run(run_dir, [1.0] * 10, seed=42)

    runs = compare.load_runs([str(run_dir)], "unilab")

    assert len(runs) == 1
    run = runs[0]
    assert run.seed == 42
    assert run.max_iterations == 2000
    assert "Loss/value" not in run.curves
    assert run.curves["Train/mean_reward"] == [(i, 1.0) for i in range(10)]


def test_load_runs_skips_empty_dirs(tmp_path: Path) -> None:
    compare = _load_compare_module()
    empty = tmp_path / "aborted_run"
    empty.mkdir()
    good = tmp_path / "good_run"
    _write_upstream_run(good, [1.0] * 10, seed=7)

    runs = compare.load_runs([str(empty), str(good)], "upstream")

    assert [r.seed for r in runs] == [7]


def test_summarize_side_final_window_and_convergence(tmp_path: Path) -> None:
    compare = _load_compare_module()
    # Linear ramp 0..9: final 20% window covers the last 2 points (8, 9) -> 8.5;
    # 80% of 8.5 = 6.8 first reached at iteration 7.
    _write_unilab_run(tmp_path / "run_a", [float(i) for i in range(10)], seed=42)
    _write_unilab_run(tmp_path / "run_b", [2.0 * i for i in range(10)], seed=43)
    runs = compare.load_runs([str(tmp_path / "run_a"), str(tmp_path / "run_b")], "unilab")

    summary = compare.summarize_side(runs)

    # run_a final window mean(8, 9) = 8.5; run_b mean(16, 18) = 17 -> cross-run 12.75.
    assert summary["Train/mean_reward"]["mean"] == 12.75
    assert summary["Train/mean_reward"]["std"] is not None
    assert summary["Train/mean_reward@convergence_iter"]["mean"] == 7.0
    ep = summary["Train/mean_episode_length"]
    assert ep["mean"] == 108.5  # per-run tail mean(108, 109) for both runs


def test_build_report_pairs_aliased_terms(tmp_path: Path) -> None:
    compare = _load_compare_module()
    _write_unilab_run(tmp_path / "uni_a", [1.0] * 10, seed=42)
    _write_upstream_run(tmp_path / "up_a", [2.0] * 10, seed=42)
    unilab_runs = compare.load_runs([str(tmp_path / "uni_a")], "unilab")
    upstream_runs = compare.load_runs([str(tmp_path / "up_a")], "upstream")

    report = compare.build_report(unilab_runs, upstream_runs)

    assert "tracking_lin_vel / track_linear_velocity" in report
    assert "action_rate / action_rate_l2" in report
    assert "tilt / fell_over" in report
    assert "上游独有 term: `head_pose_bias`" in report
    assert "只有 1 个 run" in report
    # aliased termination values: unilab tilt=1 vs upstream fell_over=3 -> -66.7%
    assert "-66.7%" in report
