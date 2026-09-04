from __future__ import annotations

import json

import pytest
import torch
from scripts.benchmark.rl import benchmark_sac_replay_buffer_sampling as bench


def _tiny_case() -> bench.BenchmarkCase:
    return bench.BenchmarkCase(
        algo="sac",
        task="g1_walk_flat",
        sim="mujoco",
        command="uv run train --algo sac --task g1_walk_flat --sim mujoco",
        training_task_name="G1WalkFlat",
        num_envs=2,
        env_steps_per_sync=1,
        replay_buffer_n=8,
        config_capacity_rows=16,
        configured_batch_size=4,
        learner_batch_size=4,
        updates_per_step=2,
        sample_count_per_rank=8,
        learning_starts=0,
        shape=bench.ReplayShape(obs_dim=3, action_dim=2, critic_dim=1),
    )


def test_sac_default_case_uses_configured_batch() -> None:
    cfg = bench._compose_offpolicy_cfg("mujoco")
    case = bench._build_case(
        cfg,
        sim="mujoco",
        shape=bench.ReplayShape(obs_dim=45, action_dim=29, critic_dim=48),
    )

    assert case.command == "uv run train --algo sac --task g1_walk_flat --sim mujoco"
    assert case.config_capacity_rows == case.num_envs * case.replay_buffer_n
    assert case.learner_batch_size == case.configured_batch_size
    assert case.sample_count_per_rank == case.learner_batch_size * case.updates_per_step
    assert case.shape.packed_width == 2 * 45 + 29 + 3 + 2 * 48


def test_default_compose_targets_motion_tracking_motrix() -> None:
    cfg = bench._compose_offpolicy_cfg()
    case = bench._build_case(
        cfg,
        task=bench.DEFAULT_TASK,
        sim=bench.DEFAULT_SIM,
        shape=bench.ReplayShape(obs_dim=160, action_dim=29, critic_dim=289),
    )

    assert bench.DEFAULT_TASK == "g1_motion_tracking"
    assert bench.DEFAULT_SIM == "motrix"
    assert bench._owner_config_exists("g1_motion_tracking", "motrix")
    assert case.command == "uv run train --algo sac --task g1_motion_tracking --sim motrix"
    assert case.training_task_name == "G1MotionTrackingSAC"
    assert case.num_envs == 2048
    assert case.env_steps_per_sync == 1
    assert case.replay_buffer_n == 512
    assert case.config_capacity_rows == 1_048_576
    assert case.configured_batch_size == 8192
    assert case.updates_per_step == 4
    assert case.learning_starts == 1
    assert case.shape.packed_width == 930
    assert case.collector_rows_per_iter == 2048
    assert case.sample_bytes_per_rank == 121_896_960
    assert case.collector_bytes_per_iter == 7_618_560
    assert case.sample_new_ratio_per_rank == 16


def test_missing_owner_config_fails_before_hydra_compose() -> None:
    missing = bench._owner_config_path("not_a_task", "motrix")

    try:
        bench._compose_offpolicy_cfg("not_a_task", "motrix")
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("expected missing owner config to raise FileNotFoundError")


def test_resolve_capacity_rows_from_multipliers_deduplicates() -> None:
    assert bench._resolve_capacity_rows(
        config_capacity_rows=100,
        capacity_rows_arg="auto",
        capacity_multipliers_arg="0.25,0.5,0.5,1",
    ) == [25, 50, 100]


def test_resolve_capacity_rows_from_explicit_values() -> None:
    assert bench._resolve_capacity_rows(
        config_capacity_rows=100,
        capacity_rows_arg="16,32,16",
        capacity_multipliers_arg="1",
    ) == [16, 32]


def test_parse_device_ids_auto_no_cuda(monkeypatch) -> None:
    monkeypatch.setattr(bench.torch.cuda, "is_available", lambda: False)

    assert bench._parse_device_ids("auto") == []


def test_resolve_device_type_prefers_cuda_then_mps(monkeypatch) -> None:
    monkeypatch.setattr(bench.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bench, "_mps_available", lambda: True)
    assert bench._resolve_device_type() == "cuda"

    monkeypatch.setattr(bench.torch.cuda, "is_available", lambda: False)
    assert bench._resolve_device_type() == "mps"

    monkeypatch.setattr(bench, "_mps_available", lambda: False)
    assert bench._resolve_device_type() == "none"


def test_run_capacity_case_portable_cpu_path_records_timings() -> None:
    result = bench._run_capacity_case(
        _tiny_case(),
        capacity_rows=16,
        devices=[torch.device("cpu")],
        warmup=0,
        repeat=1,
        prefill="none",
        pinned_host_batch=False,
        index_mode="pregenerated",
        seed=123,
        torch_threads=1,
    )

    assert result.world_size == 1
    assert result.capacity_rows == 16
    assert result.sample_bytes_per_rank > 0
    assert set(result.timings) == {
        "cpu_sample_wall",
        "cpu_sample_h2d_wall",
        "cpu_sample_then_h2d_wall",
        "gpu_sample_wall",
    }


def test_main_no_cuda_writes_skipped_json(tmp_path, monkeypatch) -> None:
    out_json = tmp_path / "results.json"
    monkeypatch.setattr(bench.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(bench, "_mps_available", lambda: False)

    rc = bench.main(
        [
            "--task",
            "g1_walk_flat",
            "--sim",
            "mujoco",
            "--gpu-counts",
            "1,2",
            "--capacity-rows",
            "16",
            "--warmup",
            "0",
            "--repeat",
            "1",
            "--obs-dim",
            "3",
            "--action-dim",
            "2",
            "--critic-dim",
            "1",
            "--out-json",
            str(out_json),
        ]
    )

    assert rc == 1
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["results"] == []
    assert [item["world_size"] for item in payload["skipped"]] == [1, 2]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
def test_main_mps_runs_single_device_and_skips_multi(tmp_path, monkeypatch) -> None:
    out_json = tmp_path / "results.json"
    monkeypatch.setattr(bench.torch.cuda, "is_available", lambda: False)

    rc = bench.main(
        [
            "--task",
            "g1_walk_flat",
            "--sim",
            "mujoco",
            "--gpu-counts",
            "1,2",
            "--capacity-rows",
            "16",
            "--warmup",
            "0",
            "--repeat",
            "1",
            "--obs-dim",
            "3",
            "--action-dim",
            "2",
            "--critic-dim",
            "1",
            "--out-json",
            str(out_json),
        ]
    )

    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["args"]["device_type"] == "mps"
    assert [item["world_size"] for item in payload["results"]] == [1]
    assert set(payload["results"][0]["timings"]) == {
        "cpu_sample_wall",
        "cpu_sample_h2d_wall",
        "cpu_sample_then_h2d_wall",
        "gpu_sample_wall",
    }
    assert [item["world_size"] for item in payload["skipped"]] == [2]
    assert payload["skipped"][0]["reason"] == "MPS backend exposes a single device"
