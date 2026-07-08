from __future__ import annotations

import json

import torch
from benchmark import benchmark_sac_replay_buffer_sampling as bench


def _tiny_case() -> bench.BenchmarkCase:
    return bench.BenchmarkCase(
        algo="sac",
        task="g1_walk_flat",
        sim="mujoco",
        command="uv run train --algo sac --task g1_walk_flat --sim mujoco",
        training_task_name="G1WalkFlat",
        num_envs=2,
        replay_buffer_n=8,
        config_capacity_rows=16,
        configured_batch_size=4,
        learner_batch_size=4,
        symmetry_batch_multiplier=1,
        updates_per_step=2,
        sample_count_per_rank=8,
        learning_starts=0,
        shape=bench.ReplayShape(obs_dim=3, action_dim=2, critic_dim=1),
    )


def test_sac_default_case_uses_effective_symmetry_batch() -> None:
    cfg = bench._compose_offpolicy_cfg("mujoco")
    case = bench._build_case(
        cfg,
        sim="mujoco",
        shape=bench.ReplayShape(obs_dim=45, action_dim=29, critic_dim=48),
        symmetry_batch_multiplier=2,
    )

    assert case.command == "uv run train --algo sac --task g1_walk_flat --sim mujoco"
    assert case.config_capacity_rows == case.num_envs * case.replay_buffer_n
    assert case.learner_batch_size == case.configured_batch_size // 2
    assert case.sample_count_per_rank == case.learner_batch_size * case.updates_per_step
    assert case.shape.packed_width == 2 * 45 + 29 + 3 + 2 * 48


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

    rc = bench.main(
        [
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
            "--symmetry-batch-multiplier",
            "2",
            "--out-json",
            str(out_json),
        ]
    )

    assert rc == 1
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["results"] == []
    assert [item["world_size"] for item in payload["skipped"]] == [1, 2]
