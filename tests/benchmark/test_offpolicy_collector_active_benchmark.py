from __future__ import annotations

import builtins
import sys
from collections import namedtuple
from types import SimpleNamespace

import pytest
from scripts.benchmark.rl import benchmark_offpolicy_collector_active as bench


def _make_result(
    *,
    algo: str = "sac",
    task: str = "g1_walk_flat",
    sim: str = "mujoco",
    runtime_sim_backend: str = "mujoco",
    num_envs: int = 8192,
    throughput: float = 123456.7,
    cpu_util_pct: float = 42.5,
    include_env_step_breakdown: bool = False,
) -> bench.CollectorResult:
    case = bench.CollectorCase(
        algo=algo,
        task=task,
        sim=sim,
        variant="default",
        runtime_sim_backend=runtime_sim_backend,
        command=f"uv run train --algo {algo} --task {task} --sim {runtime_sim_backend}",
        training_task_name="G1WalkFlat",
        num_envs=num_envs,
        replay_capacity_rows=num_envs * 2,
        replay_capacity_steps=2,
        obs_dim=3,
        critic_dim=4,
        action_dim=1,
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
    assert bench._parse_case("sac/g1_walk_flat/mujoco") == ("sac", "g1_walk_flat", "mujoco", None)

    with pytest.raises(ValueError, match="<algo>/<task>/<sim>"):
        bench._parse_case("g1_walk_flat/mujoco")


def test_parse_case_accepts_optional_owner_profile() -> None:
    assert bench._parse_case("flashsac/g1_motion_tracking/mjwarp/mimiclite_dr") == (
        "flashsac",
        "g1_motion_tracking",
        "mjwarp",
        "mimiclite_dr",
    )

    with pytest.raises(ValueError, match=r"<algo>/<task>/<sim>\[/<profile>\]"):
        bench._parse_case("flashsac/g1_motion_tracking/mjwarp/mimiclite_dr/extra")


def test_owner_config_path_resolves_profile_owner() -> None:
    plain = bench._owner_config_path("flashsac", "g1_motion_tracking", "mjwarp")
    profiled = bench._owner_config_path("flashsac", "g1_motion_tracking", "mjwarp", "mimiclite_dr")

    assert plain.name == "mjwarp.yaml"
    assert profiled.name == "mjwarp_mimiclite_dr.yaml"
    assert profiled.is_file()


def test_owner_config_path_resolves_startup_profile_owner() -> None:
    profiled = bench._owner_config_path(
        "flashsac",
        "g1_motion_tracking",
        "mjwarp",
        "mimiclite_dr_startup",
    )

    assert profiled.name == "mjwarp_mimiclite_dr_startup.yaml"
    assert profiled.is_file()


def test_profile_case_composes_profile_owner_with_sim_backend() -> None:
    cfg = bench._compose_offpolicy_cfg(
        "flashsac",
        "g1_motion_tracking",
        "mjwarp",
        "mimiclite_dr",
    )

    # The profile selects the owner YAML; the sim segment still selects the
    # runtime backend.
    assert cfg.training.sim_backend == "mjwarp"
    assert cfg.training.task_name == "G1MotionTracking"


def test_startup_profile_case_composes_startup_mass_com_owner() -> None:
    cfg = bench._compose_offpolicy_cfg(
        "flashsac",
        "g1_motion_tracking",
        "mjwarp",
        "mimiclite_dr_startup",
    )

    assert cfg.training.sim_backend == "mjwarp"
    assert cfg.env.events.body_mass.mode == "startup"
    assert cfg.env.events.body_com.mode == "startup"


def test_default_cases_cover_scoped_host_bridge() -> None:
    specs = bench._resolve_case_specs(
        "default",
        algos_arg="sac,flashsac",
        backends=("mujoco",),
    )

    assert "sac/g1_motion_tracking/mujoco" in specs
    assert "flashsac/g1_walk_flat/mujoco" in specs


def test_all_backend_selection_stays_on_scoped_default_backends() -> None:
    backends = bench._resolve_backend_selection(backend="mujoco", all_backends=True)
    specs = bench._resolve_case_specs(
        "default",
        algos_arg="sac,flashsac",
        backends=backends,
    )

    assert backends == ("mujoco",)
    assert "sac/g1_motion_tracking/mujoco" in specs
    assert "flashsac/g1_walk_flat/mujoco" in specs


def test_cuda_tensor_backends_are_opt_in() -> None:
    assert bench.OPTIONAL_BACKENDS == ("mjwarp", "genesis")
    assert set(bench.OPTIONAL_BACKENDS).isdisjoint(bench.BENCHMARK_BACKENDS)
    assert "mjwarp" not in bench._resolve_backend_selection(backend="mujoco", all_backends=True)
    assert "genesis" not in bench._resolve_backend_selection(backend="mujoco", all_backends=True)
    if bench.find_spec("mujoco_warp") is None or bench.find_spec("warp") is None:
        with pytest.raises(SystemExit, match="mjwarp extra"):
            bench._resolve_backend_selection(backend="mjwarp", all_backends=False)
    else:
        assert bench._resolve_backend_selection(backend="mjwarp", all_backends=False) == ("mjwarp",)


def test_resolve_case_specs_deduplicates_explicit_specs() -> None:
    specs = bench._resolve_case_specs(
        "sac/g1_walk_flat/mujoco,sac/g1_walk_flat/mujoco,flashsac/g1_walk_flat/mujoco",
        algos_arg="sac,flashsac",
        backends=("mujoco",),
    )

    assert specs == ["sac/g1_walk_flat/mujoco", "flashsac/g1_walk_flat/mujoco"]


def test_noise_seed_override_composes_for_target_g1_profiles() -> None:
    # Manager-Based owners seed their tensor runtime directly rather than
    # carrying the legacy observation-noise config.
    for spec in (("sac", "g1_motion_tracking", "mujoco"),):
        cfg = bench._compose_offpolicy_cfg(
            *spec,
            extra_overrides=["env.seed=123"],
        )

        assert cfg.env.seed == 123


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
    assert args.backend == "mujoco"
    assert not args.all_backends


def test_parse_args_accepts_backend_and_all_backend_modes() -> None:
    backend_args = bench.parse_args(["--backend", "mujoco"])
    all_args = bench.parse_args(["--all"])

    assert not backend_args.all_backends
    assert all_args.backend == "mujoco"
    assert all_args.all_backends


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
    assert "mujoco" in table
    assert "8,192" in table
    assert "123,457" in table
    assert "42.5" in table
    assert "Total active ms" in table
    assert "Env step ms (%)" in table
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
    assert "mba_" not in header
    for key in bench.COLLECTOR_PHASES:
        assert key in header


def test_write_csv_includes_backend_set_state_sub_timing_columns(tmp_path) -> None:
    """CSV headers must expose every backend set_state sub-key so downstream
    reporting can diff pre/post-optimization runs."""
    result = _make_result(num_envs=2, throughput=2000.0, include_env_step_breakdown=True)
    out_csv = tmp_path / "collector.csv"

    bench._write_csv(out_csv, [result])

    header = out_csv.read_text(encoding="utf-8").splitlines()[0]
    expected_keys = (
        "set_state_mask_ms",
        "set_state_data_slice_ms",
        "set_state_data_reset_ms",
        "set_state_clear_forces_ms",
        "set_state_geom_overrides_ms",
        "set_state_reset_rand_ms",
        "set_state_set_dof_vel_ms",
        "set_state_set_dof_pos_ms",
        "set_state_actuator_ctrl_ms",
        "set_state_forward_kinematic_ms",
        "set_state_refresh_pose_cache_ms",
        "set_state_invalidate_velocity_ms",
        "set_state_qpos_convert_ms",
        "set_state_pool_reset_ms",
        "set_state_state_scatter_ms",
        "set_state_reset_upload_ms",
        "set_state_reset_forward_ms",
        "set_state_host_cache_refresh_ms",
        "set_state_internal_gap_ms",
        "set_state_tensor_model_update_ms",
        "set_state_tensor_mask_ms",
        "set_state_tensor_commit_forward_ms",
        "set_state_tensor_host_cache_refresh_ms",
    )
    for key in expected_keys:
        assert key in header, f"CSV header missing {key!r}"


def test_format_set_state_detail_table_reports_sub_key_percentages() -> None:
    """The detail table shows each sub-key next to its share of
    ``dr_reset_set_state_ms`` (percentages must appear)."""
    result = _make_result(num_envs=2, throughput=2000.0, include_env_step_breakdown=True)
    result.env_step_timing_ms_per_vector_step["dr_reset_set_state_ms"] = bench.TimingStats(
        [4.0], 4.0, 4.0, 0.0, 4.0, 4.0
    )
    result.env_step_timing_ms_per_vector_step["set_state_forward_kinematic_ms"] = bench.TimingStats(
        [2.0], 2.0, 2.0, 0.0, 2.0, 2.0
    )
    result.env_step_timing_ms_per_vector_step["set_state_mask_ms"] = bench.TimingStats(
        [0.1], 0.1, 0.1, 0.0, 0.1, 0.1
    )

    table = bench._format_set_state_detail_table([result])

    # Header exposes the "FK" and "Mask" sub-columns.
    assert "FK" in table
    assert "Mask" in table
    # 2.0 ms / 4.0 ms = 50%; 0.1 ms / 4.0 ms = 2.5%.
    assert "2.000 (50.0%)" in table
    assert "0.100 (2.5%)" in table


def test_format_set_state_mujoco_table_covers_mujoco_only_keys() -> None:
    """The mujoco keyset table exposes pool_reset / state_scatter columns."""
    result = _make_result(
        runtime_sim_backend="mujoco",
        num_envs=2,
        throughput=2000.0,
        include_env_step_breakdown=True,
    )
    result.env_step_timing_ms_per_vector_step["dr_reset_set_state_ms"] = bench.TimingStats(
        [5.0], 5.0, 5.0, 0.0, 5.0, 5.0
    )
    result.env_step_timing_ms_per_vector_step["set_state_pool_reset_ms"] = bench.TimingStats(
        [4.0], 4.0, 4.0, 0.0, 4.0, 4.0
    )
    result.env_step_timing_ms_per_vector_step["set_state_state_scatter_ms"] = bench.TimingStats(
        [0.75], 0.75, 0.75, 0.0, 0.75, 0.75
    )

    table = bench._format_set_state_mujoco_table([result])

    assert "Pool reset" in table
    assert "State scatter" in table
    assert "4.000 (80.0%)" in table
    assert "0.750 (15.0%)" in table


def test_format_set_state_mjwarp_table_covers_mjwarp_only_keys() -> None:
    """The mjwarp keyset table exposes reset_upload / forward / host_cache_refresh."""
    result = _make_result(
        runtime_sim_backend="mjwarp",
        num_envs=2,
        throughput=2000.0,
        include_env_step_breakdown=True,
    )
    result.env_step_timing_ms_per_vector_step["dr_reset_set_state_ms"] = bench.TimingStats(
        [8.0], 8.0, 8.0, 0.0, 8.0, 8.0
    )
    result.env_step_timing_ms_per_vector_step["set_state_reset_upload_ms"] = bench.TimingStats(
        [4.0], 4.0, 4.0, 0.0, 4.0, 4.0
    )
    result.env_step_timing_ms_per_vector_step["set_state_host_cache_refresh_ms"] = (
        bench.TimingStats([2.0], 2.0, 2.0, 0.0, 2.0, 2.0)
    )

    table = bench._format_set_state_mjwarp_table([result])

    assert "Reset upload" in table
    assert "Reset forward" in table
    assert "Host cache refresh" in table
    assert "4.000 (50.0%)" in table
    assert "2.000 (25.0%)" in table


def test_format_set_state_mjwarp_tensor_table_covers_tensor_only_keys() -> None:
    """The mjwarp tensor keyset table exposes the set_state_tensor sub-timings."""
    result = _make_result(
        runtime_sim_backend="mjwarp",
        num_envs=2,
        throughput=2000.0,
        include_env_step_breakdown=True,
    )
    result.env_step_timing_ms_per_vector_step["dr_reset_set_state_ms"] = bench.TimingStats(
        [8.0], 8.0, 8.0, 0.0, 8.0, 8.0
    )
    result.env_step_timing_ms_per_vector_step["set_state_tensor_commit_forward_ms"] = (
        bench.TimingStats([4.0], 4.0, 4.0, 0.0, 4.0, 4.0)
    )
    result.env_step_timing_ms_per_vector_step["set_state_tensor_model_update_ms"] = (
        bench.TimingStats([2.0], 2.0, 2.0, 0.0, 2.0, 2.0)
    )

    table = bench._format_set_state_mjwarp_tensor_table([result])

    assert "Model update" in table
    assert "Tensor mask" in table
    assert "Commit forward" in table
    assert "Tensor host cache" in table
    assert "4.000 (50.0%)" in table
    assert "2.000 (25.0%)" in table


def test_active_collector_loop_keeps_transition_tensors_native() -> None:
    """The benchmark hot loop must not regress to NumPy transition carriers."""
    import ast
    from pathlib import Path

    source = Path(bench.__file__).read_text()
    tree = ast.parse(source)
    loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_run_active_window_case"
    )

    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "zeros"
        for node in ast.walk(loop)
    )
    assert not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "asarray"
        for node in ast.walk(loop)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_numpy"
        for node in ast.walk(loop)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "resolve_terminal_observation_contract"
        for node in ast.walk(loop)
    )


def test_variant_labels_require_unique_nonempty_values() -> None:
    assert bench._variant_labels(None) == ("default",)
    assert bench._variant_labels("numpy, torch") == ("numpy", "torch")

    with pytest.raises(ValueError, match="at least one non-empty label"):
        bench._variant_labels(" , ")
    with pytest.raises(ValueError, match="labels must be unique"):
        bench._variant_labels("torch,torch")


def test_variant_overrides_parse_hydra_equals_values() -> None:
    parsed = bench._parse_variant_overrides(["torch=env.seed=7", "torch=training.foo=1"])

    assert parsed == {"torch": ["env.seed=7", "training.foo=1"]}

    with pytest.raises(ValueError, match="VARIANT=HYDRA_OVERRIDE"):
        bench._parse_variant_overrides(["torch"])
    with pytest.raises(ValueError, match="non-empty VARIANT"):
        bench._parse_variant_overrides(["=env.seed=7"])


def test_parse_args_accepts_variant_labels_and_overrides() -> None:
    args = bench.parse_args(
        [
            "--variants",
            "numpy,torch",
            "--variant-override",
            "torch=env.seed=7",
            "--variant-override",
            "numpy=env.seed=8",
        ]
    )

    assert args.variants == "numpy,torch"
    assert args.variant_override == ["torch=env.seed=7", "numpy=env.seed=8"]


def test_variant_blocks_alternate_labels_without_extending_measurement(monkeypatch) -> None:
    """A/B blocks alternate labels and retain the same warmup/measure budget."""
    order: list[str] = []

    def fake_build_and_run_case(
        spec,
        *,
        warmup_steps,
        measure_steps,
        replay_capacity_steps,
        tail_steps,
        num_envs,
        extra_overrides,
        variant,
        profile_numpy_random,
    ):
        order.append(variant)
        result = _make_result(
            throughput=1000.0,
            include_env_step_breakdown=True,
        )
        result.case = bench.CollectorCase(**{**vars(result.case), "variant": variant})
        return result

    monkeypatch.setattr(bench, "_build_and_run_case", fake_build_and_run_case)
    monkeypatch.setattr(bench, "_print_result", lambda result: None)

    payload = {}
    monkeypatch.setattr(
        bench,
        "_write_json",
        lambda path, value: payload.update(value),
    )

    argv = [
        "--cases",
        "flashsac/g1_walk_flat/mujoco",
        "--warmup-steps",
        "5",
        "--measure-steps",
        "20",
        "--variants",
        "base,head",
        "--repeat-variant-blocks",
        "2",
    ]
    monkeypatch.setattr(sys, "argv", ["benchmark", *argv])
    bench.main()

    assert order == ["base", "head", "base", "head"]
    assert payload["args"]["warmup_steps"] == 5
    assert payload["args"]["measure_steps"] == 20
    assert payload["args"]["repeat_variant_blocks"] == 2
    assert [result["case"]["variant"] for result in payload["results"]] == order


def test_variant_ablation_table_compares_each_variant_to_default() -> None:
    baseline = _make_result(num_envs=2, throughput=1000.0)
    torch_result = _make_result(num_envs=2, throughput=1500.0)
    torch_result.case = bench.CollectorCase(**{**vars(torch_result.case), "variant": "torch"})

    table = bench._format_variant_ablation_table([baseline, torch_result])

    assert "default" in table
    assert "torch" in table
    assert "1.500x" in table


def test_variant_ablation_table_pairs_each_head_with_preceding_base() -> None:
    base_low = _make_result(num_envs=2, throughput=1000.0)
    head_low = _make_result(num_envs=2, throughput=1200.0)
    base_high = _make_result(num_envs=2, throughput=1500.0)
    head_high = _make_result(num_envs=2, throughput=1500.0)
    for result, variant in (
        (base_low, "base"),
        (head_low, "head"),
        (base_high, "base"),
        (head_high, "head"),
    ):
        result.case = bench.CollectorCase(**{**vars(result.case), "variant": variant})

    table = bench._format_variant_ablation_table(
        [base_low, head_low, base_high, head_high], paired=True
    )

    assert "1.200x" in table
    assert "1.500x" in table


def test_numpy_random_profiler_times_generator_bound_calls() -> None:
    import numpy as np

    profiler = bench._NumpyRandomProfiler()
    generator = np.random.default_rng(17)
    proxy = profiler.bind_generator(generator)

    profiler.begin_step()
    proxy.uniform(-1.0, 1.0, (2,))
    proxy.integers(0, 3, (2,))
    profiler.end_step()

    assert profiler.step_calls == 2
    assert profiler.step_ms >= 0.0
    assert isinstance(proxy.bit_generator, np.random.BitGenerator)
    assert profiler.restore_generator() is generator
