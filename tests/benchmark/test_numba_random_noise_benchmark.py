from __future__ import annotations

import math

import numpy as np
from benchmark import benchmark_numba_random_noise as bench


def test_numba_random_noise_profiles_match_training_shapes() -> None:
    profiles = bench.make_profiles()

    motion = profiles["sac_g1_motion_tracking_mujoco"]
    joystick = profiles["sac_g1_walk_flat_mujoco"]

    assert motion.default_num_envs == 2048
    assert motion.total_width == 64
    assert [field.name for field in motion.fields] == [
        "linvel",
        "gyro",
        "joint_pos",
        "dof_vel",
    ]
    assert joystick.total_width == 64
    assert joystick.fields[0].scale == 0.0
    assert joystick.fields[1].scale == 0.0


def test_numba_random_noise_benchmark_builds_records() -> None:
    profile = bench.make_profiles()["ppo_g1_motion_tracking_motrix"]
    threads = [1] if bench.NUMBA_AVAILABLE else []

    records = bench.bench_one(
        profile=profile,
        num_envs=32,
        dtype_name="float32",
        thread_counts=threads,
        iters=1,
        warmup=0,
        seed=0,
    )

    paths = {record.path for record in records}
    assert "numpy_random_uniform_alloc" in paths
    assert "numpy_generator_random_out" in paths
    if bench.NUMBA_AVAILABLE:
        assert "numba_random_prange" in paths
    assert all(record.values == 32 * profile.total_width for record in records)
    assert all(record.mean_ms >= 0.0 for record in records)
    assert all(math.isfinite(record.speedup_vs_numpy_alloc) for record in records)


def test_numba_random_noise_zero_scale_fields_are_zeroed() -> None:
    profile = bench.make_profiles()["sac_g1_walk_flat_mujoco"]
    buffers = bench._numpy_random_uniform_alloc(
        profile,
        num_envs=8,
        dtype=np.dtype(np.float32),
        noise_level=1.0,
    )

    assert np.max(np.abs(buffers[0])) == 0.0
    assert np.max(np.abs(buffers[1])) == 0.0
    assert np.max(np.abs(buffers[2])) > 0.0


def test_numba_random_noise_formats_records() -> None:
    record = bench.BenchCase(
        profile="sac_g1_motion_tracking_mujoco",
        owner_config="conf/offpolicy/task/sac/g1_motion_tracking/mujoco.yaml",
        num_envs=64,
        dtype="float32",
        path="numba_random_prange",
        threads=4,
        values=4096,
        mean_ms=0.25,
        min_ms=0.2,
        std_ms=0.01,
        values_per_s=16_384_000.0,
        speedup_vs_numpy_alloc=2.0,
        compile_ms=10.0,
        deterministic_same_seed=True,
    )

    payload = bench._case_to_dict(record)
    table = bench._format_table([record])

    assert payload["mvalues_per_s"] == 16.384
    assert "numba_random_prange" in table
    assert "2.00x" in table
    assert "True" in table
