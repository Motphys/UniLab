#!/usr/bin/env python3
"""Attribution harness for reset_done changes from issue #673.

This benchmark avoids full collector throughput attribution. It constructs one
env per case and alternates old/new implementations in the same process, using
the same env_ids and reset payloads.

Usage:
    uv run benchmark/benchmark_reset_done_attribution.py
    uv run benchmark/benchmark_reset_done_attribution.py --num-envs 8192 --reset-count 256
    uv run benchmark/benchmark_reset_done_attribution.py --out-json /tmp/reset_done_attr.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from benchmark.core.device_info import get_device_info_dict, get_device_info_line


@dataclass(frozen=True)
class TimingStats:
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    samples_ms: list[float]


@dataclass(frozen=True)
class BodyPoseAttributionResult:
    backend: str
    num_envs: int
    reset_count: int
    body_count: int
    repeats: int
    old_build_observation_ms: TimingStats
    new_build_observation_ms: TimingStats
    old_body_pose_ms: TimingStats
    new_body_pose_ms: TimingStats
    build_observation_speedup: float
    body_pose_speedup: float


@dataclass(frozen=True)
class SetStateAttributionResult:
    backend: str
    num_envs: int
    reset_count: int
    repeats: int
    old_set_state_ms: TimingStats
    new_set_state_ms: TimingStats
    old_refresh_cache_ms: TimingStats
    new_refresh_cache_ms: TimingStats
    old_component_ms: dict[str, TimingStats]
    new_component_ms: dict[str, TimingStats]
    set_state_speedup: float
    refresh_cache_speedup: float


@dataclass(frozen=True)
class SetStateProfileResult:
    backend: str
    num_envs: int
    reset_count: int
    repeats: int
    set_state_ms: TimingStats
    component_ms: dict[str, TimingStats]


def _stats(samples_ms: list[float]) -> TimingStats:
    if not samples_ms:
        raise ValueError("no samples")
    arr = np.asarray(samples_ms, dtype=np.float64)
    return TimingStats(
        mean_ms=float(arr.mean()),
        median_ms=float(np.median(arr)),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
        samples_ms=[float(v) for v in samples_ms],
    )


def _speedup(old: TimingStats, new: TimingStats) -> float:
    return old.mean_ms / new.mean_ms if new.mean_ms > 0.0 else float("inf")


def _sample_env_ids(num_envs: int, reset_count: int, seed: int) -> np.ndarray:
    if reset_count <= 0:
        raise ValueError("reset-count must be > 0")
    if reset_count > num_envs:
        raise ValueError("reset-count must be <= num-envs")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_envs, size=reset_count, replace=False).astype(np.int32))


def _compose_env(algo: str, task: str, backend: str, num_envs: int):
    from unilab.training import BackendAdapter, create_env, ensure_registries

    ensure_registries()
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(ROOT_DIR / "conf" / "offpolicy")):
        cfg = compose(
            "config",
            overrides=[
                f"algo={algo}",
                f"task={algo}/{task}/{backend}",
                f"algo.num_envs={num_envs}",
            ],
        )
    env_cfg_override = BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name=algo,
    ).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=num_envs, env_cfg_override=env_cfg_override)
    env.init_state()
    return env


def _body_pose_old_rows(backend: Any, rows: np.ndarray, body_ids: np.ndarray):
    pos, quat = backend.get_body_pose_w(body_ids)
    return pos[rows], quat[rows]


def _run_body_pose_once(provider: Any, env: Any, env_ids: np.ndarray) -> tuple[float, float]:
    t0 = time.perf_counter()
    provider.build_reset_observation(env, env_ids, {})
    total_ms = (time.perf_counter() - t0) * 1000.0
    timing = provider.last_reset_observation_timing_ms
    return total_ms, float(timing["dr_reset_obs_get_body_pose_ms"])


def benchmark_body_pose_attribution(
    *,
    backend: str,
    num_envs: int,
    reset_count: int,
    warmup_repeats: int,
    measure_repeats: int,
    seed: int,
) -> BodyPoseAttributionResult:
    env = _compose_env("sac", "g1_motion_tracking", backend, num_envs)
    try:
        provider = env._dr_manager._provider
        env_ids = _sample_env_ids(num_envs, reset_count, seed)
        original_rows = env._backend.get_body_pose_w_rows

        def old_rows(self, rows: np.ndarray, body_ids: np.ndarray):
            return _body_pose_old_rows(self, rows, body_ids)

        old_pos, old_quat = _body_pose_old_rows(env._backend, env_ids, env.body_ids)
        new_pos, new_quat = original_rows(env_ids, env.body_ids)
        np.testing.assert_allclose(old_pos, new_pos, rtol=0, atol=0)
        np.testing.assert_allclose(old_quat, new_quat, rtol=0, atol=0)

        old_total: list[float] = []
        new_total: list[float] = []
        old_body: list[float] = []
        new_body: list[float] = []

        for repeat_idx in range(warmup_repeats + measure_repeats):
            record = repeat_idx >= warmup_repeats

            env._backend.get_body_pose_w_rows = types.MethodType(old_rows, env._backend)
            total_ms, body_ms = _run_body_pose_once(provider, env, env_ids)
            if record:
                old_total.append(total_ms)
                old_body.append(body_ms)

            env._backend.get_body_pose_w_rows = original_rows
            total_ms, body_ms = _run_body_pose_once(provider, env, env_ids)
            if record:
                new_total.append(total_ms)
                new_body.append(body_ms)

        env._backend.get_body_pose_w_rows = original_rows
        old_total_stats = _stats(old_total)
        new_total_stats = _stats(new_total)
        old_body_stats = _stats(old_body)
        new_body_stats = _stats(new_body)
        return BodyPoseAttributionResult(
            backend=backend,
            num_envs=num_envs,
            reset_count=reset_count,
            body_count=int(len(env.body_ids)),
            repeats=measure_repeats,
            old_build_observation_ms=old_total_stats,
            new_build_observation_ms=new_total_stats,
            old_body_pose_ms=old_body_stats,
            new_body_pose_ms=new_body_stats,
            build_observation_speedup=_speedup(old_total_stats, new_total_stats),
            body_pose_speedup=_speedup(old_body_stats, new_body_stats),
        )
    finally:
        env.close()


def _old_refresh_link_pose_cache(self, env_indices: np.ndarray | None = None, data_slice=None):
    del data_slice
    if env_indices is None:
        self._link_poses = self._model.get_link_poses(self._data)
    else:
        mask = np.zeros(self._num_envs, dtype=bool)
        mask[env_indices] = True
        self._link_poses[env_indices] = self._model.get_link_poses(self._data[mask])


SET_STATE_COMPONENT_KEYS = (
    "dr_reset_set_state_alloc_state_ms",
    "dr_reset_set_state_fill_qpos_ms",
    "dr_reset_set_state_fill_qvel_ms",
    "dr_reset_set_state_env_ids_ms",
    "dr_reset_set_state_qpos_convert_ms",
    "dr_reset_set_state_slice_ms",
    "dr_reset_set_state_data_write_ms",
    "dr_reset_set_state_data_reset_ms",
    "dr_reset_set_state_clear_forces_ms",
    "dr_reset_set_state_geom_overrides_ms",
    "dr_reset_set_state_randomization_ms",
    "dr_reset_set_state_set_dof_vel_ms",
    "dr_reset_set_state_set_dof_pos_ms",
    "dr_reset_set_state_ctrl_ms",
    "dr_reset_set_state_forward_kinematic_ms",
    "dr_reset_set_state_pool_reset_ms",
    "dr_reset_set_state_physics_state_scatter_ms",
    "dr_reset_set_state_sensor_scatter_ms",
    "dr_reset_set_state_refresh_cache_ms",
    "dr_reset_set_state_invalidate_cache_ms",
    "dr_reset_set_state_backend_internal_gap_ms",
)


def _run_set_state_once(env: Any, env_ids: np.ndarray, qpos: np.ndarray, qvel: np.ndarray) -> dict:
    t0 = time.perf_counter()
    env._backend.set_state(env_ids, qpos, qvel)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    timing = env._backend.last_set_state_timing_ms
    return {"dr_reset_set_state_wall_ms": wall_ms, **timing}


def benchmark_set_state_attribution(
    *,
    num_envs: int,
    reset_count: int,
    warmup_repeats: int,
    measure_repeats: int,
    seed: int,
) -> SetStateAttributionResult:
    env = _compose_env("flashsac", "g1_walk_flat", "motrix", num_envs)
    try:
        env_ids = _sample_env_ids(num_envs, reset_count, seed)
        plan = env._dr_manager._provider.build_reset_plan(env, env_ids)
        qpos = np.asarray(plan.qpos)
        qvel = np.asarray(plan.qvel)
        original_refresh = env._backend._refresh_link_pose_cache

        old_set_state: list[float] = []
        new_set_state: list[float] = []
        old_refresh: list[float] = []
        new_refresh: list[float] = []
        old_components: dict[str, list[float]] = {key: [] for key in SET_STATE_COMPONENT_KEYS}
        new_components: dict[str, list[float]] = {key: [] for key in SET_STATE_COMPONENT_KEYS}

        for repeat_idx in range(warmup_repeats + measure_repeats):
            record = repeat_idx >= warmup_repeats

            env._backend._refresh_link_pose_cache = types.MethodType(
                _old_refresh_link_pose_cache, env._backend
            )
            timing = _run_set_state_once(env, env_ids, qpos, qvel)
            if record:
                old_set_state.append(float(timing["dr_reset_set_state_wall_ms"]))
                old_refresh.append(float(timing["dr_reset_set_state_refresh_cache_ms"]))
                for key, values in old_components.items():
                    values.append(float(timing.get(key, 0.0)))

            env._backend._refresh_link_pose_cache = original_refresh
            timing = _run_set_state_once(env, env_ids, qpos, qvel)
            if record:
                new_set_state.append(float(timing["dr_reset_set_state_wall_ms"]))
                new_refresh.append(float(timing["dr_reset_set_state_refresh_cache_ms"]))
                for key, values in new_components.items():
                    values.append(float(timing.get(key, 0.0)))

        env._backend._refresh_link_pose_cache = original_refresh
        old_set_state_stats = _stats(old_set_state)
        new_set_state_stats = _stats(new_set_state)
        old_refresh_stats = _stats(old_refresh)
        new_refresh_stats = _stats(new_refresh)
        return SetStateAttributionResult(
            backend="motrix",
            num_envs=num_envs,
            reset_count=reset_count,
            repeats=measure_repeats,
            old_set_state_ms=old_set_state_stats,
            new_set_state_ms=new_set_state_stats,
            old_refresh_cache_ms=old_refresh_stats,
            new_refresh_cache_ms=new_refresh_stats,
            old_component_ms={key: _stats(values) for key, values in old_components.items()},
            new_component_ms={key: _stats(values) for key, values in new_components.items()},
            set_state_speedup=_speedup(old_set_state_stats, new_set_state_stats),
            refresh_cache_speedup=_speedup(old_refresh_stats, new_refresh_stats),
        )
    finally:
        env.close()


def benchmark_set_state_profile(
    *,
    backend: str,
    num_envs: int,
    reset_count: int,
    warmup_repeats: int,
    measure_repeats: int,
    seed: int,
) -> SetStateProfileResult:
    env = _compose_env("flashsac", "g1_walk_flat", backend, num_envs)
    try:
        env_ids = _sample_env_ids(num_envs, reset_count, seed)
        plan = env._dr_manager._provider.build_reset_plan(env, env_ids)
        qpos = np.asarray(plan.qpos)
        qvel = np.asarray(plan.qvel)
        randomization = plan.randomization

        set_state: list[float] = []
        components: dict[str, list[float]] = {key: [] for key in SET_STATE_COMPONENT_KEYS}

        for repeat_idx in range(warmup_repeats + measure_repeats):
            record = repeat_idx >= warmup_repeats
            t0 = time.perf_counter()
            env._backend.set_state(env_ids, qpos, qvel, randomization=randomization)
            wall_ms = (time.perf_counter() - t0) * 1000.0
            timing = env._backend.last_set_state_timing_ms
            if record:
                set_state.append(wall_ms)
                for key, values in components.items():
                    values.append(float(timing.get(key, 0.0)))

        return SetStateProfileResult(
            backend=backend,
            num_envs=num_envs,
            reset_count=reset_count,
            repeats=measure_repeats,
            set_state_ms=_stats(set_state),
            component_ms={key: _stats(values) for key, values in components.items()},
        )
    finally:
        env.close()


def _print_stats(label: str, old: TimingStats, new: TimingStats, speedup: float) -> None:
    print(
        f"  {label}: old={old.mean_ms:.6f} ms new={new.mean_ms:.6f} ms "
        f"speedup={speedup:.2f}x"
    )


def _run_safely(label: str, fn: Callable[[], Any]) -> Any | None:
    try:
        return fn()
    except Exception as exc:
        print(f"{label}: ERROR {type(exc).__name__}: {exc}")
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--reset-count", type=int, default=256)
    parser.add_argument("--warmup-repeats", type=int, default=5)
    parser.add_argument("--measure-repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=673)
    parser.add_argument("--body-pose-backends", default="motrix,mujoco")
    parser.add_argument("--set-state-profile-backends", default="motrix,mujoco")
    parser.add_argument("--skip-set-state", action="store_true")
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"Device: {get_device_info_line()}")

    body_pose_results: list[BodyPoseAttributionResult] = []
    for backend in [part.strip() for part in args.body_pose_backends.split(",") if part.strip()]:
        result = _run_safely(
            f"body_pose/{backend}",
            lambda backend=backend: benchmark_body_pose_attribution(
                backend=backend,
                num_envs=args.num_envs,
                reset_count=args.reset_count,
                warmup_repeats=args.warmup_repeats,
                measure_repeats=args.measure_repeats,
                seed=args.seed,
            ),
        )
        if result is None:
            continue
        body_pose_results.append(result)
        print(f"body_pose/{backend}:")
        _print_stats(
            "build_observation_ms",
            result.old_build_observation_ms,
            result.new_build_observation_ms,
            result.build_observation_speedup,
        )
        _print_stats(
            "body_pose_ms",
            result.old_body_pose_ms,
            result.new_body_pose_ms,
            result.body_pose_speedup,
        )

    set_state_result = None
    set_state_profiles: list[SetStateProfileResult] = []
    if not args.skip_set_state:
        set_state_result = _run_safely(
            "set_state/motrix",
            lambda: benchmark_set_state_attribution(
                num_envs=args.num_envs,
                reset_count=args.reset_count,
                warmup_repeats=args.warmup_repeats,
                measure_repeats=args.measure_repeats,
                seed=args.seed,
            ),
        )
        if set_state_result is not None:
            print("set_state/motrix:")
            _print_stats(
                "set_state_ms",
                set_state_result.old_set_state_ms,
                set_state_result.new_set_state_ms,
                set_state_result.set_state_speedup,
            )
            _print_stats(
                "refresh_cache_ms",
                set_state_result.old_refresh_cache_ms,
                set_state_result.new_refresh_cache_ms,
                set_state_result.refresh_cache_speedup,
            )
            for key in SET_STATE_COMPONENT_KEYS:
                old = set_state_result.old_component_ms[key]
                new = set_state_result.new_component_ms[key]
                print(f"  {key}: old={old.mean_ms:.6f} ms new={new.mean_ms:.6f} ms")

        for backend in [
            part.strip() for part in args.set_state_profile_backends.split(",") if part.strip()
        ]:
            profile = _run_safely(
                f"set_state_profile/{backend}",
                lambda backend=backend: benchmark_set_state_profile(
                    backend=backend,
                    num_envs=args.num_envs,
                    reset_count=args.reset_count,
                    warmup_repeats=args.warmup_repeats,
                    measure_repeats=args.measure_repeats,
                    seed=args.seed,
                ),
            )
            if profile is None:
                continue
            set_state_profiles.append(profile)
            print(f"set_state_profile/{backend}:")
            print(f"  set_state_ms: current={profile.set_state_ms.mean_ms:.6f} ms")
            for key in SET_STATE_COMPONENT_KEYS:
                value = profile.component_ms[key]
                if value.mean_ms > 0.0:
                    print(f"  {key}: current={value.mean_ms:.6f} ms")

    if args.out_json is not None:
        payload = {
            "device": get_device_info_dict(),
            "num_envs": args.num_envs,
            "reset_count": args.reset_count,
            "warmup_repeats": args.warmup_repeats,
            "measure_repeats": args.measure_repeats,
            "seed": args.seed,
            "body_pose": [asdict(result) for result in body_pose_results],
            "set_state": asdict(set_state_result) if set_state_result is not None else None,
            "set_state_profiles": [asdict(result) for result in set_state_profiles],
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
