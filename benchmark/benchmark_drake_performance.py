#!/usr/bin/env python3
"""Profile Drake vs MuJoCo env-step performance on selected UniLab tasks.

This benchmark is intentionally task-level rather than raw-simulator-level. It
keeps UniLab's reset, observation, sensor-view, and body-query paths in the
loop so G1 motion tracking can expose the expensive integration points.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
DEFAULT_DRAKEUNI_SRC = ROOT_DIR.parent / "drakeuni" / "src"
DEFAULT_OUTPUT = ROOT_DIR / "benchmark" / "outputs" / "drake_performance" / "results.json"


def _prepend_import_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _install_import_paths(drakeuni_src: Path | None) -> None:
    for path in (SRC_DIR, ROOT_DIR):
        _prepend_import_path(path)
    if drakeuni_src is not None and drakeuni_src.exists():
        _prepend_import_path(drakeuni_src)


@dataclass(frozen=True)
class TaskSpec:
    env_cfg_factory: Callable[[], Any]
    env_cls_factory: Callable[[], type]


@dataclass
class BenchRecord:
    task: str
    backend: str
    num_envs: int
    nthread: int | str
    warmup_steps: int
    steps: int
    action_mode: str
    reset_events_mean: float
    reset_events_max: int
    metrics_median_ms: dict[str, float]
    metrics_mean_ms: dict[str, float]


class StepProfiler:
    """Small monkeypatch profiler for task/backend methods inside one env."""

    def __init__(self) -> None:
        self._current: dict[str, float] | None = None

    def begin(self) -> None:
        self._current = {}

    def end(self) -> dict[str, float]:
        current = self._current or {}
        self._current = None
        return current

    def add(self, key: str, seconds: float) -> None:
        if self._current is None:
            return
        self._current[key] = self._current.get(key, 0.0) + seconds * 1000.0

    def wrap(self, obj: Any, name: str, key: str) -> None:
        if not hasattr(obj, name):
            return
        original = getattr(obj, name)
        if getattr(original, "_unilab_benchmark_wrapped", False):
            return

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.add(key, time.perf_counter() - t0)

        setattr(wrapped, "_unilab_benchmark_wrapped", True)
        setattr(obj, name, wrapped)


def _task_specs() -> dict[str, TaskSpec]:
    def go1_cfg() -> Any:
        from unilab.envs.locomotion.go1.joystick import Go1JoystickCfg

        return Go1JoystickCfg()

    def go1_env() -> type:
        from unilab.envs.locomotion.go1.joystick import Go1WalkTask

        return Go1WalkTask

    def go2_cfg() -> Any:
        from unilab.envs.locomotion.go2.joystick import Go2JoystickCfg

        return Go2JoystickCfg()

    def go2_env() -> type:
        from unilab.envs.locomotion.go2.joystick import Go2WalkTask

        return Go2WalkTask

    def g1_tracking_cfg() -> Any:
        from unilab.envs.motion_tracking.g1.tracking import G1MotionTrackingEnvCfg

        return G1MotionTrackingEnvCfg()

    def g1_tracking_env() -> type:
        from unilab.envs.motion_tracking.g1.tracking import G1MotionTrackingEnv

        return G1MotionTrackingEnv

    return {
        "g1_motion_tracking": TaskSpec(g1_tracking_cfg, g1_tracking_env),
        "go1_joystick_flat": TaskSpec(go1_cfg, go1_env),
        "go2_joystick_flat": TaskSpec(go2_cfg, go2_env),
    }


def _compose_env_cfg(task: str, backend: str, spec: TaskSpec) -> Any:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from unilab.base.registry import apply_cfg_overrides
    from unilab.training import BackendAdapter

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf" / "ppo"), version_base="1.3"):
        owner_cfg = compose(
            config_name="config",
            overrides=[
                f"task={task}/{backend}",
                "hydra.run.dir=.",
                "hydra.output_subdir=null",
                "hydra/job_logging=disabled",
                "hydra/hydra_logging=disabled",
            ],
        )

    env_cfg_override = BackendAdapter(
        owner_cfg,
        root_dir=ROOT_DIR,
        algo_name="ppo",
    ).build_task_env_cfg_override()
    cfg = spec.env_cfg_factory()
    apply_cfg_overrides(cfg, env_cfg_override)
    return cfg


def _make_env(
    *,
    task: str,
    backend: str,
    num_envs: int,
    nthread: int,
    spec: TaskSpec,
) -> Any:
    cfg = _compose_env_cfg(task, backend, spec)
    if backend == "drake":
        cfg.drake_nthread = int(nthread)
    env_cls = spec.env_cls_factory()
    return env_cls(cfg, num_envs=num_envs, backend_type=backend)


def _attach_profiler(env: Any, profiler: StepProfiler) -> None:
    backend = getattr(env, "_backend", None)
    if backend is not None:
        profiler.wrap(backend, "step", "backend_step_call_ms")
        profiler.wrap(backend, "set_state", "backend_set_state_ms")
        profiler.wrap(backend, "get_sensor_data", "sensor_view_ms")
        profiler.wrap(backend, "get_sensor_data_rows", "sensor_view_ms")
        profiler.wrap(backend, "get_body_pose_w", "body_query_ms")
        profiler.wrap(backend, "get_body_pose_w_rows", "body_query_ms")
        profiler.wrap(backend, "get_body_vel_w", "body_query_ms")
        profiler.wrap(backend, "copy_body_state_w", "body_query_ms")
    profiler.wrap(env, "reset", "reset_method_ms")
    profiler.wrap(env, "_get_body_state_w", "body_query_ms")
    profiler.wrap(env, "_get_current_motion", "motion_current_ms")

    motion_sampler = getattr(env, "motion_sampler", None)
    if motion_sampler is not None:
        profiler.wrap(motion_sampler, "sample_frames", "motion_sample_ms")
        profiler.wrap(motion_sampler, "step", "motion_sampler_step_ms")
        profiler.wrap(motion_sampler, "get_current_motion", "motion_current_ms")

    motion_loader = getattr(env, "motion_loader", None)
    if motion_loader is not None:
        profiler.wrap(motion_loader, "get_motion_at_frame", "motion_lookup_ms")


def _actions(env: Any, mode: str, rng: np.random.Generator) -> np.ndarray:
    shape = (env.num_envs, int(np.prod(env.action_space.shape)))
    if mode == "zeros":
        return np.zeros(shape, dtype=np.float32)
    if mode == "random":
        return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    if mode == "small-random":
        return rng.normal(0.0, 0.05, size=shape).astype(np.float32)
    raise ValueError(f"Unknown action mode: {mode}")


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _summarize(samples: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    keys = sorted({key for sample in samples for key in sample})
    median = {key: _median([sample.get(key, 0.0) for sample in samples]) for key in keys}
    mean = {key: _mean([sample.get(key, 0.0) for sample in samples]) for key in keys}
    return median, mean


def _benchmark_one(
    *,
    task: str,
    backend: str,
    num_envs: int,
    nthread: int,
    warmup_steps: int,
    steps: int,
    action_mode: str,
    seed: int,
    spec: TaskSpec,
) -> BenchRecord:
    rng = np.random.default_rng(seed)
    env = _make_env(task=task, backend=backend, num_envs=num_envs, nthread=nthread, spec=spec)
    profiler = StepProfiler()
    _attach_profiler(env, profiler)
    env.init_state()

    samples: list[dict[str, float]] = []
    reset_counts: list[int] = []
    total_steps = warmup_steps + steps
    for step_idx in range(total_steps):
        actions = _actions(env, action_mode, rng)
        profiler.begin()
        state = env.step(actions)
        profile_sample = profiler.end()
        if step_idx < warmup_steps:
            continue
        timing = dict(state.info.get("timing", {}))
        sample = {key: float(value) for key, value in timing.items() if np.isscalar(value)}
        sample.update(profile_sample)
        samples.append(sample)
        done = np.asarray(state.terminated) | np.asarray(state.truncated)
        reset_counts.append(int(np.count_nonzero(done)))

    median, mean = _summarize(samples)
    return BenchRecord(
        task=task,
        backend=backend,
        num_envs=num_envs,
        nthread=nthread if backend == "drake" else "auto",
        warmup_steps=warmup_steps,
        steps=steps,
        action_mode=action_mode,
        reset_events_mean=_mean(reset_counts),
        reset_events_max=max(reset_counts) if reset_counts else 0,
        metrics_median_ms=median,
        metrics_mean_ms=mean,
    )


def _parse_csv(text: str) -> list[str]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError(f"Expected at least one value in {text!r}")
    return values


def _parse_int_csv(text: str) -> list[int]:
    return [int(value) for value in _parse_csv(text)]


def _print_table(records: Sequence[BenchRecord]) -> None:
    columns = [
        "task",
        "backend",
        "envs",
        "nthread",
        "env_total",
        "step_core",
        "update_state",
        "reset_done",
        "backend_step",
        "backend_physics",
        "backend_refresh",
        "body_query",
        "motion_lookup",
        "resets_mean",
    ]
    print(" | ".join(columns))
    print(" | ".join("---" for _ in columns))
    for record in records:
        m = record.metrics_median_ms
        row = [
            record.task,
            record.backend,
            str(record.num_envs),
            str(record.nthread),
            f"{m.get('env_step_total_ms', 0.0):.3f}",
            f"{m.get('step_core_ms', 0.0):.3f}",
            f"{m.get('update_state_ms', 0.0):.3f}",
            f"{m.get('reset_done_ms', 0.0):.3f}",
            f"{m.get('backend_step_ms', m.get('backend_step_call_ms', 0.0)):.3f}",
            f"{m.get('backend_physics_ms', 0.0):.3f}",
            f"{m.get('backend_refresh_cache_ms', 0.0):.3f}",
            f"{m.get('body_query_ms', 0.0):.3f}",
            f"{m.get('motion_lookup_ms', 0.0):.3f}",
            f"{record.reset_events_mean:.1f}",
        ]
        print(" | ".join(row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        default="go1_joystick_flat,go2_joystick_flat",
        help=(
            "Comma-separated task ids. Defaults stay within committed Drake task configs; "
            "pass g1_motion_tracking explicitly when its Drake config is available."
        ),
    )
    parser.add_argument("--backends", default="drake,mujoco", help="Comma-separated backends.")
    parser.add_argument("--num-envs", default="64,256,1024", help="Comma-separated env counts.")
    parser.add_argument("--nthreads", default="1,4,8,12,20", help="Comma-separated Drake threads.")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--action-mode",
        choices=("zeros", "small-random", "random"),
        default="zeros",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--drakeuni-src", type=Path, default=DEFAULT_DRAKEUNI_SRC)
    args = parser.parse_args()

    _install_import_paths(args.drakeuni_src)
    specs = _task_specs()
    task_ids = _parse_csv(args.tasks)
    backends = _parse_csv(args.backends)
    env_counts = _parse_int_csv(args.num_envs)
    nthreads = _parse_int_csv(args.nthreads)

    records: list[BenchRecord] = []
    for task in task_ids:
        if task not in specs:
            raise ValueError(f"Unknown task {task!r}; available: {sorted(specs)}")
        for num_envs in env_counts:
            for backend in backends:
                thread_values = nthreads if backend == "drake" else [0]
                for nthread in thread_values:
                    print(
                        f"Running task={task} backend={backend} envs={num_envs} "
                        f"nthread={nthread if backend == 'drake' else 'auto'}",
                        flush=True,
                    )
                    records.append(
                        _benchmark_one(
                            task=task,
                            backend=backend,
                            num_envs=num_envs,
                            nthread=nthread,
                            warmup_steps=args.warmup_steps,
                            steps=args.steps,
                            action_mode=args.action_mode,
                            seed=args.seed,
                            spec=specs[task],
                        )
                    )

    _print_table(records)
    payload = {
        "records": [asdict(record) for record in records],
        "args": {
            "tasks": task_ids,
            "backends": backends,
            "num_envs": env_counts,
            "nthreads": nthreads,
            "warmup_steps": args.warmup_steps,
            "steps": args.steps,
            "action_mode": args.action_mode,
            "seed": args.seed,
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
