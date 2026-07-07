#!/usr/bin/env python3
"""Benchmark off-policy collector active-window throughput.

This benchmark measures the collector-side hot loop without learner waiting.
It uses the same Hydra owner configs, registry-created envs, off-policy actor
factory, action sampling helpers, terminal-observation contract, and ReplayBuffer
write path used by the training collector.

Usage:
    uv run benchmark/benchmark_offpolicy_collector_active.py
    uv run benchmark/benchmark_offpolicy_collector_active.py --backend motrix
    uv run benchmark/benchmark_offpolicy_collector_active.py --all
    uv run benchmark/benchmark_offpolicy_collector_active.py --cases auto --backend mujoco
    uv run benchmark/benchmark_offpolicy_collector_active.py --cases auto --backend motrix
    uv run benchmark/benchmark_offpolicy_collector_active.py --cases sac/g1_walk_flat/mujoco
    uv run benchmark/benchmark_offpolicy_collector_active.py --cases sac/g1_walk_flat/motrixsim
    uv run benchmark/benchmark_offpolicy_collector_active.py --num-envs 1024 --measure-steps 100
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Sequence, cast

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from benchmark.core.device_info import get_device_info_dict, get_device_info_line

DEFAULT_OUTPUT_JSON = (
    ROOT_DIR / "benchmark" / "outputs" / "offpolicy_collector_active" / "results.json"
)
DEFAULT_NUM_ENVS = 8192
DEFAULT_WARMUP_STEPS = 10
DEFAULT_MEASURE_STEPS = 100
DEFAULT_CASE_TEMPLATES = (
    "sac/g1_motion_tracking",
    "flashsac/g1_walk_flat",
)
DEFAULT_ALGOS = ("sac", "flashsac", "td3")
DEFAULT_BACKEND = "motrix"
BENCHMARK_BACKENDS = ("mujoco", "motrix")
DEFAULT_COLLECTOR_CPU_THREADS = 8
COLLECTOR_CPU_THREADS_ENV = "UNILAB_COLLECTOR_TORCH_THREADS"
NUMBA_ACCELERATED_TASKS = frozenset({"g1_walk_flat", "g1_motion_tracking"})
BACKEND_ALIASES = {
    # UniLab's registry/backend contract uses "motrix"; "motrixsim" is the package
    # and benchmark-facing backend family name.
    "motrixsim": "motrix",
}
COLLECTOR_PHASES = (
    "weight_sync_ms",
    "action_select_ms",
    "env_step_ms",
    "replay_ms",
    "bookkeeping_ms",
)
NP_ENV_STEP_TIMING_KEYS = (
    "env_step_total_ms",
    "apply_action_ms",
    "step_core_ms",
    "update_state_ms",
    "reset_done_ms",
    "env_step_internal_gap_ms",
    "reset_done_terminal_obs_ms",
    "reset_done_reset_call_ms",
    "reset_done_obs_scatter_ms",
    "reset_done_info_scatter_ms",
    "reset_done_internal_gap_ms",
    "dr_reset_total_ms",
    "dr_reset_plan_ms",
    "dr_reset_payload_filter_ms",
    "dr_reset_set_state_ms",
    "dr_reset_build_observation_ms",
    "dr_reset_internal_gap_ms",
    "dr_reset_observation_getters_ms",
    "dr_reset_obs_get_motion_ms",
    "dr_reset_obs_get_local_linvel_ms",
    "dr_reset_obs_get_gyro_ms",
    "dr_reset_obs_get_gravity_ms",
    "dr_reset_obs_get_dof_pos_ms",
    "dr_reset_obs_get_dof_vel_ms",
    "dr_reset_obs_get_body_pose_ms",
    "dr_reset_observation_compute_obs_ms",
    "dr_reset_observation_internal_gap_ms",
    # Backend set_state internals (see BACKEND_SET_STATE_DETAIL_TIMING_KEYS in
    # src/unilab/base/np_env.py). All backends emit the same key set for column
    # stability; sub-keys that don't apply report 0.0.
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
    "set_state_internal_gap_ms",
)
NP_ENV_STEP_COUNT_KEYS = ("reset_done_count",)
NP_ENV_STEP_SAMPLE_KEYS = (*NP_ENV_STEP_TIMING_KEYS, *NP_ENV_STEP_COUNT_KEYS)
NP_RANDOM_PROFILE_FUNCTIONS = (
    "uniform",
    "randint",
    "random",
    "rand",
    "randn",
    "normal",
    "choice",
)
NP_ENV_STEP_TIMING_CSV_FIELDS = (
    ("env_step_total_ms", "np_env_step_total_ms"),
    ("apply_action_ms", "np_env_apply_action_ms"),
    ("step_core_ms", "np_env_step_core_ms"),
    ("update_state_ms", "np_env_update_state_ms"),
    ("reset_done_ms", "np_env_reset_done_ms"),
    ("env_step_internal_gap_ms", "np_env_internal_gap_ms"),
    ("reset_done_terminal_obs_ms", "reset_done_terminal_obs_ms"),
    ("reset_done_reset_call_ms", "reset_done_reset_call_ms"),
    ("reset_done_obs_scatter_ms", "reset_done_obs_scatter_ms"),
    ("reset_done_info_scatter_ms", "reset_done_info_scatter_ms"),
    ("reset_done_internal_gap_ms", "reset_done_internal_gap_ms"),
    ("dr_reset_total_ms", "dr_reset_total_ms"),
    ("dr_reset_plan_ms", "dr_reset_plan_ms"),
    ("dr_reset_payload_filter_ms", "dr_reset_payload_filter_ms"),
    ("dr_reset_set_state_ms", "dr_reset_set_state_ms"),
    ("dr_reset_build_observation_ms", "dr_reset_build_observation_ms"),
    ("dr_reset_internal_gap_ms", "dr_reset_internal_gap_ms"),
    ("dr_reset_observation_getters_ms", "dr_reset_observation_getters_ms"),
    ("dr_reset_obs_get_motion_ms", "dr_reset_obs_get_motion_ms"),
    ("dr_reset_obs_get_local_linvel_ms", "dr_reset_obs_get_local_linvel_ms"),
    ("dr_reset_obs_get_gyro_ms", "dr_reset_obs_get_gyro_ms"),
    ("dr_reset_obs_get_gravity_ms", "dr_reset_obs_get_gravity_ms"),
    ("dr_reset_obs_get_dof_pos_ms", "dr_reset_obs_get_dof_pos_ms"),
    ("dr_reset_obs_get_dof_vel_ms", "dr_reset_obs_get_dof_vel_ms"),
    ("dr_reset_obs_get_body_pose_ms", "dr_reset_obs_get_body_pose_ms"),
    ("dr_reset_observation_compute_obs_ms", "dr_reset_observation_compute_obs_ms"),
    ("dr_reset_observation_internal_gap_ms", "dr_reset_observation_internal_gap_ms"),
    ("set_state_mask_ms", "set_state_mask_ms"),
    ("set_state_data_slice_ms", "set_state_data_slice_ms"),
    ("set_state_data_reset_ms", "set_state_data_reset_ms"),
    ("set_state_clear_forces_ms", "set_state_clear_forces_ms"),
    ("set_state_geom_overrides_ms", "set_state_geom_overrides_ms"),
    ("set_state_reset_rand_ms", "set_state_reset_rand_ms"),
    ("set_state_set_dof_vel_ms", "set_state_set_dof_vel_ms"),
    ("set_state_set_dof_pos_ms", "set_state_set_dof_pos_ms"),
    ("set_state_actuator_ctrl_ms", "set_state_actuator_ctrl_ms"),
    ("set_state_forward_kinematic_ms", "set_state_forward_kinematic_ms"),
    ("set_state_refresh_pose_cache_ms", "set_state_refresh_pose_cache_ms"),
    ("set_state_invalidate_velocity_ms", "set_state_invalidate_velocity_ms"),
    ("set_state_qpos_convert_ms", "set_state_qpos_convert_ms"),
    ("set_state_pool_reset_ms", "set_state_pool_reset_ms"),
    ("set_state_state_scatter_ms", "set_state_state_scatter_ms"),
    ("set_state_internal_gap_ms", "set_state_internal_gap_ms"),
)


@dataclass(frozen=True)
class CollectorCase:
    algo: str
    task: str
    sim: str
    variant: str
    numba_acceleration: bool
    numba_threads: int | None
    runtime_sim_backend: str
    command: str
    training_task_name: str
    collector_algo_type: str
    num_envs: int
    replay_capacity_rows: int
    replay_capacity_steps: int
    obs_dim: int
    critic_dim: int
    action_dim: int
    actor_hidden_dim: int
    use_layer_norm: bool
    env_steps_per_sync: int


@dataclass
class TimingStats:
    samples_ms: list[float]
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float


@dataclass
class CollectorResult:
    case: CollectorCase
    warmup_steps: int
    measure_steps: int
    total_active_ms: float
    collector_active_steps_per_sec: float
    phase_ms_per_vector_step: dict[str, TimingStats]
    phase_pct: dict[str, float]
    notes: list[str]
    # Fine-grained timings reported by NpEnv.step() inside env_step_ms.
    env_step_timing_ms_per_vector_step: dict[str, TimingStats] = field(default_factory=dict)
    # Backend-internal physics time per vector step (sub-part of env_step_ms).
    # None when the backend does not report it (e.g. motrix).
    physics_ms_per_vector_step: TimingStats | None = None
    # Non-physics env.step time per vector step: env_step_ms - physics_ms.
    # None when backend-internal physics timing is unavailable.
    env_step_overhead_ms_per_vector_step: TimingStats | None = None
    # System-wide CPU utilization (%) over the measured window. NaN if unknown.
    cpu_util_pct: float = float("nan")
    # NumPy random time measured inside collector env.step() only. This is an
    # optional benchmark-side monkeypatch used for RNG/noise-buffer profiling.
    numpy_random_ms_per_vector_step: TimingStats | None = None
    numpy_random_calls_per_vector_step: TimingStats | None = None


def _stats(samples_ms: list[float]) -> TimingStats:
    if not samples_ms:
        raise ValueError("no timing samples collected")
    return TimingStats(
        samples_ms=samples_ms,
        mean_ms=mean(samples_ms),
        median_ms=median(samples_ms),
        std_ms=pstdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        min_ms=min(samples_ms),
        max_ms=max(samples_ms),
    )


def _optional_timing_ms(timing: dict[str, Any], key: str) -> float | None:
    try:
        value = float(timing.get(key, float("nan")))
    except (TypeError, ValueError):
        return None
    return value if value == value else None


def _read_cpu_times() -> tuple[float, float] | None:
    """Return system-wide (busy, total) CPU time, or None.

    System-wide (not per-process) because the MuJoCo BatchEnvPool worker
    threads run on other cores; a per-process reading would miss them and hide
    the "many cores, low utilization" scaling problem this metric exists to
    surface. Linux uses /proc/stat directly; other platforms fall back to
    psutil.cpu_times() when available.
    """
    try:
        with open("/proc/stat", encoding="utf-8") as f:
            parts = f.readline().split()[1:]
        vals = [int(x) for x in parts]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
        total = sum(vals)
        if total > 0:
            return float(total - idle), float(total)
    except (OSError, ValueError, IndexError):
        pass

    try:
        import psutil

        times = psutil.cpu_times()
        total = float(sum(times))
        idle = float(getattr(times, "idle", 0.0) + getattr(times, "iowait", 0.0))
        if total > 0.0:
            return total - idle, total
    except Exception:
        pass
    return None


def _cpu_util_pct(start: tuple[float, float] | None, end: tuple[float, float] | None) -> float:
    """System CPU utilization (%) between two _read_cpu_times snapshots."""
    if start is None or end is None:
        return float("nan")
    busy0, total0 = start
    busy1, total1 = end
    dt = total1 - total0
    if dt <= 0:
        return float("nan")
    return 100.0 * (busy1 - busy0) / dt


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


class _NumpyRandomProfiler:
    """Measure selected ``np.random`` module calls inside a benchmark window."""

    def __init__(self) -> None:
        self.enabled = False
        self.step_ms = 0.0
        self.step_calls = 0
        self._originals: dict[str, Any] = {}

    def install(self) -> None:
        if self._originals:
            return
        for name in NP_RANDOM_PROFILE_FUNCTIONS:
            original = getattr(np.random, name, None)
            if original is None:
                continue
            self._originals[name] = original
            setattr(np.random, name, self._wrap(original))

    def uninstall(self) -> None:
        for name, original in self._originals.items():
            setattr(np.random, name, original)
        self._originals.clear()
        self.enabled = False

    def begin_step(self) -> None:
        self.step_ms = 0.0
        self.step_calls = 0
        self.enabled = True

    def end_step(self) -> tuple[float, int]:
        self.enabled = False
        return self.step_ms, self.step_calls

    def _wrap(self, fn: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if not self.enabled:
                return fn(*args, **kwargs)
            t0 = time.perf_counter_ns()
            try:
                return fn(*args, **kwargs)
            finally:
                self.step_ms += (time.perf_counter_ns() - t0) / 1e6
                self.step_calls += 1

        return wrapped


def _configure_collector_cpu_threads(cap: int | None = None) -> int:
    desired = DEFAULT_COLLECTOR_CPU_THREADS if cap is None else int(cap)
    override = os.environ.get(COLLECTOR_CPU_THREADS_ENV)
    if override is not None:
        try:
            desired = int(override)
        except ValueError:
            pass
    n_threads = max(1, min(desired, os.cpu_count() or 1))
    torch.set_num_threads(n_threads)
    return n_threads


def _runtime_sim_backend(sim: str) -> str:
    return BACKEND_ALIASES.get(sim, sim)


def _compose_offpolicy_cfg(
    algo: str,
    task: str,
    sim: str,
    *,
    num_envs: int | None = None,
    extra_overrides: list[str] | None = None,
) -> DictConfig:
    owner_sim = _runtime_sim_backend(sim)
    overrides = [
        f"algo={algo}",
        f"task={algo}/{task}/{owner_sim}",
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled",
    ]
    if num_envs is not None:
        overrides.append(f"algo.num_envs={int(num_envs)}")
    if extra_overrides:
        overrides.extend(extra_overrides)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf" / "offpolicy"), version_base="1.3"):
        return compose(config_name="config", overrides=overrides)


def _owner_config_path(algo: str, task: str, sim: str) -> Path:
    return (
        ROOT_DIR / "conf" / "offpolicy" / "task" / algo / task / f"{_runtime_sim_backend(sim)}.yaml"
    )


def _discover_cases(*, algos: list[str], sim: str) -> list[str]:
    cases: list[str] = []
    owner_sim = _runtime_sim_backend(sim)
    for algo in algos:
        task_root = ROOT_DIR / "conf" / "offpolicy" / "task" / algo
        if not task_root.is_dir():
            continue
        for path in sorted(task_root.glob(f"*/{owner_sim}.yaml")):
            cases.append(f"{algo}/{path.parent.name}/{sim}")
    return cases


def _resolve_backend_selection(*, backend: str, all_backends: bool) -> tuple[str, ...]:
    if all_backends:
        return BENCHMARK_BACKENDS
    if backend not in BENCHMARK_BACKENDS:
        raise ValueError(f"unsupported backend {backend!r}; expected one of {BENCHMARK_BACKENDS}")
    return (backend,)


def _default_case_specs(backends: Sequence[str]) -> list[str]:
    return [f"{template}/{backend}" for backend in backends for template in DEFAULT_CASE_TEMPLATES]


def _parse_case(spec: str) -> tuple[str, str, str]:
    parts = [part for part in spec.split("/") if part]
    if len(parts) != 3:
        raise ValueError(f"case must use <algo>/<task>/<sim>, got {spec!r}")
    algo, task, sim = parts
    if algo not in DEFAULT_ALGOS:
        raise ValueError(f"unsupported off-policy algo {algo!r}; expected one of {DEFAULT_ALGOS}")
    return algo, task, _runtime_sim_backend(sim)


def _resolve_case_specs(
    cases_arg: str,
    *,
    algos_arg: str,
    backends: Sequence[str],
) -> list[str]:
    if cases_arg == "default":
        return _default_case_specs(backends)
    if cases_arg == "auto":
        algos = [part.strip() for part in algos_arg.split(",") if part.strip()]
        return [case for backend in backends for case in _discover_cases(algos=algos, sim=backend)]

    seen: set[str] = set()
    specs: list[str] = []
    for raw in cases_arg.split(","):
        spec = raw.strip()
        if not spec or spec in seen:
            continue
        _parse_case(spec)
        seen.add(spec)
        specs.append(spec)
    return specs


def _resolve_actor_spec(
    algo: str,
    cfg: DictConfig,
    *,
    obs_dim: int,
    critic_dim: int,
) -> tuple[str, dict[str, Any], bool]:
    if algo == "sac":
        from unilab.algos.torch.offpolicy.runtime import resolve_custom_offpolicy_runtime

        rl_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.algo, resolve=True))
        custom_runtime = resolve_custom_offpolicy_runtime(rl_cfg)
        if custom_runtime is None:
            return "sac", {}, bool(cfg.algo.use_layer_norm)
        algo_type = str(custom_runtime.algo_type or "sac")
        return (
            algo_type,
            custom_runtime.build_model_kwargs(obs_dim=obs_dim, critic_obs_dim=critic_dim),
            bool(cfg.algo.use_layer_norm),
        )

    if algo == "flashsac":
        return (
            "flashsac",
            {
                "actor_num_blocks": int(cfg.algo.algo_params.actor_num_blocks),
                "actor_noise_zeta_mu": float(cfg.algo.algo_params.actor_noise_zeta_mu),
                "actor_noise_zeta_max": int(cfg.algo.algo_params.actor_noise_zeta_max),
            },
            False,
        )

    if algo == "td3":
        return (
            "td3",
            {
                "init_scale": float(cfg.algo.algo_params.init_scale),
                "log_std_min": float(cfg.algo.algo_params.log_std_min),
                "log_std_max": float(cfg.algo.algo_params.log_std_max),
            },
            False,
        )

    raise ValueError(f"Unsupported algo={algo!r}")


def _build_case(
    cfg: DictConfig,
    *,
    algo: str,
    task: str,
    sim: str,
    variant: str,
    replay_capacity_steps: int,
    actor_algo_type: str,
    use_layer_norm: bool,
    obs_dim: int,
    critic_dim: int,
    action_dim: int,
) -> CollectorCase:
    num_envs = int(cfg.algo.num_envs)
    capacity_steps = max(1, int(replay_capacity_steps))
    numba_threads = OmegaConf.select(cfg, "env.numba_num_threads", default=None)
    return CollectorCase(
        algo=algo,
        task=task,
        sim=sim,
        variant=variant,
        numba_acceleration=bool(OmegaConf.select(cfg, "env.numba_acceleration", default=False)),
        numba_threads=int(numba_threads) if numba_threads is not None else None,
        runtime_sim_backend=str(cfg.training.sim_backend),
        command=f"uv run train --algo {algo} --task {task} --sim {cfg.training.sim_backend}",
        training_task_name=str(cfg.training.task_name),
        collector_algo_type=actor_algo_type,
        num_envs=num_envs,
        replay_capacity_rows=num_envs * capacity_steps,
        replay_capacity_steps=capacity_steps,
        obs_dim=int(obs_dim),
        critic_dim=int(critic_dim),
        action_dim=int(action_dim),
        actor_hidden_dim=int(cfg.algo.actor_hidden_dim),
        use_layer_norm=use_layer_norm,
        env_steps_per_sync=int(cfg.training.env_steps_per_sync),
    )


def _make_env_and_actor(
    cfg: DictConfig,
    *,
    algo: str,
    env_cfg_override: dict[str, Any] | None,
):
    from unilab.algos.torch.common.actor_factory import build_actor
    from unilab.base.observations import get_obs_dims
    from unilab.training import create_env

    env = create_env(cfg, num_envs=int(cfg.algo.num_envs), env_cfg_override=env_cfg_override)
    if env.state is None:
        env.init_state()
    obs_dim, critic_dim = get_obs_dims(env.obs_groups_spec)
    action_shape = env.action_space.shape
    if action_shape is None:
        env.close()
        raise ValueError("env.action_space.shape must be defined")
    action_dim = int(action_shape[0])
    actor_algo_type, actor_kwargs, use_layer_norm = _resolve_actor_spec(
        algo,
        cfg,
        obs_dim=int(obs_dim),
        critic_dim=int(critic_dim),
    )
    actor = build_actor(
        actor_algo_type,
        int(obs_dim),
        action_dim,
        int(cfg.algo.actor_hidden_dim),
        use_layer_norm,
        "cpu",
        int(cfg.algo.num_envs),
        **actor_kwargs,
    )
    actor.eval()
    return (
        env,
        actor,
        actor_algo_type,
        actor_kwargs,
        use_layer_norm,
        int(obs_dim),
        int(critic_dim),
        action_dim,
    )


def _run_active_window_case(
    case: CollectorCase,
    *,
    cfg: DictConfig,
    env,
    actor,
    warmup_steps: int,
    measure_steps: int,
    profile_numpy_random: bool = False,
) -> CollectorResult:
    from unilab.algos.torch.offpolicy.worker import (
        resolve_offpolicy_actor_priv_info,
        sample_offpolicy_actions,
    )
    from unilab.base.final_observation import resolve_terminal_observation_contract
    from unilab.base.observations import split_obs_dict
    from unilab.ipc.replay_buffer import ReplayBuffer

    replay_buffer = ReplayBuffer(
        capacity=case.replay_capacity_rows,
        obs_dim=case.obs_dim,
        action_dim=case.action_dim,
        critic_dim=case.critic_dim,
        device="cpu",
    )

    actions_np = np.zeros((case.num_envs, case.action_dim), dtype=np.float32)
    state = env.step(actions_np)
    obs_np, critic_np = split_obs_dict(state.obs)
    obs_np = np.asarray(obs_np, dtype=np.float32)
    critic_np = np.asarray(critic_np, dtype=np.float32)
    info_dict = state.info
    prev_dones_np = np.zeros(case.num_envs, dtype=np.float32)
    current_ep_rewards = np.zeros(case.num_envs, dtype=np.float32)
    current_ep_lengths = np.zeros(case.num_envs, dtype=np.int32)
    ep_rewards: list[float] = []
    ep_lengths: list[int] = []
    ep_reward_components: defaultdict[str, list[Any]] = defaultdict(list)

    samples: dict[str, list[float]] = {key: [] for key in COLLECTOR_PHASES}
    # Auxiliary env.step breakdown samples. These are sub-parts of env_step_ms,
    # so they must NOT be summed into total_active_ns (that would double-count).
    aux_samples: dict[str, list[float]] = {
        "physics_ms": [],
        "env_step_overhead_ms": [],
    }
    env_step_timing_samples: dict[str, list[float]] = {key: [] for key in NP_ENV_STEP_SAMPLE_KEYS}
    numpy_random_ms_samples: list[float] = []
    numpy_random_call_samples: list[float] = []
    random_profiler = _NumpyRandomProfiler() if profile_numpy_random else None
    total_active_ns = 0
    total_steps = int(warmup_steps) + int(measure_steps)
    if total_steps <= 0 or measure_steps <= 0:
        raise ValueError("measure_steps must be > 0 and warmup_steps must be >= 0")

    # System-wide CPU utilization across the measured window (skips warmup).
    # Must be system-wide, not per-process: the BatchEnvPool worker threads run
    # on other cores and a per-process reading would miss them, hiding the
    # "160 cores but only 25% utilized" scaling problem.
    cpu_probe_start: tuple[float, float] | None = None

    if random_profiler is not None:
        random_profiler.install()
    try:
        for step_idx in range(total_steps):
            record = step_idx >= warmup_steps
            if record and cpu_probe_start is None:
                cpu_probe_start = _read_cpu_times()  # begin measured-window CPU sampling

            phase_start_ns = time.perf_counter_ns()
            # No learner exists in this benchmark, so weight sync is intentionally a
            # zero-work phase. Keeping it explicit preserves the training collector's
            # phase schema.
            weight_sync_ms = (time.perf_counter_ns() - phase_start_ns) / 1e6

            phase_start_ns = time.perf_counter_ns()
            with torch.no_grad():
                obs_torch = torch.from_numpy(obs_np)
                dones_torch = torch.from_numpy(prev_dones_np)
                priv_info_np = resolve_offpolicy_actor_priv_info(
                    algo_type=case.collector_algo_type,
                    obs_np=obs_np,
                    critic_np=critic_np,
                    info=info_dict,
                )
                priv_info_torch = (
                    torch.from_numpy(priv_info_np) if priv_info_np is not None else None
                )
                actions_torch = sample_offpolicy_actions(
                    actor=actor,
                    algo_type=case.collector_algo_type,
                    obs_torch=obs_torch,
                    prev_dones_torch=dones_torch,
                    priv_info_torch=priv_info_torch,
                )
                actions_np = actions_torch.numpy()
            action_select_ms = (time.perf_counter_ns() - phase_start_ns) / 1e6

            phase_start_ns = time.perf_counter_ns()
            if random_profiler is not None and record:
                random_profiler.begin_step()
            try:
                state = env.step(actions_np)
            finally:
                if random_profiler is not None and record:
                    random_ms, random_calls = random_profiler.end_step()
                    numpy_random_ms_samples.append(random_ms)
                    numpy_random_call_samples.append(float(random_calls))
            env_step_ms = (time.perf_counter_ns() - phase_start_ns) / 1e6
            # Backend-internal physics time (sub-part of env_step_ms). Present for
            # MuJoCo (via backend.step timing); absent for backends that don't
            # report it -> NaN, rendered as "n/a".
            _timing = state.info.get("timing", {}) if isinstance(state.info, dict) else {}
            physics_ms = _optional_timing_ms(_timing, "backend_physics_ms")
            env_step_timing_values = {
                key: _optional_timing_ms(_timing, key)
                for key in NP_ENV_STEP_SAMPLE_KEYS
                if key != "env_step_internal_gap_ms"
            }
            internal_children = (
                env_step_timing_values["apply_action_ms"],
                env_step_timing_values["step_core_ms"],
                env_step_timing_values["update_state_ms"],
                env_step_timing_values["reset_done_ms"],
            )
            if all(value is not None for value in internal_children):
                env_step_timing_values["env_step_internal_gap_ms"] = env_step_ms - sum(
                    cast(float, value) for value in internal_children
                )
            else:
                env_step_timing_values["env_step_internal_gap_ms"] = None

            phase_start_ns = time.perf_counter_ns()
            next_obs_np, next_critic_np = split_obs_dict(state.obs)
            next_obs_np = np.asarray(next_obs_np, dtype=np.float32)
            next_critic_np = np.asarray(next_critic_np, dtype=np.float32)
            rewards_np = np.asarray(state.reward, dtype=np.float32).ravel()
            truncated_np = state.truncated.astype(np.float32, copy=False).ravel()
            combined_dones = (state.terminated | state.truncated).astype(
                np.float32, copy=False
            ).ravel()
            terminal_contract = resolve_terminal_observation_contract(
                next_obs_batch_size=next_obs_np.shape[0],
                final_observation=state.final_observation,
                done=combined_dones > 0.5,
                info=state.info,
                truncated=truncated_np,
            )
            replay_buffer.add(
                torch.from_numpy(obs_np),
                torch.from_numpy(actions_np),
                torch.from_numpy(rewards_np),
                torch.from_numpy(next_obs_np),
                torch.from_numpy(combined_dones),
                torch.from_numpy(truncated_np),
                terminal_mask=torch.from_numpy(terminal_contract.terminal_mask),
                terminal_next_obs=(
                    torch.from_numpy(terminal_contract.terminal_obs)
                    if terminal_contract.terminal_obs is not None
                    else None
                ),
                critic=torch.from_numpy(critic_np),
                next_critic=torch.from_numpy(next_critic_np),
                terminal_next_critic=(
                    torch.from_numpy(terminal_contract.terminal_critic)
                    if terminal_contract.terminal_critic is not None
                    else None
                ),
            )
            replay_ms = (time.perf_counter_ns() - phase_start_ns) / 1e6

            phase_start_ns = time.perf_counter_ns()
            current_ep_rewards += rewards_np
            current_ep_lengths += 1
            reset_indices = np.where(combined_dones > 0.5)[0]
            if len(reset_indices) > 0:
                ep_rewards.extend(current_ep_rewards[reset_indices].tolist())
                ep_lengths.extend(current_ep_lengths[reset_indices].tolist())
                current_ep_rewards[reset_indices] = 0.0
                current_ep_lengths[reset_indices] = 0

            log_info = state.info.get("log", {})
            if log_info:
                for key, value in log_info.items():
                    if key.startswith("reward/"):
                        ep_reward_components[key].append(value)
            bookkeeping_ms = (time.perf_counter_ns() - phase_start_ns) / 1e6

            obs_np = next_obs_np
            critic_np = next_critic_np
            info_dict = state.info
            prev_dones_np = combined_dones

            if record:
                phase_values = {
                    "weight_sync_ms": weight_sync_ms,
                    "action_select_ms": action_select_ms,
                    "env_step_ms": env_step_ms,
                    "replay_ms": replay_ms,
                    "bookkeeping_ms": bookkeeping_ms,
                }
                step_active_ns = int(sum(phase_values.values()) * 1e6)
                total_active_ns += step_active_ns
                for key, value in phase_values.items():
                    samples[key].append(value)
                # Env-step breakdown is aux only; env_step_ms is the additive phase.
                if physics_ms is not None:
                    aux_samples["physics_ms"].append(physics_ms)
                    aux_samples["env_step_overhead_ms"].append(env_step_ms - physics_ms)
                for key, value in env_step_timing_values.items():
                    if value is not None:
                        env_step_timing_samples[key].append(value)
    finally:
        if random_profiler is not None:
            random_profiler.uninstall()

    # Measured-window system CPU utilization (see cpu_probe_start comment).
    cpu_util_pct = _cpu_util_pct(cpu_probe_start, _read_cpu_times())

    total_active_ms = total_active_ns / 1e6
    collector_active_steps_per_sec = (case.num_envs * measure_steps) / (total_active_ms / 1000.0)
    phase_stats = {key: _stats(values) for key, values in samples.items() if values}
    phase_mean_total = sum(stat.mean_ms for stat in phase_stats.values())
    phase_pct = {
        key: (stat.mean_ms / phase_mean_total * 100.0 if phase_mean_total > 0.0 else 0.0)
        for key, stat in phase_stats.items()
    }

    del replay_buffer
    _cleanup()
    physics_stats = _stats(aux_samples["physics_ms"]) if aux_samples["physics_ms"] else None
    env_step_overhead_stats = (
        _stats(aux_samples["env_step_overhead_ms"]) if aux_samples["env_step_overhead_ms"] else None
    )
    env_step_timing_stats = {
        key: _stats(values) for key, values in env_step_timing_samples.items() if values
    }
    return CollectorResult(
        case=case,
        warmup_steps=int(warmup_steps),
        measure_steps=int(measure_steps),
        total_active_ms=total_active_ms,
        collector_active_steps_per_sec=collector_active_steps_per_sec,
        phase_ms_per_vector_step=phase_stats,
        phase_pct=phase_pct,
        notes=[],
        env_step_timing_ms_per_vector_step=env_step_timing_stats,
        physics_ms_per_vector_step=physics_stats,
        env_step_overhead_ms_per_vector_step=env_step_overhead_stats,
        cpu_util_pct=cpu_util_pct,
        numpy_random_ms_per_vector_step=(
            _stats(numpy_random_ms_samples) if numpy_random_ms_samples else None
        ),
        numpy_random_calls_per_vector_step=(
            _stats(numpy_random_call_samples) if numpy_random_call_samples else None
        ),
    )


def _build_and_run_case(
    spec: str,
    *,
    warmup_steps: int,
    measure_steps: int,
    replay_capacity_steps: int,
    num_envs: int | None,
    extra_overrides: list[str],
    variant: str = "default",
    profile_numpy_random: bool = False,
) -> CollectorResult:
    from unilab.training import BackendAdapter, ensure_registries
    from unilab.training.seed import apply_training_seed

    algo, task, sim = _parse_case(spec)
    owner_path = _owner_config_path(algo, task, sim)
    if not owner_path.is_file():
        raise FileNotFoundError(f"missing owner config for case {spec}: {owner_path}")

    cfg = _compose_offpolicy_cfg(
        algo,
        task,
        sim,
        num_envs=num_envs,
        extra_overrides=extra_overrides,
    )
    ensure_registries()
    # Mirror the real collector subprocess: cap torch CPU threads so this
    # in-process replica measures the same action_select/replay cost the
    # training collector sees.
    _configure_collector_cpu_threads()
    apply_training_seed(int(cfg.algo.seed), torch_runtime=True, cuda=True)
    env_cfg_override = BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name=algo,
    ).build_task_env_cfg_override()

    env = None
    try:
        (
            env,
            actor,
            actor_algo_type,
            _actor_kwargs,
            use_layer_norm,
            obs_dim,
            critic_dim,
            action_dim,
        ) = _make_env_and_actor(cfg, algo=algo, env_cfg_override=env_cfg_override)
        case = _build_case(
            cfg,
            algo=algo,
            task=task,
            sim=sim,
            variant=variant,
            replay_capacity_steps=replay_capacity_steps,
            actor_algo_type=actor_algo_type,
            use_layer_norm=use_layer_norm,
            obs_dim=obs_dim,
            critic_dim=critic_dim,
            action_dim=action_dim,
        )
        return _run_active_window_case(
            case,
            cfg=cfg,
            env=env,
            actor=actor,
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
            profile_numpy_random=profile_numpy_random,
        )
    finally:
        if env is not None:
            env.close()
        _cleanup()


def _numba_ab_overrides(*, enabled: bool, numba_threads: int | None) -> list[str]:
    overrides = [f"++env.numba_acceleration={'true' if enabled else 'false'}"]
    if enabled and numba_threads is not None:
        overrides.append(f"++env.numba_num_threads={int(numba_threads)}")
    return overrides


def _build_and_run_numba_ab_case(
    spec: str,
    *,
    warmup_steps: int,
    measure_steps: int,
    replay_capacity_steps: int,
    num_envs: int | None,
    extra_overrides: list[str],
    numba_threads: int | None,
    profile_numpy_random: bool = False,
) -> list[CollectorResult]:
    _algo, task, _sim = _parse_case(spec)
    if task not in NUMBA_ACCELERATED_TASKS:
        raise ValueError(
            f"--numba-ab only supports {sorted(NUMBA_ACCELERATED_TASKS)}, got task={task!r}"
        )
    records = []
    for variant, enabled in (
        ("numpy_baseline", False),
        ("numba_accelerated", True),
    ):
        records.append(
            _build_and_run_case(
                spec,
                warmup_steps=warmup_steps,
                measure_steps=measure_steps,
                replay_capacity_steps=replay_capacity_steps,
                num_envs=num_envs,
                extra_overrides=[
                    *extra_overrides,
                    *_numba_ab_overrides(enabled=enabled, numba_threads=numba_threads),
                ],
                variant=variant,
                profile_numpy_random=profile_numpy_random,
            )
        )
    return records


def _result_to_dict(result: CollectorResult) -> dict[str, Any]:
    data = asdict(result)
    data["phase_ms_per_vector_step"] = {
        key: asdict(value) for key, value in result.phase_ms_per_vector_step.items()
    }
    data["env_step_timing_ms_per_vector_step"] = {
        key: asdict(value) for key, value in result.env_step_timing_ms_per_vector_step.items()
    }
    # physics_ms_per_vector_step / cpu_util_pct are handled by asdict (None /
    # nested dict / float), so no special-casing needed beyond what asdict does.
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, results: list[CollectorResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "algo",
        "task",
        "sim",
        "variant",
        "numba_acceleration",
        "numba_threads",
        "runtime_sim_backend",
        "training_task_name",
        "collector_algo_type",
        "num_envs",
        "warmup_steps",
        "measure_steps",
        "collector_active_steps_per_sec",
        "total_active_ms",
        "action_select_ms",
        "env_step_ms",
        "replay_ms",
        "weight_sync_ms",
        "bookkeeping_ms",
        "numpy_random_ms",
        "numpy_random_calls",
        "physics_ms",
        "env_step_overhead_ms",
        "reset_done_count",
        *(field_name for _, field_name in NP_ENV_STEP_TIMING_CSV_FIELDS),
        "action_select_pct",
        "env_step_pct",
        "replay_pct",
        "weight_sync_pct",
        "bookkeeping_pct",
        "physics_pct",
        "env_step_overhead_pct",
        *(f"{field_name[:-3]}_pct" for _, field_name in NP_ENV_STEP_TIMING_CSV_FIELDS),
        "cpu_util_pct",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row: dict[str, Any] = {
                "algo": result.case.algo,
                "task": result.case.task,
                "sim": result.case.sim,
                "variant": result.case.variant,
                "numba_acceleration": result.case.numba_acceleration,
                "numba_threads": (
                    result.case.numba_threads if result.case.numba_threads is not None else ""
                ),
                "runtime_sim_backend": result.case.runtime_sim_backend,
                "training_task_name": result.case.training_task_name,
                "collector_algo_type": result.case.collector_algo_type,
                "num_envs": result.case.num_envs,
                "warmup_steps": result.warmup_steps,
                "measure_steps": result.measure_steps,
                "collector_active_steps_per_sec": result.collector_active_steps_per_sec,
                "total_active_ms": result.total_active_ms,
            }
            for key in COLLECTOR_PHASES:
                row[key] = result.phase_ms_per_vector_step.get(
                    key, TimingStats([], 0, 0, 0, 0, 0)
                ).mean_ms
            row["physics_ms"] = (
                result.physics_ms_per_vector_step.mean_ms
                if result.physics_ms_per_vector_step is not None
                else ""
            )
            row["numpy_random_ms"] = (
                result.numpy_random_ms_per_vector_step.mean_ms
                if result.numpy_random_ms_per_vector_step is not None
                else ""
            )
            row["numpy_random_calls"] = (
                result.numpy_random_calls_per_vector_step.mean_ms
                if result.numpy_random_calls_per_vector_step is not None
                else ""
            )
            row["env_step_overhead_ms"] = (
                result.env_step_overhead_ms_per_vector_step.mean_ms
                if result.env_step_overhead_ms_per_vector_step is not None
                else ""
            )
            reset_done_count = result.env_step_timing_ms_per_vector_step.get("reset_done_count")
            row["reset_done_count"] = (
                reset_done_count.mean_ms if reset_done_count is not None else ""
            )
            for timing_key, field_name in NP_ENV_STEP_TIMING_CSV_FIELDS:
                stat = result.env_step_timing_ms_per_vector_step.get(timing_key)
                row[field_name] = stat.mean_ms if stat is not None else ""
            for key in COLLECTOR_PHASES:
                row[key.replace("_ms", "_pct")] = result.phase_pct.get(key, "")
            env_step_mean = result.phase_ms_per_vector_step.get(
                "env_step_ms", TimingStats([], 0, 0, 0, 0, 0)
            ).mean_ms
            env_step_pct = result.phase_pct.get("env_step_ms", 0.0)
            row["physics_pct"] = (
                (result.physics_ms_per_vector_step.mean_ms / env_step_mean) * env_step_pct
                if result.physics_ms_per_vector_step is not None and env_step_mean > 0.0
                else ""
            )
            row["env_step_overhead_pct"] = (
                (result.env_step_overhead_ms_per_vector_step.mean_ms / env_step_mean) * env_step_pct
                if result.env_step_overhead_ms_per_vector_step is not None and env_step_mean > 0.0
                else ""
            )
            for timing_key, field_name in NP_ENV_STEP_TIMING_CSV_FIELDS:
                stat = result.env_step_timing_ms_per_vector_step.get(timing_key)
                row[f"{field_name[:-3]}_pct"] = (
                    (stat.mean_ms / env_step_mean) * env_step_pct
                    if stat is not None and env_step_mean > 0.0
                    else ""
                )
            row["cpu_util_pct"] = (
                result.cpu_util_pct if result.cpu_util_pct == result.cpu_util_pct else ""
            )
            writer.writerow(row)


def _run_text_command(command: Sequence[str]) -> str:
    try:
        return subprocess.check_output(
            list(command),
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        ).strip()
    except Exception:
        return ""


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _format_mhz(mhz: float) -> str:
    if mhz >= 1000.0:
        return f"{mhz / 1000.0:.2f} GHz"
    return f"{mhz:.0f} MHz"


def _format_khz(khz: float) -> str:
    return _format_mhz(khz / 1000.0)


def _clean_frequency(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        return "unknown"
    if value.lower() in {"unknown", "0 mhz", "0 mt/s"}:
        return "unknown"
    return value


def _is_known(value: Any) -> bool:
    return isinstance(value, str) and value.strip() and value.strip().lower() != "unknown"


def _detect_cpu_frequency() -> str:
    system = platform.system()
    if system == "Linux":
        lscpu = _run_text_command(["lscpu"])
        for key in ("CPU max MHz", "CPU MHz"):
            match = re.search(rf"^{re.escape(key)}:\s*([0-9.]+)", lscpu, re.MULTILINE)
            if match:
                return _format_mhz(float(match.group(1)))

        sys_freq = _read_text(Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"))
        if sys_freq.isdigit():
            return _format_khz(float(sys_freq))

        cpuinfo = _read_text(Path("/proc/cpuinfo"))
        match = re.search(r"^cpu MHz\s*:\s*([0-9.]+)", cpuinfo, re.MULTILINE)
        if match:
            return _format_mhz(float(match.group(1)))

    if system == "Darwin":
        for key in ("hw.cpufrequency_max", "hw.cpufrequency"):
            value = _run_text_command(["sysctl", "-n", key])
            if value.isdigit():
                return _format_mhz(float(value) / 1_000_000.0)
        brand = _run_text_command(["sysctl", "-n", "machdep.cpu.brand_string"])
        if brand.startswith("Apple "):
            return "dynamic (Apple Silicon)"

    if system == "Windows":
        output = _run_text_command(["wmic", "cpu", "get", "MaxClockSpeed", "/value"])
        match = re.search(r"MaxClockSpeed=(\d+)", output)
        if match:
            return _format_mhz(float(match.group(1)))

    return "unknown"


def _parse_memory_frequency(text: str) -> str:
    values: list[str] = []
    patterns = (
        r"Configured Memory Speed:\s*([^\n]+)",
        r"Speed:\s*([^\n]+)",
        r"clock:\s*([^\n]+)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            value = _clean_frequency(match.group(1))
            if value != "unknown" and value not in values:
                values.append(value)
    return values[0] if values else "unknown"


def _detect_memory_frequency() -> str:
    system = platform.system()
    if system == "Linux":
        for command in (["dmidecode", "--type", "17"], ["lshw", "-class", "memory"]):
            value = _parse_memory_frequency(_run_text_command(command))
            if value != "unknown":
                return value

    if system == "Darwin":
        mem_text = _run_text_command(["system_profiler", "SPMemoryDataType"])
        value = _parse_memory_frequency(mem_text)
        if value != "unknown":
            return value
        type_match = re.search(r"^\s*Type:\s*(.+)$", mem_text, re.MULTILINE)
        if type_match:
            memory_type = _clean_frequency(type_match.group(1))
            if memory_type != "unknown":
                return memory_type

    if system == "Windows":
        output = _run_text_command(["wmic", "memorychip", "get", "Speed", "/value"])
        speeds = re.findall(r"Speed=(\d+)", output)
        if speeds:
            return f"{speeds[0]} MHz"

    return "unknown"


def _get_benchmark_hardware_info() -> dict[str, str]:
    info = dict(get_device_info_dict())
    if not _is_known(info.get("cpu_frequency")):
        info["cpu_frequency"] = _detect_cpu_frequency()
    if not _is_known(info.get("memory_frequency")):
        info["memory_frequency"] = _detect_memory_frequency()
    return info


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(str(value)))

    def format_row(row: Sequence[str]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([format_row(headers), separator, *(format_row(row) for row in rows)])


def _format_hardware_table(hardware_info: dict[str, str]) -> str:
    headers = (
        "CPU model",
        "CPU cores",
        "CPU freq",
        "Memory",
        "Mem type/freq",
    )
    rows = [
        (
            hardware_info.get("chip", "unknown"),
            hardware_info.get("cpu_total_cores", "unknown"),
            hardware_info.get("cpu_frequency", "unknown"),
            hardware_info.get("memory", "unknown"),
            hardware_info.get("memory_frequency", "unknown"),
        )
    ]
    return _format_table(headers, rows)


def _phase_pct(result: CollectorResult, key: str) -> float:
    return result.phase_pct.get(key, 0.0)


def _env_step_child_pct(result: CollectorResult, child_ms: float) -> float:
    env_step = result.phase_ms_per_vector_step.get("env_step_ms")
    if env_step is None or env_step.mean_ms <= 0.0:
        return 0.0
    return (child_ms / env_step.mean_ms) * _phase_pct(result, "env_step_ms")


def _env_step_child_env_pct(result: CollectorResult, child_ms: float) -> float:
    env_step = result.phase_ms_per_vector_step.get("env_step_ms")
    if env_step is None or env_step.mean_ms <= 0.0:
        return 0.0
    return child_ms / env_step.mean_ms * 100.0


def _format_ms_pct(ms: float, pct: float) -> str:
    return f"{ms:.3f} ({pct:.1f}%)"


def _format_ms_env_active_pct(ms: float, env_pct: float, active_pct: float) -> str:
    return f"{ms:.3f} ({env_pct:.1f}%, {active_pct:.1f}%)"


def _format_np_env_timing(result: CollectorResult, key: str) -> str:
    stat = result.env_step_timing_ms_per_vector_step.get(key)
    if stat is None:
        return "n/a"
    return _format_ms_env_active_pct(
        stat.mean_ms,
        _env_step_child_env_pct(result, stat.mean_ms),
        _env_step_child_pct(result, stat.mean_ms),
    )


def _format_np_env_value(result: CollectorResult, key: str, *, digits: int = 1) -> str:
    stat = result.env_step_timing_ms_per_vector_step.get(key)
    if stat is None:
        return "n/a"
    return f"{stat.mean_ms:.{digits}f}"


def _format_set_state_sub_ms(result: CollectorResult, key: str) -> str:
    """Format a backend set_state sub-timing as ``ms (%of set_state)``.

    The percentage is relative to ``dr_reset_set_state_ms`` (the outer
    wall-clock measurement in DomainRandomizationManager), not to env_step_ms,
    so a reader can see which sub-step dominates set_state.
    """
    stat = result.env_step_timing_ms_per_vector_step.get(key)
    if stat is None:
        return "n/a"
    outer = result.env_step_timing_ms_per_vector_step.get("dr_reset_set_state_ms")
    if outer is None or outer.mean_ms <= 0.0:
        return f"{stat.mean_ms:.3f} (n/a)"
    pct = 100.0 * stat.mean_ms / outer.mean_ms
    return f"{stat.mean_ms:.3f} ({pct:.1f}%)"


def _format_throughput_table(results: list[CollectorResult]) -> str:
    headers = (
        "Algo",
        "Task",
        "Backend",
        "Variant",
        "Numba",
        "num_env",
        "Throughput env/s",
        "Total active ms",
        "Weight sync ms (% active)",
        "Action select ms (%)",
        "Env step ms (%)",
        "CPU %",
        "Replay ms (%)",
        "Bookkeeping ms (%)",
    )
    rows = []
    for result in results:
        phase_means = {
            key: result.phase_ms_per_vector_step.get(key, TimingStats([], 0, 0, 0, 0, 0)).mean_ms
            for key in COLLECTOR_PHASES
        }
        cpu_str = (
            f"{result.cpu_util_pct:.1f}"
            if result.cpu_util_pct == result.cpu_util_pct  # not NaN
            else "n/a"
        )
        rows.append(
            (
                result.case.algo,
                result.case.task,
                result.case.runtime_sim_backend,
                result.case.variant,
                (f"on/{result.case.numba_threads}T" if result.case.numba_acceleration else "off"),
                f"{result.case.num_envs:,}",
                f"{result.collector_active_steps_per_sec:,.0f}",
                f"{result.total_active_ms / result.measure_steps:.3f} (100.0%)",
                _format_ms_pct(phase_means["weight_sync_ms"], _phase_pct(result, "weight_sync_ms")),
                _format_ms_pct(
                    phase_means["action_select_ms"], _phase_pct(result, "action_select_ms")
                ),
                _format_ms_pct(phase_means["env_step_ms"], _phase_pct(result, "env_step_ms")),
                cpu_str,
                _format_ms_pct(phase_means["replay_ms"], _phase_pct(result, "replay_ms")),
                _format_ms_pct(phase_means["bookkeeping_ms"], _phase_pct(result, "bookkeeping_ms")),
            )
        )
    return _format_table(headers, rows)


def _format_reset_done_timing_table(results: list[CollectorResult]) -> str:
    headers = (
        "Algo",
        "Task",
        "Backend",
        "Reset count/step",
        "Reset done ms (% env, % active)",
        "Terminal obs ms (% env, % active)",
        "Reset call ms (% env, % active)",
        "Obs scatter ms (% env, % active)",
        "Info scatter ms (% env, % active)",
        "Reset gap ms (% env, % active)",
    )
    rows = []
    for result in results:
        env_step = result.phase_ms_per_vector_step.get("env_step_ms")
        if env_step is None:
            continue
        rows.append(
            (
                result.case.algo,
                result.case.task,
                result.case.runtime_sim_backend,
                _format_np_env_value(result, "reset_done_count"),
                _format_np_env_timing(result, "reset_done_ms"),
                _format_np_env_timing(result, "reset_done_terminal_obs_ms"),
                _format_np_env_timing(result, "reset_done_reset_call_ms"),
                _format_np_env_timing(result, "reset_done_obs_scatter_ms"),
                _format_np_env_timing(result, "reset_done_info_scatter_ms"),
                _format_np_env_timing(result, "reset_done_internal_gap_ms"),
            )
        )
    return _format_table(headers, rows)


def _format_dr_reset_timing_table(results: list[CollectorResult]) -> str:
    headers = (
        "Algo",
        "Task",
        "Backend",
        "DR reset total ms (% env, % active)",
        "Plan ms (% env, % active)",
        "Set state ms (% env, % active)",
        "Build reset obs ms (% env, % active)",
        "Motion lookup ms (% env, % active)",
        "Obs getters ms (% env, % active)",
        "Body pose getter ms (% env, % active)",
        "Obs compute ms (% env, % active)",
        "DR gap ms (% env, % active)",
    )
    rows = []
    for result in results:
        env_step = result.phase_ms_per_vector_step.get("env_step_ms")
        if env_step is None:
            continue
        rows.append(
            (
                result.case.algo,
                result.case.task,
                result.case.runtime_sim_backend,
                _format_np_env_timing(result, "dr_reset_total_ms"),
                _format_np_env_timing(result, "dr_reset_plan_ms"),
                _format_np_env_timing(result, "dr_reset_set_state_ms"),
                _format_np_env_timing(result, "dr_reset_build_observation_ms"),
                _format_np_env_timing(result, "dr_reset_obs_get_motion_ms"),
                _format_np_env_timing(result, "dr_reset_observation_getters_ms"),
                _format_np_env_timing(result, "dr_reset_obs_get_body_pose_ms"),
                _format_np_env_timing(result, "dr_reset_observation_compute_obs_ms"),
                _format_np_env_timing(result, "dr_reset_internal_gap_ms"),
            )
        )
    return _format_table(headers, rows)


_SET_STATE_MOTRIX_KEYS = (
    ("set_state_qpos_convert_ms", "QPos convert"),
    ("set_state_mask_ms", "Mask"),
    ("set_state_data_slice_ms", "Data slice"),
    ("set_state_data_reset_ms", "Data reset"),
    ("set_state_clear_forces_ms", "Clear forces"),
    ("set_state_geom_overrides_ms", "Geom overrides"),
    ("set_state_reset_rand_ms", "Reset rand"),
    ("set_state_set_dof_vel_ms", "Set dof vel"),
    ("set_state_set_dof_pos_ms", "Set dof pos"),
    ("set_state_actuator_ctrl_ms", "Actuator ctrl"),
    ("set_state_forward_kinematic_ms", "FK"),
    ("set_state_refresh_pose_cache_ms", "Refresh pose"),
    ("set_state_invalidate_velocity_ms", "Inval vel"),
    ("set_state_internal_gap_ms", "Gap"),
)

_SET_STATE_MUJOCO_KEYS = (
    ("set_state_qpos_convert_ms", "QPos convert"),
    ("set_state_pool_reset_ms", "Pool reset"),
    ("set_state_state_scatter_ms", "State scatter"),
    ("set_state_internal_gap_ms", "Gap"),
)


def _format_set_state_detail_table(results: list[CollectorResult]) -> str:
    """Backend set_state sub-timing table (motrix keyset).

    Renders the 14 motrix-oriented sub-keys next to the outer
    ``dr_reset_set_state_ms``. Backends that don't populate a key emit 0.0 so
    columns stay stable across backends. MuJoCo runs will show 0.0 for the
    motrix-only sub-keys; use :func:`_format_set_state_mujoco_table` for the
    MuJoCo-oriented view instead.
    """
    headers = (
        "Algo",
        "Task",
        "Backend",
        "Set state ms (%env, %active)",
        *(label for _, label in _SET_STATE_MOTRIX_KEYS),
    )
    rows = []
    for result in results:
        env_step = result.phase_ms_per_vector_step.get("env_step_ms")
        if env_step is None:
            continue
        rows.append(
            (
                result.case.algo,
                result.case.task,
                result.case.runtime_sim_backend,
                _format_np_env_timing(result, "dr_reset_set_state_ms"),
                *(_format_set_state_sub_ms(result, key) for key, _ in _SET_STATE_MOTRIX_KEYS),
            )
        )
    return _format_table(headers, rows)


def _format_set_state_mujoco_table(results: list[CollectorResult]) -> str:
    """Backend set_state sub-timing table (mujoco keyset)."""
    headers = (
        "Algo",
        "Task",
        "Backend",
        "Set state ms (%env, %active)",
        *(label for _, label in _SET_STATE_MUJOCO_KEYS),
    )
    rows = []
    for result in results:
        env_step = result.phase_ms_per_vector_step.get("env_step_ms")
        if env_step is None:
            continue
        rows.append(
            (
                result.case.algo,
                result.case.task,
                result.case.runtime_sim_backend,
                _format_np_env_timing(result, "dr_reset_set_state_ms"),
                *(_format_set_state_sub_ms(result, key) for key, _ in _SET_STATE_MUJOCO_KEYS),
            )
        )
    return _format_table(headers, rows)


def _format_np_env_step_timing_table(results: list[CollectorResult]) -> str:
    headers = (
        "Algo",
        "Task",
        "Backend",
        "Env step ms (% env, % active)",
        "NpEnv total ms (% env, % active)",
        "Apply action ms (% env, % active)",
        "Backend step ms (% env, % active)",
        "Update state ms (% env, % active)",
        "Reset done ms (% env, % active)",
        "Internal gap ms (% env, % active)",
    )
    rows = []
    for result in results:
        env_step = result.phase_ms_per_vector_step.get("env_step_ms")
        if env_step is None:
            continue
        rows.append(
            (
                result.case.algo,
                result.case.task,
                result.case.runtime_sim_backend,
                _format_ms_env_active_pct(
                    env_step.mean_ms,
                    100.0,
                    _phase_pct(result, "env_step_ms"),
                ),
                _format_np_env_timing(result, "env_step_total_ms"),
                _format_np_env_timing(result, "apply_action_ms"),
                _format_np_env_timing(result, "step_core_ms"),
                _format_np_env_timing(result, "update_state_ms"),
                _format_np_env_timing(result, "reset_done_ms"),
                _format_np_env_timing(result, "env_step_internal_gap_ms"),
            )
        )
    return _format_table(headers, rows)


def _format_env_step_breakdown_table(results: list[CollectorResult]) -> str:
    headers = (
        "Algo",
        "Task",
        "Backend",
        "Env step ms (% env, % active)",
        "Physics ms (% env, % active)",
        "Env overhead ms (% env, % active)",
        "Gap ms",
    )
    rows = []
    for result in results:
        env_step = result.phase_ms_per_vector_step.get("env_step_ms")
        if env_step is None:
            continue
        physics = result.physics_ms_per_vector_step
        overhead = result.env_step_overhead_ms_per_vector_step
        if physics is None or overhead is None:
            physics_str = "n/a"
            overhead_str = "n/a"
            gap_ms = "n/a"
        else:
            physics_ms_value = physics.mean_ms
            overhead_ms_value = overhead.mean_ms
            gap_ms_value = env_step.mean_ms - physics_ms_value - overhead_ms_value
            physics_str = _format_ms_env_active_pct(
                physics_ms_value,
                _env_step_child_env_pct(result, physics_ms_value),
                _env_step_child_pct(result, physics_ms_value),
            )
            overhead_str = _format_ms_env_active_pct(
                overhead_ms_value,
                _env_step_child_env_pct(result, overhead_ms_value),
                _env_step_child_pct(result, overhead_ms_value),
            )
            gap_ms = f"{gap_ms_value:.6f}"
        rows.append(
            (
                result.case.algo,
                result.case.task,
                result.case.runtime_sim_backend,
                _format_ms_env_active_pct(
                    env_step.mean_ms,
                    100.0,
                    _phase_pct(result, "env_step_ms"),
                ),
                physics_str,
                overhead_str,
                gap_ms,
            )
        )
    return _format_table(headers, rows)


def _mean_phase_ms(result: CollectorResult, key: str) -> float | None:
    stat = result.phase_ms_per_vector_step.get(key)
    return stat.mean_ms if stat is not None else None


def _mean_np_env_ms(result: CollectorResult, key: str) -> float | None:
    stat = result.env_step_timing_ms_per_vector_step.get(key)
    return stat.mean_ms if stat is not None else None


def _physics_ms(result: CollectorResult) -> float | None:
    return (
        result.physics_ms_per_vector_step.mean_ms
        if result.physics_ms_per_vector_step is not None
        else None
    )


def _collector_other_ms(result: CollectorResult) -> float:
    total = result.total_active_ms / float(result.measure_steps)
    physics = _physics_ms(result) or 0.0
    update_state = _mean_np_env_ms(result, "update_state_ms") or 0.0
    return total - physics - update_state


def _format_speedup(new: float, baseline: float) -> str:
    return f"{baseline / new:.2f}x" if new > 0.0 else "n/a"


def _format_saved(new: float, baseline: float) -> str:
    return f"{baseline - new:.3f}"


def _format_numba_ab_table(results: list[CollectorResult]) -> str:
    baseline_by_case = {
        (result.case.algo, result.case.task, result.case.sim, result.case.num_envs): result
        for result in results
        if result.case.variant == "numpy_baseline"
    }
    rows = []
    for result in results:
        if result.case.variant != "numba_accelerated":
            continue
        key = (result.case.algo, result.case.task, result.case.sim, result.case.num_envs)
        baseline = baseline_by_case.get(key)
        if baseline is None:
            continue
        base_total = baseline.total_active_ms / float(baseline.measure_steps)
        new_total = result.total_active_ms / float(result.measure_steps)
        base_env = _mean_phase_ms(baseline, "env_step_ms")
        new_env = _mean_phase_ms(result, "env_step_ms")
        base_update = _mean_np_env_ms(baseline, "update_state_ms")
        new_update = _mean_np_env_ms(result, "update_state_ms")
        base_physics = _physics_ms(baseline)
        new_physics = _physics_ms(result)
        base_other = _collector_other_ms(baseline)
        new_other = _collector_other_ms(result)
        rows.append(
            (
                result.case.algo,
                result.case.task,
                result.case.runtime_sim_backend,
                f"{result.case.num_envs:,}",
                "-" if result.case.numba_threads is None else str(result.case.numba_threads),
                f"{result.collector_active_steps_per_sec / baseline.collector_active_steps_per_sec:.2f}x",
                _format_speedup(new_total, base_total),
                _format_saved(new_total, base_total),
                "n/a"
                if base_env is None or new_env is None
                else _format_speedup(new_env, base_env),
                "n/a" if base_env is None or new_env is None else _format_saved(new_env, base_env),
                (
                    "n/a"
                    if base_update is None or new_update is None
                    else _format_speedup(new_update, base_update)
                ),
                (
                    "n/a"
                    if base_update is None or new_update is None
                    else _format_saved(new_update, base_update)
                ),
                (
                    "n/a"
                    if base_physics is None or new_physics is None
                    else _format_saved(new_physics, base_physics)
                ),
                _format_saved(new_other, base_other),
            )
        )
    if not rows:
        return ""
    headers = (
        "Algo",
        "Task",
        "Backend",
        "num_env",
        "Numba T",
        "Collector throughput",
        "Total step speedup",
        "Total saved ms",
        "Env step speedup",
        "Env saved ms",
        "Update speedup",
        "Update saved ms",
        "Physics saved ms",
        "Other saved ms",
    )
    return _format_table(headers, rows)


def _print_result(result: CollectorResult) -> None:
    case = result.case
    cpu_str = f"{result.cpu_util_pct:.1f}%" if result.cpu_util_pct == result.cpu_util_pct else "n/a"
    print(
        f"{case.algo}/{case.task}/{case.sim}: "
        f"Collector/s={result.collector_active_steps_per_sec:,.0f} "
        f"num_envs={case.num_envs:,} active_ms={result.total_active_ms:.1f} "
        f"cpu_util={cpu_str}"
    )
    for key in COLLECTOR_PHASES:
        stat = result.phase_ms_per_vector_step.get(key)
        if stat is None:
            continue
        pct = result.phase_pct.get(key, 0.0)
        print(f"  {key:<18} mean={stat.mean_ms:8.3f} ms  pct={pct:5.1f}%")
    # Physics is a sub-part of env_step_ms; the gap is upper-layer overhead
    # (obs/reward/reset) that dilutes the physics-layer speedup.
    if result.physics_ms_per_vector_step is not None:
        env_step = result.phase_ms_per_vector_step.get("env_step_ms")
        phys = result.physics_ms_per_vector_step.mean_ms
        upper = (
            result.env_step_overhead_ms_per_vector_step.mean_ms
            if result.env_step_overhead_ms_per_vector_step is not None
            else ((env_step.mean_ms - phys) if env_step is not None else float("nan"))
        )
        print(
            f"  {'physics_ms':<18} mean={phys:8.3f} ms  "
            f"pct_env={_env_step_child_env_pct(result, phys):5.1f}% "
            f"pct_active={_env_step_child_pct(result, phys):5.1f}%"
        )
        print(
            f"  {'env_step_overhead_ms':<18} mean={upper:8.3f} ms  "
            f"pct_env={_env_step_child_env_pct(result, upper):5.1f}% "
            f"pct_active={_env_step_child_pct(result, upper):5.1f}%"
        )
    if result.numpy_random_ms_per_vector_step is not None:
        random_ms = result.numpy_random_ms_per_vector_step.mean_ms
        random_calls = (
            result.numpy_random_calls_per_vector_step.mean_ms
            if result.numpy_random_calls_per_vector_step is not None
            else 0.0
        )
        print(
            f"  {'numpy_random_ms':<18} mean={random_ms:8.3f} ms  "
            f"calls={random_calls:5.1f}  "
            f"pct_env={_env_step_child_env_pct(result, random_ms):5.1f}% "
            f"pct_active={_env_step_child_pct(result, random_ms):5.1f}%"
        )
    if result.env_step_timing_ms_per_vector_step:
        reset_count = result.env_step_timing_ms_per_vector_step.get("reset_done_count")
        if reset_count is not None:
            print(f"  {'reset_done_count':<18} mean={reset_count.mean_ms:8.1f}")
        for key in (
            "apply_action_ms",
            "step_core_ms",
            "update_state_ms",
            "reset_done_ms",
            "env_step_internal_gap_ms",
            "reset_done_terminal_obs_ms",
            "reset_done_reset_call_ms",
            "reset_done_obs_scatter_ms",
            "reset_done_info_scatter_ms",
            "dr_reset_plan_ms",
            "dr_reset_set_state_ms",
            "dr_reset_build_observation_ms",
            "dr_reset_obs_get_motion_ms",
            "dr_reset_observation_getters_ms",
            "dr_reset_obs_get_body_pose_ms",
            "dr_reset_observation_compute_obs_ms",
        ):
            stat = result.env_step_timing_ms_per_vector_step.get(key)
            if stat is None:
                continue
            print(
                f"  {('np_env_' + key):<18} mean={stat.mean_ms:8.3f} ms  "
                f"pct_env={_env_step_child_env_pct(result, stat.mean_ms):5.1f}% "
                f"pct_active={_env_step_child_pct(result, stat.mean_ms):5.1f}%"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default="default",
        help=(
            "'default', 'auto', or comma-separated <algo>/<task>/<sim> cases. "
            "Default covers SAC G1 motion tracking and FlashSAC G1 walk flat on the selected backend. "
            "Run Sharpa explicitly with --cases sac/sharpa_inhand/mujoco_hora."
        ),
    )
    parser.add_argument(
        "--algos",
        default=",".join(DEFAULT_ALGOS),
        help="Comma-separated algos used only with --cases auto.",
    )
    parser.add_argument(
        "--backend",
        choices=BENCHMARK_BACKENDS,
        default=DEFAULT_BACKEND,
        help="Backend to benchmark for --cases default/auto. Default: motrix.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_backends",
        help="Benchmark all default backends (mujoco and motrix).",
    )
    parser.add_argument(
        "--sim",
        choices=(*BENCHMARK_BACKENDS, *BACKEND_ALIASES.keys()),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=DEFAULT_NUM_ENVS,
        help=f"Override algo.num_envs. Default: {DEFAULT_NUM_ENVS}.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=DEFAULT_WARMUP_STEPS,
        help=f"Warmup vector steps before timing. Default: {DEFAULT_WARMUP_STEPS}.",
    )
    parser.add_argument(
        "--measure-steps",
        type=int,
        default=DEFAULT_MEASURE_STEPS,
        help=f"Measured vector steps used for throughput statistics. Default: {DEFAULT_MEASURE_STEPS}.",
    )
    parser.add_argument(
        "--replay-capacity-steps",
        type=int,
        default=64,
        help="Replay capacity expressed in vectorized collector steps.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override. Can be passed more than once.",
    )
    parser.add_argument(
        "--numba-ab",
        action="store_true",
        help=(
            "Run numpy baseline and env.numba_acceleration=true variants for supported "
            "G1 tasks, then print an A/B bottleneck summary."
        ),
    )
    parser.add_argument(
        "--numba-threads",
        type=int,
        default=None,
        help="Numba thread count used by --numba-ab accelerated variants.",
    )
    parser.add_argument(
        "--profile-numpy-random",
        action="store_true",
        help=(
            "Measure selected np.random module calls during measured env.step() "
            "windows and report their collector active share."
        ),
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)
    if args.sim is not None:
        args.backend = _runtime_sim_backend(args.sim)
    return args


def main() -> int:
    args = parse_args()
    backends = _resolve_backend_selection(
        backend=args.backend,
        all_backends=bool(args.all_backends),
    )
    specs = _resolve_case_specs(args.cases, algos_arg=args.algos, backends=backends)
    if not specs:
        raise SystemExit("No benchmark cases resolved.")

    hardware_info = _get_benchmark_hardware_info()
    print(f"Device: {get_device_info_line()}")
    print(f"Cases: {', '.join(specs)}")

    results: list[CollectorResult] = []
    errors: list[dict[str, str]] = []
    for spec in specs:
        try:
            if args.numba_ab:
                case_results = _build_and_run_numba_ab_case(
                    spec,
                    warmup_steps=int(args.warmup_steps),
                    measure_steps=int(args.measure_steps),
                    replay_capacity_steps=int(args.replay_capacity_steps),
                    num_envs=args.num_envs,
                    extra_overrides=list(args.override),
                    numba_threads=args.numba_threads,
                    profile_numpy_random=bool(args.profile_numpy_random),
                )
            else:
                case_results = [
                    _build_and_run_case(
                        spec,
                        warmup_steps=int(args.warmup_steps),
                        measure_steps=int(args.measure_steps),
                        replay_capacity_steps=int(args.replay_capacity_steps),
                        num_envs=args.num_envs,
                        extra_overrides=list(args.override),
                        profile_numpy_random=bool(args.profile_numpy_random),
                    )
                ]
            results.extend(case_results)
            for result in case_results:
                _print_result(result)
        except Exception as exc:
            error = {"case": spec, "type": type(exc).__name__, "message": str(exc)}
            errors.append(error)
            print(f"{spec}: ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                raise

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "offpolicy_collector_active",
        "definition": (
            "active collect window excludes learner-wait time and measures the "
            "collector hot loop: action selection, env.step including reward computation, "
            "terminal-contract handling, replay writes, and collector-side bookkeeping."
        ),
        "device": hardware_info,
        "args": {
            "cases": args.cases,
            "algos": args.algos,
            "backend": args.backend,
            "all_backends": args.all_backends,
            "num_envs": args.num_envs,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "replay_capacity_steps": args.replay_capacity_steps,
            "override": args.override,
            "numba_ab": args.numba_ab,
            "numba_threads": args.numba_threads,
            "profile_numpy_random": args.profile_numpy_random,
        },
        "results": [_result_to_dict(result) for result in results],
        "errors": errors,
    }
    _write_json(args.out_json, payload)
    print(f"Wrote JSON: {args.out_json}")
    if args.out_csv is not None:
        _write_csv(args.out_csv, results)
        print(f"Wrote CSV: {args.out_csv}")
    print("\nHardware summary:")
    print(_format_hardware_table(hardware_info))
    print("\nTask throughput (active phases; phase percentages add to 100%):")
    if results:
        print(_format_throughput_table(results))
        if args.numba_ab:
            print(
                "\nNumba A/B summary (baseline is env.numba_acceleration=false; positive saved ms means the numba run is faster):"
            )
            table = _format_numba_ab_table(results)
            print(table if table else "No paired Numba A/B results.")
        print(
            "\nEnv step breakdown (subparts of Env step; do not add Env step together with its subparts):"
        )
        print(_format_env_step_breakdown_table(results))
        print(
            "\nNpEnv step timing (subparts reported by NpEnv.step; gap = external env_step_ms - listed subparts):"
        )
        print(_format_np_env_step_timing_table(results))
        print("\nReset done timing (subparts of NpEnv reset_done_ms):")
        print(_format_reset_done_timing_table(results))
        print(
            "\nDR reset timing (subparts of reset call; reset obs getters currently read full batch):"
        )
        print(_format_dr_reset_timing_table(results))
        print(
            "\nBackend set_state detail — motrix keyset "
            "(sub-timings sum to dr_reset_set_state_ms; % is share of that outer wall-clock):"
        )
        print(_format_set_state_detail_table(results))
        print("\nBackend set_state detail — mujoco keyset:")
        print(_format_set_state_mujoco_table(results))
    else:
        print("No successful benchmark cases.")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
