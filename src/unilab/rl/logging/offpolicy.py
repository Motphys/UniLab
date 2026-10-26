"""Rich-based training logger for off-policy RL algorithms (SAC, etc)."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from unilab.rl.logging.base import (
    BaseTrainingLogger,
    _fmt_number,
    _load_wandb,
    _set_backend_scalar,
)
from unilab.rl.logging.metric_schema import (
    LOGGER_OWNED_METRICS,
    normalize_metric_map,
    reward_term_key,
    validate_metric_tags,
)
from unilab.rl.logging.runtime_manifest_schema import RUNTIME_MANIFEST_SCHEMA_VERSION

_COLLECTOR_WAIT_TIMING_SPEC = (
    "Perf/learner_collector_wait_ms",
    "Collector Wait",
    "_collector_wait_time",
)
_INFERENCE_TIMING_SPEC = ("Perf/learner_inference_ms", "Inference", "_inference_time")
_COLLECTOR_RELEASE_TIMING_SPEC = (
    "Perf/learner_collector_release_ms",
    "Collector Release",
    "_sync_coordination_time",
)
_REPLAY_BATCH_WAIT_TIMING_SPEC = (
    "Perf/learner_replay_batch_wait_ms",
    "Replay Batch Wait",
    "_replay_batch_wait_time",
)
_REPLAY_STAGE_TIMING_SPEC = (
    "Perf/learner_replay_stage_ms",
    "Replay Stage",
    "_learner_replay_stage_time",
)
_REPLAY_SAMPLE_TIMING_SPEC = (
    "Perf/learner_replay_sample_ms",
    "Replay Sample",
    "_learner_replay_sample_time",
)
_TRAIN_TIMING_SPEC = ("Perf/learning_time", "Learning", "_train_time")
_WEIGHT_PUBLISH_TIMING_SPEC = (
    "Perf/learner_weight_publish_ms",
    "Weight Publish",
    "_weight_sync_time",
)

_LEARNER_TIMING_PROFILES = {
    "sac_family": (
        _COLLECTOR_WAIT_TIMING_SPEC,
        _INFERENCE_TIMING_SPEC,
        _COLLECTOR_RELEASE_TIMING_SPEC,
        _REPLAY_BATCH_WAIT_TIMING_SPEC,
        _REPLAY_SAMPLE_TIMING_SPEC,
        _TRAIN_TIMING_SPEC,
    ),
    "appo": (
        _COLLECTOR_WAIT_TIMING_SPEC,
        _REPLAY_STAGE_TIMING_SPEC,
        _REPLAY_SAMPLE_TIMING_SPEC,
        _TRAIN_TIMING_SPEC,
        _WEIGHT_PUBLISH_TIMING_SPEC,
    ),
}

_SAC_FAMILY_DETAIL_TIMING_SPECS = (
    ("Perf/learner_inference_h2d_ms", "Inference H2D", "_inference_h2d_time"),
    (
        "Perf/learner_inference_forward_ms",
        "Inference Forward",
        "_inference_forward_time",
    ),
    ("Perf/learner_inference_d2h_ms", "Inference D2H", "_inference_d2h_time"),
    (
        "Perf/replay_ingress_h2d_submit_ms",
        "Replay H2D Submit",
        "_replay_ingress_h2d_submit_time",
    ),
)

_LEARNER_DETAIL_TIMING_PROFILES = {
    "sac_family": _SAC_FAMILY_DETAIL_TIMING_SPECS,
    "appo": (),
}

_LEARNER_OTHER_TIMING_LABEL = "Other"
_ITER_WALL_TIMING_SPEC = ("Perf/iteration_time", "Iter Wall", "")

_METRIC_DISPLAY_LABELS = {
    "Loss/surrogate": "Surrogate",
    "Loss/value": "Value",
    "Loss/entropy": "Entropy",
    "Loss/actor": "Actor",
    "Loss/critic": "Critic",
    "Loss/temperature": "Temperature",
    "Loss/learning_rate": "Learning Rate",
    "Policy/mean_std": "Mean Std",
    "Policy/temperature": "Temperature",
    "Train/global_gradient_norm": "Global Grad Norm",
    "Train/actor_gradient_norm": "Actor Grad Norm",
    "Train/critic_gradient_norm": "Critic Grad Norm",
    "Train/target_q_max": "Target Q Max",
    "Train/target_q_min": "Target Q Min",
    "Train/reward_scale_std": "Reward Scale Std",
    "Train/ring_available_slots": "Ring Slots",
    "Train/rollouts_read": "Rollouts Read",
    "Train/approx_kl": "Approx KL",
    "Train/clip_fraction": "Clip Fraction",
    "Train/behavior_to_current_log_prob_delta": "LogProb Delta",
    "Train/vtrace_rho_clip_fraction": "VTrace Rho Clip",
    "Train/vtrace_rho_p99": "VTrace Rho p99",
    "Train/replay_ingress_depth": "Ingress Depth",
    "Train/replay_ingress_occupancy": "Ingress Occupancy",
    "Train/replay_ingress_high_water": "Ingress High Water",
    "Train/replay_ingress_backpressure_wait_ms": "Ingress Backpressure",
    "Train/replay_ingress_dropped_batches": "Ingress Drops",
}
_REPLAY_INGRESS_PREFIX = "Train/replay_ingress_"

_COLLECTOR_TIMING_SPECS = {
    "mlp_infer_ms": (1.0, "MLP Infer", "per_step"),
    "inference_request_ms": (1.0, "Inference Request", "cycle_phase"),
    "learner_action_wait_ms": (1.1, "Learner Action Wait", "cycle_phase"),
    "env_step_ms": (2.0, "Env Step", "cycle_phase"),
    "env_step_action_validate_ms": (2.05, "  Action Validate", "env_step_detail"),
    "env_step_apply_action_ms": (2.08, "  Apply Action", "env_step_detail"),
    "env_step_action_backend_ms": (2.1, "  Action+Backend", "env_step_detail"),
    "env_step_backend_ms": (2.1, "  Backend Step", "env_step_detail"),
    "env_step_update_state_ms": (2.2, "  Update State", "env_step_detail"),
    "env_step_reset_done_ms": (2.3, "  Reset Done", "env_step_detail"),
    "transition_extract_ms": (2.4, "Transition Extract", "cycle_phase"),
    "replay_write_ms": (3.0, "Replay Write", "cycle_phase"),
    "metrics_publish_ms": (4.0, "Metrics Publish", "cycle_phase"),
    "rollout_ms": (9.0, "Rollout Wall", "rollout_total"),
}

_COLLECTOR_INFERENCE_TAG_SPECS: dict[str, str | None] = {
    "queue_depth": "Perf/collector_inference_queue_depth",
    "action_backlog": "Perf/collector_inference_action_backlog",
    "max_action_backlog": "Perf/collector_inference_max_action_backlog",
    "in_flight": "Perf/collector_inference_in_flight",
    "max_in_flight": "Perf/collector_inference_max_in_flight",
    # Wait time is persisted by the collector timing map; keep the value in the
    # inference snapshot without emitting a duplicate canonical tag.
    "wait_time_ms": None,
    "publication_lag": "Perf/collector_inference_publication_lag",
    "max_publication_lag": "Perf/collector_inference_max_publication_lag",
}

OFFPOLICY_ENV_STEP_DETAIL_KEYS = (
    "env_step_action_validate_ms",
    "env_step_apply_action_ms",
    "env_step_action_backend_ms",
    "env_step_backend_ms",
    "env_step_update_state_ms",
    "env_step_reset_done_ms",
)

_OFFPOLICY_COLLECTOR_CYCLE_KEYS = tuple(
    key for key, (_, _, role) in _COLLECTOR_TIMING_SPECS.items() if role == "cycle_phase"
)
_COLLECTOR_CORE_TERMINAL_KEYS = frozenset(
    {
        "mlp_infer_ms",
        "learner_action_wait_ms",
        "env_step_ms",
        "replay_write_ms",
        "rollout_ms",
    }
)
_COLLECTOR_DIAGNOSTIC_TERMINAL_MIN_PCT = 1.0

_TERMINAL_AVERAGE_WINDOW_SEC = 2.0
_TERMINAL_AVERAGE_MAX_SAMPLES = 512


@dataclass(frozen=True)
class _TerminalSample:
    timestamp: float
    scalars: dict[str, float]
    metrics: dict[str, float]
    reward_components: dict[str, float]
    collector_timing: dict[str, float]


@dataclass(frozen=True)
class _TerminalSnapshot:
    iteration: int
    sample_count: int
    scalars: dict[str, float]
    metrics: dict[str, float]
    reward_components: dict[str, float]
    collector_timing: dict[str, float]
    reward_history: tuple[float, ...]
    buffer_size: int
    batch_size_per_rank: int


class OffPolicyLogger(BaseTrainingLogger):
    """Rich logger for off-policy RL algorithms (SAC, etc)."""

    def __init__(
        self,
        algo_name: str = "RL",
        max_iterations: int = 1500,
        num_envs: int = 4096,
        env_name: str = "",
        obs_dim: int = 0,
        action_dim: int = 0,
        num_gpus: int = 1,
        refresh_per_second: int = 2,
        log_dir: str = "",
        log_backend: str = "tensorboard",
        wandb_project: str = "unilab",
        wandb_entity: str | None = None,
        wandb_name: str = "",
        wandb_group: str | None = None,
        wandb_job_type: str | None = None,
        wandb_tags: list[str] | None = None,
        wandb_notes: str | None = None,
        timing_profile: str = "sac_family",
        log_interval: int = 1,
    ):
        super().__init__(
            algo_name=algo_name,
            max_iterations=max_iterations,
            num_envs=num_envs,
            env_name=env_name,
            log_dir=log_dir,
            log_backend=log_backend,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_name=wandb_name,
            wandb_group=wandb_group,
            wandb_job_type=wandb_job_type,
            wandb_tags=wandb_tags,
            wandb_notes=wandb_notes,
            refresh_per_second=refresh_per_second,
            fixed_terminal_refresh=True,
            tensorboard_subdir=None,
            log_interval=log_interval,
            wandb_config={
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "max_iterations": max_iterations,
            },
        )
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        if int(num_gpus) < 1:
            raise ValueError("num_gpus must be >= 1")
        self.num_gpus = int(num_gpus)
        self._total_steps: int = 0
        self._buffer_size: int = 0
        self._buffer_target: int = 0
        self._collector_wait_time: float = 0.0
        self._replay_batch_wait_time: float = 0.0
        self._learner_replay_stage_time: float = 0.0
        self._learner_replay_sample_time: float = 0.0
        self._sync_coordination_time: float = 0.0
        self._replay_ingress_h2d_submit_time: float = 0.0
        self._weight_sync_time: float = 0.0
        self._inference_h2d_time: float = 0.0
        self._inference_forward_time: float = 0.0
        self._inference_d2h_time: float = 0.0
        self._inference_time: float = 0.0
        self._iteration_time: float | None = None
        self._throughput_env_steps: int = 0
        # Keep a bounded tail window for a stable end-to-end throughput gate.
        # The final single iteration is intentionally not authoritative: one
        # iteration can absorb asynchronous collector/learner jitter. The
        # 100-iteration window also spans replay ingress and asynchronous
        # scheduling effects that remain visible over tens of iterations.
        self.tail_window_iterations = 100
        self._tail_iteration_times: deque[float] = deque(maxlen=self.tail_window_iterations)
        self._tail_throughput_env_steps = 0
        self._env_steps_per_sec_override: float | None = None
        self._learner_replay_rows_per_sec_override: float | None = None
        self._batch_size_per_rank: int = 0
        self._effective_batch_size: int = 0
        self._learner_replay_rows_per_iter: int = 0
        self._has_iteration_extra_info: bool = False
        self._collector_timing: dict[str, float] = {}
        self._collector_inference: dict[str, float] = {}
        self._curriculum_metrics: dict[str, float] = {}
        if timing_profile not in _LEARNER_TIMING_PROFILES:
            raise ValueError("timing_profile must be 'sac_family' or 'appo'")
        self._timing_profile = timing_profile
        self._timeout_rate: float | None = None
        self._buffer_utilization: float = 0.0
        self._runtime_manifest: dict[str, Any] = {}
        self._status: str = "Initializing..."
        self._training_timer_started: bool = False
        self._terminal_samples: deque[_TerminalSample] = deque(maxlen=_TERMINAL_AVERAGE_MAX_SAMPLES)
        self._terminal_snapshot: _TerminalSnapshot | None = None
        self._last_reward_iter: int | None = None
        self._last_reward_components_iter: int | None = None

    def _format_tensorboard_message(self, tb_dir: str) -> str:
        return f"[dim]TensorBoard logging to: {tb_dir}[/]"

    def _format_wandb_message(self, project: str, name: str) -> str:
        return f"[dim]W&B logging to project: {project}, run: {name}[/]"

    def start(self, *, status: str = "Warming up..."):
        super().start(status=status)

    def start_training_timer(self) -> float:
        """Start the measured training window after collector warm-up is complete."""
        if not self._training_timer_started:
            self._start_time = time.time()
            self._training_timer_started = True
        return self._start_time

    def finish(self, *, title: str = "Training Summary", extra_summary: str = ""):
        super().finish(
            title=title,
            extra_summary=f"  Total env steps: [yellow]{self._total_steps:,}[/]\n{extra_summary}",
        )

    def log_buffer_fill(self, current: int, target: int):
        self._buffer_size = current
        self._buffer_target = target
        pct = current / max(target, 1) * 100
        self._status = f"Buffer fill: {current:,}/{target:,} ({pct:.0f}%)"

    def _get_iter_env_steps_per_sec(self) -> float | None:
        if self._env_steps_per_sec_override is not None:
            return self._env_steps_per_sec_override
        if not self._has_iteration_extra_info or self._throughput_env_steps <= 0:
            return None
        iter_time = self._get_measured_iter_wall_time()
        if iter_time is None or iter_time <= 0:
            return None
        return self._throughput_env_steps / iter_time

    def _get_tail_env_steps_per_sec(self) -> float | None:
        """Aggregate env throughput over the bounded recent-iteration window."""
        if not self._tail_iteration_times:
            return None
        # The first queued sample predates this iteration's freshly assigned
        # step count only when the queue was empty; use the current aggregate,
        # which is exact after the per-iteration enqueue above.
        elapsed = sum(self._tail_iteration_times)
        if elapsed <= 0.0 or self._tail_throughput_env_steps <= 0:
            return None
        return self._tail_throughput_env_steps / elapsed

    def _get_learner_replay_rows_per_sec(self) -> float | None:
        if self._learner_replay_rows_per_sec_override is not None:
            return self._learner_replay_rows_per_sec_override
        if not self._has_iteration_extra_info or self._learner_replay_rows_per_iter <= 0:
            return None
        iter_time = self._get_measured_iter_wall_time()
        if iter_time is None or iter_time <= 0:
            return None
        return self._learner_replay_rows_per_iter / iter_time

    def _learner_phase_times(self) -> dict[str, float]:
        """Return mutually exclusive learner main-thread phases by backend key."""
        return {
            key: float(getattr(self, attribute))
            for key, _, attribute in self._learner_timing_specs()
        }

    def _learner_timing_specs(self) -> tuple[tuple[str, str, str], ...]:
        """Return phases applicable to the selected algorithm timing contract."""
        return _LEARNER_TIMING_PROFILES[self._timing_profile]

    def _learner_detail_times(self) -> dict[str, float]:
        """Return nested or asynchronous diagnostics that are not wall-clock slices."""
        return {
            key: float(getattr(self, attribute))
            for key, _, attribute in _LEARNER_DETAIL_TIMING_PROFILES[self._timing_profile]
        }

    def _get_learner_accounted_time(self) -> float:
        return sum(self._learner_phase_times().values())

    def _get_learner_other_time(self) -> float:
        return max(self._get_iter_wall_time() - self._get_learner_accounted_time(), 0.0)

    def _get_iter_wall_time(self) -> float:
        measured = self._get_measured_iter_wall_time()
        return measured if measured is not None else self._get_learner_accounted_time()

    def _get_measured_iter_wall_time(self) -> float | None:
        if self._iteration_time is not None and self._iteration_time > 0.0:
            return self._iteration_time
        return None

    def _get_collector_cycle_ms(
        self,
        collector_timing: dict[str, float] | None = None,
    ) -> float | None:
        if self._timing_profile != "sac_family":
            return None
        timing = self._collector_timing if collector_timing is None else collector_timing
        if not all(key in timing for key in _OFFPOLICY_COLLECTOR_CYCLE_KEYS):
            return None
        return sum(timing.get(key, 0.0) for key in _OFFPOLICY_COLLECTOR_CYCLE_KEYS)

    def _build_compact_header(
        self,
        *,
        include_status: bool,
        include_identity: bool = True,
        include_iteration: bool = True,
        extra_fields: list[tuple[str, str]] | None = None,
        terminal_snapshot: _TerminalSnapshot | None = None,
    ) -> Text:
        snapshot = terminal_snapshot or self._terminal_snapshot
        iter_steps_per_sec = (
            snapshot.scalars.get("env_steps_per_sec")
            if snapshot is not None
            else self._get_iter_env_steps_per_sec()
        )
        learner_replay_rows_per_sec = (
            snapshot.scalars.get("learner_replay_rows_per_sec")
            if snapshot is not None
            else self._get_learner_replay_rows_per_sec()
        )
        header_extra_fields: list[tuple[str, str]] = []
        if iter_steps_per_sec is not None:
            header_extra_fields.append((f"Steps/s {iter_steps_per_sec:,.0f}", "bold green"))
        if learner_replay_rows_per_sec is not None:
            header_extra_fields.append((f"Rows/s {learner_replay_rows_per_sec:,.0f}", "bold cyan"))
        if snapshot is not None:
            header_extra_fields.append(
                (
                    f"Avg {_TERMINAL_AVERAGE_WINDOW_SEC:g}s (n={snapshot.sample_count})",
                    "dim",
                )
            )
        if extra_fields:
            header_extra_fields.extend(extra_fields)
        return self._build_compact_header_state(
            include_status=include_status,
            include_identity=include_identity,
            include_iteration=include_iteration,
            extra_fields=header_extra_fields,
            ep_length=(
                snapshot.scalars.get("mean_ep_length", 0.0)
                if snapshot is not None
                else self._mean_ep_length
            ),
            current_iteration=(snapshot.iteration if snapshot is not None else self._iteration),
        )

    @staticmethod
    def _mean_sample_maps(
        samples: tuple[_TerminalSample, ...],
        attribute: str,
    ) -> dict[str, float]:
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for sample in samples:
            values = getattr(sample, attribute)
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
        return {key: total / counts[key] for key, total in totals.items()}

    def _record_terminal_sample(
        self,
        *,
        metrics: dict[str, float] | None,
        reward: float | None,
        reward_components: dict[str, float] | None,
    ) -> None:
        """Publish one immutable, time-smoothed terminal-only state snapshot."""
        scalars = {
            attribute: float(getattr(self, attribute))
            for _, _, attribute in self._learner_timing_specs()
        }
        scalars.update(
            {
                "iter_wall_time": self._get_iter_wall_time(),
                "learner_other_time": self._get_learner_other_time(),
            }
        )
        if self._timeout_rate is not None:
            scalars["timeout_rate"] = self._timeout_rate
        if self._mean_ep_length > 0.0:
            scalars["mean_ep_length"] = self._mean_ep_length
        if reward is not None:
            scalars["reward"] = float(reward)
        steps_per_sec = self._get_iter_env_steps_per_sec()
        if steps_per_sec is not None:
            scalars["env_steps_per_sec"] = steps_per_sec
        samples_per_sec = self._get_learner_replay_rows_per_sec()
        if samples_per_sec is not None:
            scalars["learner_replay_rows_per_sec"] = samples_per_sec

        now = time.monotonic()
        self._terminal_samples.append(
            _TerminalSample(
                timestamp=now,
                scalars=scalars,
                metrics=dict(metrics or {}),
                reward_components=dict(reward_components or {}),
                collector_timing=dict(self._collector_timing),
            )
        )
        cutoff = now - _TERMINAL_AVERAGE_WINDOW_SEC
        while self._terminal_samples and self._terminal_samples[0].timestamp < cutoff:
            self._terminal_samples.popleft()

        samples = tuple(self._terminal_samples)
        self._terminal_snapshot = _TerminalSnapshot(
            iteration=self._iteration,
            sample_count=len(samples),
            scalars=self._mean_sample_maps(samples, "scalars"),
            metrics=self._mean_sample_maps(samples, "metrics"),
            reward_components=self._mean_sample_maps(samples, "reward_components"),
            collector_timing=self._mean_sample_maps(samples, "collector_timing"),
            reward_history=tuple(self._reward_history),
            buffer_size=self._buffer_size,
            batch_size_per_rank=self._batch_size_per_rank,
        )

    def update_curriculum_metrics(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            if not key.startswith("Curriculum/"):
                raise ValueError(f"invalid curriculum metric key: {key}")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"curriculum metric {key} must be numeric")
        self._curriculum_metrics.update(metrics)

    def update_collector_timing(self, timing_ms: dict[str, float]):
        normalized = dict(timing_ms)
        unknown = sorted(set(normalized) - set(_COLLECTOR_TIMING_SPECS))
        if unknown:
            names = ", ".join(unknown)
            raise ValueError(f"unregistered collector timing keys: {names}")
        self._collector_timing.update(normalized)

    def update_collector_inference(self, diagnostics: dict[str, float | int]):
        unknown = sorted(set(diagnostics) - set(_COLLECTOR_INFERENCE_TAG_SPECS))
        if unknown:
            names = ", ".join(unknown)
            raise ValueError(f"unregistered collector inference keys: {names}")
        normalized: dict[str, float] = {}
        for key, value in diagnostics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"collector inference {key} must be numeric")
            number = float(value)
            if number < 0.0:
                raise ValueError(f"collector inference {key} must be non-negative")
            normalized[key] = number
        self._collector_inference.update(normalized)

    def update_timeout_rate(self, timeout_rate: float):
        self._timeout_rate = float(timeout_rate)

    def update_buffer_utilization(self, utilization: float):
        self._buffer_utilization = float(utilization)

    def update_runtime_manifest(self, manifest: dict[str, Any]) -> None:
        schema_version = manifest.get("schema_version")
        if schema_version is not None and schema_version != RUNTIME_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "unsupported runtime_manifest.schema_version "
                f"{schema_version}; expected {RUNTIME_MANIFEST_SCHEMA_VERSION}"
            )
        existing_version = self._runtime_manifest.get("schema_version")
        if (
            existing_version is not None
            and schema_version is not None
            and existing_version != schema_version
        ):
            raise ValueError("runtime_manifest.schema_version cannot change during a run")
        self._runtime_manifest.update(manifest)

    def log_collector(self, total_steps: int, buffer_size: int):
        self._total_steps = total_steps
        self._buffer_size = buffer_size

    def log_step(
        self,
        iteration: int,
        metrics: dict[str, float] | None = None,
        return_mean_ep100: float | None = None,
        reward_components: dict[str, float] | None = None,
        train_time: float = 0.0,
        collector_wait_time: float = 0.0,
        replay_batch_wait_time: float = 0.0,
        learner_replay_stage_time: float = 0.0,
        learner_replay_sample_time: float = 0.0,
        sync_coordination_time: float = 0.0,
        replay_ingress_h2d_submit_time: float = 0.0,
        weight_sync_time: float = 0.0,
        inference_h2d_time: float = 0.0,
        inference_forward_time: float = 0.0,
        inference_d2h_time: float = 0.0,
        inference_time: float = 0.0,
        iteration_time: float | None = None,
        extra_info: dict | None = None,
    ):
        if metrics:
            metrics = normalize_metric_map(metrics)
            if "Train/iteration" in metrics:
                raise ValueError(
                    "OffPolicyLogger owns Train/iteration; do not emit it from learner metrics"
                )
            owned = sorted(set(metrics) & LOGGER_OWNED_METRICS)
            if owned:
                names = ", ".join(owned)
                raise ValueError(
                    f"logger-owned metrics must use dedicated log_step inputs: {names}"
                )
        if reward_components:
            for component in reward_components:
                reward_term_key(component)
        self._iteration = iteration
        self._train_time = train_time
        self._collector_wait_time = collector_wait_time
        self._replay_batch_wait_time = replay_batch_wait_time
        self._learner_replay_stage_time = learner_replay_stage_time
        self._learner_replay_sample_time = learner_replay_sample_time
        self._sync_coordination_time = sync_coordination_time
        self._replay_ingress_h2d_submit_time = replay_ingress_h2d_submit_time
        self._weight_sync_time = weight_sync_time
        self._inference_h2d_time = inference_h2d_time
        self._inference_forward_time = inference_forward_time
        self._inference_d2h_time = inference_d2h_time
        self._inference_time = inference_time
        self._iteration_time = iteration_time
        self._has_iteration_extra_info = extra_info is not None
        if extra_info:
            self._throughput_env_steps = int(extra_info.get("throughput_steps", 0))
        else:
            self._throughput_env_steps = 0
        if extra_info:
            steps_per_sec = extra_info.get("env_steps_per_sec")
            self._env_steps_per_sec_override = (
                float(steps_per_sec) if steps_per_sec is not None else None
            )
            learner_replay_rows_per_sec = extra_info.get("learner_replay_rows_per_sec")
            self._learner_replay_rows_per_sec_override = (
                float(learner_replay_rows_per_sec)
                if learner_replay_rows_per_sec is not None
                else None
            )
            self._batch_size_per_rank = int(extra_info.get("batch_size_per_rank", 0))
            self._effective_batch_size = int(extra_info.get("effective_batch_size", 0))
            if self._effective_batch_size <= 0:
                self._effective_batch_size = self._batch_size_per_rank
            if self._batch_size_per_rank <= 0 and self._effective_batch_size > 0:
                self._batch_size_per_rank = self._effective_batch_size
            self._learner_replay_rows_per_iter = int(
                extra_info.get("learner_replay_rows_per_iter", 0)
            )
        else:
            self._env_steps_per_sec_override = None
            self._learner_replay_rows_per_sec_override = None
            self._batch_size_per_rank = 0
            self._effective_batch_size = 0
            self._learner_replay_rows_per_iter = 0
        if iteration_time is not None and iteration_time > 0.0:
            if len(self._tail_iteration_times) == self._tail_iteration_times.maxlen:
                self._tail_throughput_env_steps -= self._throughput_env_steps
            self._tail_iteration_times.append(float(iteration_time))
            self._tail_throughput_env_steps += self._throughput_env_steps
        if metrics:
            self._latest_metrics.update(metrics)
        if return_mean_ep100 is not None:
            self._reward_history.append(float(return_mean_ep100))
            self._last_reward_iter = iteration
        if reward_components:
            self._latest_reward_components = reward_components
            self._last_reward_components_iter = iteration
        self._status = "Training"
        scalars = self._build_backend_scalars(
            iteration, metrics, return_mean_ep100, reward_components
        )
        if self._should_log_backend(iteration):
            self._write_backend_scalars(scalars, iteration)
        self._record_terminal_sample(
            metrics=metrics,
            reward=self._reward_history[-1] if self._reward_history else None,
            reward_components=reward_components,
        )

    def _build_backend_scalars(
        self,
        iteration: int,
        metrics: dict[str, float] | None,
        return_mean_ep100: float | None,
        reward_components: dict[str, float] | None,
    ):
        iter_steps_per_sec = self._get_iter_env_steps_per_sec()
        iter_wall_time = self._get_measured_iter_wall_time()
        learner_timing_scalars = {
            key: (seconds if key == "Perf/learning_time" else seconds * 1000)
            for key, seconds in self._learner_phase_times().items()
        }
        learner_timing_scalars.update(
            {key: seconds * 1000 for key, seconds in self._learner_detail_times().items()}
        )
        scalars: dict[str, Any] = {}
        if self._total_steps > 0:
            _set_backend_scalar(scalars, "Train/iteration", float(iteration))
        if metrics:
            for tag, value in metrics.items():
                _set_backend_scalar(scalars, tag, value)
        if return_mean_ep100 is not None:
            _set_backend_scalar(scalars, "Train/mean_reward", float(return_mean_ep100))
        if reward_components:
            for key, value in reward_components.items():
                _set_backend_scalar(scalars, reward_term_key(key), float(value))
        if self._mean_ep_length > 0:
            _set_backend_scalar(scalars, "Train/mean_episode_length", self._mean_ep_length)
        if self._timeout_rate is not None:
            _set_backend_scalar(scalars, "Episode/timeout_rate", self._timeout_rate)
        for key, value in self._curriculum_metrics.items():
            _set_backend_scalar(scalars, key, value)
        for key, value in self._collector_timing.items():
            if key == "rollout_ms":
                _set_backend_scalar(scalars, "Perf/collection_time", value / 1000.0)
            else:
                _set_backend_scalar(scalars, f"Perf/collector_{key}", value)
        for key, inference_tag in _COLLECTOR_INFERENCE_TAG_SPECS.items():
            if inference_tag is None:
                continue
            if key in self._collector_inference:
                _set_backend_scalar(scalars, inference_tag, self._collector_inference[key])
        if iter_steps_per_sec is not None:
            _set_backend_scalar(scalars, "Perf/total_fps", iter_steps_per_sec)
        for key, value in learner_timing_scalars.items():
            _set_backend_scalar(scalars, key, value)
        if iter_wall_time is not None:
            _set_backend_scalar(scalars, "Perf/iteration_time", iter_wall_time)
        validate_metric_tags(scalars)
        return scalars

    def _write_backend_scalars(self, scalars: dict[str, Any], iteration: int) -> None:
        global_step = self._total_steps if self._total_steps > 0 else iteration
        if self._tb_writer:
            self._write_tb_scalars(list(scalars.items()), global_step)

        if self._wandb_run:
            wandb = _load_wandb()
            if wandb is None:
                return
            wandb.log(scalars, step=global_step)

    def log_status(self, status: str):
        self._status = status
        if "[red]" in status or "ERROR" in status:
            self._refresh(force=True)

    def _build_display(self) -> Panel:
        snapshot = self._terminal_snapshot
        header = self._build_compact_header(
            include_status=self._status != "Training",
            include_identity=False,
            include_iteration=False,
            terminal_snapshot=snapshot,
        )
        left = self._build_metrics_table(snapshot)
        right = self._build_reward_table(snapshot)
        bottom = self._build_timing_table(snapshot)
        if self._console.width >= 96:
            grid = Table.grid(expand=True)
            grid.add_column(ratio=1)
            grid.add_column(width=2)
            grid.add_column(ratio=1)
            grid.add_row(left, "", right)
        else:
            grid = Table.grid(expand=True)
            grid.add_column()
            grid.add_row(left)
            grid.add_row(Text(""))
            grid.add_row(right)
        title = Text()
        if self._unicode_console:
            title.append(" 🚀")
        title.append(" UniLab Off-Policy Training ", style="bold")
        title.append("|", style="dim")
        title.append(f" {self.algo_name} ", style="bold cyan")
        title.append("|", style="dim")
        title.append(f" {self.env_name} ", style="bold white")
        title.append("|", style="dim")
        title.append(f" GPUs {self.num_gpus} ", style="bold magenta")
        title.append("|", style="dim")
        iteration = snapshot.iteration if snapshot is not None else self._iteration
        title.append(f" iter {iteration}/{self.max_iterations} ", style="yellow")
        return Panel(
            Group(header, Text(""), grid, Text(""), bottom),
            title=title,
            border_style="bright_blue",
            padding=(0, 1),
        )

    def _build_metrics_table(self, snapshot: _TerminalSnapshot | None = None) -> Table:
        snapshot = snapshot or self._terminal_snapshot
        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            show_edge=False,
            header_style="bold cyan",
            expand=True,
            pad_edge=False,
        )
        compact = self._console.width < 96
        table.add_column(
            "Metrics" if compact else "Losses & Policy",
            style="white",
            no_wrap=compact,
        )
        table.add_column("Value", style="yellow", justify="right", width=13, no_wrap=True)
        metrics = snapshot.metrics if snapshot is not None else self._latest_metrics
        if not metrics:
            table.add_row("[dim]Waiting for data...[/]", "")
        else:
            loss_keys = sorted(
                key
                for key in metrics
                if key.startswith(("Loss/", "Policy/")) or "loss" in key.lower()
            )
            training_keys = sorted(
                key
                for key in metrics
                if key.startswith("Train/")
                and not key.startswith(_REPLAY_INGRESS_PREFIX)
                and key not in loss_keys
            )
            replay_keys = sorted(key for key in metrics if key.startswith(_REPLAY_INGRESS_PREFIX))
            known_keys = set(loss_keys) | set(training_keys) | set(replay_keys)
            other_keys = sorted(key for key in metrics if key not in known_keys)

            def _display_label(key: str, *, indent: str = "") -> str:
                label = _METRIC_DISPLAY_LABELS.get(key)
                if label is None:
                    label = key.split("/", 1)[-1].replace("_", " ").title()
                return f"{indent}{label}"

            for key in loss_keys:
                value = metrics[key]
                style = "red" if value > 10 else "yellow"
                table.add_row(_display_label(key), f"[{style}]{_fmt_number(value)}[/]")
            for key in training_keys:
                table.add_row(_display_label(key, indent="  "), _fmt_number(metrics[key]))
            for key in replay_keys:
                table.add_row(_display_label(key, indent="  "), _fmt_number(metrics[key]))
            for key in other_keys:
                table.add_row(_display_label(key, indent="  "), _fmt_number(metrics[key]))
        return table

    def _build_reward_table(self, snapshot: _TerminalSnapshot | None = None) -> Table:
        snapshot = snapshot or self._terminal_snapshot
        reward_components = None
        if snapshot is not None and snapshot.reward_components:
            reward_components = snapshot.reward_components
        source_iterations = [
            iteration
            for iteration in (self._last_reward_iter, self._last_reward_components_iter)
            if iteration is not None
        ]
        return self._build_reward_table_common(
            wait_message="[dim]Waiting for data...[/]",
            include_ep_length=False,
            reward_history=(snapshot.reward_history if snapshot is not None else None),
            reward_components=reward_components,
            mean_reward=(snapshot.scalars.get("reward") if snapshot is not None else None),
            compact=self._console.width < 96,
            source_iteration=max(source_iterations) if source_iterations else None,
        )

    def _cuda_mps_status_label(self) -> str | None:
        """Render rank-local CUDA MPS state from runtime-manifest evidence."""
        evidence = self._runtime_manifest.get("cuda_process_sharing")
        if not isinstance(evidence, dict):
            return None
        if evidence.get("effective") == "mps":
            return "[green]on[/]"
        if evidence.get("ambient_server"):
            return "[yellow]ambient[/]"
        return "[dim]off[/]"

    def _build_timing_table(self, snapshot: _TerminalSnapshot | None = None) -> Table:
        snapshot = snapshot or self._terminal_snapshot
        terminal_width = self._console.width
        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            show_edge=False,
            header_style="bold blue",
            expand=True,
            pad_edge=False,
        )
        if terminal_width >= 120:
            table.add_column("Learner (Iter Wall)", style="white", no_wrap=True)
            table.add_column("Value", style="yellow", justify="right", width=16, no_wrap=True)
            table.add_column("Collector (own clock)", style="white", no_wrap=True)
            table.add_column("Value", style="yellow", justify="right", width=16, no_wrap=True)
            table.add_column("System", style="white", no_wrap=True)
            table.add_column("Value", style="yellow", justify="right", width=12, no_wrap=True)
        elif terminal_width >= 96:
            table.add_column("Learner", style="white", no_wrap=True)
            table.add_column("Value", style="yellow", justify="right", width=16, no_wrap=True)
            table.add_column("Collector", style="white", no_wrap=True)
            table.add_column("Value", style="yellow", justify="right", width=16, no_wrap=True)
        else:
            table.add_column("Timing", style="white", no_wrap=True)
            table.add_column("Value", style="yellow", justify="right", width=16, no_wrap=True)

        iter_wall_time = (
            snapshot.scalars.get("iter_wall_time", self._get_iter_wall_time())
            if snapshot is not None
            else self._get_iter_wall_time()
        )

        def _scalar(attribute: str, fallback: float) -> float:
            if snapshot is None:
                return fallback
            return snapshot.scalars.get(attribute, fallback)

        def _fmt_phase(seconds: float, *, color: str | None = None) -> str:
            ms = seconds * 1000
            pct = seconds / iter_wall_time * 100.0 if iter_wall_time > 0.0 else 0.0
            text = f"{ms:>7.1f}ms  {pct:>3.0f}%"
            return f"[{color}]{text}[/]" if color else text

        collector_wait_ms = _scalar("_collector_wait_time", self._collector_wait_time) * 1000
        wait_color = "red" if collector_wait_ms > 1.0 else "yellow"
        phase_colors = {
            "Collector Wait": wait_color,
            "Learning": "green",
        }
        learner_items = [
            (
                label,
                _fmt_phase(
                    _scalar(attribute, float(getattr(self, attribute))),
                    color=phase_colors.get(label),
                ),
            )
            for _, label, attribute in self._learner_timing_specs()
        ]
        learner_items.append(
            (
                _LEARNER_OTHER_TIMING_LABEL,
                _fmt_phase(_scalar("learner_other_time", self._get_learner_other_time())),
            )
        )
        learner_items.append(
            (
                _ITER_WALL_TIMING_SPEC[1],
                f"{iter_wall_time * 1000:>7.1f}ms  100%",
            )
        )
        collector_timing = (
            snapshot.collector_timing if snapshot is not None else self._collector_timing
        )
        sorted_collector_timing = sorted(
            collector_timing.items(),
            key=lambda item: (
                _COLLECTOR_TIMING_SPECS.get(item[0], (float("inf"), "", ""))[0],
                item[0],
            ),
        )
        env_step_detail_keys = [
            key for key, _ in sorted_collector_timing if key in OFFPOLICY_ENV_STEP_DETAIL_KEYS
        ]
        last_env_step_detail_key = env_step_detail_keys[-1] if env_step_detail_keys else None
        cycle_total_ms = self._get_collector_cycle_ms(collector_timing) or 0.0

        def _show_collector_row(key: str) -> bool:
            if key in _COLLECTOR_CORE_TERMINAL_KEYS or cycle_total_ms <= 0.0:
                return True
            percentage = collector_timing[key] / cycle_total_ms * 100.0
            return percentage >= _COLLECTOR_DIAGNOSTIC_TERMINAL_MIN_PCT

        collector_items: list[tuple[str, str]] = []
        for key, value in sorted_collector_timing:
            if not _show_collector_row(key):
                continue
            _, label, role = _COLLECTOR_TIMING_SPECS.get(
                key,
                (float("inf"), key, "diagnostic"),
            )
            value_text = f"{value:.1f}ms"
            if key in OFFPOLICY_ENV_STEP_DETAIL_KEYS:
                if self._unicode_console:
                    connector = "─┘" if key == last_env_step_detail_key else "─┤"
                else:
                    connector = "-'" if key == last_env_step_detail_key else "-+"
                label = f"[dim]{label}[/]"
                if self._timing_profile == "sac_family" and cycle_total_ms > 0.0:
                    pct = value / cycle_total_ms * 100.0
                    value_text = f"[dim cyan]{value:>7.1f}ms {pct:>3.0f}%{connector}[/]"
                else:
                    value_text = f"[dim cyan]{value:>7.1f}ms {connector}[/]"
            elif (
                self._timing_profile == "sac_family"
                and role == "cycle_phase"
                and cycle_total_ms > 0.0
            ):
                pct = value / cycle_total_ms * 100.0
                value_text = f"{value:>7.1f}ms  {pct:>3.0f}%"
            collector_items.append((label, value_text))
        buffer_size = snapshot.buffer_size if snapshot is not None else self._buffer_size
        timeout_rate = _scalar("timeout_rate", 0.0)
        batch_size_per_rank = (
            snapshot.batch_size_per_rank if snapshot is not None else self._batch_size_per_rank
        )
        system_items = [
            ("Buffer", f"{buffer_size:,}"),
            ("Timeout Rate", f"{timeout_rate * 100:.1f}%"),
        ]
        system_items.append(("Envs", f"{self.num_envs:,}"))
        if batch_size_per_rank > 0:
            system_items.append(("Batch/Rank", f"{batch_size_per_rank:,}"))
        cuda_mps_status = self._cuda_mps_status_label()
        if cuda_mps_status is not None:
            system_items.append(("CUDA MPS", cuda_mps_status))
        column_items: tuple[list[tuple[str, str]], ...]
        if terminal_width >= 120:
            column_items = (learner_items, collector_items, system_items)
        elif terminal_width >= 96:
            column_items = (learner_items, collector_items)
        else:
            column_items = (learner_items + collector_items,)
        row_count = max(len(items) for items in column_items)
        for index in range(row_count):
            row: list[str] = []
            for items in column_items:
                if index < len(items):
                    row.extend(items[index])
                else:
                    row.extend(["", ""])
            table.add_row(*row)
        return table
