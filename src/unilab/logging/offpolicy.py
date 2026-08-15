"""Rich-based training logger for off-policy RL algorithms (SAC, TD3, etc)."""

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

from unilab.logging.common import BaseTrainingLogger, _fmt_number, _load_wandb

_COLLECTOR_WAIT_TIMING_SPEC = (
    "timing/learner_collector_wait_ms",
    "Collector Wait",
    "_collector_wait_time",
)
_INFERENCE_TIMING_SPEC = ("timing/learner_inference_ms", "Inference", "_inference_time")
_COLLECTOR_RELEASE_TIMING_SPEC = (
    "timing/learner_collector_release_ms",
    "Collector Release",
    "_sync_coordination_time",
)
_REPLAY_BATCH_WAIT_TIMING_SPEC = (
    "timing/learner_replay_batch_wait_ms",
    "Replay Batch Wait",
    "_replay_batch_wait_time",
)
_REPLAY_STAGE_TIMING_SPEC = (
    "timing/learner_replay_stage_ms",
    "Replay Stage",
    "_learner_replay_stage_time",
)
_REPLAY_SAMPLE_TIMING_SPEC = (
    "timing/learner_replay_sample_ms",
    "Replay Sample",
    "_learner_replay_sample_time",
)
_TRAIN_TIMING_SPEC = ("timing/learner_train_ms", "Train", "_train_time")
_WEIGHT_PUBLISH_TIMING_SPEC = (
    "timing/learner_weight_publish_ms",
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
    ("timing/learner_inference_h2d_ms", "Inference H2D", "_inference_h2d_time"),
    (
        "timing/learner_inference_forward_ms",
        "Inference Forward",
        "_inference_forward_time",
    ),
    ("timing/learner_inference_d2h_ms", "Inference D2H", "_inference_d2h_time"),
    (
        "timing/replay_ingress_h2d_submit_ms",
        "Replay H2D Submit",
        "_replay_ingress_h2d_submit_time",
    ),
)

_LEARNER_DETAIL_TIMING_PROFILES = {
    "sac_family": _SAC_FAMILY_DETAIL_TIMING_SPECS,
    "appo": (),
}

_LEARNER_OTHER_TIMING_SPEC = ("timing/learner_other_ms", "Other", "")
_ITER_WALL_TIMING_SPEC = ("perf/iter_ms", "Iter Wall", "")

_COLLECTOR_TIMING_SPECS = {
    "mlp_infer_ms": (1.0, "MLP Infer", "per_step"),
    "inference_request_ms": (1.0, "Inference Request", "cycle_phase"),
    "learner_action_wait_ms": (1.1, "Learner Action Wait", "cycle_phase"),
    "env_step_ms": (2.0, "Env Step", "cycle_phase"),
    "env_step_backend_ms": (2.1, "  Backend Step", "env_step_detail"),
    "env_step_update_state_ms": (2.2, "  Update State", "env_step_detail"),
    "env_step_reset_done_ms": (2.3, "  Reset Done", "env_step_detail"),
    "replay_write_ms": (3.0, "Replay Write", "cycle_phase"),
    "rollout_ms": (9.0, "Rollout Wall", "rollout_total"),
}

OFFPOLICY_ENV_STEP_DETAIL_KEYS = (
    "env_step_backend_ms",
    "env_step_update_state_ms",
    "env_step_reset_done_ms",
)

_OFFPOLICY_COLLECTOR_CYCLE_KEYS = tuple(
    key for key, (_, _, role) in _COLLECTOR_TIMING_SPECS.items() if role == "cycle_phase"
)

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


def _metric_backend_key(key: str) -> str:
    """Keep canonical slash metrics intact; namespace legacy flat metrics under train/."""
    return key if "/" in key else f"train/{key}"


def _reward_backend_key(key: str) -> str:
    """Keep canonical reward/* keys intact; namespace bare component names under reward/."""
    return key if key.startswith("reward/") else f"reward/{key}"


def _dedupe_metric_aliases(metrics: dict[str, float] | None) -> dict[str, float] | None:
    """Drop legacy flat APPO aliases when canonical metrics are present."""
    if not metrics:
        return metrics
    normalized = dict(metrics)
    aliases = {
        "surrogate_loss": "loss/policy_loss",
        "value_loss": "loss/value_loss",
        "entropy": "policy/entropy",
        "kl": "ppo/approx_kl",
    }
    for legacy_key, canonical_key in aliases.items():
        if canonical_key in normalized:
            normalized.pop(legacy_key, None)
    return normalized


class OffPolicyLogger(BaseTrainingLogger):
    """Rich logger for off-policy RL algorithms (SAC, TD3, etc)."""

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
        self._throughput_steps: int = 0
        self._steps_per_sec_override: float | None = None
        self._samples_per_sec_override: float | None = None
        self._collector_active_steps_per_sec: float | None = None
        self._batch_size_per_rank: int = 0
        self._effective_batch_size: int = 0
        self._replay_samples_per_iter: int = 0
        self._learner_samples_per_iter: int = 0
        self._has_iteration_extra_info: bool = False
        self._collector_timing: dict[str, float] = {}
        if timing_profile not in _LEARNER_TIMING_PROFILES:
            raise ValueError("timing_profile must be 'sac_family' or 'appo'")
        self._timing_profile = timing_profile
        self._timeout_rate: float = 0.0
        self._buffer_utilization: float = 0.0
        self._runtime_manifest: dict[str, Any] = {}
        self._staging_pool_len: int = 0
        self._staging_pool_max: int = 0
        self._status: str = "Initializing..."
        self._training_timer_started: bool = False
        self._terminal_samples: deque[_TerminalSample] = deque(maxlen=_TERMINAL_AVERAGE_MAX_SAMPLES)
        self._terminal_snapshot: _TerminalSnapshot | None = None

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

    def _get_iter_steps_per_sec(self) -> float | None:
        if self._steps_per_sec_override is not None:
            return self._steps_per_sec_override
        if not self._has_iteration_extra_info or self._throughput_steps <= 0:
            return None
        iter_time = self._get_iter_wall_time()
        if iter_time <= 0:
            return None
        return self._throughput_steps / iter_time

    def _get_effective_samples_per_sec(self) -> float | None:
        if self._samples_per_sec_override is not None:
            return self._samples_per_sec_override
        if not self._has_iteration_extra_info or self._learner_samples_per_iter <= 0:
            return None
        iter_time = self._get_iter_wall_time()
        if iter_time <= 0:
            return None
        return self._learner_samples_per_iter / iter_time

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

    def _get_iter_pct(self, seconds: float) -> float:
        iter_time = self._get_iter_wall_time()
        if iter_time <= 0.0:
            return 0.0
        return seconds / iter_time * 100.0

    def _get_iter_wall_time(self) -> float:
        if self._iteration_time is not None and self._iteration_time > 0.0:
            return self._iteration_time
        return self._get_learner_accounted_time()

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
            snapshot.scalars.get("steps_per_sec")
            if snapshot is not None
            else self._get_iter_steps_per_sec()
        )
        effective_samples_per_sec = (
            snapshot.scalars.get("samples_per_sec")
            if snapshot is not None
            else self._get_effective_samples_per_sec()
        )
        header_extra_fields: list[tuple[str, str]] = []
        if iter_steps_per_sec is not None:
            header_extra_fields.append((f"Steps/s {iter_steps_per_sec:,.0f}", "bold green"))
        if effective_samples_per_sec is not None:
            header_extra_fields.append((f"Samples/s {effective_samples_per_sec:,.0f}", "bold cyan"))
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
                "timeout_rate": self._timeout_rate,
            }
        )
        if self._mean_ep_length > 0.0:
            scalars["mean_ep_length"] = self._mean_ep_length
        if reward is not None:
            scalars["reward"] = float(reward)
        steps_per_sec = self._get_iter_steps_per_sec()
        if steps_per_sec is not None:
            scalars["steps_per_sec"] = steps_per_sec
        samples_per_sec = self._get_effective_samples_per_sec()
        if samples_per_sec is not None:
            scalars["samples_per_sec"] = samples_per_sec

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

    def update_collector_timing(self, timing_ms: dict[str, float]):
        normalized = dict(timing_ms)
        legacy_action_wait = normalized.pop("inference_wait_ms", None)
        if "learner_action_wait_ms" not in normalized and legacy_action_wait is not None:
            normalized["learner_action_wait_ms"] = legacy_action_wait
        normalized.pop("sync_idle_ms", None)
        normalized.pop("bookkeeping_ms", None)
        self._collector_timing.update(normalized)

    def update_collector_active_steps_per_sec(self, steps_per_sec: float):
        self._collector_active_steps_per_sec = float(steps_per_sec)

    def update_timeout_rate(self, timeout_rate: float):
        self._timeout_rate = float(timeout_rate)

    def update_buffer_utilization(self, utilization: float):
        self._buffer_utilization = float(utilization)

    def update_replay_queue(self, current_len: int, max_size: int):
        self.update_staging_pool(current_len, max_size)

    def update_staging_pool(self, current_len: int, max_size: int):
        self._staging_pool_len = current_len
        self._staging_pool_max = max_size

    def update_runtime_manifest(self, manifest: dict[str, Any]) -> None:
        self._runtime_manifest.update(manifest)

    def log_collector(self, total_steps: int, buffer_size: int, mean_reward: float = 0.0):
        self._total_steps = total_steps
        self._buffer_size = buffer_size
        if mean_reward != 0:
            self._reward_history.append(mean_reward)

    def log_step(
        self,
        iteration: int,
        metrics: dict[str, float] | None = None,
        reward: float | None = None,
        reward_metrics: dict[str, float] | None = None,
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
        metrics = _dedupe_metric_aliases(metrics)
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
            self._throughput_steps = int(extra_info.get("throughput_steps", 0))
            steps_per_sec = extra_info.get("steps_per_sec")
            self._steps_per_sec_override = (
                float(steps_per_sec) if steps_per_sec is not None else None
            )
            samples_per_sec = extra_info.get("learner_samples_per_sec")
            self._samples_per_sec_override = (
                float(samples_per_sec) if samples_per_sec is not None else None
            )
            collector_active_steps_per_sec = extra_info.get("collector_active_steps_per_sec")
            self._collector_active_steps_per_sec = (
                float(collector_active_steps_per_sec)
                if collector_active_steps_per_sec is not None
                else None
            )
            self._batch_size_per_rank = int(extra_info.get("batch_size_per_rank", 0))
            self._effective_batch_size = int(extra_info.get("effective_batch_size", 0))
            if self._effective_batch_size <= 0:
                self._effective_batch_size = self._batch_size_per_rank
            if self._batch_size_per_rank <= 0 and self._effective_batch_size > 0:
                self._batch_size_per_rank = self._effective_batch_size
            self._replay_samples_per_iter = int(extra_info.get("replay_samples_per_iter", 0))
            self._learner_samples_per_iter = int(extra_info.get("learner_samples_per_iter", 0))
            if self._replay_samples_per_iter <= 0:
                self._replay_samples_per_iter = self._learner_samples_per_iter
        else:
            self._throughput_steps = 0
            self._steps_per_sec_override = None
            self._samples_per_sec_override = None
            self._collector_active_steps_per_sec = None
            self._batch_size_per_rank = 0
            self._effective_batch_size = 0
            self._replay_samples_per_iter = 0
            self._learner_samples_per_iter = 0
        if metrics:
            self._latest_metrics.update(metrics)
        if reward is not None:
            self._reward_history.append(reward)
        if reward_components:
            self._latest_reward_components = reward_components
        self._status = "Training"
        self._backend_log_step(
            iteration,
            metrics,
            reward,
            reward_metrics,
            reward_components,
            train_time,
        )
        self._record_terminal_sample(
            metrics=metrics,
            reward=reward,
            reward_components=reward_components,
        )

    def _backend_log_step(
        self,
        iteration: int,
        metrics: dict[str, float] | None,
        reward: float | None,
        reward_metrics: dict[str, float] | None,
        reward_components: dict[str, float] | None,
        train_time: float,
    ):
        global_step = self._total_steps if self._total_steps > 0 else iteration
        iter_steps_per_sec = self._get_iter_steps_per_sec()
        effective_samples_per_sec = self._get_effective_samples_per_sec()
        iter_wall_time = self._get_iter_wall_time()
        learner_other_time = self._get_learner_other_time()
        learner_accounted_time = self._get_learner_accounted_time()
        learner_timing_ms = {
            key: seconds * 1000 for key, seconds in self._learner_phase_times().items()
        }
        learner_timing_ms.update(
            {key: seconds * 1000 for key, seconds in self._learner_detail_times().items()}
        )
        learner_timing_ms[_LEARNER_OTHER_TIMING_SPEC[0]] = learner_other_time * 1000
        collector_cycle_ms = self._get_collector_cycle_ms()

        if self._tb_writer:
            writer = self._tb_writer
            if metrics:
                for key, value in metrics.items():
                    writer.add_scalar(_metric_backend_key(key), value, global_step)
            if reward is not None:
                writer.add_scalar("reward/mean", reward, global_step)
            if reward_metrics:
                for key, value in reward_metrics.items():
                    writer.add_scalar(_reward_backend_key(key), value, global_step)
            if reward_components:
                for key, value in reward_components.items():
                    writer.add_scalar(_reward_backend_key(key), value, global_step)
            if self._mean_ep_length > 0:
                writer.add_scalar("episode/length", self._mean_ep_length, global_step)
            writer.add_scalar("episode/timeout_rate", self._timeout_rate, global_step)
            for key, value_ms in learner_timing_ms.items():
                writer.add_scalar(key, value_ms, global_step)
            for key, value in self._collector_timing.items():
                writer.add_scalar(f"timing/collector_{key}", value, global_step)
            if iter_steps_per_sec is not None:
                writer.add_scalar("perf/steps_per_sec", iter_steps_per_sec, global_step)
            if self._collector_active_steps_per_sec is not None:
                writer.add_scalar(
                    "perf/collector_active_steps_per_sec",
                    self._collector_active_steps_per_sec,
                    global_step,
                )
            if effective_samples_per_sec is not None:
                writer.add_scalar(
                    "perf/effective_samples_per_sec",
                    effective_samples_per_sec,
                    global_step,
                )
            writer.add_scalar("perf/iter_ms", iter_wall_time * 1000, global_step)
            writer.add_scalar("perf/learner_train_pct", self._get_iter_pct(train_time), global_step)
            writer.add_scalar(
                "perf/learner_accounted_pct",
                self._get_iter_pct(learner_accounted_time),
                global_step,
            )
            writer.add_scalar(
                "perf/learner_other_pct",
                self._get_iter_pct(learner_other_time),
                global_step,
            )
            if collector_cycle_ms is not None:
                writer.add_scalar("perf/collector_cycle_ms", collector_cycle_ms, global_step)

        if self._wandb_run:
            wandb = _load_wandb()
            if wandb is None:
                return
            log_dict: dict[str, Any] = {"iteration": iteration}
            if metrics:
                for key, value in metrics.items():
                    log_dict[_metric_backend_key(key)] = value
            if reward is not None:
                log_dict["reward/mean"] = reward
            if reward_metrics:
                for key, value in reward_metrics.items():
                    log_dict[_reward_backend_key(key)] = value
            if reward_components:
                for key, value in reward_components.items():
                    log_dict[_reward_backend_key(key)] = value
            if self._mean_ep_length > 0:
                log_dict["episode/length"] = self._mean_ep_length
            log_dict["episode/timeout_rate"] = self._timeout_rate
            log_dict.update(learner_timing_ms)
            for key, value in self._collector_timing.items():
                log_dict[f"timing/collector_{key}"] = value
            if iter_steps_per_sec is not None:
                log_dict["perf/steps_per_sec"] = iter_steps_per_sec
            if self._collector_active_steps_per_sec is not None:
                log_dict["perf/collector_active_steps_per_sec"] = (
                    self._collector_active_steps_per_sec
                )
            if effective_samples_per_sec is not None:
                log_dict["perf/effective_samples_per_sec"] = effective_samples_per_sec
            log_dict["perf/iter_ms"] = iter_wall_time * 1000
            log_dict["perf/learner_train_pct"] = self._get_iter_pct(train_time)
            log_dict["perf/learner_accounted_pct"] = self._get_iter_pct(learner_accounted_time)
            log_dict["perf/learner_other_pct"] = self._get_iter_pct(learner_other_time)
            if collector_cycle_ms is not None:
                log_dict["perf/collector_cycle_ms"] = collector_cycle_ms
            wandb.log(log_dict, step=global_step)

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
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(width=2)
        grid.add_column(ratio=1)
        grid.add_row(left, "", right)
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
        table.add_column("Losses & Metrics", style="white", ratio=2)
        table.add_column("Value", style="yellow", justify="right", ratio=1)
        metrics = snapshot.metrics if snapshot is not None else self._latest_metrics
        if not metrics:
            table.add_row("[dim]Waiting for data...[/]", "")
        else:
            loss_keys = sorted([key for key in metrics if "loss" in key.lower()])
            other_keys = sorted([key for key in metrics if "loss" not in key.lower()])
            for key in loss_keys:
                value = metrics[key]
                style = "red" if value > 10 else "yellow"
                table.add_row(key.replace("_", " ").title(), f"[{style}]{_fmt_number(value)}[/]")
            for key in other_keys:
                value = metrics[key]
                table.add_row(f"  {key.replace('_', ' ').title()}", _fmt_number(value))
        return table

    def _build_reward_table(self, snapshot: _TerminalSnapshot | None = None) -> Table:
        snapshot = snapshot or self._terminal_snapshot
        return self._build_reward_table_common(
            wait_message="[dim]Waiting for data...[/]",
            include_ep_length=False,
            reward_history=(snapshot.reward_history if snapshot is not None else None),
            reward_components=(snapshot.reward_components if snapshot is not None else None),
            mean_reward=(snapshot.scalars.get("reward") if snapshot is not None else None),
        )

    def _build_timing_table(self, snapshot: _TerminalSnapshot | None = None) -> Table:
        snapshot = snapshot or self._terminal_snapshot
        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            show_edge=False,
            header_style="bold blue",
            expand=True,
            pad_edge=False,
        )
        table.add_column("Learner (Iter Wall)", style="white", ratio=5, no_wrap=True)
        table.add_column("Value", style="yellow", justify="right", width=16, no_wrap=True)
        table.add_column("Collector (own clock)", style="white", ratio=6, no_wrap=True)
        table.add_column("Value", style="yellow", justify="right", width=16, no_wrap=True)
        table.add_column("System", style="white", ratio=4, no_wrap=True)
        table.add_column("Value", style="yellow", justify="right", width=12, no_wrap=True)

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
            "Train": "green",
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
                _LEARNER_OTHER_TIMING_SPEC[1],
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
        collector_items: list[tuple[str, str]] = []
        for key, value in sorted_collector_timing:
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
        timeout_rate = _scalar("timeout_rate", self._timeout_rate)
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
        row_count = max(len(learner_items), len(collector_items), len(system_items))
        for index in range(row_count):
            row: list[str] = []
            for items in (learner_items, collector_items, system_items):
                if index < len(items):
                    row.extend(items[index])
                else:
                    row.extend(["", ""])
            table.add_row(*row)
        return table
