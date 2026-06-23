"""Rich-based training logger for off-policy RL algorithms (SAC, TD3, etc)."""

from __future__ import annotations

from typing import Any

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from unilab.logging.common import BaseTrainingLogger, _fmt_number, _load_wandb

OFFPOLICY_COLLECTOR_TIMING_ORDER = {
    "weight_sync_ms": 0,
    "action_select_ms": 1,
    "env_step_ms": 2,
    "replay_ms": 3,
    "sync_coordination_ms": 4,
}

OFFPOLICY_COLLECTOR_TIMING_LABELS = {
    "weight_sync_ms": "Weight Sync",
    "action_select_ms": "Action Select",
    "env_step_ms": "Env Step",
    "replay_ms": "Replay",
    "sync_coordination_ms": "Sync Coordination",
}


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
        refresh_per_second: int = 4,
        log_dir: str = "",
        log_backend: str = "tensorboard",
        wandb_project: str = "unilab",
        wandb_entity: str | None = None,
        wandb_name: str = "",
        wandb_group: str | None = None,
        wandb_job_type: str | None = None,
        wandb_tags: list[str] | None = None,
        wandb_notes: str | None = None,
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
            tensorboard_subdir=None,
            wandb_config={
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "max_iterations": max_iterations,
            },
        )
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._total_steps: int = 0
        self._buffer_size: int = 0
        self._buffer_target: int = 0
        self._wait_time: float = 0.0
        self._learner_incremental_h2d_time: float = 0.0
        self._learner_param_sync_time: float = 0.0
        self._weight_sync_time: float = 0.0
        self._iteration_time: float | None = None
        self._throughput_steps: int = 0
        self._world_size: int = 1
        self._multi_gpu_sync_mode: str = ""
        self._multi_gpu_sync_interval: int = 0
        self._batch_size_per_rank: int = 0
        self._effective_batch_size: int = 0
        self._replay_samples_per_iter: int = 0
        self._learner_samples_per_iter: int = 0
        self._has_iteration_extra_info: bool = False
        self._collector_timing: dict[str, float] = {}
        self._timeout_rate: float = 0.0
        self._terminated_rate: float = 0.0
        self._buffer_utilization: float = 0.0
        self._sync_collection: bool = False
        self._env_steps_per_sync: int = 0
        self._staging_pool_len: int = 0
        self._staging_pool_max: int = 0
        self._status: str = "Initializing..."
        self._terminal_refresh_started: bool = False

    def _format_tensorboard_message(self, tb_dir: str) -> str:
        return f"[dim]TensorBoard logging to: {tb_dir}[/]"

    def _format_wandb_message(self, project: str, name: str) -> str:
        return f"[dim]W&B logging to project: {project}, run: {name}[/]"

    def start(self, *, status: str = "Warming up..."):
        super().start(status=status)

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
        if not self._terminal_refresh_started:
            self._refresh()

    def _get_iter_steps_per_sec(self) -> float | None:
        if not self._has_iteration_extra_info or self._throughput_steps <= 0:
            return None
        iter_time = self._get_iter_wall_time()
        if iter_time <= 0:
            return None
        return self._throughput_steps / iter_time

    def _get_effective_samples_per_sec(self) -> float | None:
        if not self._has_iteration_extra_info or self._learner_samples_per_iter <= 0:
            return None
        iter_time = self._get_iter_wall_time()
        if iter_time <= 0:
            return None
        return self._learner_samples_per_iter / iter_time

    def _get_learner_pipeline_time(self) -> float:
        return self._learner_incremental_h2d_time + self._train_time + self._weight_sync_time

    def _get_iter_wall_time(self) -> float:
        if self._iteration_time is not None and self._iteration_time > 0.0:
            return self._iteration_time
        return self._wait_time + self._get_learner_pipeline_time()

    def _build_compact_header(
        self,
        *,
        include_status: bool,
        extra_fields: list[tuple[str, str]] | None = None,
    ) -> Text:
        iter_steps_per_sec = self._get_iter_steps_per_sec()
        effective_samples_per_sec = self._get_effective_samples_per_sec()
        header_extra_fields: list[tuple[str, str]] = []
        if iter_steps_per_sec is not None:
            header_extra_fields.append((f"Steps/s {iter_steps_per_sec:,.0f}", "bold green"))
        if effective_samples_per_sec is not None:
            header_extra_fields.append((f"Samples/s {effective_samples_per_sec:,.0f}", "bold cyan"))
        if extra_fields:
            header_extra_fields.extend(extra_fields)
        return super()._build_compact_header(
            include_status=include_status,
            extra_fields=header_extra_fields,
        )

    def update_collector_timing(self, timing_ms: dict[str, float]):
        self._collector_timing.update(timing_ms)

    def update_done_rates(self, timeout_rate: float, terminated_rate: float):
        self._timeout_rate = float(timeout_rate)
        self._terminated_rate = float(terminated_rate)

    def update_buffer_utilization(self, utilization: float):
        self._buffer_utilization = float(utilization)

    def update_replay_queue(self, current_len: int, max_size: int):
        self.update_staging_pool(current_len, max_size)

    def update_staging_pool(self, current_len: int, max_size: int):
        self._staging_pool_len = current_len
        self._staging_pool_max = max_size

    def set_collection_sync(self, enabled: bool, env_steps_per_sync: int = 0):
        self._sync_collection = enabled
        self._env_steps_per_sync = env_steps_per_sync

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
        wait_time: float = 0.0,
        learner_incremental_h2d_time: float = 0.0,
        learner_param_sync_time: float = 0.0,
        weight_sync_time: float = 0.0,
        iteration_time: float | None = None,
        extra_info: dict | None = None,
    ):
        metrics = _dedupe_metric_aliases(metrics)
        self._iteration = iteration
        self._train_time = train_time
        self._wait_time = wait_time
        self._learner_incremental_h2d_time = learner_incremental_h2d_time
        self._learner_param_sync_time = learner_param_sync_time
        self._weight_sync_time = weight_sync_time
        self._iteration_time = iteration_time
        self._has_iteration_extra_info = extra_info is not None
        if extra_info:
            self._throughput_steps = int(extra_info.get("throughput_steps", 0))
            self._world_size = int(extra_info.get("world_size", 1))
            self._multi_gpu_sync_mode = str(extra_info.get("multi_gpu_sync_mode", ""))
            self._multi_gpu_sync_interval = int(extra_info.get("multi_gpu_sync_interval", 0))
            self._batch_size_per_rank = int(extra_info.get("batch_size_per_rank", 0))
            self._effective_batch_size = int(extra_info.get("effective_batch_size", 0))
            if self._effective_batch_size <= 0:
                self._effective_batch_size = self._batch_size_per_rank * self._world_size
            if self._batch_size_per_rank <= 0 and self._effective_batch_size > 0:
                self._batch_size_per_rank = self._effective_batch_size // max(self._world_size, 1)
            self._replay_samples_per_iter = int(extra_info.get("replay_samples_per_iter", 0))
            self._learner_samples_per_iter = int(extra_info.get("learner_samples_per_iter", 0))
            if self._replay_samples_per_iter <= 0:
                self._replay_samples_per_iter = self._learner_samples_per_iter
        else:
            self._throughput_steps = 0
            self._world_size = 1
            self._multi_gpu_sync_mode = ""
            self._multi_gpu_sync_interval = 0
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
        self._terminal_refresh_started = True
        self._refresh()
        self._backend_log_step(
            iteration,
            metrics,
            reward,
            reward_metrics,
            reward_components,
            train_time,
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
            writer.add_scalar("episode/terminated_rate", self._terminated_rate, global_step)
            writer.add_scalar("timing/learner_wait_ms", self._wait_time * 1000, global_step)
            writer.add_scalar(
                "timing/learner_incremental_h2d_ms",
                self._learner_incremental_h2d_time * 1000,
                global_step,
            )
            writer.add_scalar("timing/learner_train_ms", train_time * 1000, global_step)
            if self._world_size > 1 or self._learner_param_sync_time > 0.0:
                writer.add_scalar(
                    "timing/learner_param_sync_ms",
                    self._learner_param_sync_time * 1000,
                    global_step,
                )
            writer.add_scalar(
                "timing/learner_weight_sync_ms",
                self._weight_sync_time * 1000,
                global_step,
            )
            for key, value in self._collector_timing.items():
                writer.add_scalar(f"timing/collector_{key}", value, global_step)
            if iter_steps_per_sec is not None:
                writer.add_scalar("perf/steps_per_sec", iter_steps_per_sec, global_step)
            if effective_samples_per_sec is not None:
                writer.add_scalar(
                    "perf/effective_samples_per_sec",
                    effective_samples_per_sec,
                    global_step,
                )
            writer.add_scalar("perf/iter_ms", iter_wall_time * 1000, global_step)
            writer.add_scalar(
                "perf/learner_pipeline_ms",
                self._get_learner_pipeline_time() * 1000,
                global_step,
            )

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
            log_dict["episode/terminated_rate"] = self._terminated_rate
            log_dict["timing/learner_wait_ms"] = self._wait_time * 1000
            log_dict["timing/learner_incremental_h2d_ms"] = (
                self._learner_incremental_h2d_time * 1000
            )
            log_dict["timing/learner_train_ms"] = train_time * 1000
            if self._world_size > 1 or self._learner_param_sync_time > 0.0:
                log_dict["timing/learner_param_sync_ms"] = self._learner_param_sync_time * 1000
            log_dict["timing/learner_weight_sync_ms"] = self._weight_sync_time * 1000
            for key, value in self._collector_timing.items():
                log_dict[f"timing/collector_{key}"] = value
            if iter_steps_per_sec is not None:
                log_dict["perf/steps_per_sec"] = iter_steps_per_sec
            if effective_samples_per_sec is not None:
                log_dict["perf/effective_samples_per_sec"] = effective_samples_per_sec
            log_dict["perf/iter_ms"] = iter_wall_time * 1000
            log_dict["perf/learner_pipeline_ms"] = self._get_learner_pipeline_time() * 1000
            wandb.log(log_dict, step=global_step)

    def log_status(self, status: str):
        self._status = status
        if not self._terminal_refresh_started or "[red]" in status or "ERROR" in status:
            self._refresh(force=True)

    def _build_display(self) -> Panel:
        header = self._build_compact_header(include_status=True)
        left = self._build_metrics_table()
        right = self._build_reward_table()
        bottom = self._build_timing_table()
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(width=2)
        grid.add_column(ratio=1)
        grid.add_row(left, "", right)
        return Panel(
            Group(header, Text(""), grid, Text(""), bottom),
            title=(
                "[bold] 🚀 UniLab Off-Policy Training [/]"
                if self._unicode_console
                else "[bold] UniLab Off-Policy Training [/]"
            ),
            border_style="bright_blue",
            padding=(0, 1),
        )

    def _build_metrics_table(self) -> Table:
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
        if not self._latest_metrics:
            table.add_row("[dim]Waiting for data...[/]", "")
        else:
            loss_keys = sorted([key for key in self._latest_metrics if "loss" in key.lower()])
            other_keys = sorted([key for key in self._latest_metrics if "loss" not in key.lower()])
            for key in loss_keys:
                value = self._latest_metrics[key]
                style = "red" if value > 10 else "yellow"
                table.add_row(key.replace("_", " ").title(), f"[{style}]{_fmt_number(value)}[/]")
            for key in other_keys:
                value = self._latest_metrics[key]
                table.add_row(f"  {key.replace('_', ' ').title()}", _fmt_number(value))
        return table

    def _build_reward_table(self) -> Table:
        return self._build_reward_table_common(
            wait_message="[dim]Waiting for data...[/]",
            include_ep_length=False,
        )

    def _build_timing_table(self) -> Table:
        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            show_edge=False,
            header_style="bold blue",
            expand=True,
            pad_edge=False,
        )
        table.add_column("Learner", style="white", ratio=2, no_wrap=True)
        table.add_column("Value", style="yellow", justify="right", ratio=1, no_wrap=True)
        table.add_column("Collector", style="white", ratio=2, no_wrap=True)
        table.add_column("Value", style="yellow", justify="right", ratio=1, no_wrap=True)
        table.add_column("System", style="white", ratio=2, no_wrap=True)
        table.add_column("Value", style="yellow", justify="right", ratio=1, no_wrap=True)

        wait_ms = self._wait_time * 1000
        wait_color = "red" if wait_ms > 1.0 else "yellow"
        learner_items = [
            ("Wait", f"[{wait_color}]{wait_ms:.1f}ms[/]"),
            ("H2D Copy", f"{self._learner_incremental_h2d_time * 1000:.1f}ms"),
            ("Train", f"{self._train_time * 1000:.1f}ms"),
            ("Weight Sync", f"{self._weight_sync_time * 1000:.1f}ms"),
            ("Iter Wall", f"{self._get_iter_wall_time() * 1000:.1f}ms"),
        ]
        if self._world_size > 1 or self._learner_param_sync_time > 0.0:
            learner_items.insert(
                3,
                ("Param Sync (in Train)", f"{self._learner_param_sync_time * 1000:.1f}ms"),
            )
        collector_items = [
            (OFFPOLICY_COLLECTOR_TIMING_LABELS.get(key, key), f"{value:.1f}ms")
            for key, value in sorted(
                self._collector_timing.items(),
                key=lambda item: (
                    OFFPOLICY_COLLECTOR_TIMING_ORDER.get(
                        item[0], len(OFFPOLICY_COLLECTOR_TIMING_ORDER)
                    ),
                    item[0],
                ),
            )
        ]
        system_items = [
            ("Buffer", f"{self._buffer_size:,}"),
        ]
        system_items.extend(
            [
                ("Timeout Rate", f"{self._timeout_rate * 100:.1f}%"),
                ("Terminated Rate", f"{self._terminated_rate * 100:.1f}%"),
            ]
        )
        system_items.append(("Envs", f"{self.num_envs:,}"))
        if self._world_size > 1:
            system_items.append(("GPUs", f"{self._world_size:,}"))
            if self._multi_gpu_sync_mode:
                system_items.append(("Sync Mode", self._multi_gpu_sync_mode))
            if self._multi_gpu_sync_interval > 0:
                system_items.append(("Sync Interval", f"{self._multi_gpu_sync_interval:,}"))
        if self._batch_size_per_rank > 0:
            system_items.append(("Batch/Rank", f"{self._batch_size_per_rank:,}"))
        if (
            self._effective_batch_size > 0
            and self._effective_batch_size != self._batch_size_per_rank
        ):
            system_items.append(("Batch/Update", f"{self._effective_batch_size:,}"))
        if (
            self._replay_samples_per_iter > 0
            and self._replay_samples_per_iter != self._learner_samples_per_iter
        ):
            system_items.append(("Replay/Iter", f"{self._replay_samples_per_iter:,}"))
        if self._learner_samples_per_iter > 0:
            system_items.append(("Samples/Iter", f"{self._learner_samples_per_iter:,}"))
        yes_mark = "✓" if self._unicode_console else "yes"
        no_mark = "✗" if self._unicode_console else "no"
        sync_collect = (
            f"{yes_mark} ({self._env_steps_per_sync})" if self._sync_collection else no_mark
        )
        system_items.append(("Sync Collect", sync_collect))
        if self._staging_pool_max > 0:
            staging_color = "green" if self._staging_pool_len < self._staging_pool_max else "yellow"
            system_items.append(
                (
                    "Staging Pool",
                    f"[{staging_color}]{self._staging_pool_len}/{self._staging_pool_max}[/]",
                )
            )
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
