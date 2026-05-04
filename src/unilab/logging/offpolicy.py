"""Rich-based training logger for off-policy RL algorithms (SAC, TD3, etc)."""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table

from unilab.logging.common import BaseTrainingLogger, _fmt_number, _fmt_time, _load_wandb


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
        self._startup_wait_time: float = 0.0
        self._throughput_steps: int = 0
        self._has_iteration_extra_info: bool = False
        self._iter_times: deque = deque(maxlen=50)
        self._collector_timing: dict[str, float] = {}
        self._timeout_rate: float = 0.0
        self._terminated_rate: float = 0.0
        self._buffer_utilization: float = 0.0
        self._sync_collection: bool = False
        self._env_steps_per_sync: int = 0
        self._replay_queue_len: int = 0
        self._replay_queue_max: int = 0
        self._status: str = "Initializing..."

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
        self._refresh()

    def _get_iter_steps_per_sec(self) -> float | None:
        if not self._has_iteration_extra_info or self._throughput_steps <= 0:
            return None
        iter_time = self._collect_time + self._train_time
        if iter_time <= 0:
            return None
        return self._throughput_steps / iter_time

    def _get_wall_steps_per_sec(self, elapsed: float) -> float | None:
        if elapsed <= 0 or self._total_steps <= 0:
            return None
        return self._total_steps / elapsed

    def update_collector_timing(self, timing_ms: dict[str, float]):
        self._collector_timing.update(timing_ms)

    def update_done_rates(self, timeout_rate: float, terminated_rate: float):
        self._timeout_rate = float(timeout_rate)
        self._terminated_rate = float(terminated_rate)

    def update_buffer_utilization(self, utilization: float):
        self._buffer_utilization = float(utilization)

    def update_replay_queue(self, current_len: int, max_size: int):
        self._replay_queue_len = current_len
        self._replay_queue_max = max_size

    def set_collection_sync(self, enabled: bool, env_steps_per_sync: int = 0):
        self._sync_collection = enabled
        self._env_steps_per_sync = env_steps_per_sync

    def log_collector(self, total_steps: int, buffer_size: int, mean_reward: float = 0.0):
        self._total_steps = total_steps
        self._buffer_size = buffer_size
        if mean_reward != 0:
            self._reward_history.append(mean_reward)
        self._refresh()

    def log_step(
        self,
        iteration: int,
        metrics: dict[str, float] | None = None,
        reward: float | None = None,
        reward_components: dict[str, float] | None = None,
        collect_time: float = 0.0,
        train_time: float = 0.0,
        wait_time: float = 0.0,
        extra_info: dict | None = None,
    ):
        self._iteration = iteration
        self._collect_time = collect_time
        self._train_time = train_time
        self._wait_time = wait_time
        self._has_iteration_extra_info = extra_info is not None
        if extra_info:
            self._startup_wait_time = float(extra_info.get("startup_wait_time", 0.0))
            self._throughput_steps = int(extra_info.get("throughput_steps", 0))
        else:
            self._startup_wait_time = 0.0
            self._throughput_steps = 0
        self._iter_times.append(collect_time + train_time)
        if metrics:
            self._latest_metrics.update(metrics)
        if reward is not None:
            self._reward_history.append(reward)
        if reward_components:
            self._latest_reward_components = reward_components
        self._status = "Training"
        self._refresh()
        self._backend_log_step(
            iteration, metrics, reward, reward_components, collect_time, train_time
        )

    def _backend_log_step(
        self,
        iteration: int,
        metrics: dict[str, float] | None,
        reward: float | None,
        reward_components: dict[str, float] | None,
        collect_time: float,
        train_time: float,
    ):
        global_step = self._total_steps if self._total_steps > 0 else iteration
        elapsed = time.time() - self._start_time if self._start_time else 0
        iter_steps_per_sec = self._get_iter_steps_per_sec()
        wall_steps_per_sec = self._get_wall_steps_per_sec(elapsed)

        if self._tb_writer:
            writer = self._tb_writer
            if metrics:
                for key, value in metrics.items():
                    writer.add_scalar(f"train/{key}", value, global_step)
            if reward is not None:
                writer.add_scalar("reward/mean", reward, global_step)
            if reward_components:
                for key, value in reward_components.items():
                    writer.add_scalar(f"reward/{key}", value, global_step)
            if self._mean_ep_length > 0:
                writer.add_scalar("episode/length", self._mean_ep_length, global_step)
            writer.add_scalar("episode/timeout_rate", self._timeout_rate, global_step)
            writer.add_scalar("episode/terminated_rate", self._terminated_rate, global_step)
            writer.add_scalar("timing/learner_wait_ms", self._wait_time * 1000, global_step)
            if self._has_iteration_extra_info:
                writer.add_scalar(
                    "timing/startup_wait_ms", self._startup_wait_time * 1000, global_step
                )
            writer.add_scalar("timing/learner_collect_ms", collect_time * 1000, global_step)
            writer.add_scalar("timing/learner_train_ms", train_time * 1000, global_step)
            for key, value in self._collector_timing.items():
                writer.add_scalar(f"timing/collector_{key}", value, global_step)
            if iter_steps_per_sec is not None:
                writer.add_scalar("perf/steps_per_sec", iter_steps_per_sec, global_step)
                writer.add_scalar("perf/steps_per_sec_iter", iter_steps_per_sec, global_step)
            elif wall_steps_per_sec is not None:
                writer.add_scalar("perf/steps_per_sec", wall_steps_per_sec, global_step)
            if wall_steps_per_sec is not None:
                writer.add_scalar("perf/steps_per_sec_wall", wall_steps_per_sec, global_step)
            writer.add_scalar(
                "perf/iter_ms", (self._collect_time + self._train_time) * 1000, global_step
            )
            writer.add_scalar(
                "perf/collect_train_ratio",
                self._collect_time / max(self._train_time, 1e-6),
                global_step,
            )

        if self._wandb_run:
            wandb = _load_wandb()
            if wandb is None:
                return
            log_dict: dict[str, Any] = {"iteration": iteration}
            if metrics:
                for key, value in metrics.items():
                    log_dict[f"train/{key}"] = value
            if reward is not None:
                log_dict["reward/mean"] = reward
            if reward_components:
                for key, value in reward_components.items():
                    log_dict[f"reward/{key}"] = value
            if self._mean_ep_length > 0:
                log_dict["episode/length"] = self._mean_ep_length
            log_dict["episode/timeout_rate"] = self._timeout_rate
            log_dict["episode/terminated_rate"] = self._terminated_rate
            log_dict["timing/learner_wait_ms"] = self._wait_time * 1000
            if self._has_iteration_extra_info:
                log_dict["timing/startup_wait_ms"] = self._startup_wait_time * 1000
            log_dict["timing/learner_collect_ms"] = collect_time * 1000
            log_dict["timing/learner_train_ms"] = train_time * 1000
            for key, value in self._collector_timing.items():
                log_dict[f"timing/collector_{key}"] = value
            if iter_steps_per_sec is not None:
                log_dict["perf/steps_per_sec"] = iter_steps_per_sec
                log_dict["perf/steps_per_sec_iter"] = iter_steps_per_sec
            elif wall_steps_per_sec is not None:
                log_dict["perf/steps_per_sec"] = wall_steps_per_sec
            if wall_steps_per_sec is not None:
                log_dict["perf/steps_per_sec_wall"] = wall_steps_per_sec
            log_dict["perf/iter_ms"] = (self._collect_time + self._train_time) * 1000
            log_dict["perf/collect_train_ratio"] = self._collect_time / max(self._train_time, 1e-6)
            wandb.log(log_dict, step=global_step)

    def log_status(self, status: str):
        self._status = status
        self._refresh()

    def _build_display(self) -> Panel:
        header_panel = self._build_header(include_status=True)
        left = self._build_metrics_table()
        right = self._build_reward_table()
        bottom = self._build_timing_table()
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(left, right)
        return Panel(
            Group(header_panel, grid, bottom),
            title="[bold] 🚀 UniLab Off-Policy Training [/]",
            border_style="bright_blue",
            padding=(0, 1),
        )

    def _build_metrics_table(self) -> Table:
        table = Table(
            title="[bold]Losses & Metrics[/]",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold cyan",
            expand=True,
            pad_edge=False,
        )
        table.add_column("Metric", style="white", ratio=2)
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
        return self._build_reward_table_common(wait_message="[dim]Waiting for data...[/]")

    def _build_timing_table(self) -> Table:
        table = Table(
            title="[bold]Timing & System[/]",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold blue",
            expand=True,
            pad_edge=False,
        )
        table.add_column("Item", style="white", ratio=2, no_wrap=True)
        table.add_column("Value", style="yellow", justify="right", ratio=1, no_wrap=True)
        table.add_column("Item", style="white", ratio=2, no_wrap=True)
        table.add_column("Value", style="yellow", justify="right", ratio=1, no_wrap=True)

        elapsed = time.time() - self._start_time if self._start_time else 0
        table.add_row("Elapsed", _fmt_time(elapsed), "Buffer", f"{self._buffer_size:,}")
        wait_ms = self._wait_time * 1000
        wait_color = "red" if wait_ms > 1.0 else "yellow"
        table.add_row(
            "[dim]learner[/] Wait",
            f"[{wait_color}]{wait_ms:.1f}ms[/]",
            "[dim]learner[/] Train",
            f"{self._train_time * 1000:.1f}ms",
        )
        table.add_row(
            "[dim]learner[/] Collect",
            f"{self._collect_time * 1000:.1f}ms",
            "",
            "",
        )
        if self._startup_wait_time > 0:
            table.add_row(
                "[dim]startup[/] Wait",
                f"{self._startup_wait_time * 1000:.1f}ms",
                "",
                "",
            )
        timing_items = list(self._collector_timing.items())
        for index in range(0, len(timing_items), 2):
            left_key, left_value = timing_items[index]
            if index + 1 < len(timing_items):
                right_key, right_value = timing_items[index + 1]
                table.add_row(
                    f"[dim]collector[/] {left_key}",
                    f"{left_value:.1f}ms",
                    f"[dim]collector[/] {right_key}",
                    f"{right_value:.1f}ms",
                )
            else:
                table.add_row(f"[dim]collector[/] {left_key}", f"{left_value:.1f}ms", "", "")
        table.add_row(
            "Timeout Rate",
            f"{self._timeout_rate * 100:.1f}%",
            "Terminated Rate",
            f"{self._terminated_rate * 100:.1f}%",
        )
        utilization = self._buffer_utilization
        if utilization >= 1.5:
            utilization_str = f"[bold red]{utilization:.2f}  (collector >> learner)[/]"
        elif utilization >= 1.0:
            utilization_str = f"[yellow]{utilization:.2f}[/]"
        else:
            utilization_str = f"[green]{utilization:.2f}[/]"
        table.add_row("Write/Read", utilization_str, "", "")
        table.add_row(
            "Envs",
            f"{self.num_envs:,}",
            "Sync Collect",
            f"{'✓' if self._sync_collection else '✗'} ({self._env_steps_per_sync})"
            if self._sync_collection
            else "✗",
        )
        if self._replay_queue_max > 0:
            replay_color = "green" if self._replay_queue_len < self._replay_queue_max else "yellow"
            table.add_row(
                "Replay Queue",
                f"[{replay_color}]{self._replay_queue_len}/{self._replay_queue_max}[/]",
                "",
                "",
            )
        iter_steps_per_sec = self._get_iter_steps_per_sec()
        wall_steps_per_sec = self._get_wall_steps_per_sec(elapsed)
        if iter_steps_per_sec is not None:
            table.add_row("Steps/s", f"{iter_steps_per_sec:,.0f}", "", "")
            if wall_steps_per_sec is not None:
                table.add_row("Wall Steps/s", f"{wall_steps_per_sec:,.0f}", "", "")
        elif wall_steps_per_sec is not None:
            table.add_row("Steps/s", f"{wall_steps_per_sec:,.0f}", "", "")
        return table
