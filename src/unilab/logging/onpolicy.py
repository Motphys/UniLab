from __future__ import annotations

from typing import Any

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from unilab.logging.common import BaseTrainingLogger, _fmt_number, _load_wandb


class OnPolicyLogger(BaseTrainingLogger):
    """Rich logger for on-policy RL (PPO, A2C, etc)."""

    def __init__(
        self,
        algo_name: str = "PPO",
        max_iterations: int = 1500,
        num_envs: int = 4096,
        num_steps: int = 24,
        env_name: str = "",
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
            tensorboard_subdir="tb",
        )
        self.num_steps = num_steps

    def start(self, *, status: str = ""):
        super().start(status=status)

    def finish(self, *, title: str = "Training Summary", extra_summary: str = ""):
        super().finish(title=title, extra_summary=extra_summary)

    def log_step(
        self,
        iteration: int,
        metrics: dict[str, float] | None = None,
        reward: float | None = None,
        reward_components: dict[str, float] | None = None,
        collect_time: float = 0.0,
        train_time: float = 0.0,
    ):
        self._iteration = iteration
        self._collect_time = collect_time
        self._train_time = train_time

        if metrics:
            self._latest_metrics.update(metrics)
        if reward is not None:
            self._reward_history.append(reward)
        if reward_components:
            self._latest_reward_components = reward_components

        self._refresh()
        self._backend_log_step(iteration, metrics, reward, reward_components)

    def _backend_log_step(
        self,
        iteration: int,
        metrics: dict[str, float] | None,
        reward: float | None,
        reward_components: dict[str, float] | None,
    ):
        if self._tb_writer:
            w = self._tb_writer
            if metrics:
                for k, v in metrics.items():
                    w.add_scalar(f"train/{k}", v, iteration)
            if reward is not None:
                w.add_scalar("reward/mean", reward, iteration)
            if reward_components:
                for k, v in reward_components.items():
                    w.add_scalar(f"reward/{k}", v, iteration)
            if self._mean_ep_length > 0:
                w.add_scalar("episode/length", self._mean_ep_length, iteration)
            w.add_scalar("perf/collect_time_ms", self._collect_time * 1000, iteration)
            w.add_scalar("perf/train_time_ms", self._train_time * 1000, iteration)

        if self._wandb_run:
            wandb = _load_wandb()
            if wandb is None:
                return

            log_dict: dict[str, Any] = {"iteration": iteration}
            if metrics:
                for k, v in metrics.items():
                    log_dict[f"train/{k}"] = v
            if reward is not None:
                log_dict["reward/mean"] = reward
            if reward_components:
                for k, v in reward_components.items():
                    log_dict[f"reward/{k}"] = v
            if self._mean_ep_length > 0:
                log_dict["episode/length"] = self._mean_ep_length
            log_dict["perf/collect_time_ms"] = self._collect_time * 1000
            log_dict["perf/train_time_ms"] = self._train_time * 1000
            wandb.log(log_dict, step=iteration)

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
            title="[bold] 🚀 UniLab On-Policy Training [/]",
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

        iter_time = self._collect_time + self._train_time
        fps = int(self.num_envs * self.num_steps / max(iter_time, 1e-8)) if iter_time > 0 else 0

        learner_items = [
            ("Train", f"{self._train_time * 1000:.1f}ms"),
            ("Iter Time", f"{iter_time * 1000:.1f}ms"),
        ]
        collector_items = [
            ("Collect", f"{self._collect_time * 1000:.1f}ms"),
        ]
        system_items = [
            ("Envs", f"{self.num_envs:,}"),
            ("Steps/s", f"{fps:,}"),
        ]

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

    def _build_compact_header(self, *, include_status: bool) -> Text:
        iter_time = self._collect_time + self._train_time
        extra_fields: list[tuple[str, str]] = []
        if iter_time > 0:
            steps_per_second = self.num_envs * self.num_steps / iter_time
            extra_fields.append((f"Steps/s {steps_per_second:,.0f}", "bold green"))
        return super()._build_compact_header(
            include_status=include_status,
            extra_fields=extra_fields,
        )
