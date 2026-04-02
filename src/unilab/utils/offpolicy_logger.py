"""Rich-based training logger for off-policy RL algorithms (SAC, TD3, etc).

Usage:
    from unilab.utils.offpolicy_logger import OffPolicyLogger

    logger = OffPolicyLogger(
        algo_name="FastSAC",
        max_iterations=1500,
        num_envs=4096,
        log_dir="logs/run_01",            # for tensorboard
        log_backend="tensorboard",        # "tensorboard", "wandb", or "none"
    )

    logger.start()                       # Begin Live display

    logger.log_buffer_fill(cur, total)   # During warmup/buffer fill
    logger.log_collector(step, buf, rew) # Collector progress (from subprocess)

    logger.log_step(                     # Each training iteration
        iteration=100,
        metrics={"qf_loss": 5.1, "actor_loss": -0.3, "alpha": 0.001},
        reward=8.5,
        reward_components={"track_lin_vel": 1.2, "action_rate": -0.05},
        collect_time=0.03,
        train_time=0.15,
    )

    logger.log_save(path)                # Checkpoint saved
    logger.finish()                      # End Live display
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table

from unilab.utils.logging_common import BaseTrainingLogger, _fmt_number, _fmt_time, _load_wandb


class OffPolicyLogger(BaseTrainingLogger):
    """Rich logger for off-policy RL algorithms (SAC, TD3, etc).

    Features:
    - Real-time Live table with training metrics
    - Loss tracking (any key-value pairs)
    - Reward tracking (mean + per-component breakdown)
    - Timing: collect/train per step, total elapsed, ETA
    - Buffer fill progress bar
    - Checkpoint save notifications
    - TensorBoard / W&B backend logging
    """

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
        log_backend: str = "tensorboard",  # "tensorboard", "wandb", "none"
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

    # ---- Lifecycle ----

    def _format_tensorboard_message(self, tb_dir: str) -> str:
        return f"[dim]TensorBoard logging to: {tb_dir}[/]"

    def _format_wandb_message(self, project: str, name: str) -> str:
        return f"[dim]W&B logging to project: {project}, run: {name}[/]"

    def start(self, *, status: str = "Warming up..."):
        """Begin the Live display."""
        super().start(status=status)

    def finish(self, *, title: str = "Training Summary", extra_summary: str = ""):
        """Stop the Live display and print a summary."""
        super().finish(
            title=title,
            extra_summary=f"  Total env steps: [yellow]{self._total_steps:,}[/]\n{extra_summary}",
        )

    # ---- Logging API ----

    def log_buffer_fill(self, current: int, target: int):
        """Update buffer fill progress."""
        self._buffer_size = current
        self._buffer_target = target
        pct = current / max(target, 1) * 100
        self._status = f"Buffer fill: {current:,}/{target:,} ({pct:.0f}%)"
        self._refresh()

    def update_collector_timing(self, timing_ms: dict[str, float]):
        """Update collector-side environment timing (milliseconds)."""
        self._collector_timing.update(timing_ms)

    def update_done_rates(self, timeout_rate: float, terminated_rate: float):
        """Update timeout/terminated ratio among completed episodes in collector window."""
        self._timeout_rate = float(timeout_rate)
        self._terminated_rate = float(terminated_rate)

    def update_buffer_utilization(self, utilization: float):
        """Update buffer fill ratio (0.0–1.0). Displayed in the timing panel."""
        self._buffer_utilization = float(utilization)

    def update_replay_queue(self, current_len: int, max_size: int):
        """Update replay queue occupancy (APPO-specific)."""
        self._replay_queue_len = current_len
        self._replay_queue_max = max_size

    def set_collection_sync(self, enabled: bool, env_steps_per_sync: int = 0):
        """Set collection/training synchronization status for display."""
        self._sync_collection = enabled
        self._env_steps_per_sync = env_steps_per_sync

    def log_collector(self, total_steps: int, buffer_size: int, mean_reward: float = 0.0):
        """Update collector progress (called periodically from metrics queue drain)."""
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
        """Log one training iteration."""
        self._iteration = iteration
        self._collect_time = collect_time
        self._train_time = train_time
        self._wait_time = wait_time
        self._iter_times.append(collect_time + train_time)

        if metrics:
            self._latest_metrics.update(metrics)
        if reward is not None:
            self._reward_history.append(reward)
        if reward_components:
            self._latest_reward_components = reward_components

        self._status = "Training"
        self._refresh()

        # ---- Write to backend ----
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
        """Write metrics to TensorBoard / W&B."""
        global_step = self._total_steps if self._total_steps > 0 else iteration

        elapsed = time.time() - self._start_time if self._start_time else 0

        # ---- TensorBoard ----
        if self._tb_writer:
            w = self._tb_writer

            # train/ — model outputs (losses, alpha, etc.)
            if metrics:
                for k, v in metrics.items():
                    w.add_scalar(f"train/{k}", v, global_step)

            # reward/ — reward signals
            if reward is not None:
                w.add_scalar("reward/mean", reward, global_step)
            if reward_components:
                for k, v in reward_components.items():
                    w.add_scalar(f"reward/{k}", v, global_step)

            # episode/ — per-episode statistics
            if self._mean_ep_length > 0:
                w.add_scalar("episode/length", self._mean_ep_length, global_step)
            w.add_scalar("episode/timeout_rate", self._timeout_rate, global_step)
            w.add_scalar("episode/terminated_rate", self._terminated_rate, global_step)

            # timing/ — learner-side and collector-side timing
            w.add_scalar("timing/learner_wait_ms", self._wait_time * 1000, global_step)
            w.add_scalar("timing/learner_collect_ms", collect_time * 1000, global_step)
            w.add_scalar("timing/learner_train_ms", train_time * 1000, global_step)
            for key, val in self._collector_timing.items():
                w.add_scalar(f"timing/collector_{key}", val, global_step)

            # perf/ — throughput and efficiency
            if elapsed > 0 and self._total_steps > 0:
                w.add_scalar("perf/steps_per_sec", self._total_steps / elapsed, global_step)
            w.add_scalar(
                "perf/iter_ms", (self._collect_time + self._train_time) * 1000, global_step
            )
            w.add_scalar(
                "perf/collect_train_ratio",
                self._collect_time / max(self._train_time, 1e-6),
                global_step,
            )

        # ---- W&B ----
        if self._wandb_run:
            wandb = _load_wandb()
            if wandb is None:
                return

            log_dict: dict[str, Any] = {"iteration": iteration}
            if metrics:
                for k, v in metrics.items():
                    log_dict[f"train/{k}"] = v

            # reward/
            if reward is not None:
                log_dict["reward/mean"] = reward
            if reward_components:
                for k, v in reward_components.items():
                    log_dict[f"reward/{k}"] = v

            # episode/
            if self._mean_ep_length > 0:
                log_dict["episode/length"] = self._mean_ep_length
            log_dict["episode/timeout_rate"] = self._timeout_rate
            log_dict["episode/terminated_rate"] = self._terminated_rate

            # timing/
            log_dict["timing/learner_wait_ms"] = self._wait_time * 1000
            log_dict["timing/learner_collect_ms"] = collect_time * 1000
            log_dict["timing/learner_train_ms"] = train_time * 1000
            for key, val in self._collector_timing.items():
                log_dict[f"timing/collector_{key}"] = val

            # perf/
            if elapsed > 0 and self._total_steps > 0:
                log_dict["perf/steps_per_sec"] = self._total_steps / elapsed
            log_dict["perf/iter_ms"] = (self._collect_time + self._train_time) * 1000
            log_dict["perf/collect_train_ratio"] = self._collect_time / max(self._train_time, 1e-6)

            wandb.log(log_dict, step=global_step)

    def log_status(self, status: str):
        """Set a custom status message."""
        self._status = status
        self._refresh()

    # ---- Display Building ----

    def _build_display(self) -> Panel:
        """Build the full rich display panel."""
        header_panel = self._build_header(include_status=True)

        # Body: side-by-side tables
        left = self._build_metrics_table()
        right = self._build_reward_table()
        bottom = self._build_timing_table()

        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(left, right)

        main_group = Group(header_panel, grid, bottom)

        return Panel(
            main_group,
            title="[bold] 🚀 UniLab Off-Policy Training [/]",
            border_style="bright_blue",
            padding=(0, 1),
        )

    def _build_metrics_table(self) -> Table:
        """Build the losses/metrics table."""
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
            # Sort: losses first, then other metrics
            loss_keys = sorted([k for k in self._latest_metrics if "loss" in k.lower()])
            other_keys = sorted([k for k in self._latest_metrics if "loss" not in k.lower()])

            for k in loss_keys:
                v = self._latest_metrics[k]
                name = k.replace("_", " ").title()
                val_str = _fmt_number(v)
                style = "red" if v > 10 else "yellow"
                table.add_row(f"{name}", f"[{style}]{val_str}[/]")

            for k in other_keys:
                v = self._latest_metrics[k]
                name = k.replace("_", " ").title()
                table.add_row(f"  {name}", _fmt_number(v))

        return table

    def _build_reward_table(self) -> Table:
        """Build the reward breakdown table."""
        return self._build_reward_table_common(wait_message="[dim]Waiting for data...[/]")

    def _build_timing_table(self) -> Table:
        """Build the timing info table."""
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

        table.add_row(
            "Elapsed",
            _fmt_time(elapsed),
            "Buffer",
            f"{self._buffer_size:,}",
        )

        # Wait time with color coding
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
        timing_items = list(self._collector_timing.items())
        for i in range(0, len(timing_items), 2):
            left_key, left_val = timing_items[i]
            if i + 1 < len(timing_items):
                right_key, right_val = timing_items[i + 1]
                table.add_row(
                    f"[dim]collector[/] {left_key}",
                    f"{left_val:.1f}ms",
                    f"[dim]collector[/] {right_key}",
                    f"{right_val:.1f}ms",
                )
            else:
                table.add_row(
                    f"[dim]collector[/] {left_key}",
                    f"{left_val:.1f}ms",
                    "",
                    "",
                )
        table.add_row(
            "Timeout Rate",
            f"{self._timeout_rate * 100:.1f}%",
            "Terminated Rate",
            f"{self._terminated_rate * 100:.1f}%",
        )

        util = self._buffer_utilization
        if util >= 1.5:
            util_str = f"[bold red]{util:.2f}  (collector >> learner)[/]"
        elif util >= 1.0:
            util_str = f"[yellow]{util:.2f}[/]"
        else:
            util_str = f"[green]{util:.2f}[/]"
        table.add_row("Write/Read", util_str, "", "")

        table.add_row(
            "Envs",
            f"{self.num_envs:,}",
            "Sync Collect",
            f"{'✓' if self._sync_collection else '✗'} ({self._env_steps_per_sync})"
            if self._sync_collection
            else "✗",
        )

        if self._replay_queue_max > 0:
            rq_color = "green" if self._replay_queue_len < self._replay_queue_max else "yellow"
            table.add_row(
                "Replay Queue",
                f"[{rq_color}]{self._replay_queue_len}/{self._replay_queue_max}[/]",
                "",
                "",
            )

        # Steps per second
        if elapsed > 0 and self._total_steps > 0:
            sps = self._total_steps / elapsed
            table.add_row("Steps/s", f"{sps:,.0f}", "", "")

        return table
