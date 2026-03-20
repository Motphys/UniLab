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

import importlib
import time
from collections import deque
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _fmt_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _fmt_number(v: float, width: int = 8) -> str:
    """Smart number formatting."""
    if abs(v) == 0:
        return "0"
    if abs(v) >= 1e6:
        return f"{v:.2e}"
    if abs(v) >= 100:
        return f"{v:.1f}"
    if abs(v) >= 1:
        return f"{v:.3f}"
    if abs(v) >= 0.001:
        return f"{v:.4f}"
    return f"{v:.2e}"


def _load_wandb() -> Any | None:
    """Load wandb lazily so it remains an optional dependency."""
    try:
        return importlib.import_module("wandb")
    except ImportError:
        return None


class OffPolicyLogger:
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
        wandb_name: str = "",
    ):
        self.algo_name = algo_name
        self.max_iterations = max_iterations
        self.num_envs = num_envs
        self.env_name = env_name
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self._no_print = log_backend.lower() == "no_print"
        self._log_backend = "none" if self._no_print else log_backend.lower()

        self._console = Console()
        self._live: Live | None = None
        self._refresh_rate = refresh_per_second

        # State
        self._start_time: float = 0.0
        self._iteration: int = 0
        self._total_steps: int = 0
        self._buffer_size: int = 0
        self._mean_ep_length: float = 0.0
        self._buffer_target: int = 0

        # Metrics history (for sparkline / trend)
        self._reward_history: deque = deque(maxlen=200)
        self._latest_metrics: dict[str, float] = {}
        self._latest_reward_components: dict[str, float] = {}

        # Timing
        self._collect_time: float = 0.0
        self._train_time: float = 0.0
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

        # Status message
        self._status: str = "Initializing..."
        self._last_save: str = ""

        # ---- Backend logging ----
        self._log_dir = log_dir
        self._tb_writer: "SummaryWriter | None" = None  # type: ignore[name-defined]
        self._wandb_run = None

        if self._log_backend == "tensorboard" and log_dir:
            self._init_tensorboard(log_dir)
        elif self._log_backend == "wandb":
            self._init_wandb(
                project=wandb_project,
                name=wandb_name or f"{algo_name}_{env_name}",
                log_dir=log_dir,
            )

    def _init_tensorboard(self, log_dir: str):
        """Initialize TensorBoard SummaryWriter."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._tb_writer = SummaryWriter(log_dir=log_dir)
            if not self._no_print:
                self._console.print(f"[dim]TensorBoard logging to: {log_dir}[/]")
        except ImportError:
            if not self._no_print:
                self._console.print("[yellow]tensorboard not installed, skipping TB logging[/]")

    def _init_wandb(self, project: str, name: str, log_dir: str):
        """Initialize Weights & Biases run."""
        wandb = _load_wandb()
        if wandb is None:
            if not self._no_print:
                self._console.print("[yellow]wandb not installed, skipping W&B logging[/]")
            return

        self._wandb_run = wandb.init(
            project=project,
            name=name,
            config={
                "algo": self.algo_name,
                "env": self.env_name,
                "num_envs": self.num_envs,
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "max_iterations": self.max_iterations,
            },
            dir=log_dir or None,
            reinit=True,
        )
        if not self._no_print:
            self._console.print(f"[dim]W&B logging to project: {project}, run: {name}[/]")

    # ---- Lifecycle ----

    def start(self):
        """Begin the Live display."""
        self._start_time = time.time()
        self._status = "Warming up..."
        if not self._no_print:
            self._live = Live(
                self._build_display(),
                console=self._console,
                refresh_per_second=self._refresh_rate,
                transient=False,
            )
            self._live.start()

    def finish(self):
        """Stop the Live display and print a summary."""
        if self._live is not None:
            self._live.update(self._build_display())
            self._live.stop()
            self._live = None

        elapsed = time.time() - self._start_time
        if not self._no_print:
            self._console.print()
            self._console.print(
                Panel(
                    f"[bold green]Training complete[/]\n"
                    f"  Algo: [cyan]{self.algo_name}[/] | Env: [cyan]{self.env_name}[/]\n"
                    f"  Iterations: [yellow]{self._iteration}[/]/{self.max_iterations}\n"
                    f"  Total time: [yellow]{_fmt_time(elapsed)}[/]\n"
                    f"  Total env steps: [yellow]{self._total_steps:,}[/]\n"
                    + (f"  Last checkpoint: [dim]{self._last_save}[/]" if self._last_save else ""),
                    title="[bold]Training Summary[/]",
                    border_style="green",
                )
            )

        if self._tb_writer:
            self._tb_writer.close()
        if self._wandb_run:
            wandb = _load_wandb()
            if wandb is not None:
                wandb.finish()

    # ---- Logging API ----

    def log_buffer_fill(self, current: int, target: int):
        """Update buffer fill progress."""
        self._buffer_size = current
        self._buffer_target = target
        pct = current / max(target, 1) * 100
        self._status = f"Buffer fill: {current:,}/{target:,} ({pct:.0f}%)"
        self._refresh()

    def update_ep_length(self, length: float):
        """Update mean episode length from collector."""
        self._mean_ep_length = length

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

    def log_save(self, path: str):
        """Log a checkpoint save."""
        self._last_save = path
        self._refresh()

    def log_status(self, status: str):
        """Set a custom status message."""
        self._status = status
        self._refresh()

    # ---- Display Building ----

    def _refresh(self):
        if self._live is not None:
            self._live.update(self._build_display())

    def _build_display(self) -> Panel:
        """Build the full rich display panel."""
        # Header
        elapsed = time.time() - self._start_time if self._start_time else 0
        eta = self._estimate_eta()
        header_text = Text()
        header_text.append(f" {self.algo_name}", style="bold cyan")
        header_text.append("  │  ", style="dim")
        header_text.append(f"{self.env_name}", style="bold white")
        header_text.append("  │  ", style="dim")
        header_text.append(f"iter {self._iteration}/{self.max_iterations}", style="yellow")
        header_text.append("  │  ", style="dim")
        header_text.append(f"⏱ {_fmt_time(elapsed)}", style="green")
        if eta:
            header_text.append("  │  ETA ", style="dim")
            header_text.append(eta, style="bold magenta")
        header_text.append("  │  ", style="dim")
        header_text.append(self._status, style="dim italic")

        header_panel = Panel(header_text, style="dim", box=box.SIMPLE)

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
        table = Table(
            title="[bold]Rewards[/]",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold green",
            expand=True,
            pad_edge=False,
        )
        table.add_column("Component", style="white", ratio=2)
        table.add_column("Value", justify="right", ratio=1)

        # Mean reward
        if self._reward_history:
            recent = list(self._reward_history)
            mean_rew = sum(recent[-50:]) / max(len(recent[-50:]), 1)
            peak_rew = max(recent) if recent else 0

            # Trend indicator
            if len(recent) >= 10:
                old = sum(recent[-20:-10]) / 10
                new = sum(recent[-10:]) / 10
                if new > old * 1.05:
                    trend = "[green]▲[/]"
                elif new < old * 0.95:
                    trend = "[red]▼[/]"
                else:
                    trend = "[yellow]━[/]"
            else:
                trend = ""

            table.add_row(f"[bold]Mean Reward[/] {trend}", f"[bold green]{mean_rew:.3f}[/]")
            table.add_row("  Peak", f"[dim]{peak_rew:.3f}[/]")
            if self._mean_ep_length > 0:
                table.add_row("  Ep Len", f"[dim]{self._mean_ep_length:.1f}[/]")
            table.add_row("", "")  # spacer
        else:
            table.add_row("[dim]Waiting for data...[/]", "")

        # Sub-components
        if self._latest_reward_components:
            for name, val in sorted(self._latest_reward_components.items()):
                display = name.replace("reward/", "").replace("_", " ")
                color = "green" if val > 0 else "red" if val < 0 else "dim"
                table.add_row(f"  {display}", f"[{color}]{val:+.4f}[/]")

        return table

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

    def _estimate_eta(self) -> str:
        """Estimate time remaining."""
        if self._iteration <= 0 or not self._iter_times:
            return ""
        elapsed = time.time() - self._start_time
        remaining = self.max_iterations - self._iteration
        avg_iter = elapsed / self._iteration
        eta_s = remaining * avg_iter
        return _fmt_time(eta_s)
