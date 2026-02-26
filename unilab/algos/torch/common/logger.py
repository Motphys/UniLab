"""Rich-based training logger — modular, plug-and-play for all RL algorithms.

Usage:
    from unilab.algos.torch.common.logger import TrainingLogger

    logger = TrainingLogger(
        algo_name="FastSAC",
        max_iterations=1500,
        num_envs=4096,
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
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.layout import Layout
from rich import box


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


class TrainingLogger:
    """Modular Rich training logger for RL algorithms.

    Features:
    - Real-time Live table with training metrics
    - Loss tracking (any key-value pairs)
    - Reward tracking (mean + per-component breakdown)
    - Timing: collect/train per step, total elapsed, ETA
    - Buffer fill progress bar
    - Checkpoint save notifications
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
    ):
        self.algo_name = algo_name
        self.max_iterations = max_iterations
        self.num_envs = num_envs
        self.env_name = env_name
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self._console = Console()
        self._live: Optional[Live] = None
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
        self._latest_metrics: Dict[str, float] = {}
        self._latest_reward_components: Dict[str, float] = {}

        # Timing
        self._collect_time: float = 0.0
        self._train_time: float = 0.0
        self._iter_times: deque = deque(maxlen=50)

        # Status message
        self._status: str = "Initializing..."
        self._last_save: str = ""

    # ---- Lifecycle ----

    def start(self):
        """Begin the logging."""
        self._start_time = time.time()
        self._status = "Warming up..."
        self._console.print(self._build_display())

    def finish(self):
        """End the logging and print summary."""
        self._console.print(self._build_display())

        elapsed = time.time() - self._start_time
        self._console.print()
        self._console.print(
            Panel(
                f"[bold green]✓ Training complete[/]\n"
                f"  Algo: [cyan]{self.algo_name}[/] | Env: [cyan]{self.env_name}[/]\n"
                f"  Iterations: [yellow]{self._iteration}[/]/{self.max_iterations}\n"
                f"  Total time: [yellow]{_fmt_time(elapsed)}[/]\n"
                f"  Total env steps: [yellow]{self._total_steps:,}[/]\n"
                + (f"  Last checkpoint: [dim]{self._last_save}[/]" if self._last_save else ""),
                title="[bold]Training Summary[/]",
                border_style="green",
            )
        )

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
        metrics: Dict[str, float] | None = None,
        reward: float | None = None,
        reward_components: Dict[str, float] | None = None,
        collect_time: float = 0.0,
        train_time: float = 0.0,
        extra_info: Dict[str, Any] | None = None,
    ):
        """Log one training iteration."""
        self._iteration = iteration
        self._collect_time = collect_time
        self._train_time = train_time
        self._iter_times.append(collect_time + train_time)

        if metrics:
            self._latest_metrics.update(metrics)
        if reward is not None:
            self._reward_history.append(reward)
        if reward_components:
            self._latest_reward_components = reward_components

        self._status = "Training"
        # Print the dashboard periodically
        self._console.print(self._build_display())

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
        if self._status != "Training":
            # Only print warming up, buffer fill, or status changes, 
            # or if training, we'll let `log_step` explicitly call print if needed (or throttle it)
            pass
        # To avoid spamming, we will just print periodically in log_step instead or on major events.

    def _build_display(self) -> Panel:
        """Build the full rich display panel."""
        # Header
        elapsed = time.time() - self._start_time if self._start_time else 0
        eta = self._estimate_eta()
        header_text = Text()
        header_text.append(f" {self.algo_name}", style="bold cyan")
        header_text.append(f"  │  ", style="dim")
        header_text.append(f"{self.env_name}", style="bold white")
        header_text.append(f"  │  ", style="dim")
        header_text.append(f"iter {self._iteration}/{self.max_iterations}", style="yellow")
        header_text.append(f"  │  ", style="dim")
        header_text.append(f"⏱ {_fmt_time(elapsed)}", style="green")
        if eta:
            header_text.append(f"  │  ETA ", style="dim")
            header_text.append(eta, style="bold magenta")
        header_text.append(f"  │  ", style="dim")
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

        from rich.console import Group
        main_group = Group(
            header_panel,
            grid,
            bottom
        )

        return Panel(
            main_group,
            title=f"[bold] 🚀 UniLab Training Dashboard [/]",
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
                table.add_row(f"📉 {name}", f"[{style}]{val_str}[/]")

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

            table.add_row(
                f"[bold]⭐ Mean Reward[/] {trend}",
                f"[bold green]{mean_rew:.3f}[/]"
            )
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
        table.add_column("Item", style="white")
        table.add_column("Value", style="yellow", justify="right")
        table.add_column("Item", style="white")
        table.add_column("Value", style="yellow", justify="right")

        elapsed = time.time() - self._start_time if self._start_time else 0

        table.add_row(
            "⏱ Elapsed", _fmt_time(elapsed),
            "📦 Buffer", f"{self._buffer_size:,}",
        )
        table.add_row(
            "🔄 Collect", f"{self._collect_time * 1000:.1f}ms",
            "🧠 Train", f"{self._train_time * 1000:.1f}ms",
        )

        table.add_row(
            "🌐 Envs", f"{self.num_envs:,}",
            "", ""
        )

        # Steps per second
        if elapsed > 0 and self._total_steps > 0:
            sps = self._total_steps / elapsed
            table.add_row(
                "🚀 Steps/s", f"{sps:,.0f}",
                "", ""
            )

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
