from __future__ import annotations

import os
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _fmt_number(v: float) -> str:
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


class OnPolicyLogger:
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
        wandb_name: str = "",
    ):
        self.algo_name = algo_name
        self.max_iterations = max_iterations
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.env_name = env_name

        self._no_print = log_backend.lower() == "no_print"
        self._log_backend = "none" if self._no_print else log_backend.lower()

        self._console = Console()
        self._live: Live | None = None

        self._start_time: float = 0.0
        self._iteration: int = 0
        self._reward_history: deque = deque(maxlen=200)
        self._latest_metrics: dict[str, float] = {}
        self._latest_reward_components: dict[str, float] = {}
        self._collect_time: float = 0.0
        self._train_time: float = 0.0
        self._mean_ep_length: float = 0.0
        self._last_save: str = ""

        self._log_dir = log_dir
        self._tb_writer: "SummaryWriter | None" = None
        self._wandb_run = None

        if self._log_backend == "tensorboard" and log_dir:
            self._init_tensorboard(log_dir)
        elif self._log_backend == "wandb":
            self._init_wandb(
                project=wandb_project, name=wandb_name or f"{algo_name}_{env_name}", log_dir=log_dir
            )

    def _init_tensorboard(self, log_dir: str):
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = os.path.join(log_dir, "tb")
            os.makedirs(tb_dir, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=tb_dir)
            if not self._no_print:
                self._console.print(f"[dim]TensorBoard: {tb_dir}[/]")
        except ImportError:
            if not self._no_print:
                self._console.print("[yellow]tensorboard not installed[/]")

    def _init_wandb(self, project: str, name: str, log_dir: str):
        try:
            import wandb

            self._wandb_run = wandb.init(
                project=project,
                name=name,
                config={"algo": self.algo_name, "env": self.env_name, "num_envs": self.num_envs},
                dir=log_dir or None,
                reinit=True,
            )
            if not self._no_print:
                self._console.print(f"[dim]W&B: {project}/{name}[/]")
        except ImportError:
            if not self._no_print:
                self._console.print("[yellow]wandb not installed[/]")

    def start(self):
        self._start_time = time.time()
        if not self._no_print:
            self._live = Live(
                self._build_display(), console=self._console, refresh_per_second=4, transient=False
            )
            self._live.start()

    def finish(self):
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
                    + (f"  Last checkpoint: [dim]{self._last_save}[/]" if self._last_save else ""),
                    title="[bold]Training Summary[/]",
                    border_style="green",
                )
            )

        if self._tb_writer:
            self._tb_writer.close()
        if self._wandb_run:
            import wandb

            wandb.finish()

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
            import wandb

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

    def update_ep_length(self, length: float):
        self._mean_ep_length = length

    def log_save(self, path: str):
        self._last_save = path
        self._refresh()

    def _refresh(self):
        if self._live is not None:
            self._live.update(self._build_display())

    def _build_display(self) -> Panel:
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

        header_panel = Panel(header_text, style="dim", box=box.SIMPLE)

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
            title="[bold] 🚀 UniLab On-Policy Training [/]",
            border_style="bright_blue",
            padding=(0, 1),
        )

    def _build_metrics_table(self) -> Table:
        table = Table(
            title="[bold]Policy Metrics[/]",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold cyan",
            expand=True,
            pad_edge=False,
        )
        table.add_column("Metric", style="white", ratio=2)
        table.add_column("Value", style="yellow", justify="right", ratio=1)

        if not self._latest_metrics:
            table.add_row("[dim]Waiting...[/]", "")
        else:
            for k in sorted(self._latest_metrics.keys()):
                v = self._latest_metrics[k]
                name = k.replace("_", " ").title()
                table.add_row(name, _fmt_number(v))

        return table

    def _build_reward_table(self) -> Table:
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

        if self._reward_history:
            recent = list(self._reward_history)
            mean_rew = sum(recent[-50:]) / max(len(recent[-50:]), 1)
            peak_rew = max(recent) if recent else 0

            if len(recent) >= 10:
                old = sum(recent[-20:-10]) / 10
                new = sum(recent[-10:]) / 10
                trend = (
                    "[green]▲[/]"
                    if new > old * 1.05
                    else "[red]▼[/]"
                    if new < old * 0.95
                    else "[yellow]━[/]"
                )
            else:
                trend = ""

            table.add_row(f"[bold]Mean Reward[/] {trend}", f"[bold green]{mean_rew:.3f}[/]")
            table.add_row("  Peak", f"[dim]{peak_rew:.3f}[/]")
            if self._mean_ep_length > 0:
                table.add_row("  Ep Len", f"[dim]{self._mean_ep_length:.1f}[/]")
            table.add_row("", "")
        else:
            table.add_row("[dim]Waiting...[/]", "")

        if self._latest_reward_components:
            for name, val in sorted(self._latest_reward_components.items()):
                display = name.replace("reward/", "").replace("_", " ")
                color = "green" if val > 0 else "red" if val < 0 else "dim"
                table.add_row(f"  {display}", f"[{color}]{val:+.4f}[/]")

        return table

    def _build_timing_table(self) -> Table:
        table = Table(
            title="[bold]Timing[/]",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold blue",
            expand=True,
            pad_edge=False,
        )
        table.add_column("Item", style="white", ratio=1)
        table.add_column("Value", style="yellow", justify="right", ratio=1)
        table.add_column("Item", style="white", ratio=1)
        table.add_column("Value", style="yellow", justify="right", ratio=1)

        elapsed = time.time() - self._start_time if self._start_time else 0
        iter_time = self._collect_time + self._train_time
        fps = int(self.num_envs * self.num_steps / max(iter_time, 1e-8)) if iter_time > 0 else 0

        table.add_row("Elapsed", _fmt_time(elapsed), "Envs", f"{self.num_envs:,}")
        table.add_row(
            "Collect",
            f"{self._collect_time * 1000:.1f}ms",
            "Train",
            f"{self._train_time * 1000:.1f}ms",
        )
        table.add_row("Iter Time", f"{iter_time * 1000:.1f}ms", "Steps/s", f"{fps:,}")

        return table

    def _estimate_eta(self) -> str:
        if self._iteration <= 0:
            return ""
        elapsed = time.time() - self._start_time
        remaining = self.max_iterations - self._iteration
        avg_iter = elapsed / self._iteration
        eta_s = remaining * avg_iter
        return _fmt_time(eta_s)
