from __future__ import annotations

import importlib
import os
import sys
import time
from collections import deque
from typing import Any

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


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


def _console_supports_unicode() -> bool:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        "⏱ ▲ ▼ ━".encode(encoding)
    except UnicodeEncodeError:
        return False
    return True


def _load_wandb() -> Any | None:
    """Load wandb lazily so it remains an optional dependency."""
    try:
        return importlib.import_module("wandb")
    except ImportError:
        return None


class BaseTrainingLogger:
    """Shared lifecycle and backend logging setup for rich training loggers."""

    def __init__(
        self,
        *,
        algo_name: str,
        max_iterations: int,
        num_envs: int,
        env_name: str,
        log_dir: str,
        log_backend: str,
        wandb_project: str,
        wandb_entity: str | None,
        wandb_name: str,
        wandb_group: list[str] | None | str,
        wandb_job_type: str | None,
        wandb_tags: list[str] | None,
        wandb_notes: str | None,
        refresh_per_second: int = 4,
        fixed_terminal_refresh: bool = False,
        tensorboard_subdir: str | None = "tb",
        wandb_config: dict[str, Any] | None = None,
    ):
        self.algo_name = algo_name
        self.max_iterations = max_iterations
        self.num_envs = num_envs
        self.env_name = env_name

        self._no_print = log_backend.lower() == "no_print"
        self._log_backend = "none" if self._no_print else log_backend.lower()

        self._unicode_console = _console_supports_unicode()
        self._console = Console(force_terminal=False if not self._unicode_console else None)
        self._live: Live | None = None
        self._refresh_rate = refresh_per_second
        self._fixed_terminal_refresh = fixed_terminal_refresh
        self._last_live_refresh_time: float | None = None

        self._start_time: float = 0.0
        self._iteration: int = 0
        self._reward_history: deque[float] = deque(maxlen=200)
        self._latest_metrics: dict[str, float] = {}
        self._latest_reward_components: dict[str, float] = {}
        self._collect_time: float = 0.0
        self._train_time: float = 0.0
        self._mean_ep_length: float = 0.0
        self._last_save: str = ""
        self._status: str = ""

        self._log_dir = log_dir
        self._tb_writer: Any | None = None
        self._wandb_run = None
        self._owns_wandb_run = False
        self._finished = False
        self._closed = False

        if self._log_backend == "tensorboard" and log_dir:
            self._init_tensorboard(log_dir, tensorboard_subdir)
        elif self._log_backend == "wandb":
            self._init_wandb(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_name or f"{algo_name}_{env_name}",
                log_dir=log_dir,
                group=wandb_group,
                job_type=wandb_job_type,
                tags=wandb_tags,
                notes=wandb_notes,
                extra_config=wandb_config,
            )

    def _format_tensorboard_message(self, tb_dir: str) -> str:
        return f"[dim]TensorBoard: {tb_dir}[/]"

    def _format_wandb_message(self, project: str, name: str) -> str:
        return f"[dim]W&B: {project}/{name}[/]"

    def _init_tensorboard(self, log_dir: str, subdir: str | None):
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = log_dir if subdir is None else os.path.join(log_dir, subdir)
            os.makedirs(tb_dir, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=tb_dir)
            if not self._no_print:
                self._console.print(self._format_tensorboard_message(tb_dir))
        except ImportError:
            if not self._no_print:
                self._console.print("[yellow]tensorboard not installed[/]")

    def _init_wandb(
        self,
        *,
        project: str,
        entity: str | None,
        name: str,
        log_dir: str,
        group: str | None | list[str],
        job_type: str | None,
        tags: list[str] | None,
        notes: str | None,
        extra_config: dict[str, Any] | None = None,
    ):
        wandb = _load_wandb()
        if wandb is None:
            if not self._no_print:
                self._console.print("[yellow]wandb not installed[/]")
            return

        self._wandb_run = wandb.run
        if self._wandb_run is None:
            config: dict[str, Any] = {
                "algo": self.algo_name,
                "env": self.env_name,
                "num_envs": self.num_envs,
            }
            if extra_config:
                config.update(extra_config)

            kwargs: dict[str, Any] = {
                "project": project,
                "name": name,
                "config": config,
                "dir": log_dir or None,
                "reinit": True,
            }
            if entity:
                kwargs["entity"] = entity
            if group:
                kwargs["group"] = group
            if job_type:
                kwargs["job_type"] = job_type
            if tags:
                kwargs["tags"] = tags
            if notes:
                kwargs["notes"] = notes

            self._wandb_run = wandb.init(**kwargs)
            self._owns_wandb_run = True

        if not self._no_print:
            self._console.print(self._format_wandb_message(project, name))

    def start(self, *, status: str = ""):
        if self._live is not None:
            if status:
                self._status = status
            self._refresh()
            return

        self._start_time = time.time()
        self._status = status
        if not self._no_print:
            self._live = Live(
                self._build_display(),
                console=self._console,
                auto_refresh=self._fixed_terminal_refresh,
                refresh_per_second=self._refresh_rate,
                transient=False,
                get_renderable=(self._build_display if self._fixed_terminal_refresh else None),
            )
            self._live.start(refresh=False)

    def _stop_live(self) -> None:
        live = self._live
        if live is None:
            return

        # Drop the owned reference first so repeated cleanup stays idempotent
        # even if Rich's final render fails. ``Live.stop()`` normally restores
        # the cursor itself; the explicit final call covers failures before its
        # internal cursor-restoration block and makes Ctrl+C teardown fail-safe.
        self._live = None
        self._last_live_refresh_time = None
        try:
            live.update(self._build_display(), refresh=False)
        finally:
            try:
                live.stop()
            finally:
                self._console.show_cursor(True)

    def _close_backends(self) -> None:
        if self._tb_writer:
            self._tb_writer.close()
            self._tb_writer = None
        if self._wandb_run and self._owns_wandb_run:
            wandb = _load_wandb()
            if wandb is not None:
                wandb.finish()
            self._wandb_run = None
            self._owns_wandb_run = False

    def close(self) -> None:
        """Release live terminal state and backend handles without printing a summary."""
        if self._closed:
            return
        try:
            self._stop_live()
        finally:
            try:
                self._close_backends()
            finally:
                self._closed = True

    def finish(self, *, title: str = "Training Summary", extra_summary: str = ""):
        if self._finished:
            return
        self._stop_live()

        elapsed = time.time() - self._start_time
        if not self._no_print:
            summary = (
                f"[bold green]Training complete[/]\n"
                f"  Algo: [cyan]{self.algo_name}[/] | Env: [cyan]{self.env_name}[/]\n"
                f"  Iterations: [yellow]{self._iteration}[/]/{self.max_iterations}\n"
                f"  Total time: [yellow]{_fmt_time(elapsed)}[/]\n"
            )
            if extra_summary:
                summary += extra_summary
            if self._last_save:
                summary += f"  Last checkpoint: [dim]{self._last_save}[/]"

            self._console.print()
            self._console.print(Panel(summary, title=f"[bold]{title}[/]", border_style="green"))

        self._close_backends()
        self._closed = True
        self._finished = True

    def update_ep_length(self, length: float):
        self._mean_ep_length = length

    def log_save(self, path: str):
        self._last_save = path

    def _refresh(self, *, force: bool = False):
        if self._live is None:
            return
        if self._fixed_terminal_refresh and not force:
            return
        now = time.time()
        if not force and self._refresh_rate > 0 and self._last_live_refresh_time is not None:
            min_interval_s = 1.0 / self._refresh_rate
            if now - self._last_live_refresh_time < min_interval_s:
                return
        self._last_live_refresh_time = now
        self._live.update(self._build_display(), refresh=True)

    def _estimate_eta(self, *, iteration: int | None = None) -> str:
        current_iteration = self._iteration if iteration is None else iteration
        if current_iteration <= 0:
            return ""
        elapsed = time.time() - self._start_time
        remaining = self.max_iterations - current_iteration
        avg_iter = elapsed / current_iteration
        eta_s = remaining * avg_iter
        return _fmt_time(eta_s)

    def _build_compact_header(
        self,
        *,
        include_status: bool,
        include_identity: bool = True,
        include_iteration: bool = True,
        extra_fields: list[tuple[str, str]] | None = None,
    ) -> Text:
        return self._build_compact_header_state(
            include_status=include_status,
            include_identity=include_identity,
            include_iteration=include_iteration,
            extra_fields=extra_fields,
            ep_length=self._mean_ep_length,
            current_iteration=self._iteration,
        )

    def _build_compact_header_state(
        self,
        *,
        include_status: bool,
        include_identity: bool,
        include_iteration: bool,
        extra_fields: list[tuple[str, str]] | None,
        ep_length: float,
        current_iteration: int,
    ) -> Text:
        elapsed = time.time() - self._start_time if self._start_time else 0
        eta = self._estimate_eta(iteration=current_iteration)
        fields: list[tuple[str, str]] = []
        if include_identity:
            fields.extend(
                [
                    (f" {self.algo_name}", "bold cyan"),
                    (self.env_name, "bold white"),
                ]
            )
        if include_iteration:
            fields.append((f"iter {current_iteration}/{self.max_iterations}", "yellow"))
        fields.append(
            (
                f"{'⏱' if self._unicode_console else 'time'} {_fmt_time(elapsed)}",
                "green",
            )
        )
        if eta:
            fields.append((f"ETA {eta}", "bold magenta"))
        if ep_length > 0:
            fields.append((f"Ep Len {ep_length:.1f}", "yellow"))
        if extra_fields:
            fields.extend(extra_fields)
        if include_status and self._status:
            fields.append((self._status, "dim italic"))

        header_text = Text(no_wrap=True, overflow="ellipsis", justify="center")
        for index, (text, style) in enumerate(fields):
            if index > 0:
                header_text.append(" | ", style="dim")
            header_text.append(text, style=style)
        return header_text

    def _build_reward_table_common(
        self,
        *,
        wait_message: str,
        include_ep_length: bool = True,
        reward_history: tuple[float, ...] | None = None,
        reward_components: dict[str, float] | None = None,
        mean_reward: float | None = None,
    ) -> Table:
        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            show_edge=False,
            header_style="bold green",
            expand=True,
            pad_edge=False,
        )
        table.add_column("Rewards", style="white", width=31, no_wrap=True)
        table.add_column("Value", justify="right", ratio=2, no_wrap=True)

        recent = list(self._reward_history if reward_history is None else reward_history)
        if recent:
            mean_rew = (
                mean_reward
                if mean_reward is not None
                else sum(recent[-50:]) / max(len(recent[-50:]), 1)
            )
            peak_rew = max(recent) if recent else 0

            if len(recent) >= 10:
                old = sum(recent[-20:-10]) / 10
                new = sum(recent[-10:]) / 10
                if new > old * 1.05:
                    trend = "[green]▲[/]" if self._unicode_console else "[green]+[/]"
                elif new < old * 0.95:
                    trend = "[red]▼[/]" if self._unicode_console else "[red]-[/]"
                else:
                    trend = "[yellow]━[/]" if self._unicode_console else "[yellow]=[/]"
            else:
                trend = ""

            table.add_row(
                f"[bold]Reward[/] {trend}",
                f"Mean [bold green]{mean_rew:.3f}[/] / Peak [dim]{peak_rew:.3f}[/]",
            )
            if include_ep_length and self._mean_ep_length > 0:
                table.add_row("  Ep Len", f"[dim]{self._mean_ep_length:.1f}[/]")
            if include_ep_length:
                table.add_row("", "")
        else:
            table.add_row(wait_message, "")

        components = (
            self._latest_reward_components if reward_components is None else reward_components
        )
        if components:
            for name, val in sorted(components.items()):
                display = name.replace("reward/", "").replace("_", " ")
                color = "green" if val > 0 else "red" if val < 0 else "dim"
                table.add_row(f"  {display}", f"[{color}]{val:+.4f}[/]")

        return table

    def _build_display(self) -> Panel:
        raise NotImplementedError
