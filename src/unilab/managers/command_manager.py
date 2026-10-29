# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), src/mjlab/managers/command_manager.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and UniLab contracts; licensed under Apache-2.0.
"""Command manager for generating and updating commands."""

from __future__ import annotations

import abc
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence, cast

import torch
from prettytable import PrettyTable

from unilab.base.reset_selectors import ResetMaskBuffer, resolve_reset_env_ids
from unilab.managers.manager_base import ManagerBase, ManagerTermBase
from unilab.managers.rng import ManagerRng

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv


def _finite(values: torch.Tensor) -> bool:
    return bool(torch.isfinite(values).all())


def _mean(values: torch.Tensor) -> float:
    return float(values.mean().item())


@dataclass(kw_only=True)
class CommandTermCfg(abc.ABC):
    """Configuration for a command generator term.

    Command terms generate goal commands for the agent (e.g., target velocity,
    target position). Commands are automatically resampled at configurable
    intervals and can track metrics for logging.
    """

    resampling_time_range: tuple[float, float]
    """Time range in seconds for command resampling. When the timer expires, a new
  command is sampled and the timer is reset to a value uniformly drawn from
  ``[min, max]``. Set both values equal for fixed-interval resampling."""

    debug_vis: bool = False
    """Whether to enable debug visualization for this command term. When True,
  the command term's ``_debug_vis_impl`` method is called each frame to render
  visual aids (e.g., velocity arrows, target markers)."""

    @abc.abstractmethod
    def build(self, env: ManagerBasedRlEnv) -> CommandTerm:
        """Build the command term from this config."""
        raise NotImplementedError


class CommandTerm(ManagerTermBase):
    """Base class for command terms."""

    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedRlEnv):
        self.cfg = cfg
        super().__init__(env)
        lower, upper = cfg.resampling_time_range
        if not (math.isfinite(lower) and math.isfinite(upper)) or lower > upper:
            raise ValueError(
                f"CommandTerm '{self.name}' has invalid resampling_time_range "
                f"{cfg.resampling_time_range}."
            )
        self._resampling_time_range = (lower, upper)
        self._device = torch.device(getattr(self._env, "device", torch.device("cpu")))
        self._rng = self._require_rng()
        # Per-term reusable ids→mask adapter (roadmap #2461). Each term owns
        # its instance, so terms dispatched in sequence never clobber a mask
        # another term still holds.
        self._reset_mask_buffer = ResetMaskBuffer(self.num_envs, self._device)
        self.metrics: dict[str, torch.Tensor] = {}
        self._pending_metric_sums: dict[str, torch.Tensor] = {}
        self._pending_metric_count = torch.zeros((), dtype=torch.int64, device=self._device)
        self._pending_metric_dirty = False
        self.time_left = torch.zeros(self.num_envs, dtype=torch.float32, device=self._device)
        self.command_counter = torch.zeros(self.num_envs, dtype=torch.int64, device=self._device)
        self.last_reset_timing_ms: dict[str, float] = {}
        self.last_post_compute_timing_ms: dict[str, float] = {}
        self._command_finite_valid = torch.ones((), dtype=torch.bool, device=self._device)

    @property
    @abc.abstractmethod
    def command(self):
        raise NotImplementedError

    def reset(
        self,
        env_ids: torch.Tensor | slice | None = None,
        *,
        publish_metrics: bool = True,
    ) -> dict[str, float]:
        reset_mask = self._reset_mask_buffer.resolve(
            env_ids,
            owner=f"CommandTerm '{self.name}' reset",
        )
        extras: dict[str, float] = {}
        metrics_started = time.perf_counter()
        metric_values = list(self.metrics.items())
        if publish_metrics:
            tensor_metrics = [
                (name, value)
                for name, value in metric_values
                if not isinstance(value, torch.Tensor)
            ]
            if tensor_metrics:
                metric_name, _ = tensor_metrics[0]
                raise TypeError(
                    f"CommandTerm '{self.name}' metric '{metric_name}' must be a torch.Tensor"
                )
            if metric_values:
                # Publish reset means and the finite check through one
                # fixed-width device boundary instead of materializing rows.
                values = torch.stack([value for _, value in metric_values], dim=0)
                selected_values = torch.where(reset_mask[None, :], values, 0.0)
                if self._deferred_validation_enabled():
                    self._record_metric_finite(metric_values, reset_mask)
                    runtime_reset = int(getattr(self._env, "step_counter", 0)) > 0 or bool(
                        getattr(self._env, "_has_transition", False)
                    )
                    selected_rows = not (isinstance(env_ids, torch.Tensor) and env_ids.numel() == 0)
                    if isinstance(env_ids, slice):
                        selected_rows = bool(range(*env_ids.indices(self.num_envs)))
                    if runtime_reset and selected_rows:
                        self._pending_metric_count += reset_mask.sum()
                        for (metric_name, _), selected in zip(
                            metric_values, selected_values, strict=True
                        ):
                            pending = self._pending_metric_sums.get(metric_name)
                            if pending is None:
                                pending = torch.zeros((), dtype=torch.float32, device=self._device)
                                self._pending_metric_sums[metric_name] = pending
                            pending.add_(selected.sum())
                        self._pending_metric_dirty = True
                else:
                    reduced = torch.stack(
                        (
                            selected_values.sum(dim=1) / reset_mask.sum().clamp_min(1),
                            (torch.isfinite(values) | ~reset_mask[None, :])
                            .all(dim=1)
                            .to(torch.float32),
                        ),
                        dim=0,
                    ).detach()
                    host_reduced = reduced.cpu()
                    means = host_reduced[0].tolist()
                    finite = bool(host_reduced[1].min().item() == 1.0)
                    if not finite:
                        for metric_name, metric_value in metric_values:
                            if not bool((torch.isfinite(metric_value) | ~reset_mask).all()):
                                raise ValueError(
                                    f"CommandTerm '{self.name}' metric '{metric_name}' contains "
                                    "NaN or Inf."
                                )
                    for (metric_name, _), mean in zip(metric_values, means, strict=True):
                        extras[metric_name] = float(mean)
                for _, metric_value in metric_values:
                    metric_value.masked_fill_(reset_mask, 0.0)
        else:
            if self._deferred_validation_enabled():
                self._record_metric_finite(metric_values, reset_mask)
            self.clear_episode_metrics(reset_mask)
        metrics_ms = (time.perf_counter() - metrics_started) * 1000.0
        self.command_counter.masked_fill_(reset_mask, 0)
        self.last_reset_timing_ms.clear()
        resample_started = time.perf_counter()
        self._resample(reset_mask)
        self.last_reset_timing_ms["reset_done_command_metrics_ms"] = metrics_ms
        self.last_reset_timing_ms["reset_done_command_resample_ms"] = (
            time.perf_counter() - resample_started
        ) * 1000.0
        return extras

    def _deferred_validation_enabled(self) -> bool:
        return bool(getattr(self._env, "defer_finite_validation", False))

    def publish_pending_reset_metrics(self) -> dict[str, float]:
        """Drain completed episodes only at the environment's host boundary."""
        if not self._pending_metric_dirty:
            return {}
        names = list(self._pending_metric_sums)
        divisor = self._pending_metric_count.clamp_min(1)
        means = (
            torch.stack([self._pending_metric_sums[name] / divisor for name in names])
            .detach()
            .cpu()
            .tolist()
        )
        for pending in self._pending_metric_sums.values():
            pending.zero_()
        self._pending_metric_count.zero_()
        self._pending_metric_dirty = False
        return {name: float(value) for name, value in zip(names, means, strict=True)}

    def _record_metric_finite(
        self,
        metric_values: list[tuple[str, torch.Tensor]],
        reset_mask: torch.Tensor,
    ) -> None:
        record = getattr(self._env, "record_finite_validation", None)
        for metric_name, metric_value in metric_values:
            selected = torch.where(reset_mask, metric_value, 0.0)
            if callable(record):
                record(selected, f"command/{self.name}/metric/{metric_name}")
            elif not _finite(selected):
                raise ValueError(
                    f"CommandTerm '{self.name}' metric '{metric_name}' contains NaN or Inf."
                )

    def clear_episode_metrics(self, reset_mask: torch.Tensor) -> None:
        """Clear command metrics without a host metric reduction."""
        for metric_value in self.metrics.values():
            if not isinstance(metric_value, torch.Tensor):
                raise TypeError("CommandTerm metrics must be torch.Tensor carriers")
            metric_value.masked_fill_(reset_mask, 0.0)

    def compute(self, dt: float | torch.Tensor) -> None:
        """Advance the command state by dt.

        With env_ids=None (the per-step path) all envs are updated; with env_ids
        (the reset path) timers and the command update are scoped to those envs.
        Metrics are refreshed each call; terms may scope per-row metric work to
        env_ids since other rows are unchanged since the per-step update, or
        defer row-wise metric work to ``reset()`` when reset is the only
        consumer (e.g. MotionCommand, issue #1355).

        dt may be a scalar (all envs) or a per-env tensor (auto-reset path,
        where freshly reset envs get zero to keep their timers full). A tensor
        dt requires env_ids=None.

        Full-command and metric finite diagnostics run only on the per-step
        full-batch path. The reset refresh is row-scoped; the immediately
        preceding command reset or reset-owner transaction remains the
        authoritative validation boundary for freshly reset rows.
        """
        timing = getattr(self, "last_step_timing_ms", None)
        if timing is None:
            timing = {}
            self.last_step_timing_ms = timing
        dt_started = time.perf_counter()
        tensor_dt = isinstance(dt, torch.Tensor)
        dt_scalar: float | None = None
        dt_tensor: torch.Tensor | None
        if not tensor_dt:
            if isinstance(dt, bool) or not isinstance(dt, (int, float)):
                raise TypeError("CommandTerm dt must be a scalar or torch.Tensor")
            dt_scalar = float(dt)
            dt_tensor = None
        else:
            dt_tensor = cast(torch.Tensor, dt)
        if dt_tensor is not None:
            if dt_tensor.shape != (self.num_envs,):
                raise ValueError(
                    f"CommandTerm '{self.name}' expected dt shape ({self.num_envs},), "
                    f"received {dt_tensor.shape}."
                )
        if tensor_dt:
            assert dt_tensor is not None
            if self._deferred_validation_enabled():
                record = getattr(self._env, "record_finite_validation", None)
                if callable(record):
                    record(dt_tensor, f"command/{self.name}/dt")
                    dt_is_finite = True
                else:
                    dt_is_finite = bool(torch.isfinite(dt_tensor).all())
            else:
                dt_is_finite = bool(torch.isfinite(dt_tensor).all())
        else:
            assert dt_scalar is not None
            dt_is_finite = math.isfinite(dt_scalar)
        if not dt_is_finite:
            raise ValueError(f"CommandTerm '{self.name}' received non-finite dt.")
        timing["update_state_command_dt_validation_ms"] = (
            time.perf_counter() - dt_started
        ) * 1000.0
        metrics_started = time.perf_counter()
        self._update_metrics()
        timing["update_state_command_metrics_update_ms"] = (
            time.perf_counter() - metrics_started
        ) * 1000.0
        timer_started = time.perf_counter()
        if dt_tensor is not None:
            self.time_left -= dt_tensor
        else:
            assert dt_scalar is not None
            self.time_left -= dt_scalar
        self._resample(self.time_left <= 0.0)
        timing["update_state_command_timer_ms"] = (time.perf_counter() - timer_started) * 1000.0
        update_started = time.perf_counter()
        self._update_command()
        # The owner records detailed update-command phases in the same dict;
        # total update time is kept only when this base method created it.
        if "update_state_motion_failure_stats_ms" not in timing:
            timing["update_state_command_update_dispatch_ms"] = (
                time.perf_counter() - update_started
            ) * 1000.0

    def _validate_metrics(self) -> None:
        metric_values = list(self.metrics.items())
        for metric_name, metric_value in metric_values:
            if not isinstance(metric_value, torch.Tensor):
                raise TypeError(
                    f"CommandTerm '{self.name}' metric '{metric_name}' must be a torch.Tensor"
                )
            if metric_value.ndim == 0 or metric_value.shape[0] != self.num_envs:
                raise ValueError(
                    f"CommandTerm '{self.name}' metric '{metric_name}' returned shape "
                    f"{metric_value.shape}, expected leading dimension {self.num_envs}."
                )
        if not metric_values:
            return
        if self._deferred_validation_enabled():
            record = getattr(self._env, "record_finite_validation", None)
            if callable(record):
                for metric_name, metric_value in metric_values:
                    record(metric_value, f"command/{self.name}/metric/{metric_name}")
                return
        if metric_values:
            finite = torch.stack([torch.isfinite(value) for _, value in metric_values]).all()
            if bool(finite):
                return
        for metric_name, metric_value in metric_values:
            if not _finite(metric_value):
                raise ValueError(
                    f"CommandTerm '{self.name}' metric '{metric_name}' contains NaN or Inf."
                )

    def _resample(self, reset_mask: torch.Tensor) -> None:
        lower, upper = self._resampling_time_range
        sampled = self._rng.uniform(lower, upper, (self.num_envs,), dtype=torch.float32)
        self.time_left.copy_(torch.where(reset_mask, sampled, self.time_left))
        self._resample_command(reset_mask)
        self.command_counter += reset_mask.to(torch.int64)

    def _require_rng(self) -> ManagerRng:
        rng = self._env.rng
        if not isinstance(rng, ManagerRng):
            raise RuntimeError("CommandTerm scheduling requires the Manager-owned Torch generator")
        return rng

    @abc.abstractmethod
    def _update_metrics(self) -> None:
        """Update full-width metrics based on the current state."""
        raise NotImplementedError

    @abc.abstractmethod
    def _resample_command(self, reset_mask: torch.Tensor) -> None:
        """Resample the command while preserving rows outside the mask."""
        raise NotImplementedError

    @abc.abstractmethod
    def _update_command(self) -> None:
        """Update the command based on the current state.

        Advance per-step state with a fixed-shape carrier; the reset lifecycle
        uses the explicit mask boundary.
        """
        raise NotImplementedError

    def post_compute(self) -> None:
        """Refresh state that depends on committed command-side simulation writes."""

    def bind_read_phase(self) -> None:
        """Bind command state that requires the scene's tensor read phase.

        The default command owns no scene reads. Device-resident terms may
        override this hook to obtain stable public views after the Manager has
        compiled its packed read plan, avoiding construction-order coupling to
        ``EntityScene.read_plan``.
        """


class CommandManager(ManagerBase):
    """Manages command generation for the environment.

    The command manager generates and updates goal commands for the agent (e.g.,
    target velocity, target position). Commands are resampled at configurable
    intervals and can track metrics for logging.
    """

    _env: ManagerBasedRlEnv

    def __init__(self, cfg: dict[str, CommandTermCfg | None], env: ManagerBasedRlEnv):
        self._terms: dict[str, CommandTerm] = dict()

        self.cfg = cfg
        self._device = torch.device(getattr(env, "device", torch.device("cpu")))
        env_cfg = getattr(env, "cfg", None)
        self._defer_finite_validation = bool(
            getattr(
                env, "defer_finite_validation", getattr(env_cfg, "defer_finite_validation", False)
            )
        )
        self._finite_verdict = torch.ones((), dtype=torch.bool, device=self._device)
        self._last_term_reset_timing_ms: dict[str, float] = {}
        self.last_post_compute_timing_ms: dict[str, float] = {}
        super().__init__(env)

    def __str__(self) -> str:
        msg = f"<CommandManager> contains {len(self._terms.values())} active terms.\n"
        table = PrettyTable()
        table.title = "Active Command Terms"
        table.field_names = ["Index", "Name", "Type"]
        table.align["Name"] = "l"
        for index, (name, term) in enumerate(self._terms.items()):
            table.add_row([index, name, term.__class__.__name__])
        msg += str(table.get_string())
        msg += "\n"
        return msg

    # Properties.

    @property
    def active_terms(self) -> list[str]:
        return list(self._terms.keys())

    @property
    def last_reset_timing_ms(self) -> dict[str, float]:
        return dict(self._last_term_reset_timing_ms)

    @property
    def finite_verdict(self) -> torch.Tensor:
        """Device-resident command verdict accumulated since the last poll."""
        return self._finite_verdict

    def reset_finite_verdict(self) -> None:
        self._finite_verdict.fill_(True)

    def poll_finite_validation(self) -> bool:
        """Raise a command error when the deferred boundary finds invalid data."""
        valid = bool(self._finite_verdict.detach().cpu().item())
        self.reset_finite_verdict()
        if valid:
            return True
        for name, term in self._terms.items():
            command = term.command
            finite = torch.isfinite(command)
            if not bool(finite.all()):
                rows = ~finite.reshape(self.num_envs, -1).all(dim=1)
                invalid_rows = torch.nonzero(rows, as_tuple=False).flatten()[:10]
                raise ValueError(
                    f"CommandManager term '{name}' returned NaN or Inf for "
                    f"environments {invalid_rows.detach().cpu().tolist()}."
                )
        raise ValueError("CommandManager returned a non-finite command.")

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        terms = []
        for name, term in self._terms.items():
            command = self._validate_command(name, term.command)
            terms.append((name, command[env_idx].detach().cpu().tolist()))
        return terms

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
        """Reset command terms for the selected environment rows.

        Every term consumes the public ``env_ids`` row selector; the packed
        internals keep their fixed-width mask form behind the selector
        conversion.
        """
        ids = resolve_reset_env_ids(
            env_ids, num_envs=self.num_envs, device=self._device, owner="CommandManager.reset"
        )
        extras: dict[str, float] = {}
        for name, term in self._terms.items():
            metrics = term.reset(ids)
            for metric_name, metric_value in metrics.items():
                extras[f"Metrics/{name}/{metric_name}"] = metric_value
        self._last_term_reset_timing_ms.clear()
        for name, term in self._terms.items():
            self._last_term_reset_timing_ms.update(term.last_reset_timing_ms)
        return extras

    def publish_pending_reset_metrics(self) -> dict[str, float]:
        """Drain deferred command metrics after the shared validation poll."""
        extras: dict[str, float] = {}
        for name, term in self._terms.items():
            for metric_name, metric_value in term.publish_pending_reset_metrics().items():
                extras[f"Metrics/{name}/{metric_name}"] = metric_value
        return extras

    def compute(self, dt: float | torch.Tensor) -> None:
        step_timing: dict[str, float] = {}
        dispatch_ms = 0.0
        validation_ms = 0.0
        for name, term in self._terms.items():
            term_timing = getattr(term, "last_step_timing_ms", None)
            if term_timing is not None:
                term_timing.clear()
            dispatch_started = time.perf_counter()
            term.compute(dt)
            dispatch_ms += time.perf_counter() - dispatch_started
            step_timing.update(getattr(term, "last_step_timing_ms", {}))
            validation_started = time.perf_counter()
            self._validate_command(name, term.command)
            term._validate_metrics()
            validation_ms += time.perf_counter() - validation_started
        step_timing["update_state_command_term_dispatch_ms"] = dispatch_ms * 1000.0
        step_timing["update_state_command_validation_ms"] = validation_ms * 1000.0
        self.last_step_timing_ms = step_timing

    def refresh_reset(self, reset_mask: torch.Tensor) -> None:
        """Fixed-shape command refresh after reset."""
        step_timing: dict[str, float] = {}
        for name, term in self._terms.items():
            callable_mask = getattr(term, "refresh_reset", None)
            if not callable(callable_mask):
                raise NotImplementedError(
                    f"CommandManager term '{name}' does not define the "
                    "mask-preserving reset refresh boundary"
                )
            term_timing = getattr(term, "last_step_timing_ms", None)
            if term_timing is not None:
                term_timing.clear()
            started = time.perf_counter()
            callable_mask(reset_mask)
            step_timing.update(getattr(term, "last_step_timing_ms", {}))
            step_timing["update_state_command_term_dispatch_ms"] = (
                time.perf_counter() - started
            ) * 1000.0
        self.last_step_timing_ms = step_timing

    def post_compute(self) -> None:
        self.last_post_compute_timing_ms.clear()
        for term in self._terms.values():
            term.post_compute()
            self.last_post_compute_timing_ms.update(
                getattr(term, "last_post_compute_timing_ms", {})
            )

    def post_compute_reset(self) -> None:
        """Fixed-shape command post-compute after reset."""
        self.last_post_compute_timing_ms.clear()
        for name, term in self._terms.items():
            callable_mask = getattr(term, "post_compute_reset", None)
            if not callable(callable_mask):
                raise NotImplementedError(
                    f"CommandManager term '{name}' does not define the "
                    "mask-preserving post-compute boundary"
                )
            callable_mask()
            self.last_post_compute_timing_ms.update(
                getattr(term, "last_post_compute_timing_ms", {})
            )

    def bind_read_phase(self) -> None:
        for term in self._terms.values():
            term.bind_read_phase()

    def reset_diagnostics(self) -> dict[str, float]:
        """Collect optional term-owned reset call-graph diagnostics."""
        diagnostics: dict[str, float] = {}
        for term in self._terms.values():
            collector = getattr(term, "sampler_reset_diagnostics", None)
            if callable(collector):
                diagnostics.update(cast("dict[str, float]", collector()))
        return diagnostics

    def get_command(self, name: str) -> torch.Tensor:
        return self._validate_command(name, self._terms[name].command)

    def get_term(self, name: str) -> CommandTerm:
        return self._terms[name]

    def get_term_cfg(self, name: str) -> CommandTermCfg:
        term_cfg = self.cfg[name]
        if term_cfg is None:
            raise KeyError(f"Command term '{name}' is disabled.")
        return term_cfg

    def _prepare_terms(self) -> None:
        for term_name, term_cfg in self.cfg.items():
            if term_cfg is None:
                print(f"term: {term_name} set to None, skipping...")
                continue
            if term_cfg.debug_vis:
                raise NotImplementedError(
                    f"CommandManager term '{term_name}' requested viewer debug visualization; "
                    "viewer glue is unsupported by the UniLab manager core."
                )
            term = term_cfg.build(self._env)
            if not isinstance(term, CommandTerm):
                raise TypeError(
                    f"Returned object for the term {term_name} is not of type CommandTerm."
                )
            self._terms[term_name] = term

    def _validate_command(self, name: str, command: torch.Tensor) -> torch.Tensor:
        self._validate_command_contract(name, command)
        valid = torch.isfinite(command).all()
        term = self._terms[name]
        term._command_finite_valid.copy_(valid)
        if self._defer_finite_validation:
            self._finite_verdict.logical_and_(valid)
        elif not bool(valid):
            raise ValueError(f"CommandManager term '{name}' returned NaN or Inf.")
        return command

    def _validate_command_contract(self, name: str, command: torch.Tensor) -> None:
        if not isinstance(command, torch.Tensor):
            raise TypeError(
                f"CommandManager term '{name}' returned {type(command).__name__}, "
                "expected torch.Tensor."
            )
        if command.ndim < 1 or command.shape[0] != self.num_envs:
            raise ValueError(
                f"CommandManager term '{name}' returned shape {command.shape}, "
                f"expected leading dimension {self.num_envs}."
            )


class NullCommandManager:
    """Placeholder for absent command manager that safely no-ops all operations."""

    def __init__(self):
        self.active_terms: list[str] = []
        self._terms: dict[str, Any] = {}
        self.cfg = None

    def __str__(self) -> str:
        return "<NullCommandManager> (inactive)"

    def __repr__(self) -> str:
        return "NullCommandManager()"

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        return []

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
        del env_ids
        return {}

    def publish_pending_reset_metrics(self) -> dict[str, float]:
        return {}

    def reset_diagnostics(self) -> dict[str, float]:
        return {}

    def compute(self, dt: float | torch.Tensor) -> None:
        pass

    def refresh_reset(self, reset_mask: torch.Tensor) -> None:
        del reset_mask

    def post_compute(self) -> None:
        pass

    def post_compute_reset(self) -> None:
        pass

    def bind_read_phase(self) -> None:
        pass

    def get_command(self, name: str) -> None:
        return None

    def get_term(self, name: str) -> None:
        return None

    def get_term_cfg(self, name: str) -> None:
        return None
