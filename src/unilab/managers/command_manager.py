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

from unilab.managers.manager_base import ManagerBase, ManagerTermBase
from unilab.managers.torch_rng import TorchManagerRng

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


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
        self._torch_rng = self._require_torch_rng()
        self.metrics: dict[str, torch.Tensor] = {}
        self.time_left = torch.zeros(self.num_envs, dtype=torch.float32, device=self._device)
        self.command_counter = torch.zeros(self.num_envs, dtype=torch.int64, device=self._device)
        self.last_reset_timing_ms: dict[str, float] = {}
        self.last_post_compute_timing_ms: dict[str, float] = {}
        self._dt_finite_valid = torch.ones((), dtype=torch.bool, device=self._device)
        self._command_finite_valid = torch.ones((), dtype=torch.bool, device=self._device)
        self._metrics_finite_valid = torch.ones((), dtype=torch.bool, device=self._device)
        self._deferred_validation_boundaries = False

    @property
    @abc.abstractmethod
    def command(self):
        raise NotImplementedError

    def reset(
        self,
        reset_mask: torch.Tensor,
        *,
        publish_metrics: bool = True,
    ) -> dict[str, float]:
        if (
            not isinstance(reset_mask, torch.Tensor)
            or reset_mask.shape != (self.num_envs,)
            or reset_mask.dtype != torch.bool
            or reset_mask.device != self._device
        ):
            raise ValueError(
                f"CommandTerm '{self.name}' reset mask must have shape "
                f"{(self.num_envs,)}, bool dtype, and device {self._device}"
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
                # Publish reset means and the finite diagnostic through one
                # fixed-width device boundary instead of materializing rows.
                values = torch.stack([value for _, value in metric_values], dim=0)
                selected_values = values * reset_mask[None, :]
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
                for (metric_name, metric_value), mean in zip(metric_values, means, strict=True):
                    extras[metric_name] = float(mean)
                    metric_value.masked_fill_(reset_mask, 0.0)
        else:
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
            valid = torch.isfinite(dt_tensor).all()
            self._dt_finite_valid.copy_(valid)
            dt_is_finite = self._deferred_validation_boundaries or bool(valid)
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
            if metric_value.ndim == 0 or metric_value.shape[0] != self.num_envs:
                raise ValueError(
                    f"CommandTerm '{self.name}' metric '{metric_name}' returned shape "
                    f"{metric_value.shape}, expected leading dimension {self.num_envs}."
                )
        if not metric_values:
            return
        non_tensor = [
            (name, value) for name, value in metric_values if not isinstance(value, torch.Tensor)
        ]
        if non_tensor:
            metric_name, _ = non_tensor[0]
            raise TypeError(
                f"CommandTerm '{self.name}' metric '{metric_name}' must be a torch.Tensor"
            )
        if metric_values:
            finite = torch.stack([torch.isfinite(value) for _, value in metric_values]).all()
            self._metrics_finite_valid.copy_(finite)
            if self._deferred_validation_boundaries:
                return
            if bool(finite):
                return
        for metric_name, metric_value in metric_values:
            if not _finite(metric_value):
                raise ValueError(
                    f"CommandTerm '{self.name}' metric '{metric_name}' contains NaN or Inf."
                )

    def _resample(self, reset_mask: torch.Tensor) -> None:
        lower, upper = self._resampling_time_range
        sampled = self._torch_rng.uniform(lower, upper, (self.num_envs,), dtype=torch.float32)
        self.time_left.copy_(torch.where(reset_mask, sampled, self.time_left))
        self._resample_command(reset_mask)
        self.command_counter += reset_mask.to(torch.int64)

    def _require_torch_rng(self) -> TorchManagerRng:
        rng = self._env.torch_rng
        if not isinstance(rng, TorchManagerRng):
            raise RuntimeError("CommandTerm scheduling requires the Manager-owned Torch generator")
        return rng

    @property
    def deferred_validation_boundaries(self) -> bool:
        """Whether finite verdict publication may defer to completion."""
        return self._deferred_validation_boundaries

    def set_deferred_validation_boundaries(self, enabled: bool) -> None:
        """Move finite publication to an explicit completion boundary."""
        if not isinstance(enabled, bool):
            raise TypeError(
                f"deferred command validation boundaries must be bool, got {type(enabled)}"
            )
        self._deferred_validation_boundaries = enabled

    def publish_validation_verdict(self) -> None:
        """Fail closed on deferred command and metric finite verdicts."""
        if not self._deferred_validation_boundaries:
            return
        if not bool(self._command_finite_valid) and bool(self._metrics_finite_valid):
            raise ValueError(f"CommandTerm '{self.name}' command contains NaN or Inf.")
        if not bool(self._metrics_finite_valid):
            for metric_name, metric_value in self.metrics.items():
                if not _finite(metric_value):
                    raise ValueError(
                        f"CommandTerm '{self.name}' metric '{metric_name}' contains NaN or Inf."
                    )

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
        ``EntityScene._tensor_read_plan``.
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
        self._last_term_reset_timing_ms: dict[str, float] = {}
        self.last_post_compute_timing_ms: dict[str, float] = {}
        self._deferred_validation_boundaries = False
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

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        terms = []
        for name, term in self._terms.items():
            command = self._validate_command(name, term.command)
            terms.append((name, command[env_idx].detach().cpu().tolist()))
        return terms

    @property
    def deferred_validation_boundaries(self) -> bool:
        """Whether command finite publication may defer to completion."""
        return self._deferred_validation_boundaries

    def set_deferred_validation_boundaries(self, enabled: bool) -> None:
        """Move finite publication to an explicit completion boundary."""
        if not isinstance(enabled, bool):
            raise TypeError(
                f"deferred command validation boundaries must be bool, got {type(enabled)}"
            )
        self._deferred_validation_boundaries = enabled
        for term in self._terms.values():
            term.set_deferred_validation_boundaries(enabled)

    def reset(self, reset_mask: torch.Tensor) -> dict[str, float]:
        """Full-width command reset.

        Every term must provide a mask-preserving owner implementation; a row
        fallback would invalidate the fixed-shape contract.
        """
        self._validate_reset_mask(reset_mask)
        extras: dict[str, float] = {}
        for name, term in self._terms.items():
            if not callable(getattr(term, "reset", None)):
                raise NotImplementedError(
                    f"CommandManager term '{name}' does not define the "
                    "mask-preserving reset boundary"
                )
            metrics = term.reset(reset_mask)
            for metric_name, metric_value in metrics.items():
                extras[f"Metrics/{name}/{metric_name}"] = metric_value
        self._last_term_reset_timing_ms.clear()
        for name, term in self._terms.items():
            self._last_term_reset_timing_ms.update(term.last_reset_timing_ms)
        return extras

    def _validate_reset_mask(self, reset_mask: torch.Tensor) -> None:
        if (
            not isinstance(reset_mask, torch.Tensor)
            or reset_mask.ndim != 1
            or reset_mask.shape[0] != self.num_envs
            or reset_mask.dtype != torch.bool
            or reset_mask.device != self._device
        ):
            raise ValueError(
                f"CommandManager reset mask must have shape {(self.num_envs,)}, "
                f"bool dtype, and device {self._device}"
            )

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

    def compute_reset(self, dt: float | torch.Tensor, reset_mask: torch.Tensor) -> None:
        """Fixed-shape command refresh after reset."""
        step_timing: dict[str, float] = {}
        for name, term in self._terms.items():
            callable_mask = getattr(term, "_update_command_reset", None)
            if not callable(callable_mask):
                raise NotImplementedError(
                    f"CommandManager term '{name}' does not define the "
                    "mask-preserving compute boundary"
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
                    f"Returned object for the term {term_name} is not of type CommandType."
                )
            self._terms[term_name] = term

    def _validate_command(self, name: str, command: torch.Tensor) -> torch.Tensor:
        self._validate_command_contract(name, command)
        valid = torch.isfinite(command).all()
        term = self._terms[name]
        term._command_finite_valid.copy_(valid)
        if term.deferred_validation_boundaries:
            return command
        if not bool(valid):
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

    def set_deferred_validation_boundaries(self, enabled: bool) -> None:
        del enabled

    def __str__(self) -> str:
        return "<NullCommandManager> (inactive)"

    def __repr__(self) -> str:
        return "NullCommandManager()"

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        return []

    def reset(self, reset_mask: torch.Tensor) -> dict[str, float]:
        del reset_mask
        return {}

    def reset_diagnostics(self) -> dict[str, float]:
        return {}

    def compute(self, dt: float | torch.Tensor) -> None:
        pass

    def compute_reset(self, dt: float | torch.Tensor, reset_mask: torch.Tensor) -> None:
        del dt, reset_mask

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
