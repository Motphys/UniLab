# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), src/mjlab/managers/command_manager.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and UniLab contracts; licensed under Apache-2.0.
"""Command manager for generating and updating commands."""

from __future__ import annotations

import abc
import inspect
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence, cast

import numpy as np
import torch
from prettytable import PrettyTable

from unilab.managers.manager_base import ManagerBase, ManagerTermBase

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


def _finite(values: np.ndarray | torch.Tensor) -> bool:
    if isinstance(values, torch.Tensor):
        return bool(torch.isfinite(values).all())
    return bool(np.isfinite(values).all())


def _mean(values: np.ndarray | torch.Tensor) -> float:
    if isinstance(values, torch.Tensor):
        return float(values.mean().item())
    return float(np.mean(values))


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
        if not np.isfinite((lower, upper)).all() or lower > upper:
            raise ValueError(
                f"CommandTerm '{self.name}' has invalid resampling_time_range "
                f"{cfg.resampling_time_range}."
            )
        self._resampling_time_range = (lower, upper)
        self._check_update_command_signature()
        self._device = torch.device(getattr(self._env, "device", torch.device("cpu")))
        self._torch_rng = getattr(self._env, "torch_rng", None)
        self.metrics: dict[str, np.ndarray | torch.Tensor] = {}
        self.time_left = torch.zeros(self.num_envs, dtype=torch.float32, device=self._device)
        self.command_counter = torch.zeros(self.num_envs, dtype=torch.int64, device=self._device)
        self.last_reset_timing_ms: dict[str, float] = {}
        self.last_post_compute_timing_ms: dict[str, float] = {}

    @property
    @abc.abstractmethod
    def command(self):
        raise NotImplementedError

    def reset(
        self,
        env_ids: torch.Tensor | slice | None,
        *,
        publish_metrics: bool = True,
    ) -> dict[str, float]:
        assert isinstance(env_ids, torch.Tensor)
        extras = {}
        metrics_started = time.perf_counter()
        if publish_metrics:
            metric_values = list(self.metrics.items())
            tensor_metrics = [
                (name, value) for name, value in metric_values if isinstance(value, torch.Tensor)
            ]
            if tensor_metrics and len(tensor_metrics) == len(metric_values):
                # Publish reset means and the finite diagnostic through one
                # device boundary instead of synchronizing twice per metric.
                selected = torch.stack([value[env_ids] for _, value in tensor_metrics], dim=0)
                reduced = torch.stack(
                    (
                        selected.mean(dim=1),
                        torch.isfinite(selected).all(dim=1).to(torch.float32),
                    ),
                    dim=0,
                ).detach()
                host_reduced = reduced.cpu()
                means = host_reduced[0].tolist()
                finite = bool(host_reduced[1].min().item() == 1.0)
                if not finite:
                    for metric_name, metric_slice in zip(
                        (name for name, _ in tensor_metrics), selected, strict=True
                    ):
                        if not _finite(metric_slice):
                            raise ValueError(
                                f"CommandTerm '{self.name}' metric '{metric_name}' contains "
                                "NaN or Inf."
                            )
                for (metric_name, metric_value), mean in zip(tensor_metrics, means, strict=True):
                    extras[metric_name] = float(mean)
                    metric_value[env_ids] = 0.0
            else:
                for metric_name, metric_value in metric_values:
                    metric_slice = metric_value[env_ids]
                    if not _finite(metric_slice):
                        raise ValueError(
                            f"CommandTerm '{self.name}' metric '{metric_name}' contains NaN or Inf."
                        )
                    extras[metric_name] = float(_mean(metric_slice))
                    metric_value[env_ids] = 0.0
        else:
            self.reset_last_episode_metrics(env_ids)
        metrics_ms = (time.perf_counter() - metrics_started) * 1000.0
        self.command_counter[env_ids] = 0
        self.last_reset_timing_ms.clear()
        resample_started = time.perf_counter()
        self._resample(env_ids)
        self.last_reset_timing_ms["reset_done_command_metrics_ms"] = metrics_ms
        self.last_reset_timing_ms["reset_done_command_resample_ms"] = (
            time.perf_counter() - resample_started
        ) * 1000.0
        return extras

    def reset_last_episode_metrics(self, env_ids: torch.Tensor) -> None:
        """Clear episode metric state without publishing reset means.

        Autoreset consumes per-transition episode logs on every vector step.
        Explicit reset is a control-plane boundary and may publish those logs;
        this method gives the caller that policy choice while retaining the
        same reset-state mutation.
        """
        for metric_value in self.metrics.values():
            if isinstance(metric_value, torch.Tensor):
                metric_value[env_ids] = 0.0
            else:
                metric_value[env_ids] = 0.0

    def compute(
        self, dt: float | np.ndarray | torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
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
        """
        tensor_dt = isinstance(dt, torch.Tensor)
        dt_scalar: float | None = None
        dt_tensor: torch.Tensor | None
        if not tensor_dt:
            host_dt = np.asarray(dt)
            if host_dt.ndim == 1:
                tensor_dt = True
                dt_tensor = torch.as_tensor(host_dt, dtype=torch.float32, device=self._device)
            else:
                tensor_dt = False
                dt_tensor = None
                dt_scalar = float(host_dt.item())
        else:
            dt_tensor = cast(torch.Tensor, dt)
        if dt_tensor is not None:
            if env_ids is not None:
                raise ValueError("Per-environment command dt requires env_ids=None.")
            if dt_tensor.shape != (self.num_envs,):
                raise ValueError(
                    f"CommandTerm '{self.name}' expected dt shape ({self.num_envs},), "
                    f"received {dt_tensor.shape}."
                )
        if tensor_dt:
            assert dt_tensor is not None
            dt_is_finite = bool(torch.isfinite(dt_tensor).all())
        else:
            host_dt = np.asarray(dt)
            assert host_dt.ndim == 0
            dt_scalar = float(host_dt.item())
            dt_is_finite = bool(np.isfinite(dt_scalar))
        if not dt_is_finite:
            raise ValueError(f"CommandTerm '{self.name}' received non-finite dt.")
        self._update_metrics(env_ids)
        self._validate_metrics()
        resample_env_ids: torch.Tensor
        if env_ids is None:
            if dt_tensor is not None:
                self.time_left -= dt_tensor
            else:
                assert dt_scalar is not None
                self.time_left -= dt_scalar
            resample_env_ids = torch.nonzero(self.time_left <= 0.0, as_tuple=False).flatten()
        else:
            assert not tensor_dt
            assert dt_scalar is not None
            self.time_left[env_ids] -= dt_scalar
            resample_env_ids = env_ids[self.time_left[env_ids] <= 0.0]
        if resample_env_ids.numel() > 0:
            self._resample(resample_env_ids)
        self._update_command(env_ids)

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
        tensor_metrics = [
            (name, value) for name, value in metric_values if isinstance(value, torch.Tensor)
        ]
        if len(tensor_metrics) == len(metric_values):
            finite = torch.stack([torch.isfinite(value) for _, value in tensor_metrics]).all()
            if bool(finite):
                return
        for metric_name, metric_value in metric_values:
            if not _finite(metric_value):
                raise ValueError(
                    f"CommandTerm '{self.name}' metric '{metric_name}' contains NaN or Inf."
                )

    def _check_update_command_signature(self) -> None:
        """Fail fast with a migration hint for terms with the old signature."""
        try:
            sig = inspect.signature(self._update_command)
        except (TypeError, ValueError):
            return
        if len(sig.parameters) == 0:
            raise TypeError(
                f"{type(self).__name__}._update_command must accept env_ids: "
                "_update_command(self, env_ids: torch.Tensor | None). It receives "
                "None on the per-step update and the reset env ids on reset(); "
                "scope per-step state advances to env_ids."
            )

    def _resample(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() != 0:
            lower, upper = self._resampling_time_range
            if self._torch_rng is not None:
                sampled = self._torch_rng.uniform(
                    lower, upper, (env_ids.numel(),), dtype=torch.float32
                )
            else:
                sampled = torch.as_tensor(
                    self._env.rng.uniform(lower, upper, env_ids.numel()),
                    dtype=torch.float32,
                    device=self._device,
                )
            self.time_left[env_ids] = sampled
            self._resample_command(env_ids)
            self.command_counter[env_ids] += 1

    @abc.abstractmethod
    def _update_metrics(self, env_ids: torch.Tensor | None = None) -> None:
        """Update the metrics based on the current state.

        env_ids is None on the per-step update (all envs) and the reset env ids on
        the reset path. Terms may scope per-row metric work to env_ids; rows outside
        env_ids are unchanged since the last per-step update and stay valid.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Resample the command for the specified environments."""
        raise NotImplementedError

    @abc.abstractmethod
    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        """Update the command based on the current state.

        env_ids is None on the per-step update (all envs) and the reset env ids on reset().
        Scope per-step state advances (e.g. a motion frame index) to env_ids; pure
        functions of the current state may ignore it.
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
            command = self._validate_command(name, term.command).detach().cpu().numpy()
            terms.append((name, command[env_idx].tolist()))
        return terms

    def reset(
        self,
        env_ids: torch.Tensor | slice | None,
        *,
        publish_metrics: bool = True,
    ) -> dict[str, float]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self._device)[env_ids]
        extras = {}
        reset_commands: list[tuple[str, torch.Tensor]] = []
        self._last_term_reset_timing_ms.clear()
        for name, term in self._terms.items():
            metrics = term.reset(env_ids=env_ids, publish_metrics=publish_metrics)
            reset_commands.append((name, term.command))
            self._last_term_reset_timing_ms.update(term.last_reset_timing_ms)
            for metric_name, metric_value in metrics.items():
                extras[f"Metrics/{name}/{metric_name}"] = metric_value
        validation_started = time.perf_counter()
        commands: list[torch.Tensor] = []
        labels: list[str] = []
        for name, command in reset_commands:
            if isinstance(command, np.ndarray):
                command = torch.from_numpy(np.ascontiguousarray(command)).to(
                    dtype=torch.float32, device=self._device
                )
            self._validate_command_contract(name, command)
            commands.append(command)
            labels.append(name)
        if commands and not bool(
            torch.isfinite(
                torch.cat(tuple(value.reshape(value.shape[0], -1) for value in commands), dim=1)
            ).all()
        ):
            for name, command in zip(labels, commands, strict=True):
                if not bool(torch.isfinite(command).all()):
                    raise ValueError(f"CommandManager term '{name}' returned NaN or Inf.")
        self._last_term_reset_timing_ms["reset_done_reset_validation_ms"] = (
            time.perf_counter() - validation_started
        ) * 1000.0
        return extras

    def reset_last_episode_metrics(self, env_ids: torch.Tensor) -> None:
        """Clear each term's episode metrics without host metric reduction."""
        for term in self._terms.values():
            term.reset_last_episode_metrics(env_ids)

    def compute(
        self, dt: float | np.ndarray | torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        for name, term in self._terms.items():
            term.compute(dt, env_ids)
            self._validate_command(name, term.command)

    def post_compute(self) -> None:
        self.last_post_compute_timing_ms.clear()
        for term in self._terms.values():
            term.post_compute()
            self.last_post_compute_timing_ms.update(
                getattr(term, "last_post_compute_timing_ms", {})
            )

    def bind_read_phase(self) -> None:
        for term in self._terms.values():
            term.bind_read_phase()

    def uses_tensor_reset_rows(self) -> bool:
        """Whether every command can consume a device row selector on reset."""
        return bool(self._terms) and all(
            bool(getattr(term, "uses_tensor_reset_rows", False)) for term in self._terms.values()
        )

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
        if isinstance(command, np.ndarray):
            command = torch.from_numpy(np.ascontiguousarray(command)).to(
                dtype=torch.float32, device=self._device
            )
        self._validate_command_contract(name, command)
        if not bool(torch.isfinite(command).all()):
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

    def reset(
        self,
        env_ids: torch.Tensor | None = None,
        *,
        publish_metrics: bool = True,
    ) -> dict[str, np.ndarray]:
        del publish_metrics
        return {}

    def compute(
        self, dt: float | np.ndarray | torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        pass

    def post_compute(self) -> None:
        pass

    def bind_read_phase(self) -> None:
        pass

    def uses_tensor_reset_rows(self) -> bool:
        return False

    def get_command(self, name: str) -> None:
        return None

    def get_term(self, name: str) -> None:
        return None

    def get_term_cfg(self, name: str) -> None:
        return None
