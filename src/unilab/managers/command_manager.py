# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), src/mjlab/managers/command_manager.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and UniLab contracts; licensed under Apache-2.0.
"""Command manager for generating and updating commands."""

from __future__ import annotations

import abc
import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from prettytable import PrettyTable

from unilab.managers.manager_base import ManagerBase, ManagerTermBase

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


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

    updates_before_reward: ClassVar[bool] = False

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
        self.metrics: dict[str, np.ndarray] = {}
        self.time_left = np.zeros(self.num_envs, dtype=np.float32)
        self.command_counter = np.zeros(self.num_envs, dtype=np.int64)

    @property
    @abc.abstractmethod
    def command(self):
        raise NotImplementedError

    def reset(self, env_ids: np.ndarray | slice | None) -> dict[str, float]:
        assert isinstance(env_ids, np.ndarray)
        extras = {}
        for metric_name, metric_value in self.metrics.items():
            metric_slice = metric_value[env_ids]
            if not np.isfinite(metric_slice).all():
                raise ValueError(
                    f"CommandTerm '{self.name}' metric '{metric_name}' contains NaN or Inf."
                )
            extras[metric_name] = float(np.mean(metric_slice))
            metric_value[env_ids] = 0.0
        self.command_counter[env_ids] = 0
        self._resample(env_ids)
        return extras

    def compute(self, dt: float | np.ndarray, env_ids: np.ndarray | None = None) -> None:
        """Advance the command state by dt.

        With env_ids=None (the per-step path) all envs are updated; with env_ids
        (the reset path) timers and the command update are scoped to those envs.
        Metrics are refreshed each call; terms may scope per-row metric work to
        env_ids since other rows are unchanged since the per-step update.

        dt may be a scalar (all envs) or a per-env tensor (auto-reset path,
        where freshly reset envs get zero to keep their timers full). A tensor
        dt requires env_ids=None.
        """
        if isinstance(dt, np.ndarray):
            if env_ids is not None:
                raise ValueError("Per-environment command dt requires env_ids=None.")
            if dt.shape != (self.num_envs,):
                raise ValueError(
                    f"CommandTerm '{self.name}' expected dt shape ({self.num_envs},), "
                    f"received {dt.shape}."
                )
        dt_is_finite = np.isfinite(dt).all() if isinstance(dt, np.ndarray) else np.isfinite(dt)
        if not dt_is_finite:
            raise ValueError(f"CommandTerm '{self.name}' received non-finite dt.")
        self._update_metrics(env_ids)
        self._validate_metrics()
        if env_ids is None:
            self.time_left -= dt
            resample_env_ids = np.flatnonzero(self.time_left <= 0.0)
        else:
            assert not isinstance(dt, np.ndarray)
            self.time_left[env_ids] -= dt
            resample_env_ids = env_ids[self.time_left[env_ids] <= 0.0]
        if len(resample_env_ids) > 0:
            self._resample(resample_env_ids)
        self._update_command(env_ids)

    def _validate_metrics(self) -> None:
        for metric_name, metric_value in self.metrics.items():
            if not isinstance(metric_value, np.ndarray):
                raise TypeError(
                    f"CommandTerm '{self.name}' metric '{metric_name}' returned "
                    f"{type(metric_value).__name__}, expected np.ndarray."
                )
            if metric_value.ndim == 0 or metric_value.shape[0] != self.num_envs:
                raise ValueError(
                    f"CommandTerm '{self.name}' metric '{metric_name}' returned shape "
                    f"{metric_value.shape}, expected leading dimension {self.num_envs}."
                )
            if not np.isfinite(metric_value).all():
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
                "_update_command(self, env_ids: np.ndarray | None). It receives "
                "None on the per-step update and the reset env ids on reset(); "
                "scope per-step state advances to env_ids."
            )

    def _resample(self, env_ids: np.ndarray) -> None:
        if len(env_ids) != 0:
            lower, upper = self._resampling_time_range
            self.time_left[env_ids] = self._env.rng.uniform(lower, upper, len(env_ids))
            self._resample_command(env_ids)
            self.command_counter[env_ids] += 1

    @abc.abstractmethod
    def _update_metrics(self, env_ids: np.ndarray | None = None) -> None:
        """Update the metrics based on the current state.

        env_ids is None on the per-step update (all envs) and the reset env ids on
        the reset path. Terms may scope per-row metric work to env_ids; rows outside
        env_ids are unchanged since the last per-step update and stay valid.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _resample_command(self, env_ids: np.ndarray) -> None:
        """Resample the command for the specified environments."""
        raise NotImplementedError

    @abc.abstractmethod
    def _update_command(self, env_ids: np.ndarray | None) -> None:
        """Update the command based on the current state.

        env_ids is None on the per-step update (all envs) and the reset env ids on reset().
        Scope per-step state advances (e.g. a motion frame index) to env_ids; pure
        functions of the current state may ignore it.
        """
        raise NotImplementedError

    def post_compute(self) -> None:
        """Refresh state that depends on committed command-side simulation writes."""


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
    def requires_pre_reward_update(self) -> bool:
        """Whether a term must commit its transition state before rewards read it."""
        return any(term.updates_before_reward for term in self._terms.values())

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        terms = []
        for name, term in self._terms.items():
            command = self._validate_command(name, term.command)
            terms.append((name, command[env_idx].tolist()))
        return terms

    def reset(self, env_ids: np.ndarray | slice | None) -> dict[str, float]:
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        elif isinstance(env_ids, slice):
            env_ids = np.arange(self.num_envs)[env_ids]
        extras = {}
        for name, term in self._terms.items():
            metrics = term.reset(env_ids=env_ids)
            self._validate_command(name, term.command)
            for metric_name, metric_value in metrics.items():
                extras[f"Metrics/{name}/{metric_name}"] = metric_value
        return extras

    def compute(
        self,
        dt: float | np.ndarray,
        env_ids: np.ndarray | None = None,
        *,
        update_before_reward: bool | None = None,
    ) -> None:
        """Advance selected command terms for one lifecycle phase.

        ``None`` preserves the original all-term behavior. Passing a boolean
        runs only terms whose ``updates_before_reward`` declaration matches it;
        this lets environments place a stateful term before reward evaluation
        without moving unrelated command terms across that boundary.
        """
        for name, term in self._terms.items():
            if (
                update_before_reward is not None
                and term.updates_before_reward != update_before_reward
            ):
                continue
            term.compute(dt, env_ids)
            self._validate_command(name, term.command)

    def post_compute(self, *, update_before_reward: bool | None = None) -> None:
        for term in self._terms.values():
            if (
                update_before_reward is not None
                and term.updates_before_reward != update_before_reward
            ):
                continue
            term.post_compute()

    def get_command(self, name: str) -> np.ndarray:
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

    def _validate_command(self, name: str, command: np.ndarray) -> np.ndarray:
        if not isinstance(command, np.ndarray):
            raise TypeError(
                f"CommandManager term '{name}' returned {type(command).__name__}, "
                "expected np.ndarray."
            )
        if command.ndim < 1 or command.shape[0] != self.num_envs:
            raise ValueError(
                f"CommandManager term '{name}' returned shape {command.shape}, "
                f"expected leading dimension {self.num_envs}."
            )
        if not np.isfinite(command).all():
            raise ValueError(f"CommandManager term '{name}' returned NaN or Inf.")
        return command


class NullCommandManager:
    """Placeholder for absent command manager that safely no-ops all operations."""

    def __init__(self):
        self.active_terms: list[str] = []
        self._terms: dict[str, Any] = {}
        self.cfg = None

    @property
    def requires_pre_reward_update(self) -> bool:
        return False

    def __str__(self) -> str:
        return "<NullCommandManager> (inactive)"

    def __repr__(self) -> str:
        return "NullCommandManager()"

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        return []

    def reset(self, env_ids: np.ndarray | None = None) -> dict[str, np.ndarray]:
        return {}

    def compute(
        self,
        dt: float | np.ndarray,
        env_ids: np.ndarray | None = None,
        *,
        update_before_reward: bool | None = None,
    ) -> None:
        del update_before_reward
        pass

    def post_compute(self, *, update_before_reward: bool | None = None) -> None:
        del update_before_reward
        pass

    def get_command(self, name: str) -> None:
        return None

    def get_term(self, name: str) -> None:
        return None

    def get_term_cfg(self, name: str) -> None:
        return None
