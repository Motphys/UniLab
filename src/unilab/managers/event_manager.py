# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), src/mjlab/managers/event_manager.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and UniLab contracts; licensed under Apache-2.0.
"""Event manager for orchestrating operations based on different simulation events."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from prettytable import PrettyTable

from unilab.managers.manager_base import ManagerBase, ManagerTermBaseCfg
from unilab.utils.term_profiling import (
    profile_term,  # PROFILING_TEMP (#1293, TODO: remove after #1292)
)

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv

EventMode = Literal["startup", "reset", "interval", "step"]


@dataclass(kw_only=True)
class EventTermCfg(ManagerTermBaseCfg):
    """Configuration for an event term.

    Event terms trigger operations at specific simulation events. They're commonly
    used for domain randomization, state resets, and periodic perturbations.

    The four modes determine when the event fires:

    - ``"startup"``: Once when the environment initializes. Use for parameters that
      should be randomized per-environment but stay constant within an episode (e.g.,
      domain randomization).

    - ``"reset"``: On every episode reset. Use for parameters that should vary between
      episodes (e.g., initial robot pose, domain randomization).

    - ``"interval"``: Periodically during simulation, controlled by ``interval_range_s``.
      Use for perturbations that should happen during episodes (e.g., pushing the robot,
      external disturbances).

    - ``"step"``: Every environment step, unconditionally on all envs. Use for terms that
      manage per-step state such as force lifetimes (e.g., ``apply_body_impulse``).
    """

    mode: EventMode
    """When the event triggers: ``"startup"`` (once at init), ``"reset"`` (every
  episode), ``"interval"`` (periodically during simulation), or ``"step"`` (every
  environment step)."""

    interval_range_s: tuple[float, float] | None = None
    """Time range in seconds for interval mode. The next trigger time is uniformly
  sampled from ``[min, max]``. Required when ``mode="interval"``."""

    is_global_time: bool = False
    """Whether all environments share the same timer. If True, all envs trigger
  simultaneously. If False (default), each env has an independent timer that
  resets on episode reset. Only applies to ``mode="interval"``."""

    min_step_count_between_reset: int = 0
    """Minimum environment steps between triggers. Prevents the event from firing
  too frequently when episodes reset rapidly. Only applies to ``mode="reset"``.
  Set to 0 (default) to trigger on every reset."""


class EventManager(ManagerBase):
    """Manages event-based operations for the environment.

    The event manager triggers operations at different simulation events: startup
    (once at initialization), reset (on episode reset), or interval (periodically
    during simulation). Common uses include domain randomization and state resets.
    """

    _env: ManagerBasedRlEnv

    def __init__(self, cfg: dict[str, EventTermCfg | None], env: ManagerBasedRlEnv):
        self.cfg = deepcopy(cfg)
        self._mode_term_names: dict[EventMode, list[str]] = dict()
        self._mode_term_cfgs: dict[EventMode, list[EventTermCfg]] = dict()
        self._mode_class_term_cfgs: dict[EventMode, list[EventTermCfg]] = dict()

        super().__init__(env=env)

    def __str__(self) -> str:
        msg = f"<EventManager> contains {len(self._mode_term_names)} active terms.\n"
        for mode in self._mode_term_names:
            table = PrettyTable()
            table.title = f"Active Event Terms in Mode: '{mode}'"
            if mode == "interval":
                table.field_names = ["Index", "Name", "Interval time range (s)"]
                table.align["Name"] = "l"
                for index, (name, cfg) in enumerate(
                    zip(self._mode_term_names[mode], self._mode_term_cfgs[mode], strict=False)
                ):
                    table.add_row([index, name, cfg.interval_range_s])
            else:
                table.field_names = ["Index", "Name"]
                table.align["Name"] = "l"
                for index, name in enumerate(self._mode_term_names[mode]):
                    table.add_row([index, name])
            msg += str(table.get_string())
            msg += "\n"
        return msg

    # Properties.

    @property
    def active_terms(self) -> dict[EventMode, list[str]]:
        return self._mode_term_names

    @property
    def available_modes(self) -> list[EventMode]:
        return list(self._mode_term_names.keys())

    # Methods.

    def get_term_cfg(self, term_name: str) -> EventTermCfg:
        """Get the configuration of a specific event term by name."""
        for mode in self._mode_term_names:
            if term_name in self._mode_term_names[mode]:
                index = self._mode_term_names[mode].index(term_name)
                return self._mode_term_cfgs[mode][index]
        raise ValueError(f"Event term '{term_name}' not found in active terms.")

    def reset(self, env_ids: np.ndarray | None = None):
        for mode_cfg in self._mode_class_term_cfgs.values():
            for term_cfg in mode_cfg:
                term_cfg.func.reset(env_ids=env_ids)
        if env_ids is None:
            num_envs = self._env.num_envs
            ids: np.ndarray | slice = slice(None)
        else:
            num_envs = len(env_ids)
            ids = env_ids
        # Iterate the full interval term list: _interval_term_time_left is parallel
        # to _mode_term_cfgs["interval"], not the class-only subset.
        if "interval" in self._mode_term_cfgs:
            for index, term_cfg in enumerate(self._mode_term_cfgs["interval"]):
                if not term_cfg.is_global_time:
                    assert term_cfg.interval_range_s is not None
                    lower, upper = term_cfg.interval_range_s
                    sampled_interval = self._env.rng.uniform(lower, upper, num_envs)
                    self._interval_term_time_left[index][ids] = sampled_interval
        return {}

    def apply(
        self,
        mode: EventMode,
        env_ids: np.ndarray | slice | None = None,
        dt: float | None = None,
        global_env_step_count: int | None = None,
    ):
        if mode not in ("startup", "reset", "interval", "step"):
            raise ValueError(f"Unsupported event mode '{mode}'.")
        if mode not in self._mode_term_cfgs:
            return
        if mode == "interval" and dt is None:
            raise ValueError(f"Event mode '{mode}' requires the time-step of the environment.")
        if mode == "interval" and env_ids is not None:
            raise ValueError(
                f"Event mode '{mode}' does not require environment indices. This is an undefined behavior"
                " as the environment indices are computed based on the time left for each environment."
            )
        if mode == "reset" and global_env_step_count is None:
            raise ValueError(
                f"Event mode '{mode}' requires the total number of environment steps to be provided."
            )
        if mode == "step" and dt is None:
            raise ValueError(f"Event mode '{mode}' requires the time-step of the environment.")

        for index, term_cfg in enumerate(self._mode_term_cfgs[mode]):
            # PROFILING_TEMP (#1293, TODO: remove after #1292)
            pkey = f"event/{mode}/{self._mode_term_names[mode][index]}"
            if mode == "interval":
                time_left = self._interval_term_time_left[index]
                assert dt is not None
                time_left -= dt
                if term_cfg.is_global_time:
                    if time_left < 1e-6:
                        assert term_cfg.interval_range_s is not None
                        lower, upper = term_cfg.interval_range_s
                        sampled_interval = self._env.rng.uniform(lower, upper, 1)
                        self._interval_term_time_left[index][:] = sampled_interval
                        with profile_term(pkey):  # PROFILING_TEMP (#1293)
                            term_cfg.func(self._env, None, **term_cfg.params)
                else:
                    valid_env_ids = np.flatnonzero(time_left < 1e-6)
                    if len(valid_env_ids) > 0:
                        assert term_cfg.interval_range_s is not None
                        lower, upper = term_cfg.interval_range_s
                        sampled_time = self._env.rng.uniform(lower, upper, len(valid_env_ids))
                        self._interval_term_time_left[index][valid_env_ids] = sampled_time
                        with profile_term(pkey):  # PROFILING_TEMP (#1293)
                            term_cfg.func(self._env, valid_env_ids, **term_cfg.params)
            elif mode == "step":
                with profile_term(pkey):  # PROFILING_TEMP (#1293)
                    term_cfg.func(self._env, None, **term_cfg.params)
            elif mode == "reset":
                assert global_env_step_count is not None
                # Reset events require concrete indices: callers (e.g. ManagerBasedRlEnv)
                # resolve None to all environments upstream. Enforce that here so a future
                # caller passing None fails loudly instead of leaking a slice into event
                # functions, which only understand None or a tensor.
                if env_ids is None:
                    raise ValueError("Event mode 'reset' requires concrete environment indices.")
                min_step_count = term_cfg.min_step_count_between_reset
                if min_step_count == 0:
                    self._reset_term_last_triggered_step_id[index][env_ids] = global_env_step_count
                    self._reset_term_last_triggered_once[index][env_ids] = True
                    with profile_term(pkey):  # PROFILING_TEMP (#1293)
                        term_cfg.func(self._env, env_ids, **term_cfg.params)
                else:
                    last_triggered_step = self._reset_term_last_triggered_step_id[index][env_ids]
                    triggered_at_least_once = self._reset_term_last_triggered_once[index][env_ids]
                    steps_since_triggered = global_env_step_count - last_triggered_step
                    valid_trigger = steps_since_triggered >= min_step_count
                    valid_trigger |= (last_triggered_step == 0) & ~triggered_at_least_once
                    if isinstance(env_ids, np.ndarray):
                        valid_env_ids = env_ids[valid_trigger]
                    else:
                        valid_env_ids = np.flatnonzero(valid_trigger)
                    if len(valid_env_ids) > 0:
                        self._reset_term_last_triggered_once[index][valid_env_ids] = True
                        self._reset_term_last_triggered_step_id[index][valid_env_ids] = (
                            global_env_step_count
                        )
                        with profile_term(pkey):  # PROFILING_TEMP (#1293)
                            term_cfg.func(self._env, valid_env_ids, **term_cfg.params)
            else:
                with profile_term(pkey):  # PROFILING_TEMP (#1293)
                    term_cfg.func(self._env, env_ids, **term_cfg.params)

    def _prepare_terms(self) -> None:
        self._interval_term_time_left: list[np.ndarray] = list()
        self._reset_term_last_triggered_step_id: list[np.ndarray] = list()
        self._reset_term_last_triggered_once: list[np.ndarray] = list()

        for term_name, term_cfg in self.cfg.items():
            if term_cfg is None:
                print(f"term: {term_name} set to None, skipping...")
                continue
            self._resolve_common_term_cfg(term_name, term_cfg)
            if term_cfg.mode not in ("startup", "reset", "interval", "step"):
                raise ValueError(
                    f"EventManager term '{term_name}' has unsupported mode '{term_cfg.mode}'."
                )
            if term_cfg.mode not in self._mode_term_names:
                self._mode_term_names[term_cfg.mode] = list()
                self._mode_term_cfgs[term_cfg.mode] = list()
                self._mode_class_term_cfgs[term_cfg.mode] = list()
            self._mode_term_names[term_cfg.mode].append(term_name)
            self._mode_term_cfgs[term_cfg.mode].append(term_cfg)
            if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
                self._mode_class_term_cfgs[term_cfg.mode].append(term_cfg)
            if term_cfg.mode == "interval":
                if term_cfg.interval_range_s is None:
                    raise ValueError(
                        f"Event term '{term_name}' has mode 'interval' but 'interval_range_s' is not specified."
                    )
                lower, upper = term_cfg.interval_range_s
                if not np.isfinite((lower, upper)).all() or lower > upper:
                    raise ValueError(
                        f"EventManager term '{term_name}' has invalid interval_range_s "
                        f"{term_cfg.interval_range_s}."
                    )
                if term_cfg.is_global_time:
                    time_left = self._env.rng.uniform(lower, upper, 1)
                    self._interval_term_time_left.append(time_left)
                else:
                    time_left = self._env.rng.uniform(lower, upper, self.num_envs)
                    self._interval_term_time_left.append(time_left)
            elif term_cfg.mode == "reset":
                step_count = np.zeros(self.num_envs, dtype=np.int64)
                self._reset_term_last_triggered_step_id.append(step_count)
                no_trigger = np.zeros(self.num_envs, dtype=np.bool_)
                self._reset_term_last_triggered_once.append(no_trigger)

            func = term_cfg.func
            if hasattr(func, "model_fields"):
                raise NotImplementedError(
                    f"EventManager term '{term_name}' requests direct model-field mutation; "
                    "this capability is unsupported by the standalone UniLab manager core."
                )
