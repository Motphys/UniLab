# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), src/mjlab/managers/event_manager.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and UniLab contracts; licensed under Apache-2.0.
"""Event manager for orchestrating operations based on different simulation events."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import torch
from prettytable import PrettyTable

from unilab.base.host_transitions import record_host_transition
from unilab.base.reset_selection import (
    ResetSelection,
    assign_rows_,
    fill_rows_,
    selection_blockers,
)
from unilab.base.reset_selectors import resolve_reset_env_ids
from unilab.managers.manager_base import ManagerBase, ManagerTermBaseCfg
from unilab.managers.rng import ManagerRng

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv

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
        self._device = env.device
        rng = env.rng
        if not isinstance(rng, ManagerRng):
            raise RuntimeError(
                "EventManager interval sampling requires the Manager-owned Torch generator"
            )
        self._rng = rng
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
    def term_cfgs(self) -> dict[str, EventTermCfg]:
        """Return active event term configs keyed by their global term name."""
        return {
            name: self._mode_term_cfgs[mode][index]
            for mode, names in self._mode_term_names.items()
            for index, name in enumerate(names)
        }

    @property
    def available_modes(self) -> list[EventMode]:
        return list(self._mode_term_names.keys())

    @property
    def uses_startup_rows(self) -> bool:
        """Whether any startup term requires concrete full-width row indices."""
        return any(
            bool(getattr(term_cfg.func, "uses_startup_rows", False))
            for term_cfg in self._mode_term_cfgs.get("startup", ())
        )

    # Methods.

    def get_term_cfg(self, term_name: str) -> EventTermCfg:
        """Get the configuration of a specific event term by name."""
        for mode in self._mode_term_names:
            if term_name in self._mode_term_names[mode]:
                index = self._mode_term_names[mode].index(term_name)
                return self._mode_term_cfgs[mode][index]
        raise ValueError(f"Event term '{term_name}' not found in active terms.")

    supports_reset_selection = True

    def reset_selection_blockers(self) -> list[str]:
        """Reset-mode terms and stateful step/interval terms must opt in."""
        participants = [
            (name, cfg.func)
            for name, cfg in zip(
                self._mode_term_names.get("reset", ()),
                self._mode_term_cfgs.get("reset", ()),
                strict=True,
            )
        ]
        for mode, mode_cfg in self._mode_class_term_cfgs.items():
            if mode != "reset":
                participants.extend((f"{mode}:{cfg.func!r}", cfg.func) for cfg in mode_cfg)
        return selection_blockers("EventManager", participants)

    def reset(
        self, env_ids: torch.Tensor | slice | ResetSelection | None = None
    ) -> dict[str, float]:
        """Clear stateful event terms for the selected environment rows."""
        ids = resolve_reset_env_ids(
            env_ids, num_envs=self.num_envs, device=self._device, owner="EventManager.reset"
        )
        for mode, mode_cfg in self._mode_class_term_cfgs.items():
            if mode == "reset":
                continue
            for term_cfg in mode_cfg:
                term_cfg.func.reset(ids)
        # Iterate the full interval term list: _interval_term_time_left is parallel
        # to _mode_term_cfgs["interval"], not the class-only subset.
        for index, term_cfg in enumerate(self._mode_term_cfgs.get("interval", ())):
            if not term_cfg.is_global_time:
                assert term_cfg.interval_range_s is not None
                lower, upper = term_cfg.interval_range_s
                assign_rows_(
                    self._interval_term_time_left[index],
                    ids,
                    lambda count, lower=lower, upper=upper: self._sample_interval(
                        lower, upper, count
                    ),
                )
                # Resampled rows restart at >= lower; untouched rows keep the
                # previous host lower bound. Interval apply lowers the bound
                # between resets, so a captured reset must replay this update.
                record_host_transition(
                    self._env,
                    f"event.interval_lower_bound.{index}",
                    lambda index=index, lower=lower: self._lower_interval_bound(index, lower),
                )
        return {}

    def _lower_interval_bound(self, index: int, lower: float) -> None:
        self._interval_host_time_left_lb[index] = min(
            self._interval_host_time_left_lb[index], lower
        )

    def apply(
        self,
        mode: EventMode,
        env_ids: torch.Tensor | slice | None = None,
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

        if mode == "reset":
            # Reset events require concrete indices: callers (e.g.
            # ManagerBasedRlEnv) resolve None to all environments upstream.
            # Enforce that here so a future caller passing None fails loudly
            # instead of leaking a slice into event functions.
            if env_ids is None or isinstance(env_ids, slice):
                raise ValueError("Event mode 'reset' requires concrete environment indices.")
            if isinstance(env_ids, ResetSelection):
                self._apply_reset_selection(env_ids)
                return
            self._apply_reset(env_ids, global_env_step_count=cast(int, global_env_step_count))
            return

        for index, term_cfg in enumerate(self._mode_term_cfgs[mode]):
            if mode == "interval":
                time_left = self._interval_term_time_left[index]
                assert dt is not None
                time_left -= dt
                assert term_cfg.interval_range_s is not None
                lower, upper = term_cfg.interval_range_s
                # Host lower bound on the minimum device timer (the MotionCommand
                # _timer_lower_bound_s pattern): IEEE-754 rounding is monotone,
                # so subtracting the same float64 dt on host and device keeps the
                # bound valid. While the bound stays above the trigger threshold
                # no timer can fire and the device is not queried at all.
                host_lb = self._interval_host_time_left_lb[index] - float(dt)
                if host_lb >= 1e-6:
                    self._interval_host_time_left_lb[index] = host_lb
                    continue
                if term_cfg.is_global_time:
                    current = float(time_left.amin())
                    if current < 1e-6:
                        self._interval_term_time_left[index][:] = self._sample_interval(
                            lower, upper, 1
                        )
                        term_cfg.func(self._env, None, **term_cfg.params)
                        self._interval_host_time_left_lb[index] = lower
                    else:
                        self._interval_host_time_left_lb[index] = current
                else:
                    expired = time_left < 1e-6
                    # One packed host read publishes both the trigger verdict and
                    # the minimum surviving timer (the next lower bound).
                    probe = torch.stack(
                        (
                            expired.any().to(torch.float64),
                            torch.where(
                                expired, torch.full_like(time_left, float("inf")), time_left
                            ).amin(),
                        )
                    ).tolist()
                    if not probe[0]:
                        self._interval_host_time_left_lb[index] = probe[1]
                        continue
                    valid_env_ids = torch.nonzero(expired, as_tuple=False).flatten()
                    self._interval_term_time_left[index][valid_env_ids] = self._sample_interval(
                        lower, upper, valid_env_ids.numel()
                    )
                    term_cfg.func(self._env, valid_env_ids, **term_cfg.params)
                    # Triggered rows resampled to >= lower; the surviving rows
                    # keep the measured minimum.
                    self._interval_host_time_left_lb[index] = min(lower, probe[1])
            elif mode == "step":
                term_cfg.func(self._env, None, **term_cfg.params)
            else:
                term_cfg.func(self._env, env_ids, **term_cfg.params)

    def _apply_reset(self, env_ids: torch.Tensor, *, global_env_step_count: int) -> None:
        """Dispatch reset event terms to the selected rows (mjlab-style)."""
        for index, term_cfg in enumerate(self._mode_term_cfgs["reset"]):
            min_step_count = term_cfg.min_step_count_between_reset
            if min_step_count == 0:
                self._reset_term_last_triggered_step_id[index][env_ids] = global_env_step_count
                self._reset_term_last_triggered_once[index][env_ids] = True
                term_cfg.func(self._env, env_ids, **term_cfg.params)
            else:
                last_triggered_step = self._reset_term_last_triggered_step_id[index][env_ids]
                triggered_at_least_once = self._reset_term_last_triggered_once[index][env_ids]
                steps_since_triggered = global_env_step_count - last_triggered_step
                valid_trigger = steps_since_triggered >= min_step_count
                valid_trigger |= (last_triggered_step == 0) & ~triggered_at_least_once
                valid_env_ids = env_ids[valid_trigger]
                if valid_env_ids.numel() > 0:
                    self._reset_term_last_triggered_once[index][valid_env_ids] = True
                    self._reset_term_last_triggered_step_id[index][valid_env_ids] = (
                        global_env_step_count
                    )
                    term_cfg.func(self._env, valid_env_ids, **term_cfg.params)

    def _apply_reset_selection(self, selection: ResetSelection) -> None:
        """Fixed-shape reset dispatch: every term sees a ``ResetSelection``.

        The trigger step comes from the env's device step counter, not a host
        integer, so a captured dispatch stays correct on replay. Terms with a
        ``min_step_count_between_reset`` receive their own selection whose mask
        is narrowed to the rows allowed to trigger (possibly empty).
        """
        step_counter = getattr(self._env, "step_counter_tensor", None)
        if not isinstance(step_counter, torch.Tensor):
            raise RuntimeError(
                "EventManager ResetSelection dispatch requires env.step_counter_tensor"
            )
        step_rows = step_counter.expand(self.num_envs)
        for index, term_cfg in enumerate(self._mode_term_cfgs["reset"]):
            last_step = self._reset_term_last_triggered_step_id[index]
            once = self._reset_term_last_triggered_once[index]
            min_step_count = term_cfg.min_step_count_between_reset
            if min_step_count == 0:
                term_selection = selection
            else:
                valid_trigger = (step_rows - last_step) >= min_step_count
                valid_trigger |= (last_step == 0) & ~once
                term_selection = self._term_reset_selection(index, selection)
                torch.logical_and(selection.mask, valid_trigger, out=term_selection.mask)
            assign_rows_(last_step, term_selection, step_rows)
            fill_rows_(once, term_selection, True)
            term_cfg.func(self._env, term_selection, **term_cfg.params)

    def _term_reset_selection(self, index: int, selection: ResetSelection) -> ResetSelection:
        scratch = self._reset_term_selections.get(index)
        if scratch is None:
            scratch = ResetSelection(
                mask=torch.zeros_like(selection.mask), row_ids=selection.row_ids
            )
            self._reset_term_selections[index] = scratch
        return scratch

    def _prepare_terms(self) -> None:
        self._reset_term_selections: dict[int, ResetSelection] = {}
        self._interval_term_time_left: list[torch.Tensor] = list()
        self._interval_host_time_left_lb: list[float] = list()
        self._reset_term_last_triggered_step_id: list[torch.Tensor] = list()
        self._reset_term_last_triggered_once: list[torch.Tensor] = list()

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
            if (
                term_cfg.mode in ("step", "interval")
                and hasattr(term_cfg.func, "reset")
                and callable(term_cfg.func.reset)
            ):
                self._mode_class_term_cfgs[term_cfg.mode].append(term_cfg)
            if term_cfg.mode == "interval":
                if term_cfg.interval_range_s is None:
                    raise ValueError(
                        f"Event term '{term_name}' has mode 'interval' but 'interval_range_s' is not specified."
                    )
                lower, upper = term_cfg.interval_range_s
                if not (math.isfinite(lower) and math.isfinite(upper)) or lower > upper:
                    raise ValueError(
                        f"EventManager term '{term_name}' has invalid interval_range_s "
                        f"{term_cfg.interval_range_s}."
                    )
                shape = 1 if term_cfg.is_global_time else self.num_envs
                self._interval_term_time_left.append(self._sample_interval(lower, upper, shape))
                # Samples are drawn from [lower, upper], so lower is a valid
                # initial host lower bound for every device timer.
                self._interval_host_time_left_lb.append(lower)
            elif term_cfg.mode == "reset":
                step_count = torch.zeros(self.num_envs, dtype=torch.int64, device=self._device)
                self._reset_term_last_triggered_step_id.append(step_count)
                no_trigger = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
                self._reset_term_last_triggered_once.append(no_trigger)

            func = term_cfg.func
            if hasattr(func, "model_fields"):
                raise NotImplementedError(
                    f"EventManager term '{term_name}' requests direct model-field mutation; "
                    "this capability is unsupported by the standalone UniLab manager core."
                )

    def _sample_interval(self, lower: float, upper: float, count: int) -> torch.Tensor:
        return self._rng.uniform(lower, upper, (count,), dtype=torch.float64).reshape(count)
