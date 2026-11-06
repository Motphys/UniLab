# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), src/mjlab/managers/termination_manager.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and UniLab contracts; licensed under Apache-2.0.
"""Termination manager for computing done signals."""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch
from prettytable import PrettyTable

from unilab.base.reset_selectors import resolve_reset_env_ids
from unilab.managers.manager_base import ManagerBase, ManagerTermBaseCfg

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv


@dataclass
class TerminationTermCfg(ManagerTermBaseCfg):
    """Configuration for a termination term."""

    time_out: bool = False
    """Whether the term contributes towards episodic timeouts."""


class TerminationManager(ManagerBase):
    """Manages termination conditions for the environment.

    The termination manager aggregates multiple termination terms to compute
    episode done signals. Terms can be either truncations (time-based) or
    terminations (failure conditions).
    """

    _env: ManagerBasedRlEnv

    def __init__(self, cfg: dict[str, TerminationTermCfg | None], env: ManagerBasedRlEnv):
        self._term_names: list[str] = list()
        self._term_cfgs: list[TerminationTermCfg] = list()
        self._class_term_cfgs: list[TerminationTermCfg] = list()

        self.cfg = deepcopy(cfg)
        super().__init__(env)

        self._device = env.device
        self._term_dones: dict[str, torch.Tensor] = {}
        for term_name in self._term_names:
            self._term_dones[term_name] = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self._device
            )
        output = getattr(env, "time_out_scratch", None)
        if output is None:
            output = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self._truncated_buf = output
        self._terminated_buf = torch.zeros_like(self._truncated_buf)
        self._dones_buf = torch.zeros_like(self._truncated_buf)
        # Deferred finite validation moves the per-reset episode counts off the
        # host: counts accumulate on this device carrier and are published once
        # at the env's poll boundary via publish_pending_reset_metrics.
        self._pending_term_counts: torch.Tensor | None = None
        self._pending_counts_dirty = False

    def __str__(self) -> str:
        msg = f"<TerminationManager> contains {len(self._term_names)} active terms.\n"
        table = PrettyTable()
        table.title = "Active Termination Terms"
        table.field_names = ["Index", "Name", "Time Out"]
        table.align["Name"] = "l"
        for index, (name, term_cfg) in enumerate(
            zip(self._term_names, self._term_cfgs, strict=False)
        ):
            table.add_row([index, name, term_cfg.time_out])
        msg += str(table.get_string())
        msg += "\n"
        return msg

    # Properties.

    @property
    def active_terms(self) -> list[str]:
        return self._term_names

    @property
    def dones(self) -> torch.Tensor:
        torch.logical_or(self._truncated_buf, self._terminated_buf, out=self._dones_buf)
        return self._dones_buf

    @property
    def time_outs(self) -> torch.Tensor:
        return self._truncated_buf

    @property
    def terminated(self) -> torch.Tensor:
        return self._terminated_buf

    # Methods.

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, int]:
        """Publish episodic counts and clear terms for the selected rows."""
        ids = resolve_reset_env_ids(
            env_ids, num_envs=self.num_envs, device=self._device, owner="TerminationManager.reset"
        )
        extras: dict[str, int] = {}
        keys = list(self._term_dones)
        counts = [self._term_dones[key][ids].sum() for key in keys]
        if self._deferred_validation_enabled():
            # Accumulate counts on the device; the env's poll boundary drains
            # them through publish_pending_reset_metrics (the command/metrics
            # pending-buffer pattern), so resets never read the device.
            if counts and ids.numel() > 0:
                pending = self._pending_term_counts
                if pending is None:
                    pending = torch.zeros(len(keys), dtype=torch.int64, device=self._device)
                    self._pending_term_counts = pending
                pending.add_(torch.stack(counts))
                self._pending_counts_dirty = True
        else:
            host_counts = torch.stack(counts).detach().cpu().tolist() if counts else []
            for key, count in zip(keys, host_counts, strict=True):
                extras["Episode_Termination/" + key] = int(count)
        for term_cfg in self._class_term_cfgs:
            if not callable(getattr(term_cfg.func, "reset", None)):
                raise NotImplementedError(
                    "TerminationManager stateful term does not define the reset boundary"
                )
            term_cfg.func.reset(ids)
        return extras

    def publish_pending_reset_metrics(self) -> dict[str, int]:
        """Drain accumulated episode counts at the environment's host boundary."""
        if not self._pending_counts_dirty or self._pending_term_counts is None:
            return {}
        host_counts = self._pending_term_counts.detach().cpu().tolist()
        self._pending_term_counts.zero_()
        self._pending_counts_dirty = False
        return {
            "Episode_Termination/" + key: int(count)
            for key, count in zip(self._term_dones, host_counts, strict=True)
        }

    def _deferred_validation_enabled(self) -> bool:
        return bool(getattr(self._env, "defer_finite_validation", False))

    def compute(self) -> torch.Tensor:
        timing = getattr(self, "last_step_timing_ms", None)
        if timing is None:
            timing = {}
            self.last_step_timing_ms = timing
        timing.clear()
        self._truncated_buf[:] = False
        self._terminated_buf[:] = False
        dispatch_ms = 0.0
        aggregation_ms = 0.0
        for name, term_cfg in zip(self._term_names, self._term_cfgs, strict=False):
            dispatch_started = time.perf_counter()
            value = self._compute_term(name, term_cfg)
            dispatch_ms += time.perf_counter() - dispatch_started
            aggregation_started = time.perf_counter()
            if term_cfg.time_out:
                self._truncated_buf |= value
            else:
                self._terminated_buf |= value
            self._term_dones[name][:] = value
            aggregation_ms += time.perf_counter() - aggregation_started
        timing["update_state_termination_term_dispatch_ms"] = dispatch_ms * 1000.0
        timing["update_state_termination_aggregation_ms"] = aggregation_ms * 1000.0
        torch.logical_or(self._truncated_buf, self._terminated_buf, out=self._dones_buf)
        return self._dones_buf

    def get_term(self, name: str) -> torch.Tensor:
        return self._term_dones[name]

    def get_term_cfg(self, term_name: str) -> TerminationTermCfg:
        if term_name not in self._term_names:
            raise ValueError(f"Term '{term_name}' not found in active terms.")
        return self._term_cfgs[self._term_names.index(term_name)]

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        terms = []
        for key in self._term_dones.keys():
            terms.append((key, [self._term_dones[key][env_idx].item()]))
        return terms

    def _prepare_terms(self) -> None:
        for term_name, term_cfg in self.cfg.items():
            if term_cfg is None:
                print(f"term: {term_name} set to None, skipping...")
                continue
            self._resolve_common_term_cfg(term_name, term_cfg)
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
                self._class_term_cfgs.append(term_cfg)

    def _compute_term(self, name: str, term_cfg: TerminationTermCfg) -> torch.Tensor:
        value = term_cfg.func(self._env, **term_cfg.params)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"TerminationManager term '{name}' must return torch.Tensor, "
                f"got {type(value).__name__}."
            )
        if value.dtype != torch.bool:
            raise TypeError(
                f"TerminationManager term '{name}' returned dtype {value.dtype}, expected bool."
            )
        if value.device != self._device:
            raise ValueError(
                f"TerminationManager term '{name}' returned device {value.device}, "
                f"expected {self._device}."
            )
        result = value
        if result.shape != (self.num_envs,):
            raise ValueError(
                f"TerminationManager term '{name}' returned shape {tuple(result.shape)}; "
                f"expected ({self.num_envs},)."
            )
        return result
