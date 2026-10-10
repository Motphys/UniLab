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

from unilab.managers.manager_base import ManagerBase, ManagerTermBaseCfg

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


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

        self._device = getattr(env, "device", torch.device("cpu"))
        self._term_dones: dict[str, torch.Tensor] = {}
        for term_name in self._term_names:
            self._term_dones[term_name] = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self._device
            )
        self._truncated_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self._terminated_buf = torch.zeros_like(self._truncated_buf)

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
        return self._truncated_buf | self._terminated_buf

    @property
    def time_outs(self) -> torch.Tensor:
        return self._truncated_buf

    @property
    def terminated(self) -> torch.Tensor:
        return self._terminated_buf

    # Methods.

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, int]:
        if env_ids is None:
            env_ids = slice(None)
        extras = {}
        mask = self._reset_mask(env_ids)
        keys = list(self._term_dones)
        counts = [self._term_dones[key][mask].sum() for key in keys]
        # Publish every episodic count through one device boundary instead of
        # synchronizing once per termination term.
        host_counts = torch.stack(counts).detach().cpu().tolist() if counts else []
        for key, count in zip(keys, host_counts, strict=True):
            extras["Episode_Termination/" + key] = int(count)
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        return extras

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
        return self._truncated_buf | self._terminated_buf

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
        if getattr(term_cfg.func, "returns_transient_tensor", False):
            result = value
        else:
            result = value.clone()
        if result.shape != (self.num_envs,):
            raise ValueError(
                f"TerminationManager term '{name}' returned shape {tuple(result.shape)}; "
                f"expected ({self.num_envs},)."
            )
        return result

    def _reset_mask(self, env_ids: torch.Tensor | slice) -> torch.Tensor:
        if env_ids is None:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self._device)
        if isinstance(env_ids, slice):
            indices = torch.arange(self.num_envs, dtype=torch.int64, device=self._device)[env_ids]
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self._device).index_fill(
                0, indices, True
            )
        if (
            not isinstance(env_ids, torch.Tensor)
            or env_ids.ndim != 1
            or env_ids.dtype
            not in {
                torch.int32,
                torch.int64,
            }
        ):
            raise TypeError("TerminationManager reset rows must be one-dimensional integers")
        rows = env_ids.to(self._device)
        if rows.numel() and (rows.min() < 0 or rows.max() >= self.num_envs):
            raise IndexError(f"TerminationManager reset rows out of range: {rows.tolist()}")
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self._device).index_fill(
            0, rows.to(torch.int64), True
        )
