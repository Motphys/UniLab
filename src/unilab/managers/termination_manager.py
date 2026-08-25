# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), src/mjlab/managers/termination_manager.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and UniLab contracts; licensed under Apache-2.0.
"""Termination manager for computing done signals."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
from prettytable import PrettyTable

from unilab.managers.manager_base import ManagerBase, ManagerTermBaseCfg
from unilab.utils.term_profiling import (
    profile_term,  # PROFILING_TEMP (#1293, TODO: remove after #1292)
)

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

        self._term_dones = dict()
        for term_name in self._term_names:
            self._term_dones[term_name] = np.zeros(self.num_envs, dtype=np.bool_)
        self._truncated_buf = np.zeros(self.num_envs, dtype=np.bool_)
        self._terminated_buf = np.zeros_like(self._truncated_buf)

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
    def dones(self) -> np.ndarray:
        return self._truncated_buf | self._terminated_buf

    @property
    def time_outs(self) -> np.ndarray:
        return self._truncated_buf

    @property
    def terminated(self) -> np.ndarray:
        return self._terminated_buf

    # Methods.

    def reset(self, env_ids: np.ndarray | slice | None = None) -> dict[str, int]:
        if env_ids is None:
            env_ids = slice(None)
        extras = {}
        for key in self._term_dones.keys():
            extras["Episode_Termination/" + key] = int(
                np.count_nonzero(self._term_dones[key][env_ids])
            )
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        return extras

    def compute(self) -> np.ndarray:
        self._truncated_buf[:] = False
        self._terminated_buf[:] = False
        for name, term_cfg in zip(self._term_names, self._term_cfgs, strict=False):
            # PROFILING_TEMP (#1293, TODO: remove after #1292)
            with profile_term(f"termination/{name}"):
                value = term_cfg.func(self._env, **term_cfg.params)
                self._check_term_shape(name, value)
                if value.dtype != np.bool_:
                    raise TypeError(
                        f"TerminationManager term '{name}' returned dtype {value.dtype}, expected bool."
                    )
                if term_cfg.time_out:
                    self._truncated_buf |= value
                else:
                    self._terminated_buf |= value
                self._term_dones[name][:] = value
        return self._truncated_buf | self._terminated_buf

    def get_term(self, name: str) -> np.ndarray:
        return self._term_dones[name]

    def get_term_cfg(self, term_name: str) -> TerminationTermCfg:
        if term_name not in self._term_names:
            raise ValueError(f"Term '{term_name}' not found in active terms.")
        return self._term_cfgs[self._term_names.index(term_name)]

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        terms = []
        for key in self._term_dones.keys():
            terms.append((key, [float(self._term_dones[key][env_idx])]))
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
