# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), src/mjlab/managers/metrics_manager.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and UniLab contracts; licensed under Apache-2.0.
"""Metrics manager for logging custom per-step metrics during training."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
from prettytable import PrettyTable

from unilab.managers.manager_base import ManagerBase, ManagerTermBaseCfg

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv

REDUCE_OPTIONS = ("last", "max", "mean", "sum")


@dataclass(kw_only=True)
class MetricsTermCfg(ManagerTermBaseCfg):
    """Configuration for a metrics term.

    Attributes:
      per_substep: If True, evaluate this term once per physics substep inside
        the decimation loop and report the per-step mean. Only the integrated
        state (qpos, qvel, act) is current mid-loop; all derived quantities
        (xpos, xquat, site_xpos, actuator_force, contacts, ...) are stale.

      reduce: How to aggregate per-step values into an episode metric.
        - ``"mean"`` (default) reports ``sum / step_count``.
        - ``"last"`` reports the value from the final step of the episode,
          useful for binary success metrics that should not be averaged over
          timesteps.
        - ``"max"`` reports the highest value seen during the episode, useful
          for peak metrics like maximum power or contact force.
        - ``"sum"`` reports the accumulated total over the episode, useful for
          cumulative quantities like episodic reward or total distance
          traveled.
    """

    per_substep: bool = False
    reduce: Literal["last", "max", "mean", "sum"] = "mean"


class MetricsManager(ManagerBase):
    """Accumulates per-step metric values, reports episode averages.

    Unlike rewards, metrics have no weight, no dt scaling, and no
    normalization by episode length. Episode values are true per-step
    averages (sum / step_count), so a metric in [0,1] stays in [0,1]
    in the logger.
    """

    _env: ManagerBasedRlEnv

    def __init__(
        self,
        cfg: dict[str, MetricsTermCfg | None],
        env: ManagerBasedRlEnv,
        *,
        finite_check_interval: int = 1,
    ):
        self._term_names: list[str] = list()
        self._term_cfgs: list[MetricsTermCfg] = list()
        self._class_term_cfgs: list[MetricsTermCfg] = list()
        self._step_term_indices: list[int] = list()
        self._substep_term_indices: list[int] = list()
        # NaN/Inf check cadence: every ``finite_check_interval`` compute() calls,
        # starting with the first. 1 keeps the historical every-step behavior.
        self._finite_check_interval = max(1, int(finite_check_interval))
        self._finite_check_clock = 0

        self.cfg = deepcopy(cfg)
        super().__init__(env=env)

        self._episode_sums: dict[str, np.ndarray] = {}
        self._episode_max: dict[str, np.ndarray] = {}
        for idx, term_name in enumerate(self._term_names):
            if self._term_cfgs[idx].reduce not in REDUCE_OPTIONS:
                msg = (
                    f"The reduce method '{self._term_cfgs[idx].reduce}' for metric '{term_name}' "
                    f"is unknown. Valid options are {REDUCE_OPTIONS}."
                )
                raise ValueError(msg)

            self._episode_sums[term_name] = np.zeros(self.num_envs, dtype=np.float32)

            if self._term_cfgs[idx].reduce == "max":
                self._episode_max[term_name] = np.full(
                    self.num_envs, float("-inf"), dtype=np.float32
                )
        # Pre-resolved tensor refs for substep terms to avoid dict lookups in
        # the hot loop.
        self._substep_accum: list[np.ndarray] = []
        self._substep_episode_sums: list[np.ndarray] = []
        self._substep_episode_max: list[np.ndarray | None] = []
        for idx in self._substep_term_indices:
            name = self._term_names[idx]
            buf = np.zeros(self.num_envs, dtype=np.float32)
            self._substep_accum.append(buf)
            self._substep_episode_sums.append(self._episode_sums[name])
            self._substep_episode_max.append(self._episode_max.get(name))
        self._substep_count: int = 0
        self._step_count = np.zeros(self.num_envs, dtype=np.int64)
        self._step_values = np.zeros((self.num_envs, len(self._term_names)), dtype=np.float32)

    def __str__(self) -> str:
        msg = f"<MetricsManager> contains {len(self._term_names)} active terms.\n"
        table = PrettyTable()
        table.title = "Active Metrics Terms"
        table.field_names = ["Index", "Name"]
        table.align["Name"] = "l"
        for index, name in enumerate(self._term_names):
            table.add_row([index, name])
        msg += str(table.get_string())
        msg += "\n"
        return msg

    # Properties.

    @property
    def active_terms(self) -> list[str]:
        return self._term_names

    # Methods.

    def reset(self, env_ids: np.ndarray | slice | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        extras = {}
        counts = self._step_count[env_ids].astype(np.float32)
        # Avoid division by zero for envs that haven't stepped.
        safe_counts = np.maximum(counts, 1.0)
        for idx, key in enumerate(self._episode_sums):
            reduce = self._term_cfgs[idx].reduce
            if reduce == "max":
                extras["Episode_Metrics/" + key] = float(np.mean(self._episode_max[key][env_ids]))
                self._episode_max[key][env_ids] = float("-inf")

            elif reduce == "last":
                extras["Episode_Metrics/" + key] = float(np.mean(self._step_values[env_ids, idx]))

            elif reduce == "sum":
                extras["Episode_Metrics/" + key] = float(np.mean(self._episode_sums[key][env_ids]))

            else:
                extras["Episode_Metrics/" + key] = float(
                    np.mean(self._episode_sums[key][env_ids] / safe_counts)
                )

            self._episode_sums[key][env_ids] = 0.0
        self._step_count[env_ids] = 0

        for buf in self._substep_accum:
            buf[env_ids] = 0.0

        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)

        return extras

    def compute_substep(self) -> None:
        """Accumulate per-substep metric values inside the decimation loop.

        No-op when no ``per_substep`` terms are configured.
        """
        if not self._substep_term_indices:
            return
        for i, idx in enumerate(self._substep_term_indices):
            value = self._compute_term(idx)
            self._substep_accum[i] += value
        self._substep_count += 1

    def compute(self) -> None:
        self._step_count += 1
        # Same cadence rule as RewardManager: first compute always checks.
        self._finite_check_clock += 1
        if self._substep_term_indices and self._substep_count > 0:
            for i, idx in enumerate(self._substep_term_indices):
                avg = self._substep_accum[i] / self._substep_count
                self._substep_episode_sums[i] += avg
                self._step_values[:, idx] = avg
                max_buf = self._substep_episode_max[i]
                if max_buf is not None:
                    np.maximum(max_buf, avg, out=max_buf)
                self._substep_accum[i].fill(0.0)
            self._substep_count = 0
        for idx in self._step_term_indices:
            name = self._term_names[idx]
            value = self._compute_term(idx)
            self._episode_sums[name] += value
            self._step_values[:, idx] = value
            if name in self._episode_max:
                np.maximum(self._episode_max[name], value, out=self._episode_max[name])

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        terms = []
        for idx, name in enumerate(self._term_names):
            terms.append((name, [self._step_values[env_idx, idx].item()]))
        return terms

    def _prepare_terms(self) -> None:
        for term_name, term_cfg in self.cfg.items():
            if term_cfg is None:
                print(f"term: {term_name} set to None, skipping...")
                continue
            self._resolve_common_term_cfg(term_name, term_cfg)
            idx = len(self._term_names)
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            if term_cfg.per_substep:
                self._substep_term_indices.append(idx)
            else:
                self._step_term_indices.append(idx)
            if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
                self._class_term_cfgs.append(term_cfg)

    def _compute_term(self, idx: int) -> np.ndarray:
        name = self._term_names[idx]
        term_cfg = self._term_cfgs[idx]
        value = term_cfg.func(self._env, **term_cfg.params)
        self._check_term_shape(name, value)
        if (self._finite_check_clock - 1) % self._finite_check_interval == 0:
            self._check_term_finite(name, value)
        return value


class NullMetricsManager:
    """Placeholder for absent metrics manager that safely no-ops all operations."""

    def __init__(self):
        self.active_terms: list[str] = []
        self.cfg = None

    def __str__(self) -> str:
        return "<NullMetricsManager> (inactive)"

    def __repr__(self) -> str:
        return "NullMetricsManager()"

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        return []

    def reset(self, env_ids: np.ndarray | None = None) -> dict[str, float]:
        return {}

    def compute_substep(self) -> None:
        pass

    def compute(self) -> None:
        pass
