# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), src/mjlab/managers/reward_manager.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and UniLab contracts; licensed under Apache-2.0.
"""Reward manager for computing reward signals."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from prettytable import PrettyTable

from unilab.managers.manager_base import ManagerBase, ManagerTermBaseCfg

if TYPE_CHECKING:
    from unilab.managers._types import DebugVisualizer, ManagerBasedRlEnv


@dataclass(kw_only=True)
class RewardTermCfg(ManagerTermBaseCfg):
    """Configuration for a reward term."""

    func: Any
    """The callable that computes this reward term's value."""

    weight: float
    """Weight multiplier for this reward term."""


class RewardManager(ManagerBase):
    """Manages reward computation by aggregating weighted reward terms.

    Reward Scaling Behavior:
      By default, rewards are scaled by the environment step duration (dt). This
      normalizes cumulative episodic rewards across different simulation frequencies.
      The scaling can be disabled via the ``scale_by_dt`` parameter.

      When ``scale_by_dt=True`` (default):
        - ``reward_buf`` (returned by ``compute()``) = raw_value * weight * dt
        - ``_episode_sums`` (cumulative rewards) are scaled by dt
        - ``Episode_Reward/*`` logged metrics are scaled by dt

      When ``scale_by_dt=False``:
        - ``reward_buf`` = raw_value * weight (no dt scaling)

      Regardless of the scaling setting:
        - ``_step_reward`` (via ``get_active_iterable_terms()``) always contains
          the unscaled reward rate (raw_value * weight)

      ``step_reward_extras()`` exposes the latest ``compute()`` call's per-term
      means as ``reward/<term>`` log entries (weighted, pre-dt rate), matching
      the legacy envs' per-step reward log contract consumed by training
      runners.
    """

    _env: ManagerBasedRlEnv

    def __init__(
        self,
        cfg: dict[str, RewardTermCfg | None],
        env: ManagerBasedRlEnv,
        *,
        scale_by_dt: bool = True,
        finite_check_interval: int = 1,
    ):
        self._term_names: list[str] = list()
        self._term_cfgs: list[RewardTermCfg] = list()
        self._class_term_cfgs: list[RewardTermCfg] = list()
        self._scale_by_dt = scale_by_dt
        # NaN/Inf check cadence: every ``finite_check_interval`` compute() calls,
        # starting with the first. 1 keeps the historical every-step behavior.
        self._finite_check_interval = max(1, int(finite_check_interval))
        self._finite_check_clock = 0

        self.cfg = deepcopy(cfg)
        super().__init__(env=env)
        self._episode_sums = dict()
        for term_name in self._term_names:
            self._episode_sums[term_name] = np.zeros(self.num_envs, dtype=np.float32)
        self._reward_buf = np.zeros(self.num_envs, dtype=np.float32)
        self._step_reward = np.zeros((self.num_envs, len(self._term_names)), dtype=np.float32)

    def __str__(self) -> str:
        msg = f"<RewardManager> contains {len(self._term_names)} active terms.\n"
        table = PrettyTable()
        table.title = "Active Reward Terms"
        table.field_names = ["Index", "Name", "Weight"]
        table.align["Name"] = "l"
        table.align["Weight"] = "r"
        for index, (name, term_cfg) in enumerate(
            zip(self._term_names, self._term_cfgs, strict=False)
        ):
            table.add_row([index, name, term_cfg.weight])
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
        for key in self._episode_sums.keys():
            episodic_sum_avg = float(np.mean(self._episode_sums[key][env_ids]))
            extras["Episode_Reward/" + key] = episodic_sum_avg / self._env.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        return extras

    def compute(self, dt: float) -> np.ndarray:
        if not np.isfinite(dt) or (self._scale_by_dt and dt <= 0.0):
            raise ValueError(f"RewardManager received invalid dt {dt}.")
        self._reward_buf[:] = 0.0
        scale = dt if self._scale_by_dt else 1.0
        # Increment before the term loop so a raised NaN still advances the
        # cadence; clock 1 (first compute) always checks.
        self._finite_check_clock += 1
        check_finite = (self._finite_check_clock - 1) % self._finite_check_interval == 0
        for term_idx, (name, term_cfg) in enumerate(
            zip(self._term_names, self._term_cfgs, strict=False)
        ):
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue
            value = term_cfg.func(self._env, **term_cfg.params)
            self._check_term_shape(name, value)
            if check_finite:
                self._check_term_finite(name, value)
            value = value * term_cfg.weight * scale
            self._reward_buf += value
            self._episode_sums[name] += value
            self._step_reward[:, term_idx] = value / scale
        return self._reward_buf

    def step_reward_extras(self) -> dict[str, float]:
        """Per-term log entries of the latest ``compute()`` call.

        Returns ``reward/<term>`` -> mean weighted reward rate across envs
        (raw_value * weight, before dt scaling), mirroring the legacy envs'
        per-step reward log format.
        """
        return {
            f"reward/{name}": float(np.mean(self._step_reward[:, term_idx]))
            for term_idx, name in enumerate(self._term_names)
        }

    def get_active_iterable_terms(self, env_idx: int) -> list[tuple[str, list[float]]]:
        terms = []
        for idx, name in enumerate(self._term_names):
            terms.append((name, [self._step_reward[env_idx, idx].item()]))
        return terms

    def get_term_cfg(self, term_name: str) -> RewardTermCfg:
        if term_name not in self._term_names:
            raise ValueError(f"Term '{term_name}' not found in active terms.")
        return self._term_cfgs[self._term_names.index(term_name)]

    def _prepare_terms(self) -> None:
        for term_name, term_cfg in self.cfg.items():
            if term_cfg is None:
                print(f"term: {term_name} set to None, skipping...")
                continue
            if not np.isfinite(term_cfg.weight):
                raise ValueError(
                    f"RewardManager term '{term_name}' has non-finite weight {term_cfg.weight}."
                )
            self._resolve_common_term_cfg(term_name, term_cfg)
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
                self._class_term_cfgs.append(term_cfg)
