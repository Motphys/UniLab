# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), src/mjlab/managers/reward_manager.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and UniLab contracts; licensed under Apache-2.0.
"""Reward manager for computing reward signals."""

from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
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

    reward_pack_names: tuple[str, ...] = ()
    """Ordered outputs when ``func`` evaluates multiple terms in one carrier read."""


class RewardManager(ManagerBase):
    """Manages reward computation by aggregating weighted reward terms.

    Reward Scaling Behavior:
      By default, rewards are scaled by the environment step duration (dt). This
      normalizes cumulative episodic rewards across different simulation frequencies.
      The scaling can be disabled via the ``scale_by_dt`` parameter.

      When ``scale_by_dt=True`` (default):
        - ``reward_buf`` (returned by ``compute()``) = raw_value * weight * dt

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
    ):
        self._term_names: list[str] = list()
        self._term_cfgs: list[RewardTermCfg] = list()
        self._class_term_cfgs: list[RewardTermCfg] = list()
        self._scale_by_dt = scale_by_dt

        self.cfg = deepcopy(cfg)
        super().__init__(env=env)
        self._device = getattr(env, "device", torch.device("cpu"))
        self._reward_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self._device)
        self._step_reward = torch.zeros(
            (self.num_envs, len(self._term_names)), dtype=torch.float32, device=self._device
        )

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

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        return {}

    def compute(self, dt: float) -> torch.Tensor:
        timing = getattr(self, "last_step_timing_ms", None)
        if timing is None:
            timing = {}
            self.last_step_timing_ms = timing
        timing.clear()
        if not math.isfinite(dt) or (self._scale_by_dt and dt <= 0.0):
            raise ValueError(f"RewardManager received invalid dt {dt}.")
        reset_started = time.perf_counter()
        self._reward_buf[:] = 0.0
        reset_ms = time.perf_counter() - reset_started
        dispatch_ms = 0.0
        aggregation_ms = 0.0
        scale = dt if self._scale_by_dt else 1.0
        packed_targets = {
            packed_name
            for term_cfg in self._term_cfgs
            for packed_name in term_cfg.reward_pack_names
        }
        for term_idx, (name, term_cfg) in enumerate(
            zip(self._term_names, self._term_cfgs, strict=False)
        ):
            if term_cfg.reward_pack_names or name in packed_targets:
                continue
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue
            term_timing = getattr(term_cfg.func, "last_step_timing_ms", None)
            if term_timing is not None:
                term_timing.clear()
            dispatch_started = time.perf_counter()
            value = self._compute_term(name, term_cfg, validate=False)
            dispatch_ms += time.perf_counter() - dispatch_started
            if term_timing is not None:
                timing.update(term_timing)
            aggregation_started = time.perf_counter()
            weighted = value * float(term_cfg.weight) * scale
            self._reward_buf += weighted
            self._step_reward[:, term_idx] = weighted / scale
            aggregation_ms += time.perf_counter() - aggregation_started
        for term_idx, term_cfg in enumerate(self._term_cfgs):
            if not term_cfg.reward_pack_names:
                continue
            dispatch_started = time.perf_counter()
            pack_name = self._term_names[term_idx]
            pack_value = term_cfg.func(self._env, **term_cfg.params)
            if not isinstance(pack_value, torch.Tensor):
                raise TypeError(
                    f"RewardManager pack term '{pack_name}' must return a torch.Tensor; "
                    f"got {type(pack_value).__name__}"
                )
            if pack_value.dtype != torch.float32:
                raise TypeError(
                    f"RewardManager pack term '{pack_name}' returned dtype "
                    f"{pack_value.dtype}, expected float32."
                )
            if pack_value.device != self._device:
                raise ValueError(
                    f"RewardManager pack term '{pack_name}' returned device "
                    f"{pack_value.device}, expected {self._device}."
                )
            packed = pack_value
            dispatch_ms += time.perf_counter() - dispatch_started
            if packed.ndim != 2 or packed.shape[1] != len(term_cfg.reward_pack_names):
                raise ValueError(
                    f"RewardManager pack term '{self._term_names[term_idx]}' returned "
                    f"shape {tuple(packed.shape)}; expected "
                    f"({self.num_envs}, {len(term_cfg.reward_pack_names)})."
                )
            aggregation_started = time.perf_counter()
            for offset, packed_name in enumerate(term_cfg.reward_pack_names):
                target_idx = self._term_names.index(packed_name)
                target_cfg = self._term_cfgs[target_idx]
                weighted = packed[:, offset] * float(target_cfg.weight) * scale
                self._reward_buf += weighted
                self._step_reward[:, target_idx] = weighted / scale
            aggregation_ms += time.perf_counter() - aggregation_started
        finite_started = time.perf_counter()
        finite = bool(torch.isfinite(self._reward_buf).all())
        finite_ms = time.perf_counter() - finite_started
        if not finite:
            finite = torch.isfinite(self._step_reward)
            for term_idx, name in enumerate(self._term_names):
                if not bool(finite[:, term_idx].all()):
                    value = self._step_reward[:, term_idx]
                    has_nan = bool(torch.isnan(value).any())
                    has_inf = bool(torch.isinf(value).any())
                    invalid_kind = "NaN/Inf" if has_nan and has_inf else "NaN" if has_nan else "Inf"
                    invalid_rows = torch.nonzero(~finite[:, term_idx]).flatten()[:10]
                    raise ValueError(
                        f"RewardManager term '{name}' returned {invalid_kind} for "
                        f"environments {invalid_rows.tolist()}."
                    )
            raise ValueError("RewardManager returned a non-finite reward.")
        timing["update_state_reward_term_dispatch_ms"] = dispatch_ms * 1000.0
        timing["update_state_reward_aggregation_ms"] = aggregation_ms * 1000.0
        timing["update_state_reward_finite_validation_ms"] = finite_ms * 1000.0
        timing["update_state_reward_manager_residual_ms"] = reset_ms * 1000.0
        return self._reward_buf

    def step_reward_extras(self) -> dict[str, float]:
        """Per-term log entries of the latest ``compute()`` call.

        Returns ``reward/<term>`` -> mean weighted reward rate across envs
        (raw_value * weight, before dt scaling), mirroring the legacy envs'
        per-step reward log format.
        """
        if not self._term_names:
            return {}
        means = self.step_reward_means.detach()
        if means.numel() == 0:
            return {}
        host_means = means.cpu().tolist()
        return {
            f"reward/{name}": float(mean)
            for name, mean in zip(self._term_names, host_means, strict=True)
        }

    @property
    def step_reward_names(self) -> tuple[str, ...]:
        """Active per-term log names in Manager declaration order."""
        return tuple(self._term_names)

    @property
    def step_reward_means(self) -> torch.Tensor:
        """Latest per-term weighted means on the Manager device."""
        if not self._term_names:
            return torch.empty(0, dtype=torch.float32, device=self._device)
        return self._step_reward.mean(dim=0)

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
            if not math.isfinite(term_cfg.weight):
                raise ValueError(
                    f"RewardManager term '{term_name}' has non-finite weight {term_cfg.weight}."
                )
            self._resolve_common_term_cfg(term_name, term_cfg)
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
                self._class_term_cfgs.append(term_cfg)

    def _compute_term(
        self, name: str, term_cfg: RewardTermCfg, *, validate: bool = True
    ) -> torch.Tensor:
        value = term_cfg.func(self._env, **term_cfg.params)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"RewardManager term '{name}' must return torch.Tensor, got {type(value).__name__}."
            )
        if value.dtype != torch.float32:
            raise TypeError(
                f"RewardManager term '{name}' returned dtype {value.dtype}, expected float32."
            )
        if value.device != self._device:
            raise ValueError(
                f"RewardManager term '{name}' returned device {value.device}, "
                f"expected {self._device}."
            )
        if getattr(term_cfg.func, "returns_transient_tensor", False):
            result = value
        else:
            result = value.clone()
        if result.shape != (self.num_envs,):
            raise ValueError(
                f"RewardManager term '{name}' returned shape {tuple(result.shape)}; "
                f"expected ({self.num_envs},)."
            )
        if validate and not bool(torch.isfinite(result).all()):
            has_nan = bool(torch.isnan(result).any())
            has_inf = bool(torch.isinf(result).any())
            invalid_kind = "NaN/Inf" if has_nan and has_inf else "NaN" if has_nan else "Inf"
            invalid_rows = torch.nonzero(~torch.isfinite(result)).flatten()[:10]
            raise ValueError(
                f"RewardManager term '{name}' returned {invalid_kind} for "
                f"environments {invalid_rows.tolist()}."
            )
        return result

    @staticmethod
    def _log_mean(values: torch.Tensor) -> float:
        if values.numel() == 0:
            return 0.0
        return float(values.mean().item())
