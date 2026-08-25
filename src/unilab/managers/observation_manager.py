# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), src/mjlab/managers/observation_manager.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and UniLab contracts; licensed under Apache-2.0.
"""Observation manager for computing observations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
from prettytable import PrettyTable

from unilab.base.config_overrides import (
    CONFIG_MAPPING_POLICY_KEY,
    MANAGER_TERM_MAPPING_POLICY,
)
from unilab.managers._buffers import CircularBuffer, DelayBuffer
from unilab.managers._noise import noise_cfg, noise_model
from unilab.managers._noise.noise_cfg import NoiseCfg, NoiseModelCfg
from unilab.managers.manager_base import ManagerBase, ManagerTermBaseCfg
from unilab.utils.term_profiling import (
    profile_term,  # PROFILING_TEMP (#1293, TODO: remove after #1292)
)

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


@dataclass
class ObservationTermCfg(ManagerTermBaseCfg):
    """Configuration for an observation term.

    Processing pipeline: compute → noise → clip → scale → delay → history.
    Delay models sensor latency. History provides temporal context. Both are optional
    and can be combined.
    """

    noise: NoiseCfg | NoiseModelCfg | None = None
    """Noise model to apply to the observation."""

    clip: tuple[float, float] | None = None
    """Range (min, max) to clip the observation values."""

    scale: tuple[float, ...] | float | np.ndarray | None = None
    """Scaling factor(s) to multiply the observation by."""

    delay_min_lag: int = 0
    """Minimum lag (in steps) for delayed observations. Lag sampled uniformly from
  [min_lag, max_lag]. Convert to ms: lag * (1000 / control_hz)."""

    delay_max_lag: int = 0
    """Maximum lag (in steps) for delayed observations. Use min=max for constant delay."""

    delay_per_env: bool = True
    """If True, each environment samples its own lag. If False, all environments share
  the same lag at each step."""

    delay_hold_prob: float = 0.0
    """Probability of reusing the previous lag instead of resampling. Useful for
  temporally correlated latency patterns."""

    delay_update_period: int = 0
    """Resample lag every N steps (models multi-rate sensors). If 0, update every step."""

    delay_per_env_phase: bool = True
    """If True and update_period > 0, stagger update timing across envs to avoid
  synchronized resampling."""

    history_length: int = 0
    """Number of past observations to keep in history. 0 = no history."""

    flatten_history_dim: bool = True
    """Whether to flatten the history dimension into observation.

  When True and concatenate_terms=True, uses term-major ordering:
  [A_t0, A_t1, ..., A_tH-1, B_t0, B_t1, ..., B_tH-1, ...]
  See docs/source/observation.rst for details on ordering."""


@dataclass
class ObservationGroupCfg:
    """Configuration for an observation group.

    An observation group bundles multiple observation terms together. Groups are
    typically used to separate observations for different purposes (e.g., "actor"
    for the actor, "critic" for the value function).
    """

    terms: dict[str, ObservationTermCfg | None] = field(
        metadata={CONFIG_MAPPING_POLICY_KEY: MANAGER_TERM_MAPPING_POLICY}
    )
    """Dictionary mapping term names to their configurations."""

    concatenate_terms: bool = True
    """Whether to concatenate all terms into a single tensor. If False, returns
  a dict mapping term names to their individual tensors."""

    concatenate_dim: int = -1
    """Dimension along which to concatenate terms. Default -1 (last dimension)."""

    enable_corruption: bool = False
    """Whether to apply noise corruption to observations. Set to True during
  training for domain randomization, False during evaluation."""

    history_length: int | None = None
    """Group-level history length override. If set, applies to all terms in
  this group. If None, each term uses its own ``history_length`` setting."""

    flatten_history_dim: bool = True
    """Whether to flatten history into the observation dimension. If True,
  observations have shape ``(num_envs, obs_dim * history_length)``. If False,
  shape is ``(num_envs, history_length, obs_dim)``."""

    nan_policy: Literal["disabled", "warn", "sanitize", "error"] = "error"
    """NaN/Inf handling policy for observations in this group.

  - 'disabled': No checks (explicit opt-out)
  - 'warn': Log warning with term name and env IDs, then sanitize (debugging)
  - 'sanitize': Silent sanitization to 0.0 like reward manager (safe for production)
  - 'error': Raise ValueError on NaN/Inf (strict development mode)
  """

    nan_check_per_term: bool = True
    """If True, check each observation term individually to identify NaN source.
  If False, check only the final concatenated output (faster but less informative).
  Only applies when nan_policy != 'disabled'."""


class ObservationManager(ManagerBase):
    """Manages observation computation for the environment.

    The observation manager computes observations from multiple terms organized
    into groups. Each term can have noise, clipping, scaling, delay, and history
    applied. Groups can optionally concatenate their terms into a single tensor.
    """

    def __init__(self, cfg: dict[str, ObservationGroupCfg | None], env: ManagerBasedRlEnv):
        self.cfg = deepcopy(cfg)
        super().__init__(env=env)

        self._group_obs_dim: dict[str, tuple[int, ...] | list[tuple[int, ...]]] = dict()

        for group_name, group_term_dims in self._group_obs_term_dim.items():
            if self._group_obs_concatenate[group_name]:
                term_dims = np.stack([np.asarray(dims) for dims in group_term_dims], axis=0)
                if len(term_dims.shape) > 1:
                    if self._group_obs_concatenate_dim[group_name] >= 0:
                        dim = self._group_obs_concatenate_dim[group_name] - 1
                    else:
                        dim = self._group_obs_concatenate_dim[group_name]
                    dim_sum = np.sum(term_dims[:, dim], axis=0)
                    term_dims[0, dim] = dim_sum
                    term_dims = term_dims[0]
                else:
                    term_dims = np.sum(term_dims, axis=0)
                self._group_obs_dim[group_name] = tuple(term_dims.tolist())
            else:
                self._group_obs_dim[group_name] = group_term_dims

        self._obs_buffer: dict[str, np.ndarray | dict[str, np.ndarray]] | None = None

    def __str__(self) -> str:
        msg = f"<ObservationManager> contains {len(self._group_obs_term_names)} groups.\n"
        for group_name, group_dim in self._group_obs_dim.items():
            table = PrettyTable()
            table.title = f"Active Observation Terms in Group: '{group_name}'"
            if self._group_obs_concatenate[group_name]:
                table.title += f" (shape: {group_dim})"  # type: ignore
            table.field_names = ["Index", "Name", "Shape"]
            table.align["Name"] = "l"
            obs_terms = zip(
                self._group_obs_term_names[group_name],
                self._group_obs_term_dim[group_name],
                self._group_obs_term_cfgs[group_name],
                strict=False,
            )
            for index, (name, dims, term_cfg) in enumerate(obs_terms):
                if term_cfg.history_length > 0 and term_cfg.flatten_history_dim:
                    # Flattened history: show (9,) ← 3×(3,)
                    original_size = int(np.prod(dims)) // term_cfg.history_length
                    original_shape = (original_size,) if len(dims) == 1 else dims[1:]
                    shape_str = f"{dims}  ← {term_cfg.history_length}×{original_shape}"
                else:
                    shape_str = str(tuple(dims))
                table.add_row([index, name, shape_str])
            msg += str(table.get_string())
            msg += "\n"
        return msg

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        terms = []

        if self._obs_buffer is None:
            self.compute()
        assert self._obs_buffer is not None
        obs_buffer: dict[str, np.ndarray | dict[str, np.ndarray]] = self._obs_buffer

        for group_name, _ in self.group_obs_dim.items():
            if not self.group_obs_concatenate[group_name]:
                buffers = obs_buffer[group_name]
                assert isinstance(buffers, dict)
                for name, term in buffers.items():
                    terms.append((group_name + "-" + name, term[env_idx].tolist()))
                continue

            idx = 0
            data = obs_buffer[group_name]
            assert isinstance(data, np.ndarray)
            for name, shape in zip(
                self._group_obs_term_names[group_name],
                self._group_obs_term_dim[group_name],
                strict=False,
            ):
                data_length = np.prod(shape)
                term = data[env_idx, idx : idx + data_length]
                terms.append((group_name + "-" + name, term.tolist()))
                idx += data_length

        return terms

    # Properties.

    @property
    def active_terms(self) -> dict[str, list[str]]:
        return self._group_obs_term_names

    @property
    def group_obs_dim(self) -> dict[str, tuple[int, ...] | list[tuple[int, ...]]]:
        return self._group_obs_dim

    @property
    def group_obs_term_dim(self) -> dict[str, list[tuple[int, ...]]]:
        return self._group_obs_term_dim

    @property
    def group_obs_concatenate(self) -> dict[str, bool]:
        return self._group_obs_concatenate

    # Methods.

    def get_term_cfg(self, group_name: str, term_name: str) -> ObservationTermCfg:
        if group_name not in self._group_obs_term_names:
            raise ValueError(f"Group '{group_name}' not found in active groups.")
        if term_name not in self._group_obs_term_names[group_name]:
            raise ValueError(f"Term '{term_name}' not found in group '{group_name}'.")
        index = self._group_obs_term_names[group_name].index(term_name)
        return self._group_obs_term_cfgs[group_name][index]

    def reset(self, env_ids: np.ndarray | slice | None = None) -> dict[str, float]:
        # Invalidate cache since reset envs will have different observations.
        self._obs_buffer = None

        for group_name, group_cfg in self._group_obs_class_term_cfgs.items():
            for term_cfg in group_cfg:
                term_cfg.func.reset(env_ids=env_ids)
            for term_name in self._group_obs_term_names[group_name]:
                batch_ids = env_ids
                if term_name in self._group_obs_term_delay_buffer[group_name]:
                    self._group_obs_term_delay_buffer[group_name][term_name].reset(
                        batch_ids=batch_ids
                    )
                if term_name in self._group_obs_term_history_buffer[group_name]:
                    self._group_obs_term_history_buffer[group_name][term_name].reset(
                        batch_ids=batch_ids
                    )
        for group_mods in self._group_obs_class_instances.values():
            for mod in group_mods.values():
                mod.reset(env_ids=env_ids)
        return {}

    def _check_and_handle_nans(
        self,
        tensor: np.ndarray,
        context: str,
        policy: str,
        env_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """Check for NaN/Inf and handle according to policy.

        Args:
          tensor: Observation tensor to check. On the reset path this holds only
            the reset rows; pass env_ids so diagnostics report real env indices.
          context: Context string for error/warning messages (e.g., "actor/base_lin_vel").
          policy: NaN handling policy ("disabled", "warn", "sanitize", "error").
          env_ids: Optional mapping from tensor rows to env indices (reset path).

        Returns:
          The tensor, potentially sanitized depending on policy.

        Raises:
          ValueError: If policy is "error" and NaN/Inf detected.
        """
        if policy == "disabled":
            return tensor

        # The overwhelmingly common path is finite.  Use one full tensor scan
        # here instead of separate isnan/isinf scans for every term.  On the
        # exceptional path the same allocation is reused as the invalid mask
        # so diagnostics and sanitization retain their existing semantics.
        finite = np.isfinite(tensor)
        if finite.all():
            return tensor
        invalid = np.logical_not(finite, out=finite)
        invalid_values = tensor[invalid]
        has_nan = np.isnan(invalid_values).any()
        has_inf = np.isinf(invalid_values).any()

        def _row_env_ids(mask: np.ndarray) -> list[int]:
            rows = np.flatnonzero(np.asarray(mask, dtype=bool))
            if env_ids is not None:
                rows = np.asarray(env_ids)[rows]
            result: list[int] = rows.tolist()
            return result

        if policy == "error":
            nan_mask = np.asarray(invalid.reshape(tensor.shape[0], -1).any(axis=1))
            nan_env_ids = _row_env_ids(nan_mask)
            invalid_kind = (
                "NaN"
                if has_nan and not has_inf
                else "Inf"
                if has_inf and not has_nan
                else "NaN/Inf"
            )
            raise ValueError(
                f"{invalid_kind} detected in ObservationManager term '{context}' "
                f"for environments: {nan_env_ids[:10]}"
            )

        if policy == "warn":
            nan_mask = np.asarray(invalid.reshape(tensor.shape[0], -1).any(axis=1))
            nan_env_ids = _row_env_ids(nan_mask)
            print(
                f"[ObservationManager] NaN/Inf in '{context}' "
                f"(envs: {nan_env_ids[:5]}). Sanitizing to 0."
            )

        # Sanitize (applies to both "warn" and "sanitize" policies).
        return np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    def compute(
        self,
        update_history: bool = False,
        env_ids: np.ndarray | None = None,
    ) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
        """Compute observations for all groups.

        With env_ids=None (the per-step path), history and delay buffers advance
        for all envs and the returned arrays cover the full batch. With env_ids
        (the reset path), only the reset envs' buffers receive their post-reset
        frame (a backfill); other envs' buffers, delay schedules, and lag draws
        are untouched, so a partial reset does not advance their observation
        timelines. The returned arrays then hold only the reset rows, in env_ids
        order, and the observation cache is left invalidated (the next per-step
        compute refreshes it); noise is still drawn with full-batch shapes so
        the shared RNG stream matches the full-batch implementation exactly.
        """
        if env_ids is not None and not update_history:
            raise ValueError("env_ids is only meaningful with update_history=True.")
        # Return cached observations if not updating and cache exists.
        # This prevents double-pushing to delay buffers when compute() is called
        # multiple times per control step (e.g., in get_observations() after step()).
        if not update_history and self._obs_buffer is not None:
            return self._obs_buffer

        obs_buffer: dict[str, np.ndarray | dict[str, np.ndarray]] = dict()
        for group_name in self._group_obs_term_names:
            obs_buffer[group_name] = self.compute_group(group_name, update_history, env_ids)
        if env_ids is None:
            self._obs_buffer = obs_buffer
        return obs_buffer

    def compute_group(
        self,
        group_name: str,
        update_history: bool = False,
        env_ids: np.ndarray | None = None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        group_cfg = self.cfg[group_name]
        if group_cfg is None:
            raise KeyError(f"Observation group '{group_name}' is disabled.")
        group_term_names = self._group_obs_term_names[group_name]
        group_obs: dict[str, np.ndarray] = {}
        obs_terms = zip(group_term_names, self._group_obs_term_cfgs[group_name], strict=False)
        # Reset path (issue #1259 R2): when no term in this group uses delay or
        # history buffers, everything downstream of the term call is row
        # independent, so only the reset rows are processed. Term calls and
        # noise stay full-batch: term funcs are contracted to return
        # (num_envs, ...) and full-shape noise draws keep the shared RNG stream
        # and per-row noise values identical to the full-batch path.
        row_scoped = env_ids is not None and not self._group_obs_temporal[group_name]
        # PROFILING_TEMP (#1293, TODO: remove after #1292)
        phase = "reset" if env_ids is not None else "step"
        for term_name, term_cfg in obs_terms:
            # PROFILING_TEMP (#1293, TODO: remove after #1292)
            with profile_term(f"obs/{group_name}/{term_name}|{phase}"):
                obs = term_cfg.func(self._env, **term_cfg.params)
            if not isinstance(obs, np.ndarray):
                raise TypeError(
                    f"ObservationManager term '{group_name}/{term_name}' returned "
                    f"{type(obs).__name__}, expected np.ndarray."
                )
            if obs.ndim < 2 or obs.shape[0] != self.num_envs:
                raise ValueError(
                    f"ObservationManager term '{group_name}/{term_name}' returned shape "
                    f"{obs.shape}, expected (num_envs, ...) with num_envs={self.num_envs}."
                )
            # PROFILING_TEMP (#1293, TODO: remove after #1292): manager-level
            # per-term post-processing (copy/noise/clip/scale/nan check).
            with profile_term(f"obs_post/{group_name}/{term_name}|{phase}"):
                fresh = False
                if isinstance(term_cfg.noise, noise_cfg.NoiseCfg):
                    # NoiseCfg.apply always returns a newly allocated array.
                    obs = term_cfg.noise.apply(obs, rng=self._env.rng)
                    fresh = True
                elif isinstance(term_cfg.noise, noise_cfg.NoiseModelCfg):
                    # NoiseModel.__call__ likewise returns a new array.
                    obs = self._group_obs_class_instances[group_name][term_name](obs)
                    fresh = True
                if not row_scoped and not fresh:
                    # Terms may return backend/command-owned buffers; copy before the
                    # in-place clip/scale below. Skipped when noise already produced
                    # a fresh array (issue #1296).
                    obs = obs.copy()
                if row_scoped:
                    # Fresh row copy; safe for the in-place clip/scale below.
                    obs = obs[env_ids]
                if term_cfg.clip:
                    np.clip(obs, term_cfg.clip[0], term_cfg.clip[1], out=obs)
                if term_cfg.scale is not None:
                    scale = term_cfg.scale
                    assert isinstance(scale, np.ndarray)
                    np.multiply(obs, scale, out=obs)

                # Check for NaN/Inf before delay/history buffers (per-term checking).
                if group_cfg.nan_check_per_term and group_cfg.nan_policy != "disabled":
                    obs = self._check_and_handle_nans(
                        obs,
                        context=f"{group_name}/{term_name}",
                        policy=group_cfg.nan_policy,
                        env_ids=env_ids if row_scoped else None,
                    )

            if term_cfg.delay_max_lag > 0:
                delay_buffer = self._group_obs_term_delay_buffer[group_name][term_name]
                if env_ids is None or not delay_buffer.is_initialized:
                    delay_buffer.append(obs)
                    obs = delay_buffer.compute()
                else:
                    delay_buffer.backfill(obs, env_ids)
                    obs = delay_buffer.peek()
            if term_cfg.history_length > 0:
                circular_buffer = self._group_obs_term_history_buffer[group_name][term_name]
                if env_ids is None or not circular_buffer.is_initialized:
                    if update_history or not circular_buffer.is_initialized:
                        circular_buffer.append(obs)
                else:
                    circular_buffer.backfill(obs, env_ids)

                if term_cfg.flatten_history_dim:
                    group_obs[term_name] = circular_buffer.buffer.reshape(self._env.num_envs, -1)
                else:
                    group_obs[term_name] = circular_buffer.buffer
            else:
                group_obs[term_name] = obs

        # PROFILING_TEMP (#1293, TODO: remove after #1292): group-level
        # post-processing (group nan check / concatenate / reset row slice).
        with profile_term(f"obs_group_post/{group_name}|{phase}"):
            # Final NaN check for non-per-term checking.
            if not group_cfg.nan_check_per_term and group_cfg.nan_policy != "disabled":
                if self._group_obs_concatenate[group_name]:
                    # Will check after concatenation below.
                    pass
                else:
                    for term_name in group_obs:
                        group_obs[term_name] = self._check_and_handle_nans(
                            group_obs[term_name],
                            context=f"{group_name}/{term_name}",
                            policy=group_cfg.nan_policy,
                            env_ids=env_ids if row_scoped else None,
                        )

            if self._group_obs_concatenate[group_name]:
                result = np.concatenate(
                    list(group_obs.values()), axis=self._group_obs_concatenate_dim[group_name]
                )
                # Final check for concatenated result (non-per-term checking).
                if not group_cfg.nan_check_per_term and group_cfg.nan_policy != "disabled":
                    result = self._check_and_handle_nans(
                        result,
                        context=group_name,
                        policy=group_cfg.nan_policy,
                        env_ids=env_ids if row_scoped else None,
                    )
            else:
                result = group_obs

            if env_ids is not None and not row_scoped:
                # Groups with delay/history terms ran the full-batch pipeline above
                # (buffer readout stays full-batch); slice the reset rows to match
                # the reset-path return contract.
                if isinstance(result, dict):
                    result = {name: values[env_ids] for name, values in result.items()}
                else:
                    result = result[env_ids]

        return result

    def _prepare_terms(self) -> None:
        self._group_obs_term_names: dict[str, list[str]] = dict()
        self._group_obs_term_dim: dict[str, list[tuple[int, ...]]] = dict()
        self._group_obs_term_cfgs: dict[str, list[ObservationTermCfg]] = dict()
        self._group_obs_class_term_cfgs: dict[str, list[ObservationTermCfg]] = dict()
        self._group_obs_concatenate: dict[str, bool] = dict()
        self._group_obs_concatenate_dim: dict[str, int] = dict()
        self._group_obs_class_instances: dict[str, dict[str, noise_model.NoiseModel]] = {}
        self._group_obs_term_delay_buffer: dict[str, dict[str, DelayBuffer]] = dict()
        self._group_obs_term_history_buffer: dict[str, dict[str, CircularBuffer]] = dict()
        # Whether any term in the group uses delay/history buffers. Groups
        # without temporal terms can be row-scoped on the reset path.
        self._group_obs_temporal: dict[str, bool] = dict()

        for group_name, group_cfg in self.cfg.items():
            if group_cfg is None:
                print(f"group: {group_name} set to None, skipping...")
                continue

            if not any(t is not None for t in group_cfg.terms.values()):
                print(f"group: {group_name} has no active terms, skipping...")
                continue

            if group_cfg.nan_policy not in ("disabled", "warn", "sanitize", "error"):
                raise ValueError(
                    f"Observation group '{group_name}' has unsupported NaN policy "
                    f"'{group_cfg.nan_policy}'."
                )
            if group_cfg.history_length is not None and group_cfg.history_length < 0:
                raise ValueError(
                    f"Observation group '{group_name}' has negative history_length "
                    f"{group_cfg.history_length}."
                )

            self._group_obs_term_names[group_name] = list()
            self._group_obs_term_dim[group_name] = list()
            self._group_obs_term_cfgs[group_name] = list()
            self._group_obs_class_term_cfgs[group_name] = list()
            self._group_obs_class_instances[group_name] = {}
            group_entry_delay_buffer: dict[str, DelayBuffer] = dict()
            group_entry_history_buffer: dict[str, CircularBuffer] = dict()

            self._group_obs_concatenate[group_name] = group_cfg.concatenate_terms
            self._group_obs_concatenate_dim[group_name] = (
                group_cfg.concatenate_dim + 1
                if group_cfg.concatenate_dim >= 0
                else group_cfg.concatenate_dim
            )

            for term_name, term_cfg in group_cfg.terms.items():
                if term_cfg is None:
                    print(f"term: {term_name} set to None, skipping...")
                    continue

                if term_cfg.delay_min_lag < 0 or term_cfg.delay_max_lag < term_cfg.delay_min_lag:
                    raise ValueError(
                        f"ObservationManager term '{group_name}/{term_name}' has invalid "
                        f"delay range [{term_cfg.delay_min_lag}, {term_cfg.delay_max_lag}]."
                    )
                if term_cfg.history_length < 0:
                    raise ValueError(
                        f"ObservationManager term '{group_name}/{term_name}' has negative "
                        f"history_length {term_cfg.history_length}."
                    )
                if term_cfg.clip is not None and term_cfg.clip[0] > term_cfg.clip[1]:
                    raise ValueError(
                        f"ObservationManager term '{group_name}/{term_name}' has invalid "
                        f"clip range {term_cfg.clip}."
                    )

                # NOTE: This deepcopy is important to avoid cross-group contamination of term
                # configs.
                term_cfg = deepcopy(term_cfg)
                self._resolve_common_term_cfg(term_name, term_cfg)

                if not group_cfg.enable_corruption:
                    term_cfg.noise = None
                if group_cfg.history_length is not None:
                    term_cfg.history_length = group_cfg.history_length
                    term_cfg.flatten_history_dim = group_cfg.flatten_history_dim
                self._group_obs_term_names[group_name].append(term_name)
                self._group_obs_term_cfgs[group_name].append(term_cfg)
                if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
                    self._group_obs_class_term_cfgs[group_name].append(term_cfg)

                initial_obs = term_cfg.func(self._env, **term_cfg.params)
                if not isinstance(initial_obs, np.ndarray):
                    raise TypeError(
                        f"ObservationManager term '{group_name}/{term_name}' returned "
                        f"{type(initial_obs).__name__}, expected np.ndarray."
                    )
                if initial_obs.ndim < 2 or initial_obs.shape[0] != self.num_envs:
                    raise ValueError(
                        f"ObservationManager term '{group_name}/{term_name}' returned shape "
                        f"{initial_obs.shape}, expected (num_envs, ...) with "
                        f"num_envs={self.num_envs}."
                    )
                obs_dims = tuple(initial_obs.shape)

                if term_cfg.scale is not None:
                    term_cfg.scale = np.asarray(term_cfg.scale, dtype=np.float32).copy()

                if term_cfg.noise is not None and isinstance(
                    term_cfg.noise, noise_cfg.NoiseModelCfg
                ):
                    noise_model_cls = term_cfg.noise.class_type
                    if not issubclass(noise_model_cls, noise_model.NoiseModel):
                        raise TypeError(
                            f"ObservationManager term '{group_name}/{term_name}' noise model "
                            f"{noise_model_cls} is not a NoiseModel subclass."
                        )
                    self._group_obs_class_instances[group_name][term_name] = noise_model_cls(
                        term_cfg.noise, num_envs=self._env.num_envs, rng=self._env.rng
                    )

                if term_cfg.delay_max_lag > 0:
                    group_entry_delay_buffer[term_name] = DelayBuffer(
                        min_lag=term_cfg.delay_min_lag,
                        max_lag=term_cfg.delay_max_lag,
                        batch_size=self._env.num_envs,
                        per_env=term_cfg.delay_per_env,
                        hold_prob=term_cfg.delay_hold_prob,
                        update_period=term_cfg.delay_update_period,
                        per_env_phase=term_cfg.delay_per_env_phase,
                        generator=self._env.rng,
                    )

                if term_cfg.history_length > 0:
                    group_entry_history_buffer[term_name] = CircularBuffer(
                        max_len=term_cfg.history_length,
                        batch_size=self._env.num_envs,
                    )
                    old_dims = list(obs_dims)
                    old_dims.insert(1, term_cfg.history_length)
                    obs_dims = tuple(old_dims)
                    if term_cfg.flatten_history_dim:
                        obs_dims = (obs_dims[0], int(np.prod(obs_dims[1:])))

                self._group_obs_term_dim[group_name].append(obs_dims[1:])

            self._group_obs_term_delay_buffer[group_name] = group_entry_delay_buffer
            self._group_obs_term_history_buffer[group_name] = group_entry_history_buffer
            self._group_obs_temporal[group_name] = any(
                term_cfg.delay_max_lag > 0 or term_cfg.history_length > 0
                for term_cfg in self._group_obs_term_cfgs[group_name]
            )
