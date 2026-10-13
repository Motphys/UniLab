# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), src/mjlab/managers/observation_manager.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and UniLab contracts; licensed under Apache-2.0.
"""Observation manager for computing observations."""

from __future__ import annotations

import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Sequence, cast

import torch
from prettytable import PrettyTable

from unilab.base.config_overrides import (
    CONFIG_MAPPING_POLICY_KEY,
    MANAGER_TERM_MAPPING_POLICY,
)
from unilab.managers._buffers import CircularBuffer, DelayBuffer
from unilab.managers._noise import noise_cfg, noise_model
from unilab.managers._noise.noise_cfg import NoiseCfg, NoiseModelCfg
from unilab.managers.manager_base import ManagerBase, ManagerTermBaseCfg

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

    scale: tuple[float, ...] | float | torch.Tensor | None = None
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


def _freeze_param_value(value):
    """Recursively convert param containers into hashable equivalents.

    Cross-group term sharing keys on ``(func, params)``; params such as
    ``future_steps: [0, 1]`` arrive as lists and would otherwise disable sharing
    for exactly the wide future-window terms that benefit most.
    """
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_param_value(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_param_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_param_value(item) for item in value)
    if isinstance(value, torch.Tensor):
        return ("__tensor__", tuple(value.shape), str(value.dtype), value.flatten().tolist())
    return value


def _shape_product(shape: Sequence[int]) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result


class ObservationManager(ManagerBase):
    """Manages observation computation for the environment.

    The observation manager computes observations from multiple terms organized
    into groups. Each term can have noise, clipping, scaling, delay, and history
    applied. Groups can optionally concatenate their terms into a single tensor.
    """

    def __init__(self, cfg: dict[str, ObservationGroupCfg | None], env: ManagerBasedRlEnv):
        self.cfg = deepcopy(cfg)
        self._device = getattr(env, "device", torch.device("cpu"))
        self._torch_rng = getattr(env, "torch_rng", None)
        if self._torch_rng is not None and self._torch_rng.device != self._device:
            raise ValueError(
                f"ObservationManager Torch RNG device {self._torch_rng.device} "
                f"does not match observations on {self._device}."
            )
        self._torch_generator = self._torch_rng.generator if self._torch_rng is not None else None
        super().__init__(env=env)

        self._group_obs_dim: dict[str, tuple[int, ...] | list[tuple[int, ...]]] = dict()
        self._group_term_noise_carriers: dict[tuple[str, str], list[torch.Tensor]] = dict()
        self._group_term_noise_generation: dict[tuple[str, str], int] = dict()
        self._deferred_validation_boundaries = False
        self._group_finite_valid: dict[str, torch.Tensor] = {}

        for group_name, group_term_dims in self._group_obs_term_dim.items():
            if self._group_obs_concatenate[group_name]:
                term_dims = self._concatenated_group_dim(
                    group_term_dims,
                    concatenate_dim=self._group_obs_concatenate_dim[group_name],
                )
                self._group_obs_dim[group_name] = term_dims
            else:
                self._group_obs_dim[group_name] = group_term_dims

        self._obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] | None = None
        for group_name, group_dim in self._group_obs_dim.items():
            group_tail = group_dim[1:]
            verdict_shape = (self.num_envs, *(size for size in group_tail if isinstance(size, int)))
            self._group_finite_valid[group_name] = torch.ones(
                verdict_shape, dtype=torch.bool, device=self._device
            )

    @property
    def deferred_validation_boundaries(self) -> bool:
        """Whether observation finite publication may defer to completion."""
        return self._deferred_validation_boundaries

    def set_deferred_validation_boundaries(self, enabled: bool) -> None:
        """Move finite publication to an explicit completion boundary."""
        if not isinstance(enabled, bool):
            raise TypeError(
                f"deferred observation validation boundaries must be bool, got {type(enabled)}"
            )
        self._deferred_validation_boundaries = enabled

    def finite_valid(self, group_name: str) -> torch.Tensor:
        """Return the latest device finite verdict for one observation group."""
        try:
            return self._group_finite_valid[group_name]
        except KeyError as exc:
            raise KeyError(f"unknown observation group '{group_name}'") from exc

    def publish_validation_verdict(self) -> None:
        """Fail closed on persistent group finite verdicts."""
        if not self._deferred_validation_boundaries:
            return
        for group_name, verdict in self._group_finite_valid.items():
            if bool(verdict.all()):
                continue
            raise ValueError(f"ObservationManager group '{group_name}' contains NaN or Inf")

    @staticmethod
    def _concatenated_group_dim(
        term_dims: Sequence[Sequence[int]], *, concatenate_dim: int
    ) -> tuple[int, ...]:
        if not term_dims:
            raise ValueError("cannot concatenate an empty observation group")
        if len({len(dims) for dims in term_dims}) != 1:
            raise ValueError(
                f"cannot concatenate observation terms with inconsistent ranks: {term_dims}"
            )
        result = list(term_dims[0])
        resolved_dim = concatenate_dim if concatenate_dim < 0 else concatenate_dim - 1
        if resolved_dim < -len(result) or resolved_dim >= len(result):
            raise IndexError(f"observation concatenate_dim {concatenate_dim} is out of range")
        for dims in term_dims[1:]:
            result[resolved_dim] += dims[resolved_dim]
        return tuple(result)

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
                    original_size = int(torch.tensor(dims).prod().item()) // term_cfg.history_length
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
        obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] = self._obs_buffer

        for group_name, _ in self.group_obs_dim.items():
            if not self.group_obs_concatenate[group_name]:
                buffers = obs_buffer[group_name]
                assert isinstance(buffers, dict)
                for name, term in buffers.items():
                    terms.append((group_name + "-" + name, term[env_idx].tolist()))
                continue

            idx = 0
            data = obs_buffer[group_name]
            assert isinstance(data, torch.Tensor)
            for name, shape in zip(
                self._group_obs_term_names[group_name],
                self._group_obs_term_dim[group_name],
                strict=False,
            ):
                data_length = _shape_product(shape)
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

    def reset(self, reset_mask: torch.Tensor) -> dict[str, float]:
        """Fixed-shape observation state reset."""
        self._validate_reset_mask(reset_mask)
        self._obs_buffer = None
        for group_cfg in self._group_obs_class_term_cfgs.values():
            for term_cfg in group_cfg:
                if not callable(getattr(term_cfg.func, "reset", None)):
                    raise NotImplementedError(
                        f"ObservationManager term '{term_cfg.func}' does not define "
                        "the mask-preserving reset boundary"
                    )
                term_cfg.func.reset(reset_mask)
        for group_name in self._group_obs_term_names:
            for term_name in self._group_obs_term_names[group_name]:
                delay = self._group_obs_term_delay_buffer[group_name].get(term_name)
                if delay is not None:
                    raise NotImplementedError("the mask reset does not support delay buffers")
                history = self._group_obs_term_history_buffer[group_name].get(term_name)
                if history is not None:
                    raise NotImplementedError("the mask reset does not support history buffers")
        for group_mods in self._group_obs_class_instances.values():
            for mod in group_mods.values():
                if not callable(getattr(mod, "reset", None)):
                    raise NotImplementedError(
                        f"ObservationManager term '{type(mod).__name__}' does not define "
                        "the mask-preserving reset boundary"
                    )
                mod.reset(reset_mask)
        return {}

    def _validate_reset_mask(self, reset_mask: torch.Tensor) -> None:
        if (
            reset_mask.ndim != 1
            or reset_mask.shape[0] != self.num_envs
            or reset_mask.dtype != torch.bool
            or reset_mask.device != self._device
        ):
            raise ValueError(
                f"ObservationManager reset mask must have shape {(self.num_envs,)}, "
                f"bool dtype, and device {self._device}"
            )

    def _check_and_handle_nans(
        self,
        tensor: torch.Tensor,
        context: str,
        policy: str,
    ) -> torch.Tensor:
        """Check for NaN/Inf and handle according to policy.

        Args:
          tensor: Full-width observation tensor to check.
          context: Context string for error/warning messages (e.g., "actor/base_lin_vel").
          policy: NaN handling policy ("disabled", "warn", "sanitize", "error").

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
        finite = torch.isfinite(tensor)
        if bool(finite.all()):
            return tensor
        invalid = ~finite
        invalid_values = tensor[invalid]
        has_nan = bool(torch.isnan(invalid_values).any())
        has_inf = bool(torch.isinf(invalid_values).any())
        row_any = invalid.reshape(tensor.shape[0], -1).any(dim=1)
        invalid_rows = row_any.detach().cpu()

        def _row_env_ids(mask: torch.Tensor) -> list[int]:
            rows = torch.nonzero(mask, as_tuple=False).flatten()
            return [int(row) for row in rows.tolist()]

        if policy == "error":
            nan_env_ids = _row_env_ids(invalid_rows)
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
            nan_env_ids = _row_env_ids(invalid_rows)
            print(
                f"[ObservationManager] NaN/Inf in '{context}' "
                f"(envs: {nan_env_ids[:5]}). Sanitizing to 0."
            )

        # Sanitize (applies to both "warn" and "sanitize" policies).
        return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    def compute(
        self,
        update_history: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute observations for all groups.

        Every computation is full-width. History and delay buffers are shared
        reset-runtime state and fail closed during construction until they gain
        a fixed-width mask boundary.
        """
        # Return cached observations if not updating and cache exists.
        # This prevents double-pushing to delay buffers when compute() is called
        # multiple times per control step (e.g., in get_observations() after step()).
        if not update_history and self._obs_buffer is not None:
            return self._obs_buffer

        timing = getattr(self, "last_step_timing_ms", None)
        if timing is None:
            timing = {}
            self.last_step_timing_ms = timing
        timing.clear()
        child_keys = (
            "update_state_observation_term_dispatch_ms",
            "update_state_observation_validation_ms",
            "update_state_observation_noise_ms",
            "update_state_observation_transform_ms",
            "update_state_observation_temporal_ms",
            "update_state_observation_concatenation_ms",
            "update_state_observation_boundary_ms",
        )
        groups_started = time.perf_counter()
        obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] = dict()
        # Cross-group sharing of identical term computations (issue #1351):
        # within one compute() call, terms with the same func and params yield
        # the same raw output (the per-term pipeline never mutates func outputs
        # in place), so later groups reuse the first group's raw result.
        share_cache: dict[tuple, torch.Tensor] = {}
        for group_name in self._group_obs_term_names:
            obs_buffer[group_name] = self.compute_group(
                group_name, update_history, share_cache=share_cache
            )
        groups_ms = time.perf_counter() - groups_started
        cache_started = time.perf_counter()
        self._obs_buffer = obs_buffer
        cache_ms = time.perf_counter() - cache_started
        # Exclude phases already published by nested compute_group calls.
        # The remainder retains top-level group iteration, setup, timing
        # calls, and any uninstrumented group branch; it is not GPU work.
        child_ms = sum(float(timing[key]) for key in child_keys if key in timing)
        timing["update_state_observation_manager_residual_ms"] = max(
            (groups_ms + cache_ms - child_ms) * 1000.0, 0.0
        )
        return obs_buffer

    def compute_group(
        self,
        group_name: str,
        update_history: bool = False,
        *,
        share_cache: dict[tuple, torch.Tensor] | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        group_cfg = self.cfg[group_name]
        if group_cfg is None:
            raise KeyError(f"Observation group '{group_name}' is disabled.")
        group_term_names = self._group_obs_term_names[group_name]
        group_obs: dict[str, torch.Tensor] = {}
        obs_terms = zip(group_term_names, self._group_obs_term_cfgs[group_name], strict=False)
        if share_cache is None:
            share_cache = {}
        timing = getattr(self, "last_step_timing_ms", None)
        if timing is None:
            timing = {}
            self.last_step_timing_ms = timing
        group_timing = timing
        dispatch_ms = 0.0
        validation_ms = 0.0
        noise_ms = 0.0
        transform_ms = 0.0
        temporal_ms = 0.0
        share_map = self._group_obs_term_share.get(group_name, {})
        # In the strict default policy a finite result is by far the common
        # case.  For concatenated groups, scan the assembled output once and
        # only inspect individual slices when an error is actually found; this
        # retains the per-term diagnostic while removing repeated full scans
        # from the hot path.
        defer_error_nan_check = (
            self._group_obs_concatenate[group_name]
            and not self._group_obs_temporal[group_name]
            and group_cfg.nan_check_per_term
            and group_cfg.nan_policy == "error"
        )
        for term_name, term_cfg in obs_terms:
            share_key = share_map.get(term_name)
            if share_key is not None and share_key in share_cache:
                obs = share_cache[share_key]
            else:
                dispatch_started = time.perf_counter()
                obs = term_cfg.func(self._env, **term_cfg.params)
                dispatch_ms += time.perf_counter() - dispatch_started
                if share_key is not None:
                    share_cache[share_key] = obs
            validation_started = time.perf_counter()
            if not isinstance(obs, torch.Tensor):
                raise TypeError(
                    f"ObservationManager term '{group_name}/{term_name}' returned "
                    f"{type(obs).__name__}, expected torch.Tensor."
                )
            tensor = cast("torch.Tensor", obs)
            if tensor.dtype != torch.float32:
                raise TypeError(
                    f"ObservationManager term '{group_name}/{term_name}' must return "
                    f"float32 Torch observations; got {tensor.dtype}."
                )
            if tensor.device != self._device:
                raise ValueError(
                    f"ObservationManager term '{group_name}/{term_name}' must return "
                    f"observations on {self._device}; got {tensor.device}."
                )
            expected_rows = self.num_envs
            if obs.ndim < 2 or obs.shape[0] != expected_rows:
                raise ValueError(
                    f"ObservationManager term '{group_name}/{term_name}' returned shape "
                    f"{obs.shape}, expected ({expected_rows}, ...) with "
                    f"num_envs={self.num_envs}."
                )
            validation_ms += time.perf_counter() - validation_started
            fresh = False
            transform_started = time.perf_counter()
            noise_started = time.perf_counter()
            if isinstance(term_cfg.noise, noise_cfg.NoiseCfg):
                # The manager owns the noise publication carrier; random draws
                # themselves remain generated by the public noise config.
                source = cast("torch.Tensor", obs)
                noise_carriers = self._group_term_noise_carriers.setdefault(
                    (group_name, term_name), []
                )
                if not noise_carriers:
                    noise_carriers.extend(torch.empty_like(source) for _ in range(2))
                generation = self._group_term_noise_generation.get((group_name, term_name), 0)
                noisy = noise_carriers[generation % len(noise_carriers)]
                self._group_term_noise_generation[(group_name, term_name)] = generation + 1
                noisy.copy_(source)
                obs = term_cfg.noise.apply(noisy, torch_rng=self._torch_generator)
                fresh = True
                noise_ms += time.perf_counter() - noise_started
            elif isinstance(term_cfg.noise, noise_cfg.NoiseModelCfg):
                # Noise models likewise return a fresh carrier allocation.
                obs = self._group_obs_class_instances[group_name][term_name](obs)
                fresh = True
                noise_ms += time.perf_counter() - noise_started
            sanitizes_per_term = group_cfg.nan_check_per_term and group_cfg.nan_policy in (
                "warn",
                "sanitize",
            )
            exposes_term_output = (
                not self._group_obs_concatenate[group_name]
                and term_cfg.delay_max_lag == 0
                and term_cfg.history_length == 0
            )
            if not fresh and (
                term_cfg.clip is not None
                or term_cfg.scale is not None
                or sanitizes_per_term
                or exposes_term_output
            ):
                # Concatenation and temporal buffers already copy their inputs.
                # Only take a defensive copy when this pipeline may mutate the
                # term or expose it directly to callers.
                obs = cast(torch.Tensor, obs).clone()
            if term_cfg.clip:
                tensor_obs_value = cast("torch.Tensor", obs)
                torch.clamp(
                    tensor_obs_value,
                    min=term_cfg.clip[0],
                    max=term_cfg.clip[1],
                    out=tensor_obs_value,
                )
            if term_cfg.scale is not None:
                tensor_obs_value = cast("torch.Tensor", obs)
                scale_tensor = self._scale_tensors[id(term_cfg)]
                torch.multiply(
                    tensor_obs_value,
                    scale_tensor,
                    out=tensor_obs_value,
                )
            transform_ms += time.perf_counter() - transform_started

            temporal_started = time.perf_counter()

            # Check for NaN/Inf before delay/history buffers (per-term checking).
            if (
                group_cfg.nan_check_per_term
                and group_cfg.nan_policy != "disabled"
                and not defer_error_nan_check
            ):
                obs = self._check_and_handle_nans(
                    obs,
                    context=f"{group_name}/{term_name}",
                    policy=group_cfg.nan_policy,
                )

            if term_cfg.delay_max_lag > 0:
                delay_buffer = self._group_obs_term_delay_buffer[group_name][term_name]
                delay_buffer.append(cast("torch.Tensor", obs))
                obs = delay_buffer.compute()
            if term_cfg.history_length > 0:
                circular_buffer = self._group_obs_term_history_buffer[group_name][term_name]
                if update_history or not circular_buffer.is_initialized:
                    circular_buffer.append(cast("torch.Tensor", obs))

                if term_cfg.flatten_history_dim:
                    history = circular_buffer.buffer
                    group_obs[term_name] = history.reshape(self._env.num_envs, -1)
                else:
                    group_obs[term_name] = circular_buffer.buffer
            else:
                group_obs[term_name] = obs
            temporal_ms += time.perf_counter() - temporal_started

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
                    )

        if self._group_obs_concatenate[group_name]:
            concatenation_started = time.perf_counter()
            values = list(group_obs.values())
            tensors = cast("list[torch.Tensor]", values)
            if len(tensors) == 1:
                # A one-term concatenated group already has final layout;
                # preserve its exact dtype/device without a defensive copy.
                result = tensors[0]
            else:
                result = torch.cat(
                    tensors,
                    dim=self._group_obs_concatenate_dim[group_name],
                )
            if defer_error_nan_check:
                finite = torch.isfinite(result)
                verdict = self._group_finite_valid[group_name]
                torch.all(finite, dim=tuple(range(1, finite.ndim)), out=verdict)
                if self._deferred_validation_boundaries:
                    pass
                elif not bool(verdict.all()):
                    axis = self._group_obs_concatenate_dim[group_name]
                    axis = axis if axis >= 0 else result.ndim + axis
                    offset = 0
                    for term_name, term_dims in zip(
                        group_term_names,
                        self._group_obs_term_dim[group_name],
                        strict=True,
                    ):
                        width = int(term_dims[axis - 1])
                        selectors = [slice(None)] * result.ndim
                        selectors[axis] = slice(offset, offset + width)
                        selector = tuple(selectors)
                        if not finite[selector].all():
                            # Reuse the established diagnostic path so the
                            # error still names the first offending term and
                            # reports reset-row IDs when applicable.
                            self._check_and_handle_nans(
                                result[selector],
                                context=f"{group_name}/{term_name}",
                                policy=group_cfg.nan_policy,
                            )
                            break
                        offset += width
            # Final check for concatenated result (non-per-term checking).
            if not group_cfg.nan_check_per_term and group_cfg.nan_policy != "disabled":
                result = self._check_and_handle_nans(
                    result,
                    context=group_name,
                    policy=group_cfg.nan_policy,
                )
            if group_timing is not None:
                group_timing["update_state_observation_concatenation_ms"] = (
                    group_timing.get("update_state_observation_concatenation_ms", 0.0)
                    + (time.perf_counter() - concatenation_started) * 1000.0
                )
        else:
            result = group_obs

        boundary_started = time.perf_counter()
        public_result = self._observations_to_tensor_boundary(result)
        if group_timing is not None:
            group_timing["update_state_observation_term_dispatch_ms"] = (
                group_timing.get("update_state_observation_term_dispatch_ms", 0.0)
                + dispatch_ms * 1000.0
            )
            group_timing["update_state_observation_validation_ms"] = (
                group_timing.get("update_state_observation_validation_ms", 0.0)
                + validation_ms * 1000.0
            )
            group_timing["update_state_observation_noise_ms"] = (
                group_timing.get("update_state_observation_noise_ms", 0.0) + noise_ms * 1000.0
            )
            group_timing["update_state_observation_transform_ms"] = (
                group_timing.get("update_state_observation_transform_ms", 0.0)
                + transform_ms * 1000.0
            )
            group_timing["update_state_observation_temporal_ms"] = (
                group_timing.get("update_state_observation_temporal_ms", 0.0) + temporal_ms * 1000.0
            )
            group_timing["update_state_observation_boundary_ms"] = (
                group_timing.get("update_state_observation_boundary_ms", 0.0)
                + (time.perf_counter() - boundary_started) * 1000.0
            )
        return public_result

    def _observations_to_tensor_boundary(
        self, values: torch.Tensor | dict[str, torch.Tensor]
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Publish completed observation pipelines on the public Torch carrier."""
        if isinstance(values, dict):
            mapped = {
                name: self._observations_to_tensor_boundary(value) for name, value in values.items()
            }
            return cast("torch.Tensor | dict[str, torch.Tensor]", mapped)
        if isinstance(values, torch.Tensor):
            if values.dtype != torch.float32 or values.device != self._device:
                return values.to(device=self._device, dtype=torch.float32)
            if not values.is_contiguous():
                return values.contiguous()
            return values
        raise TypeError("ObservationManager results must be torch.Tensor carriers")

    def _prepare_terms(self) -> None:
        self._group_obs_term_names: dict[str, list[str]] = dict()
        self._group_obs_term_dim: dict[str, list[tuple[int, ...]]] = dict()
        self._group_obs_term_cfgs: dict[str, list[ObservationTermCfg]] = dict()
        self._group_obs_class_term_cfgs: dict[str, list[ObservationTermCfg]] = dict()
        self._group_obs_concatenate: dict[str, bool] = dict()
        self._group_obs_concatenate_dim: dict[str, int] = dict()
        self._scale_tensors: dict[int, torch.Tensor] = dict()
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
                if term_cfg.delay_max_lag > 0 or term_cfg.history_length > 0:
                    raise NotImplementedError(
                        f"ObservationManager term '{group_name}/{term_name}' uses delay or "
                        "history state, which has no fixed-width mask reset boundary"
                    )
                if hasattr(term_cfg.func, "compute_reset_rows"):
                    raise NotImplementedError(
                        f"ObservationManager term '{group_name}/{term_name}' declares the "
                        "removed observation reset-row contract"
                    )
                self._group_obs_term_names[group_name].append(term_name)
                self._group_obs_term_cfgs[group_name].append(term_cfg)
                if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
                    self._group_obs_class_term_cfgs[group_name].append(term_cfg)

                initial_obs = term_cfg.func(self._env, **term_cfg.params)
                if not isinstance(initial_obs, torch.Tensor):
                    raise TypeError(
                        f"ObservationManager term '{group_name}/{term_name}' returned "
                        f"{type(initial_obs).__name__}, expected torch.Tensor."
                    )
                if initial_obs.dtype != torch.float32 or initial_obs.device != self._device:
                    raise TypeError(
                        f"ObservationManager term '{group_name}/{term_name}' must return "
                        f"float32 observations on {self._device}; got "
                        f"{initial_obs.dtype} on {initial_obs.device}."
                    )
                if initial_obs.ndim < 2 or initial_obs.shape[0] != self.num_envs:
                    raise ValueError(
                        f"ObservationManager term '{group_name}/{term_name}' returned shape "
                        f"{initial_obs.shape}, expected (num_envs, ...) with "
                        f"num_envs={self.num_envs}."
                    )
                obs_dims = tuple(initial_obs.shape)

                if term_cfg.scale is not None:
                    scale = torch.as_tensor(term_cfg.scale, dtype=torch.float32)
                    self._scale_tensors[id(term_cfg)] = (
                        scale.broadcast_to(initial_obs.shape).clone().to(device=self._device)
                    )

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
                        term_cfg.noise,
                        num_envs=self._env.num_envs,
                        torch_rng=cast("torch.Generator", self._torch_generator),
                        device=self._device,
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
                        torch_generator=self._torch_generator,
                        device=self._device,
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
                        obs_dims = (obs_dims[0], _shape_product(obs_dims[1:]))

                self._group_obs_term_dim[group_name].append(obs_dims[1:])

            self._group_obs_term_delay_buffer[group_name] = group_entry_delay_buffer
            self._group_obs_term_history_buffer[group_name] = group_entry_history_buffer
            self._group_obs_temporal[group_name] = any(
                term_cfg.delay_max_lag > 0 or term_cfg.history_length > 0
                for term_cfg in self._group_obs_term_cfgs[group_name]
            )

        # Cross-group sharing of identical term computations (issue #1351):
        # within one compute() call, terms with the same func and params yield
        # the same raw output, so later groups reuse the first group's result.
        # Class-based terms (possibly stateful per call) and terms with
        # unhashable params are never shared.
        self._group_obs_term_share: dict[str, dict[str, tuple]] = {}
        for share_group, share_terms in self._group_obs_term_cfgs.items():
            share_entry: dict[str, tuple] = {}
            for share_name, share_cfg in zip(
                self._group_obs_term_names[share_group], share_terms, strict=False
            ):
                func = share_cfg.func
                if hasattr(func, "reset") and callable(func.reset):
                    continue
                try:
                    share_key = (func, _freeze_param_value(dict(share_cfg.params)))
                    hash(share_key)
                except TypeError:
                    continue
                share_entry[share_name] = share_key
            self._group_obs_term_share[share_group] = share_entry
