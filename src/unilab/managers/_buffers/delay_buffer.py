# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), src/mjlab/utils/buffers/delay_buffer.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and UniLab contracts; licensed under Apache-2.0.
"""Delay buffer for stochastically delayed observations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from unilab.managers._buffers import CircularBuffer


class DelayBuffer:
    """Serve stochastically delayed observations from a rolling history.

    Wraps a CircularBuffer to simulate observation delays by returning frames from T-lag
    timesteps ago, where lag is sampled from [min_lag, max_lag].

    Core Behavior
    =============

    At each timestep:
      1. Append new observation to history
      2. Sample or hold lag value (0 = no delay, 3 = 3 timesteps old)
      3. Return observation from T-lag

    Example with lag=2:
      t=0: append obs_0 → return obs_0 (not enough history)
      t=1: append obs_1 → return obs_0 (clamped to available history)
      t=2: append obs_2 → return obs_0 (lag=2, so T-2 = 0)
      t=3: append obs_3 → return obs_1 (lag=2, so T-2 = 1)

    Lag Update Policy
    =================

    Lags can be refreshed every step or periodically:

      **Every-step updates (update_period=0)**
        Each timestep may sample a new lag (subject to hold_prob).

      **Periodic updates (update_period=N)**
        Lags refresh only every N steps per environment:
          if (step_count + phase_offset) % N == 0:
              sample new lag
          else:
              keep previous lag

      **Staggered updates (per_env_phase=True)**
        Each environment gets a random phase_offset ∈ [0, N), causing
        lag updates to occur on different timesteps:
          Env 0: updates at t=0, N, 2N, ...
          Env 1: updates at t=3, N+3, 2N+3, ...
          Env 2: updates at t=7, N+7, 2N+7, ...

      **Hold probability (hold_prob=0.2)**
        Even when an update would occur, keep previous lag with 20% chance.
        Creates temporal correlation in delay patterns.

    Per-Environment vs Shared Lags
    ==============================

      **per_env=True** (default)
        Each environment has independent lag:
          Batch 0: lag=1 → returns obs from t-1
          Batch 1: lag=3 → returns obs from t-3
          Batch 2: lag=0 → returns current obs

      **per_env=False**
        All environments share one sampled lag:
          All batches: lag=2 → all return obs from t-2

    Reset Behavior
    ==============

      reset(batch_ids=[1]) clears history for specified environments:
        - Sets lag and step counter to zero
        - Clears circular buffer for those rows
        - Next append backfills their history with first new value
        - Until that append, compute() returns zeros for reset rows

    Args:
      min_lag (int, optional): Minimum lag (inclusive). Must be >= 0.
      max_lag (int, optional): Maximum lag (inclusive). Must be >= `min_lag`.
      batch_size (int, optional): Number of parallel environments (leading
        dimension of inputs).
      per_env (bool, optional): If True, sample a separate lag per environment;
        otherwise sample one lag and share it across environments.
      hold_prob (float, optional): Probability in `[0.0, 1.0]` to keep the previous
        lag when an update would occur. Creates temporal correlation in delays.
      update_period (int, optional): If > 0, refresh lags every N steps per
        environment; if 0, consider updating every step.
      per_env_phase (bool, optional): If True and `update_period > 0`, each
        environment uses a different phase offset in `[0, update_period)`, causing
        staggered refresh steps across the batch.
      generator (np.random.Generator | None, optional): RNG for sampling lags.

    Examples:
      Constant delay (lag = 2):
        >>> buf = DelayBuffer(min_lag=2, max_lag=2, batch_size=4)
        >>> buf.append(obs)                # obs.shape == (4, ...)
        >>> delayed = buf.compute()        # delayed[t] = obs[t-2]

      Stochastic delay (uniform 0-3):
        >>> buf = DelayBuffer(min_lag=0, max_lag=3, batch_size=4)
        >>> buf.append(obs)
        >>> delayed = buf.compute()        # per-env lag sampled in {0,1,2,3}

      Periodic updates with staggering:
        >>> buf = DelayBuffer(
        ...     min_lag=1, max_lag=5, batch_size=8,
        ...     update_period=10,           # refresh every 10 steps
        ...     per_env_phase=True,         # stagger across envs
        ...     hold_prob=0.2               # 20% chance to hold lag
        ... )
        >>> # Env 0 refreshes at t=0,10,20,...
        >>> # Env 1 refreshes at t=3,13,23,... (random offset)
        >>> # But each refresh has 20% chance to keep previous lag
    """

    def __init__(
        self,
        min_lag: int = 0,
        max_lag: int = 3,
        batch_size: int = 1,
        per_env: bool = True,
        hold_prob: float = 0.0,
        update_period: int = 0,
        per_env_phase: bool = True,
        generator: np.random.Generator | None = None,
    ) -> None:
        if min_lag < 0:
            raise ValueError(f"min_lag must be >= 0, got {min_lag}")
        if max_lag < min_lag:
            raise ValueError(f"max_lag ({max_lag}) must be >= min_lag ({min_lag})")
        if not 0.0 <= hold_prob <= 1.0:
            raise ValueError(f"hold_prob must be in [0, 1], got {hold_prob}")
        if update_period < 0:
            raise ValueError(f"update_period must be >= 0, got {update_period}")

        self.min_lag = min_lag
        self.max_lag = max_lag
        self.batch_size = batch_size
        self.per_env = per_env
        self.hold_prob = hold_prob
        self.update_period = update_period
        self.per_env_phase = per_env_phase
        self.generator = generator

        buffer_size = max_lag + 1 if max_lag > 0 else 1
        self._buffer = CircularBuffer(max_len=buffer_size, batch_size=batch_size)
        self._current_lags = np.zeros(batch_size, dtype=np.int64)
        self._step_count = np.zeros(batch_size, dtype=np.int64)

        if update_period > 0 and per_env_phase:
            self._phase_offsets = self._require_generator().integers(
                0, update_period, size=batch_size, dtype=np.int64
            )
        else:
            self._phase_offsets = np.zeros(batch_size, dtype=np.int64)

    @property
    def is_initialized(self) -> bool:
        """Check if buffer has been initialized with at least one append."""
        return self._buffer.is_initialized

    @property
    def current_lags(self) -> np.ndarray:
        """Current lag per environment. Shape: (batch_size,)."""
        return self._current_lags

    def set_lags(
        self,
        lags: np.ndarray,
        batch_ids: Sequence[int] | np.ndarray | slice | None = None,
    ) -> None:
        """Set lag values for specified environments.

        Args:
          lags: Lag values to set. Shape: (num_batch_ids,) or scalar.
          batch_ids: Batch indices to set, or None to set all.
        """
        idx = slice(None) if batch_ids is None else batch_ids
        self._current_lags[idx] = np.clip(lags, self.min_lag, self.max_lag)

    def reset(self, batch_ids: Sequence[int] | np.ndarray | slice | None = None) -> None:
        """Reset specified environments to initial state.

        Args:
          batch_ids: Batch indices to reset, or None to reset all.
        """
        if isinstance(batch_ids, slice):
            indices = range(*batch_ids.indices(self.batch_size))
            batch_ids = list(indices)

        self._buffer.reset(batch_ids=batch_ids)
        idx = slice(None) if batch_ids is None else batch_ids
        self._current_lags[idx] = 0
        self._step_count[idx] = 0
        if self.update_period > 0 and self.per_env_phase:
            new_phases = self._require_generator().integers(
                0, self.update_period, size=self.batch_size, dtype=np.int64
            )
            self._phase_offsets[idx] = new_phases[idx]

    def append(self, data: np.ndarray) -> None:
        """Append new observation to buffer.

        Args:
          data: Observation tensor of shape (batch_size, ...).
        """
        self._buffer.append(data)

    def backfill(self, data: np.ndarray, batch_ids: np.ndarray) -> None:
        """Backfill the given rows with one frame, without advancing time.

        Used after a partial reset: the reset rows (whose lags and step counters
        were just zeroed by reset) get their first post-reset frame while the
        remaining rows keep their history, lags, and update schedule intact.

        Args:
          data: Tensor of shape (batch_size, ...); only rows at batch_ids are read.
          batch_ids: Batch indices to backfill.
        """
        self._buffer.backfill(data, batch_ids)

    def compute(self) -> np.ndarray:
        """Compute delayed observation for current step.

        Advances the lag update schedule, then returns the delayed observation.

        Returns:
          Delayed observation with shape (batch_size, ...).
        """
        if not self.is_initialized:
            raise RuntimeError("Buffer not initialized. Call append() first.")

        self._update_lags()
        return self.peek()

    def peek(self) -> np.ndarray:
        """Return the delayed observation using current lags, without advancing.

        Unlike compute, this neither steps the update schedule nor resamples lags,
        so it is safe to call outside the once-per-step cadence.

        Returns:
          Delayed observation with shape (batch_size, ...).
        """
        if not self.is_initialized:
            raise RuntimeError("Buffer not initialized. Call append() first.")

        # Clamp lags to valid range [0, buffer_length - 1].
        # Buffer may not be full yet (e.g., only 2 frames but sampled lag=3).
        valid_lags = np.minimum(self._current_lags, self._buffer.current_length - 1)
        valid_lags = np.maximum(valid_lags, 0)

        return self._buffer[valid_lags]

    def _update_lags(self) -> None:
        """Update current lags according to configured policy."""
        if self.update_period > 0:
            phase_adjusted_count = (self._step_count + self._phase_offsets) % (self.update_period)
            should_update = phase_adjusted_count == 0
        else:
            should_update = np.ones(self.batch_size, dtype=np.bool_)
        new_lags = self._sample_lags(should_update)
        self._current_lags = np.where(should_update, new_lags, self._current_lags)
        self._step_count += 1

    def _sample_lags(self, mask: np.ndarray) -> np.ndarray:
        """Sample new lags for specified environments.

        Args:
          mask: Boolean mask of shape (batch_size,) indicating which envs to sample.

        Returns:
          New lags with shape (batch_size,).
        """
        if self.min_lag == self.max_lag:
            candidate_lags = np.full(self.batch_size, self.min_lag, dtype=np.int64)
        elif self.per_env:
            candidate_lags = self._require_generator().integers(
                self.min_lag, self.max_lag + 1, size=self.batch_size, dtype=np.int64
            )
        else:
            shared_lag = self._require_generator().integers(self.min_lag, self.max_lag + 1)
            candidate_lags = np.full(self.batch_size, shared_lag, dtype=np.int64)

        if self.hold_prob > 0.0:
            should_sample = self._require_generator().random(self.batch_size) >= self.hold_prob
            update_mask = mask & should_sample
        else:
            update_mask = mask

        return np.where(update_mask, candidate_lags, self._current_lags)

    def _require_generator(self) -> np.random.Generator:
        if self.generator is None:
            raise ValueError(
                "DelayBuffer stochastic sampling requires an env-owned NumPy generator."
            )
        return self.generator
