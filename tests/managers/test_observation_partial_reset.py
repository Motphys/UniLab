"""Row-scoped partial-reset parity tests for ObservationManager (issue #1259 R2).

On the partial-reset path (compute(update_history=True, env_ids=...)) the
manager returns only the reset rows and processes them row-scoped whenever the
group has no delay/history terms. These tests pin that contract:

- un-noised reset rows are bit-identical to a full-batch compute sliced to
  those rows; noise is drawn for the reset rows only — issue #1349 removed the
  full-batch RNG-stream parity requirement, so noised reset rows match neither
  the full-batch noise values nor its RNG consumption;
- groups with delay/history terms fall back to the full-batch pipeline and
  only slice the final output; untouched rows' buffers are not advanced;
- NaN diagnostics on the row-scoped path report real env indices.
"""

from __future__ import annotations

import numpy as np
import pytest

from unilab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg
from unilab.managers._noise import UniformNoiseCfg

from .conftest import FakeEnv


def _noisy_cfg() -> dict[str, ObservationGroupCfg]:
    return {
        "policy": ObservationGroupCfg(
            terms={
                "state": ObservationTermCfg(
                    func=lambda env: env.obs,
                    noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1),
                    clip=(-10.0, 10.0),
                    scale=2.0,
                ),
                "bias": ObservationTermCfg(
                    func=lambda env: np.ones((env.num_envs, 1), dtype=np.float32)
                ),
            },
            enable_corruption=True,
        ),
    }


def test_partial_reset_row_scoped_noise() -> None:
    env = FakeEnv(seed=11)
    manager = ObservationManager(_noisy_cfg(), env)
    manager.compute(update_history=True)  # populate caches like a step would

    ids = np.array([0, 2], dtype=np.int32)
    rng_state = env.rng.bit_generator.state
    rows = manager.compute(update_history=True, env_ids=ids)
    rng_after_rows = env.rng.bit_generator.state

    env.rng.bit_generator.state = rng_state
    full = manager.compute(update_history=True)
    rng_after_full = env.rng.bit_generator.state

    assert rows["policy"].shape == (len(ids), full["policy"].shape[1])
    # Issue #1349: reset-path noise is drawn for the reset rows only, so the
    # shared RNG stream is consumed strictly less than the full-batch draw.
    assert rng_after_rows != rng_after_full
    # The un-noised trailing term stays bit-identical to the full-batch compute.
    state_dim = env.obs.shape[1]
    np.testing.assert_array_equal(rows["policy"][:, state_dim:], full["policy"][ids][:, state_dim:])
    # Noised columns differ from the full-batch slice (row-scoped draws).
    assert not np.array_equal(rows["policy"][:, :state_dim], full["policy"][ids][:, :state_dim])
    # The same RNG state reproduces the same reset rows deterministically.
    env.rng.bit_generator.state = rng_state
    rows_again = manager.compute(update_history=True, env_ids=ids)
    np.testing.assert_array_equal(rows["policy"], rows_again["policy"])


def test_partial_reset_does_not_populate_obs_cache() -> None:
    env = FakeEnv(seed=3)
    manager = ObservationManager(_noisy_cfg(), env)
    manager.compute(update_history=True)
    assert manager._obs_buffer is not None

    manager.reset(np.array([1], dtype=np.int32))
    assert manager._obs_buffer is None
    manager.compute(update_history=True, env_ids=np.array([1], dtype=np.int32))
    # The reset path leaves the cache invalidated; the next per-step compute
    # refreshes it with a full-batch entry.
    assert manager._obs_buffer is None
    manager.compute(update_history=True)
    assert manager._obs_buffer is not None
    assert manager._obs_buffer["policy"].shape[0] == env.num_envs


def test_partial_reset_temporal_group_falls_back_and_preserves_rows() -> None:
    env = FakeEnv(seed=5)
    cfg = {
        "policy": ObservationGroupCfg(
            terms={
                "state": ObservationTermCfg(func=lambda env: env.obs, history_length=3),
                "delayed": ObservationTermCfg(
                    func=lambda env: env.obs, delay_min_lag=1, delay_max_lag=1
                ),
            },
        ),
    }
    manager = ObservationManager(cfg, env)
    for step in range(4):
        env.obs = env.obs + 1
        manager.compute(update_history=True)

    ids = np.array([1], dtype=np.int32)
    keep_ids = np.array([0, 2, 3], dtype=np.int32)
    history_before = manager._group_obs_term_history_buffer["policy"]["state"].buffer.copy()
    delay_before = manager._group_obs_term_delay_buffer["policy"]["delayed"].peek().copy()

    rng_state = env.rng.bit_generator.state
    rows = manager.compute(update_history=True, env_ids=ids)

    env.rng.bit_generator.state = rng_state
    full = manager.compute(update_history=True, env_ids=ids)
    np.testing.assert_array_equal(rows["policy"], full["policy"])
    assert rows["policy"].shape[0] == len(ids)

    # Untouched rows keep their history and delayed values bit-identically;
    # the reset row is backfilled with its post-reset frame in every slot.
    history_after = manager._group_obs_term_history_buffer["policy"]["state"].buffer
    np.testing.assert_array_equal(history_after[keep_ids], history_before[keep_ids])
    reset_slots = history_after[ids[0]]
    np.testing.assert_array_equal(reset_slots, np.broadcast_to(reset_slots[-1], reset_slots.shape))
    np.testing.assert_array_equal(
        manager._group_obs_term_delay_buffer["policy"]["delayed"].peek()[keep_ids],
        delay_before[keep_ids],
    )

    # Reference: the full-batch pipeline sliced to the reset rows agrees.
    env.rng.bit_generator.state = rng_state
    reference = manager.compute(update_history=True)["policy"][ids]
    env.rng.bit_generator.state = rng_state
    np.testing.assert_array_equal(rows["policy"], reference)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_partial_reset_nan_error_reports_env_ids(bad: float) -> None:
    def invalid(env: FakeEnv) -> np.ndarray:
        result = env.obs.copy()
        result[2, 0] = bad
        return result

    env = FakeEnv(seed=7)
    manager = ObservationManager(
        {"policy": ObservationGroupCfg(terms={"bad": ObservationTermCfg(func=invalid)})},
        env,
    )
    with pytest.raises(ValueError, match=r"for environments: \[2\]"):
        manager.compute(update_history=True, env_ids=np.array([0, 2], dtype=np.int32))


def test_partial_reset_nan_on_untouched_row_is_not_rechecked() -> None:
    def invalid(env: FakeEnv) -> np.ndarray:
        result = env.obs.copy()
        result[1, 0] = np.nan
        return result

    env = FakeEnv(seed=7)
    manager = ObservationManager(
        {
            "policy": ObservationGroupCfg(
                terms={"bad": ObservationTermCfg(func=invalid)}, nan_policy="error"
            )
        },
        env,
    )
    # Row-scoped NaN checks cover only the reset rows; untouched rows were
    # already checked by the per-step compute of their control step.
    rows = manager.compute(update_history=True, env_ids=np.array([0, 2], dtype=np.int32))
    assert rows["policy"].shape == (2, env.obs.shape[1])
    assert np.isfinite(rows["policy"]).all()
