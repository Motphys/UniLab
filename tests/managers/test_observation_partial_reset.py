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
import torch

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
                    func=lambda env: torch.ones((env.num_envs, 1), dtype=torch.float32)
                ),
            },
            enable_corruption=True,
        ),
    }


def _row_scale_cfg() -> dict[str, ObservationGroupCfg]:
    """Exercise a per-environment scale on the row-scoped reset path."""
    return {
        "policy": ObservationGroupCfg(
            terms={
                "scaled": ObservationTermCfg(
                    func=lambda env: env.obs,
                    scale=np.array([[1.0], [10.0], [100.0], [1000.0]], dtype=np.float32),
                ),
            },
        ),
    }


def test_partial_reset_row_scoped_noise() -> None:
    env = FakeEnv(seed=11)
    manager = ObservationManager(_noisy_cfg(), env)
    manager.compute(update_history=True)  # populate caches like a step would

    ids = torch.tensor([0, 2], dtype=torch.int64)
    rng_state = env.torch_rng.get_state()
    rows = manager.compute(update_history=True, env_ids=ids)
    rng_after_rows = env.torch_rng.get_state()

    env.torch_rng.set_state(rng_state)
    full = manager.compute(update_history=True)
    rng_after_full = env.torch_rng.get_state()

    assert rows["policy"].shape == (len(ids), full["policy"].shape[1])
    # Issue #1349: reset-path noise is drawn for the reset rows only, so the
    # shared RNG stream is consumed strictly less than the full-batch draw.
    assert not torch.equal(rng_after_rows, rng_after_full)
    # The un-noised trailing term stays bit-identical to the full-batch compute.
    state_dim = env.obs.shape[1]
    np.testing.assert_array_equal(rows["policy"][:, state_dim:], full["policy"][ids][:, state_dim:])
    # Noised columns differ from the full-batch slice (row-scoped draws).
    assert not np.array_equal(rows["policy"][:, :state_dim], full["policy"][ids][:, :state_dim])
    # The same RNG state reproduces the same reset rows deterministically.
    env.torch_rng.set_state(rng_state)
    rows_again = manager.compute(update_history=True, env_ids=ids)
    np.testing.assert_array_equal(rows["policy"], rows_again["policy"])


def test_partial_reset_does_not_populate_obs_cache() -> None:
    env = FakeEnv(seed=3)
    manager = ObservationManager(_noisy_cfg(), env)
    manager.compute(update_history=True)
    assert manager._obs_buffer is not None

    manager.reset(torch.tensor([1], dtype=torch.int64))
    assert manager._obs_buffer is None
    manager.compute(update_history=True, env_ids=torch.tensor([1], dtype=torch.int64))
    # The reset path leaves the cache invalidated; the next per-step compute
    # refreshes it with a full-batch entry.
    assert manager._obs_buffer is None
    manager.compute(update_history=True)
    assert manager._obs_buffer is not None
    assert manager._obs_buffer["policy"].shape[0] == env.num_envs


def test_partial_reset_slices_per_env_scale() -> None:
    env = FakeEnv(seed=13)
    manager = ObservationManager(_row_scale_cfg(), env)
    full = manager.compute(update_history=True)
    ids = torch.tensor([1, 3], dtype=torch.int64)
    rows = manager.compute(update_history=True, env_ids=ids)

    assert rows["policy"].shape == (len(ids), full["policy"].shape[1])
    np.testing.assert_allclose(rows["policy"].cpu().numpy(), full["policy"][ids].cpu().numpy())


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

    ids = torch.tensor([1], dtype=torch.int64)
    keep_ids = torch.tensor([0, 2, 3], dtype=torch.int64)
    history_before = manager._group_obs_term_history_buffer["policy"]["state"].buffer.clone()
    delay_before = manager._group_obs_term_delay_buffer["policy"]["delayed"].peek().clone()

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
        result = env.obs.clone()
        result[2, 0] = bad
        return result

    env = FakeEnv(seed=7)
    manager = ObservationManager(
        {"policy": ObservationGroupCfg(terms={"bad": ObservationTermCfg(func=invalid)})},
        env,
    )
    with pytest.raises(ValueError, match=r"for environments: \[2\]"):
        manager.compute(update_history=True, env_ids=torch.tensor([0, 2], dtype=torch.int64))


def test_partial_reset_nan_on_untouched_row_is_not_rechecked() -> None:
    def invalid(env: FakeEnv) -> np.ndarray:
        result = env.obs.clone()
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
    rows = manager.compute(update_history=True, env_ids=torch.tensor([0, 2], dtype=torch.int64))
    assert rows["policy"].shape == (2, env.obs.shape[1])
    assert isinstance(rows["policy"], torch.Tensor)
    assert torch.isfinite(rows["policy"]).all()


INSTANCES: list["RowScopedTerm"] = []


class RowScopedTerm:
    """Callable class-based observation term with reset-row execution."""

    def __init__(self, cfg: ObservationTermCfg | None = None, env: FakeEnv | None = None) -> None:
        del cfg, env
        self.reset_row_calls: list[torch.Tensor] = []
        self.full_calls = 0
        INSTANCES.append(self)

    def __call__(self, env: FakeEnv) -> torch.Tensor:
        self.full_calls += 1
        return env.obs.clone()

    def compute_reset_rows(self, env: FakeEnv, env_ids: torch.Tensor) -> torch.Tensor:
        self.reset_row_calls.append(env_ids.clone())
        return env.obs.index_select(0, env_ids.to(env.obs.device))

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        del env_ids


def test_opt_in_reset_row_term_executes_only_requested_rows() -> None:
    env = FakeEnv(seed=17)
    manager = ObservationManager(
        {"policy": ObservationGroupCfg(terms={"state": ObservationTermCfg(func=RowScopedTerm)})},
        env,
    )
    term = INSTANCES[-1]
    full = manager.compute(update_history=True)
    ids = torch.tensor([1, 3], dtype=torch.int64)
    rows = manager.compute(update_history=True, env_ids=ids)

    assert term.full_calls == 2  # construction plus the explicit full compute
    assert len(term.reset_row_calls) == 1
    torch.testing.assert_close(term.reset_row_calls[0], ids)
    assert rows["policy"].shape == (len(ids), full["policy"].shape[1])
    torch.testing.assert_close(rows["policy"], full["policy"][ids])


def test_stateful_reset_row_terms_are_not_shared_across_groups() -> None:
    env = FakeEnv(seed=19)
    cfg = {
        "actor": ObservationGroupCfg(terms={"state": ObservationTermCfg(func=RowScopedTerm)}),
        "critic": ObservationGroupCfg(terms={"state": ObservationTermCfg(func=RowScopedTerm)}),
    }
    manager = ObservationManager(cfg, env)
    manager.compute(update_history=True)
    ids = torch.tensor([0, 2], dtype=torch.int64)
    rows = manager.compute(update_history=True, env_ids=ids)

    actor_term, critic_term = INSTANCES[-2:]
    assert len(actor_term.reset_row_calls) == 1
    torch.testing.assert_close(actor_term.reset_row_calls[0], ids)
    assert len(critic_term.reset_row_calls) == 1
    torch.testing.assert_close(critic_term.reset_row_calls[0], ids)
    torch.testing.assert_close(rows["actor"], rows["critic"])


def test_opt_in_reset_row_term_rejects_wrong_leading_dimension() -> None:
    class BadRows(RowScopedTerm):
        def compute_reset_rows(self, env: FakeEnv, env_ids: torch.Tensor) -> torch.Tensor:
            del env_ids
            return env.obs

    env = FakeEnv(seed=23)
    manager = ObservationManager(
        {"policy": ObservationGroupCfg(terms={"state": ObservationTermCfg(func=BadRows())})},
        env,
    )
    with pytest.raises(ValueError, match="expected \\(1, \\.\\.\\.\\)"):
        manager.compute(update_history=True, env_ids=torch.tensor([2], dtype=torch.int64))


def test_opt_in_reset_row_term_temporal_group_falls_back() -> None:
    env = FakeEnv(seed=29)
    term = RowScopedTerm()
    manager = ObservationManager(
        {
            "policy": ObservationGroupCfg(
                terms={"state": ObservationTermCfg(func=term, history_length=2)},
            ),
        },
        env,
    )
    manager.compute(update_history=True)
    ids = torch.tensor([1], dtype=torch.int64)
    rows = manager.compute(update_history=True, env_ids=ids)

    assert term.reset_row_calls == []
    assert rows["policy"].shape[0] == len(ids)
