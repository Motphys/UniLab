"""Edge-case tests for ``run_reward_dispatch`` (Phase 4.3 safety net).

Total-reward golden cannot isolate the dispatch branches; these pin them directly
against the current implementation (rewards.py):
- ``scale == 0`` / missing-key skip (line 355)
- ``scale * fn(ctx) * ctrl_dt`` reduction (lines 357-360, 366)
- ``only_positive`` clamp before ``ctrl_dt`` (line 364)
- log cadence: components written only when ``step % log_every_n_steps == 0`` (line 347)

Pure-function — no simulator, no Hydra. Not marked ``slow``.
"""

from __future__ import annotations

import numpy as np

from unilab.envs.locomotion.common.rewards import RewardContext, run_reward_dispatch


def _ctx(num_envs: int = 2) -> RewardContext:
    """Minimal RewardContext: info/linvel/gyro/dof_pos are required (no defaults)."""
    return RewardContext(
        info={},
        num_envs=num_envs,
        linvel=np.zeros((num_envs, 3), dtype=np.float32),
        gyro=np.zeros((num_envs, 3), dtype=np.float32),
        dof_pos=np.zeros((num_envs, 12), dtype=np.float32),
    )


def _ones(ctx: RewardContext) -> np.ndarray:
    return np.ones(ctx.num_envs)


class _SpyFn:
    """Reward fn that counts calls — proves the dispatch short-circuit, not just the output."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, ctx: RewardContext) -> np.ndarray:
        self.calls += 1
        return np.ones(ctx.num_envs)


def test_dispatch_skips_zero_scale_and_missing_key():
    out = run_reward_dispatch(
        scales={"a": 0.0, "b": 1.0},  # a: scale 0 -> skip; b: not in fns -> skip
        fns={"a": _ones},
        ctx=_ctx(2),
        info={"steps": np.zeros(2, dtype=np.uint32)},
        enable_log=True,
        ctrl_dt=0.02,
    )
    np.testing.assert_array_equal(out, np.zeros(2))


def test_dispatch_zero_scale_does_not_call_fn():
    """rewards.py:355 short-circuits on ``scale == 0`` *before* calling fn — a perf contract.

    Output alone can't prove this (``fn`` * 0 is also 0); the spy proves fn was never run.
    """
    spy = _SpyFn()
    run_reward_dispatch(
        scales={"a": 0.0},
        fns={"a": spy},
        ctx=_ctx(2),
        info={"steps": np.zeros(2, dtype=np.uint32)},
        enable_log=True,
        ctrl_dt=0.02,
    )
    assert spy.calls == 0  # scale==0 -> `continue` before fns[name](ctx)


def test_dispatch_applies_scale_and_ctrl_dt():
    out = run_reward_dispatch(
        scales={"a": 2.0},
        fns={"a": _ones},
        ctx=_ctx(2),
        info={"steps": np.zeros(2, dtype=np.uint32)},
        enable_log=False,
        ctrl_dt=0.5,
    )
    np.testing.assert_allclose(out, np.full(2, 1.0))  # 1 * 2 * 0.5


def test_dispatch_only_positive_clamps():
    out = run_reward_dispatch(
        scales={"a": -1.0},
        fns={"a": _ones},
        ctx=_ctx(2),
        info={"steps": np.zeros(2, dtype=np.uint32)},
        enable_log=False,
        ctrl_dt=1.0,
        only_positive=True,
    )
    np.testing.assert_array_equal(out, np.zeros(2))  # max(reward, 0) before ctrl_dt


def test_dispatch_logs_on_cadence_step():
    info = {"steps": np.zeros(2, dtype=np.uint32)}  # step 0 % 4 == 0 -> logs
    run_reward_dispatch(
        scales={"a": 1.0},
        fns={"a": _ones},
        ctx=_ctx(2),
        info=info,
        enable_log=True,
        ctrl_dt=1.0,
        log_every_n_steps=4,
    )
    assert "reward/a" in info["log"]


def test_dispatch_log_value_is_pre_ctrl_dt():
    """log stores ``mean(scale*fn)`` BEFORE ctrl_dt (rewards.py:361); the return is *after*
    (rewards.py:366). scale=2, fn=1, ctrl_dt=0.5 -> component logged 2.0 but reward 1.0.

    Locks the semantic that dashboard component values are pre-ctrl_dt and intentionally
    differ from the summed reward — a refactor moving the log past the ctrl_dt scale breaks it.
    """
    info = {"steps": np.zeros(2, dtype=np.uint32)}  # step 0 -> logs
    out = run_reward_dispatch(
        scales={"a": 2.0},
        fns={"a": _ones},
        ctx=_ctx(2),
        info=info,
        enable_log=True,
        ctrl_dt=0.5,
    )
    np.testing.assert_allclose(out, np.full(2, 1.0))  # 1 * 2 * 0.5  (post-ctrl_dt)
    assert info["log"]["reward/a"] == 2.0  # 1 * 2 (pre-ctrl_dt mean)


def test_dispatch_skips_log_off_cadence_step():
    info = {"steps": np.ones(2, dtype=np.uint32)}  # step 1 % 4 != 0 -> no fresh component
    run_reward_dispatch(
        scales={"a": 1.0},
        fns={"a": _ones},
        ctx=_ctx(2),
        info=info,
        enable_log=True,
        ctrl_dt=1.0,
        log_every_n_steps=4,
    )
    assert "reward/a" not in info.get("log", {})


def test_dispatch_off_cadence_preserves_prior_log():
    """Off-cadence keeps the prior ``info["log"]`` untouched (rewards.py:352 carry-forward).

    ``run_env_trajectory`` relies on this: components written on a cadence step persist
    across the off-cadence steps in between. A refactor that cleared the log every step
    (or appended fresh components off-cadence) would break that carry-forward.
    """
    info = {
        "steps": np.ones(2, dtype=np.uint32),  # step 1 % 4 != 0 -> off-cadence
        "log": {"reward/old": 5.0},  # component from a prior cadence step
    }
    run_reward_dispatch(
        scales={"a": 1.0},
        fns={"a": _ones},
        ctx=_ctx(2),
        info=info,
        enable_log=True,
        ctrl_dt=1.0,
        log_every_n_steps=4,
    )
    assert info["log"] == {"reward/old": 5.0}  # preserved verbatim, no fresh "reward/a"
