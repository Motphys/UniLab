"""Tests for the shared standing-command helper and Go2 flat regressions.

The zero-command standstill behaviour itself is A2-specific and lives on
``A2JoystickFlatEnv`` (see ``tests/envs/locomotion/a2/test_a2_joystick_contract.py``).
This file keeps only:

- ``sample_commands_with_standing`` (zero-xy + standing fraction helper) — a
  shared building block used by A2 reset + A2 resample.
- Go2-flat-unchanged regressions asserting ``Go2WalkTask`` reverted to the main
  baseline: the phase clock advances unconditionally and ``RewardConfig`` carries
  no ``command_threshold``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from unilab.envs.locomotion.common.commands import (
    sample_commands_with_standing,
    zero_small_xy_commands,
)

# ── sample_commands_with_standing (shared helper) ─────────────────────


def test_sample_commands_with_standing_all_standing():
    low = np.array([-1.0, -1.0, -1.0])
    high = np.array([1.0, 1.0, 1.0])
    out = sample_commands_with_standing(low, high, 64, rel_standing_envs=1.0)
    assert out.shape == (64, 3)
    np.testing.assert_array_equal(out, np.zeros((64, 3)))


def test_sample_commands_with_standing_none_standing_zeroes_small_xy():
    np.random.seed(7)
    low = np.array([-1.0, -1.0, -1.0])
    high = np.array([1.0, 1.0, 1.0])
    out = sample_commands_with_standing(low, high, 2000, rel_standing_envs=0.0)
    xy_norm = np.linalg.norm(out[:, :2], axis=1)
    # no row below the zero-xy threshold may retain a nonzero xy.
    assert np.all((xy_norm == 0.0) | (xy_norm >= 0.08))
    # with no standing fraction, not every row is fully zero (yaw survives).
    assert np.any(np.abs(out[:, 2]) > 0.0)


def test_sample_commands_with_standing_matches_rough_block():
    """Same construction as rough.py's provider standing block, so reset (provider)
    and resampling stay a single source of truth."""
    np.random.seed(3)
    low = np.array([-1.0, -1.0, -1.0])
    high = np.array([1.0, 1.0, 1.0])
    out = sample_commands_with_standing(low, high, 5, rel_standing_envs=0.5)

    np.random.seed(3)
    ref = np.asarray(np.random.uniform(low=low, high=high, size=(5, 3)))
    zero_small_xy_commands(ref, threshold=0.08)
    standing = np.random.uniform(size=(5,)) < 0.5
    ref[standing] = 0.0
    np.testing.assert_allclose(out, ref)


# ── Go2 flat regressions (reverted to main baseline) ──────────────────


def test_go2_advance_phase_is_unconditional():
    """Go2WalkTask advances the gait clock every step regardless of command —
    the A2 standing freeze must not have leaked into the Go2 owner."""
    from unilab.envs.locomotion.go2.joystick import Go2WalkTask

    stub = SimpleNamespace(_cfg=SimpleNamespace(ctrl_dt=0.02), gait_frequency=2.0)
    phase = np.array([0.3, 0.3])
    out = Go2WalkTask._advance_phase(stub, phase)
    expected = np.fmod(phase + 0.02 * 2.0, 1.0)
    np.testing.assert_allclose(out, expected)


def test_go2_reward_config_has_no_command_threshold():
    """command_threshold is A2-owned (A2RewardConfig); the Go2 base RewardConfig
    must not declare it."""
    import dataclasses

    from unilab.envs.locomotion.go2.joystick import RewardConfig

    names = {f.name for f in dataclasses.fields(RewardConfig)}
    assert "command_threshold" not in names
