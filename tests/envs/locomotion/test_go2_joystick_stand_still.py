"""Go2 legacy-base regressions around the A2 Manager-Based migration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def test_go2_advance_phase_is_unconditional():
    """Go2WalkTask advances the gait clock every step regardless of command —
    the A2 standing freeze must not have leaked into the Go2 owner."""
    from unilab.tasks.locomotion.go2.joystick import Go2WalkTask

    stub = SimpleNamespace(_cfg=SimpleNamespace(ctrl_dt=0.02), gait_frequency=2.0)
    phase = np.array([0.3, 0.3])
    out = Go2WalkTask._advance_phase(stub, phase)
    expected = np.fmod(phase + 0.02 * 2.0, 1.0)
    np.testing.assert_allclose(out, expected)


def test_go2_reward_config_has_no_command_threshold():
    """A2's threshold stays in Hydra Manager terms, not the shared legacy config."""
    import dataclasses

    from unilab.tasks.locomotion.go2.joystick import RewardConfig

    names = {f.name for f in dataclasses.fields(RewardConfig)}
    assert "command_threshold" not in names
