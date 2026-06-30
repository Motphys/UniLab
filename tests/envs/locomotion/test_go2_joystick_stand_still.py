"""Unit tests for the command-gated standing behaviour shared by Go2WalkTask.

These cover the config-gated changes that let the A2 (and any future) joystick
task stand still at zero command without retraining the gait clock:

- ``_reward_swing_feet_z`` / ``_reward_contact`` are gated by
  ``RewardConfig.command_threshold`` (default 0.0 -> Go2 byte-for-byte unchanged).
- ``_reward_hip_deviation`` (L1 over hip DOF indices [0, 3, 6, 9]).
- ``sample_commands_with_standing`` (zero-xy + standing fraction helper).
- ``_update_commands`` mid-episode resampling gated by ``resampling_time``.

The reward methods are unbound and only read a handful of attributes, so we
bind them onto a light stub instead of constructing a MuJoCo env. This keeps the
tests fast and backend-free while exercising the exact production code paths.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from unilab.envs.locomotion.common.commands import (
    sample_commands_with_standing,
    zero_small_xy_commands,
)
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.go2.joystick import Go2WalkTask, RewardConfig

# ── stub helpers ──────────────────────────────────────────────────────


def _sensor(num_feet: int = 4):
    return SimpleNamespace(
        feet_force=["FL", "FR", "RL", "RR"][:num_feet],
        feet_pos=["FL", "FR", "RL", "RR"][:num_feet],
    )


def _reward_stub(
    *,
    num_envs: int,
    feet_phase: np.ndarray,
    feet_pos: np.ndarray | None = None,
    feet_force: np.ndarray | None = None,
    gait_frequency: float = 2.0,
    command_threshold: float = 0.0,
    default_angles: np.ndarray | None = None,
):
    """Build an object exposing exactly the attributes the reward methods read."""
    num_feet = feet_phase.shape[1]
    if feet_pos is None:
        feet_pos = np.zeros((num_envs, num_feet, 3), dtype=np.float32)
    if feet_force is None:
        feet_force = np.zeros((num_envs, num_feet, 3), dtype=np.float32)
    if default_angles is None:
        default_angles = np.zeros((12,), dtype=np.float64)
    reward_cfg = RewardConfig(
        scales={},
        tracking_sigma=0.25,
        base_height_target=0.4,
        command_threshold=command_threshold,
    )
    return SimpleNamespace(
        _num_envs=num_envs,
        feet_phase=feet_phase,
        feet_pos=feet_pos,
        feet_force=feet_force,
        gait_frequency=gait_frequency,
        _reward_cfg=reward_cfg,
        default_angles=default_angles,
        _cfg=SimpleNamespace(sensor=_sensor(num_feet)),
    )


def _ctx(
    commands: np.ndarray,
    dof_pos: np.ndarray | None = None,
    gyro: np.ndarray | None = None,
    linvel: np.ndarray | None = None,
) -> RewardContext:
    num_envs = commands.shape[0]
    if dof_pos is None:
        dof_pos = np.zeros((num_envs, 12), dtype=np.float64)
    if gyro is None:
        gyro = np.zeros((num_envs, 3))
    if linvel is None:
        linvel = np.zeros((num_envs, 3))
    return RewardContext(
        info={"commands": commands},
        linvel=linvel,
        gyro=gyro,
        dof_pos=dof_pos,
        num_envs=num_envs,
    )


# Pre-change reference formulas (snapshot for the regression test).
def _ref_swing_feet_z(stub) -> np.ndarray:
    is_swing = stub.feet_phase >= 0.6
    target_height = 0.1
    height_error = np.square(stub.feet_pos[:, :, 2] - target_height)
    swing_rew = np.exp(-height_error / 0.01) * is_swing
    return np.sum(swing_rew, axis=1) / len(stub._cfg.sensor.feet_pos)


def _ref_contact(stub) -> np.ndarray:
    contact = stub.feet_force[:, :, 2] > 0.1
    res = np.zeros(stub._num_envs, dtype=np.float32)
    for i in range(len(stub._cfg.sensor.feet_force)):
        is_contact = (stub.feet_phase[:, i] < 0.6) | (stub.gait_frequency < 1.0e-8)
        res += (contact[:, i] == is_contact).astype(np.float32)
    return res / len(stub._cfg.sensor.feet_force)


# ── swing_feet_z gating ───────────────────────────────────────────────


def test_swing_feet_z_gated_zeroes_standing_rows():
    """With threshold 0.1, a zero-command row earns 0 even at swing height while a
    large-command row keeps the original ungated value."""
    # both feet in swing at the target height -> max raw reward.
    feet_phase = np.array([[0.7, 0.7, 0.7, 0.7], [0.7, 0.7, 0.7, 0.7]], dtype=np.float32)
    feet_pos = np.zeros((2, 4, 3), dtype=np.float32)
    feet_pos[:, :, 2] = 0.1  # exactly target -> exp(0) = 1
    stub = _reward_stub(num_envs=2, feet_phase=feet_phase, feet_pos=feet_pos, command_threshold=0.1)
    commands = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    out = Go2WalkTask._reward_swing_feet_z(stub, _ctx(commands))

    raw = _ref_swing_feet_z(stub)
    assert out[0] == 0.0  # standing row gated to zero
    np.testing.assert_allclose(out[1], raw[1])  # active row unchanged
    assert raw[1] > 0.0


def test_swing_feet_z_threshold_zero_is_unchanged():
    """Default threshold 0.0: any nonzero command keeps active True -> Go2 path
    is byte-for-byte identical to the pre-change formula."""
    rng = np.random.default_rng(0)
    feet_phase = rng.uniform(0.0, 1.0, size=(5, 4)).astype(np.float32)
    feet_pos = rng.uniform(-0.05, 0.2, size=(5, 4, 3)).astype(np.float32)
    stub = _reward_stub(num_envs=5, feet_phase=feet_phase, feet_pos=feet_pos, command_threshold=0.0)
    commands = rng.uniform(0.1, 1.0, size=(5, 3)).astype(np.float64)
    out = Go2WalkTask._reward_swing_feet_z(stub, _ctx(commands))
    np.testing.assert_array_equal(out, _ref_swing_feet_z(stub))


# ── contact gating ────────────────────────────────────────────────────


def test_contact_gated_full_reward_when_standing():
    """At zero command (threshold 0.1) every foot is expected-in-contact, so a
    robot with all feet planted earns the full 1.0 contact reward."""
    # phase puts feet 0,3 in stance (<0.6) and feet 1,2 in swing (>=0.6).
    feet_phase = np.array([[0.1, 0.7, 0.7, 0.1]], dtype=np.float32)
    feet_force = np.zeros((1, 4, 3), dtype=np.float32)
    feet_force[:, :, 2] = 5.0  # all feet planted
    stub = _reward_stub(
        num_envs=1, feet_phase=feet_phase, feet_force=feet_force, command_threshold=0.1
    )
    commands = np.zeros((1, 3), dtype=np.float64)
    out = Go2WalkTask._reward_contact(stub, _ctx(commands))
    assert out[0] == 1.0


def test_contact_threshold_zero_is_unchanged():
    """Default threshold 0.0: standing mask ``cmd_norm <= 0`` is False for any
    nonzero command -> Go2 contact reward identical to the pre-change formula."""
    rng = np.random.default_rng(1)
    feet_phase = rng.uniform(0.0, 1.0, size=(5, 4)).astype(np.float32)
    feet_force = rng.uniform(-1.0, 5.0, size=(5, 4, 3)).astype(np.float32)
    stub = _reward_stub(
        num_envs=5, feet_phase=feet_phase, feet_force=feet_force, command_threshold=0.0
    )
    commands = rng.uniform(0.1, 1.0, size=(5, 3)).astype(np.float64)
    out = Go2WalkTask._reward_contact(stub, _ctx(commands))
    np.testing.assert_array_equal(out, _ref_contact(stub))


# ── hip_deviation ─────────────────────────────────────────────────────


def test_hip_deviation_l1_over_hip_indices():
    default = np.array(
        [0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8], dtype=np.float64
    )
    stub = _reward_stub(
        num_envs=2, feet_phase=np.zeros((2, 4), dtype=np.float32), default_angles=default
    )
    dof_pos = np.tile(default, (2, 1))
    # perturb only hip joints (indices 0,3,6,9) on row 1.
    dof_pos[1, 0] += 0.2
    dof_pos[1, 3] -= 0.3
    dof_pos[1, 6] += 0.1
    dof_pos[1, 9] -= 0.4
    # perturb a non-hip joint on row 0 -> must NOT be counted.
    dof_pos[0, 1] += 5.0
    out = Go2WalkTask._reward_hip_deviation(stub, _ctx(np.zeros((2, 3)), dof_pos=dof_pos))
    np.testing.assert_allclose(out[0], 0.0)
    np.testing.assert_allclose(out[1], 0.2 + 0.3 + 0.1 + 0.4)


# ── stand_feet_air (penalize stepping while standing) ─────────────────


def test_stand_feet_air_counts_lifted_feet_when_standing():
    """At zero command (threshold 0.1) the penalty counts feet off the ground."""
    feet_force = np.zeros((1, 4, 3), dtype=np.float32)
    feet_force[:, 0, 2] = 5.0  # FL planted
    feet_force[:, 1, 2] = 5.0  # FR planted (RL, RR have ~0 vertical force -> in air)
    stub = _reward_stub(
        num_envs=1,
        feet_phase=np.zeros((1, 4), dtype=np.float32),
        feet_force=feet_force,
        command_threshold=0.1,
    )
    out = Go2WalkTask._reward_stand_feet_air(stub, _ctx(np.zeros((1, 3))))
    assert out[0] == 2.0  # two feet in the air


def test_stand_feet_air_zero_when_all_planted():
    feet_force = np.zeros((1, 4, 3), dtype=np.float32)
    feet_force[:, :, 2] = 5.0
    stub = _reward_stub(
        num_envs=1,
        feet_phase=np.zeros((1, 4), dtype=np.float32),
        feet_force=feet_force,
        command_threshold=0.1,
    )
    out = Go2WalkTask._reward_stand_feet_air(stub, _ctx(np.zeros((1, 3))))
    assert out[0] == 0.0


def test_stand_feet_air_inactive_during_locomotion():
    """Command above threshold -> term is exactly zero, so it costs the walking
    gait nothing (decoupled from locomotion / lateral / yaw tracking)."""
    feet_force = np.zeros((1, 4, 3), dtype=np.float32)  # all four feet in the air
    stub = _reward_stub(
        num_envs=1,
        feet_phase=np.zeros((1, 4), dtype=np.float32),
        feet_force=feet_force,
        command_threshold=0.1,
    )
    out = Go2WalkTask._reward_stand_feet_air(stub, _ctx(np.array([[0.5, 0.0, 0.0]])))
    assert out[0] == 0.0


# ── _advance_phase (freeze gait clock while standing) ─────────────────


def test_advance_phase_freezes_standing_envs():
    """At/below threshold the gait phase is frozen (no 2 Hz drive at standstill);
    above threshold it advances by ctrl_dt * gait_frequency as before."""
    phase = np.array([0.30, 0.30, 0.30], dtype=np.float32)
    commands = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float64)
    out = Go2WalkTask._advance_phase(
        phase, ctrl_dt=0.02, gait_frequency=2.0, commands=commands, command_threshold=0.1
    )
    # rows 0 (zero) and 1 (||cmd||=0.05<=0.1) frozen; row 2 advances by 0.02*2=0.04
    np.testing.assert_allclose(out[0], 0.30)
    np.testing.assert_allclose(out[1], 0.30)
    np.testing.assert_allclose(out[2], 0.34)


def test_advance_phase_threshold_zero_advances_all():
    """Default threshold 0.0: any nonzero command advances -> Go2 path unchanged."""
    phase = np.array([0.30, 0.98], dtype=np.float32)
    commands = np.array([[0.6, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=np.float64)
    out = Go2WalkTask._advance_phase(
        phase, ctrl_dt=0.02, gait_frequency=2.0, commands=commands, command_threshold=0.0
    )
    # both advance by 0.04; row 1 wraps via fmod (0.98+0.04=1.02 -> 0.02)
    np.testing.assert_allclose(out[0], 0.34, atol=1e-6)
    np.testing.assert_allclose(out[1], 0.02, atol=1e-6)


# ── sample_commands_with_standing ─────────────────────────────────────


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
    and resampling (_update_commands) stay a single source of truth."""
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


# ── _update_commands ──────────────────────────────────────────────────


def _update_commands_stub(commands: np.ndarray, *, resampling_time: float):
    num_envs = commands.shape[0]
    cfg = SimpleNamespace(
        ctrl_dt=0.02,
        commands=SimpleNamespace(
            resampling_time=resampling_time,
            heading_command=False,
            rel_standing_envs=0.0,
            vel_limit=[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        ),
    )
    return SimpleNamespace(_num_envs=num_envs, _cfg=cfg)


def test_update_commands_noop_when_resampling_disabled():
    commands = np.array([[0.5, 0.0, 0.3], [0.0, 0.0, 0.0]], dtype=np.float64)
    stub = _update_commands_stub(commands.copy(), resampling_time=0.0)
    info = {"commands": commands.copy(), "steps": np.array([10, 10], dtype=np.uint32)}
    Go2WalkTask._update_commands(stub, info)
    np.testing.assert_array_equal(info["commands"], commands)


def test_update_commands_resamples_at_interval_boundary():
    np.random.seed(11)
    commands = np.array([[0.5, 0.2, 0.3], [0.5, 0.2, 0.3]], dtype=np.float64)
    stub = _update_commands_stub(commands.copy(), resampling_time=5.0)
    # interval_steps = round(5.0 / 0.02) = 250; step 250 is a resample boundary.
    info = {"commands": commands.copy(), "steps": np.array([250, 1], dtype=np.uint32)}
    Go2WalkTask._update_commands(stub, info)
    # row 0 is at the boundary -> resampled (changed); row 1 is not -> unchanged.
    assert not np.allclose(info["commands"][0], commands[0])
    # _update_commands casts to the global dtype; row 1 is value-preserved.
    np.testing.assert_allclose(info["commands"][1], commands[1])
