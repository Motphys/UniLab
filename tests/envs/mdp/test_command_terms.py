"""Contract tests for the generic pose/posture command terms."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from unilab.envs.mdp import (
    GroundPickPhaseCommand,
    GroundPickPhaseCommandCfg,
    SitStandCommand,
    SitStandCommandCfg,
    UniformPoseCommand,
    UniformPoseCommandCfg,
    UniformVelocityCommandCfg,
)


def _velocity_ranges() -> UniformVelocityCommandCfg.Ranges:
    return UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.0, 1.0),
    )


def _env(num_envs: int = 2) -> SimpleNamespace:
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=np.asarray(
                [[0.0, 0.0, 0.115], [0.0, 0.0, 0.060]],
                dtype=np.float32,
            ),
            root_link_lin_vel_b=np.zeros((num_envs, 3), dtype=np.float32),
            root_link_ang_vel_b=np.zeros((num_envs, 3), dtype=np.float32),
        )
    )
    return SimpleNamespace(
        num_envs=num_envs,
        rng=np.random.default_rng(11),
        scene={"robot": robot},
        step_dt=0.02,
        episode_length_buf=np.zeros(num_envs, dtype=np.int64),
    )


def test_uniform_pose_command_samples_configured_width_and_zero_bucket() -> None:
    env = _env()
    cfg = UniformPoseCommandCfg(
        resampling_time_range=(1.0, 1.0),
        ranges=((-1.0, 1.0), (2.0, 2.0)),
        zero_command_prob=1.0,
    )
    term = cfg.build(env)
    assert isinstance(term, UniformPoseCommand)

    term.reset(np.asarray([0, 1], dtype=np.int32))
    assert term.command.shape == (2, 2)
    np.testing.assert_array_equal(term.command, 0.0)


def test_ground_pick_phase_is_continuous_and_does_not_resample() -> None:
    env = _env()
    cfg = GroundPickPhaseCommandCfg(
        entity_name="robot",
        resampling_time_range=(1.0, 1.0),
        ranges=_velocity_ranges(),
        period=4.0,
        randomize_phase=False,
    )
    term = cfg.build(env)
    assert isinstance(term, GroundPickPhaseCommand)

    ids = np.asarray([0, 1], dtype=np.int32)
    term.reset(ids)
    np.testing.assert_array_equal(term.phase, 0.0)
    np.testing.assert_array_equal(term.command_counter, 0)

    term.compute(1.0)
    np.testing.assert_allclose(term.phase, 0.25)
    np.testing.assert_allclose(term.command, [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], atol=1e-6)
    np.testing.assert_array_equal(term.command_counter, 0)

    with pytest.raises(ValueError, match="non-finite dt"):
        term.compute(float("nan"))


def test_sit_stand_command_slews_from_measured_height_to_sampled_target() -> None:
    env = _env()
    cfg = SitStandCommandCfg(
        entity_name="robot",
        resampling_time_range=(1.0, 1.0),
        ranges=_velocity_ranges(),
        sit_prob=1.0,
        ramp_s=2.0,
        sit_z=0.060,
        stand_z=0.115,
    )
    term = cfg.build(env)
    assert isinstance(term, SitStandCommand)

    ids = np.asarray([0, 1], dtype=np.int32)
    term.reset(ids)
    # The command is SIT for both rows, but alpha starts from the measured
    # posture: standing remains 0 and seated remains 1.
    term.compute(0.0, env_ids=ids)
    np.testing.assert_allclose(term.command[:, 0], 1.0)
    np.testing.assert_allclose(term.alpha, [0.0, 1.0])

    term.compute(1.0)
    np.testing.assert_allclose(term.alpha, [0.5, 1.0])


def test_posture_command_rejects_non_finite_or_non_positive_tuning() -> None:
    with pytest.raises(ValueError, match="period must be finite"):
        GroundPickPhaseCommandCfg(
            entity_name="robot",
            resampling_time_range=(1.0, 1.0),
            ranges=_velocity_ranges(),
            period=float("nan"),
        ).build(_env())

    with pytest.raises(TypeError, match="ramp_s must be a real number"):
        SitStandCommandCfg(
            entity_name="robot",
            resampling_time_range=(1.0, 1.0),
            ranges=_velocity_ranges(),
            ramp_s=True,
        ).build(_env())
