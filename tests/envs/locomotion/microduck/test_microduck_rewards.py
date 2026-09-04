"""Focused reward tests for MicroDuck stateful manager terms."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from unilab.envs.mdp import UniformPoseCommandCfg, UniformVelocityCommand, UniformVelocityCommandCfg
from unilab.managers import RewardTermCfg
from unilab.managers._types import ManagerBasedRlEnv
from unilab.tasks.locomotion.microduck.manager_terms import (
    MicroduckVelocityCommandCfg,
    body_pose_tracking,
    flight_phase,
    foot_air_time_biped,
)


class _SensorScene:
    def __init__(self, values: dict[str, np.ndarray]) -> None:
        self.values = values
        self.bound_names: tuple[str, ...] | None = None

    def bind_sensor_data(self, names: tuple[str, ...]):
        self.bound_names = names
        arrays = [self.values[name] for name in names]
        return SimpleNamespace(
            dimensions=tuple(array.shape[1] for array in arrays),
            backend_type="fake",
            read=lambda: np.concatenate(arrays, axis=1),
        )


def _env(contact: np.ndarray, command: np.ndarray) -> tuple[ManagerBasedRlEnv, _SensorScene]:
    scene = _SensorScene(
        {
            "left_foot_contact": contact[:, 0:1],
            "right_foot_contact": contact[:, 1:2],
        }
    )
    command_manager = SimpleNamespace(get_command=lambda name: command)
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=contact.shape[0],
            step_dt=0.1,
            scene=scene,
            command_manager=command_manager,
        ),
    )
    return env, scene


def _term(term_type: type, env: ManagerBasedRlEnv, **params: Any):
    return term_type(
        RewardTermCfg(func=term_type, weight=1.0, params=params),
        env,
    )


def test_foot_air_time_biped_rewards_single_stance_only() -> None:
    env, scene = _env(
        np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        np.asarray([[0.2, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
    )
    term = _term(
        foot_air_time_biped,
        env,
        threshold=0.3,
        command_threshold=0.01,
        command_name="twist",
    )

    np.testing.assert_allclose(term(env), [0.1, 0.0])
    assert scene.bound_names == ("left_foot_contact", "right_foot_contact")
    term.reset(np.asarray([0], dtype=np.int32))
    np.testing.assert_array_equal(term._air_time[0], 0.0)
    np.testing.assert_array_equal(term._contact_time[0], 0.0)


def test_flight_phase_penalizes_only_double_air() -> None:
    env, _ = _env(
        np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
    )
    term = _term(flight_phase, env)
    np.testing.assert_array_equal(term(env), [1.0, 0.0])


def test_turn_in_place_bucket_overrides_independent_standing_sample(monkeypatch) -> None:
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=2,
            step_dt=0.02,
            rng=np.random.default_rng(4),
            scene={
                "robot": SimpleNamespace(
                    data=SimpleNamespace(
                        root_link_lin_vel_b=np.zeros((2, 3), dtype=np.float32),
                        root_link_ang_vel_b=np.zeros((2, 3), dtype=np.float32),
                    )
                )
            },
        ),
    )
    cfg = MicroduckVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(1.0, 1.0),
        rel_standing_envs=1.0,
        turn_in_place_fraction=1.0,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.4, 0.4),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-1.0, 1.0),
        ),
    )
    term = cfg.build(env)

    def sample_standing(self, env_ids):
        self.vel_command_b[env_ids] = (0.2, 0.1, 0.0)
        self.vel_command_w[env_ids] = self.vel_command_b[env_ids]
        self.is_standing_env[env_ids] = True

    monkeypatch.setattr(UniformVelocityCommand, "_resample_command", sample_standing)
    ids = np.asarray([0, 1], dtype=np.int32)
    term._resample_command(ids)
    term._update_command()

    np.testing.assert_array_equal(term.is_standing_env, False)
    np.testing.assert_array_equal(term.command[:, :2], 0.0)
    assert np.all(np.abs(term.command[:, 2]) >= 0.4)
    np.testing.assert_array_equal(term.vel_command_w, term.command)


def test_contact_terms_read_first_component_from_vector_force_sensors() -> None:
    scene = _SensorScene(
        {
            "left_foot_contact": np.asarray([[0.0, 4.0, 5.0], [1.0, 0.0, 0.0]]),
            "right_foot_contact": np.asarray([[0.0, 6.0, 7.0], [0.0, 8.0, 9.0]]),
        }
    )
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(num_envs=2, step_dt=0.1, scene=scene),
    )

    term = _term(flight_phase, env)

    np.testing.assert_array_equal(term(env), [1.0, 0.0])


def test_contact_terms_fail_closed_when_sensor_is_missing() -> None:
    env, scene = _env(
        np.ones((1, 2), dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
    )
    del scene.values["right_foot_contact"]
    with pytest.raises(KeyError, match="right_foot_contact"):
        _term(flight_phase, env)


def test_uniform_pose_command_resamples_from_live_ranges() -> None:
    """Step-staged command curricula mutate the live term cfg; the next
    resample must sample from the widened ranges without rebuilding the term."""
    env = cast(ManagerBasedRlEnv, SimpleNamespace(num_envs=8, rng=np.random.default_rng(0)))
    cfg = UniformPoseCommandCfg(
        resampling_time_range=(2.0, 5.0),
        ranges=[[-0.05, 0.05], [-0.05, 0.05], [-0.07, 0.07], [-0.015, 0.015]],
    )
    term = cfg.build(env)
    env_ids = np.arange(8)
    term.reset(env_ids)
    assert float(np.abs(term.command).max()) <= 0.07

    cfg.ranges = [[-1.1, 1.1], [-1.1, 1.1], [-1.4, 1.4], [-0.31, 0.31]]
    term.reset(env_ids)
    assert float(np.abs(term.command[:, 2]).max()) > 0.07
    assert float(np.abs(term.command).max()) <= 1.4


def test_uniform_pose_command_rejects_range_width_change() -> None:
    env = cast(ManagerBasedRlEnv, SimpleNamespace(num_envs=2, rng=np.random.default_rng(0)))
    cfg = UniformPoseCommandCfg(
        resampling_time_range=(2.0, 5.0),
        ranges=[[-0.05, 0.05], [-0.05, 0.05], [-0.07, 0.07], [-0.015, 0.015]],
    )
    term = cfg.build(env)
    cfg.ranges = [[-1.0, 1.0]] * 3
    with pytest.raises(ValueError, match="width changed"):
        term.reset(np.arange(2))


class _PoseScene:
    def __init__(self, position: np.ndarray, quat: np.ndarray) -> None:
        self.env_origins = np.zeros((position.shape[0], 3), dtype=np.float32)
        self._robot = SimpleNamespace(
            data=SimpleNamespace(root_link_pos_w=position, root_link_quat_w=quat)
        )

    def __getitem__(self, name: str):
        if name != "robot":
            raise KeyError(name)
        return self._robot


def _pose_env(position: np.ndarray, quat: np.ndarray, command: np.ndarray) -> ManagerBasedRlEnv:
    return cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=position.shape[0],
            scene=_PoseScene(position, quat),
            command_manager=SimpleNamespace(get_command=lambda name: command),
        ),
    )


def test_body_pose_tracking_matches_upstream_6d_gaussian_mean() -> None:
    identity = np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    # env0 is exactly at the nominal pose with a zero command; env1 is one
    # xy_std off in x and one z_std off in z: two axes at exp(-1), four at 1.
    position = np.asarray([[0.0, 0.0, 0.095], [0.05, 0.0, 0.095 + 0.02]], dtype=np.float32)
    command = np.zeros((2, 6), dtype=np.float32)
    env = _pose_env(position, identity, command)
    expected = (4.0 + 2.0 * np.exp(-1.0)) / 6.0
    np.testing.assert_allclose(body_pose_tracking(env), [1.0, expected], rtol=1e-6)

    # One angle_std of roll error: quat for roll=15deg about x.
    half = np.deg2rad(15.0) / 2.0
    rolled = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [np.cos(half), np.sin(half), 0.0, 0.0]], dtype=np.float32
    )
    position = np.asarray([[0.0, 0.0, 0.095], [0.0, 0.0, 0.095]], dtype=np.float32)
    env = _pose_env(position, rolled, command)
    expected = (5.0 + np.exp(-1.0)) / 6.0
    np.testing.assert_allclose(body_pose_tracking(env), [1.0, expected], rtol=1e-5)


def test_body_pose_tracking_uses_env_origins_and_command_targets() -> None:
    # A root parked at the env origin plus the commanded xyz delta scores 1.
    position = np.asarray([[2.01, -1.01, 0.105]], dtype=np.float32)
    quat = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    command = np.asarray([[0.01, -0.01, 0.01, 0.0, 0.0, 0.0]], dtype=np.float32)
    env = _pose_env(position, quat, command)
    scene = cast(Any, env.scene)
    scene.env_origins = np.asarray([[2.0, -1.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(body_pose_tracking(env), [1.0], rtol=1e-6)

    bad_env = _pose_env(position, quat, np.zeros((1, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="width 6"):
        body_pose_tracking(bad_env)
