"""Focused reward tests for MicroDuck stateful manager terms."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from unilab.envs.mdp import UniformPoseCommandCfg
from unilab.managers import RewardTermCfg
from unilab.managers._types import ManagerBasedRlEnv
from unilab.tasks.locomotion.microduck.manager_terms import flight_phase, foot_air_time_biped


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
