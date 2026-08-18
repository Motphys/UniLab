"""Focused parity tests for task-owned quadruped Manager-Based terms."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from unilab.base.backend.base import BackendSensorView
from unilab.envs.locomotion.common import manager_terms
from unilab.managers import (
    ObservationGroupCfg,
    ObservationManager,
    ObservationTermCfg,
    RewardManager,
    RewardTermCfg,
)
from unilab.managers._types import ManagerBasedRlEnv

CONTACTS = ("fl_contact", "fr_contact", "rl_contact", "rr_contact")
POSITIONS = ("fl_pos", "fr_pos", "rl_pos", "rr_pos")


class _Scene:
    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.values = {
            "fl_contact": np.array([[0.2], [0.0]], dtype=np.float32),
            "fr_contact": np.array([[0, 0, 0], [0, 0, 0.2]], dtype=np.float32),
            "rl_contact": np.array([[0.0], [0.2]], dtype=np.float32),
            "rr_contact": np.array([[0, 0, 0.2], [0, 0, 0]], dtype=np.float32),
            "fl_pos": np.array([[0, 0, 0.1], [0, 0, 0]], dtype=np.float32),
            "fr_pos": np.array([[0, 0, 0.2], [0, 0, 0.1]], dtype=np.float32),
            "rl_pos": np.array([[0, 0, 0.1], [0, 0, 0.2]], dtype=np.float32),
            "rr_pos": np.array([[0, 0, 0], [0, 0, 0.1]], dtype=np.float32),
        }

    def bind_sensor_data(self, names) -> BackendSensorView:
        names = tuple(names)
        self.calls["bind"] += 1
        dimensions = tuple(self.values[name].shape[1] for name in names)

        def read() -> np.ndarray:
            self.calls["read"] += 1
            return np.concatenate([self.values[name] for name in names], axis=1)

        view = BackendSensorView("fake", names, dimensions, 2, read)
        view.read()  # Mirror SimBackend.bind_sensor_data materialization validation.
        return view


def _env(counter: int = 0, scene: _Scene | None = None) -> ManagerBasedRlEnv:
    return cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=2,
            common_step_counter=counter,
            episode_length_buf=np.array([counter, 0]),
            step_dt=0.02,
            scene=scene or _Scene(),
        ),
    )


def _observations(env: ManagerBasedRlEnv, **params: Any) -> ObservationManager:
    term = ObservationTermCfg(
        func=manager_terms.quadruped_gait_phase,
        params={"frequency": 2.0, **params},
    )
    return ObservationManager({"policy": ObservationGroupCfg(terms={"gait_phase": term})}, env)


def _rewards(env: ManagerBasedRlEnv) -> RewardManager:
    terms: dict[str, RewardTermCfg | None] = {
        "contact": RewardTermCfg(
            func=manager_terms.feet_phase_contact,
            weight=1.0,
            params={"sensor_names": CONTACTS, "frequency": 2.0},
        ),
        "swing": RewardTermCfg(
            func=manager_terms.feet_phase_swing_height,
            weight=1.0,
            params={"sensor_names": POSITIONS, "frequency": 2.0},
        ),
    }
    return RewardManager(terms, env, scale_by_dt=False)


def test_gait_phase_matches_global_legacy_clock_and_ignores_episode_reset() -> None:
    env = _env()
    manager = _observations(env)
    assert manager.group_obs_dim == {"policy": (4,)}
    initial = manager.compute_group("policy")
    assert isinstance(initial, np.ndarray)
    np.testing.assert_array_equal(initial, [[0.0, 0.5, 0.5, 0.0]] * 2)

    cast(Any, env).common_step_counter = 25
    env.episode_length_buf[:] = [0, 999]  # Partial reset does not reset the global clock.
    advanced = manager.compute_group("policy")
    assert isinstance(advanced, np.ndarray)
    np.testing.assert_allclose(
        advanced, [[1.1920929e-7, 0.5000001, 0.5000001, 1.1920929e-7]] * 2, atol=1e-8
    )


def test_foot_rewards_match_legacy_equations_and_read_only_bound_views() -> None:
    scene = _Scene()
    manager = _rewards(_env(5, scene))
    assert scene.calls == {"bind": 2, "read": 2}
    result = manager.compute(dt=0.02)
    assert scene.calls == {"bind": 2, "read": 4}

    phase = np.array([[0.2, 0.7, 0.7, 0.2]] * 2)
    contact = np.array([[True, False, False, True], [False, True, True, False]])
    heights = np.array([[0.1, 0.2, 0.1, 0.0], [0.0, 0.1, 0.2, 0.1]])
    expected = np.mean(contact == (phase < 0.6), axis=1)
    expected += np.mean(np.exp(-np.square(heights - 0.1) / 0.01) * (phase >= 0.6), axis=1)
    np.testing.assert_allclose(result, expected, atol=1e-7)
    manager.compute(dt=0.02)
    assert scene.calls == {"bind": 2, "read": 6}


@pytest.mark.parametrize(
    ("params", "error", "match"),
    [
        ({"frequency": -1}, ValueError, "frequency must be at least 0.0"),
        ({"phase_offsets": (0, 0.5)}, ValueError, "must contain 4 values"),
        ({"phase_offsets": (0, True, 0.5, 0)}, TypeError, "must be a real number"),
        ({"unknown": 1}, TypeError, "unsupported parameters"),
    ],
)
def test_gait_phase_invalid_config_fails_at_construction(params, error, match: str) -> None:
    with pytest.raises(error, match=match):
        _observations(_env(), **params)


def test_foot_sensor_contracts_fail_at_construction() -> None:
    env = _env()
    short = RewardTermCfg(
        func=manager_terms.feet_phase_contact,
        weight=1,
        params={"sensor_names": CONTACTS[:3]},
    )
    with pytest.raises(ValueError, match="must contain 4 names"):
        RewardManager({"feet": short}, env)

    scene = _Scene()
    scene.values["fr_contact"] = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="must each expose 1-D found or 3-D force"):
        _rewards(_env(scene=scene))

    scene = _Scene()
    scene.values["rl_pos"] = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="must each expose 3-D xyz"):
        _rewards(_env(scene=scene))


def test_runtime_nonfinite_sensor_data_reports_term_and_backend() -> None:
    scene = _Scene()
    manager = _rewards(_env(scene=scene))
    scene.values["fl_contact"][1, 0] = np.nan
    with pytest.raises(ValueError, match="feet_phase_contact.*backend 'fake'.*NaN or Inf"):
        manager.compute(dt=0.02)
