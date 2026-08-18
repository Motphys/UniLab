"""Focused parity tests for task-owned quadruped Manager-Based terms."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from unilab.base.backend.base import BackendSensorView
from unilab.dtype_config import get_global_dtype
from unilab.managers import (
    ObservationGroupCfg,
    ObservationManager,
    ObservationTermCfg,
    RewardManager,
    RewardTermCfg,
)
from unilab.managers._types import ManagerBasedRlEnv
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.tasks.locomotion.common import manager_terms

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


class _ParityScene:
    def __init__(self) -> None:
        self.robot = SimpleNamespace(
            data=SimpleNamespace(
                root_link_lin_vel_b=np.array(
                    [[0.2, 0.1, -0.3], [0.3, -0.2, 0.5]], dtype=np.float32
                ),
                root_link_ang_vel_b=np.array(
                    [[0.1, -0.2, 0.4], [-0.3, 0.2, -0.1]], dtype=np.float32
                ),
                root_link_pos_w=np.array([[1.0, 2.0, 0.4], [-1.0, 0.5, 0.2]], dtype=np.float32),
                joint_pos=np.array([[0.2, -0.1, 0.5], [-0.4, 0.3, 0.1]], dtype=np.float32),
                default_joint_pos=np.array([[0.1, -0.2, 0.5], [-0.1, 0.1, 0.0]], dtype=np.float32),
            )
        )

    def __getitem__(self, name: str):
        if name != "robot":
            raise KeyError(name)
        return self.robot


class _Commands:
    def __init__(self, command: np.ndarray | None = None) -> None:
        self.command = (
            np.array([[0.5, -0.2, 0.3], [-0.1, 0.4, -0.2]], dtype=np.float32)
            if command is None
            else command
        )

    def get_command(self, name: str) -> np.ndarray:
        if name != "twist":
            raise KeyError(name)
        return self.command


def _env(counter: int = 0, scene: _Scene | None = None) -> ManagerBasedRlEnv:
    return cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=2,
            common_step_counter=counter,
            episode_length_buf=np.array([counter, 0]),
            step_dt=0.02,
            scene=scene or _Scene(),
            command_manager=_Commands(),
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


def _parity_env() -> ManagerBasedRlEnv:
    return cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=2,
            scene=_ParityScene(),
            command_manager=_Commands(),
            max_episode_length_s=20.0,
        ),
    )


def _reward_value(env: ManagerBasedRlEnv, func, **params: Any) -> np.ndarray:
    cfg = RewardTermCfg(func=func, weight=1.0, params=params)
    return RewardManager({"parity": cfg}, env, scale_by_dt=False).compute(dt=0.02)


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


def test_standing_aware_gait_freezes_only_standing_environments() -> None:
    env = _env()
    cast(Any, env).command_manager.command[:] = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
    manager = _observations(env, command_name="twist", command_threshold=0.1)

    cast(Any, env).common_step_counter = 1
    phase = manager.compute_group("policy")
    assert isinstance(phase, np.ndarray)
    np.testing.assert_allclose(
        phase,
        [[0.0, 0.5, 0.5, 0.0], [0.04, 0.54, 0.54, 0.04]],
        atol=1e-7,
    )


def test_standing_aware_foot_rewards_gate_swing_and_expect_planted_feet() -> None:
    env = _env(counter=5)
    cast(Any, env).command_manager.command[:] = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
    params = {"frequency": 2.0, "command_name": "twist", "command_threshold": 0.1}
    manager = RewardManager(
        {
            "contact": RewardTermCfg(
                func=manager_terms.feet_phase_contact,
                weight=1.0,
                params={"sensor_names": CONTACTS, **params},
            ),
            "swing": RewardTermCfg(
                func=manager_terms.feet_phase_swing_height,
                weight=1.0,
                params={"sensor_names": POSITIONS, **params},
            ),
        },
        env,
        scale_by_dt=False,
    )

    value = manager.compute(dt=0.02)
    expected_moving_swing = (1.0 + np.exp(-1.0)) / 4.0
    np.testing.assert_allclose(value, [0.5, expected_moving_swing], atol=1e-7)


def test_standing_penalties_match_a2_legacy_gates() -> None:
    parity_env = _parity_env()
    cast(Any, parity_env).command_manager.command[:] = [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
    ]
    stand_still = _reward_value(
        parity_env,
        manager_terms.stand_still_l1,
        command_name="twist",
        command_threshold=0.1,
    )
    np.testing.assert_allclose(stand_still, [0.2, 0.0], atol=1e-7)

    foot_env = _env()
    cast(Any, foot_env).command_manager.command[:] = [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
    ]
    feet_air = _reward_value(
        foot_env,
        manager_terms.feet_air_while_standing,
        sensor_names=CONTACTS,
        command_name="twist",
        command_threshold=0.1,
    )
    np.testing.assert_array_equal(feet_air, [2.0, 0.0])


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
        ({"command_name": ""}, ValueError, "command_name must be a non-empty string"),
        ({"command_threshold": -0.1}, ValueError, "must be at least 0.0"),
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


def test_base_reward_terms_match_go2_flat_equations() -> None:
    env = _parity_env()
    asset = SceneEntityCfg("robot")
    data = cast(Any, env).scene.robot.data
    command = cast(Any, env).command_manager.command
    std = 0.5
    actual = {
        "track_xy": _reward_value(
            env,
            manager_terms.track_lin_vel_xy_exp,
            std=std,
            command_name="twist",
            asset_cfg=asset,
        ),
        "track_yaw": _reward_value(
            env,
            manager_terms.track_ang_vel_z_exp,
            std=std,
            command_name="twist",
            asset_cfg=SceneEntityCfg("robot"),
        ),
        "lin_z": _reward_value(env, manager_terms.lin_vel_z_l2, asset_cfg=SceneEntityCfg("robot")),
        "ang_xy": _reward_value(
            env, manager_terms.ang_vel_xy_l2, asset_cfg=SceneEntityCfg("robot")
        ),
        "height": _reward_value(
            env,
            manager_terms.base_height_l2,
            target_height=0.3,
            asset_cfg=SceneEntityCfg("robot"),
        ),
        "pose": _reward_value(
            env, manager_terms.joint_deviation_l1, asset_cfg=SceneEntityCfg("robot")
        ),
    }
    expected = {
        "track_xy": np.exp(
            -np.sum(np.square(command[:, :2] - data.root_link_lin_vel_b[:, :2]), axis=1) / std**2
        ),
        "track_yaw": np.exp(-np.square(command[:, 2] - data.root_link_ang_vel_b[:, 2]) / std**2),
        "lin_z": np.square(data.root_link_lin_vel_b[:, 2]),
        "ang_xy": np.sum(np.square(data.root_link_ang_vel_b[:, :2]), axis=1),
        "height": np.square(data.root_link_pos_w[:, 2] - 0.3),
        "pose": np.sum(np.abs(data.joint_pos - data.default_joint_pos), axis=1),
    }
    for name in expected:
        assert actual[name].shape == (2,)
        assert actual[name].dtype == np.dtype(get_global_dtype())
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-6, atol=1e-7)


def test_base_reward_terms_fail_closed_at_nearest_boundary() -> None:
    env = _parity_env()
    with pytest.raises(ValueError, match="track_lin_vel_xy_exp std must be greater than 0.0"):
        _reward_value(env, manager_terms.track_lin_vel_xy_exp, std=0.0, command_name="twist")
    with pytest.raises(KeyError, match="track_ang_vel_z_exp command capability 'missing'"):
        _reward_value(env, manager_terms.track_ang_vel_z_exp, std=0.5, command_name="missing")
    with pytest.raises(ValueError, match="base_height_l2 target_height must be finite"):
        _reward_value(env, manager_terms.base_height_l2, target_height=np.nan)

    cast(Any, env).scene.robot.data.root_link_lin_vel_b[1, 2] = np.inf
    with pytest.raises(ValueError, match="lin_vel_z_l2 root linear velocity contains NaN or Inf"):
        _reward_value(env, manager_terms.lin_vel_z_l2)

    with pytest.raises(KeyError, match="RewardManager term 'parity'.*asset_cfg.*missing"):
        _reward_value(
            _parity_env(),
            manager_terms.joint_deviation_l1,
            asset_cfg=SceneEntityCfg("missing"),
        )
