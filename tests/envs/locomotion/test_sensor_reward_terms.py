"""Focused tests for the shared named-sensor reward terms."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from unisim.backend.base import BackendSensorView

from unilab.dtype_config import get_global_dtype
from unilab.managers import RewardTermCfg
from unilab.managers._types import ManagerBasedRlEnv
from unilab.tasks.locomotion.common import manager_terms, sensor_reward_terms


class _Scene:
    def __init__(self, values: dict[str, np.ndarray]) -> None:
        self.values = values
        self.bound_names: tuple[str, ...] | None = None

    def bind_sensor_data(self, names) -> BackendSensorView:
        names = tuple(names)
        self.bound_names = names
        dimensions = tuple(self.values[name].shape[1] for name in names)

        def read() -> np.ndarray:
            return np.concatenate([self.values[name] for name in names], axis=1)

        view = BackendSensorView("fake", names, dimensions, 2, read)
        view.read()  # Mirror SimBackend.bind_sensor_data materialization validation.
        return view


class _Commands:
    def __init__(self, command: np.ndarray) -> None:
        self.command = command

    def get_command(self, name: str) -> np.ndarray:
        if name != "twist":
            raise KeyError(name)
        return self.command


def _env(
    sensors: dict[str, np.ndarray] | None = None,
    command: np.ndarray | None = None,
) -> tuple[ManagerBasedRlEnv, _Scene]:
    scene = _Scene(
        sensors
        or {
            "local_linvel": np.array([[0.2, 0.1, -0.3], [0.0, 0.0, 0.5]], dtype=np.float32),
            "gyro": np.array([[0.1, -0.2, 0.4], [0.0, 0.0, -0.2]], dtype=np.float32),
            "upvector": np.array([[0.0, 0.0, 1.0], [0.3, -0.4, 0.9]], dtype=np.float32),
        }
    )
    if command is None:
        command = np.array([[0.5, -0.2, 0.3], [-0.1, 0.4, -0.2]], dtype=np.float32)
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(num_envs=2, scene=scene, command_manager=_Commands(command)),
    )
    return env, scene


def _term(term_type: type, env: ManagerBasedRlEnv, **params: Any):
    return term_type(RewardTermCfg(func=term_type, weight=1.0, params=params), env)


def test_track_lin_vel_matches_legacy_equation() -> None:
    env, scene = _env()
    term = _term(
        sensor_reward_terms.track_lin_vel, env, sensor_name="local_linvel", tracking_sigma=0.25
    )
    assert scene.bound_names == ("local_linvel",)
    expected = np.exp(-np.array([0.18, 0.17]) / 0.25)
    np.testing.assert_allclose(term(env), expected, rtol=1e-6)


def test_track_ang_vel_matches_legacy_equation() -> None:
    env, _ = _env()
    term = _term(sensor_reward_terms.track_ang_vel, env, sensor_name="gyro", tracking_sigma=0.25)
    expected = np.exp(-np.array([(0.3 - 0.4) ** 2, (-0.2 - -0.2) ** 2]) / 0.25)
    np.testing.assert_allclose(term(env), expected, rtol=1e-6)


def test_lin_vel_z_ang_vel_xy_and_orientation_read_declared_sensor() -> None:
    env, scene = _env()
    lin_vel_z = _term(sensor_reward_terms.lin_vel_z, env, sensor_name="local_linvel")
    ang_vel_xy = _term(sensor_reward_terms.ang_vel_xy, env, sensor_name="gyro")
    orientation = _term(sensor_reward_terms.orientation, env, sensor_name="upvector")

    np.testing.assert_allclose(lin_vel_z(env), [0.09, 0.25], rtol=1e-6)
    np.testing.assert_allclose(ang_vel_xy(env), [0.05, 0.0], rtol=1e-6)
    np.testing.assert_allclose(orientation(env), [0.0, 0.25], rtol=1e-6)
    assert scene.bound_names == ("upvector",)


def test_tracking_terms_validate_params_at_construction() -> None:
    env, _ = _env()
    with pytest.raises(ValueError, match="tracking_sigma"):
        _term(
            sensor_reward_terms.track_lin_vel, env, sensor_name="local_linvel", tracking_sigma=0.0
        )
    with pytest.raises(ValueError, match="command_name"):
        _term(sensor_reward_terms.track_lin_vel, env, sensor_name="local_linvel", command_name="")
    with pytest.raises(KeyError, match="unavailable"):
        _term(
            sensor_reward_terms.track_lin_vel,
            env,
            sensor_name="local_linvel",
            command_name="missing",
        )
    with pytest.raises(TypeError, match="unsupported parameters"):
        _term(sensor_reward_terms.lin_vel_z, env, sensor_name="local_linvel", bogus=1.0)


def test_terms_fail_closed_on_missing_or_misshapen_sensor() -> None:
    env, _ = _env()
    with pytest.raises(KeyError, match="missing_linvel"):
        _term(sensor_reward_terms.lin_vel_z, env, sensor_name="missing_linvel")
    with pytest.raises(ValueError, match="sensor_name"):
        _term(sensor_reward_terms.lin_vel_z, env)


def test_terms_reject_non_vec3_sensor() -> None:
    env, _ = _env(sensors={"scalar": np.ones((2, 1), dtype=np.float32)})
    with pytest.raises(ValueError, match="3 values"):
        _term(sensor_reward_terms.orientation, env, sensor_name="scalar")


def test_alive_returns_unconditional_ones() -> None:
    env, _ = _env()
    result = manager_terms.alive(env)
    np.testing.assert_array_equal(result, np.ones(2))
    assert result.dtype == np.dtype(get_global_dtype())


def test_hot_paths_use_only_cached_runtime_objects() -> None:
    for term_type in (
        sensor_reward_terms.track_lin_vel,
        sensor_reward_terms.track_ang_vel,
        sensor_reward_terms.lin_vel_z,
        sensor_reward_terms.ang_vel_xy,
        sensor_reward_terms.orientation,
    ):
        source = inspect.getsource(term_type.__call__)
        for forbidden in (
            "ASSETS_ROOT_PATH",
            "model_file",
            "getattr(",
            "hasattr(",
            "._backend",
        ):
            assert forbidden not in source, f"{term_type.__name__} hot path references {forbidden}"
