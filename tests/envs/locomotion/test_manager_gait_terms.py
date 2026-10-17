"""Focused parity tests for task-owned quadruped Manager-Based terms."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from unisim.backend.base import BackendSensorView

from unilab.managers import (
    ObservationGroupCfg,
    ObservationManager,
    ObservationTermCfg,
    RewardManager,
    RewardTermCfg,
)
from unilab.managers._types import ManagerBasedRlEnv
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.tasks.locomotion.common import manager_terms, sensor_reward_terms

CONTACTS = ("fl_contact", "fr_contact", "rl_contact", "rr_contact")
POSITIONS = ("fl_pos", "fr_pos", "rl_pos", "rr_pos")


class _Scene:
    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.values = {
            "fl_contact": np.array([[0.2], [0.0]], dtype=np.float32),
            # Width-3 contact ``force`` sensors follow the MuJoCo contact-frame
            # convention: column 0 is the normal force, columns 1-2 are
            # tangential and must not gate contact.
            "fr_contact": np.array([[0, 0.9, 0.9], [0.2, 0, 0]], dtype=np.float32),
            "rl_contact": np.array([[0.0], [0.2]], dtype=np.float32),
            "rr_contact": np.array([[0.2, 0, 0], [0, 0.9, 0.9]], dtype=np.float32),
            "fl_pos": np.array([[0, 0, 0.1], [0, 0, 0]], dtype=np.float32),
            "fr_pos": np.array([[0, 0, 0.2], [0, 0, 0.1]], dtype=np.float32),
            "rl_pos": np.array([[0, 0, 0.1], [0, 0, 0.2]], dtype=np.float32),
            "rr_pos": np.array([[0, 0, 0], [0, 0, 0.1]], dtype=np.float32),
        }
        self.refresh_tensor_read_plan()

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

    def refresh_tensor_read_plan(self, device: torch.device = torch.device("cpu")) -> None:
        values = {
            name: torch.as_tensor(value, device=device) for name, value in self.values.items()
        }
        requests: list[tuple[str, tuple[str, ...]]] = []

        def sensor_tensor_views(entity, names):
            requests.append((entity, tuple(names)))
            return SimpleNamespace(values={name: values[name] for name in names})

        self._tensor_read_plan = SimpleNamespace(
            device=device,
            sensor_names={"robot": tuple(self.values)},
            sensor_tensor_views=sensor_tensor_views,
        )


class _ParityScene:
    def __init__(self) -> None:
        self._tensor_read_plan = SimpleNamespace(
            device=torch.device("cpu"),
            sensor_names={"robot": ("local_linvel", "gyro")},
            sensor_tensor_views=lambda entity, names: SimpleNamespace(
                values={name: torch.as_tensor(self.values[name]) for name in names}
            ),
            body_tensor_view=lambda entity, names: SimpleNamespace(
                pos_w=torch.tensor([[[1.0, 2.0, 0.4]], [[-1.0, 0.5, 0.2]]], dtype=torch.float32),
                quat_w=torch.tensor(
                    [[[0.70710678, 0.0, 0.0, -0.70710678]], [[1.0, 0.0, 0.0, 0.0]]],
                    dtype=torch.float32,
                ),
                lin_vel_w=torch.tensor(
                    [[[0.0, 2.0, 3.0]], [[0.3, -0.2, 0.5]]], dtype=torch.float32
                ),
            ),
            joint_tensor_view=lambda entity: SimpleNamespace(
                joint_pos=torch.tensor([[0.2, -0.1, 0.5], [-0.4, 0.3, 0.1]], dtype=torch.float32)
            ),
        )
        self.values = {
            "local_linvel": np.array([[0.2, 0.1, -0.3], [0.3, -0.2, 0.5]], dtype=np.float32),
            "gyro": np.array([[0.1, -0.2, 0.4], [-0.3, 0.2, -0.1]], dtype=np.float32),
        }
        self.robot = SimpleNamespace(
            data=SimpleNamespace(
                root_link_lin_vel_b=np.array(
                    [[0.2, 0.1, -0.3], [0.3, -0.2, 0.5]], dtype=np.float32
                ),
                root_link_ang_vel_b=np.array(
                    [[0.1, -0.2, 0.4], [-0.3, 0.2, -0.1]], dtype=np.float32
                ),
                default_joint_pos_torch=lambda device: torch.tensor(
                    [[0.1, -0.2, 0.5], [-0.1, 0.1, 0.0]], dtype=torch.float32
                ),
            )
        )

    def __getitem__(self, name: str):
        if name != "robot":
            raise KeyError(name)
        return self.robot

    def bind_sensor_data(self, names):
        names = tuple(names)
        dimensions = tuple(self.values[name].shape[1] for name in names)

        def read() -> np.ndarray:
            return np.concatenate([self.values[name] for name in names], axis=1)

        view = BackendSensorView("fake", names, dimensions, 2, read)
        view.read()
        return view


class _Commands:
    def __init__(self, command: torch.Tensor | None = None) -> None:
        self.command = (
            torch.tensor([[0.5, -0.2, 0.3], [-0.1, 0.4, -0.2]], dtype=torch.float32)
            if command is None
            else command
        )

    def get_command(self, name: str) -> torch.Tensor:
        if name != "twist":
            raise KeyError(name)
        return self.command


def _env(counter: int = 0, scene: _Scene | None = None, device: str = "cpu") -> ManagerBasedRlEnv:
    scene = scene or _Scene()
    scene.refresh_tensor_read_plan(torch.device(device))
    return cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=2,
            device=torch.device(device),
            common_step_counter=counter,
            episode_length_buf=np.array([counter, 0]),
            step_dt=0.02,
            scene=scene,
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
            device=torch.device("cpu"),
            cfg=SimpleNamespace(
                scene=SimpleNamespace(entities={"robot": SimpleNamespace(root_body_name="base")})
            ),
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
    assert isinstance(initial, torch.Tensor)
    np.testing.assert_array_equal(initial, [[0.0, 0.5, 0.5, 0.0]] * 2)

    cast(Any, env).common_step_counter = 25
    env.episode_length_buf[:] = [0, 999]  # Partial reset does not reset the global clock.
    advanced = manager.compute_group("policy")
    assert isinstance(advanced, torch.Tensor)
    np.testing.assert_allclose(
        advanced, [[1.1920929e-7, 0.5000001, 0.5000001, 1.1920929e-7]] * 2, atol=1e-8
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_standing_aware_gait_freezes_only_standing_environments(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA or ROCm GPU is unavailable")
    env = _env(device=device)
    cast(Any, env).command_manager.command = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float32, device=device
    )
    manager = _observations(env, command_name="twist", command_threshold=0.1)

    cast(Any, env).common_step_counter = 1
    phase = manager.compute_group("policy")
    assert isinstance(phase, torch.Tensor)
    assert phase.device == env.device
    np.testing.assert_allclose(
        phase.cpu(),
        [[0.0, 0.5, 0.5, 0.0], [0.04, 0.54, 0.54, 0.04]],
        atol=1e-7,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_standing_aware_foot_rewards_gate_swing_and_expect_planted_feet(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA or ROCm GPU is unavailable")
    env = _env(counter=5, device=device)
    cast(Any, env).command_manager.command = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float32, device=device
    )
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
    assert value.device == env.device
    expected_moving_swing = (1.0 + np.exp(-1.0)) / 4.0
    np.testing.assert_allclose(value.cpu(), [0.5, expected_moving_swing], atol=1e-7)


def test_standing_penalties_match_a2_legacy_gates() -> None:
    parity_env = _parity_env()
    cast(Any, parity_env).command_manager.command.copy_(
        torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float32)
    )
    stand_still = _reward_value(
        parity_env,
        manager_terms.stand_still_l1,
        command_name="twist",
        command_threshold=0.1,
    )
    np.testing.assert_allclose(stand_still, [0.2, 0.0], atol=1e-7)

    foot_env = _env()
    cast(Any, foot_env).command_manager.command.copy_(
        torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float32)
    )
    feet_air = _reward_value(
        foot_env,
        manager_terms.feet_air_while_standing,
        sensor_names=CONTACTS,
        command_name="twist",
        command_threshold=0.1,
    )
    np.testing.assert_array_equal(feet_air, [2.0, 0.0])


def test_feet_air_while_standing_fails_closed_without_packed_sensors() -> None:
    env = _env()
    cfg = RewardTermCfg(
        func=manager_terms.feet_air_while_standing,
        weight=1.0,
        params={"sensor_names": CONTACTS, "command_name": "twist"},
    )
    built = manager_terms.feet_air_while_standing(cfg=cfg, env=cast(Any, env))
    cast(Any, env).scene._tensor_read_plan.sensor_names = {"robot": ()}

    with pytest.raises(NotImplementedError, match="packed Torch read plan"):
        built(env)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_foot_rewards_match_legacy_equations_and_read_only_bound_views(dtype) -> None:
    scene = _Scene()
    scene.values = {name: values.astype(dtype) for name, values in scene.values.items()}
    scene.refresh_tensor_read_plan()
    manager = _rewards(_env(5, scene))
    assert scene.calls == {"bind": 2, "read": 2}
    result = manager.compute(dt=0.02)
    assert scene.calls == {"bind": 2, "read": 2}

    phase = np.array([[0.2, 0.7, 0.7, 0.2]] * 2)
    contact = np.array([[True, False, False, True], [False, True, True, False]])
    heights = np.array([[0.1, 0.2, 0.1, 0.0], [0.0, 0.1, 0.2, 0.1]])
    expected = np.mean(contact == (phase < 0.6), axis=1)
    expected += np.mean(np.exp(-np.square(heights - 0.1) / 0.01) * (phase >= 0.6), axis=1)
    np.testing.assert_allclose(result, expected, atol=1e-7)
    manager.compute(dt=0.02)
    assert scene.calls == {"bind": 2, "read": 2}


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_foot_rewards_use_compiled_tensor_sensors_without_host_reads(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA or ROCm GPU is unavailable")
    scene = _Scene()
    manager = _rewards(_env(5, scene, device=device))
    expected = manager.compute(dt=0.02).clone()
    calls = scene.calls.copy()
    values = {name: torch.as_tensor(value, device=device) for name, value in scene.values.items()}
    requests = []

    def sensor_tensor_views(entity, names):
        requests.append((entity, names))
        return SimpleNamespace(values={name: values[name] for name in names})

    cast(Any, scene)._tensor_read_plan = SimpleNamespace(
        device=torch.device(device),
        sensor_names={"robot": CONTACTS + POSITIONS},
        sensor_tensor_views=sensor_tensor_views,
    )
    torch.testing.assert_close(manager.compute(dt=0.02), expected)
    assert scene.calls == calls
    assert requests == [("robot", CONTACTS), ("robot", POSITIONS)]


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
    values = {name: torch.as_tensor(value) for name, value in scene.values.items()}

    def sensor_tensor_views(entity, names):
        # Mirror the public read-plan validation boundary: non-finite sensor
        # values fail closed with the requesting sensor named in the error.
        if not bool(torch.isfinite(values["fl_contact"]).all()):
            raise ValueError("fl_contact has NaN or Inf")
        return SimpleNamespace(values={name: values[name] for name in names})

    cast(Any, scene)._tensor_read_plan = SimpleNamespace(
        device=torch.device("cpu"),
        sensor_names={"robot": tuple(scene.values)},
        sensor_tensor_views=sensor_tensor_views,
    )
    with pytest.raises(ValueError, match="fl_contact has NaN or Inf"):
        manager.compute(dt=0.02)


def test_base_reward_terms_match_go2_flat_equations() -> None:
    env = _parity_env()
    data = cast(Any, env).scene
    local_linvel = data.values["local_linvel"]
    gyro = data.values["gyro"]
    root_pos_w = torch.tensor([[[1.0, 2.0, 0.4]], [[-1.0, 0.5, 0.2]]], dtype=torch.float32)[:, 0]
    root_lin_vel_b = torch.tensor([[-2.0, 0.0, 3.0], [0.3, -0.2, 0.5]], dtype=torch.float32)
    joint_pos = torch.tensor([[0.2, -0.1, 0.5], [-0.4, 0.3, 0.1]], dtype=torch.float32)
    default_joint_pos = torch.tensor([[0.1, -0.2, 0.5], [-0.1, 0.1, 0.0]], dtype=torch.float32)
    command = cast(Any, env).command_manager.command
    sigma = 0.5
    actual = {
        "track_xy": _reward_value(
            env,
            sensor_reward_terms.track_lin_vel,
            tracking_sigma=sigma,
            sensor_name="local_linvel",
            command_name="twist",
        ),
        "root_track_xy": _reward_value(
            env,
            sensor_reward_terms.root_track_lin_vel,
            tracking_sigma=sigma,
            command_name="twist",
        ),
        "track_yaw": _reward_value(
            env,
            sensor_reward_terms.track_ang_vel,
            tracking_sigma=sigma,
            sensor_name="gyro",
            command_name="twist",
        ),
        "lin_z": _reward_value(
            env,
            sensor_reward_terms.lin_vel_z,
            sensor_name="local_linvel",
        ),
        "root_lin_z": _reward_value(env, sensor_reward_terms.root_lin_vel_z),
        "ang_xy": _reward_value(
            env,
            sensor_reward_terms.ang_vel_xy,
            sensor_name="gyro",
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
        "track_xy": torch.exp(-(command[:, :2] - local_linvel[:, :2]).square().sum(dim=1) / sigma),
        "root_track_xy": torch.exp(
            -(command[:, :2] - root_lin_vel_b[:, :2]).square().sum(dim=1) / sigma
        ),
        "track_yaw": np.exp(-np.square(command[:, 2] - gyro[:, 2]) / sigma),
        "lin_z": np.square(local_linvel[:, 2]),
        "root_lin_z": torch.square(root_lin_vel_b[:, 2]),
        "ang_xy": np.sum(np.square(gyro[:, :2]), axis=1),
        "height": torch.square(root_pos_w[:, 2] - 0.3),
        "pose": (joint_pos - default_joint_pos).abs().sum(dim=1),
    }
    for name in expected:
        assert actual[name].shape == (2,)
        assert actual[name].dtype == torch.float32
        expected_tensor = expected[name]
        if isinstance(expected_tensor, torch.Tensor):
            expected_tensor = expected_tensor.to(device=actual[name].device)
        if isinstance(expected_tensor, np.ndarray):
            expected_tensor = torch.from_numpy(np.ascontiguousarray(expected_tensor))
        torch.testing.assert_close(actual[name], expected_tensor, rtol=1e-6, atol=1e-7)


def test_base_reward_terms_fail_closed_at_nearest_boundary() -> None:
    env = _parity_env()
    with pytest.raises(ValueError, match="track_lin_vel tracking_sigma must be greater than 0.0"):
        _reward_value(
            env,
            sensor_reward_terms.track_lin_vel,
            tracking_sigma=0.0,
            sensor_name="local_linvel",
            command_name="twist",
        )
    with pytest.raises(KeyError, match="track_ang_vel command capability 'missing'"):
        _reward_value(
            env,
            sensor_reward_terms.track_ang_vel,
            tracking_sigma=0.5,
            sensor_name="gyro",
            command_name="missing",
        )
    with pytest.raises(ValueError, match="base_height_l2 target_height must be finite"):
        _reward_value(env, manager_terms.base_height_l2, target_height=np.nan)

    scene = cast(Any, env).scene
    scene.values = {
        "local_linvel": cast(Any, env).scene.robot.data.root_link_lin_vel_b.copy(),
        "gyro": cast(Any, env).scene.robot.data.root_link_ang_vel_b.copy(),
    }
    scene.values["local_linvel"][1, 2] = np.inf
    scene._tensor_read_plan = SimpleNamespace(
        device=torch.device("cpu"),
        sensor_names={"robot": ("local_linvel", "gyro")},
        sensor_tensor_views=lambda entity, names: SimpleNamespace(
            values={name: torch.as_tensor(scene.values[name]) for name in names}
        ),
    )
    with pytest.raises(ValueError, match="named-sensor capability.*NaN or Inf"):
        _reward_value(
            env,
            sensor_reward_terms.lin_vel_z,
            sensor_name="local_linvel",
        )

    with pytest.raises(KeyError, match="RewardManager term 'parity'.*asset_cfg.*missing"):
        _reward_value(
            _parity_env(),
            manager_terms.joint_deviation_l1,
            asset_cfg=SceneEntityCfg("missing"),
        )
