"""Upstream-derived NumPy tests for basic manager observation terms."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from unisim.backend.base import SimBackend

from unilab.base.entity import EntityCfg, EntityScene
from unilab.envs import mdp
from unilab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg
from unilab.managers._types import ManagerBasedRlEnv
from unilab.managers.scene_entity_config import SceneEntityCfg


class _Backend:
    backend_type = "fake"
    num_envs = 2
    num_actuators = 0

    def __init__(self) -> None:
        self.sensor_calls: Counter[str] = Counter()
        self.sensor_values = {
            "gyro": np.asarray([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]], dtype=np.float32),
            "upvector": np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        }
        self.joint_names = ("hip", "knee", "ankle")
        self.dof_pos = np.asarray(
            [[0.4, 0.1, -0.2], [0.0, 0.3, 0.7]],
            dtype=np.float32,
        )
        self.dof_vel = np.asarray(
            [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]],
            dtype=np.float32,
        )
        self.body_pos = np.zeros((2, 1, 3), dtype=np.float32)
        self.body_quat = np.zeros((2, 1, 4), dtype=np.float32)
        self.body_quat[0, 0, 0] = 1.0
        half = np.sqrt(0.5)
        self.body_quat[1, 0] = [half, half, 0.0, 0.0]
        self.body_lin_vel_w = np.zeros((2, 1, 3), dtype=np.float32)
        self.body_ang_vel_w = np.zeros((2, 1, 3), dtype=np.float32)
        self.body_lin_vel_b = np.asarray(
            [[[0.5, 0.25, 0.0]], [[-0.5, -0.25, 0.1]]], dtype=np.float32
        )
        self.body_ang_vel_b = np.asarray(
            [[[0.1, 0.2, 0.3]], [[-0.1, -0.2, -0.3]]], dtype=np.float32
        )

    def get_joint_dof_pos_indices(self, names) -> np.ndarray:
        return np.asarray([self.joint_names.index(name) for name in names], dtype=np.int32)

    def get_joint_dof_vel_indices(self, names) -> np.ndarray:
        return self.get_joint_dof_pos_indices(names)

    def get_body_ids(self, names) -> np.ndarray:
        if tuple(names) != ("base",):
            raise KeyError(names)
        return np.asarray([0], dtype=np.int32)

    def get_dof_pos(self) -> np.ndarray:
        return self.dof_pos

    def get_dof_vel(self) -> np.ndarray:
        return self.dof_vel

    def get_default_dof_pos(self) -> np.ndarray:
        return np.asarray([0.1, 0.2, 0.3], dtype=np.float32)

    def get_joint_range(self) -> np.ndarray:
        return np.tile(np.asarray([[-1.0, 1.0]], dtype=np.float32), (3, 1))

    def get_body_pos_w(self, ids: np.ndarray) -> np.ndarray:
        return self.body_pos[:, ids]

    def get_body_quat_w(self, ids: np.ndarray) -> np.ndarray:
        return self.body_quat[:, ids]

    def get_body_lin_vel_w(self, ids: np.ndarray) -> np.ndarray:
        return self.body_lin_vel_w[:, ids]

    def get_body_ang_vel_w(self, ids: np.ndarray) -> np.ndarray:
        return self.body_ang_vel_w[:, ids]

    def get_body_lin_vel_b(self, ids: np.ndarray) -> np.ndarray:
        return self.body_lin_vel_b[:, ids]

    def get_body_ang_vel_b(self, ids: np.ndarray) -> np.ndarray:
        return self.body_ang_vel_b[:, ids]

    def get_sensor_data(self, name: str) -> np.ndarray:
        self.sensor_calls["single"] += 1
        try:
            return self.sensor_values[name]
        except KeyError as exc:
            raise KeyError(f"unknown sensor {name!r}") from exc

    def get_sensor_data_batch(self, names: tuple[str, ...]) -> np.ndarray:
        self.sensor_calls["batch"] += 1
        values = [self.sensor_values[name].reshape(self.num_envs, -1) for name in names]
        return np.concatenate(values, axis=1)

    def _bind_sensor_data_reader(self, names: tuple[str, ...]):
        return lambda: self.get_sensor_data_batch(names)

    def bind_sensor_data(self, names):
        self.sensor_calls["bind"] += 1
        return SimBackend.bind_sensor_data(self, names)  # type: ignore[arg-type]


class _ActionManager:
    def __init__(self) -> None:
        self.action = np.arange(6, dtype=np.float32).reshape(2, 3)
        self._terms = {
            "legs": SimpleNamespace(raw_action=self.action[:, [0, 2]]),
        }

    def get_term(self, name: str):
        return self._terms[name]


class _CommandManager:
    def __init__(self) -> None:
        self.command = np.asarray([[1.0, 0.0, 0.2], [0.5, -0.1, -0.2]], dtype=np.float32)

    def get_command(self, name: str) -> np.ndarray:
        if name != "twist":
            raise KeyError(name)
        return self.command


class _FakeEnv:
    """Attribute container with identity hash/eq for per-env term caches."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def _env() -> tuple[ManagerBasedRlEnv, _Backend]:
    backend = _Backend()
    scene = EntityScene(
        {
            "robot": EntityCfg(
                root_body_name="base",
                joint_names=backend.joint_names,
            )
        },
        cast(SimBackend, backend),
    )
    env = cast(
        ManagerBasedRlEnv,
        _FakeEnv(
            num_envs=backend.num_envs,
            scene=scene,
            action_manager=_ActionManager(),
            command_manager=_CommandManager(),
            rng=np.random.default_rng(4),
        ),
    )
    return env, backend


def test_root_joint_action_and_command_terms_match_numpy_contract() -> None:
    env, backend = _env()
    robot = cast(Any, env.scene["robot"])
    robot.data.encoder_bias[:] = [[0.01, 0.02, 0.03], [-0.01, -0.02, -0.03]]

    np.testing.assert_array_equal(mdp.base_lin_vel(env), backend.body_lin_vel_b[:, 0])
    np.testing.assert_array_equal(mdp.base_ang_vel(env), backend.body_ang_vel_b[:, 0])
    np.testing.assert_allclose(
        mdp.projected_gravity(env),
        [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0]],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        mdp.joint_pos_rel(env),
        backend.dof_pos - [0.1, 0.2, 0.3],
    )
    np.testing.assert_allclose(
        mdp.joint_pos_rel(env, biased=True),
        backend.dof_pos + robot.data.encoder_bias - [0.1, 0.2, 0.3],
    )
    np.testing.assert_array_equal(mdp.joint_vel_rel(env), backend.dof_vel)
    np.testing.assert_array_equal(mdp.last_action(env), env.action_manager.action)
    np.testing.assert_array_equal(
        mdp.last_action(env, "legs"), env.action_manager.get_term("legs").raw_action
    )
    np.testing.assert_array_equal(
        mdp.generated_commands(env, "twist"), env.command_manager.get_command("twist")
    )


def test_scene_entity_selector_is_resolved_once_by_observation_manager() -> None:
    env, backend = _env()
    selector = SceneEntityCfg("robot", joint_names=("ankle", "hip"), preserve_order=True)
    manager = ObservationManager(
        {
            "policy": ObservationGroupCfg(
                terms={
                    "joint_pos": ObservationTermCfg(
                        func=mdp.joint_pos_rel,
                        params={"asset_cfg": selector},
                    ),
                    "joint_vel": ObservationTermCfg(
                        func=mdp.joint_vel_rel,
                        params={"asset_cfg": selector},
                    ),
                    "command": ObservationTermCfg(
                        func=mdp.generated_commands,
                        params={"command_name": "twist"},
                    ),
                }
            )
        },
        env,
    )

    result = manager.compute()["policy"]
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 7)
    np.testing.assert_allclose(result[:, :2], backend.dof_pos[:, [2, 0]] - [0.3, 0.1])
    np.testing.assert_array_equal(result[:, 2:4], backend.dof_vel[:, [2, 0]])
    assert selector.joint_ids == slice(None)
    resolved = manager.get_term_cfg("policy", "joint_pos").params["asset_cfg"]
    assert resolved.joint_ids == [2, 0]


def _sensor_manager(env: ManagerBasedRlEnv) -> ObservationManager:
    return ObservationManager(
        {
            "policy": ObservationGroupCfg(
                terms={
                    "gyro": ObservationTermCfg(
                        func=mdp.builtin_sensor,
                        params={"sensor_name": "gyro"},
                    ),
                    "gravity": ObservationTermCfg(
                        func=mdp.projected_gravity_from_sensor,
                        params={"sensor_name": "upvector"},
                    ),
                }
            )
        },
        env,
    )


def test_named_sensor_terms_bind_once_and_only_read_cached_views() -> None:
    env, backend = _env()

    manager = _sensor_manager(env)

    assert manager.group_obs_dim == {"policy": (6,)}
    assert backend.sensor_calls == {"bind": 2, "single": 2, "batch": 4}

    first = manager.compute_group("policy")
    second = manager.compute_group("policy")
    assert isinstance(first, np.ndarray)
    assert isinstance(second, np.ndarray)
    expected = np.concatenate(
        [backend.sensor_values["gyro"], -backend.sensor_values["upvector"]], axis=1
    )
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)
    assert backend.sensor_calls == {"bind": 2, "single": 2, "batch": 8}

    gyro_term = manager.get_term_cfg("policy", "gyro").func
    gravity_term = manager.get_term_cfg("policy", "gravity").func
    assert isinstance(gyro_term, mdp.builtin_sensor)
    assert isinstance(gravity_term, mdp.projected_gravity_from_sensor)


@pytest.mark.parametrize(
    ("sensor_name", "error", "message"),
    [
        ("", ValueError, "builtin_sensor.*non-empty sensor_name"),
        ("missing", KeyError, "builtin_sensor.*missing.*backend.*fake"),
    ],
)
def test_named_sensor_term_materialization_fails_closed(
    sensor_name: str, error: type[Exception], message: str
) -> None:
    env, _ = _env()
    with pytest.raises(error, match=message):
        ObservationManager(
            {
                "policy": ObservationGroupCfg(
                    terms={
                        "sensor": ObservationTermCfg(
                            func=mdp.builtin_sensor,
                            params={"sensor_name": sensor_name},
                        )
                    }
                )
            },
            env,
        )


def test_named_sensor_scene_seam_rejects_duplicate_names() -> None:
    env, _ = _env()
    with pytest.raises(ValueError, match="named-sensor.*unique"):
        env.scene.bind_sensor_data(("gyro", "gyro"))


def test_projected_gravity_sensor_requires_one_three_dimensional_view() -> None:
    env, backend = _env()
    backend.sensor_values["upvector"] = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(
        ValueError,
        match=r"projected_gravity_from_sensor.*3-D.*dimensions \(2,\).*backend 'fake'",
    ):
        ObservationManager(
            {
                "policy": ObservationGroupCfg(
                    terms={
                        "gravity": ObservationTermCfg(
                            func=mdp.projected_gravity_from_sensor,
                            params={"sensor_name": "upvector"},
                        )
                    }
                )
            },
            env,
        )


def test_named_sensor_runtime_drift_reports_term_and_backend() -> None:
    env, backend = _env()
    manager = ObservationManager(
        {
            "policy": ObservationGroupCfg(
                terms={
                    "gyro": ObservationTermCfg(
                        func=mdp.builtin_sensor,
                        params={"sensor_name": "gyro"},
                    )
                }
            )
        },
        env,
    )
    backend.sensor_values["gyro"] = np.full((2, 3), np.nan, dtype=np.float32)

    with pytest.raises(ValueError, match="builtin_sensor.*gyro.*backend 'fake'.*NaN or Inf"):
        manager.compute_group("policy")


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda env: mdp.last_action(env, "missing"), "Action term 'missing' not found"),
        (
            lambda env: mdp.generated_commands(env, "missing"),
            "Command term 'missing' not found",
        ),
        (
            lambda env: mdp.joint_pos_rel(env, biased=1),  # type: ignore[arg-type]
            "biased must be bool",
        ),
    ],
)
def test_invalid_term_requests_fail_explicitly(call, message: str) -> None:
    env, _ = _env()
    with pytest.raises((KeyError, TypeError), match=message):
        call(env)


def test_missing_entity_capability_fails_instead_of_returning_zeros() -> None:
    backend = _Backend()
    scene = EntityScene(
        {"robot": EntityCfg(joint_names=backend.joint_names)},
        cast(SimBackend, backend),
    )
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=2,
            scene=scene,
            action_manager=_ActionManager(),
            command_manager=_CommandManager(),
        ),
    )
    with pytest.raises(NotImplementedError, match="projected gravity.*not materialized"):
        mdp.projected_gravity(env)


def _imu_misalignment_manager(
    env: ManagerBasedRlEnv, max_angle_deg: float = 6.0
) -> ObservationManager:
    """Actor sees IMU-misaligned gyro/gravity; critic keeps the true values."""
    return ObservationManager(
        {
            "policy": ObservationGroupCfg(
                terms={
                    "base_ang_vel": ObservationTermCfg(
                        func=mdp.base_ang_vel_imu_misaligned,
                        params={"max_angle_deg": max_angle_deg},
                    ),
                    "projected_gravity": ObservationTermCfg(
                        func=mdp.projected_gravity_imu_misaligned,
                        params={"max_angle_deg": max_angle_deg},
                    ),
                }
            ),
            "critic": ObservationGroupCfg(
                terms={
                    "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
                    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
                }
            ),
        },
        env,
    )


def _rotation_angle_rad(raw: np.ndarray, rotated: np.ndarray) -> np.ndarray:
    cos = np.sum(raw * rotated, axis=-1) / (
        np.linalg.norm(raw, axis=-1) * np.linalg.norm(rotated, axis=-1)
    )
    return np.arccos(np.clip(cos, -1.0, 1.0))


def test_imu_misaligned_terms_share_one_per_env_constant_quaternion() -> None:
    env, backend = _env()
    manager = _imu_misalignment_manager(env, max_angle_deg=6.0)

    assert manager.group_obs_dim == {"policy": (6,), "critic": (6,)}
    obs = manager.compute()
    policy = obs["policy"]
    critic = obs["critic"]
    assert isinstance(policy, np.ndarray)
    assert isinstance(critic, np.ndarray)
    gyro_raw, gravity_raw = critic[:, :3], critic[:, 3:]
    gyro_rot, gravity_rot = policy[:, :3], policy[:, 3:]

    # The critic group keeps the true (unrotated) values.
    np.testing.assert_array_equal(gyro_raw, backend.body_ang_vel_b[:, 0])
    np.testing.assert_allclose(gravity_raw, [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0]], atol=1e-6)

    # A rotation preserves vector norms...
    np.testing.assert_allclose(
        np.linalg.norm(gyro_rot, axis=-1), np.linalg.norm(gyro_raw, axis=-1), atol=1e-6
    )
    np.testing.assert_allclose(np.linalg.norm(gravity_rot, axis=-1), 1.0, atol=1e-6)
    # ...and the gyro/gravity inner product, proving both actor terms were
    # rotated by the SAME per-env quaternion.
    np.testing.assert_allclose(
        np.sum(gyro_rot * gravity_rot, axis=-1),
        np.sum(gyro_raw * gravity_raw, axis=-1),
        atol=1e-6,
    )

    # Magnitude is bounded by max_angle_deg for every env, and the rotation is
    # actually applied (seeded RNG keeps this deterministic).
    max_angle_rad = np.deg2rad(6.0)
    assert (_rotation_angle_rad(gyro_raw, gyro_rot) <= max_angle_rad + 1e-5).all()
    assert (_rotation_angle_rad(gravity_raw, gravity_rot) <= max_angle_rad + 1e-5).all()
    assert _rotation_angle_rad(gravity_raw, gravity_rot).max() > 1e-4


def test_imu_misalignment_is_constant_across_calls_and_episode_resets() -> None:
    env, _ = _env()
    manager = _imu_misalignment_manager(env)

    first = manager.compute_group("policy")
    second = manager.compute_group("policy")
    manager.reset(np.arange(env.num_envs))
    after_reset = manager.compute_group("policy")

    assert isinstance(first, np.ndarray)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, after_reset)


def test_imu_misalignment_reproducible_for_same_env_seed() -> None:
    env_a, _ = _env()
    env_b, _ = _env()

    obs_a = _imu_misalignment_manager(env_a).compute_group("policy")
    obs_b = _imu_misalignment_manager(env_b).compute_group("policy")

    np.testing.assert_array_equal(obs_a, obs_b)


def test_imu_misalignment_zero_angle_is_identity() -> None:
    env, backend = _env()

    policy = _imu_misalignment_manager(env, max_angle_deg=0.0).compute_group("policy")

    assert isinstance(policy, np.ndarray)
    np.testing.assert_array_equal(policy[:, :3], backend.body_ang_vel_b[:, 0])
    np.testing.assert_allclose(policy[:, 3:], [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0]], atol=1e-6)


@pytest.mark.parametrize("max_angle_deg", [-1.0, float("nan"), True, "6"])
def test_imu_misalignment_rejects_invalid_max_angle(max_angle_deg) -> None:
    env, _ = _env()
    with pytest.raises(ValueError, match="max_angle_deg"):
        _imu_misalignment_manager(env, max_angle_deg=max_angle_deg)


def test_imu_misaligned_term_rejects_call_with_mismatched_angle() -> None:
    env, _ = _env()
    manager = _imu_misalignment_manager(env, max_angle_deg=6.0)
    term = manager.get_term_cfg("policy", "base_ang_vel").func
    assert isinstance(term, mdp.base_ang_vel_imu_misaligned)

    with pytest.raises(ValueError, match="bound to max_angle_deg"):
        term(env, max_angle_deg=3.0)


def test_observation_module_has_no_forbidden_runtime_dependencies() -> None:
    path = (
        Path(__file__).resolve().parents[3] / "src" / "unilab" / "envs" / "mdp" / "observations.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = ("torch", "unilab.ipc", "unilab.algos", "unilab.training", "unilab.base.backend")
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)] + [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not [name for name in imports if name.startswith(forbidden)]
