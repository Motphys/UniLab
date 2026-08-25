"""Unit contracts for T800-specific Manager-Based terms."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from unilab.base.backend.base import SimBackend
from unilab.base.entity import EntityCfg, EntityScene
from unilab.managers import RewardTermCfg
from unilab.managers._types import ManagerBasedRlEnv
from unilab.tasks.locomotion.t800.manager_terms import (
    T800JointPositionActionCfg,
    compute_lateral_feet_penalty,
    penalty_close_feet_lateral,
)


class _ActionBackend:
    backend_type = "fake"
    num_envs = 2
    num_actuators = 3

    def __init__(self) -> None:
        self.actuator_names = ("knee_motor", "hip_motor", "head_motor")
        self.target_joint_names = ("knee", "hip", "head")
        self.joint_index = {"hip": 0, "knee": 1, "head": 2}
        self.dof_pos = np.zeros((self.num_envs, 3), dtype=np.float32)

    def get_actuator_names(self) -> tuple[str, ...]:
        return self.actuator_names

    def get_actuator_joint_names(self) -> tuple[str, ...]:
        return self.target_joint_names

    def get_actuator_ctrl_range(self) -> np.ndarray:
        return np.tile(np.asarray([[-10.0, 10.0]], dtype=np.float32), (3, 1))

    def get_joint_dof_pos_indices(self, names: list[str]) -> np.ndarray:
        return np.asarray([self.joint_index[name] for name in names], dtype=np.int32)

    def get_joint_dof_vel_indices(self, names: list[str]) -> np.ndarray:
        return self.get_joint_dof_pos_indices(names)

    def get_dof_pos(self) -> np.ndarray:
        return self.dof_pos

    def get_dof_vel(self) -> np.ndarray:
        return np.zeros_like(self.dof_pos)

    def get_default_dof_pos(self) -> np.ndarray:
        return np.asarray([0.1, 0.2, 0.3], dtype=np.float32)

    def get_joint_range(self) -> np.ndarray:
        return np.tile(np.asarray([[-1.0, 1.0]], dtype=np.float32), (3, 1))


def _action(
    *,
    active: tuple[str, ...] = ("hip", "knee"),
    held: tuple[str, ...] = ("head",),
    clip: dict[str, tuple[float, float]] | None = None,
) -> tuple[Any, np.ndarray, EntityScene]:
    backend = _ActionBackend()
    control = np.zeros((backend.num_envs, backend.num_actuators), dtype=np.float32)
    scene = EntityScene(
        {
            "robot": EntityCfg(
                joint_names=("hip", "knee", "head"),
                actuator_names=backend.actuator_names,
            )
        },
        cast(SimBackend, backend),
        control,
    )
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(num_envs=backend.num_envs, scene=scene),
    )
    scale = {name: {"hip": 2.0, "knee": 3.0}.get(name, 1.0) for name in active}
    cfg = T800JointPositionActionCfg(
        entity_name="robot",
        actuator_names=active,
        held_actuator_names=held,
        scale=scale,
        use_default_offset=True,
        clip=clip,
    )
    return cfg.build(env), control, scene


def test_t800_action_writes_active_targets_and_holds_non_policy_defaults() -> None:
    action, control, scene = _action()
    scene["robot"].data.encoder_bias[:] = np.asarray([[0.01, 0.02, 0.03]])
    raw = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    action.process_actions(raw)
    action.apply_actions()

    assert action.action_dim == 2
    assert action.target_names == ["hip", "knee"]
    np.testing.assert_allclose(action.processed_action, raw * [2.0, 3.0] + [0.1, 0.2])
    np.testing.assert_allclose(control[:, 1], action.processed_action[:, 0] - 0.01)
    np.testing.assert_allclose(control[:, 0], action.processed_action[:, 1] - 0.02)
    np.testing.assert_allclose(control[:, 2], 0.3 - 0.03)


def test_t800_action_clips_active_targets_before_mapping_and_keeps_held_target() -> None:
    action, control, scene = _action(clip={"hip": (-1.0, 1.0), "knee": (-2.0, 2.0)})
    scene["robot"].data.encoder_bias[:] = np.asarray(
        [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]], dtype=np.float32
    )
    raw = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    action.process_actions(raw)
    action.apply_actions()

    np.testing.assert_allclose(action.processed_action, [[1.0, 2.0], [1.0, 2.0]])
    np.testing.assert_allclose(control[:, 1], [1.0 - 0.01, 1.0 - 0.04])
    np.testing.assert_allclose(control[:, 0], [2.0 - 0.02, 2.0 - 0.05])
    np.testing.assert_allclose(control[:, 2], [0.3 - 0.03, 0.3 - 0.06])


@pytest.mark.parametrize(
    ("active", "held", "message"),
    [
        (("hip", "head"), ("head",), "overlap"),
        (("hip",), ("head",), "partition"),
        (("hip", "hip"), ("knee", "head"), "duplicate"),
        (("hip", "knee"), ("head", "head"), "duplicate"),
        (("hip", "knee"), ("missing",), "match"),
    ],
)
def test_t800_action_partition_fails_closed(
    active: tuple[str, ...],
    held: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _action(active=active, held=held)


class _LateralScene:
    def __init__(self, sensor_data: dict[str, np.ndarray]) -> None:
        self.sensor_data = sensor_data
        self.bound_names: tuple[str, ...] | None = None

    def bind_sensor_data(self, names: tuple[str, ...]):
        self.bound_names = tuple(names)
        arrays = [self.sensor_data[name] for name in names]
        return SimpleNamespace(
            dimensions=tuple(array.shape[1] for array in arrays),
            backend_type="fake",
            read=lambda: np.concatenate(arrays, axis=1),
        )


def _lateral_env(
    *,
    left: np.ndarray | None = None,
    right: np.ndarray | None = None,
    base_quat: np.ndarray | None = None,
) -> tuple[ManagerBasedRlEnv, _LateralScene]:
    left = (
        np.asarray([[0.0, 0.05, 0.0], [0.0, -0.10, 0.0]], dtype=np.float32)
        if left is None
        else left
    )
    right = (
        np.asarray([[0.0, -0.05, 0.0], [0.0, 0.10, 0.0]], dtype=np.float32)
        if right is None
        else right
    )
    base_quat = (
        np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        if base_quat is None
        else base_quat
    )
    scene = _LateralScene(
        {
            "left_foot_pos": left,
            "right_foot_pos": right,
            "base_link_quaternion": base_quat,
        }
    )
    env = cast(ManagerBasedRlEnv, SimpleNamespace(num_envs=left.shape[0], scene=scene))
    return env, scene


def _lateral_term(env: ManagerBasedRlEnv, **params: Any) -> penalty_close_feet_lateral:
    return penalty_close_feet_lateral(
        RewardTermCfg(
            func=penalty_close_feet_lateral,
            weight=-1.0,
            params={"min_width": 0.20, "sigma": 0.04, **params},
        ),
        env,
    )


def test_lateral_penalty_is_invariant_to_fore_aft_foot_offsets() -> None:
    left = np.asarray([[0.0, 0.05, 0.0], [3.0, 0.05, 1.0]], dtype=np.float32)
    right = np.asarray([[0.0, -0.05, 0.0], [-4.0, -0.05, -2.0]], dtype=np.float32)
    quat = np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=np.float32)

    penalty = compute_lateral_feet_penalty(left, right, quat, min_width=0.20, sigma=0.04)

    assert penalty[0] == pytest.approx(penalty[1])


def test_lateral_penalty_is_invariant_to_world_yaw() -> None:
    local_left = np.asarray([[1.0, 0.05, 0.0]], dtype=np.float32)
    local_right = np.asarray([[-2.0, -0.05, 0.0]], dtype=np.float32)
    yaw_90_left = np.asarray([[-0.05, 1.0, 0.0]], dtype=np.float32)
    yaw_90_right = np.asarray([[0.05, -2.0, 0.0]], dtype=np.float32)
    identity = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    yaw_90 = np.asarray([[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]], dtype=np.float32)

    local_penalty = compute_lateral_feet_penalty(
        local_left, local_right, identity, min_width=0.20, sigma=0.04
    )
    yawed_penalty = compute_lateral_feet_penalty(
        yaw_90_left, yaw_90_right, yaw_90, min_width=0.20, sigma=0.04
    )

    np.testing.assert_allclose(yawed_penalty, local_penalty, atol=1.0e-6)


def test_lateral_penalty_is_zero_at_or_above_minimum_width() -> None:
    left = np.asarray([[0.0, 0.10, 0.0], [0.0, 0.20, 0.0]], dtype=np.float32)
    right = np.asarray([[0.0, -0.10, 0.0], [0.0, -0.10, 0.0]], dtype=np.float32)
    quat = np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=np.float32)

    penalty = compute_lateral_feet_penalty(left, right, quat, min_width=0.20, sigma=0.04)

    np.testing.assert_array_equal(penalty, 0.0)


def test_lateral_penalty_is_near_one_for_crossed_feet() -> None:
    left = np.asarray([[0.0, -0.10, 0.0]], dtype=np.float32)
    right = np.asarray([[0.0, 0.10, 0.0]], dtype=np.float32)
    quat = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    penalty = compute_lateral_feet_penalty(left, right, quat, min_width=0.20, sigma=0.04)

    assert penalty[0] > 0.999


def test_lateral_term_binds_named_sensors_and_matches_helper() -> None:
    env, scene = _lateral_env()
    term = _lateral_term(env)

    result = term(env)

    assert scene.bound_names == (
        "left_foot_pos",
        "right_foot_pos",
        "base_link_quaternion",
    )
    expected = compute_lateral_feet_penalty(
        scene.sensor_data["left_foot_pos"],
        scene.sensor_data["right_foot_pos"],
        scene.sensor_data["base_link_quaternion"],
        min_width=0.20,
        sigma=0.04,
    )
    np.testing.assert_allclose(result, expected)


def test_lateral_term_fails_closed_when_a_named_sensor_is_missing() -> None:
    env, scene = _lateral_env()
    del scene.sensor_data["right_foot_pos"]

    with pytest.raises(KeyError, match="right_foot_pos"):
        _lateral_term(env)


def test_lateral_term_rejects_wrong_sensor_dimensions_at_construction() -> None:
    env, _ = _lateral_env(base_quat=np.zeros((2, 3), dtype=np.float32))

    with pytest.raises(ValueError, match=r"dimensions.*\(3, 3, 3\)"):
        _lateral_term(env)


@pytest.mark.parametrize(
    ("params", "error", "message"),
    [
        ({"min_width": -0.01}, ValueError, "min_width"),
        ({"sigma": 0.0}, ValueError, "sigma"),
        ({"sigma": -0.01}, ValueError, "sigma"),
        ({"min_width": float("nan")}, ValueError, "finite"),
        ({"sigma": float("inf")}, ValueError, "finite"),
        ({"min_width": True}, TypeError, "real number"),
    ],
)
def test_lateral_term_rejects_invalid_parameters_at_construction(
    params: dict[str, Any], error: type[Exception], message: str
) -> None:
    env, _ = _lateral_env()

    with pytest.raises(error, match=message):
        _lateral_term(env, **params)
