"""Upstream-derived NumPy tests for sensor-free manager reward terms."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from unilab.envs import mdp
from unilab.managers import RewardManager, RewardTermCfg
from unilab.managers._types import ManagerBasedRlEnv
from unilab.managers.scene_entity_config import SceneEntityCfg


class _Entity:
    joint_names = ("hip", "knee", "ankle")
    body_names = ("base", "torso")
    num_joints = 3
    num_bodies = 2

    def __init__(self) -> None:
        self.data = SimpleNamespace(
            joint_pos=np.asarray(
                [[0.1, -0.2, 0.3], [0.0, 0.4, -0.1], [0.2, 0.0, 0.5]],
                dtype=np.float32,
            ),
            default_joint_pos=np.asarray(
                [[0.1, 0.0, 0.2], [0.1, 0.0, 0.2], [0.1, 0.0, 0.2]],
                dtype=np.float32,
            ),
            soft_joint_pos_limits=np.asarray(
                [[-0.15, 0.15], [-0.3, 0.3], [-0.4, 0.4]],
                dtype=np.float32,
            ),
            joint_vel=np.asarray(
                [[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0], [0.0, -2.0, 1.0]],
                dtype=np.float32,
            ),
            projected_gravity_b=np.asarray(
                [[0.0, 0.0, -1.0], [0.3, 0.4, -0.866], [0.6, 0.0, -0.8]],
                dtype=np.float32,
            ),
            root_link_lin_vel_b=np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 0.5, 0.25], [-0.5, 0.0, -0.2]],
                dtype=np.float32,
            ),
            root_link_ang_vel_b=np.asarray(
                [[0.0, 0.0, 0.5], [0.1, 0.2, -0.25], [-0.3, 0.0, 0.1]],
                dtype=np.float32,
            ),
            body_link_ang_vel_w=np.asarray(
                [
                    [[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]],
                    [[-0.1, 0.4, 0.2], [-1.0, 0.5, 0.0]],
                    [[0.0, -0.2, 0.7], [0.25, -0.75, 0.1]],
                ],
                dtype=np.float32,
            ),
        )

    def find_joints(self, keys, preserve_order: bool = False):
        patterns = (keys,) if isinstance(keys, str) else tuple(keys)
        ids = [self.joint_names.index(name) for name in patterns]
        return ids, [self.joint_names[index] for index in ids]


class _CommandManager:
    def __init__(self) -> None:
        self.command = np.asarray(
            [[1.0, 0.0, 0.5], [0.25, 0.0, -0.5], [0.0, 0.0, 0.0]], dtype=np.float32
        )

    def get_command(self, name: str) -> np.ndarray:
        if name != "twist":
            raise KeyError(name)
        return self.command


def _env() -> ManagerBasedRlEnv:
    action = np.asarray(
        [[1.0, 2.0], [0.5, -0.5], [-1.0, 0.25]],
        dtype=np.float32,
    )
    return cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=3,
            scene={"robot": _Entity()},
            action_manager=SimpleNamespace(
                action=action,
                prev_action=action - 0.25,
                prev_prev_action=action - 0.75,
            ),
            command_manager=_CommandManager(),
            termination_manager=SimpleNamespace(
                terminated=np.asarray([False, True, False], dtype=np.bool_)
            ),
            max_episode_length_s=2.0,
        ),
    )


def test_generic_reward_terms_match_pinned_numpy_semantics() -> None:
    env = _env()
    entity = cast(Any, env.scene["robot"])

    np.testing.assert_array_equal(mdp.is_alive(env), [1.0, 0.0, 1.0])
    np.testing.assert_array_equal(mdp.is_terminated(env), [0.0, 1.0, 0.0])
    np.testing.assert_allclose(
        mdp.joint_vel_l2(env, SceneEntityCfg("robot", joint_ids=[2, 0])),
        np.sum(np.square(entity.data.joint_vel[:, [2, 0]]), axis=1),
    )
    np.testing.assert_allclose(mdp.action_rate_l2(env), 2 * 0.25**2)
    np.testing.assert_allclose(mdp.action_acc_l2(env), 2 * 0.25**2)
    np.testing.assert_allclose(
        mdp.flat_orientation_l2(env),
        np.sum(np.square(entity.data.projected_gravity_b[:, :2]), axis=1),
    )


def test_velocity_tracking_terms_match_pinned_equations() -> None:
    env = _env()
    entity = cast(Any, env.scene["robot"])
    command = cast(np.ndarray, env.command_manager.get_command("twist"))

    linear_error = np.sum(
        np.square(command[:, :2] - entity.data.root_link_lin_vel_b[:, :2]), axis=1
    ) + np.square(entity.data.root_link_lin_vel_b[:, 2])
    angular_error = np.square(command[:, 2] - entity.data.root_link_ang_vel_b[:, 2]) + np.sum(
        np.square(entity.data.root_link_ang_vel_b[:, :2]), axis=1
    )
    np.testing.assert_allclose(
        mdp.track_linear_velocity(env, std=0.5, command_name="twist"),
        np.exp(-linear_error / 0.5**2),
    )
    np.testing.assert_allclose(
        mdp.track_angular_velocity(env, std=0.4, command_name="twist"),
        np.exp(-angular_error / 0.4**2),
    )


def test_body_angular_velocity_requires_one_selected_body() -> None:
    env = _env()
    np.testing.assert_allclose(
        mdp.body_angular_velocity_penalty(env, SceneEntityCfg("robot", body_ids=[1])),
        [5.0, 1.25, 0.625],
    )
    with pytest.raises(ValueError, match="requires exactly one body"):
        mdp.body_angular_velocity_penalty(env)


def test_joint_pos_limits_matches_mjlab_soft_limit_equation() -> None:
    env = _env()
    entity = cast(Any, env.scene["robot"])
    joint_pos = entity.data.joint_pos
    limits = entity.data.soft_joint_pos_limits
    expected = -np.clip(joint_pos - limits[:, 0], min=None, max=0.0)
    expected += np.clip(joint_pos - limits[:, 1], min=0.0, max=None)
    np.testing.assert_allclose(mdp.joint_pos_limits(env), np.sum(expected, axis=1))
    selector = SceneEntityCfg("robot", joint_ids=[1])
    expected_knee = -np.clip(joint_pos[:, [1]] - limits[[1], 0], min=None, max=0.0)
    expected_knee += np.clip(joint_pos[:, [1]] - limits[[1], 1], min=0.0, max=None)
    np.testing.assert_allclose(mdp.joint_pos_limits(env, selector), np.sum(expected_knee, axis=1))


def _posture_expected(entity: Any, std: np.ndarray, joint_ids: list[int]) -> np.ndarray:
    error_squared = np.square(
        entity.data.joint_pos[:, joint_ids] - entity.data.default_joint_pos[:, joint_ids]
    )
    return np.exp(-np.mean(error_squared / np.square(std), axis=1))


def test_posture_matches_mjlab_per_joint_std_kernel() -> None:
    env = _env()
    entity = cast(Any, env.scene["robot"])
    std = {".*hip.*": 0.5, ".*knee.*": 0.35, ".*ankle.*": 0.25}
    manager = RewardManager(
        {"posture": RewardTermCfg(func=mdp.posture, weight=1.0, params={"std": std})},
        env,
        scale_by_dt=False,
    )
    expected = _posture_expected(entity, np.asarray([0.5, 0.35, 0.25]), [0, 1, 2])
    np.testing.assert_allclose(manager.compute(dt=0.02), expected)


def test_variable_posture_selects_std_by_command_speed_regime() -> None:
    env = _env()
    entity = cast(Any, env.scene["robot"])
    # Commands (1.0,0,0.5) / (0.25,0,-0.5) / (0,0,0) give total speeds 1.5 /
    # 0.75 / 0.0: running, walking, standing for thresholds 0.5 and 1.5.
    params = {
        "std_standing": {".*": 0.1},
        "std_walking": {".*": 0.3},
        "std_running": {".*": 0.6},
        "command_name": "twist",
        "walking_threshold": 0.5,
        "running_threshold": 1.5,
    }
    manager = RewardManager(
        {"pose": RewardTermCfg(func=mdp.variable_posture, weight=1.0, params=params)},
        env,
        scale_by_dt=False,
    )
    expected = np.stack(
        [
            _posture_expected(entity, np.full(3, 0.6), [0, 1, 2])[0],
            _posture_expected(entity, np.full(3, 0.3), [0, 1, 2])[1],
            _posture_expected(entity, np.full(3, 0.1), [0, 1, 2])[2],
        ]
    )
    np.testing.assert_allclose(manager.compute(dt=0.02), expected)


def test_posture_std_mapping_fail_closed() -> None:
    env = _env()
    with pytest.raises(ValueError, match="must match joint 'hip' exactly once"):
        RewardManager(
            {
                "posture": RewardTermCfg(
                    func=mdp.posture, weight=1.0, params={"std": {".*knee.*": 0.3}}
                )
            },
            env,
        )
    with pytest.raises(ValueError, match="std must be finite and positive"):
        RewardManager(
            {"posture": RewardTermCfg(func=mdp.posture, weight=1.0, params={"std": {".*": 0.0}})},
            env,
        )
    with pytest.raises(ValueError, match="walking_threshold must be below running_threshold"):
        RewardManager(
            {
                "pose": RewardTermCfg(
                    func=mdp.variable_posture,
                    weight=1.0,
                    params={
                        "std_standing": {".*": 0.1},
                        "std_walking": {".*": 0.3},
                        "std_running": {".*": 0.6},
                        "command_name": "twist",
                        "walking_threshold": 2.0,
                        "running_threshold": 1.5,
                    },
                )
            },
            env,
        )


def test_terms_integrate_with_reward_manager_and_cold_selector_resolution() -> None:
    env = _env()
    selector = SceneEntityCfg("robot", joint_names=("ankle", "hip"), preserve_order=True)
    manager = RewardManager(
        {
            "joint_velocity": RewardTermCfg(
                func=mdp.joint_vel_l2,
                weight=-0.5,
                params={"asset_cfg": selector},
            ),
            "track_linear": RewardTermCfg(
                func=mdp.track_linear_velocity,
                weight=2.0,
                params={"std": 0.5, "command_name": "twist"},
            ),
        },
        env,
        scale_by_dt=False,
    )

    result = manager.compute(dt=0.02)
    entity = cast(Any, env.scene["robot"])
    expected = -0.5 * np.sum(np.square(entity.data.joint_vel[:, [2, 0]]), axis=1)
    expected += 2.0 * mdp.track_linear_velocity(env, std=0.5, command_name="twist")
    np.testing.assert_allclose(result, expected)
    resolved = manager.get_term_cfg("joint_velocity").params["asset_cfg"]
    assert resolved.joint_ids == [2, 0]


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda env: mdp.track_linear_velocity(env, std=0.0, command_name="twist"),
            "std must be finite and positive",
        ),
        (
            lambda env: mdp.track_angular_velocity(env, std=np.nan, command_name="twist"),
            "std must be finite and positive",
        ),
        (
            lambda env: mdp.track_linear_velocity(env, std=0.5, command_name="missing"),
            "Command term 'missing' not found",
        ),
    ],
)
def test_invalid_reward_requests_fail_explicitly(call, message: str) -> None:
    with pytest.raises((KeyError, ValueError), match=message):
        call(_env())


def test_reward_manager_reports_nonfinite_term_and_name() -> None:
    env = _env()
    cast(Any, env.scene["robot"]).data.joint_vel[1, 0] = np.nan
    manager = RewardManager(
        {"joint_velocity": RewardTermCfg(func=mdp.joint_vel_l2, weight=-1.0)},
        env,
    )
    with pytest.raises(ValueError, match="RewardManager term 'joint_velocity'.*NaN"):
        manager.compute(dt=0.02)
