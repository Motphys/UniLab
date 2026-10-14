"""Upstream-derived NumPy tests for basic manager termination terms."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from unilab.base.entity import EntityTensorBodyStateView, EntityTensorStateView
from unilab.envs import mdp
from unilab.managers import TerminationManager, TerminationTermCfg
from unilab.managers._types import ManagerBasedRlEnv
from unilab.managers.scene_entity_config import SceneEntityCfg


def _env() -> ManagerBasedRlEnv:
    angles = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    gravity = np.zeros((3, 3), dtype=np.float32)
    gravity[:, 2] = -np.cos(angles)
    root_pos = np.zeros((3, 3), dtype=np.float32)
    root_pos[:, 2] = [0.2, 0.4, 0.6]
    entity = _Entity(gravity, root_pos)
    scene = _Scene(entity)
    return cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=3,
            episode_length_buf=torch.tensor([9, 10, 11], dtype=torch.int64),
            max_episode_length=10,
            device=torch.device("cpu"),
            scene=scene,
        ),
    )


class _Entity:
    name = "robot"
    _root_body_name = "base"

    def __init__(self, projected_gravity: np.ndarray, root_pos: np.ndarray):
        self.projected_gravity = projected_gravity
        self.root_pos = root_pos
        self.joint_pos = torch.zeros((3, 2))
        self.joint_vel = torch.zeros((3, 2))
        self.data = self

    def gravity_vec_w_torch(self, device):
        gravity = np.zeros((3, 3), dtype=np.float32)
        gravity[:, 2] = -1.0
        return torch.from_numpy(gravity).to(device=device)


class _ReadPlan:
    def __init__(self, entity: _Entity):
        self.entity = entity

    def body_tensor_view(self, entity, body_names):
        assert entity is self.entity and body_names == ("base",)
        quat = np.zeros((3, 4), dtype=np.float32)
        # Orient each root so inverse rotation of world -Z gives the pinned
        # projected-gravity fixture rows.
        angles = np.arccos(-self.entity.projected_gravity[:, 2])
        quat[:, 0] = np.cos(0.5 * angles)
        quat[:, 1] = np.sin(0.5 * angles)
        return EntityTensorBodyStateView(
            body_names=("base",),
            pos_w=torch.from_numpy(self.entity.root_pos[:, None, :]),
            quat_w=torch.from_numpy(quat[:, None, :]),
            lin_vel_w=torch.zeros((3, 1, 3)),
            ang_vel_w=torch.zeros((3, 1, 3)),
        )

    def joint_tensor_view(self, entity):
        assert entity is self.entity
        return EntityTensorStateView(
            joint_pos=self.entity.joint_pos, joint_vel=self.entity.joint_vel
        )


class _Scene(dict):
    def __init__(self, entity: _Entity):
        super().__init__({"robot": entity})
        self._tensor_read_plan = _ReadPlan(entity)


def test_basic_termination_terms_match_pinned_semantics() -> None:
    env = _env()

    np.testing.assert_array_equal(mdp.time_out(env), [False, True, True])
    np.testing.assert_array_equal(mdp.bad_orientation(env, limit_angle=0.6), [False, False, True])
    np.testing.assert_array_equal(
        mdp.root_height_below_minimum(env, minimum_height=0.4),
        [True, False, False],
    )


def test_time_out_reuses_environment_persistent_scratch() -> None:
    env = _env()
    scratch = torch.empty((env.num_envs,), dtype=torch.bool)
    setattr(env, "time_out_scratch", scratch)

    result = mdp.time_out(env)

    assert result is scratch
    np.testing.assert_array_equal(result, [False, True, True])


def test_terms_integrate_with_termination_manager_and_selector_resolution() -> None:
    env = _env()
    selector = SceneEntityCfg("robot")
    manager = TerminationManager(
        {
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
            "bad_orientation": TerminationTermCfg(
                func=mdp.bad_orientation,
                params={"limit_angle": 0.6, "asset_cfg": selector},
            ),
            "low_height": TerminationTermCfg(
                func=mdp.root_height_below_minimum,
                params={"minimum_height": 0.4},
            ),
        },
        env,
    )

    np.testing.assert_array_equal(manager.compute(), [True, True, True])
    np.testing.assert_array_equal(manager.time_outs, [False, True, True])
    np.testing.assert_array_equal(manager.terminated, [True, False, True])
    assert selector.name == "robot"


@pytest.mark.parametrize(
    ("term", "parameter", "message"),
    [
        (mdp.bad_orientation, np.nan, "limit_angle must be finite"),
        (mdp.bad_orientation, True, "limit_angle must be a real number"),
        (
            mdp.root_height_below_minimum,
            np.inf,
            "minimum_height must be finite",
        ),
    ],
)
def test_invalid_scalar_parameters_fail_explicitly(term, parameter, message: str) -> None:
    env = _env()
    keyword = "limit_angle" if term is mdp.bad_orientation else "minimum_height"
    with pytest.raises((TypeError, ValueError), match=message):
        term(env, **{keyword: parameter})


@pytest.mark.parametrize("field", ["projected_gravity_b", "root_link_pos_w"])
def test_invalid_entity_state_fails_instead_of_becoming_false(field: str) -> None:
    env = _env()
    entity = cast(Any, env.scene["robot"])
    if field == "projected_gravity_b":
        entity.projected_gravity[:] = np.nan
    else:
        entity.root_pos[:] = np.nan

    with pytest.raises(ValueError, match="NaN or Inf entity state"):
        if field == "projected_gravity_b":
            mdp.bad_orientation(env, limit_angle=0.5)
        else:
            mdp.root_height_below_minimum(env, minimum_height=0.5)


def test_missing_entity_is_not_silently_replaced() -> None:
    env = _env()
    with pytest.raises(KeyError, match="missing"):
        mdp.bad_orientation(env, 0.5, SceneEntityCfg("missing"))


def test_nan_detection_flags_only_nonfinite_envs() -> None:
    env = _env()
    np.testing.assert_array_equal(mdp.nan_detection(env), [False, False, False])

    entity = cast(Any, env.scene["robot"])
    entity.joint_vel[1, 0] = torch.nan
    entity.root_pos[2, 2] = np.inf
    np.testing.assert_array_equal(mdp.nan_detection(env), [False, True, True])


def test_nan_detection_integrates_with_termination_manager() -> None:
    env = _env()
    manager = TerminationManager(
        {"nan": TerminationTermCfg(func=mdp.nan_detection)},
        env,
    )
    np.testing.assert_array_equal(manager.compute(), [False, False, False])
    entity = cast(Any, env.scene["robot"])
    entity.joint_pos[0, 1] = torch.nan
    np.testing.assert_array_equal(manager.compute(), [True, False, False])
    np.testing.assert_array_equal(manager.terminated, [True, False, False])
