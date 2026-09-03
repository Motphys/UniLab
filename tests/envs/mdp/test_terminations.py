"""Upstream-derived NumPy tests for basic manager termination terms."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

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
    entity = SimpleNamespace(
        data=SimpleNamespace(
            projected_gravity_b=gravity,
            root_link_pos_w=root_pos,
            root_link_lin_vel_w=np.zeros((3, 3), dtype=np.float32),
            root_link_ang_vel_w=np.zeros((3, 3), dtype=np.float32),
            joint_pos=np.zeros((3, 2), dtype=np.float32),
            joint_vel=np.zeros((3, 2), dtype=np.float32),
        )
    )
    return cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=3,
            episode_length_buf=np.asarray([9, 10, 11], dtype=np.int64),
            max_episode_length=10,
            scene={"robot": entity},
        ),
    )


def test_basic_termination_terms_match_pinned_semantics() -> None:
    env = _env()

    np.testing.assert_array_equal(mdp.time_out(env), [False, True, True])
    np.testing.assert_array_equal(mdp.bad_orientation(env, limit_angle=0.6), [False, False, True])
    np.testing.assert_array_equal(
        mdp.root_height_below_minimum(env, minimum_height=0.4),
        [True, False, False],
    )


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
    setattr(entity.data, field, np.full((3, 3), np.nan, dtype=np.float32))

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
    entity.data.joint_vel[1, 0] = np.nan
    entity.data.root_link_pos_w[2, 2] = np.inf
    np.testing.assert_array_equal(mdp.nan_detection(env), [False, True, True])


def test_nan_detection_integrates_with_termination_manager() -> None:
    env = _env()
    manager = TerminationManager(
        {"nan": TerminationTermCfg(func=mdp.nan_detection)},
        env,
    )
    np.testing.assert_array_equal(manager.compute(), [False, False, False])
    entity = cast(Any, env.scene["robot"])
    entity.data.joint_pos[0, 1] = np.nan
    np.testing.assert_array_equal(manager.compute(), [True, False, False])
    np.testing.assert_array_equal(manager.terminated, [True, False, False])
