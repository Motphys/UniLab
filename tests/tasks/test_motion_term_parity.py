"""Issue #1296: bit-parity tests for the temp-array-eliminated motion tracking
terms. Each optimized term is compared against the naive NumPy expression it
replaced; results must be exactly equal (same op order, only fewer temps).

``_command`` is monkeypatched to return a stub because the terms type-check
against the real MotionCommand; the stub provides the same buffer attributes.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from unilab.managers import RewardTermCfg
from unilab.tasks.motion_tracking.common import manager_terms as mt


def _make_env(command: Any) -> SimpleNamespace:
    return SimpleNamespace(num_envs=command.num_envs)


@pytest.fixture
def body_setup(monkeypatch: pytest.MonkeyPatch):
    rng = np.random.default_rng(42)
    num_envs, num_bodies = 8, 12
    command = SimpleNamespace(
        num_envs=num_envs,
        cfg=SimpleNamespace(body_names=tuple(f"b{i}" for i in range(num_bodies))),
        anchor_pos_w=rng.standard_normal((num_envs, 3), dtype=np.float32),
        robot_anchor_pos_w=rng.standard_normal((num_envs, 3), dtype=np.float32),
        joint_pos=rng.standard_normal((num_envs, 29), dtype=np.float32),
        robot_joint_pos=rng.standard_normal((num_envs, 29), dtype=np.float32),
        body_pos_relative_w=rng.standard_normal((num_envs, num_bodies, 3), dtype=np.float32),
        robot_body_pos_w=rng.standard_normal((num_envs, num_bodies, 3), dtype=np.float32),
        body_lin_vel_w=rng.standard_normal((num_envs, num_bodies, 3), dtype=np.float32),
        robot_body_lin_vel_w=rng.standard_normal((num_envs, num_bodies, 3), dtype=np.float32),
    )
    # Snapshot inputs so tests can assert the terms never mutate them.
    snapshots = {
        key: value.copy() for key, value in vars(command).items() if isinstance(value, np.ndarray)
    }
    monkeypatch.setattr(mt, "_command", lambda env, name: command)
    return command, _make_env(command), snapshots


def test_anchor_position_error_exp_bit_parity(body_setup) -> None:
    command, env, snapshots = body_setup
    out = mt.motion_global_anchor_position_error_exp(env, "motion", std=0.3)
    expected = np.exp(
        -np.sum(np.square(snapshots["anchor_pos_w"] - snapshots["robot_anchor_pos_w"]), axis=-1)
        / 0.3**2
    )
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(command.anchor_pos_w, snapshots["anchor_pos_w"])


def test_joint_position_error_exp_bit_parity(body_setup) -> None:
    command, env, snapshots = body_setup
    out = mt.motion_joint_position_error_exp(env, "motion", std=0.2)
    expected = np.exp(
        -np.square(snapshots["joint_pos"] - snapshots["robot_joint_pos"]).mean(axis=-1) / 0.2**2
    )
    np.testing.assert_array_equal(out, expected)


def test_body_term_error_exp_bit_parity_and_scratch_reuse(body_setup) -> None:
    command, env, snapshots = body_setup
    cfg = RewardTermCfg(func=None, weight=1.0, params={"command_name": "motion"})
    term = mt.motion_relative_body_position_error_exp(cfg, env)

    out = term(env, "motion", std=0.3)
    expected = np.exp(
        -np.square(snapshots["body_pos_relative_w"] - snapshots["robot_body_pos_w"])
        .sum(axis=-1)
        .mean(axis=-1)
        / 0.3**2
    )
    np.testing.assert_array_equal(out, expected)

    # Repeat calls reuse scratch and stay correct after the buffers change.
    rng = np.random.default_rng(7)
    command.body_pos_relative_w[:] = rng.standard_normal(
        command.body_pos_relative_w.shape, dtype=np.float32
    )
    out2 = term(env, "motion", std=0.3)
    expected2 = np.exp(
        -np.square(command.body_pos_relative_w - snapshots["robot_body_pos_w"])
        .sum(axis=-1)
        .mean(axis=-1)
        / 0.3**2
    )
    np.testing.assert_array_equal(out2, expected2)

    # Body-subset selection changes the scratch shape but stays bit-identical.
    cfg_sub = RewardTermCfg(
        func=None,
        weight=1.0,
        params={"command_name": "motion", "body_names": ("b0", "b3", "b11")},
    )
    term_sub = mt.motion_global_body_linear_velocity_error_exp(cfg_sub, env)
    out_sub = term_sub(env, "motion", std=1.0, body_names=("b0", "b3", "b11"))
    ids = [0, 3, 11]
    expected_sub = np.exp(
        -np.square(snapshots["body_lin_vel_w"][:, ids] - snapshots["robot_body_lin_vel_w"][:, ids])
        .sum(axis=-1)
        .mean(axis=-1)
        / 1.0**2
    )
    np.testing.assert_array_equal(out_sub, expected_sub)


def test_joint_pos_limits_bit_parity() -> None:
    rng = np.random.default_rng(123)
    joint_pos = rng.standard_normal((8, 5), dtype=np.float32)
    limits = np.asarray([[-1.0, 1.0]] * 5, dtype=np.float32)
    asset = SimpleNamespace(data=SimpleNamespace(joint_pos=joint_pos, soft_joint_pos_limits=limits))
    env = SimpleNamespace(scene={"robot": asset})
    asset_cfg = SimpleNamespace(name="robot", joint_ids=np.array([4, 2, 0], dtype=np.intp))

    out = mt.joint_pos_limits(env, asset_cfg)
    selected = joint_pos[:, [4, 2, 0]]
    selected_limits = limits[[4, 2, 0]]
    expected = np.sum(
        np.square(
            np.maximum(selected_limits[:, 0] - selected, 0.0)
            + np.maximum(selected - selected_limits[:, 1], 0.0)
        ),
        axis=-1,
    )
    np.testing.assert_array_equal(out, expected)
