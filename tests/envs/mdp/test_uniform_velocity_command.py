"""Upstream-derived NumPy tests for uniform velocity commands."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from unisim.backend.base import SimBackend

from unilab.base.entity import EntityCfg, EntityScene
from unilab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from unilab.envs.mdp.commands import (
    UniformVelocityCommand as ExportedUniformVelocityCommand,
)
from unilab.envs.mdp.commands import (
    UniformVelocityCommandCfg as ExportedUniformVelocityCommandCfg,
)
from unilab.managers import CommandManager
from unilab.managers._types import ManagerBasedRlEnv
from unilab.utils.rotation import np_yaw_to_quat


class _Backend:
    backend_type = "fake"
    num_actuators = 0

    def __init__(self, num_envs: int = 4, *, unsupported: str | None = None) -> None:
        self.num_envs = num_envs
        self.unsupported = unsupported
        self.body_pos = np.zeros((num_envs, 1, 3), dtype=np.float32)
        self.body_quat = np.zeros((num_envs, 1, 4), dtype=np.float32)
        self.body_quat[..., 0] = 1.0
        self.body_lin_vel_w = np.zeros((num_envs, 1, 3), dtype=np.float32)
        self.body_ang_vel_w = np.zeros((num_envs, 1, 3), dtype=np.float32)
        self.body_lin_vel_b = np.zeros((num_envs, 1, 3), dtype=np.float32)
        self.body_ang_vel_b = np.zeros((num_envs, 1, 3), dtype=np.float32)

    def get_body_ids(self, names) -> np.ndarray:
        if tuple(names) != ("base",):
            raise KeyError(names)
        return np.asarray([0], dtype=np.int32)

    def get_dof_pos(self) -> np.ndarray:
        return np.empty((self.num_envs, 0), dtype=np.float32)

    def get_dof_vel(self) -> np.ndarray:
        return np.empty((self.num_envs, 0), dtype=np.float32)

    def get_body_pos_w(self, ids: np.ndarray) -> np.ndarray:
        return self.body_pos[:, ids]

    def get_body_quat_w(self, ids: np.ndarray) -> np.ndarray:
        return self.body_quat[:, ids]

    def get_body_lin_vel_w(self, ids: np.ndarray) -> np.ndarray:
        return self.body_lin_vel_w[:, ids]

    def get_body_ang_vel_w(self, ids: np.ndarray) -> np.ndarray:
        return self.body_ang_vel_w[:, ids]

    def get_body_lin_vel_b(self, ids: np.ndarray) -> np.ndarray:
        if self.unsupported == "body-frame linear velocity":
            raise NotImplementedError("fake lacks body-frame linear velocity")
        return self.body_lin_vel_b[:, ids]

    def get_body_ang_vel_b(self, ids: np.ndarray) -> np.ndarray:
        return self.body_ang_vel_b[:, ids]


def _env(seed: int = 7) -> tuple[ManagerBasedRlEnv, _Backend]:
    backend = _Backend()
    scene = EntityScene(
        {"robot": EntityCfg(root_body_name="base")},
        cast(SimBackend, backend),
    )
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=backend.num_envs,
            rng=np.random.default_rng(seed),
            scene=scene,
            step_dt=0.02,
        ),
    )
    return env, backend


def _cfg(**overrides: Any) -> UniformVelocityCommandCfg:
    values: dict[str, Any] = {
        "entity_name": "robot",
        "resampling_time_range": (1.0, 1.0),
        "ranges": UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.8, 0.8),
        ),
        **overrides,
    }
    return UniformVelocityCommandCfg(**values)


def _manager(env: ManagerBasedRlEnv, **overrides: Any) -> CommandManager:
    return CommandManager({"twist": _cfg(**overrides)}, env)


def test_public_exports_and_cfg_build_are_canonical() -> None:
    env, _ = _env()
    term = _cfg().build(env)

    assert UniformVelocityCommand is ExportedUniformVelocityCommand
    assert UniformVelocityCommandCfg is ExportedUniformVelocityCommandCfg
    assert isinstance(term, UniformVelocityCommand)
    assert term.command.shape == (env.num_envs, 3)
    assert term.command.dtype == np.float32


def test_resampling_is_seeded_and_partial_reset_is_local() -> None:
    left_env, _ = _env(seed=19)
    right_env, _ = _env(seed=19)
    left = _manager(left_env, resampling_time_range=(0.5, 1.5))
    right = _manager(right_env, resampling_time_range=(0.5, 1.5))
    all_ids = np.arange(left_env.num_envs, dtype=np.int32)

    left.reset(all_ids)
    right.reset(all_ids)
    np.testing.assert_array_equal(left.get_command("twist"), right.get_command("twist"))
    np.testing.assert_array_equal(
        left.get_term("twist").time_left,
        right.get_term("twist").time_left,
    )

    before_command = left.get_command("twist").copy()
    before_counter = left.get_term("twist").command_counter.copy()
    left.reset(np.asarray([1, 3], dtype=np.int32))
    np.testing.assert_array_equal(left.get_command("twist")[[0, 2]], before_command[[0, 2]])
    np.testing.assert_array_equal(
        left.get_term("twist").command_counter[[0, 2]], before_counter[[0, 2]]
    )
    np.testing.assert_array_equal(left.get_term("twist").command_counter[[1, 3]], 1)


def test_metrics_and_fixed_interval_resampling_follow_manager_schedule() -> None:
    env, backend = _env()
    backend.body_lin_vel_b[:, 0, :2] = [0.5, -0.25]
    backend.body_ang_vel_b[:, 0, 2] = 0.1
    manager = _manager(
        env,
        resampling_time_range=(0.2, 0.2),
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(1.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.2, 0.2),
        ),
    )
    all_ids = np.arange(env.num_envs, dtype=np.int32)
    manager.reset(all_ids)
    term = manager.get_term("twist")

    manager.compute(0.1)
    np.testing.assert_allclose(term.metrics["error_vel_xy"], np.hypot(0.5, 0.25) / 10.0)
    np.testing.assert_allclose(term.metrics["error_vel_yaw"], 0.1 / 10.0)
    np.testing.assert_array_equal(term.command_counter, 1)
    manager.compute(0.1)
    np.testing.assert_array_equal(term.command_counter, 2)
    np.testing.assert_allclose(term.time_left, 0.2)


def test_heading_world_forward_and_standing_modes() -> None:
    all_ids = np.arange(4, dtype=np.int32)

    heading_env, _ = _env()
    heading = _manager(
        heading_env,
        heading_command=True,
        rel_heading_envs=1.0,
        heading_control_stiffness=1.0,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.5, 0.5),
            heading=(np.pi / 2, np.pi / 2),
        ),
    )
    heading.reset(all_ids)
    heading.compute(0.0, env_ids=all_ids)
    np.testing.assert_allclose(heading.get_command("twist")[:, 2], 0.5)

    world_env, world_backend = _env()
    world_backend.body_quat[:, 0] = np_yaw_to_quat(np.full(4, np.pi / 2))
    world = _manager(
        world_env,
        rel_world_envs=1.0,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(1.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )
    world.reset(all_ids)
    world.compute(0.0, env_ids=all_ids)
    np.testing.assert_allclose(world.get_command("twist")[:, :2], [[0.0, -1.0]] * 4, atol=1e-6)

    forward_env, _ = _env()
    forward = _manager(
        forward_env,
        rel_forward_envs=1.0,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, -0.1),
            lin_vel_y=(0.4, 0.4),
            ang_vel_z=(0.2, 0.2),
        ),
    )
    forward.reset(all_ids)
    np.testing.assert_allclose(forward.get_command("twist"), [[0.3, 0.0, 0.0]] * 4)

    standing_env, _ = _env()
    standing = _manager(standing_env, rel_standing_envs=1.0)
    standing.reset(all_ids)
    standing.compute(0.0, env_ids=all_ids)
    np.testing.assert_array_equal(standing.get_command("twist"), 0.0)


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"rel_standing_envs": 1.1}, ValueError, "within.*0, 1"),
        ({"heading_command": True}, ValueError, "heading_command=True"),
        (
            {
                "ranges": UniformVelocityCommandCfg.Ranges(
                    lin_vel_x=(1.0, -1.0),
                    lin_vel_y=(0.0, 0.0),
                    ang_vel_z=(0.0, 0.0),
                )
            },
            ValueError,
            "exceeds upper",
        ),
        ({"heading_control_stiffness": float("nan")}, ValueError, "must be finite"),
        ({"resampling_time_range": (0.0, 0.0)}, ValueError, "must be positive"),
        ({"init_velocity_prob": 0.1}, NotImplementedError, "root velocity write"),
    ],
)
def test_invalid_or_unsupported_config_fails_at_construction(
    overrides: dict[str, Any], error: type[Exception], message: str
) -> None:
    env, _ = _env()
    with pytest.raises(error, match=message):
        _cfg(**overrides).build(env)


def test_root_body_frame_capability_and_finite_values_fail_closed() -> None:
    unsupported = _Backend(unsupported="body-frame linear velocity")
    with pytest.raises(
        NotImplementedError,
        match="body-frame linear velocity state.*backend 'fake'",
    ):
        EntityScene(
            {"robot": EntityCfg(root_body_name="base")},
            cast(SimBackend, unsupported),
        )

    non_finite = _Backend()
    non_finite.body_ang_vel_b[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="body-frame angular velocity state.*NaN or Inf"):
        EntityScene(
            {"robot": EntityCfg(root_body_name="base")},
            cast(SimBackend, non_finite),
        )
