"""Pinned mjlab-semantics tests for the shared locomotion gait terms."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from unilab.base.backend.base import BackendSensorView
from unilab.managers import RewardManager, RewardTermCfg
from unilab.managers._types import ManagerBasedRlEnv
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.tasks.locomotion.common import gait_terms

LEFT = ("left_contact_0", "left_contact_1")
RIGHT = ("right_contact_0", "right_contact_1")
GROUPS = (LEFT, RIGHT)


class _Scene:
    """Two-foot contact sensors plus an entity with two foot bodies."""

    def __init__(self, num_envs: int = 2) -> None:
        self.calls: Counter[str] = Counter()
        self.contacts = {
            "left_contact_0": np.zeros((num_envs, 1), dtype=np.float32),
            "left_contact_1": np.zeros((num_envs, 1), dtype=np.float32),
            "right_contact_0": np.zeros((num_envs, 1), dtype=np.float32),
            "right_contact_1": np.zeros((num_envs, 1), dtype=np.float32),
        }
        self.robot = SimpleNamespace(
            joint_names=("left_knee", "right_knee"),
            body_names=("left_foot", "right_foot"),
            num_joints=2,
            num_bodies=2,
            data=SimpleNamespace(
                body_link_pos_w=np.zeros((num_envs, 2, 3), dtype=np.float32),
                body_link_lin_vel_w=np.zeros((num_envs, 2, 3), dtype=np.float32),
            ),
        )

    def __getitem__(self, name: str):
        if name != "robot":
            raise KeyError(name)
        return self.robot

    def bind_sensor_data(self, names) -> BackendSensorView:
        names = tuple(names)
        self.calls["bind"] += 1
        dimensions = tuple(self.contacts[name].shape[1] for name in names)

        def read() -> np.ndarray:
            self.calls["read"] += 1
            return np.concatenate([self.contacts[name] for name in names], axis=1)

        view = BackendSensorView("fake", names, dimensions, 2, read)
        view.read()  # Mirror SimBackend.bind_sensor_data materialization validation.
        return view

    def set_foot_contact(self, foot: int, in_contact: bool, env_ids=(0, 1)) -> None:
        names = LEFT if foot == 0 else RIGHT
        for name in names:
            self.contacts[name][list(env_ids), 0] = 1.0 if in_contact else 0.0


class _Commands:
    def __init__(self, command: np.ndarray | None = None) -> None:
        self.command = (
            np.array([[0.8, 0.0, 0.0], [0.8, 0.0, 0.0]], dtype=np.float32)
            if command is None
            else command
        )

    def get_command(self, name: str) -> np.ndarray:
        if name != "twist":
            raise KeyError(name)
        return self.command


def _env(scene: _Scene | None = None, command: np.ndarray | None = None) -> ManagerBasedRlEnv:
    return cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=2,
            step_dt=0.02,
            scene=scene or _Scene(),
            command_manager=_Commands(command),
        ),
    )


def _term(func: Any, env: ManagerBasedRlEnv, **params: Any) -> Any:
    cfg = RewardTermCfg(func=func, weight=1.0, params=params)
    return func(cfg=cfg, env=env)


# ---------------------------------------------------------------------------
# feet_air_time: mjlab window-form counting semantics
# ---------------------------------------------------------------------------


def _mjlab_air_time_sequence(
    contacts: list[np.ndarray], step_dt: float, threshold_min: float, threshold_max: float
) -> list[np.ndarray]:
    """Replicate mjlab ContactSensor air-time updates plus the window count."""
    air_time = np.zeros_like(contacts[0], dtype=np.float64)
    rewards = []
    for contact in contacts:
        air_time = np.where(contact, 0.0, air_time + step_dt)
        in_range = (air_time > threshold_min) & (air_time < threshold_max)
        rewards.append(np.sum(in_range.astype(np.float64), axis=1))
    return rewards


def test_feet_air_time_matches_mjlab_window_counting() -> None:
    scene = _Scene()
    env = _env(scene)
    term = _term(gait_terms.feet_air_time, env, sensor_groups=GROUPS)

    # Foot 0 stays in contact; foot 1 lifts for four steps then lands.
    sequence = [(True, False)] * 4 + [(True, True)]
    contacts = []
    for left, right in sequence:
        scene.set_foot_contact(0, left)
        scene.set_foot_contact(1, right)
        contacts.append(np.array([[left, right], [left, right]]))
        expected = _mjlab_air_time_sequence(contacts, 0.02, 0.05, 0.5)[-1]
        np.testing.assert_allclose(term(env), expected)
    # Air time after the fourth air step is 0.08: inside (0.05, 0.5).
    assert _mjlab_air_time_sequence(contacts, 0.02, 0.05, 0.5)[3][0] == 1.0
    # Landing resets the air time to zero, back out of the window.
    assert _mjlab_air_time_sequence(contacts, 0.02, 0.05, 0.5)[4][0] == 0.0


def test_feet_air_time_window_excludes_long_flight() -> None:
    scene = _Scene()
    env = _env(scene)
    term = _term(
        gait_terms.feet_air_time,
        env,
        sensor_groups=GROUPS,
        threshold_min=0.01,
        threshold_max=0.05,
    )
    for step in range(5):  # air times 0.02, 0.04, 0.06, 0.08, 0.10
        reward = term(env)
        expected = 1.0 if step in (0, 1) else 0.0
        np.testing.assert_allclose(reward, [expected * 2, expected * 2])


def test_feet_air_time_command_gate_and_reset() -> None:
    scene = _Scene()
    standing = np.zeros((2, 3), dtype=np.float32)
    env = _env(scene, command=standing)
    term = _term(
        gait_terms.feet_air_time,
        env,
        sensor_groups=GROUPS,
        command_name="twist",
        command_threshold=0.5,
    )
    for _ in range(4):
        np.testing.assert_allclose(term(env), [0.0, 0.0])  # Standing: gated off.

    moving_env = _env(scene)
    term.reset()  # Episode reset clears accumulated air time.
    np.testing.assert_allclose(term(moving_env), [0.0, 0.0])  # 0.02 <= threshold_min.
    for _ in range(2):
        term(moving_env)
    np.testing.assert_allclose(term(moving_env), [2.0, 2.0])  # 0.08 in window.


def test_feet_air_time_group_any_reduce_matches_mjlab_slots() -> None:
    scene = _Scene()
    env = _env(scene)
    term = _term(gait_terms.feet_air_time, env, sensor_groups=GROUPS)
    # Only one sensor of the left group reports contact: the whole foot counts
    # as in contact (mjlab ContactSensor any-reduce across slots), so only the
    # right foot accumulates air time.
    scene.contacts["left_contact_1"][:, 0] = 1.0
    np.testing.assert_allclose(term(env), [0.0, 0.0])  # Right at 0.02.
    np.testing.assert_allclose(term(env), [0.0, 0.0])  # Right at 0.04.
    np.testing.assert_allclose(term(env), [1.0, 1.0])  # Right at 0.06, in window.


# ---------------------------------------------------------------------------
# feet_clearance / feet_swing_height / feet_slip / angular_momentum
# ---------------------------------------------------------------------------


def test_feet_clearance_matches_mjlab_velocity_weighted_error() -> None:
    scene = _Scene()
    scene.robot.data.body_link_pos_w[:] = [[[0, 0, 0.12], [0, 0, 0.05]]]
    scene.robot.data.body_link_lin_vel_w[:] = [[[0.3, 0.4, 0.0], [0.0, -0.5, 0.0]]]
    env = _env(scene)
    cost = gait_terms.feet_clearance(env, target_height=0.1, command_name="twist")
    expected = np.abs(np.array([0.12, 0.05]) - 0.1) * np.array([0.5, 0.5])
    np.testing.assert_allclose(cost, [np.sum(expected)] * 2, rtol=1e-6)

    standing = _env(scene, command=np.zeros((2, 3), dtype=np.float32))
    np.testing.assert_allclose(
        gait_terms.feet_clearance(standing, target_height=0.1, command_name="twist"),
        [0.0, 0.0],
    )


def test_feet_swing_height_charges_peak_error_at_landing() -> None:
    scene = _Scene()
    env = _env(scene)
    term = _term(
        gait_terms.feet_swing_height,
        env,
        sensor_groups=GROUPS,
        target_height=0.1,
        command_name="twist",
        command_threshold=0.05,
    )
    # Swing: both feet in the air, left peaks at 0.12, right at 0.08.
    scene.robot.data.body_link_pos_w[:, 0, 2] = 0.12
    scene.robot.data.body_link_pos_w[:, 1, 2] = 0.08
    np.testing.assert_allclose(term(env), [0.0, 0.0])  # In air: nothing charged.
    scene.robot.data.body_link_pos_w[:, 0, 2] = 0.10  # Lower later: peak persists.
    np.testing.assert_allclose(term(env), [0.0, 0.0])
    # Landing: first contact charges (peak / target - 1)^2 per foot.
    scene.set_foot_contact(0, True)
    scene.set_foot_contact(1, True)
    scene.robot.data.body_link_pos_w[:, :, 2] = 0.0
    expected = (0.12 / 0.1 - 1.0) ** 2 + (0.08 / 0.1 - 1.0) ** 2
    np.testing.assert_allclose(term(env), [expected] * 2, rtol=1e-6)
    # After landing the peaks reset; continued contact charges nothing.
    np.testing.assert_allclose(term(env), [0.0, 0.0])

    term.reset()
    standing = _env(scene, command=np.zeros((2, 3), dtype=np.float32))
    scene.set_foot_contact(0, False)
    scene.set_foot_contact(1, False)
    scene.robot.data.body_link_pos_w[:, 0, 2] = 0.15
    scene.robot.data.body_link_pos_w[:, 1, 2] = 0.15
    term(standing)
    scene.set_foot_contact(0, True)
    scene.set_foot_contact(1, True)
    np.testing.assert_allclose(term(standing), [0.0, 0.0])  # Standing: gated off.


def test_feet_slip_matches_mjlab_contact_weighted_xy_speed() -> None:
    scene = _Scene()
    scene.set_foot_contact(0, True)
    scene.robot.data.body_link_lin_vel_w[:] = [[[0.3, 0.4, 9.0], [1.0, 0.0, 0.0]]]
    env = _env(scene)
    term = _term(
        gait_terms.feet_slip,
        env,
        sensor_groups=GROUPS,
        command_name="twist",
        command_threshold=0.01,
    )
    # Only the left foot is in contact; z velocity is ignored.
    np.testing.assert_allclose(term(env), [0.3**2 + 0.4**2] * 2, rtol=1e-6)

    standing = _env(scene, command=np.zeros((2, 3), dtype=np.float32))
    np.testing.assert_allclose(term(standing), [0.0, 0.0])


def test_angular_momentum_penalty_matches_mjlab_squared_norm() -> None:
    scene = _Scene()
    scene.contacts["angmom"] = np.array([[1.0, 2.0, 3.0], [-0.5, 0.0, 0.25]], dtype=np.float32)
    env = _env(scene)
    term = _term(gait_terms.angular_momentum_penalty, env, sensor_name="angmom")
    np.testing.assert_allclose(term(env), [14.0, 0.3125], rtol=1e-6)


# ---------------------------------------------------------------------------
# Validation and manager integration
# ---------------------------------------------------------------------------


def test_gait_terms_fail_closed_on_bad_configuration() -> None:
    env = _env()
    with pytest.raises(ValueError, match="sensor_groups must declare at least one foot"):
        _term(gait_terms.feet_air_time, env, sensor_groups=[])
    with pytest.raises(ValueError, match="command_name must be a non-empty string"):
        _term(gait_terms.feet_slip, env, sensor_groups=GROUPS)
    with pytest.raises(ValueError, match="threshold_min must be below threshold_max"):
        _term(
            gait_terms.feet_air_time,
            env,
            sensor_groups=GROUPS,
            threshold_min=0.5,
            threshold_max=0.05,
        )
    with pytest.raises(ValueError, match="selects 1 foot bodies"):
        _term(
            gait_terms.feet_slip,
            env,
            sensor_groups=GROUPS,
            command_name="twist",
            asset_cfg=SceneEntityCfg("robot", body_ids=[0]),
        )
    with pytest.raises(ValueError, match="must expose 3 values"):
        _term(gait_terms.angular_momentum_penalty, env, sensor_name="left_contact_0")


def test_gait_terms_integrate_with_reward_manager() -> None:
    scene = _Scene()
    env = _env(scene)
    manager = RewardManager(
        {
            "air_time": RewardTermCfg(
                func=gait_terms.feet_air_time,
                weight=0.25,
                params={"sensor_groups": [list(LEFT), list(RIGHT)], "command_name": "twist"},
            ),
            "slip": RewardTermCfg(
                func=gait_terms.feet_slip,
                weight=-0.1,
                params={"sensor_groups": [list(LEFT), list(RIGHT)], "command_name": "twist"},
            ),
        },
        env,
        scale_by_dt=False,
    )
    scene.set_foot_contact(0, True)
    scene.robot.data.body_link_lin_vel_w[:] = [[[0.3, 0.4, 0.0], [0.0, 0.0, 0.0]]]
    for _ in range(3):
        result = manager.compute(dt=0.02)
    expected = 0.25 * 1.0 - 0.1 * 0.25  # Right foot at 0.06 s in window; left slips.
    np.testing.assert_allclose(result, [expected, expected], rtol=1e-6)


def test_gait_term_module_has_no_forbidden_runtime_dependencies() -> None:
    import ast
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "unilab"
        / "tasks"
        / "locomotion"
        / "common"
        / "gait_terms.py"
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


# ---------------------------------------------------------------------------
# Privileged foot observations (mjlab velocity critic terms, issue #1402)
# ---------------------------------------------------------------------------


def test_foot_height_returns_per_foot_body_z() -> None:
    scene = _Scene()
    scene.robot.data.body_link_pos_w[:] = [[[0.0, 0.0, 0.031], [0.0, 0.0, 0.052]]]
    env = _env(scene)
    np.testing.assert_allclose(
        gait_terms.foot_height(env),
        [[0.031, 0.052], [0.031, 0.052]],
        rtol=1e-6,
    )


def test_foot_air_time_tracks_current_air_time() -> None:
    scene = _Scene()
    env = _env(scene)
    term = _term(gait_terms.foot_air_time, env, sensor_groups=GROUPS)
    scene.set_foot_contact(0, True)
    scene.set_foot_contact(1, False)
    np.testing.assert_allclose(term(env), [[0.0, 0.02], [0.0, 0.02]])
    np.testing.assert_allclose(term(env), [[0.0, 0.04], [0.0, 0.04]])
    scene.set_foot_contact(1, True)
    np.testing.assert_allclose(term(env), [[0.0, 0.0], [0.0, 0.0]])


def test_foot_air_time_reset_clears_only_targeted_rows() -> None:
    scene = _Scene()
    env = _env(scene)
    term = _term(gait_terms.foot_air_time, env, sensor_groups=GROUPS)
    scene.set_foot_contact(0, False)
    scene.set_foot_contact(1, False)
    term(env)  # Both feet at 0.02 s air time in both envs.
    term.reset(np.array([1]))
    np.testing.assert_allclose(term(env), [[0.04, 0.04], [0.02, 0.02]])


def test_foot_contact_reports_float_flags() -> None:
    scene = _Scene()
    env = _env(scene)
    term = _term(gait_terms.foot_contact, env, sensor_groups=GROUPS)
    scene.set_foot_contact(0, True, env_ids=(0,))
    scene.set_foot_contact(1, False)
    np.testing.assert_allclose(term(env), [[1.0, 0.0], [0.0, 0.0]])


def test_foot_contact_forces_log_compressed_in_group_order() -> None:
    scene = _Scene()
    scene.contacts = {
        "left_contact_0": np.array([[1.0, 0.0, -1.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        "left_contact_1": np.zeros((2, 3), dtype=np.float32),
        "right_contact_0": np.array([[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        "right_contact_1": np.zeros((2, 3), dtype=np.float32),
    }
    env = _env(scene)
    term = _term(gait_terms.foot_contact_forces, env, sensor_groups=GROUPS)
    values = term(env)
    assert values.shape == (2, 12)
    expected_left = np.sign([1.0, 0.0, -1.0]) * np.log1p(np.abs([1.0, 0.0, -1.0]))
    np.testing.assert_allclose(values[0, :3], expected_left, rtol=1e-6)
    np.testing.assert_allclose(values[0, 3:6], 0.0, atol=1e-8)
    np.testing.assert_allclose(values[0, 6:9], [0.0, np.log1p(2.0), 0.0], rtol=1e-6)
    np.testing.assert_allclose(values[1], 0.0, atol=1e-8)


def test_foot_contact_forces_rejects_scalar_found_sensors() -> None:
    scene = _Scene()  # width-1 "found" sensors
    env = _env(scene)
    with pytest.raises(ValueError, match="3-D force"):
        _term(gait_terms.foot_contact_forces, env, sensor_groups=GROUPS)
