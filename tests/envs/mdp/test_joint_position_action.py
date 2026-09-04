"""Upstream-derived NumPy tests for the manager joint-position action."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from unisim.backend.base import SimBackend

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend_factory import create_backend
from unilab.base.entity import EntityCfg, EntityScene
from unilab.base.scene import SceneCfg
from unilab.envs.mdp import (
    JointEffortAction,
    JointEffortActionCfg,
    JointPositionAction,
    JointPositionActionCfg,
    JointVelocityAction,
    JointVelocityActionCfg,
    RelativeJointPositionAction,
    RelativeJointPositionActionCfg,
)
from unilab.envs.mdp.actions import (
    JointPositionAction as ExportedJointPositionAction,
)
from unilab.envs.mdp.actions import (
    JointPositionActionCfg as ExportedJointPositionActionCfg,
)
from unilab.managers._types import ManagerBasedRlEnv


class _Backend:
    backend_type = "fake"
    num_envs = 2
    num_actuators = 3

    def __init__(self) -> None:
        self.actuator_names = ("knee_motor", "hip_motor", "ankle_motor")
        self.target_joint_names = ("knee", "hip", "ankle")
        self.joint_index = {"hip": 0, "knee": 1, "ankle": 2}
        self.dof_pos = np.zeros((self.num_envs, 3), dtype=np.float32)

    def get_actuator_names(self) -> tuple[str, ...]:
        return self.actuator_names

    def get_actuator_joint_names(self) -> tuple[str, ...]:
        return self.target_joint_names

    def get_actuator_ctrl_range(self) -> np.ndarray:
        return np.tile(np.asarray([[-10.0, 10.0]], dtype=np.float32), (3, 1))

    def get_joint_dof_pos_indices(self, names) -> np.ndarray:
        return np.asarray([self.joint_index[name] for name in names], dtype=np.int32)

    def get_joint_dof_vel_indices(self, names) -> np.ndarray:
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
    **overrides,
) -> tuple[JointPositionAction, np.ndarray, EntityScene]:
    return _build_action(JointPositionActionCfg, **overrides)


def _build_action(action_cfg_type, **overrides):
    backend = _Backend()
    control = np.zeros((backend.num_envs, backend.num_actuators), dtype=np.float32)
    scene = EntityScene(
        {
            "robot": EntityCfg(
                joint_names=("hip", "knee", "ankle"),
                actuator_names=backend.actuator_names,
            )
        },
        cast(SimBackend, backend),
        control,
    )
    cfg_values = {
        "entity_name": "robot",
        "actuator_names": ("hip|knee",),
        **overrides,
    }
    cfg = action_cfg_type(**cfg_values)
    env = cast(ManagerBasedRlEnv, SimpleNamespace(num_envs=backend.num_envs, scene=scene))
    return cfg.build(env), control, scene


def test_public_exports_are_canonical_objects() -> None:
    assert JointPositionAction is ExportedJointPositionAction
    assert JointPositionActionCfg is ExportedJointPositionActionCfg


def test_default_offset_encoder_bias_and_control_order() -> None:
    action, control, scene = _action(scale=2.0)
    raw = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    scene["robot"].data.encoder_bias[:, 0] = np.asarray([0.05, 0.1])

    action.process_actions(raw)
    action.apply_actions()

    assert action.target_names == ["hip", "knee"]
    np.testing.assert_array_equal(action.target_ids, [0, 1])
    np.testing.assert_allclose(action.processed_action, raw * 2.0 + [0.1, 0.2])
    np.testing.assert_allclose(control[:, 1], action.processed_action[:, 0] - [0.05, 0.1])
    np.testing.assert_allclose(control[:, 0], action.processed_action[:, 1])
    np.testing.assert_array_equal(control[:, 2], 0.0)


def test_regex_scale_offset_clip_and_local_reset() -> None:
    action, _, _ = _action(
        scale={"hip": 2.0, "knee": 3.0},
        offset={"hip": 0.5, "knee": -0.5},
        clip={"hip": (-1.0, 1.0)},
        use_default_offset=False,
    )
    raw = np.asarray([[2.0, 2.0], [-2.0, -2.0]], dtype=np.float32)

    action.process_actions(raw)

    np.testing.assert_allclose(action.processed_action, [[1.0, 5.5], [-1.0, -6.5]])
    np.testing.assert_array_equal(action.raw_action, raw)
    action.reset(np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(action.raw_action[0], raw[0])
    np.testing.assert_array_equal(action.raw_action[1], 0.0)


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"scale": {"missing": 1.0}}, ValueError, "match no targets"),
        (
            {"scale": {"hip|knee": 1.0, ".*": 2.0}},
            ValueError,
            "both match target",
        ),
        ({"clip": {"hip": (1.0, -1.0)}}, ValueError, "exceeds upper"),
        ({"offset": float("nan")}, ValueError, "must be finite"),
        ({"use_default_offset": 1}, TypeError, "must be bool"),
    ],
)
def test_invalid_action_config_fails_at_construction(overrides, error, message) -> None:
    with pytest.raises(error, match=message):
        _action(**overrides)


def test_non_finite_and_wrong_shape_actions_fail_before_control_write() -> None:
    action, control, _ = _action()
    with pytest.raises(ValueError, match="expected action shape"):
        action.process_actions(np.zeros((2, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="NaN or Inf"):
        action.process_actions(np.full((2, 2), np.nan, dtype=np.float32))
    np.testing.assert_array_equal(control, 0.0)


@pytest.mark.parametrize(
    ("action_cfg_type", "action_type"),
    [
        (JointVelocityActionCfg, JointVelocityAction),
        (JointEffortActionCfg, JointEffortAction),
    ],
)
def test_velocity_and_effort_actions_use_the_shared_joint_control_mapping(
    action_cfg_type, action_type
) -> None:
    action, control, _ = _build_action(action_cfg_type, scale=2.0)
    assert isinstance(action, action_type)
    raw = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    action.process_actions(raw)
    action.apply_actions()

    np.testing.assert_allclose(control[:, 1], raw[:, 0] * 2.0)
    np.testing.assert_allclose(control[:, 0], raw[:, 1] * 2.0)
    np.testing.assert_array_equal(control[:, 2], 0.0)


def test_relative_joint_position_action_reads_current_position_at_apply_time() -> None:
    action, control, scene = _build_action(RelativeJointPositionActionCfg, scale=0.25)
    assert isinstance(action, RelativeJointPositionAction)
    scene["robot"]._backend.dof_pos[:] = np.asarray(
        [[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32
    )
    raw = np.asarray([[0.4, -0.8], [0.8, -0.4]], dtype=np.float32)

    action.process_actions(raw)
    action.apply_actions()

    # Entity joint order is hip, knee, ankle while backend actuator order is
    # knee, hip, ankle.
    np.testing.assert_allclose(control[:, 1], [0.4 + 0.4 * 0.25, 0.7 + 0.8 * 0.25])
    np.testing.assert_allclose(control[:, 0], [0.5 - 0.8 * 0.25, 0.8 - 0.4 * 0.25])


def test_relative_joint_position_action_rejects_nonzero_offsets() -> None:
    with pytest.raises(ValueError, match="does not support a non-zero offset"):
        _build_action(RelativeJointPositionActionCfg, offset={"hip": 0.1})


@pytest.mark.parametrize("backend_type", ["mujoco", "motrix"])
def test_go2_joint_targets_are_mapped_to_backend_control_order(backend_type: str) -> None:
    if backend_type == "motrix":
        pytest.importorskip("motrixsim")
    joint_names = (
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    )
    scene_cfg = SceneCfg(
        model_file=str(ASSETS_ROOT_PATH / "robots" / "go2" / "scene_flat.xml"),
        entities={
            "robot": EntityCfg(
                joint_names=joint_names,
                actuator_names=(
                    "FR_hip",
                    "FR_thigh",
                    "FR_calf",
                    "FL_hip",
                    "FL_thigh",
                    "FL_calf",
                    "RR_hip",
                    "RR_thigh",
                    "RR_calf",
                    "RL_hip",
                    "RL_thigh",
                    "RL_calf",
                ),
            )
        },
    )
    backend = create_backend(
        backend_type,
        scene_cfg,
        2,
        0.01,
        base_name="base",
    )
    control = np.zeros((2, backend.num_actuators), dtype=np.float32)
    scene = EntityScene.from_scene_cfg(scene_cfg, backend, control)
    env = cast(ManagerBasedRlEnv, SimpleNamespace(num_envs=2, scene=scene))
    action = JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=0.25,
        offset={".*_hip_joint": 0.1, ".*_thigh_joint": 0.2, ".*_calf_joint": -0.3},
        use_default_offset=False,
    ).build(env)
    raw = np.arange(24, dtype=np.float32).reshape(2, 12) / 10.0

    action.process_actions(raw)
    action.apply_actions()

    target_index = {name: index for index, name in enumerate(joint_names)}
    expected = np.column_stack(
        [
            action.processed_action[:, target_index[name]]
            for name in backend.get_actuator_joint_names()
        ]
    )
    np.testing.assert_allclose(control, expected)


def test_action_module_has_no_runtime_or_backend_private_dependencies() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "unilab"
        / "envs"
        / "mdp"
        / "actions"
        / "actions.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = ("torch", "uni_rl", "unilab.training", "unilab.base.backend")
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)] + [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not [name for name in imports if name.startswith(forbidden)]
