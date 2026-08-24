"""Training-contract tests for the Manager-Based A2Arm position-force task."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest


def _robot_dir():
    from unilab.assets import ASSETS_ROOT_PATH

    return ASSETS_ROOT_PATH / "robots" / "a2arm"


@pytest.fixture(scope="module", autouse=True)
def _resolve_a2arm_meshes() -> None:
    from unilab.assets.hub import resolve_robot_asset_dir

    resolve_robot_asset_dir("robots/a2arm/meshes", marker="adapter_plate.STL")


def test_a2arm_mjcf_preserves_joint_and_actuator_contract() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(_robot_dir() / "scene_pos_force.xml"))
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(model.njnt)
    ]

    assert [name for name in joint_names if name and name.startswith("joint")] == [
        "joint1",
        "joint2",
        "joint4",
        "joint6",
        "joint7",
    ]
    assert model.nu == 17
    np.testing.assert_allclose(
        model.actuator_forcerange[12:],
        [[-30.0, 30.0], [-30.0, 30.0], [-30.0, 30.0], [-10.0, 10.0], [-10.0, 10.0]],
    )


def test_a2arm_keyframe_is_owned_by_task_scene() -> None:
    robot_root = ET.parse(_robot_dir() / "a2arm.xml").getroot()
    scene_root = ET.parse(_robot_dir() / "scene_pos_force.xml").getroot()

    assert robot_root.find("keyframe") is None
    assert scene_root.find("keyframe") is not None


def test_a2arm_external_mesh_marker_is_ignored_but_directory_is_tracked() -> None:
    marker = _robot_dir() / "meshes" / ".gitkeep"
    assert marker.is_file()


def test_a2arm_manager_reset_runs_state_mutation_inside_reset_transaction() -> None:
    """The task command term must be constructible before reset lifecycle starts."""
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from unilab.base.backend.mujoco.xml import materialize_scene_visual_override
    from unilab.base.config_adapter import BackendAdapter, create_env
    from unilab.training import ensure_registries

    root = Path(__file__).resolve().parents[4]
    GlobalHydra.instance().clear()
    ensure_registries()
    with initialize_config_dir(config_dir=str(root / "conf" / "ppo_cse"), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=[
                "task=a2arm_pos_force/mujoco",
                "algo.num_envs=1",
                "training.no_play=true",
            ],
        )
    adapter = BackendAdapter(
        cfg,
        root_dir=root,
        algo_name="ppo_cse",
        scene_materializer=materialize_scene_visual_override,
    )
    env = create_env(
        cfg,
        num_envs=1,
        env_cfg_override=adapter.build_task_env_cfg_override(),
    )
    obs, info = env.reset()
    assert isinstance(info["log"], dict)
    assert obs["obs"].shape == (1, 2336)
    assert obs["critic"].shape == (1, 402)


def test_a2arm_typed_teleop_override_owns_command_and_external_force() -> None:
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from unilab.base.backend.mujoco.xml import materialize_scene_visual_override
    from unilab.base.config_adapter import BackendAdapter, create_env
    from unilab.tasks.locomotion.a2arm.state import A2ArmTeleopCommand
    from unilab.training import ensure_registries

    root = Path(__file__).resolve().parents[4]
    GlobalHydra.instance().clear()
    ensure_registries()
    with initialize_config_dir(config_dir=str(root / "conf" / "ppo_cse"), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=["task=a2arm_pos_force/mujoco", "algo.num_envs=1", "training.no_play=true"],
        )
    adapter = BackendAdapter(
        cfg,
        root_dir=root,
        algo_name="ppo_cse",
        scene_materializer=materialize_scene_visual_override,
    )
    env = create_env(cfg, num_envs=1, env_cfg_override=adapter.build_task_env_cfg_override())
    try:
        env.reset()
        state = env.command_manager.get_term("task_state")
        state.set_teleop_override(
            A2ArmTeleopCommand(
                velocity=np.asarray([[0.2, -0.1, 0.05]], dtype=np.float32),
                ee_sphere=np.asarray([[0.4, 0.3, -0.1]], dtype=np.float32),
                ee_force=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
                base_force=np.asarray([[-2.0, 1.0, 0.5]], dtype=np.float32),
            )
        )
        env.step(np.zeros((1, 17), dtype=np.float32))
        np.testing.assert_allclose(state.command[0, 0:3], [0.2, -0.1, 0.05])
        np.testing.assert_allclose(state.command[0, 3:6], [0.4, 0.3, -0.1])
        np.testing.assert_allclose(state.force_ee_world[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(state.force_base_world[0], [-2.0, 1.0, 0.5])
        state.clear_teleop_override()
        assert np.all(state.force_ee_world == 0.0)
        assert np.all(state.force_base_world == 0.0)
    finally:
        env.close()
