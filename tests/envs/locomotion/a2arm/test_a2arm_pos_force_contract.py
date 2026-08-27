"""Training-contract tests for the Manager-Based A2Arm position-force task."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _robot_dir():
    from unilab.assets import ASSETS_ROOT_PATH

    return ASSETS_ROOT_PATH / "robots" / "a2arm"


def _create_a2arm_env(num_envs: int = 1):
    """Build a fresh Manager-Based A2Arm environment for contract tests."""
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
                f"algo.num_envs={num_envs}",
                "training.no_play=true",
            ],
        )
    adapter = BackendAdapter(
        cfg,
        root_dir=root,
        algo_name="ppo_cse",
        scene_materializer=materialize_scene_visual_override,
    )
    return create_env(
        cfg,
        num_envs=num_envs,
        env_cfg_override=adapter.build_task_env_cfg_override(),
    )


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


def test_a2arm_ee_sensor_view_contains_only_fk_inputs() -> None:
    env = _create_a2arm_env()
    try:
        env.reset()
        state = env.command_manager.get_term("task_state")
        assert state._sensor_view.names == (
            "endpoint_pos",
            "endpoint_quat",
            "armbasepoint_world_pos",
            "armbasepoint_world_quat",
        )
    finally:
        env.close()


def test_a2arm_manager_reset_runs_state_mutation_inside_reset_transaction() -> None:
    """The task command term must be constructible before reset lifecycle starts."""
    env = _create_a2arm_env()
    try:
        obs, info = env.reset()
        assert isinstance(info["log"], dict)
        assert obs["obs"].shape == (1, 2336)
        assert obs["critic"].shape == (1, 402)
    finally:
        env.close()


def test_a2arm_history_terms_own_only_their_history_role() -> None:
    """Actor and critic terms must not carry an unused second history buffer."""
    env = _create_a2arm_env()
    try:
        actor_term = env.observation_manager._group_obs_term_cfgs["policy"][0].func
        critic_term = env.observation_manager._group_obs_term_cfgs["critic"][0].func
        assert hasattr(actor_term, "_history")
        assert not hasattr(actor_term, "_actor_history")
        assert hasattr(critic_term, "_history")
        assert not hasattr(critic_term, "_critic_history")
    finally:
        env.close()


def test_a2arm_typed_teleop_override_owns_command_and_external_force() -> None:
    from unilab.tasks.locomotion.a2arm.state import A2ArmTeleopCommand

    env = _create_a2arm_env()
    try:
        env.reset()
        state = env.command_manager.get_term("task_state")
        applied: list[tuple[str, np.ndarray]] = []

        def record_ee_force(values: np.ndarray, *, term_name: str) -> None:
            del term_name
            applied.append(("ee", values.copy()))

        def record_base_force(values: np.ndarray, *, term_name: str) -> None:
            del term_name
            applied.append(("base", values.copy()))

        state._ee_entity.apply_body_force = record_ee_force  # type: ignore[method-assign]
        state._entity.apply_body_force = record_base_force  # type: ignore[method-assign]
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
        assert np.all(state._teleop_ee_sphere == 0.0)
        np.testing.assert_allclose(applied[-2][1], 0.0)
        np.testing.assert_allclose(applied[-1][1], 0.0)
    finally:
        env.close()


def test_a2arm_actor_angular_velocity_noise_is_scaled_after_injection() -> None:
    from unilab.tasks.locomotion.a2arm.observations import _add_actor_noise

    core = np.zeros((1, 73), dtype=np.float32)
    rng = type(
        "FixedRng",
        (),
        {"uniform": lambda self, low, high, size: np.full(size, 0.2, dtype=np.float32)},
    )()
    result = _add_actor_noise(core, rng, noise_level=1.0)

    np.testing.assert_allclose(result[0, 2:5], 0.05)


def test_a2arm_standing_velocity_push_preserves_legacy_absolute_target() -> None:
    """Standing push must scale the sampled target before submitting a delta."""
    from unilab.tasks.locomotion.a2arm.state import A2ArmPosForceState

    applied: list[np.ndarray] = []

    class _Rng:
        def uniform(self, low, high, size):
            del low, high
            return np.full(size, 3.0, dtype=np.float32)

    class _Entity:
        data = SimpleNamespace(root_link_lin_vel_w=np.asarray([[1.0, 2.0, 0.0]], dtype=np.float32))

        def apply_root_linear_velocity_delta_to_sim(self, values, *, env_ids, term_name):
            del env_ids, term_name
            applied.append(values.copy())

    state = object.__new__(A2ArmPosForceState)
    state.cfg = SimpleNamespace(
        velocity_push=True,
        push_interval=1,
        max_push_vel_xy=4.0,
        velocity_push_standing_scale=0.5,
        velocity_clip=(0.1, 0.1, 0.2),
    )
    state._env = SimpleNamespace(rng=_Rng())
    state._entity = _Entity()
    state._all_env_ids = np.asarray([0], dtype=np.intp)
    state._command = np.zeros((1, 15), dtype=np.float32)

    state._maybe_apply_velocity_push(1)

    assert len(applied) == 1
    # Legacy behavior scales the sampled absolute target before replacing the
    # current velocity.  Sampled target is [3, 3], standing scale is 0.5, and
    # current velocity is [1, 2], so the submitted delta is [0.5, -0.5].
    np.testing.assert_allclose(applied[0], [[0.5, -0.5, 0.0]])


def test_a2arm_partial_reset_history_keeps_untouched_rows() -> None:
    env = _create_a2arm_env(2)
    try:
        env.reset()
        env.step(np.zeros((2, 17), dtype=np.float32))
        actor_term = env.observation_manager._group_obs_term_cfgs["policy"][0].func
        critic_term = env.observation_manager._group_obs_term_cfgs["critic"][0].func
        actor_before = actor_term._history[1].copy()
        critic_before = critic_term._history[1].copy()
        env.reset(np.asarray([0], dtype=np.int32))
        np.testing.assert_array_equal(actor_term._history[1], actor_before)
        np.testing.assert_array_equal(critic_term._history[1], critic_before)
    finally:
        env.close()


def test_a2arm_partial_reset_history_accepts_nonzero_env_id() -> None:
    """Row-scoped history terms must scatter selected frames by local row."""
    env = _create_a2arm_env(3)
    try:
        env.reset()
        env.step(np.zeros((3, 17), dtype=np.float32))
        actor_term = env.observation_manager._group_obs_term_cfgs["policy"][0].func
        critic_term = env.observation_manager._group_obs_term_cfgs["critic"][0].func
        actor_before = actor_term._history[[0, 1]].copy()
        critic_before = critic_term._history[[0, 1]].copy()

        env.reset(np.asarray([2], dtype=np.int32))

        np.testing.assert_array_equal(actor_term._history[[0, 1]], actor_before)
        np.testing.assert_array_equal(critic_term._history[[0, 1]], critic_before)
    finally:
        env.close()


def test_a2arm_partial_reset_actor_noise_keeps_rng_stream() -> None:
    """Row-scoped history must consume the same actor-noise draws as a full call."""
    env = _create_a2arm_env(2)
    try:
        env.reset()
        actor_term = env.observation_manager._group_obs_term_cfgs["policy"][0].func
        rng_state = copy.deepcopy(env.rng.bit_generator.state)

        actor_term.reset()
        actor_term(env)
        full_rng_state = copy.deepcopy(env.rng.bit_generator.state)

        env.rng.bit_generator.state = rng_state
        actor_term.reset()
        actor_term(env, env_ids=np.asarray([0], dtype=np.int32))

        assert env.rng.bit_generator.state == full_rng_state
    finally:
        env.close()


def test_a2arm_partial_reset_preserves_unreset_velocity_history() -> None:
    """A partial reset must not discard transition state for other rows."""
    env = _create_a2arm_env(2)
    try:
        env.reset()
        zero_actions = np.zeros((2, 17), dtype=np.float32)
        env.step(zero_actions)
        state = env.command_manager.get_term("task_state")
        expected = state._pending_dof_vel[1].copy()

        env.reset(np.asarray([0], dtype=np.int32))
        env.step(zero_actions)

        np.testing.assert_allclose(state._last_dof_vel[1], expected)
    finally:
        env.close()


def test_a2arm_command_term_owns_transition_advance() -> None:
    env = _create_a2arm_env()
    try:
        env.reset()
        state = env.command_manager.get_term("task_state")
        state._gait_phase.fill(0.0)
        state._command[:, 0] = 0.5
        state._last_transition_token.fill(-1)
        state._update_command(None)
        np.testing.assert_allclose(state._gait_phase, env.step_dt / state.cfg.gait_cycle_time)
    finally:
        env.close()


def test_a2arm_command_reset_clears_manager_counter_per_row() -> None:
    env = _create_a2arm_env(2)
    try:
        env.reset()
        state = env.command_manager.get_term("task_state")
        state.command_counter[:] = 7
        env.reset(np.asarray([1], dtype=np.int32))
        assert state.command_counter.tolist() == [7, 0]
    finally:
        env.close()


def test_a2arm_state_reset_slice_only_resets_selected_rows() -> None:
    env = _create_a2arm_env(3)
    try:
        env.reset()
        state = env.command_manager.get_term("task_state")
        state._command[1] = 123.0
        with env._reset_state.scoped(np.asarray([0], dtype=np.int32)):
            state.reset(slice(0, 1))
        np.testing.assert_array_equal(state.command[1], np.full(15, 123.0, dtype=np.float32))
    finally:
        env.close()


def test_a2arm_state_reset_restarts_command_cadence_per_row() -> None:
    env = _create_a2arm_env(2)
    try:
        env.reset()
        state = env.command_manager.get_term("task_state")
        state._command_timer[:] = 999
        with env._reset_state.scoped(np.asarray([0], dtype=np.int32)):
            state.reset(np.asarray([0], dtype=np.int32))
        assert 0 <= state._command_timer[0] < state.cfg.command_resample_steps
        assert state._command_timer[1] == 999
    finally:
        env.close()


def test_a2arm_episode_length_setter_updates_manager_state() -> None:
    env = _create_a2arm_env(2)
    try:
        env.reset()
        values = np.asarray([3, 5], dtype=np.int64)
        env.set_episode_length_buf(values)
        np.testing.assert_array_equal(env.episode_length_buf, values)
        assert env.state is not None
        np.testing.assert_array_equal(env.state.info["steps"], values)
    finally:
        env.close()
