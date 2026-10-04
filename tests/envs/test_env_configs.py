"""Tests for env config completeness and env instantiation.

The whole module is marked slow: Hydra composition and ``registry.make()``
over every owner are CPU-bound and too slow for the single-core CI runner.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from unilab.base.registry import ensure_registries

# CPU-bound on the single-core CI runner; kept in the slow lane (make test-slow).
pytestmark = pytest.mark.slow


def _require_mujoco_runtime() -> None:
    pytest.importorskip("mujoco", reason="mujoco not installed")
    pytest.importorskip("mjbatch", reason="mjbatch not installed")
    pytest.importorskip(
        "unisim.backend.mujoco.backend",
        reason="unisim-core MuJoCo adapter (mjbatch build) not available",
    )


def _require_mjwarp_runtime() -> None:
    from unisim.backend.mjwarp.dependencies import load_mjwarp_dependencies

    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.fail("mjwarp runtime tests require an active CUDA Warp device")


def _require_genesis_runtime() -> None:
    from unisim.backend.genesis.dependencies import genesis_dependencies_available

    if not genesis_dependencies_available():
        pytest.skip("genesis requires the genesis-world extra")
    if not torch.cuda.is_available():
        pytest.skip("genesis runtime tests require a CUDA device")


def _require_newton_runtime() -> None:
    from unisim.backend.newton.dependencies import load_newton_dependencies

    dependencies = load_newton_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.skip("newton runtime tests require an active CUDA Warp device")


def _require_motrix_runtime() -> None:
    pytest.importorskip("motrixsim", reason="motrixsim not installed")


def _allegro_manager_override(
    backend: str = "mujoco",
    *,
    config_root: str = "ppo",
    task: str = "allegro_inhand",
) -> dict[str, Any]:
    from hydra import compose, initialize_config_dir

    from unilab.base.config_adapter import BackendAdapter

    repo_root = Path(__file__).parents[2]
    with initialize_config_dir(
        config_dir=str(repo_root / "src" / "unilab" / "conf" / config_root), version_base="1.3"
    ):
        cfg = compose("config", overrides=[f"task={task}/{backend}"])
    return BackendAdapter(
        cfg, root_dir=repo_root, algo_name=config_root
    ).build_task_env_cfg_override()


def _g1_manager_override(
    task: str = "g1_walk_flat", backend: str = "mujoco", config_group: str = "ppo"
) -> dict[str, Any]:
    from hydra import compose, initialize_config_dir

    from unilab.base.config_adapter import BackendAdapter

    repo_root = Path(__file__).parents[2]
    with initialize_config_dir(
        config_dir=str(repo_root / "src" / "unilab" / "conf" / config_group), version_base="1.3"
    ):
        cfg = compose("config", overrides=[f"task={task}/{backend}"])
    return BackendAdapter(
        cfg, root_dir=repo_root, algo_name=config_group
    ).build_task_env_cfg_override()


def _motion_manager_override(
    task: str,
    backend: str,
    *,
    config_root: str = "ppo",
) -> tuple[str, dict[str, Any]]:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from unilab.base.config_adapter import BackendAdapter

    repo_root = Path(__file__).parents[2]
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(repo_root / "src" / "unilab" / "conf" / config_root), version_base="1.3"
    ):
        overrides = [f"task={task}/{backend}"]
        cfg = compose("config", overrides=overrides)
    return str(cfg.training.task_name), BackendAdapter(
        cfg,
        root_dir=repo_root,
        algo_name=config_root,
    ).build_task_env_cfg_override()


# ---------------------------------------------------------------------------
# Config attribute completeness (no env.step(), no MuJoCo sim)
# ---------------------------------------------------------------------------


def test_registry_bootstrap_and_config_imports_do_not_require_mujoco():
    repo_root = Path(__file__).parents[2]
    script = textwrap.dedent(
        """
        import builtins

        real_import = builtins.__import__

        def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "mujoco" or name.startswith("mujoco."):
                raise ImportError("mujoco blocked by test")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = blocked_import

        from unilab.base import registry
        from unilab.base.backend_factory import create_backend
        from unilab.base.registry import ensure_registries

        ensure_registries()
        assert callable(create_backend)
        assert registry.contains("G1MotionTracking")
        assert registry.contains("X2WallFlipTracking")
        assert registry.contains("AllegroInhandRotation")
        metadata = registry.list_registered_envs()
        assert metadata["G1MotionTracking"]["config_factory"] == "ManagerBasedRlEnvCfg"
        assert metadata["X2WallFlipTracking"]["config_factory"] == "ManagerBasedRlEnvCfg"
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_g1_walk_flat_assets_define_contact_sensors_for_gait_rewards():
    repo_root = Path(__file__).parents[2]
    scene_text = (
        repo_root / "src" / "unilab" / "assets" / "robots" / "g1" / "scene_flat.xml"
    ).read_text()
    model_text = (repo_root / "src" / "unilab" / "assets" / "robots" / "g1" / "g1.xml").read_text()

    for name in (
        "left_foot_contact_0",
        "left_foot_contact_1",
        "left_foot_contact_2",
        "left_foot_contact_3",
        "right_foot_contact_0",
        "right_foot_contact_1",
        "right_foot_contact_2",
        "right_foot_contact_3",
    ):
        assert name in scene_text

    for name in (
        "pelvis_local_linvel",
        "pelvis_gyro",
        "pelvis_acceleration",
        "pelvis_upvector",
        "torso_gyro",
        "torso_acceleration",
        "torso_upvector",
    ):
        assert name in model_text

    for name in (
        "left_foot_contact_0_geom",
        "left_foot_contact_1_geom",
        "left_foot_contact_2_geom",
        "left_foot_contact_3_geom",
        "right_foot_contact_0_geom",
        "right_foot_contact_1_geom",
        "right_foot_contact_2_geom",
        "right_foot_contact_3_geom",
    ):
        assert name in model_text


def test_g1_sphere_hand_assets_align_with_current_g1_sensor_names():
    repo_root = Path(__file__).parents[2]
    model_text = (
        repo_root / "src" / "unilab" / "assets" / "robots" / "g1" / "g1_sphere_hand.xml"
    ).read_text()

    for name in (
        "pelvis_local_linvel",
        "pelvis_gyro",
        "pelvis_acceleration",
        "pelvis_upvector",
        "torso_gyro",
        "torso_acceleration",
        "torso_upvector",
        "left_foot_quat",
        "right_foot_quat",
    ):
        assert name in model_text


def test_g1_box_tracking_scene_compiles_with_pelvis_imu_sensor_names():
    mujoco = pytest.importorskip("mujoco")

    repo_root = Path(__file__).parents[2]
    scene_xml = (
        repo_root / "src" / "unilab" / "assets" / "robots" / "g1" / "scene_flat_with_largebox.xml"
    )

    model = mujoco.MjModel.from_xml_path(str(scene_xml))

    for sensor_name in (
        "pelvis_local_linvel",
        "pelvis_gyro",
        "pelvis_acceleration",
        "pelvis_upvector",
        "torso_gyro",
        "torso_acceleration",
        "torso_upvector",
    ):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name) >= 0


def test_g1_box_tracking_scene_uses_sphere_hand_and_box_tracking_mesh():
    repo_root = Path(__file__).parents[2]
    scene_text = (
        repo_root / "src" / "unilab" / "assets" / "robots" / "g1" / "scene_flat_with_largebox.xml"
    ).read_text()

    for snippet in (
        '<include file="g1_sphere_hand.xml"/>',
        'mesh name="largebox_mesh" file="box_tracking/largebox.obj"',
        '<freejoint name="largebox_joint"/>',
        '<geom name="largebox" type="mesh" mesh="largebox_mesh"',
        "0.0 0.5 0.85",
    ):
        assert snippet in scene_text

    for name in (
        "left_foot_contact_0",
        "left_foot_contact_1",
        "left_foot_contact_2",
        "left_foot_contact_3",
        "right_foot_contact_0",
        "right_foot_contact_1",
        "right_foot_contact_2",
        "right_foot_contact_3",
    ):
        assert name in scene_text


def test_allegro_rotation_and_grasp_registries_are_manager_only():
    from unilab.base import registry
    from unilab.base.config_materialization import apply_cfg_overrides
    from unilab.envs import ManagerBasedRlEnvCfg

    ensure_registries()
    metadata = registry.list_registered_envs()
    assert metadata["AllegroInhandRotation"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mujoco"],
    }
    assert metadata["AllegroInhandRotationGrasp"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mujoco"],
    }

    cfg = registry.materialize_env_config("AllegroInhandRotation")
    assert isinstance(cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(cfg, _allegro_manager_override())
    assert cfg.policy_observation_group == "policy"
    assert cfg.critic_observation_group is None
    assert cfg.observations["policy"].history_length == 3
    assert list(cfg.actions) == ["hand"]
    assert list(cfg.terminations) == ["dropped", "time_out"]
    assert list(cfg.rewards) == [
        "rotate",
        "obj_linvel",
        "pose_diff",
        "torque",
        "work",
        "drop",
    ]
    assert not hasattr(cfg, "reward_config")

    grasp_cfg = registry.materialize_env_config("AllegroInhandRotationGrasp")
    assert isinstance(grasp_cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(
        grasp_cfg,
        _allegro_manager_override(task="allegro_inhand_grasp"),
    )
    assert grasp_cfg.observations["policy"].history_length == 3
    assert grasp_cfg.actions["hand"].action_scale == 0.0
    assert list(grasp_cfg.terminations) == ["dropped", "time_out", "invalid_grasp"]
    assert list(grasp_cfg.metrics) == [
        "fingertips_close",
        "enough_contacts",
        "ball_held",
        "valid",
    ]
    assert list(grasp_cfg.recorders) == ["grasp_cache"]


def test_allegro_manager_configured_missing_grasp_cache_fails_closed(tmp_path: Path):
    from unilab.managers import EventTermCfg
    from unilab.tasks.manipulation.allegro_inhand.manager_terms import AllegroHandBallReset

    entity = SimpleNamespace(
        num_joints=16,
        data=SimpleNamespace(
            default_root_state=np.zeros((2, 13), dtype=np.float32),
            actuator_ctrl_range=np.tile([-1.0, 1.0], (16, 1)),
        ),
    )
    env = SimpleNamespace(num_envs=2, scene={"robot": entity})
    cfg = EventTermCfg(
        func=AllegroHandBallReset,
        mode="reset",
        params={
            "entity_name": "robot",
            "grasp_cache_path": str(tmp_path / "missing.npy"),
            "joint_noise": 0.0,
            "ball_velocity_noise": 0.0,
            "ball_z_offset": 0.0,
        },
    )

    with pytest.raises(FileNotFoundError, match="configured grasp cache does not exist"):
        AllegroHandBallReset(cfg, cast(Any, env))


def _allegro_grasp_term_fixture() -> tuple[Any, Any, np.ndarray]:
    from unilab.managers import TerminationTermCfg
    from unilab.tasks.manipulation.allegro_inhand.grasp_gen import (
        AllegroGraspQualityTermination,
    )
    from unilab.tasks.manipulation.allegro_inhand.manager_terms import (
        AllegroRotationObservation,
    )

    num_envs = 3
    states = np.arange(num_envs * 23, dtype=np.float32).reshape(num_envs, 23)
    observation = object.__new__(AllegroRotationObservation)
    observation.dof_pos = states[:, :16]
    observation.ball_pos = np.array(
        [[0.0, 0.0, 0.2], [0.0, 0.0, 0.2], [0.0, 0.0, 0.1]], dtype=np.float32
    )
    observation.ball_quat = states[:, 19:23]
    observation._last_counter = 1

    body_pos = np.repeat(observation.ball_pos[:, None, :], 4, axis=1)
    body_pos[:, :, 0] += 0.05
    body_pos[1, 0, 0] += 0.2
    contacts = np.array(
        [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    entity = SimpleNamespace(
        data=SimpleNamespace(body_link_pos_w=body_pos),
        find_bodies=lambda names, preserve_order: (list(range(4)), list(names)),
    )

    class _Scene(dict):
        def bind_sensor_data(self, names):
            assert tuple(names) == ("ff_contact", "mf_contact", "rf_contact", "th_contact")
            return SimpleNamespace(dimensions=(1, 1, 1, 1), read=lambda: contacts)

    env = SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        common_step_counter=1,
        scene=_Scene(robot=entity),
        observation_manager=SimpleNamespace(
            get_term_cfg=lambda group, name: SimpleNamespace(func=observation)
        ),
        action_manager=SimpleNamespace(),
        reset_time_outs=np.ones(num_envs, dtype=np.bool_),
        reset_terminated=np.zeros(num_envs, dtype=np.bool_),
        extras={"log": {}},
    )
    cfg = TerminationTermCfg(
        func=AllegroGraspQualityTermination,
        params={
            "entity_name": "robot",
            "observation_group": "policy",
            "observation_term": "rotation",
            "fingertip_body_names": ["ff_tip", "mf_tip", "rf_tip", "th_tip"],
            "contact_sensor_names": [
                "ff_contact",
                "mf_contact",
                "rf_contact",
                "th_contact",
            ],
            "max_fingertip_distance": 0.1,
            "minimum_contacts": 2,
            "minimum_ball_height": 0.125,
            "enabled": True,
        },
    )
    term = AllegroGraspQualityTermination(cfg, cast(Any, env))
    env.termination_manager = SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=term))
    return env, term, states


def test_allegro_grasp_quality_term_matches_legacy_conditions():
    from unilab.managers import MetricsTermCfg
    from unilab.tasks.manipulation.allegro_inhand.grasp_gen import AllegroGraspQualityMetric

    env, term, _ = _allegro_grasp_term_fixture()

    np.testing.assert_array_equal(term(cast(Any, env)), [False, True, True])
    np.testing.assert_array_equal(term.fingertips_close, [True, False, True])
    np.testing.assert_array_equal(term.enough_contacts, [True, True, False])
    np.testing.assert_array_equal(term.ball_held, [True, True, False])
    metric = AllegroGraspQualityMetric(
        MetricsTermCfg(
            func=AllegroGraspQualityMetric,
            params={"quality_term_name": "invalid_grasp", "condition": "valid"},
        ),
        cast(Any, env),
    )
    np.testing.assert_array_equal(metric(cast(Any, env)), [1.0, 0.0, 0.0])


def test_allegro_grasp_recorder_saves_target_and_raises_run_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unilab.base.run_control import RunComplete
    from unilab.managers import RecorderTermCfg
    from unilab.tasks.manipulation.allegro_inhand import grasp_gen

    env, term, states = _allegro_grasp_term_fixture()
    term(cast(Any, env))
    cache_path = tmp_path / "allegro.npy"
    cfg = RecorderTermCfg(
        func=grasp_gen.AllegroGraspRecorder,
        params={
            "quality_term_name": "invalid_grasp",
            "output_path": str(cache_path),
            "collection_target": 2,
            "auto_save": True,
        },
    )
    recorder = grasp_gen.AllegroGraspRecorder(cfg, cast(Any, env))
    save_calls: list[Path] = []
    real_save = grasp_gen.np.save

    def save_once(path: str | Path, values: np.ndarray) -> None:
        save_calls.append(Path(path))
        real_save(path, values)

    monkeypatch.setattr(grasp_gen.np, "save", save_once)
    with pytest.raises(RunComplete) as caught:
        recorder.record_pre_reset(torch.arange(3, dtype=torch.int64))

    expected = np.concatenate(
        (states[:, :16], term.observation.ball_pos, states[:, 19:23]), axis=1, dtype=np.float32
    )
    np.testing.assert_array_equal(np.load(cache_path), expected[:2])
    assert save_calls == [cache_path]
    assert env.extras["log"] == {
        "grasp_cache/saved": 1.0,
        "grasp_cache/num_states": 2.0,
        "grasp/target_reached": 1.0,
    }
    assert dict(caught.value.summary) == {
        "collected_grasps": 3,
        "saved_grasps": 2,
        "grasp_collection_target": 2,
    }

    recorder.close()
    assert save_calls == [cache_path]


def test_allegro_grasp_recorder_close_autosaves_and_io_failure_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unilab.managers import RecorderTermCfg
    from unilab.tasks.manipulation.allegro_inhand import grasp_gen

    env, term, _ = _allegro_grasp_term_fixture()
    term(cast(Any, env))
    cache_path = tmp_path / "allegro.npy"
    cfg = RecorderTermCfg(
        func=grasp_gen.AllegroGraspRecorder,
        params={
            "quality_term_name": "invalid_grasp",
            "output_path": str(cache_path),
            "collection_target": 3,
            "auto_save": True,
        },
    )
    recorder = grasp_gen.AllegroGraspRecorder(cfg, cast(Any, env))
    env.reset_terminated[1] = True
    recorder.record_pre_reset(torch.tensor([0, 1], dtype=torch.int64))
    assert recorder.total_saved_grasps == 1
    assert not cache_path.exists()
    recorder.close()
    assert np.load(cache_path).shape == (1, 23)

    failed_path = tmp_path / "failed.npy"
    failed_cfg = RecorderTermCfg(
        func=grasp_gen.AllegroGraspRecorder,
        params={
            "quality_term_name": "invalid_grasp",
            "output_path": str(failed_path),
            "collection_target": 1,
            "auto_save": True,
        },
    )
    failed = grasp_gen.AllegroGraspRecorder(failed_cfg, cast(Any, env))
    sentinel = OSError("disk full")
    monkeypatch.setattr(
        grasp_gen.np, "save", lambda *_args, **_kwargs: (_ for _ in ()).throw(sentinel)
    )
    with pytest.raises(OSError) as caught:
        failed.record_pre_reset(torch.tensor([0], dtype=torch.int64))
    assert caught.value is sentinel
    assert failed.cache_saved is False


# ---------------------------------------------------------------------------
# Fast env/backend smoke tests
# ---------------------------------------------------------------------------

# Environments that don't need special config overrides
_STANDARD_ENVS = [
    "G1WalkFlat",
    "AllegroInhandRotation",
    "AllegroInhandRotationGrasp",
]


@pytest.mark.parametrize("env_name", _STANDARD_ENVS)
def test_env_reset_and_step(env_name: str):
    """Every registered env must be constructible, resetable, and steppable.

    Verifies:
    - observation/action spaces are valid
    - init_state + reset produces dict obs with correct keys and shapes
    - step with zero actions produces dict obs, scalar reward, bool done
    """
    _require_mujoco_runtime()
    ensure_registries()
    from unilab.base import registry

    # Provide config overrides for envs that require them via Hydra
    env_cfg_override = None
    if env_name == "G1WalkFlat":
        env_cfg_override = _g1_manager_override("g1_walk_flat")
    elif env_name == "AllegroInhandRotation":
        env_cfg_override = _allegro_manager_override()
    elif env_name == "AllegroInhandRotationGrasp":
        env_cfg_override = _allegro_manager_override(task="allegro_inhand_grasp")

    env = cast(
        Any,
        registry.make(
            env_name, num_envs=2, sim_backend="mujoco", env_cfg_override=env_cfg_override
        ),
    )
    try:
        # 1. Spaces
        obs_space = env.observation_space
        act_space = env.action_space
        assert obs_space.shape is not None and obs_space.shape[0] > 0
        assert act_space.shape is not None and act_space.shape[0] > 0

        # obs_groups_spec must sum to observation_space total dim
        spec = env.obs_groups_spec
        assert isinstance(spec, dict)
        assert sum(spec.values()) == obs_space.shape[0]

        # 2. Reset
        state = env.init_state()
        assert isinstance(state.obs, dict)
        for key, dim in spec.items():
            assert key in state.obs, f"obs missing group '{key}'"
            assert state.obs[key].shape == (2, dim), (
                f"obs['{key}'] shape mismatch: {state.obs[key].shape} != (2, {dim})"
            )

        # 3. Step with zero actions
        actions = torch.zeros((2, int(act_space.shape[0])), dtype=torch.float32)
        state = env.step(actions)
        assert isinstance(state.obs, dict)
        for key, dim in spec.items():
            assert state.obs[key].shape == (2, dim)
        assert state.reward.shape == (2,)
        assert state.terminated.shape == (2,)
        assert state.truncated.shape == (2,)
    finally:
        env.close()


def test_allegro_manager_runtime_transition_contract():
    _require_mujoco_runtime()
    ensure_registries()
    from unilab.base import registry
    from unilab.envs import ManagerBasedRlEnv
    from unilab.tasks.manipulation.allegro_inhand.manager_terms import (
        AllegroIncrementalPositionAction,
        AllegroRotationObservation,
    )

    manager_override = _allegro_manager_override()
    manager_override["observations"]["policy"]["terms"]["rotation"]["params"]["joint_noise"] = 0.0
    env = registry.make(
        "AllegroInhandRotation",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override=manager_override,
    )
    assert isinstance(env, ManagerBasedRlEnv)
    try:
        manager_initial = env.init_state()
        history = manager_initial.obs["obs"].reshape(2, 3, 35)
        np.testing.assert_array_equal(history[:, 0], history[:, 1])
        np.testing.assert_array_equal(history[:, 1], history[:, 2])

        action = env.action_manager.get_term("hand")
        assert isinstance(action, AllegroIncrementalPositionAction)
        target_before = action.target.clone()
        actions = torch.full((2, 16), 0.25, dtype=torch.float32)
        manager_state = env.step(actions)
        expected_target = torch.clamp(
            target_before + 0.25 / 24.0,
            min=action.ctrl_lower,
            max=action.ctrl_upper,
        )
        torch.testing.assert_close(action.target, expected_target, rtol=0.0, atol=1.0e-7)

        observation = env.observation_manager.get_term_cfg("policy", "rotation").func
        assert isinstance(observation, AllegroRotationObservation)
        current_frame = observation(env)
        np.testing.assert_allclose(
            manager_state.obs["obs"][:, -35:].cpu().numpy(),
            current_frame,
            rtol=0.0,
            atol=1.0e-7,
        )
        assert torch.isfinite(manager_state.obs["obs"]).all()
        assert torch.isfinite(manager_state.reward).all()
        assert manager_state.terminated.dtype == torch.bool
        assert action.action_dim == 16
        assert env.obs_groups_spec == {"obs": 105}
    finally:
        env.close()


def test_allegro_incremental_action_uses_device_tensors_and_partial_reset():
    _require_mujoco_runtime()
    ensure_registries()
    from unilab.base import registry
    from unilab.envs import ManagerBasedRlEnv
    from unilab.tasks.manipulation.allegro_inhand.manager_terms import (
        AllegroIncrementalPositionAction,
    )

    env = registry.make(
        "AllegroInhandRotation",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override=_allegro_manager_override(),
    )
    assert isinstance(env, ManagerBasedRlEnv)
    try:
        env.reset(seed=23)
        action = env.action_manager.get_term("hand")
        assert isinstance(action, AllegroIncrementalPositionAction)
        for value in (action.raw_action, action.target, action.ctrl_lower, action.ctrl_upper):
            assert value.dtype == torch.float32
            assert value.device == env.device

        target_before = action.target.clone()
        action.process_actions(torch.full((2, 16), 24.0, dtype=torch.float32, device=env.device))
        torch.testing.assert_close(action.raw_action, torch.full_like(action.raw_action, 24.0))
        torch.testing.assert_close(action.clipped_action, torch.ones_like(action.raw_action))
        torch.testing.assert_close(
            action.target,
            torch.clamp(
                target_before + 0.041666666666666664,
                min=action.ctrl_lower,
                max=action.ctrl_upper,
            ),
        )
        action.reset(torch.asarray([1], dtype=torch.int64, device=env.device))
        torch.testing.assert_close(action.raw_action[1], torch.zeros_like(action.raw_action[1]))
        torch.testing.assert_close(action.clipped_action[1], torch.zeros_like(action.raw_action[1]))
        torch.testing.assert_close(action.target[1], target_before[1])
        torch.testing.assert_close(
            action.raw_action[0], torch.full_like(action.raw_action[0], 24.0)
        )

        with pytest.raises(TypeError, match="expected float32"):
            action.process_actions(torch.ones((2, 16), dtype=torch.float64))
        with pytest.raises(ValueError, match="expected device"):
            action.process_actions(
                torch.ones((2, 16), dtype=torch.float32, device=torch.device("cpu"))
                if env.device.type != "cpu"
                else torch.ones((2, 16), dtype=torch.float32, device=torch.device("meta"))
            )
    finally:
        env.close()


@pytest.mark.parametrize("sim_backend", ["mujoco"])
def test_allegro_grasp_manager_runtime_uses_zero_increment_action(sim_backend: str, tmp_path: Path):
    if sim_backend == "mujoco":
        _require_mujoco_runtime()
    else:
        pytest.importorskip("motrixsim")
    ensure_registries()
    from unilab.base import registry
    from unilab.envs import ManagerBasedRlEnv
    from unilab.tasks.manipulation.allegro_inhand.grasp_gen import (
        AllegroGraspQualityTermination,
        AllegroGraspRecorder,
    )
    from unilab.tasks.manipulation.allegro_inhand.manager_terms import (
        AllegroIncrementalPositionAction,
    )

    override = _allegro_manager_override(sim_backend, task="allegro_inhand_grasp")
    override["auto_reset"] = False
    override["terminations"]["invalid_grasp"]["params"]["enabled"] = False
    override["recorders"]["grasp_cache"]["params"].update(
        {"output_path": str(tmp_path / f"{sim_backend}.npy"), "auto_save": False}
    )
    env = registry.make(
        "AllegroInhandRotationGrasp",
        num_envs=2,
        sim_backend=sim_backend,
        env_cfg_override=override,
    )
    assert isinstance(env, ManagerBasedRlEnv)
    try:
        initial = env.init_state()
        action = env.action_manager.get_term("hand")
        quality = env.termination_manager.get_term_cfg("invalid_grasp").func
        recorder = env.recorder_manager.get_term("grasp_cache")
        assert isinstance(action, AllegroIncrementalPositionAction)
        assert isinstance(quality, AllegroGraspQualityTermination)
        assert isinstance(recorder, AllegroGraspRecorder)
        target = action.target.clone()

        state = env.step(torch.ones((2, 16), dtype=torch.float32))
        torch.testing.assert_close(action.target, target)
        torch.testing.assert_close(state.reward, torch.zeros(2))
        assert initial.obs["obs"].shape == (2, 105)
        assert state.obs["obs"].shape == (2, 105)
        assert quality.last_counter == env.common_step_counter
    finally:
        env.close()


_MOTION_CORE_RUNTIME_CASES = (
    pytest.param("ppo", "g1_motion_tracking", "G1MotionTracking", 160, 286, 29, False),
    pytest.param("appo", "g1_motion_tracking", "G1MotionTracking", 160, 286, 29, False),
)


def test_g1_motion_core_registrations_are_manager_only() -> None:
    from unilab.base import registry

    ensure_registries()
    metadata = registry.list_registered_envs()
    for task_name in ("G1MotionTracking",):
        assert metadata[task_name] == {
            "config_factory": "ManagerBasedRlEnvCfg",
            "available_backends": ["mujoco"],
        }
    # MJWarp and Genesis are the scoped DEVICE_RESIDENT tensor owners.
    assert metadata["G1MotionTrackingSAC"]["config_factory"] == "ManagerBasedRlEnvCfg"
    assert set(metadata["G1MotionTrackingSAC"]["available_backends"]) >= {
        "mujoco",
        "mjwarp",
        "genesis",
    }


def test_g1_motion_manager_ppo_wraps_only_active_rows_in_one_state_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_registries()
    _require_mujoco_runtime()
    from unilab.base import registry

    _, override = _motion_manager_override("g1_motion_tracking", "mujoco")
    env = registry.make(
        "G1MotionTracking",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override=override,
    )
    try:
        env.init_state()
        command = env.command_manager.get_term("motion")
        command.time_steps[:] = command.sampler.current_clip_end_frames
        env.reset_buf.copy_(torch.tensor([True, False], device=env.device))

        set_state_env_ids: list[np.ndarray] = []
        original_set_state = env._backend.set_state

        def record_set_state(
            env_ids: np.ndarray,
            qpos: np.ndarray,
            qvel: np.ndarray,
            *,
            randomization: Any = None,
        ) -> Any:
            set_state_env_ids.append(env_ids.copy())
            return original_set_state(
                env_ids,
                qpos,
                qvel,
                randomization=randomization,
            )

        monkeypatch.setattr(env._backend, "set_state", record_set_state)
        all_ids = np.arange(env.num_envs, dtype=np.int32)
        with env._reset_state.scoped(all_ids):
            env.command_manager.compute(dt=0.0)
        env.command_manager.post_compute()

        assert len(set_state_env_ids) == 1
        np.testing.assert_array_equal(set_state_env_ids[0], [1])
        assert command.time_steps[0] == command.sampler.current_clip_end_frames[0]
        assert command.time_steps[1] <= command.sampler.current_clip_end_frames[1]
        expected_motion = command.motion.get_motion_at_frame(command.time_steps)
        np.testing.assert_array_equal(command.joint_pos, expected_motion.joint_pos)
        np.testing.assert_array_equal(
            command._robot_body_pos_w,
            command.robot.data.body_link_pos_w[:, command._robot_body_ids],
        )
        assert command._robot_cache_step == env.common_step_counter
    finally:
        env.close()


def test_g1_motion_manager_sac_clip_end_is_truncation() -> None:
    ensure_registries()
    _require_mujoco_runtime()
    from unilab.base import registry

    _, override = _motion_manager_override(
        "g1_motion_tracking",
        "mujoco",
        config_root="sac",
    )
    override["auto_reset"] = False
    env = registry.make(
        "G1MotionTrackingSAC",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override=override,
    )
    try:
        env.init_state()
        command = env.command_manager.get_term("motion")
        command.time_steps[:] = command.sampler.current_clip_end_frames

        state = env.step(torch.zeros((2, 29), dtype=torch.float32))

        np.testing.assert_array_equal(state.terminated, [False, False])
        np.testing.assert_array_equal(state.truncated, [True, True])
        np.testing.assert_array_equal(command.time_steps, command.sampler.current_clip_end_frames)
    finally:
        env.close()


def test_sac_g1_motion_mjwarp_dr_runtime_applies_reset_and_interval_dr() -> None:
    ensure_registries()
    _require_mjwarp_runtime()
    from unilab.base import registry

    _, override = _motion_manager_override(
        "g1_motion_tracking",
        "mjwarp",
        config_root="sac",
    )
    push_robot = override["events"]["push_robot"]
    push_robot["interval_range_s"] = [0.0, 0.0]
    env = registry.make(
        "G1MotionTrackingSAC",
        num_envs=2,
        sim_backend="mjwarp",
        env_cfg_override=override,
    )
    try:
        state = env.init_state()
        assert torch.isfinite(state.obs["obs"]).all()
        assert torch.isfinite(state.obs["critic"]).all()

        for _ in range(3):
            state = env.step(torch.zeros((2, 29), dtype=torch.float32, device=env.device))
            assert torch.isfinite(state.obs["obs"]).all()
            assert torch.isfinite(state.obs["critic"]).all()
            assert torch.isfinite(state.reward).all()

        velocity_range = env.event_manager.get_term_cfg("push_robot").params["velocity_range"]
        assert velocity_range["x"] == [-0.5, 0.5]
        assert velocity_range["z"] == [-0.2, 0.2]
        assert all(velocity_range[axis] == [0.0, 0.0] for axis in ("roll", "pitch", "yaw"))
    finally:
        env.close()


def test_flashsac_g1_motion_mjwarp_tensor_anchor_observations_roll_out() -> None:
    """The canonical FlashSAC MJWarp owner exercises tensor-native anchors."""
    ensure_registries()
    _require_mjwarp_runtime()
    from unilab.base import registry
    from unilab.envs import ManagerBasedRlEnv

    _, override = _motion_manager_override(
        "g1_motion_tracking",
        "mjwarp",
        config_root="flashsac",
    )
    env = registry.make(
        "G1MotionTrackingSAC",
        num_envs=2,
        sim_backend="mjwarp",
        env_cfg_override=override,
    )
    assert isinstance(env, ManagerBasedRlEnv)
    command = env.command_manager.get_term("motion")
    assert command.tensor_carrier is True
    assert env.command_manager.uses_tensor_reset_rows()
    try:
        assert env.obs_groups_spec == {"obs": 160, "critic": 289}

        state = env.init_state()
        for _ in range(3):
            state = env.step(torch.zeros((2, 29), dtype=torch.float32, device=env.device))

        from unisim.backend.base import SelectedResetPublication

        capabilities = env.backend.get_tensor_capabilities()
        assert capabilities.selected_reset_publication is (
            SelectedResetPublication.AUTHORITATIVE_VIEWS
        )
        step_calls = 0
        original_step = env.backend.step_tensor

        def count_step(*args: Any, **kwargs: Any) -> Any:
            nonlocal step_calls
            step_calls += 1
            return original_step(*args, **kwargs)

        env.backend.step_tensor = count_step  # type: ignore[method-assign]
        env.reset(env_indices=torch.tensor([1], dtype=torch.int64, device=env.device))
        assert step_calls == 0
        assert state.obs["obs"].shape == (2, 160)
        assert state.obs["critic"].shape == (2, 289)
        assert all(torch.isfinite(values).all() for values in state.obs.values())
        assert torch.isfinite(state.reward).all()
    finally:
        env.close()


def test_flashsac_g1_motion_mujoco_tensor_command_roll_out() -> None:
    """The scoped MuJoCo host bridge owns a tensor motion command carrier."""
    ensure_registries()
    _require_mujoco_runtime()
    script = textwrap.dedent(
        """
        from pathlib import Path
        import torch
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from unilab.base import registry
        from unilab.base.config_adapter import BackendAdapter
        from unilab.envs import ManagerBasedRlEnv

        registry.ensure_registries()
        GlobalHydra.instance().clear()
        root = Path.cwd()
        with initialize_config_dir(
            config_dir=str(root / "src/unilab/conf/flashsac"), version_base="1.3"
        ):
            owner = compose("config", overrides=["task=g1_motion_tracking/mujoco"])
        override = BackendAdapter(
            owner, root_dir=root, algo_name="flashsac"
        ).build_task_env_cfg_override()
        env = registry.make(
            "G1MotionTrackingSAC",
            num_envs=2,
            sim_backend="mujoco",
            env_cfg_override=override,
        )
        assert isinstance(env, ManagerBasedRlEnv)
        try:
            command = env.command_manager.get_term("motion")
            assert command.tensor_carrier is True
            assert env.torch_rng is not None
            state = env.reset(seed=7)[0]
            for _ in range(3):
                state = env.step(torch.zeros((2, 29), dtype=torch.float32))
            assert state.obs["obs"].shape == (2, 160)
            assert torch.isfinite(state.obs["obs"]).all()
        finally:
            env.close()
        print("[mujoco tensor motion rollout] OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "[mujoco tensor motion rollout] OK" in result.stdout


def test_flashsac_g1_motion_genesis_manager_tensor_command_roll_out() -> None:
    """The canonical FlashSAC Genesis owner exercises the Manager tensor path."""
    ensure_registries()
    _require_genesis_runtime()
    from unilab.base import registry
    from unilab.envs import ManagerBasedRlEnv

    _, override = _motion_manager_override(
        "g1_motion_tracking",
        "genesis",
        config_root="flashsac",
    )
    env = registry.make(
        "G1MotionTrackingSAC",
        num_envs=2,
        sim_backend="genesis",
        env_cfg_override=override,
    )
    assert isinstance(env, ManagerBasedRlEnv)
    try:
        assert env.obs_groups_spec == {"obs": 160, "critic": 289}
        command = env.command_manager.get_term("motion")
        assert command.tensor_carrier is True
        assert env.command_manager.uses_tensor_reset_rows()

        state = env.init_state()
        for _ in range(3):
            state = env.step(torch.zeros((2, 29), dtype=torch.float32, device=env.device))

        assert state.obs["obs"].shape == (2, 160)
        assert state.obs["critic"].shape == (2, 289)
        assert all(torch.isfinite(values).all() for values in state.obs.values())
        assert torch.isfinite(state.reward).all()
    finally:
        env.close()
        env._backend.close()


def test_flashsac_g1_motion_newton_manager_tensor_command_roll_out() -> None:
    """The canonical FlashSAC Newton owner exercises the Manager tensor path."""
    ensure_registries()
    _require_newton_runtime()
    from unilab.base import registry
    from unilab.envs import ManagerBasedRlEnv

    _, override = _motion_manager_override(
        "g1_motion_tracking",
        "newton",
        config_root="flashsac",
    )
    env = registry.make(
        "G1MotionTrackingSAC",
        num_envs=2,
        sim_backend="newton",
        env_cfg_override=override,
    )
    assert isinstance(env, ManagerBasedRlEnv)
    try:
        assert env.obs_groups_spec == {"obs": 160, "critic": 289}
        command = env.command_manager.get_term("motion")
        assert command.tensor_carrier is True
        assert env.command_manager.uses_tensor_reset_rows()

        state = env.init_state()
        for _ in range(3):
            state = env.step(torch.zeros((2, 29), dtype=torch.float32, device=env.device))

        assert state.obs["obs"].shape == (2, 160)
        assert state.obs["critic"].shape == (2, 289)
        assert all(torch.isfinite(values).all() for values in state.obs.values())
        assert torch.isfinite(state.reward).all()
    finally:
        env.close()
        env._backend.close()


def test_flashsac_g1_motion_motrix_manager_tensor_command_roll_out() -> None:
    """The canonical FlashSAC Motrix owner exercises the packed host bridge."""
    ensure_registries()
    _require_motrix_runtime()
    from unilab.base import registry
    from unilab.envs import ManagerBasedRlEnv

    _, override = _motion_manager_override(
        "g1_motion_tracking",
        "motrix",
        config_root="flashsac",
    )
    env = registry.make(
        "G1MotionTrackingSAC",
        num_envs=2,
        sim_backend="motrix",
        env_cfg_override=override,
    )
    assert isinstance(env, ManagerBasedRlEnv)
    try:
        assert env.obs_groups_spec == {"obs": 160, "critic": 289}
        command = env.command_manager.get_term("motion")
        assert command.tensor_carrier is True
        assert env.command_manager.uses_tensor_reset_rows()

        state = env.init_state()
        for _ in range(3):
            state = env.step(torch.zeros((2, 29), dtype=torch.float32, device=env.device))

        assert state.obs["obs"].shape == (2, 160)
        assert state.obs["critic"].shape == (2, 289)
        assert all(torch.isfinite(values).all() for values in state.obs.values())
        assert torch.isfinite(state.reward).all()
    finally:
        env.close()
        env._backend.close()


@pytest.mark.parametrize(
    ("task", "backend"), [("g1_motion_tracking", "mjwarp"), ("g1_motion_tracking", "newton")]
)
def test_selected_reset_publication_requires_no_manager_readiness_step(
    task: str, backend: str
) -> None:
    """Selected reset publishes authoritative views without a control step."""
    ensure_registries()
    if backend == "mjwarp":
        _require_mjwarp_runtime()
    elif backend == "newton":
        _require_newton_runtime()
    else:
        _require_genesis_runtime()
    from unisim.backend.base import SelectedResetPublication

    from unilab.base import registry

    _, override = _motion_manager_override(
        task,
        backend,
        config_root="flashsac",
    )
    env = registry.make(
        "G1MotionTrackingSAC",
        num_envs=2,
        sim_backend=backend,
        env_cfg_override=override,
    )
    try:
        capabilities = env.backend.get_tensor_capabilities()
        assert capabilities.selected_reset
        assert (
            capabilities.selected_reset_publication is SelectedResetPublication.AUTHORITATIVE_VIEWS
        )
        env.init_state()

        step_calls = 0
        original_step = env.backend.step_tensor

        def count_step(*args: Any, **kwargs: Any) -> Any:
            nonlocal step_calls
            step_calls += 1
            return original_step(*args, **kwargs)

        env.backend.step_tensor = count_step  # type: ignore[method-assign]
        env.reset(env_indices=torch.tensor([1], dtype=torch.int64, device=env.device))
        assert step_calls == 0

        state_views = env.backend.get_state_views(("qpos", "qvel"), device=env.device)
        assert state_views["qpos"].shape == (2, env.backend.get_public_state_widths().nq)
        assert state_views["qvel"].shape == (2, env.backend.get_public_state_widths().nv)
        assert torch.isfinite(state_views["qpos"]).all()
        assert torch.isfinite(state_views["qvel"]).all()
    finally:
        env.close()
        env._backend.close()


def test_newton_backend_close_disables_graphs_and_fails_closed() -> None:
    """Newton teardown publishes disabled graph diagnostics before rejecting use."""
    ensure_registries()
    _require_newton_runtime()
    from unilab.base import registry

    _, override = _motion_manager_override(
        "g1_motion_tracking",
        "newton",
        config_root="flashsac",
    )
    env = registry.make(
        "G1MotionTrackingSAC",
        num_envs=2,
        sim_backend="newton",
        env_cfg_override=override,
    )
    try:
        env.init_state()
        before = env.backend.get_tensor_runtime_diagnostics()["cuda_graph"]
        assert before.requested is True
    finally:
        env.close()
        env.backend.close()

    after = env.backend.get_tensor_runtime_diagnostics()["cuda_graph"]
    assert after.enabled is False
    assert after.disable_reason is not None
    with pytest.raises(RuntimeError, match="newton backend is closed"):
        env.backend.get_state_views(("qpos", "qvel"), device=env.device)


@pytest.mark.parametrize(
    ("config_root", "task", "identity", "actor_dim", "critic_dim", "action_dim", "truncate"),
    _MOTION_CORE_RUNTIME_CASES,
)
@pytest.mark.parametrize("sim_backend", ["mujoco"])
def test_g1_motion_core_manager_reset_and_step(
    config_root: str,
    task: str,
    identity: str,
    actor_dim: int,
    critic_dim: int,
    action_dim: int,
    truncate: bool,
    sim_backend: str,
) -> None:
    ensure_registries()
    from unilab.base import registry
    from unilab.envs import ManagerBasedRlEnv

    if sim_backend == "mujoco":
        _require_mujoco_runtime()
    else:
        pytest.importorskip("motrixsim")

    task_name, override = _motion_manager_override(
        task,
        sim_backend,
        config_root=config_root,
    )
    assert task_name == identity
    env = registry.make(
        identity,
        num_envs=2,
        sim_backend=sim_backend,
        env_cfg_override=override,
    )
    assert isinstance(env, ManagerBasedRlEnv)
    try:
        assert env.obs_groups_spec == {"obs": actor_dim, "critic": critic_dim}
        assert env.action_space.shape == (action_dim,)
        command = env.command_manager.get_term("motion")
        assert command.cfg.params.truncate_on_clip_end is truncate
        if sim_backend == "motrix" and "deploy" in task:
            assert env._cfg.events["foot_friction"] is None
            assert env._cfg.events["push_robot"] is None
        if sim_backend == "motrix" and config_root in {"ppo", "appo"}:
            root_pos = env._cfg.rewards["motion_global_root_pos"]
            action_rate = env._cfg.rewards["action_rate_l2"]
            assert root_pos is not None
            assert action_rate is not None
            expected_weights = (1.0, -0.05) if config_root == "ppo" else (0.5, -0.1)
            assert (root_pos.weight, action_rate.weight) == pytest.approx(expected_weights)

        state = env.init_state()
        assert state.obs["obs"].shape == (2, actor_dim)
        assert state.obs["critic"].shape == (2, critic_dim)

        state = env.step(torch.zeros((2, action_dim), dtype=torch.float32))
        assert state.reward.shape == (2,)
        assert state.terminated.shape == (2,)
        assert state.truncated.shape == (2,)
        assert torch.isfinite(state.reward).all()
        assert all(torch.isfinite(values).all() for values in state.obs.values())
    finally:
        env.close()


def test_flashsac_motion_reset_publishes_call_graph_counts() -> None:
    """Selected-reset diagnostics count Manager dispatch, not only wall time."""
    ensure_registries()
    _require_mjwarp_runtime()
    from unilab.base import registry
    from unilab.envs import ManagerBasedRlEnv

    _, override = _motion_manager_override(
        "g1_motion_tracking",
        "mjwarp",
        config_root="flashsac",
    )
    env = registry.make(
        "G1MotionTrackingSAC",
        num_envs=4,
        sim_backend="mjwarp",
        env_cfg_override=override,
    )
    assert isinstance(env, ManagerBasedRlEnv)
    try:
        env.init_state()
        # Force every row done so the selected-reset path executes deterministically.
        original_compute = env._compute_truncated

        def terminate_all(state):
            del state
            return torch.ones((env.num_envs,), dtype=torch.bool, device=env.device)

        env._compute_truncated = terminate_all
        state = env.step(torch.zeros((4, 29), dtype=torch.float32, device=env.device))
        env._compute_truncated = original_compute
        timing = state.info["timing"]
        assert timing["reset_done_event_term_count"] == 0.0
        assert timing["reset_done_command_term_count"] == 1.0
        assert timing["reset_done_manager_reset_count"] == 4.0
        assert timing["reset_done_observation_term_count"] == 2.0
        assert timing["reset_done_sampler_host_transfer_count"] == 0.0
    finally:
        env.close()
        env._backend.close()


def test_flashsac_motion_reset_owner_matches_generic_command_and_action_state() -> None:
    """Owned command/action/metric reset state matches generic Manager reset."""
    ensure_registries()
    _require_mjwarp_runtime()
    from unilab.base import registry
    from unilab.envs import ManagerBasedRlEnv
    from unilab.tasks.motion_tracking.common.manager_terms import (
        TensorMotionCommand,
    )

    def make(owner: bool) -> ManagerBasedRlEnv:
        _, override = _motion_manager_override(
            "g1_motion_tracking", "mjwarp", config_root="flashsac"
        )
        if not owner:
            override["reset_owners"] = {}
        env = registry.make(
            "G1MotionTrackingSAC",
            num_envs=8,
            sim_backend="mjwarp",
            env_cfg_override=override,
        )
        assert isinstance(env, ManagerBasedRlEnv)
        return env

    generic = make(False)
    owned = make(True)
    try:
        generic.reset(seed=1811)
        owned.reset(seed=1811)
        # One step gives action history and command metrics nonzero state.
        actions = torch.zeros((8, 29), dtype=torch.float32, device=generic.device)
        generic.step(actions)
        owned.step(actions)
        # Autoreset can clear counts before manual partial reset; force the
        # selected rows done so both environments execute selected reset.
        rows = torch.tensor([0, 2, 5, 7], dtype=torch.int64, device=generic.device)
        generic.reset_terminated.fill_(False)
        generic.reset_time_outs.fill_(False)
        owned.reset_terminated.fill_(False)
        owned.reset_time_outs.fill_(False)
        generic.reset_terminated[rows] = True
        generic.reset_time_outs[rows] = True
        owned.reset_terminated[rows] = True
        owned.reset_time_outs[rows] = True
        generic.reset_buf.fill_(False)
        owned.reset_buf.fill_(False)
        generic.reset_buf[rows] = True
        owned.reset_buf[rows] = True
        generic_obs, _generic_info = generic.reset(env_indices=rows)
        owned_obs, _owned_info = owned.reset(env_indices=rows)

        for name in generic_obs:
            assert name in owned_obs
            assert torch.isfinite(owned_obs[name]).all()
            assert owned_obs[name].shape == generic_obs[name].shape
        torch.testing.assert_close(owned_obs, generic_obs, rtol=2e-6, atol=2e-6)

        generic_action = generic.action_manager
        owned_action = owned.action_manager
        for field in ("action", "prev_action", "prev_prev_action"):
            torch.testing.assert_close(
                getattr(owned_action, field),
                getattr(generic_action, field),
                rtol=0,
                atol=0,
            )
        generic_command = generic.command_manager.get_term("motion")
        owned_command = owned.command_manager.get_term("motion")
        assert isinstance(generic_command, TensorMotionCommand)
        assert isinstance(owned_command, TensorMotionCommand)
        torch.testing.assert_close(owned_command.command, generic_command.command, rtol=0, atol=0)
        torch.testing.assert_close(
            owned_command.time_steps, generic_command.time_steps, rtol=0, atol=0
        )
        torch.testing.assert_close(
            owned_command.joint_default_bias,
            generic_command.joint_default_bias,
            rtol=0,
            atol=0,
        )
        generic_timing = generic._last_reset_manager_timing_ms
        owned_timing = owned._last_reset_manager_timing_ms
        assert generic_timing["reset_done_manager_reset_count"] == 7.0
        assert owned_timing["reset_done_manager_reset_count"] == 4.0
        assert owned_timing["reset_done_sampler_host_transfer_count"] == 0.0
    finally:
        generic.close()
        generic._backend.close()
        owned.close()
        owned._backend.close()
