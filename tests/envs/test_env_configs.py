"""Tests for env config completeness and env instantiation.

Config-attribute tests (non-slow) verify that config dataclasses expose every
attribute accessed by their paired env class, WITHOUT running a simulation.

Slow tests actually call registry.make() and run reset + step.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from unilab.base.registry import ensure_registries


def _require_mujoco_runtime() -> None:
    pytest.importorskip("mujoco", reason="mujoco not installed")
    try:
        from mujoco_uni.batch_env import BatchEnvPool as _  # noqa: F401
    except Exception:
        pytest.skip("mujoco_uni.batch_env not available (platform/libstdc++ issue)")


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
    if task == "g1_walk_rough":
        # There is no ppo g1_walk_rough owner; use the SAC owner instead.
        with initialize_config_dir(
            config_dir=str(repo_root / "src" / "unilab" / "conf" / "sac"), version_base="1.3"
        ):
            cfg = compose("config", overrides=[f"task={task}/mujoco"])
        return BackendAdapter(
            cfg, root_dir=repo_root, algo_name="sac"
        ).build_task_env_cfg_override()
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
# Non-slow: config attribute completeness (no env.step(), no MuJoCo sim)
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
        assert registry.contains("G1MotionTrackingDeploy")
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


def test_g1_walk_tasks_register_to_manager_based_env():
    from unilab.base import registry

    ensure_registries()
    metadata = registry.list_registered_envs()
    assert metadata["G1WalkFlat"]["config_factory"] == "ManagerBasedRlEnvCfg"
    assert metadata["G1WalkFlat"]["available_backends"] == [
        "mujoco",
        "mjwarp",
        "motrix",
        "isaacgym",
        "genesis",
        "isaacsim",
    ]
    assert metadata["G1WalkRough"]["available_backends"] == ["mujoco", "motrix"]


def test_g1_walk_flat_isaacgym_owner_composes_and_materializes():
    """The isaacgym owner composes and materializes a plain manager env cfg."""
    from unilab.base import registry
    from unilab.base.config_materialization import apply_cfg_overrides
    from unilab.envs import ManagerBasedRlEnvCfg

    ensure_registries()
    override = _g1_manager_override("g1_walk_flat", backend="isaacgym")
    env_cfg = registry.materialize_env_config("G1WalkFlat")
    assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(env_cfg, override)
    env_cfg.validate()

    assert env_cfg.scene is not None
    # The subprocess backend consumes the self-contained MJCF scene directly;
    # scene fragments and generated terrain stay unset.
    assert env_cfg.scene.model_file.endswith("robots/g1/scene_flat.xml")
    assert env_cfg.scene.fragment_files == []
    assert env_cfg.scene.terrain is None
    assert env_cfg.scene.default_keyframe_name == "stand"
    assert env_cfg.isaacgym_device_id == 0
    # Effort-mode dofs carry no PD gains, so the owner disables kp/kd
    # randomization like the mjwarp/motrix owners.
    assert env_cfg.events["pd_gains"] is None


@pytest.mark.parametrize("config_group", ("ppo", "sac"))
def test_g1_walk_flat_genesis_owner_composes_and_materializes(config_group):
    """The genesis owners compose and materialize a plain manager env cfg."""
    from unilab.base import registry
    from unilab.base.config_materialization import apply_cfg_overrides
    from unilab.envs import ManagerBasedRlEnvCfg

    ensure_registries()
    override = _g1_manager_override("g1_walk_flat", backend="genesis", config_group=config_group)
    env_cfg = registry.materialize_env_config("G1WalkFlat")
    assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(env_cfg, override)
    env_cfg.validate()

    assert env_cfg.scene is not None
    # The in-process backend consumes the self-contained MJCF scene directly;
    # scene fragments and generated terrain stay unset.
    assert env_cfg.scene.model_file.endswith("robots/g1/scene_flat.xml")
    assert env_cfg.scene.fragment_files == []
    assert env_cfg.scene.terrain is None
    assert env_cfg.scene.default_keyframe_name == "stand"
    # The owner re-declares the MJCF <option integrator="implicitfast"> that
    # Genesis drops at import; the other global options stay at Genesis
    # defaults (None).
    assert env_cfg.genesis_integrator == "implicitfast"
    assert env_cfg.genesis_constraint_solver is None
    assert env_cfg.genesis_friction_cone is None
    assert env_cfg.genesis_solver_iterations is None
    # kp/kd reset randomization stays enabled: the backend declares the
    # measured RESET_TERM_KP/KD DR terms.
    assert env_cfg.events["pd_gains"] is not None


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
        "available_backends": ["mujoco", "motrix", "drake"],
    }
    assert metadata["AllegroInhandRotationGrasp"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mujoco", "motrix"],
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
        recorder.record_pre_reset(np.arange(3, dtype=np.int32))

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
    recorder.record_pre_reset(np.array([0, 1], dtype=np.int32))
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
        failed.record_pre_reset(np.array([0], dtype=np.int32))
    assert caught.value is sentinel
    assert failed.cache_saved is False


def _patch_sharpa_rotation_constructor(
    monkeypatch: pytest.MonkeyPatch,
    initialized_providers: list[Any],
) -> Any:
    from unilab.tasks.manipulation.sharpa_inhand import rotation as sharpa_rotation_module
    from unilab.tasks.manipulation.sharpa_inhand.base import SharpaInhandBaseEnv

    def fake_base_init(self, cfg, backend, num_envs):
        self._cfg = cfg
        self._backend = backend
        self._num_envs = num_envs
        self._np_dtype = np.float64
        self._num_action = 22
        self._num_tactile = 5
        self._num_scales = len(cfg.domain_rand.scale_list)
        self.scale_values = np.asarray(cfg.domain_rand.scale_list, dtype=np.float64)
        self.scale_ids = np.zeros((num_envs,), dtype=np.int32)
        self._object_body_ids = np.zeros((0,), dtype=np.int32)

    def unsupported_backend_metadata(*args, **kwargs):
        raise NotImplementedError("fake backend does not expose Sharpa MuJoCo metadata")

    monkeypatch.setattr(
        sharpa_rotation_module,
        "create_backend",
        lambda *args, **kwargs: SimpleNamespace(
            backend_type="motrix",
            get_actuator_gains=lambda: (
                np.ones(22, dtype=np.float64),
                np.ones(22, dtype=np.float64),
            ),
            get_geom_id=unsupported_backend_metadata,
            get_body_id=unsupported_backend_metadata,
            get_body_subtree_ids=unsupported_backend_metadata,
            get_geom_body_ids=unsupported_backend_metadata,
            get_geom_contact_masks=unsupported_backend_metadata,
            get_geom_names=unsupported_backend_metadata,
            get_geom_friction=unsupported_backend_metadata,
            get_gravity=unsupported_backend_metadata,
            get_body_mass=unsupported_backend_metadata,
            get_body_ipos=unsupported_backend_metadata,
        ),
    )
    monkeypatch.setattr(SharpaInhandBaseEnv, "__init__", fake_base_init)
    monkeypatch.setattr(
        sharpa_rotation_module.SharpaInhandRotationEnv,
        "_init_domain_randomization",
        lambda self, provider: initialized_providers.append(provider),
    )
    return sharpa_rotation_module


def test_sharpa_rotation_explicit_default_provider_materializes_cache(monkeypatch):
    from unilab.tasks.manipulation.sharpa_inhand.rotation import (
        RewardConfig,
        SharpaInhandRotationCfg,
        SharpaInhandRotationDRProvider,
        SharpaInhandRotationEnv,
    )

    initialized_providers: list[Any] = []
    rotation = _patch_sharpa_rotation_constructor(monkeypatch, initialized_providers)
    materialize_calls: list[tuple[str, np.ndarray]] = []
    sentinel_cache = (np.zeros((1, 29), dtype=np.float64),)

    def materialize(path: str, scale_values: np.ndarray) -> tuple[np.ndarray, ...]:
        materialize_calls.append((path, scale_values.copy()))
        return sentinel_cache

    monkeypatch.setattr(rotation, "_materialize_grasp_caches", materialize)
    cfg = SharpaInhandRotationCfg(reward_config=RewardConfig())
    provider = SharpaInhandRotationDRProvider()

    env = cast(Any, SharpaInhandRotationEnv)(
        cfg,
        num_envs=4,
        backend_type="mujoco",
        dr_provider=provider,
    )

    assert initialized_providers == [provider]
    assert len(materialize_calls) == 1
    assert materialize_calls[0][0] == cfg.grasp_cache_path
    np.testing.assert_array_equal(materialize_calls[0][1], env.scale_values)
    assert env._grasp_cache is sentinel_cache


def test_sharpa_grasp_env_initializes_dr_once_with_grasp_provider(monkeypatch):
    from unilab.tasks.manipulation.sharpa_inhand.grasp_gen import (
        SharpaInhandRotationGraspCfg,
        SharpaInhandRotationGraspEnv,
    )

    initialized_providers: list[Any] = []
    rotation = _patch_sharpa_rotation_constructor(monkeypatch, initialized_providers)

    def fail_materialization(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("grasp generation must not materialize rotation caches")

    monkeypatch.setattr(rotation, "_materialize_grasp_caches", fail_materialization)

    cfg = SharpaInhandRotationGraspCfg()
    assert cfg.domain_rand.randomize_pd_gains is False
    assert cfg.domain_rand.randomize_friction is False
    assert cfg.domain_rand.randomize_com is False
    assert cfg.domain_rand.randomize_mass is True
    assert cfg.domain_rand.randomize_mass_lower == pytest.approx(0.05)
    assert cfg.domain_rand.randomize_mass_upper == pytest.approx(0.051)
    assert cfg.domain_rand.force_scale == pytest.approx(0.0)
    assert cfg.domain_rand.random_force_prob_scalar == pytest.approx(0.0)
    assert cfg.domain_rand.joint_noise_scale == pytest.approx(0.02)
    assert cfg.domain_rand.contact_latency == pytest.approx(0.005)
    assert cfg.domain_rand.contact_sensor_noise == pytest.approx(0.01)
    assert cfg.control_config.torque_control is False
    assert cfg.control_config.dof_limits_scale == pytest.approx(0.9)
    assert cfg.obs.enable_tactile is True
    assert cfg.obs.binary_contact is False
    assert cfg.obs.enable_contact_pos is False
    assert cfg.obs.contact_smooth == pytest.approx(0.5)
    assert cfg.obs.contact_threshold == pytest.approx(0.05)
    assert cfg.obs.tactile_force_clip_max == pytest.approx(4.0)
    assert cfg.priv_info.include_friction_scale is True
    assert cfg.priv_info.include_gravity_direction is False
    env = cast(Any, SharpaInhandRotationGraspEnv)(cfg, num_envs=4, backend_type="mujoco")

    assert [provider.__class__.__name__ for provider in initialized_providers] == [
        "SharpaInhandGraspDRProvider"
    ]
    assert len(env._saved_grasping_states) == env._num_scales


def test_sharpa_grasp_target_saves_cache_then_raises_run_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from unilab.base.run_control import RunComplete
    from unilab.tasks.manipulation.sharpa_inhand.grasp_gen import (
        SharpaInhandRotationGraspCfg,
        SharpaInhandRotationGraspEnv,
    )

    cache_path = tmp_path / "sharpa.npy"
    env = cast(Any, object.__new__(SharpaInhandRotationGraspEnv))
    env._cfg = SharpaInhandRotationGraspCfg(
        grasp_collection_target=1,
        grasp_cache_path=str(cache_path),
        grasp_auto_save=True,
    )
    env._grasp_target_per_scale = 1
    env._num_scales = 1
    env.scale_values = np.asarray([0.8], dtype=np.float64)
    env.scale_ids = np.asarray([0], dtype=np.int32)
    env._saved_grasping_states = [[]]
    env._grasp_cache_saved = False
    env._grasp_target_reached_notified = False
    env._last_grasp_progress_counts = ()
    env._last_grasp_progress_step = -1
    env._state = SimpleNamespace(
        truncated=np.asarray([True]),
        terminated=np.asarray([False]),
        info={"steps": np.asarray([1], dtype=np.uint32), "log": {}},
    )
    env.get_hand_dof_pos = lambda: np.arange(22, dtype=np.float64).reshape(1, 22)
    env.get_object_pos = lambda: np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)
    env.get_object_quat = lambda: np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)

    save_calls: list[Path] = []
    real_save = np.save

    def save_once(path: str | Path, values: np.ndarray) -> None:
        save_calls.append(Path(path))
        real_save(path, values)

    monkeypatch.setattr(
        "unilab.tasks.manipulation.sharpa_inhand.grasp_gen.np.save",
        save_once,
    )

    with caplog.at_level(
        logging.INFO,
        logger="unilab.tasks.manipulation.sharpa_inhand.grasp_gen",
    ):
        with pytest.raises(RunComplete) as caught:
            env._collect_successful_grasps(np.asarray([0], dtype=np.int32))

    saved_path = tmp_path / "sharpa_0.8.npy"
    saved = np.load(saved_path)
    expected = np.concatenate(
        [
            np.arange(22, dtype=np.float32),
            np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ]
    )[None, :]
    np.testing.assert_array_equal(saved, expected)
    assert saved.dtype == np.float32
    assert save_calls == [saved_path]
    assert "grasp progress total=1/1, per_scale=[scale=0.8:1]" in caplog.text
    assert "target reached (saved=1, configured_target=1)" in caplog.text
    assert env.state.info["log"]["grasp_cache/saved"] == 1.0
    assert env.state.info["log"]["grasp_cache/num_states"] == 1.0
    assert env.state.info["log"]["grasp/target_reached"] == 1.0
    assert dict(caught.value.summary) == {
        "collected_grasps": 1,
        "grasp_collection_target": 1,
        "grasp_collection_counts_by_scale": (1,),
    }

    env._collect_successful_grasps(np.asarray([0], dtype=np.int32))
    env._stop_collection()
    assert save_calls == [saved_path]


def test_sharpa_run_complete_reports_effective_collection_target() -> None:
    from unilab.base.run_control import RunComplete
    from unilab.tasks.manipulation.sharpa_inhand.grasp_gen import (
        SharpaInhandRotationGraspCfg,
        SharpaInhandRotationGraspEnv,
    )

    env = cast(Any, object.__new__(SharpaInhandRotationGraspEnv))
    env._cfg = SharpaInhandRotationGraspCfg(grasp_collection_target=0)
    env._grasp_target_per_scale = 1
    env._saved_grasping_states = [[np.zeros((1, 29), dtype=np.float32)]]
    env._grasp_target_reached_notified = False
    env._state = None

    with pytest.raises(RunComplete) as caught:
        env._stop_collection()

    assert dict(caught.value.summary) == {
        "collected_grasps": 1,
        "grasp_collection_target": 1,
        "grasp_collection_counts_by_scale": (1,),
    }


# ---------------------------------------------------------------------------
# Fast env/backend smoke tests
# ---------------------------------------------------------------------------

# Environments that don't need special config overrides
_STANDARD_ENVS = [
    "G1WalkFlat",
    "G1WalkRough",
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
    elif env_name == "G1WalkRough":
        env_cfg_override = _g1_manager_override("g1_walk_rough")
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
        actions = np.zeros((2, act_space.shape[0]))
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
        target_before = action.target.copy()
        actions = np.full((2, 16), 0.25, dtype=np.float32)
        manager_state = env.step(actions)
        expected_target = np.clip(
            target_before + 0.25 / 24.0,
            action.ctrl_lower,
            action.ctrl_upper,
        )
        np.testing.assert_allclose(action.target, expected_target, rtol=0.0, atol=1.0e-7)

        observation = env.observation_manager.get_term_cfg("policy", "rotation").func
        assert isinstance(observation, AllegroRotationObservation)
        current_frame = observation(env)
        np.testing.assert_allclose(
            manager_state.obs["obs"][:, -35:], current_frame, rtol=0.0, atol=1.0e-7
        )
        assert np.isfinite(manager_state.obs["obs"]).all()
        assert np.isfinite(manager_state.reward).all()
        assert manager_state.terminated.dtype == np.bool_
        assert action.action_dim == 16
        assert env.obs_groups_spec == {"obs": 105}
    finally:
        env.close()


@pytest.mark.parametrize("sim_backend", ["mujoco", "motrix"])
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
        target = action.target.copy()

        state = env.step(np.ones((2, 16), dtype=np.float32))
        np.testing.assert_array_equal(action.target, target)
        np.testing.assert_array_equal(state.reward, np.zeros(2, dtype=state.reward.dtype))
        assert initial.obs["obs"].shape == (2, 105)
        assert state.obs["obs"].shape == (2, 105)
        assert quality.last_counter == env.common_step_counter
    finally:
        env.close()


_MOTION_CORE_RUNTIME_CASES = (
    pytest.param("ppo", "g1_motion_tracking", "G1MotionTracking", 160, 286, 29, False),
    pytest.param(
        "ppo",
        "g1_motion_tracking_deploy",
        "G1MotionTrackingDeploy",
        154,
        286,
        29,
        False,
    ),
    pytest.param("ppo", "g1_23dof_motion_tracking", "G1MotionTracking23Dof", 130, 256, 23, False),
    pytest.param(
        "ppo",
        "g1_23dof_motion_tracking_deploy",
        "G1MotionTracking23DofDeploy",
        124,
        256,
        23,
        False,
    ),
    pytest.param("appo", "g1_motion_tracking", "G1MotionTracking", 160, 286, 29, False),
    pytest.param("appo", "g1_23dof_motion_tracking", "G1MotionTracking23Dof", 130, 256, 23, False),
    pytest.param(
        "sac",
        "g1_motion_tracking",
        "G1MotionTrackingSAC",
        160,
        289,
        29,
        True,
    ),
    pytest.param(
        "sac",
        "g1_23dof_motion_tracking",
        "G1MotionTrackingSAC23Dof",
        130,
        259,
        23,
        True,
    ),
)


def test_g1_motion_core_registrations_are_manager_only() -> None:
    from unilab.base import registry

    ensure_registries()
    metadata = registry.list_registered_envs()
    for task_name in (
        "G1MotionTracking",
        "G1MotionTrackingDeploy",
        "G1MotionTracking23Dof",
        "G1MotionTracking23DofDeploy",
        "G1MotionTrackingSAC23Dof",
    ):
        assert metadata[task_name] == {
            "config_factory": "ManagerBasedRlEnvCfg",
            "available_backends": ["mujoco", "motrix"],
        }
    # mjwarp is registered for G1MotionTrackingSAC only (benchmark scope, #1292).
    assert metadata["G1MotionTrackingSAC"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mujoco", "motrix", "mjwarp"],
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
        env.reset_buf[:] = [True, False]

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

        state = env.step(np.zeros((2, 29), dtype=np.float32))

        np.testing.assert_array_equal(state.terminated, [False, False])
        np.testing.assert_array_equal(state.truncated, [True, True])
        np.testing.assert_array_equal(command.time_steps, command.sampler.current_clip_end_frames)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("config_root", "task", "identity", "actor_dim", "critic_dim", "action_dim", "truncate"),
    _MOTION_CORE_RUNTIME_CASES,
)
@pytest.mark.parametrize("sim_backend", ["mujoco", "motrix"])
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

        state = env.step(np.zeros((2, action_dim), dtype=np.float32))
        assert state.reward.shape == (2,)
        assert state.terminated.shape == (2,)
        assert state.truncated.shape == (2,)
        assert np.isfinite(state.reward).all()
        assert all(np.isfinite(values).all() for values in state.obs.values())
    finally:
        env.close()
