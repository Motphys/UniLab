from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.envs import ManagerBasedRlEnvCfg
from unilab.envs.mdp.actions import JointPositionAction

# CPU-bound on the single-core CI runner; kept in the slow lane (make test-slow).
pytestmark = pytest.mark.slow

_ROOT = Path(__file__).parents[2]

_PROFILE_IDENTITIES = (
    "G1BoxTracking",
    "G1BoxTracking23Dof",
    "G1ClimbTracking",
    "G1ClimbTracking23Dof",
    "G1FlipTracking",
    "G1FlipTracking23Dof",
    "G1FlipTrackingSAC",
    "G1FlipTrackingSAC23Dof",
    "G1WallFlipTracking",
    "G1WallFlipTracking23Dof",
    "G1WallFlipTrackingSAC",
    "G1WallFlipTrackingSAC23Dof",
    "G1WBTObs",
    "G1WBTObs23Dof",
    "X2WallFlipTracking",
)

_PPO_PROFILES = (
    (
        "g1_box_tracking",
        "G1BoxTracking",
        "scene_flat_with_largebox.xml",
        "sub3_largebox_003_boxconverted.npz",
    ),
    (
        "g1_23dof_box_tracking",
        "G1BoxTracking23Dof",
        "scene_flat_23dof_with_largebox.xml",
        "sub3_largebox_003_boxconverted_23dof.npz",
    ),
    (
        "g1_climb_tracking",
        "G1ClimbTracking",
        "scene_climb_20_z_scale_1.xml",
        "climb_20_z_scale_1.0.npz",
    ),
    (
        "g1_23dof_climb_tracking",
        "G1ClimbTracking23Dof",
        "scene_climb_20_z_scale_1_23dof.xml",
        "climb_20_z_scale_1.0_23dof.npz",
    ),
    ("g1_flip_tracking", "G1FlipTracking", "scene_flat.xml", "flip_360_001__A304.npz"),
    (
        "g1_23dof_flip_tracking",
        "G1FlipTracking23Dof",
        "scene_flat_23dof.xml",
        "flip_360_001__A304_23dof.npz",
    ),
    (
        "g1_wall_flip_tracking",
        "G1WallFlipTracking",
        "scene_flat_with_wall.xml",
        "flip_from_wall_104__A304.npz",
    ),
    (
        "g1_23dof_wall_flip_tracking",
        "G1WallFlipTracking23Dof",
        "scene_flat_23dof_with_wall.xml",
        "flip_from_wall_104__A304_23dof.npz",
    ),
    (
        "x2_wall_flip_tracking",
        "X2WallFlipTracking",
        "scene_flat_with_wall.xml",
        "tictacflip_6-3_g1format.npz",
    ),
)

_APPO_PROFILES = tuple(
    profile
    for profile in _PPO_PROFILES
    if "box" not in profile[0] and not profile[0].startswith("x2")
)

_SAC_PROFILES = (
    ("g1_flip_tracking", "G1FlipTrackingSAC", "scene_flat.xml", "flip_360_001__A304.npz"),
    (
        "g1_23dof_flip_tracking",
        "G1FlipTrackingSAC23Dof",
        "scene_flat_23dof.xml",
        "flip_360_001__A304_23dof.npz",
    ),
    (
        "g1_wall_flip_tracking",
        "G1WallFlipTrackingSAC",
        "scene_flat_with_wall.xml",
        "flip_from_wall_104__A304.npz",
    ),
    (
        "g1_23dof_wall_flip_tracking",
        "G1WallFlipTrackingSAC23Dof",
        "scene_flat_23dof_with_wall.xml",
        "flip_from_wall_104__A304_23dof.npz",
    ),
    ("g1_wbt_obs", "G1WBTObs", "scene_flat.xml", "dance1_subject2_part.npz"),
    (
        "g1_23dof_wbt_obs",
        "G1WBTObs23Dof",
        "scene_flat_23dof.xml",
        "dance1_subject2_part_23dof.npz",
    ),
)

_OWNER_CASES = (
    tuple(
        ("ppo", task, backend, identity, scene, motion)
        for task, identity, scene, motion in _PPO_PROFILES
        for backend in ("mujoco", "motrix")
    )
    + tuple(
        ("appo", task, backend, identity, scene, motion)
        for task, identity, scene, motion in _APPO_PROFILES
        for backend in ("mujoco", "motrix")
    )
    + tuple(
        ("sac", task, "mujoco", identity, scene, motion)
        for task, identity, scene, motion in _SAC_PROFILES
    )
)

_LEGACY_G1_ACTION_SCALE = (
    (r".*_(hip_pitch|hip_yaw)_joint", 0.5475464629911068),
    (r".*_(hip_roll|knee)_joint", 0.35066146637882434),
    (r".*_ankle_(pitch|roll)_joint", 0.43857731392336724),
    (r"waist_yaw_joint", 0.5475464629911068),
    (r"waist_(roll|pitch)_joint", 0.43857731392336724),
    (r".*_(shoulder_(pitch|roll|yaw)|elbow|wrist_roll)_joint", 0.43857731392336724),
    (r".*_wrist_(pitch|yaw)_joint", 0.07450087032950714),
)

_LEGACY_SCALAR_ACTION_SCALE = {
    ("ppo", "g1_flip_tracking", "motrix"): 0.25,
    ("ppo", "g1_23dof_flip_tracking", "motrix"): 0.25,
    ("appo", "g1_wall_flip_tracking", "motrix"): 0.25,
    ("appo", "g1_23dof_wall_flip_tracking", "motrix"): 0.25,
}


def _compose_owner(config_root: str, task: str, backend: str) -> Any:
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(_ROOT / "src" / "unilab" / "conf" / config_root), version_base="1.3"
    ):
        overrides = [f"task={task}/{backend}"]
        return compose("config", overrides=overrides)


def _materialize_profile(
    config_root: str,
    task: str,
    backend: str,
    identity: str,
) -> tuple[Any, ManagerBasedRlEnvCfg, dict[str, Any]]:
    registry.ensure_registries()
    owner = _compose_owner(config_root, task, backend)
    cfg = registry.materialize_env_config(identity)
    assert isinstance(cfg, ManagerBasedRlEnvCfg)
    override = BackendAdapter(
        owner,
        root_dir=_ROOT,
        algo_name=config_root,
    ).build_task_env_cfg_override()
    apply_cfg_overrides(
        cfg,
        override,
    )
    return owner, cfg, override


def _resolved_action_scale(cfg: ManagerBasedRlEnvCfg) -> tuple[list[str], np.ndarray]:
    joint_names = list(cfg.scene.entities["robot"].joint_names or ())
    entity = SimpleNamespace(
        data=SimpleNamespace(
            default_joint_pos=np.zeros((1, len(joint_names)), dtype=np.float32),
        ),
        find_joints_by_actuator_names=lambda actuator_names: (
            list(range(len(joint_names))),
            joint_names,
        ),
    )
    env = SimpleNamespace(num_envs=1, scene={"robot": entity})
    action = JointPositionAction(cfg.actions["joint_pos"], cast(Any, env))
    scale = action.scale
    if np.isscalar(scale):
        scale = np.full(len(joint_names), scale, dtype=np.float32)
    else:
        scale = np.asarray(scale[0])
    return joint_names, scale


def _legacy_g1_action_scale(joint_names: list[str]) -> np.ndarray:
    import re

    expected = []
    for joint_name in joint_names:
        matches = [
            value for pattern, value in _LEGACY_G1_ACTION_SCALE if re.fullmatch(pattern, joint_name)
        ]
        assert len(matches) == 1, (joint_name, matches)
        expected.append(matches[0])
    return np.asarray(expected, dtype=np.float32)


@pytest.mark.parametrize(
    ("config_root", "task", "backend", "identity", "scene_file", "motion_file"),
    _OWNER_CASES,
)
def test_motion_profile_owner_composes_to_manager_runtime(
    config_root: str,
    task: str,
    backend: str,
    identity: str,
    scene_file: str,
    motion_file: str,
) -> None:
    registry.ensure_registries()
    owner, cfg, _ = _materialize_profile(config_root, task, backend, identity)

    assert owner.training.task_name == identity
    assert owner.training.sim_backend == backend
    assert cfg.scene.model_file.endswith(scene_file)
    assert str(cfg.commands["motion"].params.motion_file).endswith(motion_file)
    assert list(cfg.actions) == ["joint_pos"]
    assert cfg.policy_observation_group == "actor"
    assert cfg.critic_observation_group == "critic"


@pytest.mark.parametrize(
    ("config_root", "task", "backend", "identity", "scene_file", "motion_file"),
    _OWNER_CASES,
)
def test_motion_profile_action_scale_matches_legacy_runtime(
    config_root: str,
    task: str,
    backend: str,
    identity: str,
    scene_file: str,
    motion_file: str,
) -> None:
    del scene_file, motion_file
    _, cfg, _ = _materialize_profile(config_root, task, backend, identity)
    joint_names, actual = _resolved_action_scale(cfg)

    scalar = _LEGACY_SCALAR_ACTION_SCALE.get((config_root, task, backend))
    if "box_tracking" in task or task == "x2_wall_flip_tracking":
        scalar = 0.25
    elif "wbt_obs" in task:
        scalar = 2.0
    expected = (
        np.full(len(joint_names), scalar, dtype=np.float32)
        if scalar is not None
        else _legacy_g1_action_scale(joint_names)
    )

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    (
        "task",
        "backend",
        "adaptive_kl_factor",
        "adaptive_lr_factor",
        "enable_compile",
        "steps_per_env",
        "replay_queue_size",
    ),
    (
        ("g1_climb_tracking", "mujoco", 1.2, 1.1, False, 24, None),
        ("g1_climb_tracking", "motrix", 1.2, 1.1, False, 24, None),
        ("g1_23dof_climb_tracking", "mujoco", 1.2, 1.1, False, 24, None),
        ("g1_23dof_climb_tracking", "motrix", 1.2, 1.1, False, 24, None),
        ("g1_23dof_flip_tracking", "mujoco", 1.2, 1.1, False, 24, None),
        ("g1_23dof_flip_tracking", "motrix", 2.0, 1.5, True, 24, None),
        ("g1_wall_flip_tracking", "motrix", 2.0, 1.5, True, 24, None),
        ("g1_23dof_wall_flip_tracking", "mujoco", 2.0, 1.5, True, 20, 5),
        ("g1_23dof_wall_flip_tracking", "motrix", 2.0, 1.5, True, 24, None),
    ),
)
def test_appo_profiles_preserve_training_owner_contract(
    task: str,
    backend: str,
    adaptive_kl_factor: float,
    adaptive_lr_factor: float,
    enable_compile: bool,
    steps_per_env: int,
    replay_queue_size: int | None,
) -> None:
    owner = _compose_owner("appo", task, backend)

    assert owner.algo.algorithm.adaptive_kl_factor == pytest.approx(adaptive_kl_factor)
    assert owner.algo.algorithm.adaptive_lr_factor == pytest.approx(adaptive_lr_factor)
    assert owner.algo.algorithm.enable_compile is enable_compile
    assert owner.algo.steps_per_env == steps_per_env
    assert owner.training.replay_queue_size == replay_queue_size


@pytest.mark.parametrize("task", ("g1_flip_tracking", "g1_23dof_flip_tracking"))
def test_ppo_motrix_flip_profiles_keep_actor_normalization_disabled(task: str) -> None:
    owner = _compose_owner("ppo", task, "motrix")

    assert owner.algo.empirical_normalization is False
    assert owner.algo.obs_groups.actor == ["actor"]
    assert owner.algo.obs_groups.critic == ["critic"]


@pytest.mark.parametrize(
    "task",
    (
        "g1_23dof_box_tracking",
        "g1_23dof_climb_tracking",
        "g1_23dof_flip_tracking",
        "g1_23dof_wall_flip_tracking",
        "x2_wall_flip_tracking",
    ),
)
def test_ppo_profiles_without_legacy_play_overrides_stay_disabled(task: str) -> None:
    owner = _compose_owner("ppo", task, "mujoco")

    assert owner.play_profile.enabled is False
    assert owner.play_profile.env is None


@pytest.mark.parametrize("task", ("g1_box_tracking", "g1_23dof_box_tracking"))
def test_box_motrix_drops_unconsumed_algorithm_noise_config(task: str) -> None:
    owner = _compose_owner("ppo", task, "motrix")

    assert "noise_config" not in owner.algo


def test_all_motion_profiles_have_one_manager_factory_and_both_backends() -> None:
    registry.ensure_registries()
    metadata = registry.list_registered_envs()

    for identity in _PROFILE_IDENTITIES:
        assert metadata[identity] == {
            "config_factory": "ManagerBasedRlEnvCfg",
            "available_backends": ["mujoco", "motrix"],
        }


def test_box_wall_wbt_and_x2_profiles_keep_only_owner_differences() -> None:
    from unilab.tasks.motion_tracking.g1.manager_terms import BoxMotionCommandCfg

    _, box, _ = _materialize_profile("ppo", "g1_box_tracking", "mujoco", "G1BoxTracking")
    _, flip, _ = _materialize_profile("sac", "g1_flip_tracking", "mujoco", "G1FlipTrackingSAC")
    _, wall, _ = _materialize_profile(
        "sac",
        "g1_wall_flip_tracking",
        "mujoco",
        "G1WallFlipTrackingSAC",
    )
    _, wbt, _ = _materialize_profile("sac", "g1_wbt_obs", "mujoco", "G1WBTObs")
    _, x2, _ = _materialize_profile("ppo", "x2_wall_flip_tracking", "mujoco", "X2WallFlipTracking")

    assert set(box.scene.entities) == {"robot", "object"}
    assert isinstance(box.commands["motion"], BoxMotionCommandCfg)
    assert box.observations["actor"].terms["motion_anchor_pos_b"] is None
    assert box.observations["actor"].terms["base_ang_vel"].params == {"sensor_name": "pelvis_gyro"}
    assert box.observations["critic"].terms["base_ang_vel"].params == {"sensor_name": "pelvis_gyro"}
    assert box.observations["critic"].terms["object_state"] is not None

    assert flip.commands["motion"].params.sampling_mode == "mixed"
    assert flip.commands["motion"].params.sampling_start_ratio == pytest.approx(0.1)
    assert wall.commands["motion"].params.sampling_mode == "uniform"
    assert wall.terminations["undesired_contacts"] is None
    assert wall.terminations["anchor_pos"].params["threshold"] == pytest.approx(1.0e9)

    actor_terms = wbt.observations["actor"].terms
    critic_terms = wbt.observations["critic"].terms
    assert actor_terms["motion_anchor_pos_b"] is None
    assert actor_terms["base_lin_vel"] is None
    assert [
        actor_terms[name].history_length
        for name in ("base_ang_vel", "joint_pos", "joint_vel", "actions")
    ] == [5, 5, 5, 5]
    assert actor_terms["joint_pos"].func.__name__ == "motion_joint_pos_rel_biased"
    assert critic_terms["joint_pos"].func.__name__ == "motion_joint_pos_rel"
    assert wbt.actions["joint_pos"].simulate_action_latency is True
    assert list(wbt.events) == [
        "base_mass",
        "base_com",
        "pd_gains",
        "foot_friction",
        "encoder_bias",
        "push_robot",
    ]

    assert len(x2.scene.entities["robot"].joint_names) == 29
    assert x2.scene.entities["robot"].geom_names is None
    assert x2.scene.visual_model_file is not None
    assert x2.observations["actor"].terms["motion_anchor_pos_b"] is None
    assert x2.observations["actor"].terms["base_lin_vel"] is None


def test_manager_factory_selects_robot_from_multiple_floating_entities() -> None:
    from unilab.base.entity import EntityCfg
    from unilab.base.scene import SceneCfg
    from unilab.envs.manager_based_rl_env import _resolve_backend_entity_contract

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            model_file="scene.xml",
            entities={
                "robot": EntityCfg(root_body_name="pelvis", body_names=("pelvis",)),
                "object": EntityCfg(root_body_name="largebox"),
            },
        )
    )

    assert _resolve_backend_entity_contract(cfg) == ("pelvis", True)

    cfg.scene.entities = {
        "first": EntityCfg(root_body_name="first"),
        "second": EntityCfg(root_body_name="second"),
    }
    with pytest.raises(ValueError, match="conventional 'robot' root entity"):
        _resolve_backend_entity_contract(cfg)


def test_x2_factory_resolves_meshes_only_before_manager_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unilab.tasks.motion_tracking import x2

    calls: list[tuple[str, str]] = []
    sentinel = object()
    monkeypatch.setattr(
        x2,
        "resolve_robot_asset_dir",
        lambda path, marker: calls.append((path, marker)),
    )
    monkeypatch.setattr(x2, "make_manager_based_rl_env", lambda *args, **kwargs: sentinel)

    result = x2.make_x2_wall_flip_env(
        ManagerBasedRlEnvCfg(),
        num_envs=3,
        backend_type="motrix",
    )

    assert result is sentinel
    assert calls == [("robots/x2/meshes", "pelvis.STL")]


def test_joint_acc_reset_updates_selected_rows_without_pairwise_indexing() -> None:
    from unilab.managers import ManagerTermBaseCfg, SceneEntityCfg
    from unilab.tasks.motion_tracking.g1.manager_terms import joint_acc_l2

    velocity = np.arange(12, dtype=np.float32).reshape(4, 3)
    entity = SimpleNamespace(num_joints=3, data=SimpleNamespace(joint_vel=velocity))
    env = SimpleNamespace(num_envs=4, step_dt=0.02, scene={"robot": entity})
    cfg = ManagerTermBaseCfg(
        func=joint_acc_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_ids=[0, 2])},
    )
    term = joint_acc_l2(cfg, cast(Any, env))
    velocity[[1, 3]] += 10.0

    term.reset(np.array([1, 3], dtype=np.int32))

    np.testing.assert_array_equal(term._previous[[1, 3]], velocity[np.ix_([1, 3], [0, 2])])
    np.testing.assert_array_equal(term._previous[[0, 2]], np.array([[0.0, 2.0], [6.0, 8.0]]))


@pytest.mark.parametrize(
    ("config_root", "task", "identity", "backend", "actor_dim", "critic_dim", "action_dim"),
    (
        ("ppo", "g1_box_tracking", "G1BoxTracking", "mujoco", 154, 298, 29),
        ("ppo", "g1_box_tracking", "G1BoxTracking", "motrix", 154, 298, 29),
        ("ppo", "x2_wall_flip_tracking", "X2WallFlipTracking", "mujoco", 154, 430, 29),
        ("ppo", "x2_wall_flip_tracking", "X2WallFlipTracking", "motrix", 154, 430, 29),
        ("sac", "g1_wbt_obs", "G1WBTObs", "mujoco", 514, 289, 29),
        ("sac", "g1_23dof_wbt_obs", "G1WBTObs23Dof", "mujoco", 412, 259, 23),
    ),
)
def test_representative_motion_profiles_reset_and_step(
    config_root: str,
    task: str,
    identity: str,
    backend: str,
    actor_dim: int,
    critic_dim: int,
    action_dim: int,
) -> None:
    if backend == "mujoco":
        pytest.importorskip("mujoco")
        try:
            from mujoco_uni.batch_env import BatchEnvPool as _  # noqa: F401
        except Exception:
            pytest.skip("mujoco_uni.batch_env not available")
    else:
        pytest.importorskip("motrixsim")

    _, _, override = _materialize_profile(config_root, task, backend, identity)
    env = registry.make(identity, num_envs=2, sim_backend=backend, env_cfg_override=override)
    try:
        initial = env.init_state()
        assert env.obs_groups_spec == {"obs": actor_dim, "critic": critic_dim}
        assert initial.obs["obs"].shape == (2, actor_dim)
        assert initial.obs["critic"].shape == (2, critic_dim)

        state = env.step(np.zeros((2, action_dim), dtype=np.float32))
        assert state.reward.shape == (2,)
        assert np.isfinite(state.reward).all()
        assert all(np.isfinite(value).all() for value in state.obs.values())
    finally:
        env.close()
