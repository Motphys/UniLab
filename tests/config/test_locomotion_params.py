"""Tests for structured configs and Hydra YAML loading."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# CPU-bound on the single-core CI runner; kept in the slow lane (make test-slow).
pytestmark = pytest.mark.slow

CONF_DIR = Path(__file__).parent.parent.parent / "src" / "unilab" / "conf"

G1_BEYONDMIMIC_ACTION_SCALE = [
    0.5475464629911068,
    0.35066146637882434,
    0.5475464629911068,
    0.35066146637882434,
    0.43857731392336724,
    0.43857731392336724,
    0.5475464629911068,
    0.35066146637882434,
    0.5475464629911068,
    0.35066146637882434,
    0.43857731392336724,
    0.43857731392336724,
    0.5475464629911068,
    0.43857731392336724,
    0.43857731392336724,
    0.43857731392336724,
    0.43857731392336724,
    0.43857731392336724,
    0.43857731392336724,
    0.43857731392336724,
    0.07450087032950714,
    0.07450087032950714,
    0.43857731392336724,
    0.43857731392336724,
    0.43857731392336724,
    0.43857731392336724,
    0.43857731392336724,
    0.07450087032950714,
    0.07450087032950714,
]
G1_23DOF_BEYONDMIMIC_ACTION_SCALE = G1_BEYONDMIMIC_ACTION_SCALE[:13] + [0.43857731392336724] * 10
X2_ACTION_SCALE = [0.25] * 29


# ---------------------------------------------------------------------------
# structured_configs dataclass defaults
# ---------------------------------------------------------------------------


def test_sac_config_defaults():
    from unilab.structured_configs import SACAlgoParams, SACConfig

    cfg = SACConfig()
    assert cfg.algo == "sac"
    assert cfg.num_envs == 4096
    assert cfg.batch_size == 8192
    assert cfg.obs_normalization is False
    assert isinstance(cfg.algo_params, SACAlgoParams)
    assert cfg.algo_params.alpha_init == 0.01
    assert cfg.algo_params.use_compile is True


def test_td3_config_defaults():
    from unilab.structured_configs import TD3Config

    cfg = TD3Config()
    assert cfg.algo == "td3"
    assert cfg.num_envs == 4096
    assert cfg.use_layer_norm is False
    assert cfg.algo_params.weight_decay == 0.1


def test_flashsac_config_defaults():
    from unilab.structured_configs import FlashSACAlgoParams, FlashSACConfig

    cfg = FlashSACConfig()
    assert cfg.algo == "flashsac"
    assert cfg.num_envs == 1024
    assert cfg.batch_size == 2048
    assert cfg.learning_starts == 98
    assert cfg.gamma == pytest.approx(0.97)
    assert cfg.obs_normalization is False
    assert isinstance(cfg.algo_params, FlashSACAlgoParams)
    assert cfg.algo_params.normalize_reward is True
    assert cfg.algo_params.amp_dtype == "auto"
    assert cfg.algo_params.use_compile is True


def test_ppo_config_defaults():
    from unilab.structured_configs import PPOConfig

    cfg = PPOConfig()
    assert cfg.algo == "ppo"
    assert cfg.max_iterations == 101
    assert cfg.algorithm.clip_param == 0.2
    assert cfg.algorithm.class_name == "uni_rl.algos.rsl_rl_ppo:FinalObservationAwarePPO"
    assert cfg.algorithm.enable_compile is True
    assert cfg.policy.class_name == "ActorCritic"


def test_appo_config_defaults():
    from unilab.structured_configs import APPOConfig

    cfg = APPOConfig()
    assert cfg.algo == "appo"
    assert cfg.num_envs == 2048
    assert cfg.actor.class_name == "rsl_rl.models.MLPModel"
    assert cfg.algorithm.enable_compile is True


def test_base_config_to_dict():
    from unilab.structured_configs import SACConfig

    cfg = SACConfig()
    d = cfg.to_dict()
    assert isinstance(d, dict)
    assert d["algo"] == "sac"
    assert "algo_params" in d
    assert isinstance(d["algo_params"], dict)


# ---------------------------------------------------------------------------
# Hydra YAML loading — offpolicy
# ---------------------------------------------------------------------------


def test_offpolicy_sac_defaults():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "sac"), version_base="1.3"):
        cfg = compose("config")
    assert cfg.algo.algo == "sac"
    assert cfg.algo.num_envs == 2048


def test_offpolicy_sac_g1_task_overrides():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "sac"), version_base="1.3"):
        cfg = compose("config", overrides=["task=g1_walk_flat/mujoco"])
    assert cfg.algo.num_envs == 2048
    assert cfg.algo.max_iterations == 5000
    assert cfg.algo.algo_params.target_entropy_ratio == pytest.approx(0.0)
    assert cfg.algo.algo_params.use_compile is True
    assert cfg.training.task_name == "G1WalkFlat"

    assert cfg.env.actions.joint_pos.scale == pytest.approx(1.0)
    assert cfg.env.observations.policy.terms.gait_phase.params.init_mode == "offset_phase"
    assert cfg.env.events.reset_root_state_uniform.params.velocity_range.x == [-0.5, 0.5]


def test_offpolicy_td3_defaults():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "td3"), version_base="1.3"):
        cfg = compose("config")
    assert cfg.algo.algo == "td3"
    assert cfg.algo.use_layer_norm is False
    assert cfg.algo.algo_params.weight_decay == pytest.approx(0.1)
    assert cfg.algo.tau == pytest.approx(0.1)
    assert cfg.algo.algo_params.policy_noise == pytest.approx(0.2)
    assert cfg.algo.algo_params.noise_clip == pytest.approx(0.5)
    assert cfg.algo.algo_params.log_std_min == pytest.approx(-1.6)


def test_offpolicy_td3_g1_task_overrides():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "td3"), version_base="1.3"):
        cfg = compose("config", overrides=["task=g1_walk_flat/mujoco"])
    assert cfg.training.task_name == "G1WalkFlat"
    assert cfg.algo.max_iterations == 100000
    assert cfg.env.actions.joint_pos.scale == pytest.approx(1.0)


def test_offpolicy_flashsac_g1_task_overrides():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "flashsac"), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=["task=g1_walk_flat/mujoco"],
        )
    assert cfg.algo.algo == "flashsac"
    assert cfg.training.task_name == "G1WalkFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.algo.algo_params.actor_num_blocks == 2
    assert cfg.algo.algo_params.normalize_reward is True
    assert cfg.algo.algo_params.amp_dtype == "auto"


def test_offpolicy_flashsac_go2_task_overrides():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "flashsac"), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=["task=go2_joystick_flat/mujoco"],
        )
    assert cfg.algo.algo == "flashsac"
    assert cfg.training.task_name == "Go2JoystickFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.algo.num_envs == 1024
    assert cfg.algo.max_iterations == 4000
    assert cfg.algo.tau == pytest.approx(0.05)
    assert cfg.algo.replay_buffer_n == 4096
    assert cfg.algo.updates_per_step == 2
    assert cfg.reward.swing_feet_z.weight == pytest.approx(4.0)
    assert cfg.env.actions.joint_pos.scale == pytest.approx(0.4)


def test_offpolicy_g1_rough_terrain_task_overrides():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "sac"), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=["task=g1_walk_rough/mujoco"],
        )
    assert cfg.algo.algo == "sac"
    assert cfg.training.task_name == "G1WalkRough"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.env.scene.model_file.endswith("scene_rough.xml")


def test_g1_task_owner_yamls_preserve_legacy_and_walk_observation_profiles():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    def uses_walk_profile(config_group: str, overrides: list[str]) -> bool:
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(CONF_DIR / config_group), version_base="1.3"):
            cfg = compose("config", overrides=overrides)
        gyro_scale = cfg.env.observations.policy.terms.base_ang_vel.get("scale")
        if gyro_scale is None:
            return False
        assert gyro_scale == pytest.approx(0.25)
        assert cfg.env.observations.policy.terms.joint_vel.scale == pytest.approx(0.05)
        assert cfg.env.observations.critic.terms.base_lin_vel.scale == pytest.approx(2.0)
        return True

    assert uses_walk_profile("ppo", ["task=g1_walk_flat/mujoco"]) is False
    assert uses_walk_profile("appo", ["task=g1_walk_flat/mujoco"]) is False
    assert uses_walk_profile("sac", ["task=g1_walk_flat/mujoco"]) is True
    assert uses_walk_profile("sac", ["task=g1_walk_flat/motrix"]) is True
    assert uses_walk_profile("sac", ["task=g1_walk_rough/mujoco"]) is True
    assert uses_walk_profile("td3", ["task=g1_walk_flat/mujoco"]) is True
    assert uses_walk_profile("flashsac", ["task=g1_walk_flat/mujoco"]) is True


# ---------------------------------------------------------------------------
# Hydra YAML loading — appo
# ---------------------------------------------------------------------------


def test_appo_defaults():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "appo"), version_base="1.3"):
        cfg = compose("config")
    assert cfg.algo.algo == "appo"
    assert cfg.algo.max_iterations == 150
    assert cfg.algo.algorithm.enable_compile is False


def test_appo_g1_task_overrides():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "appo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=g1_walk_flat/mujoco"])
    assert cfg.algo.max_iterations == 500
    assert cfg.algo.save_interval == 100
    assert cfg.training.task_name == "G1WalkFlat"
    assert "obs_profile" not in cfg.env
    assert "curriculum" not in cfg.env


# ---------------------------------------------------------------------------
# Hydra YAML loading — ppo
# ---------------------------------------------------------------------------


def test_ppo_go1_max_iterations():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=go1_joystick_flat/mujoco"])
    assert cfg.algo.max_iterations == 151
    assert "actor" in cfg.algo.obs_groups
    assert cfg.algo.algorithm.enable_compile is False


def test_ppo_compile_overrides():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=[
                "task=go1_joystick_flat/mujoco",
                "algo.algorithm.enable_compile=false",
            ],
        )
    assert cfg.algo.algorithm.enable_compile is False


def test_ppo_g1_num_envs():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=g1_walk_flat/mujoco"])
    assert cfg.algo.num_envs == 2048
    assert cfg.algo.max_iterations == 2200
    assert cfg.training.task_name == "G1WalkFlat"
    assert "obs_profile" not in cfg.env
    assert "curriculum" not in cfg.env


def test_ppo_go2_num_envs():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=go2_joystick_flat/mujoco"])
    assert cfg.algo.num_envs == 1024
    assert cfg.algo.max_iterations == 151


def test_ppo_go2_footstand_uses_hydra_owned_manager_task():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=go2_footstand/mujoco"])

    assert cfg.training.task_name == "Go2FootStand"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.algo.num_envs == 4096
    assert cfg.env.sim_dt == pytest.approx(0.004)
    assert cfg.env.ctrl_dt == pytest.approx(0.02)
    assert cfg.env.max_episode_seconds == pytest.approx(10.0)
    assert cfg.env.adaptive_chunk_size is False
    assert cfg.env.observations.policy.terms.frame.history_length == 15
    assert cfg.env.observations.critic.terms.frame.history_length == 15
    assert cfg.env.actions.joint_pos.action_scale == pytest.approx(0.3)
    assert cfg.env.actions.joint_pos.clip_actions == pytest.approx(1.0)
    assert cfg.env.terminations.footstand.params.energy_threshold == pytest.approx(200.0)
    assert cfg.reward.footstand.params.scales.energy == pytest.approx(-0.003)
    assert cfg.reward.footstand.params.scales.dof_acc == pytest.approx(-2.5e-7)
    assert cfg.reward.footstand.params.scales.rear_leg_symmetry == pytest.approx(-0.2)
    assert cfg.reward.footstand.params.scales.knee_clearance == pytest.approx(-0.5)
    assert cfg.reward.footstand.params.knee_height_target == pytest.approx(0.08)
    assert cfg.env.events.floor_friction is not None
    assert cfg.env.events.link_mass is not None
    assert cfg.env.events.torso_com is not None
    assert cfg.env.events.joint_armature is not None
    assert cfg.env.events.reset_joints is not None


def test_ppo_g1_motion_tracking():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=g1_motion_tracking/mujoco"])
    assert cfg.training.task_name == "G1MotionTracking"
    assert cfg.algo.max_iterations == 15000
    assert cfg.algo.algorithm.entropy_coef == pytest.approx(0.005)


def test_ppo_g1_motion_tracking_deploy():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=g1_motion_tracking_deploy/mujoco"])
    assert cfg.training.task_name == "G1MotionTrackingDeploy"
    assert cfg.algo.max_iterations == 15000
    assert cfg.algo.algorithm.entropy_coef == pytest.approx(0.005)
    assert cfg.env.sim_dt == pytest.approx(0.005)
    assert cfg.env.observations.actor.terms.base_ang_vel.params.sensor_name == "pelvis_gyro"
    assert cfg.env.actions.joint_pos.scale[".*_(hip_pitch|hip_yaw)_joint"] == pytest.approx(
        G1_BEYONDMIMIC_ACTION_SCALE[0]
    )
    assert cfg.env.actions.joint_pos.scale[".*_wrist_(pitch|yaw)_joint"] == pytest.approx(
        G1_BEYONDMIMIC_ACTION_SCALE[20]
    )


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("g1_motion_tracking_deploy", G1_BEYONDMIMIC_ACTION_SCALE),
        ("g1_23dof_motion_tracking_deploy", G1_23DOF_BEYONDMIMIC_ACTION_SCALE),
    ],
)
def test_ppo_g1_motion_tracking_deploy_action_scale_expands_in_joint_order(
    task: str,
    expected: list[float],
) -> None:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=[f"task={task}/mujoco"])

    scales = cfg.env.actions.joint_pos.scale
    resolved: list[float] = []
    for joint_name in cfg.env.scene.entities.robot.joint_names:
        matches = [
            float(value) for pattern, value in scales.items() if re.fullmatch(pattern, joint_name)
        ]
        assert len(matches) == 1, f"{joint_name} matched {len(matches)} action-scale patterns"
        resolved.append(matches[0])
    assert resolved == pytest.approx(expected)


def test_ppo_g1_box_tracking():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=g1_box_tracking/mujoco"])
    assert cfg.training.task_name == "G1BoxTracking"
    assert cfg.algo.max_iterations == 30000
    assert cfg.algo.algorithm.entropy_coef == pytest.approx(0.005)
    assert cfg.env.scene.entities.object.root_body_name == "largebox"
    assert cfg.env.commands.motion.object_entity_name == "object"
    assert cfg.reward.object_global_ref_position_error_exp.weight == pytest.approx(2.0)
    assert cfg.reward.object_global_ref_orientation_error_exp.weight == pytest.approx(2.0)
    assert cfg.reward.object_global_ref_position_error_exp.params.std == pytest.approx(0.2)
    assert cfg.reward.object_global_ref_orientation_error_exp.params.std == pytest.approx(0.3)


def test_ppo_g1_flip_tracking():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=g1_flip_tracking/mujoco"])
    assert cfg.training.task_name == "G1FlipTracking"
    assert cfg.algo.num_envs == 1024
    assert cfg.algo.max_iterations == 20000
    assert cfg.algo.empirical_normalization is True
    assert cfg.algo.obs_groups.critic == ["critic"]
    assert cfg.algo.algorithm.entropy_coef == pytest.approx(0.005)
    assert cfg.algo.algorithm.desired_kl == pytest.approx(0.01)
    assert cfg.env.commands.motion.params.sampling_mode == "start"
    assert cfg.env.commands.motion.params.truncate_on_clip_end is False
    assert cfg.env.sim_dt == pytest.approx(0.005)
    assert cfg.env.actions.joint_pos.scale[".*_(hip_pitch|hip_yaw)_joint"] == pytest.approx(
        G1_BEYONDMIMIC_ACTION_SCALE[0]
    )
    assert cfg.env.terminations.anchor_pos.params.threshold == pytest.approx(0.5)
    assert cfg.env.terminations.ee_body_pos.params.threshold == pytest.approx(0.5)
    assert cfg.env.terminations.undesired_contacts is not None
    assert cfg.reward.motion_body_pos.weight == pytest.approx(2.0)
    assert cfg.reward.motion_body_ori.weight == pytest.approx(1.5)
    assert cfg.reward.motion_ee_body_pos_z.weight == pytest.approx(2.0)
    assert cfg.reward.action_rate_l2.weight == pytest.approx(-0.005)
    assert cfg.reward.undesired_contacts.weight == pytest.approx(-0.1)


def test_ppo_g1_wall_flip_tracking():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=g1_wall_flip_tracking/mujoco"])
    assert cfg.training.task_name == "G1WallFlipTracking"
    assert cfg.algo.num_envs == 1024
    assert cfg.algo.max_iterations == 20000
    assert cfg.algo.empirical_normalization is True
    assert cfg.algo.obs_groups.critic == ["critic"]
    assert cfg.algo.algorithm.entropy_coef == pytest.approx(0.005)
    assert cfg.algo.algorithm.desired_kl == pytest.approx(0.01)
    assert cfg.env.commands.motion.params.sampling_mode == "start"
    assert cfg.env.commands.motion.params.truncate_on_clip_end is False
    assert cfg.env.sim_dt == pytest.approx(0.005)
    assert cfg.env.actions.joint_pos.scale[".*_(hip_pitch|hip_yaw)_joint"] == pytest.approx(
        G1_BEYONDMIMIC_ACTION_SCALE[0]
    )
    assert cfg.env.scene.model_file.endswith("scene_flat_with_wall.xml")
    assert cfg.reward.motion_joint_pos.weight == pytest.approx(0.5)
    assert cfg.reward.motion_joint_vel.weight == pytest.approx(0.25)
    assert cfg.reward.motion_body_pos.weight == pytest.approx(2.0)
    assert cfg.reward.motion_body_ori.weight == pytest.approx(1.5)
    assert cfg.reward.motion_ee_body_pos_z.weight == pytest.approx(2.0)
    assert cfg.reward.action_rate_l2.weight == pytest.approx(-0.005)
    assert cfg.reward.undesired_contacts.weight == pytest.approx(-0.1)


def test_ppo_x2_wall_flip_tracking():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=x2_wall_flip_tracking/mujoco"])
    assert cfg.training.task_name == "X2WallFlipTracking"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.algo.num_envs == 1024
    assert cfg.algo.max_iterations == 9500
    assert cfg.algo.empirical_normalization is True
    assert cfg.algo.obs_groups.critic == ["critic"]
    assert cfg.algo.algorithm.entropy_coef == pytest.approx(0.005)
    assert cfg.algo.algorithm.desired_kl == pytest.approx(0.01)
    # Interactive playback defaults to policy mode for this task.
    assert cfg.interactive.action_mode == "policy"
    assert cfg.env.commands.motion.params.sampling_mode == "start"
    assert cfg.env.commands.motion.params.truncate_on_clip_end is False
    assert cfg.env.sim_dt == pytest.approx(0.005)
    assert cfg.env.actions.joint_pos.scale == pytest.approx(X2_ACTION_SCALE[0])
    assert cfg.env.terminations.anchor_pos.params.threshold == pytest.approx(0.5)
    assert cfg.env.terminations.ee_body_pos.params.threshold == pytest.approx(0.5)
    assert cfg.env.terminations.undesired_contacts is not None
    assert cfg.reward.motion_joint_pos.weight == pytest.approx(0.5)
    assert cfg.reward.motion_joint_vel.weight == pytest.approx(0.25)
    assert cfg.reward.motion_body_pos.weight == pytest.approx(2.0)
    assert cfg.reward.motion_body_ori.weight == pytest.approx(1.5)
    assert cfg.reward.motion_ee_body_pos_z.weight == pytest.approx(2.0)
    assert cfg.reward.action_rate_l2.weight == pytest.approx(-0.005)
    assert cfg.reward.undesired_contacts.weight == pytest.approx(-0.1)
