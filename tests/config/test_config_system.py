"""Config system verification tests.

These tests enforce that:
1. Base Hydra configs compose without legacy config groups.
2. Every supported runtime variant resolves through exactly one task owner file.
3. Final reward/env/algo sections are present on the composed config, not mounted by Python glue.
4. Backend-specific hyperparameters preserve the intended pre-refactor behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

CONF_DIR = Path(__file__).parent.parent.parent / "conf"
_BACKENDS = ("mujoco", "mjwarp", "motrix", "isaacgym", "isaacsim")


def _expected_backend_from_variant(name: str) -> str | None:
    for backend in _BACKENDS:
        if name == backend or name.startswith(f"{backend}_"):
            return backend
    return None


def _compose(algo_dir: str, config_name: str = "config", overrides: list[str] | None = None):
    normalized_overrides = _normalize_overrides(algo_dir, overrides)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / algo_dir), version_base="1.3"):
        return compose(config_name, overrides=normalized_overrides)


def _normalize_overrides(algo_dir: str, overrides: list[str] | None) -> list[str]:
    normalized: list[str] = []
    task_selected = False

    for override in overrides or []:
        if override.startswith("task="):
            task_selected = True
            normalized.append(override)
            continue
        normalized.append(override)

    if not task_selected:
        if algo_dir in ("sac", "td3", "flashsac"):
            normalized.append("task=g1_walk_flat/mujoco")
        else:
            normalized.append("task=go1_joystick_flat/mujoco")

    return normalized


def _assert_reward_populated(cfg, label: str):
    assert hasattr(cfg, "reward"), f"{label} missing cfg.reward"
    reward_dict = OmegaConf.to_container(cfg.reward, resolve=True)
    assert isinstance(reward_dict, dict), f"{label} reward must resolve to mapping"
    if "scales" in reward_dict:
        assert len(reward_dict["scales"]) > 0, f"{label} reward.scales must be non-empty"
        return

    active_terms = {name: term for name, term in reward_dict.items() if term is not None}
    assert active_terms, f"{label} Manager-Based reward terms must be non-empty"
    for term_name, term in active_terms.items():
        assert isinstance(term, dict), f"{label} reward.{term_name} must be a mapping"
        assert "func" in term, f"{label} reward.{term_name} must declare func"
        assert "weight" in term, f"{label} reward.{term_name} must declare weight"


def _supported_task_cases() -> list[tuple[str, str, str, str, str, list[str]]]:
    cases: list[tuple[str, str, str, str, str, list[str]]] = []

    for algo_dir in ["ppo", "appo"]:
        root = CONF_DIR / algo_dir / "task"
        for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            for backend_file in sorted(task_dir.glob("*.yaml")):
                expected_backend = _expected_backend_from_variant(backend_file.stem)
                if expected_backend is None:
                    continue
                cases.append(
                    (
                        algo_dir,
                        "config",
                        task_dir.name,
                        expected_backend,
                        str(backend_file.relative_to(CONF_DIR)),
                        [f"task={task_dir.name}/{backend_file.stem}"],
                    )
                )

    for algo_dir in ["sac", "td3", "flashsac"]:
        root = CONF_DIR / algo_dir / "task"
        for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            for backend_file in sorted(task_dir.glob("*.yaml")):
                expected_backend = _expected_backend_from_variant(backend_file.stem)
                if expected_backend is None:
                    continue
                cases.append(
                    (
                        algo_dir,
                        "config",
                        task_dir.name,
                        expected_backend,
                        str(backend_file.relative_to(CONF_DIR)),
                        [f"task={task_dir.name}/{backend_file.stem}"],
                    )
                )

    return cases


@pytest.mark.parametrize(
    "algo_dir,config_name",
    [
        ("sac", "config"),
        ("td3", "config"),
        ("flashsac", "config"),
        ("appo", "config"),
        ("ppo", "config"),
    ],
)
def test_algo_config_composes(algo_dir: str, config_name: str):
    cfg = _compose(algo_dir, config_name)
    assert cfg.training.task_name
    assert cfg.training.sim_backend == "mujoco"


def test_legacy_config_groups_removed():
    for path in [
        CONF_DIR / "ppo" / "reward",
        CONF_DIR / "ppo" / "backend_task_preset",
        CONF_DIR / "ppo" / "algo_preset",
        CONF_DIR / "ppo" / "sim_backend",
        CONF_DIR / "appo" / "reward",
        CONF_DIR / "appo" / "backend_task_preset",
        CONF_DIR / "appo" / "sim_backend",
        CONF_DIR / "sac" / "reward",
        CONF_DIR / "sac" / "backend_task_preset",
        CONF_DIR / "sac" / "algo_preset",
        CONF_DIR / "sac" / "sim_backend",
    ]:
        assert not path.exists(), f"legacy config group should be removed: {path}"


def test_task_files_keep_full_identity_without_hidden_backend_marker():
    for path in sorted(CONF_DIR.glob("*/task/**/*.yaml")):
        cfg = OmegaConf.load(path)
        cfg_dict_raw = OmegaConf.to_container(cfg, resolve=True) or {}
        assert isinstance(cfg_dict_raw, dict)
        assert "_selected_sim_backend" not in cfg_dict_raw, (
            f"task has hidden backend marker: {path}"
        )
        if path.stem not in _BACKENDS:
            continue
        training_raw = cfg_dict_raw.get("training", {})
        assert isinstance(training_raw, dict)
        assert "task_name" in training_raw, f"task missing task_name: {path}"
        assert "sim_backend" in training_raw, f"task missing sim_backend: {path}"


def test_motrix_task_files_do_not_declare_post_step_forward_sensor():
    for path in sorted(CONF_DIR.glob("*/task/**/*motrix*.yaml")):
        cfg = OmegaConf.load(path)

        assert OmegaConf.select(cfg, "env.post_step_forward_sensor") is None, (
            "post_step_forward_sensor is routed only to MuJoCo backends: "
            f"{path.relative_to(CONF_DIR)}"
        )


@pytest.mark.parametrize(
    "algo_dir,config_name,task,backend,task_file,overrides",
    _supported_task_cases(),
)
def test_supported_task_composes(
    algo_dir: str,
    config_name: str,
    task: str,
    backend: str,
    task_file: str,
    overrides: list[str],
):
    cfg = _compose(algo_dir, config_name, overrides=overrides)

    assert cfg.training.task_name, f"{task_file} should resolve task_name"
    assert cfg.training.sim_backend == backend, f"{task_file} should set backend"
    _assert_reward_populated(cfg, task_file)


def test_ppo_go2_arm_manip_loco_motrix_preserves_backend_overrides():
    cfg = _compose("ppo", overrides=["task=go2_arm_manip_loco/motrix"])

    assert cfg.training.task_name == "Go2ArmManipLoco"
    assert cfg.training.sim_backend == "motrix"
    assert cfg.algo.num_envs == 4096
    assert cfg.algo.max_iterations == 3000
    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(2.0)
    assert cfg.env.domain_rand.randomize_dof_armature is False
    assert cfg.env.domain_rand.randomize_kp is False
    assert cfg.env.domain_rand.randomize_kd is False


def test_offpolicy_g1_walk_flat_motrix_sac_preserves_backend_overrides():
    cfg = _compose("sac", overrides=["task=g1_walk_flat/motrix"])

    assert cfg.algo.num_envs == 2048
    assert cfg.algo.max_iterations == 5000
    assert cfg.reward.tracking_lin_vel.weight == pytest.approx(2.2)
    assert cfg.env.events.pd_gains is None


def test_offpolicy_g1_walk_flat_mujoco_td3_uses_td3_task_owner():
    cfg = _compose("td3", overrides=["task=g1_walk_flat/mujoco"])

    assert cfg.training.task_name == "G1WalkFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.algo.max_iterations == 100000
    assert cfg.algo.tau == pytest.approx(0.1)
    assert cfg.algo.actor_hidden_dim == 512
    assert cfg.algo.critic_hidden_dim == 1024
    assert cfg.reward.tracking_lin_vel.weight == pytest.approx(2.0)
    assert cfg.env.actions.joint_pos.scale == pytest.approx(1.0)


def test_offpolicy_td3_go2_joystick_flat_motrix_composes():
    cfg = _compose(
        "td3",
        overrides=["task=go2_joystick_flat/motrix"],
    )

    assert cfg.training.task_name == "Go2JoystickFlat"
    assert cfg.training.sim_backend == "motrix"
    assert cfg.algo.algo == "td3"
    assert cfg.algo.tau == pytest.approx(0.1)
    assert cfg.algo.algo_params.weight_decay == pytest.approx(0.1)
    assert cfg.algo.algo_params.policy_noise == pytest.approx(0.2)
    assert cfg.reward.tracking_lin_vel.weight == pytest.approx(1.0)
    assert cfg.reward.base_height.params.target_height == pytest.approx(0.3)


def test_offpolicy_td3_go1_joystick_flat_motrix_composes():
    cfg = _compose(
        "td3",
        overrides=["task=go1_joystick_flat/motrix"],
    )

    assert cfg.training.task_name == "Go1JoystickFlat"
    assert cfg.training.sim_backend == "motrix"
    assert cfg.algo.algo == "td3"
    assert cfg.reward.tracking_lin_vel.weight == pytest.approx(1.0)
    assert cfg.reward.contact is None
    assert cfg.env.events.push_robot is None


def test_offpolicy_g1_walk_flat_mjwarp_owner_preserves_sac_contract():
    mujoco_cfg = _compose("sac", overrides=["task=g1_walk_flat/mujoco"])
    mjwarp_cfg = _compose("sac", overrides=["task=g1_walk_flat/mjwarp"])

    assert mjwarp_cfg.training.sim_backend == "mjwarp"
    assert mjwarp_cfg.training.no_play is False
    assert mjwarp_cfg.training.play_render_mode == "record"
    assert mjwarp_cfg.algo.num_envs == mujoco_cfg.algo.num_envs
    assert mjwarp_cfg.env.actions.joint_pos.scale == pytest.approx(
        mujoco_cfg.env.actions.joint_pos.scale
    )
    assert mjwarp_cfg.env.mjwarp_nconmax == 128
    assert mjwarp_cfg.env.mjwarp_njmax == 256
    assert mjwarp_cfg.env.render_spacing == pytest.approx(2.0)
    assert mjwarp_cfg.env.events.pd_gains is None
    assert OmegaConf.to_container(mjwarp_cfg.reward, resolve=True) == OmegaConf.to_container(
        mujoco_cfg.reward, resolve=True
    )


def test_ppo_g1_mjwarp_inherits_enabled_playback_default():
    cfg = _compose("ppo", overrides=["task=g1_walk_flat/mjwarp"])

    assert cfg.training.no_play is False
    assert cfg.training.play_render_mode == "record"


def test_ppo_g1_backend_specific_hyperparams_remain_separate():
    mujoco_cfg = _compose("ppo", overrides=["task=g1_walk_flat/mujoco"])
    motrix_cfg = _compose("ppo", overrides=["task=g1_walk_flat/motrix"])

    assert mujoco_cfg.algo.max_iterations == 2200
    assert mujoco_cfg.algo.empirical_normalization is False
    assert mujoco_cfg.algo.obs_groups.actor == ["actor"]

    assert motrix_cfg.algo.max_iterations == 2200
    assert motrix_cfg.algo.empirical_normalization is True
    assert motrix_cfg.algo.obs_groups.actor == ["policy"]
    assert OmegaConf.select(motrix_cfg, "env.motrix_max_iterations") is None
    assert motrix_cfg.env.actions.joint_pos.scale == pytest.approx(0.5)
    assert motrix_cfg.env.commands.twist.ranges.lin_vel_x == [0.4, 0.7]
    assert motrix_cfg.env.observations.policy.terms.gait_phase.params.init_mode == "offset_phase"
    assert motrix_cfg.reward.tracking_lin_vel.weight == pytest.approx(2.0)
    assert motrix_cfg.reward.tracking_ang_vel.weight == pytest.approx(0.25)
    assert motrix_cfg.reward.forward_progress.weight == pytest.approx(0.0)
    assert motrix_cfg.reward.under_speed.weight == pytest.approx(-0.2)
    assert motrix_cfg.reward.penalty_feet_ori.weight == pytest.approx(0.0)
    assert motrix_cfg.reward.feet_phase.weight == pytest.approx(1.2)
    assert motrix_cfg.reward.feet_phase_contrast.weight == pytest.approx(1.5)
    assert motrix_cfg.reward.feet_phase_contact.weight == pytest.approx(1.0)
    assert motrix_cfg.reward.feet_double_stance.weight == pytest.approx(-1.0)
    assert motrix_cfg.reward.base_height.weight == pytest.approx(-120.0)
    assert motrix_cfg.reward.pose.weight == pytest.approx(-0.05)
    assert motrix_cfg.reward.base_height.params.target_height == pytest.approx(0.765)
    assert motrix_cfg.reward.feet_phase.params.min_forward_speed == pytest.approx(0.05)
    assert motrix_cfg.env.terminations.base_height.params.minimum_height == pytest.approx(0.5)
    assert motrix_cfg.env.terminations.tilt.params.max_tilt_deg == pytest.approx(35.0)


@pytest.mark.parametrize(
    ("algo_dir", "overrides"),
    [
        ("ppo", ["task=g1_walk_flat/mujoco"]),
        ("ppo_him", ["task=go2_arm_manip_loco/mujoco"]),
        ("appo", ["task=g1_walk_flat/mujoco"]),
        ("sac", ["task=g1_walk_flat/mujoco"]),
        ("flashsac", ["task=g1_walk_flat/mujoco"]),
    ],
)
def test_post_step_forward_sensor_defaults_false_outside_sharpa_mujoco(
    algo_dir: str, overrides: list[str]
):
    cfg = _compose(algo_dir, overrides=overrides)

    assert cfg.env.post_step_forward_sensor is False


@pytest.mark.parametrize(
    ("algo_dir", "overrides"),
    [
        ("ppo", ["task=sharpa_inhand/mujoco"]),
        ("ppo", ["task=sharpa_inhand/mujoco_hora"]),
        ("ppo", ["task=sharpa_inhand_grasp/mujoco"]),
        ("appo", ["task=sharpa_inhand/mujoco"]),
        ("appo", ["task=sharpa_inhand/mujoco_hora"]),
        ("sac", ["task=sharpa_inhand/mujoco_hora"]),
        ("hora_distill", ["task=sharpa_inhand/mujoco"]),
    ],
)
def test_post_step_forward_sensor_enabled_for_sharpa_mujoco(algo_dir: str, overrides: list[str]):
    cfg = _compose(algo_dir, overrides=overrides)

    assert cfg.env.post_step_forward_sensor is True


def test_mujoco_post_step_forward_sensor_can_be_overridden():
    override_cfg = _compose(
        "appo",
        overrides=["task=sharpa_inhand/mujoco_hora", "env.post_step_forward_sensor=false"],
    )

    assert override_cfg.env.post_step_forward_sensor is False


def test_appo_adaptive_lr_factors_are_overridden_only_by_dex_hand_owners():
    g1_cfg = _compose("appo", overrides=["task=g1_walk_flat/mujoco"])
    allegro_cfg = _compose("appo", overrides=["task=allegro_inhand/mujoco"])
    allegro_motrix_cfg = _compose("appo", overrides=["task=allegro_inhand/motrix"])
    sharpa_cfg = _compose("appo", overrides=["task=sharpa_inhand/mujoco"])
    sharpa_hora_cfg = _compose("appo", overrides=["task=sharpa_inhand/mujoco_hora"])

    assert g1_cfg.algo.algorithm.adaptive_kl_factor == pytest.approx(1.2)
    assert g1_cfg.algo.algorithm.adaptive_lr_factor == pytest.approx(1.1)
    assert allegro_cfg.algo.algorithm.adaptive_kl_factor == pytest.approx(2.0)
    assert allegro_cfg.algo.algorithm.adaptive_lr_factor == pytest.approx(1.5)
    assert allegro_motrix_cfg.algo.algorithm.adaptive_kl_factor == pytest.approx(2.0)
    assert allegro_motrix_cfg.algo.algorithm.adaptive_lr_factor == pytest.approx(1.5)
    assert sharpa_cfg.algo.algorithm.adaptive_kl_factor == pytest.approx(1.2)
    assert sharpa_cfg.algo.algorithm.adaptive_lr_factor == pytest.approx(1.1)
    assert sharpa_hora_cfg.algo.algorithm.adaptive_kl_factor == pytest.approx(1.2)
    assert sharpa_hora_cfg.algo.algorithm.adaptive_lr_factor == pytest.approx(1.1)


def test_ppo_go1_motrix_preserves_reward_and_algo_values():
    cfg = _compose("ppo", overrides=["task=go1_joystick_flat/motrix"])

    assert cfg.algo.max_iterations == 151
    assert cfg.algo.empirical_normalization is True
    assert cfg.algo.policy.init_noise_std == pytest.approx(0.5)
    assert cfg.algo.algorithm.learning_rate == pytest.approx(3.0e-4)
    assert cfg.reward.tracking_lin_vel.weight == pytest.approx(1.0)
    assert cfg.reward.contact is None
    assert cfg.env.commands.twist.ranges.lin_vel_x == [0.5, 0.5]
    assert cfg.env.commands.twist.ranges.lin_vel_y == [0.0, 0.0]
    assert cfg.env.commands.twist.ranges.ang_vel_z == [0.0, 0.0]
    assert cfg.env.events.push_robot is None


def test_ppo_go2_motrix_preserves_backend_env_overrides():
    cfg = _compose("ppo", overrides=["task=go2_joystick_flat/motrix"])

    assert cfg.algo.num_envs == 1024
    assert cfg.algo.empirical_normalization is True
    assert cfg.env.events.pd_gains is None
    assert cfg.env.commands.twist.ranges.lin_vel_x == [0.5, 0.5]
    assert cfg.env.commands.twist.ranges.lin_vel_y == [0.0, 0.0]
    assert cfg.env.commands.twist.ranges.ang_vel_z == [0.0, 0.0]


def test_ppo_go2w_mujoco_uses_motor_owner_dr_path():
    cfg = _compose("ppo", overrides=["task=go2w_joystick_flat/mujoco"])

    assert cfg.training.task_name == "Go2WJoystickFlat"
    assert cfg.training.sim_backend == "mujoco"
    command = cfg.env.commands.twist
    assert command.ranges.lin_vel_x == [0.0, 1.0]
    assert command.ranges.lin_vel_y == [0.0, 0.0]
    assert command.ranges.ang_vel_z == [-1.0, 1.0]
    action = cfg.env.actions.motor
    assert action.leg_action_scale == pytest.approx(0.5)
    assert action.leg_kp == pytest.approx(50.0)
    assert action.leg_kd == pytest.approx(1.5)
    assert action.wheel_action_scale == pytest.approx(10.0)
    assert action.wheel_kd == pytest.approx(0.5)
    gains = cfg.env.events.motor_gains.params
    assert gains.kp_multiplier_range == [1.0, 1.0]
    assert gains.kd_multiplier_range == [1.0, 1.0]
    assert cfg.reward.tracking_ang_vel.weight == pytest.approx(0.75)
    assert cfg.reward.orientation.weight == pytest.approx(-2.0)
    assert cfg.reward.upward.weight == pytest.approx(1.0)
    assert cfg.reward.base_height.params.target_height == pytest.approx(0.4)
    assert cfg.reward.torques.weight < 0.0


def test_ppo_go2w_motrix_uses_motor_owner_dr_path():
    cfg = _compose("ppo", overrides=["task=go2w_joystick_flat/motrix"])

    assert cfg.training.task_name == "Go2WJoystickFlat"
    assert cfg.training.sim_backend == "motrix"
    assert cfg.env.render_offset_mode == "zero"
    command = cfg.env.commands.twist
    assert command.ranges.lin_vel_x == [0.0, 1.0]
    assert command.ranges.lin_vel_y == [0.0, 0.0]
    assert command.ranges.ang_vel_z == [-1.0, 1.0]
    action = cfg.env.actions.motor
    assert action.leg_action_scale == pytest.approx(0.5)
    assert action.leg_kp == pytest.approx(50.0)
    assert action.leg_kd == pytest.approx(1.5)
    assert action.wheel_action_scale == pytest.approx(10.0)
    assert action.wheel_kd == pytest.approx(0.5)
    assert cfg.reward.tracking_ang_vel.weight == pytest.approx(0.75)
    assert cfg.reward.orientation.weight == pytest.approx(-2.0)
    assert cfg.reward.upward.weight == pytest.approx(1.0)
    assert cfg.reward.torques.weight < 0.0


def test_ppo_go2w_motrix_uses_motor_owner_scene_path():
    cfg = _compose("ppo", overrides=["task=go2w_joystick_flat/motrix"])

    assert cfg.training.task_name == "Go2WJoystickFlat"
    assert cfg.training.sim_backend == "motrix"
    assert str(cfg.env.scene.model_file).endswith("src/unilab/assets/robots/go2w/scene_flat.xml")
    assert cfg.env.scene.default_keyframe_name == "home"
    assert cfg.env.actions.motor.wheel_action_scale == pytest.approx(10.0)
    assert cfg.reward.torques.weight < 0.0


def test_offpolicy_g1_walk_flat_motrix_preserves_backend_env_overrides():
    cfg = _compose("sac", overrides=["task=g1_walk_flat/motrix"])

    assert cfg.training.sim_backend == "motrix"
    assert cfg.algo.num_envs == 2048
    assert cfg.algo.max_iterations == 5000
    assert cfg.env.events.pd_gains is None
    assert cfg.reward.tracking_lin_vel.weight == pytest.approx(2.2)


def test_offpolicy_flashsac_go2_joystick_mujoco_enables_full_dr_stack():
    mujoco_cfg = _compose(
        "flashsac",
        overrides=["task=go2_joystick_flat/mujoco"],
    )

    assert mujoco_cfg.training.task_name == "Go2JoystickFlat"
    assert mujoco_cfg.training.sim_backend == "mujoco"

    assert mujoco_cfg.env.events.pd_gains.func == "unilab.envs.mdp.pd_gains"
    assert (
        mujoco_cfg.env.events.randomize_rigid_body_mass.func
        == "unilab.envs.mdp.randomize_rigid_body_mass"
    )
    assert (
        mujoco_cfg.env.events.randomize_rigid_body_com.func
        == "unilab.envs.mdp.randomize_rigid_body_com"
    )
    assert (
        mujoco_cfg.env.events.randomize_physics_scene_gravity.func
        == "unilab.envs.mdp.randomize_physics_scene_gravity"
    )
    assert (
        mujoco_cfg.env.events.push_by_setting_velocity.func
        == "unilab.envs.mdp.push_by_setting_velocity"
    )
    assert mujoco_cfg.env.events.push_by_setting_velocity.mode == "interval"
    assert mujoco_cfg.env.events.push_by_setting_velocity.is_global_time is True
    assert mujoco_cfg.env.observations.policy.enable_corruption is True
    assert mujoco_cfg.env.observations.policy.terms.joint_pos.noise.n_min == pytest.approx(-0.01)
    assert mujoco_cfg.env.observations.policy.terms.joint_vel.noise.n_min == pytest.approx(-0.1)


def test_cli_override_beats_task_defaults():
    cfg = _compose(
        "ppo",
        overrides=["task=g1_walk_flat/motrix", "algo.max_iterations=1"],
    )

    assert cfg.algo.max_iterations == 1
    assert cfg.algo.empirical_normalization is True
