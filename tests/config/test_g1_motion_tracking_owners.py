from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

CONF_DIR = Path(__file__).parents[2] / "src" / "unilab" / "conf"
ROOT_DIR = Path(__file__).parents[2]


def _compose_sac(task: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "sac"), version_base="1.3"):
        return compose("config", overrides=[f"task={task}"])


def _compose_flashsac(task: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "flashsac"), version_base="1.3"):
        return compose("config", overrides=[f"task={task}"])


def test_sac_g1_motion_tracking_split_keeps_dr_in_backend_owner() -> None:
    base = _compose_sac("g1_motion_tracking/base")
    assert not hasattr(base.env, "events")
    assert base.env.commands.motion.params.motion_file == "motions/g1/dance1_subject2_part.npz"
    assert (
        base.env.observations.actor.terms.joint_pos.func
        == "unilab.tasks.motion_tracking.common.manager_terms.motion_joint_pos_rel_biased"
    )
    assert (
        base.env.observations.critic.terms.joint_pos.func
        == "unilab.tasks.motion_tracking.common.manager_terms.motion_joint_pos_rel"
    )

    cfg = _compose_sac("g1_motion_tracking/mujoco")
    events = cfg.env.events
    assert set(events) == {"base_com", "encoder_bias", "foot_friction", "push_robot"}
    assert events.base_com.mode == "reset"
    assert events.base_com.params.asset_cfg.body_names == "torso_link"
    assert events.base_com.params.com_range == {
        "x": [-0.025, 0.025],
        "y": [-0.05, 0.05],
        "z": [-0.05, 0.05],
    }
    assert events.encoder_bias.params.bias_range == [-0.01, 0.01]
    assert events.foot_friction.params.ranges == [0.3, 1.2]
    assert events.foot_friction.params.shared_random is True
    assert events.push_robot.mode == "interval"
    assert events.push_robot.interval_range_s == [1.0, 3.0]
    assert events.push_robot.params.velocity_range.z == [-0.2, 0.2]


def test_flashsac_g1_motion_tracking_uses_comparable_dr_free_owner() -> None:
    mujoco_cfg = _compose_flashsac("g1_motion_tracking/mujoco")
    mjwarp_cfg = _compose_flashsac("g1_motion_tracking/mjwarp")

    assert not hasattr(mujoco_cfg.env, "events")
    assert not hasattr(mujoco_cfg.env.scene.entities.robot, "geom_names")
    assert (
        mujoco_cfg.env.observations.actor.terms.joint_pos.func
        == "unilab.tasks.motion_tracking.common.manager_terms.motion_joint_pos_rel"
    )
    assert (
        mujoco_cfg.env.observations.critic.terms.joint_pos.func
        == "unilab.tasks.motion_tracking.common.manager_terms.motion_joint_pos_rel"
    )
    assert mjwarp_cfg.training.sim_backend == "mjwarp"
    assert mjwarp_cfg.env.commands.motion._target_ == (
        "unilab.tasks.motion_tracking.common.manager_terms.TensorMotionCommandCfg"
    )
    assert mjwarp_cfg.env == mujoco_cfg.env
    assert mjwarp_cfg.reward == mujoco_cfg.reward


def test_flashsac_g1_motion_tracking_mjwarp_is_managed_term_baseline() -> None:
    mujoco_cfg = _compose_flashsac("g1_motion_tracking/mujoco")
    cfg = _compose_flashsac("g1_motion_tracking/mjwarp")

    assert cfg.training.sim_backend == "mjwarp"
    assert cfg.env == mujoco_cfg.env
    assert cfg.reward == mujoco_cfg.reward
    assert cfg.algo == mujoco_cfg.algo


def test_flashsac_motion_rewards_publish_canonical_component_names() -> None:
    cfg = _compose_flashsac("g1_motion_tracking/mujoco")

    for name, weight in (
        ("motion_global_root_pos", 1.0),
        ("motion_global_root_ori", 0.5),
        ("motion_body_pos", 2.0),
        ("motion_body_ori", 1.0),
        ("motion_body_lin_vel", 1.0),
        ("motion_body_ang_vel", 1.0),
    ):
        assert cfg.reward[name].weight == pytest.approx(weight)


def test_flashsac_g1_motion_tracking_newton_keeps_mujoco_parity() -> None:
    mujoco_cfg = _compose_flashsac("g1_motion_tracking/mujoco")
    cfg = _compose_flashsac("g1_motion_tracking/newton")

    assert cfg.training.task_name == "G1MotionTracking"
    assert cfg.training.sim_backend == "newton"
    assert cfg.training.play_render_mode == "record"
    assert cfg.env.commands.motion._target_ == (
        "unilab.tasks.motion_tracking.common.manager_terms.TensorMotionCommandCfg"
    )
    assert cfg.env.newton_device == "cuda:0"
    assert cfg.env.newton_nconmax == 320
    assert cfg.env.newton_njmax == 512
    assert cfg.env.newton_capacity_check_steps == 1
    assert cfg.env.newton_use_cuda_graph is True
    # The Newton owner changes only backend identity/placement and the tensor
    # command implementation; it does not add reset DR or task semantics.
    newton_env = OmegaConf.to_container(cfg.env)
    mujoco_env = OmegaConf.to_container(mujoco_cfg.env)
    assert isinstance(newton_env, dict) and isinstance(mujoco_env, dict)
    del newton_env["newton_device"]
    del newton_env["newton_nconmax"]
    del newton_env["newton_njmax"]
    del newton_env["newton_capacity_check_steps"]
    del newton_env["newton_use_cuda_graph"]
    del newton_env["commands"]
    del mujoco_env["commands"]
    assert newton_env == mujoco_env
    assert cfg.algo == mujoco_cfg.algo


def test_flashsac_g1_motion_tracking_motrix_uses_tensor_command() -> None:
    mujoco_cfg = _compose_flashsac("g1_motion_tracking/mujoco")
    cfg = _compose_flashsac("g1_motion_tracking/motrix")

    assert cfg.training.task_name == "G1MotionTracking"
    assert cfg.training.sim_backend == "motrix"
    assert cfg.training.play_render_mode == "record"
    assert cfg.env.commands.motion._target_ == (
        "unilab.tasks.motion_tracking.common.manager_terms.TensorMotionCommandCfg"
    )
    assert cfg.env == mujoco_cfg.env
    assert cfg.algo == mujoco_cfg.algo


def test_flashsac_g1_motion_tracking_contact_policy_is_reward_only() -> None:
    cfg = _compose_flashsac("g1_motion_tracking/mujoco")

    assert "undesired_contacts" not in cfg.env.terminations
    assert cfg.reward.undesired_contacts.weight == pytest.approx(-0.1)


def test_sac_g1_motion_tracking_mjwarp_inherits_full_dr() -> None:
    cfg = _compose_sac("g1_motion_tracking/mjwarp")
    assert set(cfg.env.events) == {"base_com", "encoder_bias", "foot_friction", "push_robot"}
    velocity_range = cfg.env.events.push_robot.params.velocity_range
    assert velocity_range.x == [-0.5, 0.5]
    assert velocity_range.z == [-0.2, 0.2]
    assert velocity_range.roll == [0.0, 0.0]
    assert velocity_range.pitch == [0.0, 0.0]
    assert velocity_range.yaw == [0.0, 0.0]
    assert cfg.training.sim_backend == "mjwarp"


def test_sac_g1_motion_tracking_genesis_inherits_mujoco_parity() -> None:
    mujoco_cfg = _compose_sac("g1_motion_tracking/mujoco")
    cfg = _compose_sac("g1_motion_tracking/genesis")
    assert cfg.training.task_name == "G1MotionTracking"
    assert cfg.training.sim_backend == "genesis"
    assert cfg.training.play_render_mode == "auto"
    assert cfg.env.genesis_device_id == 0
    assert cfg.env.genesis_integrator == "implicitfast"
    # Genesis legacy scenes declare no model-field reset terms and no interval
    # velocity delta; only the host-side encoder_bias bias stays enabled.
    assert set(cfg.env.events) == {"base_com", "encoder_bias", "foot_friction", "push_robot"}
    assert cfg.env.events.foot_friction is None
    assert cfg.env.events.base_com is None
    assert cfg.env.events.push_robot is None
    assert cfg.env.events.encoder_bias is not None
    assert cfg.env.scene.entities.robot.geom_names is None
    # Algo block inherits the MuJoCo owner verbatim (DENYLIST parity).
    assert cfg.algo.num_envs == mujoco_cfg.algo.num_envs
    assert cfg.algo.max_iterations == mujoco_cfg.algo.max_iterations
    assert cfg.algo.updates_per_step == mujoco_cfg.algo.updates_per_step


def test_sac_g1_motion_tracking_mjwarp_uses_tensor_motion_owner() -> None:
    cfg = _compose_sac("g1_motion_tracking/mjwarp")

    assert cfg.env.commands.motion._target_ == (
        "unilab.tasks.motion_tracking.common.manager_terms.TensorMotionCommandCfg"
    )
    actor_terms = cfg.env.observations.actor.terms
    critic_terms = cfg.env.observations.critic.terms
    assert actor_terms.command.func == "unilab.envs.mdp.generated_commands"
    assert actor_terms.motion_anchor_pos_b.func == (
        "unilab.tasks.motion_tracking.common.manager_terms.motion_anchor_pos_b"
    )
    assert actor_terms.motion_anchor_ori_b.func == (
        "unilab.tasks.motion_tracking.common.manager_terms.motion_anchor_ori_b"
    )
    assert actor_terms.base_lin_vel.noise.n_min == pytest.approx(-0.1)
    assert actor_terms.base_ang_vel.noise.n_min == pytest.approx(-0.2)
    assert actor_terms.joint_pos.noise.n_min == pytest.approx(-0.01)
    assert actor_terms.joint_vel.noise.n_min == pytest.approx(-1.5)
    assert critic_terms.motion_anchor_pos_b.func == (
        "unilab.tasks.motion_tracking.common.manager_terms.motion_anchor_pos_b"
    )
    assert critic_terms.motion_anchor_ori_b.func == (
        "unilab.tasks.motion_tracking.common.manager_terms.motion_anchor_ori_b"
    )
