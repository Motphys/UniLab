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


def _compose_warpsac(task: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "warpsac"), version_base="1.3"):
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
    assert (
        mjwarp_cfg.env.observations.actor.terms.motion_anchor_pack.func
        == "unilab.tasks.motion_tracking.common.manager_terms.MotionObservationPack"
    )
    assert (
        mjwarp_cfg.env.observations.actor.terms.motion_anchor_pack._target_
        == "unilab.tasks.motion_tracking.common.manager_terms.MotionObservationPackCfg"
    )
    assert (
        mjwarp_cfg.env.observations.critic.terms.motion_critic_pack.func
        == "unilab.tasks.motion_tracking.common.manager_terms.MotionCriticObservationPack"
    )
    assert mjwarp_cfg.env.observations.critic.terms.motion_critic_pack._target_ == (
        "unilab.tasks.motion_tracking.common.manager_terms.MotionCriticObservationPackCfg"
    )
    assert mjwarp_cfg.env.reset_owners.motion._target_ == (
        "unilab.tasks.motion_tracking.common.manager_terms.MotionResetOwnerCfg"
    )
    assert (
        mjwarp_cfg.reward.motion_penalty_pack.func
        == "unilab.tasks.motion_tracking.common.manager_terms.MotionPenaltyRewardPack"
    )
    assert mjwarp_cfg.reward.motion_penalty_pack._target_ == (
        "unilab.tasks.motion_tracking.common.manager_terms.MotionPenaltyRewardPackCfg"
    )
    # Anchor term execution, the MJWARP-only tensor command implementation, and
    # its fused selected-reset owner are intended cross-backend differences;
    # the semantic owner remains DR-free.
    mujoco_env = OmegaConf.to_container(mujoco_cfg.env)
    mjwarp_env = OmegaConf.to_container(mjwarp_cfg.env)
    assert isinstance(mujoco_env, dict) and isinstance(mjwarp_env, dict)
    del mjwarp_env["observations"]
    del mujoco_env["observations"]
    del mjwarp_env["commands"]
    del mujoco_env["commands"]
    del mjwarp_env["reset_owners"]
    del mjwarp_env["terminations"]
    del mujoco_env["terminations"]
    assert mjwarp_env == mujoco_env
    mjwarp_reward = OmegaConf.to_container(mjwarp_cfg.reward)
    mujoco_reward = OmegaConf.to_container(mujoco_cfg.reward)
    assert isinstance(mjwarp_reward, dict) and isinstance(mujoco_reward, dict)
    del mjwarp_reward["motion_penalty_pack"]
    for fused_name in ("action_rate_l2", "joint_limit", "undesired_contacts"):
        del mjwarp_reward[fused_name]
        del mujoco_reward[fused_name]
    assert mjwarp_reward == mujoco_reward


def test_flashsac_g1_motion_tracking_newton_keeps_mujoco_parity() -> None:
    mujoco_cfg = _compose_flashsac("g1_motion_tracking/mujoco")
    cfg = _compose_flashsac("g1_motion_tracking/newton")

    assert cfg.training.task_name == "G1MotionTrackingSAC"
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
    assert cfg.training.task_name == "G1MotionTrackingSAC"
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


def test_sac_g1_flip_tracking_stays_dr_free() -> None:
    cfg = _compose_sac("g1_flip_tracking/mujoco")
    assert all(term is None for term in cfg.env.events.values())


def test_warpsac_g1_motion_tracking_owners_share_policy_contract() -> None:
    mujoco_cfg = _compose_warpsac("g1_motion_tracking/mujoco")
    mjwarp_cfg = _compose_warpsac("g1_motion_tracking/mjwarp")

    assert mujoco_cfg.training.task_name == "G1MotionTrackingSAC"
    assert mujoco_cfg.training.sim_backend == "mujoco"
    assert mjwarp_cfg.training.sim_backend == "mjwarp"
    assert mjwarp_cfg.training.play_render_mode == "record"
    assert mujoco_cfg.algo.num_envs == 2048
    assert mujoco_cfg.algo.max_iterations == 25000
    assert mujoco_cfg.algo.updates_per_step == 4
    assert mujoco_cfg.algo.gamma == 0.99
    assert mujoco_cfg.algo.tau == 0.05
    assert mujoco_cfg.algo.decay_step == 0
    assert mujoco_cfg.algo.replay_min_weight == 0.05
    assert mujoco_cfg.algo.algo_params.n_step == 1
    assert mujoco_cfg.training.replay_prefetch_mode == "one_tick"

    for section in ("env", "reward", "algo"):
        assert OmegaConf.to_container(mjwarp_cfg[section]) == OmegaConf.to_container(
            mujoco_cfg[section]
        )


def test_sac_g1_motion_tracking_mjwarp_uses_tensor_motion_owner() -> None:
    cfg = _compose_sac("g1_motion_tracking/mjwarp")

    assert cfg.env.commands.motion._target_ == (
        "unilab.tasks.motion_tracking.common.manager_terms.TensorMotionCommandCfg"
    )
    assert (
        cfg.env.observations.actor.terms.motion_anchor_pack.func
        == "unilab.tasks.motion_tracking.common.manager_terms.MotionObservationPack"
    )
    assert cfg.env.observations.actor.terms.motion_anchor_pack._target_ == (
        "unilab.tasks.motion_tracking.common.manager_terms.MotionObservationPackCfg"
    )
    assert cfg.env.observations.critic.terms.motion_anchor_pack.func == (
        "unilab.tasks.motion_tracking.common.manager_terms.MotionAnchorObservationPack"
    )
    noise = cfg.env.observations.actor.terms.motion_anchor_pack.noise
    assert len(noise.ranges) == 160
    assert noise.ranges[58] == pytest.approx((-0.0, 0.0))
    assert noise.ranges[67] == pytest.approx((-0.1, 0.1))
    assert noise.ranges[70] == pytest.approx((-0.2, 0.2))
    assert noise.ranges[73] == pytest.approx((-0.01, 0.01))
    assert noise.ranges[102] == pytest.approx((-1.5, 1.5))
    assert noise.ranges[131] == pytest.approx((-0.0, 0.0))
