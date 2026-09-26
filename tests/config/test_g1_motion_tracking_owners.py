from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

CONF_DIR = Path(__file__).parents[2] / "src" / "unilab" / "conf"


def _compose_sac(task: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "sac"), version_base="1.3"):
        return compose("config", overrides=[f"task={task}"])


def _compose_warpsac(task: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "warpsac"), version_base="1.3"):
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


def test_sac_g1_motion_tracking_motrix_keeps_supported_dr() -> None:
    cfg = _compose_sac("g1_motion_tracking/motrix")
    assert cfg.env.events.base_com is not None
    assert cfg.env.events.encoder_bias is not None
    assert cfg.env.events.foot_friction is not None
    assert cfg.env.events.push_robot is None
    assert cfg.training.sim_backend == "motrix"


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


def test_sac_g1_motion_tracking_isaacgym_disables_unsupported_dr() -> None:
    cfg = _compose_sac("g1_motion_tracking/isaacgym")
    assert cfg.training.task_name == "G1MotionTrackingSAC"
    assert cfg.training.sim_backend == "isaacgym"
    assert cfg.training.play_render_mode == "auto"
    assert cfg.env.isaacgym_device_id == 0
    assert cfg.env.render_spacing == 2.0
    # The isaacgym legacy path declares an empty DR capability set (fail-closed).
    assert set(cfg.env.events) == {"base_com", "encoder_bias", "foot_friction", "push_robot"}
    assert all(term is None for term in cfg.env.events.values())


def test_sac_g1_motion_tracking_newton_keeps_full_dr() -> None:
    cfg = _compose_sac("g1_motion_tracking/newton")
    assert cfg.training.task_name == "G1MotionTrackingSAC"
    assert cfg.training.sim_backend == "newton"
    assert cfg.env.newton_device is None
    assert cfg.env.newton_nconmax == 320
    assert cfg.env.newton_njmax == 512
    assert cfg.env.newton_use_cuda_graph is True
    # Newton declares an empty DR capability set; only the host-side
    # encoder_bias observation bias stays enabled.
    assert set(cfg.env.events) == {"base_com", "encoder_bias", "foot_friction", "push_robot"}
    assert cfg.env.events.foot_friction is None
    assert cfg.env.events.base_com is None
    assert cfg.env.events.push_robot is None
    assert cfg.env.events.encoder_bias is not None
    assert cfg.env.scene.entities.robot.geom_names is None


def test_sac_g1_motion_tracking_isaacsim_disables_unsupported_dr() -> None:
    cfg = _compose_sac("g1_motion_tracking/isaacsim")
    assert cfg.training.task_name == "G1MotionTrackingSAC"
    assert cfg.training.sim_backend == "isaacsim"
    assert cfg.training.play_render_mode == "auto"
    assert cfg.env.isaacsim_device_id == 0
    assert cfg.env.isaacsim_worker_timeout_s == 120.0
    assert cfg.play_profile.enabled is False
    # The isaacsim legacy path declares an empty DR capability set (fail-closed).
    assert set(cfg.env.events) == {"base_com", "encoder_bias", "foot_friction", "push_robot"}
    assert all(term is None for term in cfg.env.events.values())


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
