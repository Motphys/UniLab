from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from unilab.base.config_adapter import BackendAdapter
from unilab.base.registry import apply_cfg_overrides
from unilab.envs import ManagerBasedRlEnvCfg

CONF_DIR = Path(__file__).parents[2] / "src" / "unilab" / "conf"
ROOT_DIR = Path(__file__).parents[2]
ISAACSIM_TENSOR_FIXTURE_DIR = ROOT_DIR / "tests" / "fixtures" / "isaacsim_g1_tensor_cuda_ipc"


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


def _stand_key_values(path: Path) -> tuple[list[float], list[float]]:
    key = ET.parse(path).find("./keyframe/key[@name='stand']")
    assert key is not None
    return (
        [float(value) for value in key.attrib["qpos"].split()],
        [float(value) for value in key.attrib["ctrl"].split()],
    )


def _compose_isaacsim_tensor_fixture():
    base = _compose_flashsac("g1_motion_tracking/mujoco")
    overlay = OmegaConf.load(ISAACSIM_TENSOR_FIXTURE_DIR / "isaacsim_candidate_overlay.yaml")
    # The production MuJoCo owner is struct mode; a fixture overlay may add
    # backend-only fields exactly as a Hydra child owner would.
    OmegaConf.set_struct(base, False)
    base.merge_with(overlay)
    return base


def _structural_robot_signature(path: Path) -> bytes:
    root = ET.parse(path).getroot()
    root.attrib.pop("model", None)
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)
    ET.indent(root, space="")
    return ET.tostring(root, encoding="unicode").encode()


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
    for section in ("env", "reward"):
        assert OmegaConf.to_container(mjwarp_cfg[section]) == OmegaConf.to_container(
            mujoco_cfg[section]
        )


def test_flashsac_g1_motion_tracking_contact_policy_is_reward_only() -> None:
    cfg = _compose_flashsac("g1_motion_tracking/mujoco")

    assert "undesired_contacts" not in cfg.env.terminations
    assert cfg.reward.undesired_contacts.weight == pytest.approx(-0.1)


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


def test_flashsac_g1_motion_tracking_isaacsim_opts_into_cuda_ipc_candidate() -> None:
    cfg = _compose_isaacsim_tensor_fixture()
    assert cfg.training.sim_backend == "isaacsim"
    assert cfg.env.tensor_runtime is True
    assert cfg.env.isaacsim_tensor_cuda_ipc is True
    assert cfg.env.isaacsim_share_friction_materials is True

    scene = cfg.env.scene
    assert scene.model_file is None
    assert scene.default_keyframe_name == "stand"
    assert [
        (entity.name, entity.get("kind", "articulation"), entity.root_mode)
        for entity in scene.entity_assets
    ] == [("robot", "articulation", "floating"), ("floor", "rigid", "fixed")]
    robot = scene.entities.robot
    assert robot.physical_entity == "robot"
    assert robot.root_body_name == "robot/pelvis"
    assert robot.joint_names[0] == "robot/left_hip_pitch_joint"
    assert robot.body_names[0] == "robot/pelvis"
    assert all(name.startswith("robot/") for name in robot.joint_names)
    assert all(name.startswith("robot/") for name in robot.body_names)
    assert cfg.env.commands.motion.params.anchor_body_name == "robot/torso_link"
    assert cfg.env.commands.motion.params.body_names[0] == "robot/pelvis"
    assert (
        cfg.env.observations.actor.terms.base_lin_vel.params.sensor_name
        == "robot/pelvis_local_linvel"
    )
    assert cfg.env.observations.actor.terms.base_ang_vel.params.sensor_name == "robot/torso_gyro"
    assert (
        cfg.env.observations.critic.terms.sac_base_lin_vel.params.sensor_name
        == "robot/pelvis_local_linvel"
    )

    fixture_robot = ROOT_DIR / str(scene.entity_assets[0].source.model_file)
    canonical_robot = ROOT_DIR / "src/unilab/assets/robots/g1/g1.xml"
    canonical_qpos, canonical_ctrl = _stand_key_values(
        ROOT_DIR / "src/unilab/assets/robots/g1/scene_flat.xml"
    )
    mapped_qpos, mapped_ctrl = _stand_key_values(fixture_robot)
    assert mapped_qpos == canonical_qpos
    assert mapped_ctrl == canonical_ctrl
    assert _structural_robot_signature(fixture_robot) == _structural_robot_signature(
        canonical_robot
    )
    assert list(scene.entity_assets[0].initial_state.position) == [0.0, 0.0, mapped_qpos[2]]


def test_isaacsim_tensor_candidate_stays_outside_production_owner_discovery() -> None:
    production_owner = ROOT_DIR / "src/unilab/conf/flashsac/task/g1_motion_tracking/isaacsim.yaml"

    assert not production_owner.exists()
    assert (ISAACSIM_TENSOR_FIXTURE_DIR / "isaacsim_candidate_overlay.yaml").is_file()
    assert (ISAACSIM_TENSOR_FIXTURE_DIR / "g1_stand_entity.xml").is_file()


def test_isaacsim_tensor_fixture_materializes_into_manager_config() -> None:
    owner_cfg = _compose_isaacsim_tensor_fixture()
    override = BackendAdapter(
        owner_cfg, root_dir=ROOT_DIR, algo_name="flashsac"
    ).build_task_env_cfg_override()
    cfg = ManagerBasedRlEnvCfg()
    apply_cfg_overrides(cfg, override)

    cfg.validate()
    assert cfg.isaacsim_tensor_cuda_ipc is True
    assert cfg.isaacsim_share_friction_materials is True
    assert cfg.tensor_runtime is True
    assert cfg.scene is not None
    assert cfg.scene.entity_assets


def test_isaacsim_cuda_ipc_requires_manager_tensor_runtime() -> None:
    cfg = ManagerBasedRlEnvCfg(isaacsim_tensor_cuda_ipc=True)

    with pytest.raises(ValueError, match="isaacsim_tensor_cuda_ipc requires tensor_runtime"):
        cfg.validate()


def test_flashsac_g1_motion_tracking_isaacgym_uses_gpu_tensor_candidate() -> None:
    cfg = _compose_flashsac("g1_motion_tracking/isaacgym")
    assert cfg.training.sim_backend == "isaacgym"
    assert cfg.env.tensor_runtime is True
    assert cfg.env.isaacgym_device_id == 0
    assert cfg.env.scene.model_file.endswith("robots/g1/scene_flat.xml")
    assert cfg.env.scene.default_keyframe_name == "stand"
    assert not cfg.env.scene.get("entity_assets", [])
    assert cfg.env.scene.entities.robot.get("physical_entity") is None
    assert cfg.env.scene.entities.robot.root_body_name == "pelvis"


@pytest.mark.parametrize("backend", ["superdex", "drake"])
def test_flashsac_g1_motion_tracking_host_bridge_candidates_opt_in(backend: str) -> None:
    cfg = _compose_flashsac(f"g1_motion_tracking/{backend}")
    assert cfg.training.sim_backend == backend
    assert cfg.env.tensor_runtime is True
    assert cfg.env.scene.model_file.endswith("robots/g1/scene_flat.xml")
    assert cfg.env.scene.default_keyframe_name == "stand"


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
