"""Hydra, Registry, and term-level contracts for MicroduckStandupFlat.

Locks the port of the upstream microduck_rl standup recipe (anchor commit
29e887e, microduck_standup_env_cfg.py + tasks/mdp.py): the fixed-target stand
reward layer, the 6D body-pose command/tracking pair, the four-bucket
set_ground_state reset with the sitting keyframe overrides, the curriculum
stage tables (steps = iters x 24), and the NaN termination with contact-force
sensor coverage.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
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
from unilab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from unilab.managers._types import ManagerBasedRlEnv
from unilab.tasks.locomotion.microduck import recovery_terms, standup_terms
from unilab.tasks.locomotion.microduck.deploy_contract import (
    MICRODUCK_ACTOR_OBS_DIM,
    MICRODUCK_CRITIC_OBS_DIM,
    MICRODUCK_NUM_ACTION,
)

ROOT_DIR = Path(__file__).parents[4]
CONF_DIR = ROOT_DIR / "src" / "unilab" / "conf" / "ppo"
ASSET_DIR = ROOT_DIR / "src" / "unilab" / "assets" / "robots" / "microduck"

# Upstream standup critic: actor 61 + base_lin_vel 3 + foot terms WITHOUT the
# terrain-sensor foot_height (2) the walk critic carries.
STANDUP_CRITIC_OBS_DIM = MICRODUCK_CRITIC_OBS_DIM - 2

# Upstream SITTING_JOINT_OVERRIDES by servo index (mdp layout), expressed by
# joint name in the UniLab owner.
SITTING_JOINT_OVERRIDES = {
    "left_hip_roll": 0.0,
    "left_hip_pitch": -0.4079,
    "left_knee": 1.35,
    "left_ankle": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_pitch": 0.4079,
    "right_knee": -1.35,
    "right_ankle": 0.0,
}

LEG_JOINTS = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)


def _compose_owner(task: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose("config", overrides=[f"task={task}"])


def _materialize(task: str, task_name: str) -> tuple[Any, Any]:
    cfg = _compose_owner(task)
    registry.ensure_registries()
    override = BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="ppo").build_task_env_cfg_override()
    env_cfg = registry.materialize_env_config(task_name)
    assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(env_cfg, override)
    env_cfg.validate()
    return cfg, env_cfg


def _materialize_standup() -> tuple[Any, Any]:
    return _materialize("microduck_standup_flat/mujoco", "MicroduckStandupFlat")


def test_standup_owner_materializes_stand_layer() -> None:
    cfg, env_cfg = _materialize_standup()

    assert cfg.training.task_name == "MicroduckStandupFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.algo.seed == 42
    assert cfg.algo.num_envs == 4096
    assert cfg.algo.max_iterations == 15000
    assert cfg.algo.save_interval == 250
    assert cfg.algo.experiment_name == "microduck_stand"
    assert cfg.algo.run_name == "microduck_stand"
    assert cfg.algo.num_steps_per_env == 24
    assert cfg.algo.empirical_normalization is True
    assert cfg.algo.obs_groups.actor == ["policy"]
    assert cfg.algo.obs_groups.critic == ["critic"]
    assert cfg.env.seed == 42
    assert cfg.env.max_episode_seconds == pytest.approx(6.0)

    assert env_cfg.scene.model_file.endswith("robots/microduck/scene_flat_groundcontact_bam.xml")
    assert env_cfg.scene.fragment_files[0].endswith("robots/microduck/locomotion_task.xml")
    assert env_cfg.scene.default_keyframe_name == "home"

    # Obs layout: policy stays the shared 61D contract; the critic drops the
    # terrain-sensor foot_height term like upstream (no height scanner here).
    expected = (
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "twist_command",
        "head_pose_command",
        "body_pose_command",
    )
    assert tuple(env_cfg.observations["policy"].terms) == expected
    assert tuple(env_cfg.observations["critic"].terms) == (
        *expected,
        "base_lin_vel",
        "foot_air_time",
        "foot_contact",
        "foot_contact_forces",
    )

    # The walk layer is gone; the reward set is the upstream stand layer.
    assert tuple(env_cfg.rewards) == (
        "pose_stand_legs",
        "head_pose_tracking",
        "head_pose_bias",
        "pose_stand_l1",
        "height_stand",
        "height_stand_sharp",
        "height_stand_l1",
        "com_upward_velocity",
        "gentle_rise",
        "arrival_damping",
        "upright_linear",
        "upright_sharp",
        "standing_composite",
        "body_pose_tracking",
        "action_rate",
        "joint_torque_rate_l2",
        "body_ang_vel",
        "angular_momentum",
        "self_collisions",
        "dof_pos_limits",
    )

    # Sign convention: self-negating penalties carry POSITIVE weights; plain
    # cost functions carry negative (or zero-starting) weights.
    pose_stand = env_cfg.rewards["pose_stand_legs"]
    assert pose_stand.func is standup_terms.pose_target_match
    assert pose_stand.weight == pytest.approx(2.0)
    assert pose_stand.params["std"] == pytest.approx(0.5)
    assert tuple(pose_stand.params["asset_cfg"].joint_names) == LEG_JOINTS
    pose_l1 = env_cfg.rewards["pose_stand_l1"]
    assert pose_l1.func is standup_terms.pose_l1_penalty
    assert pose_l1.weight == pytest.approx(1.25)

    head_tracking = env_cfg.rewards["head_pose_tracking"]
    assert head_tracking.weight == pytest.approx(0.75)
    assert head_tracking.params["std"] == pytest.approx(0.5)
    bias = env_cfg.rewards["head_pose_bias"]
    assert bias.weight == pytest.approx(0.0)
    assert bias.params["tau_s"] == pytest.approx(1.0)
    assert bias.params["gate_height_low"] == pytest.approx(0.09)
    assert bias.params["gate_height_high"] == pytest.approx(0.11)
    assert bias.params["gate_tilt_full_deg"] == pytest.approx(20.0)
    assert bias.params["gate_tilt_zero_deg"] == pytest.approx(45.0)

    height_stand = env_cfg.rewards["height_stand"]
    assert height_stand.func is standup_terms.height_target_gaussian
    assert height_stand.weight == pytest.approx(1.0)
    assert height_stand.params["std"] == pytest.approx(0.04)
    assert height_stand.params["target_height"] == pytest.approx(0.115)
    sharp = env_cfg.rewards["height_stand_sharp"]
    assert sharp.func is standup_terms.height_target_gaussian
    assert sharp.weight == pytest.approx(1.0)
    assert sharp.params["std"] == pytest.approx(0.015)
    height_l1 = env_cfg.rewards["height_stand_l1"]
    assert height_l1.func is standup_terms.height_l1_penalty
    assert height_l1.weight == pytest.approx(7.5)

    # com_upward_velocity: standup variant — ungated, no max_vz cap.
    com_upward = env_cfg.rewards["com_upward_velocity"]
    assert com_upward.func is recovery_terms.com_upward_velocity
    assert com_upward.weight == pytest.approx(0.75)
    assert com_upward.params["max_height"] == pytest.approx(0.125)
    assert "gate_z_below" not in com_upward.params
    assert "max_vz" not in com_upward.params

    gentle = env_cfg.rewards["gentle_rise"]
    assert gentle.func is standup_terms.trunk_vertical_accel_penalty
    assert gentle.weight == pytest.approx(0.005)
    assert gentle.weight > 0.0  # self-negating: positive weight is the penalty

    damping = env_cfg.rewards["arrival_damping"]
    assert damping.func is standup_terms.body_ang_vel_at_height
    assert damping.weight == pytest.approx(0.0)
    assert damping.params["height_low"] == pytest.approx(0.09)
    assert damping.params["height_high"] == pytest.approx(0.11)
    assert damping.params["tilt_full_deg"] == pytest.approx(20.0)
    assert damping.params["tilt_zero_deg"] == pytest.approx(45.0)

    upright_linear = env_cfg.rewards["upright_linear"]
    assert upright_linear.func is standup_terms.body_upright_linear
    assert upright_linear.weight == pytest.approx(1.5)
    assert "gate_z_below" not in upright_linear.params
    upright_sharp = env_cfg.rewards["upright_sharp"]
    assert upright_sharp.func is standup_terms.upright_gaussian_at_height
    assert upright_sharp.weight == pytest.approx(1.5)
    assert upright_sharp.params["std"] == pytest.approx(0.3)
    assert upright_sharp.params["height_low"] == pytest.approx(0.06)
    assert upright_sharp.params["height_high"] == pytest.approx(0.115)

    composite = env_cfg.rewards["standing_composite"]
    assert composite.func is standup_terms.standing_composite_score
    assert composite.weight == pytest.approx(3.75)
    assert composite.params["target_height"] == pytest.approx(0.115)
    assert composite.params["height_std"] == pytest.approx(0.04)
    assert composite.params["upright_std"] == pytest.approx(0.4)
    assert composite.params["pose_std"] == pytest.approx(0.4)

    tracking = env_cfg.rewards["body_pose_tracking"]
    assert tracking.func is standup_terms.body_pose_tracking_locomotion
    assert tracking.weight == pytest.approx(0.0)
    assert tracking.params["command_name"] == "body_pose"
    assert tracking.params["nominal_height"] == pytest.approx(0.115)
    assert tracking.params["z_std"] == pytest.approx(0.01)
    assert tracking.params["angle_std"] == pytest.approx(math.radians(5))
    assert list(tracking.params["axis_weights"]) == [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    assert "vel_gate_command_name" not in tracking.params

    assert env_cfg.rewards["action_rate"].weight == pytest.approx(-0.1)
    torque_rate = env_cfg.rewards["joint_torque_rate_l2"]
    assert torque_rate.func is recovery_terms.joint_torque_rate_l2
    assert torque_rate.weight == pytest.approx(0.0)
    assert env_cfg.rewards["body_ang_vel"].weight == pytest.approx(-0.05)
    assert env_cfg.rewards["angular_momentum"].weight == pytest.approx(-0.02)
    assert env_cfg.rewards["self_collisions"].weight == pytest.approx(-1.0)
    assert env_cfg.rewards["dof_pos_limits"].weight == pytest.approx(-1.0)

    # Commands: near-zero twist, 4D head_pose, 6D body_pose with exact-zero
    # sampling for the deployment idle case.
    twist = env_cfg.commands["twist"]
    assert list(twist.resampling_time_range) == [6.0, 12.0]
    assert twist.heading_command is False
    assert twist.rel_standing_envs == 0.0
    assert twist.rel_forward_envs == 0.0
    assert twist.turn_in_place_fraction == 0.0
    assert tuple(twist.ranges.lin_vel_x) == (-0.01, 0.01)
    assert tuple(twist.ranges.lin_vel_y) == (-0.01, 0.01)
    assert tuple(twist.ranges.ang_vel_z) == (-0.05, 0.05)
    body_pose = env_cfg.commands["body_pose"]
    assert body_pose.zero_command_prob == pytest.approx(0.3)
    assert len(body_pose.ranges) == 6

    # Events: set_ground_state replaces random_prone_init, before push_robot.
    assert tuple(env_cfg.events) == (
        "reset_scene_to_default",
        "reset_base",
        "base_com",
        "head_com",
        "encoder_bias",
        "foot_friction",
        "randomize_armature",
        "randomize_mass_inertia",
        "set_ground_state",
        "push_robot",
    )
    ground = env_cfg.events["set_ground_state"]
    assert ground.func is standup_terms.set_random_ground_state
    assert ground.mode == "reset"
    params = ground.params
    assert params["face_down_prob"] == pytest.approx(0.20)
    assert params["face_up_prob"] == pytest.approx(0.0)
    assert params["sitting_prob"] == pytest.approx(0.40)
    assert params["standing_prob"] == pytest.approx(0.40)
    assert params["prone_z_min"] == pytest.approx(0.05)
    assert params["prone_z_max"] == pytest.approx(0.09)
    assert params["sitting_z_min"] == pytest.approx(0.05)
    assert params["sitting_z_max"] == pytest.approx(0.09)
    assert params["standing_z_min"] == pytest.approx(0.11)
    assert params["standing_z_max"] == pytest.approx(0.12)
    assert params["face_up_roll_max"] == pytest.approx(math.radians(90))
    assert params["sitting_joint_noise_std"] == pytest.approx(0.12)
    assert params["sitting_tilt_max"] == pytest.approx(math.radians(10))
    # Sitting overrides: the sit-policy end pose, keyed by joint name.
    assert {k: float(v) for k, v in params["sitting_joint_overrides"].items()} == {
        name: pytest.approx(angle) for name, angle in SITTING_JOINT_OVERRIDES.items()
    }

    push = env_cfg.events["push_robot"]
    assert push.mode == "interval"
    assert list(push.interval_range_s) == [3.0, 6.0]
    assert push.params["velocity_range"]["x"] == [0.0, 0.0]

    # Terminations: no fell_over / fallen_too_long; nan_state carries the
    # upstream contact-sensor coverage.
    assert tuple(env_cfg.terminations) == ("time_out", "nan_state")
    nan_term = env_cfg.terminations["nan_state"]
    assert nan_term.func is standup_terms.robot_state_is_nan
    assert nan_term.time_out is False
    assert list(nan_term.params["sensor_names"]) == ["left_foot_contact", "right_foot_contact"]

    # Curriculum tables: steps are upstream iterations x 24.
    assert tuple(env_cfg.curriculum) == (
        "ground_state_mix",
        "head_pose_range",
        "base_com_range",
        "head_com_range",
        "push_magnitude",
        "action_rate_weight",
        "arrival_damping_weight",
        "head_pose_bias_weight",
        "torque_rate_weight",
        "body_pose_tracking_weight",
        "body_pose_range",
        "height_stand_sharp_weight",
        "upright_sharp_weight",
        "standing_composite_weight",
    )
    mix = env_cfg.curriculum["ground_state_mix"].params["stages"]
    assert [stage["step"] for stage in mix] == [0, 600 * 24, 1500 * 24, 2500 * 24]
    assert [stage["params"]["standing_prob"] for stage in mix] == [0.40, 0.25, 0.20, 0.15]
    assert [stage["params"]["sitting_prob"] for stage in mix] == [0.40, 0.30, 0.25, 0.20]
    assert [stage["params"]["face_down_prob"] for stage in mix] == [0.20, 0.35, 0.30, 0.30]
    assert [stage["params"]["face_up_prob"] for stage in mix] == [0.00, 0.10, 0.25, 0.35]

    push_stages = env_cfg.curriculum["push_magnitude"].params["stages"]
    assert [stage["step"] for stage in push_stages] == [0, 500 * 24, 1000 * 24]
    assert push_stages[1]["params"]["velocity_range"]["x"] == [-0.08, 0.08]
    assert push_stages[2]["params"]["velocity_range"]["x"] == [-0.3, 0.3]

    action_rate = env_cfg.curriculum["action_rate_weight"].params["stages"]
    assert [stage["step"] for stage in action_rate] == [
        0,
        500 * 24,
        750 * 24,
        1000 * 24,
        1250 * 24,
        1500 * 24,
    ]
    assert [stage["weight"] for stage in action_rate] == [-0.1, -0.2, -0.4, -0.6, -0.8, -1.0]

    for name, steps, weights in (
        ("arrival_damping_weight", [0, 3000 * 24, 4000 * 24], [0.0, -0.025, -0.05]),
        ("head_pose_bias_weight", [0, 3000 * 24, 4000 * 24], [0.0, 0.5, 1.5]),
        ("torque_rate_weight", [0, 3000 * 24], [0.0, -0.001]),
        ("body_pose_tracking_weight", [0, 2500 * 24, 3000 * 24, 4000 * 24], [0.0, 1.5, 3.0, 4.0]),
        ("height_stand_sharp_weight", [0, 3000 * 24, 4000 * 24], [1.0, 0.5, 0.2]),
        ("upright_sharp_weight", [0, 3000 * 24, 4000 * 24], [1.5, 1.0, 0.5]),
        ("standing_composite_weight", [0, 3000 * 24, 4000 * 24], [3.75, 2.5, 1.5]),
    ):
        stages = env_cfg.curriculum[name].params["stages"]
        assert [stage["step"] for stage in stages] == steps, name
        assert [stage["weight"] for stage in stages] == weights, name

    body_ranges = env_cfg.curriculum["body_pose_range"].params["stages"]
    assert [stage["step"] for stage in body_ranges] == [0, 2500 * 24, 3000 * 24, 4000 * 24]
    final_ranges = body_ranges[-1]["ranges"]
    assert final_ranges[2] == [-0.04, 0.03]
    assert final_ranges[3][0] == pytest.approx(-math.radians(15))
    assert final_ranges[3][1] == pytest.approx(math.radians(15))
    assert final_ranges[5] == [-0.05, 0.05]  # yaw stays at the alive range


def test_standup_sitting_overrides_resolve_against_the_model() -> None:
    """The sitting override joint names exist on the groundcontact BAM model."""
    root = ET.parse(ASSET_DIR / "microduck_groundcontact_bam.xml").getroot()
    joint_names = {joint.get("name") for joint in root.iter("joint") if joint.get("name")}
    assert set(SITTING_JOINT_OVERRIDES) <= joint_names
    assert set(LEG_JOINTS) <= joint_names
    # The 14-servo layout: 0-4 left leg, 5-8 neck/head, 9-13 right leg.
    servo_names = [
        joint.get("name")
        for joint in root.find("worldbody").iter("joint")
        if joint.get("name") and not joint.get("name").startswith("passive_")
    ]
    assert servo_names == [
        "left_hip_yaw",
        "left_hip_roll",
        "left_hip_pitch",
        "left_knee",
        "left_ankle",
        "neck_pitch",
        "head_pitch",
        "head_yaw",
        "head_roll",
        "right_hip_yaw",
        "right_hip_roll",
        "right_hip_pitch",
        "right_knee",
        "right_ankle",
    ]


def test_standup_registry_entry_is_mujoco_only() -> None:
    registry.ensure_registries()
    assert registry.list_registered_envs()["MicroduckStandupFlat"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mujoco"],
    }


# ── Term-level semantics (mock envs, upstream mdp.py parity) ────────────────

JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)
LEG_JOINT_IDS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]


def _tilt_quat(tilt_deg: float, num_envs: int) -> np.ndarray:
    half = math.radians(tilt_deg) / 2.0
    quat = np.zeros((num_envs, 4))
    quat[:, 0] = math.cos(half)
    quat[:, 1] = math.sin(half)
    return quat


class _MockEntity:
    def __init__(self, num_envs: int, z: list[float], tilt_deg: list[float]) -> None:
        self.num_joints = 14
        pos = np.zeros((num_envs, 3))
        pos[:, 2] = z
        quat = np.concatenate([_tilt_quat(tilt, 1) for tilt in tilt_deg], axis=0)
        self.data = SimpleNamespace(
            root_link_pos_w=pos,
            root_link_quat_w=quat,
            root_link_lin_vel_w=np.zeros((num_envs, 3)),
            root_link_ang_vel_w=np.zeros((num_envs, 3)),
            joint_pos=np.zeros((num_envs, 14)),
            joint_vel=np.zeros((num_envs, 14)),
            default_joint_pos=np.zeros((num_envs, 14)),
            body_link_pos_w=np.zeros((num_envs, 8, 3)),
            body_link_quat_w=np.zeros((num_envs, 8, 4)),
            body_link_ang_vel_w=np.zeros((num_envs, 8, 3)),
        )
        self.data.body_link_quat_w[..., 0] = 1.0

    def find_joints(self, keys, preserve_order: bool = False):
        key = keys.strip("^$")
        return [JOINT_NAMES.index(key)], [key]


def _standup_env(
    z: list[float],
    tilt_deg: list[float],
    *,
    step_dt: float = 0.1,
    episode_length: int = 10,
    commands: dict[str, np.ndarray] | None = None,
    sensor_data: dict[str, np.ndarray] | None = None,
) -> ManagerBasedRlEnv:
    num_envs = len(z)
    entity = _MockEntity(num_envs, z, tilt_deg)

    class _Scene:
        def __init__(self) -> None:
            self.env_origins = np.zeros((num_envs, 3))

        def __getitem__(self, name: str):
            assert name == "robot"
            return entity

        def bind_sensor_data(self, names: tuple[str, ...]):
            arrays = [(sensor_data or {})[name] for name in names]
            return SimpleNamespace(
                dimensions=tuple(array.shape[1] for array in arrays),
                backend_type="fake",
                read=lambda: np.concatenate(arrays, axis=1),
            )

    return cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=num_envs,
            step_dt=step_dt,
            episode_length_buf=np.full(num_envs, episode_length, dtype=np.int64),
            scene=_Scene(),
            command_manager=SimpleNamespace(get_command=lambda name: (commands or {})[name]),
        ),
    )


def _term(term_type: type, env: ManagerBasedRlEnv, **params: Any):
    return term_type(RewardTermCfg(func=term_type, weight=1.0, params=params), env)


def test_pose_target_match_gaussian_and_l1_sign() -> None:
    env = _standup_env([0.115, 0.115], [0.0, 0.0])
    entity = env.scene["robot"]
    entity.data.joint_pos[:, LEG_JOINT_IDS] = 0.0
    entity.data.joint_pos[1, LEG_JOINT_IDS] = 0.5
    asset_cfg = SceneEntityCfg("robot", joint_ids=LEG_JOINT_IDS)

    gaussian = _term(standup_terms.pose_target_match, env, asset_cfg=asset_cfg, std=0.5)
    np.testing.assert_allclose(gaussian(env), [1.0, math.exp(-1.0)], atol=1e-6)
    l1 = _term(standup_terms.pose_l1_penalty, env, asset_cfg=asset_cfg)
    np.testing.assert_allclose(l1(env), [0.0, -0.5], atol=1e-6)
    assert np.all(l1(env) <= 0.0)  # self-negating contract

    # Overrides shift the target for the named joints only.
    entity.data.joint_pos[:, :] = 0.0
    entity.data.joint_pos[:, 3] = 1.35  # left_knee
    overridden = _term(
        standup_terms.pose_l1_penalty,
        env,
        asset_cfg=asset_cfg,
        target_overrides={"left_knee": 1.35},
    )
    np.testing.assert_allclose(overridden(env), [0.0, 0.0], atol=1e-6)
    with pytest.raises(ValueError, match="not in the"):
        _term(
            standup_terms.pose_l1_penalty,
            env,
            asset_cfg=asset_cfg,
            target_overrides={"neck_pitch": 0.1},
        )


def test_height_terms_track_target() -> None:
    env = _standup_env([0.115, 0.075], [0.0, 0.0])
    gaussian = standup_terms.height_target_gaussian(
        env, target_height=0.115, std=0.04, asset_cfg=SceneEntityCfg("robot")
    )
    np.testing.assert_allclose(gaussian, [1.0, math.exp(-1.0)], atol=1e-6)
    l1 = standup_terms.height_l1_penalty(
        env, target_height=0.115, asset_cfg=SceneEntityCfg("robot")
    )
    np.testing.assert_allclose(l1, [0.0, -0.04], atol=1e-6)


def test_trunk_vertical_accel_penalty_is_self_negating_and_reset_safe() -> None:
    env = _standup_env([0.06, 0.06], [0.0, 0.0])
    term = _term(standup_terms.trunk_vertical_accel_penalty, env, asset_cfg=SceneEntityCfg("robot"))
    # First call baselines: zero.
    np.testing.assert_allclose(term(env), [0.0, 0.0], atol=1e-6)
    entity = env.scene["robot"]
    entity.data.root_link_lin_vel_w[:, 2] = [0.2, -0.1]
    # a_z = dvz / step_dt = [2.0, -1.0] -> -|a_z|.
    np.testing.assert_allclose(term(env), [-2.0, -1.0], atol=1e-6)
    # Freshly reset envs pay nothing even with a velocity jump.
    env.episode_length_buf[:] = 0
    entity.data.root_link_lin_vel_w[:, 2] = [0.5, 0.5]
    np.testing.assert_allclose(term(env), [0.0, 0.0], atol=1e-6)


def test_body_ang_vel_at_height_gates_on_z_and_tilt() -> None:
    env = _standup_env([0.12, 0.05, 0.12, 0.12], [0.0, 0.0, 50.0, 10.0])
    entity = env.scene["robot"]
    entity.data.body_link_ang_vel_w[:, 0, :2] = 1.0  # cost 2.0 everywhere
    cost = standup_terms.body_ang_vel_at_height(
        env,
        height_low=0.09,
        height_high=0.11,
        tilt_full_deg=20.0,
        tilt_zero_deg=45.0,
        asset_cfg=SceneEntityCfg("robot", body_ids=[0]),
    )
    # High + vertical: full cost; low: zeroed; tilt 50 deg > 45: zeroed;
    # tilt 10 deg <= 20: full cost.
    np.testing.assert_allclose(cost, [2.0, 0.0, 0.0, 2.0], atol=1e-6)
    assert np.all(cost >= 0.0)  # positive cost, negative weight in config


def test_upright_linear_and_gaussian_at_height() -> None:
    env = _standup_env([0.115, 0.115, 0.06], [0.0, 90.0, 0.0])
    linear = standup_terms.body_upright_linear(env, asset_cfg=SceneEntityCfg("robot"))
    np.testing.assert_allclose(linear, [1.0, 0.0, 1.0], atol=1e-6)
    # Gated variant (velstand parity): only fallen envs score.
    gated = standup_terms.body_upright_linear(
        env,
        asset_cfg=SceneEntityCfg("robot"),
        gate_z_below=0.08,
        gate_tilt_above_deg=40.0,
    )
    np.testing.assert_allclose(gated, [0.0, 0.0, 1.0], atol=1e-6)
    # Gaussian at height: the low vertical crouch (env 2, z = height_low) is
    # gated to zero; the prone env scores the Gaussian but stays low-gated too.
    sharp = standup_terms.upright_gaussian_at_height(
        env,
        std=0.3,
        height_low=0.06,
        height_high=0.115,
        asset_cfg=SceneEntityCfg("robot"),
    )
    assert sharp[0] == pytest.approx(1.0)
    assert sharp[1] < 1e-4  # 90 deg tilt: Gaussian ~exp(-11), height gate open
    assert sharp[2] == pytest.approx(0.0, abs=1e-6)


def test_standing_composite_collapses_on_any_factor() -> None:
    env = _standup_env([0.115, 0.075], [0.0, 0.0])
    composite = _term(
        standup_terms.standing_composite_score,
        env,
        target_height=0.115,
        height_std=0.04,
        upright_std=0.4,
        pose_std=0.4,
        asset_cfg=SceneEntityCfg("robot", joint_ids=LEG_JOINT_IDS),
    )
    score = composite(env)
    # Goal state scores 1.0; a 4 cm height deficit collapses the product even
    # with perfect upright/pose.
    np.testing.assert_allclose(score, [1.0, math.exp(-1.0)], atol=1e-6)


def test_body_pose_tracking_locomotion_tracks_z_roll_pitch() -> None:
    num_envs = 3
    commands = {"body_pose": np.zeros((num_envs, 6))}
    # env 2 is rotated 5 deg about x (roll == tilt for an x-axis rotation).
    env = _standup_env([0.115, 0.125, 0.115], [0.0, 0.0, 5.0], commands=commands)
    tracking = standup_terms.body_pose_tracking_locomotion(
        env,
        command_name="body_pose",
        nominal_height=0.115,
        z_std=0.01,
        angle_std=math.radians(5),
        axis_weights=(0.0, 0.0, 1.0, 1.0, 1.0, 0.0),
        asset_cfg=SceneEntityCfg("robot"),
        feet_cfg=SceneEntityCfg("robot", body_ids=[1, 2]),
    )
    # env 0 at the nominal stand: 1.0; env 1 is 1 cm high: r_z = e^-1;
    # env 2 rolled 5 deg about x: r_r = e^-1.
    np.testing.assert_allclose(
        tracking, [1.0, (math.exp(-1.0) + 2.0) / 3.0, (math.exp(-1.0) + 2.0) / 3.0], atol=1e-6
    )
    # The feet reference only feeds zero-weighted axes here: moving the ankle
    # bodies cannot change the reward.
    env.scene["robot"].data.body_link_pos_w[:, 1:3, :] = 0.5
    np.testing.assert_allclose(
        standup_terms.body_pose_tracking_locomotion(
            env,
            command_name="body_pose",
            nominal_height=0.115,
            z_std=0.01,
            angle_std=math.radians(5),
            axis_weights=(0.0, 0.0, 1.0, 1.0, 1.0, 0.0),
            asset_cfg=SceneEntityCfg("robot"),
            feet_cfg=SceneEntityCfg("robot", body_ids=[1, 2]),
        ),
        tracking,
        atol=1e-6,
    )


def test_robot_state_is_nan_covers_state_and_sensor_forces() -> None:
    sensors = {
        "left_foot_contact": np.zeros((2, 3)),
        "right_foot_contact": np.zeros((2, 3)),
    }
    env = _standup_env([0.115, 0.115], [0.0, 0.0], sensor_data=sensors)
    term = standup_terms.robot_state_is_nan(
        TerminationTermCfg(
            func=standup_terms.robot_state_is_nan,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_names": ["left_foot_contact", "right_foot_contact"],
            },
        ),
        env,
    )
    np.testing.assert_array_equal(term(env), [False, False])
    # NaN joint state fires.
    env.scene["robot"].data.joint_pos[1, 3] = np.nan
    np.testing.assert_array_equal(term(env), [False, True])
    env.scene["robot"].data.joint_pos[1, 3] = 0.0
    # NaN contact force fires a step before the integrated state diverges.
    sensors["left_foot_contact"][0, 0] = np.inf
    np.testing.assert_array_equal(term(env), [True, False])


# ── set_random_ground_state reset event ──────────────────────────────────────


class _RecordingEntity:
    """Entity mock recording reset-transaction writes for set_ground_state."""

    def __init__(self, num_envs: int) -> None:
        self.poses = np.zeros((num_envs, 7))
        self.poses[:, 3] = 1.0
        self.poses[:, 0] = np.arange(num_envs) * 0.1  # staged reset_base x
        self.poses[:, 2] = 0.125
        # Distinct per-joint defaults so overrides are detectable.
        defaults = np.tile(np.arange(14, dtype=np.float64) * 0.01, (num_envs, 1))
        self.data = SimpleNamespace(default_joint_pos=defaults)
        self.pose_writes: list[tuple[np.ndarray, np.ndarray]] = []
        self.velocity_writes: list[tuple[np.ndarray, np.ndarray]] = []
        self.joint_writes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def read_reset_root_pose(self, env_ids=None) -> np.ndarray:
        return self.poses[env_ids].copy()

    def write_root_link_pose_to_sim(self, pose, env_ids=None) -> None:
        self.poses[env_ids] = pose
        self.pose_writes.append((np.asarray(env_ids), pose.copy()))

    def write_root_link_velocity_to_sim(self, velocity, env_ids=None) -> None:
        self.velocity_writes.append((np.asarray(env_ids), velocity.copy()))

    def write_joint_state_to_sim(self, position, velocity, joint_ids=None, env_ids=None) -> None:
        self.joint_writes.append((np.asarray(env_ids), position.copy(), velocity.copy()))

    def find_joints(self, keys, preserve_order: bool = False):
        key = keys.strip("^$")
        return [JOINT_NAMES.index(key)], [key]


def _event_env(num_envs: int = 8) -> tuple[ManagerBasedRlEnv, _RecordingEntity]:
    entity = _RecordingEntity(num_envs)

    class _Scene:
        env_origins = np.zeros((num_envs, 3))

        def __getitem__(self, name: str):
            assert name == "robot"
            return entity

    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=num_envs,
            rng=np.random.default_rng(0),
            scene=_Scene(),
        ),
    )
    return env, entity


def test_ground_state_validates_probabilities() -> None:
    env, _ = _event_env()
    with pytest.raises(ValueError, match="must not all be zero"):
        standup_terms.set_random_ground_state(
            env,
            np.arange(8, dtype=np.int32),
            face_down_prob=0.0,
            face_up_prob=0.0,
            sitting_prob=0.0,
            standing_prob=0.0,
        )


def test_ground_state_face_down_bucket() -> None:
    env, entity = _event_env()
    ids = np.arange(8, dtype=np.int32)
    standup_terms.set_random_ground_state(
        env,
        ids,
        face_down_prob=1.0,
        face_up_prob=0.0,
        sitting_prob=0.0,
        standing_prob=0.0,
        prone_z_min=0.05,
        prone_z_max=0.09,
    )
    quat = entity.poses[:, 3:7]
    # Face-down family [s*cy, -s*sy, s*cy, s*sy]: qw == qy, qx == -qz.
    np.testing.assert_allclose(quat[:, 0], quat[:, 2], atol=1e-7)
    np.testing.assert_allclose(quat[:, 1], -quat[:, 3], atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(quat, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), 0.0, atol=1e-6)
    assert np.all(entity.poses[:, 2] >= 0.05) and np.all(entity.poses[:, 2] <= 0.09)
    # Staged xy preserved; velocities zeroed; joints untouched (no write).
    np.testing.assert_allclose(entity.poses[:, 0], np.arange(8) * 0.1, atol=1e-7)
    assert len(entity.velocity_writes) == 1
    np.testing.assert_array_equal(entity.velocity_writes[0][1], np.zeros((8, 6)))
    assert entity.joint_writes == []


def test_ground_state_face_up_bucket_roll_noise() -> None:
    env, entity = _event_env()
    ids = np.arange(8, dtype=np.int32)
    standup_terms.set_random_ground_state(
        env,
        ids,
        face_down_prob=0.0,
        face_up_prob=1.0,
        sitting_prob=0.0,
        standing_prob=0.0,
        prone_z_min=0.05,
        prone_z_max=0.09,
        face_up_roll_max=math.radians(90),
    )
    quat = entity.poses[:, 3:7]
    np.testing.assert_allclose(np.linalg.norm(quat, axis=1), 1.0, atol=1e-6)
    # Supine family with a body-long-axis roll: trunk stays horizontal
    # (cos tilt ~ 0) regardless of the roll angle.
    np.testing.assert_allclose(1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), 0.0, atol=1e-6)
    assert np.all(entity.poses[:, 2] >= 0.05) and np.all(entity.poses[:, 2] <= 0.09)
    assert entity.joint_writes == []


def test_ground_state_sitting_bucket_applies_overrides_and_noise() -> None:
    env, entity = _event_env()
    ids = np.arange(8, dtype=np.int32)
    standup_terms.set_random_ground_state(
        env,
        ids,
        face_down_prob=0.0,
        face_up_prob=0.0,
        sitting_prob=1.0,
        standing_prob=0.0,
        sitting_z_min=0.05,
        sitting_z_max=0.09,
        sitting_joint_overrides=SITTING_JOINT_OVERRIDES,
        sitting_joint_noise_std=0.0,
        sitting_tilt_max=0.0,
    )
    # Upright yaw-only quats, sitting z band, one joint write.
    quat = entity.poses[:, 3:7]
    np.testing.assert_allclose(quat[:, 1:3], 0.0, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(quat, axis=1), 1.0, atol=1e-6)
    assert np.all(entity.poses[:, 2] >= 0.05) and np.all(entity.poses[:, 2] <= 0.09)
    assert len(entity.joint_writes) == 1
    write_ids, position, velocity = entity.joint_writes[0]
    np.testing.assert_array_equal(write_ids, ids)
    np.testing.assert_array_equal(velocity, np.zeros((8, 14)))
    defaults = np.arange(14, dtype=np.float64) * 0.01
    expected = np.tile(defaults, (8, 1))
    for name, angle in SITTING_JOINT_OVERRIDES.items():
        expected[:, JOINT_NAMES.index(name)] = angle
    np.testing.assert_allclose(position, expected, atol=1e-9)

    # Gaussian noise on every joint when the std is non-zero.
    env2, entity2 = _event_env()
    standup_terms.set_random_ground_state(
        env2,
        ids,
        face_down_prob=0.0,
        face_up_prob=0.0,
        sitting_prob=1.0,
        standing_prob=0.0,
        sitting_joint_overrides=SITTING_JOINT_OVERRIDES,
        sitting_joint_noise_std=0.12,
    )
    noisy = entity2.joint_writes[0][1]
    # Untouched joints are default + zero-mean noise; overridden joints center
    # on the override value with the same noise spread.
    knee = noisy[:, JOINT_NAMES.index("left_knee")]
    assert abs(float(np.mean(knee)) - 1.35) < 0.1
    assert float(np.std(knee)) > 0.05
    hip_yaw = noisy[:, JOINT_NAMES.index("left_hip_yaw")]
    assert abs(float(np.mean(hip_yaw)) - 0.0) < 0.1
    assert float(np.std(hip_yaw)) > 0.05


def test_ground_state_standing_bucket_keeps_home_joints() -> None:
    env, entity = _event_env()
    ids = np.arange(8, dtype=np.int32)
    standup_terms.set_random_ground_state(
        env,
        ids,
        face_down_prob=0.0,
        face_up_prob=0.0,
        sitting_prob=0.0,
        standing_prob=1.0,
        standing_z_min=0.11,
        standing_z_max=0.12,
        sitting_tilt_max=math.radians(10),
    )
    assert np.all(entity.poses[:, 2] >= 0.11) and np.all(entity.poses[:, 2] <= 0.12)
    np.testing.assert_allclose(np.linalg.norm(entity.poses[:, 3:7], axis=1), 1.0, atol=1e-6)
    # Pitch and roll are each bounded at ±10 deg, so the combined tilt is
    # bounded at sqrt(2) * 10 deg.
    quat = entity.poses[:, 3:7]
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    assert np.all(cos_tilt >= math.cos(math.radians(10) * math.sqrt(2)) - 1e-6)
    # Joints stay at the staged HOME keyframe: no joint write at all.
    assert entity.joint_writes == []


@pytest.mark.slow
def test_standup_owner_builds_and_steps_real_mujoco_env() -> None:
    pytest.importorskip("mujoco")
    cfg, _ = _materialize_standup()
    env = cast(
        Any,
        registry.make(
            "MicroduckStandupFlat",
            sim_backend="mujoco",
            num_envs=2,
            env_cfg_override=BackendAdapter(
                cfg,
                root_dir=ROOT_DIR,
                algo_name="ppo",
            ).build_task_env_cfg_override(),
        ),
    )
    try:
        obs, info = env.reset()
        assert isinstance(info, dict)
        assert env.action_space.shape == (MICRODUCK_NUM_ACTION,)
        assert obs["obs"].shape == (2, MICRODUCK_ACTOR_OBS_DIM)
        assert obs["critic"].shape == (2, STANDUP_CRITIC_OBS_DIM)

        for _ in range(3):
            state = env.step(np.zeros((2, MICRODUCK_NUM_ACTION), dtype=np.float32))
        assert np.isfinite(state.obs["obs"]).all()
        assert np.isfinite(state.obs["critic"]).all()
        assert np.isfinite(state.reward).all()
    finally:
        env.close()
