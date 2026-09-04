"""Hydra, Registry, and term-level contracts for MicroduckVelstandFlat.

Locks the port of the upstream microduck_rl velstand recipe (anchor commit
29e887e): the ground-contact asset delta, the recovery reward/termination/
event/curriculum layer on top of the BAM velocity base, and the NumPy
semantics of the stateful recovery terms.
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
from unilab.managers import RewardTermCfg, SceneEntityCfg
from unilab.managers._types import ManagerBasedRlEnv
from unilab.tasks.locomotion.microduck import recovery_terms
from unilab.tasks.locomotion.microduck.bam_action import BamVoltageAction
from unilab.tasks.locomotion.microduck.deploy_contract import (
    MICRODUCK_ACTOR_OBS_DIM,
    MICRODUCK_CRITIC_OBS_DIM,
    MICRODUCK_NUM_ACTION,
)

ROOT_DIR = Path(__file__).parents[4]
CONF_DIR = ROOT_DIR / "src" / "unilab" / "conf" / "ppo"
ASSET_DIR = ROOT_DIR / "src" / "unilab" / "assets" / "robots" / "microduck"


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


def _materialize_velstand() -> tuple[Any, Any]:
    return _materialize("microduck_velstand_flat/mujoco", "MicroduckVelstandFlat")


def test_velstand_owner_materializes_recovery_layer() -> None:
    cfg, env_cfg = _materialize_velstand()

    assert cfg.training.task_name == "MicroduckVelstandFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.algo.seed == 42
    assert cfg.algo.num_envs == 4096
    assert cfg.algo.max_iterations == 20000
    assert cfg.algo.save_interval == 250
    assert cfg.algo.experiment_name == "velstand"
    assert cfg.algo.run_name == "velstand"
    assert cfg.algo.num_steps_per_env == 24
    assert cfg.algo.empirical_normalization is True
    assert cfg.algo.obs_groups.actor == ["policy"]
    assert cfg.algo.obs_groups.critic == ["critic"]
    assert cfg.env.seed == 42

    assert env_cfg.scene.model_file.endswith("robots/microduck/scene_flat_groundcontact_bam.xml")
    assert env_cfg.scene.fragment_files[0].endswith("robots/microduck/locomotion_task.xml")
    assert env_cfg.scene.default_keyframe_name == "home"

    # Obs layout stays the shared 61D/76D contract (policy hot-swap invariant).
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
        "foot_height",
        "foot_air_time",
        "foot_contact",
        "foot_contact_forces",
    )

    # The walk layer is verbatim; the recovery layer appends after action_rate.
    assert tuple(env_cfg.rewards) == (
        "tracking_lin_vel",
        "tracking_ang_vel",
        "upright",
        "head_pose_tracking",
        "head_pose_bias",
        "body_pose_tracking",
        "leg_pose",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "body_ang_vel",
        "self_collisions",
        "dof_pos_limits",
        "angular_momentum",
        "action_rate",
        "upright_progress",
        "height_progress",
        "com_upward_velocity",
        "joint_torque_rate_l2",
        "fallen_tax",
        "recovery_success",
    )

    # Sign convention: shaping terms positive, taxes/jackpot-guards as upstream.
    assert env_cfg.rewards["upright_progress"].func is recovery_terms.upright_progress
    assert env_cfg.rewards["upright_progress"].weight == pytest.approx(5.0)
    assert env_cfg.rewards["height_progress"].func is recovery_terms.height_progress
    assert env_cfg.rewards["height_progress"].weight == pytest.approx(30.0)
    assert env_cfg.rewards["height_progress"].params["ceiling"] == pytest.approx(0.115)
    com_upward = env_cfg.rewards["com_upward_velocity"]
    assert com_upward.func is recovery_terms.com_upward_velocity
    assert com_upward.weight == pytest.approx(0.0)
    assert com_upward.params["max_height"] == pytest.approx(0.125)
    assert com_upward.params["gate_z_below"] == pytest.approx(0.0)
    assert com_upward.params["gate_tilt_above_deg"] == pytest.approx(40.0)
    assert env_cfg.rewards["joint_torque_rate_l2"].func is recovery_terms.joint_torque_rate_l2
    assert env_cfg.rewards["joint_torque_rate_l2"].weight == pytest.approx(-2.0e-3)

    # air_time -> feet_air_time_upright, same weight/window plus the tilt gate.
    air_time = env_cfg.rewards["air_time"]
    assert air_time.func is recovery_terms.feet_air_time_upright
    assert air_time.weight == pytest.approx(3.0)
    assert air_time.params["threshold_min"] == pytest.approx(0.125)
    assert air_time.params["threshold_max"] == pytest.approx(0.3)
    assert air_time.params["gate_tilt_above_deg"] == pytest.approx(40.0)

    # head_pose_bias carries the velstand upright gate.
    bias = env_cfg.rewards["head_pose_bias"]
    assert bias.weight == pytest.approx(0.0)
    assert bias.params["gate_height_low"] == pytest.approx(0.09)
    assert bias.params["gate_height_high"] == pytest.approx(0.11)
    assert bias.params["gate_tilt_full_deg"] == pytest.approx(20.0)
    assert bias.params["gate_tilt_zero_deg"] == pytest.approx(40.0)

    # Recovery economics: hysteretic tax + one-shot bounty, weights ramped in.
    fallen_tax = env_cfg.rewards["fallen_tax"]
    assert fallen_tax.func is recovery_terms.fallen_state_penalty
    assert fallen_tax.weight == pytest.approx(0.0)
    assert fallen_tax.params["gate_tilt_above_deg"] == pytest.approx(40.0)
    assert fallen_tax.params["release_tilt_below_deg"] == pytest.approx(25.0)
    assert fallen_tax.params["release_z_above"] == pytest.approx(0.09)
    recovery = env_cfg.rewards["recovery_success"]
    assert recovery.func is recovery_terms.recovery_success
    assert recovery.weight == pytest.approx(0.0)
    assert recovery.params["fallen_tilt_deg"] == pytest.approx(40.0)
    assert recovery.params["min_fallen_s"] == pytest.approx(0.5)
    assert recovery.params["up_tilt_deg"] == pytest.approx(25.0)
    assert recovery.params["up_z"] == pytest.approx(0.09)

    # Prone/crouch reset event sits after the DR events, before push_robot.
    assert tuple(env_cfg.events) == (
        "reset_scene_to_default",
        "reset_base",
        "base_com",
        "head_com",
        "encoder_bias",
        "foot_friction",
        "randomize_armature",
        "randomize_mass_inertia",
        "random_prone_init",
        "push_robot",
    )
    prone = env_cfg.events["random_prone_init"]
    assert prone.func is recovery_terms.maybe_set_random_prone_orientation
    assert prone.mode == "reset"
    assert prone.params["prone_prob"] == pytest.approx(0.0)
    assert prone.params["face_down_prob"] == pytest.approx(1.0)
    assert prone.params["prone_z_min"] == pytest.approx(0.05)
    assert prone.params["prone_z_max"] == pytest.approx(0.09)
    assert prone.params["crouch_prob"] == pytest.approx(0.0)

    # Terminations: tilt stays at the legacy 70 deg until the curriculum
    # disables it; fallen_too_long is the non-timeout recycling backstop.
    assert tuple(env_cfg.terminations) == ("time_out", "tilt", "fallen_too_long", "nan_state")
    assert env_cfg.terminations["tilt"].params["limit_angle"] == pytest.approx(1.2217304763960306)
    fallen_term = env_cfg.terminations["fallen_too_long"]
    assert fallen_term.func is recovery_terms.fallen_too_long
    assert fallen_term.time_out is False
    assert fallen_term.params["gate_z_below"] == pytest.approx(0.08)
    assert fallen_term.params["gate_tilt_above_deg"] == pytest.approx(40.0)
    assert fallen_term.params["max_duration_s"] == pytest.approx(8.0)

    # Velstand curricula precede the legacy walk curricula; stage steps are
    # upstream iterations x 24 (num_steps_per_env).
    assert tuple(env_cfg.curriculum) == (
        "fell_over_disable",
        "prone_init_prob",
        "fallen_tax_weight",
        "recovery_success_weight",
        "com_upward_weight",
        "action_rate_weight",
        "head_pose_bias_weight",
        "standing_envs",
        "head_pose_range",
        "base_com_range",
        "head_com_range",
    )
    fell_over = env_cfg.curriculum["fell_over_disable"].params["stages"]
    assert fell_over == [
        {"step": 0, "params": {"limit_angle": pytest.approx(1.2217304763960306)}},
        {"step": 500 * 24, "params": {"limit_angle": pytest.approx(math.pi)}},
    ]
    prone_stages = env_cfg.curriculum["prone_init_prob"].params["stages"]
    assert [stage["step"] for stage in prone_stages] == [
        0,
        800 * 24,
        1500 * 24,
        2000 * 24,
        2500 * 24,
    ]
    assert [stage["params"]["prone_prob"] for stage in prone_stages] == [0.0, 0.0, 0.15, 0.30, 0.45]
    assert [stage["params"]["face_down_prob"] for stage in prone_stages] == [
        1.0,
        1.0,
        0.80,
        0.65,
        0.50,
    ]
    assert [stage["params"]["crouch_prob"] for stage in prone_stages] == [
        0.0,
        0.15,
        0.15,
        0.15,
        0.15,
    ]
    for name, final_weight in (
        ("fallen_tax_weight", -0.5),
        ("recovery_success_weight", 10.0),
        ("com_upward_weight", 2.0),
    ):
        stages = env_cfg.curriculum[name].params["stages"]
        assert [stage["step"] for stage in stages] == [0, 1200 * 24]
        assert [stage["weight"] for stage in stages] == [0.0, final_weight]


def test_velocity_owners_keep_head_pose_bias_ungated() -> None:
    """Regression: the gate params default to off so the walk tasks are unchanged."""
    for task, task_name in (
        ("microduck_velocity_flat/mujoco", "MicroduckVelocityFlat"),
        ("microduck_velocity_bam_flat/mujoco", "MicroduckVelocityBamFlat"),
    ):
        _, env_cfg = _materialize(task, task_name)
        params = env_cfg.rewards["head_pose_bias"].params
        assert "gate_height_low" not in params
        assert "gate_tilt_zero_deg" not in params


def test_velstand_registry_entry_is_mujoco_only() -> None:
    registry.ensure_registries()
    assert registry.list_registered_envs()["MicroduckVelstandFlat"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mujoco"],
    }


def test_groundcontact_xml_carries_full_body_collisions() -> None:
    root = ET.parse(ASSET_DIR / "microduck_groundcontact.xml").getroot()
    assert root.get("model") == "microduck_groundcontact"

    geoms = {geom.get("name"): geom for geom in root.iter("geom") if geom.get("name") is not None}
    # The six upstream walk->groundcontact additions land as collision geoms.
    for name in (
        "np_f970_collision",
        "left_hip_collision",
        "right_hip_collision",
        "top_head_shell_collision",
        "jaw_collision",
        "bottom_head_shell_collision",
    ):
        assert name in geoms, name
        assert geoms[name].get("class") == "collision"
    # The leg geoms keep their self_collision names (locomotion_task.xml
    # sensors resolve against them) but move to the collision class.
    assert geoms["left_leg_self_collision"].get("class") == "collision"
    assert geoms["right_leg_self_collision"].get("class") == "collision"
    assert geoms["trunk_base_self_collision"].get("class") == "self_collision_only"

    # The collision geoms sit on the bodies the upstream model attaches them to.
    body_geoms: dict[str, set[str]] = {}
    for body in root.iter("body"):
        body_geoms[body.get("name")] = {
            geom.get("name") for geom in body.findall("geom") if geom.get("name")
        }
    assert "np_f970_collision" in body_geoms["trunk_base"]
    assert "left_hip_collision" in body_geoms["hip_l"]
    assert "right_hip_collision" in body_geoms["hip_l_2"]
    for name in ("top_head_shell_collision", "jaw_collision", "bottom_head_shell_collision"):
        assert name in body_geoms["jaw_soft"]


def test_groundcontact_bam_xml_applies_the_bam_actuator_transform() -> None:
    root = ET.parse(ASSET_DIR / "microduck_groundcontact_bam.xml").getroot()
    assert root.get("model") == "microduck_groundcontact_bam"

    actuator = root.find("actuator")
    assert actuator is not None
    motors = actuator.findall("motor")
    assert len(motors) == 14
    assert all(motor.get("class") == "chosen_actuator" for motor in motors)
    assert actuator.findall("position") == []

    for default in root.iter("default"):
        if default.get("class") == "chosen_actuator":
            joint = default.find("joint")
            assert joint is not None
            assert joint.get("damping") == "0.0"
            assert joint.get("frictionloss") == "0.0"
            motor = default.find("motor")
            assert motor is not None
            assert motor.get("forcerange") == "-1.0676 1.0676"
            break
    else:
        raise AssertionError("chosen_actuator default class not found")

    scene = ET.parse(ASSET_DIR / "scene_flat_groundcontact_bam.xml").getroot()
    include = scene.find("include")
    assert include is not None
    assert include.get("file") == "microduck_groundcontact_bam.xml"


# ── Term-level semantics (mock envs, upstream mdp.py parity) ────────────────


def _tilt_quat(tilt_deg: float, num_envs: int) -> np.ndarray:
    """Quaternion pitched ``tilt_deg`` about x: cos_tilt == cos(tilt_deg)."""
    half = math.radians(tilt_deg) / 2.0
    quat = np.zeros((num_envs, 4))
    quat[:, 0] = math.cos(half)
    quat[:, 1] = math.sin(half)
    return quat


def _recovery_env(
    z: list[float],
    tilt_deg: list[float],
    vz: list[float] | None = None,
    *,
    step_dt: float = 0.1,
    episode_length: int = 10,
) -> ManagerBasedRlEnv:
    num_envs = len(z)
    pos = np.zeros((num_envs, 3))
    pos[:, 2] = z
    lin_vel = np.zeros((num_envs, 3))
    if vz is not None:
        lin_vel[:, 2] = vz
    quat = np.concatenate([_tilt_quat(tilt, 1) for tilt in tilt_deg], axis=0)
    entity = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=pos,
            root_link_quat_w=quat,
            root_link_lin_vel_w=lin_vel,
        )
    )

    class _Scene:
        def __init__(self) -> None:
            self.env_origins = np.zeros((num_envs, 3))

        def __getitem__(self, name: str):
            assert name == "robot"
            return entity

    return cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=num_envs,
            step_dt=step_dt,
            episode_length_buf=np.full(num_envs, episode_length, dtype=np.int64),
            scene=_Scene(),
        ),
    )


def _set_pose(env: ManagerBasedRlEnv, z: list[float], tilt_deg: list[float]) -> None:
    entity = env.scene["robot"]
    entity.data.root_link_pos_w[:, 2] = z
    entity.data.root_link_quat_w = np.concatenate([_tilt_quat(t, 1) for t in tilt_deg], axis=0)


def _term(term_type: type, env: ManagerBasedRlEnv, **params: Any):
    return term_type(RewardTermCfg(func=term_type, weight=1.0, params=params), env)


def test_fallen_state_penalty_hysteresis() -> None:
    env = _recovery_env([0.12, 0.12], [0.0, 0.0])
    term = _term(
        recovery_terms.fallen_state_penalty,
        env,
        asset_cfg=SceneEntityCfg("robot"),
        gate_tilt_above_deg=40.0,
        release_tilt_below_deg=25.0,
        release_z_above=0.09,
    )
    # Nobody fallen -> no tax.
    np.testing.assert_array_equal(term(env), [0.0, 0.0])
    # Env 0 falls past 40 deg -> armed.
    _set_pose(env, [0.05, 0.12], [50.0, 0.0])
    np.testing.assert_array_equal(term(env), [1.0, 0.0])
    # Recovers under the arming gate but not genuinely up (low z) -> still taxed.
    _set_pose(env, [0.05, 0.12], [20.0, 0.0])
    np.testing.assert_array_equal(term(env), [1.0, 0.0])
    # Genuinely up (tilt < 25 AND z > 0.09) -> released.
    _set_pose(env, [0.11, 0.12], [20.0, 0.0])
    np.testing.assert_array_equal(term(env), [0.0, 0.0])
    # A fresh reset clears the latch.
    _set_pose(env, [0.05, 0.05], [50.0, 50.0])
    env.episode_length_buf[:] = 0
    np.testing.assert_array_equal(term(env), [1.0, 1.0])
    env.episode_length_buf[:] = 0
    _set_pose(env, [0.12, 0.12], [0.0, 0.0])
    np.testing.assert_array_equal(term(env), [0.0, 0.0])


def test_recovery_success_fires_once_after_min_fallen_time() -> None:
    env = _recovery_env([0.05, 0.12], [50.0, 0.0])
    term = _term(
        recovery_terms.recovery_success,
        env,
        asset_cfg=SceneEntityCfg("robot"),
        fallen_tilt_deg=40.0,
        min_fallen_s=0.25,
        up_tilt_deg=25.0,
        up_z=0.09,
    )
    # 0.1 + 0.1 fallen: not yet armed (needs >= 0.25 s); upright env never fires.
    np.testing.assert_array_equal(term(env), [0.0, 0.0])
    np.testing.assert_array_equal(term(env), [0.0, 0.0])
    # Third fallen step (0.3 s) arms; still down -> no fire.
    np.testing.assert_array_equal(term(env), [0.0, 0.0])
    # Genuinely up -> one-shot bounty.
    _set_pose(env, [0.12, 0.12], [0.0, 0.0])
    np.testing.assert_array_equal(term(env), [1.0, 0.0])
    # Hysteresis: staying up does not re-fire.
    np.testing.assert_array_equal(term(env), [0.0, 0.0])
    # A brief dip under the gate neither re-arms (< 0.25 s) nor pays.
    _set_pose(env, [0.05, 0.12], [50.0, 0.0])
    term(env)
    _set_pose(env, [0.12, 0.12], [0.0, 0.0])
    np.testing.assert_array_equal(term(env), [0.0, 0.0])


def test_fallen_too_long_recycles_only_continuously_fallen_envs() -> None:
    env = _recovery_env([0.05, 0.12], [50.0, 0.0])
    term = _term(
        recovery_terms.fallen_too_long,
        env,
        asset_cfg=SceneEntityCfg("robot"),
        gate_z_below=0.08,
        gate_tilt_above_deg=40.0,
        max_duration_s=0.25,
    )
    np.testing.assert_array_equal(term(env), [False, False])  # 0.1 s
    np.testing.assert_array_equal(term(env), [False, False])  # 0.2 s
    np.testing.assert_array_equal(term(env), [True, False])  # 0.3 s >= 0.25
    # Standing back up resets the timer.
    _set_pose(env, [0.12, 0.12], [0.0, 0.0])
    np.testing.assert_array_equal(term(env), [False, False])
    # The z gate alone (sit-height, upright) also counts as fallen.
    _set_pose(env, [0.05, 0.12], [0.0, 0.0])
    for expected in (False, False, True):
        np.testing.assert_array_equal(term(env), [expected, False])
    # Freshly reset envs start with a clean timer.
    _set_pose(env, [0.05, 0.05], [50.0, 50.0])
    env.episode_length_buf[:] = 0
    np.testing.assert_array_equal(term(env), [False, False])


def test_upright_progress_is_potential_based() -> None:
    env = _recovery_env([0.12, 0.05], [0.0, 90.0])
    term = _term(recovery_terms.upright_progress, env, asset_cfg=SceneEntityCfg("robot"))
    # First call baselines: zero everywhere regardless of pose.
    np.testing.assert_allclose(term(env), [0.0, 0.0], atol=1e-6)
    # Holding any pose pays zero.
    np.testing.assert_allclose(term(env), [0.0, 0.0], atol=1e-6)
    # Falling charges exactly -delta cos(tilt); rising pays it back.
    _set_pose(env, [0.05, 0.12], [90.0, 0.0])
    np.testing.assert_allclose(term(env), [-1.0, 1.0], atol=1e-6)
    # A fresh reset re-baselines (no cross-episode delta).
    env.episode_length_buf[:] = 0
    np.testing.assert_allclose(term(env), [0.0, 0.0], atol=1e-6)


def test_height_progress_caps_at_ceiling() -> None:
    env = _recovery_env([0.05, 0.05], [0.0, 0.0])
    term = _term(
        recovery_terms.height_progress,
        env,
        asset_cfg=SceneEntityCfg("robot"),
        ceiling=0.115,
    )
    np.testing.assert_allclose(term(env), [0.0, 0.0], atol=1e-6)
    entity = env.scene["robot"]
    entity.data.root_link_pos_w[:, 2] = [0.10, 0.20]
    # Rising pays delta z; the capped env only gains up to the ceiling.
    np.testing.assert_allclose(term(env), [0.05, 0.065], atol=1e-6)
    # Hopping above the ceiling pays nothing.
    entity.data.root_link_pos_w[:, 2] = [0.10, 0.30]
    np.testing.assert_allclose(term(env), [0.0, 0.0], atol=1e-6)


def test_com_upward_velocity_gates_and_caps() -> None:
    env = _recovery_env([0.05, 0.05, 0.20, 0.05], [50.0, 0.0, 50.0, 50.0], vz=[0.5, 0.5, 0.5, -0.5])
    reward = recovery_terms.com_upward_velocity(
        env,
        asset_cfg=SceneEntityCfg("robot"),
        max_height=0.125,
        gate_z_below=0.0,
        gate_tilt_above_deg=40.0,
    )
    # Fallen + rising + below target pays vz; upright is gated out; above
    # max_height pays zero; downward velocity is clipped to zero.
    np.testing.assert_allclose(reward, [0.5, 0.0, 0.0, 0.0], atol=1e-6)
    # Without the fallen gate the upright env scores too.
    reward = recovery_terms.com_upward_velocity(
        env,
        asset_cfg=SceneEntityCfg("robot"),
        max_height=0.125,
    )
    np.testing.assert_allclose(reward, [0.5, 0.5, 0.0, 0.0], atol=1e-6)
    # max_vz caps the rewarded velocity.
    reward = recovery_terms.com_upward_velocity(
        env,
        asset_cfg=SceneEntityCfg("robot"),
        max_height=0.125,
        max_vz=0.2,
    )
    np.testing.assert_allclose(reward, [0.2, 0.2, 0.0, 0.0], atol=1e-6)


def test_feet_air_time_upright_zeroed_while_fallen() -> None:
    num_envs = 1
    contacts = {"left_foot_contact": np.zeros((1, 1)), "right_foot_contact": np.zeros((1, 1))}
    entity = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=np.array([[0.0, 0.0, 0.12]]),
            root_link_quat_w=_tilt_quat(0.0, num_envs),
        )
    )

    class _Scene:
        def __init__(self) -> None:
            self.env_origins = np.zeros((num_envs, 3))

        def __getitem__(self, name: str):
            assert name == "robot"
            return entity

        def bind_sensor_data(self, names: tuple[str, ...]):
            arrays = [contacts[name] for name in names]
            return SimpleNamespace(
                dimensions=tuple(array.shape[1] for array in arrays),
                backend_type="fake",
                read=lambda: np.concatenate(arrays, axis=1),
            )

    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=num_envs,
            step_dt=0.1,
            episode_length_buf=np.full(num_envs, 10, dtype=np.int64),
            scene=_Scene(),
            command_manager=SimpleNamespace(get_command=lambda name: np.array([[0.2, 0.0, 0.0]])),
        ),
    )
    term = _term(
        recovery_terms.feet_air_time_upright,
        env,
        threshold_min=0.05,
        threshold_max=0.5,
        command_threshold=0.01,
        command_name="twist",
        gate_tilt_above_deg=40.0,
        sensor_groups=[["left_foot_contact"], ["right_foot_contact"]],
        asset_cfg=SceneEntityCfg("robot"),
    )
    # Both feet airborne for 0.1 s (inside the window) -> 2.0 while upright.
    np.testing.assert_allclose(term(env), [2.0])
    # Fallen past the tilt gate -> zeroed even though the window still matches.
    entity.data.root_link_quat_w = _tilt_quat(50.0, num_envs)
    np.testing.assert_allclose(term(env), [0.0])
    # Upright again -> scores (air time kept accumulating while fallen).
    entity.data.root_link_quat_w = _tilt_quat(0.0, num_envs)
    np.testing.assert_allclose(term(env), [2.0])


def test_joint_torque_rate_l2_tracks_applied_torque_delta() -> None:
    bam = object.__new__(BamVoltageAction)
    bam._prev_applied_torque = np.array([[0.1, -0.2], [0.0, 0.0]])
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=2,
            action_manager=SimpleNamespace(get_term=lambda name: bam),
        ),
    )
    term = _term(recovery_terms.joint_torque_rate_l2, env)
    # First call baselines and pays zero.
    np.testing.assert_array_equal(term(env), [0.0, 0.0])
    bam._prev_applied_torque = np.array([[0.3, -0.2], [0.1, 0.1]])
    np.testing.assert_allclose(term(env), [0.04, 0.02], atol=1e-6)

    bad_env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=2,
            action_manager=SimpleNamespace(get_term=lambda name: object()),
        ),
    )
    with pytest.raises(TypeError, match="BamVoltageAction"):
        _term(recovery_terms.joint_torque_rate_l2, bad_env)


class _RecordingEntity:
    """Entity mock recording reset-transaction writes for the prone/crouch events."""

    def __init__(self, num_envs: int) -> None:
        self.poses = np.zeros((num_envs, 7))
        self.poses[:, 3] = 1.0
        self.poses[:, 0] = np.arange(num_envs) * 0.1  # staged reset_base x
        self.poses[:, 2] = 0.125
        self.num_joints = 14
        self.data = SimpleNamespace(default_joint_pos=np.zeros((num_envs, 14)))
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
        names = (
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
        key = keys.strip("^$")
        return [names.index(key)], [key]


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


def test_prone_event_is_inert_at_zero_probabilities() -> None:
    env, entity = _event_env()
    recovery_terms.maybe_set_random_prone_orientation(
        env, np.arange(8, dtype=np.int32), prone_prob=0.0, crouch_prob=0.0
    )
    assert entity.pose_writes == []
    assert entity.velocity_writes == []
    assert entity.joint_writes == []


def test_prone_event_lays_robot_down_and_preserves_xy() -> None:
    env, entity = _event_env()
    ids = np.arange(8, dtype=np.int32)
    recovery_terms.maybe_set_random_prone_orientation(
        env,
        ids,
        prone_prob=1.0,
        face_down_prob=1.0,
        prone_z_min=0.05,
        prone_z_max=0.09,
    )
    quat = entity.poses[:, 3:7]
    # Face-down family [s*cy, -s*sy, s*cy, s*sy]: qw == qy, qx == -qz, unit norm.
    np.testing.assert_allclose(quat[:, 0], quat[:, 2], atol=1e-7)
    np.testing.assert_allclose(quat[:, 1], -quat[:, 3], atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(quat, axis=1), 1.0, atol=1e-6)
    # 90 deg pitch: cos(tilt) = 1 - 2(qx^2 + qy^2) == 0.
    np.testing.assert_allclose(1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), 0.0, atol=1e-6)
    # z lifted into the clearance band; staged x/y preserved; velocity zeroed.
    assert np.all(entity.poses[:, 2] >= 0.05) and np.all(entity.poses[:, 2] <= 0.09)
    np.testing.assert_allclose(entity.poses[:, 0], np.arange(8) * 0.1, atol=1e-7)
    np.testing.assert_allclose(entity.poses[:, 1], 0.0, atol=1e-7)
    assert len(entity.velocity_writes) == 1
    np.testing.assert_array_equal(entity.velocity_writes[0][1], np.zeros((8, 6)))
    assert entity.joint_writes == []


def test_crouch_event_lerps_joints_and_preserves_xy() -> None:
    env, entity = _event_env()
    ids = np.arange(8, dtype=np.int32)
    recovery_terms.maybe_set_random_prone_orientation(env, ids, prone_prob=0.0, crouch_prob=1.0)
    assert len(entity.joint_writes) == 1
    _, position, velocity = entity.joint_writes[0]
    np.testing.assert_array_equal(velocity, np.zeros((8, 14)))
    # lam in [0.35, 1.0], defaults 0: anchor joints lerp toward the anchor with
    # +-0.12 noise; untouched joints stay within the noise band.
    left_knee = position[:, 3]
    assert np.all(left_knee >= 0.35 * 1.25 - 0.12) and np.all(left_knee <= 1.25 + 0.12)
    assert np.all(np.abs(position[:, 0]) <= 0.12)  # left_hip_yaw: no anchor
    # Height band between z_deep and z_stand (plus the settle margin).
    assert np.all(entity.poses[:, 2] >= 0.06) and np.all(entity.poses[:, 2] <= 0.125)
    np.testing.assert_allclose(entity.poses[:, 0], np.arange(8) * 0.1, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(entity.poses[:, 3:7], axis=1), 1.0, atol=1e-6)


@pytest.mark.slow
def test_velstand_owner_builds_and_steps_real_mujoco_env() -> None:
    pytest.importorskip("mujoco")
    cfg, _ = _materialize_velstand()
    env = cast(
        Any,
        registry.make(
            "MicroduckVelstandFlat",
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
        assert obs["critic"].shape == (2, MICRODUCK_CRITIC_OBS_DIM)

        for _ in range(3):
            state = env.step(np.zeros((2, MICRODUCK_NUM_ACTION), dtype=np.float32))
        assert np.isfinite(state.obs["obs"]).all()
        assert np.isfinite(state.obs["critic"]).all()
        assert np.isfinite(state.reward).all()
    finally:
        env.close()
