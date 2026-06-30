"""Contract tests for the A2JoystickFlat environment (leg-only Unitree A2).

The A2 leg-only MJCF mirrors the Go2 joystick sensor/geom/leg-ordering
contract (legs FL,FR,RL,RR; foot geoms+sites FL/FR/RL/RR; Go2-named IMU/foot
sensors) and uses <position> actuators, so the env reuses Go2WalkTask
unchanged. These tests prove the A2 model + scene + config + env chain
constructs and steps in MuJoCo as a 12-DOF joystick task."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH

# mjlab INIT_STATE: pos z=0.4, thigh=0.9 (all), calf=-1.8 (all), R_hip=+0.1, L_hip=-0.1.
# Asset/actuator order is FL,FR,RL,RR x (hip,thigh,calf).
_MJLAB_HOME_HEIGHT = 0.4
_MJLAB_LEG_ANGLES = [
    -0.1, 0.9, -1.8,  # FL
    0.1, 0.9, -1.8,   # FR
    -0.1, 0.9, -1.8,  # RL
    0.1, 0.9, -1.8,   # RR
]  # fmt: skip
# mjlab per-joint PD gains: hip/thigh kp=100/kd=4, calf kp=150/kd=6.
_MJLAB_KP = [100.0, 100.0, 150.0] * 4
_MJLAB_KD = [4.0, 4.0, 6.0] * 4
# DR ranges referencing mjlab events: joint_armature scale [0.9,1.1], foot
# friction [0.3,1.6] (UniLab realises it as a multiplier on the floor geom,
# which is made the priority geom so it dictates the foot-ground friction).
_MJLAB_ARMATURE_RANGE = [0.9, 1.1]
_MJLAB_FRICTION_RANGE = [0.3, 1.6]


def _skip_if_no_mujoco():
    pytest.importorskip("mujoco", reason="mujoco not installed")
    try:
        from mujoco.batch_env import BatchEnvPool  # noqa: F401
    except Exception:
        pytest.skip("mujoco.batch_env not available")


def test_a2_robot_xml_compiles_with_12_position_actuators():
    """a2.xml loads standalone and exposes exactly 12 position-style leg
    actuators in the FL,FR,RL,RR x hip,thigh,calf order."""
    mujoco = pytest.importorskip("mujoco")
    xml = ASSETS_ROOT_PATH / "robots" / "a2" / "a2.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    assert model.nu == 12
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    assert names == [
        "FL_hip",
        "FL_thigh",
        "FL_calf",
        "FR_hip",
        "FR_thigh",
        "FR_calf",
        "RL_hip",
        "RL_thigh",
        "RL_calf",
        "RR_hip",
        "RR_thigh",
        "RR_calf",
    ]
    # Position actuators carry an affine bias (kp in gainprm[0]); motor actuators do not.
    affine = int(mujoco.mjtBias.mjBIAS_AFFINE)
    assert all(int(model.actuator_biastype[i]) == affine for i in range(model.nu))


def test_a2_scene_loads_with_foot_contacts_and_home_keyframe():
    """scene_flat.xml includes a2.xml + floor, exposes the four foot-contact
    sensors and the joystick foot-pos/IMU sensors, and a home keyframe whose
    qpos is base(7)+12 leg = 19."""
    mujoco = pytest.importorskip("mujoco")
    xml = ASSETS_ROOT_PATH / "robots" / "a2" / "scene_flat.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))

    sensor_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(model.nsensor)
    }
    for required in [
        "gyro",
        "local_linvel",
        "upvector",
        "FL_pos",
        "FR_pos",
        "RL_pos",
        "RR_pos",
        "FL_foot_contact",
        "FR_foot_contact",
        "RL_foot_contact",
        "RR_foot_contact",
    ]:
        assert required in sensor_names, f"missing sensor {required}"

    # home keyframe present, qpos length = 7 (free base) + 12 (legs).
    assert model.nkey >= 1
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert key_id >= 0
    assert model.nq == 19
    # foot geoms used by the contact sensors exist.
    for g in ["FL", "FR", "RL", "RR", "floor"]:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) >= 0


def test_a2_home_keyframe_matches_mjlab_pose():
    """The home keyframe is aligned to mjlab's INIT_STATE: base height 0.4,
    thigh 0.9, calf -1.8 on all legs, hips +-0.1 (R/L)."""
    mujoco = pytest.importorskip("mujoco")
    xml = ASSETS_ROOT_PATH / "robots" / "a2" / "scene_flat.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert key_id >= 0
    qpos = np.asarray(model.key_qpos[key_id])
    assert qpos.shape == (19,)
    assert qpos[2] == pytest.approx(_MJLAB_HOME_HEIGHT)
    np.testing.assert_allclose(qpos[7:19], _MJLAB_LEG_ANGLES)
    # ctrl targets the same standing pose so position actuators hold it at reset.
    ctrl = np.asarray(model.key_ctrl[key_id])
    np.testing.assert_allclose(ctrl, _MJLAB_LEG_ANGLES)


def test_a2_control_config_per_joint_gains():
    """A2JoystickControlConfig.position_gains() yields per-joint arrays matching
    mjlab (calf 150/6, hip/thigh 100/4) in actuator order."""
    from unilab.envs.locomotion.a2.joystick import A2JoystickControlConfig

    gains = A2JoystickControlConfig().position_gains()
    np.testing.assert_allclose(np.asarray(gains["kp"]), _MJLAB_KP)
    np.testing.assert_allclose(np.asarray(gains["kd"]), _MJLAB_KD)


def test_pd_control_config_position_gains_default_is_scalar():
    """Base PdControlConfig keeps the scalar gain contract (Go2 path unchanged)."""
    from unilab.envs.locomotion.common.base import PdControlConfig

    gains = PdControlConfig(Kp=35.0, Kd=0.5).position_gains()
    assert gains == {"kp": 35.0, "kd": 0.5}


def test_a2_dr_provider_returns_per_joint_base_gains():
    """The A2 DR provider exposes per-joint base kp/kd so randomize_kp/kd scales
    each joint off the correct baseline (calf off 150, not 100)."""
    from unilab.envs.locomotion.a2.joystick import (
        A2JoystickControlConfig,
        A2JoystickDomainRandomizationProvider,
    )

    env = SimpleNamespace(
        cfg=SimpleNamespace(control_config=A2JoystickControlConfig()),
        _num_action=12,
    )
    base_kp, base_kd = A2JoystickDomainRandomizationProvider()._get_base_actuator_gains(env)
    np.testing.assert_allclose(np.asarray(base_kp), _MJLAB_KP)
    np.testing.assert_allclose(np.asarray(base_kd), _MJLAB_KD)


def test_a2_floor_geom_dominates_for_friction_dr():
    """The floor geom is the priority geom (priority=2 > feet's 1) and carries
    condim=6, so it dictates the foot-ground friction. This makes randomizing the
    floor geom's friction actually move the contact friction (otherwise the
    priority-1 feet would override it)."""
    mujoco = pytest.importorskip("mujoco")
    xml = ASSETS_ROOT_PATH / "robots" / "a2" / "scene_flat.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "FL")
    assert int(model.geom_priority[floor]) == 2
    assert int(model.geom_priority[floor]) > int(model.geom_priority[foot])
    # condim must stay 6 (feet use 6); floor would default to 3 without this.
    assert int(model.geom_condim[floor]) == 6


def test_a2_dr_provider_caches_friction_and_armature_baselines():
    """The A2 DR provider caches the pristine geom-friction + dof-armature tables
    (and the floor geom id) from the backend so randomize_ground_friction /
    randomize_dof_armature can multiply against them. body_mass stays uncached."""
    mujoco = pytest.importorskip("mujoco")
    from unilab.envs.locomotion.a2.joystick import A2JoystickDomainRandomizationProvider

    xml = ASSETS_ROOT_PATH / "robots" / "a2" / "scene_flat.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    class _StubBackend:
        def get_geom_friction(self):
            return np.asarray(model.geom_friction, dtype=np.float64).copy()

        def get_dof_armature(self):
            return np.asarray(model.dof_armature, dtype=np.float64).copy()

        def get_geom_id(self, name):
            return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))

    env = SimpleNamespace(
        _backend=_StubBackend(),
        cfg=SimpleNamespace(asset=SimpleNamespace(ground="floor")),
    )
    base_body_mass, base_geom_friction, ground_geom_id, base_dof_armature = (
        A2JoystickDomainRandomizationProvider()._get_reset_randomization_baselines(env)
    )
    assert base_body_mass is None  # body_mass DR not enabled
    assert ground_geom_id == floor_id
    assert base_geom_friction.shape == (model.ngeom, 3)
    np.testing.assert_allclose(base_geom_friction, model.geom_friction)
    assert base_dof_armature.shape == (model.nv,)
    np.testing.assert_allclose(base_dof_armature, model.dof_armature)


def _ensure_registered() -> None:
    from unilab.base import registry

    registry.ensure_registries()
    if not registry.contains("A2JoystickFlat"):
        importlib.import_module("unilab.envs.locomotion.a2.joystick")


def test_a2_joystick_registered():
    """Registers without MuJoCo (decorators run on module import)."""
    from unilab.base import registry

    _ensure_registered()
    assert registry.contains("A2JoystickFlat")


def test_a2_joystick_yaml_composes_and_targets_a2():
    """The owner YAML composes under Hydra and selects the A2JoystickFlat task
    with a reward block that injects into the env's reward_config."""
    from hydra import compose, initialize

    with initialize(config_path="../../../../conf/ppo", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["task=a2_joystick_flat/mujoco"])
    assert cfg.training.task_name == "A2JoystickFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert "tracking_lin_vel" in cfg.reward.scales


def _default_reward_cfg():
    from unilab.envs.locomotion.go2.joystick import RewardConfig

    return RewardConfig(
        scales={
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -5.0,
            "ang_vel_xy": -0.1,
            "base_height": -100.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            "contact": 0.24,
            "swing_feet_z": 4.0,
        },
        tracking_sigma=0.25,
        base_height_target=0.45,
    )


def _make_a2_env(num_envs: int = 2, domain_rand=None):
    from unilab.base import registry

    _ensure_registered()
    override = {"reward_config": _default_reward_cfg()}
    if domain_rand is not None:
        override["domain_rand"] = domain_rand
    return registry.make(
        "A2JoystickFlat",
        sim_backend="mujoco",
        num_envs=num_envs,
        env_cfg_override=override,
    )


@pytest.mark.slow
def test_a2_joystick_obs_layout_and_12_dof():
    _skip_if_no_mujoco()
    env = _make_a2_env(num_envs=2)
    assert env._num_action == 12
    assert env.default_angles.shape == (12,)
    assert env.obs_groups_spec == {"obs": 49, "critic": 52}


@pytest.mark.slow
def test_a2_joystick_model_gains_are_per_joint():
    """End-to-end: the per-joint gains reach the compiled MuJoCo model — calf
    actuators carry kp=150/kd=6, hip/thigh kp=100/kd=4 — and default_angles
    (derived from the home keyframe) match the mjlab standing pose."""
    _skip_if_no_mujoco()
    env = _make_a2_env(num_envs=2)

    # default_angles come from the home keyframe -> mjlab pose.
    np.testing.assert_allclose(env.default_angles, _MJLAB_LEG_ANGLES)

    # The env forwarded per-joint arrays to the backend.
    stored = env._backend._position_actuator_gains
    np.testing.assert_allclose(np.asarray(stored["kp"]), _MJLAB_KP)
    np.testing.assert_allclose(np.asarray(stored["kd"]), _MJLAB_KD)

    # ...and they are written into the compiled model: position actuators store
    # kp in gainprm[0], and -kp / -kd in biasprm[1] / biasprm[2].
    model = env._backend._model
    np.testing.assert_allclose(model.actuator_gainprm[:, 0], _MJLAB_KP)
    np.testing.assert_allclose(model.actuator_biasprm[:, 1], [-v for v in _MJLAB_KP])
    np.testing.assert_allclose(model.actuator_biasprm[:, 2], [-v for v in _MJLAB_KD])


@pytest.mark.slow
def test_a2_joystick_init_step_runs_finite():
    """End-to-end: init + steps must run (all A2 sensors/geoms resolve) with
    finite obs/reward, proving the leg-only A2 asset satisfies the joystick
    sensor contract on the hot path."""
    _skip_if_no_mujoco()

    env = _make_a2_env(num_envs=2)
    state = env.init_state()
    assert state.obs["obs"].shape == (2, 49)
    assert state.obs["critic"].shape == (2, 52)
    for _ in range(10):
        state = env.step(np.zeros((2, 12), dtype=np.float64))
    assert np.isfinite(state.reward).all()
    assert np.isfinite(state.obs["obs"]).all()
    assert np.isfinite(state.obs["critic"]).all()


@pytest.mark.slow
def test_a2_joystick_dr_on_constructs_and_steps_finite():
    """With DR on (incl. base_link interval push, dof-armature and ground-friction
    randomization), the env constructs and steps with finite obs/reward — proving
    push_body_name resolves to a real body and the mass/COM/kp-kd/armature/friction
    randomization path is sound.

    A2JoystickDomainRandomizationProvider caches the dof-armature + geom-friction
    baselines, so randomize_dof_armature / randomize_ground_friction are now ON.
    randomize_body_mass stays off (base_body_mass baseline not cached). The
    YAML-surface is covered by test_a2_joystick_domain_rand_fully_configured."""
    _skip_if_no_mujoco()
    from unilab.envs.locomotion.a2.joystick import A2JoystickDomainRandConfig

    dr_on = A2JoystickDomainRandConfig(
        randomize_base_mass=True,
        added_mass_range=[0.0, 8.0],
        randomize_body_mass=False,  # provider does not cache base_body_mass
        random_com=True,
        com_offset_x=[-0.08, 0.08],
        com_offset_y=[-0.08, 0.08],
        com_offset_z=[-0.08, 0.08],
        randomize_ground_friction=True,
        ground_friction_multiplier_range=_MJLAB_FRICTION_RANGE,
        randomize_dof_armature=True,
        dof_armature_multiplier_range=_MJLAB_ARMATURE_RANGE,
        randomize_kp=True,
        randomize_kd=True,
        push_robots=True,
        push_interval=400,
        push_body_name="base_link",
    )
    env = _make_a2_env(num_envs=4, domain_rand=dr_on)

    # DR fields are active on the constructed config.
    assert env._cfg.domain_rand.push_robots is True
    assert env._cfg.domain_rand.push_body_name == "base_link"
    assert env._cfg.domain_rand.randomize_base_mass is True
    assert env._cfg.domain_rand.randomize_dof_armature is True
    assert env._cfg.domain_rand.randomize_ground_friction is True
    assert list(env._cfg.domain_rand.com_offset_z) == [-0.08, 0.08]

    state = env.init_state()
    assert state.obs["obs"].shape == (4, 49)
    # 10 steps exercises reset-time DR (mass/friction/COM/armature/kp-kd) + stepping.
    for _ in range(10):
        state = env.step(np.zeros((4, 12), dtype=np.float64))
    assert np.isfinite(state.reward).all()
    assert np.isfinite(state.obs["obs"]).all()
    assert np.isfinite(state.obs["critic"]).all()


def test_a2_joystick_domain_rand_fully_configured():
    """Owner YAML enables DR switches supported by A2JoystickDomainRandomizationProvider,
    with mjlab-referenced ranges, 3-axis COM, base_link push target, and the
    500-iteration budget.

    randomize_dof_armature + randomize_ground_friction are ON: the A2 provider caches
    their baselines (dof_armature, geom_friction + floor geom id). randomize_body_mass
    stays OFF (base_body_mass not cached); gravity OFF (constant on flat ground)."""
    from hydra import compose, initialize

    with initialize(config_path="../../../../conf/ppo", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["task=a2_joystick_flat/mujoco"])

    dr = cfg.env.domain_rand
    assert dr.randomize_base_mass is True
    assert dr.randomize_body_mass is False  # provider does not cache base_body_mass
    assert dr.random_com is True
    assert dr.randomize_gravity is False
    assert dr.randomize_ground_friction is True
    assert dr.randomize_dof_armature is True
    assert dr.randomize_kp is True
    assert dr.randomize_kd is True
    assert dr.push_robots is True
    # 3-axis COM present + A2-scale value
    assert list(dr.com_offset_x) == [-0.08, 0.08]
    assert list(dr.com_offset_y) == [-0.08, 0.08]
    assert list(dr.com_offset_z) == [-0.08, 0.08]
    # mjlab-referenced ranges + push target
    assert list(dr.added_mass_range) == [0.0, 8.0]
    assert list(dr.ground_friction_multiplier_range) == _MJLAB_FRICTION_RANGE
    assert list(dr.dof_armature_multiplier_range) == _MJLAB_ARMATURE_RANGE
    assert dr.push_interval == 400
    assert dr.push_body_name == "base_link"
    # bumped budget
    assert cfg.algo.max_iterations == 500
