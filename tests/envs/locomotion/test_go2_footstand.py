from __future__ import annotations

import numpy as np
import pytest

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.base.registry import ensure_registries
from unilab.dr import ResetRandomizationPayload
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.go2.footstand import (
    FootstandControlConfig,
    FootstandSensor,
    Go2FootStandCfg,
    Go2FootStandDomainRandConfig,
    Go2FootStandDomainRandomizationProvider,
    Go2FootStandTask,
)
from unilab.envs.locomotion.go2.handstand import RewardConfig


def test_go2_footstand_registers_mujoco_only() -> None:
    ensure_registries()

    meta = registry.list_registered_envs()["Go2FootStand"]

    assert meta["available_backends"] == ["mujoco"]


class _OrientationBackend:
    pass


class _JointRangeBackend:
    def get_joint_range(self) -> np.ndarray:
        return np.array([[-2.0, 2.0], [0.0, 2.0]], dtype=np.float32)


class _BaseMotionBackend:
    def get_base_lin_vel(self) -> np.ndarray:
        return np.array([[1.0, 2.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)

    def get_base_ang_vel(self) -> np.ndarray:
        return np.array([[0.0, 0.0, 2.0], [0.0, 2.0, 0.0]], dtype=np.float32)


class _KneeHeightBackend:
    def __init__(self, knee_height: np.ndarray) -> None:
        self._knee_height = knee_height

    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        assert body_ids.shape == (4,)
        out = np.zeros((*self._knee_height.shape, 3), dtype=np.float32)
        out[:, :, 2] = self._knee_height
        return out


def test_go2_footstand_cfg_uses_rear_body_contact_termination() -> None:
    cfg = Go2FootStandCfg()

    assert isinstance(cfg.sensor, FootstandSensor)
    assert "base1_contact" not in cfg.sensor.ternamate_contact
    assert "RL_calf_contact1" in cfg.sensor.ternamate_contact
    assert "RR_calf_contact2" in cfg.sensor.ternamate_contact
    assert "FL_calf_contact1" in cfg.sensor.penalty_contact
    assert "FR_calf_contact2" in cfg.sensor.penalty_contact
    assert cfg.noise_config.level == pytest.approx(1.0)
    assert cfg.noise_config.scale_joint_angle == pytest.approx(0.01)
    assert cfg.noise_config.scale_joint_vel == pytest.approx(1.5)
    assert cfg.add_body_sensors is True
    assert isinstance(cfg.control_config, FootstandControlConfig)
    assert cfg.control_config.action_scale == pytest.approx(0.3)
    assert cfg.control_config.clip_actions == pytest.approx(1.0)
    assert isinstance(cfg.domain_rand, Go2FootStandDomainRandConfig)
    assert cfg.domain_rand.randomize_kp is False
    assert cfg.domain_rand.randomize_floor_friction is True
    assert cfg.obs_history_len == 15
    assert cfg.soft_joint_pos_limit_factor == pytest.approx(0.9)
    assert cfg.energy_termination_threshold == np.inf
    assert cfg.termination_grace_steps == 100
    assert cfg.termination_height_fraction == pytest.approx(0.8)
    assert cfg.termination_orientation_threshold == pytest.approx(0.2)
    assert cfg.max_episode_seconds == pytest.approx(10.0)


def test_go2_footstand_orientation_flips_handstand_target() -> None:
    env = object.__new__(Go2FootStandTask)
    env._backend = _OrientationBackend()
    env._desired_forward_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    env._get_body_forward = lambda: np.array(  # type: ignore[method-assign]
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=np.float32
    )

    reward = env._reward_orientation(
        RewardContext(
            info={},
            linvel=np.zeros((2, 3), dtype=np.float32),
            gyro=np.zeros((2, 3), dtype=np.float32),
            dof_pos=np.zeros((2, 12), dtype=np.float32),
        )
    )

    np.testing.assert_allclose(reward, np.array([1.0, 0.0], dtype=np.float32))


def test_go2_footstand_soft_joint_limits_use_playground_factor() -> None:
    env = object.__new__(Go2FootStandTask)
    cfg = Go2FootStandCfg()
    cfg.soft_joint_pos_limit_factor = 0.5
    env._cfg = cfg
    env._num_action = 2
    env._backend = _JointRangeBackend()

    env._init_soft_joint_limits()

    np.testing.assert_allclose(env._soft_lowers, np.array([-1.0, 0.5], dtype=np.float32))
    np.testing.assert_allclose(env._soft_uppers, np.array([1.0, 1.5], dtype=np.float32))


def test_go2_footstand_reward_functions_include_stability_terms() -> None:
    env = object.__new__(Go2FootStandTask)

    env._init_reward_functions()

    assert "tar" in env._reward_fns
    assert "penalty_contact" in env._reward_fns
    assert "termination" in env._reward_fns
    assert "rear_feet_contact" in env._reward_fns
    assert "rear_leg_symmetry" in env._reward_fns
    assert "front_leg_motion" in env._reward_fns
    assert "upright_stability" in env._reward_fns
    assert "knee_clearance" in env._reward_fns


def test_go2_footstand_pose_targets_front_legs_and_supports_rear() -> None:
    env = object.__new__(Go2FootStandTask)

    env._init_footstand_pose_targets()

    assert env.feet_geom_names == [0, 1]
    assert env._joint_ids == [6, 7, 8, 9, 10, 11]
    assert env._tar_ids == [0, 1, 2, 3, 4, 5]
    np.testing.assert_allclose(
        env.target_angle, np.array([0.0, 1.82, -1.16, 0.0, 1.82, -1.16], dtype=np.float32)
    )


def test_go2_footstand_obs_matches_playground_state_layout() -> None:
    env = object.__new__(Go2FootStandTask)
    cfg = Go2FootStandCfg()
    cfg.noise_config.level = 0.0
    env._cfg = cfg
    env._num_envs = 1
    env.default_angles = np.arange(12, dtype=np.float32).reshape(1, 12)
    env._obs_history = np.zeros((1, cfg.obs_history_len, 45), dtype=np.float32)

    linvel = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    gyro = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
    gravity = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
    dof_pos = env.default_angles + 0.5
    dof_vel = np.arange(12, dtype=np.float32).reshape(1, 12) + 10.0
    last_actions = np.full((1, 12), 0.25, dtype=np.float32)
    current_actions = np.full((1, 12), 0.75, dtype=np.float32)
    accelerometer = np.array([[7.0, 8.0, 9.0]], dtype=np.float32)
    global_angvel = np.array([[10.0, 11.0, 12.0]], dtype=np.float32)
    torques = np.arange(12, dtype=np.float32).reshape(1, 12) + 20.0

    obs = env._compute_obs(
        {"last_actions": last_actions, "current_actions": current_actions, "torques": torques},
        linvel,
        gyro,
        gravity,
        dof_pos,
        dof_vel,
        np.array([[0.53]], dtype=np.float32),
        accelerometer,
        global_angvel,
    )

    current_frame = obs["obs"][:, -45:]
    assert obs["obs"].shape == (1, 675)
    assert obs["critic"].shape == (1, 724)
    np.testing.assert_allclose(obs["obs"][:, : 14 * 45], 0.0)
    np.testing.assert_allclose(current_frame[:, 0:3], linvel)
    np.testing.assert_allclose(current_frame[:, 3:6], gyro)
    np.testing.assert_allclose(current_frame[:, 6:9], gravity)
    np.testing.assert_allclose(current_frame[:, -12:], last_actions)


def test_go2_footstand_reset_obs_fills_history_with_current_frame() -> None:
    env = object.__new__(Go2FootStandTask)
    cfg = Go2FootStandCfg()
    cfg.noise_config.level = 0.0
    env._cfg = cfg
    env._num_envs = 2
    env.default_angles = np.zeros((1, 12), dtype=np.float32)
    env._obs_history = np.zeros((2, cfg.obs_history_len, 45), dtype=np.float32)

    linvel = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    gyro = np.array([[0.0, 2.0, 0.0]], dtype=np.float32)
    gravity = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
    dof_pos = np.ones((1, 12), dtype=np.float32)
    dof_vel = np.full((1, 12), 3.0, dtype=np.float32)

    obs = env._compute_obs(
        {"last_actions": np.full((1, 12), 0.5, dtype=np.float32)},
        linvel,
        gyro,
        gravity,
        dof_pos,
        dof_vel,
        np.array([[0.53]], dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
        env_ids=np.array([1], dtype=np.int32),
    )

    frames = obs["obs"].reshape(1, cfg.obs_history_len, 45)
    np.testing.assert_allclose(frames[:, 0, :], frames[:, -1, :])
    np.testing.assert_allclose(env._obs_history[0], 0.0)
    np.testing.assert_allclose(env._obs_history[1], frames[0])


class _EnergyTerminationBackend:
    def __init__(self) -> None:
        self._sensors = {
            "local_linvel": np.zeros((1, 3), dtype=np.float32),
            "gyro": np.zeros((1, 3), dtype=np.float32),
            "upvector": np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            "accelerometer": np.zeros((1, 3), dtype=np.float32),
            "global_angvel": np.zeros((1, 3), dtype=np.float32),
            "global_position": np.array([[0.0, 0.0, 0.53]], dtype=np.float32),
        }
        for name in FootstandSensor.feet_force:
            self._sensors[name] = np.zeros((1, 1), dtype=np.float32)
        for name in FootstandSensor.feet_pos:
            self._sensors[name] = np.zeros((1, 3), dtype=np.float32)
        for name in FootstandSensor.ternamate_contact:
            self._sensors[name] = np.zeros((1, 1), dtype=np.float32)

    def get_sensor_data(self, name: str) -> np.ndarray:
        return self._sensors[name]

    def get_dof_pos(self) -> np.ndarray:
        return np.zeros((1, 12), dtype=np.float32)

    def get_dof_vel(self) -> np.ndarray:
        return np.full((1, 12), 10.0, dtype=np.float32)

    def get_base_quat(self) -> np.ndarray:
        return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    def get_base_pos(self) -> np.ndarray:
        return np.array([[0.0, 0.0, 0.53]], dtype=np.float32)


def test_go2_footstand_energy_threshold_terminates() -> None:
    env = object.__new__(Go2FootStandTask)
    cfg = Go2FootStandCfg(
        reward_config=RewardConfig(scales={}, tracking_sigma=0.25, base_height_target=0.3)
    )
    cfg.noise_config.level = 0.0
    cfg.energy_termination_threshold = 1.0
    env._cfg = cfg
    env._reward_cfg = cfg.reward_config
    env._backend = _EnergyTerminationBackend()
    env._num_envs = 1
    env._num_action = 12
    env._z_des = 0.53
    env._desired_forward_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    env.default_angles = np.zeros((1, 12), dtype=np.float32)
    env.feet_force = np.zeros((1, 4, 1), dtype=np.float32)
    env.feet_pos = np.zeros((1, 4, 3), dtype=np.float32)
    env.torso_height = np.zeros((1,), dtype=np.float32)
    env._last_dof_vel_for_acc = np.zeros((1, 12), dtype=np.float32)
    env._motor_targets = np.zeros((1, 12), dtype=np.float32)
    env._last_terminated = np.zeros((1,), dtype=bool)
    env._enable_reward_log = False

    state = NpEnvState(
        obs={},
        reward=np.zeros((1,), dtype=np.float32),
        terminated=np.zeros((1,), dtype=bool),
        truncated=np.zeros((1,), dtype=bool),
        info={
            "current_actions": np.zeros((1, 12), dtype=np.float32),
            "last_actions": np.zeros((1, 12), dtype=np.float32),
        },
    )

    updated = env.update_state(state)

    assert updated.terminated[0]


def test_go2_footstand_action_updates_incremental_motor_targets() -> None:
    env = object.__new__(Go2FootStandTask)
    env._cfg = Go2FootStandCfg()
    env._motor_targets = np.zeros((1, 12), dtype=np.float32)
    state = NpEnvState(
        obs={},
        reward=np.zeros((1,), dtype=np.float32),
        terminated=np.zeros((1,), dtype=bool),
        truncated=np.zeros((1,), dtype=bool),
        info={"current_actions": np.full((1, 12), 0.1, dtype=np.float32)},
    )

    ctrl = env.apply_action(np.full((1, 12), 0.5, dtype=np.float32), state)

    np.testing.assert_allclose(ctrl, np.full((1, 12), 0.15, dtype=np.float32))
    np.testing.assert_allclose(state.info["last_actions"], np.full((1, 12), 0.1, dtype=np.float32))
    np.testing.assert_allclose(
        state.info["current_actions"], np.full((1, 12), 0.5, dtype=np.float32)
    )


def test_go2_footstand_action_clips_policy_actions_and_motor_targets() -> None:
    env = object.__new__(Go2FootStandTask)
    env._cfg = Go2FootStandCfg()
    env._motor_targets = np.zeros((1, 2), dtype=np.float32)
    env._target_lowers = np.array([-0.2, -0.4], dtype=np.float32)
    env._target_uppers = np.array([0.2, 0.4], dtype=np.float32)
    state = NpEnvState(
        obs={},
        reward=np.zeros((1,), dtype=np.float32),
        terminated=np.zeros((1,), dtype=bool),
        truncated=np.zeros((1,), dtype=bool),
        info={},
    )

    ctrl = env.apply_action(np.array([[10.0, -10.0]], dtype=np.float32), state)

    np.testing.assert_allclose(
        state.info["current_actions"], np.array([[1.0, -1.0]], dtype=np.float32)
    )
    np.testing.assert_allclose(ctrl, np.array([[0.2, -0.3]], dtype=np.float32))

    ctrl = env.apply_action(np.array([[10.0, -10.0]], dtype=np.float32), state)

    np.testing.assert_allclose(ctrl, np.array([[0.2, -0.4]], dtype=np.float32))


def test_go2_footstand_playground_reset_randomization_payload_shapes() -> None:
    np.random.seed(0)
    env = object.__new__(Go2FootStandTask)
    env._cfg = Go2FootStandCfg()
    env._num_action = 12
    env._floor_geom_id = 0
    env._base_body_id = 1
    env._base_geom_friction = np.ones((3, 3), dtype=np.float64)
    env._base_body_mass = np.ones((4,), dtype=np.float64)
    env._base_body_ipos = np.zeros((4, 3), dtype=np.float64)
    env._base_dof_armature = np.ones((18,), dtype=np.float64)

    payload = env._build_playground_reset_randomization(num_reset=2)

    assert payload is not None
    assert payload.geom_friction is not None and payload.geom_friction.shape == (2, 3, 3)
    assert payload.body_mass is not None and payload.body_mass.shape == (2, 4)
    assert payload.body_ipos is not None and payload.body_ipos.shape == (2, 4, 3)
    assert payload.dof_armature is not None and payload.dof_armature.shape == (2, 18)
    assert np.all((payload.geom_friction[:, 0, 0] >= 0.4) & (payload.geom_friction[:, 0, 0] <= 1.0))
    np.testing.assert_allclose(payload.dof_armature[:, :6], 1.0)


def test_go2_footstand_reset_randomization_merges_common_and_playground_terms() -> None:
    base = ResetRandomizationPayload(
        base_mass_delta=np.array([0.1], dtype=np.float64),
        kp=np.ones((1, 12), dtype=np.float64),
    )
    playground = ResetRandomizationPayload(
        body_mass=np.full((1, 4), 2.0, dtype=np.float64),
        geom_friction=np.full((1, 3, 3), 0.7, dtype=np.float64),
    )

    merged = Go2FootStandDomainRandomizationProvider._merge_reset_randomization(base, playground)

    assert merged is not None
    np.testing.assert_allclose(merged.base_mass_delta, base.base_mass_delta)
    np.testing.assert_allclose(merged.kp, base.kp)
    np.testing.assert_allclose(merged.body_mass, playground.body_mass)
    np.testing.assert_allclose(merged.geom_friction, playground.geom_friction)


def test_go2_footstand_height_reward_matches_playground_shape() -> None:
    env = object.__new__(Go2FootStandTask)
    env._z_des = 0.53
    env.torso_height = np.array([0.53, 0.33, 0.63], dtype=np.float32)

    reward = env._reward_height(
        RewardContext(
            info={},
            linvel=np.zeros((3, 3), dtype=np.float32),
            gyro=np.zeros((3, 3), dtype=np.float32),
            dof_pos=np.zeros((3, 12), dtype=np.float32),
        )
    )

    np.testing.assert_allclose(
        reward, np.array([1.0, np.exp(-2.0), np.exp(-1.0)], dtype=np.float32), rtol=1e-6
    )


def test_go2_footstand_contact_cost_only_penalizes_front_feet() -> None:
    env = object.__new__(Go2FootStandTask)
    env.feet_geom_names = [0, 1]
    env.feet_force = np.zeros((3, 4, 1), dtype=np.float32)
    env.feet_force[0, 0, 0] = 1.0
    env.feet_force[1, 2, 0] = 1.0
    env.feet_force[2, 3, 0] = 1.0

    cost = env._cost_contact(
        RewardContext(
            info={},
            linvel=np.zeros((3, 3), dtype=np.float32),
            gyro=np.zeros((3, 3), dtype=np.float32),
            dof_pos=np.zeros((3, 12), dtype=np.float32),
        )
    )

    np.testing.assert_allclose(cost, np.array([1.0, 0.0, 0.0], dtype=np.float32))


def test_go2_footstand_rear_feet_contact_rewards_support_feet() -> None:
    env = object.__new__(Go2FootStandTask)
    env.feet_force = np.zeros((3, 4, 1), dtype=np.float32)
    env.feet_force[0, 2:4, 0] = 1.0
    env.feet_force[1, 2, 0] = 1.0
    env.feet_force[2, 0:2, 0] = 1.0

    reward = env._reward_rear_feet_contact(
        RewardContext(
            info={},
            linvel=np.zeros((3, 3), dtype=np.float32),
            gyro=np.zeros((3, 3), dtype=np.float32),
            dof_pos=np.zeros((3, 12), dtype=np.float32),
        )
    )

    np.testing.assert_allclose(reward, np.array([1.0, 0.5, 0.0], dtype=np.float32))


def test_go2_footstand_rear_leg_symmetry_mirrors_hip_sign_only() -> None:
    env = object.__new__(Go2FootStandTask)
    env._standing_mask = lambda: np.array([0.0, 0.0, 1.0], dtype=np.float32)  # type: ignore[method-assign]
    dof_pos = np.zeros((3, 12), dtype=np.float32)
    dof_pos[0, 6:9] = np.array([0.2, 1.0, -1.5], dtype=np.float32)
    dof_pos[0, 9:12] = np.array([-0.2, 1.0, -1.5], dtype=np.float32)
    dof_pos[1, 6:9] = np.array([0.2, 1.0, -1.5], dtype=np.float32)
    dof_pos[1, 9:12] = np.array([0.3, 1.0, -1.5], dtype=np.float32)
    dof_pos[2, 6:9] = np.array([0.0, 1.2, -1.1], dtype=np.float32)
    dof_pos[2, 9:12] = np.array([0.0, 0.9, -1.7], dtype=np.float32)

    cost = env._cost_rear_leg_symmetry(
        RewardContext(
            info={},
            linvel=np.zeros((3, 3), dtype=np.float32),
            gyro=np.zeros((3, 3), dtype=np.float32),
            dof_pos=dof_pos,
        )
    )

    np.testing.assert_allclose(
        cost,
        np.array([0.0, 0.25 / 3.0, 0.0], dtype=np.float32),
        rtol=1e-6,
    )


def test_go2_footstand_front_leg_motion_only_penalizes_standing_pose() -> None:
    env = object.__new__(Go2FootStandTask)
    env._z_des = 0.53
    env.torso_height = np.array([0.53, 0.53, 0.2], dtype=np.float32)
    env._orientation_score = lambda: np.array([1.0, 0.4, 1.0], dtype=np.float32)  # type: ignore[method-assign]
    dof_vel = np.zeros((3, 12), dtype=np.float32)
    dof_vel[:, 0:6] = 2.0

    cost = env._cost_front_leg_motion(
        RewardContext(
            info={},
            linvel=np.zeros((3, 3), dtype=np.float32),
            gyro=np.zeros((3, 3), dtype=np.float32),
            dof_pos=np.zeros((3, 12), dtype=np.float32),
            dof_vel=dof_vel,
        )
    )

    np.testing.assert_allclose(cost, np.array([4.0, 0.0, 0.0], dtype=np.float32))


def test_go2_footstand_upright_stability_is_standing_gated() -> None:
    env = object.__new__(Go2FootStandTask)
    env._backend = _BaseMotionBackend()
    env._z_des = 0.53
    env.torso_height = np.array([0.53, 0.53], dtype=np.float32)
    env._orientation_score = lambda: np.array([1.0, 0.4], dtype=np.float32)  # type: ignore[method-assign]

    cost = env._cost_upright_stability(
        RewardContext(
            info={},
            linvel=np.zeros((2, 3), dtype=np.float32),
            gyro=np.zeros((2, 3), dtype=np.float32),
            dof_pos=np.zeros((2, 12), dtype=np.float32),
        )
    )

    np.testing.assert_allclose(cost, np.array([6.0, 0.0], dtype=np.float32))


def test_go2_footstand_knee_clearance_penalizes_low_knees() -> None:
    env = object.__new__(Go2FootStandTask)
    env._reward_cfg = RewardConfig(
        scales={},
        tracking_sigma=0.25,
        base_height_target=0.3,
        knee_height_target=0.1,
    )
    env._knee_body_ids = np.array([1, 2, 3, 4], dtype=np.int32)
    env._backend = _KneeHeightBackend(
        np.array(
            [
                [0.1, 0.12, 0.2, 0.3],
                [0.05, 0.05, 0.05, 0.05],
                [0.0, 0.05, 0.1, 0.15],
            ],
            dtype=np.float32,
        )
    )

    cost = env._cost_knee_clearance(
        RewardContext(
            info={},
            linvel=np.zeros((3, 3), dtype=np.float32),
            gyro=np.zeros((3, 3), dtype=np.float32),
            dof_pos=np.zeros((3, 12), dtype=np.float32),
        )
    )

    np.testing.assert_allclose(cost, np.array([0.0, 0.25, 0.3125], dtype=np.float32))


def test_go2_footstand_post_grace_low_height_terminates() -> None:
    env = object.__new__(Go2FootStandTask)
    cfg = Go2FootStandCfg(
        reward_config=RewardConfig(scales={}, tracking_sigma=0.25, base_height_target=0.3)
    )
    cfg.noise_config.level = 0.0
    cfg.termination_grace_steps = 10
    env._cfg = cfg
    env._reward_cfg = cfg.reward_config
    env._backend = _EnergyTerminationBackend()
    env._backend._sensors["global_position"] = np.array([[0.0, 0.0, 0.2]], dtype=np.float32)
    env._num_envs = 1
    env._num_action = 12
    env._z_des = 0.53
    env.default_angles = np.zeros((1, 12), dtype=np.float32)
    env.feet_force = np.zeros((1, 4, 1), dtype=np.float32)
    env.feet_pos = np.zeros((1, 4, 3), dtype=np.float32)
    env.torso_height = np.zeros((1,), dtype=np.float32)
    env._last_dof_vel_for_acc = np.zeros((1, 12), dtype=np.float32)
    env._motor_targets = np.zeros((1, 12), dtype=np.float32)
    env._last_terminated = np.zeros((1,), dtype=bool)
    env._enable_reward_log = False
    env._orientation_score = lambda: np.array([1.0], dtype=np.float32)  # type: ignore[method-assign]
    state = NpEnvState(
        obs={},
        reward=np.zeros((1,), dtype=np.float32),
        terminated=np.zeros((1,), dtype=bool),
        truncated=np.zeros((1,), dtype=bool),
        info={
            "steps": np.array([10], dtype=np.uint32),
            "current_actions": np.zeros((1, 12), dtype=np.float32),
            "last_actions": np.zeros((1, 12), dtype=np.float32),
        },
    )

    updated = env.update_state(state)

    assert updated.terminated[0]


def test_go2_footstand_returned_termination_does_not_alias_reset_bookkeeping() -> None:
    env = object.__new__(Go2FootStandTask)
    cfg = Go2FootStandCfg(
        reward_config=RewardConfig(scales={}, tracking_sigma=0.25, base_height_target=0.3)
    )
    cfg.noise_config.level = 0.0
    cfg.termination_grace_steps = 10
    env._cfg = cfg
    env._reward_cfg = cfg.reward_config
    env._backend = _EnergyTerminationBackend()
    env._backend._sensors["global_position"] = np.array([[0.0, 0.0, 0.2]], dtype=np.float32)
    env._num_envs = 1
    env._num_action = 12
    env._z_des = 0.53
    env.default_angles = np.zeros((1, 12), dtype=np.float32)
    env.feet_force = np.zeros((1, 4, 1), dtype=np.float32)
    env.feet_pos = np.zeros((1, 4, 3), dtype=np.float32)
    env.torso_height = np.zeros((1,), dtype=np.float32)
    env._last_dof_vel_for_acc = np.zeros((1, 12), dtype=np.float32)
    env._motor_targets = np.zeros((1, 12), dtype=np.float32)
    env._last_terminated = np.zeros((1,), dtype=bool)
    env._enable_reward_log = False
    env._orientation_score = lambda: np.array([1.0], dtype=np.float32)  # type: ignore[method-assign]
    state = NpEnvState(
        obs={},
        reward=np.zeros((1,), dtype=np.float32),
        terminated=np.zeros((1,), dtype=bool),
        truncated=np.zeros((1,), dtype=bool),
        info={
            "steps": np.array([10], dtype=np.uint32),
            "current_actions": np.zeros((1, 12), dtype=np.float32),
            "last_actions": np.zeros((1, 12), dtype=np.float32),
        },
    )

    updated = env.update_state(state)
    env._last_terminated[0] = False

    assert updated.terminated[0]


def test_go2_footstand_joint_limit_cost_uses_soft_limits() -> None:
    env = object.__new__(Go2FootStandTask)
    env._soft_lowers = np.array([-1.0, -1.0], dtype=np.float32)
    env._soft_uppers = np.array([1.0, 1.0], dtype=np.float32)

    cost = env._cost_joint_pos_limits(
        RewardContext(
            info={},
            linvel=np.zeros((2, 3), dtype=np.float32),
            gyro=np.zeros((2, 3), dtype=np.float32),
            dof_pos=np.array([[0.0, 1.5], [-1.25, 0.0]], dtype=np.float32),
        )
    )

    np.testing.assert_allclose(cost, np.array([0.5, 0.25], dtype=np.float32))


def test_go2_footstand_reset_critic_height_uses_backend_sensor() -> None:
    pytest.importorskip("mujoco", reason="mujoco not installed")
    try:
        from mujoco.batch_env import BatchEnvPool as _  # noqa: F401
    except Exception:
        pytest.skip("mujoco.batch_env not available")

    ensure_registries()
    env = registry.make(
        "Go2FootStand",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={
            "reward_config": RewardConfig(
                scales={"height": 1.0},
                tracking_sigma=0.25,
                base_height_target=0.3,
            )
        },
    )
    try:
        state = env.init_state()
        assert state.obs["critic"][0, -1] > 0.1
    finally:
        env.close()
