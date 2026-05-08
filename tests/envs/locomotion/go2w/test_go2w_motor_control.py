from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from unilab.base.np_env import NpEnvState
from unilab.envs.locomotion.go2w.base import (
    DEFAULT_GO2W_ANGLES,
    JOINT_SENSOR_PREFIXES,
    NUM_GO2W_ACTIONS,
    NUM_LEG_ACTIONS,
    NUM_WHEEL_ACTIONS,
    Go2WBaseEnv,
    compute_go2w_motor_ctrl,
)
from unilab.envs.locomotion.go2w.joystick import (
    Go2WJoystickCfg,
    Go2WJoystickDomainRandomizationProvider,
    Go2WJoystickEnv,
    RewardConfig,
    build_go2w_backend_reset_randomization,
)


def _reward_config() -> RewardConfig:
    return RewardConfig(scales={"alive": 1.0}, tracking_sigma=0.25, base_height_target=0.3)


class _ConcreteGo2WBaseEnv(Go2WBaseEnv):
    def update_state(self, state: NpEnvState) -> NpEnvState:
        return state


def test_compute_go2w_motor_ctrl_converts_legs_and_tracks_wheel_velocity() -> None:
    policy_ctrl = np.zeros((1, NUM_GO2W_ACTIONS), dtype=np.float64)
    policy_ctrl[:, :NUM_LEG_ACTIONS] = 0.5
    policy_ctrl[:, NUM_LEG_ACTIONS:] = np.array([[2.0, -2.0, 20.0, -20.0]])
    joint_pos = np.zeros_like(policy_ctrl)
    joint_vel = np.ones_like(policy_ctrl) * 0.1
    leg_kp = np.ones((1, NUM_LEG_ACTIONS), dtype=np.float64) * 10.0
    leg_kd = np.ones((1, NUM_LEG_ACTIONS), dtype=np.float64) * 2.0
    wheel_kd = np.ones((1, NUM_WHEEL_ACTIONS), dtype=np.float64) * 2.0
    ctrl_range = np.tile(np.array([-15.0, 15.0]), (NUM_GO2W_ACTIONS, 1))
    out = np.zeros_like(policy_ctrl)

    result = compute_go2w_motor_ctrl(
        policy_ctrl,
        joint_pos,
        joint_vel,
        leg_kp,
        leg_kd,
        wheel_kd,
        ctrl_range[:, 0],
        ctrl_range[:, 1],
        out,
    )

    assert result is out
    np.testing.assert_allclose(result[:, :NUM_LEG_ACTIONS], 4.8)
    np.testing.assert_allclose(result[:, NUM_LEG_ACTIONS:], [[3.8, -4.2, 15.0, -15.0]])


def test_go2w_joint_state_reads_named_sensors() -> None:
    calls: list[str] = []

    class FakeBackend:
        def get_sensor_data(self, name: str) -> np.ndarray:
            calls.append(name)
            idx = len(calls)
            return np.array([[float(idx)], [float(idx + 100)]], dtype=np.float32)

    env = cast(Any, object.__new__(_ConcreteGo2WBaseEnv))
    env._backend = FakeBackend()
    env.default_angles = DEFAULT_GO2W_ANGLES.astype(np.float32)

    pos = env.get_dof_pos()
    vel = env.get_dof_vel()

    assert pos.shape == (2, NUM_GO2W_ACTIONS)
    assert vel.shape == (2, NUM_GO2W_ACTIONS)
    assert calls[:NUM_GO2W_ACTIONS] == [f"{prefix}_pos" for prefix in JOINT_SENSOR_PREFIXES]
    assert calls[NUM_GO2W_ACTIONS:] == [f"{prefix}_vel" for prefix in JOINT_SENSOR_PREFIXES]


def test_go2w_backend_reset_randomization_excludes_kp_kd_payload() -> None:
    cfg = Go2WJoystickCfg()
    cfg.domain_rand.randomize_base_mass = True
    cfg.domain_rand.random_com = True
    cfg.domain_rand.randomize_gravity = True
    cfg.domain_rand.randomize_kp = True
    cfg.domain_rand.randomize_kd = True

    payload = build_go2w_backend_reset_randomization(SimpleNamespace(cfg=cfg), num_reset=3)

    assert payload is not None
    assert payload.kp is None
    assert payload.kd is None
    assert payload.base_mass_delta is not None
    assert payload.base_com_offset is not None
    assert payload.gravity is not None


def test_go2w_reset_plan_can_disable_initial_yaw_randomization() -> None:
    cfg = Go2WJoystickCfg(reward_config=_reward_config())
    cfg.domain_rand.randomize_init_yaw = False
    env = SimpleNamespace(
        cfg=cfg,
        _env_origins=np.zeros((2, 3), dtype=np.float32),
        _init_qpos=np.concatenate(
            [np.array([0.0, 0.0, 0.42, 1.0, 0.0, 0.0, 0.0]), np.zeros(NUM_GO2W_ACTIONS)]
        ),
        _init_qvel=np.zeros(6 + NUM_GO2W_ACTIONS),
        _num_action=NUM_GO2W_ACTIONS,
        sample_reset_motor_gains=lambda num_reset: (
            np.ones((num_reset, NUM_LEG_ACTIONS)),
            np.ones((num_reset, NUM_LEG_ACTIONS)),
        ),
        set_motor_gains=lambda env_ids, motor_kp, motor_kd: None,
    )

    plan = Go2WJoystickDomainRandomizationProvider().build_reset_plan(
        env, np.array([0, 1], dtype=np.int32)
    )

    expected = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (2, 1))
    np.testing.assert_allclose(plan.qpos[:, 3:7], expected)


def test_go2w_init_does_not_pass_position_actuator_gains(monkeypatch: pytest.MonkeyPatch) -> None:
    from unilab.envs.locomotion.go2w import joystick as go2w_module

    captured: dict[str, Any] = {}

    class FakeBackend:
        backend_type = "mujoco"
        num_actuators = NUM_GO2W_ACTIONS

        def get_actuator_ctrl_range(self) -> np.ndarray:
            return np.tile(np.array([-15.0, 15.0]), (NUM_GO2W_ACTIONS, 1))

        def get_sensor_data(self, name: str) -> np.ndarray:
            if name.endswith("_pos") or name.endswith("_vel"):
                return np.zeros((2, 1), dtype=np.float32)
            raise KeyError(name)

        def set_pre_step_control(self, fn) -> None:
            captured["pre_step_control"] = fn

    def fake_create_backend(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeBackend()

    def fake_base_init(self, cfg, backend, num_envs):
        self._cfg = cfg
        self._backend = backend
        self._num_envs = num_envs
        self._num_action = NUM_GO2W_ACTIONS
        self.default_angles = DEFAULT_GO2W_ANGLES.astype(np.float32)

    monkeypatch.setattr(go2w_module, "create_backend", fake_create_backend)
    monkeypatch.setattr(Go2WBaseEnv, "__init__", fake_base_init)
    monkeypatch.setattr(Go2WJoystickEnv, "_init_domain_randomization", lambda self, provider: None)

    cfg = Go2WJoystickCfg(reward_config=_reward_config())
    Go2WJoystickEnv(cfg, num_envs=2, backend_type="mujoco")

    assert captured["args"][0] == "mujoco"
    assert "position_actuator_gains" not in captured["kwargs"]
    assert callable(captured["pre_step_control"])


def test_go2w_apply_action_maps_legs_to_targets_and_wheels_to_velocity_targets() -> None:
    env = cast(Any, object.__new__(Go2WJoystickEnv))
    env._cfg = Go2WJoystickCfg(reward_config=_reward_config())
    env._np_dtype = np.float32
    env._num_envs = 1
    env._num_action = NUM_GO2W_ACTIONS
    env.default_angles = DEFAULT_GO2W_ANGLES.astype(np.float32)
    state = NpEnvState(
        obs={},
        reward=np.zeros((1,), dtype=np.float32),
        terminated=np.zeros((1,), dtype=bool),
        truncated=np.zeros((1,), dtype=bool),
        info={},
    )
    action = np.ones((1, NUM_GO2W_ACTIONS), dtype=np.float32)

    ctrl = env.apply_action(action, state)

    expected_leg_targets = (DEFAULT_GO2W_ANGLES[:NUM_LEG_ACTIONS] + 0.25).reshape(1, -1)
    np.testing.assert_allclose(ctrl[:, :NUM_LEG_ACTIONS], expected_leg_targets)
    np.testing.assert_allclose(ctrl[:, NUM_LEG_ACTIONS:], 10.0)
    np.testing.assert_allclose(state.info["current_actions"], action)


def test_go2w_compute_obs_accepts_reset_subset_info_shapes() -> None:
    env = cast(Any, object.__new__(Go2WJoystickEnv))
    env._cfg = Go2WJoystickCfg(reward_config=_reward_config())
    env._num_envs = 2
    env._num_action = NUM_GO2W_ACTIONS
    env.default_angles = DEFAULT_GO2W_ANGLES.astype(np.float32)
    info = {
        "current_actions": np.zeros((1, NUM_GO2W_ACTIONS), dtype=np.float32),
        "commands": np.zeros((1, 3), dtype=np.float32),
        "torques": np.ones((1, NUM_GO2W_ACTIONS), dtype=np.float32),
    }
    linvel = np.zeros((1, 3), dtype=np.float32)
    gyro = np.zeros((1, 3), dtype=np.float32)
    gravity = np.zeros((1, 3), dtype=np.float32)
    dof_pos = DEFAULT_GO2W_ANGLES.reshape(1, -1).astype(np.float32)
    dof_vel = np.zeros((1, NUM_GO2W_ACTIONS), dtype=np.float32)

    obs = env._compute_obs(info, linvel, gyro, gravity, dof_pos, dof_vel)

    assert obs["obs"].shape == (1, env.obs_groups_spec["obs"])
    assert obs["critic"].shape == (1, env.obs_groups_spec["critic"])
    np.testing.assert_allclose(obs["critic"][:, -NUM_GO2W_ACTIONS:], 1.0)


def test_go2w_heading_command_updates_yaw_rate_from_heading_error() -> None:
    class FakeBackend:
        def get_base_quat(self) -> np.ndarray:
            return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    env = cast(Any, object.__new__(Go2WJoystickEnv))
    env._cfg = Go2WJoystickCfg(reward_config=_reward_config())
    env._cfg.commands.heading_command = True
    env._num_envs = 1
    env._backend = FakeBackend()
    info = {
        "commands": np.zeros((1, 3), dtype=np.float32),
        "heading_commands": np.array([np.pi / 2.0], dtype=np.float32),
        "steps": np.zeros((1,), dtype=np.uint32),
    }

    env._update_commands(info)

    np.testing.assert_allclose(info["commands"][:, 2], [0.25 * np.pi], rtol=1e-6)


def test_go2w_pre_step_motor_control_reads_from_passed_backend() -> None:
    calls: list[str] = []

    class FakeBackend:
        def get_sensor_data(self, name: str) -> np.ndarray:
            calls.append(name)
            if name.endswith("_pos"):
                return np.zeros((1, 1), dtype=np.float32)
            if name.endswith("_vel"):
                return np.ones((1, 1), dtype=np.float32) * 0.1
            raise KeyError(name)

    class PoisonBackend:
        def get_sensor_data(self, name: str) -> np.ndarray:
            raise AssertionError(f"unexpected env backend read: {name}")

    env = cast(Any, object.__new__(Go2WJoystickEnv))
    env._backend = PoisonBackend()
    env.default_angles = DEFAULT_GO2W_ANGLES.astype(np.float32)
    env._np_dtype = np.float32
    env._motor_kp = np.ones((1, NUM_LEG_ACTIONS), dtype=np.float64) * 10.0
    env._motor_kd = np.ones((1, NUM_LEG_ACTIONS), dtype=np.float64) * 2.0
    env._wheel_kd = np.ones((1, NUM_WHEEL_ACTIONS), dtype=np.float64) * 2.0
    ctrl_range = np.tile(np.array([-15.0, 15.0]), (NUM_GO2W_ACTIONS, 1))
    env._ctrl_lower = ctrl_range[:, 0].astype(np.float32)
    env._ctrl_upper = ctrl_range[:, 1].astype(np.float32)
    env._last_motor_ctrl = np.zeros((1, NUM_GO2W_ACTIONS), dtype=np.float32)
    policy_ctrl = np.zeros((1, NUM_GO2W_ACTIONS), dtype=np.float32)
    policy_ctrl[:, :NUM_LEG_ACTIONS] = 0.5
    policy_ctrl[:, NUM_LEG_ACTIONS:] = np.array([[1.0, -1.0, 2.0, -2.0]], dtype=np.float32)

    motor_ctrl = env._pre_step_motor_control(FakeBackend(), policy_ctrl)

    assert calls[:NUM_GO2W_ACTIONS] == [f"{prefix}_pos" for prefix in JOINT_SENSOR_PREFIXES]
    assert calls[NUM_GO2W_ACTIONS:] == [f"{prefix}_vel" for prefix in JOINT_SENSOR_PREFIXES]
    np.testing.assert_allclose(motor_ctrl[:, :NUM_LEG_ACTIONS], 4.8)
    np.testing.assert_allclose(motor_ctrl[:, NUM_LEG_ACTIONS:], [[1.8, -2.2, 3.8, -4.2]])
