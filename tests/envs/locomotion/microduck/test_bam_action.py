"""Focused tests for the MicroDuck BAM voltage-actuator action term."""

from __future__ import annotations

import inspect
import math
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
from unilab.managers._types import ManagerBasedRlEnv
from unilab.tasks.locomotion.microduck import bam_action
from unilab.tasks.locomotion.microduck.bam_action import (
    BamVoltageAction,
    BamVoltageActionCfg,
)

ROOT_DIR = Path(__file__).parents[4]
CONF_DIR = ROOT_DIR / "src" / "unilab" / "conf" / "ppo"

KT = 0.36601349688984386
RESISTANCE = 2.8113923539223227
ERROR_GAIN = (4096.0 / (2.0 * math.pi)) / (256.0 * 885.0)
KP_FW = 200.0
MAX_CURRENT = 1.75
VIN = 7.5

_JOINT_NAMES = ("left_knee", "right_knee")


class _FakeEntityData:
    def __init__(self, num_envs: int, num_joints: int) -> None:
        self.joint_pos = np.zeros((num_envs, num_joints), dtype=np.float32)
        self.joint_vel = np.zeros((num_envs, num_joints), dtype=np.float32)
        self.default_joint_pos = np.zeros((num_envs, num_joints), dtype=np.float32)
        self.encoder_bias = np.zeros((num_envs, num_joints), dtype=np.float32)
        self.actuator_ctrl_range = np.tile([-1.0676, 1.0676], (num_joints, 1)).astype(np.float32)
        self.written: np.ndarray | None = None
        self.written_actuator_ids: np.ndarray | None = None

    def write_ctrl(self, values, env_ids=None, *, actuator_ids=None) -> None:
        self.written = np.asarray(values, dtype=np.float64).copy()
        self.written_actuator_ids = None if actuator_ids is None else np.asarray(actuator_ids)


class _FakeEntity:
    def __init__(self, num_envs: int, joint_names: tuple[str, ...]) -> None:
        self.joint_names = list(joint_names)
        self.actuator_names = list(joint_names)
        self.data = _FakeEntityData(num_envs, len(joint_names))

    def find_actuators(self, keys, preserve_order: bool = False):
        return list(range(len(self.actuator_names))), list(self.actuator_names)

    def find_joints_by_actuator_names(self, keys):
        return list(range(len(self.joint_names))), list(self.joint_names)


def _make_env(num_envs: int = 2, seed: int = 0) -> tuple[ManagerBasedRlEnv, _FakeEntity]:
    entity = _FakeEntity(num_envs, _JOINT_NAMES)
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=num_envs,
            physics_dt=0.005,
            rng=np.random.default_rng(seed),
            scene={"robot": entity},
        ),
    )
    return env, entity


def _frictionless_cfg(**overrides: Any) -> BamVoltageActionCfg:
    params: dict[str, Any] = dict(
        entity_name="robot",
        actuator_names=[".*"],
        scale=1.0,
        kp_fw=KP_FW,
        vin_range=(VIN, VIN),
        vin_drop_gain_range=(0.0, 0.0),
        delay_min_lag=0,
        delay_max_lag=0,
        friction_scale_range=(1.0, 1.0),
        friction_base=0.0,
        friction_stribeck=0.0,
        load_friction_motor=0.0,
        load_friction_external=0.0,
        load_friction_motor_stribeck=0.0,
        load_friction_external_stribeck=0.0,
        load_friction_motor_quad=0.0,
        load_friction_external_quad=0.0,
        friction_viscous=0.0,
    )
    params.update(overrides)
    return BamVoltageActionCfg(**params)


def _make_term(
    cfg: BamVoltageActionCfg | None = None, num_envs: int = 2, seed: int = 0
) -> tuple[BamVoltageAction, _FakeEntity]:
    env, entity = _make_env(num_envs=num_envs, seed=seed)
    term = BamVoltageAction(cfg or _frictionless_cfg(), env)
    return term, entity


def _expected_torque(delta_q: float, dq: float, vin: float = VIN) -> float:
    duty = delta_q * KP_FW * ERROR_GAIN
    center = KT * dq / vin
    span = RESISTANCE * MAX_CURRENT / vin
    duty = min(max(duty, center - span), center + span)
    duty = min(max(duty, -1.0), 1.0)
    volts = vin * duty
    return (KT * volts - KT**2 * dq) / RESISTANCE


def test_bam_voltage_action_declares_substep_state_feedback() -> None:
    assert BamVoltageAction.requires_substep_state_feedback is True


def test_voltage_law_matches_firmware_p_controller() -> None:
    term, entity = _make_term()
    entity.data.joint_vel[:] = 0.5
    term.process_actions(np.full((2, 2), 0.1, dtype=np.float32))
    term.apply_actions()
    expected = _expected_torque(0.1, 0.5)
    np.testing.assert_allclose(entity.data.written, expected, rtol=1e-5)
    np.testing.assert_allclose(term.motor_torque, expected, rtol=1e-5)


def test_current_limit_window_clamps_duty() -> None:
    term, entity = _make_term()
    # dq < 0 shifts the duty window down so the P-term output saturates the
    # upper window edge instead of the PWM limit.
    entity.data.joint_vel[:] = -3.0
    term.process_actions(np.full((2, 2), 1.0, dtype=np.float32))
    term.apply_actions()
    duty = 1.0 * KP_FW * ERROR_GAIN
    upper = KT * -3.0 / VIN + RESISTANCE * MAX_CURRENT / VIN
    assert duty > upper  # window clamp actually engaged
    np.testing.assert_allclose(entity.data.written, _expected_torque(1.0, -3.0), rtol=1e-5)


def test_pwm_limit_clamps_duty_to_unit_range() -> None:
    term, entity = _make_term()
    # Large dq pushes the current-limit window past the PWM bound (window
    # upper edge = kt*dq/vin + R*Imax/vin = 0.488 + 0.656 = 1.144 > 1), so the
    # physical PWM clamp is the binding constraint.
    entity.data.joint_vel[:] = 10.0
    term.process_actions(np.full((2, 2), 5.0, dtype=np.float32))
    term.apply_actions()
    assert 5.0 * KP_FW * ERROR_GAIN > 1.0  # duty demand exceeds the PWM limit
    expected = (KT * VIN - KT**2 * 10.0) / RESISTANCE
    np.testing.assert_allclose(entity.data.written, expected, rtol=1e-5)


def test_back_emf_produces_braking_torque_at_zero_error() -> None:
    term, entity = _make_term()
    entity.data.joint_vel[:] = 2.0
    term.process_actions(np.zeros((2, 2), dtype=np.float32))
    term.apply_actions()
    np.testing.assert_allclose(entity.data.written, -(KT**2) * 2.0 / RESISTANCE, rtol=1e-5)


def test_encoder_bias_shifts_target_before_delay_buffer() -> None:
    term, entity = _make_term()
    entity.data.default_joint_pos[:] = 0.3
    entity.data.encoder_bias[:] = 0.02
    term.process_actions(np.zeros((2, 2), dtype=np.float32))
    term.apply_actions()
    # target = 0 + 0.3 - 0.02 -> delta_q = 0.28, dq = 0.
    np.testing.assert_allclose(entity.data.written, _expected_torque(0.28, 0.0), rtol=1e-5)


def test_delay_buffer_serves_lifo_frame_with_fixed_lag() -> None:
    env, entity = _make_env(num_envs=1)
    cfg = _frictionless_cfg(delay_min_lag=3, delay_max_lag=3)
    term = BamVoltageAction(cfg, env)
    values = [0.1 * k for k in range(1, 8)]
    served = []
    for value in values:
        term.process_actions(np.full((1, 2), value, dtype=np.float32))
        term.apply_actions()
        served.append(term._delay_read().copy())
    # Append+read per substep: with lag=3 the read at append t serves the
    # frame from t-3 (clamped to the oldest available while history fills).
    np.testing.assert_allclose(served[0], values[0])
    np.testing.assert_allclose(served[1], values[0])
    np.testing.assert_allclose(served[2], values[0])
    np.testing.assert_allclose(served[3], values[0])
    np.testing.assert_allclose(served[4], values[1])
    np.testing.assert_allclose(served[5], values[2])
    np.testing.assert_allclose(served[6], values[3])


def test_delay_buffer_reset_backfills_history_with_first_frame() -> None:
    env, entity = _make_env(num_envs=2)
    cfg = _frictionless_cfg(delay_min_lag=3, delay_max_lag=6)
    term = BamVoltageAction(cfg, env)
    term.process_actions(np.full((2, 2), 0.5, dtype=np.float32))
    for _ in range(8):
        term.apply_actions()
    term.reset(np.asarray([0], dtype=np.int32))
    assert term._delay_pushes[0] == 0
    assert term._delay_pushes[1] == 8
    term.process_actions(np.array([[0.9, 0.9], [0.5, 0.5]], dtype=np.float32))
    term.apply_actions()
    served = term._delay_read()
    # Reset row: lag clamps to the single available frame (the fresh target).
    np.testing.assert_allclose(served[0], 0.9, rtol=1e-6)
    # Untouched row keeps serving its own (stale) history.
    np.testing.assert_allclose(served[1], 0.5, rtol=1e-6)


def test_delay_lag_stays_within_sampled_range() -> None:
    env, _ = _make_env(num_envs=4)
    cfg = _frictionless_cfg(delay_min_lag=3, delay_max_lag=6)
    term = BamVoltageAction(cfg, env)
    # Fill the buffer with 32 distinct constant frames; each read must return
    # one of the frames 3..6 appends back.
    for k in range(32):
        term.process_actions(np.full((4, 2), float(k), dtype=np.float32))
        term.apply_actions()
    read = term._delay_read()
    allowed = {float(k) for k in range(32 - 7, 32 - 2)}
    assert set(np.unique(read)) <= allowed


def test_friction_budget_matches_m6_formula() -> None:
    term, _ = _make_term(
        cfg=_frictionless_cfg(
            friction_base=0.004771183165566,
            friction_stribeck=0.004676345799486616,
            load_friction_motor=0.2667860954283698,
            load_friction_external=8.515871897059342e-06,
            load_friction_motor_stribeck=1.0722918395099123e-05,
            load_friction_external_stribeck=0.08077928978935671,
            load_friction_motor_quad=0.009972471242139415,
            load_friction_external_quad=0.004902565732332559,
        )
    )
    term._friction_scale[:] = 1.0
    motor = np.array([[0.5, 0.5]])
    external = np.array([[0.1, 0.6]])
    stribeck = np.ones((1, 2))
    budget = term._friction_budget(motor, external, stribeck)

    def expected(tau_mot: float, tau_ext: float) -> float:
        gearbox = abs(tau_ext * 8.515871897059342e-06 - tau_mot * 0.2667860954283698)
        gearbox_stribeck = abs(tau_ext * 0.08077928978935671 - tau_mot * 1.0722918395099123e-05)
        drive = 1.0 if abs(tau_mot) > abs(tau_ext) else 0.0
        quad = (
            drive * 0.004902565732332559 * abs(tau_ext) ** 2
            + (1.0 - drive) * 0.009972471242139415 * abs(tau_mot) ** 2
        )
        return 0.004771183165566 + 0.004676345799486616 + gearbox + gearbox_stribeck + quad

    np.testing.assert_allclose(budget[0, 0], expected(0.5, 0.1), rtol=1e-6)
    np.testing.assert_allclose(budget[0, 1], expected(0.5, 0.6), rtol=1e-6)


def test_friction_scale_multiplies_budget_and_is_per_env() -> None:
    term, _ = _make_term(cfg=_frictionless_cfg(friction_base=0.01))
    term._friction_scale[0] = 1.1
    term._friction_scale[1] = 0.9
    zeros = np.zeros((2, 2))
    budget = term._friction_budget(zeros, zeros, np.ones((2, 2)))
    np.testing.assert_allclose(budget[:, 0], [0.011, 0.009], rtol=1e-6)


def test_stiction_clip_holds_quasi_static_joint() -> None:
    cfg = _frictionless_cfg(
        friction_base=0.004771183165566,
        friction_stribeck=0.004676345799486616,
        load_friction_motor=0.2667860954283698,
        load_friction_external=8.515871897059342e-06,
        load_friction_motor_stribeck=1.0722918395099123e-05,
        load_friction_external_stribeck=0.08077928978935671,
        load_friction_motor_quad=0.009972471242139415,
        load_friction_external_quad=0.004902565732332559,
        friction_viscous=0.005359668274599504,
    )
    term, entity = _make_term(cfg=cfg)
    entity.data.joint_vel[:] = 5.0e-4  # below the quasi-static threshold
    # Row 0: tiny error -> torque below the friction budget -> held at zero.
    # Row 1: large error -> torque breaks friction -> dry + viscous friction apply.
    term.process_actions(np.array([[0.001, 0.001], [0.5, 0.5]], dtype=np.float32))
    term.apply_actions()
    # Row 0 holds via stiction; only the unconditional viscous term remains.
    np.testing.assert_allclose(entity.data.written[0], -0.005359668274599504 * 5.0e-4, rtol=1e-5)
    motor_row1 = _expected_torque(0.5, 5.0e-4)
    # First substep: prev motor/applied torques are zero, so tau_ext ~= 0 and
    # all load-dependent budget terms vanish; only base + Stribeck remain.
    stribeck = math.exp(-((5.0e-4 / 2.890372094130307) ** 8.683259907618984))
    budget_row1 = 0.004771183165566 + stribeck * 0.004676345799486616
    expected_row1 = motor_row1 - budget_row1 * 1.0 - 0.005359668274599504 * 5.0e-4
    np.testing.assert_allclose(entity.data.written[1], expected_row1, rtol=1e-4)


def test_reset_clears_episode_state_and_resamples_friction_scale() -> None:
    env, entity = _make_env(num_envs=2)
    cfg = _frictionless_cfg(
        friction_base=0.01, delay_min_lag=3, delay_max_lag=6, friction_scale_range=(0.9, 1.1)
    )
    term = BamVoltageAction(cfg, env)
    vin_before = term._vin.copy()
    term.process_actions(np.full((2, 2), 0.4, dtype=np.float32))
    for _ in range(4):
        term.apply_actions()
    assert np.any(term._prev_motor_torque != 0.0)
    scale_before = term._friction_scale.copy()

    term.reset(np.asarray([0], dtype=np.int32))
    assert term._delay_pushes[0] == 0
    np.testing.assert_allclose(term._prev_motor_torque[0], 0.0)
    np.testing.assert_allclose(term._prev_applied_torque[0], 0.0)
    np.testing.assert_allclose(term._prev_joint_vel[0], 0.0)
    np.testing.assert_allclose(term._motor_torque[0], 0.0)
    assert 0.9 <= term._friction_scale[0, 0] <= 1.1
    np.testing.assert_allclose(term._vin, vin_before)  # startup semantics: held
    # Row 1 untouched.
    np.testing.assert_allclose(term._friction_scale[1], scale_before[1])
    assert term._delay_pushes[1] == 4

    # After reset the delayed target is the fresh target, not stale history.
    term.process_actions(np.array([[0.7, 0.7], [0.4, 0.4]], dtype=np.float32))
    term.apply_actions()
    np.testing.assert_allclose(term._delay_read()[0], 0.7, rtol=1e-6)


def test_process_actions_rejects_bad_input() -> None:
    term, _ = _make_term()
    with pytest.raises(ValueError, match="NaN or Inf"):
        term.process_actions(np.full((2, 2), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="expected action shape"):
        term.process_actions(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(TypeError, match="np.ndarray"):
        term.process_actions([[0.0, 0.0]])


def test_cfg_validation_fails_closed() -> None:
    base: dict[str, Any] = dict(entity_name="robot", actuator_names=[".*"])
    env, _ = _make_env()
    with pytest.raises(ValueError, match="kp_fw"):
        BamVoltageAction(BamVoltageActionCfg(**base, kp_fw=-1.0), env)
    with pytest.raises(ValueError, match="delay_min_lag"):
        BamVoltageAction(BamVoltageActionCfg(**base, delay_min_lag=7, delay_max_lag=3), env)
    with pytest.raises(ValueError, match="vin_range"):
        BamVoltageAction(BamVoltageActionCfg(**base, vin_range=(8.2, 6.5)), env)
    with pytest.raises(TypeError, match="scale"):
        BamVoltageAction(BamVoltageActionCfg(**base, scale="1.0"), env)
    with pytest.raises(ValueError, match="kt"):
        BamVoltageAction(BamVoltageActionCfg(**base, kt=0.0), env)


def test_bam_action_source_does_not_probe_backend_privates() -> None:
    source = inspect.getsource(bam_action)
    for forbidden in ("._backend", "getattr(", "hasattr(", "qpos", "qvel", "ASSETS_ROOT_PATH"):
        assert forbidden not in source


def _materialize(task: str, task_name: str) -> tuple[Any, ManagerBasedRlEnvCfg]:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose("config", overrides=[f"task={task}/mujoco"])
    registry.ensure_registries()
    override = BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="ppo").build_task_env_cfg_override()
    env_cfg = registry.materialize_env_config(task_name)
    assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(env_cfg, override)
    env_cfg.validate()
    return cfg, env_cfg


def test_bam_owner_materializes_and_matches_pd_recipe() -> None:
    cfg, env_cfg = _materialize("microduck_velocity_bam_flat", "MicroduckVelocityBamFlat")
    _, pd_env_cfg = _materialize("microduck_velocity_flat", "MicroduckVelocityFlat")

    assert cfg.training.task_name == "MicroduckVelocityBamFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert env_cfg.scene is not None
    assert env_cfg.scene.model_file.endswith("robots/microduck/scene_flat_bam.xml")
    assert env_cfg.sim_dt == pytest.approx(0.005)
    assert env_cfg.ctrl_dt == pytest.approx(0.02)

    assert list(env_cfg.actions) == ["joint_pos"]
    action = env_cfg.actions["joint_pos"]
    assert isinstance(action, BamVoltageActionCfg)
    assert action.kp_fw == pytest.approx(200.0)
    assert tuple(action.vin_range) == (6.5, 8.2)
    assert tuple(action.vin_drop_gain_range) == (0.0, 0.2)
    assert action.vin_min == pytest.approx(6.0)
    assert action.delay_min_lag == 3
    assert action.delay_max_lag == 6
    assert tuple(action.friction_scale_range) == (0.9, 1.1)

    # Controlled actuator experiment: reward / observation / command /
    # termination / curriculum stacks are identical to the PD owner.
    assert {name: term.weight for name, term in env_cfg.rewards.items()} == {
        name: term.weight for name, term in pd_env_cfg.rewards.items()
    }
    assert list(env_cfg.observations) == list(pd_env_cfg.observations)
    assert list(env_cfg.commands) == list(pd_env_cfg.commands)
    assert list(env_cfg.terminations) == list(pd_env_cfg.terminations)
    assert list(env_cfg.curriculum) == list(pd_env_cfg.curriculum)
    # Events: same DR stack as the PD owner (encoder bias, armature, CoM,
    # mass/inertia, foot friction); BAM-specific DR lives inside the term.
    assert list(env_cfg.events) == list(pd_env_cfg.events)

    registered = registry.list_registered_envs()
    assert registered["MicroduckVelocityBamFlat"]["available_backends"] == ["mujoco"]


@pytest.mark.slow
def test_bam_owner_builds_and_steps_real_mujoco_env() -> None:
    pytest.importorskip("mujoco")
    cfg, _ = _materialize("microduck_velocity_bam_flat", "MicroduckVelocityBamFlat")
    env = cast(
        Any,
        registry.make(
            "MicroduckVelocityBamFlat",
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
        obs, info = env.reset(seed=7)
        assert isinstance(info, dict)
        action = env.action_manager.get_term("joint_pos")
        assert isinstance(action, BamVoltageAction)
        assert env._uses_pre_step_control is True
        state = env.step(np.zeros((2, 14), dtype=np.float32))
        assert np.isfinite(state.obs["obs"]).all()
        assert np.isfinite(state.reward).all()
        assert np.isfinite(action.motor_torque).all()
        assert np.isfinite(env._control).all()
    finally:
        env.close()
