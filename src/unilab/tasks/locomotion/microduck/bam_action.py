"""BAM voltage-controlled actuator action term owned by the MicroDuck task.

NumPy port of the upstream ``bam`` xl330-m6 voltage servo (microduck_rl,
``bam.mjlab.BamActuator`` + ``bam.actuator.VoltageControlledActuator``) onto the
Manager-Based runtime.  The term recomputes motor torque on every physics
substep (``requires_substep_state_feedback=True``) through the
``SimBackend.set_pre_step_control`` contract, mirroring the firmware's 200 Hz
position loop under the 50 Hz policy.

Per-substep pipeline (all shapes ``(num_envs, num_joints)``):

1. Command delay: the encoder-biased position target is pushed into a LIFO
   ring buffer (capacity ``delay_max_lag + 1``); each substep resamples a
   per-env lag uniformly in ``[delay_min_lag, delay_max_lag]`` and serves the
   frame ``lag`` substeps old (clamped to available history).  Reset clears a
   row's history; the next append backfills every slot with the first value.
2. Per-env supply voltage: ``vin_eff = max(vin - vin_drop_gain * Σ|τ_motor_prev|,
   vin_min)`` where ``vin`` / ``vin_drop_gain`` are sampled once at startup and
   held across resets.
3. Firmware P law: ``duty = (q_target - q) * kp_fw * error_gain``, clipped to
   the firmware current-limit duty window
   ``[kt·dq/vin_eff ± R·max_current/vin_eff]`` and then to the physical PWM
   range; ``volts = vin_eff * duty``.
4. DC motor torque with back-EMF: ``τ_motor = (kt·volts - kt²·dq) / R``.  The
   XML ``<motor>`` actuators carry ``forcerange = max(vin_range)·kt/R`` so the
   solver performs the final truncation; the term does not clip its output.
5. BAM m6 friction budget (Coulomb + Stribeck + directional load terms +
   quadratic term), scaled per env by ``friction_scale`` (resampled at every
   episode reset; non-accumulating by assignment).

Approximation boundaries versus upstream (recorded for issue #1474):

- Upstream writes the friction budget into ``dof_frictionloss``/``dof_damping``
  and lets MuJoCo's constraint solver apply static-friction clipping.  The
  UniLab ``SimBackend`` contract has no per-substep write channel for those
  fields, so the budget is folded into the output torque instead: the stiction
  clip is replicated in the torque domain (zero output when the joint is
  quasi-static and the motor torque cannot break friction), and the viscous
  term is subtracted as ``friction_viscous * dq``.
- Upstream estimates the gearbox external torque from ``qfrc_bias`` /
  ``qfrc_constraint`` / ``qfrc_friction``; none of those are exposed by the
  ``SimBackend`` contract.  Here ``τ_ext`` is a finite-difference estimate
  ``I_eff·(dq_t - dq_{t-1})/sim_dt - τ_applied_prev`` with ``I_eff`` equal to
  the nominal ``dof_armature`` (link inertia deliberately not faked).  Because
  the applied-torque path already contains the modeled friction, the estimate
  excludes it, matching the upstream exclusion of ``qfrc_friction``.  The
  steady-state behavior is unaffected; only fast transients underestimate the
  inertial reaction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.managers import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv

# XL330 firmware scaling (bam/dynamixel/actuator.py): the position error is
# converted to a PWM duty cycle by kp * error_gain, where error_gain maps
# radians to encoder counts and then to the KP divisor / PWM limit domain.
_XL330_ERROR_GAIN = (4096.0 / (2.0 * math.pi)) / (256.0 * 885.0)

# Quasi-static velocity threshold for the torque-domain stiction clip.
_DQ_EPS = 1.0e-3


def _real(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and (result <= minimum if strict_minimum else result < minimum):
        relation = "greater than" if strict_minimum else "at least"
        raise ValueError(f"{label} must be {relation} {minimum}")
    return result


def _int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer, got {type(value).__name__}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _range(value: Any, *, label: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{label} must be a two-value range")
    lower = _real(value[0], label=f"{label} lower", minimum=0.0)
    upper = _real(value[1], label=f"{label} upper", minimum=0.0)
    if lower > upper:
        raise ValueError(f"{label} lower {lower} exceeds upper {upper}")
    return lower, upper


@dataclass(kw_only=True)
class BamVoltageActionCfg(ActionTermCfg):
    """Configure the BAM xl330-m6 voltage servo action term.

    The electrical/friction defaults are the fitted upstream parameters
    (``bam/params/xl330/m6.json``); the task-facing knobs (firmware gain,
    voltage/delay/friction DR ranges) are declared explicitly in the owner
    YAML.
    """

    actuator_names: tuple[str, ...] | list[str]
    scale: float = 1.0
    # Firmware P gain (Dynamixel KP register domain; microduck keeps 200).
    kp_fw: float = 200.0
    error_gain: float = _XL330_ERROR_GAIN
    # Fitted electrical parameters (bam xl330 m6).
    kt: float = 0.36601349688984386
    resistance: float = 2.8113923539223227
    armature: float = 0.0018077432831600838
    max_current: float = 1.75
    max_pwm: float = 1.0
    # Per-env battery model (startup-sampled, held across resets).
    vin_range: tuple[float, float] | list[float] = (6.5, 8.2)
    vin_drop_gain_range: tuple[float, float] | list[float] = (0.0, 0.2)
    vin_min: float = 6.0
    # Command delay in physics substeps; lag resampled every substep.
    delay_min_lag: int = 3
    delay_max_lag: int = 6
    # Per-episode multiplier on the velocity-independent friction budget.
    friction_scale_range: tuple[float, float] | list[float] = (0.9, 1.1)
    # Fitted BAM m6 friction parameters (bam xl330 m6).
    friction_base: float = 0.004771183165566
    friction_stribeck: float = 0.004676345799486616
    load_friction_motor: float = 0.2667860954283698
    load_friction_external: float = 8.515871897059342e-06
    load_friction_motor_stribeck: float = 1.0722918395099123e-05
    load_friction_external_stribeck: float = 0.08077928978935671
    load_friction_motor_quad: float = 0.009972471242139415
    load_friction_external_quad: float = 0.004902565732332559
    dtheta_stribeck: float = 2.890372094130307
    stribeck_alpha: float = 8.683259907618984
    friction_viscous: float = 0.005359668274599504

    def build(self, env: ManagerBasedRlEnv) -> BamVoltageAction:
        return BamVoltageAction(self, env)


class BamVoltageAction(ActionTerm):
    """Convert policy actions into BAM voltage-servo motor torques per substep."""

    requires_substep_state_feedback: ClassVar[bool] = True
    cfg: BamVoltageActionCfg
    _entity: Entity

    def __init__(self, cfg: BamVoltageActionCfg, env: ManagerBasedRlEnv):
        self._validate_cfg(cfg)
        super().__init__(cfg=cfg, env=env)
        actuator_ids, actuator_names = self._entity.find_actuators(cfg.actuator_names)
        joint_ids, joint_names = self._entity.find_joints_by_actuator_names(cfg.actuator_names)
        if not actuator_ids or len(actuator_ids) != len(joint_ids):
            raise ValueError(
                "BamVoltageAction requires a non-empty 1:1 actuator/joint selection; "
                f"received actuators={actuator_names}, joints={joint_names}"
            )
        self._actuator_ids = np.asarray(actuator_ids, dtype=np.intp)
        self._joint_ids = np.asarray(joint_ids, dtype=np.intp)
        self._actuator_ids.setflags(write=False)
        self._joint_ids.setflags(write=False)
        num_joints = len(self._joint_ids)

        dtype = get_global_dtype()
        shape = (self.num_envs, num_joints)
        self._raw_action = np.zeros(shape, dtype=dtype)
        self._processed_action = np.zeros(shape, dtype=dtype)
        self._motor_torque = np.zeros(shape, dtype=dtype)
        self._prev_motor_torque = np.zeros(shape, dtype=dtype)
        self._prev_applied_torque = np.zeros(shape, dtype=dtype)
        self._prev_joint_vel = np.zeros(shape, dtype=dtype)

        # LIFO command-delay ring buffer with per-env history depth; frames are
        # appended every physics substep (the 50 Hz policy target repeats for
        # all substeps of one control step).
        capacity = self._delay_max_lag + 1
        self._delay_buffer = np.zeros((capacity, self.num_envs, num_joints), dtype=dtype)
        self._delay_pointer = -1
        self._delay_pushes = np.zeros((self.num_envs,), dtype=np.int64)

        # Startup-sampled battery model: constant per env across resets
        # (mirrors the upstream BamActuator.initialize semantics).
        self._vin = np.asarray(
            self._env.rng.uniform(*self._vin_range, size=(self.num_envs, 1)), dtype=dtype
        )
        self._vin_drop_gain = np.asarray(
            self._env.rng.uniform(*self._vin_drop_gain_range, size=(self.num_envs, 1)),
            dtype=dtype,
        )
        self._friction_scale = np.ones((self.num_envs, 1), dtype=dtype)

        ctrl_range = self._entity.data.actuator_ctrl_range[self._actuator_ids]
        expected_range_shape = (num_joints, 2)
        if ctrl_range.shape != expected_range_shape:
            raise ValueError(
                "BamVoltageAction actuator control range must have shape "
                f"{expected_range_shape}, got {ctrl_range.shape}"
            )

    def _validate_cfg(self, cfg: BamVoltageActionCfg) -> None:
        label = "BamVoltageActionCfg"
        self._scale = _real(cfg.scale, label=f"{label} scale")
        self._kp_fw = _real(cfg.kp_fw, label=f"{label} kp_fw", minimum=0.0)
        self._error_gain = _real(cfg.error_gain, label=f"{label} error_gain", minimum=0.0)
        self._kt = _real(cfg.kt, label=f"{label} kt", minimum=0.0, strict_minimum=True)
        self._resistance = _real(
            cfg.resistance, label=f"{label} resistance", minimum=0.0, strict_minimum=True
        )
        self._armature = _real(cfg.armature, label=f"{label} armature", minimum=0.0)
        self._max_current = _real(
            cfg.max_current, label=f"{label} max_current", minimum=0.0, strict_minimum=True
        )
        self._max_pwm = _real(
            cfg.max_pwm, label=f"{label} max_pwm", minimum=0.0, strict_minimum=True
        )
        self._vin_range = _range(cfg.vin_range, label=f"{label} vin_range")
        self._vin_drop_gain_range = _range(
            cfg.vin_drop_gain_range, label=f"{label} vin_drop_gain_range"
        )
        self._vin_min = _real(
            cfg.vin_min, label=f"{label} vin_min", minimum=0.0, strict_minimum=True
        )
        self._delay_min_lag = _int(cfg.delay_min_lag, label=f"{label} delay_min_lag")
        self._delay_max_lag = _int(cfg.delay_max_lag, label=f"{label} delay_max_lag")
        if self._delay_min_lag > self._delay_max_lag:
            raise ValueError(
                f"{label} delay_min_lag {self._delay_min_lag} exceeds "
                f"delay_max_lag {self._delay_max_lag}"
            )
        self._friction_scale_range = _range(
            cfg.friction_scale_range, label=f"{label} friction_scale_range"
        )
        for name, value in (
            ("friction_base", cfg.friction_base),
            ("friction_stribeck", cfg.friction_stribeck),
            ("load_friction_motor", cfg.load_friction_motor),
            ("load_friction_external", cfg.load_friction_external),
            ("load_friction_motor_stribeck", cfg.load_friction_motor_stribeck),
            ("load_friction_external_stribeck", cfg.load_friction_external_stribeck),
            ("load_friction_motor_quad", cfg.load_friction_motor_quad),
            ("load_friction_external_quad", cfg.load_friction_external_quad),
            ("dtheta_stribeck", cfg.dtheta_stribeck),
            ("stribeck_alpha", cfg.stribeck_alpha),
            ("friction_viscous", cfg.friction_viscous),
        ):
            _real(value, label=f"{label} {name}", minimum=0.0)

    @property
    def action_dim(self) -> int:
        return len(self._joint_ids)

    @property
    def raw_action(self) -> np.ndarray:
        return self._raw_action

    @property
    def processed_action(self) -> np.ndarray:
        return self._processed_action

    @property
    def motor_torque(self) -> np.ndarray:
        return self._motor_torque

    @property
    def applied_torque(self) -> np.ndarray:
        """Last joint torque written to ctrl after the friction/static clip.

        Closest observable to the upstream ``actuator_force`` used by
        ``joint_torque_rate_l2``: the torque actually applied for the current
        control step, zeroed on reset.
        """
        return self._prev_applied_torque

    @property
    def friction_scale(self) -> np.ndarray:
        return self._friction_scale

    def process_actions(self, actions: np.ndarray) -> None:
        if not isinstance(actions, np.ndarray):
            raise TypeError(f"BamVoltageAction expected np.ndarray, got {type(actions).__name__}")
        if actions.shape != self._raw_action.shape:
            raise ValueError(
                f"BamVoltageAction expected action shape {self._raw_action.shape}, "
                f"got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("BamVoltageAction received NaN or Inf actions")
        self._raw_action[:] = actions
        # q_des = action * scale + default_joint_pos, mirroring the PD recipe's
        # JointPositionAction (scale=1.0, use_default_offset=True).
        np.multiply(self._raw_action, self._scale, out=self._processed_action)
        self._processed_action += self._entity.data.default_joint_pos[:, self._joint_ids]

    def _delay_append(self, q_target: np.ndarray) -> None:
        capacity = self._delay_buffer.shape[0]
        self._delay_pointer = (self._delay_pointer + 1) % capacity
        self._delay_buffer[self._delay_pointer] = q_target
        # First append after a reset backfills the whole history with the
        # first frame (upstream CircularBuffer semantics).
        first = self._delay_pushes == 0
        if np.any(first):
            self._delay_buffer[:, first] = q_target[first]
        self._delay_pushes += 1

    def _delay_read(self) -> np.ndarray:
        capacity = self._delay_buffer.shape[0]
        lag = self._env.rng.integers(
            self._delay_min_lag, self._delay_max_lag + 1, size=self.num_envs
        )
        valid = np.minimum(lag, np.maximum(self._delay_pushes, 1) - 1)
        index = (self._delay_pointer - valid) % capacity
        return self._delay_buffer[index, np.arange(self.num_envs)]

    def _friction_budget(
        self,
        motor_torque: np.ndarray,
        external_torque: np.ndarray,
        stribeck_coeff: np.ndarray,
    ) -> np.ndarray:
        """BAM m6 velocity-independent friction budget, shape ``(N, J)``."""
        cfg = self.cfg
        abs_ext = np.abs(external_torque)
        abs_mot = np.abs(motor_torque)
        gearbox = np.abs(
            external_torque * cfg.load_friction_external - motor_torque * cfg.load_friction_motor
        )
        gearbox_stribeck = np.abs(
            external_torque * cfg.load_friction_external_stribeck
            - motor_torque * cfg.load_friction_motor_stribeck
        )
        # Directional quadratic term: equal loads are classed as backdrive.
        drive_mask = (abs_mot > abs_ext).astype(get_global_dtype())
        quad = (
            drive_mask * cfg.load_friction_external_quad * abs_ext**2
            + (1.0 - drive_mask) * cfg.load_friction_motor_quad * abs_mot**2
        )
        frictionloss = (
            cfg.friction_base
            + stribeck_coeff * cfg.friction_stribeck
            + gearbox
            + stribeck_coeff * gearbox_stribeck
            + stribeck_coeff * quad
        )
        return frictionloss * self._friction_scale

    def apply_actions(self) -> None:
        joint_pos = self._entity.data.joint_pos[:, self._joint_ids]
        joint_vel = self._entity.data.joint_vel[:, self._joint_ids]
        if not np.isfinite(joint_pos).all() or not np.isfinite(joint_vel).all():
            raise ValueError(
                f"BamVoltageAction on entity '{self.cfg.entity_name}' received non-finite "
                "joint state from the backend"
            )
        dtype = get_global_dtype()
        q = np.asarray(joint_pos, dtype=dtype)
        dq = np.asarray(joint_vel, dtype=dtype)

        # Encoder bias is subtracted before the delay buffer, matching the PD
        # recipe's JointPositionAction ordering (target = processed - bias).
        q_target = self._processed_action - self._entity.data.encoder_bias[:, self._joint_ids]
        self._delay_append(q_target)
        q_target_delayed = self._delay_read()

        # Per-env battery voltage with load-dependent sag (previous substep's
        # motor torque), floored at vin_min.
        load = np.abs(self._prev_motor_torque).sum(axis=1, keepdims=True)
        vin_eff = np.maximum(self._vin - self._vin_drop_gain * load, self._vin_min)

        # Firmware P law with the current-limit duty window, then the physical
        # PWM limit (upstream VoltageControlledActuator.compute_control).
        duty = (q_target_delayed - q) * (self._kp_fw * self._error_gain)
        center = self._kt * dq / vin_eff
        span = self._resistance * self._max_current / vin_eff
        duty = np.clip(duty, center - span, center + span)
        duty = np.clip(duty, -self._max_pwm, self._max_pwm)
        volts = vin_eff * duty

        # DC motor torque with back-EMF (upstream compute_torque).
        motor_torque = (self._kt * volts - self._kt**2 * dq) / self._resistance

        # External gearbox torque via finite difference (see module docstring
        # for the approximation boundary): I_eff is the nominal dof_armature.
        accel_fd = (dq - self._prev_joint_vel) / self._env.physics_dt
        external_torque = self._armature * accel_fd - self._prev_applied_torque

        stribeck_coeff = np.exp(
            -((np.abs(dq) / self.cfg.dtheta_stribeck) ** self.cfg.stribeck_alpha)
        )
        frictionloss = self._friction_budget(
            self._prev_motor_torque, external_torque, stribeck_coeff
        )

        # Torque-domain replica of the solver's static-friction clip: a
        # quasi-static joint whose motor torque cannot break the friction
        # budget holds position (zero output); otherwise dry friction opposes
        # motion. The viscous term is subtracted unconditionally.
        static = (np.abs(dq) < _DQ_EPS) & (np.abs(motor_torque) < frictionloss)
        applied = np.where(static, 0.0, motor_torque - frictionloss * np.sign(dq))
        applied = applied - self.cfg.friction_viscous * dq

        self._entity.data.write_ctrl(applied, actuator_ids=self._actuator_ids)

        self._motor_torque[:] = motor_torque
        self._prev_motor_torque[:] = motor_torque
        self._prev_applied_torque[:] = applied
        self._prev_joint_vel[:] = dq

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_action[env_ids] = 0.0
        self._processed_action[env_ids] = 0.0
        self._motor_torque[env_ids] = 0.0
        self._prev_motor_torque[env_ids] = 0.0
        self._prev_applied_torque[env_ids] = 0.0
        self._prev_joint_vel[env_ids] = 0.0
        # Clear the delay history; the next append backfills all slots.
        rows = np.arange(self.num_envs)[env_ids] if isinstance(env_ids, slice) else env_ids
        self._delay_buffer[:, rows] = 0.0
        self._delay_pushes[rows] = 0
        # Per-episode friction DR: assignment (not accumulation), so the scale
        # is exactly one fresh sample per reset. vin / vin_drop_gain are
        # startup semantics and intentionally untouched.
        ids = np.asarray(rows, dtype=np.intp).ravel()
        if ids.size:
            self._friction_scale[ids] = np.asarray(
                self._env.rng.uniform(*self._friction_scale_range, size=(ids.size, 1)),
                dtype=get_global_dtype(),
            )


__all__ = ["BamVoltageAction", "BamVoltageActionCfg"]
