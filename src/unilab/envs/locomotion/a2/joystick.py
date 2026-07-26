"""A2 joystick task (leg-only Unitree A2).

The A2 leg-only MJCF (robots/a2/scene_flat.xml) mirrors the Go2 joystick
sensor/geom/leg-ordering contract and uses <position> actuators, so this
task reuses Go2WalkTask unchanged. Only the A2 identity differs: scene path,
standing pose, and per-joint PD gains.

Asset values are aligned to the official unitree_rl_mjlab A2 (a2_constants.py):
the home keyframe matches its INIT_STATE (height 0.4, thigh 0.9, calf -1.8,
hips +-0.1), and the PD gains match its BuiltinPositionActuatorCfg — hip/thigh
kp=100/kd=4, calf kp=150/kd=6 — applied per joint at init via
position_actuator_gains and used as the per-joint baseline for kp/kd domain
randomization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import sample_commands_with_standing
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.go2.base import Asset, ControlConfig
from unilab.envs.locomotion.go2.joystick import (
    Go2DomainRandConfig,
    Go2JoystickCfg,
    Go2JoystickDomainRandomizationProvider,
    Go2WalkTask,
    RewardConfig,
)

# Actuator/keyframe leg order: FL, FR, RL, RR x (hip, thigh, calf).
_NUM_LEGS = 4


def _per_leg_gains(hip: float, thigh: float, calf: float) -> np.ndarray:
    """Tile (hip, thigh, calf) gains across the four legs in actuator order."""
    return np.asarray([hip, thigh, calf] * _NUM_LEGS, dtype=np.float64)


@dataclass
class A2InitState:
    pos = [0.0, 0.0, 0.4]


@dataclass
class A2Asset(Asset):
    # The A2 base body is named "base_link" in its MJCF, whereas Go2 uses "base".
    base_name: str = "base_link"  # type: ignore[assignment]


@dataclass
class A2JoystickControlConfig(ControlConfig):
    # Per-joint PD gains aligned to unitree_rl_mjlab: hip/thigh share Kp/Kd, the
    # calf is stiffer. position_gains() expands these into 12-actuator arrays
    # forwarded to the backend (overriding the static per-class kp in a2.xml).
    Kp: float = 100.0
    Kd: float = 4.0
    calf_Kp: float = 150.0  # noqa: N815 - matches the Kp/Kd Hydra config convention.
    calf_Kd: float = 6.0  # noqa: N815 - matches the Kp/Kd Hydra config convention.

    def position_gains(self) -> dict[str, float | np.ndarray]:
        return {
            "kp": _per_leg_gains(self.Kp, self.Kp, self.calf_Kp),
            "kd": _per_leg_gains(self.Kd, self.Kd, self.calf_Kd),
        }


def _a2_scene() -> SceneCfg:
    return SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots" / "a2" / "scene_flat.xml"))


@dataclass
class A2JoystickDomainRandConfig(Go2DomainRandConfig):
    # A2's base COM is uncertain in all 3 axes on the real robot. dr_utils reads
    # com_offset_y/z via getattr, so they must be declared here to be settable
    # from the owner YAML (Hydra struct mode rejects undeclared keys). Inherits
    # com_offset_x + every other DR switch/range from Go2DomainRandConfig and
    # the base DomainRandConfig; on/off + ranges are set in the owner YAML.
    com_offset_y: list[float] = field(default_factory=lambda: [-0.08, 0.08])
    com_offset_z: list[float] = field(default_factory=lambda: [-0.08, 0.08])


class A2JoystickDomainRandomizationProvider(Go2JoystickDomainRandomizationProvider):
    """A2 reuses the Go2 joystick DR logic but supplies per-joint base gains so
    randomize_kp/kd scales each actuator off its true baseline (calf off 150,
    not the shared scalar 100). Without this, kp/kd DR would fall back to a
    uniform ``control_config.Kp``/``Kd`` and silently weaken the calf."""

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        """Standing-aware reset commands for A2 (Go2 base stays pure-uniform).

        Mirrors rough.py's provider override: draw from ``commands.vel_limit`` and
        zero a ``rel_standing_envs`` fraction so the policy trains on genuine
        zero-command samples. Uses the shared ``sample_commands_with_standing`` so
        the reset and mid-episode resampling paths stay a single source of truth."""
        low = np.asarray(env.cfg.commands.vel_limit[0], dtype=np.float64)
        high = np.asarray(env.cfg.commands.vel_limit[1], dtype=np.float64)
        return sample_commands_with_standing(
            low, high, num_reset, rel_standing_envs=env.cfg.commands.rel_standing_envs
        )

    def _get_base_actuator_gains(self, env: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        gains = env.cfg.control_config.position_gains()
        num_actuators = env._num_action
        base_kp = np.broadcast_to(
            np.asarray(gains["kp"], dtype=np.float64), (num_actuators,)
        ).copy()
        base_kd = np.broadcast_to(
            np.asarray(gains["kd"], dtype=np.float64), (num_actuators,)
        ).copy()
        return base_kp, base_kd

    def _get_reset_randomization_baselines(
        self, env: Any
    ) -> tuple[np.ndarray | None, np.ndarray | None, int | None, np.ndarray | None]:
        """Snapshot the pristine model tables that reset-time DR multiplies against.

        Caches once per env (the base model is not mutated by per-env reset
        randomization) via the public backend getters — no infra change, no
        feature leak. Enables randomize_ground_friction (floor geom is the
        priority geom, see scene_flat.xml) and randomize_dof_armature.
        body_mass stays uncached (that DR switch is intentionally off)."""
        cached = getattr(self, "_reset_baselines", None)
        if cached is None:
            backend = env._backend
            base_geom_friction = backend.get_geom_friction()
            ground_geom_id = backend.get_geom_id(env.cfg.asset.ground)
            base_dof_armature = backend.get_dof_armature()
            cached = (None, base_geom_friction, ground_geom_id, base_dof_armature)
            self._reset_baselines = cached
        return cached


@dataclass
class A2RewardConfig(RewardConfig):
    # Command norm below which the phase-driven gait rewards (swing_feet_z /
    # contact) switch to standing behaviour and the gait clock freezes, so the
    # A2 stands still at zero command instead of marching in place. Set via the
    # A2 owner YAML; default 0.0 leaves gating off.
    command_threshold: float = 0.0


@registry.envcfg("A2JoystickFlat")
@dataclass
class A2JoystickCfg(Go2JoystickCfg):
    scene: SceneCfg = field(default_factory=_a2_scene)
    init_state: A2InitState = field(default_factory=A2InitState)  # type: ignore[assignment]
    asset: A2Asset = field(default_factory=A2Asset)  # type: ignore[assignment]
    control_config: A2JoystickControlConfig = field(  # type: ignore[assignment]
        default_factory=A2JoystickControlConfig
    )
    domain_rand: A2JoystickDomainRandConfig = field(  # type: ignore[assignment]
        default_factory=A2JoystickDomainRandConfig
    )
    reward_config: A2RewardConfig | None = None  # type: ignore[assignment]


@registry.env("A2JoystickFlat", sim_backend="mujoco")
class A2JoystickFlatEnv(Go2WalkTask):
    """Leg-only A2 joystick task. Reuses Go2WalkTask locomotion; adds
    zero-command standstill (phase freeze + gated gait rewards + standing
    resample) gated by A2RewardConfig.command_threshold."""

    _cfg: A2JoystickCfg
    _reward_cfg: A2RewardConfig

    def _make_dr_provider(self) -> LocomotionDRProvider:
        return A2JoystickDomainRandomizationProvider()

    def _advance_phase(self, phase: np.ndarray) -> np.ndarray:
        """Advance the gait phase, freezing envs whose command is at/below
        ``command_threshold`` so a standing A2 holds phase instead of swaying."""
        cmd_norm = np.linalg.norm(self._latest_commands, axis=1)
        moving = cmd_norm > self._reward_cfg.command_threshold
        increment = self._cfg.ctrl_dt * self.gait_frequency * moving
        return np.fmod(phase + increment, 1.0)

    def _init_reward_functions(self) -> None:
        super()._init_reward_functions()
        self._reward_fns.update(
            {
                "stand_still": rewards.stand_still,
                "hip_deviation": self._reward_hip_deviation,
                "stand_feet_air": self._reward_stand_feet_air,
                "swing_feet_z": self._gated_swing_feet_z,
                "contact": self._gated_contact,
            }
        )

    def _gated_swing_feet_z(self, ctx: RewardContext) -> np.ndarray:
        """Base swing reward, zeroed while standing (command at/below threshold)."""
        reward = super()._reward_swing_feet_z(ctx)
        cmd_norm = np.linalg.norm(ctx.info["commands"], axis=1)
        active = cmd_norm > self._reward_cfg.command_threshold
        return reward * active

    def _gated_contact(self, ctx: RewardContext) -> np.ndarray:
        """Contact reward; while standing every foot is expected planted so a
        planted robot earns full contact reward (standing branch is interleaved
        per-foot, so this re-implements rather than wraps the base loop)."""
        contact = self.feet_force[:, :, 2] > 0.1
        cmd_norm = np.linalg.norm(ctx.info["commands"], axis=1)
        standing = cmd_norm <= self._reward_cfg.command_threshold
        res = np.zeros(self._num_envs, dtype=np.float32)
        for i in range(len(self._cfg.sensor.feet_force)):
            is_contact = (self.feet_phase[:, i] < 0.6) | (self.gait_frequency < 1.0e-8) | standing
            res += (contact[:, i] == is_contact).astype(np.float32)
        return res / len(self._cfg.sensor.feet_force)

    def _reward_hip_deviation(self, ctx: RewardContext) -> np.ndarray:
        """L1 deviation of the hip DOFs ([0, 3, 6, 9]) from the default pose."""
        hip_indices = [0, 3, 6, 9]
        diff = ctx.dof_pos[:, hip_indices] - self.default_angles[hip_indices]
        return np.asarray(np.sum(np.abs(diff), axis=1), dtype=get_global_dtype())

    def _reward_stand_feet_air(self, ctx: RewardContext) -> np.ndarray:
        """Penalize feet leaving the ground while standing (||command|| <= threshold)."""
        cmd_norm = np.linalg.norm(ctx.info["commands"], axis=1)
        standing = cmd_norm <= self._reward_cfg.command_threshold
        in_air = np.sum(self.feet_force[:, :, 2] <= 0.1, axis=1)
        return np.asarray(in_air * standing, dtype=get_global_dtype())

    def _update_commands(self, info: dict) -> None:
        """Standing-aware mid-episode resample (gated by ``resampling_time``),
        then stamp ``self._latest_commands`` for the phase-freeze read."""
        resampling_time = float(self._cfg.commands.resampling_time)
        if resampling_time > 0.0:
            commands_arr = np.asarray(info["commands"], dtype=get_global_dtype())
            interval_steps = max(int(round(resampling_time / self._cfg.ctrl_dt)), 1)
            steps = np.asarray(info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32)))
            resample_mask = (steps > 0) & ((steps % interval_steps) == 0)
            if np.any(resample_mask):
                num_resample = int(np.count_nonzero(resample_mask))
                low = np.asarray(self._cfg.commands.vel_limit[0], dtype=get_global_dtype())
                high = np.asarray(self._cfg.commands.vel_limit[1], dtype=get_global_dtype())
                sampled = sample_commands_with_standing(
                    low, high, num_resample, rel_standing_envs=self._cfg.commands.rel_standing_envs
                )
                commands_arr[resample_mask] = sampled
                if self._cfg.commands.heading_command:
                    commands_arr[resample_mask, 2] = 0.0
            info["commands"] = commands_arr
        self._latest_commands = np.asarray(info["commands"], dtype=get_global_dtype())
