"""Shared public schema for compiled G1 walking executors.

This module owns the cold-path constants, selectors, reset descriptors, and
validated kernel configuration shared by the reference and fused-host
executors.  Executor implementations may evolve independently
without importing private symbols from one another.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from unilab.base.backend import (
    BufferContract,
    BufferLayout,
    BufferLifetime,
    BufferMutability,
    BufferOwner,
    BufferPlacement,
    MutationBaseline,
    MutationCommitPhase,
    MutationFieldKind,
    MutationOperation,
    MutationPersistence,
    MutationRecomputeLevel,
    MutationTargetKind,
    MutationTrigger,
    PhysicalUnit,
    ReferenceFrame,
    RowSelection,
    StateFieldKind,
)
from unilab.base.backend.base import SimBackend
from unilab.dtype_config import get_global_dtype
from unilab.manager import (
    EntityKind,
    EntitySelector,
    ManagerContractError,
    MutationTemplate,
    QuaternionOrder,
    StateRequirement,
    TensorSpec,
)

from .joystick import (
    G1WalkEnvCfg,
    G1WalkRewardConfig,
    build_upper_body_pose_weights,
)
from .managed_reward_terms import unsupported_g1_reward_terms

G1_ROOT_NAME = "pelvis"
G1_ACTOR_OBSERVATION_WIDTH = 98
G1_CRITIC_OBSERVATION_WIDTH = 101
G1_RESET_TERM = "g1_reset_state"
G1_ROOT_RESET_SPECS = (
    ("root_position", "state.root.position", MutationFieldKind.POSITION, (3,)),
    (
        "root_orientation",
        "state.root.orientation",
        MutationFieldKind.ORIENTATION,
        (4,),
    ),
    (
        "root_linear_velocity",
        "state.root.linear_velocity",
        MutationFieldKind.LINEAR_VELOCITY,
        (3,),
    ),
    (
        "root_angular_velocity",
        "state.root.angular_velocity",
        MutationFieldKind.ANGULAR_VELOCITY,
        (3,),
    ),
)
G1_STATE_KEYS = (
    "g1.dof.angular_velocity",
    "g1.dof.position",
    "g1.root.angular_velocity",
    "g1.root.linear_velocity",
    "g1.root.orientation",
    "g1.root.position",
    "g1.sensor.left_foot_pos",
    "g1.sensor.pelvis_local_linvel",
    "g1.sensor.right_foot_pos",
    "g1.sensor.torso_gyro",
    "g1.sensor.torso_upvector",
)


@dataclass(frozen=True)
class G1KernelConfig:
    action_scale: np.ndarray
    default_angles: np.ndarray
    initial_qpos: np.ndarray
    initial_qvel: np.ndarray
    reward_terms: tuple[tuple[str, float], ...]
    ctrl_dt: float
    tracking_sigma: float
    gait_phase_delta: float
    gait_phase_init_mode: str
    reset_base_qvel_limit: float
    command_low: np.ndarray
    command_high: np.ndarray
    standing_probability: float
    base_height_target: float
    min_base_height: float
    max_tilt_rad: float
    feet_phase_swing_height: float
    feet_phase_tracking_sigma: float
    min_forward_speed_for_gait_reward: float
    close_feet_threshold: float
    pose_weights: np.ndarray
    upper_body_pose_weights: np.ndarray
    walk_observation_profile: bool
    observation_noise_level: float
    observation_noise_scale_joint_angle: float
    observation_noise_scale_joint_vel: float
    observation_noise_scale_gyro: float
    observation_noise_scale_gravity: float
    reset_seed: int
    observation_noise_seed: int | None


@dataclass(frozen=True)
class G1StateViews:
    """Named views in the canonical compiled G1 state-field order."""

    dof_angular_velocity: np.ndarray
    dof_position: np.ndarray
    root_angular_velocity: np.ndarray
    root_linear_velocity: np.ndarray
    root_orientation: np.ndarray
    root_position: np.ndarray
    left_foot_position: np.ndarray
    pelvis_local_linear_velocity: np.ndarray
    right_foot_position: np.ndarray
    torso_gyro: np.ndarray
    torso_upvector: np.ndarray


@dataclass(frozen=True)
class G1ResetSample:
    rows: RowSelection


def manager_buffer_contract(
    *, row_shape: tuple[int, ...], lifetime: BufferLifetime
) -> BufferContract:
    return BufferContract(
        row_shape=row_shape,
        dtype=np.dtype(get_global_dtype()).name,
        layout=BufferLayout.C_CONTIGUOUS,
        placement=BufferPlacement.host(),
        owner=BufferOwner.MANAGER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=lifetime,
        dlpack_exportable=False,
    )


def state_requirement(
    *,
    key: str,
    selector: EntitySelector,
    field_kind: StateFieldKind,
    shape: tuple[int, ...],
    frame: ReferenceFrame,
    unit: PhysicalUnit,
    quaternion_order: QuaternionOrder = QuaternionOrder.NONE,
    entity_axis: int | None = None,
) -> StateRequirement:
    return StateRequirement(
        semantic_key=key,
        selector=selector,
        field_kind=field_kind,
        tensor=TensorSpec(
            shape,
            np.dtype(get_global_dtype()).name,
            frame=frame,
            unit=unit,
            quaternion_order=quaternion_order,
        ),
        entity_axis=entity_axis,
    )


def validate_g1_managed_profile(
    cfg: G1WalkEnvCfg,
    *,
    profile_name: str,
    error_type: type[ManagerContractError],
    allow_pd_randomization: bool = False,
    allow_dof_armature_randomization: bool = False,
    allow_body_gravity_compensation_randomization: bool = False,
) -> G1WalkRewardConfig:
    """Reject legacy features absent from the named compiled executor."""

    if not isinstance(allow_pd_randomization, bool):
        raise error_type("allow_pd_randomization must be a bool")
    if not isinstance(allow_dof_armature_randomization, bool):
        raise error_type("allow_dof_armature_randomization must be a bool")
    if not isinstance(allow_body_gravity_compensation_randomization, bool):
        raise error_type("allow_body_gravity_compensation_randomization must be a bool")

    reward = cfg.reward_config
    if not isinstance(reward, G1WalkRewardConfig):
        raise error_type(f"G1 {profile_name} requires a G1WalkRewardConfig")
    unsupported_rewards = unsupported_g1_reward_terms(reward.scales)
    if unsupported_rewards:
        raise error_type(
            f"G1 {profile_name} does not implement reward terms: " + ", ".join(unsupported_rewards)
        )
    if any(not np.isfinite(float(scale)) for scale in reward.scales.values()):
        raise error_type(f"G1 {profile_name} reward scales must be finite")
    if cfg.curriculum.enabled:
        raise error_type(f"G1 {profile_name} does not implement the legacy penalty curriculum")
    if cfg.numba_acceleration:
        raise error_type(f"G1 {profile_name} does not select the legacy Numba executor")
    if cfg.commands.heading_command:
        raise error_type(f"G1 {profile_name} does not implement heading-command task state")
    if cfg.commands.resampling_time != 0.0:
        raise error_type(f"G1 {profile_name} only supports reset-sampled velocity commands")
    if cfg.gait_phase_init_mode not in {"offset_phase", "independent"}:
        raise error_type(
            f"G1 {profile_name} requires gait_phase_init_mode='offset_phase' or 'independent'"
        )

    dr = cfg.domain_rand
    enabled_dr = tuple(
        name
        for name, enabled in (
            ("randomize_base_mass", dr.randomize_base_mass),
            ("randomize_body_mass", dr.randomize_body_mass),
            ("random_com", dr.random_com),
            ("randomize_gravity", dr.randomize_gravity),
            ("randomize_ground_friction", dr.randomize_ground_friction),
            (
                "randomize_dof_armature",
                dr.randomize_dof_armature and not allow_dof_armature_randomization,
            ),
            (
                "randomize_body_gravity_compensation",
                dr.randomize_body_gravity_compensation
                and not allow_body_gravity_compensation_randomization,
            ),
            ("push_robots", dr.push_robots),
            ("randomize_kp", dr.randomize_kp and not allow_pd_randomization),
            ("randomize_kd", dr.randomize_kd and not allow_pd_randomization),
        )
        if enabled
    )
    if enabled_dr:
        raise error_type(
            f"G1 {profile_name} has no typed DR/Event implementation for: " + ", ".join(enabled_dr)
        )
    return reward


def g1_action_scale(
    cfg: G1WalkEnvCfg,
    action_dim: int,
    *,
    error_type: type[ManagerContractError],
) -> np.ndarray:
    raw = np.asarray(cfg.control_config.action_scale, dtype=get_global_dtype())
    if raw.ndim == 0:
        values = np.full((action_dim,), raw.item(), dtype=get_global_dtype())
    elif raw.shape == (action_dim,):
        values = np.asarray(raw, dtype=get_global_dtype())
    else:
        raise error_type(
            f"G1 action_scale must be scalar or shape ({action_dim},), got {raw.shape}"
        )
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise error_type("G1 action_scale must contain finite positive values")
    return values


def walk_observation_profile(reward: G1WalkRewardConfig, cfg: G1WalkEnvCfg) -> bool:
    scales = reward.scales
    if any(
        key in scales
        for key in (
            "penalty_orientation",
            "penalty_ang_vel_xy",
            "penalty_action_rate",
            "alive",
        )
    ):
        return True
    if any(key in scales for key in ("orientation", "ang_vel_xy", "action_rate")):
        return False
    return cfg.curriculum.enabled


def reset_suffix_for_dof(*, kind: str, index: int) -> str:
    return f"dof_{kind}_{index:02d}"


def reset_term_key(*, suffix: str) -> str:
    return f"{G1_RESET_TERM}.{suffix}"


def g1_selectors(
    actuator_names: tuple[str, ...],
) -> tuple[
    EntitySelector,
    EntitySelector,
    tuple[EntitySelector, ...],
    tuple[EntitySelector, ...],
]:
    root = EntitySelector(
        key="g1.root",
        entity="g1",
        kind=EntityKind.ROOT,
        expressions=(G1_ROOT_NAME,),
    )
    dofs = EntitySelector(
        key="g1.actuated_dofs",
        entity="g1",
        kind=EntityKind.DOF,
        expressions=actuator_names,
    )
    reset_position = tuple(
        EntitySelector(
            key=f"g1.reset.dof_position.{index:02d}",
            entity="g1",
            kind=EntityKind.DOF,
            expressions=(name,),
        )
        for index, name in enumerate(actuator_names)
    )
    reset_velocity = tuple(
        EntitySelector(
            key=f"g1.reset.dof_velocity.{index:02d}",
            entity="g1",
            kind=EntityKind.DOF,
            expressions=(name,),
        )
        for index, name in enumerate(actuator_names)
    )
    return root, dofs, reset_position, reset_velocity


def g1_state_requirements(
    *, root: EntitySelector, dofs: EntitySelector, action_dim: int
) -> tuple[StateRequirement, ...]:
    sensors = (
        (
            "pelvis_local_linvel",
            ReferenceFrame.SENSOR,
            PhysicalUnit.METER_PER_SECOND,
        ),
        ("torso_gyro", ReferenceFrame.SENSOR, PhysicalUnit.RADIAN_PER_SECOND),
        ("torso_upvector", ReferenceFrame.WORLD, PhysicalUnit.UNITLESS),
        ("left_foot_pos", ReferenceFrame.WORLD, PhysicalUnit.METER),
        ("right_foot_pos", ReferenceFrame.WORLD, PhysicalUnit.METER),
    )
    requirements = [
        state_requirement(
            key="g1.root.position",
            selector=root,
            field_kind=StateFieldKind.POSITION,
            shape=(3,),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.METER,
        ),
        state_requirement(
            key="g1.root.orientation",
            selector=root,
            field_kind=StateFieldKind.ORIENTATION,
            shape=(4,),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.QUATERNION,
            quaternion_order=QuaternionOrder.WXYZ,
        ),
        state_requirement(
            key="g1.root.linear_velocity",
            selector=root,
            field_kind=StateFieldKind.LINEAR_VELOCITY,
            shape=(3,),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.METER_PER_SECOND,
        ),
        state_requirement(
            key="g1.root.angular_velocity",
            selector=root,
            field_kind=StateFieldKind.ANGULAR_VELOCITY,
            shape=(3,),
            frame=ReferenceFrame.WORLD,
            unit=PhysicalUnit.RADIAN_PER_SECOND,
        ),
        state_requirement(
            key="g1.dof.position",
            selector=dofs,
            field_kind=StateFieldKind.POSITION,
            shape=(action_dim,),
            frame=ReferenceFrame.JOINT,
            unit=PhysicalUnit.RADIAN,
            entity_axis=0,
        ),
        state_requirement(
            key="g1.dof.angular_velocity",
            selector=dofs,
            field_kind=StateFieldKind.ANGULAR_VELOCITY,
            shape=(action_dim,),
            frame=ReferenceFrame.JOINT,
            unit=PhysicalUnit.RADIAN_PER_SECOND,
            entity_axis=0,
        ),
    ]
    for name, frame, unit in sensors:
        requirements.append(
            state_requirement(
                key=f"g1.sensor.{name}",
                selector=EntitySelector(
                    key=f"g1.sensor.{name}",
                    entity="g1",
                    kind=EntityKind.SENSOR,
                    expressions=(name,),
                ),
                field_kind=StateFieldKind.VALUE,
                shape=(3,),
                frame=frame,
                unit=unit,
            )
        )
    return tuple(requirements)


def g1_reset_templates(
    *,
    root: EntitySelector,
    reset_position: tuple[EntitySelector, ...],
    reset_velocity: tuple[EntitySelector, ...],
) -> tuple[MutationTemplate, ...]:
    templates: list[MutationTemplate] = []
    for suffix, target_key, field_kind, row_shape in G1_ROOT_RESET_SPECS:
        templates.append(
            MutationTemplate(
                key_suffix=suffix,
                target_key=target_key,
                target_kind=MutationTargetKind.SIMULATION_STATE,
                selector=root,
                field_kind=field_kind,
                trigger=MutationTrigger.RESET,
                commit_phase=MutationCommitPhase.RESET,
                operation=MutationOperation.SET,
                baseline=MutationBaseline.DEFAULT,
                persistence=MutationPersistence.EPISODE,
                recompute=MutationRecomputeLevel.KINEMATICS,
                value_template=manager_buffer_contract(
                    row_shape=row_shape, lifetime=BufferLifetime.UNTIL_COMMIT
                ),
            )
        )
    for index, selector in enumerate(reset_position):
        templates.append(
            MutationTemplate(
                key_suffix=reset_suffix_for_dof(kind="position", index=index),
                target_key="state.dof.position",
                target_kind=MutationTargetKind.SIMULATION_STATE,
                selector=selector,
                field_kind=MutationFieldKind.POSITION,
                trigger=MutationTrigger.RESET,
                commit_phase=MutationCommitPhase.RESET,
                operation=MutationOperation.SET,
                baseline=MutationBaseline.DEFAULT,
                persistence=MutationPersistence.EPISODE,
                recompute=MutationRecomputeLevel.KINEMATICS,
                value_template=manager_buffer_contract(
                    row_shape=(1,), lifetime=BufferLifetime.UNTIL_COMMIT
                ),
            )
        )
    for index, selector in enumerate(reset_velocity):
        templates.append(
            MutationTemplate(
                key_suffix=reset_suffix_for_dof(kind="velocity", index=index),
                target_key="state.dof.angular_velocity",
                target_kind=MutationTargetKind.SIMULATION_STATE,
                selector=selector,
                field_kind=MutationFieldKind.ANGULAR_VELOCITY,
                trigger=MutationTrigger.RESET,
                commit_phase=MutationCommitPhase.RESET,
                operation=MutationOperation.SET,
                baseline=MutationBaseline.DEFAULT,
                persistence=MutationPersistence.EPISODE,
                recompute=MutationRecomputeLevel.KINEMATICS,
                value_template=manager_buffer_contract(
                    row_shape=(1,), lifetime=BufferLifetime.UNTIL_COMMIT
                ),
            )
        )
    return tuple(templates)


def build_g1_kernel_config(
    *,
    backend: SimBackend,
    cfg: G1WalkEnvCfg,
    reset_seed: int,
    observation_noise_seed: int | None,
    profile_name: str,
    error_type: type[ManagerContractError],
    allow_pd_randomization: bool = False,
    allow_dof_armature_randomization: bool = False,
    allow_body_gravity_compensation_randomization: bool = False,
) -> G1KernelConfig:
    reward = validate_g1_managed_profile(
        cfg,
        profile_name=profile_name,
        error_type=error_type,
        allow_pd_randomization=allow_pd_randomization,
        allow_dof_armature_randomization=allow_dof_armature_randomization,
        allow_body_gravity_compensation_randomization=(
            allow_body_gravity_compensation_randomization
        ),
    )
    if not np.isfinite(float(cfg.sim_dt)) or float(cfg.sim_dt) <= 0.0:
        raise error_type("G1 sim_dt must be finite and positive")
    if not np.isfinite(float(cfg.ctrl_dt)) or float(cfg.ctrl_dt) <= 0.0:
        raise error_type("G1 ctrl_dt must be finite and positive")
    if cfg.sim_substeps <= 0:
        raise error_type(f"G1 {profile_name} requires positive sim_substeps")
    if isinstance(reset_seed, bool) or not isinstance(reset_seed, int) or reset_seed < 0:
        raise error_type("G1 reset_seed must be a non-negative integer")
    if observation_noise_seed is None:
        observation_noise_seed = cfg.noise_config.seed
    if observation_noise_seed is not None and (
        isinstance(observation_noise_seed, bool)
        or not isinstance(observation_noise_seed, int)
        or observation_noise_seed < 0
    ):
        raise error_type("G1 observation_noise_seed must be non-negative or None")
    if cfg.noise_config.level > 0.0 and observation_noise_seed is None:
        raise error_type(
            f"G1 {profile_name} requires an explicit observation noise seed when noise is enabled"
        )
    actuator_names = backend.get_actuator_names()
    action_dim = len(actuator_names)
    default_qpos = np.asarray(backend.get_keyframe_qpos("stand"), dtype=get_global_dtype())
    initial_qvel = np.asarray(backend.get_init_qvel(), dtype=get_global_dtype())
    if default_qpos.shape != (7 + action_dim,) or initial_qvel.shape != (6 + action_dim,):
        raise error_type(
            f"G1 {profile_name} requires one floating root and one coordinate per actuator"
        )
    if not np.isfinite(default_qpos).all() or not np.isfinite(initial_qvel).all():
        raise error_type("G1 initial state must be finite")
    command_low = np.asarray(cfg.commands.vel_limit[0], dtype=get_global_dtype())
    command_high = np.asarray(cfg.commands.vel_limit[1], dtype=get_global_dtype())
    if command_low.shape != (3,) or command_high.shape != (3,):
        raise error_type("G1 command limits must have shape (2, 3)")
    if (
        not np.isfinite(command_low).all()
        or not np.isfinite(command_high).all()
        or np.any(command_high < command_low)
    ):
        raise error_type("G1 command maximum must be >= minimum")
    standing_probability = float(cfg.commands.rel_standing_envs)
    if not np.isfinite(standing_probability) or standing_probability < 0.0:
        raise error_type("G1 standing probability must be finite and non-negative")
    pose_weights = np.asarray(reward.pose_weights, dtype=get_global_dtype())
    if pose_weights.shape != (action_dim,) or not np.isfinite(pose_weights).all():
        raise error_type(
            f"G1 pose_weights must be finite with shape ({action_dim},), got {pose_weights.shape}"
        )
    finite_positive = {
        "tracking_sigma": reward.tracking_sigma,
        "feet_phase_tracking_sigma": reward.feet_phase_tracking_sigma,
    }
    for name, value in finite_positive.items():
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise error_type(f"G1 {name} must be finite and positive")
    finite_non_negative = {
        "reset_base_qvel_limit": cfg.reset_base_qvel_limit,
        "feet_phase_swing_height": reward.feet_phase_swing_height,
        "min_forward_speed_for_gait_reward": reward.min_forward_speed_for_gait_reward,
        "close_feet_threshold": reward.close_feet_threshold,
        "noise_config.level": cfg.noise_config.level,
    }
    for name, value in finite_non_negative.items():
        if not np.isfinite(float(value)) or float(value) < 0.0:
            raise error_type(f"G1 {name} must be finite and non-negative")
    finite_values = {
        "gait_frequency": reward.gait_frequency,
        "base_height_target": reward.base_height_target,
        "min_base_height": reward.min_base_height,
        "max_tilt_deg": reward.max_tilt_deg,
        "noise_config.scale_joint_angle": cfg.noise_config.scale_joint_angle,
        "noise_config.scale_joint_vel": cfg.noise_config.scale_joint_vel,
        "noise_config.scale_gyro": cfg.noise_config.scale_gyro,
        "noise_config.scale_gravity": cfg.noise_config.scale_gravity,
    }
    for name, value in finite_values.items():
        if not np.isfinite(float(value)):
            raise error_type(f"G1 {name} must be finite")
    return G1KernelConfig(
        action_scale=np.asarray(
            g1_action_scale(cfg, action_dim, error_type=error_type), dtype=get_global_dtype()
        ),
        default_angles=np.asarray(default_qpos[-action_dim:], dtype=get_global_dtype()),
        initial_qpos=np.asarray(default_qpos, dtype=get_global_dtype()),
        initial_qvel=np.asarray(initial_qvel, dtype=get_global_dtype()),
        reward_terms=tuple((name, float(scale)) for name, scale in reward.scales.items()),
        ctrl_dt=float(cfg.ctrl_dt),
        tracking_sigma=float(reward.tracking_sigma),
        gait_phase_delta=float(2.0 * math.pi * reward.gait_frequency * cfg.ctrl_dt),
        gait_phase_init_mode=cfg.gait_phase_init_mode,
        reset_base_qvel_limit=float(cfg.reset_base_qvel_limit),
        command_low=command_low,
        command_high=command_high,
        standing_probability=min(standing_probability, 1.0),
        base_height_target=float(reward.base_height_target),
        min_base_height=float(reward.min_base_height),
        max_tilt_rad=float(np.deg2rad(reward.max_tilt_deg)),
        feet_phase_swing_height=float(reward.feet_phase_swing_height),
        feet_phase_tracking_sigma=float(reward.feet_phase_tracking_sigma),
        min_forward_speed_for_gait_reward=float(reward.min_forward_speed_for_gait_reward),
        close_feet_threshold=float(reward.close_feet_threshold),
        pose_weights=pose_weights,
        upper_body_pose_weights=build_upper_body_pose_weights(pose_weights.tolist()),
        walk_observation_profile=walk_observation_profile(reward, cfg),
        observation_noise_level=float(cfg.noise_config.level),
        observation_noise_scale_joint_angle=float(cfg.noise_config.scale_joint_angle),
        observation_noise_scale_joint_vel=float(cfg.noise_config.scale_joint_vel),
        observation_noise_scale_gyro=float(cfg.noise_config.scale_gyro),
        observation_noise_scale_gravity=float(cfg.noise_config.scale_gravity),
        reset_seed=reset_seed,
        observation_noise_seed=observation_noise_seed,
    )


__all__ = [
    "G1_ACTOR_OBSERVATION_WIDTH",
    "G1_CRITIC_OBSERVATION_WIDTH",
    "G1_RESET_TERM",
    "G1_ROOT_RESET_SPECS",
    "G1_STATE_KEYS",
    "G1KernelConfig",
    "G1ResetSample",
    "G1StateViews",
    "build_g1_kernel_config",
    "g1_action_scale",
    "g1_reset_templates",
    "g1_selectors",
    "g1_state_requirements",
    "manager_buffer_contract",
    "reset_suffix_for_dof",
    "reset_term_key",
    "validate_g1_managed_profile",
]
