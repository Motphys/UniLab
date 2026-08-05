from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from unilab.term import (
    NamedTensorSpec,
    NumpyTermContext,
    ParameterKind,
    ParameterSpec,
    ResolvedTermPlan,
    TensorSpec,
    TermConfig,
    TermDefinition,
    TermKind,
    TermPlanError,
    TermRegistry,
    resolve_term_plan,
)

F32 = np.float32

REWARD_TERM_KEYS = {
    name: f"g1.walk.reward.{implementation}.v1"
    for name, implementation in {
        "tracking_lin_vel": "tracking_lin_vel",
        "tracking_ang_vel": "tracking_ang_vel",
        "forward_progress": "forward_progress",
        "under_speed": "under_speed",
        "lin_vel_z": "lin_vel_z",
        "orientation": "orientation",
        "penalty_orientation": "orientation",
        "ang_vel_xy": "ang_vel_xy",
        "penalty_ang_vel_xy": "ang_vel_xy",
        "action_rate": "action_rate",
        "penalty_action_rate": "action_rate",
        "base_height": "base_height",
        "pose": "pose",
        "upper_body_pose": "upper_body_pose",
        "penalty_feet_ori": "feet_ori",
        "feet_phase": "feet_phase",
        "feet_phase_contrast": "feet_phase_contrast",
        "feet_phase_contact": "feet_phase_contact",
        "feet_double_stance": "feet_double_stance",
        "alive": "alive",
    }.items()
}

DEFAULT_OBSERVATION_TERMS: dict[str, list[str]] = {
    "obs": [
        "g1.walk.observation.actor_gyro.v1",
        "g1.walk.observation.actor_gravity.v1",
        "g1.walk.observation.actor_dof_pos.v1",
        "g1.walk.observation.actor_dof_vel.v1",
        "g1.walk.observation.actor_actions.v1",
        "g1.walk.observation.actor_commands.v1",
        "g1.walk.observation.actor_gait_phase.v1",
    ],
    "critic": [
        "g1.walk.observation.critic_gyro.v1",
        "g1.walk.observation.critic_gravity.v1",
        "g1.walk.observation.critic_dof_pos.v1",
        "g1.walk.observation.critic_dof_vel.v1",
        "g1.walk.observation.critic_actions.v1",
        "g1.walk.observation.critic_commands.v1",
        "g1.walk.observation.critic_gait_phase.v1",
        "g1.walk.observation.critic_linvel.v1",
    ],
}
DEFAULT_TERMINATION_TERMS = ["g1.walk.termination.tilt_or_height.v1"]

_OBS_META = {
    key: (group, label, source, width)
    for group, entries in {
        "obs": (
            ("actor_gyro", "gyro", "noisy_gyro", 3),
            ("actor_gravity", "gravity", "noisy_gravity", 3),
            ("actor_dof_pos", "dof_pos", "noisy_dof_pos", -1),
            ("actor_dof_vel", "dof_vel", "noisy_dof_vel", -1),
            ("actor_actions", "actions", "current_actions", -1),
            ("actor_commands", "command", "commands", 3),
            ("actor_gait_phase", "gait_phase", "gait_phase", 2),
        ),
        "critic": (
            ("critic_gyro", "gyro", "gyro", 3),
            ("critic_gravity", "gravity", "gravity", 3),
            ("critic_dof_pos", "dof_pos", "dof_pos_diff", -1),
            ("critic_dof_vel", "dof_vel", "dof_vel", -1),
            ("critic_actions", "actions", "current_actions", -1),
            ("critic_commands", "command", "commands", 3),
            ("critic_gait_phase", "gait_phase", "gait_phase", 2),
            ("critic_linvel", "linvel", "linvel", 3),
        ),
    }.items()
    for name, label, source, width in entries
    for key in (f"g1.walk.observation.{name}.v1",)
}


def compute_feet_phase_height_targets(
    gait_phase: np.ndarray, swing_height: float
) -> tuple[np.ndarray, np.ndarray]:
    def height(phi: np.ndarray) -> np.ndarray:
        normalized = np.fmod(phi + np.pi, 2 * np.pi) - np.pi
        x = (normalized + np.pi) / (2 * np.pi)
        stance_t = 2 * x
        swing_t = 2 * x - 1
        stance = swing_height * (stance_t**3 + 3 * stance_t**2 * (1 - stance_t))
        swing = swing_height * (1 - (swing_t**3 + 3 * swing_t**2 * (1 - swing_t)))
        return np.where(x <= 0.5, stance, swing)

    return height(gait_phase[:, 0]), height(gait_phase[:, 1])


def compute_feet_phase_contact_targets(
    gait_phase: np.ndarray, swing_height: float
) -> tuple[np.ndarray, np.ndarray]:
    left, right = compute_feet_phase_height_targets(gait_phase, swing_height)
    threshold = swing_height * 0.5
    return left <= threshold, right <= threshold


def compute_forward_speed_gate(linvel: np.ndarray, minimum: float) -> np.ndarray:
    return np.asarray(np.maximum(linvel[:, 0], 0.0) >= minimum, dtype=F32)


def compute_forward_command_mask(commands: np.ndarray) -> np.ndarray:
    return np.asarray(np.maximum(commands[:, 0], 0.0) > 1.0e-6, dtype=F32)


def _parameter(context: NumpyTermContext, name: str) -> float:
    return cast(float, context.parameters[name])


def _store(context: NumpyTermContext, value: np.ndarray) -> None:
    np.copyto(context.output, value, casting="same_kind")


def _tracking_lin_vel(context: NumpyTermContext) -> None:
    error = np.sum(
        np.square(context.inputs["commands"][:, :2] - context.inputs["linvel"][:, :2]), axis=1
    )
    _store(context, np.exp(-error / _parameter(context, "tracking_sigma")))


def _tracking_ang_vel(context: NumpyTermContext) -> None:
    error = np.square(context.inputs["commands"][:, 2] - context.inputs["gyro"][:, 2])
    _store(context, np.exp(-error / _parameter(context, "tracking_sigma")))


def _forward_progress(context: NumpyTermContext) -> None:
    commanded = np.maximum(context.inputs["commands"][:, 0], 1.0e-6)
    forward = np.maximum(context.inputs["linvel"][:, 0], 0.0)
    _store(context, np.minimum(forward / commanded, 1.0))


def _under_speed(context: NumpyTermContext) -> None:
    commanded = np.maximum(context.inputs["commands"][:, 0], 1.0e-6)
    forward = np.maximum(context.inputs["linvel"][:, 0], 0.0)
    _store(context, np.maximum(context.inputs["commands"][:, 0] - forward, 0.0) / commanded)


def _squared_columns(source: str, columns: slice):
    def term(context: NumpyTermContext) -> None:
        np.sum(np.square(context.inputs[source][:, columns]), axis=1, out=context.output)

    return term


def _action_rate(context: NumpyTermContext) -> None:
    diff = context.inputs["current_actions"] - context.inputs["last_actions"]
    np.sum(np.square(diff), axis=1, out=context.output)


def _base_height(context: NumpyTermContext) -> None:
    np.square(
        context.inputs["base_height"] - _parameter(context, "base_height_target"),
        out=context.output,
    )


def _weighted_pose(weights: str):
    def term(context: NumpyTermContext) -> None:
        diff = context.inputs["dof_pos"] - context.inputs["default_angles"]
        np.sum(context.inputs[weights] * np.square(diff), axis=1, out=context.output)

    return term


def _feet_ori(context: NumpyTermContext) -> None:
    left, right = context.inputs["left_foot_quat"], context.inputs["right_foot_quat"]
    _store(
        context,
        np.square(left[:, 1])
        + np.square(left[:, 2])
        + np.square(right[:, 1])
        + np.square(right[:, 2]),
    )


def _gait_gate(context: NumpyTermContext) -> np.ndarray:
    return compute_forward_speed_gate(
        context.inputs["linvel"], _parameter(context, "min_forward_speed")
    )


def _feet_phase(context: NumpyTermContext) -> None:
    left, right = compute_feet_phase_height_targets(
        context.inputs["gait_phase"], _parameter(context, "swing_height")
    )
    error = np.square(context.inputs["left_foot_pos"][:, 2] - left)
    error += np.square(context.inputs["right_foot_pos"][:, 2] - right)
    _store(context, np.exp(-error / _parameter(context, "tracking_sigma")) * _gait_gate(context))


def _feet_phase_contrast(context: NumpyTermContext) -> None:
    left, right = compute_feet_phase_height_targets(
        context.inputs["gait_phase"], _parameter(context, "swing_height")
    )
    actual = context.inputs["left_foot_pos"][:, 2] - context.inputs["right_foot_pos"][:, 2]
    _store(
        context,
        np.exp(-np.square(actual - (left - right)) / _parameter(context, "tracking_sigma"))
        * _gait_gate(context),
    )


def _feet_phase_contact(context: NumpyTermContext) -> None:
    left, right = compute_feet_phase_contact_targets(
        context.inputs["gait_phase"], _parameter(context, "swing_height")
    )
    matches = 0.5 * (
        (context.inputs["left_contact"] == left).astype(F32)
        + (context.inputs["right_contact"] == right).astype(F32)
    )
    _store(context, matches * _gait_gate(context))


def _feet_double_stance(context: NumpyTermContext) -> None:
    stance = np.logical_and(context.inputs["left_contact"], context.inputs["right_contact"])
    _store(context, stance.astype(F32) * compute_forward_command_mask(context.inputs["commands"]))


def _alive(context: NumpyTermContext) -> None:
    context.output.fill(1.0)


def _scaled_copy(context: NumpyTermContext) -> None:
    np.multiply(
        next(iter(context.inputs.values())),
        _parameter(context, "multiplier"),
        out=context.output,
    )


def _tilt_or_height(context: NumpyTermContext) -> None:
    tilt = np.arccos(np.clip(context.inputs["gravity"][:, 2], -1.0, 1.0))
    np.logical_or(
        tilt > _parameter(context, "max_tilt_rad"),
        context.inputs["base_height"] < _parameter(context, "min_base_height"),
        out=context.output,
    )


@dataclass(frozen=True)
class G1WalkResolvedTermPlan:
    plan: ResolvedTermPlan
    reward_outputs: tuple[tuple[str, str], ...]
    observation_outputs: Mapping[str, tuple[str, ...]]
    observation_labels: Mapping[str, tuple[str, ...]]
    termination_outputs: tuple[str, ...]

    @property
    def observation_dims(self) -> dict[str, int]:
        return {
            group: sum(self.plan.output_specs[name].shape[0] for name in names)
            for group, names in self.observation_outputs.items()
        }


def _build_registry(num_action: int) -> TermRegistry:
    from . import joystick_numba

    registry = TermRegistry()
    shapes = {
        "linvel": (3,),
        "gyro": (3,),
        "gravity": (3,),
        "dof_pos": (num_action,),
        "dof_vel": (num_action,),
        "dof_pos_diff": (num_action,),
        "noisy_gyro": (3,),
        "noisy_gravity": (3,),
        "noisy_dof_pos": (num_action,),
        "noisy_dof_vel": (num_action,),
        "base_height": (),
        "commands": (3,),
        "current_actions": (num_action,),
        "last_actions": (num_action,),
        "gait_phase": (2,),
        "default_angles": (num_action,),
        "pose_weights": (num_action,),
        "upper_body_pose_weights": (num_action,),
        "left_foot_pos": (3,),
        "right_foot_pos": (3,),
        "left_foot_quat": (4,),
        "right_foot_quat": (4,),
        "left_contact": (),
        "right_contact": (),
    }

    def inputs(*names: str) -> tuple[NamedTensorSpec, ...]:
        return tuple(
            NamedTensorSpec(name, TensorSpec(shapes[name], np.bool_ if "contact" in name else F32))
            for name in names
        )

    float_param = lambda name: ParameterSpec(name, ParameterKind.FLOAT)  # noqa: E731
    definitions = (
        ("tracking_lin_vel", _tracking_lin_vel, ("linvel", "commands"), ("tracking_sigma",)),
        ("tracking_ang_vel", _tracking_ang_vel, ("gyro", "commands"), ("tracking_sigma",)),
        ("forward_progress", _forward_progress, ("linvel", "commands"), ()),
        ("under_speed", _under_speed, ("linvel", "commands"), ()),
        ("lin_vel_z", _squared_columns("linvel", slice(2, 3)), ("linvel",), ()),
        ("orientation", _squared_columns("gravity", slice(0, 2)), ("gravity",), ()),
        ("ang_vel_xy", _squared_columns("gyro", slice(0, 2)), ("gyro",), ()),
        ("action_rate", _action_rate, ("current_actions", "last_actions"), ()),
        ("base_height", _base_height, ("base_height",), ("base_height_target",)),
        ("pose", _weighted_pose("pose_weights"), ("dof_pos", "default_angles", "pose_weights"), ()),
        (
            "upper_body_pose",
            _weighted_pose("upper_body_pose_weights"),
            ("dof_pos", "default_angles", "upper_body_pose_weights"),
            (),
        ),
        ("feet_ori", _feet_ori, ("left_foot_quat", "right_foot_quat"), ()),
        (
            "feet_phase",
            _feet_phase,
            ("linvel", "gait_phase", "left_foot_pos", "right_foot_pos"),
            ("swing_height", "tracking_sigma", "min_forward_speed"),
        ),
        (
            "feet_phase_contrast",
            _feet_phase_contrast,
            ("linvel", "gait_phase", "left_foot_pos", "right_foot_pos"),
            ("swing_height", "tracking_sigma", "min_forward_speed"),
        ),
        (
            "feet_phase_contact",
            _feet_phase_contact,
            ("linvel", "gait_phase", "left_contact", "right_contact"),
            ("swing_height", "min_forward_speed"),
        ),
        (
            "feet_double_stance",
            _feet_double_stance,
            ("commands", "left_contact", "right_contact"),
            (),
        ),
        ("alive", _alive, (), ()),
    )
    for name, function, input_names, parameter_names in definitions:
        registry.register(
            TermDefinition(
                f"g1.walk.reward.{name}.v1",
                TermKind.REWARD,
                function,
                TensorSpec((), F32),
                inputs=inputs(*input_names),
                parameters=tuple(float_param(p) for p in parameter_names),
                numba_item_fn=joystick_numba.NUMBA_REWARD_ITEMS.get(name),
            )
        )

    for key, (_, _, source, width) in _OBS_META.items():
        registry.register(
            TermDefinition(
                key,
                TermKind.OBSERVATION,
                _scaled_copy,
                TensorSpec((num_action if width == -1 else width,), F32),
                inputs=inputs(source),
                parameters=(float_param("multiplier"),),
                numba_item_fn=joystick_numba.NUMBA_OBSERVATION_ITEM,
            )
        )
    registry.register(
        TermDefinition(
            DEFAULT_TERMINATION_TERMS[0],
            TermKind.TERMINATION,
            _tilt_or_height,
            TensorSpec((), np.bool_),
            inputs=inputs("gravity", "base_height"),
            parameters=(float_param("max_tilt_rad"), float_param("min_base_height")),
            numba_item_fn=joystick_numba.NUMBA_TERMINATION_ITEM,
        )
    )
    return registry


def _reward_parameters(name: str, cfg: Any) -> dict[str, float]:
    if name in ("tracking_lin_vel", "tracking_ang_vel"):
        return {"tracking_sigma": float(cfg.tracking_sigma)}
    if name == "base_height":
        return {"base_height_target": float(cfg.base_height_target)}
    if name in ("feet_phase", "feet_phase_contrast"):
        return {
            "swing_height": float(cfg.feet_phase_swing_height),
            "tracking_sigma": float(cfg.feet_phase_tracking_sigma),
            "min_forward_speed": float(cfg.min_forward_speed_for_gait_reward),
        }
    if name == "feet_phase_contact":
        return {
            "swing_height": float(cfg.feet_phase_swing_height),
            "min_forward_speed": float(cfg.min_forward_speed_for_gait_reward),
        }
    return {}


def resolve_g1_walk_term_plan(
    *,
    num_action: int,
    reward_cfg: Any,
    observations: Mapping[str, Sequence[str]],
    terminations: Sequence[str],
    walk_profile: bool,
) -> G1WalkResolvedTermPlan:
    """Resolve the task config into one reward/observation/termination plan."""
    registry = _build_registry(num_action)
    configs: list[TermConfig] = []
    reward_outputs: list[tuple[str, str]] = []
    for name, scale in reward_cfg.scales.items():
        key = REWARD_TERM_KEYS.get(name)
        if key is None:
            if scale != 0.0:
                raise TermPlanError(f"G1 walk has unknown nonzero reward term {name!r}")
            continue
        output_name = f"reward.{name}"
        configs.append(
            TermConfig(
                output_name, key, scale=scale, parameters=_reward_parameters(name, reward_cfg)
            )
        )
        reward_outputs.append((name, output_name))

    if set(observations) != {"obs", "critic"}:
        raise TermPlanError("G1 walk observation layout must define exactly 'obs' and 'critic'")
    observation_outputs: dict[str, tuple[str, ...]] = {}
    observation_labels: dict[str, tuple[str, ...]] = {}
    multipliers = {
        "actor_gyro": 0.25 if walk_profile else 1.0,
        "actor_dof_vel": 0.05 if walk_profile else 1.0,
        "critic_gyro": 0.25 if walk_profile else 1.0,
        "critic_dof_vel": 0.05 if walk_profile else 1.0,
        "critic_linvel": 2.0 if walk_profile else 1.0,
        "actor_gravity": -1.0,
        "critic_gravity": -1.0,
    }
    for group in ("obs", "critic"):
        keys = observations[group]
        if not isinstance(keys, (list, tuple)) or not keys:
            raise TermPlanError(f"G1 walk observation group {group!r} must be a non-empty list")
        if len(keys) != len(set(keys)):
            raise TermPlanError(f"G1 walk observation group {group!r} contains duplicate terms")
        names, labels = [], []
        for key in keys:
            meta = _OBS_META.get(key)
            if meta is None:
                registry.resolve(key)
            assert meta is not None
            declared_group, label, _, _ = meta
            if declared_group != group:
                raise TermPlanError(f"observation term {key!r} does not belong to group {group!r}")
            name = f"observation.{group}.{label}"
            configs.append(
                TermConfig(
                    name,
                    key,
                    parameters={"multiplier": multipliers.get(key.rsplit(".", 2)[-2], 1.0)},
                )
            )
            names.append(name)
            labels.append(label)
        observation_outputs[group] = tuple(names)
        observation_labels[group] = tuple(labels)

    if not isinstance(terminations, (list, tuple)) or not terminations:
        raise TermPlanError("G1 walk termination layout must be a non-empty list")
    termination_outputs = []
    for index, key in enumerate(terminations):
        name = f"termination.term{index}"
        configs.append(
            TermConfig(
                name,
                key,
                parameters={
                    "max_tilt_rad": math.radians(float(reward_cfg.max_tilt_deg)),
                    "min_base_height": float(reward_cfg.min_base_height),
                },
            )
        )
        termination_outputs.append(name)

    return G1WalkResolvedTermPlan(
        plan=resolve_term_plan(registry, configs),
        reward_outputs=tuple(reward_outputs),
        observation_outputs=MappingProxyType(observation_outputs),
        observation_labels=MappingProxyType(observation_labels),
        termination_outputs=tuple(termination_outputs),
    )
