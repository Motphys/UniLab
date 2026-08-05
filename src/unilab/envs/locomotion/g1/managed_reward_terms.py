"""Shared reward-term definitions for managed G1 execution profiles."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
import torch


class G1RewardMath(Protocol):
    """In-place tensor primitives required by the shared reward formulas."""

    def add(self, left: Any, right: Any, *, out: Any) -> None: ...

    def subtract(self, left: Any, right: Any, *, out: Any) -> None: ...

    def multiply(self, left: Any, right: Any, *, out: Any) -> None: ...

    def divide(self, left: Any, right: Any, *, out: Any) -> None: ...

    def square(self, value: Any, *, out: Any) -> None: ...

    def exp(self, value: Any, *, out: Any) -> None: ...

    def sqrt(self, value: Any, *, out: Any) -> None: ...

    def remainder(self, value: Any, divisor: float, *, out: Any) -> None: ...

    def sum(self, value: Any, *, axis: int, out: Any) -> None: ...

    def maximum(self, value: Any, minimum: float, *, out: Any) -> None: ...

    def minimum(self, value: Any, maximum: float, *, out: Any) -> None: ...

    def less(self, left: Any, right: Any, *, out: Any) -> None: ...

    def less_equal(self, left: Any, right: Any, *, out: Any) -> None: ...

    def greater_equal(self, left: Any, right: Any, *, out: Any) -> None: ...

    def where(self, condition: Any, when_true: Any, when_false: Any, *, out: Any) -> None: ...

    def copy(self, source: Any, *, out: Any) -> None: ...

    def fill(self, out: Any, value: float) -> None: ...


class NumpyG1RewardMath:
    def add(self, left: Any, right: Any, *, out: np.ndarray) -> None:
        np.add(left, right, out=out)

    def subtract(self, left: Any, right: Any, *, out: np.ndarray) -> None:
        np.subtract(left, right, out=out)

    def multiply(self, left: Any, right: Any, *, out: np.ndarray) -> None:
        np.multiply(left, right, out=out)

    def divide(self, left: Any, right: Any, *, out: np.ndarray) -> None:
        np.divide(left, right, out=out)

    def square(self, value: Any, *, out: np.ndarray) -> None:
        np.square(value, out=out)

    def exp(self, value: Any, *, out: np.ndarray) -> None:
        np.exp(value, out=out)

    def sqrt(self, value: Any, *, out: np.ndarray) -> None:
        np.sqrt(value, out=out)

    def remainder(self, value: Any, divisor: float, *, out: np.ndarray) -> None:
        np.remainder(value, divisor, out=out)

    def sum(self, value: Any, *, axis: int, out: np.ndarray) -> None:
        np.sum(value, axis=axis, out=out)

    def maximum(self, value: Any, minimum: float, *, out: np.ndarray) -> None:
        np.maximum(value, minimum, out=out)

    def minimum(self, value: Any, maximum: float, *, out: np.ndarray) -> None:
        np.minimum(value, maximum, out=out)

    def less(self, left: Any, right: Any, *, out: np.ndarray) -> None:
        np.less(left, right, out=out)

    def less_equal(self, left: Any, right: Any, *, out: np.ndarray) -> None:
        np.less_equal(left, right, out=out)

    def greater_equal(self, left: Any, right: Any, *, out: np.ndarray) -> None:
        np.greater_equal(left, right, out=out)

    def where(
        self,
        condition: np.ndarray,
        when_true: Any,
        when_false: Any,
        *,
        out: np.ndarray,
    ) -> None:
        np.copyto(out, when_false, casting="unsafe")
        np.copyto(out, when_true, where=condition, casting="unsafe")

    def copy(self, source: Any, *, out: np.ndarray) -> None:
        np.copyto(out, source, casting="unsafe")

    def fill(self, out: np.ndarray, value: float) -> None:
        out.fill(value)


class TorchG1RewardMath:
    def add(self, left: Any, right: Any, *, out: torch.Tensor) -> None:
        torch.add(left, right, out=out)

    def subtract(self, left: Any, right: Any, *, out: torch.Tensor) -> None:
        torch.sub(left, right, out=out)

    def multiply(self, left: Any, right: Any, *, out: torch.Tensor) -> None:
        torch.mul(left, right, out=out)

    def divide(self, left: Any, right: Any, *, out: torch.Tensor) -> None:
        torch.div(left, right, out=out)

    def square(self, value: Any, *, out: torch.Tensor) -> None:
        torch.square(value, out=out)

    def exp(self, value: Any, *, out: torch.Tensor) -> None:
        torch.exp(value, out=out)

    def sqrt(self, value: Any, *, out: torch.Tensor) -> None:
        torch.sqrt(value, out=out)

    def remainder(self, value: Any, divisor: float, *, out: torch.Tensor) -> None:
        torch.remainder(value, divisor, out=out)

    def sum(self, value: Any, *, axis: int, out: torch.Tensor) -> None:
        torch.sum(value, dim=axis, out=out)

    def maximum(self, value: Any, minimum: float, *, out: torch.Tensor) -> None:
        torch.clamp_min(value, minimum, out=out)

    def minimum(self, value: Any, maximum: float, *, out: torch.Tensor) -> None:
        torch.clamp_max(value, maximum, out=out)

    def less(self, left: Any, right: Any, *, out: torch.Tensor) -> None:
        torch.lt(left, right, out=out)

    def less_equal(self, left: Any, right: Any, *, out: torch.Tensor) -> None:
        torch.le(left, right, out=out)

    def greater_equal(self, left: Any, right: Any, *, out: torch.Tensor) -> None:
        torch.ge(left, right, out=out)

    def where(
        self,
        condition: torch.Tensor,
        when_true: Any,
        when_false: Any,
        *,
        out: torch.Tensor,
    ) -> None:
        torch.where(condition, when_true, when_false, out=out)

    def copy(self, source: Any, *, out: torch.Tensor) -> None:
        out.copy_(source, non_blocking=True)

    def fill(self, out: torch.Tensor, value: float) -> None:
        out.fill_(value)


NUMPY_G1_REWARD_MATH = NumpyG1RewardMath()
TORCH_G1_REWARD_MATH = TorchG1RewardMath()


@dataclass(frozen=True)
class G1RewardScratch:
    bool_a: Any
    bool_b: Any
    scalar_b: Any
    scalar_c: Any
    scalar_d: Any
    vector2: Any
    action: Any
    left_height: Any
    right_height: Any


@dataclass(frozen=True)
class G1RewardContext:
    commands: Any
    current_actions: Any
    last_actions: Any
    gait_phase: Any
    root_position: Any
    dof_position: Any
    linear_velocity: Any
    gyro: Any
    upvector: Any
    left_foot_position: Any
    right_foot_position: Any
    default_angles: Any
    pose_weights: Any
    upper_body_pose_weights: Any
    tracking_sigma: float
    base_height_target: float
    feet_phase_swing_height: float
    feet_phase_tracking_sigma: float
    min_forward_speed_for_gait_reward: float
    close_feet_threshold: float
    scratch: G1RewardScratch


class G1RewardEvaluator(Protocol):
    def __call__(
        self,
        math_ops: G1RewardMath,
        context: G1RewardContext,
        *,
        out: Any,
    ) -> None: ...


@dataclass(frozen=True)
class G1RewardTerm:
    key: str
    evaluate: G1RewardEvaluator


def _tracking_lin_vel(math_ops: G1RewardMath, context: G1RewardContext, *, out: Any) -> None:
    vector2 = context.scratch.vector2
    math_ops.subtract(context.commands[:, :2], context.linear_velocity[:, :2], out=vector2)
    math_ops.square(vector2, out=vector2)
    math_ops.sum(vector2, axis=1, out=out)
    math_ops.multiply(out, -1.0 / context.tracking_sigma, out=out)
    math_ops.exp(out, out=out)


def _tracking_ang_vel(math_ops: G1RewardMath, context: G1RewardContext, *, out: Any) -> None:
    math_ops.subtract(context.commands[:, 2], context.gyro[:, 2], out=out)
    math_ops.square(out, out=out)
    math_ops.multiply(out, -1.0 / context.tracking_sigma, out=out)
    math_ops.exp(out, out=out)


def _forward_progress(math_ops: G1RewardMath, context: G1RewardContext, *, out: Any) -> None:
    commanded_speed = context.scratch.scalar_b
    math_ops.maximum(context.linear_velocity[:, 0], 0.0, out=out)
    math_ops.maximum(context.commands[:, 0], 1.0e-6, out=commanded_speed)
    math_ops.divide(out, commanded_speed, out=out)
    math_ops.minimum(out, 1.0, out=out)


def _under_speed(math_ops: G1RewardMath, context: G1RewardContext, *, out: Any) -> None:
    commanded_speed = context.scratch.scalar_b
    gap = context.scratch.scalar_c
    math_ops.maximum(context.commands[:, 0], 1.0e-6, out=commanded_speed)
    math_ops.maximum(context.linear_velocity[:, 0], 0.0, out=out)
    math_ops.subtract(context.commands[:, 0], out, out=gap)
    math_ops.maximum(gap, 0.0, out=gap)
    math_ops.divide(gap, commanded_speed, out=out)


def _lin_vel_z(math_ops: G1RewardMath, context: G1RewardContext, *, out: Any) -> None:
    math_ops.square(context.linear_velocity[:, 2], out=out)


def _orientation(math_ops: G1RewardMath, context: G1RewardContext, *, out: Any) -> None:
    vector2 = context.scratch.vector2
    math_ops.square(context.upvector[:, :2], out=vector2)
    math_ops.sum(vector2, axis=1, out=out)


def _ang_vel_xy(math_ops: G1RewardMath, context: G1RewardContext, *, out: Any) -> None:
    vector2 = context.scratch.vector2
    math_ops.square(context.gyro[:, :2], out=vector2)
    math_ops.sum(vector2, axis=1, out=out)


def _action_rate(math_ops: G1RewardMath, context: G1RewardContext, *, out: Any) -> None:
    action = context.scratch.action
    math_ops.subtract(context.current_actions, context.last_actions, out=action)
    math_ops.square(action, out=action)
    math_ops.sum(action, axis=1, out=out)


def _base_height(math_ops: G1RewardMath, context: G1RewardContext, *, out: Any) -> None:
    math_ops.subtract(context.root_position[:, 2], context.base_height_target, out=out)
    math_ops.square(out, out=out)


def _weighted_pose(
    math_ops: G1RewardMath,
    context: G1RewardContext,
    *,
    weights: Any,
    out: Any,
) -> None:
    action = context.scratch.action
    math_ops.subtract(context.dof_position, context.default_angles, out=action)
    math_ops.square(action, out=action)
    math_ops.multiply(action, weights, out=action)
    math_ops.sum(action, axis=1, out=out)


def _pose(math_ops: G1RewardMath, context: G1RewardContext, *, out: Any) -> None:
    _weighted_pose(math_ops, context, weights=context.pose_weights, out=out)


def _upper_body_pose(math_ops: G1RewardMath, context: G1RewardContext, *, out: Any) -> None:
    _weighted_pose(math_ops, context, weights=context.upper_body_pose_weights, out=out)


def _penalty_close_feet_xy(math_ops: G1RewardMath, context: G1RewardContext, *, out: Any) -> None:
    scratch = context.scratch
    math_ops.subtract(
        context.left_foot_position[:, :2],
        context.right_foot_position[:, :2],
        out=scratch.vector2,
    )
    math_ops.square(scratch.vector2, out=scratch.vector2)
    math_ops.sum(scratch.vector2, axis=1, out=out)
    math_ops.sqrt(out, out=out)
    math_ops.less(out, context.close_feet_threshold, out=scratch.bool_a)
    math_ops.subtract(out, context.close_feet_threshold, out=scratch.scalar_b)
    math_ops.square(scratch.scalar_b, out=scratch.scalar_b)
    math_ops.fill(scratch.scalar_c, 0.0)
    math_ops.where(scratch.bool_a, scratch.scalar_b, scratch.scalar_c, out=out)


def _bezier_height(
    math_ops: G1RewardMath,
    context: G1RewardContext,
    phase: Any,
    *,
    out: Any,
    work: Any,
) -> None:
    scratch = context.scratch
    math_ops.add(phase, math.pi, out=work)
    math_ops.remainder(work, 2.0 * math.pi, out=work)
    math_ops.divide(work, 2.0 * math.pi, out=work)
    math_ops.less_equal(work, 0.5, out=scratch.bool_a)

    math_ops.multiply(work, 2.0, out=scratch.scalar_b)
    math_ops.square(scratch.scalar_b, out=out)
    math_ops.multiply(out, 3.0, out=out)
    math_ops.square(scratch.scalar_b, out=scratch.scalar_c)
    math_ops.multiply(scratch.scalar_c, scratch.scalar_b, out=scratch.scalar_c)
    math_ops.multiply(scratch.scalar_c, 2.0, out=scratch.scalar_c)
    math_ops.subtract(out, scratch.scalar_c, out=out)

    math_ops.subtract(scratch.scalar_b, 1.0, out=scratch.scalar_b)
    math_ops.square(scratch.scalar_b, out=scratch.scalar_c)
    math_ops.multiply(scratch.scalar_c, 3.0, out=scratch.scalar_c)
    math_ops.square(scratch.scalar_b, out=scratch.scalar_d)
    math_ops.multiply(scratch.scalar_d, scratch.scalar_b, out=scratch.scalar_d)
    math_ops.multiply(scratch.scalar_d, 2.0, out=scratch.scalar_d)
    math_ops.subtract(scratch.scalar_c, scratch.scalar_d, out=scratch.scalar_c)
    math_ops.multiply(scratch.scalar_c, -1.0, out=scratch.scalar_c)
    math_ops.add(scratch.scalar_c, 1.0, out=scratch.scalar_c)
    math_ops.where(scratch.bool_a, out, scratch.scalar_c, out=scratch.scalar_d)
    math_ops.copy(scratch.scalar_d, out=out)
    math_ops.multiply(out, context.feet_phase_swing_height, out=out)


def _feet_targets(math_ops: G1RewardMath, context: G1RewardContext, *, work: Any) -> None:
    _bezier_height(
        math_ops,
        context,
        context.gait_phase[:, 0],
        out=context.scratch.left_height,
        work=work,
    )
    _bezier_height(
        math_ops,
        context,
        context.gait_phase[:, 1],
        out=context.scratch.right_height,
        work=work,
    )


def _apply_gait_gate(math_ops: G1RewardMath, context: G1RewardContext, *, out: Any) -> None:
    scratch = context.scratch
    math_ops.maximum(context.linear_velocity[:, 0], 0.0, out=scratch.scalar_c)
    math_ops.greater_equal(
        scratch.scalar_c,
        context.min_forward_speed_for_gait_reward,
        out=scratch.bool_b,
    )
    math_ops.copy(scratch.bool_b, out=scratch.scalar_d)
    math_ops.multiply(out, scratch.scalar_d, out=out)


def _feet_phase(math_ops: G1RewardMath, context: G1RewardContext, *, out: Any) -> None:
    scratch = context.scratch
    _feet_targets(math_ops, context, work=out)
    math_ops.subtract(context.left_foot_position[:, 2], scratch.left_height, out=out)
    math_ops.square(out, out=out)
    math_ops.subtract(context.right_foot_position[:, 2], scratch.right_height, out=scratch.scalar_b)
    math_ops.square(scratch.scalar_b, out=scratch.scalar_b)
    math_ops.add(out, scratch.scalar_b, out=out)
    math_ops.multiply(out, -1.0 / context.feet_phase_tracking_sigma, out=out)
    math_ops.exp(out, out=out)
    _apply_gait_gate(math_ops, context, out=out)


def _feet_phase_contrast(math_ops: G1RewardMath, context: G1RewardContext, *, out: Any) -> None:
    scratch = context.scratch
    _feet_targets(math_ops, context, work=out)
    math_ops.subtract(context.left_foot_position[:, 2], context.right_foot_position[:, 2], out=out)
    math_ops.subtract(scratch.left_height, scratch.right_height, out=scratch.scalar_b)
    math_ops.subtract(out, scratch.scalar_b, out=out)
    math_ops.square(out, out=out)
    math_ops.multiply(out, -1.0 / context.feet_phase_tracking_sigma, out=out)
    math_ops.exp(out, out=out)
    _apply_gait_gate(math_ops, context, out=out)


def _alive(math_ops: G1RewardMath, context: G1RewardContext, *, out: Any) -> None:
    del context
    math_ops.fill(out, 1.0)


def _term(
    key: str,
    evaluate: G1RewardEvaluator,
) -> G1RewardTerm:
    return G1RewardTerm(key=key, evaluate=evaluate)


_TERMS = (
    _term("tracking_lin_vel", _tracking_lin_vel),
    _term("tracking_ang_vel", _tracking_ang_vel),
    _term("forward_progress", _forward_progress),
    _term("under_speed", _under_speed),
    _term("lin_vel_z", _lin_vel_z),
    _term("orientation", _orientation),
    _term("penalty_orientation", _orientation),
    _term("ang_vel_xy", _ang_vel_xy),
    _term("penalty_ang_vel_xy", _ang_vel_xy),
    _term("action_rate", _action_rate),
    _term("penalty_action_rate", _action_rate),
    _term("base_height", _base_height),
    _term("pose", _pose),
    _term("upper_body_pose", _upper_body_pose),
    _term("penalty_close_feet_xy", _penalty_close_feet_xy),
    _term("feet_phase", _feet_phase),
    _term("feet_phase_contrast", _feet_phase_contrast),
    _term("alive", _alive),
)
G1_REWARD_TERM_REGISTRY: Mapping[str, G1RewardTerm] = MappingProxyType(
    {term.key: term for term in _TERMS}
)


def unsupported_g1_reward_terms(names: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(names) - set(G1_REWARD_TERM_REGISTRY)))


def bind_g1_reward_terms(
    configured_terms: Iterable[tuple[str, float]],
) -> tuple[tuple[G1RewardTerm, float], ...]:
    bound: list[tuple[G1RewardTerm, float]] = []
    for name, scale in configured_terms:
        try:
            term = G1_REWARD_TERM_REGISTRY[name]
        except KeyError as exc:
            raise ValueError(f"G1 reward term {name!r} is not registered") from exc
        bound.append((term, float(scale)))
    return tuple(bound)


__all__ = [
    "G1_REWARD_TERM_REGISTRY",
    "G1RewardContext",
    "G1RewardScratch",
    "G1RewardTerm",
    "NUMPY_G1_REWARD_MATH",
    "TORCH_G1_REWARD_MATH",
    "bind_g1_reward_terms",
    "unsupported_g1_reward_terms",
]
