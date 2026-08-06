from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
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
from unilab.utils.geometry import (
    np_gravity_z_in_body_from_quat,
    np_write_relative_anchor_transform_pos_rot6d,
)

from . import rewards
from .observations import write_body_ori6_in_anchor_frame, write_body_pos_in_anchor_frame
from .transforms import write_relative_transforms

PREAMBLE_KEY = "motion.tracking.preamble.transforms.v1"
REWARD_KEYS = {
    name: f"motion.tracking.reward.{name}.v1"
    for name in (
        "motion_global_root_pos",
        "motion_global_root_ori",
        "motion_body_pos",
        "motion_body_ori",
        "motion_body_lin_vel",
        "motion_body_ang_vel",
        "motion_ee_body_pos_z",
        "motion_joint_pos",
        "motion_joint_vel",
        "action_rate_l2",
        "joint_limit",
        "undesired_contacts",
    )
}
DEFAULT_TERMINATIONS = ("motion.tracking.termination.failures.v1",)

_OBS_META = {
    f"motion.tracking.observation.{key}.v1": (group, label, source, width, workspace)
    for group, entries in {
        "obs": (
            ("actor_command", "command", "motion_command", -2, True),
            ("actor_anchor_pos", "anchor_pos", "motion_anchor_pos_b", 3, True),
            ("actor_anchor_ori", "anchor_ori", "motion_anchor_ori_b", 6, True),
            ("actor_linvel", "linvel", "noisy_linvel", 3, False),
            ("actor_gyro", "gyro", "noisy_gyro", 3, False),
            ("actor_joint_pos", "joint_pos", "noisy_joint_pos_rel", -1, False),
            ("actor_dof_vel", "dof_vel", "noisy_dof_vel", -1, False),
            ("actor_actions", "actions", "obs_actions", -1, False),
        ),
        "critic": (
            ("critic_command", "command", "motion_command", -2, True),
            ("critic_anchor_pos", "anchor_pos", "motion_anchor_pos_b", 3, True),
            ("critic_anchor_ori", "anchor_ori", "motion_anchor_ori_b", 6, True),
            ("critic_linvel", "linvel", "linvel", 3, False),
            ("critic_gyro", "gyro", "gyro", 3, False),
            ("critic_joint_pos", "joint_pos", "joint_pos_rel", -1, True),
            ("critic_dof_vel", "dof_vel", "dof_vel", -1, False),
            ("critic_actions", "actions", "obs_actions", -1, False),
            ("critic_body_pos", "body_pos", "robot_body_pos_b", -3, True),
            ("critic_body_ori", "body_ori", "robot_body_ori_b", -6, True),
        ),
    }.items()
    for key, label, source, width, workspace in entries
}
DEFAULT_OBSERVATIONS = {
    group: [key for key, meta in _OBS_META.items() if meta[0] == group]
    for group in ("obs", "critic")
}


def _param(ctx: NumpyTermContext, name: str) -> float:
    return cast(float, ctx.parameters[name])


# fmt: off
def _preamble(ctx: NumpyTermContext) -> None:
    inputs, work = ctx.inputs, ctx.workspace
    i = cast(int, ctx.parameters["anchor_body_idx"])
    write_relative_transforms(
        motion_body_pos_w=inputs["motion_body_pos_w"], motion_body_quat_w=inputs["motion_body_quat_w"],
        robot_body_pos_w=inputs["robot_body_pos_w"], robot_body_quat_w=inputs["robot_body_quat_w"],
        anchor_body_idx=i, delta_pos_w=work["delta_pos_w"], delta_ori_w=work["delta_ori_w"],
        body_vec_error=work["body_vec_error"], scalar_scratch=work["env_error"],
        scalar_scratch2=work["term_scratch"], out_body_pos_w=work["ref_body_pos_w"],
        out_body_quat_w=work["ref_body_quat_w"],
    )
    motion_pos, motion_quat = inputs["motion_body_pos_w"][:, i], inputs["motion_body_quat_w"][:, i]
    robot_pos, robot_quat = inputs["robot_body_pos_w"][:, i], inputs["robot_body_quat_w"][:, i]
    np_write_relative_anchor_transform_pos_rot6d(
        robot_pos, robot_quat, motion_pos, motion_quat,
        work["motion_anchor_pos_b"], work["motion_anchor_ori_b"],
    )
    n = inputs["dof_pos"].shape[1]
    work["motion_command"][:, :n], work["motion_command"][:, n:] = inputs["motion_joint_pos"], inputs["motion_joint_vel"]
    np.subtract(inputs["dof_pos"], inputs["effective_default_angles"], out=work["joint_pos_rel"])
    write_body_pos_in_anchor_frame(
        robot_pos, robot_quat, inputs["robot_body_pos_w"], work["robot_body_pos_b"],
        body_vec_error=work["body_vec_error"],
    )
    write_body_ori6_in_anchor_frame(robot_quat, inputs["robot_body_quat_w"], work["robot_body_ori_b"])
    ctx.output.fill(0.0)
# fmt: on


def _copy(ctx: NumpyTermContext) -> None:
    np.copyto(
        ctx.output,
        next(iter((*ctx.inputs.values(), *ctx.workspace.values()))).reshape(ctx.output.shape),
    )


# fmt: off
class _RewardAdapter:
    """Cold-bound adapter around the existing vectorized owner term."""

    def __init__(self, fn: Any) -> None:
        self.fn = fn
        self.bound: rewards.RewardContext | None = None

    def __call__(self, ctx: NumpyTermContext) -> None:
        if self.bound is None:
            get = lambda name: ctx.inputs.get(name)  # noqa: E731
            work = lambda name: ctx.workspace.get(name)  # noqa: E731
            std = cast(float, ctx.parameters.get("std", 1.0))
            std_cfg = SimpleNamespace(**{name: std for name in (
                "std_root_pos", "std_root_ori", "std_body_pos", "std_body_ori",
                "std_body_lin_vel", "std_body_ang_vel", "std_joint_pos", "std_joint_vel",
            )})
            indices = np.asarray(ctx.parameters.get("indices", ()), dtype=np.intp)
            env_error, indexed_error, indexed_mask = work("env_error"), work("indexed_error"), work("indexed_mask")
            direct_fields = (
                "robot_body_pos_w", "robot_body_quat_w", "robot_body_lin_vel_w",
                "robot_body_ang_vel_w", "dof_pos", "dof_vel", "joint_lower", "joint_upper",
            )
            scratch_fields = (
                "body_vec_error", "joint_error", "joint_error_upper", "quat_error_w", "quat_error_x",
            )
            motion_fields = (
                "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "joint_pos", "joint_vel",
            )
            self.bound = rewards.RewardContext(
                info={"current_actions": get("current_actions"), "last_actions": get("last_actions")},
                motion_data=SimpleNamespace(**{name: get(f"motion_{name}") for name in motion_fields}),
                ref_body_pos_w=work("ref_body_pos_w"), ref_body_quat_w=work("ref_body_quat_w"), reward_config=std_cfg,
                anchor_body_idx=cast(int, ctx.parameters.get("anchor_body_idx", 0)),
                ee_body_indices=indices, undesired_contact_body_indices=indices,
                undesired_contact_z_threshold=cast(float, ctx.parameters.get("threshold", 0.0)),
                num_envs=ctx.output.shape[0], env_error=env_error if env_error is not None else ctx.output,
                reward_term=ctx.output,
                ee_pos_error_z=None if indexed_error is None else indexed_error[:, : len(indices)],
                undesired_contact_mask=None if indexed_mask is None else indexed_mask[:, : len(indices)],
                **{name: get(name) for name in direct_fields},
                **{name: work(name) for name in scratch_fields},
            )
        result = self.fn(self.bound)
        if result is not ctx.output:
            np.copyto(ctx.output, result)
# fmt: on


# fmt: off
def _termination(ctx: NumpyTermContext) -> None:
    i = cast(int, ctx.parameters["anchor_body_idx"])
    error = ctx.workspace["env_error"]
    np.subtract(ctx.inputs["motion_body_pos_w"][:, i, 2], ctx.inputs["robot_body_pos_w"][:, i, 2], out=error)
    np.abs(error, out=error)
    np.greater(error, _param(ctx, "anchor_pos_z_threshold"), out=ctx.output)
    if cast(bool, ctx.parameters["check_anchor_ori"]):
        np.subtract(np_gravity_z_in_body_from_quat(ctx.inputs["motion_body_quat_w"][:, i]),
                    np_gravity_z_in_body_from_quat(ctx.inputs["robot_body_quat_w"][:, i]), out=error)
        np.abs(error, out=error)
        np.logical_or(ctx.output, error > _param(ctx, "anchor_ori_threshold"), out=ctx.output)
    for prefix, source in (("ee", ctx.workspace["ref_body_pos_w"]), ("contact", None)):
        indices = cast(tuple[int, ...], ctx.parameters[f"{prefix}_indices"])
        if not indices:
            continue
        mask = ctx.workspace["indexed_mask"][:, : len(indices)]
        if source is None:
            np.less(ctx.inputs["robot_body_pos_w"][:, indices, 2], _param(ctx, "contact_threshold"), out=mask)
        else:
            indexed_error = ctx.workspace["indexed_error"][:, : len(indices)]
            np.subtract(source[:, indices, 2], ctx.inputs["robot_body_pos_w"][:, indices, 2], out=indexed_error)
            np.abs(indexed_error, out=indexed_error)
            np.greater(indexed_error, _param(ctx, "ee_threshold"), out=mask)
        np.logical_or.reduce(mask, axis=1, out=error)
        np.logical_or(ctx.output, error, out=ctx.output)
# fmt: on


@dataclass(frozen=True)
class MotionResolvedTermPlan:
    plan: ResolvedTermPlan
    preamble_outputs: tuple[str, ...]
    reward_outputs: tuple[tuple[str, str], ...]
    reward_available: Mapping[str, bool]
    observation_outputs: Mapping[str, tuple[str, ...]]
    observation_labels: Mapping[str, tuple[str, ...]]
    termination_outputs: tuple[str, ...]

    @property
    def observation_dims(self) -> dict[str, int]:
        return {
            group: sum(self.plan.output_specs[name].shape[0] for name in names)
            for group, names in self.observation_outputs.items()
        }


def _build_registry(*, n_action: int, n_body: int, n_indexed: int, dtype: np.dtype) -> TermRegistry:
    from . import numba as motion_numba

    registry = TermRegistry()
    # fmt: off
    shapes = {
        "motion_body_pos_w": (n_body, 3), "motion_body_quat_w": (n_body, 4),
        "motion_body_lin_vel_w": (n_body, 3), "motion_body_ang_vel_w": (n_body, 3),
        "motion_joint_pos": (n_action,), "motion_joint_vel": (n_action,),
        "robot_body_pos_w": (n_body, 3), "robot_body_quat_w": (n_body, 4),
        "robot_body_lin_vel_w": (n_body, 3), "robot_body_ang_vel_w": (n_body, 3),
        "dof_pos": (n_action,), "dof_vel": (n_action,), "linvel": (3,), "gyro": (3,),
        "noisy_linvel": (3,), "noisy_gyro": (3,), "noisy_joint_pos_rel": (n_action,),
        "noisy_dof_vel": (n_action,), "current_actions": (n_action,),
        "last_actions": (n_action,), "obs_actions": (n_action,),
        "effective_default_angles": (n_action,), "joint_lower": (n_action,),
        "joint_upper": (n_action,),
    }
    work_shapes = {
        "delta_pos_w": (3,), "delta_ori_w": (4,), "body_vec_error": (n_body, 3),
        "env_error": (), "term_scratch": (), "ref_body_pos_w": (n_body, 3),
        "ref_body_quat_w": (n_body, 4), "motion_anchor_pos_b": (3,),
        "motion_anchor_ori_b": (6,), "motion_command": (2 * n_action,),
        "joint_pos_rel": (n_action,), "robot_body_pos_b": (n_body, 3),
        "robot_body_ori_b": (n_body, 6), "quat_error_w": (n_body,),
        "quat_error_x": (n_body,), "joint_error": (n_action,),
        "joint_error_upper": (n_action,), "indexed_error": (n_indexed,),
        "indexed_mask": (n_indexed,),
    }
    # fmt: on
    # The remaining declarations are a compact registry table.
    # fmt: off
    def ins(*names: str) -> tuple[NamedTensorSpec, ...]:
        return tuple(NamedTensorSpec(name, TensorSpec(shapes[name], dtype)) for name in names)

    def work(*names: str) -> tuple[NamedTensorSpec, ...]:
        return tuple(NamedTensorSpec(name, TensorSpec(work_shapes[name], np.bool_ if name == "indexed_mask" else dtype)) for name in names)

    def register(key: str, kind: TermKind, fn: Any, output: TensorSpec,
                 inputs: tuple[str, ...] = (), workspace: tuple[str, ...] = (),
                 parameters: tuple[ParameterSpec, ...] = (), numba_item_fn: Any = None) -> None:
        registry.register(TermDefinition(key, kind, fn, output, inputs=ins(*inputs),
                                         workspace=work(*workspace), parameters=parameters,
                                         numba_item_fn=numba_item_fn))

    scalar = TensorSpec((), dtype)

    def float_param(name: str) -> ParameterSpec:
        return ParameterSpec(name, ParameterKind.FLOAT)

    def int_param(name: str, many: bool = False) -> ParameterSpec:
        return ParameterSpec(name, ParameterKind.INTEGER, tuple_value=many)

    def bool_param(name: str) -> ParameterSpec:
        return ParameterSpec(name, ParameterKind.BOOLEAN)

    preamble_inputs = (
        "motion_body_pos_w", "motion_body_quat_w", "motion_joint_pos", "motion_joint_vel",
        "robot_body_pos_w", "robot_body_quat_w", "dof_pos", "effective_default_angles",
    )
    preamble_work = (
        "delta_pos_w", "delta_ori_w", "body_vec_error", "env_error", "term_scratch",
        "ref_body_pos_w", "ref_body_quat_w", "motion_anchor_pos_b", "motion_anchor_ori_b",
        "motion_command", "joint_pos_rel", "robot_body_pos_b", "robot_body_ori_b",
    )
    register(PREAMBLE_KEY, TermKind.OBSERVATION, _preamble, scalar, preamble_inputs,
             preamble_work, (int_param("anchor_body_idx"),), motion_numba.NUMBA_PREAMBLE_ITEM)

    # (function, inputs, workspace, parameters); kept compact so the paired
    # authoring surface is auditable as one table.
    reward_defs = {
        "motion_global_root_pos": (rewards.motion_global_root_pos, ("motion_body_pos_w", "robot_body_pos_w"), ("env_error",), (int_param("anchor_body_idx"), float_param("std"))),
        "motion_global_root_ori": (rewards.motion_global_root_ori, ("motion_body_quat_w", "robot_body_quat_w"), ("env_error",), (int_param("anchor_body_idx"), float_param("std"))),
        "motion_body_pos": (rewards.motion_body_pos, ("robot_body_pos_w",), ("ref_body_pos_w", "body_vec_error", "env_error"), (float_param("std"),)),
        "motion_body_ori": (rewards.motion_body_ori, ("robot_body_quat_w",), ("ref_body_quat_w", "quat_error_w", "quat_error_x", "env_error"), (float_param("std"),)),
        "motion_body_lin_vel": (rewards.motion_body_lin_vel, ("motion_body_lin_vel_w", "robot_body_lin_vel_w"), ("body_vec_error", "env_error"), (float_param("std"),)),
        "motion_body_ang_vel": (rewards.motion_body_ang_vel, ("motion_body_ang_vel_w", "robot_body_ang_vel_w"), ("body_vec_error", "env_error"), (float_param("std"),)),
        "motion_ee_body_pos_z": (rewards.motion_ee_body_pos_z, ("robot_body_pos_w",), ("ref_body_pos_w", "indexed_error", "env_error"), (int_param("indices", True), float_param("std"))),
        "motion_joint_pos": (rewards.motion_joint_pos, ("motion_joint_pos", "dof_pos"), ("joint_error", "env_error"), (float_param("std"),)),
        "motion_joint_vel": (rewards.motion_joint_vel, ("motion_joint_vel", "dof_vel"), ("joint_error", "env_error"), (float_param("std"),)),
        "action_rate_l2": (rewards.action_rate_l2, ("current_actions", "last_actions"), ("joint_error", "env_error"), ()),
        "joint_limit": (rewards.joint_limit, ("joint_lower", "joint_upper", "dof_pos"), ("joint_error", "joint_error_upper"), ()),
        "undesired_contacts": (rewards.undesired_contacts, ("robot_body_pos_w",), ("indexed_mask", "env_error"), (int_param("indices", True), float_param("threshold"))),
    }
    for name, (fn, inputs, workspace, params) in reward_defs.items():
        register(REWARD_KEYS[name], TermKind.REWARD, _RewardAdapter(fn), scalar, inputs, workspace,
                 params, motion_numba.NUMBA_REWARD_ITEMS.get(name))

    for key, (group, _label, source, width, is_workspace) in _OBS_META.items():
        resolved_width = {-1: n_action, -2: 2 * n_action, -3: 3 * n_body, -6: 6 * n_body}.get(width, width)
        register(key, TermKind.OBSERVATION, _copy, TensorSpec((resolved_width,), dtype),
                 workspace=(source,) if is_workspace else (),
                 inputs=() if is_workspace else (source,),
                 numba_item_fn=motion_numba.NUMBA_OBSERVATION_ITEM)

    termination_params = (
        int_param("anchor_body_idx"), float_param("anchor_pos_z_threshold"),
        bool_param("check_anchor_ori"), float_param("anchor_ori_threshold"),
        int_param("ee_indices", True), float_param("ee_threshold"),
        int_param("contact_indices", True), float_param("contact_threshold"),
    )
    register(DEFAULT_TERMINATIONS[0], TermKind.TERMINATION, _termination, TensorSpec((), np.bool_),
             ("motion_body_pos_w", "motion_body_quat_w", "robot_body_pos_w", "robot_body_quat_w"),
             ("ref_body_pos_w", "env_error", "indexed_error", "indexed_mask"), termination_params,
             motion_numba.NUMBA_TERMINATION_ITEM)
    return registry


# fmt: on


def resolve_motion_term_plan(
    *,
    cfg: Any,
    n_action: int,
    n_body: int,
    anchor_body_idx: int,
    ee_indices: np.ndarray,
    undesired_indices: np.ndarray,
    config: Any,
    dtype: np.dtype,
) -> MotionResolvedTermPlan:
    """Resolve the task-owned motion tracking numeric plan."""
    n_indexed = max(1, len(ee_indices), len(undesired_indices))
    registry = _build_registry(n_action=n_action, n_body=n_body, n_indexed=n_indexed, dtype=dtype)
    configs: list[TermConfig] = []
    preamble_outputs = []
    for index, key in enumerate(config.preambles):
        name = f"preamble.term{index}"
        configs.append(
            TermConfig(
                name,
                key,
                parameters={"anchor_body_idx": anchor_body_idx} if key == PREAMBLE_KEY else {},
            )
        )
        preamble_outputs.append(name)

    # fmt: off
    std_names = {
        "motion_global_root_pos": "std_root_pos", "motion_global_root_ori": "std_root_ori",
        "motion_body_pos": "std_body_pos", "motion_body_ori": "std_body_ori",
        "motion_body_lin_vel": "std_body_lin_vel", "motion_body_ang_vel": "std_body_ang_vel",
        "motion_ee_body_pos_z": "std_body_pos", "motion_joint_pos": "std_joint_pos",
        "motion_joint_vel": "std_joint_vel",
    }
    # fmt: on
    available = {
        "motion_ee_body_pos_z": bool(len(ee_indices)),
        "joint_limit": cfg._joint_lower is not None,
        "undesired_contacts": bool(len(undesired_indices)),
    }
    reward_outputs = []
    for name, scale in cfg._cfg.reward_config.scales.items():
        key = REWARD_KEYS.get(name)
        if key is None:
            if scale != 0.0:
                raise TermPlanError(f"motion tracking has unknown nonzero reward term {name!r}")
            continue
        params: dict[str, object] = {}
        if name in std_names:
            params["std"] = float(getattr(cfg._cfg.reward_config, std_names[name]))
        if name in ("motion_global_root_pos", "motion_global_root_ori"):
            params["anchor_body_idx"] = anchor_body_idx
        if name == "motion_ee_body_pos_z":
            params["indices"] = ee_indices.tolist()
        if name == "undesired_contacts":
            params.update(
                indices=undesired_indices.tolist(), threshold=cfg._cfg.undesired_contact_z_threshold
            )
        output = f"reward.{name}"
        configs.append(
            TermConfig(
                output, key, scale=scale if available.get(name, True) else 0.0, parameters=params
            )
        )
        reward_outputs.append((name, output))

    termination_outputs = []
    termination_params = {
        "anchor_body_idx": anchor_body_idx,
        "anchor_pos_z_threshold": cfg._cfg.anchor_pos_z_threshold,
        "check_anchor_ori": cfg._cfg.anchor_ori_threshold < 2.0,
        "anchor_ori_threshold": cfg._cfg.anchor_ori_threshold,
        "ee_indices": ee_indices.tolist(),
        "ee_threshold": cfg._cfg.ee_body_pos_z_threshold,
        "contact_indices": undesired_indices.tolist()
        if cfg._cfg.terminate_on_undesired_contacts
        else [],
        "contact_threshold": cfg._cfg.undesired_contact_z_threshold,
    }
    for index, key in enumerate(config.terminations):
        name = f"termination.term{index}"
        configs.append(
            TermConfig(
                name, key, parameters=termination_params if key == DEFAULT_TERMINATIONS[0] else {}
            )
        )
        termination_outputs.append(name)

    if set(config.observations) != {"obs", "critic"}:
        raise TermPlanError("motion observation layout must define exactly 'obs' and 'critic'")
    observation_outputs, observation_labels = {}, {}
    for group in ("obs", "critic"):
        keys = config.observations[group]
        if not isinstance(keys, (list, tuple)) or not keys or len(keys) != len(set(keys)):
            raise TermPlanError(
                f"motion observation group {group!r} must be a non-empty unique list"
            )
        names, labels = [], []
        for index, key in enumerate(keys):
            meta = _OBS_META.get(key)
            if meta is None:
                registry.resolve(key)
            assert meta is not None
            if meta[0] != group:
                raise TermPlanError(f"observation term {key!r} does not belong to group {group!r}")
            name = f"observation.{group}.term{index}"
            configs.append(TermConfig(name, key))
            names.append(name)
            labels.append(meta[1])
        observation_outputs[group] = tuple(names)
        observation_labels[group] = tuple(labels)

    return MotionResolvedTermPlan(
        plan=resolve_term_plan(registry, configs),
        preamble_outputs=tuple(preamble_outputs),
        reward_outputs=tuple(reward_outputs),
        reward_available=MappingProxyType(available),
        observation_outputs=MappingProxyType(observation_outputs),
        observation_labels=MappingProxyType(observation_labels),
        termination_outputs=tuple(termination_outputs),
    )
