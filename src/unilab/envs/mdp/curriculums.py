# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), src/mjlab/envs/mdp/curriculums.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and UniLab contracts; licensed under Apache-2.0.
"""Generic stage-based curriculum terms for the NumPy manager runtime.

These terms let owner YAMLs ramp any reward/termination term's ``weight``
and/or ``params`` by training step (``env.common_step_counter``) through a
declarative stage table, so tasks no longer need private step-based
curriculum terms. Stage scheduling is validated fail-closed at manager
construction time.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv
    from unilab.managers.curriculum_manager import CurriculumTermCfg


# Stage schemas.


class _RewardCurriculumStageOptional(TypedDict, total=False):
    weight: float
    params: dict[str, Any]


class RewardCurriculumStage(_RewardCurriculumStageOptional):
    step: int


class _TerminationCurriculumStageOptional(TypedDict, total=False):
    params: dict[str, Any]
    time_out: bool


class TerminationCurriculumStage(_TerminationCurriculumStageOptional):
    step: int


class _CommandCurriculumStageOptional(TypedDict, total=False):
    params: dict[str, Any]


class CommandCurriculumStage(_CommandCurriculumStageOptional):
    """Stage for ``command_curriculum``.

    Any key beyond ``step``/``params`` is applied as a top-level field on the
    live command term config (e.g. ``rel_standing_envs`` or ``ranges``); the
    set of valid fields is the target term's config schema.
    """

    step: int


class _EventCurriculumStageOptional(TypedDict, total=False):
    params: dict[str, Any]


class EventCurriculumStage(_EventCurriculumStageOptional):
    """Stage for ``event_curriculum``.

    Any key beyond ``step``/``params`` is applied as a top-level field on the
    live event term config; the set of valid fields is the target term's
    config schema.
    """

    step: int


# Shared engine. Stage dicts are passed directly from the public TypedDict
# schemas. Any key that isn't "step" or "params" is treated as a top-level
# field on the target term config (e.g. "weight" on RewardTermCfg).

_RESERVED_KEYS = {"step", "params"}


def _validate_stages(
    term_cfg: Any,
    term_name: str,
    stages: Sequence[Any],
) -> None:
    """Validate stage ordering, field existence, and param keys."""
    for i in range(1, len(stages)):
        if stages[i]["step"] < stages[i - 1]["step"]:
            raise ValueError(
                f"Curriculum stages must be in nondecreasing step order,"
                f" but stage {i} has step"
                f" {stages[i]['step']} < {stages[i - 1]['step']}."
            )
    for stage in stages:
        for key in stage:
            if key not in _RESERVED_KEYS and not hasattr(term_cfg, key):
                raise AttributeError(
                    f"Field '{key}' does not exist on the resolved term config for '{term_name}'."
                )
    # Command term configs carry plain dataclass fields and no params dict.
    term_params = getattr(term_cfg, "params", None)
    for stage in stages:
        known = term_params.keys() if term_params is not None else ()
        unknown = stage.get("params", {}).keys() - known
        if unknown:
            raise KeyError(
                f"Stage at step {stage['step']} sets unknown param(s)"
                f" {unknown} on term '{term_name}'. Check for typos."
            )


def _apply_stages(
    term_cfg: Any,
    step_counter: int,
    stages: Sequence[Any],
) -> dict[str, Any]:
    """Apply staged updates and return a logging snapshot."""
    for stage in stages:
        if step_counter >= stage["step"]:
            for key, value in stage.items():
                if key not in _RESERVED_KEYS:
                    setattr(term_cfg, key, value)
            if "params" in stage:
                term_cfg.params.update(stage["params"])
    # Only log values that stages actually reference.
    logged_fields: set[str] = set()
    logged_params: set[str] = set()
    for stage in stages:
        for key in stage:
            if key not in _RESERVED_KEYS:
                logged_fields.add(key)
        for key in stage.get("params", {}):
            logged_params.add(key)
    result: dict[str, Any] = {}
    for key in logged_fields:
        value = getattr(term_cfg, key)
        if isinstance(value, (int, float, bool, np.number)):
            result[key] = value
    term_params = getattr(term_cfg, "params", {})
    for key in logged_params:
        value = term_params[key]
        if isinstance(value, (int, float, bool, np.number)):
            result[key] = value
    return result


# Public wrappers.


class reward_curriculum:
    """Update a reward term's weight and/or params based on training steps.

    Each stage specifies a ``step`` threshold and optionally a ``weight``
    and/or ``params`` dict. When ``env.common_step_counter`` reaches a
    stage's ``step``, the corresponding values are applied. Later stages
    take precedence when multiple thresholds are reached.

    Example owner YAML::

      curriculum:
        action_rate_ramp:
          func: unilab.envs.mdp.reward_curriculum
          params:
            reward_name: action_rate
            stages:
              - {step: 0, weight: -0.1}
              - {step: 12000, weight: -0.4}
              - {step: 24000, weight: -1.0, params: {max_vel: 1.0}}
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        reward_name: str = cfg.params["reward_name"]
        stages: list[RewardCurriculumStage] = cfg.params["stages"]
        self._term_cfg = env.reward_manager.get_term_cfg(reward_name)
        self._stages = stages
        _validate_stages(self._term_cfg, reward_name, self._stages)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | slice,
        reward_name: str,
        stages: list[RewardCurriculumStage],
    ) -> dict[str, Any]:
        del env_ids, reward_name, stages
        return _apply_stages(self._term_cfg, env.common_step_counter, self._stages)


class termination_curriculum:
    """Update a termination term's params and/or time_out based on training steps.

    Each stage specifies a ``step`` threshold and optionally a ``params``
    dict and/or ``time_out`` flag. When ``env.common_step_counter`` reaches
    a stage's ``step``, the values are applied. Later stages take precedence.

    Example owner YAML::

      curriculum:
        tilt_threshold:
          func: unilab.envs.mdp.termination_curriculum
          params:
            termination_name: tilt
            stages:
              - {step: 12000, params: {max_tilt_deg: 80.0}}
              - {step: 24000, params: {max_tilt_deg: 65.0}}
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        termination_name: str = cfg.params["termination_name"]
        stages: list[TerminationCurriculumStage] = cfg.params["stages"]
        self._term_cfg = env.termination_manager.get_term_cfg(termination_name)
        self._stages = stages
        _validate_stages(self._term_cfg, termination_name, self._stages)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | slice,
        termination_name: str,
        stages: list[TerminationCurriculumStage],
    ) -> dict[str, Any]:
        del env_ids, termination_name, stages
        return _apply_stages(self._term_cfg, env.common_step_counter, self._stages)


class command_curriculum:
    """Update a command term's config fields and/or params based on training steps.

    Command terms read their live ``self.cfg`` at resample time, so mutating
    the resolved term config (e.g. ``rel_standing_envs`` or ``ranges``) takes
    effect on the next command resample. Stage semantics match
    ``reward_curriculum``: every stage whose ``step`` has been reached is
    applied in order, so later stages win.

    Example owner YAML::

      curriculum:
        standing_envs:
          func: unilab.envs.mdp.command_curriculum
          params:
            command_name: twist
            stages:
              - {step: 0, rel_standing_envs: 0.02}
              - {step: 12000, rel_standing_envs: 0.1}
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        command_name: str = cfg.params["command_name"]
        stages: list[CommandCurriculumStage] = cfg.params["stages"]
        self._term_cfg = env.command_manager.get_term_cfg(command_name)
        if self._term_cfg is None:
            raise ValueError(f"Command term '{command_name}' not found in active terms.")
        self._stages = stages
        _validate_stages(self._term_cfg, command_name, self._stages)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | slice,
        command_name: str,
        stages: list[CommandCurriculumStage],
    ) -> dict[str, Any]:
        del env_ids, command_name, stages
        return _apply_stages(self._term_cfg, env.common_step_counter, self._stages)


class event_curriculum:
    """Update an event term's params and/or config fields based on training steps.

    Event terms are invoked with their live ``cfg.params`` on every apply, so
    staged ``params`` updates (e.g. a widened ``com_range``) take effect on the
    next event application. Stage semantics match ``reward_curriculum``.

    Example owner YAML::

      curriculum:
        com_range:
          func: unilab.envs.mdp.event_curriculum
          params:
            event_name: base_com
            stages:
              - {step: 0, params: {com_range: {x: [-0.003, 0.003]}}}
              - {step: 24000, params: {com_range: {x: [-0.01, 0.01]}}}
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        event_name: str = cfg.params["event_name"]
        stages: list[EventCurriculumStage] = cfg.params["stages"]
        self._term_cfg = env.event_manager.get_term_cfg(event_name)
        if self._term_cfg is None:
            raise ValueError(f"Event term '{event_name}' not found in active terms.")
        self._stages = stages
        _validate_stages(self._term_cfg, event_name, self._stages)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | slice,
        event_name: str,
        stages: list[EventCurriculumStage],
    ) -> dict[str, Any]:
        del env_ids, event_name, stages
        return _apply_stages(self._term_cfg, env.common_step_counter, self._stages)
