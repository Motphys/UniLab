from __future__ import annotations

from dataclasses import dataclass

from unilab.envs.locomotion.g1.joystick import G1JoystickPPOCfg
from unilab.envs.locomotion.go1.joystick import Go1JoystickCfg
from unilab.envs.locomotion.go2.joystick import Go2JoystickCfg
from unilab.envs.manipulation.sharpa_inhand.rotation import SharpaInhandRotationCfg


@dataclass(frozen=True)
class LocomotionTaskSpec:
    owner_task_id: str
    env_task_name: str
    display_name: str
    config_cls: type


_TASK_SPECS = {
    "go1_joystick_flat": LocomotionTaskSpec(
        owner_task_id="go1_joystick_flat",
        env_task_name="Go1JoystickFlat",
        display_name="go1_joystick_flat",
        config_cls=Go1JoystickCfg,
    ),
    "go2_joystick_flat": LocomotionTaskSpec(
        owner_task_id="go2_joystick_flat",
        env_task_name="Go2JoystickFlat",
        display_name="go2_joystick_flat",
        config_cls=Go2JoystickCfg,
    ),
    "g1_joystick_flat": LocomotionTaskSpec(
        owner_task_id="g1_joystick_flat",
        env_task_name="G1JoystickFlat",
        display_name="g1_joystick_flat",
        config_cls=G1JoystickPPOCfg,
    ),
    "sharpa_inhand": LocomotionTaskSpec(
        owner_task_id="sharpa_inhand",
        env_task_name="SharpaInhandRotation",
        display_name="sharpa_inhand",
        config_cls=SharpaInhandRotationCfg,
    ),
}
_TASK_ALIASES = {spec.env_task_name: spec.owner_task_id for spec in _TASK_SPECS.values()}
_TASK_ALIASES.update({f"task={task_id}/mujoco": task_id for task_id in _TASK_SPECS})
_TASK_ALIASES.update({f"{task_id}/mujoco": task_id for task_id in _TASK_SPECS})


def canonical_locomotion_task_ids() -> list[str]:
    return list(_TASK_SPECS.keys())


def normalize_locomotion_task_id(task_name: str) -> str:
    normalized = task_name.strip()
    if normalized.startswith("task="):
        normalized = normalized[len("task=") :]
    if normalized.endswith("/motrix"):
        raise ValueError(
            f"Task '{task_name}' targets motrix, but this benchmark only measures MuJoCo paths."
        )
    if normalized in _TASK_SPECS:
        return normalized
    alias_target = _TASK_ALIASES.get(normalized)
    if alias_target is not None:
        return alias_target
    raise ValueError(
        f"Unknown task '{task_name}'. Available task ids: {canonical_locomotion_task_ids()}. "
        "Accepted aliases also include the legacy env names and task=<name>/mujoco forms."
    )


def locomotion_task_spec(task_name: str) -> LocomotionTaskSpec:
    return _TASK_SPECS[normalize_locomotion_task_id(task_name)]


def locomotion_env_name(task_name: str) -> str:
    return locomotion_task_spec(task_name).env_task_name
