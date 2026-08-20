from __future__ import annotations

from dataclasses import dataclass

from unilab.envs import ManagerBasedRlEnvCfg
from unilab.tasks.manipulation.sharpa_inhand.rotation import SharpaInhandRotationCfg


@dataclass(frozen=True)
class LocomotionTaskSpec:
    owner_task_id: str
    env_task_name: str
    display_name: str
    config_cls: type
    model_file: str | None = None


_TASK_SPECS = {
    "go1_joystick_flat": LocomotionTaskSpec(
        owner_task_id="go1_joystick_flat",
        env_task_name="Go1JoystickFlat",
        display_name="go1_joystick_flat",
        config_cls=ManagerBasedRlEnvCfg,
        model_file="src/unilab/assets/robots/go1/scene_flat.xml",
    ),
    "go2_joystick_flat": LocomotionTaskSpec(
        owner_task_id="go2_joystick_flat",
        env_task_name="Go2JoystickFlat",
        display_name="go2_joystick_flat",
        config_cls=ManagerBasedRlEnvCfg,
        model_file="src/unilab/assets/robots/go2/scene_flat.xml",
    ),
    "g1_walk_flat": LocomotionTaskSpec(
        owner_task_id="g1_walk_flat",
        env_task_name="G1WalkFlat",
        display_name="g1_walk_flat",
        config_cls=ManagerBasedRlEnvCfg,
        model_file="src/unilab/assets/robots/g1/scene_flat.xml",
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


def locomotion_task_model_file(task_name: str) -> str:
    spec = locomotion_task_spec(task_name)
    if spec.model_file is not None:
        return spec.model_file
    cfg = spec.config_cls()
    scene = getattr(cfg, "scene", None)
    model_file = getattr(scene, "model_file", None)
    if model_file:
        return str(model_file)

    raise ValueError(f"{type(cfg).__name__} does not define scene.model_file")


def locomotion_env_name(task_name: str) -> str:
    return locomotion_task_spec(task_name).env_task_name
