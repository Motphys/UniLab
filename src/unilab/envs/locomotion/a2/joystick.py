"""A2 joystick task (leg-only Unitree A2).

The A2 leg-only MJCF (robots/a2/scene_flat.xml) mirrors the Go2 joystick
sensor/geom/leg-ordering contract and uses <position> actuators, so this
task reuses Go2WalkTask unchanged. Only the A2 identity differs: scene path,
standing height, and PD gains (A2 legs are stronger than Go2's, so a single
scalar Kp/Kd is raised vs Go2 and applied to all 12 leg actuators at init via
position_actuator_gains)."""

from __future__ import annotations

from dataclasses import dataclass, field

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.go2.base import Asset, ControlConfig
from unilab.envs.locomotion.go2.joystick import (
    Go2JoystickCfg,
    Go2WalkTask,
)


@dataclass
class A2InitState:
    pos = [0.0, 0.0, 0.465]


@dataclass
class A2Asset(Asset):
    # The A2 base body is named "base_link" in its MJCF, whereas Go2 uses "base".
    base_name: str = "base_link"  # type: ignore[assignment]


@dataclass
class A2JoystickControlConfig(ControlConfig):
    # A2 legs are far stronger than Go2's, so raise the scalar PD gains. The env
    # forwards these to the backend as position_actuator_gains, which applies
    # them uniformly to all 12 leg actuators (overriding any per-class kp in
    # a2.xml). Keep a2.xml's <position> kp consistent with this scalar.
    Kp: float = 100.0
    Kd: float = 4.0


def _a2_scene() -> SceneCfg:
    return SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots" / "a2" / "scene_flat.xml"))


@registry.envcfg("A2JoystickFlat")
@dataclass
class A2JoystickCfg(Go2JoystickCfg):
    scene: SceneCfg = field(default_factory=_a2_scene)
    init_state: A2InitState = field(default_factory=A2InitState)  # type: ignore[assignment]
    asset: A2Asset = field(default_factory=A2Asset)  # type: ignore[assignment]
    control_config: A2JoystickControlConfig = field(  # type: ignore[assignment]
        default_factory=A2JoystickControlConfig
    )


@registry.env("A2JoystickFlat", sim_backend="mujoco")
class A2JoystickFlatEnv(Go2WalkTask):
    """Leg-only A2 joystick task. Identical logic to Go2WalkTask; only the
    config (asset path, standing height, gains) differs."""

    _cfg: A2JoystickCfg
