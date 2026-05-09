from typing import Any, cast

from unilab.base.scene import SceneCfg

from .base import SimBackend
from .motrix_scene import (
    add_motrix_tracking_frame_sensors,
    materialize_motrix_hfield_attached_scene,
    materialize_motrix_scene,
)
from .xml import (
    add_sensor,
    create_discardvisual_xml,
    get_named_body_ids,
    inject_mujoco_tracking_sensors,
    materialize_mujoco_hfield_attached_scene,
    materialize_scene_fragments,
    materialize_scene_visual_override,
    processed_xml,
)


def _load_mujoco_backend() -> Any:
    from .mujoco_backend import MuJoCoBackend

    return MuJoCoBackend


def _load_motrix_backend() -> tuple[Any, bool]:
    from .motrix_backend import MOTRIX_AVAILABLE, MotrixBackend

    return MotrixBackend, bool(MOTRIX_AVAILABLE)


def create_backend(
    backend_type: str,
    scene: SceneCfg,
    num_envs: int,
    sim_dt: float,
    **kwargs,
) -> SimBackend:
    """创建仿真后端

    Args:
        backend_type: "mujoco" 或 "motrix"
        scene: SceneCfg，静态 scene 或组合式 scene 都通过这个 contract 表达
        num_envs: 环境数量
        sim_dt: 仿真时间步长
        **kwargs: 其他参数（position_actuator_gains, motrix_max_iterations 等）

    Returns:
        SimBackend 实例
    """
    if scene is None:
        raise ValueError("SceneCfg must be provided")

    position_actuator_gains = kwargs.pop("position_actuator_gains", None)
    motrix_max_iterations = kwargs.pop("motrix_max_iterations", None)
    if backend_type == "mujoco":
        MuJoCoBackend = _load_mujoco_backend()
        if position_actuator_gains is not None:
            kwargs["position_actuator_gains"] = position_actuator_gains
        return cast(SimBackend, MuJoCoBackend(scene, num_envs, sim_dt, **kwargs))
    if backend_type == "motrix":
        MotrixBackend, motrix_available = _load_motrix_backend()
        if not motrix_available:
            raise ImportError("MotrixSim not available, install motrixsim package")
        if motrix_max_iterations is not None:
            kwargs["max_iterations"] = motrix_max_iterations
        return cast(SimBackend, MotrixBackend(scene, num_envs, sim_dt, **kwargs))
    raise ValueError(f"Unknown backend: {backend_type}")


def __getattr__(name: str):
    if name == "MuJoCoBackend":
        return _load_mujoco_backend()
    if name == "MotrixBackend":
        return _load_motrix_backend()[0]
    if name == "MOTRIX_AVAILABLE":
        return _load_motrix_backend()[1]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SimBackend",
    "MuJoCoBackend",
    "MotrixBackend",
    "add_sensor",
    "create_discardvisual_xml",
    "create_backend",
    "get_named_body_ids",
    "inject_mujoco_tracking_sensors",
    "add_motrix_tracking_frame_sensors",
    "materialize_motrix_hfield_attached_scene",
    "materialize_motrix_scene",
    "materialize_mujoco_hfield_attached_scene",
    "materialize_scene_fragments",
    "materialize_scene_visual_override",
    "processed_xml",
]
