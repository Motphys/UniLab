from typing import Any, cast

from .base import SimBackend
from .motrix_backend import MOTRIX_AVAILABLE, MotrixBackend
from .xml import (
    add_sensor,
    create_discardvisual_xml,
    create_motrix_compatible_xml,
    get_named_body_ids,
    inject_motrix_tracking_sensors,
    inject_mujoco_tracking_sensors,
    materialize_scene_visual_override,
    processed_xml,
)


def _load_mujoco_backend() -> Any:
    from .mujoco_backend import MuJoCoBackend

    return MuJoCoBackend


def create_backend(
    backend_type: str, model_file: str, num_envs: int, sim_dt: float, **kwargs
) -> SimBackend:
    """创建仿真后端

    Args:
        backend_type: "mujoco" 或 "motrix"
        model_file: 模型文件路径
        num_envs: 环境数量
        sim_dt: 仿真时间步长
        **kwargs: 其他参数（iterations, position_actuator_gains 等）

    Returns:
        SimBackend 实例
    """
    position_actuator_gains = kwargs.pop("position_actuator_gains", None)
    if backend_type == "mujoco":
        MuJoCoBackend = _load_mujoco_backend()
        if position_actuator_gains is not None:
            kwargs["position_actuator_gains"] = position_actuator_gains
        return cast(SimBackend, MuJoCoBackend(model_file, num_envs, sim_dt, **kwargs))
    elif backend_type == "motrix":
        if not MOTRIX_AVAILABLE:
            raise ImportError("MotrixSim not available, install motrixsim package")
        return MotrixBackend(model_file, num_envs, sim_dt, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend_type}")


def __getattr__(name: str):
    if name == "MuJoCoBackend":
        return _load_mujoco_backend()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SimBackend",
    "MuJoCoBackend",
    "MotrixBackend",
    "add_sensor",
    "create_discardvisual_xml",
    "create_motrix_compatible_xml",
    "create_backend",
    "get_named_body_ids",
    "inject_motrix_tracking_sensors",
    "inject_mujoco_tracking_sensors",
    "materialize_scene_visual_override",
    "processed_xml",
]
