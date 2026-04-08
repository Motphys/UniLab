from .base import SimBackend
from .motrix_backend import MOTRIX_AVAILABLE, MotrixBackend
from .mujoco_backend import MuJoCoBackend


def create_backend(
    backend_type: str, model_file: str, num_envs: int, sim_dt: float, **kwargs
) -> SimBackend:
    """创建仿真后端

    Args:
        backend_type: "mujoco" 或 "motrix"
        model_file: 模型文件路径
        num_envs: 环境数量
        sim_dt: 仿真时间步长
        **kwargs: 其他参数

    Returns:
        SimBackend 实例
    """
    position_actuator_gains = kwargs.pop("position_actuator_gains", None)
    if backend_type == "mujoco":
        if position_actuator_gains is not None:
            kwargs["position_actuator_gains"] = position_actuator_gains
        return MuJoCoBackend(model_file, num_envs, sim_dt, **kwargs)
    elif backend_type == "motrix":
        if not MOTRIX_AVAILABLE:
            raise ImportError("MotrixSim not available, install motrixsim package")
        return MotrixBackend(model_file, num_envs, sim_dt, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend_type}")


__all__ = [
    "SimBackend",
    "MuJoCoBackend",
    "MotrixBackend",
    "create_backend",
]
