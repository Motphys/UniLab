import abc

import numpy as np


class SimBackend(abc.ABC):
    """仿真后端统一接口"""

    @abc.abstractmethod
    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> None:
        """执行物理步进

        Args:
            ctrl: 控制输入 (num_envs, nu)
            nsteps: 步进次数
        """

    @abc.abstractmethod
    def get_dof_pos(self) -> np.ndarray:
        """获取关节位置（不含 base）

        Returns:
            (num_envs, num_dof)
        """

    @abc.abstractmethod
    def get_dof_vel(self) -> np.ndarray:
        """获取关节速度（不含 base）

        Returns:
            (num_envs, num_dof)
        """

    @abc.abstractmethod
    def get_qpos(self) -> np.ndarray:
        """获取完整位置状态（含 base）

        Returns:
            (num_envs, nq)
        """

    @abc.abstractmethod
    def get_sensor_data(self, name: str) -> np.ndarray:
        """获取传感器数据

        Args:
            name: 传感器名称

        Returns:
            传感器数据数组
        """

    @abc.abstractmethod
    def set_state(self, env_indices: np.ndarray, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """设置指定环境的物理状态

        Args:
            env_indices: 环境索引
            qpos: 位置状态
            qvel: 速度状态
        """

    @property
    @abc.abstractmethod
    def num_envs(self) -> int:
        """环境数量"""

    @property
    @abc.abstractmethod
    def model(self):
        """底层物理模型"""
