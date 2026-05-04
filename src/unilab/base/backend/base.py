import abc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from unilab.dr.types import (
    DomainRandomizationCapabilities,
    InitRandomizationPlan,
    IntervalRandomizationPlan,
    ResetRandomizationPayload,
)


@dataclass(frozen=True)
class BackendPlayCapabilities:
    """Backend-native play/render capabilities surfaced through env contracts."""

    supports_native_interactive_renderer: bool = False
    supports_physics_state_playback: bool = False


class SimBackend(abc.ABC):
    """仿真后端统一接口"""

    _model_file: str

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    @abc.abstractmethod
    def num_envs(self) -> int:
        """环境数量"""

    @property
    @abc.abstractmethod
    def model(self):
        """底层物理模型"""

    # ------------------------------------------------------------------ #
    # Model properties                                                     #
    # ------------------------------------------------------------------ #

    @property
    @abc.abstractmethod
    def num_actuators(self) -> int:
        """执行器数量"""

    @property
    @abc.abstractmethod
    def num_dof_vel(self) -> int:
        """关节速度自由度数量（不含浮动基座）"""

    @abc.abstractmethod
    def get_actuator_ctrl_range(self) -> np.ndarray:
        """获取执行器控制范围

        Returns:
            (num_actuators, 2) 数组，列为 [low, high]
        """

    @abc.abstractmethod
    def get_keyframe_qpos(self, name: str) -> np.ndarray:
        """获取指定关键帧的完整 qpos（含浮动基座）

        Args:
            name: 关键帧名称（如 "stand"、"home"）

        Returns:
            (nq,) 数组
        """

    def get_default_qpos(self) -> np.ndarray:
        """Return the backend/model default qpos through a stable contract."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose default qpos")

    @abc.abstractmethod
    def get_init_qvel(self) -> np.ndarray:
        """获取零初始化的 qvel 向量，维度与 set_state 期望一致

        Returns:
            全零数组
        """

    @abc.abstractmethod
    def get_body_ids(self, names: Sequence[str]) -> np.ndarray:
        """将 body/link 名称解析为后端整数 ID

        Args:
            names: body/link 名称序列

        Returns:
            (len(names),) int32 数组

        Raises:
            ValueError: 若名称未找到
        """

    def get_body_id(self, name: str) -> int:
        """Resolve one body/link name through the backend contract."""
        return int(self.get_body_ids([name])[0])

    def get_geom_id(self, name: str) -> int:
        """Resolve one geom name through the backend contract."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom ids")

    def get_geom_size(self, name: str) -> np.ndarray:
        """Return one geom size vector through the backend contract."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom sizes")

    def get_body_subtree_ids(self, root_body_id: int) -> np.ndarray:
        """Return body ids in the subtree rooted at ``root_body_id``."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose body subtree ids")

    def get_geom_names(self) -> tuple[str, ...]:
        """Return backend geom names in backend id order."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom names")

    def get_geom_body_ids(self) -> np.ndarray:
        """Return the owning body id for each geom."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom body ids")

    def get_geom_contact_masks(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-geom contact type and affinity masks."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom contact masks")

    def get_geom_friction(self) -> np.ndarray:
        """Return the backend geom-friction table."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom friction")

    def get_gravity(self) -> np.ndarray:
        """Return the backend gravity vector."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose gravity")

    def get_body_mass(self) -> np.ndarray:
        """Return the backend body-mass table."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose body mass")

    def get_body_ipos(self) -> np.ndarray:
        """Return the backend body inertial-position table."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose body ipos")

    def get_motion_body_ids(self, names: Sequence[str]) -> np.ndarray:
        """Resolve MuJoCo-style body IDs used by motion datasets."""
        from unilab.base.backend.xml import get_named_body_ids

        return np.asarray(get_named_body_ids(self._model_file, names), dtype=np.int32)

    @abc.abstractmethod
    def get_joint_range(self) -> np.ndarray | None:
        """获取关节位置限制（不含浮动基座）

        Returns:
            (num_dof, 2) 数组，列为 [low, high]；若后端不支持则返回 None
        """

    # ------------------------------------------------------------------ #
    # Simulation control                                                   #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict | None:
        """执行物理步进

        Args:
            ctrl: 控制输入 (num_envs, nu)
            nsteps: 步进次数

        Returns:
            可选的 dict，可包含 "timing" key 记录各阶段耗时（ms）
        """

    @abc.abstractmethod
    def set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: ResetRandomizationPayload | None = None,
    ) -> None:
        """设置指定环境的物理状态

        Args:
            env_indices: 环境索引
            qpos: 位置状态
            qvel: 速度状态
            randomization: 可选的后端随机化 payload
        """

    @abc.abstractmethod
    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        """Return supported domain-randomization capabilities for this backend."""

    def apply_init_randomization(self, plan: InitRandomizationPlan) -> None:
        """Apply cold-path model/materialization randomization."""
        if plan.is_empty():
            return
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support init-lifecycle randomization"
        )

    def materialize(self) -> None:
        """Finalize cold-path backend resources before reset/step."""

    @abc.abstractmethod
    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        """Apply a scheduled interval randomization plan."""

    def apply_body_linear_velocity_delta(
        self,
        body_ids: np.ndarray,
        velocity_delta: np.ndarray,
    ) -> None:
        """Apply a world-frame linear-velocity delta to specific bodies.

        Args:
            body_ids: Body ids whose linear velocities should be perturbed.
            velocity_delta: Velocity delta with shape ``(num_envs, len(body_ids), 3)``.

        Returns:
            None. Backends that support this mutate their pending simulation state.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support interval body velocity perturbation"
        )

    def apply_body_force(
        self,
        body_ids: np.ndarray,
        force: np.ndarray,
    ) -> None:
        """Apply a world-frame force to specific bodies for the upcoming step.

        Args:
            body_ids: Body ids whose external forces should be perturbed.
            force: Force values with shape ``(num_envs, len(body_ids), 3)``.

        Returns:
            None. Backends that support this mutate their pending simulation state.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support interval body force perturbation"
        )

    def get_play_capabilities(self) -> BackendPlayCapabilities:
        """Return backend-native play/render capabilities."""
        return BackendPlayCapabilities()

    def init_renderer(self, spacing: float = 1.0) -> None:
        """Initialize a backend-native interactive renderer."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support native interactive rendering"
        )

    def render(self) -> None:
        """Render one frame through a backend-native interactive renderer."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support native interactive rendering"
        )

    def get_physics_state(self) -> np.ndarray:
        """Return a physics snapshot suitable for offline playback/video export."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support physics-state playback"
        )

    def get_playback_model(self, env_index: int | None = None) -> Any:
        """Return the playback model for a specific env when variants exist.

        Args:
            env_index: Optional vectorized environment index.

        Returns:
            The backend model object used by playback tooling.
        """
        return self.model

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-joint (kp, kd) arrays from the backend model."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support reading actuator gains"
        )

    # ------------------------------------------------------------------ #
    # Base kinematics                                                      #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def get_base_pos(self) -> np.ndarray:
        """获取 base 在世界系下的位置

        Returns:
            (num_envs, 3)
        """

    @abc.abstractmethod
    def get_base_quat(self) -> np.ndarray:
        """获取 base 在世界系下的四元数（wxyz）

        Returns:
            (num_envs, 4)
        """

    @abc.abstractmethod
    def get_base_lin_vel(self) -> np.ndarray:
        """获取 base 在世界系下的线速度

        即广义速度 qvel 的前 3 维，表达在世界坐标系中。

        Returns:
            (num_envs, 3)
        """

    @abc.abstractmethod
    def get_base_ang_vel(self) -> np.ndarray:
        """获取 base 在世界系下的角速度

        即广义速度 qvel 的第 3-5 维，表达在世界坐标系中。
        注意与陀螺仪（gyro）读数的区别：陀螺仪返回的是角速度在 body/sensor
        局部坐标系下的分量（即 body frame 表达），而本接口返回的是世界系表达。
        若需要 body frame 下的角速度，请使用对应的传感器接口gyro。

        Returns:
            (num_envs, 3)
        """

    # ------------------------------------------------------------------ #
    # DOF state                                                            #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # Body kinematics — world frame                                        #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        """获取指定 body 在世界系下的位置

        Args:
            body_ids: body 索引数组

        Returns:
            (num_envs, len(body_ids), 3)
        """

    @abc.abstractmethod
    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        """获取指定 body 在世界系下的四元数（wxyz）

        Args:
            body_ids: body 索引数组

        Returns:
            (num_envs, len(body_ids), 4)
        """

    @abc.abstractmethod
    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        """获取指定 body 在世界系下的线速度

        Args:
            body_ids: body 索引数组

        Returns:
            (num_envs, len(body_ids), 3)
        """

    @abc.abstractmethod
    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        """获取指定 body 在世界系下的角速度

        Args:
            body_ids: body 索引数组

        Returns:
            (num_envs, len(body_ids), 3)
        """

    # ------------------------------------------------------------------ #
    # Body kinematics — baselink frame                                     #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        """获取指定 body 在 baselink 系下的位置

        Args:
            body_ids: body 索引数组

        Returns:
            (num_envs, len(body_ids), 3)
        """

    @abc.abstractmethod
    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        """获取指定 body 在 baselink 系下的四元数（wxyz）

        Args:
            body_ids: body 索引数组

        Returns:
            (num_envs, len(body_ids), 4)
        """

    @abc.abstractmethod
    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        """获取指定 body 在 baselink 系下的线速度

        Args:
            body_ids: body 索引数组

        Returns:
            (num_envs, len(body_ids), 3)
        """

    @abc.abstractmethod
    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        """获取指定 body 在 baselink 系下的角速度

        Args:
            body_ids: body 索引数组

        Returns:
            (num_envs, len(body_ids), 3)
        """

    # ------------------------------------------------------------------ #
    # Sensors                                                              #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def get_sensor_data(self, name: str) -> np.ndarray:
        """获取传感器数据

        Args:
            name: 传感器名称

        Returns:
            传感器数据数组
        """
