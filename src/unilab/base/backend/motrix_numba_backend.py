import numpy as np

try:
    import motrixsim as mtx
    from motrixsim.render import RenderApp, RenderSettings

    MOTRIX_AVAILABLE = True
except ImportError:
    MOTRIX_AVAILABLE = False

from .base import SimBackend

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# Numba 加速的内部函数
if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _convert_quaternion_wxyz_to_xyzw(qpos):
        """四元数格式转换: MuJoCo (wxyz) -> Motrix (xyzw)"""
        result = qpos.copy()
        n_envs = qpos.shape[0]
        for i in range(n_envs):
            result[i, 3] = qpos[i, 4]
            result[i, 4] = qpos[i, 5]
            result[i, 5] = qpos[i, 6]
            result[i, 6] = qpos[i, 3]
        return result

    @njit(cache=True)
    def _make_contiguous(ctrl):
        """确保数组内存连续"""
        n_envs = ctrl.shape[0]
        n_ctrls = ctrl.shape[1]
        result = np.empty((n_envs, n_ctrls), dtype=ctrl.dtype)
        for i in range(n_envs):
            for j in range(n_ctrls):
                result[i, j] = ctrl[i, j]
        return result
else:

    def _convert_quaternion_wxyz_to_xyzw(qpos):
        """NumPy fallback"""
        qpos_motrix = qpos.copy()
        qpos_motrix[:, 3:7] = qpos[:, [4, 5, 6, 3]]
        return qpos_motrix

    def _make_contiguous(ctrl):
        """NumPy fallback"""
        return np.ascontiguousarray(ctrl)


class MotrixNumbaBackend(SimBackend):
    """MotrixSim 后端实现 (Numba 优化版本)"""

    def __init__(
        self,
        model_file: str,
        num_envs: int,
        sim_dt: float,
        body_name: str = "base",
        np_dtype=np.float32,
    ):
        if not MOTRIX_AVAILABLE:
            raise ImportError("motrixsim not available")

        self._model = mtx.load_model(model_file)
        self._model.options.timestep = sim_dt
        self._num_envs = num_envs
        self._np_dtype = np_dtype

        self._data = mtx.SceneData(self._model, batch=[num_envs])
        self._body = self._model.get_body(body_name)
        self._render_app: "RenderApp | None" = None

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> None:
        self._data.actuator_ctrls = _make_contiguous(ctrl)
        for _ in range(nsteps):
            self._model.step(self._data)

    def get_dof_pos(self) -> np.ndarray:
        return np.asarray(self._body.get_joint_dof_pos(self._data))

    def get_dof_vel(self) -> np.ndarray:
        return np.asarray(self._body.get_joint_dof_vel(self._data))

    def get_qpos(self) -> np.ndarray:
        return np.asarray(self._data.dof_pos)

    def get_sensor_data(self, name: str) -> np.ndarray:
        return np.asarray(self._model.get_sensor_value(name, self._data))

    def set_state(self, env_indices: np.ndarray, qpos: np.ndarray, qvel: np.ndarray) -> None:
        # Convert quaternion from mujoco (wxyz) to motrix (xyzw)
        qpos_motrix = _convert_quaternion_wxyz_to_xyzw(qpos)

        # Create mask for batch operation
        mask = np.zeros(self._num_envs, dtype=bool)
        mask[env_indices] = True
        data_slice = self._data[mask]

        # Batch set state
        data_slice.reset(self._model)
        data_slice.set_dof_vel(qvel)
        data_slice.set_dof_pos(qpos_motrix, self._model)

        # Set control to joint positions (actuator target for PD control)
        ctrl = qpos_motrix[:, 7:]
        data_slice.actuator_ctrls = _make_contiguous(ctrl)

        self._model.forward_kinematic(self._data)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    def init_renderer(self, spacing: float = 1.0):
        """Initialize interactive renderer for visualization"""
        if self._render_app is not None:
            return

        cols = int(np.ceil(np.sqrt(self._num_envs)))
        offsets = []
        for i in range(self._num_envs):
            row = i // cols
            col = i % cols
            offsets.append([col * spacing, row * spacing, 0.0])

        self._render_app = RenderApp()
        settings = RenderSettings.performance()
        settings.enable_shadow = True
        self._render_app.launch(
            self._model,
            batch=self._num_envs,
            render_offset=offsets,
            render_settings=settings,
        )

    def get_base_pos(self) -> np.ndarray:
        return np.asarray(self._body.floatingbase.get_translation(self._data))

    def get_base_quat(self) -> np.ndarray:
        xyzw = np.asarray(self._body.floatingbase.get_rotation(self._data))
        return xyzw[..., [3, 0, 1, 2]]

    def get_base_lin_vel(self) -> np.ndarray:
        return np.asarray(self._body.floatingbase.get_global_linear_velocity(self._data))

    def get_base_ang_vel(self) -> np.ndarray:
        return np.asarray(self._body.floatingbase.get_global_angular_velocity(self._data))

    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                np.asarray(self._model.get_link(int(bid)).get_position(self._data))
                for bid in body_ids
            ],
            axis=1,
        )

    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                np.asarray(self._model.get_link(int(bid)).get_rotation(self._data))[
                    ..., [3, 0, 1, 2]
                ]
                for bid in body_ids
            ],
            axis=1,
        )

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                np.asarray(self._model.get_link(int(bid)).get_linear_velocity(self._data))
                for bid in body_ids
            ],
            axis=1,
        )

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                np.asarray(self._model.get_link(int(bid)).get_angular_velocity(self._data))
                for bid in body_ids
            ],
            axis=1,
        )

    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError("MotrixNumbaBackend does not support baselink-frame body queries")

    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError("MotrixNumbaBackend does not support baselink-frame body queries")

    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError("MotrixNumbaBackend does not support baselink-frame body queries")

    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError("MotrixNumbaBackend does not support baselink-frame body queries")

    def render(self):
        """Render current state (interactive visualization)"""
        if self._render_app is None:
            self.init_renderer()
        assert self._render_app is not None
        self._render_app.sync(data=self._data)
