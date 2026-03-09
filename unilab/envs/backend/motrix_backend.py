import numpy as np
try:
    import motrixsim as mtx
    from motrixsim.render import RenderApp, RenderSettings
    MOTRIX_AVAILABLE = True
except ImportError:
    MOTRIX_AVAILABLE = False

from .base import ISimBackend


class MotrixBackend(ISimBackend):
    """MotrixSim 后端实现"""

    def __init__(self, model_file: str, num_envs: int, sim_dt: float, body_name: str = "base", np_dtype=np.float32):
        if not MOTRIX_AVAILABLE:
            raise ImportError("motrixsim not available")

        self._model = mtx.load_model(model_file)
        self._model.options.timestep = sim_dt
        self._num_envs = num_envs
        self._np_dtype = np_dtype

        self._data = mtx.SceneData(self._model, batch=[num_envs])
        self._body = self._model.get_body(body_name)
        self._render_app = None

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> None:
        self._data.actuator_ctrls[:] = ctrl
        for _ in range(nsteps):
            self._model.step(self._data)

    def get_dof_pos(self) -> np.ndarray:
        return self._body.get_joint_dof_pos(self._data)

    def get_dof_vel(self) -> np.ndarray:
        return self._body.get_joint_dof_vel(self._data)

    def get_qpos(self) -> np.ndarray:
        return self._data.dof_pos

    def get_sensor_data(self, name: str) -> np.ndarray:
        return self._model.get_sensor_value(name, self._data)

    def set_state(self, env_indices: np.ndarray, qpos: np.ndarray, qvel: np.ndarray) -> None:
        mask = np.zeros(self._num_envs, dtype=bool)
        mask[env_indices] = True
        data_slice = self._data[mask]
        data_slice.set_dof_pos(qpos, self._model)
        data_slice.set_dof_vel(qvel)
        self._model.forward_kinematic(data_slice)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    def init_renderer(self, spacing: float = 5.0):
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

    def render(self):
        """Render current state (interactive visualization)"""
        if self._render_app is None:
            self.init_renderer()
        self._render_app.sync(data=self._data)
