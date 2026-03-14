import numpy as np

try:
    import motrixsim as mtx
    from motrixsim.render import RenderApp, RenderSettings

    MOTRIX_AVAILABLE = True
except ImportError:
    MOTRIX_AVAILABLE = False

from .base import SimBackend


class MotrixBackend(SimBackend):
    """MotrixSim 后端实现"""

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
        self._body_link = self._model.get_link(body_name)
        self._render_app = None
        self.backend_type = "motrix"

    def _process_rigid_body_props(self, cfg) -> None:
        if cfg.domain_rand.randomize_base_mass == True:
            mass = self._model.get_link(cfg.asset.body_name).get_mass_override(self._data)
            mass_low = cfg.domain_rand.added_mass_range[0]
            mass_high = cfg.domain_rand.added_mass_range[1]
            random_mass = mass + np.random.uniform(mass_low, mass_high, size=(self._num_envs,))
            self._model.get_link(cfg.asset.body_name).set_mass_override(self._data, random_mass)
            mass = self._model.get_link(cfg.asset.body_name).get_mass_override(self._data)

        if cfg.domain_rand.random_com == True:
            com_offset = np.zeros(
                (self._num_envs, 3), dtype=np.float32
            )  # [x_offset, y_offset, z_offset]
            x_low = cfg.domain_rand.com_offset_x[0]
            x_high = cfg.domain_rand.com_offset_x[1]
            com_offset[:, 0] = np.random.uniform(x_low, x_high, self._num_envs)
            self._model.get_link(cfg.asset.body_name).set_center_of_mass_override(
                self._data, com_offset
            )
            com_get = self._model.get_link(cfg.asset.body_name).get_center_of_mass_override(
                self._data
            )

    def push_robots(self, force_range):
        ex_force = np.random.rand(self.num_envs, 3) * 2 - 1  # [x_force, y_force, z_force]
        ex_force[:, 0] *= force_range[0]
        ex_force[:, 1] *= force_range[1]
        ex_force[:, 2] *= force_range[2]
        self._body_link.add_external_force(self._data, ex_force, local=True)

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> None:
        self._data.actuator_ctrls = np.ascontiguousarray(ctrl)
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
        # Convert quaternion from mujoco (wxyz) to motrix (xyzw)
        qpos_motrix = qpos.copy()
        qpos_motrix[:, 3:7] = qpos[:, [4, 5, 6, 3]]

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
        data_slice.actuator_ctrls = np.ascontiguousarray(ctrl)

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

    def render(self):
        """Render current state (interactive visualization)"""
        if self._render_app is None:
            self.init_renderer()
        self._render_app.sync(data=self._data)
