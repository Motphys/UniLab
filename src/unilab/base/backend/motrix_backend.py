import os

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
        add_body_sensors: bool = False,
    ):
        if not MOTRIX_AVAILABLE:
            raise ImportError("motrixsim not available")

        self.add_body_sensors = add_body_sensors

        if self.add_body_sensors:
            from unilab.utils.xml_utils import inject_motrix_tracking_sensors

            tmp_path, self._tracked_body_ids, valid_bnames = inject_motrix_tracking_sensors(
                model_file, baselink_name=body_name
            )
            self._model = mtx.load_model(tmp_path)  # pyright: ignore[reportPossiblyUnbound]
            os.remove(tmp_path)

            self._body_id_to_name: dict[int, str] = {
                bid: name for bid, name in zip(self._tracked_body_ids, valid_bnames)
            }
        else:
            self._model = mtx.load_model(model_file)  # pyright: ignore[reportPossiblyUnbound]

        self._model.options.timestep = sim_dt
        self._num_envs = num_envs
        self._np_dtype = np_dtype

        self._data = mtx.SceneData(self._model, batch=[num_envs])  # pyright: ignore[reportPossiblyUnbound]
        self._body = self._model.get_body(body_name)
        self._body_link = self._model.get_link(body_name)
        self._render_app: "RenderApp | None" = None
        self.backend_type = "motrix"

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    # ------------------------------------------------------------------ #
    # Simulation control                                                   #
    # ------------------------------------------------------------------ #

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> None:
        self._data.actuator_ctrls = np.ascontiguousarray(ctrl)
        for _ in range(nsteps):
            self._model.step(self._data)

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

    # ------------------------------------------------------------------ #
    # Base kinematics                                                      #
    # ------------------------------------------------------------------ #

    def get_qpos(self) -> np.ndarray:
        return np.asarray(self._data.dof_pos)

    def get_base_pos(self) -> np.ndarray:
        return np.asarray(self._body.floatingbase.get_translation(self._data))

    def get_base_quat(self) -> np.ndarray:
        return self._xyzw_to_wxyz(np.asarray(self._body.floatingbase.get_rotation(self._data)))

    def get_base_lin_vel(self) -> np.ndarray:
        return np.asarray(self._body.floatingbase.get_global_linear_velocity(self._data))

    def get_base_ang_vel(self) -> np.ndarray:
        return np.asarray(self._body.floatingbase.get_global_angular_velocity(self._data))

    # ------------------------------------------------------------------ #
    # DOF state                                                            #
    # ------------------------------------------------------------------ #

    def get_dof_pos(self) -> np.ndarray:
        return np.asarray(self._body.get_joint_dof_pos(self._data))

    def get_dof_vel(self) -> np.ndarray:
        return np.asarray(self._body.get_joint_dof_vel(self._data))

    # ------------------------------------------------------------------ #
    # Body kinematics — world frame                                        #
    # ------------------------------------------------------------------ #

    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.asarray(self._model.get_body(int(bid)).get_position(self._data)) for bid in body_ids],
            axis=1,
        )

    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        return np.stack(
            [self._xyzw_to_wxyz(np.asarray(self._model.get_body(int(bid)).get_rotation(self._data))) for bid in body_ids],
            axis=1,
        )

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.asarray(self._model.get_body(int(bid)).get_linear_velocity(self._data)) for bid in body_ids],
            axis=1,
        )

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.asarray(self._model.get_body(int(bid)).get_angular_velocity(self._data)) for bid in body_ids],
            axis=1,
        )

    # ------------------------------------------------------------------ #
    # Body kinematics — baselink frame                                     #
    # ------------------------------------------------------------------ #

    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.asarray(self._model.get_sensor_value(f"track_pos_b_{self._body_id_to_name[int(bid)]}", self._data)) for bid in body_ids],
            axis=1,
        )

    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        # framequat sensor 遵循 MuJoCo 约定，输出 wxyz
        return np.stack(
            [np.asarray(self._model.get_sensor_value(f"track_quat_b_{self._body_id_to_name[int(bid)]}", self._data)) for bid in body_ids],
            axis=1,
        )

    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.asarray(self._model.get_sensor_value(f"track_linvel_b_{self._body_id_to_name[int(bid)]}", self._data)) for bid in body_ids],
            axis=1,
        )

    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.asarray(self._model.get_sensor_value(f"track_angvel_b_{self._body_id_to_name[int(bid)]}", self._data)) for bid in body_ids],
            axis=1,
        )

    # ------------------------------------------------------------------ #
    # Sensors                                                              #
    # ------------------------------------------------------------------ #

    def get_sensor_data(self, name: str) -> np.ndarray:
        return np.asarray(self._model.get_sensor_value(name, self._data))

    # ------------------------------------------------------------------ #
    # MotrixSim-specific                                                   #
    # ------------------------------------------------------------------ #

    def _xyzw_to_wxyz(self, q: np.ndarray) -> np.ndarray:
        """motrix xyzw → wxyz"""
        return q[..., [3, 0, 1, 2]]

    def _process_rigid_body_props(self, cfg) -> None:
        if cfg.domain_rand.randomize_base_mass:
            mass = self._model.get_link(cfg.asset.body_name).get_mass_override(self._data)
            mass_low = cfg.domain_rand.added_mass_range[0]
            mass_high = cfg.domain_rand.added_mass_range[1]
            random_mass = mass + np.random.uniform(mass_low, mass_high, size=(self._num_envs,))
            self._model.get_link(cfg.asset.body_name).set_mass_override(self._data, random_mass)
            mass = self._model.get_link(cfg.asset.body_name).get_mass_override(self._data)

        if cfg.domain_rand.random_com:
            com_offset = np.zeros(
                (self._num_envs, 3), dtype=np.float32
            )  # [x_offset, y_offset, z_offset]
            x_low = cfg.domain_rand.com_offset_x[0]
            x_high = cfg.domain_rand.com_offset_x[1]
            com_offset[:, 0] = np.random.uniform(x_low, x_high, self._num_envs)
            self._model.get_link(cfg.asset.body_name).set_center_of_mass_override(
                self._data, com_offset
            )
            self._model.get_link(cfg.asset.body_name).get_center_of_mass_override(self._data)

    def push_robots(self, force_range):
        ex_force = np.random.rand(self.num_envs, 3) * 2 - 1  # [x_force, y_force, z_force]
        ex_force[:, 0] *= force_range[0]
        ex_force[:, 1] *= force_range[1]
        ex_force[:, 2] *= force_range[2]
        self._body_link.add_external_force(self._data, ex_force, local=True)

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
        assert self._render_app is not None
        self._render_app.sync(data=self._data)
