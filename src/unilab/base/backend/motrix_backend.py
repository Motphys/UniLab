import os
from collections.abc import Sequence

import numpy as np

from unilab.dr.types import (
    RESET_TERM_BASE_COM,
    RESET_TERM_BASE_MASS,
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    ResetRandomizationPayload,
)

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
        base_name: str = "base",
        np_dtype=np.float32,
        add_body_sensors: bool = False,
    ):
        if not MOTRIX_AVAILABLE:
            raise ImportError("motrixsim not available")

        self.add_body_sensors = add_body_sensors
        self._base_name = base_name

        if self.add_body_sensors:
            from unilab.utils.xml_utils import inject_motrix_tracking_sensors

            tmp_path, _, valid_bnames = inject_motrix_tracking_sensors(
                model_file, baselink_name=base_name
            )
            try:
                self._model = mtx.load_model(tmp_path)  # pyright: ignore[reportPossiblyUnbound]
            finally:
                os.remove(tmp_path)

            # 用 motrixsim link index 作 key（Link 覆盖所有关节体，Body 只有 freejoint 根体）
            self._body_id_to_name: dict[int, str] = {
                idx: name
                for name in valid_bnames
                if (idx := self._model.get_link_index(name)) is not None
            }
        else:
            self._model = mtx.load_model(model_file)  # pyright: ignore[reportPossiblyUnbound]

            # 枚举所有具名 link，用 link index 作 key
            self._body_id_to_name = {  # type: ignore[assignment]
                link.index: link.name for link in self._model.links if link.name
            }

        self._model.options.timestep = sim_dt
        self._num_envs = num_envs
        self._np_dtype = np_dtype

        self._data = mtx.SceneData(self._model, batch=[num_envs])  # pyright: ignore[reportPossiblyUnbound]
        self._body = self._model.get_body(base_name)
        self._body_link = self._model.get_link(base_name)
        self._default_base_mass_override = np.asarray(
            self._body_link.get_mass_override(self._data)
        ).copy()
        self._default_base_com_override = np.asarray(
            self._body_link.get_center_of_mass_override(self._data)
        ).copy()
        self._render_app: "RenderApp | None" = None
        self.backend_type = "motrix"

        # Pre-cache link objects to avoid repeated get_link() lookups.
        self._link_cache: dict[int, "mtx.Link"] = {}
        for link in self._model.links:
            if link.name:
                self._link_cache[link.index] = link

        # 运行一次正向运动学，确保初始 link 位置和传感器数据有效。
        self._model.forward_kinematic(self._data)
        self._refresh_link_pose_cache()

    # ------------------------------------------------------------------ #
    # Properties                                                         #
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
    # Model properties                                                   #
    # ------------------------------------------------------------------ #

    @property
    def num_actuators(self) -> int:
        return int(self._model.num_actuators)

    @property
    def num_dof_vel(self) -> int:
        # model.num_dof_vel includes 6 floating-base DOFs; exclude them
        # to match get_dof_vel() which returns joint velocities only.
        return int(self._model.num_dof_vel) - 6

    def get_actuator_ctrl_range(self) -> np.ndarray:
        arr: np.ndarray = np.array(self._model.actuator_ctrl_limits, dtype=self._np_dtype)
        result: np.ndarray = arr.T.copy()
        return result

    def get_keyframe_qpos(self, name: str) -> np.ndarray:
        if hasattr(self._model, "keyframes") and self._model.num_keyframes > 0:
            return np.array(self._model.keyframes[0].dof_pos, dtype=self._np_dtype)
        return np.array(self._model.compute_init_dof_pos(), dtype=self._np_dtype)

    def get_init_qvel(self) -> np.ndarray:
        return np.zeros((self._model.num_dof_vel,), dtype=self._np_dtype)

    def get_body_ids(self, names: Sequence[str]) -> np.ndarray:
        ids: list[int] = []
        for name in names:
            bid = self._model.get_link_index(name)
            if bid is None or bid < 0:
                raise ValueError(f"Body '{name}' not found in Motrix model")
            ids.append(int(bid))
        return np.array(ids, dtype=np.int32)

    def get_joint_range(self) -> np.ndarray | None:
        return None

    # ------------------------------------------------------------------ #
    # Simulation control                                                 #
    # ------------------------------------------------------------------ #

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> None:
        self._data.actuator_ctrls = np.ascontiguousarray(ctrl)
        for _ in range(nsteps):
            self._model.step(self._data)
        self._refresh_link_pose_cache()

    def set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: ResetRandomizationPayload | None = None,
    ) -> None:
        # Convert quaternion from mujoco (wxyz) to motrix (xyzw)
        qpos_motrix = qpos.copy()
        qpos_motrix[:, 3:7] = qpos[:, [4, 5, 6, 3]]

        # Create mask for batch operation
        mask = np.zeros(self._num_envs, dtype=bool)
        mask[env_indices] = True
        data_slice = self._data[mask]

        # Batch set state
        data_slice.reset(self._model)
        self._apply_reset_randomization(data_slice, env_indices, randomization)
        data_slice.set_dof_vel(qvel)
        data_slice.set_dof_pos(qpos_motrix, self._model)

        # Set control to joint positions (actuator target for PD control)
        ctrl = qpos_motrix[:, 7:]
        data_slice.actuator_ctrls = np.ascontiguousarray(ctrl)

        self._model.forward_kinematic(self._data)
        self._refresh_link_pose_cache()

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        return DomainRandomizationCapabilities(
            supported_reset_terms=frozenset({RESET_TERM_BASE_MASS, RESET_TERM_BASE_COM}),
            supports_interval_push=True,
        )

    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        if plan.push_perturbation_limit is None:
            return
        self.push_robots(np.asarray(plan.push_perturbation_limit))

    # ------------------------------------------------------------------ #
    # Base kinematics                                                    #
    # ------------------------------------------------------------------ #

    def get_base_pos(self) -> np.ndarray:
        return np.asarray(self._body.floatingbase.get_translation(self._data))

    def get_base_quat(self) -> np.ndarray:
        return self._xyzw_to_wxyz(np.asarray(self._body.floatingbase.get_rotation(self._data)))

    def get_base_lin_vel(self) -> np.ndarray:
        return np.asarray(self._body.floatingbase.get_global_linear_velocity(self._data))

    def get_base_ang_vel(self) -> np.ndarray:
        return np.asarray(self._body.floatingbase.get_global_angular_velocity(self._data))

    # ------------------------------------------------------------------ #
    # DOF state                                                          #
    # ------------------------------------------------------------------ #

    def get_dof_pos(self) -> np.ndarray:
        return np.asarray(self._body.get_joint_dof_pos(self._data))

    def get_dof_vel(self) -> np.ndarray:
        return np.asarray(self._body.get_joint_dof_vel(self._data))

    # ------------------------------------------------------------------ #
    # Body kinematics — world frame                                      #
    # ------------------------------------------------------------------ #

    def _as_body_ids(self, body_ids: np.ndarray) -> np.ndarray:
        return np.asarray(body_ids, dtype=np.int32)

    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._get_link_poses_w(body_ids)[:, :, :3]

    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._xyzw_to_wxyz(self._get_link_poses_w(body_ids)[:, :, 3:])

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._get_link_lin_vel_w(body_ids)

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._get_link_ang_vel_w(body_ids)

    # ------------------------------------------------------------------ #
    # Body kinematics — baselink frame                                   #
    # ------------------------------------------------------------------ #

    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        return self._get_body_sensor_values(body_ids, "track_pos_b")

    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        # motrixsim framequat sensor 输出 xyzw，转换为接口约定的 wxyz
        return self._xyzw_to_wxyz(self._get_body_sensor_values(body_ids, "track_quat_b"))

    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        return self._get_body_sensor_values(body_ids, "track_linvel_b")

    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        return self._get_body_sensor_values(body_ids, "track_angvel_b")

    # ------------------------------------------------------------------ #
    # Sensors                                                            #
    # ------------------------------------------------------------------ #

    def get_sensor_data(self, name: str) -> np.ndarray:
        return np.asarray(self._model.get_sensor_value(name, self._data))

    # ------------------------------------------------------------------ #
    # MotrixSim-specific                                                 #
    # ------------------------------------------------------------------ #

    def _get_body_names(self, body_ids: np.ndarray) -> list[str]:
        return [self._body_id_to_name[int(bid)] for bid in self._as_body_ids(body_ids)]

    def _get_link_poses_w(self, body_ids: np.ndarray) -> np.ndarray:
        ids = self._as_body_ids(body_ids)
        return np.ascontiguousarray(self._link_poses[:, ids, :])

    def _get_link_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        ids = self._as_body_ids(body_ids)
        return np.stack(
            [np.asarray(self._link_cache[int(bid)].get_linear_velocity(self._data)) for bid in ids],
            axis=1,
        )

    def _get_link_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        ids = self._as_body_ids(body_ids)
        return np.stack(
            [
                np.asarray(self._link_cache[int(bid)].get_angular_velocity(self._data))
                for bid in ids
            ],
            axis=1,
        )

    def _get_body_sensor_values(self, body_ids: np.ndarray, prefix: str) -> np.ndarray:
        return np.stack(
            [
                np.asarray(self._model.get_sensor_value(f"{prefix}_{name}", self._data))
                for name in self._get_body_names(body_ids)
            ],
            axis=1,
        )

    def _xyzw_to_wxyz(self, q: np.ndarray) -> np.ndarray:
        """motrix xyzw → wxyz"""
        return q[..., [3, 0, 1, 2]]

    def _refresh_link_pose_cache(self) -> None:
        self._link_poses = np.ascontiguousarray(np.asarray(self._model.get_link_poses(self._data)))

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

    def _apply_reset_randomization(
        self,
        data_slice,
        env_indices: np.ndarray,
        randomization: ResetRandomizationPayload | None,
    ) -> None:
        if randomization is None or randomization.is_empty():
            return
        unsupported = randomization.requested_terms() - frozenset(
            {RESET_TERM_BASE_MASS, RESET_TERM_BASE_COM}
        )
        if unsupported:
            terms = ", ".join(sorted(unsupported))
            raise NotImplementedError(
                f"{self.backend_type} backend does not support reset randomization terms: {terms}"
            )

        env_ids = np.asarray(env_indices, dtype=np.intp)
        if randomization.base_mass_delta is not None:
            base_mass = self._default_base_mass_override[env_ids].copy()
            randomized_mass = base_mass + np.asarray(randomization.base_mass_delta)
            self._body_link.set_mass_override(data_slice, randomized_mass)

        if randomization.base_com_offset is not None:
            base_com = self._default_base_com_override[env_ids].copy()
            randomized_com = base_com + np.asarray(randomization.base_com_offset)
            self._body_link.set_center_of_mass_override(data_slice, randomized_com)
