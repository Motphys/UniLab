"""Device-resident robot state reads for tensor motion-tracking owners.

The store consumes only ``SimBackend`` public tensor APIs.  It owns stable view
negotiation and full/selected-row ingestion, while task owners remain responsible
for their semantic contracts and downstream finite checks.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from unisim.backend.base import (
    HostBridgeTransferPlan,
    SimBackend,
    TensorExecution,
    TensorIOSpec,
    tensor_device_matches,
)


class TensorDeviceStateStore:
    """Read qpos/qvel, scalar sensors, and tracked body state on one device."""

    def __init__(
        self,
        *,
        backend: SimBackend,
        device: torch.device,
        num_envs: int,
        joint_qpos_ids: np.ndarray,
        joint_qvel_ids: np.ndarray,
        body_names: tuple[str, ...],
        body_ids: np.ndarray,
    ) -> None:
        if min(num_envs, len(body_names)) <= 0:
            raise ValueError("tensor state store dimensions must be positive")
        if joint_qpos_ids.shape != joint_qvel_ids.shape or joint_qpos_ids.ndim != 1:
            raise ValueError("tensor state joint indices must be aligned one-dimensional arrays")
        if body_ids.shape != (len(body_names),):
            raise ValueError("tensor state body IDs must match the body layout")
        self.backend = backend
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.joint_qpos_ids = np.asarray(joint_qpos_ids, dtype=np.int64)
        self.joint_qvel_ids = np.asarray(joint_qvel_ids, dtype=np.int64)
        self.body_names = tuple(body_names)
        self.body_ids = np.asarray(body_ids, dtype=np.intp)

        self.device = torch.device(device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA tensor state storage requested but CUDA is unavailable")
            if self.device.index is None:
                self.device = torch.device("cuda", index=torch.cuda.current_device())

        capabilities = self.backend.get_tensor_capabilities()
        self._execution = capabilities.execution
        if self._execution is TensorExecution.UNSUPPORTED:
            raise RuntimeError(
                "Tensor state store requires a DEVICE_RESIDENT or HOST_BRIDGE backend"
            )
        if not tensor_device_matches(capabilities.torch_devices, self.device):
            raise RuntimeError(
                f"backend did not accept Tensor device {str(self.device)!r}; "
                f"supported devices are {capabilities.torch_devices}"
            )

        joint_shape = (self.num_envs, self.joint_qpos_ids.size)
        body_shape = (self.num_envs, len(self.body_names))
        self.qpos: torch.Tensor | None = None
        self.qvel: torch.Tensor | None = None
        self.joint_pos = torch.empty(joint_shape, dtype=torch.float32, device=self.device)
        self.joint_vel = torch.empty_like(self.joint_pos)
        self.linvel = torch.empty((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.gyro = torch.empty_like(self.linvel)
        self.robot_body_pos = torch.empty((*body_shape, 3), device=self.device)
        self.robot_body_quat = torch.empty((*body_shape, 4), device=self.device)
        self.robot_body_lin_vel = torch.empty_like(self.robot_body_pos)
        self.robot_body_ang_vel = torch.empty_like(self.robot_body_pos)
        self._host_bridge_plan: HostBridgeTransferPlan | None = None
        self.last_backend_result: dict | None = None
        if self._execution is TensorExecution.HOST_BRIDGE:
            if not capabilities.packed_host_bridge:
                raise RuntimeError(
                    "Tensor state store requires packed I/O for host-bridge physics backends"
                )
            sensor_names = (
                "pelvis_local_linvel",
                "torso_gyro",
                *(
                    f"{prefix}_{name}"
                    for prefix in (
                        "track_pos_w",
                        "track_quat_w",
                        "track_linvel_w",
                        "track_angvel_w",
                    )
                    for name in self.body_names
                ),
            )
            self._host_bridge_plan = self.backend.compile_host_bridge_io(
                TensorIOSpec(
                    state_fields=("qpos", "qvel"),
                    sensor_names=sensor_names,
                    device=self.device,
                )
            )
        self._device_state_views: dict[str, torch.Tensor] | None = None
        self._device_sensor_views: dict[str, object] | None = None
        self._device_linvel_view: torch.Tensor | None = None
        self._device_gyro_view: torch.Tensor | None = None
        self._views_require_readiness_barrier = False

    def _require_qviews(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.qpos is None or self.qvel is None:
            raise RuntimeError("tensor state store has not performed a full read")
        return self.qpos, self.qvel

    def qviews(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the current backend qpos/qvel views after a full read."""
        return self._validate_qviews()

    def _validate_rows(self, rows: torch.Tensor | None) -> torch.Tensor | None:
        if rows is None:
            return None
        if rows.ndim != 1 or rows.device != self.device or rows.dtype != torch.int64:
            raise ValueError("tensor state rows must be a one-dimensional int64 device tensor")
        if rows.numel() == 0:
            return rows
        minimum_row, maximum_row = (
            int(value) for value in torch.stack((rows.min(), rows.max())).tolist()
        )
        if minimum_row < 0 or maximum_row >= self.num_envs:
            raise IndexError(
                "tensor state rows are outside the environment range "
                f"[0, {self.num_envs}); got [{minimum_row}, {maximum_row}]"
            )
        return rows

    def _validate_qviews(self) -> tuple[torch.Tensor, torch.Tensor]:
        qpos, qvel = self._require_qviews()
        expected_qvel = (self.num_envs, qpos.shape[1] - 1)
        if qpos.shape != (self.num_envs, qvel.shape[1] + 1):
            raise RuntimeError(
                f"backend qpos/qvel shapes are inconsistent: {tuple(qpos.shape)} vs {tuple(qvel.shape)}"
            )
        if expected_qvel != tuple(qvel.shape):
            raise RuntimeError(
                f"backend qvel shape is {tuple(qvel.shape)}, expected {expected_qvel}"
            )
        if qpos.device != self.device or qvel.device != self.device:
            raise RuntimeError("backend qpos/qvel views live on the wrong device")
        return qpos, qvel

    def _read_device_resident(self, rows: torch.Tensor | None) -> None:
        state_views = self._device_state_views
        sensor_views = self._device_sensor_views
        if state_views is None or sensor_views is None:
            requested_views = self.backend.get_state_views(("qpos", "qvel"), device=self.device)
            qpos = requested_views["qpos"]
            qvel = requested_views["qvel"]
            if not isinstance(qpos, torch.Tensor) or not isinstance(qvel, torch.Tensor):
                raise TypeError("device-resident state views did not return tensors")
            state_views = {"qpos": qpos, "qvel": qvel}
            sensor_views = {}
            linvel_view = self.backend.get_sensor_view("pelvis_local_linvel", device=self.device)
            gyro_view = self.backend.get_sensor_view("torso_gyro", device=self.device)
            if not isinstance(linvel_view, torch.Tensor) or not isinstance(gyro_view, torch.Tensor):
                raise TypeError("device-resident scalar sensor views did not return tensors")
            sensor_views["linvel"] = linvel_view
            sensor_views["gyro"] = gyro_view
            for prefix in ("track_pos_w", "track_quat_w", "track_linvel_w", "track_angvel_w"):
                sensor_views[prefix] = tuple(
                    self.backend.get_sensor_view(f"{prefix}_{name}", device=self.device)
                    for name in self.body_names
                )
            self._device_state_views = state_views
            self._device_sensor_views = sensor_views
            self._device_linvel_view = linvel_view
            self._device_gyro_view = gyro_view
            self.linvel = linvel_view.clone()
            self.gyro = gyro_view.clone()
        else:
            assert self._device_linvel_view is not None
            assert self._device_gyro_view is not None
            # Some device-resident adapters refresh tracked-body state through
            # the public sensor lifecycle. Preserve their authored frame-sensor
            # boundary semantics while refreshing only tracked body views.
            self.linvel.copy_(self._device_linvel_view)
            self.gyro.copy_(self._device_gyro_view)
            # Stable views do not negotiate lifecycle state on dereference. One
            # tracked-sensor read refreshes all injected frame sensors.
            self.backend.get_sensor_view(f"track_pos_w_{self.body_names[0]}", device=self.device)
            if rows is not None:
                self.linvel[rows] = self._device_linvel_view.index_select(0, rows)
                self.gyro[rows] = self._device_gyro_view.index_select(0, rows)

        self.qpos = state_views["qpos"]
        self.qvel = state_views["qvel"]
        self._validate_qviews()
        self.joint_pos = self.qpos[:, self.joint_qpos_ids]
        self.joint_vel = self.qvel[:, self.joint_qvel_ids]
        destinations = (
            self.robot_body_pos,
            self.robot_body_quat,
            self.robot_body_lin_vel,
            self.robot_body_ang_vel,
        )
        prefixes = ("track_pos_w", "track_quat_w", "track_linvel_w", "track_angvel_w")
        for destination, prefix in zip(destinations, prefixes, strict=True):
            views = sensor_views[prefix]
            if not isinstance(views, tuple) or len(views) != len(self.body_names):
                raise TypeError("device-resident body sensor views did not match the body layout")
            if rows is None:
                torch.stack(views, dim=1, out=destination)
            else:
                selected = torch.stack(tuple(view.index_select(0, rows) for view in views), dim=1)
                destination.index_copy_(0, rows, selected)

    def _read_host_bridge(self, rows: torch.Tensor | None) -> None:
        if self._host_bridge_plan is not None:
            values = (
                self._host_bridge_plan.read_state_sensors()
                if rows is None
                else self._host_bridge_plan.read_selected_state_sensors()
            )
            self.qpos = values["qpos"]
            self.qvel = values["qvel"]
            qpos, qvel = self._validate_qviews()
            self.joint_pos = qpos[:, self.joint_qpos_ids]
            self.joint_vel = qvel[:, self.joint_qvel_ids]
            self.linvel = values["pelvis_local_linvel"]
            self.gyro = values["torso_gyro"]
            destinations = (
                self.robot_body_pos,
                self.robot_body_quat,
                self.robot_body_lin_vel,
                self.robot_body_ang_vel,
            )
            prefixes = ("track_pos_w", "track_quat_w", "track_linvel_w", "track_angvel_w")
            for destination, prefix in zip(destinations, prefixes, strict=True):
                torch.stack(
                    tuple(values[f"{prefix}_{name}"] for name in self.body_names),
                    dim=1,
                    out=destination,
                )
            self.last_backend_result = {"timing": dict(self._host_bridge_plan.last_timing)}
            return

        if rows is None:
            state = self.backend.get_state_views(("qpos", "qvel"), device=self.device)
            self.qpos = state["qpos"]
            self.qvel = state["qvel"]
            qpos, qvel = self._validate_qviews()
            self.joint_pos = qpos[:, self.joint_qpos_ids]
            self.joint_vel = qvel[:, self.joint_qvel_ids]
            self.linvel = self.backend.get_sensor_view("pelvis_local_linvel", device=self.device)
            self.gyro = self.backend.get_sensor_view("torso_gyro", device=self.device)
        else:
            self._validate_qviews()
            host_rows = rows.detach().cpu().numpy()
            linvel = self.backend.get_sensor_data_rows("pelvis_local_linvel", host_rows)
            gyro = self.backend.get_sensor_data_rows("torso_gyro", host_rows)
            self.linvel[rows] = torch.as_tensor(
                np.ascontiguousarray(linvel), dtype=torch.float32, device=self.device
            )
            self.gyro[rows] = torch.as_tensor(
                np.ascontiguousarray(gyro), dtype=torch.float32, device=self.device
            )

        for name, value in (("linvel", self.linvel), ("gyro", self.gyro)):
            if not isinstance(value, torch.Tensor) or value.shape != (self.num_envs, 3):
                raise RuntimeError(f"backend scalar sensor {name!r} has an invalid view")

        if rows is None:
            values = (
                self.backend.get_body_pos_w(self.body_ids),
                self.backend.get_body_quat_w(self.body_ids),
                self.backend.get_body_lin_vel_w(self.body_ids),
                self.backend.get_body_ang_vel_w(self.body_ids),
            )
            destinations = (
                self.robot_body_pos,
                self.robot_body_quat,
                self.robot_body_lin_vel,
                self.robot_body_ang_vel,
            )
            for destination, value in zip(destinations, values, strict=True):
                destination.copy_(
                    torch.as_tensor(
                        np.ascontiguousarray(value), dtype=torch.float32, device=self.device
                    )
                )
        else:
            host_rows = rows.detach().cpu().numpy()
            pos, quat = self.backend.get_body_pose_w_rows(host_rows, self.body_ids)
            lin_vel = self.backend.get_body_lin_vel_w_rows(host_rows, self.body_ids)
            ang_vel = self.backend.get_body_ang_vel_w_rows(host_rows, self.body_ids)
            destinations = (
                self.robot_body_pos,
                self.robot_body_quat,
                self.robot_body_lin_vel,
                self.robot_body_ang_vel,
            )
            for destination, value in zip(destinations, (pos, quat, lin_vel, ang_vel), strict=True):
                destination[rows] = torch.as_tensor(
                    np.ascontiguousarray(value), dtype=torch.float32, device=self.device
                )

    def read(self, rows: torch.Tensor | None = None) -> None:
        selected_rows = self._validate_rows(rows)
        if selected_rows is not None and selected_rows.numel() == 0:
            return
        if self._execution is TensorExecution.DEVICE_RESIDENT:
            self._read_device_resident(selected_rows)
        elif self._execution is TensorExecution.HOST_BRIDGE:
            self._read_host_bridge(selected_rows)
        else:  # Guarded in the constructor; retained for exhaustive dispatch.
            raise RuntimeError("Tensor state store received an unsupported tensor execution")

    def step_tensor(self, ctrl: torch.Tensor, nsteps: int) -> dict | None:
        """Step through the negotiated packed plan when one is available."""
        if self._host_bridge_plan is not None:
            self._host_bridge_plan.write_control(ctrl)
            return cast(dict | None, self._host_bridge_plan.step(nsteps))
        return cast(dict | None, self.backend.step_tensor(ctrl, nsteps=nsteps))

    def apply_reset(
        self, rows: torch.Tensor, qpos: torch.Tensor, qvel: torch.Tensor
    ) -> dict | None:
        """Apply selected reset through the negotiated packed plan."""
        if self._host_bridge_plan is not None:
            return cast(dict | None, self._host_bridge_plan.apply_reset(rows, qpos, qvel))
        result = cast(dict | None, self.backend.set_state_tensor(rows, qpos, qvel))
        self._views_require_readiness_barrier = True
        self.last_backend_result = result
        return result

    def refresh_after_selected_reset(self, ctrl: torch.Tensor, nsteps: int) -> dict | None:
        """Cross the backend's public selected-reset readiness boundary.

        A backend may invalidate derived sensor/body views after
        ``set_state_tensor`` even though qpos/qvel views are already
        authoritative. This method performs that backend-owned readiness
        boundary explicitly through the same public tensor step API used by the
        normal control path. It is a lifecycle barrier, not a hidden data copy:
        control is published through the negotiated tensor plane, and the task
        performs no host/device fallback.
        """
        if self._execution is not TensorExecution.DEVICE_RESIDENT:
            raise RuntimeError(
                "selected-reset readiness requires DEVICE_RESIDENT tensor execution; "
                f"received {self._execution}"
            )
        if self._host_bridge_plan is not None:
            raise RuntimeError(
                "HOST_BRIDGE selected reset owns its paired post-reset read boundary"
            )
        if not self._views_require_readiness_barrier:
            return None
        result = self.step_tensor(ctrl, nsteps=nsteps)
        self._views_require_readiness_barrier = False
        self.last_backend_result = result
        return result

    def validate_finite(self) -> None:
        qpos, qvel = self._require_qviews()
        values = (
            qpos,
            qvel,
            self.joint_pos,
            self.joint_vel,
            self.linvel,
            self.gyro,
            self.robot_body_pos,
            self.robot_body_quat,
            self.robot_body_lin_vel,
            self.robot_body_ang_vel,
        )
        if not bool(torch.isfinite(values[0]).all()):
            raise ValueError("tensor robot state contains NaN or Inf")
        for value in values[1:]:
            if not bool(torch.isfinite(value).all()):
                raise ValueError("tensor robot state contains NaN or Inf")


__all__ = ["TensorDeviceStateStore"]
