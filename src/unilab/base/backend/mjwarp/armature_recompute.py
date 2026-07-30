"""Batched reference-constant recompute for per-world armature mutation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Iterator

import numpy as np
import torch
import warp as wp

from ..batch import BackendBatchContractError

_MAX_TILE_DOF = 64
_TILE_BLOCK_DIM = 128
_SUPPORTED_DIRECT_FIELDS = {
    ("dof_armature",),
    ("body_gravcomp", "dof_armature"),
}
_SUPPORTED_KINDS = {"set_const_0", "set_const"}


@cache
def _inverse_kernel(nv: int) -> Any:
    matrix_area = nv * nv

    @wp.kernel(module="unique", enable_backward=False)
    def kernel(
        active_mask: wp.array[bool],
        reference_mass: wp.array2d[float],
        armature: wp.array2d[float],
        default_armature: wp.array[float],
        identity: wp.array2d[float],
        inverse_mass: wp.array2d[float],
    ):
        worldid = wp.tid()
        if not active_mask[worldid]:
            return
        base = wp.tile_load(reference_mass, shape=(nv, nv), storage="shared")
        current = wp.tile_load(armature[worldid], shape=(nv,), storage="shared")
        default = wp.tile_load(default_armature, shape=(nv,), storage="shared")
        delta = wp.tile_map(wp.sub, current, default)
        mass = wp.tile_diag_add(base, delta)
        wp.tile_cholesky_inplace(mass, fill_mode="upper")
        rhs = wp.tile_load(identity, shape=(nv, nv), storage="shared")
        wp.tile_cholesky_solve_inplace(mass, rhs, fill_mode="upper")
        wp.tile_store(
            inverse_mass[worldid],
            wp.tile_reshape(rhs, (matrix_area,)),
        )

    return kernel


@cache
def _dof_invweight_kernel(nv: int) -> Any:
    @wp.kernel(module="unique", enable_backward=False)
    def kernel(
        active_mask: wp.array[bool],
        dof_jntid: wp.array[int],
        jnt_type: wp.array[int],
        jnt_dofadr: wp.array[int],
        inverse_mass: wp.array2d[float],
        dof_invweight0: wp.array2d[float],
    ):
        worldid, dofid = wp.tid()
        if not active_mask[worldid]:
            return
        jntid = dof_jntid[dofid]
        dofadr = jnt_dofadr[jntid]
        joint_type = jnt_type[jntid]
        value = inverse_mass[worldid, dofid * nv + dofid]
        if joint_type == 0:
            offset = 0
            if dofid >= dofadr + 3:
                offset = 3
            value = (
                inverse_mass[worldid, (dofadr + offset) * nv + dofadr + offset]
                + inverse_mass[worldid, (dofadr + offset + 1) * nv + dofadr + offset + 1]
                + inverse_mass[worldid, (dofadr + offset + 2) * nv + dofadr + offset + 2]
            ) / 3.0
        elif joint_type == 1:
            value = (
                inverse_mass[worldid, dofadr * nv + dofadr]
                + inverse_mass[worldid, (dofadr + 1) * nv + dofadr + 1]
                + inverse_mass[worldid, (dofadr + 2) * nv + dofadr + 2]
            ) / 3.0
        dof_invweight0[worldid % dof_invweight0.shape[0], dofid] = value

    return kernel


@cache
def _body_invweight_kernel(nv: int) -> Any:
    @wp.kernel(module="unique", enable_backward=False)
    def kernel(
        active_mask: wp.array[bool],
        body_weldid: wp.array[int],
        row_adr: wp.array[int],
        col_ind: wp.array[int],
        values: wp.array[float],
        inverse_mass: wp.array2d[float],
        body_invweight0: wp.array2d[wp.vec2],
    ):
        worldid, bodyid = wp.tid()
        if not active_mask[worldid]:
            return
        output_world = worldid % body_invweight0.shape[0]
        if bodyid == 0 or body_weldid[bodyid] == 0:
            body_invweight0[output_world, bodyid] = wp.vec2(0.0, 0.0)
            return

        translation = float(0.0)
        rotation = float(0.0)
        for component in range(6):
            row = bodyid * 6 + component
            start = row_adr[row]
            end = row_adr[row + 1]
            quadratic = float(0.0)
            for left_id in range(start, end):
                left_col = col_ind[left_id]
                transformed = float(0.0)
                for right_id in range(start, end):
                    right_col = col_ind[right_id]
                    transformed += (
                        inverse_mass[worldid, left_col * nv + right_col] * values[right_id]
                    )
                quadratic += values[left_id] * transformed
            if component < 3:
                translation += quadratic
            else:
                rotation += quadratic

        translation /= 3.0
        rotation /= 3.0
        if translation < 1.0e-15 and rotation > 1.0e-15:
            translation = rotation
        elif rotation < 1.0e-15 and translation > 1.0e-15:
            rotation = translation
        body_invweight0[output_world, bodyid] = wp.vec2(translation, rotation)

    return kernel


@cache
def _actuator_acceleration_kernel(nv: int) -> Any:
    @wp.kernel(module="unique", enable_backward=False)
    def kernel(
        active_mask: wp.array[bool],
        row_adr: wp.array[int],
        col_ind: wp.array[int],
        values: wp.array[float],
        inverse_mass: wp.array2d[float],
        actuator_acc0: wp.array2d[float],
    ):
        worldid, actuatorid = wp.tid()
        if not active_mask[worldid]:
            return
        start = row_adr[actuatorid]
        end = row_adr[actuatorid + 1]
        norm_squared = float(0.0)
        for row in range(nv):
            result = float(0.0)
            for value_id in range(start, end):
                result += inverse_mass[worldid, row * nv + col_ind[value_id]] * values[value_id]
            norm_squared += result * result
        actuator_acc0[worldid % actuator_acc0.shape[0], actuatorid] = wp.sqrt(norm_squared)

    return kernel


@cache
def _tendon_invweight_kernel(nv: int) -> Any:
    @wp.kernel(module="unique", enable_backward=False)
    def kernel(
        active_mask: wp.array[bool],
        row_adr: wp.array[int],
        col_ind: wp.array[int],
        values: wp.array[float],
        inverse_mass: wp.array2d[float],
        tendon_invweight0: wp.array2d[float],
    ):
        worldid, tendonid = wp.tid()
        if not active_mask[worldid]:
            return
        start = row_adr[tendonid]
        end = row_adr[tendonid + 1]
        quadratic = float(0.0)
        for left_id in range(start, end):
            left_col = col_ind[left_id]
            transformed = float(0.0)
            for right_id in range(start, end):
                transformed += (
                    inverse_mass[worldid, left_col * nv + col_ind[right_id]] * values[right_id]
                )
            quadratic += values[left_id] * transformed
        tendon_invweight0[worldid % tendon_invweight0.shape[0], tendonid] = quadratic

    return kernel


@cache
def _mean_inertia_kernel(nv: int) -> Any:
    @wp.kernel(module="unique", enable_backward=False)
    def kernel(
        active_mask: wp.array[bool],
        reference_mass: wp.array2d[float],
        armature: wp.array2d[float],
        default_armature: wp.array[float],
        meaninertia: wp.array[float],
    ):
        worldid = wp.tid()
        if not active_mask[worldid]:
            return
        source_world = worldid % armature.shape[0]
        total = float(0.0)
        for dofid in range(nv):
            total += (
                reference_mass[dofid, dofid]
                + armature[source_world, dofid]
                - default_armature[dofid]
            )
        meaninertia[worldid] = total / float(nv)

    return kernel


def _dense_rows_to_csr(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_adr = np.empty((values.shape[0] + 1,), dtype=np.int32)
    columns: list[int] = []
    entries: list[float] = []
    row_adr[0] = 0
    for row_id, row in enumerate(values):
        nonzero = np.flatnonzero(row)
        columns.extend(int(value) for value in nonzero)
        entries.extend(float(row[value]) for value in nonzero)
        row_adr[row_id + 1] = len(columns)
    return (
        row_adr,
        np.asarray(columns, dtype=np.int32),
        np.asarray(entries, dtype=np.float32),
    )


def _dampratio_actuators(model: Any, mujoco: Any) -> np.ndarray:
    gain = np.asarray(model.actuator_gainprm)
    bias = np.asarray(model.actuator_biasprm)
    affine = np.asarray(model.actuator_biastype) == int(mujoco.mjtBias.mjBIAS_AFFINE)
    return np.flatnonzero(
        affine
        & np.isclose(gain[:, 0], -bias[:, 1], rtol=0.0, atol=float(mujoco.mjMINVAL))
        & (bias[:, 2] > 0.0)
    )


@dataclass
class MjwarpArmatureRecomputeWorkspace:
    """Cold-compiled batched inverse and reference Jacobian workspace."""

    mujoco_warp: Any = field(repr=False)
    model: Any = field(repr=False)
    data: Any = field(repr=False)
    num_worlds: int
    nv: int
    nbody: int
    nu: int
    ntendon: int
    active_mask: Any = field(repr=False)
    reference_mass: Any = field(repr=False)
    default_armature: Any = field(repr=False)
    identity: Any = field(repr=False)
    inverse_mass: Any = field(repr=False)
    body_row_adr: Any = field(repr=False)
    body_col_ind: Any = field(repr=False)
    body_values: Any = field(repr=False)
    actuator_row_adr: Any = field(repr=False)
    actuator_col_ind: Any = field(repr=False)
    actuator_values: Any = field(repr=False)
    tendon_row_adr: Any = field(repr=False)
    tendon_col_ind: Any = field(repr=False)
    tendon_values: Any = field(repr=False)
    _prepared: bool = field(default=False, init=False, repr=False)
    _capturing: bool = field(default=False, init=False, repr=False)

    @classmethod
    def supports(cls, direct_fields: tuple[str, ...], kind: str, model: Any, mujoco: Any) -> bool:
        """Return whether the exact plan and model can use the specialized path."""

        nv = int(model.nv)
        return (
            direct_fields in _SUPPORTED_DIRECT_FIELDS
            and kind in _SUPPORTED_KINDS
            and 0 < nv <= _MAX_TILE_DOF
            and _dampratio_actuators(model, mujoco).size == 0
        )

    @classmethod
    def create(
        cls,
        *,
        mujoco: Any,
        mujoco_warp: Any,
        model: Any,
        device_model: Any,
        device_data: Any,
        active_mask: Any,
        num_worlds: int,
    ) -> MjwarpArmatureRecomputeWorkspace:
        """Compile qpos0 dynamics and upload immutable sparse reference rows."""

        nv = int(model.nv)
        nbody = int(model.nbody)
        nu = int(model.nu)
        ntendon = int(model.ntendon)
        if not 0 < nv <= _MAX_TILE_DOF:
            raise BackendBatchContractError(
                f"mjwarp armature tile recompute requires 1..{_MAX_TILE_DOF} DoFs"
            )

        cpu_data = mujoco.MjData(model)
        cpu_data.qpos[:] = model.qpos0
        mujoco.mj_forward(model, cpu_data)
        reference_mass = np.empty((nv, nv), dtype=np.float64)
        mujoco.mj_fullM(model, cpu_data, reference_mass)

        body_jacobian = np.zeros((nbody * 6, nv), dtype=np.float64)
        for bodyid in range(1, nbody):
            translation = body_jacobian[bodyid * 6 : bodyid * 6 + 3]
            rotation = body_jacobian[bodyid * 6 + 3 : bodyid * 6 + 6]
            mujoco.mj_jacBodyCom(model, cpu_data, translation, rotation, bodyid)

        actuator_moment = np.zeros((nu, nv), dtype=np.float64)
        if nu:
            mujoco.mju_sparse2dense(
                actuator_moment,
                cpu_data.actuator_moment,
                cpu_data.moment_rownnz,
                cpu_data.moment_rowadr,
                cpu_data.moment_colind,
            )
        tendon_jacobian = np.zeros((ntendon, nv), dtype=np.float64)
        if ntendon:
            mujoco.mju_sparse2dense(
                tendon_jacobian,
                cpu_data.ten_J,
                cpu_data.ten_J_rownnz,
                cpu_data.ten_J_rowadr,
                cpu_data.ten_J_colind,
            )

        body_csr = _dense_rows_to_csr(body_jacobian)
        actuator_csr = _dense_rows_to_csr(actuator_moment)
        tendon_csr = _dense_rows_to_csr(tendon_jacobian)
        device = device_model.dof_armature.device
        if (
            not isinstance(active_mask, torch.Tensor)
            or tuple(active_mask.shape) != (num_worlds,)
            or active_mask.dtype is not torch.bool
            or not active_mask.is_cuda
            or not active_mask.is_contiguous()
        ):
            raise BackendBatchContractError(
                "mjwarp armature recompute active mask has an invalid CUDA ABI"
            )
        warp_active_mask = wp.from_torch(active_mask)
        if warp_active_mask.device != device or int(warp_active_mask.ptr) != int(
            active_mask.data_ptr()
        ):
            raise BackendBatchContractError(
                "mjwarp armature recompute active mask has an invalid CUDA ABI"
            )

        def upload(value: np.ndarray, dtype: Any) -> Any:
            return wp.array(np.ascontiguousarray(value), dtype=dtype, device=device)

        return cls(
            mujoco_warp=mujoco_warp,
            model=device_model,
            data=device_data,
            num_worlds=num_worlds,
            nv=nv,
            nbody=nbody,
            nu=nu,
            ntendon=ntendon,
            active_mask=warp_active_mask,
            reference_mass=upload(reference_mass.astype(np.float32), float),
            default_armature=upload(np.asarray(model.dof_armature, dtype=np.float32), float),
            identity=upload(np.eye(nv, dtype=np.float32), float),
            inverse_mass=wp.empty(
                (num_worlds, nv * nv),
                dtype=float,
                device=device,
            ),
            body_row_adr=upload(body_csr[0], int),
            body_col_ind=upload(body_csr[1], int),
            body_values=upload(body_csr[2], float),
            actuator_row_adr=upload(actuator_csr[0], int),
            actuator_col_ind=upload(actuator_csr[1], int),
            actuator_values=upload(actuator_csr[2], float),
            tendon_row_adr=upload(tendon_csr[0], int),
            tendon_col_ind=upload(tendon_csr[1], int),
            tendon_values=upload(tendon_csr[2], float),
        )

    @staticmethod
    def _kind_value(kind: Any) -> str:
        return str(getattr(kind, "value", kind))

    def prepare(self, kind: Any, model: Any, data: Any) -> None:
        if (
            self._prepared
            or self._capturing
            or model is not self.model
            or data is not self.data
            or self._kind_value(kind) not in _SUPPORTED_KINDS
        ):
            raise BackendBatchContractError(
                "mjwarp armature recompute workspace preparation is invalid"
            )
        self._prepared = True

    @contextmanager
    def capture_body(self, kind: Any) -> Iterator[Any]:
        if not self._prepared or self._capturing:
            raise BackendBatchContractError("mjwarp armature recompute workspace is not prepared")
        self._capturing = True
        try:
            yield lambda model, data: self._recompute(kind, model, data)
        finally:
            self._capturing = False

    def _recompute(self, kind: Any, model: Any, data: Any) -> None:
        if model is not self.model or data is not self.data:
            raise BackendBatchContractError(
                "mjwarp armature recompute owner changed during capture"
            )
        if self._kind_value(kind) == "set_const":
            self.mujoco_warp.set_const_fixed(model, data)

        wp.launch_tiled(
            _inverse_kernel(self.nv),
            dim=self.num_worlds,
            inputs=[
                self.active_mask,
                self.reference_mass,
                model.dof_armature,
                self.default_armature,
                self.identity,
            ],
            outputs=[self.inverse_mass],
            block_dim=_TILE_BLOCK_DIM,
        )
        wp.launch(
            _dof_invweight_kernel(self.nv),
            dim=(self.num_worlds, self.nv),
            inputs=[
                self.active_mask,
                model.dof_jntid,
                model.jnt_type,
                model.jnt_dofadr,
                self.inverse_mass,
            ],
            outputs=[model.dof_invweight0],
        )
        wp.launch(
            _body_invweight_kernel(self.nv),
            dim=(self.num_worlds, self.nbody),
            inputs=[
                self.active_mask,
                model.body_weldid,
                self.body_row_adr,
                self.body_col_ind,
                self.body_values,
                self.inverse_mass,
            ],
            outputs=[model.body_invweight0],
        )
        if self.nu:
            wp.launch(
                _actuator_acceleration_kernel(self.nv),
                dim=(self.num_worlds, self.nu),
                inputs=[
                    self.active_mask,
                    self.actuator_row_adr,
                    self.actuator_col_ind,
                    self.actuator_values,
                    self.inverse_mass,
                ],
                outputs=[model.actuator_acc0],
            )
        if self.ntendon:
            wp.launch(
                _tendon_invweight_kernel(self.nv),
                dim=(self.num_worlds, self.ntendon),
                inputs=[
                    self.active_mask,
                    self.tendon_row_adr,
                    self.tendon_col_ind,
                    self.tendon_values,
                    self.inverse_mass,
                ],
                outputs=[model.tendon_invweight0],
            )
        wp.launch(
            _mean_inertia_kernel(self.nv),
            dim=model.stat.meaninertia.shape[0],
            inputs=[
                self.active_mask,
                self.reference_mass,
                model.dof_armature,
                self.default_armature,
            ],
            outputs=[model.stat.meaninertia],
        )

    @property
    def numeric_buffer_addresses(self) -> tuple[int, ...]:
        values = (
            self.active_mask,
            self.reference_mass,
            self.default_armature,
            self.identity,
            self.inverse_mass,
            self.body_row_adr,
            self.body_col_ind,
            self.body_values,
            self.actuator_row_adr,
            self.actuator_col_ind,
            self.actuator_values,
            self.tendon_row_adr,
            self.tendon_col_ind,
            self.tendon_values,
        )
        return tuple(int(value.ptr) for value in values if int(value.ptr or 0))


__all__ = ["MjwarpArmatureRecomputeWorkspace"]
