"""Contract tests for the owner-owned fused root+joint tensor reset write."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import torch
from unisim.backend.base import (
    BackendRootStateLayout,
    SimBackend,
    TensorDataPlane,
    TensorExecution,
    TensorLifecycleCapabilities,
    TensorProcessTopology,
)

from unilab.base.reset_state import ResetStateTransaction


class _TensorBackend:
    backend_type = "fake-tensor"
    num_envs = 4

    def __init__(self) -> None:
        self.default_qpos = np.array(
            [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3], dtype=np.float32
        )
        self.default_qvel = np.arange(9, dtype=np.float32) * 0.1
        self.reset_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def get_default_qpos(self) -> np.ndarray:
        return self.default_qpos.copy()

    def get_init_qvel(self) -> np.ndarray:
        return self.default_qvel.copy()

    def get_tensor_capabilities(self) -> TensorLifecycleCapabilities:
        return TensorLifecycleCapabilities(
            execution=TensorExecution.DEVICE_RESIDENT,
            state_fields=frozenset({"qpos", "qvel"}),
            selected_reset=True,
            process_topology=TensorProcessTopology.IN_PROCESS,
            data_plane=TensorDataPlane.DIRECT,
            stream_event_ownership="test",
            torch_devices=("cuda", "cpu"),
        )

    def set_state_tensor(self, rows, qpos, qvel, randomization=None) -> None:
        assert randomization is None
        self.reset_calls.append((rows.clone(), qpos.clone(), qvel.clone()))


def _transaction(backend: _TensorBackend, device: torch.device) -> ResetStateTransaction:
    transaction = ResetStateTransaction(cast(SimBackend, backend))
    transaction.declare_packed_reset_device(device)
    return transaction


def _valid_inputs(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    root = torch.tensor(
        [[0.1, 0.2, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, -0.5, 0.25, 0.1, -0.1, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    position = torch.tensor([[0.7, -0.2, 0.3]], dtype=torch.float32, device=device)
    velocity = torch.tensor([[-0.4, 0.5, 0.6]], dtype=torch.float32, device=device)
    return root, position, velocity


def test_fused_write_stages_root_and_joint_columns_once() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = _TensorBackend()
    transaction = _transaction(backend, device)
    root, position, velocity = _valid_inputs(device)

    with transaction.scoped_device_event_tensor(torch.tensor([2], device=device)):
        transaction.write_root_and_joint_state_tensor(
            torch.tensor([2], device=device),
            BackendRootStateLayout(tuple(range(7)), tuple(range(6))),
            np.array([7, 8, 9], dtype=np.int32),
            np.array([6, 7, 8], dtype=np.int32),
            root,
            position,
            velocity,
            term_name="motion.write_articulation_state_tensor_to_sim",
        )

    assert len(backend.reset_calls) == 1
    rows, qpos, qvel = backend.reset_calls[0]
    torch.testing.assert_close(rows, torch.tensor([2], device=device))
    torch.testing.assert_close(qpos[0, :7], root[0, :7])
    torch.testing.assert_close(qpos[0, 7:], position[0])
    torch.testing.assert_close(qvel[0, :6], root[0, 7:])
    torch.testing.assert_close(qvel[0, 6:], velocity[0])


def test_fused_write_fail_closed_for_invalid_values_and_scope() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layout = BackendRootStateLayout(tuple(range(7)), tuple(range(6)))
    qpos_columns = np.array([7, 8, 9], dtype=np.int32)
    qvel_columns = np.array([6, 7, 8], dtype=np.int32)
    cases: list[tuple[torch.Tensor, torch.Tensor, type[Exception], str]] = []
    nan_root, _, _ = _valid_inputs(device)
    nan_root = nan_root.clone()
    nan_root[0, 0] = torch.nan
    valid_root, valid_position, valid_velocity = _valid_inputs(device)
    cases.append((nan_root, valid_position, ValueError, "root state.*NaN or Inf"))
    bad_quat = valid_root.clone()
    bad_quat[0, 3] = 2.0
    cases.append((bad_quat, valid_position, ValueError, "root quaternion must be unit length"))
    nan_position = valid_position.clone()
    nan_position[0, 0] = torch.nan
    cases.append((valid_root, nan_position, ValueError, "joint position.*NaN or Inf"))

    for root, position, _, _ in cases:
        backend = _TensorBackend()
        transaction = _transaction(backend, device)
        with pytest.raises(ValueError):
            with transaction.scoped_device_event_tensor(torch.tensor([0], device=device)):
                transaction.write_root_and_joint_state_tensor(
                    torch.tensor([0], device=device),
                    layout,
                    qpos_columns,
                    qvel_columns,
                    root,
                    position,
                    valid_velocity,
                    term_name="bad",
                )
        assert backend.reset_calls == []

    backend = _TensorBackend()
    transaction = _transaction(backend, device)
    with pytest.raises(ValueError, match="outside the active tensor reset"):
        with transaction.scoped_device_event_tensor(torch.tensor([0], device=device)):
            transaction.write_root_and_joint_state_tensor(
                torch.tensor([1], device=device),
                layout,
                qpos_columns,
                qvel_columns,
                valid_root,
                valid_position,
                valid_velocity,
                term_name="scope",
            )
    assert backend.reset_calls == []


def test_fused_write_rejects_overlapping_root_and_joint_columns() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = _TensorBackend()
    transaction = _transaction(backend, device)
    root, position, velocity = _valid_inputs(device)

    with pytest.raises(ValueError, match="root and joint columns overlap"):
        with transaction.scoped_device_event_tensor(torch.tensor([0], device=device)):
            transaction.write_root_and_joint_state_tensor(
                torch.tensor([0], device=device),
                BackendRootStateLayout(tuple(range(7)), tuple(range(6))),
                np.array([6, 7, 8], dtype=np.int32),
                np.array([5, 7, 8], dtype=np.int32),
                root,
                position,
                velocity,
                term_name="overlap",
            )
    assert backend.reset_calls == []
