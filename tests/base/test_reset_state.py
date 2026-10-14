"""Focused tests for the base-owned Manager-Based reset transaction."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
import torch
from unisim.backend.base import (
    BackendMocapPoseBinding,
    BackendRootStateLayout,
    PublicStateWidths,
    SelectedResetPublication,
    SimBackend,
    TensorDataPlane,
    TensorExecution,
    TensorLifecycleCapabilities,
    TensorProcessTopology,
)
from unisim.dr.types import (
    RESET_TERM_BODY_IPOS,
    RESET_TERM_BODY_MASS,
    RESET_TERM_KD,
    RESET_TERM_KP,
    DomainRandomizationCapabilities,
    ResetRandomizationPayload,
    TensorResetRandomizationPayload,
)

from unilab.base.reset_state import ResetStateTransaction


class _Backend:
    backend_type = "fake"
    num_actuators = 3

    def __init__(
        self,
        *,
        qpos: Any = None,
        qvel: Any = None,
        fail_set_state: bool = False,
    ) -> None:
        self.num_envs = 4
        self.qpos = np.array([1.0, 2.0, 3.0]) if qpos is None else qpos
        self.qvel = np.array([0.0, 0.0]) if qvel is None else qvel
        self.fail_set_state = fail_set_state
        self.default_qpos_calls = 0
        self.init_qvel_calls = 0
        self.set_state_calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.randomization_calls: list[ResetRandomizationPayload | None] = []
        self.default_kp = np.array([10.0, 20.0, 30.0])
        self.default_kd = np.array([1.0, 2.0, 3.0])

    def get_default_qpos(self):
        self.default_qpos_calls += 1
        return self.qpos

    def get_init_qvel(self):
        self.init_qvel_calls += 1
        return self.qvel

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        return DomainRandomizationCapabilities(
            supported_reset_terms=frozenset((RESET_TERM_KP, RESET_TERM_KD, "gravity"))
        )

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        return self.default_kp.copy(), self.default_kd.copy()

    def get_reset_term_default(self, term: str) -> np.ndarray:
        if term == RESET_TERM_KP:
            return self.default_kp.copy()
        if term == RESET_TERM_KD:
            return self.default_kd.copy()
        if term == "gravity":
            return np.array([-1.0, 0.0, 9.81])
        raise NotImplementedError(term)

    def set_state_tensor(self, env_indices, qpos, qvel, randomization=None) -> dict:
        return self.set_state(
            np.asarray(env_indices),
            np.asarray(qpos),
            np.asarray(qvel),
            randomization=randomization,
        )

    def set_state(
        self,
        env_ids: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization=None,
    ) -> dict:
        if self.fail_set_state:
            raise NotImplementedError("reset upload disabled")
        self.set_state_calls.append((env_ids.copy(), qpos.copy(), qvel.copy()))
        self.randomization_calls.append(randomization)
        return {"timing": {"set_state_ms": 1.0}}


class _TensorResetBackend(_Backend):
    """CPU fake exposing the public packed tensor-reset boundary."""

    def __init__(self) -> None:
        super().__init__(
            qpos=np.array([0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.1, -0.2]),
            qvel=np.zeros(8),
        )
        self.default_body_mass = np.array([2.0, 4.0])
        self.tensor_reset_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def tensor_execution(self):
        return TensorExecution.DEVICE_RESIDENT

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        return DomainRandomizationCapabilities(
            supported_reset_terms=frozenset(
                (RESET_TERM_KP, RESET_TERM_KD, RESET_TERM_BODY_MASS, "gravity")
            )
        )

    def get_reset_term_default(self, term: str) -> np.ndarray:
        if term == RESET_TERM_BODY_MASS:
            return self.default_body_mass.copy()
        return super().get_reset_term_default(term)

    def get_tensor_capabilities(self):
        return TensorLifecycleCapabilities(
            execution=TensorExecution.DEVICE_RESIDENT,
            state_views=True,
            state_fields=frozenset({"qpos", "qvel"}),
            sensor_views=False,
            stepping=True,
            selected_reset=True,
            reset_randomization=True,
            process_topology=TensorProcessTopology.IN_PROCESS,
            data_plane=TensorDataPlane.DIRECT,
            stream_event_ownership="fake synchronous stream",
            torch_devices=("cpu",),
            device_reset_randomization=True,
            selected_reset_publication=SelectedResetPublication.AUTHORITATIVE_VIEWS,
            packed_reset=True,
            packed_reset_fields=frozenset(
                (RESET_TERM_KP, RESET_TERM_KD, RESET_TERM_BODY_MASS, "gravity")
            ),
        )

    def begin_packed_reset(self, *, device=None):
        from unisim.backend.packed_reset import TorchPackedResetComposer

        return TorchPackedResetComposer(self, device=device)

    def set_state_tensor(self, env_indices, qpos, qvel, randomization=None) -> dict:
        rows = torch.as_tensor(env_indices, dtype=torch.int64)
        self.tensor_reset_calls.append((rows.clone(), qpos.clone(), qvel.clone(), randomization))
        return {"ok": True}

    def set_state_tensor_mask(self, reset_mask, qpos, qvel, randomization=None, **_kwargs):
        rows = torch.nonzero(reset_mask, as_tuple=False).flatten()
        selected_randomization = randomization
        if randomization is not None and any(
            value is not None and value.shape[:1] == reset_mask.shape
            for value in vars(randomization).values()
        ):
            selected_randomization = type(randomization)(
                **{
                    name: value.index_select(0, rows)
                    for name, value in vars(randomization).items()
                    if value is not None and value.shape[:1] == reset_mask.shape
                }
            )
        return self.set_state_tensor(
            rows, qpos.index_select(0, rows), qvel.index_select(0, rows), selected_randomization
        )

    def get_public_state_widths(self):
        return PublicStateWidths(len(self.qpos), len(self.qvel))


class _PackedTensorResetBackend(_TensorResetBackend):
    """CPU packed fake adding the public mocap tensor-write boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.mocap_poses = np.tile([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], (self.num_envs, 1))
        self.tensor_mocap_calls: list[tuple[str, torch.Tensor, torch.Tensor]] = []

    def get_tensor_capabilities(self):
        capabilities = super().get_tensor_capabilities()
        return replace(
            capabilities,
            packed_reset_mocap=True,
        )

    def bind_mocap_pose(self, body_name: str):
        return BackendMocapPoseBinding(
            self.backend_type,
            body_name,
            self.num_envs,
            self.mocap_poses[0].copy(),
            lambda: self.mocap_poses,
            lambda ids, poses: self.mocap_poses.__setitem__(ids, poses),
        )

    def set_mocap_pose_tensor(self, body_name, env_indices, poses):
        self.tensor_mocap_calls.append((body_name, env_indices.clone(), poses.clone()))
        ids = env_indices.detach().cpu().numpy()
        self.mocap_poses[ids] = poses.detach().cpu().numpy()


class _RecordingPackedTensorResetBackend(_PackedTensorResetBackend):
    """Public-composer adapter that retains submitted state payload layouts."""

    def __init__(self) -> None:
        super().__init__()
        self.state_write_payloads: list[tuple[torch.Tensor, torch.Tensor]] = []

    def begin_packed_reset(self, *, device=None):
        composer = super().begin_packed_reset(device=device)
        write_state = composer.write_state
        write_state_mask = composer.write_state_mask

        def record_state_write(rows, *, qpos_values=None, qvel_values=None, **kwargs):
            if qpos_values is not None and qvel_values is not None:
                self.state_write_payloads.append((qpos_values, qvel_values))
            return write_state(rows, qpos_values=qpos_values, qvel_values=qvel_values, **kwargs)

        def record_state_write_mask(mask, *, qpos_values=None, qvel_values=None, **kwargs):
            if qpos_values is not None and qvel_values is not None:
                rows = torch.nonzero(mask, as_tuple=False).flatten()
                self.state_write_payloads.append(
                    (qpos_values.index_select(0, rows), qvel_values.index_select(0, rows))
                )
            return write_state_mask(
                mask, qpos_values=qpos_values, qvel_values=qvel_values, **kwargs
            )

        composer.write_state = record_state_write
        composer.write_state_mask = record_state_write_mask
        return composer


def _transaction(backend: _Backend) -> ResetStateTransaction:
    return ResetStateTransaction(cast(SimBackend, backend))


class _ManipulationBackend(_Backend):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []
        self.poses = np.tile([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], (self.num_envs, 1))
        self.default_calls = 0

    def get_dr_capabilities(self):
        return DomainRandomizationCapabilities(
            supported_reset_terms=frozenset(
                (
                    "geom_size",
                    "geom_solref",
                    "geom_solimp",
                    "dof_damping",
                    "dof_frictionloss",
                )
            )
        )

    def _defaults(self, width):
        self.default_calls += 1
        return np.ones((3, width)) if width else np.ones(2)

    def get_reset_term_default(self, term: str) -> np.ndarray:
        widths = {
            "geom_size": 3,
            "geom_solref": 2,
            "geom_solimp": 5,
            "dof_damping": 0,
            "dof_frictionloss": 0,
        }
        if term not in widths:
            raise NotImplementedError(term)
        return self._defaults(widths[term])

    def set_state(self, env_ids, qpos, qvel, randomization=None):
        self.events.append("state")
        self.poses[env_ids, 0] = 0.0
        return super().set_state(env_ids, qpos, qvel, randomization)

    def bind_mocap_pose(self, body_name):
        def write(ids, poses):
            self.events.append("mocap")
            self.poses[ids] = poses

        return BackendMocapPoseBinding(
            self.backend_type,
            body_name,
            self.num_envs,
            self.poses[0].copy(),
            lambda: self.poses,
            write,
        )


class _PerWorldBodyMassBackend(_Backend):
    def __init__(self) -> None:
        super().__init__()
        self.default_calls = 0

    def get_dr_capabilities(self):
        return DomainRandomizationCapabilities(
            supported_reset_terms=frozenset((RESET_TERM_BODY_MASS,))
        )

    def get_tensor_capabilities(self):
        return TensorLifecycleCapabilities(
            execution=TensorExecution.DEVICE_RESIDENT,
            state_views=True,
            state_fields=frozenset({"qpos", "qvel"}),
            sensor_views=False,
            stepping=True,
            selected_reset=True,
            reset_randomization=True,
            process_topology=TensorProcessTopology.IN_PROCESS,
            data_plane=TensorDataPlane.DIRECT,
            stream_event_ownership="fake synchronous stream",
            torch_devices=("cpu",),
            device_reset_randomization=True,
            selected_reset_publication=SelectedResetPublication.AUTHORITATIVE_VIEWS,
            packed_reset=True,
            packed_reset_fields=frozenset((RESET_TERM_BODY_MASS,)),
        )

    def get_public_state_widths(self):
        return PublicStateWidths(len(self.qpos), len(self.qvel))

    def begin_packed_reset(self, *, device=None):
        from unisim.backend.packed_reset import TorchPackedResetComposer

        return TorchPackedResetComposer(self, device=device)

    def get_reset_term_default(self, term: str) -> np.ndarray:
        if term != RESET_TERM_BODY_MASS:
            raise NotImplementedError(term)
        self.default_calls += 1
        return np.asarray(
            [
                [1.0, 10.0, 100.0],
                [2.0, 20.0, 200.0],
                [3.0, 30.0, 300.0],
                [4.0, 40.0, 400.0],
            ]
        )


class _StartupModelFieldBackend(_PerWorldBodyMassBackend):
    """Expose canonical mass and CoM defaults for startup baseline tests."""

    def get_dr_capabilities(self):
        return DomainRandomizationCapabilities(
            supported_reset_terms=frozenset((RESET_TERM_BODY_MASS, RESET_TERM_BODY_IPOS))
        )

    def get_tensor_capabilities(self):
        return replace(
            super().get_tensor_capabilities(),
            packed_reset_fields=frozenset((RESET_TERM_BODY_MASS, RESET_TERM_BODY_IPOS)),
        )

    def get_reset_term_default(self, term: str) -> np.ndarray:
        if term == RESET_TERM_BODY_IPOS:
            return np.asarray([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]], dtype=np.float64)
        return super().get_reset_term_default(term)


class _InvalidDefaultBackend(_Backend):
    def __init__(self, field: str, value: Any) -> None:
        super().__init__()
        self.field = field
        self.value = value

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        return DomainRandomizationCapabilities(supported_reset_terms=frozenset((self.field,)))

    def get_reset_term_default(self, term: str) -> np.ndarray:
        if term != self.field:
            raise NotImplementedError(term)
        return self.value


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("body_mass", np.ones((2, 3), dtype=np.float64), "canonical 1-D table"),
        (
            "body_mass",
            np.ones((3, 2), dtype=np.float64),
            "canonical 1-D table or a per-environment \\(4, \\*canonical\\)",
        ),
        ("body_mass", np.ones(2, dtype=np.int32), "must be floating"),
        ("gravity", np.ones(2, dtype=np.float64), "canonical 1-D table"),
    ],
)
def test_reset_term_default_shapes_fail_closed(field: str, value: Any, match: str) -> None:
    transaction = _transaction(_InvalidDefaultBackend(field, value))
    with pytest.raises((TypeError, ValueError), match=match):
        if field == "gravity":
            transaction.bind_gravity_write(term_name="bad_default")
        else:
            transaction.bind_body_mass_write(
                np.array([0], dtype=np.int32),
                term_name="bad_default",
            )


def test_packed_reset_combines_default_and_joint_writes() -> None:
    backend = _TensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    reset_mask = torch.tensor([False, True, True, False])

    with transaction.scoped_device_event_tensor(reset_mask):
        transaction.reset_to_default(reset_mask, term_name="scene")
        transaction.write_joint_state_mask(
            reset_mask,
            np.array([7, 8], dtype=np.int32),
            np.array([6, 7], dtype=np.int32),
            torch.tensor([[0.0, 0.0], [0.1, 0.2], [0.3, 0.4], [0.0, 0.0]]),
            torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]),
            term_name="joints",
        )

    assert len(backend.tensor_reset_calls) == 1
    committed_rows, qpos, qvel, randomization = backend.tensor_reset_calls[0]
    torch.testing.assert_close(committed_rows, torch.tensor([1, 2]))
    assert randomization is None
    np.testing.assert_allclose(qpos[:, :7], np.tile(backend.qpos[:7], (2, 1)), rtol=0, atol=1e-7)
    np.testing.assert_allclose(qpos[:, 7:], [[0.1, 0.2], [0.3, 0.4]], rtol=0, atol=1e-7)
    np.testing.assert_allclose(qvel[:, :6], np.tile(backend.qvel[:6], (2, 1)), rtol=0, atol=1e-7)
    np.testing.assert_allclose(qvel[:, 6:], [[1.0, 2.0], [3.0, 4.0]], rtol=0, atol=1e-7)


def test_packed_reset_without_writes_does_not_commit() -> None:
    backend = _TensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))

    with transaction.scoped_device_event_tensor(torch.zeros(4, dtype=torch.bool)):
        pass

    assert backend.tensor_reset_calls == []
    assert not transaction.last_commit_had_writes


def test_packed_reset_commits_selected_rows_with_default_fill() -> None:
    backend = _TensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    reset_mask = torch.tensor([True, False, True, False])
    layout = BackendRootStateLayout(tuple(range(7)), tuple(range(6)))
    root_state = torch.zeros((4, 13), dtype=torch.float32)
    root_state[:, 3] = 1.0
    root_state[:, 0] = torch.tensor([0.1, 0.2, 0.3, 0.4])

    with transaction.scoped_device_event_tensor(reset_mask):
        transaction.write_root_state_tensor(reset_mask, layout, root_state, term_name="root")

    committed_rows, qpos, _qvel, randomization = backend.tensor_reset_calls[0]
    torch.testing.assert_close(committed_rows, torch.tensor([0, 2]))
    assert randomization is None
    np.testing.assert_allclose(qpos[:, 0], [0.1, 0.3], rtol=0, atol=1e-7)


def test_packed_reset_randomization_fails_closed_without_field_support() -> None:
    class _MissingFieldBackend(_TensorResetBackend):
        def get_tensor_capabilities(self):
            return replace(
                super().get_tensor_capabilities(),
                packed_reset_fields=frozenset((RESET_TERM_KP, RESET_TERM_KD, "gravity")),
            )

    backend = _MissingFieldBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    _bind_device_mass(transaction)
    reset_mask = torch.tensor([False, True, False, False])

    with pytest.raises(NotImplementedError, match="does not support packed reset field"):
        with transaction.scoped_device_event_tensor(reset_mask):
            transaction.write_randomization_tensor_mask(
                RESET_TERM_BODY_MASS,
                reset_mask,
                np.array([0], dtype=np.int32),
                torch.zeros((4, 1)),
                term_name="randomize_rigid_body_mass",
            )

    assert backend.tensor_reset_calls == []


def test_owner_tensor_subset_preserves_unwritten_active_rows() -> None:
    backend = _TensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    active_mask = torch.tensor([False, True, True, False])
    owner_mask = torch.tensor([False, True, False, False])

    with pytest.raises(ValueError, match="outside the active mask"):
        with transaction.scoped_device_owner_tensor(active_mask):
            transaction.write_joint_state_mask(
                torch.tensor([True, False, False, False]),
                np.array([7], dtype=np.int32),
                np.array([6], dtype=np.int32),
                torch.zeros((4, 1)),
                torch.zeros((4, 1)),
                term_name="outside_owner",
            )

    with transaction.scoped_device_owner_tensor(active_mask):
        transaction.write_joint_state_mask(
            owner_mask,
            np.array([7], dtype=np.int32),
            np.array([6], dtype=np.int32),
            torch.tensor([[0.0], [0.7], [0.9], [0.0]]),
            torch.tensor([[0.0], [1.7], [1.9], [0.0]]),
            term_name="owner",
        )

    committed_rows, qpos, qvel, randomization = backend.tensor_reset_calls[0]
    torch.testing.assert_close(committed_rows, torch.tensor([1]))
    assert randomization is None
    np.testing.assert_allclose(qpos[:, 7], [0.7], rtol=0, atol=1e-7)
    np.testing.assert_allclose(qvel[:, 6], [1.7], rtol=0, atol=1e-7)


def test_owner_tensor_subset_commits_disjoint_write_union() -> None:
    backend = _TensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    active_mask = torch.tensor([False, True, True, True])

    with transaction.scoped_device_owner_tensor(active_mask):
        for owner_mask, position, velocity in (
            (
                torch.tensor([False, True, False, False]),
                torch.tensor([[0.0], [0.7], [0.0], [0.0]]),
                torch.tensor([[0.0], [1.7], [0.0], [0.0]]),
            ),
            (
                torch.tensor([False, False, False, True]),
                torch.tensor([[0.0], [0.0], [0.0], [0.9]]),
                torch.tensor([[0.0], [0.0], [0.0], [1.9]]),
            ),
        ):
            transaction.write_joint_state_mask(
                owner_mask,
                np.array([7], dtype=np.int32),
                np.array([6], dtype=np.int32),
                position,
                velocity,
                term_name="owner",
            )

    committed_rows, qpos, qvel, randomization = backend.tensor_reset_calls[0]
    torch.testing.assert_close(committed_rows, torch.tensor([1, 3]))
    assert randomization is None
    np.testing.assert_allclose(qpos[:, 7], [0.7, 0.9], rtol=0, atol=1e-7)
    np.testing.assert_allclose(qvel[:, 6], [1.7, 1.9], rtol=0, atol=1e-7)


def test_owner_tensor_subset_without_actual_rows_does_not_commit() -> None:
    backend = _TensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))

    with transaction.scoped_device_owner_tensor(torch.ones(4, dtype=torch.bool)):
        transaction.write_joint_state_mask(
            torch.zeros(4, dtype=torch.bool),
            np.array([7], dtype=np.int32),
            np.array([6], dtype=np.int32),
            torch.zeros((4, 1)),
            torch.zeros((4, 1)),
            term_name="empty_owner",
        )

    assert backend.tensor_reset_calls == []
    assert not transaction.last_commit_had_writes


def test_owner_tensor_subset_empty_motion_write_does_not_commit() -> None:
    backend = _TensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))

    with transaction.scoped_device_owner_tensor(torch.ones(4, dtype=torch.bool)):
        transaction.write_motion_state_tensor(
            torch.zeros(4, dtype=torch.bool),
            BackendRootStateLayout(tuple(range(7)), tuple(range(6))),
            np.array([7], dtype=np.int32),
            np.array([6], dtype=np.int32),
            torch.tensor((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)).repeat(
                4, 1
            ),
            torch.zeros((4, 1)),
            torch.zeros((4, 1)),
            term_name="empty_motion_owner",
        )

    assert backend.tensor_reset_calls == []
    assert not transaction.last_commit_had_writes


def test_owner_tensor_subset_default_reset_commits_only_requested_rows() -> None:
    backend = _TensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    active_mask = torch.tensor([False, True, True, False])
    default_mask = torch.tensor([False, True, False, False])

    with transaction.scoped_device_owner_tensor(active_mask):
        transaction.reset_to_default(default_mask, term_name="owner_default")

    committed_rows, qpos, _qvel, randomization = backend.tensor_reset_calls[0]
    torch.testing.assert_close(committed_rows, torch.tensor([1]))
    assert randomization is None
    np.testing.assert_allclose(qpos[0], backend.qpos, rtol=0, atol=1e-7)


def test_startup_model_field_commit_becomes_reset_baseline() -> None:
    backend = _StartupModelFieldBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    all_rows = torch.arange(backend.num_envs, dtype=torch.int64)
    columns = np.arange(3, dtype=np.int32)
    _, mass_defaults = transaction.bind_body_mass_write(columns, term_name="startup_mass")
    startup_mass = np.asarray(mass_defaults) * 2.0
    _, ipos_defaults = transaction.bind_body_ipos_write(columns, term_name="startup_com")
    startup_ipos = np.broadcast_to(np.asarray(ipos_defaults), (backend.num_envs, 3, 3)) + 0.1

    with transaction.scoped_startup(torch.arange(backend.num_envs)):
        transaction.write_body_mass(
            all_rows,
            columns,
            startup_mass,
            term_name="startup_mass",
        )
        transaction.record_startup_randomization(
            RESET_TERM_BODY_MASS,
            all_rows,
            columns,
            startup_mass,
            term_name="startup_mass",
        )
        transaction.write_body_ipos(all_rows, columns, startup_ipos, term_name="startup_com")
        transaction.record_startup_randomization(
            RESET_TERM_BODY_IPOS,
            all_rows,
            columns,
            startup_ipos,
            term_name="startup_com",
        )

    startup_payload = backend.randomization_calls[-1]
    assert isinstance(startup_payload, TensorResetRandomizationPayload)
    assert startup_payload.body_mass is not None
    assert startup_payload.body_ipos is not None


def test_startup_model_field_baseline_promotion_is_deferred_until_commit() -> None:
    backend = _StartupModelFieldBackend()
    backend.fail_set_state = True
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    rows = torch.arange(backend.num_envs, dtype=torch.int64)
    columns = np.array([1], dtype=np.int32)
    _, selected_defaults = transaction.bind_body_mass_write(columns, term_name="startup_mass")

    with pytest.raises(NotImplementedError, match="reset upload disabled"):
        with transaction.scoped_startup(torch.arange(backend.num_envs)):
            transaction.write_body_mass(
                rows, columns, selected_defaults.clone(), term_name="startup_mass"
            )
            transaction.record_startup_randomization(
                RESET_TERM_BODY_MASS,
                rows,
                columns,
                selected_defaults.clone(),
                term_name="startup_mass",
            )

    # On failure, promotion did not occur: the default remains the backend's
    # per-environment construction table, not the staged startup rows.
    np.testing.assert_array_equal(
        transaction._randomization_defaults[RESET_TERM_BODY_MASS],
        backend.get_reset_term_default(RESET_TERM_BODY_MASS),
    )


@pytest.mark.parametrize(
    ("ids", "error", "match"),
    [
        ([0], TypeError, "must be torch.Tensor"),
        (torch.tensor([[0]], dtype=torch.int64), TypeError, "1-D integer"),
        (torch.tensor([True]), TypeError, "1-D integer"),
        (torch.tensor([-1], dtype=torch.int64), IndexError, "out of range"),
        (torch.tensor([4], dtype=torch.int64), IndexError, "out of range"),
        (torch.tensor([1, 1], dtype=torch.int64), ValueError, "contain duplicates"),
    ],
)
def test_begin_rejects_invalid_environment_ids(ids, error, match: str) -> None:
    with pytest.raises(error, match=match):
        _transaction(_Backend()).begin(ids)


def _bind_device_mass(transaction: ResetStateTransaction) -> None:
    transaction.bind_body_mass_write(np.array([0, 1], dtype=np.int32), term_name="mass")


def test_packed_transaction_commits_state_through_public_composer() -> None:
    backend = _PackedTensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    reset_mask = torch.tensor([False, True, False, True])
    root_state = torch.zeros((4, 13), dtype=torch.float32)
    root_state[:, 3] = 1.0

    with transaction.scoped_device_event_tensor(reset_mask):
        transaction.write_joint_state_mask(
            reset_mask,
            np.array([1], dtype=np.int64),
            np.array([2], dtype=np.int64),
            torch.tensor([[0.0], [0.25], [0.0], [0.5]]),
            torch.tensor([[0.0], [1.5], [0.0], [2.5]]),
            term_name="joint_reset",
        )

    assert len(backend.tensor_reset_calls) == 1
    committed_rows, qpos, qvel, randomization = backend.tensor_reset_calls[0]
    torch.testing.assert_close(committed_rows, torch.tensor([1, 3]))
    torch.testing.assert_close(qpos[:, 1], torch.tensor([0.25, 0.5]))
    torch.testing.assert_close(qvel[:, 2], torch.tensor([1.5, 2.5]))
    assert randomization is None
    assert transaction.last_set_state_timing_ms["packed_reset_total_ms"] >= 0.0


def test_packed_root_state_payloads_remain_contiguous() -> None:
    backend = _RecordingPackedTensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    reset_mask = torch.tensor([False, True, False, True])
    original_qpos = backend.qpos.copy()
    original_qvel = backend.qvel.copy()
    layout = BackendRootStateLayout(tuple(range(7)), tuple(range(6)))
    root_state = torch.zeros((4, 13), dtype=torch.float32)
    root_state[:, 3] = 1.0
    root_state[1, :3] = torch.tensor([0.1, 0.2, 0.3])
    root_state[3, :3] = torch.tensor([0.4, 0.5, 0.6])
    root_state[1, 7:10] = torch.tensor([1.0, 2.0, 3.0])
    root_state[3, 7:10] = torch.tensor([4.0, 5.0, 6.0])

    with transaction.scoped_device_event_tensor(reset_mask):
        transaction.write_root_state_tensor(reset_mask, layout, root_state, term_name="root_reset")

    assert len(backend.state_write_payloads) == 1
    qpos_payload, qvel_payload = backend.state_write_payloads[0]
    assert tuple(qpos_payload.shape) == (2, 7)
    assert tuple(qvel_payload.shape) == (2, 6)
    assert qpos_payload.dtype is torch.float32 and qpos_payload.is_contiguous()
    assert qvel_payload.dtype is torch.float32 and qvel_payload.is_contiguous()
    assert len(backend.tensor_reset_calls) == 1
    _, qpos, qvel, _ = backend.tensor_reset_calls[0]
    torch.testing.assert_close(qpos[:, :3], torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
    torch.testing.assert_close(qvel[:, :3], torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    np.testing.assert_allclose(backend.qpos[[0, 2]], original_qpos[[0, 2]])
    np.testing.assert_allclose(backend.qvel[[0, 2]], original_qvel[[0, 2]])


@pytest.mark.parametrize(
    ("root_state", "error", "match"),
    [
        (torch.zeros((4, 12), dtype=torch.float32), ValueError, "shape"),
        (torch.zeros((4, 13), dtype=torch.float64), TypeError, "contiguous float32"),
        (torch.zeros((13, 4), dtype=torch.float32).T, TypeError, "contiguous float32"),
        (torch.zeros((4, 13), device="meta"), ValueError, "must live on"),
        (torch.full((4, 13), float("nan")), ValueError, "NaN or Inf"),
    ],
)
def test_packed_root_state_rejects_invalid_payloads(root_state, error, match: str) -> None:
    backend = _RecordingPackedTensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    reset_mask = torch.tensor([False, True, False, True])
    layout = BackendRootStateLayout(tuple(range(7)), tuple(range(6)))

    with transaction.scoped_device_event_tensor(reset_mask):
        with pytest.raises(error, match=match):
            transaction.write_root_state_tensor(
                reset_mask, layout, root_state, term_name="invalid_root_reset"
            )

    assert backend.state_write_payloads == []
    assert backend.tensor_reset_calls == []


def test_deferred_reset_payload_validation_fails_closed_at_boundary() -> None:
    backend = _RecordingPackedTensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    transaction.set_deferred_payload_validation(True)
    assert transaction.deferred_payload_validation is True
    reset_mask = torch.tensor([False, True, False, True])
    layout = BackendRootStateLayout(tuple(range(7)), tuple(range(6)))
    root_state = torch.zeros((4, 13), dtype=torch.float32)
    root_state[:, 3] = 0.5

    with transaction.scoped_device_event_tensor(reset_mask):
        transaction.write_root_state_tensor(
            reset_mask, layout, root_state, term_name="deferred_invalid_root"
        )
        assert transaction.tensor_root_payload_valid.item() is False
        with pytest.raises(ValueError, match="non-unit quaternion"):
            transaction.publish_payload_validation_verdicts(term_name="deferred_invalid_root")


def test_deferred_mask_validation_rejects_outside_rows_at_boundary() -> None:
    backend = _RecordingPackedTensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    transaction.set_deferred_mask_validation(True)
    assert transaction.deferred_mask_validation is True
    active_mask = torch.tensor([False, True, True, False])
    owner_mask = torch.tensor([False, True, False, True])
    root_state = torch.zeros((4, 13), dtype=torch.float32)
    root_state[:, 3] = 1.0

    with transaction.scoped_device_owner_tensor(active_mask):
        transaction.write_root_state_tensor(
            owner_mask,
            BackendRootStateLayout(tuple(range(7)), tuple(range(6))),
            root_state,
            term_name="outside_owner_reset",
        )
        assert transaction._mask_outside_rows.item() is True
        with pytest.raises(ValueError, match="outside the active mask"):
            transaction.publish_mask_validation_verdicts()


def test_deferred_mask_validation_skips_completed_no_write_owner_scope() -> None:
    transaction = _transaction(_TensorResetBackend())
    transaction.declare_packed_reset_device(torch.device("cpu"))
    transaction.set_deferred_mask_validation(True)

    with transaction.scoped_device_owner_tensor(torch.ones(4, dtype=torch.bool)):
        pass
        transaction.publish_mask_validation_verdicts()

    assert not transaction.last_commit_had_writes


def test_deferred_mask_validation_does_not_republish_completed_commit() -> None:
    backend = _TensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    transaction.set_deferred_mask_validation(True)
    reset_mask = torch.tensor([False, True, False, True])
    root_state = torch.zeros((4, 13), dtype=torch.float32)
    root_state[:, 3] = 1.0

    with transaction.scoped_device_owner_tensor(torch.ones(4, dtype=torch.bool)):
        transaction.write_root_state_tensor(
            reset_mask,
            BackendRootStateLayout(tuple(range(7)), tuple(range(6))),
            root_state,
            term_name="first_commit",
        )
    assert transaction.last_commit_had_writes
    transaction.publish_mask_validation_verdicts()

    transaction._mask_count.zero_()
    transaction._completed_tensor_has_writes = False
    transaction.publish_mask_validation_verdicts()

    assert len(backend.tensor_reset_calls) == 1


def test_deferred_mask_validation_clears_previous_commit_verdict_on_tensor_begin() -> None:
    transaction = _transaction(_TensorResetBackend())
    transaction.declare_packed_reset_device(torch.device("cpu"))
    transaction.set_deferred_mask_validation(True)
    event_mask = torch.tensor([True, True, False, False])
    root_state = torch.zeros((4, 13), dtype=torch.float32)
    root_state[:, 3] = 1.0

    with transaction.scoped_device_event_tensor(event_mask):
        transaction.write_root_state_tensor(
            event_mask,
            BackendRootStateLayout(tuple(range(7)), tuple(range(6))),
            root_state,
            term_name="first_event_reset",
        )
    assert transaction.last_commit_had_writes

    with transaction.scoped_device_owner_tensor(torch.ones(4, dtype=torch.bool)):
        transaction.publish_mask_validation_verdicts()

    assert not transaction.last_commit_had_writes


def test_deferred_mask_validation_commits_nonempty_owner_subset() -> None:
    backend = _TensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    transaction.set_deferred_mask_validation(True)
    active_mask = torch.tensor([False, True, True, False])
    owner_mask = torch.tensor([False, True, False, False])

    with transaction.scoped_device_owner_tensor(active_mask):
        transaction.write_joint_state_mask(
            owner_mask,
            np.array([7], dtype=np.int32),
            np.array([6], dtype=np.int32),
            torch.tensor([[0.0], [0.7], [0.0], [0.0]]),
            torch.tensor([[0.0], [1.7], [0.0], [0.0]]),
            term_name="deferred_owner",
        )
        transaction.publish_mask_validation_verdicts()

    committed_rows, qpos, _qvel, _randomization = backend.tensor_reset_calls[0]
    torch.testing.assert_close(committed_rows, torch.tensor([1]))
    assert qpos[0, 7] == 0.7


def test_owner_tensor_reuses_persistent_write_mask_scratch() -> None:
    backend = _TensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    scratch = transaction._owner_tensor_write_mask
    assert scratch is not None
    active_mask = torch.tensor([False, True, True, False])
    owner_mask = torch.tensor([False, True, False, False])

    with transaction.scoped_device_owner_tensor(active_mask):
        transaction.write_joint_state_mask(
            owner_mask,
            np.array([7], dtype=np.int32),
            np.array([6], dtype=np.int32),
            torch.tensor([[0.0], [0.7], [0.0], [0.0]]),
            torch.tensor([[0.0], [1.7], [0.0], [0.0]]),
            term_name="first_owner",
        )
        assert transaction._tensor_write_mask is scratch

    with transaction.scoped_device_owner_tensor(active_mask):
        assert transaction._tensor_write_mask is scratch
        assert not scratch.any()
        transaction.publish_mask_validation_verdicts()

    assert len(backend.tensor_reset_calls) == 1
    assert not transaction.last_commit_had_writes


def test_deferred_mask_validation_cannot_change_during_active_transaction() -> None:
    transaction = _transaction(_TensorResetBackend())
    transaction.declare_packed_reset_device(torch.device("cpu"))

    with transaction.scoped_device_owner_tensor(torch.ones(4, dtype=torch.bool)):
        with pytest.raises(RuntimeError, match="during an active reset"):
            transaction.set_deferred_mask_validation(True)


def test_packed_reset_rejects_invalid_mask_layout() -> None:
    backend = _RecordingPackedTensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))

    with pytest.raises(ValueError, match="bool dtype"):
        transaction.begin_tensor(torch.zeros(4, dtype=torch.float32))

    assert backend.state_write_payloads == []
    assert backend.tensor_reset_calls == []


def test_tensor_transaction_reuses_mask_count_carrier() -> None:
    backend = _TensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))

    with transaction.scoped_device_event_tensor(torch.tensor([False, True, False, True])):
        pass

    assert isinstance(transaction._mask_count, torch.Tensor)
    assert transaction._mask_count.dtype is torch.int64
    assert transaction._mask_count.item() == 0


def test_packed_motion_root_state_payloads_remain_contiguous() -> None:
    backend = _RecordingPackedTensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    reset_mask = torch.tensor([False, True, False, True])
    original_qpos = backend.qpos.copy()
    original_qvel = backend.qvel.copy()
    layout = BackendRootStateLayout(tuple(range(7)), tuple(range(6)))
    root_state = torch.zeros((4, 13), dtype=torch.float32)
    root_state[:, 3] = 1.0
    root_state[1, 2] = 0.3
    root_state[3, 2] = 0.6
    joint_position = torch.zeros((4, 2), dtype=torch.float32)
    joint_velocity = torch.zeros((4, 2), dtype=torch.float32)
    joint_position[[1, 3]] = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    joint_velocity[[1, 3]] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    with transaction.scoped_device_event_tensor(reset_mask):
        transaction.write_motion_state_tensor(
            reset_mask,
            layout,
            np.array([7, 8], dtype=np.int64),
            np.array([6, 7], dtype=np.int64),
            root_state,
            joint_position,
            joint_velocity,
            term_name="motion_reset",
        )

    assert len(backend.state_write_payloads) == 2
    root_qpos_payload, root_qvel_payload = backend.state_write_payloads[0]
    joint_qpos_payload, joint_qvel_payload = backend.state_write_payloads[1]
    assert tuple(root_qpos_payload.shape) == (2, 7)
    assert tuple(root_qvel_payload.shape) == (2, 6)
    assert tuple(joint_qpos_payload.shape) == (2, 2)
    assert tuple(joint_qvel_payload.shape) == (2, 2)
    assert all(
        payload.dtype is torch.float32 and payload.is_contiguous()
        for payload in (
            root_qpos_payload,
            root_qvel_payload,
            joint_qpos_payload,
            joint_qvel_payload,
        )
    )
    assert len(backend.tensor_reset_calls) == 1
    _, qpos, qvel, _ = backend.tensor_reset_calls[0]
    torch.testing.assert_close(qpos[:, 2], torch.tensor([0.3, 0.6]))
    torch.testing.assert_close(qpos[:, 7:], torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
    torch.testing.assert_close(qvel[:, 6:], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_allclose(backend.qpos[[0, 2]], original_qpos[[0, 2]])
    np.testing.assert_allclose(backend.qvel[[0, 2]], original_qvel[[0, 2]])


def test_packed_tensor_transaction_commits_mocap_only_through_public_composer() -> None:
    backend = _PackedTensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    transaction.bind_mocap_pose("palm")
    reset_mask = torch.tensor([False, True, False, True])
    poses = np.array(
        ((1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0), (4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0)),
        dtype=np.float32,
    )

    with transaction.scoped_device_event_tensor(reset_mask):
        transaction.write_mocap_pose(
            "palm", torch.tensor([1, 3], dtype=torch.int64), poses, term_name="mocap_reset"
        )

    assert backend.tensor_reset_calls == []
    assert len(backend.tensor_mocap_calls) == 1
    body_name, committed_rows, committed_poses = backend.tensor_mocap_calls[0]
    assert body_name == "palm"
    torch.testing.assert_close(committed_rows, torch.tensor([1, 3]))
    torch.testing.assert_close(committed_poses, torch.as_tensor(poses))
    np.testing.assert_allclose(backend.mocap_poses[[1, 3]], poses, rtol=0.0)
    assert transaction.last_commit_had_writes
    assert transaction.last_set_state_timing_ms["packed_reset_total_ms"] >= 0.0


def test_packed_transaction_owns_dr_merge_and_baselines() -> None:
    backend = _PackedTensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    _bind_device_mass(transaction)

    first_mask = torch.tensor([True, True, False, False])
    with transaction.scoped_device_event_tensor(first_mask):
        transaction.write_randomization_tensor_mask(
            RESET_TERM_BODY_MASS,
            first_mask,
            np.array([0], dtype=np.int32),
            torch.tensor([[7.0], [8.0], [0.0], [0.0]]),
            term_name="randomize_rigid_body_mass",
        )
    second_mask = torch.tensor([False, True, True, False])
    with transaction.scoped_device_event_tensor(second_mask):
        transaction.write_randomization_tensor_mask(
            RESET_TERM_BODY_MASS,
            second_mask,
            np.array([1], dtype=np.int32),
            torch.tensor([[0.0], [9.0], [10.0], [0.0]]),
            term_name="randomize_rigid_body_mass",
        )

    assert len(backend.tensor_reset_calls) == 2
    randomization = backend.tensor_reset_calls[1][3]
    assert isinstance(randomization, TensorResetRandomizationPayload)
    torch.testing.assert_close(
        randomization.body_mass,
        torch.tensor([[8.0, 9.0], [2.0, 10.0]]),
    )


def test_host_commit_promotes_public_packed_reset_baseline() -> None:
    backend = _PackedTensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    rows = torch.arange(backend.num_envs, dtype=torch.int64)
    columns = np.array([0, 1], dtype=np.int32)
    transaction.bind_body_mass_write(columns, term_name="startup_mass")

    with transaction.scoped_startup(rows):
        startup_mass = torch.full((backend.num_envs, 2), 6.0)
        transaction.write_body_mass(rows, columns, startup_mass, term_name="startup_mass")
        transaction.record_startup_randomization(
            RESET_TERM_BODY_MASS,
            rows,
            columns,
            startup_mass,
            term_name="startup_mass",
        )

    reset_mask = torch.tensor([False, True, True, False])
    with transaction.scoped_device_event_tensor(reset_mask):
        transaction.write_randomization_tensor_mask(
            RESET_TERM_BODY_MASS,
            reset_mask,
            np.array([1], dtype=np.int32),
            torch.tensor([[0.0], [9.0], [10.0], [0.0]]),
            term_name="later_reset_mass",
        )

    randomization = backend.tensor_reset_calls[-1][3]
    assert isinstance(randomization, TensorResetRandomizationPayload)
    torch.testing.assert_close(randomization.body_mass[:, 0], torch.tensor([6.0, 6.0]))
    torch.testing.assert_close(randomization.body_mass[:, 1], torch.tensor([9.0, 10.0]))
