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
    """CPU fake exposing the public selected tensor-reset boundary."""

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
        )

    def set_state_tensor(self, env_indices, qpos, qvel, randomization=None) -> dict:
        rows = torch.as_tensor(env_indices, dtype=torch.int64)
        self.tensor_reset_calls.append((rows.clone(), qpos.clone(), qvel.clone(), randomization))
        return {"ok": True}


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


@pytest.mark.parametrize(
    "field,width",
    [
        ("geom_size", 3),
        ("geom_solref", 2),
        ("geom_solimp", 5),
        ("dof_damping", 0),
        ("dof_frictionloss", 0),
    ],
)
def test_manipulation_fields_preserve_unselected_columns_and_bind_once(field, width):
    backend = _ManipulationBackend()
    transaction = _transaction(backend)
    columns = np.array([1], dtype=np.int32)
    _, defaults = getattr(transaction, f"bind_{field}_write")(columns, term_name=field)
    assert not defaults.flags.writeable
    ids = np.array([2, 0], dtype=np.int32)
    values = np.full((2, 1, width) if width else (2, 1), 0.25)
    for _ in range(2):
        with transaction.scoped(ids):
            getattr(transaction, f"write_{field}")(ids, columns, values, term_name=field)
    assert backend.default_calls == 1
    payload = backend.randomization_calls[-1]
    np.testing.assert_array_equal(backend.set_state_calls[-1][0], [0, 2])
    np.testing.assert_array_equal(getattr(payload, field)[:, 1], 0.25)
    np.testing.assert_array_equal(getattr(payload, field)[:, 0], 1.0)


def test_per_world_body_mass_preserves_each_selected_rows_baseline() -> None:
    backend = _PerWorldBodyMassBackend()
    transaction = _transaction(backend)
    columns = np.array([1, 2], dtype=np.int32)
    _, defaults = transaction.bind_body_mass_write(columns, term_name="variant_mass")

    assert defaults.shape == (backend.num_envs, columns.size)
    np.testing.assert_allclose(defaults[:, 0], [10.0, 20.0, 30.0, 40.0])
    np.testing.assert_allclose(defaults[:, 1], [100.0, 200.0, 300.0, 400.0])

    ids = np.array([2, 0], dtype=np.int32)
    with transaction.scoped(ids):
        transaction.write_body_mass(
            ids,
            columns[:1],
            np.full((ids.size, 1), 0.5),
            term_name="variant_mass",
        )

    assert backend.default_calls == 1
    payload = backend.randomization_calls[-1]
    assert payload is not None and payload.body_mass is not None
    np.testing.assert_allclose(payload.body_mass, [[1.0, 0.5, 100.0], [3.0, 0.5, 300.0]])


def test_startup_model_field_commit_becomes_reset_baseline() -> None:
    backend = _StartupModelFieldBackend()
    transaction = _transaction(backend)
    all_rows = np.arange(backend.num_envs, dtype=np.int32)
    columns = np.arange(3, dtype=np.int32)
    _, mass_defaults = transaction.bind_body_mass_write(columns, term_name="startup_mass")
    startup_mass = np.asarray(mass_defaults) * 2.0
    _, ipos_defaults = transaction.bind_body_ipos_write(columns, term_name="startup_com")
    startup_ipos = np.broadcast_to(np.asarray(ipos_defaults), (backend.num_envs, 3, 3)) + 0.1

    with transaction.scoped(torch.arange(backend.num_envs)):
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
    assert startup_payload is not None
    assert startup_payload.body_mass is not None
    assert startup_payload.body_ipos is not None

    selected = np.asarray([2, 0], dtype=np.int32)
    with transaction.scoped(torch.from_numpy(selected.astype(np.int64))):
        transaction.write_body_mass(
            selected,
            columns[:1],
            np.full((selected.size, 1), 0.75),
            term_name="later_reset_mass",
        )

    reset_payload = backend.randomization_calls[-1]
    assert reset_payload is not None and reset_payload.body_mass is not None
    np.testing.assert_allclose(reset_payload.body_mass[:, 0], [0.75, 0.75])
    # The transaction commits rows in sorted order.
    np.testing.assert_allclose(
        reset_payload.body_mass[:, 1:],
        startup_mass[np.sort(selected)][:, 1:],
    )


def test_startup_model_field_baseline_promotion_is_deferred_until_commit() -> None:
    backend = _StartupModelFieldBackend()
    backend.fail_set_state = True
    transaction = _transaction(backend)
    rows = np.arange(backend.num_envs, dtype=np.int32)
    columns = np.array([1], dtype=np.int32)
    _, selected_defaults = transaction.bind_body_mass_write(columns, term_name="startup_mass")

    with pytest.raises(NotImplementedError, match="reset upload disabled"):
        with transaction.scoped(torch.arange(backend.num_envs)):
            transaction.write_body_mass(
                rows, columns, selected_defaults.copy(), term_name="startup_mass"
            )
            transaction.record_startup_randomization(
                RESET_TERM_BODY_MASS,
                rows,
                columns,
                selected_defaults.copy(),
                term_name="startup_mass",
            )

    # On failure, promotion did not occur: the default remains the backend's
    # per-environment construction table, not the staged startup rows.
    np.testing.assert_array_equal(
        transaction._randomization_defaults[RESET_TERM_BODY_MASS],
        backend.get_reset_term_default(RESET_TERM_BODY_MASS),
    )


def test_mocap_pose_is_staged_then_committed_after_generalized_state():
    backend = _ManipulationBackend()
    transaction = _transaction(backend)
    transaction.bind_mocap_pose("palm")
    ids = np.array([2], dtype=np.int32)
    poses = backend.poses[ids].copy()
    poses[:, 0] = 0.7
    with transaction.scoped(ids):
        transaction.write_mocap_pose("palm", ids, poses, term_name="wrist")
        transaction.reset_to_default(ids, term_name="scene")
        np.testing.assert_array_equal(transaction.read_mocap_pose("palm")[ids], poses)
        assert backend.events == []
    assert backend.events == ["state", "mocap"]
    np.testing.assert_array_equal(backend.poses[ids], poses)
    np.testing.assert_array_equal(backend.poses[[0, 1, 3], 0], 0.0)
    assert transaction.last_commit_had_writes


def test_mocap_only_write_does_not_reset_physics_and_abort_discards_pose():
    backend = _ManipulationBackend()
    transaction = _transaction(backend)
    transaction.bind_mocap_pose("palm")
    ids = np.array([1], dtype=np.int32)
    poses = backend.poses[ids].copy()
    poses[:, 0] = 0.4
    with pytest.raises(RuntimeError, match="cancel"):
        with transaction.scoped(ids):
            transaction.write_mocap_pose("palm", ids, poses, term_name="wrist")
            raise RuntimeError("cancel")
    assert backend.events == []
    with transaction.scoped(ids):
        transaction.write_mocap_pose("palm", ids, poses, term_name="wrist")
    assert backend.events == ["mocap"]
    assert backend.default_qpos_calls == 0
    np.testing.assert_array_equal(backend.poses[ids], poses)


def test_invalid_mocap_write_fails_before_any_backend_mutation():
    backend = _ManipulationBackend()
    transaction = _transaction(backend)
    transaction.bind_mocap_pose("palm")
    ids = np.array([0], dtype=np.int32)
    with pytest.raises(ValueError, match="quaternion"):
        with transaction.scoped(ids):
            transaction.reset_to_default(ids, term_name="scene")
            transaction.write_mocap_pose("palm", ids, np.zeros((1, 7)), term_name="bad")
    assert backend.events == []


def test_transaction_is_lazy_and_combines_terms_into_one_commit() -> None:
    backend = _Backend()
    transaction = _transaction(backend)

    with transaction.scoped(torch.tensor([0, 2, 3], dtype=torch.int64)):
        assert transaction.active
    assert backend.default_qpos_calls == 0
    assert backend.init_qvel_calls == 0
    assert backend.set_state_calls == []

    with transaction.scoped(torch.tensor([0, 2, 3], dtype=torch.int64)):
        transaction.reset_to_default(
            np.array([2], dtype=np.int32),
            term_name="first",
        )
        transaction.reset_to_default(
            np.array([3, 0], dtype=np.int32),
            term_name="second",
        )
        assert backend.set_state_calls == []

    assert not transaction.active
    assert backend.default_qpos_calls == 1
    assert backend.init_qvel_calls == 1
    assert len(backend.set_state_calls) == 1
    ids, qpos, qvel = backend.set_state_calls[0]
    np.testing.assert_array_equal(ids, [0, 2, 3])
    np.testing.assert_array_equal(qpos, np.tile(backend.qpos, (3, 1)))
    np.testing.assert_array_equal(qvel, np.tile(backend.qvel, (3, 1)))

    with transaction.scoped(torch.tensor([1], dtype=torch.int64)):
        transaction.reset_to_default(np.array([1], dtype=np.int32), term_name="third")
    assert backend.default_qpos_calls == 1
    assert backend.init_qvel_calls == 1
    assert len(backend.set_state_calls) == 2


def test_transaction_reports_only_committed_dirty_rows_as_writes() -> None:
    backend = _Backend()
    transaction = _transaction(backend)

    assert not transaction.last_commit_had_writes
    with transaction.scoped(torch.tensor([0], dtype=torch.int64)):
        pass
    assert not transaction.last_commit_had_writes

    with transaction.scoped(torch.tensor([0], dtype=torch.int64)):
        transaction.reset_to_default(np.array([0], dtype=np.int32), term_name="dirty")
    assert transaction.last_commit_had_writes

    with pytest.raises(RuntimeError, match="abort"):
        with transaction.scoped(torch.tensor([1], dtype=torch.int64)):
            transaction.reset_to_default(np.array([1], dtype=np.int32), term_name="aborted")
            raise RuntimeError("abort")
    assert not transaction.last_commit_had_writes


def test_exception_aborts_without_backend_mutation_and_next_reset_is_clean() -> None:
    backend = _Backend()
    transaction = _transaction(backend)

    with pytest.raises(RuntimeError, match="term failed"):
        with transaction.scoped(torch.tensor([0, 1], dtype=torch.int64)):
            transaction.reset_to_default(np.array([0], dtype=np.int32), term_name="broken")
            raise RuntimeError("term failed")

    assert not transaction.active
    assert backend.set_state_calls == []

    with transaction.scoped(torch.tensor([1], dtype=torch.int64)):
        transaction.reset_to_default(np.array([1], dtype=np.int32), term_name="healthy")
    assert len(backend.set_state_calls) == 1
    np.testing.assert_array_equal(backend.set_state_calls[0][0], [1])


def test_actuator_gains_compose_with_state_in_one_reset_commit() -> None:
    backend = _Backend()
    transaction = _transaction(backend)
    columns, default_kp, default_kd = transaction.bind_actuator_gain_write(
        np.array([2, 0], dtype=np.int32),
        term_name="pd_gains:robot",
    )
    np.testing.assert_array_equal(columns, [2, 0])
    np.testing.assert_array_equal(default_kp, [30.0, 10.0])
    np.testing.assert_array_equal(default_kd, [3.0, 1.0])

    with transaction.scoped(torch.tensor([0, 2], dtype=torch.int64)):
        transaction.reset_to_default(np.array([0, 2], dtype=np.int32), term_name="default")
        transaction.write_actuator_gains(
            np.array([2, 0], dtype=np.int32),
            columns,
            np.array([[5.0, 6.0], [7.0, 8.0]]),
            np.array([[0.5, 0.6], [0.7, 0.8]]),
            term_name="pd_gains:robot",
        )

    payload = backend.randomization_calls[0]
    assert payload is not None
    np.testing.assert_array_equal(payload.kp, [[8.0, 20.0, 7.0], [6.0, 20.0, 5.0]])
    np.testing.assert_array_equal(payload.kd, [[0.8, 2.0, 0.7], [0.6, 2.0, 0.5]])


def test_actuator_gain_sparse_rows_abort_without_backend_mutation() -> None:
    backend = _Backend()
    transaction = _transaction(backend)
    columns, _, _ = transaction.bind_actuator_gain_write(
        np.array([0], dtype=np.int32),
        term_name="pd_gains:robot",
    )

    with pytest.raises(RuntimeError, match=r"cannot represent sparse rows.*missing env IDs \[1\]"):
        with transaction.scoped(torch.tensor([0, 1], dtype=torch.int64)):
            transaction.reset_to_default(
                np.array([0, 1], dtype=np.int32),
                term_name="default",
            )
            transaction.write_actuator_gains(
                np.array([0], dtype=np.int32),
                columns,
                np.array([[11.0]]),
                np.array([[1.1]]),
                term_name="pd_gains:robot",
            )

    assert backend.set_state_calls == []


def test_joint_writes_initialize_defaults_and_compose_by_column() -> None:
    backend = _Backend()
    transaction = _transaction(backend)

    with transaction.scoped(torch.tensor([0, 2], dtype=torch.int64)):
        transaction.write_joint_state(
            np.array([2, 0], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([[9.0], [8.0]], dtype=np.float32),
            np.array([[-1.0], [-2.0]], dtype=np.float32),
            term_name="robot.write_joint_state_to_sim",
        )

    ids, qpos, qvel = backend.set_state_calls[0]
    np.testing.assert_array_equal(ids, [0, 2])
    np.testing.assert_array_equal(qpos, [[1.0, 8.0, 3.0], [1.0, 9.0, 3.0]])
    np.testing.assert_array_equal(qvel, [[-2.0, 0.0], [-1.0, 0.0]])


def test_root_pose_and_world_velocity_compose_at_nonzero_columns() -> None:
    default_qpos = np.array([99.0, 1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 88.0])
    default_qvel = np.array([77.0, 66.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 55.0])
    backend = _Backend(qpos=default_qpos, qvel=default_qvel)
    transaction = _transaction(backend)
    layout = BackendRootStateLayout(tuple(range(1, 8)), tuple(range(2, 8)))
    half_sqrt = np.sqrt(0.5)
    poses = np.array(
        [
            [10.0, 11.0, 12.0, half_sqrt, 0.0, 0.0, half_sqrt],
            [20.0, 21.0, 22.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    velocities_w = np.array(
        [
            [1.0, 2.0, 3.0, 1.0, 0.0, 0.0],
            [4.0, 5.0, 6.0, 0.0, 1.0, 2.0],
        ]
    )

    with transaction.scoped(torch.tensor([0, 2], dtype=torch.int64)):
        transaction.write_root_pose(
            np.array([2, 0], dtype=np.int32),
            layout,
            poses,
            term_name="root_pose",
        )
        transaction.write_root_velocity(
            np.array([2, 0], dtype=np.int32),
            layout,
            velocities_w,
            term_name="root_velocity",
        )

    assert len(backend.set_state_calls) == 1
    ids, qpos, qvel = backend.set_state_calls[0]
    np.testing.assert_array_equal(ids, [0, 2])
    np.testing.assert_array_equal(qpos[:, [0, 8]], [[99.0, 88.0], [99.0, 88.0]])
    np.testing.assert_allclose(qpos[0, 1:8], poses[1])
    np.testing.assert_allclose(qpos[1, 1:8], poses[0])
    np.testing.assert_array_equal(qvel[:, [0, 1, 8]], [[77.0, 66.0, 55.0]] * 2)
    np.testing.assert_allclose(qvel[0, 2:8], velocities_w[1])
    np.testing.assert_allclose(qvel[1, 2:5], velocities_w[0, :3])
    np.testing.assert_allclose(qvel[1, 5:8], [0.0, -1.0, 0.0], atol=1e-7)


def test_read_root_pose_returns_staged_or_default_pose_without_dirtying() -> None:
    default_qpos = np.array([99.0, 1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 88.0])
    default_qvel = np.zeros(9)
    backend = _Backend(qpos=default_qpos, qvel=default_qvel)
    transaction = _transaction(backend)
    layout = BackendRootStateLayout(tuple(range(1, 8)), tuple(range(2, 8)))

    with transaction.scoped(torch.tensor([0, 2], dtype=torch.int64)):
        transaction.write_root_pose(
            np.array([2], dtype=np.int32),
            layout,
            np.array([[10.0, 11.0, 12.0, 1.0, 0.0, 0.0, 0.0]]),
            term_name="reset_base",
        )
        staged = transaction.read_root_pose(
            np.array([0, 2], dtype=np.int32),
            layout,
            term_name="random_prone_init",
        )
        # Env 2 reflects the earlier term's write; env 0 the backend default.
        np.testing.assert_allclose(staged[0], default_qpos[1:8])
        np.testing.assert_allclose(staged[1], [10.0, 11.0, 12.0, 1.0, 0.0, 0.0, 0.0])
        # The read is a detached copy: mutating it must not leak back.
        staged[1, 0] = -5.0
        reread = transaction.read_root_pose(
            np.array([2], dtype=np.int32),
            layout,
            term_name="random_prone_init",
        )
        assert reread[0, 0] == 10.0

    # Reading alone does not dirty a row: only the written env is committed.
    assert len(backend.set_state_calls) == 1
    ids, _, _ = backend.set_state_calls[0]
    np.testing.assert_array_equal(ids, [2])


def test_read_root_pose_fails_closed_outside_reset_scope() -> None:
    backend = _Backend(
        qpos=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        qvel=np.zeros(6),
    )
    transaction = _transaction(backend)
    layout = BackendRootStateLayout(tuple(range(7)), tuple(range(6)))

    with pytest.raises(RuntimeError, match="requires an active reset event"):
        transaction.read_root_pose(np.array([0], dtype=np.int32), layout, term_name="t")

    with transaction.scoped(torch.tensor([0], dtype=torch.int64)):
        with pytest.raises(ValueError, match="outside the active reset"):
            transaction.read_root_pose(np.array([1], dtype=np.int32), layout, term_name="t")


def test_combined_root_state_uses_staged_pose_for_angular_velocity() -> None:
    backend = _Backend(
        qpos=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        qvel=np.zeros(6),
    )
    transaction = _transaction(backend)
    layout = BackendRootStateLayout(tuple(range(7)), tuple(range(6)))
    half_sqrt = np.sqrt(0.5)
    root_state = np.array(
        [[1.0, 2.0, 3.0, half_sqrt, 0.0, 0.0, half_sqrt, 4.0, 5.0, 6.0, 1.0, 0.0, 0.0]]
    )

    with transaction.scoped(torch.tensor([1], dtype=torch.int64)):
        transaction.write_root_state(
            np.array([1], dtype=np.int32),
            layout,
            root_state,
            term_name="root_state",
        )

    _, qpos, qvel = backend.set_state_calls[0]
    np.testing.assert_allclose(qpos[0], root_state[0, :7])
    np.testing.assert_allclose(qvel[0, :3], root_state[0, 7:10])
    np.testing.assert_allclose(qvel[0, 3:6], [0.0, -1.0, 0.0], atol=1e-7)


@pytest.mark.parametrize(
    ("root_state", "error", "match"),
    [
        (np.zeros((1, 12)), ValueError, "root state.*expected"),
        (np.zeros((1, 13), dtype=np.int32), TypeError, "root state.*floating"),
        (np.full((1, 13), np.nan), ValueError, "root state.*NaN or Inf"),
        (
            np.array([[0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            ValueError,
            "root quaternion must be unit length",
        ),
    ],
)
def test_root_state_values_fail_closed(root_state, error, match: str) -> None:
    backend = _Backend(
        qpos=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        qvel=np.zeros(6),
    )
    transaction = _transaction(backend)
    layout = BackendRootStateLayout(tuple(range(7)), tuple(range(6)))
    with pytest.raises(error, match=match):
        with transaction.scoped(torch.tensor([0], dtype=torch.int64)):
            transaction.write_root_state(
                np.array([0], dtype=np.int32),
                layout,
                root_state,
                term_name="bad_root",
            )
    assert backend.set_state_calls == []


def test_root_layout_bounds_and_reset_scope_fail_closed() -> None:
    backend = _Backend(
        qpos=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        qvel=np.zeros(6),
    )
    transaction = _transaction(backend)
    out_of_bounds = BackendRootStateLayout(tuple(range(1, 8)), tuple(range(6)))

    with pytest.raises(IndexError, match="root qpos indices out of range"):
        with transaction.scoped(torch.tensor([0], dtype=torch.int64)):
            transaction.write_root_pose(
                np.array([0], dtype=np.int32),
                out_of_bounds,
                np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
                term_name="bad_layout",
            )

    valid = BackendRootStateLayout(tuple(range(7)), tuple(range(6)))
    with pytest.raises(ValueError, match="root-pose mutation outside the active reset"):
        with transaction.scoped(torch.tensor([0], dtype=torch.int64)):
            transaction.write_root_pose(
                np.array([1], dtype=np.int32),
                valid,
                np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
                term_name="outside",
            )


@pytest.mark.parametrize(
    ("position", "velocity", "error", "match"),
    [
        (np.zeros((1, 2)), np.zeros((1, 1)), ValueError, "joint position.*expected"),
        (np.zeros((1, 1)), np.zeros((2, 1)), ValueError, "joint velocity.*expected"),
        (np.zeros((1, 1), dtype=np.int32), np.zeros((1, 1)), TypeError, "must be floating"),
        (np.full((1, 1), np.nan), np.zeros((1, 1)), ValueError, "NaN or Inf"),
    ],
)
def test_joint_write_values_fail_closed(position, velocity, error, match: str) -> None:
    transaction = _transaction(_Backend())
    with pytest.raises(error, match=match):
        with transaction.scoped(torch.tensor([0], dtype=torch.int64)):
            transaction.write_joint_state(
                np.array([0], dtype=np.int32),
                np.array([1], dtype=np.int32),
                np.array([0], dtype=np.int32),
                position,
                velocity,
                term_name="joint_term",
            )


def test_mutation_must_stay_inside_active_reset() -> None:
    transaction = _transaction(_Backend())
    with transaction.scoped(torch.tensor([1, 2], dtype=torch.int64)):
        with pytest.raises(ValueError, match="outside the active reset.*3"):
            transaction.reset_to_default(np.array([3], dtype=np.int32), term_name="bad")

    with pytest.raises(RuntimeError, match="requires an active reset event"):
        transaction.reset_to_default(np.array([1], dtype=np.int32), term_name="late")


@pytest.mark.parametrize(
    ("ids", "error", "match"),
    [
        ([0], TypeError, "must be torch.Tensor"),
        (torch.tensor([[0]], dtype=torch.int64), TypeError, "1-D integer"),
        (torch.tensor([True]), TypeError, "1-D integer"),
        (np.array([-1], dtype=np.int32), IndexError, "out of range"),
        (np.array([4], dtype=np.int32), IndexError, "out of range"),
        (np.array([1, 1], dtype=np.int32), ValueError, "contain duplicates"),
    ],
)
def test_begin_rejects_invalid_environment_ids(ids, error, match: str) -> None:
    with pytest.raises(error, match=match):
        _transaction(_Backend()).begin(ids)


@pytest.mark.parametrize(
    ("field", "value", "error", "match"),
    [
        ("qpos", [1.0], TypeError, "default qpos.*np.ndarray"),
        ("qpos", np.zeros((1, 1)), ValueError, "default qpos.*expected 1-D"),
        ("qpos", np.array([1], dtype=np.int32), TypeError, "default qpos.*floating"),
        ("qpos", np.array([np.nan]), ValueError, "default qpos.*NaN or Inf"),
        ("qvel", np.array([np.inf]), ValueError, "initial qvel.*NaN or Inf"),
    ],
)
def test_backend_default_state_contract_fails_at_mutation_boundary(
    field: str,
    value,
    error,
    match: str,
) -> None:
    kwargs = {field: value}
    transaction = _transaction(_Backend(**kwargs))
    with pytest.raises(error, match=match):
        with transaction.scoped(torch.tensor([0], dtype=torch.int64)):
            transaction.reset_to_default(
                np.array([0], dtype=np.int32),
                term_name="reset_scene_to_default",
            )
    assert not transaction.active


def test_missing_default_and_set_state_capabilities_name_term_and_backend() -> None:
    missing = type("MissingBackend", (), {"num_envs": 1, "backend_type": "missing"})()
    transaction = ResetStateTransaction(cast(SimBackend, missing))
    with pytest.raises(
        NotImplementedError,
        match="EventManager term 'reset_scene_to_default'.*default qpos.*backend 'missing'",
    ):
        with transaction.scoped(torch.tensor([0], dtype=torch.int64)):
            transaction.reset_to_default(
                np.array([0], dtype=np.int32),
                term_name="reset_scene_to_default",
            )

    backend = _Backend(fail_set_state=True)
    transaction = _transaction(backend)
    with pytest.raises(
        NotImplementedError,
        match="SimBackend.set_state.*reset_scene_to_default.*backend 'fake'",
    ):
        with transaction.scoped(torch.tensor([0], dtype=torch.int64)):
            transaction.reset_to_default(
                np.array([0], dtype=np.int32),
                term_name="reset_scene_to_default",
            )
    assert not transaction.active


def test_tensor_motion_state_write_combines_root_and_joint_boundaries() -> None:
    transaction = _transaction(_TensorResetBackend())
    transaction.declare_packed_reset_device(torch.device("cpu"))
    rows = torch.tensor([1, 3], dtype=torch.int64)
    layout = BackendRootStateLayout(tuple(range(7)), tuple(range(6)))
    root_state = torch.tensor(
        (
            (0.1, 0.2, 0.31, 1.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.01, 0.0, 0.0),
            (0.3, -0.2, 0.32, 0.0, 1.0, 0.0, 0.0, -0.05, 0.0, 0.0, -0.01, 0.0, 0.0),
        ),
        dtype=torch.float32,
    )
    position = torch.tensor(((0.2, -0.3), (0.4, 0.5)), dtype=torch.float32)
    velocity = torch.tensor(((0.6, -0.7), (-0.8, 0.9)), dtype=torch.float32)

    with transaction.scoped_device_event_tensor(rows):
        transaction.write_motion_state_tensor(
            rows,
            layout,
            np.array([7, 8], dtype=np.int32),
            np.array([6, 7], dtype=np.int32),
            root_state,
            position,
            velocity,
            term_name="motion_owner",
        )

    backend = transaction._backend  # noqa: SLF001 - scoped test owns the fake
    assert len(backend.tensor_reset_calls) == 1
    committed_rows, qpos, qvel, committed_randomization = backend.tensor_reset_calls[0]
    assert committed_randomization is None
    torch.testing.assert_close(committed_rows, rows)
    expected_qpos = torch.zeros((2, 9), dtype=torch.float32)
    expected_qpos[:, :7] = root_state[:, :7]
    expected_qpos[:, 7:] = position
    expected_qvel = torch.zeros((2, 8), dtype=torch.float32)
    expected_qvel[:, :6] = root_state[:, 7:]
    expected_qvel[:, 6:] = velocity
    torch.testing.assert_close(qpos, expected_qpos)
    torch.testing.assert_close(qvel, expected_qvel)


def test_tensor_motion_state_write_fails_closed() -> None:
    transaction = _transaction(_TensorResetBackend())
    transaction.declare_packed_reset_device(torch.device("cpu"))
    rows = torch.tensor([1], dtype=torch.int64)
    layout = BackendRootStateLayout(tuple(range(7)), tuple(range(6)))
    root_state = torch.tensor([[0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0]])
    position = torch.tensor([[0.1]], dtype=torch.float32)
    velocity = torch.tensor([[0.2]], dtype=torch.float32)

    with pytest.raises(ValueError, match="outside the active tensor reset"):
        with transaction.scoped_device_event_tensor(torch.tensor([2], dtype=torch.int64)):
            transaction.write_motion_state_tensor(
                rows,
                layout,
                np.array([7], dtype=np.int32),
                np.array([6], dtype=np.int32),
                root_state,
                position,
                velocity,
                term_name="motion_owner",
            )

    bad_root = root_state.clone()
    bad_root[0, 3] = 0.5
    with pytest.raises(ValueError, match="quaternion"):
        with transaction.scoped_device_event_tensor(rows):
            transaction.write_motion_state_tensor(
                rows,
                layout,
                np.array([7], dtype=np.int32),
                np.array([6], dtype=np.int32),
                bad_root,
                position,
                velocity,
                term_name="motion_owner",
            )

    nonfinite_position = position.clone()
    nonfinite_position[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        with transaction.scoped_device_event_tensor(rows):
            transaction.write_motion_state_tensor(
                rows,
                layout,
                np.array([7], dtype=np.int32),
                np.array([6], dtype=np.int32),
                root_state,
                nonfinite_position,
                velocity,
                term_name="motion_owner",
            )

    nonfinite_root = root_state.clone()
    nonfinite_root[0, 0] = float("inf")
    with pytest.raises(ValueError, match="root state contains NaN or Inf"):
        with transaction.scoped_device_event_tensor(rows):
            transaction.write_motion_state_tensor(
                rows,
                layout,
                np.array([7], dtype=np.int32),
                np.array([6], dtype=np.int32),
                nonfinite_root,
                position,
                velocity,
                term_name="motion_owner",
            )


def test_tensor_motion_state_success_stays_device_resident(monkeypatch: pytest.MonkeyPatch) -> None:
    transaction = _transaction(_TensorResetBackend())
    transaction.declare_packed_reset_device(torch.device("cpu"))
    rows = torch.tensor([1], dtype=torch.int64)
    layout = BackendRootStateLayout(tuple(range(7)), tuple(range(6)))
    root_state = torch.tensor([[0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0]])
    position = torch.tensor([[0.1]], dtype=torch.float32)
    velocity = torch.tensor([[0.2]], dtype=torch.float32)
    scalar_conversions = 0
    original_bool = torch.Tensor.__bool__

    def counted_bool(self: torch.Tensor) -> bool:
        nonlocal scalar_conversions
        scalar_conversions += 1
        return original_bool(self)

    monkeypatch.setattr(torch.Tensor, "__bool__", counted_bool)
    with transaction.scoped_device_event_tensor(rows):
        transaction.write_motion_state_tensor(
            rows,
            layout,
            np.array([7], dtype=np.int32),
            np.array([6], dtype=np.int32),
            root_state,
            position,
            velocity,
            term_name="motion_owner",
        )

    assert scalar_conversions == 0


def test_tensor_reset_commit_carries_randomization_payload() -> None:
    backend = _TensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    rows = torch.tensor([1, 3], dtype=torch.int64)
    transaction.bind_gravity_write(term_name="gravity_owner")

    with transaction.scoped_device_event_tensor(rows):
        transaction.write_gravity(
            rows.detach().cpu().numpy(),
            np.array([[0.0, 0.0, -9.5], [0.0, 0.0, -10.0]], dtype=np.float64),
            term_name="gravity_owner",
        )

    assert len(backend.tensor_reset_calls) == 1
    committed_rows, _qpos, _qvel, randomization = backend.tensor_reset_calls[0]
    torch.testing.assert_close(committed_rows, rows)
    assert randomization is not None
    np.testing.assert_allclose(randomization.gravity, [[0.0, 0.0, -9.5], [0.0, 0.0, -10.0]])


def test_tensor_reset_commit_rejects_randomization_without_capability() -> None:
    class NoRandomizationBackend(_TensorResetBackend):
        def get_tensor_capabilities(self):
            capabilities = super().get_tensor_capabilities()
            # unisim validates device DR implies reset_randomization.
            return replace(
                capabilities,
                reset_randomization=False,
                device_reset_randomization=False,
            )

    transaction = _transaction(NoRandomizationBackend())
    transaction.declare_packed_reset_device(torch.device("cpu"))
    rows = torch.tensor([0], dtype=torch.int64)
    transaction.bind_gravity_write(term_name="gravity_owner")

    with pytest.raises(NotImplementedError, match="does not support reset randomization"):
        with transaction.scoped_device_event_tensor(rows):
            transaction.write_gravity(
                rows.detach().cpu().numpy(),
                np.array([[0.0, 0.0, -9.8]], dtype=np.float64),
                term_name="gravity_owner",
            )


def _bind_device_mass(transaction: ResetStateTransaction) -> None:
    transaction.bind_body_mass_write(np.array([0, 1], dtype=np.int32), term_name="mass")


def test_tensor_reset_commit_stages_device_body_mass_payload() -> None:
    backend = _TensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    _bind_device_mass(transaction)
    rows = torch.tensor([1, 3], dtype=torch.int64)

    with transaction.scoped_device_event_tensor(rows):
        transaction.write_randomization_tensor(
            RESET_TERM_BODY_MASS,
            rows,
            np.array([0], dtype=np.int32),
            torch.tensor([[7.0], [8.0]]),
            term_name="randomize_rigid_body_mass",
        )

    assert len(backend.tensor_reset_calls) == 1
    _, _, _, randomization = backend.tensor_reset_calls[0]
    assert isinstance(randomization, TensorResetRandomizationPayload)
    torch.testing.assert_close(
        randomization.body_mass,
        torch.tensor([[7.0, 4.0], [8.0, 4.0]]),
    )
    assert randomization.kp is None


def test_tensor_reset_randomization_carries_committed_rows_forward() -> None:
    backend = _TensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    _bind_device_mass(transaction)

    with transaction.scoped_device_event_tensor(torch.tensor([0, 1], dtype=torch.int64)):
        transaction.write_randomization_tensor(
            RESET_TERM_BODY_MASS,
            torch.tensor([0, 1], dtype=torch.int64),
            np.array([0], dtype=np.int32),
            torch.tensor([[7.0], [8.0]]),
            term_name="randomize_rigid_body_mass",
        )
    with transaction.scoped_device_event_tensor(torch.tensor([1, 2], dtype=torch.int64)):
        transaction.write_randomization_tensor(
            RESET_TERM_BODY_MASS,
            torch.tensor([1, 2], dtype=torch.int64),
            np.array([1], dtype=np.int32),
            torch.tensor([[9.0], [10.0]]),
            term_name="randomize_rigid_body_mass",
        )

    assert len(backend.tensor_reset_calls) == 2
    randomization = backend.tensor_reset_calls[1][3]
    assert isinstance(randomization, TensorResetRandomizationPayload)
    # Row 1 keeps the first reset's committed column-0 value; untouched
    # column rows fall back to the backend default table.
    torch.testing.assert_close(
        randomization.body_mass,
        torch.tensor([[8.0, 9.0], [2.0, 10.0]]),
    )


def test_tensor_reset_randomization_merges_column_disjoint_terms() -> None:
    backend = _TensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    rows = torch.tensor([1, 3], dtype=torch.int64)

    with transaction.scoped_device_event_tensor(rows):
        transaction.write_randomization_tensor(
            RESET_TERM_KP,
            rows,
            np.array([0], dtype=np.int32),
            torch.tensor([[11.0], [12.0]]),
            term_name="pd_gains_lower",
        )
        transaction.write_randomization_tensor(
            RESET_TERM_KP,
            rows,
            np.array([2], dtype=np.int32),
            torch.tensor([[31.0], [32.0]]),
            term_name="pd_gains_upper",
        )

    randomization = backend.tensor_reset_calls[0][3]
    assert isinstance(randomization, TensorResetRandomizationPayload)
    # Untouched middle column keeps the backend default gain.
    torch.testing.assert_close(
        randomization.kp,
        torch.tensor([[11.0, 20.0, 31.0], [12.0, 20.0, 32.0]]),
    )
    assert randomization.kd is None


def test_tensor_reset_device_randomization_requires_declared_capability() -> None:
    class NoDeviceRandomizationBackend(_TensorResetBackend):
        def get_tensor_capabilities(self):
            capabilities = super().get_tensor_capabilities()
            return replace(capabilities, device_reset_randomization=False)

    backend = NoDeviceRandomizationBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    _bind_device_mass(transaction)
    rows = torch.tensor([0], dtype=torch.int64)

    with pytest.raises(NotImplementedError, match="device_reset_randomization"):
        with transaction.scoped_device_event_tensor(rows):
            transaction.write_randomization_tensor(
                RESET_TERM_BODY_MASS,
                rows,
                np.array([0], dtype=np.int32),
                torch.tensor([[7.0]]),
                term_name="randomize_rigid_body_mass",
            )
    assert backend.tensor_reset_calls == []


def test_tensor_reset_rejects_mixing_host_and_device_randomization() -> None:
    backend = _TensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    transaction.bind_gravity_write(term_name="gravity_owner")
    _bind_device_mass(transaction)
    rows = torch.tensor([0], dtype=torch.int64)

    with pytest.raises(NotImplementedError, match="device-staged"):
        with transaction.scoped_device_event_tensor(rows):
            transaction.write_randomization_tensor(
                RESET_TERM_BODY_MASS,
                rows,
                np.array([0], dtype=np.int32),
                torch.tensor([[7.0]]),
                term_name="randomize_rigid_body_mass",
            )
            transaction.write_gravity(
                np.array([0], dtype=np.int32),
                np.array([[0.0, 0.0, -9.5]]),
                term_name="gravity_owner",
            )
    assert backend.tensor_reset_calls == []


def test_host_randomization_commit_invalidates_device_committed_baseline() -> None:
    backend = _TensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    _bind_device_mass(transaction)
    rows = torch.tensor([0, 1], dtype=torch.int64)

    with transaction.scoped_device_event_tensor(rows):
        transaction.write_randomization_tensor(
            RESET_TERM_BODY_MASS,
            rows,
            np.array([0], dtype=np.int32),
            torch.tensor([[7.0], [8.0]]),
            term_name="randomize_rigid_body_mass",
        )
    # A host-path commit is newer than the device baseline: the next
    # device-staged reset must rebuild from backend defaults.
    with transaction.scoped(torch.tensor([0], dtype=torch.int64)):
        transaction.write_body_mass(
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([[99.0]]),
            term_name="host_mass",
        )
    with transaction.scoped_device_event_tensor(rows):
        transaction.write_randomization_tensor(
            RESET_TERM_BODY_MASS,
            rows,
            np.array([1], dtype=np.int32),
            torch.tensor([[9.0], [10.0]]),
            term_name="randomize_rigid_body_mass",
        )

    randomization = backend.tensor_reset_calls[-1][3]
    assert isinstance(randomization, TensorResetRandomizationPayload)
    torch.testing.assert_close(
        randomization.body_mass,
        torch.tensor([[2.0, 9.0], [2.0, 10.0]]),
    )


def test_tensor_reset_without_randomization_writes_skips_payload_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _TensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    # Binding allocates the host dirty masks; with no host randomization
    # writes the tensor commit must not build a NumPy payload.
    transaction.bind_gravity_write(term_name="gravity_owner")

    def _fail_payload_build() -> None:
        raise AssertionError("host randomization payload build must not run")

    monkeypatch.setattr(transaction, "_tensor_randomization_payload", _fail_payload_build)
    rows = torch.tensor([1], dtype=torch.int64)
    layout = BackendRootStateLayout(tuple(range(7)), tuple(range(6)))
    root_state = torch.tensor(
        [[0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0]],
        dtype=torch.float32,
    )

    with transaction.scoped_device_event_tensor(rows):
        transaction.write_root_state_tensor(rows, layout, root_state, term_name="root_owner")

    assert len(backend.tensor_reset_calls) == 1
    assert backend.tensor_reset_calls[0][3] is None
