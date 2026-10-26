"""Upstream-derived NumPy tests for Manager-Based reset event terms."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from unisim.backend.base import (
    PublicStateWidths,
    RootStateLayout,
    SelectedResetPublication,
    SimBackend,
    TensorDataPlane,
    TensorExecution,
    TensorLifecycleCapabilities,
    TensorProcessTopology,
)
from unisim.dr.types import (
    INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA,
    INTERVAL_TERM_BODY_FORCE,
    INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA,
    INTERVAL_TERM_BODY_TORQUE,
    RESET_TERM_BODY_INERTIA,
    RESET_TERM_BODY_IPOS,
    RESET_TERM_BODY_MASS,
    RESET_TERM_DOF_ARMATURE,
    RESET_TERM_GEOM_FRICTION,
    RESET_TERM_GRAVITY,
    RESET_TERM_KD,
    RESET_TERM_KP,
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    TensorResetRandomizationPayload,
)

from unilab.base.entity import EntityCfg, EntityScene, SceneTensorReadSpec
from unilab.base.reset_selectors import ResetMaskBuffer
from unilab.base.reset_state import ResetStateTransaction
from unilab.envs import mdp
from unilab.managers import EventManager, EventTermCfg, SceneEntityCfg
from unilab.managers.rng import ManagerRng
from unilab.managers.types import ManagerBasedRlEnv

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


def _interval_op(plan: IntervalRandomizationPlan, term: str):
    return next((op for op in plan.ops if op.term == term), None)


class _CaptureEntity:
    def __init__(self, default_root_state: np.ndarray) -> None:
        self._default_root_state = default_root_state
        self._default_root_state_tensor: torch.Tensor | None = None
        self.writes: list[tuple[torch.Tensor, torch.Tensor]] = []

    @property
    def data(self) -> SimpleNamespace:
        return SimpleNamespace(
            default_root_state=self.default_root_state,
        )

    def default_root_state(self, device: torch.device) -> torch.Tensor:
        if (
            self._default_root_state_tensor is None
            or self._default_root_state_tensor.device != device
        ):
            self._default_root_state_tensor = torch.from_numpy(
                np.array(self._default_root_state, copy=True)
            ).to(device=device)
        return self._default_root_state_tensor

    def write_root_state_tensor_to_sim(
        self,
        root_state: torch.Tensor,
        reset_mask: torch.Tensor,
    ) -> None:
        root = root_state.clone()
        self.writes.append((root, reset_mask.clone()))


class _CaptureScene:
    def __init__(self, entity: _CaptureEntity, env_origins: np.ndarray) -> None:
        self.entities = {"robot": entity}
        self._env_origins = env_origins

    def __getitem__(self, name: str) -> _CaptureEntity:
        return self.entities[name]

    def env_origins(self, device: torch.device) -> torch.Tensor:
        return torch.from_numpy(self._env_origins).to(device=device)


def _capture_env(
    *,
    seed: int = 17,
    default_root_state: np.ndarray | None = None,
    env_origins: np.ndarray | None = None,
) -> tuple[ManagerBasedRlEnv, _CaptureEntity]:
    if default_root_state is None:
        default_root_state = np.zeros((3, 13), dtype=np.float32)
        default_root_state[:, 3] = 1.0
    default_root_state.setflags(write=False)
    entity = _CaptureEntity(default_root_state)
    if env_origins is None:
        env_origins = np.zeros((3, 3), dtype=np.float32)
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=3,
            device=torch.device("cpu"),
            rng=ManagerRng.seeded(seed),
            scene=_CaptureScene(entity, env_origins),
            event_term_mask_buffer=ResetMaskBuffer(3, torch.device("cpu")),
        ),
    )
    return env, entity


def test_uniform_root_state_applies_pinned_pose_velocity_and_origin_semantics() -> None:
    half_sqrt = np.float32(np.sqrt(0.5))
    defaults = np.zeros((3, 13), dtype=np.float32)
    defaults[:, :3] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    defaults[:, 3:7] = [half_sqrt, 0.0, 0.0, half_sqrt]
    defaults[:, 7:13] = np.arange(18, dtype=np.float32).reshape(3, 6)
    origins = np.asarray(
        [[100.0, 0.0, 0.0], [0.0, 200.0, 0.0], [0.0, 0.0, 300.0]],
        dtype=np.float32,
    )
    env, entity = _capture_env(default_root_state=defaults, env_origins=origins)
    reset_mask = torch.tensor([False, True, True])

    mdp.reset_root_state_uniform(
        env,
        torch.tensor([1, 2], dtype=torch.int64),
        pose_range={
            "x": (1.0, 1.0),
            "y": (-2.0, -2.0),
            "z": (0.5, 0.5),
            "roll": (np.pi / 2.0, np.pi / 2.0),
        },
        velocity_range={
            "x": (0.1, 0.1),
            "y": (0.2, 0.2),
            "z": (0.3, 0.3),
            "roll": (0.4, 0.4),
            "pitch": (0.5, 0.5),
            "yaw": (0.6, 0.6),
        },
    )

    assert len(entity.writes) == 1
    root_state, written_mask = entity.writes[0]
    torch.testing.assert_close(written_mask, reset_mask)
    np.testing.assert_allclose(
        root_state[[0, 2], :3],
        defaults[[0, 2], :3] + [1.0, -2.0, 0.5] + origins[[0, 2]],
    )
    np.testing.assert_allclose(
        root_state[[0, 2], 3:7],
        [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        root_state[[0, 2], 7:13],
        defaults[[0, 2], 7:13] + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    np.testing.assert_array_equal(defaults[:, :3], [[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test_generic_event_randomization_fails_closed_without_rng() -> None:
    env, _entity = _capture_env()
    env.rng = None
    with pytest.raises(RuntimeError, match="Manager-owned Torch generator"):
        mdp.reset_root_state_uniform(
            env, torch.tensor([0, 1], dtype=torch.int64), {"x": (0.1, 0.2)}
        )


def test_physics_scene_gravity_diagnostic_names_public_term() -> None:
    env, _backend, _transaction = _transaction_env()
    manager = EventManager(
        {
            "randomize_physics_scene_gravity": EventTermCfg(
                func=mdp.randomize_physics_scene_gravity,
                mode="reset",
                params={
                    "gravity_distribution_params": (
                        [0.0, 0.0, -10.0],
                        [0.0, 0.0, -9.0],
                    ),
                    "operation": "abs",
                },
            )
        },
        env,
    )
    env.rng = None

    with pytest.raises(
        RuntimeError,
        match="term 'randomize_physics_scene_gravity' requires",
    ):
        manager.apply(
            "reset", env_ids=torch.tensor([0, 1], dtype=torch.int64), global_env_step_count=0
        )


def test_uniform_root_state_uses_env_rng_and_preserves_single_env_shape() -> None:
    pose_range = {"x": (-1.0, 1.0), "yaw": (-0.5, 0.5)}
    velocity_range = {"x": (-2.0, 2.0), "roll": (-0.25, 0.25)}
    left, left_entity = _capture_env(seed=9)
    right, right_entity = _capture_env(seed=9)
    reset_mask = torch.tensor([False, True, False])
    env_ids = torch.tensor([1], dtype=torch.int64)

    mdp.reset_root_state_uniform(left, env_ids, pose_range, velocity_range)
    mdp.reset_root_state_uniform(right, env_ids, pose_range, velocity_range)

    left_state, left_mask = left_entity.writes[0]
    right_state, right_mask = right_entity.writes[0]
    assert left_state.shape == (3, 13)
    torch.testing.assert_close(left_mask, reset_mask)
    torch.testing.assert_close(right_mask, reset_mask)
    np.testing.assert_array_equal(left_state, right_state)


def test_uniform_root_state_none_ids_targets_all_environments() -> None:
    env, entity = _capture_env()

    reset_mask = torch.ones(3, dtype=torch.bool)
    mdp.reset_root_state_uniform(env, torch.arange(3, dtype=torch.int64), pose_range={})

    root_state, written_mask = entity.writes[0]
    torch.testing.assert_close(written_mask, reset_mask)
    assert root_state.shape == (3, 13)


class _Backend:
    backend_type = "fake"
    num_envs = 3
    num_actuators = 3

    def __init__(
        self,
        *,
        root_layout_supported: bool = True,
        gain_supported: bool = True,
        randomization_supported: bool = True,
        interval_velocity_supported: bool = True,
        interval_angular_velocity_supported: bool = False,
        interval_force_supported: bool = False,
        interval_torque_supported: bool = False,
        per_world_defaults: bool = False,
    ) -> None:
        self.root_layout_supported = root_layout_supported
        self.gain_supported = gain_supported
        self.randomization_supported = randomization_supported
        self.interval_velocity_supported = interval_velocity_supported
        self.interval_angular_velocity_supported = interval_angular_velocity_supported
        self.interval_force_supported = interval_force_supported
        self.interval_torque_supported = interval_torque_supported
        self.per_world_defaults = per_world_defaults
        self.default_qpos = np.asarray([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
        self.init_qvel = np.zeros(6)
        self.set_state_calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.randomization_calls: list[TensorResetRandomizationPayload | None] = []
        self.body_pos = np.zeros((self.num_envs, 1, 3))
        self.body_quat = np.zeros((self.num_envs, 1, 4))
        self.body_quat[:, :, 0] = 1.0
        self.body_velocity = np.zeros((self.num_envs, 1, 3))
        self.dof_pos = np.zeros((self.num_envs, 3))
        self.dof_vel = np.zeros((self.num_envs, 3))
        canonical_friction = np.array([[0.5, 0.01, 0.001], [0.7, 0.02, 0.002], [0.9, 0.03, 0.003]])
        canonical_armature = np.array([0.0] * 6 + [1.0, 2.0, 3.0])
        if per_world_defaults:
            self.body_mass = np.asarray([[10.0 + env_id] for env_id in range(self.num_envs)])
            self.body_ipos = np.asarray(
                [[0.01 * env_id, 0.0, 0.0] for env_id in range(self.num_envs)]
            )[:, None, :]
            self.gravity = np.asarray(
                [[0.0, 0.0, -9.81 - 0.1 * env_id] for env_id in range(self.num_envs)]
            )
            self.dof_armature = np.stack(
                [canonical_armature * (1.0 + 0.1 * env_id) for env_id in range(self.num_envs)]
            )
            self.geom_friction = np.stack(
                [canonical_friction * (1.0 + 0.1 * env_id) for env_id in range(self.num_envs)]
            )
            self.default_kp = np.stack(
                [
                    np.asarray([10.0, 20.0, 30.0]) * (1.0 + 0.1 * env_id)
                    for env_id in range(self.num_envs)
                ]
            )
            self.default_kd = np.stack(
                [
                    np.asarray([1.0, 2.0, 3.0]) * (1.0 + 0.1 * env_id)
                    for env_id in range(self.num_envs)
                ]
            )
        else:
            self.body_mass = np.array([10.0])
            self.body_ipos = np.array([[0.0, 0.0, 0.0]])
            self.gravity = np.array([0.0, 0.0, -9.81])
            self.dof_armature = canonical_armature
            self.geom_friction = canonical_friction
            self.default_kp = np.asarray([10.0, 20.0, 30.0])
            self.default_kd = np.asarray([1.0, 2.0, 3.0])
        self.interval_plans: list[IntervalRandomizationPlan] = []

    def get_body_ids(self, names) -> np.ndarray:
        if tuple(names) != ("base",):
            raise KeyError(names)
        return np.asarray([0], dtype=np.int32)

    def get_root_state_layout(self, root_body_name: str) -> RootStateLayout:
        if not self.root_layout_supported:
            raise NotImplementedError("fake fixed-base entity has no free-root layout")
        if root_body_name != "base":
            raise ValueError(root_body_name)
        return RootStateLayout(tuple(range(7)), tuple(range(6)))

    def get_default_qpos(self) -> np.ndarray:
        return self.default_qpos.copy()

    def get_init_qvel(self) -> np.ndarray:
        return self.init_qvel.copy()

    def get_public_state_widths(self) -> PublicStateWidths:
        return PublicStateWidths(len(self.default_qpos), len(self.init_qvel))

    def get_dof_pos(self) -> np.ndarray:
        return np.zeros((self.num_envs, 3))

    def get_dof_vel(self) -> np.ndarray:
        return np.zeros((self.num_envs, 3))

    def get_default_dof_pos(self) -> np.ndarray:
        return np.zeros(3)

    def get_joint_range(self) -> np.ndarray:
        return np.tile([-1.0, 1.0], (3, 1))

    def get_joint_dof_indices(self, names) -> np.ndarray:
        table = {"j0": 6, "j1": 7, "j2": 8}
        return np.asarray([table[name] for name in names], dtype=np.int32)

    def get_joint_dof_pos_indices(self, names) -> np.ndarray:
        table = {"j0": 0, "j1": 1, "j2": 2}
        return np.asarray([table[name] for name in names], dtype=np.int32)

    def get_joint_dof_vel_indices(self, names) -> np.ndarray:
        return self.get_joint_dof_pos_indices(names)

    def get_joint_state_qpos_indices(self, names) -> np.ndarray:
        return self.get_joint_dof_pos_indices(names)

    def get_joint_state_qvel_indices(self, names) -> np.ndarray:
        return self.get_joint_dof_pos_indices(names)

    def get_state_views(self, fields, device=None) -> dict[str, torch.Tensor]:
        target = torch.device(device) if device is not None else torch.device("cpu")
        return {
            field: torch.from_numpy(getattr(self, source).copy()).to(target)
            for field, source in (("qpos", "dof_pos"), ("qvel", "dof_vel"))
            if field in fields
        }

    def get_sensor_view(self, name, device=None) -> torch.Tensor:
        sources = {
            "track_pos_w_base": self.body_pos[:, 0],
            "track_quat_w_base": self.body_quat[:, 0],
            "track_linvel_w_base": self.body_velocity[:, 0],
            "track_angvel_w_base": self.body_velocity[:, 0],
        }
        target = torch.device(device) if device is not None else torch.device("cpu")
        return torch.from_numpy(np.array(sources[name], copy=True)).to(target)

    def get_geom_names(self) -> tuple[str, ...]:
        return ("floor", "foot", "base_geom")

    def get_actuator_names(self) -> tuple[str, ...]:
        return ("a0", "a1", "a2")

    def get_actuator_joint_names(self) -> tuple[str, ...]:
        return ("j0", "j1", "j2")

    def get_actuator_ctrl_range(self) -> np.ndarray:
        return np.tile([-1.0, 1.0], (self.num_actuators, 1))

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        terms: set[str] = set((RESET_TERM_KP, RESET_TERM_KD)) if self.gain_supported else set()
        if self.randomization_supported:
            terms.update(
                (
                    RESET_TERM_BODY_MASS,
                    RESET_TERM_BODY_IPOS,
                    RESET_TERM_DOF_ARMATURE,
                    RESET_TERM_GEOM_FRICTION,
                    RESET_TERM_GRAVITY,
                )
            )
        interval_terms = {
            term
            for term, supported in (
                (INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA, self.interval_velocity_supported),
                (
                    INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA,
                    self.interval_angular_velocity_supported,
                ),
                (INTERVAL_TERM_BODY_FORCE, self.interval_force_supported),
                (INTERVAL_TERM_BODY_TORQUE, self.interval_torque_supported),
            )
            if supported
        }
        return DomainRandomizationCapabilities(
            supported_reset_terms=frozenset(terms),
            supported_interval_terms=frozenset(interval_terms),
        )

    def get_tensor_capabilities(self) -> TensorLifecycleCapabilities:
        from unisim.backend.packed_reset import PACKED_RESET_MODEL_FIELDS

        packed_fields = frozenset(
            self.get_dr_capabilities().supported_reset_terms & PACKED_RESET_MODEL_FIELDS
        )
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
            packed_reset_fields=packed_fields,
        )

    def bind_packed_reset(self, *, device=None):
        from unisim.backend.packed_reset import PackedResetTransaction

        return PackedResetTransaction(self, device=device)

    def get_reset_term_default(self, term: str) -> np.ndarray:
        if not self.get_dr_capabilities().supports_reset_term(term):
            raise NotImplementedError(term)
        values = {
            RESET_TERM_KP: self.default_kp,
            RESET_TERM_KD: self.default_kd,
            RESET_TERM_BODY_MASS: self.body_mass,
            RESET_TERM_BODY_IPOS: self.body_ipos,
            RESET_TERM_DOF_ARMATURE: self.dof_armature,
            RESET_TERM_GEOM_FRICTION: self.geom_friction,
            RESET_TERM_GRAVITY: self.gravity,
        }
        try:
            return np.array(values[term], copy=True)
        except KeyError as exc:
            raise NotImplementedError(term) from exc

    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        self.interval_plans.append(plan)

    def set_state_tensor(
        self,
        env_indices: torch.Tensor,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        randomization=None,
    ) -> None:
        self.set_state_calls.append(
            (
                env_indices.detach().cpu().numpy().copy(),
                qpos.detach().cpu().numpy().copy(),
                qvel.detach().cpu().numpy().copy(),
            )
        )
        self.randomization_calls.append(randomization)

    def set_state_tensor_mask(
        self,
        reset_mask: torch.Tensor,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        randomization=None,
        **_kwargs,
    ) -> None:
        """Apply the public full-width mask contract through selected rows."""
        rows = torch.nonzero(reset_mask, as_tuple=False).flatten()
        selected_randomization = randomization
        if randomization is not None:
            selected_randomization = type(randomization)(
                **{
                    name: value.index_select(0, rows)
                    for name, value in vars(randomization).items()
                    if value is not None and value.shape[:1] == reset_mask.shape
                }
            )
        return self.set_state_tensor(
            rows,
            qpos.index_select(0, rows),
            qvel.index_select(0, rows),
            selected_randomization,
        )


def _transaction_env(
    *,
    root_layout_supported: bool = True,
    gain_supported: bool = True,
    randomization_supported: bool = True,
    interval_velocity_supported: bool = True,
    interval_angular_velocity_supported: bool = False,
    interval_force_supported: bool = False,
    interval_torque_supported: bool = False,
    per_world_defaults: bool = False,
    body_names: tuple[str, ...] | None = ("base",),
    rng_seed: int = 5,
    step_dt: float = 0.02,
) -> tuple[ManagerBasedRlEnv, _Backend, ResetStateTransaction]:
    backend = _Backend(
        root_layout_supported=root_layout_supported,
        gain_supported=gain_supported,
        randomization_supported=randomization_supported,
        interval_velocity_supported=interval_velocity_supported,
        interval_angular_velocity_supported=interval_angular_velocity_supported,
        interval_force_supported=interval_force_supported,
        interval_torque_supported=interval_torque_supported,
        per_world_defaults=per_world_defaults,
    )
    transaction = ResetStateTransaction(cast(SimBackend, backend))
    scene = EntityScene(
        {
            "robot": EntityCfg(
                root_body_name="base",
                joint_names=("j0", "j1", "j2"),
                body_names=body_names,
                geom_names=("floor", "foot", "base_geom"),
                actuator_names=("a0", "a1", "a2"),
            )
        },
        cast(SimBackend, backend),
        reset_state=transaction,
    )
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=backend.num_envs,
            device=torch.device("cpu"),
            rng=ManagerRng.seeded(rng_seed),
            scene=scene,
            step_dt=step_dt,
            backend=backend,
            event_term_mask_buffer=ResetMaskBuffer(backend.num_envs, torch.device("cpu")),
        ),
    )
    return env, backend, transaction


def test_uniform_root_state_composes_in_one_reset_transaction_commit() -> None:
    env, backend, transaction = _transaction_env()
    transaction.declare_packed_reset_device(torch.device("cpu"))
    reset_mask = torch.tensor([True, False, True])

    env_ids = torch.tensor([0, 2], dtype=torch.int64)
    with transaction.scoped_device_event_tensor(reset_mask):
        mdp.reset_scene_to_default(env, env_ids)
        mdp.reset_root_state_uniform(
            env,
            env_ids,
            pose_range={"x": (1.0, 1.0)},
            velocity_range={"x": (0.5, 0.5)},
        )
        assert backend.set_state_calls == []

    assert len(backend.set_state_calls) == 1
    committed_rows, qpos, qvel = backend.set_state_calls[0]
    np.testing.assert_array_equal(committed_rows, [0, 2])
    np.testing.assert_array_equal(qpos[0], backend.default_qpos + [1.0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(qpos[1], backend.default_qpos + [1.0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(qvel[0], backend.init_qvel + [0.5, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(qvel[1], backend.init_qvel + [0.5, 0, 0, 0, 0, 0])


def test_pd_gains_reset_randomization_commits_in_one_packed_transaction() -> None:
    env, backend, transaction = _transaction_env(rng_seed=11)
    manager = EventManager(
        {
            "randomize_pd": EventTermCfg(
                func=mdp.pd_gains,
                mode="reset",
                params={"kp_range": (2.0, 2.0), "kd_range": (3.0, 3.0)},
            )
        },
        env,
    )
    transaction.declare_packed_reset_device(torch.device("cpu"))
    reset_mask = torch.tensor([True, False, True])

    with transaction.scoped_device_event_tensor(reset_mask):
        manager.apply(
            "reset",
            env_ids=torch.nonzero(reset_mask, as_tuple=False).flatten(),
            global_env_step_count=0,
        )
        assert backend.set_state_calls == []

    assert len(backend.set_state_calls) == 1
    committed_rows, _, _ = backend.set_state_calls[0]
    np.testing.assert_array_equal(committed_rows, [0, 2])
    payload = backend.randomization_calls[0]
    assert payload is not None
    np.testing.assert_allclose(payload.kp.numpy(), [[20.0, 40.0, 60.0]] * 2, rtol=1e-6)
    np.testing.assert_allclose(payload.kd.numpy(), [[3.0, 6.0, 9.0]] * 2, rtol=1e-6)


@pytest.mark.parametrize(
    ("func", "params", "field", "expected"),
    [
        (
            mdp.randomize_rigid_body_mass,
            {
                "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
                "mass_distribution_params": (1.5, 1.5),
                "operation": "scale",
                "recompute_inertia": False,
            },
            "body_mass",
            np.asarray([[15.0]]),
        ),
        (
            mdp.randomize_rigid_body_com,
            {
                "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
                "com_range": {"x": (0.1, 0.1)},
            },
            "body_ipos",
            np.asarray([[[0.1, 0.0, 0.0]]]),
        ),
        (
            mdp.randomize_physics_scene_gravity,
            {
                "gravity_distribution_params": (
                    [0.0, 0.0, -10.0],
                    [0.0, 0.0, -10.0],
                ),
                "operation": "abs",
            },
            "gravity",
            np.asarray([[0.0, 0.0, -10.0]]),
        ),
        (
            mdp.dof_armature,
            {"ranges": (2.0, 2.0), "operation": "scale"},
            "dof_armature",
            np.asarray([[0.0] * 6 + [2.0, 4.0, 6.0]]),
        ),
        (
            mdp.geom_friction,
            {"ranges": (2.0, 2.0), "operation": "scale"},
            "geom_friction",
            np.asarray([[[1.0, 0.01, 0.001], [1.4, 0.02, 0.002], [1.8, 0.03, 0.003]]]),
        ),
    ],
)
def test_reset_dr_terms_compose_in_one_packed_transaction(
    func,
    params: dict[str, Any],
    field: str,
    expected: np.ndarray,
) -> None:
    env, backend, transaction = _transaction_env()
    manager = EventManager({"field": EventTermCfg(func=func, mode="reset", params=params)}, env)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    reset_mask = torch.tensor([True, False, True])

    with transaction.scoped_device_event_tensor(reset_mask):
        manager.apply(
            "reset",
            env_ids=torch.nonzero(reset_mask, as_tuple=False).flatten(),
            global_env_step_count=0,
        )
        assert backend.set_state_calls == []

    assert len(backend.set_state_calls) == 1
    committed_rows, _, _ = backend.set_state_calls[0]
    np.testing.assert_array_equal(committed_rows, [0, 2])
    payload = backend.randomization_calls[0]
    assert payload is not None
    values = getattr(payload, field)
    assert values is not None
    np.testing.assert_allclose(
        values.numpy(),
        np.broadcast_to(expected, (2, *expected.shape[1:])),
        rtol=1e-6,
    )


def test_randomize_body_mass_inertia_reset_commits_scaled_mass_and_inertia() -> None:
    env, backend, transaction = _mass_inertia_env()
    manager = _mass_inertia_manager(env)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    reset_mask = torch.tensor([True, False, True])

    with transaction.scoped_device_event_tensor(reset_mask):
        manager.apply(
            "reset",
            env_ids=torch.nonzero(reset_mask, as_tuple=False).flatten(),
            global_env_step_count=0,
        )
        assert backend.set_state_calls == []

    assert len(backend.set_state_calls) == 1
    committed_rows, _, _ = backend.set_state_calls[0]
    np.testing.assert_array_equal(committed_rows, [0, 2])
    payload = backend.randomization_calls[0]
    assert payload is not None
    mass = payload.body_mass.numpy()
    inertia = payload.body_inertia.numpy()
    assert mass.shape == (2, 2)
    assert inertia.shape == (2, 2, 3)
    np.testing.assert_allclose(mass[:, 0], 0.0, atol=1e-7)
    np.testing.assert_allclose(inertia[:, 0, :], 0.0, atol=1e-7)
    scales = mass[:, 1] / 10.0
    assert np.all(scales >= 0.95) and np.all(scales <= 1.05)
    np.testing.assert_allclose(inertia[:, 1, :], scales[:, None] * [0.1, 0.2, 0.3], rtol=1e-6)

    # The scale factor is sampled once per environment and replayed on later
    # resets: recommitting row 0 reproduces the identical scaled values.
    with transaction.scoped_device_event_tensor(torch.tensor([True, False, False])):
        manager.apply(
            "reset", env_ids=torch.tensor([0], dtype=torch.int64), global_env_step_count=1
        )
    replay = backend.randomization_calls[-1]
    assert replay is not None
    np.testing.assert_allclose(replay.body_mass.numpy(), mass[[0]], rtol=1e-6)
    np.testing.assert_allclose(replay.body_inertia.numpy(), inertia[[0]], rtol=1e-6)


@pytest.mark.parametrize(
    ("cfg_kwargs", "match"),
    [
        ({"mode": "startup"}, "only supports mode='reset'"),
        ({"params": {"kp_range": (2.0, 1.0), "kd_range": (1.0, 1.0)}}, "minimum"),
    ],
)
def test_pd_gains_invalid_config_fails_during_manager_construction(
    cfg_kwargs: dict[str, Any],
    match: str,
) -> None:
    env, backend, _ = _transaction_env(rng_seed=11)
    values: dict[str, Any] = {
        "mode": "reset",
        "params": {"kp_range": (1.0, 1.0), "kd_range": (1.0, 1.0)},
    }
    values.update(cfg_kwargs)

    with pytest.raises((ValueError, NotImplementedError), match=match):
        EventManager({"gain": EventTermCfg(func=mdp.pd_gains, **values)}, env)
    assert backend.set_state_calls == []


def test_pd_gains_missing_backend_capability_fails_during_manager_construction() -> None:
    env, backend, _ = _transaction_env(gain_supported=False, rng_seed=11)
    cfg = EventTermCfg(
        func=mdp.pd_gains,
        mode="reset",
        params={"kp_range": (1.0, 1.0), "kd_range": (1.0, 1.0)},
    )

    with pytest.raises(
        NotImplementedError,
        match="pd_gains:robot.*actuator gain randomization.*backend 'fake'",
    ):
        EventManager({"gain": cfg}, env)
    assert backend.set_state_calls == []


def test_pd_gains_commits_masked_packed_model_fields() -> None:
    env, backend, transaction = _transaction_env(rng_seed=11)
    manager = EventManager(
        {
            "gain": EventTermCfg(
                func=mdp.pd_gains,
                mode="reset",
                params={
                    "kp_range": (2.0, 2.0),
                    "kd_range": (3.0, 3.0),
                    "operation": "abs",
                },
            )
        },
        env,
    )
    reset_mask = torch.tensor([False, True, True])
    transaction.declare_packed_reset_device(torch.device("cpu"))

    with transaction.scoped_device_event_tensor(reset_mask):
        manager.apply(
            "reset",
            env_ids=torch.nonzero(reset_mask, as_tuple=False).flatten(),
            global_env_step_count=0,
        )

    assert len(backend.set_state_calls) == 1
    rows, _qpos, _qvel = backend.set_state_calls[0]
    np.testing.assert_array_equal(rows, np.asarray([1, 2]))
    payload = backend.randomization_calls[0]
    assert payload is not None
    assert payload.kp is not None
    assert payload.kd is not None
    np.testing.assert_allclose(payload.kp.numpy(), [[2.0] * 3, [2.0] * 3])
    np.testing.assert_allclose(payload.kd.numpy(), [[3.0] * 3, [3.0] * 3])


def test_model_field_capability_gaps_fail_cold() -> None:
    env, backend, _ = _transaction_env(randomization_supported=False)

    with pytest.raises(
        NotImplementedError,
        match="dof_armature:robot.*dof_armature randomization.*backend 'fake'",
    ):
        EventManager(
            {
                "armature": EventTermCfg(
                    func=mdp.dof_armature,
                    mode="reset",
                    params={"ranges": (0.9, 1.1)},
                )
            },
            env,
        )
    assert backend.set_state_calls == []


@pytest.mark.parametrize(
    ("func", "params", "match"),
    [
        (mdp.geom_friction, {"ranges": (0.5, 1.0), "axes": [3]}, "invalid axes"),
        (mdp.dof_armature, {"ranges": (-1.0, -0.5)}, "produced negative"),
    ],
)
def test_model_field_invalid_requests_fail_explicitly(func, params, match: str) -> None:
    env, backend, transaction = _transaction_env()
    if func is mdp.geom_friction:
        with pytest.raises(ValueError, match=match):
            EventManager({"field": EventTermCfg(func=func, mode="reset", params=params)}, env)
    else:
        manager = EventManager({"field": EventTermCfg(func=func, mode="reset", params=params)}, env)
        transaction.declare_packed_reset_device(torch.device("cpu"))
        reset_mask = torch.tensor([True, False, True])
        with pytest.raises(ValueError, match=match):
            with transaction.scoped_device_event_tensor(reset_mask):
                manager.apply(
                    "reset",
                    env_ids=torch.nonzero(reset_mask, as_tuple=False).flatten(),
                    global_env_step_count=0,
                )
    assert backend.set_state_calls == []


@pytest.mark.parametrize(
    ("func", "params", "match"),
    [
        (
            mdp.randomize_rigid_body_mass,
            {
                "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
                "mass_distribution_params": (0.9, 1.1),
                "operation": "scale",
            },
            "recompute_inertia=True",
        ),
        (
            mdp.randomize_rigid_body_mass,
            {
                "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
                "mass_distribution_params": (0.9, 1.1),
                "operation": "scale",
                "recompute_inertia": False,
            },
            "body_mass randomization.*unsupported",
        ),
    ],
)
def test_mass_randomization_capability_gaps_fail_during_construction(
    func, params: dict[str, Any], match: str
) -> None:
    env, backend, _ = _transaction_env(randomization_supported=False)
    with pytest.raises(NotImplementedError, match=match):
        EventManager(
            {"mass": EventTermCfg(func=func, mode="reset", params=params)},
            env,
        )
    assert backend.set_state_calls == []


def test_startup_mass_and_com_sample_once_and_persist_across_resets() -> None:
    env, backend, transaction = _transaction_env(rng_seed=61)
    uniform_calls = 0
    original_uniform = env.rng.uniform

    def counted_uniform(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal uniform_calls
        uniform_calls += 1
        return original_uniform(*args, **kwargs)

    env.rng.uniform = counted_uniform  # type: ignore[method-assign]
    asset_cfg = SceneEntityCfg("robot", body_names=("base",))
    manager = EventManager(
        {
            "mass": EventTermCfg(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": asset_cfg,
                    "mass_distribution_params": (1.5, 1.5),
                    "operation": "scale",
                    "recompute_inertia": False,
                },
            ),
            "com": EventTermCfg(
                func=mdp.randomize_rigid_body_com,
                mode="startup",
                params={
                    "asset_cfg": asset_cfg,
                    "com_range": {"x": (0.1, 0.1), "z": (-0.2, -0.2)},
                },
            ),
        },
        env,
    )
    all_rows = torch.arange(backend.num_envs, dtype=torch.int64)
    transaction.declare_packed_reset_device(torch.device("cpu"))

    with transaction.scoped_startup(all_rows):
        manager.apply(mode="startup", env_ids=all_rows)
        assert backend.set_state_calls == []

    assert len(backend.set_state_calls) == 1
    startup = backend.randomization_calls[0]
    assert startup is not None
    assert startup.body_mass is not None
    assert startup.body_ipos is not None
    np.testing.assert_allclose(startup.body_mass, [[15.0]] * backend.num_envs)
    np.testing.assert_allclose(startup.body_ipos, [[[0.1, 0.0, -0.2]]] * backend.num_envs)
    assert uniform_calls == 2, "mass and COM must each sample exactly once"

    selected_mask = torch.tensor([False, True, True])
    with transaction.scoped_device_event_tensor(selected_mask):
        # Startup terms are not reset terms. A masked state reset must not
        # resample or recommit their model fields.
        mdp.reset_scene_to_default(env, torch.tensor([1, 2], dtype=torch.int64))
    assert len(backend.set_state_calls) == 2
    replay = backend.randomization_calls[-1]
    assert replay is None or replay.body_mass is None
    assert replay is None or replay.body_ipos is None
    assert uniform_calls == 2


@pytest.mark.parametrize("mode", ["interval", "step"])
def test_startup_capable_model_field_terms_reject_other_modes(mode: str) -> None:
    env, backend, _ = _transaction_env()

    with pytest.raises(
        NotImplementedError,
        match="only supports mode='reset/startup'",
    ):
        EventManager(
            {
                "mass": EventTermCfg(
                    func=mdp.randomize_rigid_body_mass,
                    mode=mode,
                    params={
                        "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
                        "mass_distribution_params": (0.9, 1.1),
                        "operation": "scale",
                        "recompute_inertia": False,
                    },
                )
            },
            env,
        )
    with pytest.raises(
        NotImplementedError,
        match="only supports mode='reset/startup'",
    ):
        EventManager(
            {
                "com": EventTermCfg(
                    func=mdp.randomize_rigid_body_com,
                    mode=mode,
                    params={
                        "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
                        "com_range": {"x": (-0.1, 0.1)},
                    },
                )
            },
            env,
        )
    assert backend.set_state_calls == []


def test_apply_body_impulse_lifecycle_stages_sustains_and_expires() -> None:
    env, backend, _ = _transaction_env(rng_seed=31, interval_force_supported=True)
    manager = EventManager(
        {
            "impulse": EventTermCfg(
                func=mdp.apply_body_impulse,
                mode="step",
                params={
                    "force_range": (5.0, 5.0),
                    "torque_range": (0.0, 0.0),
                    "duration_s": (0.05, 0.05),
                    "cooldown_s": (0.06, 0.06),
                },
            )
        },
        env,
    )

    # step_dt=0.02: the pre-sampled 0.06 cooldown triggers on the third tick;
    # the 0.05 duration sustains through the fifth tick and expires on the sixth.
    manager.apply(mode="step", dt=0.02)
    manager.apply(mode="step", dt=0.02)
    assert backend.interval_plans == []

    manager.apply(mode="step", dt=0.02)
    assert len(backend.interval_plans) == 1
    plan = backend.interval_plans[0]
    force_op = _interval_op(plan, INTERVAL_TERM_BODY_FORCE)
    assert force_op is not None
    np.testing.assert_array_equal(force_op.body_ids, [0])
    np.testing.assert_allclose(force_op.payload[:, 0], 5.0)
    assert _interval_op(plan, INTERVAL_TERM_BODY_TORQUE) is None

    manager.apply(mode="step", dt=0.02)
    manager.apply(mode="step", dt=0.02)
    assert len(backend.interval_plans) == 3
    for plan in backend.interval_plans[1:3]:
        force_op = _interval_op(plan, INTERVAL_TERM_BODY_FORCE)
        assert force_op is not None
        np.testing.assert_allclose(force_op.payload[:, 0], 5.0)

    # Expiry stages zeros once (persistent wrench channels clear), then the
    # term stays idle through the next cooldown instead of re-staging.
    manager.apply(mode="step", dt=0.02)
    assert len(backend.interval_plans) == 4
    force_op = _interval_op(backend.interval_plans[3], INTERVAL_TERM_BODY_FORCE)
    assert force_op is not None
    np.testing.assert_allclose(force_op.payload[[1], :], 0.0)
    manager.apply(mode="step", dt=0.02)
    assert len(backend.interval_plans) == 4


def test_apply_body_impulse_offset_generates_cross_torque() -> None:
    env, backend, _ = _tensor_dr_env(device_reset_randomization=True, device=torch.device("cpu"))
    backend.interval_force_supported = True
    backend.interval_torque_supported = True
    env.scene.read_plan = scene_compile = env.scene.compile_tensor_reads(
        env.device,
        (SceneTensorReadSpec(entity="robot", body_names=("base",)),),
    )
    scene_compile.refresh()
    manager = EventManager(
        {
            "impulse": EventTermCfg(
                func=mdp.apply_body_impulse,
                mode="step",
                params={
                    "force_range": (1.0, 1.0),
                    "torque_range": (0.0, 0.0),
                    "duration_s": (1.0, 1.0),
                    "cooldown_s": (0.0, 0.0),
                    "body_point_offset": (0.0, 0.0, 0.1),
                },
            )
        },
        env,
    )

    manager.apply(mode="step", dt=0.02)

    assert len(backend.interval_plans) == 1
    plan = backend.interval_plans[0]
    force_op = _interval_op(plan, INTERVAL_TERM_BODY_FORCE)
    torque_op = _interval_op(plan, INTERVAL_TERM_BODY_TORQUE)
    assert force_op is not None and torque_op is not None
    np.testing.assert_allclose(force_op.payload[:, 0], 1.0)
    # Identity link orientation: offset_w = (0, 0, 0.1), force = (1, 1, 1),
    # cross(offset_w, force) = (-0.1, 0.1, 0).
    np.testing.assert_allclose(
        torque_op.payload[:, 0],
        [[-0.1, 0.1, 0.0]] * 3,
        atol=1e-12,
    )


@requires_cuda
def test_apply_body_impulse_cuda_payload_stays_on_declared_device_plane() -> None:
    env, backend, _ = _tensor_dr_env(device_reset_randomization=True, device=torch.device("cuda"))
    backend.interval_force_supported = True
    backend.interval_torque_supported = True
    manager = EventManager(
        {
            "impulse": EventTermCfg(
                func=mdp.apply_body_impulse,
                mode="step",
                params={
                    "force_range": (1.0, 1.0),
                    "torque_range": (0.25, 0.25),
                    "duration_s": (1.0, 1.0),
                    "cooldown_s": (0.0, 0.0),
                },
            )
        },
        env,
    )

    manager.apply(mode="step", dt=0.02)

    assert len(backend.interval_plans) == 1
    plan = backend.interval_plans[0]
    for term in (INTERVAL_TERM_BODY_FORCE, INTERVAL_TERM_BODY_TORQUE):
        op = _interval_op(plan, term)
        assert op is not None
        assert op.payload.device.type == "cuda", term
        assert op.body_ids.device.type == "cuda", term


def test_apply_body_impulse_reset_clears_active_wrench() -> None:
    env, backend, _ = _transaction_env(rng_seed=43, interval_force_supported=True)
    manager = EventManager(
        {
            "impulse": EventTermCfg(
                func=mdp.apply_body_impulse,
                mode="step",
                params={
                    "force_range": (5.0, 5.0),
                    "torque_range": (0.0, 0.0),
                    "duration_s": (1.0, 1.0),
                    "cooldown_s": (10.0, 10.0),
                },
            )
        },
        env,
    )
    term = manager.get_term_cfg("impulse").func
    term._interval_time_left[:] = 0.0
    manager.apply(mode="step", dt=0.02)
    assert len(backend.interval_plans) == 1

    manager.reset(torch.tensor([1], dtype=torch.int64))
    assert len(backend.interval_plans) == 2
    force_op = _interval_op(backend.interval_plans[1], INTERVAL_TERM_BODY_FORCE)
    assert force_op is not None
    np.testing.assert_allclose(force_op.payload[[1], :], 0.0)

    manager.apply(mode="step", dt=0.02)
    sustained = _interval_op(backend.interval_plans[-1], INTERVAL_TERM_BODY_FORCE)
    assert sustained is not None
    np.testing.assert_allclose(sustained.payload[1, 0], 0.0)
    np.testing.assert_allclose(sustained.payload[[0, 2], 0], 5.0)


@pytest.mark.parametrize(
    ("env_overrides", "params", "cfg_kwargs", "match"),
    [
        ({}, {"force_range": (1.0, 1.0)}, {"mode": "interval"}, "only supports mode='step'"),
        (
            {},
            {
                "force_range": (1.0, 1.0),
                "torque_range": (0.0, 0.0),
                "duration_s": (0.1, 0.1),
            },
            {},
            "missing parameters",
        ),
        (
            {},
            {
                "force_range": (1.0, 1.0),
                "torque_range": (0.0, 0.0),
                "duration_s": (-0.1, 0.1),
                "cooldown_s": (0.0, 0.0),
            },
            {},
            "0 <= min <= max",
        ),
        (
            {"interval_force_supported": False},
            {
                "force_range": (1.0, 1.0),
                "torque_range": (0.0, 0.0),
                "duration_s": (0.1, 0.1),
                "cooldown_s": (0.0, 0.0),
            },
            {},
            "unsupported backend capability",
        ),
        (
            {"interval_force_supported": True, "interval_torque_supported": False},
            {
                "force_range": (1.0, 1.0),
                "torque_range": (0.1, 0.1),
                "duration_s": (0.1, 0.1),
                "cooldown_s": (0.0, 0.0),
            },
            {},
            "interval body torque",
        ),
        (
            {"interval_force_supported": True, "interval_torque_supported": False},
            {
                "force_range": (1.0, 1.0),
                "torque_range": (0.0, 0.0),
                "duration_s": (0.1, 0.1),
                "cooldown_s": (0.0, 0.0),
                "body_point_offset": (0.0, 0.0, 0.1),
            },
            {},
            "interval body torque",
        ),
    ],
)
def test_apply_body_impulse_invalid_configs_fail_closed(
    env_overrides: dict[str, Any],
    params: dict[str, Any],
    cfg_kwargs: dict[str, Any],
    match: str,
) -> None:
    env, backend, _ = _transaction_env(**env_overrides)
    values: dict[str, Any] = {"mode": "step", "params": params}
    values.update(cfg_kwargs)
    if values["mode"] == "interval":
        values["interval_range_s"] = (1.0, 1.0)

    with pytest.raises((ValueError, NotImplementedError), match=match):
        EventManager({"impulse": EventTermCfg(func=mdp.apply_body_impulse, **values)}, env)
    assert backend.interval_plans == []


def test_velocity_push_uses_env_rng_and_interval_subset_plan() -> None:
    env, backend, _ = _transaction_env(rng_seed=19)
    manager = EventManager(
        {
            "push": EventTermCfg(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(1.0, 1.0),
                params={
                    "velocity_range": {
                        "x": (0.2, 0.2),
                        "y": (-0.3, -0.3),
                        "z": (0.4, 0.4),
                    }
                },
            )
        },
        env,
    )
    manager._interval_term_time_left[0][:] = torch.tensor([0.0, 1.0, 0.0])

    manager.apply(mode="interval", dt=0.1)

    assert len(backend.interval_plans) == 1
    plan = backend.interval_plans[0]
    velocity_op = _interval_op(plan, INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA)
    assert velocity_op is not None
    np.testing.assert_array_equal(velocity_op.body_ids, [0])
    np.testing.assert_allclose(
        velocity_op.payload[:, 0],
        [[0.2, -0.3, 0.4], [0.0, 0.0, 0.0], [0.2, -0.3, 0.4]],
    )


def test_velocity_push_dispatches_angular_delta_when_supported() -> None:
    env, backend, _ = _transaction_env(rng_seed=19, interval_angular_velocity_supported=True)
    manager = EventManager(
        {
            "push": EventTermCfg(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(1.0, 1.0),
                params={
                    "velocity_range": {
                        "x": (0.2, 0.2),
                        "yaw": (-0.5, -0.5),
                    }
                },
            )
        },
        env,
    )
    manager._interval_term_time_left[0][:] = torch.tensor([0.0, 1.0, 0.0])

    manager.apply(mode="interval", dt=0.1)

    assert len(backend.interval_plans) == 1
    plan = backend.interval_plans[0]
    linear_op = _interval_op(plan, INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA)
    angular_op = _interval_op(plan, INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA)
    assert linear_op is not None and angular_op is not None
    np.testing.assert_array_equal(linear_op.body_ids, [0])
    np.testing.assert_allclose(
        linear_op.payload[:, 0],
        [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
    )
    np.testing.assert_allclose(
        angular_op.payload[:, 0],
        [[0.0, 0.0, -0.5], [0.0, 0.0, 0.0], [0.0, 0.0, -0.5]],
    )


@pytest.mark.parametrize(
    (
        "velocity_supported",
        "angular_supported",
        "velocity_range",
        "match",
    ),
    [
        (False, False, {"x": (-0.1, 0.1)}, "unsupported backend capability"),
        (True, False, {"yaw": (-0.1, 0.1)}, "unsupported backend capability"),
    ],
)
def test_velocity_push_capability_gaps_fail_during_construction(
    velocity_supported: bool,
    angular_supported: bool,
    velocity_range: dict[str, tuple[float, float]],
    match: str,
) -> None:
    env, backend, _ = _transaction_env(
        interval_velocity_supported=velocity_supported,
        interval_angular_velocity_supported=angular_supported,
    )
    with pytest.raises(NotImplementedError, match=match):
        EventManager(
            {
                "push": EventTermCfg(
                    func=mdp.push_by_setting_velocity,
                    mode="interval",
                    interval_range_s=(1.0, 1.0),
                    params={"velocity_range": velocity_range},
                )
            },
            env,
        )
    assert backend.interval_plans == []


def test_uniform_root_state_fixed_or_mocap_capability_fails_closed() -> None:
    env, backend, transaction = _transaction_env(root_layout_supported=False)

    with pytest.raises(
        NotImplementedError,
        match="reset_root_state_uniform.*entity 'robot'.*fixed-base/mocap.*backend 'fake'",
    ):
        mdp.reset_root_state_uniform(env, torch.tensor([1], dtype=torch.int64), pose_range={})

    assert backend.set_state_calls == []


@pytest.mark.parametrize(
    ("pose_range", "message"),
    [
        ({"x": (np.nan, 1.0)}, "finite"),
        ({"yaw": (1.0, -1.0)}, "minimum exceeds maximum"),
        ({"z": (0.0, 1.0, 2.0)}, r"numeric \(min, max\) pair"),
    ],
)
def test_uniform_root_state_invalid_ranges_fail_before_write(
    pose_range: dict[str, Any], message: str
) -> None:
    env, entity = _capture_env()

    with pytest.raises(ValueError, match=message):
        mdp.reset_root_state_uniform(env, torch.tensor([0], dtype=torch.int64), pose_range)

    assert entity.writes == []


class _BiasEntity:
    def __init__(self, num_envs: int, num_joints: int, *, tensor_control: bool) -> None:
        self.num_joints = num_joints
        self.joint_names = [f"joint_{index}" for index in range(num_joints)]
        self._bias = np.zeros((num_envs, num_joints), dtype=np.float32)
        self._bias_tensor = torch.from_numpy(self._bias)
        self.control_buffer = (
            torch.zeros((num_envs, num_joints), dtype=torch.float32) if tensor_control else None
        )
        self.data = self

    def update_encoder_bias_tensor_mask(self, reset_mask, columns, values) -> None:
        current = self._bias_tensor[:, columns]
        self._bias_tensor[:, columns] = torch.where(reset_mask[:, None], values, current)

    @property
    def encoder_bias(self) -> np.ndarray:
        return self._bias_tensor.numpy()

    def find_joints(self, names: list[str]) -> list[int]:
        return [self.joint_names.index(name) for name in names]


class _BiasScene:
    def __init__(self, entity: _BiasEntity) -> None:
        self._entity = entity

    def __getitem__(self, name: str) -> _BiasEntity:
        if name != "robot":
            raise KeyError(name)
        return self._entity


def _bias_env(
    num_envs: int = 3,
    num_joints: int = 4,
    seed: int = 7,
    *,
    tensor_control: bool = True,
) -> ManagerBasedRlEnv:
    return cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=num_envs,
            device=torch.device("cpu"),
            rng=ManagerRng.seeded(seed),
            scene=_BiasScene(_BiasEntity(num_envs, num_joints, tensor_control=tensor_control)),
        ),
    )


def test_randomize_encoder_bias_samples_selected_joints_within_range() -> None:
    env = _bias_env()
    manager = EventManager(
        {
            "bias": EventTermCfg(
                func=mdp.randomize_encoder_bias,
                mode="reset",
                params={
                    "bias_range": (-0.02, 0.02),
                    "asset_cfg": SceneEntityCfg("robot", joint_ids=[1, 3]),
                },
            )
        },
        env,
    )

    manager.apply("reset", env_ids=torch.tensor([0, 2], dtype=torch.int64), global_env_step_count=0)

    bias = env.scene["robot"].data.encoder_bias
    assert bias.shape == (3, 4)
    np.testing.assert_array_equal(bias[1], np.zeros(4))
    np.testing.assert_array_equal(bias[:, [0, 2]], np.zeros((3, 2)))
    assert np.all(np.abs(bias[[0, 2]][:, [1, 3]]) <= 0.02)
    assert np.any(bias[[0, 2]][:, [1, 3]] != 0.0)


def test_randomize_encoder_bias_fails_closed_without_torch_control_plane() -> None:
    env = _bias_env(tensor_control=False)
    manager = EventManager(
        {
            "bias": EventTermCfg(
                func=mdp.randomize_encoder_bias,
                mode="reset",
                params={
                    "bias_range": (-0.02, 0.02),
                    "asset_cfg": SceneEntityCfg("robot", joint_ids=[1, 3]),
                },
            )
        },
        env,
    )

    with pytest.raises(NotImplementedError, match="Torch control plane"):
        manager.apply(
            "reset", env_ids=torch.tensor([0, 2], dtype=torch.int64), global_env_step_count=0
        )


def test_randomize_encoder_bias_rejects_invalid_cfg_at_construction() -> None:
    env = _bias_env()
    with pytest.raises(NotImplementedError, match="mode='reset'"):
        EventManager(
            {
                "bias": EventTermCfg(
                    func=mdp.randomize_encoder_bias,
                    mode="interval",
                    interval_range_s=(1.0, 1.0),
                    params={
                        "bias_range": (-0.02, 0.02),
                        "asset_cfg": SceneEntityCfg("robot", joint_ids=[1]),
                    },
                )
            },
            env,
        )
    with pytest.raises(ValueError, match="finite pair"):
        EventManager(
            {
                "bias": EventTermCfg(
                    func=mdp.randomize_encoder_bias,
                    mode="reset",
                    params={
                        "bias_range": (0.0, np.nan),
                        "asset_cfg": SceneEntityCfg("robot", joint_ids=[1]),
                    },
                )
            },
            env,
        )
    with pytest.raises(ValueError, match="unknown parameters"):
        EventManager(
            {
                "bias": EventTermCfg(
                    func=mdp.randomize_encoder_bias,
                    mode="reset",
                    params={
                        "bias_range": (-0.02, 0.02),
                        "asset_cfg": SceneEntityCfg("robot", joint_ids=[1]),
                        "bogus": 1.0,
                    },
                )
            },
            env,
        )


def test_manager_model_field_terms_do_not_import_engine_packages() -> None:
    source = inspect.getsource(mdp.events)
    assert "import mujoco" not in source
    assert "import mjbatch" not in source


class _InertiaBackend(_Backend):
    """Fake backend with authoritative world/body inertial default rows."""

    def __init__(
        self,
        *,
        inertia_supported: bool = True,
        per_world_defaults: bool = False,
    ) -> None:
        super().__init__(per_world_defaults=per_world_defaults)
        self.inertia_supported = inertia_supported
        # Row 0 is the world body, matching the compiled MJCF body table.
        self.body_mass = (
            np.asarray([[0.0, 10.0 + 0.5 * env_id] for env_id in range(self.num_envs)])
            if per_world_defaults
            else np.array([0.0, 10.0])
        )
        if per_world_defaults:
            self.body_inertia = np.stack(
                [
                    np.asarray(
                        [
                            [0.0, 0.0, 0.0],
                            [
                                0.1 + 0.01 * env_id,
                                0.2 + 0.01 * env_id,
                                0.3 + 0.01 * env_id,
                            ],
                        ]
                    )
                    for env_id in range(self.num_envs)
                ],
                axis=0,
            )
        else:
            self.body_inertia = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
        self.body_pos = np.zeros((self.num_envs, 2, 3))
        self.body_quat = np.zeros((self.num_envs, 2, 4))
        self.body_quat[:, :, 0] = 1.0
        self.body_velocity = np.zeros((self.num_envs, 2, 3))

    def get_body_ids(self, names) -> np.ndarray:
        if tuple(names) != ("base",):
            raise KeyError(names)
        return np.asarray([1], dtype=np.int32)

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        capabilities = super().get_dr_capabilities()
        if not self.inertia_supported:
            return capabilities
        return DomainRandomizationCapabilities(
            supported_reset_terms=frozenset(
                set(capabilities.supported_reset_terms) | {RESET_TERM_BODY_INERTIA}
            ),
            supported_interval_terms=capabilities.supported_interval_terms,
        )

    def get_reset_term_default(self, term: str) -> np.ndarray:
        if term == RESET_TERM_BODY_INERTIA and self.inertia_supported:
            return self.body_inertia.copy()
        return super().get_reset_term_default(term)


def _mass_inertia_env(
    *,
    inertia_supported: bool = True,
    per_world_defaults: bool = False,
    rng_seed: int = 5,
) -> tuple[ManagerBasedRlEnv, _InertiaBackend, ResetStateTransaction]:
    backend = _InertiaBackend(
        inertia_supported=inertia_supported,
        per_world_defaults=per_world_defaults,
    )
    transaction = ResetStateTransaction(cast(SimBackend, backend))
    scene = EntityScene(
        {
            "robot": EntityCfg(
                root_body_name="base",
                joint_names=("j0", "j1", "j2"),
                body_names=("base",),
                geom_names=("floor", "foot", "base_geom"),
                actuator_names=("a0", "a1", "a2"),
            )
        },
        cast(SimBackend, backend),
        reset_state=transaction,
    )
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=backend.num_envs,
            device=torch.device("cpu"),
            rng=ManagerRng.seeded(rng_seed),
            scene=scene,
            step_dt=0.02,
        ),
    )
    return env, backend, transaction


def _mass_inertia_manager(env: ManagerBasedRlEnv) -> EventManager:
    return EventManager(
        {
            "mass_inertia": EventTermCfg(
                func=mdp.randomize_body_mass_inertia,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
                    "scale_range": (0.95, 1.05),
                },
            )
        },
        env,
    )


def test_randomize_body_mass_inertia_capability_gap_fails_during_construction() -> None:
    env, backend, _ = _mass_inertia_env(inertia_supported=False)
    with pytest.raises(NotImplementedError, match="body_inertia randomization.*unsupported"):
        _mass_inertia_manager(env)
    assert backend.set_state_calls == []


@pytest.mark.parametrize(
    "scale_range",
    [(0.9, 0.1), (0.0, 1.05)],
)
def test_randomize_body_mass_inertia_rejects_invalid_scale_range(scale_range) -> None:
    env, backend, _ = _mass_inertia_env()
    with pytest.raises(ValueError, match="scale_range"):
        EventManager(
            {
                "mass_inertia": EventTermCfg(
                    func=mdp.randomize_body_mass_inertia,
                    mode="reset",
                    params={
                        "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
                        "scale_range": scale_range,
                    },
                )
            },
            env,
        )
    assert backend.set_state_calls == []


class _TensorBackend(_Backend):
    """Fake backend exposing the public selected tensor-reset boundary."""

    def __init__(
        self,
        *,
        device_reset_randomization: bool = False,
        device_plane: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.device_reset_randomization = device_reset_randomization
        self.device_plane = device_plane
        self.tensor_reset_calls: list[tuple[Any, Any, Any, Any]] = []
        view_device = torch.device("cuda" if device_plane else "cpu")
        self.view_device = view_device
        self.body_tensor_views = {
            "pos_w": torch.zeros((self.num_envs, 1, 3), dtype=torch.float32, device=view_device),
            "quat_w": torch.zeros((self.num_envs, 1, 4), dtype=torch.float32, device=view_device),
            "lin_vel_w": torch.zeros(
                (self.num_envs, 1, 3), dtype=torch.float32, device=view_device
            ),
            "ang_vel_w": torch.zeros(
                (self.num_envs, 1, 3), dtype=torch.float32, device=view_device
            ),
        }
        self.body_tensor_views["quat_w"][..., 0] = 1.0

    def get_tensor_capabilities(self) -> TensorLifecycleCapabilities:
        from unisim.backend.packed_reset import PACKED_RESET_MODEL_FIELDS

        packed_fields = frozenset(
            self.get_dr_capabilities().supported_reset_terms & PACKED_RESET_MODEL_FIELDS
        )
        return TensorLifecycleCapabilities(
            execution=TensorExecution.DEVICE_RESIDENT,
            state_views=True,
            state_fields=frozenset({"qpos", "qvel"}),
            sensor_views=True,
            stepping=True,
            selected_reset=True,
            reset_randomization=True,
            tracked_body_views=True,
            process_topology=TensorProcessTopology.IN_PROCESS,
            data_plane=TensorDataPlane.DIRECT,
            stream_event_ownership="fake synchronous stream",
            torch_devices=("cpu",),
            device_reset_randomization=self.device_reset_randomization,
            packed_reset=True,
            packed_reset_fields=packed_fields,
        )

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        capabilities = super().get_dr_capabilities()
        if not self.device_plane:
            return capabilities
        return DomainRandomizationCapabilities(
            supported_reset_terms=capabilities.supported_reset_terms,
            supported_interval_terms=capabilities.supported_interval_terms,
            supported_device_reset_terms=capabilities.supported_reset_terms,
            supported_device_interval_terms=capabilities.supported_interval_terms,
        )

    def set_state_tensor(self, env_indices, qpos, qvel, randomization=None) -> dict:
        self.tensor_reset_calls.append((env_indices, qpos, qvel, randomization))
        return {"ok": True}

    def get_tracked_body_views(self, body_names=None, device=None):
        assert tuple(body_names) == ("base",)
        return SimpleNamespace(body_names=tuple(body_names), **self.body_tensor_views)

    def get_state_views(self, names, device=None):
        target = self.view_device if device is None else torch.device(device)
        return {
            "qpos": torch.zeros((self.num_envs, 3), dtype=torch.float32, device=target),
            "qvel": torch.zeros((self.num_envs, 3), dtype=torch.float32, device=target),
        }

    def get_sensor_view(self, name, device="cpu"):
        field = next(
            value
            for prefix, value in (
                ("track_pos_w_", self.body_tensor_views["pos_w"]),
                ("track_quat_w_", self.body_tensor_views["quat_w"]),
                ("track_linvel_w_", self.body_tensor_views["lin_vel_w"]),
                ("track_angvel_w_", self.body_tensor_views["ang_vel_w"]),
            )
            if name.startswith(prefix)
        )
        return field[:, 0]


def _tensor_dr_env(
    *,
    device_reset_randomization: bool,
    device: torch.device,
) -> tuple[ManagerBasedRlEnv, _TensorBackend, ResetStateTransaction]:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", index=torch.cuda.current_device())
    backend = _TensorBackend(
        device_reset_randomization=device_reset_randomization,
        device_plane=device.type != "cpu",
    )
    transaction = ResetStateTransaction(cast(SimBackend, backend))
    scene = EntityScene(
        {
            "robot": EntityCfg(
                root_body_name="base",
                joint_names=("j0", "j1", "j2"),
                body_names=("base",),
                geom_names=("floor", "foot", "base_geom"),
                actuator_names=("a0", "a1", "a2"),
            )
        },
        cast(SimBackend, backend),
        reset_state=transaction,
    )
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=backend.num_envs,
            device=device,
            rng=ManagerRng.seeded(5, device=device),
            scene=scene,
            step_dt=0.02,
            backend=backend,
        ),
    )
    transaction.declare_packed_reset_device(device)
    return env, backend, transaction


def _dr_event_cfg() -> dict[str, EventTermCfg]:
    return {
        "mass": EventTermCfg(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "mass_distribution_params": (0.8, 1.2),
                "operation": "scale",
                "recompute_inertia": False,
            },
        ),
        "com": EventTermCfg(
            func=mdp.randomize_rigid_body_com,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "com_range": {"x": (-0.05, 0.05)},
            },
        ),
        "friction": EventTermCfg(
            func=mdp.geom_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "ranges": {0: (0.3, 1.2)},
                "operation": "abs",
                "shared_random": True,
            },
        ),
        "pd_lower": EventTermCfg(
            func=mdp.pd_gains,
            mode="reset",
            params={
                "kp_range": (2.0, 2.0),
                "kd_range": (2.0, 2.0),
                "asset_cfg": SceneEntityCfg("robot", actuator_names=["a0"]),
            },
        ),
        "pd_upper": EventTermCfg(
            func=mdp.pd_gains,
            mode="reset",
            params={
                "kp_range": (0.5, 1.5),
                "kd_range": (0.5, 1.5),
                "asset_cfg": SceneEntityCfg("robot", actuator_names=["a1", "a2"]),
            },
        ),
    }


@requires_cuda
def test_reset_randomization_payload_commits_on_cuda_without_host_detour() -> None:
    env, backend, transaction = _tensor_dr_env(
        device_reset_randomization=True, device=torch.device("cuda")
    )
    manager = EventManager(_dr_event_cfg(), env)
    reset_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    with transaction.scoped_device_event_tensor(reset_mask):
        manager.apply(
            "reset",
            env_ids=torch.nonzero(reset_mask, as_tuple=False).flatten(),
            global_env_step_count=0,
        )
        assert backend.tensor_reset_calls == []

    assert len(backend.tensor_reset_calls) == 1
    env_indices, qpos, qvel, payload = backend.tensor_reset_calls[0]
    assert env_indices.device.type == "cuda"
    assert qpos.device.type == "cuda"
    assert qvel.device.type == "cuda"
    assert isinstance(payload, TensorResetRandomizationPayload)
    requested = payload.requested_terms()
    assert {
        RESET_TERM_BODY_MASS,
        RESET_TERM_BODY_IPOS,
        RESET_TERM_GEOM_FRICTION,
        RESET_TERM_KP,
        RESET_TERM_KD,
    } <= requested
    for term in requested:
        assert getattr(payload, term).device.type == "cuda", term
