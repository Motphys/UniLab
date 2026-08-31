"""Upstream-derived NumPy tests for Manager-Based reset event terms."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from unilab.base.backend.base import BackendRootStateLayout, SimBackend
from unilab.base.entity import EntityCfg, EntityScene
from unilab.base.reset_state import ResetStateTransaction
from unilab.dr.types import (
    RESET_TERM_BODY_IPOS,
    RESET_TERM_BODY_MASS,
    RESET_TERM_DOF_ARMATURE,
    RESET_TERM_GEOM_FRICTION,
    RESET_TERM_GRAVITY,
    RESET_TERM_KD,
    RESET_TERM_KP,
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    ResetRandomizationPayload,
)
from unilab.envs import mdp
from unilab.managers import EventManager, EventTermCfg, SceneEntityCfg
from unilab.managers._types import ManagerBasedRlEnv


class _CaptureEntity:
    def __init__(self, default_root_state: np.ndarray) -> None:
        self.data = SimpleNamespace(default_root_state=default_root_state)
        self.writes: list[tuple[np.ndarray, np.ndarray]] = []

    def write_root_state_to_sim(
        self,
        root_state: np.ndarray,
        env_ids: np.ndarray | None = None,
    ) -> None:
        assert env_ids is not None
        self.writes.append((root_state.copy(), env_ids.copy()))


class _CaptureScene:
    def __init__(self, entity: _CaptureEntity, env_origins: np.ndarray) -> None:
        self.entities = {"robot": entity}
        self.env_origins = env_origins

    def __getitem__(self, name: str) -> _CaptureEntity:
        return self.entities[name]


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
            rng=np.random.default_rng(seed),
            scene=_CaptureScene(entity, env_origins),
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
    ids = np.asarray([2, 0], dtype=np.int32)

    mdp.reset_root_state_uniform(
        env,
        ids,
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
    root_state, written_ids = entity.writes[0]
    np.testing.assert_array_equal(written_ids, ids)
    np.testing.assert_allclose(
        root_state[:, :3],
        defaults[ids, :3] + [1.0, -2.0, 0.5] + origins[ids],
    )
    np.testing.assert_allclose(
        root_state[:, 3:7],
        [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        root_state[:, 7:13],
        defaults[ids, 7:13] + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    np.testing.assert_array_equal(defaults[:, :3], [[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test_uniform_root_state_uses_env_rng_and_preserves_single_env_shape() -> None:
    pose_range = {"x": (-1.0, 1.0), "yaw": (-0.5, 0.5)}
    velocity_range = {"x": (-2.0, 2.0), "roll": (-0.25, 0.25)}
    left, left_entity = _capture_env(seed=9)
    right, right_entity = _capture_env(seed=9)
    ids = np.asarray([1], dtype=np.int32)

    mdp.reset_root_state_uniform(left, ids, pose_range, velocity_range)
    mdp.reset_root_state_uniform(right, ids, pose_range, velocity_range)

    left_state, left_ids = left_entity.writes[0]
    right_state, right_ids = right_entity.writes[0]
    assert left_state.shape == (1, 13)
    np.testing.assert_array_equal(left_ids, ids)
    np.testing.assert_array_equal(right_ids, ids)
    np.testing.assert_array_equal(left_state, right_state)


def test_uniform_root_state_none_ids_targets_all_environments() -> None:
    env, entity = _capture_env()

    mdp.reset_root_state_uniform(env, None, pose_range={})

    root_state, ids = entity.writes[0]
    np.testing.assert_array_equal(ids, np.arange(3, dtype=np.int32))
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
    ) -> None:
        self.root_layout_supported = root_layout_supported
        self.gain_supported = gain_supported
        self.randomization_supported = randomization_supported
        self.interval_velocity_supported = interval_velocity_supported
        self.interval_angular_velocity_supported = interval_angular_velocity_supported
        self.interval_force_supported = interval_force_supported
        self.interval_torque_supported = interval_torque_supported
        self.default_qpos = np.asarray([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
        self.init_qvel = np.zeros(6)
        self.set_state_calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.randomization_calls: list[ResetRandomizationPayload | None] = []
        self.body_pos = np.zeros((self.num_envs, 1, 3))
        self.body_quat = np.zeros((self.num_envs, 1, 4))
        self.body_quat[:, :, 0] = 1.0
        self.body_velocity = np.zeros((self.num_envs, 1, 3))
        self.body_mass = np.array([10.0])
        self.body_ipos = np.array([[0.0, 0.0, 0.0]])
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.dof_armature = np.array([0.0] * 6 + [1.0, 2.0, 3.0])
        self.geom_friction = np.array([[0.5, 0.01, 0.001], [0.7, 0.02, 0.002], [0.9, 0.03, 0.003]])
        self.interval_plans: list[IntervalRandomizationPlan] = []

    def get_body_ids(self, names) -> np.ndarray:
        if tuple(names) != ("base",):
            raise KeyError(names)
        return np.asarray([0], dtype=np.int32)

    def get_root_state_layout(self, root_body_name: str) -> BackendRootStateLayout:
        if not self.root_layout_supported:
            raise NotImplementedError("fake fixed-base entity has no free-root layout")
        if root_body_name != "base":
            raise ValueError(root_body_name)
        return BackendRootStateLayout(tuple(range(7)), tuple(range(6)))

    def get_default_qpos(self) -> np.ndarray:
        return self.default_qpos.copy()

    def get_init_qvel(self) -> np.ndarray:
        return self.init_qvel.copy()

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
        return DomainRandomizationCapabilities(
            supported_reset_terms=frozenset(terms),
            supports_interval_body_velocity_delta=self.interval_velocity_supported,
            supports_interval_body_angular_velocity_delta=self.interval_angular_velocity_supported,
            supports_interval_body_force=self.interval_force_supported,
            supports_interval_body_torque=self.interval_torque_supported,
        )

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([10.0, 20.0, 30.0]), np.array([1.0, 2.0, 3.0])

    def get_body_mass(self) -> np.ndarray:
        return self.body_mass.copy()

    def get_body_ipos(self) -> np.ndarray:
        return self.body_ipos.copy()

    def get_gravity(self) -> np.ndarray:
        return self.gravity.copy()

    def get_dof_armature(self) -> np.ndarray:
        return self.dof_armature.copy()

    def get_geom_friction(self) -> np.ndarray:
        return self.geom_friction.copy()

    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        self.interval_plans.append(plan)

    def get_body_pos_w(self, ids: np.ndarray) -> np.ndarray:
        return self.body_pos[:, ids]

    def get_body_quat_w(self, ids: np.ndarray) -> np.ndarray:
        return self.body_quat[:, ids]

    def get_body_lin_vel_w(self, ids: np.ndarray) -> np.ndarray:
        return self.body_velocity[:, ids]

    def get_body_ang_vel_w(self, ids: np.ndarray) -> np.ndarray:
        return self.body_velocity[:, ids]

    def get_body_lin_vel_b(self, ids: np.ndarray) -> np.ndarray:
        return self.body_velocity[:, ids]

    def get_body_ang_vel_b(self, ids: np.ndarray) -> np.ndarray:
        return self.body_velocity[:, ids]

    def set_state(
        self,
        env_ids: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization=None,
    ) -> None:
        self.set_state_calls.append((env_ids.copy(), qpos.copy(), qvel.copy()))
        self.randomization_calls.append(randomization)


def _transaction_env(
    *,
    root_layout_supported: bool = True,
    gain_supported: bool = True,
    randomization_supported: bool = True,
    interval_velocity_supported: bool = True,
    interval_angular_velocity_supported: bool = False,
    interval_force_supported: bool = False,
    interval_torque_supported: bool = False,
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
            rng=np.random.default_rng(rng_seed),
            scene=scene,
            step_dt=step_dt,
        ),
    )
    return env, backend, transaction


def test_uniform_root_state_composes_in_one_reset_transaction_commit() -> None:
    env, backend, transaction = _transaction_env()
    active_ids = np.asarray([0, 2], dtype=np.int32)

    with transaction.scoped(active_ids):
        mdp.reset_scene_to_default(env, active_ids)
        mdp.reset_root_state_uniform(
            env,
            np.asarray([2], dtype=np.int32),
            pose_range={"x": (1.0, 1.0)},
            velocity_range={"x": (0.5, 0.5)},
        )
        assert backend.set_state_calls == []

    assert len(backend.set_state_calls) == 1
    ids, qpos, qvel = backend.set_state_calls[0]
    np.testing.assert_array_equal(ids, active_ids)
    np.testing.assert_array_equal(qpos[0], backend.default_qpos)
    np.testing.assert_allclose(qpos[1], backend.default_qpos + [1.0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(qvel[0], backend.init_qvel)
    np.testing.assert_allclose(qvel[1], backend.init_qvel + [0.5, 0, 0, 0, 0, 0])


def test_pd_gains_event_uses_selector_scale_and_exactly_once_reset_payload() -> None:
    env, backend, transaction = _transaction_env(rng_seed=11)
    manager = EventManager(
        {
            "randomize_pd": EventTermCfg(
                func=mdp.pd_gains,
                mode="reset",
                params={
                    "kp_range": (2.0, 2.0),
                    "kd_range": (3.0, 3.0),
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        actuator_names=["a2", "a0"],
                        preserve_order=True,
                    ),
                },
            )
        },
        env,
    )
    ids = np.array([0, 2], dtype=np.int32)

    with transaction.scoped(ids):
        manager.apply(mode="reset", env_ids=ids, global_env_step_count=0)
        assert backend.set_state_calls == []

    assert len(backend.set_state_calls) == 1
    payload = backend.randomization_calls[0]
    assert payload is not None
    np.testing.assert_array_equal(payload.kp, [[20.0, 20.0, 60.0]] * 2)
    np.testing.assert_array_equal(payload.kd, [[3.0, 2.0, 9.0]] * 2)


def test_pd_gains_event_supports_log_uniform_absolute_sampling() -> None:
    env, backend, transaction = _transaction_env(rng_seed=11)
    manager = EventManager(
        {
            "gain": EventTermCfg(
                func=mdp.pd_gains,
                mode="reset",
                params={
                    "kp_range": (0.25, 4.0),
                    "kd_range": (0.5, 2.0),
                    "distribution": "log_uniform",
                    "operation": "abs",
                },
            )
        },
        env,
    )
    ids = np.arange(3, dtype=np.int32)

    with transaction.scoped(ids):
        manager.apply(mode="reset", env_ids=ids, global_env_step_count=0)

    payload = backend.randomization_calls[0]
    assert payload is not None and payload.kp is not None and payload.kd is not None
    assert np.all((payload.kp >= 0.25) & (payload.kp <= 4.0))
    assert np.all((payload.kd >= 0.5) & (payload.kd <= 2.0))
    assert np.unique(payload.kp[0]).size > 1


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


def test_reset_randomization_terms_compose_with_state_and_gains_exactly_once() -> None:
    env, backend, transaction = _transaction_env(rng_seed=13)
    asset_cfg = SceneEntityCfg("robot", body_names=("base",))
    manager = EventManager(
        {
            "mass": EventTermCfg(
                func=mdp.randomize_rigid_body_mass,
                mode="reset",
                params={
                    "asset_cfg": asset_cfg,
                    "mass_distribution_params": (1.5, 1.5),
                    "operation": "scale",
                    "recompute_inertia": False,
                },
            ),
            "com": EventTermCfg(
                func=mdp.randomize_rigid_body_com,
                mode="reset",
                params={
                    "asset_cfg": asset_cfg,
                    "com_range": {"x": (0.1, 0.1), "z": (-0.2, -0.2)},
                },
            ),
            "gravity": EventTermCfg(
                func=mdp.randomize_physics_scene_gravity,
                mode="reset",
                params={
                    "gravity_distribution_params": (
                        [0.0, 0.0, -10.0],
                        [0.0, 0.0, -10.0],
                    ),
                    "operation": "abs",
                },
            ),
            "gains": EventTermCfg(
                func=mdp.pd_gains,
                mode="reset",
                params={"kp_range": (2.0, 2.0), "kd_range": (3.0, 3.0)},
            ),
        },
        env,
    )
    ids = np.array([0, 2], dtype=np.int32)

    with transaction.scoped(ids):
        mdp.reset_scene_to_default(env, ids)
        manager.apply(mode="reset", env_ids=ids, global_env_step_count=0)
        assert backend.set_state_calls == []

    assert len(backend.set_state_calls) == 1
    payload = backend.randomization_calls[0]
    assert payload is not None
    assert payload.body_mass is not None
    assert payload.body_ipos is not None
    assert payload.gravity is not None
    assert payload.kp is not None
    assert payload.kd is not None
    np.testing.assert_allclose(payload.body_mass, [[15.0], [15.0]])
    np.testing.assert_allclose(payload.body_ipos, [[[0.1, 0.0, -0.2]]] * 2)
    np.testing.assert_allclose(payload.gravity, [[0.0, 0.0, -10.0]] * 2)
    np.testing.assert_allclose(payload.kp, [[20.0, 40.0, 60.0]] * 2)
    np.testing.assert_allclose(payload.kd, [[3.0, 6.0, 9.0]] * 2)


def test_model_field_terms_use_cached_selectors_and_one_dense_reset_payload() -> None:
    env, backend, transaction = _transaction_env(rng_seed=23)
    manager = EventManager(
        {
            "armature": EventTermCfg(
                func=mdp.joint_armature,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=("j2", "j0"),
                        preserve_order=True,
                    ),
                    "ranges": (2.0, 2.0),
                    "operation": "scale",
                },
            ),
            "friction": EventTermCfg(
                func=mdp.geom_friction,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot", geom_names=".*"),
                    "ranges": {"floor": (2.0, 2.0), "base_.*": (3.0, 3.0)},
                    "operation": "scale",
                    "shared_random": True,
                },
            ),
        },
        env,
    )
    ids = np.array([0, 2], dtype=np.int32)

    with transaction.scoped(ids):
        mdp.reset_scene_to_default(env, ids)
        manager.apply(mode="reset", env_ids=ids, global_env_step_count=0)
        assert backend.set_state_calls == []

    assert len(backend.set_state_calls) == 1
    payload = backend.randomization_calls[0]
    assert payload is not None
    assert payload.dof_armature is not None
    assert payload.geom_friction is not None
    np.testing.assert_allclose(
        payload.dof_armature,
        [[0.0] * 6 + [2.0, 2.0, 6.0]] * 2,
    )
    expected_friction = backend.geom_friction.copy()
    expected_friction[0, 0] *= 2.0
    expected_friction[2, 0] *= 3.0
    np.testing.assert_allclose(payload.geom_friction, [expected_friction] * 2)


def test_model_field_aliases_are_identical_and_capability_gaps_fail_cold() -> None:
    assert mdp.dof_armature is mdp.joint_armature
    env, backend, _ = _transaction_env(randomization_supported=False)

    with pytest.raises(
        NotImplementedError,
        match="joint_armature:robot.*dof_armature randomization.*backend 'fake'",
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
        (mdp.joint_armature, {"ranges": (-1.0, -0.5)}, "produced negative"),
    ],
)
def test_model_field_invalid_requests_fail_explicitly(func, params, match: str) -> None:
    env, backend, transaction = _transaction_env()
    if func is mdp.geom_friction:
        with pytest.raises(ValueError, match=match):
            EventManager({"field": EventTermCfg(func=func, mode="reset", params=params)}, env)
    else:
        manager = EventManager(
            {"field": EventTermCfg(func=func, mode="reset", params=params)},
            env,
        )
        ids = np.array([0, 1], dtype=np.int32)
        with pytest.raises(ValueError, match=match):
            with transaction.scoped(ids):
                mdp.reset_scene_to_default(env, ids)
                manager.apply(mode="reset", env_ids=ids, global_env_step_count=0)
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


def test_reset_randomization_sparse_rows_abort_without_backend_mutation() -> None:
    env, backend, transaction = _transaction_env()
    manager = EventManager(
        {
            "gravity": EventTermCfg(
                func=mdp.randomize_physics_scene_gravity,
                mode="reset",
                params={
                    "gravity_distribution_params": ([0.0, 0.0, -10.0],) * 2,
                    "operation": "abs",
                },
            )
        },
        env,
    )
    cfg = manager.get_term_cfg("gravity")
    ids = np.array([0, 1], dtype=np.int32)

    with pytest.raises(RuntimeError, match=r"gravity payload.*sparse rows.*missing env IDs \[1\]"):
        with transaction.scoped(ids):
            mdp.reset_scene_to_default(env, ids)
            cfg.func(env, np.array([0], dtype=np.int32), **cfg.params)
    assert backend.set_state_calls == []


def test_min_step_count_gating_reuses_committed_field_values() -> None:
    env, backend, transaction = _transaction_env(rng_seed=29)
    manager = EventManager(
        {
            "mass": EventTermCfg(
                func=mdp.randomize_rigid_body_mass,
                mode="reset",
                min_step_count_between_reset=100,
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
    first_ids = np.array([0, 1], dtype=np.int32)
    with transaction.scoped(first_ids):
        mdp.reset_scene_to_default(env, first_ids)
        manager.apply(mode="reset", env_ids=first_ids, global_env_step_count=0)
    first = backend.randomization_calls[-1]
    assert first is not None and first.body_mass is not None

    # Env 2 has never triggered, so the first-trigger override fires it while
    # envs 0 and 1 stay gated; their payload rows reuse committed values.
    ids = np.array([0, 1, 2], dtype=np.int32)
    with transaction.scoped(ids):
        mdp.reset_scene_to_default(env, ids)
        manager.apply(mode="reset", env_ids=ids, global_env_step_count=50)
    second = backend.randomization_calls[-1]
    assert second is not None and second.body_mass is not None
    np.testing.assert_allclose(second.body_mass[:2], first.body_mass)
    np.testing.assert_array_equal(manager._reset_term_last_triggered_step_id[0], [0, 0, 50])

    # Fully gated resets skip the field entirely; the backend keeps the
    # previously applied per-env values, so no dense payload is needed.
    with transaction.scoped(first_ids):
        mdp.reset_scene_to_default(env, first_ids)
        manager.apply(mode="reset", env_ids=first_ids, global_env_step_count=60)
    third = backend.randomization_calls[-1]
    assert third is None or third.body_mass is None

    # Once enough steps elapsed the gate reopens and the term resamples.
    with transaction.scoped(first_ids):
        mdp.reset_scene_to_default(env, first_ids)
        manager.apply(mode="reset", env_ids=first_ids, global_env_step_count=200)
    fourth = backend.randomization_calls[-1]
    assert fourth is not None and fourth.body_mass is not None
    np.testing.assert_array_equal(manager._reset_term_last_triggered_step_id[0], [200, 200, 50])
    assert not np.allclose(fourth.body_mass, second.body_mass[:2])


def test_min_step_count_gating_applies_to_pd_gains_payload_rows() -> None:
    env, backend, transaction = _transaction_env(rng_seed=41)
    manager = EventManager(
        {
            "gains": EventTermCfg(
                func=mdp.pd_gains,
                mode="reset",
                min_step_count_between_reset=50,
                params={"kp_range": (1.5, 1.5), "kd_range": (2.5, 2.5)},
            )
        },
        env,
    )
    first_ids = np.array([0, 1], dtype=np.int32)
    with transaction.scoped(first_ids):
        manager.apply(mode="reset", env_ids=first_ids, global_env_step_count=0)
    first = backend.randomization_calls[-1]
    assert first is not None and first.kp is not None and first.kd is not None

    ids = np.array([0, 1, 2], dtype=np.int32)
    with transaction.scoped(ids):
        mdp.reset_scene_to_default(env, ids)
        manager.apply(mode="reset", env_ids=ids, global_env_step_count=10)
    second = backend.randomization_calls[-1]
    assert second is not None and second.kp is not None and second.kd is not None
    np.testing.assert_allclose(second.kp[:2], first.kp)
    np.testing.assert_allclose(second.kd[:2], first.kd)
    np.testing.assert_array_equal(manager._reset_term_last_triggered_step_id[0], [0, 0, 10])


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
    np.testing.assert_array_equal(plan.body_ids, [0])
    assert plan.body_force is not None
    np.testing.assert_allclose(plan.body_force[:, 0], 5.0)
    assert plan.body_torque is None

    manager.apply(mode="step", dt=0.02)
    manager.apply(mode="step", dt=0.02)
    assert len(backend.interval_plans) == 3
    np.testing.assert_allclose(backend.interval_plans[1].body_force[:, 0], 5.0)
    np.testing.assert_allclose(backend.interval_plans[2].body_force[:, 0], 5.0)

    # Expiry stages zeros once (persistent wrench channels clear), then the
    # term stays idle through the next cooldown instead of re-staging.
    manager.apply(mode="step", dt=0.02)
    assert len(backend.interval_plans) == 4
    np.testing.assert_allclose(backend.interval_plans[3].body_force[:, 0], 0.0)
    manager.apply(mode="step", dt=0.02)
    assert len(backend.interval_plans) == 4


def test_apply_body_impulse_offset_generates_cross_torque() -> None:
    env, backend, _ = _transaction_env(
        rng_seed=37,
        interval_force_supported=True,
        interval_torque_supported=True,
    )
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
    assert plan.body_force is not None and plan.body_torque is not None
    np.testing.assert_allclose(plan.body_force[:, 0], 1.0)
    # Identity link orientation: offset_w = (0, 0, 0.1), force = (1, 1, 1),
    # cross(offset_w, force) = (-0.1, 0.1, 0).
    np.testing.assert_allclose(
        plan.body_torque[:, 0],
        [[-0.1, 0.1, 0.0]] * 3,
        atol=1e-12,
    )


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

    manager.reset(np.array([1], dtype=np.int32))
    assert len(backend.interval_plans) == 2
    np.testing.assert_allclose(backend.interval_plans[1].body_force[:, 0], 0.0)

    manager.apply(mode="step", dt=0.02)
    sustained = backend.interval_plans[-1].body_force
    assert sustained is not None
    np.testing.assert_allclose(sustained[1, 0], 0.0)
    np.testing.assert_allclose(sustained[[0, 2], 0], 5.0)


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
    manager._interval_term_time_left[0][:] = [0.0, 1.0, 0.0]

    manager.apply(mode="interval", dt=0.1)

    assert len(backend.interval_plans) == 1
    plan = backend.interval_plans[0]
    np.testing.assert_array_equal(plan.body_ids, [0])
    assert plan.body_linear_velocity_delta is not None
    np.testing.assert_allclose(
        plan.body_linear_velocity_delta[:, 0],
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
    manager._interval_term_time_left[0][:] = [0.0, 1.0, 0.0]

    manager.apply(mode="interval", dt=0.1)

    assert len(backend.interval_plans) == 1
    plan = backend.interval_plans[0]
    np.testing.assert_array_equal(plan.body_ids, [0])
    assert plan.body_linear_velocity_delta is not None
    np.testing.assert_allclose(
        plan.body_linear_velocity_delta[:, 0],
        [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
    )
    assert plan.body_angular_velocity_delta is not None
    np.testing.assert_allclose(
        plan.body_angular_velocity_delta[:, 0],
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
        with transaction.scoped(np.asarray([1], dtype=np.int32)):
            mdp.reset_root_state_uniform(env, np.asarray([1], dtype=np.int32), pose_range={})

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
        mdp.reset_root_state_uniform(env, np.asarray([0], dtype=np.int32), pose_range)

    assert entity.writes == []


def test_events_module_has_no_forbidden_runtime_dependencies() -> None:
    path = Path(__file__).resolve().parents[3] / "src" / "unilab" / "envs" / "mdp" / "events.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = ("torch", "unilab.ipc", "unilab.algos", "unilab.training", "unilab.base.backend")
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)] + [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not [name for name in imports if name.startswith(forbidden)]


class _BiasEntity:
    def __init__(self, num_envs: int, num_joints: int) -> None:
        self.num_joints = num_joints
        self.joint_names = [f"joint_{index}" for index in range(num_joints)]
        self.data = SimpleNamespace(encoder_bias=np.zeros((num_envs, num_joints), dtype=np.float32))

    def find_joints(self, names: list[str]) -> list[int]:
        return [self.joint_names.index(name) for name in names]


class _BiasScene:
    def __init__(self, entity: _BiasEntity) -> None:
        self._entity = entity

    def __getitem__(self, name: str) -> _BiasEntity:
        if name != "robot":
            raise KeyError(name)
        return self._entity


def _bias_env(num_envs: int = 3, num_joints: int = 4, seed: int = 7) -> ManagerBasedRlEnv:
    return cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=num_envs,
            rng=np.random.default_rng(seed),
            scene=_BiasScene(_BiasEntity(num_envs, num_joints)),
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

    manager.apply(mode="reset", env_ids=np.asarray([0, 2], dtype=np.int32), global_env_step_count=0)

    bias = env.scene["robot"].data.encoder_bias
    assert bias.shape == (3, 4)
    np.testing.assert_array_equal(bias[1], np.zeros(4))
    np.testing.assert_array_equal(bias[:, [0, 2]], np.zeros((3, 2)))
    assert np.all(np.abs(bias[[0, 2]][:, [1, 3]]) <= 0.02)
    assert np.any(bias[[0, 2]][:, [1, 3]] != 0.0)


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
