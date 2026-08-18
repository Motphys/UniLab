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
    RESET_TERM_KD,
    RESET_TERM_KP,
    DomainRandomizationCapabilities,
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
    ) -> None:
        self.root_layout_supported = root_layout_supported
        self.gain_supported = gain_supported
        self.default_qpos = np.asarray([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
        self.init_qvel = np.zeros(6)
        self.set_state_calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.randomization_calls: list[ResetRandomizationPayload | None] = []
        self.body_pos = np.zeros((self.num_envs, 1, 3))
        self.body_quat = np.zeros((self.num_envs, 1, 4))
        self.body_quat[:, :, 0] = 1.0
        self.body_velocity = np.zeros((self.num_envs, 1, 3))

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
        return np.empty((self.num_envs, 0))

    def get_dof_vel(self) -> np.ndarray:
        return np.empty((self.num_envs, 0))

    def get_actuator_names(self) -> tuple[str, ...]:
        return ("a0", "a1", "a2")

    def get_actuator_ctrl_range(self) -> np.ndarray:
        return np.tile([-1.0, 1.0], (self.num_actuators, 1))

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        terms = frozenset((RESET_TERM_KP, RESET_TERM_KD)) if self.gain_supported else frozenset()
        return DomainRandomizationCapabilities(supported_reset_terms=terms)

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([10.0, 20.0, 30.0]), np.array([1.0, 2.0, 3.0])

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
    *, root_layout_supported: bool = True, gain_supported: bool = True, rng_seed: int = 5
) -> tuple[ManagerBasedRlEnv, _Backend, ResetStateTransaction]:
    backend = _Backend(
        root_layout_supported=root_layout_supported,
        gain_supported=gain_supported,
    )
    transaction = ResetStateTransaction(cast(SimBackend, backend))
    scene = EntityScene(
        {"robot": EntityCfg(root_body_name="base", actuator_names=("a0", "a1", "a2"))},
        cast(SimBackend, backend),
        reset_state=transaction,
    )
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=backend.num_envs,
            rng=np.random.default_rng(rng_seed),
            scene=scene,
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
        ({"min_step_count_between_reset": 2}, "min_step_count_between_reset=0"),
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
