"""Real-CUDA effect tests for ``mjwarp`` per-world domain randomization.

Each test verifies one of three evidence classes required by issue #1401:

- per-world model rows hold the staged payload values while complement rows
  keep the immutable defaults (write path + world isolation);
- the mutation changes the physics of the mutated worlds only (effect);
- the graded ``set_const*`` recompute refreshed the derived per-world fields.

The cross-backend test compares distribution-level statistics against the
MuJoCo backend on identical sampled payloads rather than bitwise state.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from unilab.base.backend import create_backend
from unilab.base.backend.mjwarp.dependencies import load_mjwarp_dependencies
from unilab.base.backend.mjwarp.randomization import EXPANDED_MODEL_FIELDS
from unilab.base.scene import SceneCfg
from unilab.dr.types import (
    RESET_TERM_BASE_COM,
    RESET_TERM_BASE_MASS,
    RESET_TERM_BODY_INERTIA,
    RESET_TERM_BODY_IPOS,
    RESET_TERM_BODY_IQUAT,
    RESET_TERM_BODY_MASS,
    RESET_TERM_DOF_ARMATURE,
    RESET_TERM_GEOM_FRICTION,
    RESET_TERM_KD,
    RESET_TERM_KP,
    IntervalRandomizationPlan,
    ResetRandomizationPayload,
)

pytestmark = pytest.mark.slow

_SIM_DT = 0.02 / 3.0


def _require_cuda_mjwarp() -> None:
    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.fail("mjwarp DR tests require an active CUDA Warp device")


def _scene() -> SceneCfg:
    from unilab.assets import ASSETS_ROOT_PATH

    return SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml"))


@pytest.fixture
def backend() -> Any:
    _require_cuda_mjwarp()
    return create_backend("mjwarp", _scene(), 4, _SIM_DT, base_name="pelvis")


def _stand_state(backend: Any, count: int) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.tile(backend.get_keyframe_qpos("stand"), (count, 1))
    qvel = np.zeros((count, backend.get_init_qvel().size), dtype=np.float32)
    return qpos.astype(np.float32), qvel


def _reset_all(backend: Any) -> None:
    qpos, qvel = _stand_state(backend, backend.num_envs)
    backend.set_state(np.arange(backend.num_envs, dtype=np.int32), qpos, qvel)


def _stand_ctrl(backend: Any) -> np.ndarray:
    qpos = backend.get_keyframe_qpos("stand")
    return np.tile(qpos[-backend.num_actuators :].astype(np.float32), (backend.num_envs, 1))


def _broadcast_rows(values: np.ndarray, count: int) -> np.ndarray:
    return np.broadcast_to(values, (count, *values.shape)).copy()


def _device_field(backend: Any, name: str) -> np.ndarray:
    return np.asarray(getattr(backend._device_model, name).numpy())


def _device_xfrc(backend: Any) -> np.ndarray:
    return np.asarray(backend._device_data.xfrc_applied.numpy())


def test_dr_capabilities_advertise_supported_terms(backend: Any) -> None:
    capabilities = backend.get_dr_capabilities()
    assert capabilities.supported_reset_terms == frozenset(
        {
            RESET_TERM_BASE_MASS,
            RESET_TERM_BASE_COM,
            RESET_TERM_BODY_IQUAT,
            RESET_TERM_BODY_INERTIA,
            RESET_TERM_BODY_IPOS,
            RESET_TERM_BODY_MASS,
            RESET_TERM_DOF_ARMATURE,
            RESET_TERM_GEOM_FRICTION,
            RESET_TERM_KP,
            RESET_TERM_KD,
        }
    )
    assert capabilities.supports_interval_push
    assert capabilities.supports_interval_body_velocity_delta
    assert capabilities.supports_interval_body_force


def test_declared_model_fields_are_expanded_per_world(backend: Any) -> None:
    assert EXPANDED_MODEL_FIELDS
    for name in EXPANDED_MODEL_FIELDS:
        array = getattr(backend._device_model, name)
        assert array.shape[0] == backend.num_envs, name
    np.testing.assert_allclose(
        _device_field(backend, "body_mass"),
        np.broadcast_to(backend.get_body_mass(), (backend.num_envs, backend.get_body_mass().size)),
        rtol=1e-6,
    )
    # Expansion happened before graph capture; replaying the captured step on
    # the expanded arrays must stay finite and isolated.
    _reset_all(backend)
    backend.step(_stand_ctrl(backend), nsteps=3)
    assert np.isfinite(backend.get_base_pos()).all()


def test_reset_payload_writes_only_selected_rows(backend: Any) -> None:
    rows = np.asarray([1, 3], dtype=np.int32)
    complement = np.asarray([0, 2], dtype=np.int32)
    qpos, qvel = _stand_state(backend, rows.size)

    body_mass = _broadcast_rows(backend.get_body_mass(), rows.size)
    body_mass *= 1.2
    body_ipos = _broadcast_rows(backend.get_body_ipos(), rows.size)
    body_ipos[..., 0] += 0.01
    dof_armature = _broadcast_rows(backend.get_dof_armature(), rows.size) + 0.02
    geom_friction = _broadcast_rows(backend.get_geom_friction(), rows.size)
    geom_friction[..., 0] *= 0.5
    kp = np.full((rows.size, backend.num_actuators), 17.0, dtype=np.float32)
    kd = np.full((rows.size, backend.num_actuators), 1.5, dtype=np.float32)

    backend.set_state(
        rows,
        qpos,
        qvel,
        randomization=ResetRandomizationPayload(
            body_mass=body_mass,
            body_ipos=body_ipos,
            dof_armature=dof_armature,
            geom_friction=geom_friction,
            kp=kp,
            kd=kd,
        ),
    )

    np.testing.assert_allclose(_device_field(backend, "body_mass")[rows], body_mass, rtol=1e-6)
    np.testing.assert_allclose(_device_field(backend, "body_ipos")[rows], body_ipos, rtol=1e-6)
    np.testing.assert_allclose(
        _device_field(backend, "dof_armature")[rows], dof_armature, rtol=1e-6
    )
    np.testing.assert_allclose(
        _device_field(backend, "geom_friction")[rows], geom_friction, rtol=1e-6
    )
    np.testing.assert_allclose(
        _device_field(backend, "actuator_gainprm")[rows, :, 0], kp, rtol=1e-6
    )
    np.testing.assert_allclose(
        _device_field(backend, "actuator_biasprm")[rows, :, 2], kd, rtol=1e-6
    )

    default_mass = backend.get_body_mass()
    np.testing.assert_allclose(
        _device_field(backend, "body_mass")[complement],
        _broadcast_rows(default_mass, complement.size),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        _device_field(backend, "body_ipos")[complement],
        _broadcast_rows(backend.get_body_ipos(), complement.size),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        _device_field(backend, "dof_armature")[complement],
        _broadcast_rows(backend.get_dof_armature(), complement.size),
        rtol=1e-6,
    )

    # The graded recompute refreshed derived per-world fields on every world.
    subtree_mass = _device_field(backend, "body_subtreemass")
    np.testing.assert_allclose(
        subtree_mass[rows, 0],
        body_mass.sum(axis=1),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        subtree_mass[complement, 0],
        np.full(complement.size, default_mass.sum(), dtype=np.float32),
        rtol=1e-4,
    )
    default_invweight0 = _device_field(backend, "dof_invweight0")[complement]
    mutated_invweight0 = _device_field(backend, "dof_invweight0")[rows]
    assert not np.allclose(mutated_invweight0, default_invweight0[:1])

    # Re-randomizing one of the rows restores it to the defaults + new delta.
    base_id = backend._base_body_id
    qpos1, qvel1 = _stand_state(backend, 1)
    backend.set_state(
        np.asarray([1], dtype=np.int32),
        qpos1,
        qvel1,
        randomization=ResetRandomizationPayload(base_mass_delta=np.asarray([0.5])),
    )
    np.testing.assert_allclose(
        _device_field(backend, "body_mass")[1, base_id],
        default_mass[base_id] + 0.5,
        rtol=1e-6,
    )


def test_base_com_offset_applies_default_plus_delta(backend: Any) -> None:
    rows = np.asarray([0, 2], dtype=np.int32)
    qpos, qvel = _stand_state(backend, rows.size)
    offset = np.asarray([[0.01, -0.02, 0.03], [-0.015, 0.005, -0.025]], dtype=np.float32)
    backend.set_state(
        rows,
        qpos,
        qvel,
        randomization=ResetRandomizationPayload(base_com_offset=offset),
    )
    base_id = backend._base_body_id
    expected = backend.get_body_ipos()[base_id] + offset
    np.testing.assert_allclose(_device_field(backend, "body_ipos")[rows, base_id], expected)
    complement = np.asarray([1, 3], dtype=np.int32)
    np.testing.assert_allclose(
        _device_field(backend, "body_ipos")[complement],
        _broadcast_rows(backend.get_body_ipos(), complement.size),
        rtol=1e-6,
    )


def test_body_inertia_and_iquat_recompute_invweight0(backend: Any) -> None:
    rows = np.asarray([2], dtype=np.int32)
    qpos, qvel = _stand_state(backend, 1)
    default_invweight0 = _device_field(backend, "body_invweight0")[0].copy()

    inertia = np.broadcast_to(
        np.asarray(backend._cpu_model.body_inertia, dtype=np.float32),
        (1, backend._nbody, 3),
    ).copy()
    inertia[:, 1:, :] *= 4.0
    iquat = np.broadcast_to(
        np.asarray(backend._cpu_model.body_iquat, dtype=np.float32),
        (1, backend._nbody, 4),
    ).copy()
    backend.set_state(
        rows,
        qpos,
        qvel,
        randomization=ResetRandomizationPayload(
            body_inertia=inertia,
            body_iquat=iquat,
        ),
    )
    np.testing.assert_allclose(_device_field(backend, "body_inertia")[rows], inertia, rtol=1e-6)
    assert not np.allclose(_device_field(backend, "body_invweight0")[rows[0]], default_invweight0)
    complement = np.asarray([0, 1, 3], dtype=np.int32)
    # set_const_0 recomputes every world's derived fields; complement rows stay
    # at the defaults up to float32 recompute noise (no mutated values leak).
    np.testing.assert_allclose(
        _device_field(backend, "body_invweight0")[complement],
        _broadcast_rows(default_invweight0, complement.size),
        rtol=1e-3,
        atol=2e-4,
    )


def test_heavier_world_accelerates_less_under_identical_body_force(backend: Any) -> None:
    """Mass mutation changes physics for the mutated world only."""
    _reset_all(backend)
    heavy_rows = np.asarray([1, 3], dtype=np.int32)
    qpos, qvel = _stand_state(backend, heavy_rows.size)
    body_mass = _broadcast_rows(backend.get_body_mass(), heavy_rows.size)
    body_mass *= 4.0
    backend.set_state(
        heavy_rows,
        qpos,
        qvel,
        randomization=ResetRandomizationPayload(body_mass=body_mass),
    )

    force = np.zeros((backend.num_envs, 1, 3), dtype=np.float32)
    force[:, 0, 0] = 300.0
    backend.apply_interval_randomization(
        IntervalRandomizationPlan(body_ids=np.asarray([backend._base_body_id]), body_force=force)
    )
    backend.step(_stand_ctrl(backend), nsteps=3)

    delta_v = backend.get_base_lin_vel()[:, 0]
    assert np.all(np.abs(delta_v) > 1e-4)
    # Heavier worlds accelerate less; identical unmutated worlds stay paired
    # up to contact-solver reduction noise.
    assert delta_v[1] < delta_v[0]
    assert delta_v[3] < delta_v[2]
    np.testing.assert_allclose(delta_v[0], delta_v[2], rtol=1e-3)
    np.testing.assert_allclose(delta_v[1], delta_v[3], rtol=1e-3)
    # The staged force was cleared after the step barrier.
    assert not np.any(_device_xfrc(backend))
    assert not backend._xfrc_pending


def test_push_robots_stages_one_step_force(backend: Any) -> None:
    _reset_all(backend)
    backend.apply_interval_randomization(
        IntervalRandomizationPlan(push_perturbation_limit=np.asarray([80.0, 0.0, 0.0]))
    )
    assert backend._xfrc_pending
    before = backend.get_base_lin_vel().copy()
    backend.step(_stand_ctrl(backend), nsteps=3)
    after = backend.get_base_lin_vel()
    assert not np.allclose(before, after)
    assert not np.any(_device_xfrc(backend))
    assert not backend._xfrc_pending


def test_interval_velocity_delta_kicks_selected_rows(backend: Any) -> None:
    _reset_all(backend)
    before = backend.get_base_lin_vel().copy()
    delta = np.zeros((backend.num_envs, 1, 3), dtype=np.float32)
    delta[[0, 3], 0, :] = np.asarray([[0.4, -0.2, 0.0], [-0.3, 0.1, 0.0]], dtype=np.float32)
    backend.apply_interval_randomization(
        IntervalRandomizationPlan(
            body_ids=np.asarray([backend._base_body_id], dtype=np.int32),
            body_linear_velocity_delta=delta,
        )
    )
    after = backend.get_base_lin_vel()
    np.testing.assert_allclose(after[[0, 3]], before[[0, 3]] + delta[[0, 3], 0, :], atol=1e-6)
    np.testing.assert_allclose(after[[1, 2]], before[[1, 2]], atol=1e-6)
    # The kick commits through the forward barrier, keeping sensors coherent.
    assert np.isfinite(backend.get_sensor_data("torso_upvector")).all()


def test_kp_randomization_changes_tracking_response(backend: Any) -> None:
    _reset_all(backend)
    rows = np.asarray([0, 2], dtype=np.int32)
    qpos, qvel = _stand_state(backend, rows.size)
    default_kp, _ = backend.get_actuator_gains()
    backend.set_state(
        rows,
        qpos,
        qvel,
        randomization=ResetRandomizationPayload(
            kp=np.broadcast_to(default_kp * 0.05, (rows.size, default_kp.size)).copy(),
        ),
    )
    target_offset = 0.2
    ctrl = _stand_ctrl(backend) + target_offset
    for _ in range(5):
        backend.step(ctrl, nsteps=3)
    error = np.abs(backend.get_dof_pos() - ctrl)
    weak = error[rows].mean(axis=1)
    strong = error[[1, 3]].mean(axis=1)
    assert np.all(weak > strong + 0.05)


def test_low_friction_world_changes_kick_response(backend: Any) -> None:
    _reset_all(backend)
    rows = np.asarray([1, 3], dtype=np.int32)
    qpos, qvel = _stand_state(backend, rows.size)
    geom_friction = _broadcast_rows(backend.get_geom_friction(), rows.size)
    geom_friction[..., 0] = 0.01
    backend.set_state(
        rows,
        qpos,
        qvel,
        randomization=ResetRandomizationPayload(geom_friction=geom_friction),
    )
    start = backend.get_base_pos()[:, :2].copy()
    delta = np.zeros((backend.num_envs, 1, 3), dtype=np.float32)
    delta[:, 0, 0] = 1.5
    backend.apply_interval_randomization(
        IntervalRandomizationPlan(
            body_ids=np.asarray([backend._base_body_id], dtype=np.int32),
            body_linear_velocity_delta=delta,
        )
    )
    for _ in range(6):
        backend.step(_stand_ctrl(backend), nsteps=3)
    assert np.isfinite(backend.get_base_pos()).all()
    displacement = np.linalg.norm(backend.get_base_pos()[:, :2] - start, axis=1)
    # Near-zero slide friction changes the kick response of the mutated worlds
    # while the default-friction worlds stay pairwise consistent.
    assert abs(displacement[1] - displacement[0]) > 0.003
    # Same-condition worlds agree up to contact-solver reduction noise.
    np.testing.assert_allclose(displacement[0], displacement[2], rtol=1e-3)
    np.testing.assert_allclose(displacement[1], displacement[3], rtol=1e-3)
    assert displacement[1] < displacement[0]


def test_dr_reset_bypasses_reset_scratch_graphs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model-row mutation forces the full-forward route on large batches."""
    _require_cuda_mjwarp()
    backend = create_backend("mjwarp", _scene(), 1024, _SIM_DT, base_name="pelvis")
    if not backend._cuda_graph_enabled:
        pytest.skip(f"CUDA graphs unavailable: {backend._cuda_graph_disable_reason}")
    assert backend._reset_scratch_forward_graph is not None

    rows = np.arange(32, dtype=np.int32)
    qpos, qvel = _stand_state(backend, rows.size)

    graph_launches: list[Any] = []
    original_capture_launch = backend._warp.capture_launch

    def capture_launch(graph: Any) -> None:
        graph_launches.append(graph)
        original_capture_launch(graph)

    monkeypatch.setattr(backend._warp, "capture_launch", capture_launch)
    backend.set_state(rows, qpos, qvel)
    assert backend._reset_scratch_reset_graph in graph_launches

    graph_launches.clear()
    body_mass = _broadcast_rows(backend.get_body_mass(), rows.size)
    backend.set_state(
        rows,
        qpos,
        qvel,
        randomization=ResetRandomizationPayload(body_mass=body_mass),
    )
    assert backend._reset_scratch_reset_graph not in graph_launches
    assert backend._reset_scratch_forward_graph not in graph_launches
    assert backend._reset_graph in graph_launches
    assert backend._forward_graph in graph_launches
    assert np.isfinite(backend.get_base_pos()).all()


def test_mass_and_com_randomization_match_mujoco_statistics() -> None:
    """Same sampled payloads on both backends drift to the same statistics."""
    _require_cuda_mjwarp()
    from unilab.base.backend.mujoco.backend import MuJoCoBackend

    num_envs = 16
    scene = _scene()
    mujoco_backend = MuJoCoBackend(scene, num_envs, _SIM_DT, base_name="pelvis")
    mujoco_backend.materialize()
    mjwarp_backend = create_backend("mjwarp", scene, num_envs, _SIM_DT, base_name="pelvis")

    rng = np.random.default_rng(0)
    rows = np.arange(num_envs, dtype=np.int32)
    qpos, qvel = _stand_state(mjwarp_backend, num_envs)
    mass_delta = rng.uniform(-0.5, 1.5, size=num_envs)
    com_offset = rng.uniform(-0.02, 0.02, size=(num_envs, 3))
    payload = ResetRandomizationPayload(
        base_mass_delta=mass_delta,
        base_com_offset=com_offset,
    )

    mujoco_backend.set_state(rows, qpos, qvel, randomization=payload)
    mjwarp_backend.set_state(rows, qpos, qvel, randomization=payload)
    np.testing.assert_allclose(
        _device_field(mjwarp_backend, "body_mass")[:, mjwarp_backend._base_body_id],
        mujoco_backend.get_body_mass()[mujoco_backend._base_body_id] + mass_delta,
        rtol=1e-5,
    )

    ctrl = _stand_ctrl(mjwarp_backend)
    for _ in range(30):
        mujoco_backend.step(ctrl, nsteps=3)
        mjwarp_backend.step(ctrl, nsteps=3)

    for name, mujoco_values, mjwarp_values in (
        ("base_pos", mujoco_backend.get_base_pos(), mjwarp_backend.get_base_pos()),
        (
            "base_lin_vel",
            mujoco_backend.get_base_lin_vel(),
            mjwarp_backend.get_base_lin_vel(),
        ),
    ):
        mujoco_values = np.asarray(mujoco_values, dtype=np.float64)
        mjwarp_values = np.asarray(mjwarp_values, dtype=np.float64)
        # The sampled payload visibly spreads the per-world trajectories.
        assert np.std(mjwarp_values, axis=0).max() > 1e-4, name
        assert np.std(mujoco_values, axis=0).max() > 1e-4, name
        # Distribution-level agreement, not bitwise parity across float32/float64.
        np.testing.assert_allclose(
            mjwarp_values.mean(axis=0), mujoco_values.mean(axis=0), atol=0.05, err_msg=name
        )
        np.testing.assert_allclose(
            mjwarp_values.std(axis=0), mujoco_values.std(axis=0), atol=0.05, err_msg=name
        )
