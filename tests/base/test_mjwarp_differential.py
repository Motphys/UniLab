"""Production short-trajectory differential for independent ``mjwarp``.

The oracle intentionally observes only public backend state and sensors.  It
does not inspect Warp arrays, MuJoCo pool state, or the benchmark monkey-patch
path, so an indexing/control/cache error must be visible at the public
``SimBackend`` boundary.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from unilab.base.backend import create_backend
from unilab.base.backend.mjwarp.dependencies import load_mjwarp_dependencies
from unilab.base.scene import SceneCfg

pytestmark = pytest.mark.slow

_STEPS = 100
_ATOL = 1.0e-4
_RTOL = 1.0e-3
_SENSOR_NAMES = ("torso_upvector", "pelvis_local_linvel", "torso_gyro")


def _require_cuda_mjwarp() -> None:
    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.fail("mjwarp trajectory differential requires an active CUDA Warp device")


def _scene() -> SceneCfg:
    from unilab.assets import ASSETS_ROOT_PATH

    return SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml"))


def _initial_state(backend: Any, num_envs: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a frozen, public G1 standing state without backend-private metadata."""
    stand_qpos = backend.get_keyframe_qpos("stand")
    qpos = np.broadcast_to(stand_qpos, (num_envs, stand_qpos.size)).copy()
    qvel = np.zeros((num_envs, backend.get_init_qvel().size), dtype=np.float32)
    return np.asarray(qpos, dtype=np.float32), qvel


def _control(backend: Any, step: int, seed: int) -> np.ndarray:
    """Return a stable, deterministic action schedule shared by every world."""
    actuator_ids = np.arange(backend.num_actuators, dtype=np.float32)
    # This low-amplitude schedule is deliberately contact-stable for all
    # preregistered seeds/batches while still exercising nonzero actuator
    # control at every step.  It avoids turning a backend parity oracle into
    # a chaotic long-horizon benchmark with a different acceptance question.
    action = 0.005 * np.sin(0.13 * actuator_ids + 0.17 * step + 0.13 * seed)
    return np.broadcast_to(action, (backend.num_envs, backend.num_actuators)).copy()


def _snapshot(backend: Any) -> dict[str, np.ndarray]:
    """Copy public observables so later backend mutations cannot change evidence."""
    fields = {
        "base_pos": backend.get_base_pos(),
        "base_quat": backend.get_base_quat(),
        "base_lin_vel": backend.get_base_lin_vel(),
        "base_ang_vel": backend.get_base_ang_vel(),
        "dof_pos": backend.get_dof_pos(),
        "dof_vel": backend.get_dof_vel(),
    }
    fields.update({f"sensor.{name}": backend.get_sensor_data(name) for name in _SENSOR_NAMES})
    return {name: np.asarray(value).copy() for name, value in fields.items()}


def _run_mjwarp(num_envs: int, seed: int) -> dict[str, np.ndarray]:
    backend = create_backend("mjwarp", _scene(), num_envs, 0.02 / 3.0, base_name="pelvis")
    qpos, qvel = _initial_state(backend, num_envs)
    backend.set_state(np.arange(num_envs, dtype=np.int32), qpos, qvel)

    samples: dict[str, list[np.ndarray]] = {}
    for step in range(_STEPS + 1):
        for name, value in _snapshot(backend).items():
            samples.setdefault(name, []).append(value)
        if step < _STEPS:
            backend.step(_control(backend, step, seed))
    return {name: np.stack(values) for name, values in samples.items()}


def _run_pair(num_envs: int, seed: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Run the factory-routed production backends from the same public state."""
    mujoco_backend = create_backend("mujoco", _scene(), num_envs, 0.02 / 3.0, base_name="pelvis")
    mujoco_backend.materialize()
    mjwarp_backend = create_backend("mjwarp", _scene(), num_envs, 0.02 / 3.0, base_name="pelvis")

    qpos, qvel = _initial_state(mujoco_backend, num_envs)
    rows = np.arange(num_envs, dtype=np.int32)
    mujoco_backend.set_state(rows, qpos, qvel)
    mjwarp_backend.set_state(rows, qpos, qvel)

    mujoco_samples: dict[str, list[np.ndarray]] = {}
    mjwarp_samples: dict[str, list[np.ndarray]] = {}
    for step in range(_STEPS + 1):
        for name, value in _snapshot(mujoco_backend).items():
            mujoco_samples.setdefault(name, []).append(value)
        for name, value in _snapshot(mjwarp_backend).items():
            mjwarp_samples.setdefault(name, []).append(value)
        if step < _STEPS:
            ctrl = _control(mujoco_backend, step, seed)
            mujoco_backend.step(ctrl)
            mjwarp_backend.step(ctrl)

    return (
        {name: np.stack(values) for name, values in mujoco_samples.items()},
        {name: np.stack(values) for name, values in mjwarp_samples.items()},
    )


def _align_quaternion_sign(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Treat `q` and `-q` as the same public orientation representation."""
    aligned = candidate.copy()
    signs = np.where(np.sum(reference * aligned, axis=-1, keepdims=True) < 0.0, -1.0, 1.0)
    return aligned * signs


def _assert_close(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    *,
    num_envs: int,
    seed: int,
    comparison: str,
) -> None:
    assert reference.keys() == candidate.keys()
    for name, expected in reference.items():
        actual = candidate[name]
        if name == "base_quat":
            actual = _align_quaternion_sign(expected, actual)
        close = np.isclose(actual, expected, rtol=_RTOL, atol=_ATOL)
        if bool(np.all(close)):
            continue
        mismatch = tuple(
            int(value) for value in np.unravel_index(np.flatnonzero(~close)[0], close.shape)
        )
        max_error = float(np.max(np.abs(actual - expected)))
        raise AssertionError(
            f"{comparison} public trajectory mismatch "
            f"seed={seed} num_envs={num_envs} field={name!r} "
            f"first_index={mismatch} max_abs_error={max_error:.9g} "
            f"atol={_ATOL} rtol={_RTOL}"
        )


@pytest.mark.parametrize("num_envs", (1, 32), ids=("batch-1", "batch-32"))
@pytest.mark.parametrize("seed", (0, 1, 2))
def test_g1_short_trajectory_matches_mujoco(num_envs: int, seed: int) -> None:
    """Compare frozen G1 public trajectories and repeat-run Warp determinism."""
    _require_cuda_mjwarp()
    mujoco_trajectory, mjwarp_trajectory = _run_pair(num_envs, seed)
    _assert_close(
        mujoco_trajectory,
        mjwarp_trajectory,
        num_envs=num_envs,
        seed=seed,
        comparison="mujoco/mjwarp",
    )

    repeated_mjwarp_trajectory = _run_mjwarp(num_envs, seed)
    _assert_close(
        mjwarp_trajectory,
        repeated_mjwarp_trajectory,
        num_envs=num_envs,
        seed=seed,
        comparison="mjwarp repeat-run",
    )
