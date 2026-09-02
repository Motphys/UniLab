"""CPU affinity wiring tests for the MuJoCo BatchEnvPool adapter (issue #959).

Covers the UniLab side of the contract: ``EnvCfg.cpu_ids`` validation,
``env_backend_kwargs``/``create_backend`` routing, cold-path validation in
``MuJoCoBackend``, and the actual worker pinning exposed by mujoco-uni.
"""

import inspect
import os
from pathlib import Path

import pytest

from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.base.base import EnvCfg

pytest.importorskip("mujoco", reason="mujoco not installed")

try:
    from mujoco_uni.batch_env import BatchEnvPool
except Exception:
    pytest.skip(
        "mujoco_uni.batch_env not available (platform/libstdc++ issue)", allow_module_level=True
    )

if "cpu_ids" not in inspect.signature(BatchEnvPool.__init__).parameters:
    pytest.skip(
        "installed mujoco-uni-runtime has no cpu_ids support (pre-0.3.1)",
        allow_module_level=True,
    )

from unisim.backend.mujoco.backend import MuJoCoBackend

from unilab.base.scene import SceneCfg

_MODEL_FILE = str(
    Path(__file__).resolve().parents[2] / "fixtures" / "mjlab_cartpole" / "cartpole.xml"
)
_NUM_ENVS = 4
_BASE_NAME = "cart"

if not hasattr(os, "sched_getaffinity"):
    pytest.skip(
        "os.sched_getaffinity unavailable on this platform (e.g. macOS)",
        allow_module_level=True,
    )

_AVAILABLE_CPUS = sorted(os.sched_getaffinity(0))


def _build_small_backend(**backend_kwargs):
    backend = MuJoCoBackend(
        SceneCfg(model_file=_MODEL_FILE),
        num_envs=_NUM_ENVS,
        sim_dt=0.01,
        base_name=_BASE_NAME,
        adaptive_chunk_size=False,
        **backend_kwargs,
    )
    backend.materialize()
    return backend


def test_envcfg_cpu_ids_default_none():
    cfg = EnvCfg()
    assert cfg.cpu_ids is None
    cfg.validate()


def test_envcfg_cpu_ids_overridable():
    cfg = EnvCfg(cpu_ids=[0, 1])
    assert cfg.cpu_ids == [0, 1]
    cfg.validate()


@pytest.mark.parametrize(
    "cpu_ids",
    (
        [],  # empty
        [-1],  # negative
        [True],  # bool is not a CPU id
        [0, 0],  # duplicates
        ["0"],  # non-integer entry
    ),
)
def test_envcfg_cpu_ids_validation_rejects_bad_shapes(cpu_ids):
    with pytest.raises(ValueError):
        EnvCfg(cpu_ids=cpu_ids).validate()


def test_env_backend_kwargs_maps_cpu_ids():
    cfg = EnvCfg(cpu_ids=[2, 3])
    kw = env_backend_kwargs(cfg)
    assert kw["cpu_ids"] == [2, 3]
    assert env_backend_kwargs(EnvCfg())["cpu_ids"] is None


def test_create_backend_routes_cpu_ids():
    cpu_ids = _AVAILABLE_CPUS[:2]
    backend = create_backend(
        "mujoco",
        SceneCfg(model_file=_MODEL_FILE),
        _NUM_ENVS,
        0.01,
        base_name=_BASE_NAME,
        adaptive_chunk_size=False,
        cpu_ids=cpu_ids,
    )
    assert isinstance(backend, MuJoCoBackend)
    assert backend._cpu_ids == tuple(cpu_ids)
    assert backend._n_threads == len(cpu_ids)


@pytest.mark.parametrize("cpu_ids", ([0, 0], [-1], []))
def test_backend_rejects_invalid_cpu_ids_on_cold_path(cpu_ids):
    with pytest.raises(ValueError):
        MuJoCoBackend(
            SceneCfg(model_file=_MODEL_FILE),
            num_envs=_NUM_ENVS,
            sim_dt=0.01,
            base_name=_BASE_NAME,
            cpu_ids=cpu_ids,
        )


def test_workers_pinned_to_configured_cpus():
    cpu_ids = _AVAILABLE_CPUS[:2]
    backend = _build_small_backend(cpu_ids=cpu_ids)
    try:
        # Configured mapping is queryable and workers were observed on those CPUs.
        assert tuple(backend._pool.cpu_ids) == tuple(cpu_ids)
        assert backend._pool.worker_cpu_ids() == tuple(cpu_ids)
    finally:
        backend._pool.close()


def test_unavailable_cpu_id_fails_at_pool_creation():
    unavailable = max(_AVAILABLE_CPUS) + 4096
    backend = MuJoCoBackend(
        SceneCfg(model_file=_MODEL_FILE),
        num_envs=_NUM_ENVS,
        sim_dt=0.01,
        base_name=_BASE_NAME,
        cpu_ids=[unavailable],
    )
    with pytest.raises(ValueError, match="not available"):
        backend.materialize()


def test_default_path_keeps_os_scheduling():
    backend = _build_small_backend()
    try:
        assert backend._cpu_ids is None
        assert backend._pool.cpu_ids is None
        assert backend._pool.worker_cpu_ids() == ()
    finally:
        backend._pool.close()


def test_default_nthread_sized_to_effective_cpus():
    """Default pool sizing uses the CPUs usable by this process (#1328),
    not 2x the machine-wide count; explicit ``cpu_ids`` still fixes nthread."""
    backend = MuJoCoBackend(
        SceneCfg(model_file=_MODEL_FILE),
        num_envs=10_000,
        sim_dt=0.01,
        base_name=_BASE_NAME,
    )
    assert backend._cpu_ids is None
    assert backend._n_threads == len(os.sched_getaffinity(0))


def test_default_nthread_capped_by_num_envs():
    backend = MuJoCoBackend(
        SceneCfg(model_file=_MODEL_FILE),
        num_envs=2,
        sim_dt=0.01,
        base_name=_BASE_NAME,
    )
    assert backend._n_threads == 2
