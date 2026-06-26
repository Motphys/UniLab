import numpy as np
import pytest

from unilab.base.backend import env_backend_kwargs
from unilab.base.base import EnvCfg

pytest.importorskip("mujoco", reason="mujoco not installed")

try:
    from mujoco.batch_env import BatchEnvPool  # noqa: F401
except Exception:
    pytest.skip(
        "mujoco.batch_env not available (platform/libstdc++ issue)", allow_module_level=True
    )

import mujoco

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend.mujoco.backend import MuJoCoBackend
from unilab.base.scene import SceneCfg

_MODEL_FILE = str(ASSETS_ROOT_PATH / "robots" / "go2_arm" / "scene_flat.xml")
_NUM_ENVS = 4


def test_envcfg_chunk_size_defaults():
    cfg = EnvCfg()
    assert cfg.adaptive_chunk_size is True
    assert cfg.chunk_size is None


def test_envcfg_chunk_size_overridable():
    cfg = EnvCfg(adaptive_chunk_size=False, chunk_size=128)
    assert cfg.adaptive_chunk_size is False
    assert cfg.chunk_size == 128


def test_env_backend_kwargs_maps_fields():
    cfg = EnvCfg(ctrl_dt=0.02, sim_dt=0.005, chunk_size=64, adaptive_chunk_size=False)
    kw = env_backend_kwargs(cfg)
    assert kw["chunk_size"] == 64
    assert kw["adaptive_chunk_size"] is False
    assert kw["bench_nsteps"] == cfg.sim_substeps == 4  # round(0.02/0.005)
    assert kw["post_step_forward_sensor"] == cfg.post_step_forward_sensor
    assert "motrix_max_iterations" in kw


def _build_small_backend(**backend_kwargs):
    """Build a minimal real MuJoCoBackend.

    Mirrors the SceneCfg + construction used in
    ``tests/base/backend/test_mujoco_site_jacobian.py``, forwarding
    ``**backend_kwargs`` into ``MuJoCoBackend(...)`` and calling ``.materialize()``.
    """
    backend = MuJoCoBackend(
        SceneCfg(model_file=_MODEL_FILE),
        num_envs=_NUM_ENVS,
        sim_dt=0.01,
        base_name="base",
        **backend_kwargs,
    )
    backend.materialize()
    return backend


def test_step_passes_resolved_chunk_size(monkeypatch):
    backend = _build_small_backend(chunk_size=7, adaptive_chunk_size=False)
    assert backend._chunk_size == 7

    seen = {}
    real_step = backend._pool.step

    def _spy(state, **kw):
        seen["chunk_size"] = kw.get("chunk_size")
        return real_step(state, **kw)

    monkeypatch.setattr(backend._pool, "step", _spy)
    nu = backend._model.nu
    backend.step(np.zeros((backend.num_envs, nu), dtype=np.float64), nsteps=1)
    assert seen["chunk_size"] == 7


def test_hot_path_does_no_xml_parse(monkeypatch):
    """Acceptance ③: step/reset must not parse asset/XML (any entrypoint)."""
    backend = _build_small_backend(adaptive_chunk_size=False)  # cold path done

    # Install all parse spies AFTER the cold-path materialize so only hot-path
    # (step) parses are counted. Spy multiple XML entrypoints, not just MjSpec.
    spec_calls = {"n": 0}
    model_calls = {"n": 0}
    orig_from_file = mujoco.MjSpec.from_file
    orig_from_xml_path = mujoco.MjModel.from_xml_path

    def _counting_from_file(*a, **k):
        spec_calls["n"] += 1
        return orig_from_file(*a, **k)

    def _counting_from_xml_path(*a, **k):
        model_calls["n"] += 1
        return orig_from_xml_path(*a, **k)

    monkeypatch.setattr(mujoco.MjSpec, "from_file", staticmethod(_counting_from_file))
    monkeypatch.setattr(mujoco.MjModel, "from_xml_path", staticmethod(_counting_from_xml_path))
    nu = backend._model.nu
    backend.step(np.zeros((backend.num_envs, nu), dtype=np.float64), nsteps=1)
    assert spec_calls["n"] == 0
    assert model_calls["n"] == 0


def test_benchmark_runs_and_logs_table_on_adaptive(caplog, monkeypatch, tmp_path):
    """Acceptance ④: adaptive path benchmarks and emits a per-candidate table.

    Point the chunk_size cache at an empty ``tmp_path`` file so the resolve is a
    guaranteed MISS (no warm-cache short-circuit) and never pollutes the real
    ``~/.cache/unilab/chunk_size.json``. A clean miss forces the benchmark path,
    so we can assert the benchmark-table INFO record specifically.
    """
    import logging

    from unilab.base.backend.mujoco import backend as backend_mod

    # Force nthread < num_envs so there is genuinely something to tune; otherwise the
    # resolve short-circuits (num_envs <= nthread => one chunk => no benchmark). With
    # cpu_count()==1, nthread = min(_NUM_ENVS, 2) = 2 < _NUM_ENVS, deterministically.
    monkeypatch.setattr(backend_mod, "cpu_count", lambda: 1)
    monkeypatch.setenv("UNILAB_CHUNK_SIZE_CACHE", str(tmp_path / "chunk_size.json"))
    with caplog.at_level(logging.INFO, logger="unilab.base.backend.mujoco.chunk_tuner"):
        backend = _build_small_backend(adaptive_chunk_size=True, chunk_size=None)
    assert backend._chunk_size is None or isinstance(backend._chunk_size, int)
    # Forced miss -> _log_benchmark_table emits the "chunk_size benchmark" record.
    assert any("chunk_size benchmark" in r.message for r in caplog.records)
