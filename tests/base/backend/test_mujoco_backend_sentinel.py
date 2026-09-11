"""Guard-integrity sentinels for the mujoco backend test safety net (#1554).

Every heavy mujoco suite gates itself on ``mjbatch`` + the unisim MuJoCo
adapter being importable, so a broken or missing install would silently skip
the whole safety net and leave the executor swap unverified. These sentinels
run unguarded in the fast lane and fail (never skip) when the guard cannot
admit a real backend, and prove the guard path end to end with a one-env
materialize/step — the same operations the gated suites exist to protect.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_MODEL_FILE = str(
    Path(__file__).resolve().parents[2] / "fixtures" / "mjlab_cartpole" / "cartpole.xml"
)


def test_mjbatch_and_unisim_mujoco_adapter_importable() -> None:
    """The exact guard expression that admits the gated mujoco suites."""
    import mjbatch  # noqa: F401
    import unisim.backend.mujoco.backend  # noqa: F401


def test_mujoco_backend_materializes_and_steps_under_suite_guard() -> None:
    """A real one-env backend materialize/step must succeed under that guard.

    Deliberately NOT importorskip-gated: if this errors, the gated heavy
    suites would have silently skipped, which is the failure mode #1554 pins.
    """
    from unisim.backend.mujoco.backend import MuJoCoBackend

    from unilab.base.scene import SceneCfg

    backend = MuJoCoBackend(
        SceneCfg(model_file=_MODEL_FILE),
        num_envs=1,
        sim_dt=0.01,
        base_name="cart",
    )
    backend.materialize()
    nu = backend.model.nu
    backend.step(np.zeros((1, nu), dtype=np.float64), nsteps=2)
    assert np.all(np.isfinite(backend.get_dof_pos()))
    assert np.all(np.isfinite(backend.get_dof_vel()))
