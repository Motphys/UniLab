"""IsaacGym fixed-variant integration through the deterministic protocol mock."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from unisim.backend.isaacgym.backend import IsaacGymBackend
from unisim.dr.types import FixedVariantPlan, ModelSourceDescriptor

from unilab.base.backend_factory import create_backend
from unilab.base.scene import SceneCfg

_MOCK_WORKER = Path(__file__).resolve().parent / "isaacgym_mock_worker.py"
_SIM_DT = 0.005
_DOF_NAMES = ("j0", "j1", "j2")
_BODY_NAMES = ("base", "link0", "link1", "link2")


def _variant_xml(
    *,
    mass: float,
    size: float,
    key_dof: tuple[float, float, float],
    kp: tuple[float, float, float],
) -> str:
    return f"""<mujoco model="IsaacGymFixedVariant">
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="base">
      <freejoint/>
      <geom name="base_geom" type="box" size="{size} {size} {size}" mass="{mass}"/>
      <body name="link0">
        <joint name="j0" type="hinge" range="-1.5 1.5"/>
        <geom name="g0" type="box" size="0.1 0.1 0.1"/>
        <body name="link1">
          <joint name="j1" type="hinge" range="-1.5 1.5"/>
          <geom name="g1" type="box" size="0.1 0.1 0.1"/>
          <body name="link2">
            <joint name="j2" type="hinge" range="-1.5 1.5"/>
            <geom name="g2" type="box" size="0.1 0.1 0.1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="j0" joint="j0" kp="{kp[0]}" kv="0.5" forcerange="-100 100"/>
    <position name="j1" joint="j1" kp="{kp[1]}" kv="0.5" forcerange="-100 100"/>
    <position name="j2" joint="j2" kp="{kp[2]}" kv="0.5" forcerange="-100 100"/>
  </actuator>
  <keyframe>
    <key name="home" qpos="0 0 0.8 1 0 0 0 {key_dof[0]} {key_dof[1]} {key_dof[2]}"/>
  </keyframe>
</mujoco>
"""


def _write_variants(root: Path, *, count: int = 3) -> tuple[Path, ...]:
    files: list[Path] = []
    for index in range(count):
        path = root / f"variant_{index}.xml"
        path.write_text(
            _variant_xml(
                mass=1.0 + 0.25 * index,
                size=0.08 + 0.01 * index,
                key_dof=(0.1 * index, 0.2, -0.1),
                kp=(20.0 + 10.0 * index, 30.0, 40.0),
            ),
            encoding="utf-8",
        )
        files.append(path)
    return tuple(files)


def _make_backend(
    sources: tuple[Path, ...],
    assignment: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> IsaacGymBackend:
    monkeypatch.setenv("UNILAB_ISAACGYM_MOCK_DOF_NAMES", ",".join(_DOF_NAMES))
    monkeypatch.setenv("UNILAB_ISAACGYM_MOCK_BODY_NAMES", ",".join(_BODY_NAMES))
    plan = FixedVariantPlan(
        assignment=np.asarray(assignment, dtype=np.int32),
        variants=tuple(ModelSourceDescriptor(str(path)) for path in sources),
    )
    backend = create_backend(
        "isaacgym",
        SceneCfg(model_file=str(sources[0]), fixed_variant_plan=plan),
        len(assignment),
        _SIM_DT,
        base_name="base",
        worker_command=[sys.executable, str(_MOCK_WORKER)],
        worker_timeout_s=30.0,
    )
    assert isinstance(backend, IsaacGymBackend)
    return backend


def test_mock_protocol_realizes_immutable_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _write_variants(tmp_path)
    backend = _make_backend(sources, (1, 0, 2, 1), monkeypatch)
    try:
        capabilities = backend.get_dr_capabilities()
        assert capabilities.supports_fixed_variants
        assert capabilities.supports_per_env_playback

        backend.materialize()
        assert backend.model.dof_names == _DOF_NAMES
        assert backend.model.body_names == _BODY_NAMES
        assert [backend.get_playback_model(index) for index in range(4)] == [
            str(sources[1]),
            str(sources[0]),
            str(sources[2]),
            str(sources[1]),
        ]
        # Assignment row 0 selects variant 1, so the canonical default is that
        # variant's task-initial keyframe rather than scene.model_file's row.
        np.testing.assert_allclose(
            backend.get_default_dof_pos(), np.asarray([0.1, 0.2, -0.1]), atol=1e-7
        )
        np.testing.assert_allclose(
            backend.get_dof_pos(),
            np.asarray(
                [
                    [0.1, 0.2, -0.1],
                    [0.0, 0.2, -0.1],
                    [0.2, 0.2, -0.1],
                    [0.1, 0.2, -0.1],
                ],
                dtype=np.float32,
            ),
            atol=1e-7,
        )
        backend.step(np.zeros((4, 3), dtype=np.float32), nsteps=1)
        assert np.isfinite(backend.get_dof_pos()).all()
    finally:
        backend.close()


def test_public_layout_drift_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = list(_write_variants(tmp_path, count=2))
    drifting = tmp_path / "drifting.xml"
    drifting.write_text(
        _variant_xml(
            mass=2.0,
            size=0.2,
            key_dof=(0.3, 0.2, -0.1),
            kp=(50.0, 30.0, 40.0),
        ).replace(
            '<joint name="j2" type="hinge" range="-1.5 1.5"/>',
            '<joint name="j2" type="hinge" range="-1.5 1.5"/>'
            '<body name="extra_link">'
            '<joint name="extra" type="hinge" range="-1 1"/>'
            '<geom name="extra_geom" type="box" size="0.1 0.1 0.1"/>'
            "</body>",
        ),
        encoding="utf-8",
    )
    sources.append(drifting)
    backend = _make_backend(tuple(sources), (0, 1, 0), monkeypatch)
    try:
        with pytest.raises(ValueError, match="changes public joint_names"):
            backend.materialize()
    finally:
        backend.close()


@pytest.mark.slow
def test_real_isaacgym_runtime_fixed_variants(tmp_path: Path) -> None:
    """Real Preview-4 lane; skips when the dedicated runtime is unavailable."""
    from unisim.backend.isaacgym.dependencies import isaacgym_runtime_available

    if not isaacgym_runtime_available():
        pytest.skip("IsaacGym Preview 4 runtime is not installed")

    sources = _write_variants(tmp_path)
    plan = FixedVariantPlan(
        assignment=np.asarray([0, 1, 2], dtype=np.int32),
        variants=tuple(ModelSourceDescriptor(str(path)) for path in sources),
    )
    backend = create_backend(
        "isaacgym",
        SceneCfg(model_file=str(sources[0]), fixed_variant_plan=plan),
        3,
        _SIM_DT,
        base_name="base",
        device_id=-1,
        worker_timeout_s=120.0,
    )
    assert isinstance(backend, IsaacGymBackend)
    try:
        backend.materialize()
        assert [backend.get_playback_model(index) for index in range(3)] == [
            str(path) for path in sources
        ]
        backend.step(np.zeros((3, 3), dtype=np.float32), nsteps=1)
        assert np.isfinite(backend.get_dof_pos()).all()
        assert np.isfinite(backend.get_base_pos()).all()
    finally:
        backend.close()
