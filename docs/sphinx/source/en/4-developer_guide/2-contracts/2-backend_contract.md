# Backend Capability Contract

Backend differences are contract boundaries, not script-level special cases.
The play/render decision is recorded in
{doc}`/adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot`.

## Stable Backend Interface

All env-facing backend calls should go through `SimBackend` in
`unisim.backend.base`. The interface includes base state, DOF state,
body state in world and baselink frames, named sensors, state reset, physics
stepping, domain-randomization hooks, and optional playback/render methods.

Optional capabilities are explicit:

- `BackendPlayCapabilities` reports native interactive rendering,
  physics-state playback, and native video capture support.
- `BackendHeightScanner` and `create_hfield_scanner(...)` expose terrain scan
  support through a reusable backend-owned object.
- `BackendSensorView` and `bind_sensor_data(...)` validate ordered named sensors
  on the cold path and retain a backend-owned reader for finite, shape-stable
  NumPy batches. Manager hot paths do not inspect XML or model metadata.
- Domain randomization support is surfaced through `get_dr_capabilities()` and
  the init, reset, and interval randomization methods.
- Unsupported optional methods raise `NotImplementedError` from the base class.

## Rules For New Capability

- If shared env logic needs a new backend operation, add it to `SimBackend`
  first, with a default `NotImplementedError` if not every backend can support
  it immediately.
- Keep MuJoCo/Motrix differences in backend implementations, env adapters, and
  owner YAMLs. Do not add hot-path probes of backend private methods in env code.
- Asset/XML/model metadata access belongs to cold paths such as scene
  materialization, backend init, or cache creation.

## Physical entities and selected reset

The M2 consumer in issue #1599 uses UniSim physical entity declarations separately from UniLab logical selectors. `SceneCfg` materializes typed entity/variant values, calls the parent contract validation, and the asset factory collects physical and catalog source paths. `EntityCfg.physical_entity` binds a logical facade explicitly; `primary_entity` selects the scene's primary root without encoding a task name in the backend.

A mapped logical root must name its physical entity's declared root exactly. Binding a descendant body as the root is rejected during initialization, because root queries, defaults and reset writes must reference the same object. Reset staging maps selected rows in linear time and stores only requested fields; a current-state snapshot is needed only to fill missing columns when merging different joint position/velocity selections. The transaction still validates before its single public backend commit.

The existing `ResetStateTransaction` stages one public `SceneResetRequest` for mapped scenes. Missing fields and unselected entities/environments remain unchanged. Per-environment defaults come from `get_entity_default_state`, and `restore_default_controls` restores keyframe controls in the same commit; controls need not equal joint positions. No engine-private tensors or asset parsing enter manager terms. Scalar hinge/slide joints and a common selected environment set per transaction are the current consumer boundary; unsupported mixed DR/mocap or row patterns fail explicitly.

`tests/envs/test_multi_entity_consumer.py` registers one primitive task with the same pickleable EnvFactory for MuJoCo and IsaacSim. It checks observation/action dimensions, passive joints, selected resets, variants and a kinematic mirror. The native IsaacSim cases require `UNILAB_TEST_M2_ISAACSIM=1`. The consumer requires released `unisim-core>=1.5.0`; normal and ROCm lock profiles resolve the PyPI package without a Git source override. `UNILAB_LOCAL_UNISIM` remains an explicit alternative for local development. See [UniSim roadmap #108](https://github.com/unilabsim/unisim/issues/108) for implementation and verification scope.

## Evidence In Repo

- Configuration and asset preparation: `src/unilab/base/scene.py`, `src/unilab/base/backend_factory.py`.
- Public state/reset bindings: `src/unilab/base/entity.py`, `src/unilab/base/reset_state.py`.
- Registered runtime tests: `tests/base/test_entity_scene_consumer.py`, `tests/envs/test_multi_entity_consumer.py`.

- Backend interface and play capabilities: `unisim.backend.base`
- Backend factory: `src/unilab/base/backend_factory.py`
- MuJoCo backend: `unisim.backend.mujoco.backend`
- Motrix backend: `unisim.backend.motrix.backend`
- Backend contract tests: `tests/base/test_backend_sensor_view.py`,
  `tests/base/test_backend_conformance.py`, `tests/base/test_sim_backend.py`,
  `tests/base/test_backend_imports.py`, `tests/base/test_motrix_backend_options.py`
