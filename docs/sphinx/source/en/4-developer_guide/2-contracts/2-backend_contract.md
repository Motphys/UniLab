# Backend Capability Contract

Backend differences are contract boundaries, not script-level special cases.
The play/render decision is recorded in
{doc}`/adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot`.

## Stable Backend Interface

All env-facing backend calls should go through `SimBackend` in
`src/unilab/base/backend/base.py`. The interface includes base state, DOF state,
body state in world and baselink frames, named sensors, state reset, physics
stepping, domain-randomization hooks, and optional playback/render methods.

Optional capabilities are explicit:

- `BackendPlayCapabilities` reports native interactive rendering,
  physics-state playback, and native video capture support.
- `BackendHeightScanner` and `create_hfield_scanner(...)` expose terrain scan
  support through a reusable backend-owned object.
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

## Evidence In Repo

- Backend interface and play capabilities: `src/unilab/base/backend/base.py`
- Backend factory: `src/unilab/base/backend/__init__.py`
- MuJoCo backend: `src/unilab/base/backend/mujoco/backend.py`
- Motrix backend: `src/unilab/base/backend/motrix/backend.py`
- Backend contract tests: `tests/base/test_sim_backend.py`,
  `tests/base/test_backend_imports.py`, `tests/base/test_motrix_backend_options.py`
