# Extending UniLab: New Backend

Read {doc}`../contracts/backend_capability` and
{doc}`/adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot` before
adding backend code.

## Current Backend Shape

The repository currently recognizes `mujoco` and `motrix` in two important
places:

- `registry.register_env(...)` in `src/unilab/base/registry.py`
- `create_backend(...)` in `src/unilab/base/backend/__init__.py`

A third backend is therefore an architecture change, not just a new package.

## Implementation Checklist

1. Implement a `SimBackend` subclass under `src/unilab/base/backend/<backend>/`.
2. Add backend construction to `create_backend(...)`.
3. Update registry validation if the new backend should be accepted by
   `@registry.env(..., sim_backend=...)`.
4. Expose backend-specific optional capabilities through `SimBackend` methods,
   `BackendPlayCapabilities`, domain-randomization capability methods, or a new
   explicit abstract method.
5. Add task owner YAMLs for supported task/backend pairs. Do not rely on
   `training.sim_backend=<backend>` as an override.
6. Keep XML, asset, and model inspection in backend init or materialization code.
   Hot env paths should receive cached IDs, arrays, or declared backend methods.

## Validation Near Risk

- Backend import and interface tests: `tests/base/test_backend_imports.py`,
  `tests/base/test_sim_backend.py`
- Backend-specific behavior tests near the feature boundary, such as
  `tests/base/test_motrix_backend_options.py`
- Task/backend config composition: `tests/config/test_config_system.py`

## Evidence In Repo

- Backend interface: `src/unilab/base/backend/base.py`
- Backend factory: `src/unilab/base/backend/__init__.py`
- MuJoCo backend package: `src/unilab/base/backend/mujoco/`
- Motrix backend package: `src/unilab/base/backend/motrix/`
