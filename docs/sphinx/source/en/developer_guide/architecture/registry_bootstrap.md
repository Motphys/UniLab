# Registry Bootstrap

Registry bootstrap is an explicit import contract for environments. It is
defined by {doc}`/adr/ADR-0004-registry-bootstrap-contract` and implemented in
`src/unilab/base/registry.py`.

## Runtime Flow

1. Training entrypoints call `unilab.training.common.ensure_registries()`.
2. That helper delegates to `unilab.base.registry.ensure_registries()`.
3. The registry imports declared bootstrap packages:
   `unilab.envs.locomotion`, `unilab.envs.manipulation`, and
   `unilab.envs.motion_tracking`.
4. Each package exposes `__unilab_registry_modules__`, a tuple of modules that
   contain registration side effects.
5. Imported modules register configs with `@registry.envcfg(...)` and env
   implementations with `@registry.env(..., sim_backend=...)` or
   `registry.register_env(...)`.
6. Runtime construction goes through `registry.make(...)`, which applies env
   config overrides, validates the env config, selects the requested backend,
   and instantiates the registered env class.

## Extension Rules

- Add new env modules to the package-level `__unilab_registry_modules__` tuple
  if they live in a new module that is not imported by an existing bootstrap
  entry.
- Keep registration cheap. Scene materialization, XML processing, asset access,
  and backend construction belong after `registry.make(...)`, not in decorator
  registration.
- Duplicate env configs and duplicate `(env, sim_backend)` registrations raise
  `ValueError` in `src/unilab/base/registry.py`; preserve that failure boundary.

## Evidence In Repo

- Bootstrap helper: `src/unilab/base/registry.py`
- Training helper: `src/unilab/training/common.py`
- Package declarations: `src/unilab/envs/locomotion/__init__.py`,
  `src/unilab/envs/manipulation/__init__.py`,
  `src/unilab/envs/motion_tracking/__init__.py`
- Tests: `tests/base/test_registry.py`, `tests/utils/test_algo_utils.py`
