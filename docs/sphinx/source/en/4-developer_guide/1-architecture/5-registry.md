# Registry Bootstrap

Registry bootstrap is an explicit import contract for environments. It is
defined by {doc}`/adr/ADR-0004-registry-bootstrap-contract` and implemented in
`src/unilab/base/registry.py`.

## Runtime Flow

1. Training entrypoints call `unilab.training.common.ensure_registries()`.
2. That helper delegates to `unilab.base.registry.ensure_registries()`.
3. The registry collects bootstrap packages from three sources: the built-in
   `unilab.tasks`, third-party packages declaring an entry point under
   `[project.entry-points."unilab.tasks"]` (the value is an importable package
   name), and the `UNILAB_EXTRA_REGISTRY_PACKAGES` environment variable
   (mainly for test fixtures).
4. Each bootstrap package exposes `__unilab_registry_modules__`, an explicit
   tuple of task leaf modules that contain registration side effects.
5. Imported modules register configs with `@registry.envcfg(...)` and env
   implementations with `@registry.env(..., sim_backend=...)` or
   `registry.register_env(...)`.
6. Runtime construction goes through `registry.make(...)`, which applies env
   config overrides, validates the env config, selects the requested backend,
   and instantiates the registered env class.

## Extension Rules

- Add new task leaves to `unilab.tasks.__unilab_registry_modules__` when they
  are not imported by an existing bootstrap entry.
- Third-party task packages living outside this repo self-register by declaring
  `[project.entry-points."unilab.tasks"]` in their own `pyproject.toml` (e.g.
  `microduck = "microduck_rl_unilab.tasks"`); the declared package exposes the
  same `__unilab_registry_modules__` tuple. Entry-point metadata lives in
  site-packages, so spawn-based collector subprocesses discover the same
  packages without any env-var forwarding; import failures of entry-point
  packages fail closed (installing the package is a deliberate act). The
  `UNILAB_EXTRA_REGISTRY_PACKAGES` env var remains available for test fixtures
  and ad-hoc debugging.
- Keep registration cheap. Scene materialization, XML processing, asset access,
  and backend construction belong after `registry.make(...)`, not in decorator
  registration.
- Duplicate env configs and duplicate `(env, sim_backend)` registrations raise
  `ValueError` in `src/unilab/base/registry.py`; preserve that failure boundary.

## Evidence In Repo

- Bootstrap helper: `src/unilab/base/registry.py`
- Training helper: `src/unilab/training/common.py`
- Task bootstrap declaration: `src/unilab/tasks/__init__.py`
- Tests: `tests/base/test_registry.py`, `tests/utils/test_algo_utils.py`
