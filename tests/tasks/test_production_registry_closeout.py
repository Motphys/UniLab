"""Repository boundary tests for the production Manager-Based registry.

Pin the post-migration production registry so no legacy fallback or dual
registration can come back:

- the production registry matches the declared production task matrix,
- every registered factory is one of the canonical manager-runtime
  callables (generic factory plus the maintainer-approved wrappers),

The legacy ``EnvCfg -> NpEnv`` factory seam (``unilab.tasks.compatibility``)
has been removed: legacy task factories cannot coexist with the canonical
Manager-Based runtime, and this suite keeps it that way.

Scope note: the registry has no unregister API and no provenance tracking, and
the pytest session pollutes it with fixture-only envs (``DummyFlatTest`` via
``UNILAB_EXTRA_REGISTRY_PACKAGES``, the cartpole fixtures reusing
``ManagerBasedRlEnvCfg``/``make_manager_based_rl_env``). The registry snapshot
is therefore taken in a fresh subprocess with that env var scrubbed
(``tests/base/test_backend_imports.py`` idiom), so only the production
``unilab.tasks`` bootstrap contributes registrations.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

from unilab.tasks.migration_matrix import PRODUCTION_TASK_NAMES

CANONICAL_MANAGER_RUNTIME_FACTORIES = (
    ("unilab.envs.manager_based_rl_env", "make_manager_based_rl_env"),
    # Approved wrapper: G1WalkManagerBasedEnv subclass owning the G1 walk
    # manager-based production runtime.
    ("unilab.tasks.locomotion.g1.manager_terms", "make_g1_walk_env"),
    # Approved wrapper: cold-path untracked X2 mesh resolution before
    # delegating to the generic factory.
    ("unilab.tasks.motion_tracking.x2", "make_x2_wall_flip_env"),
)

_SNAPSHOT_CODE = textwrap.dedent(
    """
    import json

    from unilab.base import registry

    registry.ensure_registries()
    snapshot = {
        name: {
            backend: [
                getattr(factory, "__module__", None) or type(factory).__module__,
                getattr(factory, "__qualname__", None) or type(factory).__qualname__,
            ]
            for backend, factory in meta.env_factory_dict.items()
        }
        for name, meta in registry._envs.items()
    }
    print(json.dumps(snapshot))
    """
)

_snapshot_cache: dict[str, dict[str, tuple[str, str]]] | None = None


def _production_factories() -> dict[str, dict[str, tuple[str, str]]]:
    """Snapshot the production registry in a clean interpreter.

    Returns ``{task: {backend: (factory_module, factory_qualname)}}``. The
    subprocess scrubs ``UNILAB_EXTRA_REGISTRY_PACKAGES`` so fixture-only test
    envs injected by ``tests/conftest.py`` cannot leak into the snapshot.
    """
    global _snapshot_cache
    if _snapshot_cache is None:
        env = {
            key: value
            for key, value in os.environ.items()
            if key != "UNILAB_EXTRA_REGISTRY_PACKAGES"
        }
        result = subprocess.run(
            [sys.executable, "-c", _SNAPSHOT_CODE],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        raw = json.loads(result.stdout.strip().splitlines()[-1])
        _snapshot_cache = {
            name: {backend: tuple(factory) for backend, factory in backends.items()}
            for name, backends in raw.items()
        }
    return _snapshot_cache


def test_production_registry_matches_declared_task_matrix() -> None:
    factories = _production_factories()

    assert set(factories) == set(PRODUCTION_TASK_NAMES), (
        "production registry must match the declared production task matrix: "
        f"missing={sorted(set(PRODUCTION_TASK_NAMES) - set(factories))}, "
        f"stray={sorted(set(factories) - set(PRODUCTION_TASK_NAMES))}"
    )
    empty = sorted(name for name, backends in factories.items() if not backends)
    assert empty == [], f"registered tasks without any backend: {empty}"


def test_all_factories_are_the_canonical_manager_runtime_factories() -> None:
    factories = _production_factories()

    offenders = [
        f"{task_name}/{backend_type}: {factory[0]}.{factory[1]}"
        for task_name, backends in sorted(factories.items())
        for backend_type, factory in sorted(backends.items())
        if factory not in CANONICAL_MANAGER_RUNTIME_FACTORIES
    ]
    assert offenders == [], (
        "registered factories must be one of the canonical manager-runtime "
        f"factories {[qualname for _, qualname in CANONICAL_MANAGER_RUNTIME_FACTORIES]}; "
        "legacy task factories do not coexist with the Manager-Based runtime: "
        f"{offenders}"
    )
