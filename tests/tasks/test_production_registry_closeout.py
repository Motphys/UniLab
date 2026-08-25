"""Repository boundary tests closing out the Manager-Based migration (#1042).

Pin the post-migration production registry so no legacy fallback or dual
registration can come back:

- the production registry matches the #1042 migration matrix exactly,
- only the three approved families use the frozen LegacyFactoryAdapter seam,
- every other registered factory is one of the canonical manager-runtime
  callables (generic factory plus the three maintainer-approved wrappers),
- the deleted ``unilab.envs.{locomotion,manipulation,motion_tracking}``
  packages stay removed (``unilab.envs`` itself remains: it owns the
  manager-based runtime such as ``manager_based_rl_env`` and ``mdp``).

Scope note: the registry has no unregister API and no provenance tracking, and
the pytest session pollutes it with fixture-only envs (``DummyFlatTest`` via
``UNILAB_EXTRA_REGISTRY_PACKAGES``, the cartpole fixtures reusing
``ManagerBasedRlEnvCfg``/``make_manager_based_rl_env``). The registry snapshot
is therefore taken in a fresh subprocess with that env var scrubbed
(``tests/base/test_backend_imports.py`` idiom), so only the production
``unilab.tasks`` bootstrap contributes registrations.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap

from unilab.tasks.migration_matrix import PRODUCTION_TASK_NAMES

APPROVED_COMPATIBILITY_TASKS = {
    "Go2ArmManipLoco",
    "SharpaInhandRotation",
    "SharpaInhandRotationGrasp",
}

ADAPTER_FACTORY = ("unilab.tasks.compatibility", "LegacyFactoryAdapter")

CANONICAL_MANAGER_RUNTIME_FACTORIES = (
    ("unilab.envs.manager_based_rl_env", "make_manager_based_rl_env"),
    # Approved wrapper: G1WalkManagerBasedEnv subclass owning the G1 walk
    # manager-based production runtime.
    ("unilab.tasks.locomotion.g1.manager_terms", "make_g1_walk_env"),
    # Approved wrapper: cold-path untracked X2 mesh resolution before
    # delegating to the generic factory.
    ("unilab.tasks.motion_tracking.x2", "make_x2_wall_flip_env"),
    # Approved wrapper: cold-path T800 OBJ/texture resolution before
    # delegating to the generic factory.
    ("unilab.tasks.locomotion.t800", "make_t800_walk_env"),
)

REMOVED_LEGACY_ENV_PACKAGES = (
    "unilab.envs.locomotion",
    "unilab.envs.manipulation",
    "unilab.envs.motion_tracking",
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


def test_production_registry_matches_migration_matrix_exactly() -> None:
    factories = _production_factories()

    assert set(factories) == set(PRODUCTION_TASK_NAMES), (
        "production registry must match the #1042 migration matrix exactly: "
        f"missing={sorted(set(PRODUCTION_TASK_NAMES) - set(factories))}, "
        f"stray={sorted(set(factories) - set(PRODUCTION_TASK_NAMES))}"
    )
    empty = sorted(name for name, backends in factories.items() if not backends)
    assert empty == [], f"registered tasks without any backend: {empty}"


def test_only_approved_families_use_the_frozen_compatibility_seam() -> None:
    factories = _production_factories()
    adapter_tasks = {
        task_name
        for task_name, backends in factories.items()
        if any(factory == ADAPTER_FACTORY for factory in backends.values())
    }

    assert adapter_tasks == APPROVED_COMPATIBILITY_TASKS, (
        "LegacyFactoryAdapter is a frozen seam: only the approved families may use it, "
        f"unexpected={sorted(adapter_tasks - APPROVED_COMPATIBILITY_TASKS)}, "
        f"missing={sorted(APPROVED_COMPATIBILITY_TASKS - adapter_tasks)}"
    )

    offenders = [
        f"{task_name}/{backend_type}: {factory[1]}"
        for task_name in sorted(APPROVED_COMPATIBILITY_TASKS)
        for backend_type, factory in factories[task_name].items()
        if factory != ADAPTER_FACTORY
    ]
    assert offenders == [], (
        "approved compatibility families must route every backend through "
        f"LegacyFactoryAdapter: {offenders}"
    )


def test_all_other_factories_are_the_canonical_manager_runtime_factories() -> None:
    factories = _production_factories()

    offenders = [
        f"{task_name}/{backend_type}: {factory[0]}.{factory[1]}"
        for task_name, backends in sorted(factories.items())
        for backend_type, factory in sorted(backends.items())
        if factory != ADAPTER_FACTORY and factory not in CANONICAL_MANAGER_RUNTIME_FACTORIES
    ]
    assert offenders == [], (
        "non-adapter factories must be one of the canonical manager-runtime "
        f"factories {[qualname for _, qualname in CANONICAL_MANAGER_RUNTIME_FACTORIES]}: "
        f"{offenders}"
    )


def test_removed_legacy_env_packages_stay_removed() -> None:
    for module_name in REMOVED_LEGACY_ENV_PACKAGES:
        assert importlib.util.find_spec(module_name) is None, module_name
