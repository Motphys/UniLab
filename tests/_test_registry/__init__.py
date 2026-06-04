"""Test-only registry package.

Exists so that ``ensure_registries`` running inside a spawn collector subprocess
can register fixture envs (e.g. ``DummyFlatTest``) that pytest conftest would
normally register in the parent process. Activated via the
``UNILAB_EXTRA_REGISTRY_PACKAGES`` env var, which ``tests/conftest.py`` sets
during test collection.
"""

# Module list consumed by unilab.base.registry.ensure_registries().
__unilab_registry_modules__ = ("tests._test_registry.dummy_flat_env",)
