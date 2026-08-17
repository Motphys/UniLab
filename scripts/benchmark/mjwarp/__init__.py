"""Benchmark-only mjwarp backend adapter.

Lives outside ``src/unilab/`` because it is wired in via a monkey-patch from
``scripts/benchmark/env/benchmark_env_step.py`` and is not part of the training-path
``SimBackend`` factory.
"""
