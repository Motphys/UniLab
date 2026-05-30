# Contributing To UniLab

This page summarizes the repository workflow for contributors. Contract and
architecture details live in {doc}`1-architecture/1-overview`.

## Environment

```bash
uv sync
uv sync --extra motrix
make sync-rocm
make sync-xpu
```

Use `uv run` for commands. Do not invoke `python` directly outside `uv run`.

## Common Commands

```bash
make format
make type
make check
make test
make test-cov
make test-slow
make test-all
```

For docs-only changes, run:

```bash
uv run pytest tests/scripts/test_check_docs.py -q
cd docs/sphinx
UNILAB_DOCS_SKIP_AUTODOC=1 uv run --no-project --with-requirements requirements.txt sphinx-build -b html -n source build/html
```

The `Docs` GitHub Actions workflow runs the same prose-only build on matching
PRs and pushes, and it can also be started from the GitHub Actions web UI via
`workflow_dispatch`. It does not install UniLab with `pip install -e .`, does
not generate API reference pages, and does not publish the external docs
repository.

For a final local refresh of the full site, including API reference pages for
the `UniLab-doc` publication flow, use a parallel Sphinx build from a synced
developer environment:

```bash
uv sync
uv pip install -r docs/sphinx/requirements.txt
cd docs/sphinx
uv run --no-sync sphinx-build -j auto -b html -n source build/html
```

## Commit And PR Expectations

- Use Conventional Commits such as `feat:`, `fix:`, `docs:`, `refactor:`,
  `test:`, and `chore:`.
- Link the driving issue in the PR.
- List the validation commands actually run.
- State whether behavior differs between MuJoCo, Motrix, macOS, or Linux.
- For code/config changes, run the nearest tests for the changed contract before
  relying on top-level smoke commands.

## Documentation Expectations

- Commands must point to checked-in scripts, package entrypoints, Makefile
  targets, or config owners.
- Backend and task support claims should use evidence grades such as
  `Registered`, `Configured`, `Tested`, `Benchmarked`, or `Recommended`.
- Do not describe `training.sim_backend=<backend>` as a standalone backend
  switch. Use `--sim <backend>` in user-facing commands and select the owner
  YAML path internally.
- Keep English pages free of manual navigation blocks.

## Configuration Changes

Task, backend, reward, and algorithm selection belongs in Hydra owner YAMLs.
When adding or changing a runnable path, update the relevant owner config under
`conf/` and verify script composition with tests under `tests/config/` or
`tests/scripts/`.

See {doc}`2-contracts/3-task_owner` and
{doc}`../2-user_guide/1-training/2-hydra_config`.
