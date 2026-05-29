# Contributing To UniLab

This page summarizes the repository workflow for contributors. Contract and
architecture details live in {doc}`architecture/overview`.

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
UNILAB_DOCS_SKIP_AUTODOC=1 uv run sphinx-build -b html -n source build/html
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
  switch. Select the owner YAML path instead.
- Keep English pages free of manual navigation blocks.

## Configuration Changes

Task, backend, reward, and algorithm selection belongs in Hydra owner YAMLs.
When adding or changing a runnable path, update the relevant owner config under
`conf/` and verify script composition with tests under `tests/config/` or
`tests/scripts/`.

See {doc}`contracts/task_owner` and
{doc}`../user_guide/training/hydra_config`.
