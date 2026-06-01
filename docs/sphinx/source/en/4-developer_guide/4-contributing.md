# Contributing To UniLab

This page summarizes the repository workflow for contributors. Contract and
architecture details live in {doc}`1-architecture/1-overview`.

## Environment

Install dependencies for your platform:

- macOS (MPS, PyPI torch wheel): `uv sync`
- Linux with NVIDIA (PyTorch cu128 wheel): `uv sync`
- Linux AMD / ROCm: `make sync-rocm`, then run commands with `uv run ...`. To
  return to the default CUDA / macOS profile, `git restore -- pyproject.toml
  uv.lock` and re-run `uv sync`.
- Linux Intel XPU: `make sync-xpu`
- Add `--extra motrix` when you need the Motrix backend, e.g. `uv sync --extra
  motrix`.

```bash
uv sync
uv sync --extra motrix
make sync-rocm
make sync-xpu
```

Use `uv run` for commands. Do not invoke `python` directly outside `uv run`.

## Development Rules

- Always use `uv run`. Run `make check` before any code-related commit.
- Do not commit backup or scratch files: no `*.bak`, `*.tmp`, `*.old`, `*.orig`,
  or editor backups ending in `~`.
- Do not add new owner logic to `src/unilab/utils/`. The modules there are a
  transitional shim; lift durable logic to its owner layer or `src/unilab/base/`
  instead.
- Module naming expresses owner responsibility: use singular nouns by default,
  plural only when the semantics are themselves a collection contract, and a
  `_factory` suffix for factory modules.
- When a change affects a user-visible workflow, keep `README.md`,
  `CONTRIBUTING.md`, and the matching pages under `docs/sphinx/source/en/` and
  `docs/sphinx/source/zh_CN/` in sync.

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

## Testing

Tests are grouped by owner area under `tests/`:

```text
tests/
├── base/         # registry, backend selection, env contract
├── config/       # Hydra / dataclass / reward injection
├── envs/         # environment configuration and instantiation
├── dr/           # domain-randomization types and managers
├── terrains/     # terrain generators and scene materialization
├── ipc/          # shared-memory and async-runner primitives
├── scripts/      # training-script configs and entrypoint tooling
├── algos/        # runner integration, RSL-RL PPO, MLX PPO
├── integration/  # cross-module reward / config integration
├── training/     # training-run helpers
└── utils/        # helpers and experiment tracking
```

Markers and skips:

- Unmarked tests are fast unit / contract / env smoke tests run by `make test`.
- `@pytest.mark.slow` marks full training/script smoke runs or cumulatively
  expensive backend matrices. CI skips them; run them locally with
  `make test-slow`. The `slow` marker is registered in `pyproject.toml`.
- The MLX PPO tests (`tests/algos/test_mlx_ppo.py`) use
  `pytest.importorskip(...)` so they skip automatically when MLX is unavailable,
  which keeps them macOS-only in practice.

## CI Workflow

Pull requests to `main` run five jobs in `.github/workflows/ci.yml`:
`ruff-lint`, `ruff-format`, `mypy`, `pyright`, and `test`. Each is a required
check, and the workflow can also be triggered manually via `workflow_dispatch`.
In-progress runs on the same branch are cancelled automatically.

| Job | What it runs |
| --- | --- |
| `ruff-lint` | `uv run --no-sync ruff check --output-format=github .` |
| `ruff-format` | `uv run --no-sync ruff format --check .` |
| `mypy` | `uv run mypy src/unilab` |
| `pyright` | `uv run pyright` |
| `test` | `uv sync --extra motrix` (CPU torch), then `uv run --no-sync pytest -m "not slow" --cov=unilab --cov-fail-under=25` |

The `test` job enforces a coverage gate (`--cov-fail-under=25`); the floor only
ratchets up as test guardrails improve. Documentation changes are validated by
`tests/scripts/test_check_docs.py` in the same suite. The separate `Docs`
workflow runs the prose-only Sphinx build.

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
