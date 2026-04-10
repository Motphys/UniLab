# Contributing to UniLab

Languages: English | [简体中文](docs/zh_CN/CONTRIBUTING.md)

## Development Environment Setup

1. Fork and clone the repository.
2. Install dependencies for your platform:
   - macOS (MPS, installs PyPI torch wheels): `uv sync`
   - Linux default (installs PyTorch cu128 wheels; requires an NVIDIA GPU/driver supported by current PyTorch cu128 wheels): `uv sync`
   - When you need Motrix, append `--extra motrix`
3. Create a branch such as `git checkout -b docs/improve-readme` or `git checkout -b fix/backend-bug`.

## Development Rules

- Always use `uv run`; do not invoke `python` outside `uv run`
- Run `make check` before code-related commits
- Keep backup files, temporary exports, and legacy compatibility copies out of the source tree; do not commit artifacts such as `*.bak`, `*.tmp`, `*.old`, `*.orig`, or editor backup files ending in `~`
- For user-facing workflow changes, keep `README.md`, `CONTRIBUTING.md`, and the matching localized docs under `docs/zh_CN/` in sync

## Read Before You Start

- Before changing training entrypoints, runners, env contracts, or backend paths, read [RL Infrastructure Development Standard](docs/zh_CN/00-development-architecture.md)
- Before changing collaboration flow or issue / milestone rules, read [docs/zh_CN/06-collaboration.md](docs/zh_CN/06-collaboration.md)

## Common Commands

```bash
make format         # ruff format + ruff check --fix
make type           # mypy src/unilab + pyright
make check          # format + type (required before code-related commits)
make test           # non-slow tests
make test-cov       # non-slow tests + coverage report
make test-slow      # slow integration tests (requires MuJoCo)
make test-veryslow  # full training smoke tests (minutes)
make test-all       # make check && make test-cov
```

## Commit Conventions

Use Conventional Commits:

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation update
- `style:` formatting only, no logic change
- `refactor:` code refactor
- `test:` test-related change
- `chore:` build or tooling

## Testing

### Test Layout

```text
tests/
├── base/         # registry, backend selection, env contracts
├── config/       # Hydra / dataclass / reward injection
├── envs/         # env configuration and instantiation
├── ipc/          # shared-memory and async-runner primitives
├── scripts/      # training script config and entrypoint tools
├── algos/        # runner integration, RSL-RL PPO, MLX PPO
├── integration/  # cross-module reward / config integration
└── utils/        # helper utilities and experiment tracking
```

### Test Markers

- Regular tests (no marker): do not require MuJoCo, run with `make test`
- `@pytest.mark.slow`: requires a MuJoCo environment, skipped in CI, run locally with `make test-slow`
- `@pytest.mark.veryslow`: full training iteration or script smoke tests, run explicitly with `make test-veryslow`
- macOS only: `test_mlx_ppo.py` uses `pytest.importorskip("mlx")` and auto-skips on non-macOS platforms

### Test Writing Principles

1. IPC or pure compute logic: place under `tests/ipc/` or the matching module test directory, without `slow`
2. Tests that depend on Runner or a real Env: place under `tests/algos/` and mark with `@pytest.mark.slow`
3. Training script smoke tests: place under `tests/scripts/` and use `pytest.importorskip` for optional dependencies
4. For multiprocessing tests, use `_SPAWN_CTX = mp.get_context("spawn")`
5. For single-process `SharedObsNormStats` tests, use `_ThreadingCtx` because `multiprocessing.Queue.empty()` is unreliable within the same process

### Running Tests

```bash
# Fast path (same coverage as CI)
uv run pytest -m "not slow and not veryslow"

# With coverage
uv run pytest -m "not slow and not veryslow" --cov=unilab --cov-report=term-missing

# Integration tests (requires MuJoCo)
uv run pytest -m "slow and not veryslow" -v

# Full training smoke tests
uv run pytest -m veryslow -v
```

## CI Workflow

PRs targeting `main` trigger five jobs automatically: `ruff-lint`, `ruff-format`, `mypy`, `pyright`, and `test`. The workflow also enables manual runs through `workflow_dispatch`, validates docs through the pytest suite on docs changes, and cancels older in-progress runs for the same PR branch.

| Job | Content | Blocking on failure |
|-----|---------|---------------------|
| `ruff-lint` | `uv sync --only-group dev` + `uv run --no-sync ruff check --output-format=github .` on `ubuntu-slim` | ✅ |
| `ruff-format` | `uv sync --only-group dev` + `uv run --no-sync ruff format --check .` on `ubuntu-slim` | ✅ |
| `mypy` | `uv sync` + `uv run mypy src/unilab` on `macos-26` | ✅ |
| `pyright` | `uv sync` + `uv run pyright` on `macos-26` | ✅ |
| `test` | `uv sync --extra motrix` + `uv run pytest -m "not slow and not veryslow" --cov=unilab --cov-report markdown-append:$GITHUB_STEP_SUMMARY --cov-fail-under=10` on `ubuntu-slim` with Python 3.11 | ✅ |

Collaboration-metadata-only changes such as `LICENSE`, issue templates, `CODEOWNERS`, and `.github/pull_request_template.md` still skip CI. Docs changes do trigger CI and are validated by `tests/scripts/test_check_docs.py`.

## Documentation Expectations

- Every documented command must match a real script, config, or Makefile target in the current repo
- When you describe backend support, prefer `Registered`, `Configured`, `Benchmarked`, or `Recommended`
- Use relative links so GitHub renders the docs correctly
- If you change user-facing docs, keep the English, zh_CN, Japanese, and Korean copies structurally aligned
- If you mention CI, log roots, or support matrices, verify them against `.github/workflows/ci.yml`, `scripts/`, and `conf/`

## GitHub Collaboration Model

- **Issue**: one executable work item per issue
- **Milestone**: a phase target such as `M1`
- **PR**: must link the driving issue and list validation commands plus impact scope
- **CODEOWNERS**: review ownership, not execution ownership

More collaboration rules live in [docs/zh_CN/06-collaboration.md](docs/zh_CN/06-collaboration.md).

## Pull Request Workflow

1. For code or config changes, run `make check` locally so lint, mypy, and pyright pass.
2. For code changes, run `make test` locally so non-slow tests pass.
3. If you touched IPC, Runner, or Config, add or update the matching tests.
4. For docs-only changes, run `uv run pytest tests/scripts/test_check_docs.py -q` at minimum.
5. If you touched repository hygiene rules, run `uv run pytest tests/scripts/test_repo_hygiene.py -q`.
6. Link the relevant GitHub issue and fill in validation plus impact scope in the PR template.
7. Open the PR against `main` and wait for green CI.
8. Wait for code review.

## Issue Reports

Use GitHub Issues to report bugs or propose features.

## Configuration System

UniLab uses Hydra + dataclass configuration:

- **Add a new task entry**: create a single owner YAML under `conf/{algo}/task/...` and use `# @package _global_`
- **Change hyperparameters**: edit the matching YAML or use CLI overrides such as `algo.num_envs=2048`
- **Add a new algorithm**: add the dataclass in `structured_configs.py` and create the matching `conf/` directory

For more detail, see the Hydra section in [Training Guide](docs/zh_CN/03-training.md) and [Development Architecture](docs/zh_CN/00-development-architecture.md).
