# Contributing to UniLab

Languages: English | [简体中文](docs/zh_CN/CONTRIBUTING.md) | [日本語](docs/ja/CONTRIBUTING.md) | [한국어](docs/ko/CONTRIBUTING.md)

## Development Environment Setup

1. Fork and clone the repository.
2. Install dependencies for your platform:
   - macOS (MPS): `uv sync`
   - Linux (CUDA 11.8 / 12.4 / 12.6 / 12.8): `uv sync --extra cu124`
   - When you need Motrix, append `--extra motrix`
3. Create a branch such as `git checkout -b docs/improve-readme` or `git checkout -b fix/backend-bug`.

## Development Rules

- Always use `uv run`; do not invoke `python` outside `uv run`
- Run `make check` before code-related commits
- For user-facing workflow changes, keep `README.md`, `CONTRIBUTING.md`, and the matching localized docs under `docs/{en,zh_CN,ja,ko}/` in sync

## Read Before You Start

- Before changing training entrypoints, runners, env contracts, or backend paths, read [RL Infrastructure Development Standard](docs/en/00-development-architecture.md)
- Before changing collaboration flow or issue / milestone rules, read [docs/en/06-collaboration.md](docs/en/06-collaboration.md)

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

PRs targeting `main` trigger three jobs automatically. The current workflow does not rerun the same CI set again after the PR is merged into `main`.

| Job | Content | Blocking on failure |
|-----|---------|---------------------|
| `lint` | `ruff check` + `ruff format --check` | ✅ |
| `typecheck` | `mypy src/unilab` + `pyright` | ✅ |
| `test` | `pytest -m "not slow and not veryslow" --cov --cov-fail-under=10` | ✅ |

Docs-only and collaboration-metadata changes such as `*.md`, `docs/**`, issue templates, and `CODEOWNERS` do not trigger CI.

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

More collaboration rules live in [docs/en/06-collaboration.md](docs/en/06-collaboration.md).

## Pull Request Workflow

1. For code or config changes, run `make check` locally so lint, mypy, and pyright pass.
2. For code changes, run `make test` locally so non-slow tests pass.
3. If you touched IPC, Runner, or Config, add or update the matching tests.
4. For docs-only changes, at minimum re-check Markdown links, file paths, script names, and command arguments.
5. Link the relevant GitHub issue and fill in validation plus impact scope in the PR template.
6. Open the PR against `main` and wait for green CI.
7. Wait for code review.

## Issue Reports

Use GitHub Issues to report bugs or propose features.

## Configuration System

UniLab uses Hydra + dataclass configuration:

- **Add a new task**: create YAML under `conf/{algo}/task/` and use `# @package _global_`
- **Change hyperparameters**: edit the matching YAML or use CLI overrides such as `algo.num_envs=2048`
- **Add a new algorithm**: add the dataclass in `structured_configs.py` and create the matching `conf/` directory

For more detail, see the Hydra section in [Training Guide](docs/en/03-training.md) and [Development Architecture](docs/en/00-development-architecture.md).
