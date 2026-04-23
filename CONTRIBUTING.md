# Contributing to UniLab

Languages: English | [简体中文](docs/developers/zh_CN/CONTRIBUTING.md)

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
- For user-facing workflow changes, keep `README.md`, `CONTRIBUTING.md`, and the matching localized docs under `docs/` in sync

## Read Before You Start

- Before changing training entrypoints, runners, env contracts, or backend paths, read [RL Infrastructure Development Standard](docs/developers/zh_CN/development-standard.md)
- Before changing collaboration flow or issue / milestone rules, read [Collaboration Workflow](docs/developers/zh_CN/collaboration.md)

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

## Deep References

- **Architecture & contracts**: [RL Infrastructure Development Standard](docs/developers/zh_CN/development-standard.md)
- **Collaboration & ADR governance**: [Collaboration Workflow](docs/developers/zh_CN/collaboration.md)
- **Test layout & markers**: [Development Standard §Testing](docs/developers/zh_CN/development-standard.md)
- **Configuration system**: [Development Standard §Configuration](docs/developers/zh_CN/development-standard.md)
