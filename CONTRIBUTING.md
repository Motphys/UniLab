# Contributing to UniLab

Languages: English | [简体中文](docs/sphinx/source/zh_CN/4-developer_guide/4-contributing.md)

## Development Environment Setup

1. Fork and clone the repository.
2. Install the pinned WheelNext-enabled uv build:
   `scripts/tools/install_uv_wheelnext.sh .uv-wheelnext`
3. Put that build first on `PATH` and install dependencies:
   `export PATH="$PWD/.uv-wheelnext:$PATH" && make setup`
   The variant-aware lock selects the CPU, CUDA, ROCm, or XPU Torch wheel.
4. Physics adapters are supplied by the production PyPI package
   `unisim-core==2.1.1` (import namespace `unisim`).
5. Create a branch such as `git checkout -b docs/improve-readme` or `git checkout -b fix/backend-bug`.

## Development Rules

- Always use `uv run`; do not invoke `python` outside `uv run`
- Run `make check` before code-related commits
- Keep backup files, temporary exports, and legacy compatibility copies out of the source tree; do not commit artifacts such as `*.bak`, `*.tmp`, `*.old`, `*.orig`, or editor backup files ending in `~`
- For user-facing workflow changes, keep `README.md`, `CONTRIBUTING.md`, and the matching localized docs under `docs/` in sync
- Do not add new owner logic under `src/unilab/utils/`; the current `src/unilab/utils/*.py` files are transition shims only and are scheduled for removal in `0.2.0`
- Name new owner modules and packages after their responsibility: prefer singular nouns, use plural only for collection-valued contracts, and reserve suffixes such as `_factory` for factory modules
- Use English for code comments, public API docstrings, internal implementation notes, TODO/FIXME entries, and config comments. Keep Chinese prose in Chinese documentation under `docs/sphinx/source/zh_CN/`; do not duplicate localized explanations inside source comments.

## Read Before You Start

- Before changing training entrypoints, runners, env contracts, or backend paths, read [RL Infrastructure Development Standard](docs/sphinx/source/zh_CN/4-developer_guide/0-index.md)
- Before changing collaboration flow or issue / milestone rules, read [Collaboration Workflow](docs/sphinx/source/en/4-developer_guide/5-contributing_workflow.md)

## Common Commands

```bash
make format         # ruff format + ruff check --fix
uv run unilab doctor # Check the installed Torch build for an AMD/ROCm mismatch
make type           # mypy src/unilab + pyright
make check          # format + type (required before code-related commits)
make test           # non-slow tests
make test-cov       # non-slow tests + coverage report
make test-slow      # slow integration and training smoke tests
make test-all       # make check + make test-cov + benchmark entrypoint smoke
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

1. Choose and record the intended PR base before development.
2. Run focused checks for the changed contract, then `make test-all` on the
   final local head before creating or updating the PR.
3. Link the driving issue and record the exact validation commands, results, and
   backend/platform impact in the PR template.
4. Complete review. A PR to `main` also waits for applicable current-head remote
   CI; another base uses local validation and review, with remote CI run by the
   later PR that reaches `main`.

Scope, authorization, roadmap branch, ADR, and release rules live in the
[Collaboration Workflow](docs/sphinx/source/en/4-developer_guide/5-contributing_workflow.md).

## Issue Reports

Use GitHub Issues to report bugs or propose features.

## Deep References

- **Architecture & contracts**: [RL Infrastructure Development Standard](docs/sphinx/source/zh_CN/4-developer_guide/0-index.md)
- **Collaboration & ADR governance**: [Collaboration Workflow](docs/sphinx/source/en/4-developer_guide/5-contributing_workflow.md)
- **Test layout & markers**: [Development Standard §Testing](docs/sphinx/source/zh_CN/4-developer_guide/0-index.md)
- **Configuration system**: [Development Standard §Configuration](docs/sphinx/source/zh_CN/4-developer_guide/0-index.md)
