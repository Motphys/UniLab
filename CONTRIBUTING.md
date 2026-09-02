# Contributing to UniLab

Languages: English | [简体中文](docs/sphinx/source/zh_CN/4-developer_guide/4-contributing.md)

## Development Environment Setup

1. Fork and clone the repository.
2. Install dependencies for your platform:
   - macOS (MPS, installs PyPI torch wheels): `uv sync`
   - Linux default (installs PyTorch cu128 wheels; requires an NVIDIA GPU/driver supported by current PyTorch cu128 wheels): `uv sync`
   - Linux AMD / ROCm workstation: `make sync-rocm`, then run commands with `uv run --no-sync ...`
   - When you need Motrix, append `--extra motrix`
   - Physics adapters are supplied by the production PyPI package
     `unisim-core>=0.1.14` (import namespace `unisim`).
3. Create a branch such as `git checkout -b docs/improve-readme` or `git checkout -b fix/backend-bug`.

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
make sync-rocm      # Linux AMD / ROCm >= 7.1: sync deps and install torch==2.11.0+rocm7.2
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

1. Choose and record the PR base before development. A roadmap child branches from its current integration branch; other work branches from its intended target branch.
2. Run the tests nearest the changed contract. IPC, Runner, Config, docs, and repository-hygiene changes include their matching focused tests.
3. Run `make test-all` on the final local head before creating or updating every PR, then record the command and result in the PR template.
4. Link the driving issue and describe validation plus impact scope in the PR template.
5. Open the PR against its intended base and complete code review.
6. A PR whose base is `main` completes the applicable remote CI for its current head. A PR to another base uses the recorded local `make test-all` result as its complete test gate; remote execution occurs when an integrated result later reaches a `main`-base PR.

## Roadmap Integration Workflow

Once a roadmap issue is approved for development, record its declared base
branch and create `dev/issue-<roadmap-number>-<slug>` from that base's latest
head. The declared base may be `main` or another roadmap's integration branch.
Create each child-issue branch from the latest integration branch using the
repository's conventional type prefix, such as
`feat/issue-<number>-<slug>` or `fix/issue-<number>-<slug>`, and set the child
PR base to the integration branch. After the approved child issues are
integrated, open the roadmap's final PR back to its declared base. Remote CI is
required when that actual PR base is `main`.

The detailed scope, authorization, and branch-update rules live in
[Collaboration Workflow](docs/sphinx/source/en/4-developer_guide/5-contributing_workflow.md).

## Issue Reports

Use GitHub Issues to report bugs or propose features.

## Deep References

- **Architecture & contracts**: [RL Infrastructure Development Standard](docs/sphinx/source/zh_CN/4-developer_guide/0-index.md)
- **Collaboration & ADR governance**: [Collaboration Workflow](docs/sphinx/source/en/4-developer_guide/5-contributing_workflow.md)
- **Test layout & markers**: [Development Standard §Testing](docs/sphinx/source/zh_CN/4-developer_guide/0-index.md)
- **Configuration system**: [Development Standard §Configuration](docs/sphinx/source/zh_CN/4-developer_guide/0-index.md)
