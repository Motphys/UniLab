# Development Standards

## Package Management

**Always use `uv run`, not python**.

```bash
# ✅ Correct
uv run python script.py
uv run pytest

# ❌ Incorrect
python script.py
pytest
```

## Installation

```bash
# macOS (MPS)
uv sync --extra dev

# Linux (CUDA 12.4)
uv sync --extra dev --extra cu124
```

## Development Workflow

### Quick Commands (Makefile)

```bash
make format     # Format and lint code (ruff format + ruff check --fix)
make type       # Type check with mypy
make check      # make format && make type
make test       # Run all non-slow tests (default)
make test-cov   # Run non-slow tests with coverage report
make test-slow  # Run slow integration tests (requires MuJoCo)
make test-all   # make check && make test-cov
```

### Manual Commands

```bash
# Format
uv run ruff format .
uv run ruff check --fix

# Type check
uv run mypy unilab

# Test (non-slow)
uv run pytest -m "not slow"

# Test with coverage
uv run pytest -m "not slow" --cov=unilab --cov-report=term-missing

# Slow integration tests (need MuJoCo installed)
uv run pytest -m slow -v
```

## Test Structure

```
tests/
├── conftest.py                    # shared fixtures + DummyFlatEnv stub
├── ipc/                           # IPC primitives unit tests
│   ├── test_replay_buffer.py
│   ├── test_shared_onpolicy_storage.py
│   ├── test_shared_weight_sync.py
│   └── test_shared_obs_stats.py
├── base/
│   └── test_registry.py
├── config/
│   └── test_locomotion_params.py
└── algos/
    ├── test_appo_runner.py        # @pytest.mark.slow
    └── test_offpolicy_runner.py   # @pytest.mark.slow
```

Tests marked `@pytest.mark.slow` require a real MuJoCo environment and are excluded from CI
by default. Run them locally when working on runner/learner code.

## Git Commits

Use Conventional Commits:
- `feat:` 新功能
- `fix:` 修复 bug
- `docs:` 文档
- `style:` 格式化
- `refactor:` 重构
- `test:` 测试
- `chore:` 构建/工具

## Pre-commit

```bash
pre-commit install  # Optional
```

**Always run `make check` before committing.**
