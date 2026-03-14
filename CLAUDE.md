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
make format     # Format and lint code
make type       # Type check with mypy
make check      # make format && make type
make test       # Run all tests
make test-fast  # Run tests excluding slow ones
make test-all   # make check && make test
```

### Manual Commands

```bash
# Format
uv run ruff format .
uv run ruff check --fix

# Type check
uv run mypy unilab

# Test
uv run pytest
```

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
