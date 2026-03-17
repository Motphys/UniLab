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
make type       # Type check with mypy + pyright
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
uv run pyright

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
│   ├── test_shared_obs_stats.py
│   └── test_async_runner.py
├── base/
│   └── test_registry.py
├── config/
│   └── test_locomotion_params.py
├── scripts/
│   └── test_train_scripts.py
└── algos/
    ├── test_appo_runner.py        # @pytest.mark.slow
    ├── test_offpolicy_runner.py   # @pytest.mark.slow
    └── test_mlx_ppo.py            # macOS only (MLX backend)
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

## Configuration System

UniLab uses **Hydra + dataclass** for type-safe, composable configs:

- **Structured configs**: `src/unilab/config/structured_configs.py` (typed dataclasses)
- **YAML configs**: `conf/` directory (offpolicy/appo/ppo)
- **CLI overrides**: `algo.num_envs=2048 training.device=cuda`

### Adding New Tasks

1. Create YAML file: `conf/{algo}/task/my_task.yaml`
2. Use `# @package _global_` directive
3. Override only deltas from base config

### Adding New Algorithms

1. Add dataclass to `structured_configs.py`
2. Create `conf/{algo}/config.yaml` with defaults
3. Update training script with `@hydra.main()`
