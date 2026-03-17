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
make test-veryslow  # Run full training iteration tests (minutes per test)
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
uv run pytest -m "slow and not veryslow" -v

# Very slow: full training iteration tests (minutes per test)
uv run pytest -m veryslow -v
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
│   ├── test_registry.py
│   └── test_np_env.py             # NpEnvState + NpEnv dict-obs contract
├── config/
│   ├── test_locomotion_params.py
│   └── test_manipulation_params.py
├── envs/
│   └── test_env_configs.py        # obs_groups_spec dims + env instantiation
├── utils/
│   └── test_obs_utils.py          # flatten_obs_dict
├── scripts/
│   └── test_train_scripts.py
└── algos/
    ├── test_appo_runner.py        # @pytest.mark.slow
    ├── test_offpolicy_runner.py   # @pytest.mark.slow
    └── test_mlx_ppo.py            # macOS only (MLX backend)
```

Tests marked `@pytest.mark.slow` require a real MuJoCo environment and are excluded from CI
by default. Run them locally when working on runner/learner code.

Tests marked `@pytest.mark.veryslow` run full training iterations (minutes per test). They are
excluded by default even when running `make test-slow`. Run `make test-veryslow` explicitly.

## Testing

**New features must ship with tests.** When developing a new feature or refactoring, design and write comprehensive unit tests alongside the feature code — not as an afterthought. Tests should cover:

- Normal behaviour and edge cases
- Error paths and invalid inputs
- Contract verification (e.g. interface shapes, types, key presence)
- Integration with neighbouring modules when relevant (`@pytest.mark.slow` for MuJoCo-dependent tests)

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
