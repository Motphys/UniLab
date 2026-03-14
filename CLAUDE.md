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
uv sync --extra dev
```

## Development Workflow

### 1. 代码开发
```bash
uv run python scripts/train_rsl_rl.py --task Go1JoystickFlatTerrain
```

### 2. 提交前检查
```bash
# 格式化
uv run ruff format .

# 检查
uv run ruff check .
```

### 3. Git 提交
使用 Conventional Commits：
- `feat:` 新功能
- `fix:` 修复 bug
- `docs:` 文档
- `style:` 格式化
- `refactor:` 重构
- `test:` 测试
- `chore:` 构建/工具

### 4. 自动化
- **Pre-commit hooks**（可选）：`pre-commit install`
- **GitHub CI**：push 到 main 或 PR 时自动运行 lint + test
