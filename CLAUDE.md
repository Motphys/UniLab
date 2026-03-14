# Development Standards

## Package Management

**Always use `uv run`, not python**.

This project uses `uv` for package management and execution. All Python commands should be run through `uv run`:

```bash
# ✅ Correct
uv run python script.py
uv run pytest
uv run python -m module

# ❌ Incorrect
python script.py
pytest
python -m module
```

## Installation

Install dependencies with:
```bash
uv sync
```

For development dependencies:
```bash
uv sync --extra dev
```

For motrix support:
```bash
uv sync --extra motrix
```

## Linux CUDA 支持

macOS 默认支持 MPS。Linux 需要手动安装 CUDA 版本 PyTorch：

```bash
# CUDA 12.1
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
uv pip install torch --index-url https://download.pytorch.org/whl/cu118
```
