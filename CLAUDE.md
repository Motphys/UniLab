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

## PyTorch 加速后端配置

```bash
# macOS - MPS 加速（Apple Silicon）
uv sync  # 默认已支持 MPS

# Linux - CUDA 12.1
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# Linux - CUDA 11.8
uv pip install torch --index-url https://download.pytorch.org/whl/cu118
```
