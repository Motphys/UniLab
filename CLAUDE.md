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

## Cross-Platform Support

### macOS (MPS)
- PyTorch with MPS backend (default from PyPI)
- MLX framework (Apple Silicon only)

### Linux (CUDA)
For CUDA support, install PyTorch with CUDA after `uv sync`:
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
