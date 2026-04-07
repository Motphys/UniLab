# Getting Started

Languages: English | [简体中文](../zh_CN/01-getting-started.md) | [日本語](../ja/01-getting-started.md) | [한국어](../ko/01-getting-started.md)

This page answers only three questions:

1. How do you get UniLab running?
2. How do installation steps differ between macOS and Linux?
3. Which command should you run first to confirm the environment works?

## Install

### Use uv

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repository
git clone https://github.com/unilabsim/UniLab.git
cd UniLab

# 3. Install system dependencies
brew install cmake  # macOS
# sudo apt-get install cmake  # Ubuntu / Debian
```

### Sync Dependencies

```bash
# macOS (MPS)
uv sync --extra dev

# Linux (CUDA 11.8 / 12.4 / 12.6 / 12.8)
uv sync --extra dev --extra cu118
uv sync --extra dev --extra cu124
uv sync --extra dev --extra cu126
uv sync --extra dev --extra cu128

# Optional: Motrix backend
uv sync --extra dev --extra motrix
uv sync --extra dev --extra cu124 --extra motrix
```

## Mainland China Mirror

```bash
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv sync --extra dev --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## First Run

### Train A Minimal Task

```bash
uv run python scripts/train_rsl_rl.py task=go1_joystick
```

### Common Entrypoints

```bash
# PPO (RSL-RL)
uv run python scripts/train_rsl_rl.py task=go1_joystick

# APPO
uv run python scripts/train_appo.py task=go1_joystick

# SAC / TD3
uv run python scripts/train_offpolicy.py algo=sac task=go1_joystick
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick
```

### Validate The Environment

```bash
make check
uv run pytest -m "not slow and not veryslow"
```

## Navigation

- Previous: [Development Architecture](00-development-architecture.md)
- Next: [Simulation Backends](02-simulation-backends.md)
