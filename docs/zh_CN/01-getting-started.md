# 快速开始

语言: [English](../en/01-getting-started.md) | 简体中文 | [日本語](../ja/01-getting-started.md) | [한국어](../ko/01-getting-started.md)

本页只回答三个问题:

1. 怎么把 UniLab 跑起来？
2. macOS 和 Linux 的安装步骤有什么差别？
3. 第一次应该跑什么命令来确认环境正常？

## Install

### 使用 uv

```bash
# 1. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 克隆仓库
git clone https://github.com/unilabsim/UniLab.git
cd UniLab

# 3. 安装系统依赖
brew install cmake  # macOS
# sudo apt-get install cmake  # Ubuntu / Debian
```

### 同步依赖

```bash
# macOS (MPS)
uv sync --extra dev

# Linux (CUDA 11.8 / 12.4 / 12.6 / 12.8)
uv sync --extra dev --extra cu118
uv sync --extra dev --extra cu124
uv sync --extra dev --extra cu126
uv sync --extra dev --extra cu128

# 可选: Motrix 后端
uv sync --extra dev --extra motrix
uv sync --extra dev --extra cu124 --extra motrix
```

## 中国大陆镜像

```bash
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv sync --extra dev --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## First Run

### 训练一个最小任务

```bash
uv run python scripts/train_rsl_rl.py task=go1_joystick
```

### 常用入口脚本

```bash
# PPO (RSL-RL)
uv run python scripts/train_rsl_rl.py task=go1_joystick

# APPO
uv run python scripts/train_appo.py task=go1_joystick

# SAC / TD3
uv run python scripts/train_offpolicy.py algo=sac task=go1_joystick
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick
```

### 验证环境

```bash
make check
uv run pytest -m "not slow and not veryslow"
```

## Navigation

- Previous: [Development Architecture](00-development-architecture.md)
- Next: [Simulation Backends](02-simulation-backends.md)
