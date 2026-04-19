# 快速开始

语言: 简体中文

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
# macOS（MPS，默认安装 PyPI 的 torch wheel）
uv sync --extra motrix

# Linux 默认（安装 PyTorch 官方 cu128 wheel）
# 需要当前 PyTorch cu128 wheel 所支持的 NVIDIA 显卡与驱动栈
uv sync --extra motrix
```

## 中国大陆镜像

```bash
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv sync --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## First Run

### 训练一个最小任务

在 macOS / MacBook 上，只要命令会打开 MotrixSim 原生 renderer，就需要用 `uv run mxpython` 启动；不需要可视化的训练仍可使用 `uv run ... training.no_play=true`。

```bash
# Linux
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/motrix

# MacOS(其他后端为Motrix的训练同理，需要用 uv run mxpython)
uv run mxpython scripts/train_rsl_rl.py task=go2_joystick_flat/motrix
```

### 常用入口脚本

```bash
# PPO (RSL-RL)
uv run scripts/train_rsl_rl.py task=go1_joystick_flat/motrix

# APPO
uv run scripts/train_appo.py task=go1_joystick_flat/mujoco

# SAC / TD3
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco
uv run scripts/train_offpolicy.py algo=td3 task=td3/go1_joystick_flat/mujoco
```

### 验证环境

```bash
make test-all
```

## Navigation

- Next: [Simulation Backends](02-simulation-backends.md)
