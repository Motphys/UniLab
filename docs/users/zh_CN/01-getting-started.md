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

# Linux AMD / ROCm 工作站
# 要求 ROCm >= 7.1；当前安装 PyTorch 官方 ROCm 7.2 wheel。
make sync-rocm
```

`make sync-rocm` 会基于 `pyproject.rocm.toml` 从 PyTorch ROCm 源同步并安装 `torch==2.11.0+rocm7.2`、`torchvision` 和 `triton-rocm==3.6.0`。后续在该环境里运行训练命令时使用 `uv run --no-sync ...`，避免 `uv run` 自动把 Linux 默认 CUDA wheel 同步回来。PyTorch ROCm 运行时仍使用 `cuda` 设备类型，所以训练配置里的 `training.device=cuda` / 自动检测路径不需要改成 `rocm`。

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

# Linux AMD / ROCm
uv run --no-sync scripts/train_rsl_rl.py task=go2_joystick_flat/motrix

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
uv run scripts/train_offpolicy.py algo=td3 task=td3/g1_walk_flat/mujoco
```

### 验证环境

```bash
make test-all
```

## 统一 CLI

UniLab 提供了 `train`、`eval` 和 `demo` 命令行入口，无需记忆各训练脚本路径：

```bash
# 训练
uv run train --algo ppo --task go2_joystick_flat --sim mujoco

# 评估（回放最新 checkpoint）
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1

# demo（使用本地训练产出的 checkpoint 回放）
uv run demo
```

支持的算法：`ppo`、`mlx_ppo`、`appo`、`sac`、`td3`、`flashsac`
支持的模拟器：`mujoco`、`motrix`

详细用法见 [训练指南](03-training.md)。

## Docker

当你希望在不改动本地 Python 环境的前提下获得隔离的 Linux NVIDIA/CUDA 运行环境时，可以使用 Docker。Linux 训练容器需要 NVIDIA 显卡、兼容的宿主机驱动，以及 NVIDIA Container Toolkit。macOS Docker 目前不支持。

### 构建 image

```bash
docker build -t unilab:latest .
```

该 image 默认安装 UniLab 运行依赖、`mujoco-uni`、`motrix` extra，以及 dev/test 工具。

### 运行 image

```bash
# 检查统一 CLI 入口
docker run --rm unilab:latest

# Linux NVIDIA 训练 shell，挂载当前仓库
# 将 .venv 放在 named volume 中，避免容器覆盖宿主机虚拟环境
docker run --rm --gpus all -it \
  -v "$(pwd):/workspace/UniLab" \
  -v unilab-venv:/workspace/UniLab/.venv \
  -w /workspace/UniLab \
  unilab:latest bash
```

将 `/workspace/UniLab/.venv` 放到 named volume 中，可以避免把容器内创建的虚拟环境混入宿主机仓库目录，进而避免 `/workspace/UniLab/.venv` 与本地 `.venv` 的路径不一致，以及回到本地 `uv` 或 `make test-all` 工作流时出现权限问题。

验证容器内 CUDA 是否可用：

```bash
docker run --rm --gpus all unilab:latest uv run python -c "import torch; print(torch.cuda.is_available())"
```

ROCm 容器请使用 AMD 官方 `rocm/pytorch` 镜像，并按 ROCm Docker 要求挂载 `/dev/kfd`、`/dev/dri`、`--group-add=video` 和 `--ipc=host`。当前仓库根目录的 `Dockerfile` 保持 NVIDIA/CUDA 镜像，不用于 ROCm。

## Navigation

- Index: [Documentation](../../README.md)
- Next: [Simulation Backends](02-simulation-backends.md)
