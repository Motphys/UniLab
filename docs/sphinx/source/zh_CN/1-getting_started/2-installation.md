# 安装

语言: 简体中文

本页仅涉及依赖配置。训练命令和回放细节请参阅快速上手与算法相关页面。

## 环境要求

- Python `>=3.10,<3.14`，来自 `pyproject.toml`。
- `uv`，用于依赖同步和命令执行。
- `cmake`，本地安装流程所需，详见
  `docs/sphinx/source/zh_CN/1-getting_started/2-installation.md`。

## 克隆与同步

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/unilabsim/UniLab.git
cd UniLab
```

```bash
brew install cmake
# Ubuntu / Debian:
# sudo apt-get install cmake
```

选择一条同步路径：

```bash
make setup
make setup-motrix
```

`make setup` 会运行 `uv sync` 并安装 shell 自动补全。`make setup-motrix`
会运行 `uv sync --extra motrix` 并安装相同的补全条目。如果 `make`
不可用，可直接运行底层的同步命令：

```bash
uv sync
uv sync --extra motrix
```

## 平台配置档

Linux CUDA 和 macOS 使用默认的 `pyproject.toml`。默认的 Linux torch
wheel 来源是在 `pyproject.toml` 中配置的 PyTorch `cu128` 索引。

ROCm 和 Intel XPU 有各自显式的 Makefile 目标：

```bash
make sync-rocm
make sync-xpu
```

`make sync-rocm` 会将 `pyproject.rocm.toml` 复制为 `pyproject.toml` 并同步
ROCm 配置档。`make sync-xpu` 会同步 Motrix 依赖但不安装默认的 torch 包，然后通过 `uv pip` 安装 XPU 版本的 torch wheel。

## 软件包镜像

如需使用本地软件包镜像，请在同步前设置 uv 索引：

```bash
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv sync --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## 冒烟检查

同步完成后，通过顶层 CLI 运行一次小型检查：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.max_iterations=1 \
  algo.num_envs=16 \
  training.no_play=true
```

对于 Motrix，请先安装相应 extra，然后通过 `--sim` 切换：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim motrix \
  algo.max_iterations=1 \
  algo.num_envs=16 \
  training.no_play=true
```

不要单独使用 `training.sim_backend` 字段来切换后端；请通过 `--sim` 选择后端。

## Navigation

- Index: [文档](0-index.md)
