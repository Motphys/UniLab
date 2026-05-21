# 安装与环境

语言: 简体中文

本页说明安装步骤、平台差异和环境准备。平台选择以顶层 `README.md` 为准。

## 1. 安装 uv 并克隆仓库

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/unilabsim/UniLab.git
cd UniLab
```

## 2. 安装系统依赖

```bash
brew install cmake
# Ubuntu / Debian:
# sudo apt-get install cmake
```

## 3. 按平台同步依赖

| 平台 | 同步命令 | 运行说明 |
|------|----------|----------|
| Linux CUDA / macOS | `uv sync --extra motrix` | 常规训练直接用 `uv run ...` |
| Linux AMD / ROCm | `make sync-rocm` | 后续命令用 `uv run --no-sync ...` |
| Linux Intel Arc / iGPU | `make sync-xpu` | 后续命令用 `uv run --no-sync ...` |

ROCm 路径的额外约束：

- `make sync-rocm` 要求 ROCm >= 7.1，并按仓库的 ROCm 依赖文件安装对应 PyTorch wheel。
- 之后不要直接用裸 `uv run ...`，避免被默认 Linux wheel 覆盖。
- 训练配置里的设备字段仍然沿用 `cuda` 语义，不需要改成 `rocm`。

Intel XPU 路径同样建议保持 `uv run --no-sync ...`，避免把默认 Linux 依赖重新同步回来。Ubuntu 24.04+ / 26.04 上还需要系统驱动包，例如 `intel-opencl-icd` 和 `libze-intel-gpu1`；off-policy 训练可按需加 `training.use_amp=true`。

## 中国大陆镜像

```bash
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv sync --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## 4. 安装后先确认什么

- 需要 Motrix 路径时，先确认 `uv sync --extra motrix` 已完成。
- 需要 MuJoCo 路径时，确认本机 MuJoCo runtime 可用。
- 第一次训练命令建议先跑一个最小任务，再决定是否做 `make test-all`。
- 想直接确认脚本入口也可用时，可先跑 `uv run scripts/train_rsl_rl.py task=go2_joystick_flat/motrix`。

## 下一步

安装完成后继续看 [首次运行](02-first-run.md)。

## Navigation

- Index: [Documentation](../../../README.md)
- Previous: [快速开始](../01-getting-started.md)
- Next: [首次运行](02-first-run.md)
