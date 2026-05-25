# 安装与环境

语言: 简体中文

本页说明安装步骤、平台差异和环境准备。平台选择以顶层 `README.md` 为准。

## 1. 安装 uv 并克隆仓库

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/unilabsim/UniLab.git
cd UniLab
```

## conda / pip 用户说明

当前推荐路径仍然是源码仓库内的 `make setup` / `uv` 工作流：优先用 `make setup` 或 `make setup-motrix` 同步依赖并安装 Tab 补全，后续用 `uv run train` / `uv run eval` / `uv run demo` 运行命令。conda 可以作为外层 Python、CUDA 或系统库隔离环境，但进入环境后仍建议继续使用本仓库的 `make` / `uv` 命令。

```bash
conda create -n unilab python=3.13
conda activate unilab
pip install uv
git clone https://github.com/unilabsim/UniLab.git
cd UniLab
make setup-motrix
```

如果不需要 Motrix，可使用 `make setup`；如果只想手动同步依赖、不写 shell rc，可继续使用底层命令 `uv sync` 或 `uv sync --extra motrix`。ROCm / XPU 仍使用下方专用 `make` 路径。中国大陆镜像用户可同时配置 conda、pip 和 uv 镜像，但最终仍以 `uv sync` 生成的仓库环境为准。

`pip install -e .` 和 `pip install .` 当前只适合源码 checkout 内的开发验证，不代表已经支持在任意目录通过 wheel / sdist 直接运行训练。训练入口仍依赖仓库中的 `conf/` 和 `scripts/`；pip-only 安装、构建包后仓库外运行，以及正式发布 wheel 的验证路径由 #360 跟踪。

可选后端依赖由对应同步路径安装：Motrix 使用 `uv sync --extra motrix`，MuJoCo 仍需要本机 runtime 可用，ROCm / XPU 依赖按下方平台命令处理。后端选择仍通过 `--sim` 路由到 task owner 配置，不要单独 override `training.sim_backend` 来切换后端。

## 2. 安装系统依赖

```bash
brew install cmake
# Ubuntu / Debian:
# sudo apt-get install cmake
```

## 3. 按平台同步依赖

| 平台 | 同步命令 | 运行说明 |
|------|----------|----------|
| Linux CUDA / macOS | `make setup-motrix` | 同步 Motrix 依赖并安装 Tab 补全；底层命令是 `uv sync --extra motrix` |
| Linux AMD / ROCm | `make sync-rocm` | 后续命令用 `uv run ...` |
| Linux Intel Arc / iGPU | `make sync-xpu` | 后续命令用 `uv run --no-sync ...` |

ROCm 路径的额外约束：

- `make sync-rocm` 要求 ROCm >= 7.1，并按仓库的 ROCm 依赖文件安装对应 PyTorch wheel。
- `make sync-rocm` 会把 `pyproject.rocm.toml` / `uv.rocm.lock` 激活为当前 `pyproject.toml` / `uv.lock`，所以后续可以直接用裸 `uv run ...`。
- 切回默认 CUDA / macOS profile 时运行 `git restore -- pyproject.toml uv.lock`，然后重新执行 `make setup-motrix` 或 `uv sync --extra motrix`；提交非 ROCm 依赖改动前应确认当前 profile。
- 训练配置里的设备字段仍然沿用 `cuda` 语义，不需要改成 `rocm`。

Intel XPU 路径同样建议保持 `uv run --no-sync ...`，避免把默认 Linux 依赖重新同步回来。Ubuntu 24.04+ / 26.04 上还需要系统驱动包，例如 `intel-opencl-icd` 和 `libze-intel-gpu1`；off-policy 训练可按需加 `training.use_amp=true`。

## 中国大陆镜像

```bash
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv sync --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## 4. 安装后先确认什么

- 需要 Motrix 路径时，先确认 `make setup-motrix` 或 `uv sync --extra motrix` 已完成。
- 需要 MuJoCo 路径时，确认本机 MuJoCo runtime 可用。
- 第一次训练命令建议先跑一个最小任务，再决定是否做 `make test-all`。
- 想直接确认脚本入口也可用时，可先跑 `uv run scripts/train_rsl_rl.py task=go2_joystick_flat/motrix`。

## 下一步

安装完成后继续看 [首次运行](02-first-run.md)。

## Navigation

- Index: [Documentation](../../../README.md)
- Previous: [快速开始](../01-getting-started.md)
- Next: [首次运行](02-first-run.md)
