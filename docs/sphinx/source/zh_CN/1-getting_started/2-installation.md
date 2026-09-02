# 安装

本页仅涉及依赖配置。训练命令和回放细节请参阅快速上手与算法相关页面。

## 环境要求

- Python `>=3.10,<3.14`，来自 `pyproject.toml`。
- `uv`，用于依赖同步和命令执行。
- `cmake`，本地安装流程所需，详见
  `docs/sphinx/source/zh_CN/1-getting_started/2-installation.md`。
- 使用 `mujoco` extra 时：需要 C++17 工具链和 Python 开发头文件，
  因为 `mujoco-uni-runtime` 仅以源码分发，`uv sync` 时会就地编译其原生扩展
  （针对 lock 钉住的 mujoco 版本）。缺少这些依赖时，`make setup` 会在编译
  `mujoco-uni-runtime` 时失败，报错 `fatal error: Python.h: No such file or directory`。
  - macOS：`xcode-select --install`
  - Ubuntu / Debian：`sudo apt-get install build-essential python3-dev`
  - Windows：MSVC Build Tools
  - 提示：使用 uv 托管的 Python（`uv python install`）自带头文件，
    只有系统 Python 才需要额外安装 `python3-dev`。

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

## Drake 批量后端（Linux-first）

Drake 后端是可选路径，不包含在默认 setup 中。它使用 PyPI 包
`drake-uni`（`import drake_uni`）以及针对独立安装的 Drake C++ 前缀编译的
原生扩展。

```bash
make setup-drake
# 等价命令：
uv sync --extra drake
```

如果要在 Linux x86_64 上完整安装，可使用可恢复的外部运行时脚本。它会下载
官方 Drake 前缀、安装 `drake` extra、构建本地原生扩展并执行 import 诊断：

```bash
bash scripts/tools/setup_drake_env.sh --download-drake
```

脚本可重复执行，下载、标记和日志默认保存在 `~/.unilab/drake`。使用已有
Drake 前缀和本地 DrakeUni checkout 时，可传入 `--drake-home`、`--deps-root`
和 `--drake-uni-source`；脚本不会使用 `sudo` 安装系统软件包。

构建扩展前，需要准备包含 `include/drake/`、`include/pybind11/` 和
`lib/libdrake.so` 的 Drake 前缀，并使用与 UniLab 相同的 Python 解释器：

```bash
/path/to/UniLab/.venv/bin/python \
  /path/to/drake_uni/scripts/build_drake_batch.py \
  --drake-home /path/to/drake/install
```

该构建路径当前以 Linux 为主，不会自动安装 Drake C++。构造 Drake batch
backend 前请保持进程未导入 `pydrake`。资产准备和短时训练探针见
{doc}`../2-user_guide/3-backends/7-drake`。

## conda 与 pip

当前推荐路径仍然是源码仓库内的 `make setup` / `make setup-motrix`（或 `uv`）工作
流。conda 可以作为外层 Python、CUDA 或系统库的隔离环境，但进入环境后仍建议继续使
用本仓库的 `make` / `uv` 命令：

```bash
conda create -n unilab python=3.13
conda activate unilab
pip install uv
git clone https://github.com/unilabsim/UniLab.git
cd UniLab
make setup-motrix
```

如果不需要 Motrix，可使用 `make setup`；ROCm / XPU 仍走下方专用的 `make` 路径。

`pip install -e .` 和 `pip install .` 构建的 wheel 已打包任务配置
（`unilab/conf/`）与训练入口，因此 `train` / `eval` / `demo` 命令可以在任意目录运
行；日志与 checkpoint 写入当前工作目录。仍有两项限制：isaacgym / isaacsim 后端和
HORA 多卡提交路径仍假设源码 checkout；任务配置中的机器人资产路径仍相对 checkout
解析，需等待 #1326 的资产外置子任务落地。

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

ROCm 说明：

- `make sync-rocm` 要求 ROCm `>= 7.1`，并按仓库的 ROCm 依赖文件安装对应的 PyTorch
  wheel。
- 它会把 `pyproject.rocm.toml` / `uv.rocm.lock` 激活为当前的 `pyproject.toml` /
  `uv.lock`，因此之后可以直接运行裸 `uv run ...`。
- 切回默认 CUDA / macOS 配置档时，运行 `git restore -- pyproject.toml uv.lock`，然
  后重新执行 `make setup-motrix`（或 `uv sync --extra motrix`）；提交任何非 ROCm
  依赖改动前先确认当前配置档。
- 训练配置里的设备字段仍沿用 `cuda` 语义，不要改成 `rocm`。

Intel XPU 说明：

- 保持使用 `uv run --no-sync ...`，避免把默认的 Linux 依赖重新同步回来。
- Ubuntu 24.04+ 上还需要系统驱动包 `intel-opencl-icd` 和 `libze-intel-gpu1`。
- off-policy 训练可按需加 `training.use_amp=true`。

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
