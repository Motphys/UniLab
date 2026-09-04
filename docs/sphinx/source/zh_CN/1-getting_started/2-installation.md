# 安装

本页仅涉及依赖配置。训练命令和回放细节请参阅快速上手与算法相关页面。

## 环境要求

- Python `>=3.10,<3.14`，来自 `pyproject.toml`。
- `uv`，用于依赖同步和命令执行。
- Git 和 `curl`，用于克隆仓库及下载 runtime asset。
- `cmake`，构建 Drake 原生 batch extension 时需要。Drake setup 脚本使用 CMake
  和 C++ 工具链。
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
# Linux / macOS：
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell：
# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

git clone https://github.com/unilabsim/UniLab.git
cd UniLab
# 推荐的主环境解释器：
uv python install 3.13
```

UniLab 支持 Python `3.10` 到 `3.13`；主环境推荐使用 `3.13`。IsaacGym 和 IsaacSim
在下方使用各自独立的 worker Python 版本。

如果计划使用 Drake，还需要在主机上安装 CMake：

```bash
# macOS：
brew install cmake

# Ubuntu / Debian：
# sudo apt-get install cmake
```

选择一条核心安装路径：

```bash
# 完整默认环境：MuJoCo + Motrix，并安装 shell 自动补全。
make setup

# 运行第一次 Motrix demo 的最快路径。
# make setup-motrix

# 仅安装 MuJoCo。
# make setup-mujoco
```

`make setup` 会运行 `uv sync --extra mujoco --extra motrix` 并安装 shell 自动补全。
`make setup-motrix` 会运行 `uv sync --extra motrix` 并安装相同的补全条目。
`make setup-mujoco` 会运行 `uv sync --extra mujoco` 并安装补全。三条路径只选择一条。
如果 `make` 不可用，可运行对应的底层命令：

```bash
# 完整默认环境：
uv sync --extra mujoco --extra motrix
uv run --no-sync unilab-complete install

# 仅 Motrix：
# uv sync --extra motrix && uv run --no-sync unilab-complete install

# 仅 MuJoCo：
# uv sync --extra mujoco && uv run --no-sync unilab-complete install
```

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

如果不需要 Motrix，可使用 `make setup-mujoco`；ROCm / XPU 仍走下方专用的 `make` 路径。

从源码 checkout 使用 pip 时，这是备用路径。请先安装 package，再显式添加所需 runtime：

```bash
# 本地开发的 editable install：
pip install -e .

# wheel 风格的常规安装（去掉 -e）：
# pip install .

# 需要 Motrix 时：
pip install motrixsim-core==0.8.2

# 需要 MuJoCo 时（分两步安装 runtime）：
pip install "mujoco>=3.5,<3.11" pybind11 wheel
pip install --no-build-isolation "mujoco-uni-runtime==0.4.0"
```

editable install 会指向源码 checkout；常规安装会把 package 和任务配置
（`unilab/conf/`）复制进环境。两种方式都支持在任意目录运行 `train` / `eval` / `demo`，
日志与 checkpoint 写入当前工作目录。MuJoCo runtime 必须在匹配的 `mujoco` package
之后安装，并使用 `--no-build-isolation`；pip 默认的隔离构建看不到该依赖。MJWarp、
Genesis、平台相关 torch index、ROCm / XPU profile 和原生扩展重建行为请优先使用上面的
uv 路径。机器人 mesh 和纹理不会打进 wheel，而是在 cold path 从
`unilabsim/unilab-robots` 数据集下载。请确保安装位置可写，或从源码 checkout 使用
`uv run unilab-pull-assets` 预拉取。isaacgym / isaacsim 后端和 HORA 多卡提交路径仍假设
源码 checkout；外部后端请使用下方专用安装页。

## 运行时 Asset

大型 asset 不打包进 wheel，而是在 cold path（所属功能首次使用时）从 Hugging Face
数据集仓库懒加载：

- [机器人网格与纹理](https://huggingface.co/datasets/unilabsim/unilab-robots)
- [动作片段](https://huggingface.co/datasets/unilabsim/unilab-motions)
- [场景](https://huggingface.co/datasets/unilabsim/unilab-scenes)
- [抓取缓存](https://huggingface.co/datasets/unilabsim/unilab-caches)
- [Demo checkpoint](https://huggingface.co/datasets/unilabsim/unilab-checkpoints)

机器人 asset 可通过 `uv run unilab-pull-assets` 预拉取。中国大陆用户在默认
Hugging Face endpoint 无法访问时，可设置 `HF_ENDPOINT=https://hf-mirror.com`。

## 后端 Extras

基础 package 会安装公开的 `unisim-core` contract；具体仿真器的依赖保持为可选项。
单后端环境请根据所用后端选择一条路径；如果要比较多个进程内后端，请在一条 `uv sync`
命令中组合对应 extras，外部 worker 脚本仍需单独执行：

例如，本地对比环境可以一次安装 MuJoCo、Motrix、MJWarp 和 Genesis：

```bash
uv sync --extra mujoco --extra motrix --extra mjwarp --extra genesis
```

| 后端 | 安装路径 | 重要前置条件 |
| --- | --- | --- |
| MuJoCo | `make setup-mujoco` 或 `uv sync --extra mujoco` | 原生扩展需要 C++17 编译器和 Python 开发头文件 |
| Motrix | `make setup-motrix` 或 `uv sync --extra motrix` | 从固定版本 Python package 安装 Motrix runtime |
| MJWarp | `uv sync --extra mujoco --extra mjwarp` | NVIDIA CUDA 和显式 CUDA process device |
| Genesis | `uv sync --extra genesis` | 已验证路径使用 Linux x86_64、NVIDIA GPU 及固定版本 torch/Genesis |
| Drake | `make setup-drake` | C++20、Eigen/fmt/spdlog，以及已有 Drake prefix 或脚本下载路径 |
| IsaacGym | `bash scripts/tools/setup_isaacgym_env.sh` | Linux x86_64、NVIDIA driver 和独立 Python 3.8 worker 环境 |
| IsaacSim | `bash scripts/tools/setup_isaacsim_env.sh` | Linux x86_64、NVIDIA CUDA、独立 Python 3.11 worker 和 Kit EULA 接受 |

Drake、IsaacGym 和 IsaacSim 的 setup 脚本会将外部 runtime 安装到仓库之外，并且可以
安全重复运行；它们不会把外部仿真器装进 UniLab 主环境。runtime 变量、渲染器要求和
验证命令见各后端页面：

- {doc}`MuJoCo <../2-user_guide/3-backends/1-mujoco>`
- {doc}`Motrix <../2-user_guide/3-backends/2-motrix>`
- {doc}`MJWarp <../2-user_guide/3-backends/0-index>`
- {doc}`Genesis <../2-user_guide/3-backends/5-genesis>`
- {doc}`Drake <../2-user_guide/3-backends/6-drake>`
- {doc}`IsaacGym <../2-user_guide/3-backends/3-isaacgym>`
- {doc}`IsaacSim <../2-user_guide/3-backends/4-isaacsim>`

## 平台配置档

Linux CUDA 和 macOS 使用默认的 `pyproject.toml`。默认的 Linux torch
wheel 来源是在 `pyproject.toml` 中配置的 PyTorch `cu128` 索引。

在 Apple Silicon macOS 上，`make setup-motrix` 是最短的交互式路径。CLI 会在需要时
通过 `mxpython` 路由 Motrix 回放；MuJoCo 回放使用官方 MuJoCo wheel 自带的
`mjpython` application。可用时 Torch 会自动选择 `mps` device；为保持配置可移植，
没有 CUDA 时，`cuda` alias 会解析到 MPS。

在 Windows 上，如果没有 GNU `make` 和 Bash，请使用上面的直接 `uv sync` 命令。
编译 MuJoCo 原生扩展需要 MSVC Build Tools 和 Python 开发头文件。如果要使用
Makefile，请另行安装 GNU Make 和 Bash（例如通过 Chocolatey 或 WSL）。

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
- 从 PyPI 安装（不克隆仓库）时，`make sync-rocm` 不适用；先从 PyTorch ROCm 索引
  安装仓库验证过的 torch build，再安装 `unilab`。发布的依赖范围是
  `torch>=2.8,<2.12`，pip 会保留已安装的 ROCm build，不会替换为 CUDA wheel：

  ```bash
  pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
  pip install unilab
  ```

Intel XPU 说明：

- 保持使用 `uv run --no-sync ...`，避免把默认的 Linux 依赖重新同步回来。
- Ubuntu 24.04+ 上还需要系统驱动包 `intel-opencl-icd` 和 `libze-intel-gpu1`。
- off-policy 训练可按需加 `training.use_amp=true`。

## 软件包镜像

如需使用本地软件包镜像，请在同步前设置 uv 索引：

```bash
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv sync --extra mujoco --extra motrix \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
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
