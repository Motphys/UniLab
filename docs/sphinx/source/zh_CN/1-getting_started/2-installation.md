# 安装

本页仅涉及依赖配置。训练命令和回放细节请参阅快速上手与算法相关页面。

## 环境要求

- Python `>=3.10,<3.14`，来自 `pyproject.toml`。
- `uv`，用于依赖同步和命令执行。
- Git 和 `curl`，用于克隆仓库及下载 runtime asset。
- 使用 `mujoco` extra 时：MuJoCo 物理后端运行在 `mjbatch` 原生 batch 引擎上，
  当前从钉住的集成 fork（`unilabsim/mjbatch`）源码构建。uv 以隔离构建
  （scikit-build-core + nanobind）编译该 git 源码，需要 C++17 工具链和
  Python 开发头文件；构建绑定 `mujoco==3.11.0`，引擎在 mujoco 版本不一致时
  拒绝导入。免编译器的安装路径取决于 fork 的最终分发渠道（预编译 wheel 还是
  git 钉版），这是 roadmap 上的待定事项（见「切换本地 MuJoCo 版本」和
  「安装错误特征对照表」）。
  - macOS：`xcode-select --install`
  - Ubuntu / Debian：`sudo apt-get install build-essential python3-dev`
  - Fedora / RHEL：`sudo dnf install gcc-c++ make python3-devel`
  - Windows：MSVC Build Tools
  - 提示：使用 uv 托管的 Python（`uv python install`）自带头文件，
    只有系统 Python 才需要额外安装 `python3-dev`。

## 克隆与同步

```bash
# Linux / macOS（仓库通过 [tool.uv] required-version 固定 uv 版本）：
curl -LsSf https://astral.sh/uv/0.12.5/install.sh | sh

# Windows PowerShell：
# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

git clone https://github.com/unilabsim/UniLab.git
cd UniLab
# 推荐的主环境解释器：
uv python install 3.13
```

UniLab 支持 Python `3.10` 到 `3.13`；主环境推荐使用 `3.13`。issue #1811
期间，外部 worker 后端不属于 scoped Manager runtime。

选择一条核心安装路径：

```bash
# 完整默认环境：MuJoCo + uni_rl，并安装 shell 自动补全。
make setup
```

`make setup` 会运行 `uv sync --extra mujoco --extra uni_rl` 并安装 shell 自动
补全。如果无法使用 `make`，可运行对应的底层命令：

```bash
uv sync --extra mujoco --extra uni_rl
uv run --no-sync unilab-complete install
```

## conda 与 pip

推荐路径是源码仓库内的 `make setup`（或 `uv`）工作流。conda 可以作为外层
Python、CUDA 或系统库的隔离环境，但进入环境后仍建议继续使
用本仓库的 `make` / `uv` 命令：

```bash
conda create -n unilab python=3.13
conda activate unilab
pip install uv
git clone https://github.com/unilabsim/UniLab.git
cd UniLab
make setup
```

ROCm / XPU 仍走下方专用的 `make` 路径。

从源码 checkout 使用 pip 时，这是备用路径。请先安装 package，再显式添加所需 runtime：

```bash
# 本地开发的 editable install：
pip install -e .

# wheel 风格的常规安装（去掉 -e）：
# pip install .

# 需要 MuJoCo 时（解析钉住的 mjbatch 集成 fork，针对 mujoco==3.11.0 构建）：
pip install "mujoco~=3.11.0" "mjbatch @ git+https://github.com/unilabsim/mjbatch.git@cf4a83d"
```

editable install 会指向源码 checkout；常规安装会把 package 和任务配置
（`unilab/conf/`）复制进环境。两种方式都支持在任意目录运行 `train` / `eval` / `demo`，
日志与 checkpoint 写入当前工作目录。`mjbatch` 引擎针对钉住的
`mujoco==3.11.0` 构建；MJWarp、Genesis、平台相关 torch index、ROCm / XPU
profile 请优先使用上面的 uv 路径。机器人 mesh 和纹理不会打进 wheel，而是在 cold path 从
`unilabsim/unilab-robots` 数据集下载。请确保安装位置可写，或从源码 checkout 使用
`uv run unilab-pull-assets` 预拉取。

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

当前 tensor-only Manager runtime 暴露五个生产后端：`mujoco`、`mjwarp`、
`genesis`、`newton`、`motrix`。各自的仿真依赖均为可选。

```bash
# 一次安装全部 scoped 后端。
uv sync --extra mujoco --extra mjwarp --extra genesis --extra newton --extra motrix
```

| 后端 | 安装路径 | 重要前置条件 |
| --- | --- | --- |
| MuJoCo | `make setup` 或 `uv sync --extra mujoco` | 从源码构建钉住的 `mjbatch` fork（绑定 `mujoco==3.11.0`）；在预编译 wheel 可用之前（roadmap 待定事项）需要 C++17 工具链和 Python 开发头文件 |
| MJWarp | `uv sync --extra mujoco --extra mjwarp` | NVIDIA CUDA；单 GPU 主机默认使用当前 CUDA 设备，多卡拓扑仍需显式配置 |
| Genesis | `uv sync --extra genesis` | 已验证路径使用 Linux x86_64、NVIDIA GPU 及固定版本 torch/Genesis |
| Newton | `uv sync --extra newton` | Linux CUDA，并钉定 newton / MuJoCo-Warp / Warp 版本 |
| Motrix | `uv sync --extra motrix` | CPU-authoritative MotrixSim 与 packed Torch HOST_BRIDGE 传输 |

`drake`、`isaacgym`、`isaacsim` 与 `superdex` 适配器在 issue #1811 期间由
`unisim-core` 暂时搁置。它们的 extras 和历史后端页面不构成生产支持声明；
在提供新的 capability、parity 与支持矩阵证据之前，UniLab train/eval CLI
会直接拒绝这些后端。

runtime 变量、渲染器要求和验证命令见 scoped 后端页面：

- {doc}`MuJoCo <../2-user_guide/3-backends/1-mujoco>`
- {doc}`MJWarp <../2-user_guide/3-backends/0-index>`
- {doc}`Genesis <../2-user_guide/3-backends/5-genesis>`
- {doc}`Newton <../2-user_guide/3-backends/7-newton>`
- {doc}`Motrix <../2-user_guide/3-backends/2-motrix>`

## 算法 Extras

PPO 训练与回放直接运行在 `rsl-rl-lib` 之上，基础 package 已包含该依赖。可选的
`uni_rl` extra 提供 `uni_rl` runtime（`unilab-rl`），仅 APPO、off-policy
算法（SAC）以及多卡数据并行 PPO（`CUDA_VISIBLE_DEVICES` 配置多项）需要：

```bash
uv sync --extra uni_rl
# 或从 PyPI 安装：
pip install unilab[uni_rl]
```

`make setup` 已包含该 extra。

## 切换本地 MuJoCo 版本

`mujoco` extra 声明 `mujoco~=3.11.0`，`mjbatch` batch 引擎针对
`mujoco==3.11.0` 构建：引擎会记录编译时的 mujoco 版本，加载时检测到不一致
会拒绝导入（见「安装错误特征对照表」中的 watchdog 行）。因此切换本地
MuJoCo 版本需要对应版本的 `mjbatch` 构建，而不是修改 UniLab 配置：

1. 在 `pyproject.toml`（并同步镜像 `pyproject.rocm.toml`）中提升 `mujoco`
   边界和 `mjbatch` 源码钉版；
2. 重新锁定（`uv lock`，ROCm lockfile 通过 `make sync-rocm`）并重新同步
   （`uv sync --extra mujoco`）。

fork 的构建在编译期钉住 `mujoco==3.11.0`，因此隔离构建总是针对匹配的 mujoco
编译。在 fork 的最终分发身份确定之前（PyPI package 还是 git 钉版——roadmap
待定事项），版本提升需要与 [fork](https://github.com/unilabsim/mjbatch)
维护者协调。

## 安装错误特征对照表

按报错文本反查原因与修复方式。

| 报错特征 | 触发场景 | 修复 |
| --- | --- | --- |
| `fatal error: Python.h: No such file or directory` | uv 从源码构建 `mjbatch` fork 时缺少 Python 开发头文件 | uv 托管的 Python（`uv python install`）自带头文件；系统 Python 需安装 `python3-dev`（Debian/Ubuntu）或 `python3-devel`（Fedora/RHEL） |
| `error: [Errno 2] No such file or directory: 'c++'`（或 `c++: No such file or directory`） | 缺少编译器，无法从钉住的 git 源码构建 `mjbatch` | 安装 C++ 工具链（见「环境要求」）后重试 `uv sync --extra mujoco` |
| `mjbatch was built against MuJoCo <version> but <other> is installed` | 版本 watchdog：引擎的编译期 mujoco 钉版与已安装的 mujoco 不一致 | 用 `uv sync --extra mujoco` 恢复 lock 钉住的组合；更换 mujoco 版本需要重建 `mjbatch`（见「切换本地 MuJoCo 版本」） |

## 平台配置档

Linux CUDA 和 macOS 使用默认的 `pyproject.toml`。默认的 Linux torch
wheel 来源是在 `pyproject.toml` 中配置的 PyTorch `cu130` 索引。

在 Apple Silicon macOS 上，使用 `make setup` 安装 MuJoCo 训练路径。MuJoCo 回放
使用官方 MuJoCo wheel 自带的 `mjpython` application。可用时 Torch 会自动选择 `mps` device；为保持配置可移植，
没有 CUDA 时，`cuda` alias 会解析到 MPS。

在 Windows 上，如果没有 GNU `make` 和 Bash，请使用上面的直接 `uv sync` 命令。
`mjbatch` 引擎不提供 Windows wheel（官方的 `mujoco.dll` 不带 import
library），因此 MuJoCo 物理后端在 Windows 上仍不可用；纯 `mujoco`
（MJCF 转换、其他后端的回放渲染）仍可安装。如果要使用 Makefile，请另行安装
GNU Make 和 Bash（例如通过 Chocolatey 或 WSL）。

ROCm 和 Intel XPU 有各自显式的 Makefile 目标：

```bash
make sync-rocm
make sync-xpu
```

`make sync-rocm` 会将 `pyproject.rocm.toml` 复制为 `pyproject.toml` 并同步
ROCm 配置档。`make sync-xpu` 会同步 scoped 后端依赖但不安装默认的 torch 包，然后通过 `uv pip` 安装 XPU 版本的 torch wheel。

ROCm 说明：

- `make sync-rocm` 要求 ROCm `>= 7.1`，并按仓库的 ROCm 依赖文件安装对应的 PyTorch
  wheel。
- 它会把 `pyproject.rocm.toml` / `uv.rocm.lock` 激活为当前的 `pyproject.toml` /
  `uv.lock`，因此之后可以直接运行裸 `uv run ...`。
- 切回默认 CUDA / macOS 配置档时，运行 `git restore -- pyproject.toml uv.lock`，然
  后重新执行 `make setup`；提交任何非 ROCm 依赖改动前先确认当前配置档。
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
uv sync --extra mujoco --extra uni_rl \
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

不要单独使用 `training.sim_backend` 字段来切换后端；请通过 `--sim` 选择后端。
