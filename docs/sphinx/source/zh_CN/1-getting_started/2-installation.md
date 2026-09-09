# 安装

本页仅涉及依赖配置。训练命令和回放细节请参阅快速上手与算法相关页面。

## 环境要求

- Python `>=3.10,<3.14`，来自 `pyproject.toml`。
- `uv`，用于依赖同步和命令执行。
- Git 和 `curl`，用于克隆仓库及下载 runtime asset。
- `cmake`，构建 Drake 原生 batch extension 时需要。Drake setup 脚本使用 CMake
  和 C++ 工具链。
- 使用 `mujoco` extra 时：默认安装路径使用 `mujoco-uni-runtime` 的预编译 wheel
  （绑定 `mujoco==3.11.0`），`make setup` / `uv sync --extra mujoco` 不需要编译器。
  只有显式的源码重建路径（切换 MuJoCo 版本，见「切换本地 MuJoCo 版本」）才需要
  C++17 工具链和 Python 开发头文件；缺少时会以
  `fatal error: Python.h: No such file or directory` 等错误失败（完整对照见
  「安装错误特征对照表」）。
  - macOS：`xcode-select --install`
  - Ubuntu / Debian：`sudo apt-get install build-essential python3-dev`
  - Fedora / RHEL：`sudo dnf install gcc-c++ make python3-devel`
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

# 需要 MuJoCo 时（默认安装绑定 mujoco==3.11.0 的预编译 wheel）：
pip install "mujoco~=3.11.0" "mujoco-uni-runtime==0.5.0"
```

editable install 会指向源码 checkout；常规安装会把 package 和任务配置
（`unilab/conf/`）复制进环境。两种方式都支持在任意目录运行 `train` / `eval` / `demo`，
日志与 checkpoint 写入当前工作目录。MuJoCo runtime 的预编译 wheel 直接通过 pip
安装即可，无需额外构建步骤；只有强制从 sdist 重建（绑定非默认 mujoco 版本）时
才需要 `pybind11` / `wheel` 与 `--no-build-isolation`，见「切换本地 MuJoCo 版本」。
MJWarp、Genesis、平台相关 torch index、ROCm / XPU profile 请优先使用上面的
uv 路径。机器人 mesh 和纹理不会打进 wheel，而是在 cold path 从
`unilabsim/unilab-robots` 数据集下载。请确保安装位置可写，或从源码 checkout 使用
`uv run unilab-pull-assets` 预拉取。isaacgym / isaacsim 后端仍假设
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

`mujoco`、`mjwarp` 和 `newton` 三个 extra 共享同一条 MuJoCo 3.11 /
MuJoCo-Warp 3.11 / Warp 1.16 版本线，可以组合进同一个环境。`newton`
extra 已包含 Newton 原生 ViewerGL 渲染依赖（离线 record + 交互式）：

```bash
uv sync --extra mujoco --extra mjwarp --extra newton
```

| 后端 | 安装路径 | 重要前置条件 |
| --- | --- | --- |
| MuJoCo | `make setup-mujoco` 或 `uv sync --extra mujoco` | 默认使用预编译 wheel（绑定 `mujoco==3.11.0`），无需编译器；仅切换版本（`make mujoco MJ=<version>`）时需要 C++17 工具链和 Python 开发头文件 |
| Motrix | `make setup-motrix` 或 `uv sync --extra motrix` | 从固定版本 Python package 安装 Motrix runtime |
| MJWarp | `uv sync --extra mujoco --extra mjwarp` | NVIDIA CUDA 和显式 CUDA process device |
| Genesis | `uv sync --extra genesis` | 已验证路径使用 Linux x86_64、NVIDIA GPU 及固定版本 torch/Genesis |
| Newton | `uv sync --extra newton` | NVIDIA CUDA；可与 `mujoco` / `mjwarp` extra 组合进同一环境 |
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
- {doc}`Newton <../2-user_guide/3-backends/7-newton>`
- {doc}`IsaacGym <../2-user_guide/3-backends/3-isaacgym>`
- {doc}`IsaacSim <../2-user_guide/3-backends/4-isaacsim>`

## 切换本地 MuJoCo 版本

`mujoco` extra 的默认安装路径使用 `mujoco-uni-runtime==0.5.0` 的预编译 wheel。
每个 runtime 发布版本只携带一个预编译 MuJoCo 绑定：0.5.0 的 wheel 针对
`mujoco==3.11.0` 编译，原生扩展会记录编译时的 mujoco 版本，加载时检测到不一致
会拒绝工作（见「安装错误特征对照表」中的 watchdog 行）。因此 MuJoCo 默认版本的
升级总是伴随一次新的 runtime 发布；发布协调细节见 mujoco-uni-runtime 仓库的
`docs/release-coordination.md`。

在支持窗口 `>=3.5,<3.12` 内切换 MuJoCo 版本总是走源码重建路径：

```bash
make mujoco MJ=3.10.0
```

`mujoco` extra 声明的 `mujoco~=3.11.0` 边界意味着 relock 无法选出窗口内的旧版本，
所以该目标直接操作当前环境（`uv pip`，不改动 `uv.lock`），依次执行：

1. `check-cxx-toolchain` 预检：缺少 C++ 编译器时立即失败，并打印各平台的安装命令；
2. `uv pip install "mujoco==3.10.0" pybind11 wheel setuptools`：把请求的 mujoco
   和 runtime 的构建依赖装进当前环境；
3. `uv cache clean mujoco-uni-runtime`：清除构建缓存（uv 的缓存无法感知该扩展
   依赖 mujoco 版本）；
4. `uv pip install --force-reinstall --no-deps --no-build-isolation
   --no-binary mujoco-uni-runtime "mujoco-uni-runtime==<当前版本>"`：从 sdist
   针对新 mujoco 就地重新编译原生扩展。

这个覆盖是**环境本地**的：`uv.lock` 不变。切回默认版本的路径是
`uv sync --extra mujoco --reinstall-package mujoco-uni-runtime`，恢复到 lock
钉住的默认状态（mujoco 3.11.0 + 预编译 wheel）；`--reinstall-package` 不可省略，
因为裸 `uv sync` 只恢复 mujoco 而保留本地重编的扩展，扩展会因此无法加载。
源码重建需要 C++17 工具链和 Python 开发头文件（见「环境要求」）。

## 安装错误特征对照表

按报错文本反查原因与修复方式。

| 报错特征 | 触发场景 | 修复 |
| --- | --- | --- |
| `error: building mujoco-uni-runtime from source requires a C++ toolchain, but 'c++' was not found.` | `make mujoco MJ=<version>` 的 `check-cxx-toolchain` 预检失败 | 安装 C++ 工具链后重试：Debian/Ubuntu `sudo apt-get install build-essential`；macOS `xcode-select --install`；Fedora/RHEL `sudo dnf install gcc-c++ make` |
| `error: [Errno 2] No such file or directory: 'c++'`（或 `c++: No such file or directory`） | 从 sdist 编译 `mujoco-uni-runtime` 时缺少编译器；只会出现在源码重建路径，默认 wheel 路径不会编译 | 同上安装工具链；或者不切换版本，直接走默认 wheel 路径 `uv sync --extra mujoco` |
| `fatal error: Python.h: No such file or directory` | 源码重建时缺少 Python 开发头文件 | uv 托管的 Python（`uv python install`）自带头文件；系统 Python 需安装 `python3-dev`（Debian/Ubuntu）或 `python3-devel`（Fedora/RHEL） |
| `MuJoCoUni native batch extension was built against mujoco '3.11.0', but loaded mujoco is '...'` | 版本 watchdog：扩展记录的编译期 mujoco 版本与当前加载的 mujoco 不一致 | 安装扩展绑定的 mujoco 版本（预编译 wheel 绑定 `3.11.0`：`uv sync --extra mujoco --reinstall-package mujoco-uni-runtime`）；或针对当前 mujoco 从源码重建：`make mujoco MJ=<version>` |
| `mujoco_uni 0.5.0 supports official mujoco>=3.5,<3.12; found mujoco '...'` | 环境中的 mujoco 超出 runtime 支持窗口 | 换装窗口 `>=3.5,<3.12` 内的 mujoco 版本（`make mujoco MJ=<version>`） |
| `MuJoCoUni native batch extension has not been built` | 原生扩展导入失败（`mujoco_uni.batch_available()` 返回 `False`）；常见诱因是版本切换后裸跑 `uv sync`：mujoco 被恢复而本地重编的扩展仍链接旧版 `libmujoco.so` | 运行 `uv run python -c "import mujoco_uni; print(mujoco_uni.batch_import_error())"` 查看底层原因；版本切换后用 `uv sync --extra mujoco --reinstall-package mujoco-uni-runtime` 恢复预编译 wheel |

## 平台配置档

Linux CUDA 和 macOS 使用默认的 `pyproject.toml`。默认的 Linux torch
wheel 来源是在 `pyproject.toml` 中配置的 PyTorch `cu128` 索引。

在 Apple Silicon macOS 上，`make setup-motrix` 是最短的交互式路径。CLI 会在需要时
通过 `mxpython` 路由 Motrix 回放；MuJoCo 回放使用官方 MuJoCo wheel 自带的
`mjpython` application。可用时 Torch 会自动选择 `mps` device；为保持配置可移植，
没有 CUDA 时，`cuda` alias 会解析到 MPS。

在 Windows 上，如果没有 GNU `make` 和 Bash，请使用上面的直接 `uv sync` 命令。
默认安装使用预编译 wheel；只有源码重建 MuJoCo 原生扩展（切换版本）时才需要
MSVC Build Tools 和 Python 开发头文件。如果要使用
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
