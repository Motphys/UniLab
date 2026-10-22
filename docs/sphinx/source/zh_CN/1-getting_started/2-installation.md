# 安装

本页仅涉及依赖配置。训练命令和回放细节请参阅快速上手与算法相关页面。

## 环境要求

- Python `>=3.10,<3.14`，来自 `pyproject.toml`。
- `uv-wheelnext 0.9.14`，即本仓库钉住、支持 WheelNext 的 uv 构建，用于依赖
  同步和命令执行。当前稳定版 uv 默认还不能正确选择这批 PyTorch wheel-variant
  metadata。
- Git、`curl`、`tar` 与 SHA-256 工具，用于克隆仓库、安装钉版 uv 并下载
  runtime asset。
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
git clone https://github.com/unilabsim/UniLab.git
cd UniLab

# Linux / macOS；Windows 请使用 WSL 或下载匹配的 WheelNext archive。
scripts/tools/install_uv_wheelnext.sh .uv-wheelnext
export PATH="$PWD/.uv-wheelnext:$PATH"

# 推荐的主环境解释器：
uv python install 3.13
```

UniLab 支持 Python `3.10` 到 `3.13`；主环境推荐使用 `3.13`。issue #1811
期间，外部 worker 后端不属于 scoped Manager runtime。

选择一条核心安装路径：

```bash
# 完整默认环境：MuJoCo 后端，并安装 shell 自动补全。
make setup
```

`make setup` 会同步默认环境（MuJoCo 后端与内置 RL 栈）并安装 shell 自动
补全。如果无法使用 `make`，可运行对应的底层命令：

```bash
uv sync --extra mujoco
uv run --no-sync unilab-complete install
```

## conda 与 pip

推荐路径是源码仓库内的 `make setup`（或 `uv`）工作流。conda 可以作为外层
Python、CUDA 或系统库的隔离环境，但进入环境后仍建议继续使
用本仓库的 `make` / `uv` 命令：

```bash
conda create -n unilab python=3.13
conda activate unilab
git clone https://github.com/unilabsim/UniLab.git
cd UniLab
scripts/tools/install_uv_wheelnext.sh .uv-wheelnext
export PATH="$PWD/.uv-wheelnext:$PATH"
make setup
```

在该 Conda 环境中，也必须让同一个 WheelNext `uv` 保持在 `PATH` 最前。

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
`mujoco==3.11.0` 构建；MJWarp 与 Genesis 请优先使用上面的 uv 路径。纯 pip
不会选择 WheelNext 加速器 variants，CUDA、ROCm 与 XPU 环境请使用钉版 uv。
机器人 mesh 和纹理不会打进 wheel，而是在 cold path 从
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

当前 tensor-only Manager runtime 暴露五个生产后端（`mujoco`、`mjwarp`、
`genesis`、`newton`、`motrix`），另有一个 scoped Drake owner。各自的仿真依赖
均为可选。

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
| Drake | `make setup-drake` | 本地 Drake C++ 前缀与 DrakeUni native batch extension；当前 scoped 到 PPO `go2_joystick_flat` |

`isaacgym`、`isaacsim` 与 `superdex` 适配器在 issue #1811 期间由
`unisim-core` 暂时搁置。它们的 extras 和历史后端页面不构成生产支持声明；
在提供新的 capability、parity 与支持矩阵证据之前，UniLab train/eval CLI
会直接拒绝这些后端。

runtime 变量、渲染器要求和验证命令见 scoped 后端页面：

- {doc}`MuJoCo <../2-user_guide/3-backends/1-mujoco>`
- {doc}`MJWarp <../2-user_guide/3-backends/0-index>`
- {doc}`Genesis <../2-user_guide/3-backends/5-genesis>`
- {doc}`Newton <../2-user_guide/3-backends/7-newton>`
- {doc}`Motrix <../2-user_guide/3-backends/2-motrix>`
- {doc}`Drake <../2-user_guide/3-backends/6-drake>`

## 算法 Extras

不再需要任何算法 extra。完整的 RL 栈已内置在 `unilab` distribution 的
`src/unilab/rl/` 中：PPO 直接运行在 `rsl-rl-lib` 之上，APPO、off-policy
（SAC / FlashSAC）与多卡数据并行 runtime 均随基础安装提供。原
`unilab-rl` 包已并入本仓库（issue #2372，决策见
{doc}`/adr/ADR-0016-rl-reintegration-into-unilab-rl`）。

旧 `uni_rl` extra 已移除。不要再将其传给 `uv sync` 或 `pip install`；
RL 栈已包含在基础发行版中。

如果你的代码直接 `import uni_rl`，请参见
{doc}`../3-deployment/3-framework_migration/7-from_unilab_rl` 中的
`uni_rl.*` → `unilab.rl.*` 迁移映射。

## 切换本地 MuJoCo 版本

`mujoco` extra 声明 `mujoco~=3.11.0`，`mjbatch` batch 引擎针对
`mujoco==3.11.0` 构建：引擎会记录编译时的 mujoco 版本，加载时检测到不一致
会拒绝导入（见「安装错误特征对照表」中的 watchdog 行）。因此切换本地
MuJoCo 版本需要对应版本的 `mjbatch` 构建，而不是修改 UniLab 配置：

1. 在 `pyproject.toml` 中提升 `mujoco` 边界和 `mjbatch` 源码钉版；
2. 重新锁定（`uv lock`）并重新同步（`uv sync --extra mujoco`）。

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

## 加速器 Variants

UniLab 使用单一的 wheel-variant-aware `uv.lock`。同步时，WheelNext provider
插件会探测宿主，并从 lock 记录的 variants 中选择 CPU、CUDA、ROCm 或 XPU Torch
wheel。没有需要激活的加速器配置档，也不需要交换任何 tracked
`pyproject.toml` / `uv.lock` 文件。见
{doc}`/adr/ADR-0020-wheel-variant-dependency-path`。

在 Apple Silicon macOS 上，使用 `make setup` 安装 MuJoCo 训练路径。MuJoCo 回放
使用官方 MuJoCo wheel 自带的 `mjpython` application。可用时 Torch 会自动选择
`mps` device；为了保持配置可移植，没有 CUDA 时，`cuda` alias 会解析到 MPS。

在 Windows 上，请使用 WSL 或下载匹配的 WheelNext uv archive 完成依赖同步。
`mjbatch` 引擎不提供 Windows wheel（官方的 `mujoco.dll` 不带 import
library），因此 MuJoCo 物理后端在 Windows 上仍不可用；纯 `mujoco`
（MJCF 转换、其他后端的回放渲染）仍可安装。

加速器说明：

- 必须使用钉版的 `uv-wheelnext 0.9.14`。用 `uv --version` 确认；当前稳定版
  uv 尚不能正确消费这批 variant metadata。
- 宿主仍需要兼容的加速器内核驱动。ROCm Torch wheel 自带用户态，宿主 ROCm
  用户态版本不必与 wheel 完全一致。`rocm7.0` Torch variant 已在 8 卡 MI300X、
  兼容 `6.19` AMD 内核驱动的主机上验证。
- Intel XPU 主机需要 Level Zero 驱动栈。Ubuntu 24.04+ 需安装
  `intel-opencl-icd` 与 `libze-intel-gpu1`；off-policy 训练可按需设置
  `training.use_amp=true`。
- 训练设备字段在 ROCm/HIP 上仍沿用 Torch 的 `cuda` 拼写；不要改成 `rocm`。
- 同步后或 GPU 行为异常时，运行 `uv run unilab doctor`。它会检测 AMD
  内核/ROCm 管理接口，报告当前 Torch 构建，并在 AMD 主机误装 CUDA 或 CPU
  Torch wheel 时 fail closed。
- 安装 ROCm PyTorch 且 GPU 可用时，声明并接受 `cuda` buffer 的
  `HOST_BRIDGE` 后端会在训练 learner 解析到 CUDA 时让 Manager/TorchEnv
  tensor 使用当前 GPU：默认 `training.collector_tensor_device=auto` 跟随
  learner 设备，显式 `training.collector_tensor_device=cuda` 也可对其他
  learner 设备强制同样的路由。直接构造环境仍为 CPU。host-bridge 物理仿真仍在
  CPU 上，通过 packed 传输连接 GPU observation、action、reward 与 reset。
  CUDA-only 物理后端仍不支持 ROCm。见
  {doc}`/adr/ADR-0012-sole-tensor-manager-and-scoped-backends`。
- 下游应用使用同一个 WheelNext uv 构建。在新项目环境中，
  `uv pip install unilab` 会选择 accelerator-aware Torch variant；纯 pip 只是
  CPU/plain-wheel fallback。

## 软件包镜像与 Variants

当前 wheel-variant 路径必须解析 WheelNext variant metadata index。同步这个
lock 时不要用普通 PyPI 镜像替换默认 index；待稳定 uv / PyPI variant metadata
集成后再重新评估镜像支持。

## 冒烟检查

同步完成后，通过顶层 CLI 运行一次小型检查：

```bash
uv run unilab doctor
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.max_iterations=1 \
  algo.num_envs=16 \
  training.no_play=true
```

不要单独使用 `training.sim_backend` 字段来切换后端；请通过 `--sim` 选择后端。
