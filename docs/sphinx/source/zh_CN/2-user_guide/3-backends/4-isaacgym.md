# IsaacGym 后端

IsaacGym（NVIDIA Preview 4）是 NVIDIA 已停止维护（EOL）的 GPU 物理仿真器，
只支持 Python 3.6–3.8。UniLab 主环境要求 Python >= 3.10，因此 IsaacGym 不能
装进主环境，必须通过外部独立的 Python 3.8 环境使用；仓库内一律通过环境变量
定位该环境，不写入任何机器本地路径。

当前状态：IsaacGym 尚未接入 registry / task owner / 训练与回放链路。仓库内
已可用的是物理性能 benchmark 脚本
`scripts/benchmark/physics/benchmark_physics_step_isaacgym.py`，它通过
`UNILAB_BENCHMARK_HOLOSOMA_DEPS` 等环境变量定位外部环境。后端接入在
[issue #1332](https://github.com/unilabsim/UniLab/issues/1332) 中规划；
本页只覆盖外部环境准备与 benchmark 验证。

## 前置条件

- Linux x86_64，装有 NVIDIA GPU 驱动。
- NVIDIA developer 账号：`IsaacGym_Preview_4_Package.tar.gz` 必须在
  <https://developer.nvidia.com/isaac-gym-preview-4> 登录后手动下载，
  脚本不会也无法自动拉取该 URL。
- 磁盘空间：miniconda、conda 环境与 IsaacGym 安装包合计约 5 GB。

## 自动化安装

在仓库根目录运行：

```bash
scripts/tools/setup_isaacgym_env.sh
```

脚本默认把所有内容安装到 `$HOME/.unilab/isaacgym`，可用环境变量
`UNILAB_ISAACGYM_HOME` 覆盖安装根目录；用 `--tarball <path>` 指定安装包位置
（默认读取 `$UNILAB_ISAACGYM_HOME/IsaacGym_Preview_4_Package.tar.gz`）。
脚本幂等，重跑时跳过已完成的步骤。

安装流程：专用 miniconda → python 3.8 `hsgym` conda 环境（含
`libstdcxx-ng`，用于修复 Ubuntu 24.04 的 GLIBCXX 问题）→ 解包 tarball →
`pip install -e isaacgym/python` → import 自检。

安装完成后，把脚本输出的 export 行写入 shell rc（如 `~/.bashrc`）：

```bash
export UNILAB_BENCHMARK_HOLOSOMA_DEPS="$HOME/.unilab/isaacgym"
export UNILAB_BENCHMARK_HSGYM_PYTHON="$UNILAB_BENCHMARK_HOLOSOMA_DEPS/miniconda3/envs/hsgym/bin/python3.8"
export UNILAB_BENCHMARK_HSGYM_LIB="$UNILAB_BENCHMARK_HOLOSOMA_DEPS/miniconda3/envs/hsgym/lib"
```

## 验证

用 benchmark 脚本验证环境可用。benchmark 从 URDF 加载机器人模型，
URDF 模型树（`go1_description/`、`g1_description/` 等）需自备，通过
`--models-root` 或 `UNILAB_BENCHMARK_MODELS_ROOT` 指向其根目录：

```bash
PYTHONPATH="$UNILAB_BENCHMARK_HOLOSOMA_DEPS/isaacgym/python" \
LD_LIBRARY_PATH="$UNILAB_BENCHMARK_HSGYM_LIB" \
uv run --no-project "$UNILAB_BENCHMARK_HSGYM_PYTHON" \
    scripts/benchmark/physics/benchmark_physics_step_isaacgym.py \
    --tasks g1_walk_flat --batch-sizes 256 --models-root "$UNILAB_BENCHMARK_MODELS_ROOT"
```

## 手动安装

自动脚本失效时，可按下面的等价命令序列手动安装：

```bash
export UNILAB_ISAACGYM_HOME="${UNILAB_ISAACGYM_HOME:-$HOME/.unilab/isaacgym}"
mkdir -p "$UNILAB_ISAACGYM_HOME"

# 1. 专用 miniconda
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -u -p "$UNILAB_ISAACGYM_HOME/miniconda3"
rm /tmp/miniconda.sh

# 2. python 3.8 conda 环境
"$UNILAB_ISAACGYM_HOME/miniconda3/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
"$UNILAB_ISAACGYM_HOME/miniconda3/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
"$UNILAB_ISAACGYM_HOME/miniconda3/bin/conda" install -y -n base -c conda-forge mamba
"$UNILAB_ISAACGYM_HOME/miniconda3/bin/mamba" create -y -n hsgym python=3.8 -c conda-forge --override-channels

# 3. Ubuntu 24.04 GLIBCXX 修复
"$UNILAB_ISAACGYM_HOME/miniconda3/bin/conda" install -y -n hsgym -c conda-forge libstdcxx-ng

# 4. 解包并安装 IsaacGym（tarball 需手动下载，见前置条件）
tar -xzf "$UNILAB_ISAACGYM_HOME/IsaacGym_Preview_4_Package.tar.gz" -C "$UNILAB_ISAACGYM_HOME"
"$UNILAB_ISAACGYM_HOME/miniconda3/envs/hsgym/bin/pip" install -e "$UNILAB_ISAACGYM_HOME/isaacgym/python"
```

## 故障排除

- **wget 直接拉 NVIDIA 下载 URL 失败**：该下载需要登录，必须手动下载，
  见前置条件。
- **Ubuntu 24.04 报 `GLIBCXX_3.4.32 not found`**：IsaacGym 预编译库链接的
  libstdc++ 比系统自带的旧；安装脚本已在 `hsgym` 环境中安装 conda-forge 的
  `libstdcxx-ng` 解决，运行时把 `LD_LIBRARY_PATH` 指向该 env 的 `lib/` 即可。
- **`from isaacgym import gymapi` import 失败**：确认 `LD_LIBRARY_PATH`
  指向 `$UNILAB_BENCHMARK_HSGYM_LIB`（即 hsgym env 的 `lib/`），且
  `PYTHONPATH` 包含 `isaacgym/python`。
