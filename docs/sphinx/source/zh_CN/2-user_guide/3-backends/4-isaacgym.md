# IsaacGym 后端

IsaacGym（NVIDIA Preview 4）是 NVIDIA 已停止维护（EOL）的 GPU 物理仿真器，
只支持 Python 3.6–3.8。UniLab 主环境要求 Python >= 3.10，因此 IsaacGym 不能
装进主环境，必须通过外部独立的 Python 3.8 环境使用；仓库内一律通过环境变量
定位该环境，不写入任何机器本地路径。

当前状态：`IsaacGymBackend`（subprocess 后端，物理跑在外部 Python 3.8
worker 中）已实现并注册到 registry；`g1_walk_flat` 已提供 isaacgym owner
配置（`conf/{ppo,appo,sac,td3,flashsac}/task/g1_walk_flat/isaacgym.yaml`），
跨后端契约审计（`scripts/audit_sim2sim_contracts.py`）覆盖 mujoco↔isaacgym。
回放渲染尚未支持（owner 配置已将 `play_render_mode` 置为 `none`）；顶层 CLI
的 `--sim` 暂不含 isaacgym（与 drake 相同，经 owner YAML 选择后端）。真机
端到端验证（MJCF 导入保真度等）依赖下文所述的外部环境，尚未在仓库 CI 中
覆盖。仓库内另有物理性能 benchmark 脚本
`scripts/benchmark/physics/benchmark_physics_step_isaacgym.py`，它通过
`UNILAB_BENCHMARK_HOLOSOMA_DEPS` 等环境变量定位外部环境。
本页覆盖外部环境准备、benchmark 验证与当前接入状态。

## 前置条件

- Linux x86_64，装有 NVIDIA GPU 驱动。
- 网络可访问 NVIDIA 下载站点：脚本会自动从
  <https://developer.nvidia.com/isaac-gym-preview-4> 下载
  `IsaacGym_Preview_4_Package.tar.gz`（无需登录）；离线机器可先自行下载，
  再用 `--tarball <path>` 指定。
- 磁盘空间：miniconda、conda 环境与 IsaacGym 安装包合计约 5 GB。

## 自动化安装

在仓库根目录运行：

```bash
scripts/tools/setup_isaacgym_env.sh
```

脚本默认把所有内容安装到 `$HOME/.unilab/isaacgym`，可用环境变量
`UNILAB_ISAACGYM_HOME` 覆盖安装根目录；安装包默认自动下载到
`$UNILAB_ISAACGYM_HOME/IsaacGym_Preview_4_Package.tar.gz`，离线机器可用
`--tarball <path>` 指定已下载的安装包。
脚本幂等，重跑时跳过已完成的步骤。

安装流程：专用 miniconda → python 3.8 `hsgym` conda 环境（含
`libstdcxx-ng`，用于修复 Ubuntu 24.04 的 GLIBCXX 问题）→ 下载并解包
tarball → `pip install -e isaacgym/python` → import 自检。

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

# 4. 下载（或复用已下载的 tarball）并安装 IsaacGym
curl -fL --retry 3 "https://developer.nvidia.com/isaac-gym-preview-4" \
  -o "$UNILAB_ISAACGYM_HOME/IsaacGym_Preview_4_Package.tar.gz"
tar -xzf "$UNILAB_ISAACGYM_HOME/IsaacGym_Preview_4_Package.tar.gz" -C "$UNILAB_ISAACGYM_HOME"
"$UNILAB_ISAACGYM_HOME/miniconda3/envs/hsgym/bin/pip" install -e "$UNILAB_ISAACGYM_HOME/isaacgym/python"
```

## 故障排除

- **tarball 下载或校验失败**：脚本会对下载结果做 gzip 校验；失败时删除
  `$UNILAB_ISAACGYM_HOME/IsaacGym_Preview_4_Package.tar.gz` 后重跑，或用
  `--tarball <path>` 指定手动下载的安装包。
- **首次 INIT 握手超时（worker 无响应）**：`gymtorch` 首次 import 会 JIT 编译
  C++ 扩展（数分钟，缓存于 `~/.cache/torch_extensions/py38_cu121/gymtorch/`）。
  安装脚本的自检已预热该编译；若编译进程曾被强杀，可能残留 `lock` 文件导致
  永久等待——删除该目录下的 `lock` 后重试。worker 环境需要 env 的 `bin/`
  在 `PATH` 上（提供 ninja），`IsaacGymBackend` 会自动注入。
- **Ubuntu 24.04 报 `GLIBCXX_3.4.32 not found`**：IsaacGym 预编译库链接的
  libstdc++ 比系统自带的旧；安装脚本已在 `hsgym` 环境中安装 conda-forge 的
  `libstdcxx-ng` 解决，运行时把 `LD_LIBRARY_PATH` 指向该 env 的 `lib/` 即可。
- **`from isaacgym import gymapi` import 失败**：确认 `LD_LIBRARY_PATH`
  指向 `$UNILAB_BENCHMARK_HSGYM_LIB`（即 hsgym env 的 `lib/`），且
  `PYTHONPATH` 包含 `isaacgym/python`。
