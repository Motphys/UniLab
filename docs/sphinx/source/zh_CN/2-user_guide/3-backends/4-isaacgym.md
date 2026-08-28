# IsaacGym 后端

IsaacGym（NVIDIA Preview 4）是 NVIDIA 已停止维护（EOL）的 GPU 物理仿真器，
只支持 Python 3.6–3.8。UniLab 主环境要求 Python >= 3.10，因此 IsaacGym 不能
装进主环境，必须通过外部独立的 Python 3.8 环境使用；仓库内一律通过环境变量
定位该环境，不写入任何机器本地路径。

当前状态：`IsaacGymBackend`（subprocess 后端，物理跑在外部 Python 3.8
worker 中）已实现并注册到 registry；`g1_walk_flat` 提供 isaacgym owner
配置（`conf/{ppo,sac}/task/g1_walk_flat/isaacgym.yaml`），跨后端契约审计
（`scripts/audit_sim2sim_contracts.py`）覆盖 mujoco↔isaacgym。回放渲染走
IsaacGym 原生渲染（viewer + camera sensor），交互与录视频两种模式均可用
（见下文「训练与评估」）。真机端到端
验证依赖下文所述的外部环境，不在仓库 CI 中覆盖。仓库内另有物理性能
benchmark 脚本
`scripts/benchmark/physics/benchmark_physics_step_isaacgym.py`，它通过
`UNILAB_BENCHMARK_HOLOSOMA_DEPS` 等环境变量定位外部环境。
本页覆盖外部环境准备、训练与评估、benchmark 验证与故障排除。

## 模型契约

后端直接消费 task 的 MJCF scene，但对 IsaacGym 的 MJCF importer 只做有限
信任：运动学（body/dof 名称与顺序）在 INIT 时与 host 侧 XML 扫描结果
对齐校验，所有影响驱动的参数都从 XML 解析而不是读取 importer 的结果。

- **控制**：只支持 `<position kp kv forcerange>` 执行器——
  `SimBackend.step(ctrl)` 携带逐 DoF 位置目标，用 PhysX `DOF_MODE_POS`
  驱动复现（force = kp·(target − q) − kv·q̇，并按对称 forcerange 截断）。
  含 `<motor>`/`<velocity>` 等其他执行器类型、非 1 的 gear、或不对称
  forcerange 的 scene 会在扫描时 fail-closed。
- **自碰撞整体关闭**（actor collision filter）。MJCF
  `<contact><exclude>` 配对（如 G1 的 elbow↔wrist、pelvis↔hip 重叠）
  无法通过 gymapi 按 link 对复现；整体关闭自碰撞是生态通行近似，且是
  这些排除项的超集。依赖自身接触的模型不会被忠实复现。
- **关节限位**：importer 会丢弃 joint range，因此 PhysX 侧没有关节
  限位；`get_joint_range()` 仍返回 XML 值。关节 `armature` 与
  `frictionloss`（经 MJCF default class 解析）会应用到 PhysX dof。

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

## 训练与评估

安装好外部环境后即可训练。worker 运行时默认从 `~/.unilab/isaacgym` 自动
发现；自定义安装根时在训练前导出 `UNILAB_ISAACGYM_HOME` 即可。当前
`g1_walk_flat` 提供 PPO 与 SAC 的 isaacgym owner 配置：

```bash
# SAC
uv run train --algo sac --task g1_walk_flat --sim isaacgym

# PPO
uv run train --algo ppo --task g1_walk_flat --sim isaacgym
```

回放渲染由 IsaacGym 原生提供：交互模式在 worker 进程里打开 gym viewer，
录制模式用 camera sensor 离屏渲染并写出 `play_video.mp4`（相机跟踪
env 0 的 root，视角可用 `training.cam_distance` / `cam_elevation` /
`cam_azimuth` 调整）。`play_render_mode=auto`（默认）在有显示器
（`DISPLAY`/`WAYLAND_DISPLAY`）时进入交互模式，无显示器的机器自动降级为
录制模式；录制需要有限的 `training.play_steps`（默认配置已给出）。

```bash
# 训练后自动进入回放（auto）；无显示器的服务器自动录视频
uv run train --algo sac --task g1_walk_flat --sim isaacgym

# 对已训练的 checkpoint 做评估：交互 viewer
uv run eval --algo sac --task g1_walk_flat --sim isaacgym \
    --render-mode interactive --load-run <run_dir_name>

# 无显示器环境录视频（强制 record）
uv run eval --algo sac --task g1_walk_flat --sim isaacgym \
    --render-mode record --load-run <run_dir_name> training.play_steps=800
```

注意：交互 viewer 与 camera 采集都要求 worker 的 sim 跑在 GPU 上
（`env.isaacgym_device_id >= 0`）；CPU pipeline 的 sim 没有图形上下文，
渲染请求会 fail-closed 并给出提示。

常用覆盖（Hydra 参数直接跟在命令后面）：

```bash
# 小规模 smoke：64 个环境、只跑 3 个 iteration
uv run train --algo sac --task g1_walk_flat --sim isaacgym \
    algo.num_envs=64 algo.max_iterations=3

# 指定 worker 使用的 GPU
uv run train --algo sac --task g1_walk_flat --sim isaacgym env.isaacgym_device_id=1
```

跨后端迁移（sim2sim）：isaacgym owner 配置与 mujoco owner 在契约守卫
（`src/unilab/utils/sim2sim.py`）审计下全部字段兼容（TRANSFERABLE），
同一 task 的 checkpoint 可跨后端使用；回放渲染在 isaacgym 上已可用，
跨后端策略评估（在 isaacgym 上播放 mujoco 训练的 checkpoint，或反向）
可直接用上面的 `uv run eval` 命令完成。

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
