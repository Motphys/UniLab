# Drake 后端

本页给出在 UniLab 中运行 Drake 批量后端的完整路径。新机器请按顺序执行：准备
原生工具链、安装 Python extra、编译扩展、运行诊断、准备资产，最后执行短时训练
冒烟测试。

Drake 负责批量物理推进；task、reward、observation、reset 策略和训练编排仍由
UniLab 负责。渲染统一使用 MuJoCo 原生 renderer：Drake 推进状态，MuJoCo 只消费
这些状态绘制画面（不会再推进第二套物理仿真）。该后端仍是实验性、Linux-first
路径；registry 条目或 owner YAML 本身不代表某个 task 已完成原生训练验证。

如果需要面向 CPU 的批量物理，并且能够准备外部 C++ 工具链，可以选择 Drake。
MuJoCo 或 Motrix 可能仍有更广的 task 覆盖，但 Drake 回放的录制和交互可视化都
统一使用 MuJoCo。使用 `--sim drake` 前，请确认算法/task 存在 Drake owner YAML，
并在 {doc}`../../5-reference/5-support_matrix` 中查看对应证据。

## 系统和编译依赖

完整原生路径当前只记录 Linux x86_64。官方 `noble` tarball 面向 Ubuntu 24.04
构建；本仓库验证过的路径是 Ubuntu 24.04（或具有兼容 glibc/libstdc++ 运行库的
发行版）。其他操作系统和 CPU 架构不在 setup 脚本覆盖范围内。

| 组件 | 要求 |
| --- | --- |
| Python | `>=3.10,<3.14`（已验证的 Drake 运行使用 Python 3.12；推荐 3.12 或 3.13） |
| Python 环境 | `uv`，以及由仓库创建的虚拟环境 |
| 编译器 | 可执行文件名为 `c++`（或通过 `CXX` 指定）的 C++20 编译器 |
| 构建/运行工具 | `git`、`curl`、`tar`、`pkg-config`（下载路径都会用到） |
| 头文件/库 | Eigen 3、fmt、spdlog 开发包，以及编译扩展所用 Python 的开发头文件 |
| Drake 前缀 | `include/drake/`、`include/pybind11/` 和 `lib/libdrake.so` |

Ubuntu 或 Debian 可先安装系统包：

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  pkg-config \
  libeigen3-dev \
  libfmt-dev \
  libspdlog-dev \
  curl \
  git
```

如果使用系统 Python，还要安装匹配版本的开发头文件（例如 `python3-dev`）。
uv 管理的 Python 通常已经包含扩展编译所需的头文件。`build_drake_batch.py`
直接调用 C++ 编译器，不需要额外安装 `cmake`。

## 推荐的完整安装

在 UniLab2 checkout 根目录运行：

```bash
bash scripts/tools/setup_drake_env.sh --download-drake
```

脚本可恢复、可重复执行。下载文件、完成标记和合并日志默认保存在
`~/.unilab/drake`。常用选项：

```text
--drake-home <path>        使用已有 Drake C++ 前缀
--deps-root <path>         使用包含 Eigen/fmt/spdlog 的 apt 风格解压目录
--drake-uni-source <path>  使用本地 DrakeUni checkout
--drake-version <version>  指定下载的 Drake 版本
--drake-platform <name>    指定官方 tarball 平台（默认 noble）
```

脚本不会通过 `sudo` 安装系统包。它会先检查前缀，再使用 UniLab 虚拟环境中的
Python 编译扩展，确保生成文件与训练时的 ABI 一致。

### 环境变量

脚本结束时会打印 export 命令，但子进程无法修改父 shell。请在新 shell 中（或
设置好下列变量后）运行 Drake：

```bash
export DRAKE_HOME="$HOME/.unilab/drake/drake-1.56.0-noble"
export UNILAB_DRAKE_HOME="$DRAKE_HOME"
export UNILAB_DRAKE_UNI_SOURCE="$PWD/../drake_uni"
export LD_LIBRARY_PATH="$DRAKE_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

如果运行时使用了 `--deps-root /path/to/deps`，将依赖库目录加入搜索路径：

```bash
export LD_LIBRARY_PATH="/path/to/deps/usr/lib/x86_64-linux-gnu:$DRAKE_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

如果 DrakeUni 不在 UniLab2 的相邻目录，请将 source 路径替换为传给
`--drake-uni-source` 的 checkout。生成的
`src/drake_uni/compiled/_drake_env_pool*` 依赖机器和 Python ABI，不能提交到
仓库。切换 Drake、Python、编译器或虚拟环境后都要重新编译。

## 已有前缀或手动编译

使用已有前缀时无需下载：

```bash
bash scripts/tools/setup_drake_env.sh \
  --drake-home /opt/drake \
  --drake-uni-source /home/user/ws/unilabsim/drake_uni
```

前缀必须包含：

```text
/opt/drake/include/drake/
/opt/drake/include/pybind11/
/opt/drake/lib/libdrake.so
```

手动重编译时，使用运行 UniLab 的同一个解释器：

```bash
uv pip install -e /home/user/ws/unilabsim/drake_uni
uv run --no-sync python \
  /home/user/ws/unilabsim/drake_uni/scripts/build_drake_batch.py \
  --drake-home /opt/drake
```

构建脚本通过 `pkg-config` 查找 Eigen/fmt。如果依赖不在系统标准路径，可设置
`EIGEN3_INCLUDE_DIR`、`FMT_INCLUDE_DIR`、`FMT_LIB_DIR` 和 `PKG_CONFIG_PATH`，
或在 setup 脚本中传入 `--deps-root`。

## 验证安装

在 Drake 库可见且没有导入 `pydrake` 的干净进程中运行诊断：

```bash
uv run --no-sync python - <<'PY'
import drake_uni
from drake_uni.runtime import batch_diagnostics

diagnostics = batch_diagnostics()
print("drake_uni:", drake_uni.__file__)
print("batch diagnostics:", diagnostics)
if not diagnostics.batch_available:
    raise SystemExit(diagnostics.batch_import_error or "Drake batch extension is unavailable")
PY
```

成功结果应包含 `batch_available=True`。如果扩展不可用，请重新运行完整 setup
脚本，并检查 `~/.unilab/drake/install.log`。

专项 contract 和训练测试：

```bash
uv run --no-sync pytest \
  tests/base/backend/test_drake_batch_pool.py -q
uv run --no-sync pytest \
  tests/scripts/test_drake_training_smoke.py -m slow -q
```

slow 冒烟测试只运行一次 PPO iteration（4 个环境、每个环境 4 步），用于检查真实
训练入口，不用于衡量策略收敛。

## 准备机器人资产

机器人 mesh 和纹理由 Hugging Face 托管，task materialize 的冷路径会按需下载。为
了可重复或离线运行，可以提前拉取：

```bash
uv run --no-sync unilab-pull-assets --robot go1
uv run --no-sync unilab-pull-assets --robot go2
# CI/离线镜像（下载所有已注册机器人）：
uv run --no-sync unilab-pull-assets --robot all
```

Stewart balance 场景不需要 Go1/Go2 mesh，适合作为第一步探针。Go1、Go2、Go2-arm
等 task 需要对应资产。UniLab 会将文件放在受管理的资产目录中，请勿手动移动 mesh。

## 运行训练冒烟

通过顶层 CLI 的 `--sim` 选择 Drake。不要单独使用
`training.sim_backend=drake` 切换后端；该字段由所选 owner YAML 设置。

先运行较小的 Stewart 场景：

```bash
uv run train --algo ppo --task stewart_balance --sim drake \
  algo.max_iterations=1 \
  algo.num_envs=8 \
  algo.num_steps_per_env=4 \
  training.no_play=true \
  env.drake_nthread=1
```

拉取资产后，用相同的有界参数检查 locomotion owner：

```bash
uv run train --algo ppo --task go1_joystick_flat --sim drake \
  algo.max_iterations=1 \
  algo.num_envs=4 \
  algo.num_steps_per_env=4 \
  training.no_play=true \
  env.drake_nthread=1

uv run train --algo ppo --task go2_joystick_flat --sim drake \
  algo.max_iterations=1 \
  algo.num_envs=4 \
  algo.num_steps_per_env=4 \
  training.no_play=true \
  env.drake_nthread=1
```

这些命令用于验证安装和 backend contract，不是生产超参数。正式训练时去掉一次
iteration 的限制，并根据主机 CPU 和内存选择环境数量：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim drake
```

评估最近一次运行且不打开回放窗口：

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim drake \
  --load-run -1 --render-mode none
```

默认 `training.play_render_mode=auto` 时，Drake 回放会自动录制 MuJoCo 渲染视频，
无需额外指定 renderer 参数：

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim drake --load-run -1
```

如果需要 Drake 物理驱动的 MuJoCo 交互窗口，使用通用 viewer 入口（viewer 是
MuJoCo；`--sim drake` 只选择物理 owner）：

```bash
uv run python -m unilab.scripts.play_interactive \
  --algo ppo --task go2_joystick_flat --sim drake \
  interactive.action_mode=policy
```

## 回放与渲染边界

- `--render-mode none`：禁用回放，适合无头训练/评估。
- Drake 没有独立 renderer。自动录制和交互 viewer 都使用 MuJoCo 原生渲染 API，
  但实际推进的物理引擎始终只有 Drake。
- `--render-mode record` 是自动录制路径的显式写法，仍需要 MuJoCo extra 和视觉
  资产。
- 不需要渲染时使用 `--render-mode none`；这是唯一需要主动指定的渲染开关。
- 这不是 sim-to-sim：MuJoCo 不会被推进，checkpoint 也不会在 MuJoCo 物理下重新
  评估；renderer 只接收当前 Drake 状态用于可视化。

## 常见问题

| 现象 | 检查/处理 |
| --- | --- |
| `ModuleNotFoundError: drake_uni` | 在 UniLab2 中重新运行 `bash scripts/tools/setup_drake_env.sh --download-drake`；使用已有前缀时传入 `--drake-home`，使用本地 checkout 时传入 `--drake-uni-source`。 |
| `DrakeEnvPool batch extension has not been built` | 用 UniLab 使用的同一个 uv Python 执行构建脚本；不要复制其他 Python ABI 生成的扩展。 |
| `libdrake.so: cannot open shared object file` | 设置 `DRAKE_HOME`，并将 `$DRAKE_HOME/lib`（以及 `--deps-root` 的库目录）加入 `LD_LIBRARY_PATH`。 |
| 找不到 `include/drake` 或 `include/pybind11` | `--drake-home` 应指向 Drake 安装前缀，而不是下载目录的父目录。 |
| `fatal error: Python.h` | 安装匹配的 `python3-dev`，或使用 uv 管理的 Python 重建环境。 |
| 找不到 Eigen/fmt/spdlog 头文件或库 | 安装 `libeigen3-dev`、`libfmt-dev`、`libspdlog-dev`，或通过 `--deps-root` 指定解压的依赖树。 |
| `robot mesh not found` | 执行 `uv run --no-sync unilab-pull-assets --robot <robot>` 预取对应资产。 |
| 诊断提示已加载 `pydrake` | 新开进程，并在导入 `pydrake` 前构造 Drake batch backend；混用进程会显式失败。 |

## 支持证据等级

支持声明按 `Registered`、`Configured`、`Tested`、`Benchmarked`、`Recommended`
分级，彼此不能混用。升级 support matrix 前，native 测试或训练记录应包含
Drake 版本、编译器、Python ABI、task、算法、环境数、线程数和结果。
