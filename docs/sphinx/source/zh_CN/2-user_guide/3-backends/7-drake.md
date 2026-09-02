# Drake 后端

UniLab 的 Drake 后端使用实验性 Python 分发包 `drake-uni`（代码中
`import drake_uni`），并加载针对本地 Drake C++ 安装编译的批量扩展。Drake
负责批量物理推进；task、reward、observation、reset 策略和训练编排仍由
UniLab 负责。

当前 registry 和部分 PPO、APPO、SAC task 已有 Drake owner YAML。该后端仍是
实验性、Linux-first 路径；存在 registry 或 owner YAML 不代表已经完成原生
训练验证。

## 前置条件

- 当前只记录 Linux x86_64 原生支持路径。
- Python `>=3.10,<3.14`、`uv`、C++20 编译器和 Python 开发头文件。
- 本地 Drake C++ 安装前缀必须包含：
  `include/drake/`、`include/pybind11/` 和 `lib/libdrake.so`。
- Eigen、fmt 的开发参数可由 `pkg-config`（或本地等价编译参数）发现。
- Drake batch 启动前不能在当前进程导入 `pydrake`。

Drake C++ 是外部工具链。安装 Python 包不会自动安装 Drake，也不会自动
编译原生扩展。

## 安装与构建

在 UniLab 仓库中安装可选 Python 依赖：

```bash
make setup-drake
# 等价命令：
uv sync --extra drake
```

如需可恢复的 Linux 完整安装（官方 Drake 前缀、Python extra、原生扩展和
import 诊断），运行：

```bash
bash scripts/tools/setup_drake_env.sh --download-drake
```

脚本可重复执行，下载和日志默认保存在 `~/.unilab/drake`。使用已有本地安装时，
可通过 `--drake-home`、`--deps-root` 和 `--drake-uni-source` 指定路径。

本地开发 DrakeUni 时可以使用 editable 安装：

```bash
uv pip install -e /path/to/drake_uni
```

必须使用与 UniLab 运行时相同的 Python 解释器构建扩展。扩展写入 DrakeUni
源码树，不是可跨环境复用的 wheel：

```bash
/path/to/unilab/.venv/bin/python \
  /path/to/drake_uni/scripts/build_drake_batch.py \
  --drake-home /path/to/drake/install
```

生成文件为 `src/drake_uni/compiled/_drake_env_pool*`。切换 Drake、Python、
编译器或虚拟环境后需要重新构建；不要提交生成的扩展。

## 资产准备

task materialize 的冷路径可能从 Hugging Face 下载机器人 mesh 和纹理。为
了可重复或离线运行，可以先预取：

```bash
uv run unilab-pull-assets --robot go1
uv run unilab-pull-assets --robot go2
```

## 训练冒烟

通过标准 CLI 选择后端，不要单独使用 `training.sim_backend=drake` 切换。

Stewart 场景较小，适合作为首次原生探针：

```bash
uv run train --algo ppo --task stewart_balance --sim drake \
  algo.max_iterations=1 algo.num_envs=8 training.no_play=true
```

准备好资产后，可使用已有 locomotion owner：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim drake \
  algo.max_iterations=1 algo.num_envs=16 training.no_play=true
```

这些命令只验证安装和 backend contract，不代表策略已经收敛。

## 回放与渲染

- `--render-mode none`：评估时不执行 playback。
- `--render-mode record`：Drake 仍负责物理 rollout，视频由 UniLab 的
  MuJoCo playback helper 离线渲染，因此录制仍需要 MuJoCo 和视觉资产。
- Drake interactive rendering 尚未实现。

示例：

```bash
uv run eval --algo ppo --task stewart_balance --sim drake \
  --load-run -1 --render-mode none
```

## 未支持边界

当前 Drake batch 对混入 `pydrake` 的进程、interactive rendering、reset
domain-randomization payload，以及非明确 body-force 的 interval 扰动均会
显式失败。除非有新的 native contract 和测试证据，不要在 Drake owner 中
启用这些能力。

## 支持证据等级

支持状态按 `Registered`、`Configured`、`Tested`、`Benchmarked`、
`Recommended` 分级。升级 support matrix 前，native 测试和训练记录必须包含
Drake 版本、编译器、Python ABI、task、算法、环境数、线程数和结果。
