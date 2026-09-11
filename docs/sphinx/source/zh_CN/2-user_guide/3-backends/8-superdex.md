# SuperDex 后端

SuperDex 是由 `unisim.backend.superdex` 拥有的可选 CPU 物理后端。UniLab 首个
owner 为固定基 `FR3JointTarget`，配置位于
`src/unilab/conf/ppo/task/fr3_joint_target/superdex.yaml`。任务使用 7 维力矩动作、
21 维观测、关节状态 reset 和标准 NumPy manager。当前支持等级为 **Configured**；
短 rollout 或少量训练迭代不能证明完整训练效果、性能或跨平台支持。
实施见 [#1534](https://github.com/Motphys/UniLab/issues/1534)，所属
roadmap 为 [#1533](https://github.com/Motphys/UniLab/issues/1533)。

## 安装

SuperDex Physics/Robotics 1.0.0 已发布 Python wheel，是 UniLab 的 optional
extra，不再需要源码编译 native extension。wheel 携带 native batch executor，
仅支持 CPython 3.12/3.13 的 Linux x86_64；其他平台上该 extra 为空，CLI 会给出
针对性的运行时诊断。CPU 物理不需要 CUDA。FR3 owner 没有 record（视频）回放——
`.superdex_bot` 资产没有 MJCF visual model——play 默认走 native interactive
viewer（`play_render_mode=interactive`，`play_env_num=1`）；无显示环境下使用
`training.play_render_mode=none`。

```bash
# 源码 checkout（默认 Python 3.13；wheel 支持 CPython 3.12/3.13）：
uv sync --extra superdex

# 从 PyPI 安装：
pip install "unilab[superdex]"
```

该 extra 通过 `unisim-core[superdex]` 委托 UniSim 钉定版本；UniSim 的
superdex extra 已包含普通 `mujoco` 包（MJCF 转换与离线回放渲染使用），不需要
`mjbatch`（仅 MuJoCo 物理后端需要）。当前 wheel 为临时
unilabsim 构建（`superdex-physics-uni`/`superdex-robotics-uni`）；上游
project_superdex 发布正式 `superdex-physics`/`superdex-robotics` wheel 后，
UniSim 会切换包名，UniLab 侧无需改动。

FR3 原生资产与其他机器人 mesh 资产一样托管在 Hugging Face
（[unilabsim/unilab-robots](https://huggingface.co/datasets/unilabsim/unilab-robots)），
wheel 不携带机器人二进制。asset hub 注册
`bots/arms/fr3_v2/fr3_v2.superdex_bot`，首次使用时自动把快照下载到
`src/unilab/assets/`，并在物理构造前验证 collision SDF、render、`LICENSE` 和
`NOTICE`。需要预拉取（如 CI 或离线准备）时：

```bash
uv run unilab-pull-assets --robot fr3_v2
```

若要审计本地 `project_superdex` checkout，可设置
`SUPERDEX_ASSETS_PATH=/absolute/path/to/project_superdex/assets`；单次运行也可通过
`env.superdex_assets_root=/absolute/path/to/project_superdex/assets` 覆盖。显式
root 优先于 Hugging Face 下载，且不会触发下载。

只有修改 SuperDex 引擎源码本身时才需要本地源码构建：
`bash scripts/tools/setup_superdex_env.sh` 会 clone integration branch、编译
native extension 并以 editable 方式链接本地 UniSim/UniLab；常规使用不需要它。

## 运行 FR3 任务

```bash
uv run --no-sync train --algo ppo --task fr3_joint_target --sim superdex \
  algo.max_iterations=2 algo.num_steps_per_env=16 \
  algo.algorithm.num_learning_epochs=1
```

目标关节角、reward、reset 范围和动作缩放由任务 `base.yaml` 声明。力矩上限
`[20,20,20,20,5,5,5]` Nm 是显式研究配置，不是硬件额定值；
`superdex_effort_limits` 在 native backend 边界声明同样的上限。SDK 固定为单线程；
下面的 native scene executor 是唯一支持的 CPU 并行层。

## 运行 Go2 任务

`go2_joystick_flat/superdex` owner 在 SuperDex 上训练和评估 Go2 四足机器人。
它继承 MuJoCo owner 的 policy I/O（49 维 actor 观测、52 维 critic 观测、
12 维位置目标动作）与控制时序，声明自己的 command 范围和 reward 权重，并显式
接受接触近似；contract 细节见下文"验证与归属"。

直接在 SuperDex 上训练（CPU 物理，自动 native worker）：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim superdex
```

owner 默认 1024 个环境、400 轮迭代。日志和 checkpoint 写入
`logs/rsl_rl_ppo/Go2JoystickFlat/<timestamp>_superdex/`。

用 `--load-run` 评估已训练的 run。默认 record 回放：跨 16 个环境推进 200 帧，
通过离线 MuJoCo renderer 把 `play_video.mp4` 写入 run 目录，物理仍由 SuperDex
执行：

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim superdex \
  --load-run 2026-09-10_10-50-00_superdex
```

如需 native SuperDex（Polyscope）viewer 而不是视频，加
`--render-mode interactive`；CLI 会强制 `training.play_env_num=1`，owner 层同时
把该次运行的 env 切到 serial executor。

在 MuJoCo 上训练的 checkpoint 可以直接在 SuperDex 上评估（sim2sim）：把它的
run 目录传给 `--load-run`。play 入口会在构造环境前按 sim2sim contract 验证来源
`run_config.json`，并拒绝不兼容的 policy I/O。

## 默认 CPU 环境并行

backend 使用 SuperDex wheel 携带的 `SceneBatchExecutor`。它是跨独立 scene 的常驻
C++ 线程池：每个子步批量写入广义力、推进 scene，并回写 articulation/link state、
contact sensor 和 solver status，不再逐环境跨越 Python binding。资产物化、reset 和
cache frame 转换仍由 UniSim adapter 负责。这是 CPU 线程并行，不是 GPU physics；它不改变
PPO/APPO collector、learner 或 policy contract。决策见
{doc}`/adr/ADR-0009-superdex-persistent-cpu-workers` 和
[unisim#41](https://github.com/unilabsim/unisim/issues/41)。

两个 task owner 默认选择自动 worker：

| Owner 选项 | 含义 |
| --- | --- |
| `env.superdex_num_workers=0` | 自动：`min(affinity 内可用物理核心数, num_envs)` |
| `env.superdex_num_workers=1` | 一个 native C++ scene worker |
| `env.superdex_num_workers=K` | 显式 C++ worker 数，最多为 `num_envs` |

native scene 的 `DebugDraw` 状态有线程亲和性，因此 batch 模式与已连接的
native SuperDex debugger 不兼容：adapter 在检测到 debugger 客户端连接时会直接
报错并给出可操作提示。需要使用 native debugger 调试时，以
`env.superdex_execution_mode=serial` 运行——该模式在环境线程上逐步推进每个
scene，完全不创建 native worker pool
（[unisim#55](https://github.com/unilabsim/unisim/issues/55)）。serial 是调试
配置，不是性能配置。

`eval --sim superdex --render-mode interactive` 走 native SuperDex（Polyscope）
viewer 渲染，而不是 MuJoCo viewer 路径。owner 层会自动把 env 切到
`superdex_execution_mode=serial`，CLI 同时强制 `training.play_env_num=1`（viewer
只绘制一个 scene）；显式传入的 `training.play_env_num=...` 会被保留。

1024 个环境、16 核 32 线程主机上，自动解析为 16 workers。多 rank 并发 collector
按完整物理核心分配，并将同一核心的 logical sibling 放在同一 rank，避免拆分 SMT 核心。
也可以通过 `training.dp_collector_cpu_ids` 为每个 rank 显式提供 CPU id 列表；该分片在
SuperDex 创建 native worker pool 前应用。

每个物理子步由 host 执行 pre-step control callback，随后进入 native batch barrier，
完成后发布新 batch state，再执行下一 callback。局部 reset 保留请求行顺序，不影响
未选择环境。native worker 错误会关闭 executor 并报告失败，不静默返回旧状态或回退
串行。env.close 会在销毁 scene 前 join C++ worker。

worker 数不能代替加速证据。吞吐对照应匹配模型、控制序列、batch 和子步，记录完整
backend/env 时间、startup、RSS、CPU 使用和实际 worker 数。吞吐按 env 控制步统计，
不能把 physics 子步重复计入样本。已有接触近似的物理边界保持不变。

固定根具有名称和可读的 body state，但 reset 只写 joint state，不要求 free-root
layout。该任务不需要接触 sensor、相机、site Jacobian 或材料 DR。SuperDex 没有 native
renderer；默认 record 回放使用离线 MuJoCo renderer，物理仍由 SuperDex 执行。
`.superdex_bot` 场景需要提供 `visual_model_file`。`play_render_mode=none` 会完全跳过
回放，不能作为 checkpoint rollout 已执行的证据。

`superdex_allow_contact_approximation` 默认 `false`，只供经过审核的 MJCF 转换配置
显式启用。启用后会警告 contact/material 近似，包括 torsional/rolling friction
不等价；这不代表任意 MJCF 任务已兼容。
接触查询返回最近一次完成求解的结果。reset 清除该结果，需要一次正时间步才会
产生新接触，不能把 reset 后的 contact 当成即时几何重叠测试。运动学 body/joint
getter 则在 reset 后立即刷新。

## 验证与归属

`go2_joystick_flat/superdex` PPO owner 是研究性质的 sim2sim 配置。它继承 MuJoCo
owner，保留 49 维 actor 观测、52 维 critic 观测、12 维位置目标动作、归一化、网络
维度和控制时序；关闭 runtime PD gain DR，并显式接受接触近似。adapter 的冷路径
MJCF 转换有明确限制，该 owner 不代表任意场景或行走效果等价。

先用 MuJoCo owner 创建一个小型来源 checkpoint，再把路径传给可选 checkpoint 测试：

```bash
uv run --no-sync train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.num_envs=2 algo.max_iterations=2 algo.num_steps_per_env=16 \
  algo.algorithm.num_mini_batches=1 algo.algorithm.num_learning_epochs=1 \
  training.device=cpu training.no_play=true
export UNILAB_SUPERDEX_GO2_CHECKPOINT=/absolute/path/to/source/run/model_1.pt
uv run --no-sync pytest tests/envs/test_go2_superdex.py -q
```

测试在构造 env 前验证来源 `run_config.json`，检查修改动作语义时确实拒绝，随后
通过 production playback session 加载真实策略，无渲染执行 64 个 SuperDex 控制步。
它验证有限数值和接口兼容性；只训练两轮的 checkpoint 不以可靠行走为验收标准。

```bash
uv run --no-sync pytest tests/assets/test_superdex_assets.py \
  tests/envs/test_fr3_superdex.py tests/test_cli_runtime_requirements.py -q
```

原生测试要求 SDK 和 FR3 资产（按需从 Hugging Face 下载，或通过
`SUPERDEX_ASSETS_PATH` 提供），覆盖有限数值 rollout、局部 reset 隔离、即时观测
刷新及 spawn `EnvFactory`。缺失 SDK/资产会明确 skip，不能将 skip 记为原生验证
通过。基础资产和配置测试不依赖原生资产 checkout。

引擎转换与物理由 UniSim 拥有；资产注册、Hydra 与任务 term 由 UniLab 拥有。
相关约束见 {doc}`/adr/ADR-0007-unisim-extraction-boundary`、
{doc}`/adr/ADR-0006-community-manager-api-on-numpy-runtime` 和
{doc}`/adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot`。
