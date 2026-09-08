# SuperDex 后端

SuperDex 是由 `unisim.backend.superdex` 拥有的可选 CPU 物理后端。UniLab 首个
owner 为固定基 `FR3JointTarget`，配置位于
`src/unilab/conf/ppo/task/fr3_joint_target/superdex.yaml`。任务使用 7 维力矩动作、
21 维观测、关节状态 reset 和标准 NumPy manager。当前支持等级为 **Configured**；
短 rollout 或少量训练迭代不能证明完整训练效果、性能或跨平台支持。
实施见 [#1534](https://github.com/Motphys/UniLab/issues/1534)，所属
roadmap 为 [#1533](https://github.com/Motphys/UniLab/issues/1533)。

## 本地开发安装

该开发配置使用本地链接的 UniSim 与 UniLab，不要求发布新版本。SuperDex
Physics/Robotics 1.0.0 要求 Python 3.12；CPU 物理不需要 CUDA。FR3 owner 暂不包含
原生渲染和视频，默认 `no_play=true`。

在 UniLab checkout 中使用已有的 Python 3.12 环境，或创建环境后安装本地包：

```bash
uv venv --python 3.12
export UNILAB_LOCAL_UNISIM=/absolute/path/to/unisim
uv pip install -e "${UNILAB_LOCAL_UNISIM}[superdex,mujoco]" -e . --group pyproject.toml:dev
export UV_NO_SYNC=1
export SUPERDEX_ASSETS_PATH=/absolute/path/to/project_superdex/assets
```

`UNILAB_LOCAL_UNISIM` 启用严格的本地依赖验证：测试同时检查 editable 安装元数据
和实际 import 路径确实指向指定 checkout。不设置时保留正常的索引发布包检查。
`UV_NO_SYNC=1` 让现有 Make 目标保留本地链接；`uv sync` 会重新解析锁定的发布依赖。

FR3 原生资产保留在上游 checkout。asset hub 注册
`bots/arms/fr3_v2/fr3_v2.superdex_bot`，在物理构造前验证 collision SDF、render、
`LICENSE` 和 `NOTICE`。UniLab 不打包或下载这些二进制。单次运行可通过
`env.superdex_assets_root=/absolute/path/to/project_superdex/assets` 覆盖环境变量。

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

## 默认 CPU 环境并行

backend 使用 SuperDex 源码构建的 `SceneBatchExecutor`。它是跨独立 scene 的常驻
C++ 线程池：每个子步批量写入广义力、推进 scene，并回写 articulation/link state、
contact sensor 和 solver status，不再逐环境跨越 Python binding。资产物化、reset 和
cache frame 转换仍由 UniSim adapter 负责。这是 CPU 线程并行，不是 GPU physics；它不改变
PPO/APPO collector、learner 或 policy contract。决策见
{doc}`/adr/ADR-0008-superdex-persistent-cpu-workers` 和
[unisim#41](https://github.com/unilabsim/unisim/issues/41)。

两个 task owner 默认选择自动 worker：

| Owner 选项 | 含义 |
| --- | --- |
| `env.superdex_num_workers=0` | 自动：`min(affinity 内可用物理核心数, num_envs)` |
| `env.superdex_num_workers=1` | 一个 native C++ scene worker |
| `env.superdex_num_workers=K` | 显式 C++ worker 数，最多为 `num_envs` |

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

原生测试要求 SDK 和 `SUPERDEX_ASSETS_PATH`，覆盖有限数值 rollout、局部 reset
隔离、即时观测刷新及 spawn `EnvFactory`。缺失 SDK/资产会明确 skip，不能将 skip
记为原生验证通过。基础资产和配置测试不依赖原生资产 checkout。

引擎转换与物理由 UniSim 拥有；资产注册、Hydra 与任务 term 由 UniLab 拥有。
相关约束见 {doc}`/adr/ADR-0007-unisim-extraction-boundary`、
{doc}`/adr/ADR-0006-community-manager-api-on-numpy-runtime` 和
{doc}`/adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot`。
