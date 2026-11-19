# 日志

训练时看到的终端面板与 TensorBoard / W&B 不是两套 logger：算法 runner 只向同一个
training logger 提交一次指标。终端面板按固定 2 Hz 时钟刷新，并显示两秒时间滑动平均；
`training.logger=tensorboard`（默认）或 `training.logger=wandb` 决定持久化 backend，
backend 保留每个 iteration 未经时间平滑的值。

本文先说明所有算法共用的日志目录和 Manager-Based reward 指标契约，再详细说明
SAC / FlashSAC 与 APPO 共用的 off-policy 终端视图。持久化 backend tag
遵循 `unilab.rl.logging.metric_schema` 持有的 canonical schema：毫秒计时字段以
`_ms` 结尾，而 `Perf/learning_time`、`Perf/collection_time` 与
`Perf/iteration_time` 单位为秒。少数终端字段（`Other`、各阶段百分比、collector
cycle 合计、`Rows/s`）是在内存中为显示推导的值，有意不持久化。

## 日志目录与 backend

使用默认 TensorBoard logger 运行训练：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
```

运行目录默认位于 `logs/<algo.algo_log_name>/<TaskName>/`，其中 `<TaskName>`
是 registry 任务名（例如 `Go2JoystickFlat`），不是 CLI `--task` 的值，除非所选
技术栈覆盖了 `training.log_root` 或 `training.log_dir`：

| 算法 | 日志根目录 | `algo_log_name` 来源 |
| --- | --- | --- |
| PPO | `logs/rsl_rl_ppo/<TaskName>/` | `src/unilab/conf/ppo/config.yaml` |
| APPO | `logs/appo/<TaskName>/` | `src/unilab/conf/appo/config.yaml` |
| SAC | `logs/sac/<TaskName>/` | `src/unilab/conf/sac/config.yaml` |
| FlashSAC | `logs/flash_sac/<TaskName>/` | `src/unilab/conf/flashsac/config.yaml` |

单个 run 目录名为 `YYYY-MM-DD_HH-MM-SS_<sim_backend>`，例如
`2026-03-09_18-30-00_mujoco`。常见产物包括 `run_config.json`、
`run_summary.json`、checkpoint，以及该 run 产生时的 `play_video.mp4`。

启用 W&B：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  training.logger=wandb \
  training.wandb_project=unilab
```

共享配置字段包括 `training.wandb_project`、`training.wandb_entity`、
`training.wandb_group`、`training.wandb_name`、`training.wandb_tags`、
`training.wandb_notes` 和 `training.wandb_mode`。`ExperimentTracker` 会写
`run_config.json` 与 `run_summary.json`；RSL-RL PPO 在 W&B 模式下还会接管
RSL-RL writer。MuJoCo run 若产生 `play_video.mp4`，该视频会上传至 W&B。

## Manager-Based reward 指标

Manager-Based 环境把结构化指标放在 `info["log"]` 中。reward 分项的路径因训练
技术栈而异：off-policy run（SAC / FlashSAC / APPO）通过 `info["reward_components"]`
发布每个 term 的值并记录为 `reward/<term>`；RSL-RL PPO 路径不发出分项 reward
key——其 TensorBoard 输出是原生聚合指标（`Loss/*`、`Perf/*`、`Policy/*`、
`Train/*`）加 episode 级 UniLab key。reward 相关 key 的契约如下：

| Key | 归属 | 含义 |
| --- | --- | --- |
| `reward/<term>` | UniLab `RewardManager`，经 off-policy 日志栈 | 当前 reward 计算的分项诊断：`raw_value * weight` 在所有 env 上的平均值，不经过 `dt` 缩放。由 SAC / FlashSAC / APPO run 发出；PPO（rsl-rl）路径不发出 |
| `Train/mean_reward` | RSL-RL PPO 原生 logger | 已完成 episode return 的聚合平均值；查看整体训练进度使用它 |
| `Train/mean_episode_length` | RSL-RL PPO 原生 logger | 已完成 episode 长度的聚合平均值 |
| `Episode_Termination/<term>` | UniLab `TerminationManager` | 本次 reset 的 env 中触发该 termination term 的数量；这是独立诊断，不是 reward |
| `Episode_Reward/<term>` | 已移除 | 历史版本中的按 term 归一化累计值；它与 `reward/<term>` 语义不同却形成重复树，因此被有意移除 |

历史 TensorBoard event 文件不可重写，因此变更前的旧 run 仍可能显示
`Episode_Reward/<term>`；新的 Manager-Based run 不应再产生它。

## Off-policy 终端视图

终端有显式宽度分层：

- 小于 96 列时，metrics / rewards / timing 区域纵向堆叠，timing 使用单一
  `Timing` 列；使用短标题，并降低 reward peak 精度，而不是截断核心标签。
- 96–119 列时，metrics 与 rewards 同行显示，timing 使用 `Learner` / `Collector`
  短标题。
- 120 列及以上时，timing 使用完整三列：
  - `Learner (Iter Wall)`：learner 主线程的一圈；所有行使用 `Iter Wall` 作分母。
  - `Collector (own clock)`：collector 子进程自己的采集时钟，与 learner 并行。
  - `System`：buffer 大小、timeout rate、env 数、每 rank batch 大小，以及 rank
    本地 CUDA MPS 状态（验证通过的请求显示 `on`，环境级透明 daemon 显示
    `ambient`，否则 `off`）。

Metric 渲染按语义分组：loss 与 policy 状态使用稳定短标签，training 诊断缩进，
replay-ingress gauge 显示为 `Ingress Backpressure`、`Ingress Depth` 等标签；它们
不放入 loss 组，也与 learner 时间线的 `Collector Wait` 语义不同。

面板边框标题显示 `GPUs N`。多卡训练中只有 rank 0 持有终端与持久化 logger；learner
指标和计时先在 rank 间取平均，再进入终端的两秒时间窗口。`Steps/s` 与 `Rows/s`
不取 rank 平均：前者把各 rank collector step rate 求和，后者把各 rank learner
replay 行速率求和，因此两者都是整个训练任务的总吞吐。其中只有 `Steps/s` 会持久化
（即 `Perf/total_fps`），`Rows/s` 仅终端显示。标题中的 `Avg 2s (n=...)` 表示当前窗口
包含多少个已经完成 rank 聚合的 learner 样本。

两列时间不能横向相加。只有 learner 列从 `Collector Wait` 到 `Other` 的行是互斥主线程
阶段；实现以 `Other = max(Iter Wall - accounted, 0)` 补齐未命名区间。正常情况下它们合计
等于 `Iter Wall`，每行百分比合计约为 100%（逐行取整可能略有误差）。
算法专有阶段不会跨算法占位或落盘；例如 `Replay Stage` 与 `Weight Publish` 只在
APPO 出现。

### Learner 主时间线

| 终端字段 | TensorBoard / W&B key | 适用路径 | 含义 |
| --- | --- | --- | --- |
| Collector Wait | `Perf/learner_collector_wait_ms` | 全部 | learner 主线程等待达到本轮 update 边界；SAC 类路径会服务 inference request，并继续等 replay ready 与 `env_steps_per_sync` tick 数满足，APPO 等 ring 中 rollout |
| Inference | `Perf/learner_inference_ms` | learner-owned inference | observation H2D、actor forward、action D2H 的总墙钟；三项嵌套明细见下表 |
| Collector Release | `Perf/learner_collector_release_ms` | learner-owned inference | 将 action response token 发给 collector；正常应很短，阻塞表示 response queue 尚未腾空 |
| Replay Batch Wait | `Perf/learner_replay_batch_wait_ms` | device replay | 等已预取的 device batch 完成 ingress commit 与 gather；预取命中时接近 0 |
| Replay Stage | `Perf/learner_replay_stage_ms` | APPO | 将 ring buffer 中本轮新到的 NumPy rollout 顺序 materialize 到 learner staging pool；这是 learner 主线程的独占阶段 |
| Replay Sample | `Perf/learner_replay_sample_ms` | 全部 | 取得 ready batch；CUDA device replay 通常只是 hot/cold swap 与 view，MPS 还可能等待 slot event |
| Learning | `Perf/learning_time`（秒） | 全部 | learner update 阶段的墙钟 |
| Weight Publish | `Perf/learner_weight_publish_ms` | APPO | 将新 actor / critic 权重写入共享内存 |
| Other | 仅终端 | 全部 | `Iter Wall` 减去上述互斥阶段的 residual，例如 metrics drain、reward stats 与 loop bookkeeping |
| Iter Wall | `Perf/iteration_time`（秒） | 全部 | 从本轮 learner loop 开始到 update 完成的墙钟，固定显示 100%；仅在 runner 测得完整迭代墙钟时写出 |

`Other` 与各行百分比是在内存中为终端推导的值，不作为 backend chart 持久化。
`accounted` 只包含上表的互斥主线程阶段，不包含任何嵌套或后台
诊断。正常采样下 `accounted + other = 100%`；如果系统时钟异常或未来埋点意外重叠使
`accounted > Iter Wall`，`Other` 会钳制为 0，而 accounted 超过 100% 是应修复的 contract
告警，不是可解释的并行占比。

### Learner 嵌套与后台诊断

下列 key 只在 TensorBoard / W&B 记录，不是额外的 `Iter Wall` 切片：

| TensorBoard / W&B key | 父级或执行线程 | 含义 |
| --- | --- | --- |
| `Perf/learner_inference_h2d_ms` | `Inference` 子项 | observation 从共享 CPU slot 拷到 learner device |
| `Perf/learner_inference_forward_ms` | `Inference` 子项 | `learner.actor` 推理；包含当前实现中的 device synchronize |
| `Perf/learner_inference_d2h_ms` | `Inference` 子项 | action 写回共享 CPU slot |
| `Perf/replay_ingress_h2d_submit_ms` | replay ingress；CUDA daemon 或 MPS learner 线程 | 最近一次 transition span 提交到 authoritative device ring 的 CPU 侧耗时；它可能落在任意主阶段内，因此绝不能再加到 learner 百分比中 |

三项 inference 明细应近似组成 `Inference`；计时调用之间的 Python 开销可能造成小差值。
`Replay H2D Submit` 则与 learner 主时间线重叠：CUDA 由
`replay_device_sync` daemon 提交 non-blocking copy，MPS 从已有 learner 调用推进。
要看 GPU copy / gather 的真实异步执行区间，应启用 trace，而不是把 submit wall time 当作
额外的 iteration 占比。

### 旧 run 的 tag 对照

历史 TensorBoard event 不会被重写。unilab-rl 1.4.1 及以后的 run 使用下列
canonical tag（权威定义：`unilab.rl.logging.metric_schema`，完整迁移表见原 unilab_rl
仓库的 `docs/metrics.md`）；打开旧 run 时仍会看到已退役的旧名：

| 旧 tag | 新 tag | 说明 |
| --- | --- | --- |
| `perf/steps_per_sec` | `Perf/total_fps` | 聚合 collector env-step 吞吐（DP 下跨 rank 求和） |
| `reward/mean`、`reward/mean_ep100` | `Train/mean_reward` | collector 最近 100 个 episode 的 return 均值；runner 的 10-report 平滑仅是 checkpoint 状态 |
| `episode/timeout_rate` | `Episode/timeout_rate` | 首个完成 episode 之前不写出 |
| `perf/iter_ms` | `Perf/iteration_time` | 毫秒 → 秒；仅在测得完整迭代墙钟时写出 |
| `timing/learner_train_ms` | `Perf/learning_time` | 毫秒 → 秒 |
| `timing/learner_<phase>_ms` | `Perf/learner_<phase>_ms` | 同样的毫秒阶段计时，canonical 命名空间 |
| `timing/inference_total_ms`、`timing/inference_{h2d,forward,d2h}_ms` | `Perf/learner_inference_ms`、`Perf/learner_inference_{h2d,forward,d2h}_ms` | 同一批字段的更早期名称 |
| `timing/learner_incremental_h2d_ms`（SAC 类）、`timing/replay_ingress_h2d_submit_ms` | `Perf/replay_ingress_h2d_submit_ms` | 可能并行的 submit 诊断，不是 learner 主阶段 |
| `timing/learner_incremental_h2d_ms`（APPO） | `Perf/learner_replay_stage_ms` | 可计入 `Iter Wall` 的同步 staging 阶段 |
| `timing/collector_<phase>_ms` | `Perf/collector_<phase>_ms` | 同样的毫秒 collector 阶段，canonical 命名空间 |
| `timing/collector_inference_wait_ms` | `Perf/collector_learner_action_wait_ms` | 等待范围还包含剩余 learner update，不等于 inference latency |
| `timing/collector_rollout_ms` | `Perf/collection_time` | 毫秒 → 秒；APPO 的整条 rollout 墙钟 |
| `train/dp_sync_time` | `Perf/dp_gradient_sync_ms_per_rank` | 秒 → 毫秒；明确为每 rank 值 |
| `train/dp_gradient_sync_calls` | `Perf/dp_gradient_sync_calls_per_rank` | 明确为每 rank 值 |

已退役且无替代 tag：`perf/effective_samples_per_sec` /
`perf/learner_samples_per_sec`（可由 run 配置与 `Perf/iteration_time` 推导 replay
行数）、`perf/collector_active_steps_per_sec`（可由 collector 计时与 run 配置推导）、
`perf/collector_cycle_ms`、`perf/learner_*_pct`、`timing/learner_other_ms` 与
`perf/learner_pipeline_ms`——它们都可以从上面的 canonical 字段推导，或本来就是仅终端
显示的推导值。提取工具 `scripts/benchmark/rl/extract_offpolicy_metrics.py` 默认只读取
schema-v1 canonical tag；`metric_schema_version` 缺失或不支持时直接失败。

### Schema 版本与兼容策略

Metric 元数据与 runtime manifest 是两个独立版本命名空间。生产端权威定义位于
`unilab.rl.logging.metric_schema` 与 `unilab.rl.logging.runtime_manifest_schema`：

| 命名空间 | 当前值 | 存储位置 |
| --- | ---: | --- |
| TensorBoard / W&B metric schema | `metric_schema_version = 1` | `run_summary.json` 顶层 |
| Runtime 进程 / 设备 / IPC manifest | `runtime_manifest.schema_version = 1` | 内嵌 runtime manifest |
| 长时间 soak 监控 artifact | `schema_version = "0.3.0"` | soak JSON artifact |

三个值刻意相互独立：soak 监控 artifact 升级不代表 metric 语义变化，metric schema
升级也不会让所有 runtime-manifest 字段失效。当前 soak monitor 只接受 metric schema 1、
runtime-manifest schema 1 与 soak artifact `0.3.0`。

Runtime-manifest v1 冻结 UniLab 已经消费的公开配置与生命周期字段：inference ring
容量、collector metric interval、`collector_backend_device`、inference-flight 计数、
replay-ingress 计数/耗时，以及存在该预算时的 CUDA inference IPC event 数。
`collector_backend_device` 在 completed manifest 中必填；只有 backend 本身不需要加速器
绑定时才允许为 `null`。`inference_flight.max_*` 与
`replay_ingress.high_water_occupancy` 的窗口语义不同：inference 最大值是相对上一次
成功发布的 collector report，replay-ingress high-water 是整个 run 的累计高水位。
未列入稳定契约的字段（包括 `runtime_manifest.shutdown`）在 v1 中都是实验性诊断。
`runtime_manifest.backend_diagnostics` 就是这样一项诊断：由
`TorchEnv.get_backend_runtime_diagnostics()` 序列化的物理 backend tensor runtime
快照（每项优化的 `requested`/`enabled`/`disable_reason`，包括 CUDA graph 状态）；
runtime manifest schema 在该 section 存在时会 fail-closed 校验，但其内容仍归
producer 所有。PPO 路径没有 runtime manifest，因此其训练脚本把同样的 payload 作为
顶层 `backend_diagnostics` 字段写入 `run_summary.json`。payload 形状与故障排查见
{doc}`7-tensor_runtime_production`。

Soak artifact `0.3.0` 顶层新增两个异常 shutdown 字段。通过时
`failure_classification` 必须为 `null`，且 `shutdown.classification` 必须是
`normal_completion`；`owner` 与 `phase` 必须是非空字符串，cleanup 不能报错。
失败时，有效的 producer 分类（`learner_failure`、`collector_failure`、
`backend_worker_failure`、`timeout/stale_tick`、`external_cancellation`、
`unknown_failure`）优先，因为它能标识发起 owner；否则 UniLab 记录 monitor
分类（例如 `timeout/stale_tick` 或 `resource_leak`）。顶层
`shutdown` 在 producer 可用时保存原始 host-only
`runtime_manifest.shutdown` mapping；teardown 无法生成诊断时为 `null`。
Producer shutdown snapshot 是 best-effort：它只追加到原始 soak failure
reason 之后，绝不替换原始错误。runtime-manifest v1 的 `shutdown` mapping
仍保持 opaque，因此该 soak artifact 契约不会把 producer 自有诊断扩展变成
稳定字段。

同一 schema major version 内，生产端可以伴随 schema、测试和文档更新增加可选字段。
稳定字段不能改名、删除、改变类型、单位或 null 语义；若必须变化，需要提升 schema
version。下游消费者遇到缺失或不支持的版本时拒绝猜测，直接失败。

历史 event 文件不可修改。唯一批准的兼容路径是
`extract_offpolicy_metrics.py --legacy-unversioned`。该模式只接受已识别的
pre-schema legacy tag，要求不存在 canonical tag 或 schema stamp，并且不能作为
未来未打版本日志的 fallback。可执行的 DP scaling benchmark 会生成并覆盖当前
run，因此只消费 schema-v1 canonical tag，不提供 historical mode。默认 metric
reader 遇到 canonical tag、canonical/legacy 混合或不支持的 metric schema version
都会 fail closed。

### Collector 自有时间线

SAC / FlashSAC 每个 vectorized env tick 记录热路径互斥阶段，终端百分比使用这些
阶段之和（collector cycle）作分母；该合计在内存中推导，不作为 backend chart 持久化：

| 终端字段 | TensorBoard / W&B key | 含义 |
| --- | --- | --- |
| Inference Request | `Perf/collector_inference_request_ms` | 发布 observation / dones 到共享 slot，并通知 learner |
| Learner Action Wait | `Perf/collector_learner_action_wait_ms` | request 发出后，等 learner 发布当前 tick action 的屏障墙钟 |
| Env Step | `Perf/collector_env_step_ms` | `env.step()` 墙钟 |
| Replay Write | `Perf/collector_replay_write_ms` | transition 后处理、打包并写入 bounded ingress |

终端始终保留上表核心 cycle 行。`Inference Request`、`Transition Extract`、
`Metrics Publish`，以及嵌套的 `Action Validate` / `Apply Action` /
`Action+Backend` 明细属于安静诊断项：它们仍是 canonical TensorBoard / W&B 字段，
但只有达到 collector cycle 的 1% 时才出现在终端。`Backend Step`、
`Update State` 与 `Reset Done` 是嵌套诊断而非额外 cycle 切片，同样按该显著性规则显示。
对于事件型异步后端（mjwarp），设备物理等待发生在 step 输出的第一次 host 读取
——update_state 内 termination 阶段的 queue drain（若 update_state 没有 host 读取，
则为 autoreset 的 done-mask 门控）——并归因到 `Backend Step`，因此该行反映真实设备
工作而非仅 kernel 发射开销。

`Learner Action Wait` 特意不叫 “Inference Wait”：它不是纯 inference latency。如果 collector
在 learner update 期间先完成 `Env Step + Replay Write` 并提交下一 request，这一项会包含
剩余 update、下一轮 learner `Collector Wait` 中的少量调度延迟，以及下一次 `Inference +
Collector Release`。所以它很长并不与“两侧并行”矛盾，反而说明 collector 比 learner
update 更早到达下一屏障。

collector 活跃吞吐诊断（`num_envs / (Inference Request + Env Step + Replay Write)`，
有意排除 `Learner Action Wait`）不再作为 backend chart 持久化；需要时可由 canonical
collector 计时字段与 run 配置推导。终端 `Steps/s` 则报告同步 collector 的总吞吐，
并持久化为 `Perf/total_fps`。`Env Step` 下缩进的明细行是父项的嵌套诊断，不参与
cycle 求和，但百分比仍使用同一 collector cycle 分母。

APPO 使用不同的采集 contract，因此 collector 只上报：

| 终端字段 | TensorBoard / W&B key | 口径 |
| --- | --- | --- |
| MLP Infer | `Perf/collector_mlp_infer_ms` | 单步策略推理 EMA |
| Env Step | `Perf/collector_env_step_ms` | 单次 `env.step()` EMA |
| Rollout Wall | `Perf/collection_time`（秒） | 完整 `steps_per_env` 步 rollout 的墙钟 |

APPO 的三个值不是一组百分比分解：前两个是单步 EMA，`Rollout Wall` 是整条 rollout 总量，
所以终端只显示 ms。backend 的活跃吞吐诊断 `(num_envs * steps_per_env) / Rollout Wall`
同样不再持久化，可由上述字段推导。

## SAC 双时间线

默认 `training.env_steps_per_sync=1`。下面的时序从 learner iteration `k` 开始；虚线横向
消息是同步点，`par` 内两侧真实并行：

```{mermaid}
sequenceDiagram
    autonumber
    participant C as Collector（CPU / NumPy Env）
    participant I as Inference Slot + Queue
    participant L as Learner 主线程
    participant D as Device Replay / CUDA daemon

    Note over C,L: iteration k 开始：collector 已发布 request(t)
    C->>I: Inference Request(t)
    I-->>L: Collector Wait 结束
    rect rgb(235, 245, 255)
        L->>I: Inference：obs H2D
        L->>L: learner.actor forward
        L->>I: action D2H
        L-->>C: Collector Release / response(t)
    end

    par Collector 迭代 t
        C->>C: Env Step(t)
        C->>D: Replay Write(t)
        C->>I: Inference Request(t+1)
        Note over C,I: Learner Action Wait(t+1) 从这里开始
    and Learner iteration k
        L->>D: Replay Batch Wait + Replay Sample(k)
        L->>L: Train(k)：固定 updates_per_step
        Note over D,L: replay ingress H2D / gather 可在后台与 Train 重叠
    end

    I-->>L: iteration k+1 的 Collector Wait 结束
    L->>I: Inference(t+1)
    L-->>C: response(t+1)，Learner Action Wait 结束
```

这里的对应关系是：

- learner `Inference(t)` 是 collector `Learner Action Wait(t)` 的末段；两者不是同一长度。
- `Env Step(t) + Replay Write(t)` 与 learner 的 replay / `Train(k)` 并行。
- 下一次 request 可在 `Train(k)` 尚未结束时到达，但 learner 只在完整 update 之后服务它；
  因此 `Learner Action Wait(t+1)` 可很长，而下一轮 learner `Collector Wait` 仍接近 0。
- inference 与 learner update 不重叠；每个 tick 使用完整的 `policy_version`。

## APPO 双时间线

APPO collector 连续产出 rollout，learner 消费 ring buffer 中已完成的 slot：

```{mermaid}
gantt
    title 一次 APPO learner iteration（示意比例）
    dateFormat x
    axisFormat %S

    section Collector（独立进程）
    rollout N+1：MLP Infer + Env Step + 其他采集工作 :active, c0, 0, 18000

    section Ring Buffer
    rollout N ready   :milestone, r0, 0, 0
    rollout N+1 ready :milestone, r1, 18000, 18000

    section Learner（Iter Wall）
    Collector Wait    :done,   l0, 0, 1000
    Replay Stage      :        l1, 1000, 3000
    Replay Sample     :        l2, 3000, 4000
    Train             :active, l3, 4000, 16000
    Weight Publish    :crit,   l4, 16000, 18000

    section Learner 总墙钟
    Iter Wall         :        l5, 0, 18000
```

collector rollout 与 learner `Iter Wall` 是独立时间线，会重叠而不能相加。稳态 ring 中已有
slot 时 `Collector Wait` 接近 0；collector 的 `Rollout Wall` 可以大于、等于或小于某次
learner `Iter Wall`，取决于 ring backlog 和两侧吞吐。

其中 `Replay Stage` 是把所有新到 slot 写入 staging pool 的主线程时间，随后
`Replay Sample` 才从 staging pool 组合本轮训练 batch；两者是相邻、可相加的 learner
阶段，不是后台 H2D 诊断。

## 如何判读瓶颈

| 现象 | 直接含义 | 优先检查 |
| --- | --- | --- |
| learner `Collector Wait` 高 | learner 到达迭代开头后，collector 数据/request 尚未就绪 | env step、transition 后处理、collector 存活与 IPC |
| collector `Learner Action Wait` 高，同时 learner `Collector Wait` 低 | collector 先到下一屏障，在等 learner 完成 update 并服务 inference | `Learning`、`updates_per_step`、batch size；再看 inference 三项明细 |
| learner `Inference` 高 | learner-owned action 路径本身慢 | H2D / forward / D2H 三项子指标 |
| learner `Replay Batch Wait` 高 | device replay 预取未赶上消费 | ingress commit、side-stream gather 与 GPU 竞争 |
| collector `Replay Write` 高 | bounded ingress 写入或 transition 后处理变慢 | ingress 槽是否耗尽、device commit 是否落后 |
| `Replay H2D Submit` 高但 learner 主阶段正常 | 后台 submit 诊断升高，不代表 iteration 多出同等墙钟 | Perfetto 中的 GPU copy、ingress commit 和重叠区间 |

## Trace 选项

off-policy 配置提供 `training.trace_enabled`、`training.trace_output_dir`、
`training.trace_thread_time` 与 `training.trace_cuda_events`。标量日志回答“每轮用了多久”；
需要判断异步 H2D、device gather、collector 与 learner 是否真正重叠时，以生成的 Perfetto
timeline 为准。
