# 日志

训练时看到的终端面板与 TensorBoard / W&B 不是两套 logger：算法 runner 只向同一个
training logger 提交一次指标。终端面板按固定 2 Hz 时钟刷新，并显示两秒时间滑动平均；
`training.logger=tensorboard`（默认）或 `training.logger=wandb` 决定持久化 backend，
backend 保留每个 iteration 未经时间平滑的值。

本文先说明所有算法共用的日志目录，再详细说明 SAC / TD3 / FlashSAC 与 APPO 共用的
off-policy 终端视图。表中的“终端字段”与 backend key 一一对应；后缀 `_ms` 均为毫秒。

## 日志目录与 backend

使用默认 TensorBoard logger 运行训练：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
```

运行目录默认位于 `logs/<algo.algo_log_name>/<task>/`，除非所选技术栈覆盖了
`training.log_root` 或 `training.log_dir`：

| 算法 | 日志根目录 | `algo_log_name` 来源 |
| --- | --- | --- |
| PPO | `logs/rsl_rl_ppo/<task>/` | `src/unilab/conf/ppo/config.yaml` |
| APPO | `logs/appo/<task>/` | `src/unilab/conf/appo/config.yaml` |
| SAC | `logs/fast_sac/<task>/` | `src/unilab/conf/sac/config.yaml` |
| FlashSAC | `logs/flash_sac/<task>/` | `src/unilab/conf/flashsac/config.yaml` |
| TD3 | `logs/fast_td3/<task>/` | `src/unilab/conf/td3/config.yaml` |

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

## Off-policy 终端视图

终端底部并排显示三列：

- `Learner (Iter Wall)`：learner 主线程的一圈；所有行使用 `Iter Wall` 作分母。
- `Collector (own clock)`：collector 子进程自己的采集时钟，与 learner 并行。
- `System`：buffer 大小、timeout rate、env 数和每 rank batch 大小。

面板边框标题显示 `GPUs N`。多卡训练中只有 rank 0 持有终端与持久化 logger；learner
指标和计时先在 rank 间取平均，再进入终端的两秒时间窗口。`Steps/s` 与 `Samples/s`
不取 rank 平均：前者把各 rank collector step rate 求和，后者把各 rank learner sample
rate 求和，因此两者都是整个训练任务的总吞吐。标题中的 `Avg 2s (n=...)` 表示当前窗口
包含多少个已经完成 rank 聚合的 learner 样本。

两列时间不能横向相加。只有 learner 列从 `Collector Wait` 到 `Other` 的行是互斥主线程
阶段；实现以 `Other = max(Iter Wall - accounted, 0)` 补齐未命名区间。正常情况下它们合计
等于 `Iter Wall`，每行百分比合计约为 100%（逐行取整可能略有误差）。
终端不再按 1% 阈值隐藏适用于当前算法的阶段，0 ms 也保留，便于与 TensorBoard / W&B
逐项对应。算法专有阶段不会跨算法占位或落盘；例如 `Replay Stage` 与 `Weight Publish` 只在
APPO / HORA-APPO 出现。

### Learner 主时间线

| 终端字段 | TensorBoard / W&B key | 适用路径 | 含义 |
| --- | --- | --- | --- |
| Collector Wait | `timing/learner_collector_wait_ms` | 全部 | learner 主线程等待达到本轮 update 边界；SAC 类路径会服务 inference request，并继续等 replay ready 与 `env_steps_per_sync` tick 数满足，APPO 等 ring 中 rollout |
| Inference | `timing/learner_inference_ms` | learner-owned inference | observation H2D、actor forward、action D2H 的总墙钟；三项嵌套明细见下表 |
| Collector Release | `timing/learner_collector_release_ms` | learner-owned inference | 将 action response token 发给 collector；正常应很短，阻塞表示 response queue 尚未腾空 |
| Replay Batch Wait | `timing/learner_replay_batch_wait_ms` | device replay | 等已预取的 device batch 完成 ingress commit 与 gather；预取命中时接近 0 |
| Replay Stage | `timing/learner_replay_stage_ms` | APPO | 将 ring buffer 中本轮新到的 NumPy rollout 顺序 materialize 到 learner staging pool；这是 learner 主线程的独占阶段 |
| Replay Sample | `timing/learner_replay_sample_ms` | 全部 | 取得 ready batch；CUDA device replay 通常只是 hot/cold swap 与 view，MPS 还可能等待 slot event |
| Train | `timing/learner_train_ms` | 全部 | learner update 阶段的墙钟 |
| Weight Publish | `timing/learner_weight_publish_ms` | APPO | 将新 actor / critic 权重写入共享内存 |
| Other | `timing/learner_other_ms` | 全部 | `Iter Wall` 减去上述互斥阶段的 residual，例如 metrics drain、reward stats 与 loop bookkeeping |
| Iter Wall | `perf/iter_ms` | 全部 | 从本轮 learner loop 开始到 update 完成的墙钟，固定显示 100% |

backend 还记录 `perf/learner_train_pct`、`perf/learner_accounted_pct` 与
`perf/learner_other_pct`。`accounted` 只包含上表的互斥主线程阶段，不包含任何嵌套或后台
诊断。正常采样下 `accounted + other = 100%`；如果系统时钟异常或未来埋点意外重叠使
`accounted > Iter Wall`，`Other` 会钳制为 0，而 `accounted_pct > 100%` 是应修复的 contract
告警，不是可解释的并行占比。

### Learner 嵌套与后台诊断

下列 key 只在 TensorBoard / W&B 记录，不是额外的 `Iter Wall` 切片：

| TensorBoard / W&B key | 父级或执行线程 | 含义 |
| --- | --- | --- |
| `timing/learner_inference_h2d_ms` | `Inference` 子项 | observation 从共享 CPU slot 拷到 learner device |
| `timing/learner_inference_forward_ms` | `Inference` 子项 | `learner.actor` 推理；包含当前实现中的 device synchronize |
| `timing/learner_inference_d2h_ms` | `Inference` 子项 | action 写回共享 CPU slot |
| `timing/replay_ingress_h2d_submit_ms` | replay ingress；CUDA daemon 或 MPS learner 线程 | 最近一次 transition span 提交到 authoritative device ring 的 CPU 侧耗时；它可能落在任意主阶段内，因此绝不能再加到 learner 百分比中 |

三项 inference 明细应近似组成 `Inference`；计时调用之间的 Python 开销可能造成小差值。
`Replay H2D Submit` 则与 learner 主时间线重叠：CUDA 由
`replay_gpu_resident_sync` daemon 提交 non-blocking copy，MPS 从已有 learner 调用推进。
要看 GPU copy / gather 的真实异步执行区间，应启用 trace，而不是把 submit wall time 当作
额外的 iteration 占比。

### 旧 run 的 tag 对照

历史 TensorBoard event 不会被重写。新 run 使用下列 canonical tag；打开旧 run 时仍会看到
旧名：

| 旧 tag | 新 tag | 变化原因 |
| --- | --- | --- |
| `timing/inference_total_ms` | `timing/learner_inference_ms` | 与终端 `Inference` 统一，并明确 owner |
| `timing/inference_{h2d,forward,d2h}_ms` | `timing/learner_inference_{h2d,forward,d2h}_ms` | 三个嵌套项统一放入 learner namespace |
| `timing/learner_incremental_h2d_ms`（SAC 类） | `timing/replay_ingress_h2d_submit_ms` | 明确它是可能并行的 submit 诊断，不是 learner 主阶段 |
| `timing/learner_incremental_h2d_ms`（APPO） | `timing/learner_replay_stage_ms` | 明确它是可计入 `Iter Wall` 的同步 staging 阶段 |
| `timing/collector_inference_wait_ms` | `timing/collector_learner_action_wait_ms` | 等待范围还包含剩余 learner update，不等于 inference latency |

`perf/learner_pipeline_ms` 已移除：它曾把互斥主阶段与后台 H2D submit 混加。主时间线请使用
`perf/iter_ms`，完整性请对照 `perf/learner_accounted_pct` 与
`perf/learner_other_pct`。

### Collector 自有时间线

SAC / TD3 / FlashSAC 每个 vectorized env tick 记录四个热路径互斥阶段，终端百分比使用
`perf/collector_cycle_ms`（这四项之和）作分母：

| 终端字段 | TensorBoard / W&B key | 含义 |
| --- | --- | --- |
| Inference Request | `timing/collector_inference_request_ms` | 发布 observation / dones 到共享 slot，并通知 learner |
| Learner Action Wait | `timing/collector_learner_action_wait_ms` | request 发出后，等 learner 发布当前 tick action 的屏障墙钟 |
| Env Step | `timing/collector_env_step_ms` | `env.step()` 墙钟 |
| Replay Write | `timing/collector_replay_write_ms` | transition 后处理、打包并写入 bounded ingress |

`Learner Action Wait` 特意不叫 “Inference Wait”：它不是纯 inference latency。如果 collector
在 learner update 期间先完成 `Env Step + Replay Write` 并提交下一 request，这一项会包含
剩余 update、下一轮 learner `Collector Wait` 中的少量调度延迟，以及下一次 `Inference +
Collector Release`。所以它很长并不与“两侧并行”矛盾，反而说明 collector 比 learner
update 更早到达下一屏障。

持久化指标 `perf/collector_active_steps_per_sec` 按 collector 活跃路径计算：
`num_envs / (Inference Request + Env Step + Replay Write)` 计算；它有意排除
`Learner Action Wait`；诊断价值较低的亚毫秒 episode / metrics bookkeeping 不再单独计时。
终端 `Steps/s` 则报告同步 collector 的总吞吐。`Env Step` 下缩进的 Backend Step /
Update State / Reset Done 是父项的嵌套明细，不参与 cycle 求和，但百分比仍使用同一
collector cycle 分母。

APPO 使用不同的采集 contract，因此 collector 只上报：

| 终端字段 | TensorBoard / W&B key | 口径 |
| --- | --- | --- |
| MLP Infer | `timing/collector_mlp_infer_ms` | 单步策略推理 EMA |
| Env Step | `timing/collector_env_step_ms` | 单次 `env.step()` EMA |
| Rollout Wall | `timing/collector_rollout_ms` | 完整 `steps_per_env` 步 rollout 的墙钟 EMA |

APPO 的三个值不是一组百分比分解：前两个是单步 EMA，`Rollout Wall` 是整条 rollout 总量，
所以终端只显示 ms。backend 的活跃吞吐诊断按
`(num_envs * steps_per_env) / Rollout Wall` 计算。

## FastSAC 双时间线

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

    par Collector tick t
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
| collector `Learner Action Wait` 高，同时 learner `Collector Wait` 低 | collector 先到下一屏障，在等 learner 完成 update 并服务 inference | `Train`、`updates_per_step`、batch size；再看 inference 三项明细 |
| learner `Inference` 高 | learner-owned action 路径本身慢 | H2D / forward / D2H 三项子指标 |
| learner `Replay Batch Wait` 高 | device replay 预取未赶上消费 | ingress commit、side-stream gather 与 GPU 竞争 |
| collector `Replay Write` 高 | bounded ingress 写入或 transition 后处理变慢 | ingress 槽是否耗尽、device commit 是否落后 |
| `Replay H2D Submit` 高但 learner 主阶段正常 | 后台 submit 诊断升高，不代表 iteration 多出同等墙钟 | Perfetto 中的 GPU copy、ingress commit 和重叠区间 |

## Trace 选项

off-policy 配置提供 `training.trace_enabled`、`training.trace_output_dir`、
`training.trace_thread_time` 与 `training.trace_cuda_events`。标量日志回答“每轮用了多久”；
需要判断异步 H2D、device gather、collector 与 learner 是否真正重叠时，以生成的 Perfetto
timeline 为准。
