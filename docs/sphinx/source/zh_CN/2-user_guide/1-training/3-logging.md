# 日志

训练配置默认使用 TensorBoard，即 `training.logger=tensorboard`。设置
`training.logger=wandb` 可启用 Weights & Biases 集成。

## TensorBoard

使用默认 logger 运行任意训练命令：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
```

运行目录会创建在 `logs/<algo.algo_log_name>/<task>/` 下，除非所选技术栈覆盖了
`training.log_root` 或 `training.log_dir`。

### 各算法的日志根目录

`algo_log_name` 由各技术栈的配置设置，并解析为具体的根目录：

| 算法 | 日志根目录 | `algo_log_name` 来源 |
| --- | --- | --- |
| PPO | `logs/rsl_rl_ppo/<task>/` | `conf/ppo/config.yaml` |
| APPO | `logs/appo/<task>/` | `conf/appo/config.yaml` |
| SAC | `logs/fast_sac/<task>/` | `conf/offpolicy/algo/sac.yaml` |
| FlashSAC | `logs/flash_sac/<task>/` | `conf/offpolicy/algo/flashsac.yaml` |
| TD3 | `logs/fast_td3/<task>/` | `conf/offpolicy/algo/td3.yaml` |

### run 目录命名

单个 run 目录以时间戳加仿真后端命名：

```text
YYYY-MM-DD_HH-MM-SS_<sim_backend>
```

例如 `2026-03-09_18-30-00_mujoco`。写入 run 目录的常见本地产物包括：

- `run_config.json`
- `run_summary.json`
- checkpoint 文件
- `play_video.mp4`（MuJoCo，当该次 run 产生了回放视频时）

## Weights & Biases

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  training.logger=wandb \
  training.wandb_project=unilab
```

受支持的共享 W&B 字段在训练配置块中声明：

- `training.wandb_project`
- `training.wandb_entity`
- `training.wandb_group`
- `training.wandb_name`
- `training.wandb_tags`
- `training.wandb_notes`
- `training.wandb_mode`

`src/unilab/training/experiment.py` 会在运行目录中写入 `run_config.json` 和
`run_summary.json`。当 `training.logger=wandb` 时，RSL-RL PPO 还会对 RSL-RL 的
W&B writer 打补丁。当后端为 MuJoCo 且该次 run 产生了 `play_video.mp4` 时，该视频会
被上传到 W&B run。

## Trace 选项

off-policy 配置暴露了 trace 字段，例如 `training.trace_enabled`、
`training.trace_output_dir`、`training.trace_thread_time` 和
`training.trace_cuda_events`。

## Off-Policy 计时字段

off-policy（SAC / TD3 / FlashSAC 与 APPO）把 learner loop 记录为已命名的墙钟阶段，所以 `Train` 可以直接看作完整迭代的占比，而不是只和 wait 指标比较。

| 计时阶段 | TensorBoard / W&B key | 含义 |
| --- | --- | --- |
| Collector Wait | `timing/learner_collector_wait_ms` | 等待 collector 提交下一次 inference request 并产出可训练数据；APPO 稳态下 ≈0，off-policy 同步路径按配置等待 collection chunk |
| Replay Batch Wait | `timing/learner_replay_batch_wait_ms` | 等上一轮末尾 prefetch 的 device batch 就绪（ingress commit + side-stream gather）；预取命中 ≈0，非零说明 device replay 流水线跟不上 learner 的消费速度 |
| Replay Sample | `timing/learner_replay_sample_ms` | 消费已 gather 完成的 hot device batch（hot/cold 槽交换 + view）；CUDA 下 ≈0，MPS 下含 slot event 同步 |
| Inference Total | `timing/inference_total_ms` | learner-owned inference 的 Observation H2D、actor forward 与 Action D2H 总耗时 |
| Collector Release | `timing/learner_collector_release_ms` | learner 在 Action D2H 后立即向同步 collector 发送 response token 的耗时；阻塞说明 collector 还没消费上一个 response |
| H2D Copy | `timing/learner_incremental_h2d_ms` | bounded ingress 增量拷入 authoritative device ring 的提交耗时（CUDA：后台线程 non_blocking 提交；MPS：learner 线程阻塞拷贝）；这部分工作在 CUDA 下与 Train 并行、或已发生在 Collector Wait / Replay Batch Wait 内，因此该行是诊断项，与其他 learner 行不严格可加 |
| Train | `timing/learner_train_ms` | 纯 SGD 计算耗时 |
| Weight Publish | `timing/learner_weight_publish_ms` | 仅 APPO：把 actor 新权重写入共享内存给其 collector；learner-owned off-policy inference 下为 0 |
| Iter Wall | `perf/iter_ms` | 整圈迭代墙钟，非各分量之和 |

终端常驻 `Collector Wait`、`Inference Total`（适用时）、`Train` 和 `Iter Wall`，用于直接判断 collector/CPU 等待、GPU inference 与训练占比。其余诊断阶段仅在占 `Iter Wall` 至少 1% 时出现，避免长期低占比指标挤占终端；TensorBoard / W&B 仍逐轮记录表中所有阶段。终端行都显示右对齐的 `ms` 与占 `Iter Wall` 百分比。后端额外记录 `perf/learner_train_pct`、`perf/learner_accounted_pct`、`perf/learner_other_pct` 和 `timing/learner_other_ms`；后两者是 residual 诊断，不占终端行。另有 `perf/learner_pipeline_ms` = learner inference + H2D + Train + APPO Weight Publish。原 `timing/learner_wait_ms` 已更名为 `timing/learner_collector_wait_ms`；原 `timing/learner_sync_coordination_ms`、`timing/learner_weight_sync_ms` 已更名为 `timing/learner_collector_release_ms`、`timing/learner_weight_publish_ms`。

collector 进程在终端 Collector 列、TensorBoard `timing/collector_*` 上报各阶段耗时；终端每行同时显示占单步采集周期（下列各行之和）的百分比。SAC / TD3 / FlashSAC：

| 终端字段 | TensorBoard / W&B key | 含义 |
| --- | --- | --- |
| Inference Request | `timing/collector_inference_request_ms` | 把 observation 与 dones 发布到共享 inference slot，并通知 learner |
| Inference Barrier Wait | `timing/collector_inference_wait_ms` | 等 learner-owned actor 发布当前 tick 的 action |
| Env Step | `timing/collector_env_step_ms` | 环境 step |
| Replay Write | `timing/collector_replay_write_ms` | 将 transition 打包并写入 bounded replay ingress 槽 |
| Sync Idle | `timing/collector_sync_idle_ms` | 每步的 episode 簿记与 metrics 上报；不计入 `Collector/s` |

`Collector/s` 表示 collector 活跃采集吞吐。SAC / TD3 / FlashSAC 使用
`num_envs / (Inference Request + Env Step + Replay Write)`，并排除
`Inference Barrier Wait` 与 `Sync Idle`。Env Step 下的三个缩进子项（Backend Step / Update State / Reset Done）以树形连接线标注，其百分比与其他行使用同一分母（单步采集周期总和）。

### 判读 CPU 与 GPU 的相对负载

流水线两侧暴露了相互对应的等待信号，可以据此判断瓶颈在哪一侧：

| 现象 | 含义 | 优化方向 |
| --- | --- | --- |
| learner Collector Wait 占比高 | learner 在等 collector 的 CPU 环境与 transition 路径 | 提高 `num_envs` 或降低环境/transition 开销 |
| learner Replay Batch Wait 高 | device replay 流水线（ingress commit + gather）跟不上 learner 消费 | GPU 已被训练占满，或 side stream 排在训练 kernel 之后 |
| collector Inference Barrier Wait 高 | collector 在等 learner inference 或 learner update phase | 检查 inference latency，并降低 `updates_per_step` / batch size |
| collector Replay Write 升高 | 写 bounded ingress 变慢，例如 ingress 槽耗尽、在等设备端 commit 释放槽位 | 与 Replay Batch Wait 同向：device 消费侧落后 |

### 单次迭代时序（以 FastSAC 为例）

下图描述 replay 已达到 `learning_starts` 后的稳态 FastSAC 迭代；默认
`training.env_steps_per_sync=1`，因此每个 learner 迭代服务一个 inference tick：

```{mermaid}
sequenceDiagram
    autonumber
    participant C as Collector（CPU / NumPy Env）
    participant I as Inference Slot + Queues
    participant L as Learner（GPU）
    participant R as Device Replay

    Note over C,R: tick t：learner 使用当前完整 policy_version
    C->>I: 发布 obs_t、dones_(t-1) 与 inference request
    I-->>L: Observation H2D
    L->>L: learner.actor.explore()
    L->>I: Action D2H + policy_version
    L->>I: 发布 inference response(t)

    par Collector 采集 transition_t
        I-->>C: 消费 action_t
        C->>C: env.step / reset / postprocess
        C->>R: 写入 bounded replay ingress
    and Learner replay 与固定更新阶段
        L->>R: Replay Batch Wait / side-stream gather
        R-->>L: iteration k 的 hot device batch
        L->>L: updates_per_step 次 critic / actor / target 更新
        L->>L: 完成后 policy_version += 1
    end

    Note over C,L: 两侧都完成后才服务 tick t+1；inference 与 learner update 不重叠
```

`Observation H2D`、`learner.actor.explore()` 和 `Action D2H` 分别对应
`Inference H2D`、`Inference Forward` 和 `Inference D2H`。learner 在 Action D2H 后立即
发布 response，因此 collector 的 `Env Step` / `Replay Write` 不仅与 `Train` 并行，
也与 learner 的 `Replay Batch Wait` / side-stream gather 重叠。本 tick 写入的
`transition_t` 进入 replay 供后续采样，并不要求被 iteration k 立即消费；下一轮
prefetch 的最小 snapshot 边界在发布 response 前冻结，不受两侧执行快慢影响。若
collector 先完成，它可以提交 tick t+1 的 request，但 learner 会等当前更新阶段结束后
才处理，所以 `Inference Barrier Wait` 同时可能包含 learner inference 和剩余 update 时间。

APPO 沿用 ring buffer，collector 上报两个**单步** EMA 和一个**整条 rollout** 的总时间：

| 终端字段 | TensorBoard / W&B key | 含义 |
| --- | --- | --- |
| MLP Infer | `timing/collector_mlp_infer_ms` | 单步策略推理耗时的 EMA（**每步**） |
| Env Step | `timing/collector_env_step_ms` | 单次 `env.step()` 耗时的 EMA（**每步**） |
| Rollout | `timing/collector_rollout_ms` | collector 产出**一条完整 rollout**（`steps_per_env` 步）的真实墙钟 EMA，列在该列最后作为总时间 |

APPO 的 `Collector/s` 使用 `(num_envs * steps_per_env) / Rollout`。

> Rollout ≈ `steps_per_env` ×（MLP Infer + Env Step）+ 每步未计时开销（如 timeout-bootstrap critic 前向、obs 处理）。它与 Learner 的 Collector Wait 是**两条独立时间线的视角**：采集与 learner 计算并行重叠，所以 Collector Wait（learner 真正阻塞的时间）通常**小于** Rollout，两者不必、也不会精确对账。想看"这一圈有多少卡在等 collector"，直接看 Collector Wait 行的百分比（= Collector Wait / Iter Wall）。所有 learner 行都使用同样的百分比格式。原 `env_step_total_ms`（`timing/collector_env_step_total_ms`）已更名为 `Env Step`（`timing/collector_env_step_ms`）。

### 单次迭代时序（以 APPO 为例）

collector 独立进程经 ring buffer 持续产 rollout；learner 每个迭代依次经历下列计时分量（括注为该指标含义）：

```{mermaid}
gantt
    title 一次 Learner 迭代的时间线（APPO）
    dateFormat x
    axisFormat %S

    section Collector（进程）
    rollout N · env interaction（mlp_infer + env_step）×steps_per_env :active, c0, 0, 12000
    rollout N+1（与 learner 并行采集）                                :active, c1, 13000, 30000

    section Ring Buffer（4 槽）
    rollout N 就绪    :milestone, r0, 12000, 12000
    rollout N+1 就绪  :milestone, r1, 30000, 30000

    section Learner（GPU）
    Collector Wait（缓冲满则约 0）    :done,   l0, 12000, 13000
    H2D Copy（ring 进 staging）       :        l1, 13000, 16000
    Train（V-trace + PPO SGD）        :active, l2, 16000, 28000
    Weight Publish 写回 collector     :crit,   l3, 28000, 30000

    section Iter Wall
    perf/iter_ms（仅 learner 这圈）   :        l4, 12000, 30000
```

> 横轴为示意相对时长（非真实 ms 比例）。collector 子进程经 4 槽 ring buffer 与 learner 并行产出 rollout，稳态下 **Collector Wait ≈ 0**。`perf/iter_ms` 仅计 learner 这一圈（含 Collector Wait，但不含 collector 的并行采集计算）；红色 Weight Publish 标志该轮迭代结束、向 collector 发布新权重。

所有 off-policy 终端视图都使用同一套数值格式。Replay Batch Wait 和 Collector Release 等诊断阶段达到 1% 阈值时才显示；Collector Release 适用于同步的 learner-owned inference 路径。
