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
| MLX PPO | `logs/mlx_rl_train/<task>/` | `conf/ppo/config_mlx.yaml` |
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

off-policy（SAC / TD3 / FlashSAC 与 APPO）把 learner 的等待拆为四个独立分量，分别记录，不合并。

| 终端字段 | TensorBoard / W&B key | 含义 |
| --- | --- | --- |
| Collector Wait | `timing/learner_collector_wait_ms` | 等待 collector 产出新数据；不含 barrier、H2D、logger 刷新（终端在该行内联显示其占 Iter Wall 的百分比） |
| Replay Batch Wait | `timing/learner_replay_batch_wait_ms` | 等待 replay pack / H2D batch 就绪；预取命中时约为 0 |
| Rank Barrier | `timing/learner_rank_barrier_ms` | 多卡 `dist.barrier()`（初始 + 最终）耗时之和 |
| Sync Coordination | `timing/learner_sync_coordination_ms` | 同步采集握手耗时；非同步采集时为 0 |
| H2D Copy | `timing/learner_incremental_h2d_ms` | host→device 批次拷贝耗时 |
| Train | `timing/learner_train_ms` | 纯 SGD 计算，不含 param sync 与 barrier |
| Param Sync | `timing/learner_param_sync_ms` | 多卡 local-SGD 参数平均耗时 |
| Weight Sync | `timing/learner_weight_sync_ms` | 向 collector 发布新权重的耗时 |
| Iter Wall | `perf/iter_ms` | 整圈迭代墙钟，非各分量之和 |

单 GPU 仅显示 Collector Wait、H2D Copy、Train、Weight Sync 与 Iter Wall；Rank Barrier、Sync Coordination、Param Sync 仅在多 GPU 出现，且计时仅由 rank 0 记录。另有 `perf/learner_pipeline_ms` = H2D + Train + Param Sync + Weight Sync。原 `timing/learner_wait_ms` 已更名为 `timing/learner_collector_wait_ms`。

collector 进程在终端 Collector 列、TensorBoard `timing/collector_*` 上报各阶段耗时。SAC / TD3：

| 终端字段 | TensorBoard / W&B key | 含义 |
| --- | --- | --- |
| Weight Sync | `timing/collector_weight_sync_ms` | 拉取并加载 learner 新权重 |
| Action Select | `timing/collector_action_select_ms` | actor 推理选动作 |
| Env Step | `timing/collector_env_step_ms` | 环境 step |
| Replay | `timing/collector_replay_ms` | 写 replay buffer 与采样打包 |
| Sync Coordination | `timing/collector_sync_coordination_ms` | 同步采集握手（通知 learner、等待 learner 完成） |

APPO 沿用 ring buffer，collector 上报两个**单步** EMA 和一个**整条 rollout** 的总时间：

| 终端字段 | TensorBoard / W&B key | 含义 |
| --- | --- | --- |
| MLP Infer | `timing/collector_mlp_infer_ms` | 单步策略推理耗时的 EMA（**每步**） |
| Env Step | `timing/collector_env_step_ms` | 单次 `env.step()` 耗时的 EMA（**每步**） |
| Rollout | `timing/collector_rollout_ms` | collector 产出**一条完整 rollout**（`steps_per_env` 步）的真实墙钟 EMA，列在该列最后作为总时间 |

> Rollout ≈ `steps_per_env` ×（MLP Infer + Env Step）+ 每步未计时开销（如 timeout-bootstrap critic 前向、obs 处理）。它与 Learner 的 Collector Wait 是**两条独立时间线的视角**：采集与 learner 计算并行重叠，所以 Collector Wait（learner 真正阻塞的时间）通常**小于** Rollout，两者不必、也不会精确对账。想看"这一圈有多少卡在等 collector"，直接看 Collector Wait 行内联的百分比（= Collector Wait / Iter Wall）。原 `env_step_total_ms`（`timing/collector_env_step_total_ms`）已更名为 `Env Step`（`timing/collector_env_step_ms`）。

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
    Weight Sync 写回 collector        :crit,   l3, 28000, 30000

    section Iter Wall
    perf/iter_ms（仅 learner 这圈）   :        l4, 12000, 30000
```

> 横轴为示意相对时长（非真实 ms 比例）。collector 子进程经 4 槽 ring buffer 与 learner 并行产出 rollout，稳态下 **Collector Wait ≈ 0**。`perf/iter_ms` 仅计 learner 这一圈（含 Collector Wait，但不含 collector 的并行采集计算）；红色 Weight Sync 标志该轮迭代结束、向 collector 发布新权重。

SAC / TD3 与多卡路径在此基础上还会出现 Replay Batch Wait（等 replay pack / H2D batch 就绪）、Rank Barrier（多卡 rank 同步）、Param Sync（多卡参数平均）与 Sync Coordination（同步采集握手）；APPO 单卡不含这些。
