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

## Off-Policy 计时字段语义

off-policy（SAC / TD3 / FlashSAC，以及 APPO）的终端面板与 TensorBoard/W&B
均把 learner 的「等待」拆分为 **4 个相互独立的分量**，不做平滑或合并，
以便分别定位真实阻塞来源（见 issue #633）。

| 终端字段 | TensorBoard / W&B key | 含义 | 单 GPU | 多 GPU |
| --- | --- | --- | --- | --- |
| Collector Wait | `timing/learner_collector_wait_ms` | learner 纯等待 collector 产出新 replay 数据 / collection tick 的时间。**不含** barrier、replay pack / H2D、logger 刷新 | ✔ | 仅 rank 0；在 rank barrier **之前**测量 |
| Replay Batch Wait | `timing/learner_replay_batch_wait_ms` | 等待预取的 replay pack / H2D batch 就绪的时间（double-buffer / 多卡 prefetch 轮询）；预取命中时 ≈ 0 | 基础 runner 恒为 0（同步 sample）；double-buffer 为实际值 | 仅 rank 0 |
| Rank Barrier | `timing/learner_rank_barrier_ms` | 多卡 `dist.barrier()`（数据后初始 + 训练后最终）耗时之和；只对已存在的 barrier 计时，不新增任何 collective | 0 | 仅 rank 0 |
| Sync Coordination | `timing/learner_sync_coordination_ms` | 同步采集握手（释放 collector 进入下一 tick）的耗时；非同步采集时为 0 | ✔ | 仅 rank 0 |
| H2D Copy | `timing/learner_incremental_h2d_ms` | 本次迭代消费数据的 host→device 拷贝耗时 | ✔ | 仅 rank 0 |
| Train | `timing/learner_train_ms` | **纯 SGD 计算**（不含 param-sync、不含最终 barrier） | ✔ | 仅 rank 0 |
| Param Sync | `timing/learner_param_sync_ms` | 多卡 local-SGD 参数平均（all-reduce）耗时 | 0 | 仅 rank 0，仅同步轮 |
| Weight Sync | `timing/learner_weight_sync_ms` | 把新权重发布到 collector 的耗时（fire-and-forget） | ✔ | 仅 rank 0 |
| Iter Wall | `perf/iter_ms` | **整圈循环的原始墙钟时间**，不是上述各项之和（还含未单独计时的 drain / batch 组装） | ✔ | 仅 rank 0 |
| — | `perf/learner_pipeline_ms` | learner 计算流水线 = H2D + Train + Param Sync + Weight Sync | ✔ | 仅 rank 0 |

要点：

- **单卡与多卡语义一致**；多卡额外把 rank barrier 与 param sync 拆成独立字段，
  非 rank 0 的计时字段恒为 0（logger 仅在 rank 0 输出）。
- **Collector Wait 不再混入** barrier / pack / H2D / logger 刷新时间。
- 旧的合并指标 `timing/learner_wait_ms` 已**重命名**为
  `timing/learner_collector_wait_ms`（破坏性变更）；依赖旧 key 的看板 / 查询需更新。
- 逐事件时间线可用 `training.trace_enabled=true` 产出
  `perfetto_offpolicy_timeline.json`，并用 `scripts/analyze_offpolicy_trace.py`
  对照各分量（对应 trace slice：`learner/wait_for_data`、
  `learner/wait_for_replay_batch` 等）。
