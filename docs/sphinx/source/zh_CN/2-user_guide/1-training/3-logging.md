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
| Collector Wait | `timing/learner_collector_wait_ms` | 等待 collector 产出新数据；不含 barrier、H2D、logger 刷新 |
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

APPO 沿用 ring buffer，collector 仅上报两项，均为**单步** EMA（非整条 rollout）：

| 终端字段 | TensorBoard / W&B key | 含义 |
| --- | --- | --- |
| env_step_total_ms | `timing/collector_env_step_total_ms` | 单次 `env.step()` 耗时的 EMA |
| mlp_infer_ms | `timing/collector_mlp_infer_ms` | 单步策略推理耗时的 EMA |
