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
