# 日志、run 目录与 W&B

语言: 简体中文

训练日志统一落在 `logs/<algo.algo_log_name>/<task>/`。

## 常见日志根目录

- PPO：`logs/rsl_rl_ppo/<task>/`
- MLX PPO：`logs/mlx_rl_train/<task>/`
- APPO：`logs/appo/<task>/`
- SAC：`logs/fast_sac/<task>/`
- FlashSAC：`logs/flash_sac/<task>/`
- TD3：`logs/fast_td3/<task>/`

## run 目录命名

单个 run 目录通常是：

```text
YYYY-MM-DD_HH-MM-SS_<sim_backend>
```

例如：

```text
2026-03-09_18-30-00_mujoco
```

常见本地产物包括：

- `run_config.json`
- `run_summary.json`
- checkpoint 文件
- MuJoCo 自动回放产出的 `play_video.mp4`（若该次训练有视频回放）

## W&B

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  training.logger=wandb \
  training.wandb_project=unilab \
  training.wandb_entity=my-team
```

常用字段：

- `training.wandb_project`
- `training.wandb_entity`
- `training.wandb_group`
- `training.wandb_name`
- `training.wandb_tags`
- `training.wandb_notes`
- `training.wandb_mode=offline`

如果 backend 是 MuJoCo 且产生了 `play_video.mp4`，当前训练日志会把该视频带进 W&B run。

## Navigation

- Index: [Documentation](../../index.md)
- Previous: [Hydra 覆盖规则](03-hydra-overrides.md)
- Next: [训练相关 Docker 用法](05-docker.md)
