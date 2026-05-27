# Allegro Inhand

语言: 简体中文

## 任务

- rotation：`allegro_inhand`
- grasp cache：`allegro_inhand_grasp`

## 典型流程

1. 先生成 grasp cache
2. 再训练 rotation policy

## 配置入口

- PPO：`conf/ppo/task/allegro_inhand/`、`conf/ppo/task/allegro_inhand_grasp/`
- APPO：`conf/appo/task/allegro_inhand/`
- 默认 grasp cache：`cache/allegro_grasp_50k.npy`

## 常用命令

```bash
uv run scripts/train_rsl_rl.py task=allegro_inhand_grasp/mujoco training.no_play=true
uv run scripts/train_rsl_rl.py task=allegro_inhand/mujoco
uv run scripts/train_appo.py task=allegro_inhand/mujoco
```

Motrix owner 也存在：

```bash
uv run scripts/train_rsl_rl.py task=allegro_inhand/motrix
uv run scripts/train_appo.py task=allegro_inhand/motrix
```

## 回放和自定义 cache

```bash
uv run scripts/train_rsl_rl.py task=allegro_inhand/mujoco training.play_only=true
uv run scripts/train_appo.py task=allegro_inhand/mujoco training.play_only=true
```

```bash
uv run scripts/train_rsl_rl.py \
  task=allegro_inhand/mujoco \
  env.grasp_cache_path=cache/my_allegro_grasp.npy
```

## 边界

- backend 选择通过 task owner 或统一 CLI 的 `--sim`
- 不要单独切 `training.sim_backend`

## Navigation

- Index: [Documentation](../../../README.md)
- Previous: [G1 Motion Tracking](02-g1-motion-tracking.md)
- Next: [Sharpa Inhand](04-sharpa-inhand.md)
